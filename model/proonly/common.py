"""Shared plumbing for the pro-only experiment (branch exp/pro-only-patch-emb).

The question this branch answers: **what does pro data alone buy, with no soloQ
pretraining at all?** Three things are pinned here so the two trainers and the
comparison script cannot drift apart:

* **Corpus.** `--corpus 2022` (default) is the documented corpus every recorded
  number is measured on (46,813 drafts, 2022-2026). `--corpus all` is the wider
  2017-2026 set (81,161 drafts) that arrived with the newer OE exports. They
  are separate caches, because alternating between two globs behind one cache
  file rescans the CSVs every time.
* **Split.** `pipeline.split_within_patch`: train = head of every patch,
  val/test = its tail. This is what makes a patch embedding meaningful at all
  - under the global date split the test patches are unseen, so every test
  row's patch id is UNK and the embedding can only memorise.
* **Eval view.** One fixed mask seed for the masked task, so "patch on" vs
  "patch off" is a paired comparison on identical slots rather than two
  independent percentages.
"""
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent            # .../model/proonly
MODEL_DIR = HERE.parent                           # .../model
ROOT = MODEL_DIR.parent                           # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"
SOLOQ_PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"

sys.path.insert(0, str(MODEL_DIR))
from soloq_model import SoloQPretrainedModel, load_encoder_state  # noqa: E402

# Fixed for every run and every arm. The masked eval masks must not depend on
# the training seed, otherwise the two arms score different slots.
EVAL_SEED = 20261101

CORPORA = {"2022": "pro_drafts.pt", "all": "pro_drafts_all.pt"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")
EP = _load("eval_protocol", PRO_DIR / "eval_protocol.py")
POLICY_MODEL = _load("policy_model", MODEL_DIR / "policy" / "model.py")

PolicyValueModel = POLICY_MODEL.PolicyValueModel


def add_corpus_args(ap):
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="2022",
                    help="2022 = the documented 46,813-draft corpus (default); "
                         "all = 2017-2026, 81,161 drafts")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="share of each patch's tail held out for val")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--min-patch-rows", type=int, default=40,
                    help="patches smaller than this stay entirely in train "
                         "(their tail would be 1-2 games)")
    ap.add_argument("--half-life", type=float, default=0.0,
                    help="recency weighting in days; 0 (default) = uniform. "
                         "Uniform on purpose: the test window is the tail of "
                         "*every* patch, so down-weighting old games would "
                         "just make the old patches unlearnable")
    ap.add_argument("--limit-drafts", type=int, default=0,
                    help="smoke runs: evenly-strided subsample of the corpus")


def load_split(args, verbose=True):
    """rows + within-patch split + patch vocabulary, in one place."""
    csv_glob = PRO_PIPE.CSV_GLOB if args.corpus == "2022" \
        else PRO_PIPE.CSV_GLOB_ALL
    rows, stats = PRO_PIPE.build_rows(
        csv_glob=csv_glob, cache_path=PRO_DIR / CORPORA[args.corpus])
    if args.limit_drafts:
        stride = max(1, len(rows) // args.limit_drafts)
        rows = rows[::stride][:args.limit_drafts]
        print("limited run: %d drafts (stride %d)" % (len(rows), stride))
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_within_patch(
        rows, val_frac=args.val_frac, test_frac=args.test_frac,
        min_patch_rows=args.min_patch_rows, verbose=verbose)
    patch_vocab, patch_index = PRO_DATA.build_patch_vocab(rows)
    if verbose:
        print("corpus %s: %d drafts, %d patches (%s..%s)"
              % (args.corpus, len(rows), len(patch_vocab),
                 patch_vocab[0], patch_vocab[-1]))
        print("split: train %d / val %d / test %d"
              % (len(splits["train"]), len(splits["val"]),
                 len(splits["test"])))
    return rows, splits, patch_vocab, patch_index


def build_encoder(num_champs, embed_dim, num_intra, num_cross, num_heads,
                  dropout, n_patches, with_value=False):
    return SoloQPretrainedModel(
        num_champs=num_champs, embed_dim=embed_dim, num_heads=num_heads,
        num_intra_layers=num_intra, num_cross_layers=num_cross,
        dropout=dropout, with_value=with_value, n_patches=n_patches)


def checkpoint_patch_space(ckpt, patch_vocab, patch_index, n_patches):
    """-> (n_patches, patch_index, vocab) honouring the checkpoint's vocabulary.

    A checkpoint's patch ids are *positions* in the vocabulary it was trained
    with. A run that initialises from a checkpoint with a patch table must use
    that id space, or the table cannot be loaded: the shapes would not match,
    and the error would be a size mismatch that says nothing about the cause.

    The run's own vocabulary can legitimately differ from the checkpoint's:

    * `--limit-drafts` subsamples the corpus, so the run sees fewer patches;
    * `--corpus all` sees more patches than a 2022 checkpoint was trained on.

    Both are handled the same way - the checkpoint's vocabulary wins, and a row
    whose patch is not in it gets UNK (id 0), which is the honest id for a
    patch the model has never seen. `patch_to_id` already falls back to UNK.

    When the checkpoint has no patch table there is nothing to inherit, and
    the run's own values stand: the table is created fresh and
    zero-initialised.

    This is a no-op for a run whose corpus and limits match the checkpoint's,
    which is every recorded run.
    """
    ckpt_vocab = list(ckpt.get("patch_vocab") or [])
    ckpt_rows = int((ckpt.get("config") or {}).get("n_patches", 0))
    if not ckpt_rows:
        return n_patches, patch_index, list(patch_vocab)
    if len(ckpt_vocab) + 1 != ckpt_rows:
        raise RuntimeError(
            "init checkpoint's patch table has %d rows but its vocabulary "
            "holds %d patches; the two must agree or the ids mean nothing"
            % (ckpt_rows, len(ckpt_vocab)))
    return ckpt_rows, PRO_DATA.patch_index_from_vocab(ckpt_vocab), ckpt_vocab


def encoder_from_checkpoint(path, n_patches, device="cpu"):
    """Rebuild an encoder from a checkpoint config, tolerating a patch mismatch.

    `n_patches` is passed in rather than read from the checkpoint so the
    policy stage can choose its own patch arm: an encoder trained with a patch
    embedding still works without one, but the table cannot be invented when
    the checkpoint has none. Mismatches that would silently change meaning
    (wrong D, wrong layer count) still raise inside `load_encoder_state`.
    """
    payload = torch.load(str(path), map_location=device, weights_only=False)
    cfg = payload["config"]
    n_p = int(n_patches if n_patches is not None else cfg.get("n_patches", 0))
    enc = build_encoder(cfg["num_champs"], cfg["embed_dim"], cfg["num_intra"],
                        cfg["num_cross"], cfg["num_heads"],
                        cfg.get("dropout", 0.0), n_p)
    info = load_encoder_state(enc, payload["model"])
    return enc, cfg, info


def param_count(model):
    return sum(p.numel() for p in model.parameters())


class SliceAccumulator:
    """Counts top-k hits per slice key, with the baseline on the same rows.

    Deliberately not `EP.RankReport`: that one is built around patch-honest
    slices (a focus patch, a same-patch reference) which do not exist here.
    What this experiment needs is the opposite question - *which* patches the
    model fails on - so the keys are the patch itself and its year.
    """

    def __init__(self, ks=(1, 5, 10)):
        self.ks = ks
        self.by_patch = {}
        self.by_year = {}

    def _bump(self, table, key, ranks, base_ranks):
        e = table.setdefault(key, {"n": 0, "hits": {k: 0 for k in self.ks},
                                   "base_hits": {k: 0 for k in self.ks}})
        e["n"] += len(ranks)
        for k in self.ks:
            e["hits"][k] += sum(1 for r in ranks if r < k)
            if base_ranks is not None:
                e["base_hits"][k] += sum(1 for r in base_ranks if r < k)

    def add(self, ranks, patches, base_ranks=None):
        for i, r in enumerate(ranks):
            p = PRO_PIPE.patch_key(patches[i])
            b = None if base_ranks is None else base_ranks[i]
            self._bump(self.by_patch, p, [r], [b] if b is not None else None)
            self._bump(self.by_year, p.split(".")[0] if p else "?", [r],
                       [b] if b is not None else None)

    def report(self):
        def table(d):
            out = {}
            for key in sorted(d):
                e = d[key]
                n = max(1, e["n"])
                row = {"n": e["n"]}
                for k in self.ks:
                    row["top%d" % k] = e["hits"][k] / n
                    if e["base_hits"][k] or e["n"]:
                        row["base_top%d" % k] = e["base_hits"][k] / n
                if row.get("base_top5"):
                    row["rel_top5"] = EP.baseline_relative(
                        row["top5"], row["base_top5"])
                out[key] = row
            return out
        return {"by_patch": table(self.by_patch), "by_year": table(self.by_year)}
