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
import json
import re
import sys
import unicodedata
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent            # .../pro/model/proonly
MODEL_DIR = HERE.parent                           # .../model
ROOT = MODEL_DIR.parents[1]                           # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"
SOLOQ_PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"
TEAM_CODES_PATH = PRO_DIR / "team_codes.json"

sys.path.insert(0, str(ROOT))
from solo_queue.model.pretraining.base_model import SiameseBase, load_encoder_state  # noqa: E402

# Fixed for every run and every arm. The masked eval masks must not depend on
# the training seed, otherwise the two arms score different slots.
EVAL_SEED = 20261101

CORPORA = {"2022": "pro_drafts.pt", "all": "pro_drafts_all.pt",
           # 2022+ exports with the corpus freeze REMOVED: every game in the
           # CSVs, including whatever the current-season export has appended
           # since CORPUS_MAX_DATE. Its own cache file, because the freeze is
           # part of what the rows mean - reusing pro_drafts.pt would either
           # serve frozen rows or silently overwrite the frozen corpus that
           # every recorded number in STATUS.md is measured on.
           "2022-full": "pro_drafts_full.pt"}

# Which corpora are frozen at `pipeline.CORPUS_MAX_DATE`. Only the unfrozen
# ones pass `max_date=""` down to the scan.
UNFROZEN_CORPORA = {"2022-full"}

# 2022 = the documented corpus; 2022-full = the same globs, no freeze; all =
# 2017-2026 (the wider exports), still frozen.
_CORPUS_GLOB = {"2022": "CSV_GLOB", "2022-full": "CSV_GLOB",
                "all": "CSV_GLOB_ALL"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")
EP = _load("eval_protocol", PRO_DIR / "eval_protocol.py")
# The policy class lives in the solo-queue track; this experiment wraps it
# rather than owning it (same reason the base encoder is imported, not copied).
POLICY_MODEL = _load("policy_model",
                     ROOT / "solo_queue" / "model" / "policy" / "model.py")

PolicyValueModel = POLICY_MODEL.PolicyValueModel


def add_corpus_args(ap):
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="2022",
                    help="2022 = the documented 46,879-draft corpus, frozen "
                         "at CORPUS_MAX_DATE (default); all = 2017-2026, "
                         "81,161 drafts; 2022-full = the 2022+ exports with "
                         "the freeze removed (every game in the CSVs, its own "
                         "cache - a different corpus, not a comparable one)")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="share of each patch held out for val (the tail, or a "
                         "random draw of the same size - see --val-split)")
    ap.add_argument("--test-frac", type=float, default=0.10,
                    help="share of each patch's tail held out for test. "
                         "IGNORED while the pinned test split is active "
                         "(pipeline.TEST_SERIES); it only applies to "
                         "--limit-drafts smoke runs, which cannot hold the "
                         "pinned series out")
    ap.add_argument("--no-test", dest="no_test", action="store_true",
                    help="hold no test split out at all: what would have been "
                         "the test tail goes to val (or to train, see "
                         "--val-scope) and the pinned TEST_SERIES are ordinary "
                         "rows. For a production refresh, not a measurement "
                         "run - the resulting checkpoint has no held-out score")
    ap.add_argument("--no-pin", dest="no_pin", action="store_true",
                    help="drop the pinned TEST_SERIES but KEEP a fractional "
                         "test tail (--test-frac). Unlike --no-test this still "
                         "holds out an out-of-sample set, it just derives it "
                         "from the fractions instead of from fixed series ids, "
                         "so the test split moves when the corpus is rebuilt")
    ap.add_argument("--val-scope", choices=("all", "latest"), default="all",
                    help="all (default) = every patch contributes val, the "
                         "protocol every recorded number uses. latest = only "
                         "the newest patch contributes val and every other "
                         "patch is 100%% train, for a model that will only be "
                         "asked about the current patch")
    ap.add_argument("--val-split", choices=("tail", "random"), default="tail",
                    help="tail (default) = val is each patch's newest "
                         "series, so the frontier of every patch is never "
                         "trained on. random = a seeded random draw of whole "
                         "series per patch, so 90%% of every patch's timeline "
                         "- including its end - is trained on and val is "
                         "spread across the patch instead of sitting on it")
    ap.add_argument("--canonical-split", dest="canonical_split",
                    action="store_true",
                    help="force the one split every recorded number is "
                         "measured on: --corpus 2022-full, --val-frac 0.10, "
                         "--test-frac 0.05, --min-patch-rows 40, --no-pin, "
                         "--val-scope all, --val-split tail. Overrides the "
                         "individual flags (and says so in the log), because "
                         "a split assembled by hand is how three incompatible "
                         "protocols accumulated in the first place. "
                         "pro/model/test_split/split.py is the definition; "
                         "the resulting checkpoint scores as 'comparable' in "
                         "that suite only if these are the counts it gets")
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


# The split every recorded number is measured on, mirroring
# `pro/model/test_split/split.py`. Kept as data rather than as defaults so
# `--canonical-split` can *force* it: a default can be overridden by accident,
# and three incompatible protocols is what that produced.
CANONICAL_SPLIT = dict(corpus="2022-full", val_frac=0.10, test_frac=0.05,
                       min_patch_rows=40, val_scope="all", val_split="tail")


def apply_canonical_split(args):
    """Overwrite the split flags with the canonical ones, out loud.

    Refuses the two combinations that would silently change the rows:
    `--no-test` (no held-out set at all) and `--limit-drafts` (a strided
    subsample, so the counts cannot match and the checkpoint is not
    comparable whatever it says).
    """
    if getattr(args, "no_test", False):
        raise SystemExit("--canonical-split holds out a test split; --no-test "
                         "contradicts it")
    if getattr(args, "limit_drafts", 0):
        raise SystemExit("--canonical-split cannot be combined with "
                         "--limit-drafts: the subsample changes the split")
    for key, want in CANONICAL_SPLIT.items():
        got = getattr(args, key, None)
        if got != want:
            print("canonical split: --%s %r -> %r"
                  % (key.replace("_", "-"), got, want))
            setattr(args, key, want)
    if hasattr(args, "no_pin") and not args.no_pin:
        print("canonical split: --no-pin on (the test split is the fractional "
              "tail, not pinned series)")
        args.no_pin = True
    return args


def load_split(args, verbose=True):
    """rows + within-patch split + patch vocabulary, in one place."""
    if getattr(args, "canonical_split", False):
        apply_canonical_split(args)
    csv_glob = getattr(PRO_PIPE, _CORPUS_GLOB[args.corpus])
    unfrozen = args.corpus in UNFROZEN_CORPORA
    rows, stats = PRO_PIPE.build_rows(
        csv_glob=csv_glob, cache_path=PRO_DIR / CORPORA[args.corpus],
        max_date="" if unfrozen else None)
    if args.limit_drafts:
        stride = max(1, len(rows) // args.limit_drafts)
        rows = rows[::stride][:args.limit_drafts]
        print("limited run: %d drafts (stride %d)" % (len(rows), stride))
    PRO_PIPE.assign_weights(rows, args.half_life)
    # A smoke run subsamples the corpus, so the pinned test series are
    # usually gone; relax the pin rather than abort, and fall back to the
    # fractional tail. Never true for a measured run.
    no_test = bool(getattr(args, "no_test", False))
    no_pin = bool(getattr(args, "no_pin", False))
    val_random = getattr(args, "val_split", "tail") == "random"
    pinned = () if (no_test or no_pin) else PRO_PIPE.TEST_SERIES
    splits = PRO_PIPE.split_within_patch(
        rows, val_frac=args.val_frac,
        test_frac=0.0 if no_test else args.test_frac,
        min_patch_rows=args.min_patch_rows,
        pinned_test=pinned,
        strict_pin=not args.limit_drafts, verbose=verbose,
        val_patches=("latest" if getattr(args, "val_scope", "all") == "latest"
                     else None),
        val_random=val_random)
    patch_vocab, patch_index = PRO_DATA.build_patch_vocab(rows)
    if verbose:
        print("corpus %s: %d drafts, %d patches (%s..%s), freeze %s"
              % (args.corpus, len(rows), len(patch_vocab),
                 patch_vocab[0], patch_vocab[-1],
                 "NONE" if unfrozen else PRO_PIPE.CORPUS_MAX_DATE))
        print("split: train %d / val %d / test %d"
              % (len(splits["train"]), len(splits["val"]),
                 len(splits["test"])))
        print("data ends at: %s (newest series in train: %s)"
              % (corpus_frontier(rows)["series_id"],
                 newest_train_series(rows, splits["train"])["series_id"]))
    return rows, splits, patch_vocab, patch_index


def split_description(val_split, has_test, pinned=True):
    """One line naming the split, for the report's own `protocol` block.

    A checkpoint records which rows selected it, because `best.pt` is chosen on
    val and two runs that differ only in `--val-split` are different
    experiments - the same reason `--epochs` is written down.
    """
    train_how, val_how = (("head", "tail") if val_split == "tail"
                          else ("the rest", "random draw per patch"))
    if not has_test:
        test_how = "no test split (--no-test)"
    elif pinned:
        test_how = "test = the pinned TEST_SERIES"
    else:
        test_how = "test = a fractional tail (--no-pin, no fixed holdout)"
    return "within-patch (train=%s, val=%s); %s" % (train_how, val_how,
                                                    test_how)


# ---------------------------------------------------------------- naming ----
# A checkpoint directory states which inference it serves and which stage
# produced it; the `-through-<league>-<a>-<b>` suffix states which series its
# data ends on, so two runs of the same stage are distinguishable at a glance
# and a stale artefact cannot pass for a fresh one.

def _ascii(text):
    """Best-effort ASCII for a directory name ('LØS' -> 'LOS', not 'LS')."""
    s = str(text or "")
    for src, dst in (("Ø", "O"), ("ø", "o"), ("Đ", "D"), ("đ", "d"),
                     ("Ł", "L"), ("ł", "l"), ("ß", "ss")):
        s = s.replace(src, dst)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _slug(text):
    """Lowercase, non-alphanumeric -> '-', collapsed, trimmed."""
    s = re.sub(r"[^a-z0-9]+", "-", _ascii(text).replace("|", "-").lower())
    return s.strip("-") or "x"


def _initialism(name):
    """Deterministic fallback for a team missing from team_codes.json.

    Not the org's tag - 'Top Esports' gives 'TE', not 'TES' - but stable, and
    a wrong tag invented here would be worse than an honest initialism.
    """
    toks = [t for t in re.split(r"[\s._-]+", _ascii(name)) if t]
    if not toks:
        return "TBD"
    if len(toks) == 1:
        # Keep '|' - a team called 'JD|XL' exists in the NLC exports, and
        # dropping the separator would make the name read 'jdxl'. _slug turns
        # it into '-'.
        return re.sub(r"[^A-Za-z0-9|]", "", toks[0]).upper()[:6] or "TBD"
    return "".join(t[0] for t in toks).upper()[:5]


def load_team_codes(path=TEAM_CODES_PATH):
    """Flatten the grouped JSON into {teamname: CODE}.

    Keys starting with '_' are comments and are skipped, so the file can
    explain itself. A team listed twice with different codes raises: silently
    taking the last one would make a name depend on dict order.
    """
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}

    def walk(node, group):
        for k, v in node.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                walk(v, k)
                continue
            prev = out.get(k)
            if prev is not None and prev != v:
                raise ValueError(
                    "team_codes.json: %r is both %r and %r (in %r)"
                    % (k, prev, v, group))
            out[k] = v

    walk(payload, "?")
    return out


_TEAM_CODES = None


def team_code(name):
    global _TEAM_CODES
    if _TEAM_CODES is None:
        _TEAM_CODES = load_team_codes()
    return _TEAM_CODES.get(name) or _initialism(name)


def corpus_frontier(rows, indices=None):
    """The newest series in the corpus, or in `indices` if given.

    Returned as a dict rather than a series_id string because `series_id` is
    built by joining team names with '|' and a team called 'JD|XL' exists in
    the NLC exports - so the id cannot be reliably split back into its parts.
    The row already carries league and both team names, which is what the name
    needs.
    """
    pool = range(len(rows)) if indices is None else list(indices)
    if not pool:
        raise ValueError("corpus_frontier: no rows")
    best = max(pool, key=lambda i: (rows[i]["date_ordinal"],
                                    rows[i].get("gameid") or ""))
    row = rows[best]
    sid = row.get("series_id")
    return {
        "series_id": sid,
        "league": row.get("league") or "?",
        "teams": tuple(sorted([row.get("blue_team") or "?",
                               row.get("red_team") or "?"])),
        "date": row.get("date") or "",
        "patch": PRO_PIPE.patch_key(row.get("patch")),
        "gameid": row.get("gameid") or "",
        "n_games": sum(1 for i in pool if rows[i].get("series_id") == sid),
    }


def newest_train_series(rows, train_idx):
    """The newest series that actually receives gradient updates.

    Different from `corpus_frontier` whenever val is non-empty: val is every
    patch's tail, so the newest games in the corpus are usually val rows. Both
    are recorded - the directory name states the data frontier, the checkpoint
    states where training stops.
    """
    return corpus_frontier(rows, train_idx)


def through_suffix(frontier):
    """'through-wlds-ig-tes' from a corpus_frontier dict."""
    return "through-%s-%s-%s" % (
        _slug(frontier["league"]),
        _slug(team_code(frontier["teams"][0])),
        _slug(team_code(frontier["teams"][1])))


def check_out_dir_name(out_dir, suffix, suggested):
    """Refuse a checkpoint directory whose name misstates the data frontier.

    The name is the only thing a later reader has; a retrain that lands in
    `masking-proonly` while the data moved is indistinguishable from one that
    did not, which is exactly the confusion the suffix exists to prevent.
    """
    if not out_dir:
        return
    if Path(out_dir).name.endswith(suffix):
        return
    raise SystemExit(
        "--out-dir must end with %r (the newest series in this corpus).\n"
        "  got:       %s\n  suggested: %s/%s"
        % (suffix, out_dir, str(Path(out_dir).parent).replace("\\", "/"),
           Path(out_dir).name + suffix))


def build_encoder(num_champs, embed_dim, num_intra, num_cross, num_heads,
                  dropout, n_patches, with_value=False):
    return SiameseBase(
        num_champs=num_champs, embed_dim=embed_dim, num_heads=num_heads,
        num_intra_layers=num_intra, num_cross_layers=num_cross,
        dropout=dropout, with_value=with_value, n_patches=n_patches)


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
