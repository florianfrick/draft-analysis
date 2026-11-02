"""Paired comparison of two pro-only checkpoints on identical evaluation rows.

    python model/proonly/compare.py --task masked --a ckpt/run_A/best.pt \
        --b ckpt/run_B/best.pt

Why this exists rather than comparing two `test_report.json` files: the two
reports are two independent percentages, and a 0.4pp difference between them
is indistinguishable from seed noise at this data size (the P3.1 attribute
ablation measured exactly that - +0.26pp between arms against 0.28-0.44pp of
seed-to-seed spread). Here both models are scored on the *same* evaluation
rows - the masked task uses one fixed mask seed, the policy uses every prefix
of every test game - so the comparison is paired and the test is McNemar on
the discordant pairs.

Two tests are reported, because they answer different questions:

* McNemar over (slot, example) pairs - the standard paired test, and the one
  the repo used for the attribute adapter. It treats slots as independent,
  which they are not: 10 slots of one draft share a context.
* A game-level bootstrap of the top-5 difference - resamples whole games, so
  the interval reflects the correlation inside a draft. If the McNemar p is
  small and this interval straddles zero, the effect is carried by a handful
  of drafts, not by the model.
"""
import argparse
import importlib.util
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

PRO_DATA, EP = C.PRO_DATA, C.EP


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STAGE_PREFIXES = ("heads.", "policy_head.", "value_head.", "econ_head.",
                  "ban_context.", "state_embed.")


def _load_encoder(enc, state):
    """Load an encoder out of a checkpoint that may carry extra stage heads.

    `compare.py` is also used to score the *existing* pipeline on this split
    (finetune `best_mask.pt` carries `heads.*`, the policy carries
    `policy_head.*`/`ban_context.*`). Those keys belong to the stage, not to
    the encoder, so they are stripped first - and then `load_encoder_state`
    decides, with its usual asymmetry: a checkpoint with a trained attribute
    adapter or patch table that the encoder cannot take still raises, because
    dropping it would score a different model than the one that was trained.
    """
    stage = {k: v for k, v in state.items() if k.startswith(STAGE_PREFIXES)}
    enc_state = {k: v for k, v in state.items() if k not in stage}
    # finetune checkpoints nest the encoder under `encoder.` (ProDraftModel
    # owns it); the policy and this branch's masked runs do not.
    if any(k.startswith("encoder.") for k in enc_state):
        enc_state = {k[len("encoder."):]: v for k, v in enc_state.items()
                     if k.startswith("encoder.")}
    C.load_encoder_state(enc, enc_state)
    return enc


def masked_ranks(ckpt, rows, splits, args, device):
    """Per-masked-slot rank of the truth, on the fixed eval view."""
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    enc = C.build_encoder(cfg["num_champs"], cfg["embed_dim"],
                          cfg["num_intra"], cfg["num_cross"],
                          cfg["num_heads"], cfg.get("dropout", 0.0),
                          cfg.get("n_patches", 0))
    _load_encoder(enc, payload["model"])
    enc.to(device).eval()
    patch_vocab = payload.get("patch_vocab") or cfg.get("patch_vocab") or []
    ds = PRO_DATA.ProMaskedDraftDataset(
        rows, splits[args.split], num_champs=cfg["num_champs"],
        mask_prob=0.15, role_keep_prob=0.1, seed=C.EVAL_SEED, epoch=0,
        econ_mask_prob=0.0,
        patch_index={k: i + 1 for i, k in enumerate(patch_vocab)} or None)
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        collate_fn=PRO_DATA.pro_collate)
    ranks, games, patches = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            out = enc(batch)
            m = batch["is_masked"]
            cl, ct = out["champ_logits"][m], batch["champ_labels"][m]
            truth = cl.gather(1, ct[:, None]).squeeze(1)
            ranks.extend((cl > truth[:, None]).sum(1).tolist())
            di = m.nonzero(as_tuple=True)[0]
            metas = batch["meta"]
            games.extend(metas[d].get("gameid") for d in di.tolist())
            patches.extend(metas[d].get("patch") for d in di.tolist())
    return ranks, games, patches


def policy_ranks(ckpt, rows, splits, args, device):
    """Per-next-move rank of the truth, over every prefix of the split."""
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    patch_vocab = (payload.get("patch_vocab") or cfg.get("patch_vocab")
                   or payload.get("config", {}).get("patch_vocab") or [])
    patch_index = {k: i + 1 for i, k in enumerate(patch_vocab)}
    enc = C.build_encoder(cfg["num_champs"], cfg["embed_dim"],
                          cfg["num_intra"], cfg["num_cross"],
                          cfg["num_heads"], cfg.get("dropout", 0.0),
                          cfg.get("n_patches", 0))
    model = C.PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg.get("state_dim", 64),
        n_econ=0, dropout=cfg.get("dropout", 0.0),
        n_patches=cfg.get("n_patches", 0), with_value=False)
    state = payload["model"]
    # A checkpoint from the soloQ pipeline may carry value/econ heads this
    # comparison does not build. Anything else missing is a real failure.
    own = set(model.state_dict())
    keep = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own - set(keep)
               if k.startswith(("encoder.", "policy_head.", "state_embed."))]
    if missing:
        raise RuntimeError("checkpoint lacks policy weights: %s" % missing[:6])
    model.load_state_dict(keep, strict=False)
    model.to(device).eval()
    ds = PRO_DATA.ProNextMoveDataset(
        rows, splits[args.split], num_champs=cfg["num_champs"],
        patch_index=patch_index, prefix_mode="all", swap_aug=False)
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        collate_fn=PRO_DATA.pro_collate)
    ranks, games, patches = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            cl = model(batch)["policy_logits"]
            tc = batch["target_champ"]
            truth = cl.gather(1, tc[:, None]).squeeze(1)
            ranks.extend((cl > truth[:, None]).sum(1).tolist())
            games.extend(m.get("gameid") for m in batch["meta"])
            patches.extend(m.get("patch") for m in batch["meta"])
    return ranks, games, patches


def mcnemar(a_hits, b_hits):
    """Two-sided McNemar on paired binary outcomes -> (b, c, chi2, p)."""
    b = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    c = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(max(0.0, chi2) / 2.0))
    return b, c, chi2, p


def check_same_patch_vocab(a, b):
    """Both checkpoints must index the patch table the same way.

    Each model scores a row's patch as a *position* in its own vocabulary.
    If the two checkpoints were trained on different vocabularies - or on the
    same patches in a different order - then "patch 12.01" is a different
    vector in A than in B, and the comparison would be measuring a vocabulary
    shift rather than the difference between the models. The rows would
    still match, so nothing would raise; the number would just be wrong.

    Comparing the two lists checks the order and the length at once, and the
    length is what the patch table's shape is.
    """
    va = torch.load(str(a), map_location="cpu", weights_only=False)
    vb = torch.load(str(b), map_location="cpu", weights_only=False)
    ta = list(va.get("patch_vocab") or [])
    tb = list(vb.get("patch_vocab") or [])
    if ta != tb:
        raise RuntimeError(
            "the two checkpoints use different patch vocabularies (%d vs %d "
            "rows); patch ids are positions in that vocabulary, so the "
            "comparison would not be paired" % (len(ta), len(tb)))
    return ta


def game_bootstrap(ra, rb, games, k=5, n_boot=2000, seed=0):
    """Top-k difference (B - A) resampling whole games."""
    by_game = {}
    for i, g in enumerate(games):
        e = by_game.setdefault(g, [0, 0, 0])
        e[0] += 1
        e[1] += int(ra[i] < k)
        e[2] += int(rb[i] < k)
    items = list(by_game.values())
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        n = na = nb = 0
        for _ in range(len(items)):
            e = items[rng.randrange(len(items))]
            n += e[0]
            na += e[1]
            nb += e[2]
        diffs.append((nb - na) / max(1, n))
    diffs.sort()
    obs_a = sum(e[1] for e in items) / max(1, sum(e[0] for e in items))
    obs_b = sum(e[2] for e in items) / max(1, sum(e[0] for e in items))
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    return obs_b - obs_a, lo, hi, len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=("masked", "policy"), required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--corpus", choices=sorted(C.CORPORA), default="2022")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--min-patch-rows", type=int, default=40)
    ap.add_argument("--half-life", type=float, default=0.0)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="",
                    help="write the comparison to this path (the kept evidence "
                         "lives in model/proonly/reports/, the weights do not)")
    args = ap.parse_args()
    device = torch.device(args.device)
    rows, splits, _, _ = C.load_split(args, verbose=False)
    check_same_patch_vocab(args.a, args.b)

    fn = masked_ranks if args.task == "masked" else policy_ranks
    ra, games, patches = fn(args.a, rows, splits, args, device)
    rb, games_b, _ = fn(args.b, rows, splits, args, device)
    assert games == games_b, "the two models must score the same examples"
    assert len(ra) == len(rb)

    print("%s, split=%s, n=%d examples, %d games"
          % (args.task, args.split, len(ra), len(set(games))))
    out = {}
    for k in (1, 5, 10):
        ha = [r < k for r in ra]
        hb = [r < k for r in rb]
        b, c, chi2, p = mcnemar(ha, hb)
        acc_a, acc_b = sum(ha) / len(ha), sum(hb) / len(hb)
        out["top%d" % k] = {"a": acc_a, "b": acc_b, "delta": acc_b - acc_a,
                            "b_only": b, "a_only": c, "p": p}
        print("  top%-2d  A %.4f  B %.4f  delta %+.4f  (B only %d / A only "
              "%d, McNemar p=%.3g)" % (k, acc_a, acc_b, acc_b - acc_a, b, c, p))
    for k in (1, 5):
        d, lo, hi, n_games = game_bootstrap(ra, rb, games, k=k)
        out["bootstrap_top%d" % k] = {"delta": d, "lo": lo, "hi": hi,
                                      "n_games": n_games}
        print("  top%d game-bootstrap delta %+.4f  95%% CI [%+.4f, %+.4f] over "
              "%d games" % (k, d, lo, hi, n_games))
    # Per year, because the two models may be trained under different recency
    # weightings: a single pooled number would then be a statement about the
    # window's year mix rather than about the models.
    years = sorted({C.PRO_PIPE.patch_key(p).split(".")[0] for p in patches})
    print("  top5 by year:  year      A      B    delta      n")
    out["by_year"] = {}
    for y in years:
        idx = [i for i, p in enumerate(patches)
               if C.PRO_PIPE.patch_key(p).split(".")[0] == y]
        if not idx:
            continue
        a5 = sum(1 for i in idx if ra[i] < 5) / len(idx)
        b5 = sum(1 for i in idx if rb[i] < 5) / len(idx)
        out["by_year"][y] = {"a": a5, "b": b5, "delta": b5 - a5,
                             "n": len(idx)}
        print("                %-5s %.4f %.4f %+.4f %6d"
              % (y, a5, b5, b5 - a5, len(idx)))
    print("\nA = %s\nB = %s" % (args.a, args.b))
    if args.json:
        import json
        payload = dict(out, task=args.task, split=args.split,
                       corpus=args.corpus, a=args.a, b=args.b,
                       n=len(ra), n_games=len(set(games)))
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
