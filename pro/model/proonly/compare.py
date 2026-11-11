"""Paired comparison of two pro-only checkpoints on identical evaluation rows.

    python pro/model/proonly/compare.py --task masked --a ckpt/run_A/best.pt \
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

`--split shared` exists because the 2026-11-04 refresh changed the split, and
two checkpoints trained under different splits do not have a common val set any
more. `shared` is the rows that are val in *both* protocols - val for the old
`--val-scope all` split and val for the refresh's newest-patch-only split - so
neither model trained on them. On this corpus that is 27 games, all patch 16.20.
`--split latest-val` is the refresh's whole val (52 games); use it only when
both models are refresh-era, because the old split trained on 25 of those.

`--mask-mode slot` hides each picked slot in turn (10 queries per game) instead
of the training objective's 15% random masking (~1.5 per game). It is the
protocol `pro/model/test_split/eval_test_split.py` uses, so its numbers are
comparable with STATUS §7; at 27 games the random mode yields ~40 masked slots,
which cannot rank two models.
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
        n_patches=cfg.get("n_patches", 0), with_value=False,
        # Read from the checkpoint, never defaulted: a ban-aware checkpoint
        # rebuilt without its flags has modules the state dict cannot fill,
        # and the missing-key check below would only catch the ones it
        # happens to name.
        ban_side=bool(cfg.get("ban_side", False)),
        ban_fearless=bool(cfg.get("ban_fearless", False)),
        counter_rank=int(cfg.get("counter_rank", 0) or 0),
        pool_order=str(cfg.get("pool_order", "absolute")))
    state = payload["model"]
    # A checkpoint from the soloQ pipeline may carry value/econ heads this
    # comparison does not build. Anything else missing is a real failure.
    own = set(model.state_dict())
    keep = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own - set(keep)
               if k.startswith(("encoder.", "policy_head.", "state_embed.",
                                "ban_context.", "counter."))]
    if missing:
        raise RuntimeError("checkpoint lacks policy weights: %s" % missing[:6])
    model.load_state_dict(keep, strict=False)
    model.to(device).eval()
    ds = PRO_DATA.ProNextMoveDataset(
        rows, splits[args.split], num_champs=cfg["num_champs"],
        patch_index=patch_index, prefix_mode="all", swap_aug=False,
        # The pool must be built from the whole corpus, not this split: a
        # val row's fearless pool comes from earlier games of its series,
        # which are train rows.
        fearless_roles=PRO_DATA.build_fearless_roles(rows))
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


def masked_ranks_slot(ckpt, rows, splits, args, device):
    """Hide each picked slot in turn - 10 queries per game, not 15% of them.

    The protocol `pro/model/test_split/eval_test_split.py` uses, so a number from
    here is comparable with the STATUS §7 table. The training objective's 15%
    random masking gives ~1.5 queries per game, which is fine on 4,768 val rows
    and useless on 27.
    """
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    enc = C.build_encoder(cfg["num_champs"], cfg["embed_dim"],
                          cfg["num_intra"], cfg["num_cross"],
                          cfg["num_heads"], cfg.get("dropout", 0.0),
                          cfg.get("n_patches", 0))
    _load_encoder(enc, payload["model"])
    enc.to(device).eval()
    patch_vocab = payload.get("patch_vocab") or cfg.get("patch_vocab") or []
    patch_index = {k: i + 1 for i, k in enumerate(patch_vocab)} or None
    n_slots = PRO_DATA.N_SLOTS
    ranks, games, patches = [], [], []
    for i in splits[args.split]:
        row = rows[i]
        champ = torch.tensor(row["champ"], dtype=torch.long)
        role = torch.tensor(row["role"], dtype=torch.long)
        pid = PRO_DATA.patch_to_id(row.get("patch"), patch_index)
        for slot in range(n_slots):
            cids = champ.clone()
            cids[slot] = PRO_DATA.CHAMP_MASK_ID
            is_m = torch.zeros(1, n_slots, dtype=torch.bool)
            is_m[0, slot] = True
            batch = {"champ_ids": cids.unsqueeze(0).to(device),
                     "role_ids": role.unsqueeze(0).to(device),
                     "pad_mask": torch.zeros(1, n_slots, dtype=torch.bool,
                                             device=device),
                     "is_masked": is_m.to(device),
                     "patch_id": torch.tensor([pid], dtype=torch.long,
                                              device=device)}
            with torch.no_grad():
                lg = enc(batch)["champ_logits"][0, slot]
            ranks.append(int((lg > lg[int(champ[slot])]).sum()))
            games.append(row["gameid"])
            patches.append(row.get("patch"))
    return ranks, games, patches


def _build_splits(rows, args):
    """Both protocols' splits, plus the rows that are honest for both.

    `shared` is the intersection of the protocol under test (`--new-scope`,
    `--val-split`) and the *other* protocol's val (`--other`): val for both, so
    neither model trained on those rows, which is the only property that makes a
    cross-protocol comparison mean anything. `--other old` is the original
    every-patch split with the pinned test held out; `--other all` is the
    every-patch split with no test holdout, which is what a `--val-scope all
    --no-test` run trains against.
    """
    latest = args.new_scope == "latest"
    new = C.PRO_PIPE.split_within_patch(
        rows, val_frac=args.val_frac_latest if latest else args.val_frac,
        test_frac=0.0, min_patch_rows=args.min_patch_rows, pinned_test=(),
        val_patches="latest" if latest else None,
        val_random=args.val_split == "random")
    if args.other == "old":
        other = C.PRO_PIPE.split_within_patch(
            rows, val_frac=args.val_frac, test_frac=args.test_frac,
            min_patch_rows=args.min_patch_rows,
            pinned_test=C.PRO_PIPE.TEST_SERIES)
    else:
        other = C.PRO_PIPE.split_within_patch(
            rows, val_frac=args.val_frac, test_frac=0.0,
            min_patch_rows=args.min_patch_rows, pinned_test=(),
            val_random=args.other_random)
    return {"val": other["val"], "test": other["test"],
            "latest-val": new["val"],
            "shared": sorted(set(other["val"]) & set(new["val"]))}


def mcnemar(a_hits, b_hits):
    """Two-sided McNemar on paired binary outcomes -> (a_only, b_only, chi2, p).

    `a_only` counts examples A got right and B got wrong; `b_only` the reverse.
    The two counts are the whole content of a paired comparison - the p-value
    only says whether their imbalance is larger than coin-flipping would give.
    """
    a_only = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    b_only = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 0.0, 1.0
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    p = math.erfc(math.sqrt(max(0.0, chi2) / 2.0))
    return a_only, b_only, chi2, p


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
    ap.add_argument("--split", default="test",
                    choices=("val", "test", "shared", "latest-val"),
                    help="val/test = the old every-patch protocol; shared = "
                         "rows that are val in BOTH protocols (the only "
                         "cross-protocol set neither model trained on); "
                         "latest-val = the protocol under test's own val, i.e. "
                         "whatever --new-scope/--val-split describe")
    ap.add_argument("--corpus", choices=sorted(C.CORPORA), default="2022")
    ap.add_argument("--new-scope", choices=("latest", "all"), default="latest",
                    help="the protocol under test: latest = val only from the "
                         "newest patch (the 2026-11-04 16.20-only arm); all = "
                         "every patch contributes val (the shipped protocol)")
    ap.add_argument("--val-split", choices=("tail", "random"), default="tail",
                    help="the protocol under test: tail = each patch's newest "
                         "series are val; random = a seeded per-patch draw. "
                         "Must match the --val-split the checkpoint trained "
                         "with, or `--split latest-val` scores the wrong rows")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="val share per patch, for --new-scope all and for the "
                         "old split")
    ap.add_argument("--val-frac-latest", type=float, default=0.20,
                    help="--new-scope latest only: that arm's val share, which "
                         "must match the --val-frac it trained with")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--min-patch-rows", type=int, default=40)
    ap.add_argument("--other", choices=("old", "all"), default="old",
                    help="which protocol --split shared intersects with: old = "
                         "every-patch val + pinned test (the original model), "
                         "all = every-patch val, no test (a --val-scope all "
                         "--no-test run)")
    ap.add_argument("--other-random", action="store_true",
                    help="--other all only: that run drew its val series at "
                         "random per patch (--val-split random) instead of "
                         "taking each patch's tail")
    ap.add_argument("--mask-mode", choices=("random", "slot"), default="random",
                    help="masked task only: random = the training objective's "
                         "15%% masking; slot = hide each pick in turn, 10 per "
                         "game (needed below ~100 games)")
    ap.add_argument("--half-life", type=float, default=0.0)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="",
                    help="write the comparison to this path (the kept evidence "
                         "lives in pro/model/proonly/reports/, the weights do not)")
    args = ap.parse_args()
    device = torch.device(args.device)
    rows, _, _, _ = C.load_split(args, verbose=False)
    splits = _build_splits(rows, args)
    if not splits[args.split]:
        raise SystemExit("split %r is empty" % args.split)

    if args.task == "masked" and args.mask_mode == "slot":
        fn = masked_ranks_slot
    elif args.task == "masked":
        fn = masked_ranks
    else:
        fn = policy_ranks
    ra, games, patches = fn(args.a, rows, splits, args, device)
    rb, games_b, _ = fn(args.b, rows, splits, args, device)
    assert games == games_b, "the two models must score the same examples"
    assert len(ra) == len(rb)

    print("%s, split=%s (%s | other=%s%s), n=%d examples, %d games"
          % (args.task, args.split,
             "%s/%s" % (args.new_scope, args.val_split),
             args.other, "+random" if args.other_random else "",
             len(ra), len(set(games))))
    out = {}
    for k in (1, 5, 10):
        ha = [r < k for r in ra]
        hb = [r < k for r in rb]
        a_only, b_only, chi2, p = mcnemar(ha, hb)
        acc_a, acc_b = sum(ha) / len(ha), sum(hb) / len(hb)
        lo_a, hi_a = EP.wilson(sum(ha), len(ha))
        lo_b, hi_b = EP.wilson(sum(hb), len(hb))
        out["top%d" % k] = {"a": acc_a, "b": acc_b, "delta": acc_b - acc_a,
                            "a_ci": [lo_a, hi_a], "b_ci": [lo_b, hi_b],
                            "a_only": a_only, "b_only": b_only, "p": p}
        print("  top%-2d  A %.4f [%.2f,%.2f]  B %.4f [%.2f,%.2f]  "
              "delta %+.4f  (A only %d / B only %d, McNemar p=%.3g)"
              % (k, acc_a, lo_a, hi_a, acc_b, lo_b, hi_b, acc_b - acc_a,
                 a_only, b_only, p))
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
                       protocol={"new_scope": args.new_scope,
                                 "val_split": args.val_split,
                                 "other": args.other,
                                 "other_random": bool(args.other_random),
                                 "val_frac": args.val_frac,
                                 "val_frac_latest": args.val_frac_latest,
                                 "mask_mode": args.mask_mode},
                       n=len(ra), n_games=len(set(games)))
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
