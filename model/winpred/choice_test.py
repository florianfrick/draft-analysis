"""Does the win model rank the champion a pro actually picked above the rest?

This is the question the win-prediction experiment never answered. Everything
in §7-§17 measures "who wins a finished 10-pick draft", where the model gets
53-54 % of games right against a 52 % constant. A draft *suggestion* needs a
different thing: given a board and a slot, is the champion the team chose
better than the ones it did not choose? That question is not the same, and a
model can be useless at one and useful at the other - the per-game error is
dominated by things that are identical across the two candidates (who is
playing, form, patch), so a comparison inside one board cancels them.

The test
--------
For every pick in every test game, score the real board, then score the same
board with that one slot swapped for each of `k` alternative champions that are
actually played in that role. Report

  * **pct** - the fraction of alternatives the chosen champion outranks. 0.50
    is "knows nothing"; 1.00 is "always right". Ties count 1/2.
  * **top1** - how often the chosen champion is the best of the k+1.
  * **delta** - how many probability points the choice is worth, i.e. the size
    of the effect a re-ranker would have to work with.

What it does *not* measure: correctness. The reference is what a pro team did,
and pro teams are sometimes wrong. A high `pct` means the model agrees with
professional decision-making - which is exactly what a suggestion model should
do, since it is trained to imitate them - not that the pick won the game.

Two controls, because a number without them is not a measurement
---------------------------------------------------------------
  * **null** - the same pipeline with a *random* champion in the "chosen"
    position. By construction it must come out at 0.50; anything else is a bug
    in the harness (an off-by-one in the candidate set, a side that is not
    flipped). Run it every time.
  * **tier list** - the champion x role Bradley-Terry prior on its own, with no
    encoder (§16). If the encoder cannot beat a linear tier list at this task
    either, that is the same conclusion as §16 from a new direction.

Sides: the model emits P(blue). A blue pick is scored with P(blue); a red pick
with 1 - P(blue), so "higher" always means "better for the side that picked".

    python model/winpred/choice_test.py --limit-games 50        # smoke
    python model/winpred/choice_test.py --out reports/choice.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent           # .../model/winpred
ROOT = HERE.parents[1]                           # .../lol draft analysis
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.winpred import data as D              # noqa: E402
from model.winpred.models import siamese_model as W          # noqa: E402
from model.winpred.evaluate import patch_remap   # noqa: E402

TEAM = D.TEAM_SIZE                                # 5
SLOTS = D.N_SLOTS                                 # 10


# ------------------------------------------------------------------ helpers
def role_pools(table, train_idx, min_games=20):
    """-> list of 5 arrays: champions played >= `min_games` times in that role.

    Built from TRAIN rows only. It only defines the candidate set - no label is
    read - so it cannot leak, but it still belongs on the train side of the
    split because a champion that appears in the test patch only would make the
    alternatives unrealistically exotic.
    """
    idx = np.asarray(train_idx, dtype=np.int64)
    role = table.role[idx].ravel().astype(np.int64)
    champ = table.champ[idx].ravel().astype(np.int64)
    n_champs = int(champ.max()) + 1
    counts = np.zeros((5, n_champs), dtype=np.int64)
    np.add.at(counts, (role, champ), 1)
    return [np.flatnonzero(counts[r] >= min_games) for r in range(5)]


def side_scores(p_blue, slot):
    """A pick's score from the perspective of the side that made it."""
    return p_blue if slot < TEAM else 1.0 - p_blue


def pct_of_first(scores):
    """scores[:, 0] is the chosen champion -> fraction of the rest it beats."""
    s0 = scores[:, :1]
    alt = scores[:, 1:]
    return ((alt < s0).sum(1) + 0.5 * (alt == s0).sum(1)) / alt.shape[1]


def top1_of_first(scores):
    s0 = scores[:, :1]
    # keepdims keeps this [N,1]: without it `max(1)` collapses to [N] and the
    # broadcast compares every row against a different row's max.
    return (s0 >= scores.max(1, keepdims=True)).ravel()


@torch.no_grad()
def score_boards(model, champ, role, patch, batch=4096, device="cpu"):
    """[M,10] champ ids + [M,10] roles + [M] patch ids -> P(blue) [M]."""
    model.eval()
    out = []
    for i in range(0, champ.shape[0], batch):
        b = {
            "champ_ids": torch.as_tensor(champ[i:i + batch], dtype=torch.long,
                                         device=device),
            "role_ids": torch.as_tensor(role[i:i + batch], dtype=torch.long,
                                        device=device),
            "patch_id": torch.as_tensor(patch[i:i + batch], dtype=torch.long,
                                        device=device),
        }
        out.append(torch.sigmoid(model(b)["logit"]).float().cpu().numpy())
    return np.concatenate(out)


def build_candidates(row_champ, role, pool, k, rng, chosen=None):
    """-> [10, 1+k] candidate champion ids, column 0 = the one under test.

    `chosen=None` draws all k+1 at random, which is the null control: every
    candidate is exchangeable, so the first one must score at percentile 0.50.
    """
    size = max([int(row_champ.max())] + [int(p.max()) for p in pool]) + 1
    taken = np.zeros(size, dtype=bool)
    taken[row_champ] = True
    cand = np.zeros((SLOTS, k + 1), dtype=np.int64)
    for slot in range(SLOTS):
        avail = pool[int(role[slot])]
        avail = avail[~taken[avail]]
        if avail.size < k + 1:
            raise SystemExit("role %d pool has %d available champions, need "
                             "%d; lower --k" % (role[slot], avail.size, k + 1))
        if chosen is None:
            draw = rng.choice(avail, size=k + 1, replace=False)
            cand[slot, 0] = draw[0]
            cand[slot, 1:] = draw[1:]
        else:
            cand[slot, 0] = chosen[slot]
            cand[slot, 1:] = rng.choice(avail, size=k, replace=False)
    return cand


def boards_for_row(row_champ, cand):
    """One board per (slot, candidate) -> [10*(1+k), 10] champion ids."""
    k1 = cand.shape[1]
    n = SLOTS * k1
    boards = np.repeat(row_champ[None, :], n, axis=0)
    slot_of = np.repeat(np.arange(SLOTS), k1)
    cand_of = np.tile(np.arange(k1), SLOTS)
    boards[np.arange(n), slot_of] = cand[slot_of, cand_of]
    return boards


def run(model, table, rows, pools, k, rng, patch_ids, device, chosen=None,
        chunk=100):
    """Score every (game, slot, candidate). -> dict of per-pick arrays."""
    pcts, tops, deltas, games, slots = [], [], [], [], []
    for start in range(0, len(rows), chunk):
        block = rows[start:start + chunk]
        boards, roles, patches = [], [], []
        for r in block:
            cand = build_candidates(table.champ[r], table.role[r],
                                    pools, k, rng, chosen=chosen)
            b = boards_for_row(table.champ[r], cand)
            boards.append(b)
            roles.append(np.repeat(table.role[r][None, :], b.shape[0], axis=0))
            patches.append(np.full(b.shape[0], patch_ids[r], dtype=np.int64))
        boards = np.concatenate(boards)
        roles = np.concatenate(roles)
        patches = np.concatenate(patches)
        p = score_boards(model, boards, roles, patches, device=device)
        p = p.reshape(len(block), SLOTS, k + 1)
        # flip the red side so "higher" always means "better for the picker"
        side = np.ones((SLOTS, k + 1), dtype=np.float32)
        side[TEAM:, :] = -1.0
        p = np.where(side[None, :, :] > 0, p, 1.0 - p)
        for j, r in enumerate(block):
            pcts.append(pct_of_first(p[j]))
            tops.append(top1_of_first(p[j]))
            deltas.append(p[j, :, 0] - p[j, :, 1:].mean(1))
            games.extend([table.series_id[r]] * SLOTS)
            slots.extend(range(SLOTS))
    return {"pct": np.concatenate(pcts), "top1": np.concatenate(tops),
            "delta": np.concatenate(deltas), "cluster": np.array(games),
            "slot": np.array(slots)}


def bt_scores(table, pools, rows, rating, k, rng, chosen=None):
    """The tier list on its own: mean(r) blue - mean(r) red, no encoder."""
    pcts, tops, deltas = [], [], []
    r2 = rating.reshape(-1, 5)
    for r in rows:
        champ, role = table.champ[r], table.role[r]
        cand = build_candidates(champ, role, pools, k, rng, chosen=chosen)
        vals = np.repeat(r2[champ, role][:, None], k + 1, axis=1)   # [10,k+1]
        for slot in range(SLOTS):
            vals[slot] = r2[cand[slot], role[slot]]                 # [k+1]
        x = vals[:TEAM].mean(0) - vals[TEAM:].mean(0)               # blue logit
        s = np.stack([x] * TEAM + [-x] * TEAM)                      # picker's view
        pcts.append(pct_of_first(s))
        tops.append(top1_of_first(s))
        deltas.append(s[:, 0] - s[:, 1:].mean(1))
    return {"pct": np.concatenate(pcts), "top1": np.concatenate(tops),
            "delta": np.concatenate(deltas)}


def cluster_boot(values, clusters, n_boot=2000, seed=0):
    """Mean of `values` with a CI from resampling whole series."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=uniq.size)
    cnts = np.bincount(inv, minlength=uniq.size)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        draws[i] = sums[pick].sum() / max(cnts[pick].sum(), 1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(ROOT / "model/checkpoints/"
                    "winpred-pro-ce-d384-seed20261105-through-20261104/best.pt"))
    ap.add_argument("--data", choices=("pro", "soloq"), default="pro")
    ap.add_argument("--k", type=int, default=32,
                    help="alternative champions per pick (resolution 1/(k+1))")
    ap.add_argument("--min-games", type=int, default=20,
                    help="a champion must have this many TRAIN games in a role "
                         "to be a candidate there")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--null-games", type=int, default=1000,
                    help="games for the random-champion control (0 = skip)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.time()
    table, meta = (D.load_pro() if args.data == "pro" else D.load_soloq())
    splits, _ = D.make_splits(table, seed=D.SPLIT_SEED, verbose=False)
    rows = np.asarray(splits["test"], dtype=np.int64)
    if args.limit_games:
        rows = rows[:args.limit_games]
    print("%s: %d test games, %d picks" % (args.data, rows.size, rows.size * SLOTS))

    pools = role_pools(table, splits["train"], args.min_games)
    print("candidate pools (train-only, >=%d games): %s"
          % (args.min_games, [int(p.size) for p in pools]))

    model, payload = W.load_checkpoint(args.ckpt, args.device)
    cfg = payload.get("config") or {}
    if cfg.get("team_feat"):
        raise SystemExit("%s carries a strength feature; this test needs a "
                         "pure draft model" % Path(args.ckpt).parent.name)
    ck_vocab = list(payload.get("patch_vocab") or [])
    data_vocab = list(meta.get("patch_vocab") or [])
    patch_ids = table.patch_id.astype(np.int64)
    if ck_vocab and ck_vocab != data_vocab:
        patch_ids = patch_remap(ck_vocab, data_vocab)[patch_ids]
        print("  patch vocabulary differs; %d of %d test rows on the UNK row"
              % (int((patch_ids[rows] == 0).sum()), rows.size))
    print("model: %s (%.2f M params)"
          % (Path(args.ckpt).parent.name,
             sum(p.numel() for p in model.parameters()) / 1e6))

    rng = np.random.default_rng(args.seed)
    res = run(model, table, rows, pools, args.k, rng, patch_ids, args.device)
    mean, lo, hi = cluster_boot(res["pct"], res["cluster"])
    print("\n== chosen champion vs %d alternatives, %d picks" % (args.k, res["pct"].size))
    print("   pct  %.4f  [%.4f, %.4f]   (0.50 = knows nothing)" % (mean, lo, hi))
    print("   top1 %.4f (chance %.4f)   delta %+.3f probability points"
          % (res["top1"].mean(), 1.0 / (args.k + 1),
             100 * res["delta"].mean()))

    out = {"ckpt": Path(args.ckpt).parent.name, "data": args.data, "k": args.k,
           "seed": args.seed, "n_picks": int(res["pct"].size),
           "model": {"pct": mean, "ci": [lo, hi],
                     "top1": float(res["top1"].mean()),
                     "delta_pp": float(100 * res["delta"].mean())}}

    if args.null_games:
        nrows = rows[:min(args.null_games, rows.size)]
        rng_n = np.random.default_rng(args.seed + 1)
        nul = run(model, table, nrows, pools, args.k, rng_n, patch_ids,
                  args.device, chosen=None)
        nm, nlo, nhi = cluster_boot(nul["pct"], nul["cluster"])
        print("\n== null control (random champion in the chosen slot), %d picks"
              % nul["pct"].size)
        print("   pct  %.4f  [%.4f, %.4f]   must contain 0.50"
              % (nm, nlo, nhi))
        out["null"] = {"pct": nm, "ci": [nlo, nhi],
                       "n_picks": int(nul["pct"].size)}

    print("\n== the tier list alone (champion x role BT prior, no encoder)")
    _feat, rating = D.champion_strength(table, splits["train"], k=20.0,
                                        by_role=True, verbose=False,
                                        return_ratings=True)
    rng_t = np.random.default_rng(args.seed + 2)
    bt = bt_scores(table, pools, rows, rating, args.k, rng_t)
    bm = float(bt["pct"].mean())
    print("   pct  %.4f   top1 %.4f" % (bm, bt["top1"].mean()))
    out["tier_list"] = {"pct": bm, "top1": float(bt["top1"].mean())}

    print("\n== by side and role (model pct)")
    for name, mask in (("blue", res["slot"] < TEAM),
                       ("red", res["slot"] >= TEAM)):
        print("   %-5s %.4f" % (name, res["pct"][mask].mean()))
    for r in range(5):
        m = (res["slot"] % TEAM) == r
        print("   role %d %.4f" % (r, res["pct"][m].mean()))
    out["by_slot"] = {"blue": float(res["pct"][res["slot"] < TEAM].mean()),
                      "red": float(res["pct"][res["slot"] >= TEAM].mean()),
                      "by_role": [float(res["pct"][(res["slot"] % TEAM) == r]
                                        .mean()) for r in range(5)]}

    print("\n== did the pick's side win? (model pct, diagnostic only)")
    y = table.y[rows]
    side_won = np.repeat(y > 0.5, SLOTS)
    for name, mask in (("won", side_won), ("lost", ~side_won)):
        print("   %-4s %.4f" % (name, res["pct"][mask].mean()))
    out["by_outcome"] = {"won": float(res["pct"][side_won].mean()),
                         "lost": float(res["pct"][~side_won].mean())}

    out["elapsed_s"] = round(time.time() - t0, 1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
