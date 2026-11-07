"""Is a next-move miss a wrong champion, or a right champion at the wrong move?

The pooled metric is exact-move top-k: the model gets credit only if it names
the champion that the very next move takes. That conflates three different
failures, and only one of them is "the model does not know the draft":

  * hit      - rank 0, nothing to explain
  * later    - the model's #1 is taken later in the SAME draft (by the acting
               team, by the opponent, or as a ban). In a double-pick block this
               is the partner pick: the team takes both champions as one joint
               decision, so an ordered label splits that decision in two and
               the model is charged for guessing the pair in the other order.
  * absent   - the champion never appears in the draft at all. This is the only
               class that is unambiguously a wrong read of the draft.

Also reports the block-specific view: for the FIRST move of a same-team
double-pick block, how often is the model's #1 the partner pick, and where does
the partner rank. If the pair is genuinely exchangeable, a model with perfect
knowledge of the pair scores 0.5 top-1 on the first move and 1.0 on the second,
and the current metric reads that as 0.5.

Usage:
    python diagnose_timing.py --ckpt <nextpick ckpt> [--split val|test]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRO_DIR = ROOT / "data" / "pro play"
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")
EP = _load("eval_protocol", PRO_DIR / "eval_protocol.py")

from solo_queue.model import PolicyValueModel                                   # noqa: E402
from solo_queue.model.pretraining.base_model import SiameseBase       # noqa: E402
from solo_queue.model.pretraining.base_model import attr_matrix_for_config     # noqa: E402

MOVE_BAN = PRO_PIPE.MOVE_BAN
MOVE_PICK = PRO_PIPE.MOVE_PICK
N_MOVES = PRO_DATA.N_MOVES
# The move order is fixed (6 bans, 6 picks, 4 bans, 4 picks) regardless of
# which side moves first, so the kind of every move index is a constant.
KIND_OF = ["ban"] * 6 + ["pick"] * 6 + ["ban"] * 4 + ["pick"] * 4


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="")
    return ap.parse_args(argv)


def build_model(blob, device):
    cfg = blob["config"]
    enc = SiameseBase(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        attr_matrix=attr_matrix_for_config(cfg))
    model = PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg["state_dim"],
        n_econ=cfg["n_econ"], dropout=cfg.get("head_dropout", 0.0),
        n_patches=cfg.get("n_patches", 0),
        with_value=cfg.get("with_value", True)).to(device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    return model, cfg


def block_role(moves):
    """For each move index: 'solo', 'first', 'second' within a double block."""
    out = ["solo"] * N_MOVES
    for m in range(N_MOVES - 1):
        a, b = moves[m], moves[m + 1]
        if a[0] == MOVE_PICK and b[0] == MOVE_PICK and a[1] == b[1]:
            out[m], out[m + 1] = "first", "second"
    return out


@torch.no_grad()
def run(model, rows, indices, patch_index, cfg, device, batch_size):
    ds = PRO_DATA.ProNextMoveDataset(
        rows, indices, num_champs=cfg["num_champs"],
        patch_index=patch_index, prefix_mode="all", swap_aug=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=PRO_DATA.pro_collate)
    # build_prefix's meta carries the gameid, not the row index
    by_game = {rows[i]["gameid"]: rows[i] for i in indices}

    # per move index counters
    n = Counter()
    hit1 = Counter()
    hit5 = Counter()
    later_own = Counter()
    later_opp = Counter()
    later_ban = Counter()
    absent = Counter()
    # block counters: only for the FIRST move of a block
    blk_n = Counter()
    blk_partner_top1 = Counter()
    blk_partner_rank = defaultdict(list)
    blk_true_rank = defaultdict(list)
    # ordering-agnostic variants: would the truth OR its exchangeable sibling
    # be in the top-k? For a double-pick block the sibling is the partner
    # pick; for a ban it is any champion banned later in the same ban phase.
    # These are upper bounds on "how much of the miss is arbitrary order",
    # not replacement metrics - they give the model more than one target.
    ord_top1 = Counter()
    ord_top5 = Counter()
    n_ord = Counter()
    # per-draft hit counts, for the clustered CI. The 20 prefixes of one draft
    # are near-duplicates, so the 50k states are not 50k independent trials and
    # a Wilson interval over states is far too narrow.
    per_draft = defaultdict(lambda: [0, 0, 0])   # gameid -> [n, hit1, hit5]
    # Is the order inside a double-pick block even observable? If the corpus
    # stores the pair sorted by role, the "first" of the two is a coin the
    # model cannot see (the target's role is not an input), and the metric
    # charges it for a label that carries no information.
    blk_role_order = defaultdict(lambda: [0, 0])  # move -> [n, role_asc]
    # block-set metrics: the model's per-move argmax collected into the set a
    # block actually needs, compared against the block's true set
    pred = {}                                     # (gameid, move) -> argmax

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        logits = model(batch)["policy_logits"]
        tc = batch["target_champ"]
        truth_logit = logits.gather(1, tc[:, None]).squeeze(1)
        ranks = (logits > truth_logit[:, None]).sum(1)
        top1 = logits.argmax(1)

        top5 = torch.topk(logits, 5, dim=1).indices
        hit5_mask = (top5 == tc[:, None]).any(1)

        metas = batch["meta"]
        for j in range(tc.numel()):
            m = int(batch["move_idx"][j])
            meta = metas[j]
            row = by_game[meta["gameid"]]
            moves = row["moves"]
            c = int(tc[j])
            t1 = int(top1[j])
            r = int(ranks[j])
            n[m] += 1
            hit1[m] += int(r == 0)
            hit5[m] += int(bool(hit5_mask[j]))
            pd = per_draft[meta["gameid"]]
            pd[0] += 1
            pd[1] += int(r == 0)
            pd[2] += int(bool(hit5_mask[j]))
            pred[(meta["gameid"], m)] = t1
            if KIND_OF[m] == "pick" and m + 1 < N_MOVES \
                    and moves[m + 1][0] == MOVE_PICK \
                    and moves[m + 1][1] == moves[m][1]:
                blk_role_order[m][0] += 1
                blk_role_order[m][1] += int(moves[m][3] < moves[m + 1][3])

            # exchangeable siblings for an ordering-agnostic check
            sib = set()
            if KIND_OF[m] == "pick" and m + 1 < N_MOVES \
                    and moves[m + 1][0] == MOVE_PICK \
                    and moves[m + 1][1] == moves[m][1]:
                sib.add(moves[m + 1][2])
            elif KIND_OF[m] == "ban":
                phase = [range(0, 6), range(12, 16)][
                    0 if m < 6 else 1]
                for mm in phase:
                    if mm > m and moves[mm][0] == MOVE_BAN:
                        sib.add(moves[mm][2])
            sib.discard(c)
            if sib:
                n_ord[m] += 1
                ord_top1[m] += int(t1 == c or t1 in sib)
                ord_top5[m] += int(bool(hit5_mask[j])
                                   or any(int(x) in sib
                                          for x in top5[j].tolist()))
            if r == 0:
                continue
            # where does the model's #1 actually go in this draft?
            rest = moves[m + 1:]
            if any(mv[0] == MOVE_PICK and mv[2] == t1 and mv[1] == moves[m][1]
                   for mv in rest):
                later_own[m] += 1
            elif any(mv[0] == MOVE_PICK and mv[2] == t1 for mv in rest):
                later_opp[m] += 1
            elif any(mv[0] == MOVE_BAN and mv[2] == t1 for mv in rest):
                later_ban[m] += 1
            else:
                absent[m] += 1

            # block analysis: partner = the next move's champion
            if moves[m][0] == MOVE_PICK and m + 1 < N_MOVES \
                    and moves[m + 1][0] == MOVE_PICK \
                    and moves[m + 1][1] == moves[m][1]:
                partner = moves[m + 1][2]
                blk_n[m] += 1
                blk_partner_top1[m] += int(t1 == partner)
                prow = int((logits[j] > logits[j, partner]).sum())
                blk_partner_rank[m].append(prow)
                blk_true_rank[m].append(r)

    return {"n": n, "hit1": hit1, "hit5": hit5, "later_own": later_own,
            "later_opp": later_opp, "later_ban": later_ban, "absent": absent,
            "blk_n": blk_n, "blk_partner_top1": blk_partner_top1,
            "blk_partner_rank": blk_partner_rank,
            "blk_true_rank": blk_true_rank,
            "n_ord": n_ord, "ord_top1": ord_top1, "ord_top5": ord_top5,
            "per_draft": dict(per_draft),
            "blk_role_order": {k: list(v) for k, v in blk_role_order.items()},
            "pred": pred}


def block_sets(stats, rows, indices):
    """Set metrics over move blocks: Jaccard vs the model's per-move argmax.

    Three groupings, because "block" is ambiguous and the answer differs:
      * pair  - the contiguous same-team double-pick block (size 2)
      * picks - one team's picks in one phase (size 2-3, not contiguous)
      * bans  - one team's bans in one phase (size 2-3, not contiguous)

    The predicted set is the model's argmax at each move of the block, so the
    set size is fixed by the block and the model cannot game |P| - except by
    repeating itself, which the size-1 column counts.
    """
    pred = stats["pred"]
    rows_by_game = {rows[i]["gameid"]: rows[i] for i in indices}
    groups = {"pair": defaultdict(list), "picks": defaultdict(list),
              "bans": defaultdict(list)}
    for i in indices:
        row = rows[i]
        g = row["gameid"]
        moves = row["moves"]
        for m in range(N_MOVES):
            kind, team, champ, _role = moves[m]
            phase = 0 if m < 6 else (1 if m < 12 else (2 if m < 16 else 3))
            key = (g, team, phase)
            groups["bans" if KIND_OF[m] == "ban" else "picks"][key].append(m)
            if KIND_OF[m] == "pick" and m + 1 < N_MOVES \
                    and moves[m + 1][0] == MOVE_PICK \
                    and moves[m + 1][1] == team:
                groups["pair"][(g, m)].append(m)
                groups["pair"][(g, m)].append(m + 1)

    print("\nblock-set metrics (predicted set = per-move argmax):")
    print("  %-6s %6s %7s %8s %8s %8s %8s %8s %6s %7s"
          % ("group", "n", "|T|", "jaccard", "J=0", "partial", "J=1",
             "exact@1", "|P|", "swap"))
    out = {}
    for name, gs in groups.items():
        js, sizes, t_sizes, exact = [], [], [], []
        swapped = 0
        for key, ms in gs.items():
            if len(ms) < 2:
                continue
            row = rows_by_game[key[0]]
            T = {row["moves"][m][2] for m in ms}
            P = {pred[(key[0], m)] for m in ms}
            js.append(len(P & T) / len(P | T))
            sizes.append(len(P))
            t_sizes.append(len(T))
            exact.extend(int(pred[(key[0], m)] == row["moves"][m][2])
                         for m in ms)
            # whole set right, order wrong: the case Jaccard forgives
            if len(P & T) == len(T) == len(ms) \
                    and any(pred[(key[0], m)] != row["moves"][m][2]
                            for m in ms):
                swapped += 1
        if not js:
            continue
        n = len(js)
        print("  %-6s %6d %7.2f %8.4f %8.3f %8.3f %8.3f %8.4f %6.2f %7.3f"
              % (name, n, sum(t_sizes) / n, sum(js) / n,
                 sum(1 for j in js if j == 0) / n,
                 sum(1 for j in js if 0 < j < 1) / n,
                 sum(1 for j in js if j == 1) / n,
                 sum(exact) / len(exact), sum(sizes) / n,
                 swapped / n))
        out[name] = {"n": n, "jaccard": sum(js) / n,
                     "zero": sum(1 for j in js if j == 0) / n,
                     "partial": sum(1 for j in js if 0 < j < 1) / n,
                     "full": sum(1 for j in js if j == 1) / n,
                     "exact_top1": sum(exact) / len(exact),
                     "mean_pred_set": sum(sizes) / n,
                     "degenerate_singleton": sum(1 for s in sizes if s == 1) / n}
    print("  (exact@1 = mean per-move top-1 over the SAME moves, for scale)")
    return out


def report(stats):
    n, hit1, hit5 = stats["n"], stats["hit1"], stats["hit5"]
    print("\n%-4s %-7s %6s %7s %7s %7s %7s %7s %7s"
          % ("move", "kind", "n", "top1", "top5", "later+", "later-",
             "laterB", "absent"))
    tot = Counter()
    for m in range(N_MOVES):
        if not n[m]:
            continue
        miss = n[m] - hit1[m]
        tot["n"] += n[m]
        tot["hit1"] += hit1[m]
        tot["hit5"] += hit5[m]
        for k in ("later_own", "later_opp", "later_ban", "absent"):
            tot[k] += stats[k][m]
        print("%-4d %-7s %6d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f"
              % (m, KIND_OF[m], n[m], hit1[m] / n[m], hit5[m] / n[m],
                 stats["later_own"][m] / max(1, miss),
                 stats["later_opp"][m] / max(1, miss),
                 stats["later_ban"][m] / max(1, miss),
                 stats["absent"][m] / max(1, miss)))
    miss = tot["n"] - tot["hit1"]
    print("%-4s %-7s %6d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f"
          % ("all", "", tot["n"], tot["hit1"] / tot["n"],
             tot["hit5"] / tot["n"],
             tot["later_own"] / max(1, miss), tot["later_opp"] / max(1, miss),
             tot["later_ban"] / max(1, miss), tot["absent"] / max(1, miss)))
    print("  (later+/later-/laterB/absent are shares of MISSES, not of n)")

    print("\nordering-agnostic upper bound (truth or an exchangeable sibling"
          " in the top-k):")
    print("  %-4s %-7s %6s %8s %8s %8s %8s"
          % ("move", "kind", "n", "top1", "top1+", "top5", "top5+"))
    for m in range(N_MOVES):
        if not stats["n_ord"][m]:
            continue
        print("  %-4d %-7s %6d %8.3f %8.3f %8.3f %8.3f"
              % (m, KIND_OF[m], stats["n_ord"][m], hit1[m] / n[m],
                 stats["ord_top1"][m] / stats["n_ord"][m],
                 hit5[m] / n[m], stats["ord_top5"][m] / stats["n_ord"][m]))
    a = sum(stats["n_ord"][m] for m in range(N_MOVES))
    b = sum(stats["ord_top1"][m] for m in range(N_MOVES))
    c = sum(stats["ord_top5"][m] for m in range(N_MOVES))
    h1 = sum(hit1[m] for m in range(N_MOVES) if stats["n_ord"][m])
    h5 = sum(hit5[m] for m in range(N_MOVES) if stats["n_ord"][m])
    print("  %-4s %-7s %6d %8.3f %8.3f %8.3f %8.3f"
          % ("all", "", a, h1 / a, b / a, h5 / a, c / a))

    # ---- clustered CI: the unit of independent information is the draft
    cl = list(stats["per_draft"].values())
    n_draft = len(cl)
    for idx, label in ((1, "top1"), (2, "top5")):
        tot_n = sum(c[0] for c in cl)
        tot_h = sum(c[idx] for c in cl)
        p = tot_h / tot_n
        lo, hi = EP.wilson(tot_h, tot_n)
        w_half = (hi - lo) / 2.0 / 1.96
        # cluster bootstrap over drafts (equal-size clusters: 20 states each)
        import random
        rng = random.Random(20261105)
        bs = []
        for _ in range(2000):
            s_n = s_h = 0
            for _ in range(n_draft):
                c = cl[rng.randrange(n_draft)]
                s_n += c[0]
                s_h += c[idx]
            bs.append(s_h / s_n)
        bs.sort()
        b_lo, b_hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]
        deff = ((b_hi - b_lo) / 2.0 / 1.96 / w_half) ** 2
        print("  %s = %.4f  state CI [%.4f, %.4f] (n=%d)  "
              "draft-clustered CI [%.4f, %.4f] (n=%d drafts, design effect "
              "%.1fx)"
              % (label, p, lo, hi, tot_n, b_lo, b_hi, n_draft, deff))

    print("\ndouble-pick block, FIRST move of the pair:")
    print("  %-4s %6s %10s %12s %12s %10s"
          % ("move", "n", "P(#1=mate)", "mean rank", "mean rank",
             "P(role"))
    print("  %-4s %6s %10s %12s %12s %10s"
          % ("", "", "", "of mate", "of truth", "ascending)"))
    for m in sorted(stats["blk_n"]):
        bn = stats["blk_n"][m]
        pr = stats["blk_partner_rank"][m]
        tr = stats["blk_true_rank"][m]
        ro = stats["blk_role_order"].get(m, [0, 0])
        print("  %-4d %6d %10.3f %12.2f %12.2f %10.3f"
              % (m, bn, stats["blk_partner_top1"][m] / bn,
                 sum(pr) / len(pr), sum(tr) / len(tr),
                 ro[1] / max(1, ro[0])))


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    blob = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    model, cfg = build_model(blob, device)
    print("checkpoint %s (step %s)" % (args.ckpt, blob.get("step")))

    rows, _ = PRO_PIPE.build_rows(limit=args.limit_drafts)
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_series_by_date(rows)
    honest, _ = EP.patch_honest_val(rows, splits, verbose=False)
    indices = honest if args.split == "val" else splits["test"]
    print("split %s: %d drafts, %d states"
          % (args.split, len(indices), len(indices) * N_MOVES))

    _, patch_index = PRO_DATA.build_patch_vocab(rows)
    stats = run(model, rows, indices, patch_index, cfg, device,
                args.batch_size)
    report(stats)
    sets = block_sets(stats, rows, indices)

    if args.json:
        out = {k: (dict(v) if isinstance(v, Counter) else
                   {kk: vv for kk, vv in v.items()})
               for k, v in stats.items()}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
