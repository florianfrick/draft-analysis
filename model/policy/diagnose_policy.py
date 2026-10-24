"""Per-move-position policy diagnostics for a ProNextMoveDataset checkpoint.

Answers three questions the aggregate test report can't:

1. Where in the 20-move draft does the model actually have signal?
2. Does the gain over baseline concentrate in a few move positions?
3. Is the model doing more than mirroring move-position popularity?

The last one is the important one. A next-move model can score well on
top-5 purely by learning "what is usually taken at move index i", which
is a lookup table, not draft understanding. We measure it two ways:

  * nov@5   -- of the model's top-5 guesses, what fraction are outside
               the position's top-5 popularity set? (higher = more
               original)
  * hit@5^nov -- top-5 hit rate restricted to cases where the truth is
               NOT in the position's top-5 popularity set. This is the
               honest "did it learn anything conditional" number; a pure
               popularity model scores 0 here by construction.

Usage:
    python diagnose_policy.py --ckpt <path> [--ckpt <path> ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PRO_DIR = HERE.parent.parent / "data" / "pro play"
sys.path.insert(0, str(HERE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")

import train as T  # noqa: E402  (needs sys.path entry above)
from model import PolicyValueModel  # noqa: E402

N_MOVES = PRO_DATA.N_MOVES
PHASE_OF = PRO_DATA.PHASE_OF
PHASE_NAMES = ["bans1", "picks1", "bans2", "picks2"]
MOVE_BAN = PRO_PIPE.MOVE_BAN


def build_popularity(rows, indices, num_champs):
    """Per-move-position champion ranking from train rows (weighted)."""
    counts = [Counter() for _ in range(N_MOVES)]
    for i in indices:
        row = rows[i]
        w = float(row.get("weight", 1.0))
        for m_idx, (_kind, _team, champ, _role) in enumerate(row["moves"]):
            counts[m_idx][int(champ)] += w
    ranks = []
    top_sets = []
    for m_idx in range(N_MOVES):
        c = counts[m_idx]
        order = sorted(range(num_champs),
                       key=lambda x: (-float(c.get(x, 0.0)), x))
        rank = torch.empty(num_champs, dtype=torch.long)
        rank[torch.tensor(order)] = torch.arange(num_champs)
        ranks.append(rank)
        top_sets.append(set(order[:5]))
    return ranks, top_sets


@torch.no_grad()
def evaluate_ckpt(ckpt_path, rows, splits, num_champs, device, batch_size=512):
    payload = torch.load(str(ckpt_path), map_location="cpu",
                         weights_only=False)
    cfg = payload["config"]
    # The policy checkpoint stores the whole PolicyValueModel state dict
    # (encoder weights under an "encoder." prefix), so construct an
    # encoder of the right shape and let load_state_dict overwrite it.
    encoder = T.SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=0.0)
    model = PolicyValueModel(
        encoder, embed_dim=cfg["embed_dim"], state_dim=cfg.get("state_dim", 64),
        n_econ=cfg.get("n_econ", 0), dropout=0.0,
        n_patches=cfg.get("n_patches", 0),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    # Two popularity references:
    #   ranks/top_sets     -- fit on TRAIN, applied to test. This is what
    #                         train.py reports. It is a *stale* prior: the
    #                         test split is the last 5% of series by date,
    #                         so it sits 6+ patches after the end of train
    #                         and its ban priorities have moved on.
    #   o_ranks/o_top_sets -- fit on TEST itself (oracle). This is the
    #                         upper bound for a static lookup table that
    #                         knows the current patch's meta. A model that
    #                         scores below this on bans has no contextual
    #                         signal to fall back on when the prior is
    #                         stale.
    ranks, top_sets = build_popularity(rows, splits["train"], num_champs)
    o_ranks, o_top_sets = build_popularity(rows, splits["test"], num_champs)

    ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["test"], num_champs=num_champs,
        patch_index=PRO_DATA.patch_index_from_vocab(
            cfg.get("patch_vocab")))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=PRO_DATA.pro_collate, num_workers=0)

    per_move = {m: {"n": 0, "top1": 0, "top5": 0, "top10": 0,
                    "pop5": 0, "pop5o": 0, "nov5": 0, "nov5_denom": 0,
                    "hit5_nov": 0, "n_nov": 0,
                    "pred1": Counter()} for m in range(N_MOVES)}
    offset = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        cl = out["policy_logits"].float().cpu()
        tc = batch["target_champ"].cpu()
        bsz = tc.numel()

        truth_logit = cl.gather(1, tc[:, None]).squeeze(1)
        truth_rank = (cl > truth_logit[:, None]).sum(1)
        top5_idx = cl.topk(5, dim=1).indices
        top1_idx = top5_idx[:, 0]

        for j in range(bsz):
            m = (offset + j) % N_MOVES
            c = int(tc[j])
            pm = per_move[m]
            pm["n"] += 1
            r = int(truth_rank[j])
            if r < 1:
                pm["top1"] += 1
            if r < 5:
                pm["top5"] += 1
            if r < 10:
                pm["top10"] += 1
            in_pop5 = c in top_sets[m]
            if in_pop5:
                pm["pop5"] += 1
            if c in o_top_sets[m]:
                pm["pop5o"] += 1
            pm["pred1"][int(top1_idx[j])] += 1
            # novelty of the top-5 guess set
            for g in top5_idx[j].tolist():
                pm["nov5_denom"] += 1
                if g not in top_sets[m]:
                    pm["nov5"] += 1
            # honest hit rate: only cases the truth is outside pop top-5
            if not in_pop5:
                pm["n_nov"] += 1
                if r < 5:
                    pm["hit5_nov"] += 1
        offset += bsz

    return payload, per_move


def report(name, payload, per_move):
    m = payload.get("metrics", {})
    print("=" * 78)
    print("%s" % name)
    print("  step=%s  epoch=%s  val_top5=%s  val_ploss=%s" % (
        payload.get("step"), m.get("epoch"),
        ("%.4f" % m["top5"]) if "top5" in m else "-",
        ("%.4f" % m["policy_loss"]) if "policy_loss" in m else "-"))
    print()
    print("  %3s %-7s %7s %7s %7s %7s %7s %7s %6s %5s" % (
        "mv", "phase", "top1", "top5", "top10", "pop5", "pop5o", "hit5nv", "n",
        "div"))
    tot = Counter()
    for mv in range(N_MOVES):
        pm = per_move[mv]
        n = max(1, pm["n"])
        nv = max(1, pm["n_nov"])
        print("  %3d %-7s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %6d %5d" % (
            mv, PHASE_NAMES[PHASE_OF[mv]],
            pm["top1"] / n, pm["top5"] / n, pm["top10"] / n,
            pm["pop5"] / n, pm["pop5o"] / n, pm["hit5_nov"] / nv, pm["n"],
            len(pm["pred1"])))
        for k in ("n", "top1", "top5", "top10", "pop5", "pop5o", "nov5",
                  "nov5_denom", "hit5_nov", "n_nov"):
            tot[k] += pm[k]
    print()
    print("  OVERALL  top1 %.4f  top5 %.4f  top10 %.4f" % (
        tot["top1"] / tot["n"], tot["top5"] / tot["n"],
        tot["top10"] / tot["n"]))
    print("  pop5 stale(train->test) %.4f | pop5 oracle(test-fit) %.4f | "
          "model-vs-oracle %+.4f" % (
              tot["pop5"] / tot["n"], tot["pop5o"] / tot["n"],
              (tot["top5"] - tot["pop5o"]) / tot["n"]))
    print("  nov@5 %.3f | hit@5 when truth outside stale pop5: %.4f (n=%d)" % (
        tot["nov5"] / tot["nov5_denom"],
        tot["hit5_nov"] / max(1, tot["n_nov"]), tot["n_nov"]))
    # diversity of top-1 predictions per move
    div = [len(per_move[m]["pred1"]) for m in range(N_MOVES)]
    print("  distinct top-1 champions per move: min=%d med=%d max=%d "
          "(of 173)" % (min(div), sorted(div)[len(div) // 2], max(div)))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint path; repeat for comparison")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rows, _stats = PRO_PIPE.build_rows(limit=args.limit_drafts)
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_series_by_date(rows)
    num_champs = 173
    print("rows %d (train %d / val %d / test %d)  device=%s" % (
        len(rows), len(splits["train"]), len(splits["val"]),
        len(splits["test"]), device))
    print()

    for path in args.ckpt:
        payload, per_move = evaluate_ckpt(
            path, rows, splits, num_champs, device, args.batch_size)
        report(Path(path).parent.name + "/" + Path(path).name,
               payload, per_move)


if __name__ == "__main__":
    main()
