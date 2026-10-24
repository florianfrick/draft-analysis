"""Diagnose the value head: where (if anywhere) does it have signal?

Loads a policy checkpoint, runs it over the test split, and reports
win AUC / Brier / calibration broken down by move index and phase.

Motivation: the aggregate test report showed win AUC ~0.497 with Brier
0.343, which is WORSE than the 0.248 you get by predicting the base
rate everywhere. Two hypotheses:
  (a) dilution  - the head works on late (informative) moves but the
      20-position average washes it out;
  (b) broken    - the head predicts confidently but uncorrelated with
      the label at every move.
This script separates them.

Usage:
    python model/policy/diagnose_value.py \
        --ckpt model/policy/ckpt/run_20261023_120714_policy-1/best_policy.pt
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train as T  # noqa: E402
from model import PolicyValueModel  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

N_MOVES = 20
PHASE_NAMES = ["bans1", "picks1", "bans2", "picks2"]
# Number of PICKS already made when move `mv` is about to happen.
PICKS_BEFORE = []
for _mv in range(N_MOVES):
    PICKS_BEFORE.append(sum(1 for _j in range(_mv)
                            if T.PRO_DATA.PHASE_OF[_j] in (1, 3)))


def auc_rank(scores, labels):
    """Mann-Whitney AUC. scores/labels are 1-D tensors, labels in {0,1}."""
    scores = scores.float()
    labels = labels.float()
    n_pos = int(labels.sum())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    n = scores.numel()
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # ranks are 1-based
            ranks[order[i:j + 1]] = avg
        i = j + 1
    pos_rank_sum = float(ranks[labels == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier(p, y):
    return float(((p - y) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--half-life", type=float, default=None)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    print("ckpt: %s" % args.ckpt)
    print("  step=%s  metrics=%s" % (blob.get("step"),
                                     {k: round(v, 4) for k, v in
                                      blob.get("metrics", {}).items()
                                      if isinstance(v, float)}))
    print("  init_ckpt=%s" % blob.get("init_ckpt"))

    # Rebuild encoder from the recorded init checkpoint
    _, init_payload = T.pick_init_ckpt(blob["init_ckpt"])
    encoder, enc_cfg = T.build_encoder(init_payload)

    model = PolicyValueModel(
        encoder, embed_dim=cfg["embed_dim"],
        state_dim=cfg["state_dim"], n_econ=cfg["n_econ"],
        dropout=cfg.get("head_dropout", 0.0),
        n_patches=cfg.get("n_patches", 0))
    model.load_state_dict(blob["model"])
    model.eval()

    # Data
    targs = argparse.Namespace(limit_drafts=args.limit_drafts or None,
                               half_life=args.half_life
                               or T.PRO_PIPE.DEFAULT_HALF_LIFE_DAYS)
    rows, splits = T.load_pro_rows(targs)
    te_ds = T.PRO_DATA.ProNextMoveDataset(
        rows, splits["test"], num_champs=enc_cfg["num_champs"],
        patch_index=T.PRO_DATA.patch_index_from_vocab(
            cfg.get("patch_vocab")))
    te_loader = DataLoader(te_ds, batch_size=args.batch_size,
                           shuffle=False,
                           collate_fn=T.PRO_DATA.pro_collate,
                           num_workers=0)
    print("test examples: %d" % len(te_ds))

    probs, labels, moves = [], [], []
    with torch.no_grad():
        for batch in te_loader:
            out = model(batch)
            p = torch.sigmoid(out["value_logit"].squeeze(-1))
            probs.append(p)
            labels.append(batch["win_label"].float())
            moves.append(batch["move_idx"])

    p = torch.cat(probs)
    y = torch.cat(labels)
    m = torch.cat(moves)

    base = float(y.mean())
    base_brier = base * (1 - base)
    print()
    print("=== aggregate ===")
    print("n=%d  base_rate=%.4f  (base-rate Brier=%.4f)"
          % (y.numel(), base, base_brier))
    print("AUC=%.4f  Brier=%.4f  mean_pred=%.4f"
          % (auc_rank(p, y), brier(p, y), float(p.mean())))

    print()
    print("=== by move index ===")
    print("%3s %-7s %6s %8s %8s %9s" %
          ("mv", "phase", "n", "AUC", "Brier", "mean_p"))
    for mv in range(N_MOVES):
        sel = m == mv
        if int(sel.sum()) == 0:
            continue
        ph = T.PRO_DATA.PHASE_OF[mv]
        print("%3d %-7s %6d %8.4f %8.4f %9.4f"
              % (mv, PHASE_NAMES[ph], int(sel.sum()),
                 auc_rank(p[sel], y[sel]), brier(p[sel], y[sel]),
                 float(p[sel].mean())))

    print()
    print("=== by phase ===")
    for ph in range(4):
        sel = torch.tensor([T.PRO_DATA.PHASE_OF[int(i)] == ph
                            for i in m])
        if int(sel.sum()) == 0:
            continue
        print("%-7s n=%5d AUC=%.4f Brier=%.4f mean_p=%.4f"
              % (PHASE_NAMES[ph], int(sel.sum()),
                 auc_rank(p[sel], y[sel]), brier(p[sel], y[sel]),
                 float(p[sel].mean())))

    print()
    print("=== by draft-completion bucket ===")
    for lo, hi, name in ((0, 2, "0-2 picks"), (3, 5, "3-5 picks"),
                         (6, 8, "6-8 picks"), (9, 10, "9-10 (full)")):
        sel = torch.tensor([lo <= PICKS_BEFORE[int(i)] <= hi
                            for i in m])
        if int(sel.sum()) == 0:
            continue
        print("%-16s n=%5d AUC=%.4f Brier=%.4f"
              % (name, int(sel.sum()),
                 auc_rank(p[sel], y[sel]), brier(p[sel], y[sel])))

    print()
    print("=== prediction spread ===")
    print("p: min=%.4f p05=%.4f median=%.4f p95=%.4f max=%.4f std=%.4f"
          % (float(p.min()), float(p.quantile(0.05)),
             float(p.median()), float(p.quantile(0.95)),
             float(p.max()), float(p.std())))
    sat = float(((p < 0.05) | (p > 0.95)).float().mean())
    print("fraction saturated (<0.05 or >0.95): %.4f" % sat)


if __name__ == "__main__":
    main()
