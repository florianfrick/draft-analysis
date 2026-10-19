"""Deep evaluation of a SoloQ pretraining checkpoint.

    python pretraining/evaluate.py [--ckpt PATH] [--device cuda]

Reports (printed + pretraining/evaluate_report.json):
  A. main val metrics under training-style masking (15% champ masks, 90/10
     role on those, plus ~10% role-only with champ visible):
     champ top-1/5/10, MRR, role-cond top-1/5, role|champ top-1 (P(role|champ)
     on role-only slots - the champ->role affinity metric), per-role top-1/5,
     val CE (champ/role), popularity baselines on the SAME masked slots.
  B. train-vs-val CE gap (under/overfit read).
  C. robustness curve: top-5 with K simultaneous mask queries per draft,
     K in {1,3,5,10} (K=10 = fully empty frontend draft) - measures the
     multi-query train/inference mismatch.
  D. five qualitative masked-slot examples (truth vs top-5 names).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))                  # soloq_model
sys.path.insert(0, str(HERE.parent.parent / "data" /
                      "solo queue" / "process matchdata"))

import pipeline as P                    # noqa: E402
import train as T                       # noqa: E402  (reuses loaders/eval)
from soloq_model import SoloQPretrainedModel  # noqa: E402

ROLE_ROWS = ["TOP", "JGL", "MID", "ADC", "SUP"]


def load_model(ckpt, device):
    payload = torch.load(str(ckpt), map_location="cpu")
    cfg = payload["config"]
    model = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


def rank_of(logits, labels):
    """Rank of the true class per row (0 = top-1); -inf col handled."""
    truth = logits.gather(1, labels[:, None]).squeeze(1)
    return (logits > truth[:, None]).sum(1)


@torch.no_grad()
def main_eval(model, loader, base, names, device, n_examples=5):
    role_rank, grank = T.build_rank_tensors(base, model.champ_head.out_features)
    role_rank, grank = role_rank.to(device), grank.to(device)
    hits = {k: 0 for k in (1, 5, 10)}
    role_hits = {k: 0 for k in (1, 5)}
    role_given_champ_hits = 0
    per_role = {r: {k: 0 for k in (1, 5)} for r in range(5)}
    per_role_n = {r: 0 for r in range(5)}
    base_role = {k: 0 for k in (1, 5, 10)}
    base_glob = {k: 0 for k in (1, 5, 10)}
    rr_sum, n, n_cond, n_role_only, n_role = 0.0, 0, 0, 0, 0
    lc_sum, lr_sum = 0.0, 0.0
    examples = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        m = batch["is_masked"]
        cl, ct = out["champ_logits"][m], batch["champ_labels"][m]
        # role CE over ALL role-labelled slots (champ-masked + role-only),
        # matching the training loss; champ metrics stay on is_masked only.
        rmask = batch["role_labels"] != -100
        rl_all, rt_all = out["role_logits"][rmask], batch["role_labels"][rmask]
        rl, rt = out["role_logits"][m], batch["role_labels"][m]
        lc = F.cross_entropy(cl, ct)
        lr = F.cross_entropy(rl_all, rt_all)
        rk = rank_of(cl, ct)
        cnt = rk.numel()
        lc_sum += float(lc) * cnt
        lr_sum += float(lr) * int(rmask.sum())
        n_role += int(rmask.sum())
        rr_sum += float((1.0 / (rk + 1)).sum())
        for k in hits:
            hits[k] += int((rk < k).sum())
        vis = ~batch["role_masked"][m]
        for k in role_hits:
            role_hits[k] += int((rk[vis] < k).sum())
        n_cond += int(vis.sum())
        # P(role | champ): champ visible, role hidden (role-only slots).
        ro = batch["role_masked"] & (~m) & rmask
        if bool(ro.any()):
            role_given_champ_hits += int(
                (out["role_logits"][ro].argmax(1)
                 == batch["role_labels"][ro]).sum())
            n_role_only += int(ro.sum())
        rks = role_rank[rt, ct]
        grs = grank[ct]
        for k in base_role:
            base_role[k] += int((rks < k).sum())
            base_glob[k] += int((grs < k).sum())
        for r in range(5):
            sel = rt == r
            per_role_n[r] += int(sel.sum())
            for k in (1, 5):
                per_role[r][k] += int((rk[sel] < k).sum())
        if len(examples) < n_examples:
            for i in range(cnt):
                if len(examples) >= n_examples:
                    break
                top5 = cl[i].topk(5)
                examples.append({
                    "truth": names[int(ct[i])],
                    "role": ROLE_ROWS[int(rt[i])],
                    "top5": [(names[int(j)], round(float(p), 4))
                             for j, p in zip(top5.indices, top5.values)]})
        n += cnt
    res = {
        "n_masked": n,
        "n_role_labelled": n_role,
        "n_role_only": n_role_only,
        "val_ce_champ": lc_sum / n, "val_ce_role": lr_sum / max(1, n_role),
        "mrr": rr_sum / n,
        "champ_top1": hits[1] / n, "champ_top5": hits[5] / n,
        "champ_top10": hits[10] / n,
        "role_cond_top1": role_hits[1] / max(1, n_cond),
        "role_cond_top5": role_hits[5] / max(1, n_cond),
        "role_given_champ_top1": role_given_champ_hits / max(1, n_role_only),
        "base_role_top1": base_role[1] / n, "base_role_top5": base_role[5] / n,
        "base_role_top10": base_role[10] / n,
        "base_global_top1": base_glob[1] / n,
        "base_global_top5": base_glob[5] / n,
        "base_global_top10": base_glob[10] / n,
        "per_role": {ROLE_ROWS[r]: {"n": per_role_n[r],
                                    "top1": per_role[r][1] / max(1, per_role_n[r]),
                                    "top5": per_role[r][5] / max(1, per_role_n[r])}
                     for r in range(5)},
        "examples": examples,
    }
    return res
