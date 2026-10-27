"""Capstone: soloQ-derived matchup fingerprints, pro-derived win weights.

The two corpora fail in opposite directions:

  soloQ  415k games -> a lane cell is reliable at ~200 games, but win/loss
                     is mostly player skill, so win-rate cells are noise.
  pro     46k games -> skill is near-constant so draft actually moves the
                     outcome, but 46k games cannot fill 30k cells.

So split the problem: estimate *what happens in a matchup* on soloQ (fp),
and estimate *how much that matters* on pro (theta, 6 params). Neither
corpus can do the other's job; together they cover it.

Usage:  python "soloq data analysis/pro_theta.py"
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STORE = os.path.join(HERE, "store")
PROC = os.path.join(ROOT, "data", "solo queue", "process matchdata")
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
FP_METRICS = ["maxCsAdvantageOnLaneOpponent", "maxLevelLeadLaneOpponent",
              "goldPerMinute", "damagePerMinute", "teamDamagePercentage",
              "killParticipation"]

with open(os.path.join(PROC, "champ_mapping.json"), encoding="utf-8") as f:
    RIOT2IX = {int(k): int(v) for k, v in json.load(f)["riot_to_idx"].items()}
NCH = len(RIOT2IX)
ID2IX = RIOT2IX


def build_fp(shrink=150.0):
    """fp[role, a, b, k], shrunk toward the role-pooled cell."""
    cols = ["match_id", "side", "pos", "champ"] + FP_METRICS
    df = pq.read_table(os.path.join(STORE, "players.parquet"),
                       columns=cols).to_pandas()
    df = df[df["pos"].isin(ROLES)].drop_duplicates(
        subset=["match_id", "side", "pos"])
    opp = df[["match_id", "side", "pos", "champ"]].copy()
    opp["side"] = opp["side"].map({"blue": "red", "red": "blue"})
    opp = opp.rename(columns={"champ": "opp"})
    df = df.merge(opp, on=["match_id", "side", "pos"], how="inner")
    df["ci"] = df["champ"].map(ID2IX)
    df["oi"] = df["opp"].map(ID2IX)
    df["ri"] = df["pos"].map({r: i for i, r in enumerate(ROLES)})
    df = df.dropna(subset=["ci", "oi"])
    print("soloQ player-games: %d" % len(df))

    K = len(FP_METRICS)
    pooled = np.zeros((NCH, NCH, K), dtype=np.float64)
    pn = np.zeros((NCH, NCH), dtype=np.float64)
    per_role = np.zeros((5, NCH, NCH, K), dtype=np.float64)
    rn = np.zeros((5, NCH, NCH), dtype=np.float64)

    for k, met in enumerate(FP_METRICS):
        base = df.groupby("ci", observed=True)[met].mean()
        cell = df.groupby(["ci", "oi"], observed=True)[met].agg(["mean", "count"])
        eff = cell["mean"] - cell.index.get_level_values("ci").map(base)
        # A cell with no non-null observation has a NaN mean. Left alone it
        # propagates: one NaN makes v.mean() NaN, which zeroes the entire
        # dimension after standardisation. That silently killed 4 of the 6
        # dims (162 all-null champion/role cells for the lane metrics).
        eff = eff.fillna(0.0)
        ci = cell.index.get_level_values("ci").to_numpy()
        oi = cell.index.get_level_values("oi").to_numpy()
        pooled[ci, oi, k] = eff.to_numpy()
        if k == 0:
            pn[ci, oi] = cell["count"].to_numpy()
        cellr = df.groupby(["ri", "ci", "oi"], observed=True)[met].agg(["mean", "count"])
        effr = cellr["mean"] - cellr.index.get_level_values("ci").map(base)
        effr = effr.fillna(0.0)
        ri = cellr.index.get_level_values("ri").to_numpy()
        per_role[ri, cellr.index.get_level_values("ci").to_numpy(),
                 cellr.index.get_level_values("oi").to_numpy(), k] = effr.to_numpy()
        if k == 0:
            rn[ri, cellr.index.get_level_values("ci").to_numpy(),
               cellr.index.get_level_values("oi").to_numpy()] = cellr["count"].to_numpy()

    # shrink role-specific cells toward the role-pooled cell
    w = (rn / (rn + shrink))[..., None]
    fp = w * per_role + (1 - w) * pooled[None, :, :, :]
    for k in range(K):
        v = fp[:, :, :, k]
        fp[:, :, :, k] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    fp = np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dead = [FP_METRICS[k] for k in range(K) if float(fp[..., k].std()) < 1e-6]
    assert not dead, "fingerprint dims collapsed to a constant: %s" % dead
    print("fp: %d role-cells with n>=100, %d pooled cells"
          % (int((rn >= 100).sum()), int((pn >= 100).sum())))
    return fp


def load_pro():
    b = torch.load(os.path.join(ROOT, "data", "pro play", "pro_drafts.pt"),
                   map_location="cpu", weights_only=False)
    rows = b["rows"]
    champ = torch.tensor([r["champ"] for r in rows], dtype=torch.long)
    y = torch.tensor([float(r["blue_win"]) for r in rows], dtype=torch.float32)
    series = [r["series_id"] for r in rows]
    return champ[:, :5].contiguous(), champ[:, 5:].contiguous(), y, series


class Model(nn.Module):
    def __init__(self, kind, fp):
        super().__init__()
        self.kind = kind
        self.v = nn.Parameter(torch.zeros(NCH))
        if kind == "free":
            self.W = nn.Parameter(torch.zeros(NCH, NCH))
        if kind == "fp":
            self.theta = nn.Parameter(torch.zeros(len(FP_METRICS)))
            self.register_buffer("fp", torch.from_numpy(fp))

    def logit(self, b, r):
        z = self.v[b].sum(1) - self.v[r].sum(1)
        if self.kind == "free":
            z = z + self.W[b, r].sum(1)
        elif self.kind == "fp":
            ar = torch.arange(5)
            # only same-role pairs are real lane matchups: fp[role, b_i, r_i]
            f = self.fp[ar, b, r, :]                 # (N, 5, K)
            z = z + (f * self.theta).sum((1, 2))
        return z


def fit(model, b, r, y, tr, va, epochs=40, l2_int=0.1, lr=0.05, bs=2048):
    groups = [{"params": [model.v], "weight_decay": 1e-4}]
    for nm in ("W", "theta"):
        if hasattr(model, nm):
            groups.append({"params": [getattr(model, nm)], "weight_decay": l2_int})
    opt = torch.optim.Adam(groups, lr=lr)
    bt, rt, yt = b[tr], r[tr], y[tr]
    n = len(tr)
    best = (9e9, None)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(
                model.logit(bt[idx], rt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.binary_cross_entropy_with_logits(
                model.logit(b[va], r[va]), y[va]).item()
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best[1] is None:
        raise RuntimeError("no finite val loss for %s" % model.kind)
    model.load_state_dict(best[1])
    return model, best[0]


def main():
    t0 = time.time()
    fp = build_fp()
    b, r, y, series = load_pro()
    print("pro games: %d  base rate %.4f" % (len(y), float(y.mean())))

    # split by series: games in a series share a draft and a patch
    uniq = sorted(set(series))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    tr_s = set(uniq[:int(.7 * len(uniq))])
    va_s = set(uniq[int(.7 * len(uniq)):int(.85 * len(uniq))])
    is_tr = np.array([s in tr_s for s in series])
    is_va = np.array([s in va_s for s in series])
    is_te = ~(is_tr | is_va)
    tr = torch.tensor(np.where(is_tr)[0])
    va = torch.tensor(np.where(is_va)[0])
    te = torch.tensor(np.where(is_te)[0])
    print("series %d -> train %d / val %d / test %d games"
          % (len(uniq), len(tr), len(va), len(te)))

    base = float(y.mean())
    const = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    print("\n%-30s %10s %10s %8s %8s" %
          ("model", "params", "test", "acc", "val"))
    print("%-30s %10s %10.5f %8.4f %8.5f" %
          ("constant", "-", const, max(base, 1 - base), const))

    for kind, label, npar in (("main", "main effects (pro)", NCH),
                              ("free", "free cells (pro)", NCH + NCH * NCH),
                              ("fp", "soloQ fp + pro theta", NCH + len(FP_METRICS))):
        best = None
        for l2 in (1e-2, 1e-1, 1.0):
            m, vl = fit(Model(kind, fp), b, r, y, tr, va, l2_int=l2)
            if best is None or vl < best[0]:
                best = (vl, m, l2)
        vl, m, l2 = best
        with torch.no_grad():
            z = m.logit(b[te], r[te])
            tl = nn.functional.binary_cross_entropy_with_logits(z, y[te]).item()
            ta = ((z > 0).float() == y[te]).float().mean().item()
        print("%-30s %10d %10.5f %8.4f %8.5f  (l2=%g)" %
              (label, npar, tl, ta, vl, l2))
        if kind == "fp":
            print("    theta (standardised fp units):")
            for met, th in zip(FP_METRICS, m.theta.detach().tolist()):
                print("      %-36s %+.4f" % (met, th))
    print("\nelapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
