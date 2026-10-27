"""Is the lane fingerprint the sufficient statistic for a matchup?

Two ways to represent "champion A against champion B in role R":

  free      W[a,b]  -- one coefficient per cell (29,929 params)
  fingerprint  theta . fp(a,b)  -- 6 params, fp estimated on disjoint data

If the fingerprint matches the free model, then matchup knowledge lives in
a handful of reliable lane dimensions rather than in cell win rates, and
you can estimate a new patch's matchups from ~200 games instead of ~1000.

Cross-fitted: fp is estimated on half A, the win model is fit and tested on
half B, so no cell contributes to both its own feature and its own label.

Usage:  python "soloq data analysis/fingerprint_test.py"
"""
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "store")
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
FP_METRICS = ["maxCsAdvantageOnLaneOpponent", "maxLevelLeadLaneOpponent",
              "goldPerMinute", "damagePerMinute", "teamDamagePercentage",
              "killParticipation"]

_cb = pq.read_table(os.path.join(STORE, "champ_base.parquet"), columns=["champ"])
RIOT = sorted(set(_cb.column("champ").to_pylist()))
ID2IX = {c: i for i, c in enumerate(RIOT)}
NCH = len(RIOT)


def load():
    cols = ["match_id", "side", "pos", "champ", "win"] + FP_METRICS
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
    return df.dropna(subset=["ci", "oi", "win"])


def build_fp(df):
    """fp[c, o, k] = cell mean minus champion mean, for each metric."""
    fp = np.zeros((NCH, NCH, len(FP_METRICS)), dtype=np.float32)
    n = np.zeros((NCH, NCH), dtype=np.int32)
    for k, met in enumerate(FP_METRICS):
        cell = df.groupby(["ci", "oi"], observed=True)[met].agg(["mean", "count"])
        base = df.groupby("ci", observed=True)[met].mean()
        eff = (cell["mean"] - cell.index.get_level_values("ci").map(base))
        # NaN cell means (no non-null observation) must not reach the
        # standardisation: a single NaN makes v.mean() NaN and zeroes the
        # whole dimension, which is how 4 of 6 dims silently died.
        eff = eff.fillna(0.0)
        ci = cell.index.get_level_values("ci").to_numpy()
        oi = cell.index.get_level_values("oi").to_numpy()
        fp[ci, oi, k] = eff.to_numpy()
        if k == 0:
            n[ci, oi] = cell["count"].to_numpy()
    # standardise each dimension so theta is comparable across metrics
    for k in range(len(FP_METRICS)):
        v = fp[:, :, k]
        fp[:, :, k] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    # cells the metric never observed stay at the mean rather than NaN
    fp = np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dead = [FP_METRICS[k] for k in range(len(FP_METRICS))
            if float(fp[..., k].std()) < 1e-6]
    assert not dead, "fingerprint dims collapsed to a constant: %s" % dead
    return fp, n


class Model(nn.Module):
    def __init__(self, kind, fp, n_cells):
        super().__init__()
        self.kind = kind
        self.v = nn.Parameter(torch.zeros(NCH))
        if kind == "free":
            self.W = nn.Parameter(torch.zeros(NCH, NCH))
        elif kind == "fp":
            self.theta = nn.Parameter(torch.zeros(len(FP_METRICS)))
            self.register_buffer("fp", torch.from_numpy(fp))
        self.n_cells = n_cells

    def logit(self, b, r):
        z = self.v[b].sum(1) - self.v[r].sum(1)
        if self.kind == "free":
            z = z + self.W[b, r].sum(1)
        elif self.kind == "fp":
            f = self.fp[b, r]                       # (N, 5 roles, K)
            z = z + (f * self.theta).sum((1, 2))
        return z


def fit(model, b, r, y, tr, va, epochs=30, l2_int=0.1, lr=0.05):
    groups = [{"params": [model.v], "weight_decay": 1e-5}]
    for name in ("W", "theta"):
        if hasattr(model, name):
            groups.append({"params": [getattr(model, name)], "weight_decay": l2_int})
    opt = torch.optim.Adam(groups, lr=lr)
    bt, rt, yt = b[tr], r[tr], y[tr]
    n, bs = len(tr), 8192
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
    model.load_state_dict(best[1])
    return model, best[0]


def main():
    t0 = time.time()
    df = load()
    print("player-games: %d   matches: %d" % (len(df), df["match_id"].nunique()))

    m = df["match_id"].unique()
    rng = np.random.default_rng(0)
    half_a = set(rng.choice(m, size=len(m) // 2, replace=False))
    A = df[df["match_id"].isin(half_a)]
    B = df[~df["match_id"].isin(half_a)]
    fp, celln = build_fp(A)
    print("fingerprint cells estimated on half A: %d with n>=100"
          % int((celln >= 100).sum()))

    # pivot half B to one row per match with role-aligned champs
    wide = B.pivot_table(index="match_id", columns=["side", "pos"],
                         values="champ", aggfunc="first")
    yv = B.drop_duplicates("match_id").set_index("match_id")["win"]
    wide = wide[wide.index.isin(yv.index)].dropna()
    yv = yv.reindex(wide.index)
    blue = wide[[("blue", r) for r in ROLES]].to_numpy()
    red = wide[[("red", r) for r in ROLES]].to_numpy()
    b = torch.tensor([[ID2IX[c] for c in row] for row in blue])
    r = torch.tensor([[ID2IX[c] for c in row] for row in red])
    y = torch.tensor(yv.to_numpy(dtype=np.float32))
    n = len(y)
    print("half B games: %d  base rate %.4f" % (n, float(y.mean())))

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    tr, va, te = perm[:int(.7 * n)], perm[int(.7 * n):int(.85 * n)], perm[int(.85 * n):]

    print("\n%-22s %10s %10s %8s %8s" %
          ("model", "params", "test", "acc", "val"))
    for kind, label, params in (("main", "main effects only", NCH),
                                ("free", "free cells W[a,b]", NCH + NCH * NCH),
                                ("fp", "fingerprint theta", NCH + len(FP_METRICS))):
        best = None
        for l2 in (3e-2, 1e-1, 3e-1):
            mod, vl = fit(Model(kind, fp, NCH), b, r, y, tr, va, l2_int=l2)
            if best is None or vl < best[0]:
                best = (vl, mod, l2)
        vl, mod, l2 = best
        with torch.no_grad():
            z = mod.logit(b[te], r[te])
            tl = nn.functional.binary_cross_entropy_with_logits(z, y[te]).item()
            ta = ((z > 0).float() == y[te]).float().mean().item()
        print("%-22s %10d %10.5f %8.4f %8.5f" % (label, params, tl, ta, vl))
        if kind == "fp":
            print("    theta (standardised units):")
            for met, th in zip(FP_METRICS, mod.theta.detach().tolist()):
                print("      %-36s %+.4f" % (met, th))
    print("\nelapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
