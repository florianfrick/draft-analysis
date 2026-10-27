"""Does role-conditioning rescue the pairwise (counter) signal?

The pooled model in additive_model.py spreads each blue champion over all
five red champions. If counter effects are lane-specific -- and they
should be, since only your lane opponent trades with you in the first
14 minutes -- then 4 of every 5 pairs are noise and the real effect is
diluted 5x while the noise is multiplied 5x. That is a representation
failure, not an absence of signal.

This script rebuilds the same games with roles attached and compares:

  M0  main effects only
  M1  main + lane counters      (5 pairs: same role only)
  M2  main + lane + cross-role  (all 25 pairs)
  M3  role-specific main effects + lane counters

Usage:  python "soloq data analysis/role_test.py"
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

_cb = pq.read_table(os.path.join(STORE, "champ_base.parquet"), columns=["champ"])
RIOT_IDS = sorted(set(_cb.column("champ").to_pylist()))
ID2IX = {c: i for i, c in enumerate(RIOT_IDS)}
NCH = len(RIOT_IDS)


def load():
    t = pq.read_table(os.path.join(STORE, "players.parquet"),
                      columns=["match_id", "side", "pos", "champ", "win"])
    df = t.to_pandas()
    df = df[df["pos"].isin(ROLES)]
    # 5,982 matches appear twice (two sources); keep one row per side+role.
    df = df.drop_duplicates(subset=["match_id", "side", "pos"])
    wide = df.pivot_table(index="match_id", columns=["side", "pos"],
                          values="champ", aggfunc="first")
    win = df.drop_duplicates("match_id").set_index("match_id")["win"]
    wide = wide[wide.index.isin(win.index)]
    wide.columns = pd.MultiIndex.from_tuples(wide.columns, names=["side", "pos"])
    wide["_win"] = win.reindex(wide.index).to_numpy()
    wide = wide.dropna()
    print("matches with all 10 roles: %d" % len(wide))

    blue = wide[[("blue", r) for r in ROLES]].to_numpy(dtype=np.int64)
    red = wide[[("red", r) for r in ROLES]].to_numpy(dtype=np.int64)
    y = torch.tensor(wide["_win"].to_numpy(dtype=np.float32))
    b = torch.tensor([[ID2IX[c] for c in row] for row in blue.tolist()])
    r = torch.tensor([[ID2IX[c] for c in row] for row in red.tolist()])
    return b, r, y


class RoleModel(nn.Module):
    def __init__(self, lane=True, cross=False, role_main=False):
        super().__init__()
        self.lane, self.cross = lane, cross
        self.v = nn.Parameter(torch.zeros(len(ROLES), NCH) if role_main
                              else torch.zeros(NCH))
        self.role_main = role_main
        self.WL = nn.Parameter(torch.zeros(NCH, NCH)) if lane else None
        self.WX = nn.Parameter(torch.zeros(NCH, NCH)) if cross else None

    def logit(self, b, r):
        if self.role_main:
            z = self.v[torch.arange(5), b].sum(1) - self.v[torch.arange(5), r].sum(1)
        else:
            z = self.v[b].sum(1) - self.v[r].sum(1)
        if self.WL is not None:
            z = z + self.WL[b, r].sum(1)          # b[:,i] vs r[:,i]: same role
        if self.WX is not None:
            z = z + self.WX[b[:, :, None], r[:, None, :]].sum((1, 2)) \
                  - self.WL[b, r].sum(1)
        return z


def fit(model, b, r, y, tr, va, epochs=30, l2_int=0.1, l2_main=1e-5, lr=0.05):
    groups = [{"params": [model.v], "weight_decay": l2_main}]
    for p in (model.WL, model.WX):
        if p is not None:
            groups.append({"params": [p], "weight_decay": l2_int})
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
    b, r, y = load()
    n = len(y)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    tr, va, te = perm[:int(.7 * n)], perm[int(.7 * n):int(.85 * n)], perm[int(.85 * n):]
    base = float(y.mean())
    const = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    print("games %d  base rate %.4f  constant logloss %.5f\n" % (n, base, const))
    print("%-34s %11s %11s %8s" % ("model", "val", "test", "acc"))
    print("%-34s %11.5f %11.5f %8.4f" % ("constant", const, const, max(base, 1 - base)))

    configs = [("M0  main effects", dict(lane=False)),
               ("M1  + lane counters", dict(lane=True)),
               ("M2  + lane + cross-role", dict(lane=True, cross=True)),
               ("M3  role main + lane", dict(lane=True, role_main=True))]
    out = {}
    for label, kw in configs:
        best = None
        for l2 in (3e-2, 1e-1, 3e-1):
            m, vl = fit(RoleModel(**kw), b, r, y, tr, va, l2_int=l2)
            if best is None or vl < best[0]:
                best = (vl, l2, m)
        vl, l2, m = best
        with torch.no_grad():
            tl = nn.functional.binary_cross_entropy_with_logits(
                m.logit(b[te], r[te]), y[te]).item()
            ta = ((m.logit(b[te], r[te]) > 0).float() == y[te]).float().mean().item()
        out[label] = (vl, tl, ta, l2)
        print("%-34s %11.5f %11.5f %8.4f  (l2=%g)" % (label, vl, tl, ta, l2))

    print("\n--- does role-conditioning help? (test) ---")
    m0 = out["M0  main effects"]
    for label, (vl, tl, ta, l2) in out.items():
        if label.startswith("M0"):
            continue
        print("  %-26s %+.5f logloss (%+.4f bits/game)  %+.2f pp acc"
              % (label, m0[1] - tl, (m0[1] - tl) / np.log(2), 100 * (ta - m0[2])))
    print("\nelapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
