"""Split-half reliability of pairwise cells, per target metric.

A cell estimate is only usable if it reproduces. Split the corpus into two
random halves, estimate every (champion, lane-opponent) effect in each
half independently, and correlate. The correlation is the ceiling on how
much any downstream explanation can trust that cell.

This separates two things the corpus conflates:
  - sparsity   (how many games a cell has)      -> fixable by pooling
  - reliability(how much signal a cell carries) -> not fixable

Win is a binary summary of 30 minutes of team chaos; lane metrics are
measured against the specific opponent and should carry far more signal
per game. If so, matchup analysis should target lane state, not outcome.

Usage:  python "soloq data analysis/reliability.py"
"""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "store")
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

METRICS = ["win", "maxCsAdvantageOnLaneOpponent", "maxLevelLeadLaneOpponent",
           "visionScoreAdvantageLaneOpponent", "goldPerMinute",
           "damagePerMinute", "deathsByEnemyChamps", "soloKills",
           "teamDamagePercentage", "killParticipation"]

_cb = pq.read_table(os.path.join(STORE, "champ_base.parquet"), columns=["champ"])
NAMES = {c: str(c) for c in _cb.column("champ").to_pylist()}


def load():
    cols = ["match_id", "side", "pos", "champ"] + METRICS
    df = pq.read_table(os.path.join(STORE, "players.parquet"),
                       columns=cols).to_pandas()
    df = df[df["pos"].isin(ROLES)]
    df = df.drop_duplicates(subset=["match_id", "side", "pos"])

    # lane opponent = the other side's champion in the same role
    opp = df[["match_id", "side", "pos", "champ"]].copy()
    opp["side"] = opp["side"].map({"blue": "red", "red": "blue"})
    opp = opp.rename(columns={"champ": "opp"})
    df = df.merge(opp, on=["match_id", "side", "pos"], how="inner")
    return df


def cell_effects(df, metric, min_n):
    """mean(metric | champ vs opp) - mean(metric | champ), per cell."""
    g = df.groupby(["champ", "opp"], observed=True)[metric]
    cell = g.agg(["mean", "count"])
    base = df.groupby("champ", observed=True)[metric].mean()
    cell["eff"] = cell["mean"] - cell.index.get_level_values("champ").map(base)
    return cell[cell["count"] >= min_n][["eff", "count"]]


def main():
    df = load()
    print("player-games with a lane opponent: %d" % len(df))
    print("matches: %d\n" % df["match_id"].nunique())

    # split by match so a game never contributes to both halves
    m = df["match_id"].unique()
    rng = np.random.default_rng(0)
    half_a = set(rng.choice(m, size=len(m) // 2, replace=False))
    is_a = df["match_id"].isin(half_a)
    A, B = df[is_a], df[~is_a]
    print("half A: %d games   half B: %d games\n"
          % (A["match_id"].nunique(), B["match_id"].nunique()))

    print("%-34s %6s %8s %8s %8s %8s" %
          ("metric", "cells", "r(n>=50)", "r(n>=200)", "r(n>=1000)", "sd"))
    rows = []
    for met in METRICS:
        out = []
        for min_n in (50, 200, 1000):
            a = cell_effects(A, met, min_n)
            b = cell_effects(B, met, min_n)
            j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            if len(j) < 30:
                out.append((np.nan, len(j)))
                continue
            out.append((np.corrcoef(j["eff_a"], j["eff_b"])[0, 1], len(j)))
        a50 = cell_effects(A, met, 50)
        rows.append((met, out, float(a50["eff"].std())))
        print("%-34s %6d %8.3f %8.3f %8.3f %8.3f" % (
            met, out[0][1], out[0][0], out[1][0], out[2][0],
            float(a50["eff"].std())))
    print("\n(each r is the correlation of the same cells estimated in "
          "disjoint halves;\n r is a hard ceiling on out-of-sample validity)")

    # how many games per cell until a metric is usable?  Spearman-Brown:
    # r_k = k*r/(1+(k-1)*r) for k times the data.
    print("\n--- games per cell needed for r = 0.7 (Spearman-Brown) ---")
    print("%-34s %10s %10s" % ("metric", "r@n=200", "games"))
    for met, out, sd in rows:
        r = out[1][0]
        if not np.isfinite(r) or r <= 0.7:
            need = ">200k" if not np.isfinite(r) else "%.0fk" % (
                0.2 * (0.7 * (1 - r)) / (r * 0.3) / 1000)
        else:
            need = "<=200"
        print("%-34s %10.3f %10s" % (met, r, need))


if __name__ == "__main__":
    main()
