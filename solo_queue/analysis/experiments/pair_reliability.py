"""Is the pairwise win-rate signal real, or noise?

If a pair's win-rate deviation survives a split-half test after removing the
two champions' own strength, the signal exists and is merely hard to estimate.
If it does not, pairwise win rates are a re-encoding of individual strength
plus noise, and neither a value function nor an attention bias built from them
can work.

Method
  1. Random half-split of games.
  2. Per half, per (champ, partner, kind): n and mean(win).
  3. Per half, per champ: n and mean(win) -> the additive baseline:
       same team ('with'):  E[win] ~ b_champ + b_partner - mu
       enemy team ('vs'):   E[win] ~ b_champ - b_partner + mu
  4. Residual = observed pair win rate - additive baseline, computed *within*
     each half so no cross-half leakage.
  5. Correlate the two halves' residuals over cells with enough games.

Run: python "solo_queue/analysis/experiments/pair_reliability.py"
"""
import json
import os

import duckdb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DB = os.path.join(ROOT, "solo_queue", "analysis", "store", "store.duckdb")


def main():
    con = duckdb.connect(DB, read_only=True)
    con.execute("SET threads=8")
    con.execute("""
      CREATE TEMP TABLE g AS
      SELECT match_id, side, champ, win, (hash(match_id) % 2) AS half
      FROM players
      WHERE patch = '16.20' AND queue_id IN (420, 440) AND win IS NOT NULL
    """)
    print("rows:", con.execute("SELECT count(*) FROM g").fetchone()[0])

    con.execute("""
      CREATE TEMP TABLE pair AS
      WITH s AS (
        SELECT match_id, side, list(champ ORDER BY champ) champs
        FROM g GROUP BY 1,2
      ), subj AS (
        SELECT g.half AS half, g.champ AS champ, unnest(o.champs) AS partner,
               CASE WHEN o.side = g.side THEN 'with' ELSE 'vs' END AS kind,
               g.win AS win
        FROM g JOIN s o ON o.match_id = g.match_id
      )
      SELECT half, champ, partner, kind, count(*) n, avg(win) wr
      FROM subj WHERE partner <> champ GROUP BY 1,2,3,4
    """)
    con.execute("""
      CREATE TEMP TABLE base AS
      SELECT half, champ, count(*) n, avg(win) wr FROM g GROUP BY 1,2
    """)
    con.execute("CREATE TEMP TABLE mu AS SELECT half, avg(win) m FROM g GROUP BY 1")
    print("pair cells:", con.execute("SELECT count(*) FROM pair").fetchone()[0])

    # de-duplicate enemy cells: keep champ < partner only (the mirror is the
    # same games with the label flipped)
    rows = con.execute("""
      WITH a AS (SELECT * FROM pair WHERE half = 0),
           b AS (SELECT * FROM pair WHERE half = 1),
           ba0 AS (SELECT champ, wr FROM base WHERE half = 0),
           ba1 AS (SELECT champ, wr FROM base WHERE half = 1),
           mu0 AS (SELECT m FROM mu WHERE half = 0),
           mu1 AS (SELECT m FROM mu WHERE half = 1)
      SELECT a.kind, a.n na, b.n nb, a.wr wra, b.wr wrb,
             x0.wr bra0, y0.wr brb0, x1.wr bra1, y1.wr brb1,
             mu0.m mu0, mu1.m mu1
      FROM a JOIN b USING (champ, partner, kind)
      JOIN ba0 x0 ON x0.champ = a.champ
      JOIN ba0 y0 ON y0.champ = a.partner
      JOIN ba1 x1 ON x1.champ = a.champ
      JOIN ba1 y1 ON y1.champ = a.partner
      CROSS JOIN mu0 CROSS JOIN mu1
      WHERE a.kind = 'with' OR a.champ < a.partner
    """).fetchall()
    con.close()
    print("matched cells:", len(rows))

    arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]]
                    for r in rows], dtype=float)
    (na, nb, wra, wrb, bra0, brb0, bra1, brb1, mu0, mu1) = arr.T
    kinds = np.array([r[0] for r in rows])

    # additive baseline per half
    add0 = np.where(kinds == "with", bra0 + brb0 - mu0, bra0 - brb0 + mu0)
    add1 = np.where(kinds == "with", bra1 + brb1 - mu1, bra1 - brb1 + mu1)
    res_a = wra - add0
    res_b = wrb - add1

    out = {}
    for kind in ("with", "vs"):
        m = kinds == kind
        for min_n in (30, 100, 300):
            sel = m & (na >= min_n) & (nb >= min_n)
            if sel.sum() < 100:
                continue
            raw = float(np.corrcoef(wra[sel], wrb[sel])[0, 1])
            res = float(np.corrcoef(res_a[sel], res_b[sel])[0, 1])
            # binomial noise floor on a single half's residual
            sd_res_a = float(np.std(res_a[sel]))
            noise = float(np.mean(np.sqrt(0.25 / na[sel])))
            out[f"{kind}_min{min_n}"] = {
                "cells": int(sel.sum()), "raw_corr": raw, "resid_corr": res,
                "sd_resid": sd_res_a, "binomial_noise_sd": noise,
                "median_n": float(np.median(na[sel])),
            }
            print(f"  {kind:<5} n>={min_n:<4} cells={int(sel.sum()):6d} "
                  f"raw_r={raw:+.3f} resid_r={res:+.3f} "
                  f"sd_resid={sd_res_a:.4f} binom_sd={noise:.4f}")

    # additive baseline quality per kind
    for kind in ("with", "vs"):
        m = (kinds == kind) & (na >= 100) & (nb >= 100)
        obs = (wra[m] + wrb[m]) / 2
        pred = (add0[m] + add1[m]) / 2
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        r2 = 1 - float(np.sum((obs - pred) ** 2)) / ss_tot
        out[f"additive_r2_{kind}"] = r2
        print(f"\nadditive baseline R2 for {kind}: {r2:+.3f} "
              f"over {int(m.sum())} cells (n>=100/half)")

    with open(os.path.join(HERE, "pair_reliability.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote pair_reliability.json")


if __name__ == "__main__":
    main()
