"""Would a *denoised* pairwise bias help, i.e. the version you would actually
feed an attention block?

Full-rank pairwise features lost to champ-strength-only (0.5449 -> 0.5353)
because 15k free parameters on 400k games is mostly variance. The attention
analogue is different: a fixed bias matrix W is added to attention logits, so
what matters is whether the *reliable part* of W carries signal.

This script isolates that:
  1. Fit the additive baseline on TRAIN (champ strength only).
  2. Residual matrix R[i,j] = observed pair win-rate deviation from baseline.
  3. Truncated SVD to rank k, reconstruct W_k  (the denoised bias).
  4. Score each TEST game with  bias = sum of W_k over its 10 same-team pairs
     (synergy) and 25 cross-team pairs (counter), signed by side.
  5. Logistic on [singles, bias] vs [singles] alone.

If AUC does not move, a pairwise attention bias cannot help and the effort is
better spent elsewhere.

Run: python "solo_queue/analysis/experiments/pair_bias_rank.py"
"""
import json
import os

import duckdb
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DB = os.path.join(ROOT, "solo_queue", "analysis", "store", "store.duckdb")


def main():
    con = duckdb.connect(DB, read_only=True)
    con.execute("SET threads=8")
    games = con.execute("""
        SELECT blue_champs, red_champs, blue_win, (hash(match_id) % 5) AS fold
        FROM drafts
        WHERE patch = '16.20' AND queue_id IN (420, 440) AND blue_win IS NOT NULL
    """).fetchall()
    con.close()
    print("games:", len(games))

    champs = sorted({c for b, r, _, _ in games for c in list(b) + list(r)})
    cidx = {c: i for i, c in enumerate(champs)}
    C = len(champs)
    print("champions:", C)

    B = np.array([[cidx[c] for c in b] for b, _, _, _ in games])
    R = np.array([[cidx[c] for c in r] for _, r, _, _ in games])
    y = np.array([w for _, _, w, _ in games], dtype=np.float64)
    fold = np.array([f for _, _, _, f in games])

    tr = fold != 0
    te = fold == 0
    print(f"train {tr.sum()}  test {te.sum()}  test base {y[te].mean():.4f}")

    # --- additive baseline from train (champ strength only) -----------------
    # per game the 10 entries are [blue x5, red x5]; values must interleave the
    # same way as the columns, hence the (n, 5) blocks rather than a flat concat
    rows = np.repeat(np.arange(len(y)), 10)
    cols = np.concatenate([B, R], axis=1).ravel()
    vals = np.concatenate([np.ones((len(y), 5)),
                           -np.ones((len(y), 5))], axis=1).ravel()
    Xs = sp.csr_matrix((vals, (rows, cols)), shape=(len(y), C))
    base = LogisticRegression(C=0.1, max_iter=300).fit(Xs[tr], y[tr])
    strength = base.coef_.ravel()
    print(f"singles-only test AUC = {roc_auc_score(y[te], base.predict_proba(Xs[te])[:, 1]):.4f}")

    # --- residual pair matrix on TRAIN -------------------------------------
    mu = y[tr].mean()
    Sw = np.zeros((C, C)); Nw = np.zeros((C, C))
    Sv = np.zeros((C, C)); Nv = np.zeros((C, C))
    for b, r, w in zip(B[tr], R[tr], y[tr]):
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                Sw[b[i], b[j]] += w; Nw[b[i], b[j]] += 1
        for i in b:
            for j in r:
                Sv[i, j] += w; Nv[i, j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        Wraw = np.where(Nw > 0, Sw / np.maximum(Nw, 1), np.nan)
        Vraw = np.where(Nv > 0, Sv / np.maximum(Nv, 1), np.nan)
    # additive baseline for the same cell (directed)
    base_w = strength[:, None] + strength[None, :] - mu
    base_v = strength[:, None] - strength[None, :] + mu
    Rw = Wraw - base_w
    Rv = Vraw - base_v

    def lowrank(Rm, Nm, k):
        m = Nm > 0
        Rf = np.where(m, Rm, 0.0)
        w = np.sqrt(np.maximum(Nm, 1)) * m
        U, s, Vt = np.linalg.svd(Rf * w, full_matrices=False)
        keep = min(k, len(s))
        rec = (U[:, :keep] * s[:keep]) @ Vt[:keep]
        # undo the weighting so the reconstruction is on the win-rate scale
        rec = np.where(m, rec / np.maximum(w, 1), 0.0)
        return rec

    out = []
    for k in (1, 4, 8, 16, 32):
        recw = lowrank(Rw, Nw, k)
        recv = lowrank(Rv, Nv, k)

        def score(Bx, Rx):
            sw = np.zeros(len(Bx)); sv = np.zeros(len(Bx))
            for g in range(len(Bx)):
                b, r = Bx[g], Rx[g]
                s = 0.0
                for i in range(5):
                    for j in range(5):
                        if i != j:
                            s += recw[b[i], b[j]]
                sw[g] = s
                t = 0.0
                for i in b:
                    for j in r:
                        t += recv[i, j]
                sv[g] = t
            return sw, sv

        sw_tr, sv_tr = score(B[tr], R[tr])
        sw_te, sv_te = score(B[te], R[te])
        Xtr = sp.hstack([Xs[tr], sp.csr_matrix(np.c_[sw_tr, sv_tr])]).tocsr()
        Xte = sp.hstack([Xs[te], sp.csr_matrix(np.c_[sw_te, sv_te])]).tocsr()
        clf = LogisticRegression(C=0.1, max_iter=400).fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y[te], p)
        out.append({"rank": k, "auc": float(auc),
                    "syn_w": float(clf.coef_.ravel()[-2]),
                    "ctr_w": float(clf.coef_.ravel()[-1])})
        print(f"  rank {k:<3} AUC={auc:.4f}  "
              f"coef(syn)={clf.coef_.ravel()[-2]:+.4f} "
              f"coef(ctr)={clf.coef_.ravel()[-1]:+.4f}")

    # reference: singles alone, same fold
    print(f"  {'singles only':<8} AUC="
          f"{roc_auc_score(y[te], base.predict_proba(Xs[te])[:, 1]):.4f}")
    with open(os.path.join(HERE, "pair_bias_rank.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote pair_bias_rank.json")


if __name__ == "__main__":
    main()
