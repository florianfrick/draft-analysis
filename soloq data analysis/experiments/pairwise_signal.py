"""Does pairwise (synergy/counter) structure add predictive signal beyond
individual champion strength?

Design: a draft is 5 blue champs vs 5 red champs. Fit logistic regression on
P(blue wins) with three nested feature sets, all symmetric under side swap:

  A  singles   : +1 per blue champ, -1 per red champ                 (C params)
  B  +synergy  : same-team unordered pair, +1 both blue / -1 both red (C(C-1)/2)
  C  +counter  : cross-team pair, +1 if the lower id is on blue       (C(C-1)/2)

If held-out AUC does not improve from A to B/C, then pairwise win-rate
structure is (mostly) a re-encoding of individual strength, and a value
function built from pairwise win rates cannot beat a champ-strength model.

Run:  python "soloq data analysis/experiments/pairwise_signal.py"
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DB = os.path.join(ROOT, "soloq data analysis", "store", "store.duckdb")


def load(patch="16.20", queues=(420, 440)):
    con = duckdb.connect(DB, read_only=True)
    q = ",".join(str(x) for x in queues)
    rows = con.execute(f"""
        SELECT blue_champs, red_champs, blue_win
        FROM drafts
        WHERE patch = ? AND queue_id IN ({q}) AND blue_win IS NOT NULL
    """, [patch]).fetchall()
    con.close()
    return rows


def build(rows, champ_idx, n_champ, use_syn=False, use_ctr=False):
    n = len(rows)
    pair_i, pair_j = np.triu_indices(n_champ, k=1)
    pair_of = -np.ones((n_champ, n_champ), dtype=np.int64)
    pair_of[pair_i, pair_j] = np.arange(len(pair_i))
    pair_of[pair_j, pair_i] = np.arange(len(pair_i))
    n_pair = len(pair_i)

    # column layout
    c_single = 0
    c_syn = n_champ if use_syn else None
    c_ctr = (n_champ + n_pair) if use_ctr else None
    n_col = n_champ + (n_pair if use_syn else 0) + (n_pair if use_ctr else 0)

    indptr, indices, data = [0], [], []
    y = np.zeros(n, dtype=np.float32)
    for k, (blue, red, win) in enumerate(rows):
        b = [champ_idx[c] for c in blue]
        r = [champ_idx[c] for c in red]
        y[k] = win
        cols, vals = [], []
        for c in b:
            cols.append(c_single + c); vals.append(1.0)
        for c in r:
            cols.append(c_single + c); vals.append(-1.0)
        if use_syn:
            for t, sign in ((b, 1.0), (r, -1.0)):
                for x in range(5):
                    for z in range(x + 1, 5):
                        cols.append(c_syn + pair_of[t[x], t[z]]); vals.append(sign)
        if use_ctr:
            for x in b:
                for z in r:
                    lo, hi = (x, z) if x < z else (z, x)
                    cols.append(c_ctr + pair_of[lo, hi])
                    vals.append(1.0 if x < z else -1.0)
        indices.extend(cols); data.extend(vals); indptr.append(len(indices))
    X = sp.csr_matrix((np.array(data, dtype=np.float32),
                       np.array(indices, dtype=np.int32),
                       np.array(indptr, dtype=np.int64)),
                      shape=(n, n_col))
    return X, y


def fit_eval(Xtr, ytr, Xte, yte, C=1.0, tag=""):
    t0 = time.time()
    clf = LogisticRegression(C=C, max_iter=200, solver="lbfgs", n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    ll = log_loss(yte, p)
    print(f"  {tag:<22} AUC={auc:.4f}  logloss={ll:.5f}  "
          f"({time.time()-t0:.0f}s, {Xtr.shape[1]} feats)", flush=True)
    return {"tag": tag, "auc": float(auc), "logloss": float(ll),
            "n_features": int(Xtr.shape[1])}


def main():
    rows = load()
    print(f"games: {len(rows)}")
    champs = sorted({c for b, r, _ in rows for c in (list(b) + list(r))})
    champ_idx = {c: i for i, c in enumerate(champs)}
    n_champ = len(champs)
    print(f"champions: {n_champ}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rows))
    cut = int(0.8 * len(rows))
    tr = [rows[i] for i in perm[:cut]]
    te = [rows[i] for i in perm[cut:]]
    print(f"train {len(tr)}  test {len(te)}")
    print(f"test base rate {np.mean([x[2] for x in te]):.4f}")

    out = []
    for name, syn, ctr in [("A singles", False, False),
                           ("B singles+synergy", True, False),
                           ("C singles+syn+ctr", True, True)]:
        Xtr, ytr = build(tr, champ_idx, n_champ, syn, ctr)
        Xte, yte = build(te, champ_idx, n_champ, syn, ctr)
        for C in (0.03, 0.1):
            out.append(fit_eval(Xtr, ytr, Xte, yte, C=C,
                                tag=f"{name} C={C}"))

    with open(os.path.join(HERE, "pairwise_signal.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote pairwise_signal.json")


if __name__ == "__main__":
    main()
