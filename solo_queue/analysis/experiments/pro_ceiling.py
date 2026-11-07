"""How much of a pro game does the draft actually explain?

Same nested logistic design as the soloQ experiment, on Oracle's Elixir pro
games, time-split (train on the past, test on the future) so the number is an
honest forward-looking ceiling rather than a random-split artifact.

  A  singles          : champ strength only
  B  +synergy+counter : pairwise structure
  C  +team identity   : how much of the residual is "who is playing", not "what
                        was drafted" (upper bound on what a draft model can own)

Run: python "solo_queue/analysis/experiments/pro_ceiling.py"
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PRO = os.path.join(ROOT, "data", "pro play")


def load():
    frames = []
    for p in sorted(glob.glob(os.path.join(PRO, "20*_LoL_esports*.csv"))):
        df = pd.read_csv(p, low_memory=False,
                         usecols=["gameid", "date", "patch", "league", "side",
                                  "position", "champion", "result", "teamname",
                                  "datacompleteness"])
        df = df[df["datacompleteness"] == "complete"]
        df = df[df["position"].isin(["top", "jng", "mid", "bot", "sup"])]
        df["side"] = df["side"].str.lower()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["champion", "result", "side"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    games = []
    for gid, g in df.groupby("gameid", sort=False):
        b = g[g["side"] == "blue"]
        r = g[g["side"] == "red"]
        if len(b) != 5 or len(r) != 5:
            continue
        bw = b["result"].iloc[0]
        if r["result"].iloc[0] == bw:      # both sides same result -> broken row
            continue
        games.append({
            "gameid": gid, "date": g["date"].iloc[0], "patch": g["patch"].iloc[0],
            "league": g["league"].iloc[0],
            "blue": sorted(b["champion"].astype(str).tolist()),
            "red": sorted(r["champion"].astype(str).tolist()),
            "blue_win": int(bw),
            "blue_team": str(b["teamname"].iloc[0]),
            "red_team": str(r["teamname"].iloc[0]),
        })
    return pd.DataFrame(games)


def build(games, champ_idx, team_idx=None, use_syn=False, use_ctr=False):
    n_champ = len(champ_idx)
    n_pair = n_champ * (n_champ - 1) // 2
    pair_of = -np.ones((n_champ, n_champ), dtype=np.int64)
    pi, pj = np.triu_indices(n_champ, k=1)
    pair_of[pi, pj] = np.arange(n_pair)
    pair_of[pj, pi] = np.arange(n_pair)

    c_syn = n_champ if use_syn else 0
    c_ctr = n_champ + (n_pair if use_syn else 0)
    c_team = n_champ + (n_pair if use_syn else 0) + (n_pair if use_ctr else 0)
    n_team = 0 if team_idx is None else len(team_idx)
    n_col = c_team + n_team

    indptr, indices, data = [0], [], []
    y = np.zeros(len(games), dtype=np.float32)
    for k, row in enumerate(games.itertuples(index=False)):
        b = [champ_idx[c] for c in row.blue]
        r = [champ_idx[c] for c in row.red]
        y[k] = row.blue_win
        cols, vals = [], []
        for c in b:
            cols.append(c); vals.append(1.0)
        for c in r:
            cols.append(c); vals.append(-1.0)
        if use_syn:
            for t, s in ((b, 1.0), (r, -1.0)):
                for x in range(5):
                    for z in range(x + 1, 5):
                        cols.append(c_syn + pair_of[t[x], t[z]]); vals.append(s)
        if use_ctr:
            for x in b:
                for z in r:
                    lo, hi = (x, z) if x < z else (z, x)
                    cols.append(c_ctr + pair_of[lo, hi])
                    vals.append(1.0 if x < z else -1.0)
        if team_idx is not None:
            cols.append(c_team + team_idx[row.blue_team]); vals.append(1.0)
            cols.append(c_team + team_idx[row.red_team]); vals.append(-1.0)
        indices.extend(cols); data.extend(vals); indptr.append(len(indices))
    return sp.csr_matrix((np.array(data, np.float32),
                          np.array(indices, np.int32),
                          np.array(indptr, np.int64)), shape=(len(games), n_col)), y


def run(Xtr, ytr, Xte, yte, C, tag, out):
    clf = LogisticRegression(C=C, max_iter=300, solver="lbfgs")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc, ll = roc_auc_score(yte, p), log_loss(yte, p)
    print(f"  {tag:<26} C={C:<6} AUC={auc:.4f}  logloss={ll:.5f}  "
          f"({Xtr.shape[1]} feats)", flush=True)
    out.append({"tag": tag, "C": C, "auc": float(auc), "logloss": float(ll),
                "n_features": int(Xtr.shape[1])})


def main():
    df = load()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"pro games: {len(df)}  {df['date'].min().date()} .. {df['date'].max().date()}")
    print("patches:", df["patch"].nunique(), " leagues:", df["league"].nunique())

    cut_date = df["date"].quantile(0.80)
    tr = df[df["date"] < cut_date]
    te = df[df["date"] >= cut_date]
    print(f"train {len(tr)} (to {cut_date.date()})  test {len(te)}  "
          f"test base rate {te['blue_win'].mean():.4f}")

    champs = sorted(set(sum(df["blue"], []) + sum(df["red"], [])))
    champ_idx = {c: i for i, c in enumerate(champs)}
    teams = sorted(set(df["blue_team"]) | set(df["red_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    print(f"champions {len(champs)}  teams {len(teams)}")

    out = []
    for tag, syn, ctr, tm in [("A singles", False, False, False),
                              ("B +synergy+counter", True, True, False),
                              ("C +team identity", True, True, True)]:
        Xtr, ytr = build(tr, champ_idx, team_idx if tm else None, syn, ctr)
        Xte, yte = build(te, champ_idx, team_idx if tm else None, syn, ctr)
        for C in (0.02, 0.1):
            run(Xtr, ytr, Xte, yte, C, tag, out)

    # constant baseline
    p0 = np.full(len(te), tr["blue_win"].mean())
    print(f"  {'constant':<26} AUC=0.5000  "
          f"logloss={log_loss(te['blue_win'], p0):.5f}")

    with open(os.path.join(HERE, "pro_ceiling.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote pro_ceiling.json")


if __name__ == "__main__":
    main()
