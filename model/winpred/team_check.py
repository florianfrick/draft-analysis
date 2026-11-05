"""How much of the pro outcome is team strength rather than draft?

Not a model - a measurement, and a cheap one, because it decides whether the
draft-only line is near its ceiling or not.

Three predictors on the *same* test rows as every other number in the README:

1. the side prior (the null);
2. **team strength alone** - each team's smoothed win rate, fitted on train rows
   only, differenced across the two sides and passed through a one-parameter
   logistic fitted on train;
3. **draft + team** - the draft model's own logit plus that team term, with the
   three coefficients fitted on *val* (never test), which is the fair way to add
   a feature to an already-trained model without re-running it.

If (2) is already well below (1) and (3) is well below the draft-only number,
then most of what a pro win predictor can know is who is playing, not what was
picked - which is worth knowing before spending effort on the draft encoder, and
worth saying out loud in any UI that shows the number.

    python model/winpred/team_check.py --ckpt model/checkpoints/winpred-pro-ce-d384-...

The team feature is computed from train rows only and the split is the usual
series-grouped one, so a team's test games never enter its own rate.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.winpred import data as D              # noqa: E402
from model.winpred import metrics as M           # noqa: E402


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def team_rate(table, train_idx, k=10.0):
    """Smoothed win rate per team, fitted on train rows only.

    `k` is the shrinkage: a team with n games is pulled toward the corpus rate
    with weight k, so a 2-game team is not treated as a 100 % winner. 10 games
    is about half a split; it is the one knob here and the result is not
    sensitive to it between 5 and 20.
    """
    rows, _ = D._pro_rows("2022", verbose=False)
    teams = {str(r["gameid"]): (str(r["blue_team"]), str(r["red_team"]))
             for r in rows}
    tb = np.array([teams[g][0] for g in table.game_id], dtype=object)
    tr = np.array([teams[g][1] for g in table.game_id], dtype=object)
    y = table.y.astype(np.float64)
    base = float(y[np.asarray(train_idx)].mean())

    wins, games = Counter(), Counter()
    for i in np.asarray(train_idx):
        b, r = tb[i], tr[i]
        games[b] += 1
        wins[b] += y[i]
        games[r] += 1
        wins[r] += 1.0 - y[i]
    rate = {t: (wins[t] + k * base) / (games[t] + k) for t in games}
    sb = np.array([rate.get(t, base) for t in tb])
    sr = np.array([rate.get(t, base) for t in tr])
    return _logit(sb), _logit(sr), len(games)


def fit_logistic(X, y, l2=1e-4, steps=400):
    """Plain logistic regression by LBFGS. Kept here so the file has no deps."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                + l2 * (w ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return (w.detach().numpy().copy(), float(b.detach()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="a winpred checkpoint dir holding val/test_preds.npz")
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--team-mode", default="rate",
                    choices=("rate", "bt", "roster", "player"),
                    help="which strength estimate to test: 'rate' is the "
                         "original marginal win rate, 'bt' Bradley-Terry, "
                         "'player' per-player ratings, 'roster' both blocks")
    ap.add_argument("--player-k", type=float, default=None,
                    help="shrinkage for the player block (default --k)")
    ap.add_argument("--seed", type=int, default=D.SPLIT_SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    table, meta = D.load_pro(corpus="2022")
    splits, _ = D.make_splits(table, seed=args.seed, verbose=False)
    tr_i, va_i, te_i = splits["train"], splits["val"], splits["test"]
    y = table.y.astype(np.float64)

    if args.team_mode == "rate":
        lb, lr, n_teams = team_rate(table, tr_i, k=args.k)
        x_team = (lb - lr).reshape(-1, 1)      # one column: the match-up
    else:
        feat = D.team_strength(table, tr_i, k=args.k, mode=args.team_mode,
                               kp=args.player_k, verbose=True)
        x_team = (feat[:, 0] - feat[:, 1]).astype(np.float64).reshape(-1, 1)
        n_teams = int(np.unique(feat[:, 0]).size)
    label = {"rate": "team", "bt": "bt", "roster": "roster",
             "player": "player"}[args.team_mode]
    print("%s with >=1 train game: %d of %d rows"
          % (label, n_teams, len(table)))

    # 1. the null
    prior = float(y[tr_i].mean())
    p_prior = np.full(len(te_i), prior)

    # 2. team strength alone
    w, b = fit_logistic(x_team[tr_i], y[tr_i])
    p_team = 1.0 / (1.0 + np.exp(-(x_team[te_i] @ w + b)))
    print("%s-only weights: %s (coef on the match-up term)" % (label,
                                                               np.round(w, 3)))

    # 3. draft, and draft + team (coefficients fitted on val)
    ck = Path(args.ckpt)
    va = D.load_predictions(ck / "val_preds.npz")
    te = D.load_predictions(ck / "test_preds.npz")
    va = D.align_predictions(va, [table.game_id[i] for i in va_i])
    te = D.align_predictions(te, [table.game_id[i] for i in te_i])
    l_draft_va, l_draft_te = _logit(va["p"]), _logit(te["p"])

    X_va = np.column_stack([l_draft_va, x_team[va_i][:, 0]])
    X_te = np.column_stack([l_draft_te, x_team[te_i][:, 0]])
    w2, b2 = fit_logistic(X_va, y[va_i])
    p_combo = 1.0 / (1.0 + np.exp(-(X_te @ w2 + b2)))
    print("draft+%s weights (fit on val): %s" % (label, np.round(w2, 3)))

    rows = [("prior", p_prior), (label + " only", p_team),
            ("draft only", te["p"].astype(np.float64)),
            ("draft + " + label, p_combo)]
    print("\n%-14s %10s %9s %9s" % ("test (n=%d)" % len(te_i), "log-loss",
                                    "Brier", "AUC"))
    out = {}
    for name, p in rows:
        rep = M.full_report(y[te_i], p)
        out[name] = rep
        print("%-14s %10.5f %9.5f %9.4f"
              % (name, rep["log_loss"], rep["brier"], rep["roc_auc"]))

    for name, p in ((label + " only", p_team), ("draft + " + label, p_combo)):
        d = M.bootstrap_delta(y[te_i], te["p"].astype(np.float64), p,
                              [table.series_id[i] for i in te_i],
                              metric="log_loss", n_boot=args.n_boot,
                              seed=args.seed)
        print("  %-12s vs draft only: dll %+.5f [%+.5f, %+.5f] %s"
              % (name, d["delta"], d["lo"], d["hi"],
                 "excludes 0" if d["excludes_zero"] else "straddles 0"))
        out[name + " vs draft"] = d

    if args.out:
        D.write_json(args.out, {"ckpt": str(ck), "k": args.k, "seed": args.seed,
                                "n_teams": n_teams, "report": out})
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
