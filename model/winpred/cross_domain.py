"""Does a champion-strength prior fitted on one domain predict the other?

The cross-domain question from README 16. The encoder transfer test (`sweep.py`)
scores a whole checkpoint on both views, but a checkpoint carries a lot of
things that are domain-specific - patch embeddings, the side prior, 2 M
parameters of attention. This asks the narrower and more useful question: is
**champion strength itself** a domain-agnostic quantity?

Why that is the question worth asking: soloQ and pro share the champion
vocabulary exactly (`data/pro play/pipeline.py` builds its name -> idx map from
the soloQ `champ_mapping.json`), so a rating vector can be moved between them
with no alignment step. If a soloQ-fitted vector ranks pro games as well as a
pro-fitted one does, then champion strength is a property of the patch and the
draft, not of the league - which is a much stronger statement than "the encoder
does not transfer" and it is cheap to check.

Two numbers per cell, because they answer different things:

* **AUC** is calibration-free: it is the ranking a vector produces on a domain
  it has never been fitted on. This is the transfer statement.
* **log-loss** after refitting the *scale* on the target's own train rows: a
  vector can rank well and still be wildly overconfident on a domain whose games
  are more predictable. Refitting one coefficient (and an intercept) on target
  train rows is the minimum honest correction, and it never touches test.

    python model/winpred/cross_domain.py --by-role --out reports/xdomain.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.winpred import data as D                      # noqa: E402
from model.winpred import metrics as M                   # noqa: E402
from model.winpred.team_check import fit_logistic, _logit  # noqa: E402


def _fit_on(table, splits, k, by_role):
    """Champion ratings fitted on this domain's train rows only."""
    _feat, rating = D.champion_strength(table, splits["train"], k=k,
                                        by_role=by_role, verbose=True,
                                        return_ratings=True)
    return rating


def _score(table, splits, rating, by_role):
    """Apply `rating` to `table`'s rows; report ranking and refitted log-loss."""
    feat = D.apply_champion_ratings(table, rating, by_role=by_role)
    x = (feat[:, 0] - feat[:, 1]).astype(np.float64).reshape(-1, 1)
    y = table.y.astype(np.float64)
    tr, te = splits["train"], splits["test"]

    # 1. ranking only - no fitting at all, so nothing can leak
    p_rank = 1.0 / (1.0 + np.exp(-x[te, 0]))
    auc = M.roc_auc(y[te], p_rank)

    # 2. one coefficient + intercept, fitted on TARGET train rows
    w, b = fit_logistic(x[tr], y[tr])
    p_cal = 1.0 / (1.0 + np.exp(-(x[te] @ w + b)))
    rep = M.full_report(y[te], p_cal)
    return {"auc_rank_only": float(auc), "log_loss": rep["log_loss"],
            "brier": rep["brier"], "accuracy": rep["accuracy"],
            "coef": float(w[0]), "intercept": float(b), "n": int(te.size)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champ-k", type=float, default=20.0)
    ap.add_argument("--by-role", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=D.SPLIT_SEED)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    tables, splits = {}, {}
    for dom in ("soloq", "pro"):
        table, _ = (D.load_soloq() if dom == "soloq"
                    else D.load_pro(corpus="2022"))
        tables[dom], splits[dom] = table, D.make_splits(table, seed=args.seed,
                                                        verbose=False)[0]
        print("%s: %d rows, blue %.4f" % (dom, len(table),
                                          float(table.y.mean())))

    print("\nfitting champion ratings on each domain's own train rows "
          "(by_role=%s, k=%.0f)" % (args.by_role, args.champ_k))
    ratings = {dom: _fit_on(tables[dom], splits[dom], args.champ_k,
                            args.by_role) for dom in tables}

    out = {}
    for target in ("soloq", "pro"):
        y = tables[target].y.astype(np.float64)
        te = splits[target]["test"]
        prior = float(y[splits[target]["train"]].mean())
        p_prior = np.full(te.size, prior)
        rep = M.full_report(y[te], p_prior)
        out["%s:prior" % target] = {"log_loss": rep["log_loss"],
                                    "auc_rank_only": 0.5, "n": int(te.size)}
        print("\n== target %s (test n=%d, prior %.5f, prior ll %.5f)"
              % (target, te.size, prior, rep["log_loss"]))
        for source in ("soloq", "pro"):
            r = _score(tables[target], splits[target], ratings[source],
                       args.by_role)
            out["%s:%s-fitted" % (target, source)] = r
            print("   ratings fitted on %-6s -> AUC %.4f  ll %.5f  brier "
                  "%.5f  (coef %.3f)" % (source, r["auc_rank_only"],
                                         r["log_loss"], r["brier"], r["coef"]))

    for target in ("soloq", "pro"):
        a = out["%s:soloq-fitted" % target]
        b = out["%s:pro-fitted" % target]
        print("\n%s: soloQ-fitted vs pro-fitted -> dAUC %+.4f  dll %+.5f"
              % (target, a["auc_rank_only"] - b["auc_rank_only"],
                 a["log_loss"] - b["log_loss"]))

    if args.out:
        D.write_json(args.out, {"by_role": args.by_role, "champ_k": args.champ_k,
                                "seed": args.seed, "results": out})
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
