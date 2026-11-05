"""Paired comparison of two saved prediction sets, on the same rows.

The three existing tools answer three different questions and none of them
answers this one:

* `evaluate.py`  - is this checkpoint better than the *prior*?
* `sweep.py`     - how does every checkpoint score on every *view*?
* this file      - is arm B better than arm A?

That last one needs a **paired** test: the same games, resampled at the series
level, with the two models' losses differenced inside each draw. Comparing two
marginal intervals would be wrong (it throws away the correlation between the
arms, which is most of the variance here).

The alignment rule is the whole content of the file, which is why it is a file
and not a flag. `data.align_predictions` refuses two prediction sets whose game
ids differ - no silent intersection - because intersecting would let two models
be compared on the rows they happen to share, which is exactly the comparison
that flatters the worse model. This was written by hand twice in throwaway
scripts before, once with an O(n^2) lookup that had to be killed mid-run.

Usage
-----
    python model/winpred/compare_arms.py \
        --ref v1=model/checkpoints/winpred-pro-ce-seed20261105-.../test_preds.npz \
        --new v2=model/checkpoints/winpred-pro-ce-v2-seed20261105-.../test_preds.npz \
        --n-boot 2000 --out model/winpred/reports/arch_v2_pro.json

The sign convention matches `bootstrap_delta`: `delta = new - ref`, so a
**negative** log-loss delta means the new arm is better.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.winpred import data as D              # noqa: E402
from model.winpred import metrics as M           # noqa: E402

DECOMPOSABLE = ("log_loss", "brier", "accuracy")


def parse_named(items):
    """`NAME=PATH` -> [(name, path)]; a bare path takes its folder name."""
    out = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name, path = Path(item).resolve().parent.name, item
        out.append((name, Path(path)))
    return out


def load_arm(spec):
    """`NAME=PATH[,PATH...]` -> (name, preds, n_members). Several paths averaged.

    Averaging happens here rather than in a new tool because it is the same
    problem: align every member onto the first member's rows, then reduce. The
    members must have been scored on identical rows - that is the contract
    `align_predictions` enforces - which is exactly why an ensemble needs
    `--init-seed` (same split) rather than `--seed` (different splits).

    The member count is returned *beside* the prediction dict, never inside it:
    every value in that dict is a per-row array that `align_predictions` will
    index by the row order, so an int in there raises "'int' object is not
    subscriptable" at the first alignment.
    """
    name, path = parse_named([spec])[0]
    paths = [Path(p) for p in str(path).split(",")]
    base = D.load_predictions(paths[0])
    if len(paths) == 1:
        return name, base, 1
    ps = [np.asarray(base["p"], dtype=np.float64)]
    for extra in paths[1:]:
        other = D.align_predictions(D.load_predictions(extra), base["game_id"])
        ps.append(np.asarray(other["p"], dtype=np.float64))
    out = dict(base)
    out["p"] = np.mean(ps, axis=0)
    return name, out, len(ps)


def groups_of(pred):
    """Resample whole series where the file has them, else whole games.

    Pro rows carry a real series id (a best-of-n is one unit of information, so
    treating its games as independent overstates n); soloQ rows are one game
    each, and `save_predictions` writes the game id into `series_id` there.
    Detecting this from the data rather than a flag keeps the caller honest -
    there is no way to accidentally claim game-level independence on pro.
    """
    sid = [str(s) for s in pred["series_id"]]
    gid = [str(g) for g in pred["game_id"]]
    if len(set(sid)) < len(sid):
        return sid, "series"
    return gid, "game"


def compare(ref_name, ref, new_name, new, n_boot, seed):
    """`new` aligned onto `ref`'s rows, then differenced draw by draw."""
    new = D.align_predictions(new, ref["game_id"])
    if not np.array_equal(np.asarray(ref["y"]), np.asarray(new["y"])):
        raise SystemExit("labels disagree after alignment - refusing")
    y = np.asarray(ref["y"], dtype=np.float64)
    p_ref = np.asarray(ref["p"], dtype=np.float64)
    p_new = np.asarray(new["p"], dtype=np.float64)
    groups, unit = groups_of(ref)

    rep_ref = M.full_report(y, p_ref)
    rep_new = M.full_report(y, p_new)
    delta = {m: M.bootstrap_delta(y, p_ref, p_new, groups, metric=m,
                                  n_boot=n_boot, seed=seed)
             for m in DECOMPOSABLE}
    return {
        "n": int(y.size),
        "unit": unit,
        "n_units": len(set(groups)),
        "ref": {"name": ref_name, "report": rep_ref},
        "new": {"name": new_name, "report": rep_new},
        "delta": delta,
        # AUC is not a per-row sum, so it gets a point estimate only - the
        # same rule `metrics.bootstrap_delta` enforces by raising.
        "auc_delta": rep_new["roc_auc"] - rep_ref["roc_auc"],
        "ref_side_bias_logit": _side(p_ref),
        "new_side_bias_logit": _side(p_new),
    }


def _side(p):
    """Mean logit, i.e. where the model put the side prior on these rows."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(np.log(p / (1.0 - p))))


def format_result(r):
    lines = []
    a, b = r["ref"], r["new"]
    lines.append("%-10s vs %-10s  n=%d  (%d %s)" % (
        b["name"], a["name"], r["n"], r["n_units"], r["unit"]))
    lines.append("%-14s %10s %10s %12s" % ("metric", a["name"], b["name"],
                                           "delta [95% CI]"))
    for m in DECOMPOSABLE:
        d = r["delta"][m]
        flag = "*" if d["excludes_zero"] else " "
        lines.append("%-14s %10.5f %10.5f  %+.5f [%+.5f,%+.5f]%s" % (
            m, a["report"][m], b["report"][m], d["delta"], d["lo"], d["hi"],
            flag))
    lines.append("%-14s %10.5f %10.5f  %+.5f (no CI: not decomposable)" % (
        "roc_auc", a["report"]["roc_auc"], b["report"]["roc_auc"],
        r["auc_delta"]))
    lines.append("%-14s %10.5f %10.5f  %+.5f" % (
        "ece", a["report"]["ece"], b["report"]["ece"],
        b["report"]["ece"] - a["report"]["ece"]))
    lines.append("  side prior on these rows (mean logit): %+.5f -> %+.5f"
                 % (r["ref_side_bias_logit"], r["new_side_bias_logit"]))
    lines.append("  * = interval excludes 0")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True,
                    help="reference arm, NAME=PATH to a test_preds.npz")
    ap.add_argument("--new", action="append", required=True,
                    help="arm(s) to compare against the reference")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=D.SPLIT_SEED)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    ref_name, ref, _ = load_arm(args.ref)
    results = {}
    for spec in args.new:
        new_name, new, members = load_arm(spec)
        r = compare(ref_name, ref, new_name, new, args.n_boot, args.seed)
        if members > 1:
            r["members"] = members
        results[new_name] = r
        print(format_result(r))
        print()

    if args.out:
        D.write_json(args.out, {"ref": ref_name, "ref_spec": args.ref,
                                "n_boot": args.n_boot, "seed": args.seed,
                                "results": results})
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
