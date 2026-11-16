"""A random forest over the flattened board - the nonparametric baseline.

The forest is trained as a **candidate scorer**: for a draft state and one
candidate champion, will that champion be the answer? Then it ranks the legal
candidates. That is not a compromise, it is the only formulation that fits:
`RandomForestClassifier` over 173 labels pays
`O(n_samples x n_classes)` per feature tried per split, which at 794k training
states is hours per tree. `forest/model.py` works the arithmetic out.

Training pairs are built by negative sampling: every state contributes its true
answer plus `--negatives` uniformly drawn legal alternatives. The consequence
is that the forest sees a *subsample* of states - `--max-states`, default
60,000 - because one state becomes `1 + negatives` rows of 548 floats. That is
a real difference from the transformer, which saw every state of every epoch,
and it is the honest reason to read this row as "what a tree can do with a
small slice of the same board" rather than "trees are worse".

Why this baseline earns its place
---------------------------------
It is the only model in the suite that is not a smooth parametric function of
the input. If it lands near the transformer, the transformer's edge is the
features (patch conditioning, role slots) rather than the architecture; if it
lands near the frequency table, the flat encoding is the bottleneck. Either
answer is more useful than another neural arm.

Usage:
    python -m pro.model.forest.train                     # both tasks
    python -m pro.model.forest.train --task next_move
    python -m pro.model.forest.train --max-states 5000 --trees 20   # smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # .../pro/model/forest
ROOT = HERE.parents[2]                          # .../lol draft analysis
sys.path.insert(0, str(ROOT))

from pro.model import baselines as B            # noqa: E402
from pro.model.forest.model import ForestScorer, n_nodes  # noqa: E402
from pro.model.test_split import models as M    # noqa: E402
from pro.model.test_split import split as S     # noqa: E402

CKPT_ROOT = ROOT / "pro" / "model" / "checkpoints"
NAMES = {"next_move": "nextpick-forest-flat-v1", "masked": "masking-forest-flat-v1"}
PER_GAME = {"next_move": B.N_MOVES, "masked": B.N_SLOTS}


def candidates_for(task, row, k):
    """Every candidate the model is allowed to name, ascending."""
    if task == "next_move":
        return B.candidates_for(row, k)
    return np.arange(B.NUM_CHAMPS, dtype=np.int64)


def training_pairs(task, rows, states, rng, negatives):
    """(state list, candidate list, labels) with one positive per state."""
    out_states, out_cands, out_y = [], [], []
    for (row, k) in states:
        pool = candidates_for(task, row, k)
        target = (int(row["moves"][k][2]) if task == "next_move"
                  else int(row["champ"][k]))
        if target not in pool:                       # should never happen
            continue
        others = pool[pool != target]
        pick = rng.choice(others, size=min(negatives, len(others)),
                          replace=False)
        cands = np.concatenate([[target], pick]).astype(np.int64)
        out_states.append((row, k))
        out_cands.append(cands)
        out_y.append(np.concatenate([[1], np.zeros(len(pick), dtype=np.int64)]))
    return out_states, out_cands, np.concatenate(out_y)


def build(task, states, candidate_lists, patch_index):
    if task == "next_move":
        return B.build_forest_next_move(states, candidate_lists, patch_index)
    return B.build_forest_masked(states, candidate_lists, patch_index)


def train_task(task, rows, splits, patch_index, args):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(args.seed)
    per_row = PER_GAME[task]

    # States, not games: the forest has no epochs, so "how much data" is the
    # only knob and it is stated in the row it produces. Drawn without
    # replacement over the (game, position) grid so the sample is distinct -
    # `max_states` is well under the 794,400 available, so the draw is cheap.
    train_idx = np.array(splits["train"])
    total = len(train_idx) * per_row
    n = min(args.max_states, total)
    pick = rng.choice(total, size=n, replace=False)
    states = [(rows[int(train_idx[p // per_row])], int(p % per_row))
              for p in pick]

    t0 = time.time()
    st, cands, y = training_pairs(task, rows, states, rng, args.negatives)
    X = build(task, st, cands, patch_index)
    print("\n== %s ==  %d states -> %d pairs (1 pos + %d neg)  X %s  %.1f GB"
          % (task, len(st), X.shape[0], args.negatives, X.shape,
             X.nbytes / 1e9))

    clf = RandomForestClassifier(
        n_estimators=args.trees, max_features=args.max_features,
        min_samples_leaf=args.min_leaf, n_jobs=args.jobs,
        class_weight="balanced_subsample", random_state=args.seed)
    clf.fit(X, y)
    fit_s = time.time() - t0
    del X
    params = B.check_budget(NAMES[task], n_nodes(clf), unit="tree nodes")
    print("   fitted in %.0fs  %d trees  %s nodes  pos rate %.3f"
          % (fit_s, args.trees, format(params, ","), y.mean()))

    # A forest has no epoch schedule, so there is nothing to select on: the fit
    # above is the model. Val is still measured, on a sample, so the report can
    # show train-side behaviour next to the test number.
    key = "steps" if task == "next_move" else "slots"
    scorer = ForestScorer(clf, task, patch_index)
    n_val = min(args.val_states, len(splits["val"]))
    val_sample = [(rows[int(i)], int(k)) for i, k in
                  zip(rng.choice(np.array(splits["val"]), size=n_val,
                                 replace=False),
                      rng.integers(0, per_row, size=n_val))]
    val = M.aggregate(M.score_flat(scorer, task, val_sample), key)
    print("   val  (sample %d states) top1 %.4f top5 %.4f rank %.2f"
          % (val["n"], val["top1"], val["top5"], val["mean_rank"]))

    t0 = time.time()
    test_states = [(rows[i], k) for i in splits["test"] for k in range(per_row)]
    test = M.aggregate(M.score_flat(scorer, task, test_states), key)
    print("   TEST top1 %.4f top5 %.4f rank %.2f  (%.0fs)"
          % (test["top1"], test["top5"], test["mean_rank"], time.time() - t0))

    run_dir = Path(args.out) / NAMES[task] if args.out else CKPT_ROOT / NAMES[task]
    run_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"estimator": clf,
                 "patch_vocab": list(M.PRO_DATA.build_patch_vocab(rows)[0])},
                run_dir / "model.joblib", compress=3)
    B.write_flat_artifact(
        run_dir, kind="forest", task=task, weights="model.joblib",
        params=params,
        meta={"trees": args.trees, "max_features": args.max_features,
              "min_samples_leaf": args.min_leaf, "negatives": args.negatives,
              "train_states": len(st), "train_pairs": int(len(y)),
              "fit_seconds": round(fit_s, 1), "seed": args.seed,
              "selection": "none (single fit)"},
        val=val, test=test,
        split_counts={k: len(v) for k, v in splits.items()})
    try:
        shown = run_dir.relative_to(ROOT)
    except ValueError:
        shown = run_dir
    print("   wrote %s" % shown)
    return {"task": task, "params": params, "val": val, "test": test}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=("next_move", "masked", "both"),
                    default="both")
    ap.add_argument("--max-states", type=int, default=60_000,
                    help="training states drawn from the train split")
    ap.add_argument("--negatives", type=int, default=6,
                    help="legal negatives per positive")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--min-leaf", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--val-states", type=int, default=4000,
                    help="val states used for the val number (a sample, not "
                         "all 95,420 - there is nothing to select on)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    rows = S.load_rows(verbose=True)
    splits = S.build(rows)
    patch_index = M.PRO_DATA.build_patch_vocab(rows)[1]
    print("split: %d train / %d val / %d test"
          % (len(splits["train"]), len(splits["val"]), len(splits["test"])))

    tasks = ["next_move", "masked"] if args.task == "both" else [args.task]
    for task in tasks:
        train_task(task, rows, splits, patch_index, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
