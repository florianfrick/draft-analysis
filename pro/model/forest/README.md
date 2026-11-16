# Random forest (`pro/model/forest/`)

The nonparametric baseline, trained as a **candidate scorer**: for one draft
state and one candidate champion, will that champion be the answer? Then rank.

| | |
|---|---|
| `next_move` | in 548 -> 110 trees, legal candidates only, 20 states/game |
| `masked` | in 369 -> 110 trees, all 173 candidates, 10 states/game |
| params | total node count across trees (a forest has no weight vector), capped at 3 M by `baselines.check_budget` |
| weights | `pro/model/checkpoints/{nextpick,masking}-forest-flat-v1/model.joblib` |

## Why it is not a 173-class classifier

`RandomForestClassifier` over 173 labels is not slow here, it is **infeasible**,
and the arithmetic is worth writing down because the natural first attempt is
exactly that:

scikit-learn's split criterion carries a class histogram, and pays
`O(n_samples_at_node x n_classes)` **per feature tried**. The canonical split's
next-move training set is 794,400 states. At the root alone that is
`794,400 x 173` = 1.4e8 operations for one feature, times `max_features`
(~23 with `sqrt`), times ~15 levels, times 100 trees: ~5e12. Hours per tree.

A forest *can* predict a 173-way label when `n x n_classes` is small - a few
thousand rows. It cannot at 794k. So the forest answers the same question in the
form trees are good at: score one (state, candidate) pair, then rank. That is a
standard tree-based recommender, and because it produces the same ranking the
suite measures - top-1 / top-5 / mean rank over the legal candidates - the row
is comparable even though the architecture is not.

## What it costs

* **Negative sampling.** Every state contributes its true answer plus
  `--negatives` uniformly drawn legal alternatives, so one state becomes seven
  rows of 548 floats.
* **A subsample of states.** `--max-states 150000` of the 794,400 available,
  because seven rows each does not fit any other way. This is a real difference
  from the transformer, which saw every state of every epoch, and it is why the
  row should be read as "what a tree does with a slice of the same board"
  rather than "trees are worse". Measured: at 60,000 states / 100 trees the
  next-move row landed *below* the frequency table; at 150,000 / 110 it reaches
  it. Capacity, not just features, is what the forest is short of.
* **No selection.** There is no epoch schedule, so there is nothing to select
  on: the single fit is the model. Val is measured on a 4,000-state sample for
  provenance, not for choosing anything.

## Training

```bash
# both tasks
docker compose run --rm -T dev python -m pro.model.forest.train

# smoke run (keep --out inside the repo: on Git Bash for Windows an absolute
# /tmp path is rewritten into the working directory)
docker compose run --rm -T dev python -m pro.model.forest.train \
    --max-states 3000 --trees 10 --val-states 300 \
    --out pro/model/checkpoints/_smoke
```

`forest/model.py` has the full argument. Same rule as the MLP: the split is
`test_split/split.py`, and the candidates come from
`baselines.candidates_for`, which is the same set the policy model's `-1e4`
masking leaves.

## Files

| file | what it is |
|---|---|
| `model.py` | `ForestScorer` (chunked scoring) and the loader |
| `train.py` | negative sampling, the fit, the artifact writer |

Features are in `pro/model/baselines.py`, shared with `pro/model/mlp/`. They are
dense and compact rather than one-hot: a tree splits on a threshold, so an
integer column is a usable feature and a 174-wide one-hot mostly is not. The
candidate still gets its own one-hot, because "is the candidate this champion"
is the strongest split available here.
