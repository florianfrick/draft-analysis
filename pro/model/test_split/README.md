# Test-split evaluation (`pro/model/test_split/`)

The one place every model class is scored, on rows nothing trained on, with the
same construction and the same error bars.

Numbers live in **[`REPORT_suite.md`](REPORT_suite.md)** and are generated from
`reports/suite.json` by `report.py`. This file describes the harness and never
restates a score, so the two cannot drift.

## The split

`split.py` is the single definition, and it is the split the shipped `-nov13-`
pair was trained on - reproduced exactly, not re-drawn:

| | |
|---|---|
| corpus | `data/pro play/pro_drafts_full.pt`, 46,939 drafts, 107 patches |
| call | `split_within_patch(val_frac=0.10, test_frac=0.05, min_patch_rows=40, pinned_test=(), val_patches=None, val_random=False)` |
| train | 39,720 (84.62 %) |
| val | 4,771 (10.16 %) |
| test | **2,448** (5.22 %), 99 patches |

Series are grouped before assignment, so a fearless context never straddles a
split. Inside each patch the newest series become test and the ones before them
val; `min_patch_rows=40` keeps 8 patches (including the 25-row newest, 16.22)
entirely in train, because their tail would be one or two games. **Consequence,
stated because it is easy to misread: the newest patch is train-only**, and the
test tail belongs to the newest *eligible* patch.

### Why not the pinned series any more

The previous harness pinned three Worlds series as a 7-game test set. It was
honest and it was useless for ranking: 7 games is 140 next-move states and 70
masked slots, where a 95 % Wilson interval is ~22 points wide, so any two models
were "tied" and the split could only detect a model that was outright broken.
It also was not the split the shipped models were trained under, which made it a
home game for some rows and an away game for others.

The fractional tail fixes all three: 2,448 games, deterministic (series are
sorted by `(date, gameid)`, no seed to get wrong), and identical to what the
incumbents were held out of.

**No holdout.** A series the corpus has never seen cannot be a test split - it
can only be a holdout nothing can be trained against, and the moment you retrain
you have to decide whether to include it. Every game stays in the corpus and the
split does the holding out.

### The manifest

`split_manifest.json` stores the counts, the arguments, the sorted test gameids
and a sha256 over them. `suite.py` verifies against it before scoring and
refuses to run on a mismatch - a corpus refresh that moved one game into test
changes the hash and stops the run, rather than silently producing numbers that
look comparable and are not.

```bash
python -m pro.model.test_split.split            # print it
python -m pro.model.test_split.split --check    # verify against the manifest
```

## The two suites

The two model families answer different questions, so they are scored by
different functions and never pooled:

| suite | question | states/game | models |
|---|---|---|---|
| `next_move` | given the first *m* moves, what is move *m*? prefix only | 20 | every `nextpick-*` |
| `masked` | given a complete draft with one pick hidden, who was it? the other nine slots - **including later picks** - are visible | 10 | every `masking-*` |

Pooling them would be a mistake with a direction: masked completion is the easier
question by construction, so a combined accuracy would rank the easier question
higher and call it a better model. This is also the sense in which a masking
model *is* a role-conditioned next-pick model - the role vector is an input and
the query is "who belongs in this slot". What differs is the context each one
gets to see, and that difference is why the two numbers are never averaged.

`models.py` holds both scorers and is the code the pre-`nov13` harness used, so
a number here is on the same construction as the ones in `STATUS.md` rather than
a quiet re-derivation. Legality is applied identically to every model and to the
frequency-table baseline; without it the table would rank already-banned
champions while every model has them at `-inf`, a ~10-place handicap that would
flatter the models. `test_suite.py::test_legality_matches_the_policy_model`
asserts that parity on real boards rather than trusting the reading.

## The baselines

The first row of each comparable table is not a checkpoint. It is the reference
point the question is asked against, and it is in the table rather than in a
paragraph because its `Δ vs ref` is a real number: how far the incumbent is
ahead of a lookup table, with a discordant count and a p.

| baseline | what it is | where |
|---|---|---|
| `00-frequency-table` | per-patch move-frequency table, train rows only, ranked under the same legality filter | `suite.py::baseline_next_move` |
| `nextpick-mlp-flat-v1` / `masking-mlp-flat-v1` | flat-feature MLP, solved to the 3 M budget (2.99 M each) | `pro/model/mlp/` |
| `nextpick-forest-flat-v1` / `masking-forest-flat-v1` | random forest used as a candidate scorer, 2.75 M / 2.12 M nodes | `pro/model/forest/` |

The two learned baselines are the interesting ones, because they answer a
question the transformer rows cannot: *how much of the result is the
architecture?* Both are given the same board, the same legality rule, the same
loss and the same rows as the shipped pair, and neither has attention or set
structure. `pro/model/baselines.py` is the single feature builder they share, so
they differ only in how they model the board.

A model that is not a set transformer ships a `flat.json` instead of a `best.pt`:

```json
{"kind": "forest", "task": "next_move", "weights": "model.joblib",
 "params": 1234567, "feature_version": 1}
```

`registry.py` treats the presence of that file as the definition of a flat
baseline - not the file extension, which would guess wrong on a `.pt` that is a
pickled estimator. `feature_version` is checked at load time: a baseline trained
on an older layout is refused rather than scored, because a shifted feature
column still produces a number, just not that number.

## Adding a model

Drop the checkpoint in `pro/model/checkpoints/<name>/` with the prefix that
states what it answers (`masking-` or `nextpick-`) and a `best.pt`. The registry
finds it by directory name, the task follows from the prefix, and it joins the
next run. Nothing needs editing. A non-neural model drops in the same way with a
`flat.json` naming its weights (see above).

For the run to count it as **comparable**, the checkpoint's sibling
`test_report.json` must record the canonical split counts - which the trainers
already write. If it records a different split the model is still scored, but it
is labelled `different split`, printed in a separate table, and given no delta
column, because a model evaluated on rows it trained on is not a measurement of
that model.

`pro/model/checkpoints/` holds only the shipped pair. Everything else is under
`pro/model/archive/checkpoints/` and is reached with `--include-archive`.

## Running it

```bash
# the full thing (both suites, every active checkpoint)
docker compose run --rm -T dev python -m pro.model.test_split.suite

# what would be scored, without loading anything
docker compose run --rm -T dev python -m pro.model.test_split.suite --list

# a fast random slice (random, not the first N: the test tail is ordered by patch)
docker compose run --rm -T dev python -m pro.model.test_split.suite --limit 200

# one model, one suite
docker compose run --rm -T dev python -m pro.model.test_split.suite \
    --task masked --only masking-proonly-nov13-through-lcp-df-sw

# the retired arms too (they will be labelled not-comparable)
docker compose run --rm -T dev python -m pro.model.test_split.suite --include-archive

# render
docker compose run --rm -T dev python -m pro.model.test_split.report
docker compose run --rm -T dev python -m pytest pro/model/test_split/ -q
```

CPU only. A full 2,448-game suite takes ~30 s per `-nov13-` model (2.5 M params);
the older soloQ-encoder arms are 3-15x slower, which is why the whole
`--include-archive` run is ~25 minutes and the shipped pair alone is ~2. The
forest is the slowest single row (~4 min) because it scores 173 candidates per
state; `--no-baseline` skips the frequency table, not the learned baselines.

## Reading it honestly

* **`params`** counts what the model stores, and the three families count
  different things: `sum(p.numel())` for a transformer, table cells for the
  frequency table (20 moves x 173 champions x one per test patch), total tree
  nodes for the forest. The column is there to size a model, not to rank one.
* Every rate carries a Wilson 95 % interval, not a normal approximation - the
  approximation collapses to zero width at 0 % and 100 %, which is exactly where
  a broken model lands.
* The **Δ vs ref** column is a paired McNemar against the incumbent. Paired is
  the only fair comparison here: items differ hugely in difficulty (ban1 is
  nearly unpredictable, pick2 nearly determined), so an unpaired difference
  mostly measures which model drew the easier games.
* `p > 0.05` with a small Δ means **not shown to be different**, not equal.
* The **discordant** column (`this_only/reference_only`) is printed next to
  every p because it is what the p is computed from; a p-value alone hides
  whether an effect rests on three items or three hundred.

## Files

| file | what it is |
|---|---|
| `split.py` | the split, the manifest, `--check` / `--write` |
| `split_manifest.json` | the frozen counts, args, test gameids and fingerprint |
| `registry.py` | checkpoint discovery, task assignment, comparability, param counts |
| `models.py` | loaders and scorers for both tasks, plus Wilson / McNemar |
| `suite.py` | the runner: scores, aggregates, pairs, writes `reports/suite.json` |
| `report.py` | `reports/suite.json` -> `REPORT_suite.md` |
| `test_suite.py` | the invariants: counts, disjointness, manifest, registry, stats |
| `reports/suite.json` | every aggregate, plus per-state detail for a random sample |

The previous pinned-split harness (`eval_test_split.py`, `render_test_split.py`,
`eval_fearless_mask.py` and its report) is in
`pro/model/archive/test_split_legacy/`. It is kept for the fearless-mask
measurement, which this suite does not redo, and for the 7-game numbers in
`STATUS.md` §7, which were measured on it.
