# Test split evaluation (`model/test_split/`)

Every model class, scored on the same seven games. This is the one split
nothing trains on.

The numbers are in **[`REPORT_test_split_evaluation.md`](REPORT_test_split_evaluation.md)**
(and the `.html` next to it); this file is about the harness. Nothing here
restates a score, so the two cannot drift.

## The split

Three series, seven games, all patch 16.20, all played on the last day of
Worlds 2026 (2026-10-31):

| series | games | gameids |
|---|---|---|
| Bilibili Gaming vs Hanwha Life Esports | 2 | `LOLTMNT02_486564`, `LOLTMNT02_486571` |
| Karmine Corp vs Movistar KOI | 2 | `LOLTMNT02_485476`, `LOLTMNT02_486541` |
| LYON vs LØS | 3 | `LOLTMNT02_485482`, `LOLTMNT02_485488`, `LOLTMNT02_486555` |

They are **`pipeline.TEST_SERIES`**, pinned by `series_id`, and
`split_within_patch` holds them out of train and val *by id*:

```python
splits = pipeline.split_within_patch(rows)      # pinned_test=TEST_SERIES
# train 42,100 / val 4,768 / test 7
```

A fractional tail would be a different set of games on every rebuild. A test
split that moves is not a test split, and a 258-game patch has nothing like
enough games for a 10% tail to be stable.

### The pin had to move, and that is the interesting part

The old `CORPUS_MAX_DATE` was `2026-10-24 23:19:46`, which is *before* these
games. The previous arrangement called them a **holdout**: outside the corpus
entirely, scored by scanning the CSV directly.

That works only while you never want to train on them. A series the corpus has
never seen cannot be a test split — it can only be a holdout that nothing can
be trained against, and the moment you retrain you have to decide whether to
include it. So the pin moved to `2026-11-01 23:59:59` (corpus 46,813 →
**46,875** drafts), the games are *in* the corpus, and the split is what keeps
them out of training. `test_test_split.py` asserts both halves: present in
`pro_drafts.pt`, absent from every train and val index.

Cost, stated plainly: raising the pin invalidates every number recorded before
it. Everything was re-measured on 2026-11-03 (`STATUS.md` §7).

## What is compared

One checkpoint per model class, all under `model/checkpoints/<name>/`. The
directory name is the contract — `frontend/server._ckpt_roles` reads the prefix
to decide which UI slot may load it.

| directory | training | answers |
|---|---|---|
| `masking-soloqueue` | soloQ pretrain only | masked completion |
| `masking-finetune` | soloQ → pro finetune | masked completion |
| `masking-proonly` | pro drafts only, from scratch | masked completion |
| `nextpick-policy` | soloQ encoder + policy head | next move |
| `nextpick-proonly` | pro drafts only, encoder + policy from scratch | next move |

`masking-soloqueue` is not a class of its own: it is the encoder
`masking-finetune` starts from, kept as the reference that says whether
fine-tuning on pro drafts bought anything.

**No value model.** The frontend has no slot for one (`_ckpt_roles` returns
`[]` for `value-*`) and the value path measured at chance in every earlier arm.
That is a gap in this comparison, not an untested result.

## What is scored

* **next move** — 20 states per game, 140 in total. Top-1, top-5, mean rank,
  and the probability given to the move that actually happened.
* **masked completion** — each of a game's ten picks hidden in turn while the
  other nine stay visible. 70 slots in total.

Baseline for next move: patch 16.20's own per-move frequency table, with
legality applied. Without the legality filter the table would rank champions
already banned or picked while every model has them masked out — a ~10-place
handicap on a 173-champion list, which would flatter the models.

## Reading it honestly

Seven games. One game is 14% of the sample, and a 95% Wilson interval on a rate
observed over seven games is roughly 45 points wide. Every rate in the report
carries its interval for exactly that reason: **at this n, two models whose
intervals overlap are tied as far as this split can tell, and most pairs will
overlap.**

So this is not a ranking. It is a smoke test on genuinely unseen games: it
catches a model that is broken or badly fitted (a next-move model at or below
the frequency table, a masked model that cannot name the champion at all) and
it refuses to flatter anything else. For the ranking, the within-patch
validation split is the number to read — 4,768 rows, and every model was
selected on it.

The part that does carry information is the per-state tables, because every
model is scored on the identical board. "Which model wins" is unanswerable
here; "here is what each one said at this state, and here is what happened" is
not.

## Files

| file | what it is |
|---|---|
| `eval_test_split.py` | scans the CSV for the pinned series, loads every checkpoint, scores every state |
| `eval_fearless_mask.py` | raw vs bans+picks-masked vs pool-masked top-5 on the four fearless games — what an output-only fearless mask buys |
| `render_test_split.py` | formats the JSON into HTML + markdown; recomputes nothing but counts and Wilson intervals |
| `reports/test_split.json` | every state, every model, every top-5 |
| `reports/fearless_mask.json` | the three mask readings per slot, per masked model |
| `REPORT_test_split_evaluation.html` | the report, colour-coded per step |
| `REPORT_test_split_evaluation.md` | the same scoreboards in markdown |
| `test_test_split.py` | the harness tests, including the split invariant and the registry/disk agreement |

## Reproduce

```bash
docker compose run --rm -T dev python model/test_split/eval_test_split.py
docker compose run --rm -T dev python model/test_split/eval_fearless_mask.py
docker compose run --rm -T dev python model/test_split/render_test_split.py
docker compose run --rm -T dev python -m pytest model/test_split/test_test_split.py -q
```

The evaluation needs the weights, which are git-ignored. Train them first — the
commands are in `model/proonly/README.md`, `model/finetune/README.md` and
`model/policy/README.md` — and `missing_checkpoints()` will name any that are
absent instead of failing inside a loader.

## Changing the split

`pipeline.TEST_SERIES` is the only place to change it. Everything downstream —
the split, this harness, the report — resolves from that tuple, so there is no
second list to forget. Two things will break loudly on purpose:

* a pinned id that is not in the corpus raises (`strict_pin=True`), because a
  typo must not become a quietly smaller test set;
* a game the pipeline rejects raises rather than being skipped.

If you add a series that arrived *after* the corpus pin, raise
`CORPUS_MAX_DATE` past it first and rebuild. If you add one that is already in
the corpus, nothing else is needed — the split will hold it out by id.
