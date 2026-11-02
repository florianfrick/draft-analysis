# Held-out series walkthrough (`model/holdout/`)

One series, replayed move by move through every checkpoint in the repo. The
question is not "how accurate is it on average" — that is what `STATUS.md` is
for — but **what did each model actually recommend at each real state**, and
what happened next.

## The series

2026 Worlds final: **Bilibili Gaming vs Hanwha Life Esports**, patch 16.20,
2026-10-31.

| | |
|---|---|
| games | `LOLTMNT02_486564` (game 1), `LOLTMNT02_486571` (game 2) |
| result | BLG won both, on blue side in both, first pick to HLE |
| fearless | on — game 2 could not re-pick any of game 1's ten champions |

## Why it is a real holdout

| | |
|---|---|
| `pro_drafts.pt` | stops at 2026-10-24 23:19:46, contains neither gameid |
| `pro_drafts_all.pt` | built 11:58, same cutoff, contains neither gameid |
| 2026 CSV | appended to at 12:59 on 2026-11-01 — *after* both caches |

`eval_series.load_series_rows` scans the CSV directly instead of calling
`pipeline.build_rows`, and `test_holdout.py` asserts the gameids are still
absent from both caches so the mistake cannot pass silently.

**This did happen once, and the fix is `pipeline.CORPUS_MAX_DATE`.** A test
that calls `build_rows()` with defaults noticed the changed CSV signature,
rescanned, and overwrote `pro_drafts.pt` with 46,875 drafts — the two final
games included. The corpus is now pinned to the date every recorded number was
measured on, so a rescan *cannot* pull them in: games after the pin are counted
as `after_corpus_freeze` and dropped. Rebuilding today returns exactly the
46,813 drafts the models trained on. Moving that pin is a deliberate act and
requires re-measuring everything.

## The confound you have to know about

Patch 16.20 has **196** corpus drafts, and the two splits place them
completely differently:

| split | train | val | test |
|---|---|---|---|
| `split_series_by_date` (kept stages) | **0** | 0 | 196 |
| `split_within_patch` (pro-only) | **155** | 21 | 20 |

So `policy-ablF` and `pro-finetune` have never seen a single patch-16.20 draft,
and ablF has no patch input at all. The pro-only models trained on 155 of them
and carry a patch-embedding row for it. The head-to-head in the report is the
**deployment** comparison — a patch ships, you have its first 80%, predict the
rest — not a like-for-like one. The gap mixes three things: better fit, a patch
prior, and strictly more information about 16.20.

## What is scored

10 distinct checkpoints — every trained artefact kept in the repo, minus
`latest.pt` (a later step of the same run, worse on the metric that run
selected on):

* **next move** — 3 policy models, 20 states per game. Top-1, top-5, the rank
  of the move that actually happened, and the probability it gave it.
* **masked completion** — 3 encoder models, 10 picks per game, each hidden in
  turn while the other nine stay visible.
* **value** — 4 value heads, P(blue wins) after each of the 20 moves.

The pro-only grid is pruned to the shipped policy, its masked init, and the
no-patch control that the ablation is measured against. The losing arms (seed
1, the wider corpus, the scratch control) were deleted with their weights;
their numbers survive in `FINDINGS.md` §4 and `model/proonly/reports/`, and
`test_holdout.py` asserts this registry and the committed report agree.

Baseline: patch 16.20's own per-move frequency table (196 rows, backed off to
patch-wide frequency to break ties), with legality applied — otherwise it would
rank champions that are already banned or picked while every model has them
masked out, a ~10-place handicap.

## Results

Next move, 40 states:

| model | training | top-1 | top-5 | mean rank | ban1 top-5 |
|---|---|---|---|---|---|
| `policy-ablF` | soloQ->pro (kept) | 0.100 | 0.450 | 17.1 | 0.17 |
| `pro-patch-s0` | pro-only +patch | **0.225** | **0.675** | 7.3 | **0.58** |
| `pro-nopatch-s0` | pro-only -patch | 0.100 | 0.550 | 13.2 | 0.17 |
| patch table | 16.20 frequency | 0.050 | 0.350 | 13.6 | 0.33 |

Masked completion, 20 hidden picks (top-5 / mean rank): `pro-patch-s0` 0.700 /
4.0, `pro-finetune` 0.650 / 10.1, `soloQ-pretrain` 0.450 / 9.2.

## Three things the pooled numbers hid

**1. `policy-ablF` is sharper but wilder than a frequency table.** Better top-5
(0.450 vs 0.350), *worse* mean rank (17.1 vs 13.6): 4 of 40 states land outside
the top 50, one at rank 148 of 173. No pro-only arm shows this — their worst is
129 with at most two such states. "Better top-5" and "better average position"
are different claims and this series separates them.

**2. ban1 is where the arms separate**, as the branch's mechanism predicts.
With an empty board the patch-conditioned models reach 0.58 top-5; the no-patch
arms sit at 0.17. That is 7/12 vs 2/12 states — direction, not magnitude.

**3. The masked models cannot express fearless.** `ProMaskedDraftDataset` never
receives `unavailable`, so on game 2 — where ten champions were already picked
in game 1 — 19 of 30 model-slot pairs put a game-1 pick in their top-5. The
next-move models, which do get `unavailable_mask`, break fearless zero times.
Scope limit of the masked task as trained, not a bug in this harness.

**Value: none of the four is usable.** `prefix-V` and `finetune-win` stay within
0.44-0.58 across all 20 moves; `pretrain-win` collapses to 0.05 at move 7 (one
team holds a single pick and PMA pools over a near-empty set);
`policy-value` swings 0.55 -> 0.01 -> 0.99 inside game 2. It ends at 0.91/0.99
and blue did win both, but the path is noise — consistent with `STATUS.md` §3
rather than contradicting it.

## Harness validation

`test_harness_reproduces_the_recorded_patch_number` scores `policy-ablF` on the
first 40 patch-16.20 test rows with this code path and requires top-1 0.0875 /
top-5 0.25625. On all 196 rows it returns **0.0793 / 0.2390**, which is exactly
the patch-16.20 figure `STATUS.md` §3 records. The ranking, the legality mask
and the patch plumbing therefore agree with the training-time evaluator, not
merely with themselves.

## Statistics discipline

Two games, 40 next-move states, 20 hidden picks. One hit is 2.5 points and a
Wilson interval on 20 samples spans about 30. There are no p-values here on
purpose. What *is* meaningful is the per-state comparison, because every model
is scored on the identical board — that is what the step tables show.

## Files

| file | what it is |
|---|---|
| `eval_series.py` | scans the CSV, loads all 10 checkpoints, scores every state |
| `render_report.py` | formats the JSON into HTML + markdown; recomputes nothing |
| `reports/blg_vs_hle.json` | every state, every model, every top-5 |
| `REPORT_blg_vs_hle.html` | the report, colour-coded per step |
| `REPORT_blg_vs_hle.md` | the same tables in markdown |
| `test_holdout.py` | 12 tests, incl. the recorded-number reproduction and the registry check |

## Reproduce

```bash
docker compose run --rm -T dev python model/holdout/eval_series.py
docker compose run --rm -T dev python model/holdout/render_report.py
docker compose run --rm -T dev python -m pytest model/holdout/test_holdout.py -q
```

## Adding another series

`SERIES_GAMES` in `eval_series.py` is the only thing to change; everything
downstream is keyed on the gameids. If the new games are *in* a cache, the
holdout claim is gone — rebuild nothing and instead point the loader at a
corpus version that predates the series, or the report becomes a train-set
readout that still looks plausible.
