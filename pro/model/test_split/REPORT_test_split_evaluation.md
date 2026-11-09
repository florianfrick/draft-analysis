# Test split evaluation

Every model class, scored on the same 7 games. This is the one split nothing trains on.

| | |
|---|---|
| corpus | `pro_drafts.pt`, pinned at `CORPUS_MAX_DATE = 2026-11-01 23:59:59` |
| split | `pipeline.split_within_patch` with `pinned_test = TEST_SERIES` |
| test series | 3, pinned by `series_id` |
| test games | 7, patch 16.20 |
| baseline | patch 16.20 move-frequency table, 258 corpus rows |

## 1. The split

Three series, 7 games, all patch 16.20, all played on 2026-10-31. The last three series of Worlds 2026, pinned by `series_id` in `pipeline.TEST_SERIES` so the test set does not move when the corpus is rebuilt.

| series | date | games | blue wins |
|---|---|---|---|
| Bilibili Gaming vs Hanwha Life Esports | 2026-10-31 | 2 | 2 |
| Karmine Corp vs Movistar KOI | 2026-10-31 | 2 | 2 |
| LYON vs LØS | 2026-10-31 | 3 | 1 |

## 2. What is compared

| checkpoint | training | answers | path |
|---|---|---|---|
| `nextpick-policy` | soloQ encoder + policy head | next move | `solo_queue/model/checkpoints/nextpick-policy/best_policy.pt` |
| `nextpick-proonly` | pro-only, from scratch | next move | `pro/model/checkpoints/nextpick-proonly/best.pt` |
| `masking-soloqueue` | soloQ pretrain only | masked | `solo_queue/model/checkpoints/masking-soloqueue/best.pt` |
| `masking-finetune` | soloQ -> pro finetune | masked | `solo_queue/model/checkpoints/masking-finetune/best_mask.pt` |
| `masking-proonly` | pro-only, from scratch | masked | `pro/model/checkpoints/masking-proonly/best.pt` |

`masking-soloqueue` is not a class of its own - it is the encoder `masking-finetune` starts from, kept as the reference that says whether fine-tuning on pro drafts bought anything.

## 3. Next move

20 states per game, 140 in total.

| model | training | top-1 | top-5 (95% CI) | mean rank | ban1 | pick1 | ban2 | pick2 |
|---|---|---|---|---|---|---|---|---|
| `nextpick-policy` | soloQ encoder + policy head | 21% | 48% (40%-56%) | 11.2 | 36% | 67% | 14% | 71% |
| `nextpick-proonly` | pro-only, from scratch | 23% | 56% (48%-64%) | 9.1 | 38% | 74% | 36% | 79% |
| patch table | patch 16.20 move-frequency table | 11% | 40% (32%-48%) | 12.1 | 45% | 55% | 11% | 39% |

## 4. Masked completion

Each of a game's ten picks hidden in turn, 70 slots per model.

| model | training | top-1 | top-5 (95% CI) | mean rank |
|---|---|---|---|---|
| `masking-soloqueue` | soloQ pretrain only | 33% | 53% (41%-64%) | 8.7 |
| `masking-finetune` | soloQ -> pro finetune | 36% | 74% (63%-83%) | 3.1 |
| `masking-proonly` | pro-only, from scratch | 36% | 71% (60%-81%) | 3.4 |

**The masked models cannot express fearless.** On the games that carry a fearless context, 85 of 120 masked slots put a champion already picked earlier in the series into their top-5. The next-move models do get `unavailable_mask` and break fearless zero times. Scope limit of the masked task as trained, not a bug in this harness.

## 5. How much seven games can say

**Almost nothing, on its own.** At 7 games one game is 14% of the sample, and a 95% interval on a rate observed over seven games is roughly 35% wide. Any two models whose intervals overlap are tied as far as this split can tell, and at this n most of them will. The intervals are printed next to every rate for exactly that reason.

What the split can do is catch a model that is broken or badly fitted - a next-move model at or below the frequency table, a masked model that cannot name the champion at all. It is a smoke test on genuinely unseen games, not a ranking. For the ranking, the within-patch validation split is the number to read: it is 100x larger and every model was selected on it.

## 6. Per-game top-5

| game | `nextpick-policy` | `nextpick-proonly` |
|---|---|---|
| Bilibili Gaming vs Hanwha Life Esports g1 | 10/20 | 12/20 |
| Bilibili Gaming vs Hanwha Life Esports g2 | 11/20 | 14/20 |
| Karmine Corp vs Movistar KOI g1 | 7/20 | 12/20 |
| Karmine Corp vs Movistar KOI g2 | 10/20 | 10/20 |
| LYON vs LØS g1 | 7/20 | 11/20 |
| LYON vs LØS g2 | 12/20 | 11/20 |
| LYON vs LØS g3 | 10/20 | 9/20 |

## Reproduce

```bash
docker compose run --rm -T dev python pro/model/test_split/eval_test_split.py
docker compose run --rm -T dev python pro/model/test_split/render_test_split.py
docker compose run --rm -T dev python -m pytest pro/model/test_split/test_test_split.py -q
```
