# Test-split evaluation

Every model class on the project's one held-out split. Generated from `reports/suite.json` by `report.py`; no number here is typed by hand.

## The split

| | |
|---|---|
| corpus | `pro_drafts_full.pt` |
| drafts | 46939 |
| split | `split_within_patch`, no pin, tail |
| train / val / test | 39720 / 4771 / 2448 |
| shares | 84.62% / 10.16% / 5.22% |
| test games scored | 2448 |
| test fingerprint | `99c72c8550489cbf` |

This is the split the shipped `-nov13-` pair was trained on, and `split.py --check` verifies it against `split_manifest.json` on every run. Nothing in this report is measured on a row any comparable model trained on.

## Next move

Given the first *m* moves, what is move *m*? 20 states per game, prefix only.

### Comparable

The first row is the reference point the question is asked against: a per-patch move-frequency table fitted on train rows only, ranked under the same legality filter every model gets. Its `Δ vs ref` is the incumbent's margin over a lookup table.

| model | params | n | top-1 | 95% CI | top-5 | top-10 | mean rank | Δ vs ref (pp) | discordant | p |
|---|---|---|---|---|---|---|---|---|---|---|
| `00-frequency-table` | 342,540 | 48,960 | 10.8% | [10.5, 11.1] | 34.4% | 50.8% | 17.83 | -8.1 | 1857/5841 | <1e-300 |
| `nextpick-proonly-nov13-k5-through-lcp-df-sw` | 3,350,879 | 48,960 | 18.9% | [18.6, 19.3] | 50.8% | 68.3% | 10.17 | - | - | - |
| `nextpick-forest-flat-v1` | 2,753,344 | 48,960 | 10.5% | [10.2, 10.7] | 34.8% | 52.5% | 17.52 | -8.5 | 2853/6998 | <1e-300 |
| `nextpick-mlp-flat-v1` | 2,996,705 | 48,960 | 17.2% | [16.9, 17.5] | 48.2% | 66.1% | 11.22 | -1.7 | 3599/4448 | 3.3e-21 |
| `nextpick-proonly-nov13-k10-through-lcp-df-sw` | 3,350,879 | 48,960 | 19.1% | [18.7, 19.4] | 51.1% | 69.3% | 9.96 | +0.1 | 2654/2591 | 0.39 |
| `nextpick-proonly-nov13-through-lcp-df-sw` | 3,350,879 | 48,960 | 17.4% | [17.1, 17.8] | 48.3% | 66.3% | 10.90 | -1.5 | 2355/3090 | 2.6e-23 |

### Not comparable

Trained on a different split, so some of these test rows are in their training data. **These are training-set numbers.** Listed for provenance, not for ranking.

| model | params | n | top-1 | 95% CI | top-5 | top-10 | mean rank | Δ vs ref (pp) | discordant | p |
|---|---|---|---|---|---|---|---|---|---|---|
| `nextpick-policy` | 31,897,631 | 48,960 | 7.2% | [6.9, 7.4] | 23.0% | 35.3% | 31.17 | - | - | - |
| `nextpick-bans-aware-through-wlds-ig-tes` | 3,368,351 | 48,960 | 19.4% | [19.1, 19.8] | 51.9% | 69.8% | 9.64 | - | - | - |
| `nextpick-bans-control-through-wlds-ig-tes` | 3,350,559 | 48,960 | 19.5% | [19.1, 19.8] | 51.3% | 69.2% | 9.85 | - | - | - |
| `nextpick-bans-counter-only-through-wlds-ig-tes` | 3,366,943 | 48,960 | 19.3% | [19.0, 19.7] | 51.5% | 69.1% | 9.77 | - | - | - |
| `nextpick-bans-fearless-only-through-wlds-ig-tes` | 3,351,199 | 48,960 | 19.5% | [19.1, 19.8] | 51.3% | 69.2% | 9.83 | - | - | - |
| `nextpick-proonly` | 3,350,559 | 48,960 | 17.9% | [17.6, 18.2] | 48.9% | 66.6% | 10.80 | - | - | - |
| `nextpick-proonly-randomvalsplit-through-wlds-ig-tes` | 3,350,559 | 48,960 | 19.4% | [19.1, 19.8] | 51.3% | 69.2% | 9.85 | - | - | - |
| `nextpick-proonly-through-wlds-ig-tes` | 3,350,559 | 48,960 | 18.0% | [17.7, 18.3] | 49.2% | 67.3% | 10.54 | - | - | - |
| `nextpick-sharedsplitthrough-las-dk-c-t1-r` | 3,368,671 | 48,960 | 22.1% | [21.7, 22.5] | 56.3% | 73.5% | 8.59 | - | - | - |

### By phase

ban1 is where the board carries no information, so it is where a patch-conditioned model should separate from a frequency table - and where the table has historically won.

| model | ban1 | pick1 | ban2 | pick2 |
|---|---|---|---|---|
| `00-frequency-table` | 42.6% | 38.9% | 22.8% | 27.0% |
| `nextpick-proonly-nov13-k5-through-lcp-df-sw` | 42.9% | 58.4% | 40.0% | 62.3% |
| `nextpick-forest-flat-v1` | 31.4% | 36.2% | 28.1% | 44.4% |
| `nextpick-mlp-flat-v1` | 39.8% | 55.3% | 39.6% | 58.7% |
| `nextpick-proonly-nov13-k10-through-lcp-df-sw` | 41.8% | 57.8% | 42.9% | 63.2% |
| `nextpick-proonly-nov13-through-lcp-df-sw` | 42.4% | 55.4% | 36.9% | 57.9% |

Cells are top-5.

## Masked completion (the role board's question)

Given a complete draft with one pick hidden, who was it? 10 states per game; the other nine slots - including later picks - are visible.

### Comparable

| model | params | n | top-1 | 95% CI | top-5 | top-10 | mean rank | Δ vs ref (pp) | discordant | p |
|---|---|---|---|---|---|---|---|---|---|---|
| `masking-proonly-nov13-through-lcp-df-sw` | 2,489,522 | 24,480 | 24.7% | [24.2, 25.2] | 65.4% | 83.4% | 5.57 | - | - | - |
| `masking-forest-flat-v1` | 2,124,412 | 24,480 | 20.6% | [20.1, 21.1] | 58.5% | 76.2% | 7.54 | -4.1 | 2460/3470 | 3.2e-39 |
| `masking-mlp-flat-v1` | 2,997,029 | 24,480 | 25.0% | [24.5, 25.6] | 64.5% | 82.7% | 5.50 | +0.3 | 2627/2549 | 0.28 |

### Not comparable

Trained on a different split, so some of these test rows are in their training data. **These are training-set numbers.** Listed for provenance, not for ranking.

| model | params | n | top-1 | 95% CI | top-5 | top-10 | mean rank | Δ vs ref (pp) | discordant | p |
|---|---|---|---|---|---|---|---|---|---|---|
| `masking-finetune` | 28,611,250 | 24,480 | 15.8% | [15.3, 16.2] | 47.1% | 66.6% | 10.64 | - | - | - |
| `masking-soloqueue` | 31,714,995 | 24,480 | 11.9% | [11.5, 12.3] | 34.5% | 51.4% | 15.35 | - | - | - |
| `masking-proonly` | 2,489,266 | 24,480 | 24.4% | [23.9, 25.0] | 65.3% | 83.7% | 5.54 | - | - | - |
| `masking-proonly-randomvalsplit-through-wlds-ig-tes` | 2,489,266 | 24,480 | 25.5% | [25.0, 26.1] | 66.9% | 84.4% | 5.28 | - | - | - |
| `masking-proonly-through-wlds-ig-tes` | 2,489,266 | 24,480 | 26.7% | [26.2, 27.3] | 67.1% | 85.0% | 5.25 | - | - | - |
| `masking-sharedsplitthrough-las-dk-c-t1-r` | 2,489,522 | 24,480 | 26.4% | [25.8, 26.9] | 66.8% | 83.9% | 5.43 | - | - | - |

## Reading it honestly

`params` counts what the model stores. For anything with a weight vector (the set transformers, the flat MLP) it is `sum(p.numel())` off the loaded module. For the frequency table it is the table's own size - 20 moves x 173 champions x one per test patch - because that is what it costs to hold. For the forest it is the total node count across all trees, a tree ensemble having no weight vector. The three are not the same kind of quantity and the column is there to size a model, not to rank one.

The intervals are Wilson 95% on the pooled rate. Two rows whose intervals overlap are not separated by this split; the `Δ vs ref` column is the paired McNemar against the incumbent, which is the sharper test because it only counts items where the two models disagree. A `p` above 0.05 with a small `Δ` means "not shown to be different", not "equal".

The two tasks are never pooled: masked completion is an easier question by construction, so a single accuracy number across both families would rank the easier question higher rather than the better model.

