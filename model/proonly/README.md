# Pro-only training + patch embedding (`exp/pro-only-patch-emb`)

What happens if the encoder never sees soloQ: pro drafts only, from scratch, on
a split where the patch is a *seen* input. Three questions, in order:

1. Does masked completion work at all on 37k pro drafts? → yes, 1.38x a
   same-patch slot-frequency table, and the patch embedding is worth
   **+3.7 to +5.5 pp top-5** (paired, p < 1e-13, CI excludes zero at the game
   level).
2. Does next-pick work? → yes, 3.14x the per-move table, and the patch
   embedding is worth **+9.5 to +10.0 pp top-5**, concentrated exactly where
   the mechanism predicts: **ban1 2.43x vs 1.07x**, i.e. the first ban is
   predictable only with the patch.
3. Is pro-only better than the existing soloQ→pro pipeline? → on this window
   yes (+19.6 pp masked, +21.7 pp next-pick pooled), but the gap is almost
   entirely 2022-2024; on 2026 rows it is +4.9 / +8.6 pp. The existing pipeline
   is trained with a 75-day recency half-life and is *asked* about recent
   patches only, so the pooled number flatters the new model. Read the
   per-year rows.

Nothing here is trained from a soloQ checkpoint, and nothing reads a soloQ
shard — `--rehearse-frac` does not exist in these scripts.

## The split is the point

`data/pro play/pipeline.split_within_patch`: for every patch, the first ~80% of
its series are train, the last 10% val, the last 10% test, series never split.

| split | test patches seen in train | patch id at test time |
|---|---|---|
| `split_series_by_date` (repo default) | no | always UNK |
| `split_within_patch` (this branch) | yes, for 99 of 106 patches | real |

Under the global date split a patch embedding can only memorise, because every
test row's patch is unseen. That is why the earlier `policy-2` run measured
patch conditioning as *harmful* (0.2029 vs 0.2558 top-5, STATUS §6): the
model was asked to use an input that was UNK at test time. This branch changes
the question to the one deployment actually asks — you always know today's
patch, and you have its first 80% — and the same input is now worth +10 pp.

46,813 drafts → 37,313 train / 4,737 val / 4,763 test (10% / 10%), 99 patches
in test, 0 series straddling a split, full row coverage.

## Setup

| | masked | next-pick |
|---|---|---|
| encoder | D=256, 2 intra + 1 cross, 8 heads, dropout 0.1, random init | same, from the masked arm's `best.pt` |
| objective | 15% masked champ CE + 0.3·role CE | CE over 173, illegal moves at −inf |
| data | complete drafts | one prefix per game per epoch, blue/red mirror |
| batch / lr | 512 / 1e-3, warmup + cosine, 40 epochs | 512 / 1e-3 head, 1e-4 encoder, 32 epochs |
| weights | uniform (no recency) | uniform |
| selection | val `rel_champ_top5` | val `rel_top5` |

Uniform weights on purpose: the test window is the tail of *every* patch, so
down-weighting old games (the 75-day half-life the other stages use) would make
the old patches unlearnable and the metric a statement about recency.

Both arms of every pair are identical except `--no-patch-emb`. The patch
embedding is zero-initialised, so the two models are bit-identical at step 0.

## Masked completion

Test, 4,763 games / 8,060 masked slots. Baseline = per-slot popularity fit on
train (the strong null: in a pro draft slot → role is fixed).

| arm | seed | top-1 | top-5 | top-10 | base@5 | rel@5 | role-cond@5 |
|---|---|---|---|---|---|---|---|
| patch | 0 | 0.2089 | **0.5669** | 0.7484 | 0.4120 | 1.376x | 0.6509 |
| patch | 1 | 0.2133 | **0.5712** | 0.7509 | 0.4120 | 1.386x | 0.6317 |
| no patch | 0 | 0.1911 | 0.5303 | 0.7083 | 0.4120 | 1.287x | 0.5806 |
| no patch | 1 | 0.1872 | 0.5163 | 0.7061 | 0.4120 | 1.253x | 0.5575 |

Paired, same masked slots (`compare.py --task masked`):

| seed | Δ top-5 (patch − no patch) | McNemar p | game bootstrap 95% CI |
|---|---|---|---|
| 0 | **+3.66 pp** | 7.7e-14 | [+2.71, +4.56] |
| 1 | **+5.50 pp** | 6.5e-29 | [+4.56, +6.48] |

Seed spread *within* an arm is 0.4 pp (patch) and 1.4 pp (no patch), so the
between-arm gap is 3-4x the noise it has to beat — the opposite of the P3.1
attribute result, where the gap was smaller than the seed spread.

The curve also says how much data this is: val top-5 peaks at **epoch 6**
(0.558 / 0.561) and then falls monotonically to 0.487 / 0.496 by epoch 40.
37k drafts against a 2.5M-parameter encoder overfits after ~440 steps.
Selection is on val, so the reported checkpoints are the epoch-6 ones.

### More data: the 2017-2026 export

The folder also holds 2014-2021 OE exports, which the documented corpus
excludes (`CSV_GLOB` is pinned to 2022+; a bare glob would swap the corpus out
from under every existing script). `--corpus all` opts in: 81,161 drafts,
2017-2026, 64,737 / 8,198 / 8,226, own cache file.

| corpus | patch rel@5 | no patch rel@5 | Δ top-5 (paired) |
|---|---|---|---|
| 2022-2026 (37,313 train) | 1.376x | 1.287x | +3.66 pp (p=7.7e-14) |
| 2017-2026 (64,737 train) | **1.493x** | 1.346x | **+5.51 pp** (p=1.1e-44) |

So the pro-only model is data-limited, as the epoch curve suggested: 1.7x the
drafts buys +0.12 rel (patch) and +0.06 rel (no patch), and the patch
embedding's value is unchanged in size. Absolute top-5 *falls* slightly
(0.5669 -> 0.5565) only because the wider window includes 2017-2021, where
the same-patch baseline is weaker and the games are harder - the rel column is
the comparable one. Seed 0 only: this is a direction, not a measurement.

## Next-pick

Test, 95,260 states (20 prefixes x 4,763 games). Baseline = per-move
popularity fit on train; same-patch leave-one-out reference = 0.1576 top-5.

| slice | n | patch s0 | no patch s0 | patch s1 | no patch s1 | rel@5 (patch / none) |
|---|---|---|---|---|---|---|
| all top-1 | 95,260 | 0.1763 | 0.1360 | 0.1757 | 0.1372 | — |
| all top-5 | 95,260 | **0.4839** | 0.3894 | **0.4868** | 0.3873 | 3.14x / 2.52x |
| all top-10 | 95,260 | 0.6676 | 0.5496 | 0.6679 | 0.5488 | — |
| ban1 | 28,578 | **0.4246** | 0.1860 | **0.4339** | 0.1860 | **2.43x / 1.07x** |
| pick1 | 28,578 | 0.5564 | 0.4705 | 0.5552 | 0.4647 | 3.64x / 3.08x |
| ban2 | 19,052 | 0.3656 | 0.3594 | 0.3700 | 0.3684 | 2.95x / 2.90x |
| pick2 | 19,052 | 0.5828 | 0.6029 | 0.5801 | 0.5920 | 3.72x / 3.85x |

Paired (`compare.py --task policy`):

| seed | Δ top-5 | McNemar p | game bootstrap 95% CI |
|---|---|---|---|
| 0 | **+9.45 pp** | <1e-300 | [+9.16, +9.74] |
| 1 | **+9.95 pp** | <1e-300 | [+9.66, +10.24] |

The per-phase split is the interesting part, because it is a mechanism test
rather than a headline:

* **ban1: 1.07x → 2.43x.** With no picks on the board, the only thing that
  determines the answer is this patch's ban priority. Without a patch input the
  model is at the table (1.07x); with one it is 2.4x the table. This is the
  same slice the soloQ-based policy lost (0.98x, STATUS §3) — and the reason it
  lost was the protocol, not the idea.
* **ban2: 2.90x → 2.95x, pick2: 3.85x → 3.72x.** Once picks are on the board,
  composition dominates and the patch prior adds nothing — pick2 is *slightly
  better without* it, consistently across both seeds (-2.0 and -1.2 pp). The
  patch vector is not free: it is 107 x 256 parameters competing with a 37k-draft
  fit, and it is a small net negative on the slice where it carries no signal.
* Per-year deltas are uniform (-6.8 to -12.6 pp), so the pooled +9.5 pp is not
  a year-mix artifact.

**Does the masked stage earn its place?** Seed 0, patch arm, same schedule:
the policy trained from `--init-ckpt scratch` scores 0.4746 top-5 against
0.4839 for the masked-initialised one — **+0.93 pp** for the masked stage
(p=1.1e-19, CI [+0.73, +1.12]). Real but small: 32 epochs of next-move CE on
37k games is enough to learn the patch prior on its own, and ban1 is
essentially identical either way (0.4275 vs 0.4246). The masked stage is worth
~1/10th of what the patch embedding is worth.

## Against the existing pipeline, on the same rows

Same within-patch test split, `compare.py` between the kept checkpoints and
this branch's. Positive = pro-only better.

| rows | masked: finetune `best_mask.pt` → pro-only | next-pick: policy `ablF` → pro-only |
|---|---|---|
| all | 0.3710 → **0.5669** (+19.6 pp) | 0.2671 → **0.4839** (+21.7 pp) |
| 2022 | 0.2721 → 0.5839 (+31.2) | 0.1309 → 0.4821 (+35.1) |
| 2023 | 0.3460 → 0.6152 (+26.9) | 0.1865 → 0.5165 (+33.0) |
| 2024 | 0.3265 → 0.5800 (+25.4) | 0.2350 → 0.4650 (+23.0) |
| 2025 | 0.4143 → 0.5182 (+10.4) | 0.3640 → 0.4649 (+10.1) |
| 2026 | 0.4939 → 0.5431 (+4.9) | 0.4091 → 0.4951 (+8.6) |

**Read the rows, not the total.** The existing stages are trained with a
75-day recency half-life and their own protocol only ever asks them about
recent patches; this split asks every year equally, so the pooled gap is
mostly "the old model was deliberately told to ignore 2022". On the 2026 rows
- the deployment target - pro-only is +4.9 pp (masked) and +8.6 pp (next-pick),
and even that is confounded by the weighting difference, not a clean
measurement of "pro-only is better".

The clean comparisons are the paired ones above, where nothing differs but the
flag.

## Reproduce

```bash
# masked, both arms, both seeds (~6 min each)
docker compose run --rm dev python model/proonly/train_masked.py --run-name m-patch-s0 --seed 0
docker compose run --rm dev python model/proonly/train_masked.py --run-name m-nopatch-s0 --seed 0 --no-patch-emb
# next-pick, initialised from the matching masked checkpoint
docker compose run --rm dev python model/proonly/train_policy.py --run-name p-patch-s0 \
    --init-ckpt model/proonly/ckpt/run_*/best.pt
docker compose run --rm dev python model/proonly/train_policy.py --run-name p-nopatch-s0 \
    --init-ckpt model/proonly/ckpt/run_*/best.pt --no-patch-emb
# paired statistics on identical rows
docker compose run --rm dev python model/proonly/compare.py --task masked --a A.pt --b B.pt
docker compose run --rm dev python model/proonly/compare.py --task policy --a A.pt --b B.pt
```

`--corpus all` switches to the wider 2017-2026 export (81,161 drafts, separate
cache). Tests: `model/proonly/test_proonly.py`, `model/tests/test_patch_embed.py`,
and the split tests in `data/pro play/test_pipeline.py`.

Kept evidence, weights deleted-safe: `reports/` holds every run's
`test_report.json` (pooled metrics + per-patch and per-year slices) and every
paired comparison as JSON + transcript, because the checkpoints themselves are
git-ignored. `ckpt/run_*/best.pt` stays local (482MB for the 9 kept runs) - the
comparisons need the losing arm too, so both arms and both seeds are kept.

## What this does not settle

* **Only one architecture size.** D=256/2+1 was chosen to match the P3.1
  budget. The epoch curve says the fit is data-limited, so a smaller model or
  heavier dropout would probably score higher; the patch-vs-no-patch gap would
  have to be re-measured there.
* **Only 2022-2026 for the main grid.** The wider corpus is seed 0 only, and
  its paired result is the same size as the 2022-2026 one, so "more data helps"
  is a direction rather than a measured curve.
* **`rel@5` is not comparable across protocols.** This window's baseline
  (0.412) is much stronger than the patch-honest one (0.373), so pro-only at
  1.38x and the finetune at 1.51x cannot be read as "the finetune is better".
  Absolute top-5 on the same rows is the only cross-protocol number here.
* **The masked stage is worth +0.9 pp to the policy**, not the +9.5 pp the
  patch embedding is worth (seed 0, paired). If the goal is next-pick, the
  masked stage is nearly optional at this data size.
