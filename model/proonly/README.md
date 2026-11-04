# `model/proonly/` — pro-only draft model

The model that ships: **pro drafts only, from scratch, no solo queue anywhere.**
Encoder D=256 / 2 intra + 1 cross / 8 heads, plus a patch embedding, trained on
a within-patch split where the patch is a *seen* input.

Results, ablations and the improvement plan live in **`STATUS.md`** at the repo
root — that is the single record. This file is only about running the folder.

## What is here

| file | what it does |
|---|---|
| `common.py` | corpus/split/eval plumbing: `CORPORA`, `load_split`, `build_encoder`, `encoder_from_checkpoint`, `SliceAccumulator`, and the naming helpers `corpus_frontier` / `newest_train_series` / `through_suffix` / `check_out_dir_name` |
| `train_masked.py` | stage 1 — masked completion; from-scratch init, `--patch-emb` / `--no-patch-emb` toggles the patch embedding |
| `train_policy.py` | stage 2 — next-pick, initialised from stage 1 |
| `compare.py` | paired comparison of two checkpoints on identical rows (McNemar + game bootstrap + per-year) |
| `test_proonly.py` | 19 tests: patch id reaches the encoder, checkpoints round-trip, both trainers resolve the same split, the team-code table and its fallback, the `-through-` name, `--val-scope latest`, and that `compare.py --split shared` really is val for both protocols |
| `ckpt/` | **local only** (git-ignored) — scratch runs from `--out-dir`-less invocations; the kept artefacts are in `model/checkpoints/` |
| `reports/` | every run's `test_report.json` and every paired comparison, committed |

## The checkpoints that are kept

The pro-only runs no longer live in `model/proonly/ckpt/`. On 2026-11-03 every
checkpoint outside the soloQ pretrain was deleted and each model class was
retrained once into `model/checkpoints/<name>/`, where the name states the
inference and the stage:

| directory | stage | what it answers |
|---|---|---|
| `masking-proonly` | this folder's `train_masked.py` | masked completion, role board |
| `nextpick-proonly` | this folder's `train_policy.py` | next-move ranking |
| `masking-proonly-through-wlds-ig-tes` | `train_masked.py --no-test --val-scope latest` | the same, on all 46,879 drafts |
| `nextpick-proonly-through-wlds-ig-tes` | `train_policy.py`, init from the row above | the same, on all 46,879 drafts |

**The `-through-<league>-<a>-<b>` suffix** states which series the training data
ends on. `--out-dir` is refused unless it ends with the suffix derived from the
corpus (`common.through_suffix`), so a retrain cannot land in an old directory
name and pass for the run it replaced. `through-wlds-ig-tes` is the Worlds 2026
final, Invictus Gaming vs Top Esports on 2026-11-04; the two team tags come from
`data/pro play/team_codes.json` and fall back to a deterministic initialism for
an unlisted org.

The suffix names the **corpus** frontier, not the last row the optimiser saw:
val is the tail of the newest patch, so the newest games are val rows. The
checkpoint's `test_report.json` records both — `through` (corpus) and
`train_through` (last row in the train split, `WLDs|Gen.G|LYON|2026-10-25` for
this pair).

The three runs this table used to list (`p-patch-s0`, `m-patch-s0`,
`p-nopatch-s0`) were the previous generation: `p-nopatch-s0` was the no-patch
ablation control and is gone with its weights, and the other two were replaced
by the retrain above. **Their numbers survive** in `reports/` and in `STATUS.md`
§4, but their paired comparisons can no longer be re-run. The current scores are
in `STATUS.md` §7 and `model/test_split/REPORT_test_split_evaluation.md`. The six
pretrained-initialisation arms were moved to `model/finetune/` +
`model/policy/` — the pipeline is now where soloQ initialisation is studied, and
this folder is from-scratch only by design.

## Train it

```bash
# stage 1: masked completion (~8 min)
docker compose run --rm dev python model/proonly/train_masked.py \
    --run-name masking-proonly --out-dir model/checkpoints/masking-proonly --seed 0

# stage 2: next-pick, from that checkpoint (~13 min)
docker compose run --rm dev python model/proonly/train_policy.py \
    --run-name nextpick-proonly --out-dir model/checkpoints/nextpick-proonly \
    --seed 0 --init-ckpt model/checkpoints/masking-proonly/best.pt --epochs 40
```

`--out-dir` is what gives the artefact its stable, self-describing name; without
it the run lands in `ckpt/run_<timestamp>_<name>/`, which the frontend would
list but nobody would be able to read.

Add `--no-patch-emb` to either for the ablation arm. The patch table is
zero-initialised, so the two arms are bit-identical at step 0.

`--epochs` is not only how long to train: it sets `total_steps`, and therefore
the whole learning-rate schedule (warmup is 3% of it, then a decay to zero).
At the default 40 epochs the warmup is 87 steps against 73 steps per epoch, so
the *first* epoch is spent ramping up; at 2 epochs it is 4 steps and the run
is at full LR almost immediately. Two runs of the same code, seed and data
therefore disagree at epoch 1 if the epoch counts differ — which makes a
shorter run look better and makes a re-run at a different `--epochs` a
*different experiment*, not a reproduction.

`--corpus all` switches to the wider 2017-2026 export (81,227 drafts, separate
cache file). It is **not** the default: `CSV_GLOB` is pinned to `202[2-9]_*` so
newly-arrived CSVs cannot silently swap the corpus out from under every
recorded number.

### Refreshing on all the data (`--no-test`, `--val-scope latest`)

```bash
# stage 1 (~7 min), then stage 2 (~8 min) from it
docker compose run --rm dev python model/proonly/train_masked.py \
    --no-test --val-scope latest --val-frac 0.20 --seed 0 \
    --out-dir model/checkpoints/masking-proonly-through-wlds-ig-tes
docker compose run --rm dev python model/proonly/train_policy.py \
    --no-test --val-scope latest --val-frac 0.20 --seed 0 \
    --init-ckpt model/checkpoints/masking-proonly-through-wlds-ig-tes/best.pt \
    --out-dir model/checkpoints/nextpick-proonly-through-wlds-ig-tes
```

Two switches, and both trade a measurement for data:

* `--no-test` drops the pinned test split entirely. The pinned series become
  ordinary rows, `test` is empty and the trainer says so instead of writing a
  report of zeros. **The result has no held-out number.** Nothing in `STATUS.md`
  §7 describes these checkpoints; §7 belongs to the originals.
* `--val-scope latest` makes only the newest patch's tail val, so every other
  patch is 100% train. That is worth ~4,700 rows (42,100 → 46,827) and points
  the only selection signal at the distribution the model is actually asked
  about. `--val-frac 0.20` is a deliberate compromise: patch 16.20 has 262 rows,
  so 10% is 26 games (78 masked slots, a coin flip) and 40% would cost 105 of
  the newest games. It buys a *correct* val, not a precise one — at 52 games
  the difference between the selected epoch and the plateau is inside the noise.

The masked stage peaks early and then overfits: val `rel_champ_top5` reaches
2.17x at epoch 9 and drifts down to 1.78x by epoch 40 while train loss falls
4.33 → 0.73. The original `masking-proonly` picked epoch 5 on a 4,768-row val,
so the early peak is a property of the task, not of the smaller val. The policy
stage is the opposite — it selected epoch 20 and was still improving.

## Compare two checkpoints

```bash
# the two kept pro-only runs, masked vs policy
docker compose run --rm dev python model/proonly/compare.py --task masked \
    --a model/checkpoints/masking-proonly/best.pt \
    --b model/checkpoints/masking-finetune/best_mask.pt
docker compose run --rm dev python model/proonly/compare.py --task policy \
    --a model/checkpoints/nextpick-proonly/best.pt \
    --b model/checkpoints/nextpick-policy/best_policy.pt
```

Two checkpoints trained under *different splits* have no common val set, so
scoring one on the other's val decides the comparison by construction. `--split
shared` intersects them instead: rows that are val in both protocols, therefore
trained by neither. `--other` picks which protocol to intersect with.

```bash
# refresh vs original: 27 games, val for both, trained by neither
docker compose run --rm dev python model/proonly/compare.py --task policy \
    --split shared --other old \
    --a model/checkpoints/nextpick-proonly/best.pt \
    --b model/checkpoints/nextpick-proonly-through-wlds-ig-tes/best.pt

# refresh vs an all-patch-val run (--other all), and exhaustive masking
docker compose run --rm dev python model/proonly/compare.py --task masked \
    --split shared --other all --mask-mode slot \
    --a model/checkpoints/masking-proonly-through-wlds-ig-tes/best.pt \
    --b model/checkpoints/masking-proonly-allval-through-wlds-ig-tes/best.pt
```

`--mask-mode slot` hides each pick in turn (10 queries per game) instead of the
training objective's 15% random masking (~1.5), which is the protocol
`model/test_split/eval_test_split.py` uses — so it reproduces §7 exactly, and it
is the only mode with enough queries to rank two models on 27 games. The verdict
that came out of these runs is in STATUS §7 ("Is the 16.20-only val helping or
hurting?").

The pruned arms (`p-patch-s0`, `p-nopatch-s0`, seed 1, the pretrained-init
six) can no longer be re-compared: their weights are gone and their numbers
live in `reports/`.

Both models are scored on the identical rows and the script asserts it, so the
delta is paired rather than two independent percentages.

## What not to change

* **The split.** `pipeline.split_within_patch` is why the patch embedding is
  worth anything. Under `split_series_by_date` every test patch is unseen, the
  patch id is `UNK` at test time, and the same input measured **−5.3 pp**.
* **The weighting.** Uniform, not the 75-day half-life the other stages use:
  the test window is the tail of *every* patch, so recency weighting would make
  old patches unlearnable and turn the metric into a statement about recency.
* **The patch table size.** `n_patches` is written into the checkpoint config
  and `load_encoder_state` refuses to invent or drop the table. A checkpoint
  trained with a patch table cannot be loaded into an encoder without one.

## Tests

```bash
docker compose run --rm -T dev python -m pytest model/proonly/test_proonly.py -q
docker compose run --rm -T dev python -m pytest model/tests/test_patch_embed.py -q
docker compose run --rm -T dev python -m pytest "data/pro play/test_pipeline.py" -q
```

Run them per file: `model/policy/model.py` shadows `model/finetune/model.py`
under a single `pytest model/`.
