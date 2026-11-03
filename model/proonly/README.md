# `model/proonly/` — pro-only draft model

The model that ships: **pro drafts only, from scratch, no solo queue anywhere.**
Encoder D=256 / 2 intra + 1 cross / 8 heads, plus a patch embedding, trained on
a within-patch split where the patch is a *seen* input.

Results, ablations and the improvement plan live in **`FINDINGS.md`** at the
repo root — that is the single record. This file is only about running the
folder.

## What is here

| file | what it does |
|---|---|
| `common.py` | corpus/split/eval plumbing: `CORPORA`, `load_split`, `build_encoder`, `encoder_from_checkpoint`, `SliceAccumulator` |
| `train_masked.py` | stage 1 — masked completion; from-scratch init, `--patch-emb` / `--no-patch-emb` toggles the patch embedding |
| `train_policy.py` | stage 2 — next-pick, initialised from stage 1 |
| `compare.py` | paired comparison of two checkpoints on identical rows (McNemar + game bootstrap + per-year) |
| `test_proonly.py` | 6 tests: patch id reaches the encoder, checkpoints round-trip, both trainers resolve the same split |
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

The three runs this table used to list (`p-patch-s0`, `m-patch-s0`,
`p-nopatch-s0`) were the previous generation: `p-nopatch-s0` was the no-patch
ablation control and is gone with its weights, and the other two were replaced
by the retrain above. **Their numbers survive** in `reports/` and in
`FINDINGS.md` §4, but their paired comparisons can no longer be re-run. The
current scores are in `STATUS.md` §9 and
`model/test_split/REPORT_test_split_evaluation.md`. The six pretrained-initialisation arms were moved to
`model/finetune/` + `model/policy/` — the pipeline is now where soloQ
initialisation is studied, and this folder is from-scratch only by design.

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

`--corpus all` switches to the wider 2017-2026 export (81,161 drafts, separate
cache file). It is **not** the default: `CSV_GLOB` is pinned to `202[2-9]_*` so
newly-arrived CSVs cannot silently swap the corpus out from under every
recorded number.

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
