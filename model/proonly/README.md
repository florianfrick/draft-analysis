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
| `train_masked.py` | stage 1 — masked completion, random init |
| `train_policy.py` | stage 2 — next-pick, initialised from stage 1 |
| `compare.py` | paired comparison of two checkpoints on identical rows (McNemar + game bootstrap + per-year) |
| `test_proonly.py` | 6 tests: patch id reaches the encoder, checkpoints round-trip, both trainers resolve the same split |
| `ckpt/` | **local only** (git-ignored) — 3 runs, 140MB |
| `reports/` | every run's `test_report.json` and every paired comparison, committed |

## The checkpoints that are kept

| run | role | test top-5 |
|---|---|---|
| `p-patch-s0` | **the shipped model** | 0.4839 (3.14x baseline) |
| `m-patch-s0` | its masked init, also the masked-completion model | 0.5669 |
| `p-nopatch-s0` | the no-patch control the ablation is measured against | 0.3894 |

Nine further runs (seed 1, the wider 2017-2026 corpus pair, and the scratch
control) were deleted as losing arms — 482MB → 140MB. **Their numbers are
preserved** in `reports/` and in `FINDINGS.md` §4, but their weights are gone,
so their paired comparisons can no longer be re-run.

## Train it

```bash
# stage 1: masked completion (~6 min)
docker compose run --rm dev python model/proonly/train_masked.py \
    --run-name m-patch-s0 --seed 0

# stage 2: next-pick, from that checkpoint (~9 min)
docker compose run --rm dev python model/proonly/train_policy.py \
    --run-name p-patch-s0 --seed 0 \
    --init-ckpt model/proonly/ckpt/run_*/best.pt --epochs 32
```

Add `--no-patch-emb` to either for the ablation arm. The patch table is
zero-initialised, so the two arms are bit-identical at step 0.

`--corpus all` switches to the wider 2017-2026 export (81,161 drafts, separate
cache file). It is **not** the default: `CSV_GLOB` is pinned to `202[2-9]_*` so
newly-arrived CSVs cannot silently swap the corpus out from under every
recorded number.

## Compare two checkpoints

```bash
docker compose run --rm dev python model/proonly/compare.py --task policy \
    --a model/proonly/ckpt/run_*_p-patch-s0/best.pt \
    --b model/proonly/ckpt/run_*_p-nopatch-s0/best.pt
docker compose run --rm dev python model/proonly/compare.py --task masked --a A.pt --b B.pt
```

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
