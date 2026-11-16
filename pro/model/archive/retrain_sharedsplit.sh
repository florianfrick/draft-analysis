#!/usr/bin/env bash
# One corpus, one split, one chain: encoder -> head -> projector -> reranker.
#
# Why this exists: until 2026-11-10 the head and the LLM track were believed to
# split the corpus differently, so no cross-model number was clean. That turned
# out to be one `seed=0` in pro/model/archive/llm/common.SPLIT overriding
# VAL_SPLIT_SEED (commit d815739). With the pin kept on both sides the two
# tracks produce *identical* train/val/test, so every stage below trains and
# validates on the same 42,145 / 4,763 / 7 games.
#
# The second change is `--pool-order to_move` on the head: the pooled vector
# becomes concat(v_me, v_them, diff, product) instead of blue-first, so "my
# team" is a fixed position rather than an XOR of `to_move` with the board. No
# new parameters, but a different function, so it is recorded in the checkpoint
# and read back by every loader (solo_queue/model/policy/test_pool_order.py).
#
# k = 20 for the reranker. The head's recall is 50.51 % at k=5 and 84.34 % at
# k=20 on the shared val, and a reranker cannot beat its own candidate list, so
# k=20 raises the ceiling by 34 pp. It also raises the share of *evaluable*
# rows from 50.5 % to 84.3 %. The risk is that the LLM's accuracy on a 20-way
# choice decays; the epoch-0 val below is the number that decides, and it is
# in-distribution (trained at k=20), which is why this could not be answered by
# scoring the k=5 adapter at k=20.
#
# Two gates, both fail-closed, because a bad encoder silently caps everything
# downstream:
#   A  stage 1 val champ_top5 within 0.5 pp of the shipped 55.98 %
#   B  stage 2 val top-5 within 1 pp of the shipped 50.56 %
#
# Run:  bash pro/model/retrain_sharedsplit.sh
# Log:  pro/model/retrain_sharedsplit.log (appended, so a failed attempt stays
#       on the record next to the run that replaced it)
set -euo pipefail
cd "$(dirname "$0")/../../.."

K=20
ROOT=pro/model/archive/checkpoints
# The proonly trainers refuse an --out-dir that does not end with the newest
# series in the corpus (`assert_out_dir_suffix`), so a checkpoint cannot be
# confused with one trained on an older frontier. The suffix is concatenated
# with no separator - `endswith`, not a hyphenated join.
SUF=through-las-dk-c-t1-r
ENC=$ROOT/masking-sharedsplit$SUF
HEAD=$ROOT/nextpick-sharedsplit$SUF
WARM=$ROOT/llm-warmup-sharedsplit
RR=$ROOT/llm-rerank-sharedsplit-k$K
PAIRS=$RR/pairs.json
LOG=pro/model/retrain_sharedsplit.log

mkdir -p "$ROOT"
touch "$LOG"
say() { printf '\n===== %s  %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
run() { echo "++ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

say "chain start: k=$K"

# ---------------------------------------------------------------- stage 1
# No --no-test: the pinned 7-game test series stays held out so this artefact
# and the three after it describe the same partition.
say "stage 1/5  encoder (masked pretraining, pinned test kept)"
run docker compose run --rm -T dev python pro/model/proonly/train_masked.py \
    --val-split random --seed 0 --no-wandb --out-dir "$ENC"

python - "$ENC/test_report.json" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
v = json.load(open(sys.argv[1]))["val"]
base5, got5 = 0.5598, v["champ_top5"]
print("gate A: encoder val champ_top5 %.4f vs shipped %.4f (%+.2f pp)"
      % (got5, base5, 100.0 * (got5 - base5)))
print("        champ_top1 %.4f (shipped 0.2075)  mask_loss %.4f (shipped "
      "3.0281, NOT gated)" % (v["champ_top1"], v["mask_loss"]))
# Gated on champ_top5, not mask_loss. Dropping --no-test removes the 7 pinned
# test games from the split pool, which re-draws the seeded per-patch 10 % -
# so this val is a *different* 4,763 games (8,060 masked slots vs 8,053) and a
# cross-run mask_loss comparison is measuring the draw, not the model. The
# first version of this gate did exactly that and tripped on +2.47 % while
# champ_top5 matched to 0.02 pp. Ranking is also what the head consumes.
if got5 < base5 - 0.005:
    sys.exit("GATE A FAILED: the encoder lost more than 0.5 pp of champ_top5 "
             "against the shipped one. Every stage below would inherit it. "
             "Stopping before spending the night.")
PY

# ---------------------------------------------------------------- stage 2
say "stage 2/5  next-pick head (ban-aware, me-first pool)"
run docker compose run --rm -T dev python pro/model/proonly/train_policy.py \
    --val-split random --seed 0 --no-wandb \
    --ban-side --ban-fearless --counter-rank 32 --pool-order to_move \
    --init-ckpt "$ENC/best.pt" --out-dir "$HEAD"

python - "$HEAD/test_report.json" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
v = json.load(open(sys.argv[1]))["val"]
base, got = 0.5056, v["top5"]
print("gate B: head val top-5 %.4f vs shipped %.4f (%+.2f pp)"
      % (got, base, 100.0 * (got - base)))
print("        top1 %.4f  loss %.4f" % (v["top1"], v["policy_loss"]))
if got < base - 0.01:
    sys.exit("GATE B FAILED: the head lost more than 1 pp of top-5 against "
             "the shipped one. The encoder and the pool change are the only "
             "differences, so one of them is the cause - look before running "
             "the LLM stages on top of it.")
PY

# ---------------------------------------------------------------- stage 3
# `--encoder` takes a checkpoint *directory name* under pro/model/checkpoints,
# not a path (pro/model/archive/llm/common.build_frozen_encoder).
#
# `--max-train 8000` is a deliberate deviation from the shipped warmup, which
# ran all 42,162 games. At batch 2 x accum 8 and ~7.5 s/step (measured on the
# reranker) that is 2,634 steps ~ 5.5 h on this card, which would eat the whole
# night for a stage whose only job is to give the adapter a projector that
# already reads the encoder's vectors. 8,000 games is ~500 steps and eight
# times the reranker's own training pool, so the mapping is learned; the
# difference is recorded here rather than left to be discovered.
say "stage 3/5  projector warmup (encoder + LLM frozen), 8k games"
run bash pro/model/archive/llm/run.sh python -m pro.model.archive.llm.finetune_lora \
    --encoder "masking-sharedsplit$SUF" --projector-lr 0.003 --llm-lr 0 \
    --epochs 1 \
    --max-train 8000 --val-split llmval --val-games 200 --out-dir "$WARM"

# ---------------------------------------------------------------- stage 4
# Head-only pass: the pairs are (game, move) rows whose truth is in the head's
# top-K, which is what makes a reranker trainable at all. Minutes, not hours.
say "stage 4/5  reranker pairs at k=$K (head only)"
mkdir -p "$RR"
run bash pro/model/archive/llm/run.sh python -m pro.model.archive.llm.rerank \
    --split rr --ckpt "$HEAD/best.pt" --pair-k "$K" --build-pairs "$PAIRS"

# ---------------------------------------------------------------- stage 5
# Two epochs because one is the whole budget at k=5 and the curve was still
# falling; at k=20 the question is whether epoch 0 is already better than the
# 33.85 % the k=5 reranker reaches on the full shared val.
say "stage 5/5  reranker at k=$K (2 epochs)"
run bash pro/model/archive/llm/run.sh python -m pro.model.archive.llm.finetune_lora \
    --encoder "masking-sharedsplit$SUF" --head "$HEAD/best.pt" \
    --init-projector "$WARM/projector.pt" --rerank "$K" --pairs "$PAIRS" \
    --val-split rr --epochs 2 --out-dir "$RR"

say "chain done. Artefacts:"
ls -1 "$ENC" "$HEAD" "$WARM" "$RR" 2>/dev/null | tee -a "$LOG"
say "read $RR/report.json: epoch 0 ft_top1 is the k=$K verdict."
