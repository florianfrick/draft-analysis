#!/usr/bin/env bash
# Run anything from pro/model/archive/llm inside the derived container.
#
# The base `lol-draft:dev` image has no compiler, and Qwen3.5's hybrid
# linear-attention layers need Triton (which JIT-compiles) to be fast enough to
# train on. `pro/model/archive/llm/Dockerfile` adds gcc and points PATH at the venv; this
# wrapper supplies the same mounts the compose file uses, so the HF cache and
# the Triton kernel cache survive between runs.
#
#   bash pro/model/archive/llm/run.sh python -m pro.model.archive.llm.warmup_projector --epochs 1
#   bash pro/model/archive/llm/run.sh python -m pro.model.archive.llm.finetune_lora --epochs 1
#
# The mount is the REPO ROOT, not this folder: `common.py` puts the repo root on
# `sys.path` so it can import `solo_queue.model.pretraining`, and the packages
# are addressed as `pro.model.archive.llm.*`. Mounting `pro/` instead (what `../..` gave
# before the folder moved from `model/llm` to `pro/model/archive/llm`) makes both fail
# with `No module named 'pro'`.
#
# `expandable_segments:True` is not a micro-optimisation here, it is the
# difference between 0.64 s/step and 5.6 s/step. The caching allocator's
# default size-bucket scheme fragments badly against this workload's variable
# sequence lengths; on WDDM it then reserves past the card's 8.19 GiB and every
# later step spills to host memory. Measured over 16 identical steps: reserved
# 7.46 -> 9.09 GB and rising without it, flat at 7.06 GB with it.
# Override with PYTORCH_CUDA_ALLOC_CONF=... in the environment if you want to
# compare.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

# Git Bash rewrites `/workspace` into a Windows path before docker sees it, so
# the container-side paths below must be exempted from MSYS path conversion.
export MSYS_NO_PATHCONV=1

docker run --rm -i --gpus all \
  -v "$PWD":/workspace \
  -v loldraftanalysis_torch-cache:/root/.cache \
  -w /workspace \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  --shm-size 2g --init \
  lol-draft:llm "$@"
