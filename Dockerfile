# syntax=docker/dockerfile:1
#
# Dev/training image for the LoL draft models.
#
# Design notes
# ------------
# * Base is plain `python:3.13-slim`, not an `nvidia/cuda` image. Since torch
#   2.0 the Linux wheels ship the whole CUDA runtime as `nvidia-*-cu12` pip
#   packages, so the only thing that must come from the host is the *driver*.
#   That pins the CUDA runtime to the same version as the wheel (12.6), which
#   the local driver supports, and avoids a second copy of CUDA in the image.
#   Note the image is still ~12GB: the CUDA stack itself (cuDNN, cuBLAS, NCCL,
#   cuSPARSELt, triton) is most of it. That is the cost of GPU torch, not waste.
# * Code and data are NOT baked in - they are bind-mounted at run time (see
#   docker-compose.yml). The repo is 9.4GB of checkpoints and CSVs; copying it
#   into a layer would make every code edit invalidate a multi-GB layer, and
#   checkpoints written inside the container would be invisible on the host.
# * No `PYTHONPATH`: the scripts manage their own sys.path. The repo root is a
#   package namespace with two tracks, `solo_queue/` and `pro/`; every entry
#   script inserts the **repo root** so `import solo_queue.model...` and
#   `import pro.model...` both resolve. The old convention relied on each
#   evaluator's own directory containing a `model.py` that shadowed the package
#   - that is gone now, and the two-level package name makes it impossible.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # torch spawns threads per core; in a container it sees all host cores and
    # oversubscribes. Leave it alone but make the OMP default explicit so it
    # can be overridden per-run.
    OMP_NUM_THREADS=8 \
    # so `nvidia-smi` / torch.cuda see the GPU through the nvidia runtime
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# libgomp1: required by torch's CPU kernels (threading runtime).
# The rest is quality-of-life for a long-lived dev container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
        git \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Dependencies first, alone, so editing code never re-installs torch.
# requirements.txt carries the cu126 extra-index and the +cu126 torch pin, so
# this single install is the whole environment.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Fail the build (not the first training run) if the wheel is broken.
# This only checks that the package imports and reports its CUDA build; the
# GPU itself is verified at run time (see DOCKER.md).
RUN python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"

CMD ["bash"]
