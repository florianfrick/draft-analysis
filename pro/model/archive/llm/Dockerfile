# Derived image for the LLM experiment (branch exp/finetune-llm).
#
# `lol-draft:dev` is deliberately minimal: python:3.13-slim plus the CUDA
# wheels, no compiler. That is fine for the set-transformer stages, which are
# pure torch, but Qwen3.5's hybrid linear-attention layers fall back to a
# reference PyTorch implementation without `flash-linear-attention`, and fla is
# Triton - which JIT-compiles its kernels and therefore needs a C compiler at
# run time.
#
# Measured, batch 4 x 160 tokens, forward+backward: ~6.0 s with the reference
# kernels, ~0.9 s with Triton once its cache is warm. The cache lives on the
# `torch-cache` volume, so the compile cost is paid once per volume, not once
# per container.
#
# The Python packages are NOT installed here: they live in the venv at
# pro/model/archive/llm/.venv, which is inside the bind mount, so they survive `--rm` and
# do not invalidate the 11.7 GB base image on every dependency change.
FROM lol-draft:dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# So `python` inside the container is the experiment's interpreter.
ENV PATH=/workspace/pro/model/archive/llm/.venv/bin:$PATH

WORKDIR /workspace
CMD ["bash"]
