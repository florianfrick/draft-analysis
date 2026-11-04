# Environment (Docker)

The Python environment for this project is defined by `Dockerfile` +
`requirements.txt`. It exists because the interpreter that ran the analysis was
a machine-level install whose `site-packages` got wiped between sessions — a
torch/CUDA rebuild is 20 minutes and unreproducible. Inside the image the
versions are pinned, so a rebuild gives back exactly the same environment.

Everything (data, checkpoints, `wandb_runs/`, `test_report.json`) is
**bind-mounted from the host**. Nothing is baked into the image, and nothing
written inside the container is lost when it exits.

## What's in it

| | |
|---|---|
| Base | `python:3.13-slim` (matches the host's 3.13.12) |
| torch | `2.14.1+cu126` from the PyTorch cu126 index |
| CUDA | 12.6 runtime, shipped inside the wheel (host needs only the driver) |
| Rest | numpy, pandas, pyarrow, scipy, scikit-learn, duckdb, wandb, aiohttp, pytest |
| GPU | passed through by the `nvidia` runtime (already registered on this host) |
| Size | **11.7GB** — ~10GB of that is the CUDA stack (cuDNN 9.10, cuBLAS, NCCL, cuSPARSELt, cuSOLVER, triton) |

No `nvidia/cuda` base image on purpose: torch wheels carry the whole CUDA
runtime as `nvidia-*-cu12` pip packages, so an NVIDIA base image would add a
second, possibly disagreeing copy of the CUDA runtime for no benefit. The size
is not avoidable while staying on GPU — a CPU-only image is ~1GB, and the
`cu126` index is the only way to get a 12.6 runtime that matches this driver.

## First run

```bash
docker compose build dev                 # ~2.5GB of wheels, once
docker compose run --rm dev python -c "
import torch; print(torch.__version__, torch.cuda.is_available(),
                    torch.cuda.get_device_name(0))"
```

That last line must print `True` and your GPU name. If it prints `False`, the
container is running without the nvidia runtime — check that Docker Desktop is
using the WSL2 backend and that `nvidia-smi` works on the host.

## Day-to-day

```bash
# interactive shell, then run anything from the README as usual
docker compose run --rm dev

# train
docker compose run --rm dev python model/pretraining/train.py --epochs 30
docker compose run --rm dev python model/finetune/train.py --no-wandb

# evaluate a checkpoint under the eval protocol
docker compose run --rm dev python model/policy/evaluate.py \
    --ckpt model/checkpoints/nextpick-proonly/best.pt
```

`--rm` matters: without it every invocation leaves a stopped container behind.
`-T` skips TTY allocation, which keeps piped output clean in a script.

### Tests

Run each file separately — several of them collide on module names
(`pipeline`, `model`) if collected in one pytest process. All of these pass in
the image (177 passed, 1 skipped):

```bash
for t in "data/pro play/test_eval_protocol.py" \
         "data/pro play/test_pipeline.py" \
         "data/solo queue/process matchdata/test_pipeline.py" \
         "model/tests" \
         "model/finetune/test_finetune.py" \
         "model/policy" \
         "frontend/test_server.py"; do
  echo "=== $t"
  docker compose run --rm -T dev python -m pytest "$t" -q | tail -2
done
```

`model/policy` goes in as a *directory*: `model/policy/model.py` shadows the
`model` package when that folder is first on `sys.path`, which is what
`pytest model/policy/test_infer.py` does.
```

`soloq data analysis/explorer/test_explorer.py` is not a pytest suite — it is a
live smoke script that curls `http://127.0.0.1:8778` and executes at import, so
pytest reports a collection error. Start the explorer first and run it directly.

## Long runs

`docker compose run --rm` attaches to the foreground, which is what you want for
a training run you intend to watch. For a run you want to leave:

```bash
docker compose run --rm -d --name pretrain dev \
    python model/pretraining/train.py --epochs 40
docker compose logs -f pretrain
```

`init: true` in `docker-compose.yml` gives the container a PID 1 reaper, so
Ctrl-C kills the process group instead of orphaning a child that keeps holding
GPU memory.

## The frontend

`frontend/server.py` needs torch and a checkpoint, so it belongs in the
container — but it binds `127.0.0.1` by default, and a port published by Docker
arrives on the container's `eth0`, not its loopback. A plain `-p 8777:8777`
therefore returns connection-refused. Pass `--host 0.0.0.0` inside the
container and keep the *published* port bound to host loopback so it stays
local:

```bash
docker compose run --rm -T -p 127.0.0.1:8777:8777 dev \
    python frontend/server.py --host 0.0.0.0 --port 8777
```

For the next-move UI, load a policy checkpoint; the header's patch selector
defaults to 16.20 from that checkpoint's own patch table, so nothing else is
needed:

```bash
docker compose run --rm -T -p 127.0.0.1:8777:8777 dev \
    python frontend/server.py --host 0.0.0.0 --port 8777
```

That is the whole command: both slots default to the pro-only artefacts
(`masking-proonly` for the role board, `nextpick-proonly` for the next move).
To pin them explicitly, or to run the role board alone:

```bash
docker compose run --rm -T -p 127.0.0.1:8777:8777 dev \
    python frontend/server.py --host 0.0.0.0 --port 8777 \
    --ckpt model/checkpoints/masking-finetune/best_mask.pt \
    --policy-ckpt model/checkpoints/nextpick-policy/best_policy.pt

docker compose run --rm -T -p 127.0.0.1:8777:8777 dev \
    python frontend/server.py --host 0.0.0.0 --port 8777 --no-policy
```

The two slots are independent: `--ckpt` takes a `masking-*` checkpoint and
`--policy-ckpt` a `nextpick-*` one. Loading a policy checkpoint into the role
slot is refused at load rather than failing later.

The `127.0.0.1:` prefix on the publish spec is what keeps it off the network —
`0.0.0.0` on the inside is only reachable through that mapping.

## Adding a package

Edit `requirements.txt`, then rebuild. Keep the pin exact (`==`) — the whole
point is that a rebuild reproduces the environment rather than re-resolving it.

```bash
docker compose build dev
```

If you need a package only for a one-off experiment, install it in a throwaway
container instead of the image:

```bash
docker compose run --rm dev pip install <pkg>     # gone when the container exits
```

## Changing CUDA / torch version

The driver on this host is 12.6, so the wheel must be `cu126`. To move:

1. Change the `--extra-index-url` in `requirements.txt` (e.g. `.../whl/cu128`).
2. Change the local version tag on the torch pin (`torch==X.Y.Z+cu128`).
3. Rebuild, then re-run the GPU check above.

`torch` and `torchvision` are not used; only `torch` is pinned.

## Troubleshooting

| symptom | cause |
|---|---|
| `torch.cuda.is_available()` is `False` | container started without the nvidia runtime; use compose (it declares the reservation) or `docker run --gpus all` |
| `permission denied` writing checkpoints | the container runs as root; on a WSL2 bind mount this is normal and harmless, files still appear on the host |
| build fails resolving `torch==2.14.1+cu126` | the cu126 index no longer serves that version for cp313; check `pip index versions torch --index-url https://download.pytorch.org/whl/cu126` |
| OOM killer on a large batch | raise `shm_size` in `docker-compose.yml` (DataLoader workers use shared memory) |
