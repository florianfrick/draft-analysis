"""Where the VRAM actually goes, measured per phase.

`--vram after build` in the trainers prints `torch.cuda.memory_allocated()`,
which is the *weight tensors and nothing else* - it is read before the first
forward pass, so it cannot include the CUDA context, the cuBLAS/Triton
workspaces, the autograd graph or anything a step allocates. Reading it as "the
model uses 3.14 GB" is how a run ends up surprised at 7.9 GB on the counter.

This prints four numbers per batch size, which together explain the gap:

  weights     memory_allocated right after loading
  +fwd        peak during a forward pass (no grad)
  +bwd        peak during forward + backward (the training step)
  reserved    the allocator's high-water mark; what nvidia-smi mostly shows

The gap between `allocated` and `reserved` is fragmentation, and the gap
between `reserved` and `nvidia-smi` is the CUDA context plus other processes -
neither is visible to `memory_allocated`.

    bash pro/model/archive/llm/run.sh python -m pro.model.archive.llm.measure_memory --batch-sizes 1 2 4 8
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import build as B   # noqa: E402


def nvidia_smi_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0]) if out else -1
    except Exception:
        return -1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--seq-len", type=int, default=132)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args(argv)

    dev = torch.cuda.current_device()
    llm = B.load_llm(args.model, quantize=not args.no_4bit, device=dev)
    B.freeze(llm)
    B.enable_checkpointing(llm)
    emb = llm.get_input_embeddings()
    d = emb.weight.shape[1]

    torch.cuda.synchronize()
    w = torch.cuda.memory_allocated() / 1e9
    print("%-6s %8s %8s %8s %9s %9s" % ("B", "weights", "+fwd", "+bwd",
                                        "reserved", "smi_total"))
    print("%-6s %8.2f %8s %8s %9.2f %9d" % (
        "-", w, "-", "-", torch.cuda.memory_reserved() / 1e9,
        nvidia_smi_used_mb()))
    print("  (nvidia-smi minus weights = CUDA context + other processes)")

    for bs in args.batch_sizes:
        soft = torch.randn(bs, 10, d, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True)
        ids = torch.randint(0, 1000, (bs, args.seq_len), device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # forward only
        with torch.no_grad():
            e = torch.cat([soft, emb(ids)[:, 10:]], dim=1)
            llm.model(inputs_embeds=e, use_cache=False)
        torch.cuda.synchronize()
        fwd = torch.cuda.max_memory_allocated() / 1e9

        # forward + backward
        torch.cuda.reset_peak_memory_stats()
        e = torch.cat([soft, emb(ids)[:, 10:]], dim=1)
        h = llm.model(inputs_embeds=e, use_cache=False).last_hidden_state
        lg = llm.get_output_embeddings()(h[:, -5:].contiguous())
        lg.float().logsumexp(-1).mean().backward()
        torch.cuda.synchronize()
        bwd = torch.cuda.max_memory_allocated() / 1e9
        res = torch.cuda.memory_reserved() / 1e9
        print("%-6d %8.2f %8.2f %8.2f %9.2f %9d"
              % (bs, w, fwd, bwd, res, nvidia_smi_used_mb()))

        del soft, ids, e, h, lg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
