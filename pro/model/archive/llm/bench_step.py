"""Steady-state throughput / allocator probe.

One real training step repeated N times, printing allocator state each step.
The point is to separate *live* tensor memory (`alloc`) from what the allocator
has reserved from the driver (`reserved`).  On WDDM those diverge: `reserved`
can creep past physical VRAM, after which every step pays for the spill and
step time is no longer a property of the batch.

    python -m pro.model.archive.llm.bench_step --model Qwen/Qwen3.5-4B --batch-size 4 \
        --steps 16 --context soft+text
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import build as B          # noqa: E402
from pro.model.archive.llm import common as C        # noqa: E402
from pro.model.archive.llm import data as D          # noqa: E402
from pro.model.archive.llm.prompt import assemble    # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--n-soft", type=int, default=10)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--context", default="soft+text", choices=["soft+text", "soft"])
    p.add_argument("--max-train", type=int, default=512)
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args(argv)

    print("PYTORCH_CUDA_ALLOC_CONF = %r"
          % os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"), flush=True)

    device = torch.device("cuda")
    torch.manual_seed(0)

    rows = C.load_rows(args.corpus)
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    train_idx = split["train"][:args.max_train]
    tr_ds = D.train_dataset(rows, train_idx, patch_index, seed=0)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=D.collate, num_workers=0, drop_last=True)

    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)
    llm = B.load_llm(args.model, quantize=not args.no_4bit, device=0)
    B.freeze(llm)
    if not args.no_checkpointing:
        B.enable_checkpointing(llm)
    d_llm = llm.get_input_embeddings().weight.shape[1]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = B.build_draft_llm(enc, llm, tok, names, cfg["embed_dim"], d_llm,
                              args.hidden, n_soft=args.n_soft).to(device)
    model.eval()
    model.encoder.eval()
    model.llm.eval()
    model.projector.train()
    params = list(model.projector.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)

    print("built | alloc %.2f GB | reserved %.2f GB | free %.2f GB"
          % (torch.cuda.memory_allocated() / 1e9,
             torch.cuda.memory_reserved() / 1e9,
             torch.cuda.mem_get_info()[0] / 1e9), flush=True)

    torch.cuda.reset_peak_memory_stats()
    print("\n%5s %8s %8s %8s %8s %8s %6s"
          % ("step", "s/step", "alloc", "reserv", "free", "peakres", "ntok"), flush=True)
    it = iter(tr_ld)
    t_prev = time.time()
    times = []
    for step in range(args.steps):
        try:
            batch, samples = next(it)
        except StopIteration:
            it = iter(tr_ld)
            batch, samples = next(it)
        batch = D.to_device(batch, device)
        pr = assemble(tok, samples, names, context=args.context,
                      with_answers=True, device=device)
        out = model(batch, pr)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        t = time.time()
        dt = t - t_prev
        t_prev = t
        if step >= 3:
            times.append(dt)
        if step < 3 or step % 4 == 0 or step == args.steps - 1:
            print("%5d %8.2f %8.2f %8.2f %8.2f %8.2f %6d"
                  % (step, dt,
                     torch.cuda.memory_allocated() / 1e9,
                     torch.cuda.memory_reserved() / 1e9,
                     torch.cuda.mem_get_info()[0] / 1e9,
                     torch.cuda.max_memory_reserved() / 1e9,
                     int(pr["attn"].sum())), flush=True)

    if times:
        med = sorted(times)[len(times) // 2]
        print("\nsteady: median %.2f s/step  min %.2f  max %.2f  | "
              "%.1f examples/s (batch %d)"
              % (med, min(times), max(times),
                 args.batch_size / med if med else 0, args.batch_size), flush=True)
    print("final: alloc %.2f  reserved %.2f  peak reserved %.2f  smi total %.2f"
          % (torch.cuda.memory_allocated() / 1e9,
             torch.cuda.memory_reserved() / 1e9,
             torch.cuda.max_memory_reserved() / 1e9,
             torch.cuda.mem_get_info()[1] / 1e9), flush=True)


if __name__ == "__main__":
    main()
