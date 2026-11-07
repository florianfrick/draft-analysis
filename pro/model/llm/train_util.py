"""Training-loop pieces shared by the warmup and the LoRA fine-tune.

Only two things live here, and both exist because the two stages must measure
the same way: the eval loop (which decides what "the number" means) and the
optimizer construction (which decides what the learning rates attach to).

The eval loop is `no_grad` and runs at a wider batch than training - eval costs
nothing but the report, while a training step pays for activations it has to
keep. Measured on this card, eval at batch 8 is ~3x the throughput of training
at batch 4.
"""
import time

import torch

from pro.model.llm import data as D
from pro.model.llm.metrics import Accumulator
from pro.model.llm.prompt import assemble


def evaluate(model, loader, tok, names, args, device, limit=0):
    model.eval()
    acc = Accumulator()
    t0 = time.time()
    with torch.no_grad():
        for i, (batch, samples) in enumerate(loader):
            if limit and i >= limit:
                break
            batch = D.to_device(batch, device)
            prompt = assemble(tok, samples, names, context=args.context,
                              loss_on_eos=args.loss_on_eos, device=device)
            out = model(batch, prompt)
            acc.update({k: v for k, v in out.items()
                        if k not in ("loss", "n_tokens")}, prompt)
    s = acc.summary()
    s["seconds"] = round(time.time() - t0, 1)
    return s


def cosine_lr(step, total, warm):
    """Linear warmup then cosine to zero. Returns a multiplier, not a rate."""
    import math
    if step < warm:
        return step / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * t))


def describe_trainables(model, projector_lr, llm_lr):
    """Two param groups: the projector and the LLM adapters learn at different
    rates, because one is a fresh 2.9M-parameter MLP and the other is a low-rank
    perturbation of a pretrained model. A single rate forces one of them to be
    wrong by an order of magnitude."""
    proj = [p for p in model.projector.parameters() if p.requires_grad]
    other = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    return [
        {"params": proj, "lr": projector_lr, "name": "projector"},
        {"params": other, "lr": llm_lr, "name": "llm"},
    ], {"projector": sum(p.numel() for p in proj),
        "llm": sum(p.numel() for p in other)}


def make_optimizer(groups, weight_decay=0.0):
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
