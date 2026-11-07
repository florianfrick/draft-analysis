"""Stage 2: QLoRA fine-tune of the LLM with the projector still trainable.

Encoder frozen throughout (it is the "vision" tower and it is already good at
what it does); the LLM gets low-rank adapters and the projector keeps its own
learning rate. The loss is the same answer-only cross-entropy as the warmup, so
the two stages are comparable on the same rows.

Why QLoRA rather than LoRA or full fine-tuning: bf16 weights alone are 8.4 GB
on this card, so a full fine-tune is impossible and a plain LoRA still has to
hold the bf16 base. nf4 weights are 3.1 GB including the 248k x 2560 embedding
table, which leaves room for adapters, optimizer state and activations at
batch 4. `bitsandbytes` 4-bit backward is slower per step than bf16, but it is
the only configuration that runs at all here - the measured throughput is in
the report rather than assumed.

`--init-projector` seeds from the warmup. Starting LoRA from a random
projector would spend the adapter's capacity learning to read noise, which is
the one thing the warmup exists to avoid.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.llm import build as B          # noqa: E402
from pro.model.llm import common as C        # noqa: E402
from pro.model.llm import data as D          # noqa: E402
from pro.model.llm import train_util as TU   # noqa: E402
from pro.model.llm.metrics import Accumulator, chance_levels  # noqa: E402
from pro.model.llm.prompt import assemble    # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--init-projector", default="",
                   help="warmup projector.pt; empty = train from scratch")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2,
                   help="2 is the measured ceiling on an 8 GB card: at 4 the "
                        "forward peak crosses the card and WDDM spills, which "
                        "surfaces as 'CUDA driver error: device not ready' "
                        "rather than an OOM. Raise it only on a bigger GPU.")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--projector-lr", type=float, default=1e-3)
    p.add_argument("--llm-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--max-train", type=int, default=3000)
    p.add_argument("--val-games", type=int, default=60)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--n-soft", type=int, default=10)
    p.add_argument("--context", default="soft+text",
                   choices=["soft+text", "soft"])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--loss-on-eos", action="store_true")
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--limit-train-steps", type=int, default=0)
    p.add_argument("--limit-val-batches", type=int, default=0)
    p.add_argument("--out-dir", default="")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = C.load_rows(args.corpus)
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    print("corpus %d drafts | train %d val %d | patches %d"
          % (len(rows), len(split["train"]), len(split["val"]), len(keys)),
          flush=True)

    train_idx = split["train"]
    if args.max_train and args.max_train < len(train_idx):
        g = torch.Generator().manual_seed(args.seed)
        pick = torch.randperm(len(train_idx), generator=g)[:args.max_train]
        train_idx = [train_idx[i] for i in sorted(pick.tolist())]

    tr_ds = D.train_dataset(rows, train_idx, patch_index, seed=args.seed)
    va_ds = D.eval_dataset(rows, split["val"], patch_index, seed=args.seed,
                           val_games=args.val_games)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=D.collate, num_workers=0, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.eval_batch_size, shuffle=False,
                       collate_fn=D.collate, num_workers=0)
    print("train games %d | val states %d" % (len(tr_ds), len(va_ds)),
          flush=True)

    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)

    llm = B.load_llm(args.model, quantize=not args.no_4bit,
                     device=device.index or 0)
    B.freeze(llm)
    llm = B.attach_lora(llm, r=args.lora_r, alpha=args.lora_alpha,
                        dropout=args.lora_dropout,
                        quantized=not args.no_4bit)
    if not args.no_checkpointing:
        B.enable_checkpointing(llm)
    d_llm = llm.get_input_embeddings().weight.shape[1]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = B.build_draft_llm(enc, llm, tok, names, cfg["embed_dim"], d_llm,
                              args.hidden, n_soft=args.n_soft).to(device)

    if args.init_projector:
        blob = B.load_projector(model, args.init_projector)
        print("projector initialised from %s (val %s)"
              % (args.init_projector,
                 json.dumps(blob.get("val", {}))[:160]), flush=True)

    groups, counts = TU.describe_trainables(model, args.projector_lr,
                                            args.llm_lr)
    opt = TU.make_optimizer(groups, weight_decay=args.weight_decay)
    total = max(1, len(tr_ld) // args.grad_accum) * args.epochs
    warm = max(1, int(total * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: TU.cosine_lr(s, total, warm))

    print("trainable: projector %.2fM, llm %.2fM | vram %.2f GB | steps %d"
          % (counts["projector"] / 1e6, counts["llm"] / 1e6,
             torch.cuda.memory_allocated() / 1e9, total), flush=True)

    out_dir = Path(args.out_dir or
                   (C.CKPT_DIR / ("llm-lora" + (("-" + args.tag)
                                                if args.tag else ""))))
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best = [], {"ce": float("inf")}
    step = 0
    for epoch in range(args.epochs):
        tr_ds.set_epoch(epoch)
        model.train()
        model.encoder.eval()
        run, epoch_run, t0 = Accumulator(), Accumulator(), time.time()
        opt.zero_grad(set_to_none=True)
        for i, (batch, samples) in enumerate(tr_ld):
            batch = D.to_device(batch, device)
            prompt = assemble(tok, samples, names, context=args.context,
                              loss_on_eos=args.loss_on_eos, device=device)
            out = model(batch, prompt)
            (out["loss"] / args.grad_accum).backward()
            per = {k: v for k, v in out.items()
                   if k not in ("loss", "n_tokens")}
            run.update(per, prompt)
            epoch_run.update(per, prompt)
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    s = run.summary()
                    print("ep %d step %d/%d | ce %.3f tok %.3f exact %.3f "
                          "ft@1 %.3f ft@5 %.3f | %.0fs"
                          % (epoch, step, total, s["ce"], s["tok_acc"],
                             s["exact"], s["ft_top1"], s["ft_top5"],
                             time.time() - t0), flush=True)
                    run.reset()
                if args.limit_train_steps and step >= args.limit_train_steps:
                    break
        val = TU.evaluate(model, va_ld, tok, names, args, device,
                          limit=args.limit_val_batches)
        print("== epoch %d | train ce %.3f | val %s"
              % (epoch, epoch_run.summary()["ce"],
                 json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in val.items()})), flush=True)
        history.append({"epoch": epoch, "step": step, "val": val})
        if val["ce"] == val["ce"] and val["ce"] < best["ce"]:
            best = dict(val, epoch=epoch, step=step)
            model.llm.save_pretrained(str(out_dir / "adapter"))
            B.save_projector(model, out_dir / "projector.pt",
                             extra={"args": vars(args),
                                    "calibration": model.calibration,
                                    "val": val})
        if args.limit_train_steps and step >= args.limit_train_steps:
            break

    first, _ = C.candidate_tables(tok, names)
    report = {
        "protocol": {
            "stage": "QLoRA fine-tune (encoder frozen, projector + adapters)",
            "model": args.model, "encoder": args.encoder,
            "init_projector": args.init_projector,
            "corpus": args.corpus, "context": args.context,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "projector_lr": args.projector_lr, "llm_lr": args.llm_lr,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "epochs": args.epochs, "seed": args.seed,
            "max_train_games": len(tr_ds),
            "val_games": len(va_ds) // C.N_MOVES,
            "trainable": counts, "loss_scope": "answer tokens only",
            "chance_first_token": chance_levels(len(first)),
            "baseline_policy_val": C.BASELINE_VAL,
        },
        "best": best, "history": history,
        "note": ("ft_top* is a %d-way readout over distinct first tokens, not "
                 "a 173-way champion ranking." % len(first)),
        "next": ("The next step is a bootstrapped CoT SFT dataset: sample "
                 "reasoning traces, keep the ones that reach the correct "
                 "champion, and train on those."),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1),
                                         encoding="utf-8")
    print("wrote %s" % out_dir, flush=True)
    return report


if __name__ == "__main__":
    main()
