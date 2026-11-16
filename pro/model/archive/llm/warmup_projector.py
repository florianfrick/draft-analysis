"""Stage 1: train the projector with the encoder and the LLM both frozen.

The question: **is the encoder's 10 slot vectors enough for a frozen 4B model
to name the next champion?** Everything except the MLP is fixed, so the only
thing that can improve is the channel - which is exactly what makes a bad
result informative. If the ceiling here is low, no amount of LoRA in stage 2
fixes it, because stage 2 can only re-arrange what arrives.

The loss is cross-entropy on the answer tokens only (`-100` on the system
text, the board, the task line and the EOS), as specified. `--loss-on-eos`
adds the `<|im_end|>` so the model also learns to stop; it is off by default
because a projector that has learned nothing yet will happily spend its
gradient on the stop token instead of the name.

Reported per split, all on the answer positions:
  ce       mean CE per answer token (the training objective)
  tok_acc  fraction of answer tokens right under teacher forcing
  exact    fraction of rows where the *whole* name is right
  ft_top1/5  136-way first-token readout (see prompt.py: only 136 distinct
           first tokens exist for 173 names, so this is NOT comparable to the
           policy head's 173-way top-5)
  *_legal  the same after removing champions that are taken or fearless
           unavailable, on the rows where the true answer survives the filter
"""
import argparse
import json
import math
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
from pro.model.archive.llm.metrics import Accumulator, chance_levels  # noqa: E402
from pro.model.archive.llm import train_util as TU    # noqa: E402
from pro.model.archive.llm.prompt import assemble    # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--max-train", type=int, default=12000,
                   help="cap on training games (0 = all 42,121)")
    p.add_argument("--val-games", type=int, default=200,
                   help="eval games; 0 = the whole 4,758-game val split")
    p.add_argument("--eval-batch-size", type=int, default=8,
                   help="eval runs under no_grad, so it can batch wider "
                        "than training; the eval set is the cost of the "
                        "report, not of the fit")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--n-soft", type=int, default=10)
    p.add_argument("--projector-dropout", type=float, default=0.0)
    p.add_argument("--context", default="soft+text",
                   choices=["soft+text", "soft"])
    p.add_argument("--prompt-format", default=C.PROMPT_FORMAT,
                   choices=["v1", "v2", "v3"],
                   help="v2 labels the turn and the pick's lane; "
                        "v1 is the format every number before "
                        "2026-11-08 was measured on")
    p.add_argument("--loss-on-eos", action="store_true")
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--limit-train-steps", type=int, default=0)
    p.add_argument("--limit-val-batches", type=int, default=0)
    p.add_argument("--out-dir", default="")
    p.add_argument("--init-projector", default="",
                   help="start from these projector weights with a FRESH "
                        "optimizer and schedule (a new cycle)")
    p.add_argument("--resume", default="",
                   help="continue a run from its out-dir: restores weights, "
                        "optimizer, schedule and step. `--epochs` larger than "
                        "the saved run extends it, because the cosine is "
                        "recomputed over the new total and the LR comes back up")
    p.add_argument("--tag", default="")
    p.add_argument("--debug-mem", action="store_true",
                   help="print allocator state at each point inside a step")
    return p.parse_args(argv)


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
                              loss_on_eos=args.loss_on_eos, device=device,
                              prompt_format=args.prompt_format)
            out = model(batch, prompt)
            acc.update({k: v for k, v in out.items()
                        if k not in ("loss", "n_tokens")}, prompt)
    s = acc.summary()
    s["seconds"] = round(time.time() - t0, 1)
    return s


def _m(tag):
    """Allocator state, for `--debug-mem`."""
    print("    %-5s alloc %.3f reserved %.3f free %.3f"
          % (tag, torch.cuda.memory_allocated() / 1e9,
             torch.cuda.memory_reserved() / 1e9,
             torch.cuda.mem_get_info()[0] / 1e9), flush=True)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = C.load_rows(args.corpus)
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    n_unk, worst = C.patch_coverage(rows, patch_index)
    if n_unk:
        print("patch: %d drafts (%.2f%%) are newer than the encoder's table "
              "-> UNK %s" % (n_unk, 100.0 * n_unk / len(rows),
                             [w[0] for w in worst][:4]), flush=True)
    names = C.champ_names()
    print("corpus %d drafts | train %d val %d test %d | patches %d"
          % (len(rows), len(split["train"]), len(split["val"]),
             len(split["test"]), len(keys)), flush=True)

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
    print("train games %d (%d steps/epoch, eff. batch %d) | val states %d"
          % (len(tr_ds), len(tr_ld) // args.grad_accum, args.batch_size
             * args.grad_accum, len(va_ds)), flush=True)

    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)
    C.assert_patch_compat(cfg, patch_index)
    print("encoder D=%d params %.2fM (frozen)"
          % (cfg["embed_dim"], C.param_count(enc) / 1e6), flush=True)

    llm = B.load_llm(args.model, quantize=not args.no_4bit,
                     device=device.index or 0)
    B.freeze(llm)
    if not args.no_checkpointing:
        B.enable_checkpointing(llm)
    d_llm = llm.get_input_embeddings().weight.shape[1]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = B.build_draft_llm(enc, llm, tok, names, cfg["embed_dim"], d_llm,
                              args.hidden, n_soft=args.n_soft,
                              dropout=args.projector_dropout).to(device)
    print("projector %.2fM params | %s"
          % (C.param_count(model.projector) / 1e6, model.calibration),
          flush=True)
    print("vram after build %.2f GB"
          % (torch.cuda.memory_allocated() / 1e9), flush=True)
    print("alloc conf %r"
          % os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"), flush=True)

    params = list(model.projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=args.weight_decay)
    total = max(1, len(tr_ld) // args.grad_accum) * args.epochs
    warm = max(1, int(total * args.warmup_frac))

    def lr_at(step):
        if step < warm:
            return step / warm
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    out_dir = Path(args.out_dir or
                   (C.CKPT_DIR / ("llm-warmup" + (("-" + args.tag)
                                                  if args.tag else ""))))
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best = [], {"ce": float("inf")}
    step = 0
    start_epoch = 0
    if args.resume:
        d, st = TU.load_resume(args.resume)
        B.load_projector(model, d / "projector.pt")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = int(st["step"])
        # The saved epoch is the one that *finished*, so training restarts on
        # the next one. Resume is epoch-granular: a crash costs the epoch in
        # flight, not the run.
        start_epoch = int(st["epoch"]) + 1
        print("resumed from %s at step %d (epoch %d done) | %d of %d steps "
              "left" % (d, step, st["epoch"], total - step, total), flush=True)
        if start_epoch >= args.epochs:
            print("  nothing to do: --epochs %d but %d epochs are already "
                  "done. Pass a larger --epochs to extend the run."
                  % (args.epochs, start_epoch), flush=True)
            return None
    elif args.init_projector:
        blob = B.load_projector(model, args.init_projector)
        ck_fmt = (blob.get("args") or {}).get("prompt_format")
        if ck_fmt and ck_fmt != args.prompt_format:
            raise SystemExit(
                "%s was warmed up with prompt_format=%s but this run uses %s. "
                "The projector learned to read that text; starting from it "
                "with different text is worse than starting from scratch."
                % (args.init_projector, ck_fmt, args.prompt_format))
        print("projector initialised from %s (val %s)"
              % (args.init_projector,
                 json.dumps(blob.get("val", {}))[:120]), flush=True)

    for epoch in range(start_epoch, args.epochs):
        tr_ds.set_epoch(epoch)
        # Set the *parts* to train rather than the whole model to train and
        # then back: `model.train()` walks every descendant, including the
        # 4-bit trunk, and the trunk's training flag decides whether the
        # hybrid-attention layers keep their recurrent state around. Doing it
        # this way makes the frozen trunk unambiguously eval.
        model.eval()
        model.projector.train()
        run, epoch_run, t0 = Accumulator(), Accumulator(), time.time()
        t_last = t0
        opt.zero_grad(set_to_none=True)
        for i, (batch, samples) in enumerate(tr_ld):
            batch = D.to_device(batch, device)
            prompt = assemble(tok, samples, names, context=args.context,
                              loss_on_eos=args.loss_on_eos, device=device,
                              prompt_format=args.prompt_format)
            out = model(batch, prompt)
            if args.debug_mem:
                torch.cuda.synchronize()
                _m("fwd")
            (out["loss"] / args.grad_accum).backward()
            if args.debug_mem:
                torch.cuda.synchronize()
                _m("bwd")
            per = {k: v for k, v in out.items()
                   if k not in ("loss", "n_tokens")}
            run.update(per, prompt)
            epoch_run.update(per, prompt)
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if args.debug_mem:
                    torch.cuda.synchronize()
                    _m("step")
                step += 1
                if step % args.log_every == 0:
                    s = run.summary()
                    now = time.time()
                    print("ep %d step %d/%d | ce %.3f tok %.3f exact %.3f "
                          "ft@1 %.3f ft@5 %.3f | %.1fs (%.2fs/step) | "
                          "mem %.2f/%.2f free %.2f"
                          % (epoch, step, total, s["ce"], s["tok_acc"],
                             s["exact"], s["ft_top1"], s["ft_top5"],
                             now - t0, (now - t_last) / args.log_every,
                             torch.cuda.memory_allocated() / 1e9,
                             torch.cuda.memory_reserved() / 1e9,
                             torch.cuda.mem_get_info()[0] / 1e9), flush=True)
                    run.reset()
                    t_last = now
                if args.limit_train_steps and step >= args.limit_train_steps:
                    break
        val = evaluate(model, va_ld, tok, names, args, device,
                       limit=args.limit_val_batches)
        print("== epoch %d | train ce %.3f | val %s"
              % (epoch, epoch_run.summary()["ce"],
                 json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in val.items()})), flush=True)
        history.append({"epoch": epoch, "step": step, "val": val})
        if val["ce"] == val["ce"] and val["ce"] < best["ce"]:
            best = dict(val, epoch=epoch, step=step)
            B.save_projector(model, out_dir / "projector.pt",
                             extra={"args": vars(args),
                                    "calibration": model.calibration,
                                    "val": val})
        # Written for every epoch, not only when the run improves: "continue
        # from here" has to mean here, not "from the best epoch".
        TU.save_resume(out_dir, model, opt, sched, step, epoch,
                       extra={"args": vars(args)})
        if args.limit_train_steps and step >= args.limit_train_steps:
            break

    first, owner = C.candidate_tables(tok, names)
    report = {
        "protocol": {
            "stage": "projector warmup (encoder + LLM frozen)",
            "model": args.model, "encoder": args.encoder,
            "corpus": args.corpus, "context": args.context,
            "prompt_format": args.prompt_format,
            "n_soft": args.n_soft, "projector_hidden": args.hidden,
            "loss_on_eos": args.loss_on_eos,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
            "max_train_games": len(tr_ds),
            "val_games": len(va_ds) // C.N_MOVES,
            "trainable_params": C.param_count(model.projector),
            "calibration": model.calibration,
            "chance_first_token": chance_levels(len(first)),
            "baseline_policy_val": C.BASELINE_VAL,
        },
        "best": best, "history": history,
        "note": ("ft_top* is a %d-way readout over distinct first tokens, not "
                 "a 173-way champion ranking; do not compare it to "
                 "baseline_policy_val." % len(first)),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1),
                                         encoding="utf-8")
    print("wrote %s" % out_dir, flush=True)
    return report


if __name__ == "__main__":
    main()
