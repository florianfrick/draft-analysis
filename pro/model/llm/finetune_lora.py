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
from pro.model.llm import rerank as R        # noqa: E402
from pro.model.llm import train_util as TU   # noqa: E402
from pro.model.llm.metrics import Accumulator, chance_levels  # noqa: E402
from pro.model.llm.prompt import assemble, candidate_first_ids  # noqa: E402


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
    p.add_argument("--prompt-format", default=None,
                   choices=["v1", "v2", "v3", "v4"],
                   help="v2 labels the turn and the pick's lane; v1 is the "
                        "format every number before 2026-11-08 was measured "
                        "on; v4 adds the candidate list and is implied by "
                        "--rerank. Default: v4 with --rerank, else %s"
                        % C.PROMPT_FORMAT)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--loss-on-eos", action="store_true")
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-every", type=int, default=0,
                   help="also validate every N steps (0 = epoch end only); a mid-run val can save a checkpoint, so a long run that turns bad is visible instead of only at the end")
    p.add_argument("--limit-train-steps", type=int, default=0)
    p.add_argument("--limit-val-batches", type=int, default=0)
    p.add_argument("--out-dir", default="")
    p.add_argument("--init-adapter", default="",
                   help="start from this saved adapter with a FRESH optimizer "
                        "and schedule (a new cycle); r/alpha/dropout then come "
                        "from the adapter, not the flags")
    p.add_argument("--resume", default="",
                   help="continue a run from its out-dir: restores projector, "
                        "adapter, optimizer, schedule and step. `--epochs` "
                        "larger than the saved run extends it")
    p.add_argument("--tag", default="")
    p.add_argument("--rerank", type=int, default=0,
                   help="K > 0 trains the reranker: the answer is restricted "
                        "to the next-pick head's top-K candidates")
    p.add_argument("--pairs", default="",
                   help="reranker training pairs, from `rerank --build-pairs`")
    p.add_argument("--moves-per-game", type=int, default=1,
                   help="reranker only: usable moves drawn per game per epoch")
    p.add_argument("--head", default=str(R.HEAD_CKPT))
    p.add_argument("--val-split", default="llmval",
                   choices=["common", "llmval", "headheld"],
                   help="common = the 478 games held out from both models, "
                        "the only set a cross-model number can use")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.prompt_format is None:
        args.prompt_format = "v4" if args.rerank else C.PROMPT_FORMAT
    if args.rerank and args.prompt_format != "v4":
        raise SystemExit("--rerank needs --prompt-format v4: the candidate "
                         "list is what the answer is constrained to")
    if args.rerank and not args.pairs:
        raise SystemExit("--rerank needs --pairs (rerank --build-pairs "
                         "--split common --build-pairs <path>)")
    # Resolved before anything is built: the adapter has to be attached at
    # construction time, and the optimizer is created from the module it
    # attaches to, so a resume that arrives later would restore Adam moments
    # onto a different parameter list.
    resume_state = None
    if args.resume:
        rdir, resume_state = TU.load_resume(args.resume)
        args.init_adapter = str(rdir / "adapter")
        args.init_projector = str(rdir / "projector.pt")
        print("resuming from %s (step %d, epoch %d done)"
              % (rdir, resume_state["step"], resume_state["epoch"]),
              flush=True)
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
    print("corpus %d drafts | train %d val %d | patches %d"
          % (len(rows), len(split["train"]), len(split["val"]), len(keys)),
          flush=True)

    train_idx = split["train"]
    if args.max_train and args.max_train < len(train_idx):
        g = torch.Generator().manual_seed(args.seed)
        pick = torch.randperm(len(train_idx), generator=g)[:args.max_train]
        train_idx = [train_idx[i] for i in sorted(pick.tolist())]

    if args.rerank:
        blob = json.loads(Path(args.pairs).read_text())
        if int(blob.get("k", 0)) != args.rerank:
            raise SystemExit("pairs were built for k=%s but --rerank is %d"
                             % (blob.get("k"), args.rerank))
        tr_ds = D.PairDataset(rows, blob["pairs"], patch_index,
                              per_game=args.moves_per_game, seed=args.seed)
        _, eval_idx = R.split_games(rows, args.val_split)
        va_ds = D.eval_dataset(rows, eval_idx, patch_index, seed=args.seed,
                               val_games=0)
        print("rerank k=%d | %d usable pairs over %d games | %d eval games"
              % (args.rerank, len(blob["pairs"]), len(tr_ds.games),
                 len(eval_idx)), flush=True)
    else:
        tr_ds = D.train_dataset(rows, train_idx, patch_index, seed=args.seed)
        if args.val_split == "llmval":
            va_ds = D.eval_dataset(rows, split["val"], patch_index,
                                   seed=args.seed, val_games=args.val_games)
        else:
            _, eval_idx = R.split_games(rows, args.val_split)
            va_ds = D.eval_dataset(rows, eval_idx, patch_index,
                                   seed=args.seed, val_games=0)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=D.collate, num_workers=0, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.eval_batch_size, shuffle=False,
                       collate_fn=D.collate, num_workers=0)
    print("train games %d | val states %d" % (len(tr_ds), len(va_ds)),
          flush=True)
    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)
    C.assert_patch_compat(cfg, patch_index)

    llm = B.load_llm(args.model, quantize=not args.no_4bit,
                     device=device.index or 0)
    B.freeze(llm)
    llm = B.attach_lora(llm, r=args.lora_r, alpha=args.lora_alpha,
                        dropout=args.lora_dropout,
                        quantized=not args.no_4bit,
                        adapter_path=args.init_adapter)
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
        ck_fmt = (blob.get("args") or {}).get("prompt_format")
        if ck_fmt and ck_fmt != args.prompt_format:
            # v4 *is* v2 plus one appended line (the board lines are
            # byte-identical, and test_v4_is_v2_plus_one_candidate_line pins
            # that), so a v2 projector is the correct initialisation for a
            # reranker - the soft tokens still describe the same board. Any
            # other mismatch is the failure this guard exists for.
            if args.rerank and ck_fmt == "v2":
                print("note: %s was warmed up on v2; v4 only appends the "
                      "candidate line, so its projector is the right start"
                      % args.init_projector, flush=True)
            else:
                raise SystemExit(
                    "%s was warmed up with prompt_format=%s but this run uses "
                    "%s. The projector learned to read that text; starting from "
                    "it with different text is worse than starting from "
                    "scratch. Pass --prompt-format %s or a matching projector."
                    % (args.init_projector, ck_fmt, args.prompt_format, ck_fmt))
        print("projector initialised from %s (val %s)"
              % (args.init_projector,
                 json.dumps(blob.get("val", {}))[:160]), flush=True)

    head = None
    if args.rerank:
        head, _head_cfg = R.load_head(args.head, device)
        head.eval()
        print("rerank: k=%d from %s"
              % (args.rerank, Path(args.head).parent.name), flush=True)
    n_vocab = int(model.llm.get_input_embeddings().weight.shape[0])

    def make_prompt(batch, samples):
        """-> (prompt, keep). The candidates come from the same batch the LLM
        sees, so side-swap augmentation and the rendered text cannot disagree -
        there is no cache to fall out of sync with an augmented row."""
        cands = None
        if head is not None:
            cid, cp = R.candidates(head(batch)["policy_logits"], args.rerank)
            cands = [[(int(c), float(p)) for c, p in zip(a, b)]
                     for a, b in zip(cid.tolist(), cp.tolist())]
        prompt = assemble(tok, samples, names, context=args.context,
                          loss_on_eos=args.loss_on_eos, device=device,
                          prompt_format=args.prompt_format, candidates=cands)
        keep = None
        if cands is not None:
            keep = R.keep_mask(
                candidate_first_ids(tok, names, cands).to(device),
                prompt["ans_ids"][:, 0], n_vocab)
        return prompt, keep

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
    start_epoch = 0
    if resume_state is not None:
        opt.load_state_dict(resume_state["opt"])
        sched.load_state_dict(resume_state["sched"])
        step = int(resume_state["step"])
        start_epoch = int(resume_state["epoch"]) + 1
        print("restored optimizer/schedule at step %d | %d of %d steps left"
              % (step, total - step, total), flush=True)
        if start_epoch >= args.epochs:
            print("  nothing to do: --epochs %d but %d epochs are already "
                  "done. Pass a larger --epochs to extend the run."
                  % (args.epochs, start_epoch), flush=True)
            return None
    last = {"step": -1, "val": None}

    def run_val(epoch, step, label):
        """Evaluate, record, and checkpoint if this is the best val ce so far.

        A full-corpus epoch is ~4 h and the only save happens on improvement,
        so without a periodic call a run that degrades is invisible until it
        ends. `model.train()` is restored here because `TU.evaluate` leaves
        the model in eval mode. A step that was already validated (mid-epoch
        val landing exactly on the last step) is not evaluated twice.
        """
        nonlocal best
        if last["step"] == step:
            return last["val"]
        val = TU.evaluate(model, va_ld, tok, names, args, device,
                          limit=args.limit_val_batches,
                          prompt_fn=make_prompt if head is not None else None)
        print("== %s | step %d | val %s"
              % (label, step,
                 json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in val.items()})), flush=True)
        history.append({"epoch": epoch, "step": step, "label": label,
                        "val": val})
        if val["ce"] == val["ce"] and val["ce"] < best["ce"]:
            best = dict(val, epoch=epoch, step=step)
            model.llm.save_pretrained(str(out_dir / "adapter"))
            B.save_projector(model, out_dir / "projector.pt",
                             extra={"args": vars(args),
                                    "calibration": model.calibration,
                                    "val": val})
            print("   best so far (ce %.4f) -> saved" % val["ce"], flush=True)
        last["step"], last["val"] = step, val
        model.train()
        return val
    for epoch in range(start_epoch, args.epochs):
        tr_ds.set_epoch(epoch)
        model.train()
        model.encoder.eval()
        run, epoch_run, t0 = Accumulator(), Accumulator(), time.time()
        opt.zero_grad(set_to_none=True)
        for i, (batch, samples) in enumerate(tr_ld):
            batch = D.to_device(batch, device)
            prompt, keep = make_prompt(batch, samples)
            out = model(batch, prompt, keep=keep)
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
                if args.val_every and step % args.val_every == 0:
                    run_val(epoch, step, "mid")
                if args.limit_train_steps and step >= args.limit_train_steps:
                    break
        print("== epoch %d | train ce %.3f"
              % (epoch, epoch_run.summary()["ce"]), flush=True)
        run_val(epoch, step, "epoch %d" % epoch)
        TU.save_resume(out_dir, model, opt, sched, step, epoch,
                       extra={"args": vars(args)})
        if args.limit_train_steps and step >= args.limit_train_steps:
            break

    first, _ = C.candidate_tables(tok, names)
    report = {
        "protocol": {
            "stage": "QLoRA fine-tune (encoder frozen, projector + adapters)",
            "model": args.model, "encoder": args.encoder,
            "init_projector": args.init_projector,
            "corpus": args.corpus, "context": args.context,
            "prompt_format": args.prompt_format,
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
