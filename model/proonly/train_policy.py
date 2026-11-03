"""Pro-only next-move policy: encoder + policy head, no soloQ anywhere.

    python model/proonly/train_policy.py --init-ckpt model/proonly/ckpt/run_X/best.pt
    python model/proonly/train_policy.py --init-ckpt scratch --patch-emb

Second half of the branch's question. `train_masked.py` asks whether a
from-scratch pro encoder can complete a draft; this asks whether it can
recommend the next move - the task the product actually serves.

Init: `--init-ckpt` points at a `train_masked.py` checkpoint (the natural
pipeline: masked completion first, then the policy, exactly like
soloQ-pretrain -> policy but with pro data only), or `scratch` to train the
encoder on the next-move loss alone. The scratch arm is the control for "does
the masked stage earn its place at all".

Patch conditioning is one flag, and it covers both places a patch can enter:
`--patch-emb` (default) sizes the encoder's patch embedding *and* the policy's
state embedding; `--no-patch-emb` turns both off. That is deliberate - a
single "can the model see the patch" switch, rather than two half-ablations.
The measured reason it matters: with no picks on the board (move 0) the only
thing that determines the answer is this patch's ban priority, which is why
the soloQ-based policy scored 0.98x a lookup table on bans1 (STATUS.md).

Split: `pipeline.split_within_patch` - same as the masked stage, so the two
stages see the same train/val/test.

Loss: policy CE over 173 champions with taken/banned/fearless illegal moves at
-inf. No value head, no econ head: both measured at chance/worse in the soloQ
pipeline and neither is part of this question.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

PRO_PIPE, PRO_DATA, EP = C.PRO_PIPE, C.PRO_DATA, C.EP
PolicyValueModel = C.PolicyValueModel
PHASE_NAMES = {0: "ban1", 1: "pick1", 2: "ban2", 3: "pick2"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pro-only next-move policy")
    C.add_corpus_args(ap)
    ap.add_argument("--init-ckpt", default="scratch",
                    help="a train_masked.py checkpoint, or 'scratch'")
    ap.add_argument("--embed-dim", type=int, default=256,
                    help="only used with --init-ckpt scratch")
    ap.add_argument("--num-intra", type=int, default=2)
    ap.add_argument("--num-cross", type=int, default=1)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--patch-emb", dest="patch_emb", action="store_true",
                    default=True)
    ap.add_argument("--no-patch-emb", dest="patch_emb", action="store_false")
    ap.add_argument("--prefix-mode", choices=("random", "all"), default="random",
                    help="random = one prefix per game per epoch (the repo "
                         "default: the 20 prefixes of a game are near-"
                         "duplicates, so unrolling all of them memorises)")
    ap.add_argument("--swap-aug", dest="swap_aug", action="store_true",
                    default=True, help="mirror blue<->red with p=0.5")
    ap.add_argument("--no-swap-aug", dest="swap_aug", action="store_false")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3, help="heads")
    ap.add_argument("--lr-enc", type=float, default=None,
                    help="encoder; default 1e-3 for scratch, 1e-4 from a "
                         "masked checkpoint (a pretrained trunk needs less)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lol-draft-proonly")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out-dir", default="",
                    help="write the run here instead of ckpt/run_<ts>_<name>. "
                         "Used to give the kept artefacts a stable, "
                         "self-describing name (model/checkpoints/nextpick-proonly).")
    return ap.parse_args(argv)


def policy_loss(out, batch):
    logits = out["policy_logits"]
    tgt = batch["target_champ"]
    # A fully illegal row (every champion taken) would make CE NaN; the
    # datasets cannot produce one, but a poisoned batch must degrade the
    # metric rather than corrupt the run.
    bad = torch.isnan(logits).any(dim=1) | ~torch.isfinite(logits).any(dim=1)
    if bool(bad.any()):
        logits = torch.where(bad.unsqueeze(1), torch.zeros_like(logits), logits)
    return F.cross_entropy(logits, tgt)


@torch.no_grad()
def evaluate(model, loader, move_rank, args, device, slices=None):
    """Next-move top-k, split by phase and by exact move index.

    The per-move breakdown is the point: a pooled number hides the phase
    structure (the soloQ policy read 1.45x on bans pooled, which was 0.98x on
    ban1 and 2.38x on ban2). ban1 is the slice a patch embedding is supposed
    to move, so it has to be visible on its own.
    """
    model.eval()
    n = 0
    hits = {1: 0.0, 5: 0.0, 10: 0.0}
    base_hits = {1: 0.0, 5: 0.0, 10: 0.0}
    by_phase = {}
    by_move = {}
    loss_sum = 0.0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        cl = out["policy_logits"]
        tc = batch["target_champ"]
        truth = cl.gather(1, tc[:, None]).squeeze(1)
        ranks = (cl > truth[:, None]).sum(1)
        moves = batch["move_idx"].tolist()
        phases = batch["phase"].tolist()
        base = [move_rank[m][c] for m, c in zip(moves, tc.tolist())]
        cnt = tc.numel()
        loss_sum += float(policy_loss(out, batch)) * cnt
        for k in hits:
            hits[k] += float((ranks < k).sum())
            base_hits[k] += sum(1 for r in base if r < k)
        for r, ph, mv, b in zip(ranks.tolist(), phases, moves, base):
            for table, key in ((by_phase, PHASE_NAMES.get(ph, str(ph))),
                               (by_move, mv)):
                e = table.setdefault(key, {"n": 0, "h5": 0, "h1": 0,
                                           "b5": 0})
                e["n"] += 1
                e["h5"] += int(r < 5)
                e["h1"] += int(r < 1)
                e["b5"] += int(b < 5)
        if slices is not None:
            metas = batch.get("meta") or [{}] * cnt
            slices.add(ranks.tolist(), [m.get("patch") for m in metas], base)
        n += cnt
    out_m = {"policy_loss": loss_sum / max(1, n), "n": n}
    for k in hits:
        out_m["top%d" % k] = hits[k] / max(1, n)
        out_m["base_top%d" % k] = base_hits[k] / max(1, n)
    out_m["rel_top5"] = EP.baseline_relative(out_m["top5"],
                                             out_m["base_top5"])
    for table, name in ((by_phase, "phase"), (by_move, "move")):
        rows = {}
        for key in sorted(table):
            e = table[key]
            rows[str(key)] = {"n": e["n"], "top1": e["h1"] / e["n"],
                              "top5": e["h5"] / e["n"],
                              "base_top5": e["b5"] / e["n"],
                              "rel_top5": EP.baseline_relative(
                                  e["h5"] / e["n"], e["b5"] / e["n"])}
        out_m[name] = rows
    return out_m


def run_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb
        key = C.MODEL_DIR / "pretraining" / "wandb_key.txt"
        if "WANDB_API_KEY" not in os.environ and key.exists():
            os.environ["WANDB_API_KEY"] = key.read_text().strip()
        return wandb.init(project=args.wandb_project,
                          name=args.run_name or None, config=vars(args),
                          dir=str(HERE), settings=wandb.Settings(init_timeout=60))
    except Exception as exc:  # noqa: BLE001
        print("wandb disabled: %s" % exc)
        return None


def save_ckpt(path, model, args, cfg, metrics, step, epoch, patch_vocab,
              resume=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": cfg, "metrics": metrics,
               "step": step, "epoch": epoch, "patch_vocab": patch_vocab,
               "corpus": args.corpus, "patch_emb": bool(args.patch_emb),
               "init_ckpt": args.init_ckpt}
    if resume is not None:
        payload["resume"] = resume
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path))


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    rows, splits, patch_vocab, patch_index = C.load_split(args)
    num_champs = 173
    n_patches = (len(patch_vocab) + 1) if args.patch_emb else 0

    # ---- encoder
    if args.init_ckpt == "scratch":
        encoder = C.build_encoder(num_champs, args.embed_dim, args.num_intra,
                                  args.num_cross, args.num_heads, args.dropout,
                                  n_patches).to(device)
        lr_enc = args.lr if args.lr_enc is None else args.lr_enc
        print("encoder: scratch (%d patch rows)" % n_patches)
    else:
        encoder, cfg, info = C.encoder_from_checkpoint(args.init_ckpt,
                                                       n_patches, device)
        encoder = encoder.to(device)
        lr_enc = (1e-4 if args.lr_enc is None else args.lr_enc)
        print("encoder: %s (val rel_top5 %.3f, patch %s)"
              % (args.init_ckpt,
                 float(torch.load(args.init_ckpt, map_location="cpu",
                                  weights_only=False).get("metrics", {})
                       .get("rel_champ_top5", float("nan"))),
                 "ON" if n_patches else "OFF"))
        if info["missing"]:
            print("  warning: encoder missing %s (random init)"
                  % info["missing"][:4])

    model = PolicyValueModel(
        encoder, embed_dim=encoder.embed.champ_lookup.embedding_dim,
        state_dim=args.state_dim, n_econ=0, dropout=args.dropout,
        n_patches=n_patches, with_value=False).to(device)
    model.set_freeze_encoder(False)
    print("model: %.2fM params, lr head %.0e / enc %.0e, device %s"
          % (C.param_count(model) / 1e6, args.lr, lr_enc, device))

    def loader(idxs, mode, shuffle, seed):
        ds = PRO_DATA.ProNextMoveDataset(
            rows, idxs, num_champs=num_champs, patch_index=patch_index,
            prefix_mode=mode, swap_aug=(args.swap_aug and shuffle), seed=seed)
        return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                              collate_fn=PRO_DATA.pro_collate, num_workers=0,
                              pin_memory=torch.cuda.is_available())

    tr_ds, tr_loader = loader(splits["train"], args.prefix_mode, True,
                              args.seed)
    _, va_loader = loader(splits["val"], "all", False, 0)
    _, te_loader = loader(splits["test"], "all", False, 0)
    print("train examples/epoch %d, val %d, test %d"
          % (len(tr_ds), len(va_loader.dataset), len(te_loader.dataset)))

    move_rank = EP.move_popularity_ranks(rows, splits["train"], num_champs)

    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters()
                  if n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": head_params, "lr": args.lr},
                             {"params": enc_params, "lr": lr_enc}],
                            weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(tr_loader))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warm = max(1, int(total_steps * args.warmup_frac))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run = run_wandb(args)
    run_dir = Path(args.out_dir) if args.out_dir else \
        HERE / "ckpt" / ("run_" + time.strftime("%Y%m%d_%H%M%S") +
                         (("_" + args.run_name) if args.run_name else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    print("checkpoints ->", run_dir)
    cfg_out = {"num_champs": num_champs,
               "embed_dim": encoder.embed.champ_lookup.embedding_dim,
               "num_intra": len(encoder.intra_team.layers),
               "num_cross": len(encoder.cross_team),
               "num_heads": args.num_heads, "dropout": args.dropout,
               "state_dim": args.state_dim, "n_patches": n_patches,
               "n_econ": 0, "with_value": False, "attr_dim": 0}

    best = -1.0
    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if hasattr(tr_ds, "set_epoch"):
            tr_ds.set_epoch(epoch)
        model.train()
        agg, nb = 0.0, 0
        for batch in tr_loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v)
                     else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = policy_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss at step %d" % global_step)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            agg += float(loss.detach())
            nb += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            va = evaluate(model, va_loader, move_rank, args, device)
            print("ep %d step %d train %.4f | val top1 %.4f top5 %.4f "
                  "base5 %.4f rel %.2fx | ban1 %.3f pick1 %.3f ban2 %.3f "
                  "pick2 %.3f (%.0fs)"
                  % (epoch, global_step, agg / max(1, nb), va["top1"],
                     va["top5"], va["base_top5"], va["rel_top5"],
                     va["phase"].get("ban1", {}).get("rel_top5", float("nan")),
                     va["phase"].get("pick1", {}).get("rel_top5", float("nan")),
                     va["phase"].get("ban2", {}).get("rel_top5", float("nan")),
                     va["phase"].get("pick2", {}).get("rel_top5", float("nan")),
                     time.time() - t0))
            if run is not None:
                import wandb
                wandb.log({"train/loss": agg / max(1, nb),
                           "val/top1": va["top1"], "val/top5": va["top5"],
                           "val/rel_top5": va["rel_top5"]}, step=global_step)
            if va["rel_top5"] > best:
                best = va["rel_top5"]
                save_ckpt(run_dir / "best.pt", model, args, cfg_out,
                          {k: v for k, v in va.items()
                           if k not in ("phase", "move")} | {"epoch": epoch},
                          global_step, epoch, patch_vocab)
        save_ckpt(run_dir / "latest.pt", model, args, cfg_out, {"epoch": epoch},
                  global_step, epoch, patch_vocab,
                  resume={"opt": opt.state_dict(), "sched": sched.state_dict(),
                          "step": global_step, "epoch": epoch, "best": best,
                          "run_dir": str(run_dir)})
        if args.max_steps and global_step >= args.max_steps:
            break

    # ---- test report
    report = {"protocol": {
        "corpus": args.corpus, "patch_emb": bool(args.patch_emb),
        "n_patches": n_patches, "init_ckpt": args.init_ckpt,
        "split": "within-patch (train=head, val=tail); test = the pinned TEST_SERIES",
        "prefix_mode": args.prefix_mode, "swap_aug": args.swap_aug,
        "seed": args.seed, "n_train": len(splits["train"]),
        "n_val": len(splits["val"]), "n_test": len(splits["test"]),
        "selection": "rel_top5 (val)", "params": C.param_count(model)}}
    blob = torch.load(str(run_dir / "best.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(blob["model"])
    report["val"] = blob["metrics"]
    slices = C.SliceAccumulator()
    te = evaluate(model, te_loader, move_rank, args, device, slices=slices)
    # leave-one-out same-patch reference: the bar that decides whether the
    # model beats "just use this patch's move frequencies"
    report["same_patch_reference"] = EP.self_move_topk(
        rows, splits["test"], num_champs, k=(1, 5))
    report["test"] = te
    report["test"]["slices"] = slices.report()
    print("TEST top1 %.4f top5 %.4f (base %.4f, rel %.2fx; same-patch ref %.4f)"
          % (te["top1"], te["top5"], te["base_top5"], te["rel_top5"],
             report["same_patch_reference"]["top5"]))
    for ph in ("ban1", "pick1", "ban2", "pick2"):
        r = te["phase"].get(ph)
        if r:
            print("  %-5s n=%-6d top5 %.4f base %.4f rel %.2fx"
                  % (ph, r["n"], r["top5"], r["base_top5"], r["rel_top5"]))
    with open(run_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("done (%s)" % run_dir)
    if run is not None:
        import wandb
        wandb.finish()
    return report


def main():
    report = train(parse_args())
    out = HERE / "last_policy.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"protocol": report["protocol"], "val": report["val"],
                   "test": {k: v for k, v in report["test"].items()
                            if k != "slices"}}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
