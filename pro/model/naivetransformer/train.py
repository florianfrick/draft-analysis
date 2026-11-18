"""Train the naive single-stack transformer on the canonical pro split.

    docker compose run --rm -T dev python pro/model/naivetransformer/train.py

Design contract (branch exp/transformer):

* No masked-draft pretraining stage. The 61-token sequence is built only from
  the prefix, so causal inference and causal training see the same inputs.
* Five *distinct, freshly drawn* prefixes per training game per epoch
  (`ProNextMoveDataset(prefix_mode='k', prefix_k=5)`). Val/test keep all 20.
* The exact same split as the nov13 model (`--canonical-split` forced here):
  39,720 / 4,771 / 2,448 games. The test_report.json records those counts so
  test_split/registry.py marks the checkpoint comparable.
* The checkpoint name is `nextpick-naivetransformer-...-through-<suffix>` so
  the existing suite discovers it as a next_move model. test_split/models.py
  dispatches on `config.arch` to the model's own loader.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.proonly import common as C  # noqa: E402
from pro.model.naivetransformer import model as NT  # noqa: E402

PRO_PIPE, PRO_DATA, EP = C.PRO_PIPE, C.PRO_DATA, C.EP
PHASE_NAMES = {0: "ban1", 1: "pick1", 2: "ban2", 3: "pick2"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_corpus_args(ap)
    # ---- architecture. Defaults fit an 8 GB card; the flags scale to ~1B
    # (which needs a much larger card + a smaller batch / sharded optimiser).
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-game-num", dest="use_game_num", action="store_false",
                    default=True)
    # ---- data
    ap.add_argument("--prefix-k", type=int, default=5,
                    help="distinct training prefixes per game per epoch (5)")
    ap.add_argument("--swap-aug", dest="swap_aug", action="store_true",
                    default=True, help="mirror blue<->red with p=0.5")
    ap.add_argument("--no-swap-aug", dest="swap_aug", action="store_false")
    # ---- optimisation
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke-run step cap (0 = full training)")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out-dir", default="",
                    help="checkpoint dir; must end in the corpus through-suffix")
    ap.add_argument("--init-ckpt", default="",
                    help="fine-tune from an existing nextpick-naivetransformer "
                         "best.pt (same architecture; the slot head is shared)")
    ap.add_argument("--slot-loss-weight", type=float, default=0.0,
                    help="weight on the empty-slot role-conditioned pick CE; "
                         "0 reproduces the plain next-move run")
    ap.add_argument("--tag", default="d512-l8-h8-k5",
                    help="architecture tag inserted in the default dir name")
    args = ap.parse_args(argv)
    # There is only one experiment split here. Force it instead of letting a
    # half-remembered CLI flag produce an incomparable checkpoint.
    args.canonical_split = True
    return args


def policy_loss(out, batch, slot_weight=0.0):
    logits = out["policy_logits"]
    tgt = batch["target_champ"]
    bad = torch.isnan(logits).any(dim=1) | ~torch.isfinite(logits).any(dim=1)
    if bool(bad.any()):
        logits = torch.where(bad.unsqueeze(1), torch.zeros_like(logits), logits)
    loss = F.cross_entropy(logits, tgt)
    if slot_weight and "slot_logits" in out:
        # Only slots still empty at this prefix have a role-conditioned label.
        empty = batch["pad_mask"].bool()
        sl = out["slot_logits"][empty]
        st = batch["slot_target"][empty]
        if sl.numel():
            loss = loss + slot_weight * F.cross_entropy(sl, st)
    return loss


@torch.no_grad()
def evaluate(model, loader, move_rank, device):
    """The same next_move metric train_policy.py uses, so numbers are comparable."""
    model.eval()
    n, loss_sum = 0, 0.0
    hits = {1: 0.0, 5: 0.0, 10: 0.0}
    base_hits = {1: 0.0, 5: 0.0, 10: 0.0}
    by_phase = {}
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        cl = out["policy_logits"]
        tc = batch["target_champ"]
        truth = cl.gather(1, tc[:, None]).squeeze(1)
        ranks = (cl > truth[:, None]).sum(1)
        base = [move_rank[m][c] for m, c in
                zip(batch["move_idx"].tolist(), tc.tolist())]
        cnt = tc.numel()
        loss_sum += float(policy_loss(out, batch)) * cnt
        for k in hits:
            hits[k] += float((ranks < k).sum())
            base_hits[k] += sum(1 for r in base if r < k)
        for r, ph, b in zip(ranks.tolist(), batch["phase"].tolist(), base):
            e = by_phase.setdefault(PHASE_NAMES.get(int(ph), str(ph)),
                                    {"n": 0, "h1": 0, "h5": 0, "b5": 0})
            e["n"] += 1
            e["h1"] += int(r < 1)
            e["h5"] += int(r < 5)
            e["b5"] += int(b < 5)
        n += cnt
    out_m = {"policy_loss": loss_sum / max(1, n), "n": n}
    for k in hits:
        out_m["top%d" % k] = hits[k] / max(1, n)
        out_m["base_top%d" % k] = base_hits[k] / max(1, n)
    out_m["rel_top5"] = EP.baseline_relative(out_m["top5"],
                                             out_m["base_top5"])
    rows = {}
    for key in sorted(by_phase):
        e = by_phase[key]
        rows[key] = {"n": e["n"], "top1": e["h1"] / e["n"],
                     "top5": e["h5"] / e["n"], "base_top5": e["b5"] / e["n"],
                     "rel_top5": EP.baseline_relative(e["h5"] / e["n"],
                                                      e["b5"] / e["n"])}
    out_m["phase"] = rows
    return out_m


def save_ckpt(path, model, cfg, metrics, step, epoch, patch_vocab, resume=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": cfg, "metrics": metrics,
               "step": step, "epoch": epoch, "patch_vocab": patch_vocab}
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
    if len(splits["train"]) != 39720 or len(splits["val"]) != 4771 \
            or len(splits["test"]) != 2448:
        raise SystemExit("not the canonical nov13 split: %s"
                         % {k: len(v) for k, v in splits.items()})

    frontier = C.corpus_frontier(rows)
    through = C.through_suffix(frontier)
    if args.out_dir:
        C.check_out_dir_name(args.out_dir, through,
                             "pro/model/checkpoints/nextpick-naivetransformer-"
                             + args.tag + "-" + through)
        run_dir = Path(args.out_dir)
    else:
        run_dir = (ROOT / "pro" / "model" / "checkpoints"
                   / ("nextpick-naivetransformer-" + args.tag + "-" + through))
    print("data frontier: %s (%s, patch %s) -> %s"
          % (frontier["series_id"], frontier["date"], frontier["patch"],
             through))

    n_patches = len(patch_vocab) + 1
    kw = NT.model_kwargs_from_args(args, n_patches)
    model = NT.NaiveTransformer(**kw).to(device)
    if args.init_ckpt:
        blob = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        old_cfg = blob.get("config") or {}
        if old_cfg.get("arch") != "naive_transformer" or \
                old_cfg.get("model") != kw:
            raise SystemExit("--init-ckpt architecture does not match this run")
        model.load_state_dict(blob["model"], strict=True)
        print("fine-tuning from %s (val rel_top5 %.3f)"
              % (args.init_ckpt,
                 float(blob.get("metrics", {}).get("rel_top5", float("nan")))))
    print("model: %s params %.2fM (%.0f layers, D=%d, FFN=%d, patch rows %d)"
          % ("naive_transformer", NT.param_count(model) / 1e6,
             args.num_layers, args.embed_dim, args.ffn_dim, n_patches))
    print("slot loss weight %.3f (shared head; 10 empty-slot targets/prefix)"
          % args.slot_loss_weight)

    fearless_roles = PRO_DATA.build_fearless_roles(rows)

    def loader(idxs, mode, shuffle, seed):
        ds = PRO_DATA.ProNextMoveDataset(
            rows, idxs, num_champs=173, patch_index=patch_index,
            prefix_mode=mode, prefix_k=args.prefix_k,
            swap_aug=(args.swap_aug and shuffle), seed=seed,
            fearless_roles=fearless_roles)
        return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                              collate_fn=PRO_DATA.pro_collate, num_workers=0,
                              pin_memory=torch.cuda.is_available())

    tr_ds, tr_loader = loader(splits["train"], "k", True, args.seed)
    _, va_loader = loader(splits["val"], "all", False, 0)
    te_loader = loader(splits["test"], "all", False, 0)[1] \
        if splits["test"] else None
    print("train examples/epoch %d (%d prefixes/game), val %d, test %d"
          % (len(tr_ds), args.prefix_k, len(va_loader.dataset),
             0 if te_loader is None else len(te_loader.dataset)))

    move_rank = EP.move_popularity_ranks(rows, splits["train"], 173)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
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
    run_dir.mkdir(parents=True, exist_ok=True)
    print("checkpoints ->", run_dir)
    cfg = {"arch": "naive_transformer", "model": kw, "n_patches": n_patches,
           "slot_head": "shared", "slot_loss_weight": args.slot_loss_weight,
           "init_ckpt": args.init_ckpt}
    best, global_step, t0 = -1.0, 0, time.time()
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
                loss = policy_loss(model(batch), batch,
                                   args.slot_loss_weight)
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
            if global_step % 100 == 0:
                print("  step %d/%d loss %.4f (%.0fs, %.1f GB)"
                      % (global_step, total_steps, agg / max(1, nb),
                         time.time() - t0,
                         torch.cuda.max_memory_allocated() / 2**30
                         if device.type == "cuda" else 0.0), flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print("  evaluating val (%d states)..." % len(va_loader.dataset),
                  flush=True)
            va = evaluate(model, va_loader, move_rank, device)
            print("ep %d step %d train %.4f | val top1 %.4f top5 %.4f "
                  "base5 %.4f rel %.2fx | ban1 %.3f pick1 %.3f ban2 %.3f "
                  "pick2 %.3f (%.0fs)"
                  % (epoch, global_step, agg / max(1, nb), va["top1"],
                     va["top5"], va["base_top5"], va["rel_top5"],
                     va["phase"].get("ban1", {}).get("rel_top5", float("nan")),
                     va["phase"].get("pick1", {}).get("rel_top5", float("nan")),
                     va["phase"].get("ban2", {}).get("rel_top5", float("nan")),
                     va["phase"].get("pick2", {}).get("rel_top5", float("nan")),
                     time.time() - t0), flush=True)
            if va["rel_top5"] > best:
                best = va["rel_top5"]
                save_ckpt(run_dir / "best.pt", model, cfg,
                          {k: v for k, v in va.items() if k != "phase"}
                          | {"epoch": epoch}, global_step, epoch, patch_vocab)
        save_ckpt(run_dir / "latest.pt", model, cfg, {"epoch": epoch},
                  global_step, epoch, patch_vocab,
                  resume={"opt": opt.state_dict(), "sched": sched.state_dict(),
                          "step": global_step, "epoch": epoch, "best": best})
        if args.max_steps and global_step >= args.max_steps:
            break

    train_end = C.newest_train_series(rows, splits["train"])
    if not (run_dir / "best.pt").exists():
        shutil.copyfile(str(run_dir / "latest.pt"), str(run_dir / "best.pt"))
    report = {"protocol": {
        "corpus": args.corpus, "arch": "naive_transformer",
        "n_patches": n_patches, "split": C.split_description(
            args.val_split, bool(splits["test"]), pinned=False),
        "val_scope": args.val_scope, "val_split": args.val_split,
        "prefix_mode": "k", "prefix_k": args.prefix_k,
        "slot_loss_weight": args.slot_loss_weight,
        "slot_head": "shared", "init_ckpt": args.init_ckpt,
        "swap_aug": args.swap_aug, "seed": args.seed,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "n_test": len(splits["test"]), "selection": "rel_top5 (val)",
        "params": NT.param_count(model), "model": kw,
        "through": frontier["series_id"], "through_suffix": through,
        "through_date": frontier["date"], "through_patch": frontier["patch"],
        "train_through": train_end["series_id"],
        "train_through_date": train_end["date"]}}
    blob = torch.load(str(run_dir / "best.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(blob["model"])
    report["val"] = blob["metrics"]
    if te_loader is not None:
        slices = C.SliceAccumulator()
        te = evaluate(model, te_loader, move_rank, device)
        report["same_patch_reference"] = EP.self_move_topk(
            rows, splits["test"], 173, k=(1, 5))
        report["test"] = te
        print("TEST top1 %.4f top5 %.4f (base %.4f, rel %.2fx; same-patch "
              "ref %.4f)"
              % (te["top1"], te["top5"], te["base_top5"], te["rel_top5"],
                 report["same_patch_reference"]["top5"]), flush=True)
    with open(run_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("done (%s)" % run_dir, flush=True)
    return report


if __name__ == "__main__":
    train(parse_args())
