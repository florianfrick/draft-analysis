"""Compact next-move trainer for the seeded split, with the experiment knobs.

Deliberately separate from `solo_queue/model/policy/train.py`. That script carries the
production contract (within-patch split, rel_top5 selection, wandb, the kept
artefact names) and is 1,000 lines of accumulated measurement. Prototyping
needs the opposite: one loop, one val number, and a flag per idea. Nothing here
writes to `pro/model/checkpoints/`, so a failed arm cannot be mistaken for a kept
one.

Knobs that exist because they target a specific weakness:

  --block-weight    auxiliary CE for the PARTNER of a double-pick block, asked
                    at the first move of the pair. Targets the block-partner
                    metric directly: it forces the model to represent the pair
                    jointly instead of picking one champion and discovering the
                    second from the board.
  --ema             exponential moving average of the weights, evaluated in
                    place of the raw ones. Cheap variance reduction; on a
                    corpus this size the seed spread is 0.4-1.4 pp, so most
                    single-arm gains are inside it without something like this.
  --unfreeze        let the encoder move (stage 1 = +cross_team, 2 = +intra).
                    The production recipe freezes it and trains ~100k of 3.3M
                    parameters; this is the obvious untested lever.
  --label-smooth    soften the CE target.
  --masked-aux      keep the masked-completion objective alive during policy
                    training (the "communication between masking and
                    next-pick" idea). Requires --unfreeze, because the champ
                    head lives inside the frozen encoder.

Selection is on val exact top-5, always. The test split is read once per arm,
after the arm is chosen - otherwise the leaderboard is just overfitting with
extra steps.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRO_DIR = ROOT / "data" / "pro play"
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")
EP = _load("eval_protocol", PRO_DIR / "eval_protocol.py")

from solo_queue.model import PolicyValueModel                                   # noqa: E402
from solo_queue.model.policy.model import CHAMP_PAD_ID, ROLE_FLEX_ID            # noqa: E402
from solo_queue.model.pretraining.base_model import SiameseBase       # noqa: E402
from solo_queue.model.pretraining.base_model import attr_matrix_for_config     # noqa: E402
from solo_queue.model.pretraining.base_model import load_encoder_state         # noqa: E402
from pro.model.proto import split as SPLIT                               # noqa: E402

NUM_CHAMPS = PRO_DATA.NUM_REAL_CHAMPS
IGNORE = -100


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="arm")
    ap.add_argument("--init", default="", help="masked encoder ckpt")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-enc", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--head-dropout", type=float, default=0.0)
    ap.add_argument("--block-weight", type=float, default=0.0)
    ap.add_argument("--slot-mask", action="store_true",
                    help="append the filled-slot and ban-slot indicators to "
                         "the policy input")
    ap.add_argument("--prior-bias", action="store_true",
                    help="add a learned multiple of the train-only empirical "
                         "log P(champion | patch, move index) to the logits")
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="0=off, e.g. 0.999")
    ap.add_argument("--unfreeze", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--masked-aux", type=float, default=0.0,
                    help="weight on a masked-completion CE over visible slots")
    ap.add_argument("--no-swap-aug", dest="swap_aug", action="store_false",
                    default=True)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--val-drafts", type=int, default=1200,
                    help="cap on val drafts for selection (0 = all)")
    ap.add_argument("--out-dir", default="pro/model/proto/runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    return ap.parse_args(argv)


# ------------------------------------------------------------------ the model
# ProtoPolicy lives in pro/model/proto/policy.py so the evaluator rebuilds the
# same class from a checkpoint (a slot_mask arm has a wider first layer).
from pro.model.proto.policy import ProtoPolicy, build_prior  # noqa: E402,F401

def build(rows, args, n_patches, device, patch_index, train_idx):
    if args.scratch:
        cfg = {"num_champs": NUM_CHAMPS, "embed_dim": 256, "num_heads": 8,
               "num_intra": 2, "num_cross": 1, "dropout": 0.1, "attr_dim": 0}
        enc = SiameseBase(
            num_champs=NUM_CHAMPS, embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
            num_cross_layers=cfg["num_cross"], dropout=cfg["dropout"],
            n_patches=n_patches)
    else:
        blob = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        enc = SiameseBase(
            num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
            num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
            attr_matrix=attr_matrix_for_config(cfg), n_patches=n_patches)
        load_encoder_state(enc, blob["model"], allow_missing_patch=True)
    model = ProtoPolicy(enc, block_head=args.block_weight > 0,
                        slot_mask=args.slot_mask,
                        prior_bias=args.prior_bias,
                        embed_dim=cfg["embed_dim"], state_dim=args.state_dim,
                        n_econ=0, dropout=args.head_dropout,
                        n_patches=n_patches, with_value=False).to(device)
    if args.prior_bias:
        model.set_prior(build_prior(rows, train_idx, patch_index,
                                    n_patches, PRO_DATA.N_MOVES,
                                    PRO_DATA.patch_to_id))
    model.set_freeze_encoder(args.unfreeze == 0)
    if args.unfreeze == 1:
        # stage 1: only the inter-team block moves; the embeddings and the
        # intra-team trunk stay at their pretrained values
        for p in model.encoder.embed.parameters():
            p.requires_grad = False
        for p in model.encoder.intra_team.parameters():
            p.requires_grad = False
    return model, cfg


def policy_loss(out, batch, smooth=0.0):
    return F.cross_entropy(out["policy_logits"], batch["target_champ"],
                           label_smoothing=smooth)


def block_loss(out, batch):
    if "block_logits" not in out:
        return out["policy_logits"].sum() * 0.0
    keep = batch["has_partner"].bool()
    if not bool(keep.any()):
        return out["block_logits"].sum() * 0.0
    return F.cross_entropy(out["block_logits"][keep],
                           batch["partner_champ"][keep])


def masked_aux_loss(model, batch):
    """Masked-completion CE on a random subset of VISIBLE slots.

    Same objective the encoder was pretrained on, kept alive while the policy
    head trains, so the trunk cannot drift into a representation that only
    serves next-move. One extra forward; the encoder is already unfrozen in
    the arms that use this.
    """
    cids, pads = batch["champ_ids"], batch["pad_mask"]
    visible = (~pads)
    if not bool(visible.any()):
        return batch["champ_ids"].sum() * 0.0
    b = cids.clone()
    pick = torch.rand(cids.shape, device=cids.device) < 0.25
    mask = pick & visible
    # guarantee at least one masked slot per row
    first = visible.float().argmax(1)
    rows = torch.arange(cids.shape[0], device=cids.device)
    mask[rows, first] = True
    b[mask] = PRO_DATA.CHAMP_MASK_ID
    sub = dict(batch)
    sub["champ_ids"] = b
    sub["is_masked"] = mask
    sub["role_masked"] = mask
    h = model.encoder._encode(sub)
    logits = model.encoder._apply_duplicate_mask(
        model.encoder.champ_head(h), sub)
    return F.cross_entropy(logits[mask], batch["champ_ids"][mask])


@torch.no_grad()
def evaluate(model, loader, device, partner_only=False):
    model.eval()
    hit = {1: 0, 5: 0, 10: 0}
    n = 0
    phit = {1: 0, 5: 0}
    pn = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        cl = model(batch)["policy_logits"]
        tc = batch["target_champ"]
        r = (cl > cl.gather(1, tc[:, None])).sum(1)
        for k in hit:
            hit[k] += int((r < k).sum())
        n += tc.numel()
        keep = batch["has_partner"].bool()
        if bool(keep.any()):
            p = batch["partner_champ"]
            pr = (cl[keep] > cl[keep].gather(1, p[keep, None])).sum(1)
            for k in phit:
                phit[k] += int((pr < k).sum())
            pn += int(keep.sum())
    model.train()
    out = {"n": n, "top1": hit[1] / n, "top5": hit[5] / n, "top10": hit[10] / n}
    if pn:
        out.update({"partner_n": pn, "partner_top1": phit[1] / pn,
                    "partner_top5": phit[5] / pn})
    return out


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def apply_to(self, model):
        backup = copy.deepcopy(model.state_dict())
        model.load_state_dict({**backup, **self.shadow})
        return backup


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()

    rows, _ = PRO_PIPE.build_rows()
    PRO_PIPE.assign_weights(rows, 75.0)
    splits, man = SPLIT.load(rows)
    _, patch_index = PRO_DATA.build_patch_vocab(rows)
    n_patches = len(patch_index) + 1

    tr_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["train"], num_champs=NUM_CHAMPS,
        patch_index=patch_index, prefix_mode="random", swap_aug=args.swap_aug,
        seed=args.seed)
    val_idx = splits["val"]
    if args.val_drafts and args.val_drafts < len(val_idx):
        step = len(val_idx) / args.val_drafts
        val_idx = [val_idx[int(i * step)] for i in range(args.val_drafts)]
    va_ds = PRO_DATA.ProNextMoveDataset(
        rows, val_idx, num_champs=NUM_CHAMPS, patch_index=patch_index,
        prefix_mode="all", swap_aug=False)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=PRO_DATA.pro_collate, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=PRO_DATA.pro_collate)

    model, cfg = build(rows, args, n_patches, device, patch_index,
                       splits["train"])
    params = [p for p in model.parameters() if p.requires_grad]
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in params if id(p) not in enc_ids]
    groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_enc})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = len(tr_loader) * args.epochs
    warm = max(1, int(steps * args.warmup_frac))

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        t = (s - warm) / max(1, steps - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, args.ema) if args.ema else None

    print("arm=%s train=%d val=%d states=%d params=%d enc_trainable=%d"
          % (args.name, len(tr_ds), len(val_idx), len(va_ds),
             sum(p.numel() for p in model.parameters()), len(enc_params)))
    Path(args.out_dir, args.name).mkdir(parents=True, exist_ok=True)
    best = (-1.0, -1)
    hist = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        tr_ds.set_epoch(epoch)
        model.train()
        run = 0.0
        nb = 0
        for batch in tr_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(batch)
                loss = policy_loss(out, batch, args.label_smooth)
                if args.block_weight:
                    loss = loss + args.block_weight * block_loss(out, batch)
                if args.masked_aux:
                    loss = loss + args.masked_aux * masked_aux_loss(model,
                                                                    batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run += float(loss.detach())
            nb += 1
            if ema:
                ema.update(model)
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        if ema:
            backup = ema.apply_to(model)
        m = evaluate(model, va_loader, device)
        if ema:
            model.load_state_dict(backup)
        hist.append({"epoch": epoch, **m})
        flag = ""
        if m["top5"] > best[0]:
            best = (m["top5"], epoch)
            flag = " *"
            torch.save({"model": model.state_dict(),
                        "config": {**cfg, "n_patches": n_patches,
                                   "state_dim": args.state_dim,
                                   "n_econ": 0, "with_value": False,
                                   "block_head": args.block_weight > 0,
                                   "slot_mask": args.slot_mask,
                                   "prior_bias": args.prior_bias},
                        "args": vars(args), "epoch": epoch,
                        "metrics": m}, Path(args.out_dir) / args.name /
                       "best.pt")
        print("ep %2d loss %.4f | val top1 %.4f top5 %.4f partner5 %.4f%s"
              % (epoch, run / max(1, nb), m["top1"], m["top5"],
                 m.get("partner_top5", float("nan")), flag))
    print("best val top5 %.4f at epoch %d (%.0fs)"
          % (best[0], best[1], time.time() - t0))
    Path(args.out_dir, args.name).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / args.name / "history.json").write_text(
        json.dumps({"args": vars(args), "best": best, "history": hist},
                   indent=1), "utf-8")
    return best


if __name__ == "__main__":
    main()
