"""Masked-completion stage on the seeded split (the encoder every arm starts from).

Same objective as `solo_queue/model/finetune/train.py` - champ CE on masked slots + 0.3
role CE - but on the 80% seeded split, so nothing in the prototype pipeline has
seen the new test rows. The production encoder (`masking-proonly`) was trained
on a within-patch split that contains 90% of the new test drafts, so it cannot
be used as an init here without leaking.

Selection is leave-one-out top-5 on a capped slice of val (one hidden slot per
draft, role visible), which is the protocol `solo_queue/model/finetune/evaluate.py` uses
and therefore comparable to it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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

from solo_queue.model.pretraining.base_model import SiameseBase       # noqa: E402
from solo_queue.model.pretraining.base_model import attr_matrix_for_config     # noqa: E402
from solo_queue.model.pretraining.base_model import load_encoder_state         # noqa: E402
from pro.model.proto import split as SPLIT                               # noqa: E402

IGNORE = -100
N_SLOTS = 10


class LeaveOneOut(Dataset):
    """One example per (draft, slot): that slot's champion is hidden."""

    def __init__(self, rows, indices, patch_index):
        self.rows = [rows[i] for i in indices]
        self.patch_index = patch_index

    def __len__(self):
        return len(self.rows) * N_SLOTS

    def __getitem__(self, k):
        row = self.rows[k // N_SLOTS]
        slot = k % N_SLOTS
        champ = torch.as_tensor(row["champ"], dtype=torch.long)
        role = torch.as_tensor(row["role"], dtype=torch.long)
        cids = champ.clone()
        cids[slot] = PRO_DATA.CHAMP_MASK_ID
        is_m = torch.zeros(N_SLOTS, dtype=torch.bool)
        is_m[slot] = True
        clab = torch.full((N_SLOTS,), IGNORE, dtype=torch.long)
        clab[slot] = champ[slot]
        return {
            "champ_ids": cids, "role_ids": role.clone(),
            "role_masked": torch.zeros(N_SLOTS, dtype=torch.bool),
            "is_masked": is_m, "champ_labels": clab,
            "role_labels": torch.full((N_SLOTS,), IGNORE, dtype=torch.long),
            "pad_mask": torch.zeros(N_SLOTS, dtype=torch.bool),
            "patch_id": torch.tensor(
                PRO_DATA.patch_to_id(row.get("patch"), self.patch_index),
                dtype=torch.long),
            "weight": torch.tensor(float(row.get("weight", 1.0))),
            "meta": {"gameid": row["gameid"], "patch": row["patch"]},
        }


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="masked-pro")
    ap.add_argument("--init", default="", help="soloQ encoder ckpt (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--lambda-role", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--val-drafts", type=int, default=300)
    ap.add_argument("--out-dir", default="pro/model/proto/runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    return ap.parse_args(argv)


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

    tr_ds = PRO_DATA.ProMaskedDraftDataset(
        rows, splits["train"], num_champs=PRO_DATA.NUM_REAL_CHAMPS,
        mask_prob=args.mask_prob, role_keep_prob=0.1, seed=args.seed,
        patch_index=patch_index)
    val_idx = splits["val"]
    if args.val_drafts and args.val_drafts < len(val_idx):
        step = len(val_idx) / args.val_drafts
        val_idx = [val_idx[int(i * step)] for i in range(args.val_drafts)]
    va_ds = LeaveOneOut(rows, val_idx, patch_index)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=PRO_DATA.pro_collate, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=512, shuffle=False,
                           collate_fn=PRO_DATA.pro_collate)

    if args.init:
        blob = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        enc = SiameseBase(
            num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
            num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
            attr_matrix=attr_matrix_for_config(cfg), n_patches=n_patches)
        load_encoder_state(enc, blob["model"], allow_missing_patch=True)
        init_note = args.init
    else:
        cfg = {"num_champs": PRO_DATA.NUM_REAL_CHAMPS, "embed_dim": 256,
               "num_heads": 8, "num_intra": 2, "num_cross": 1, "dropout": 0.1,
               "attr_dim": 0}
        enc = SiameseBase(
            num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
            num_cross_layers=cfg["num_cross"], dropout=cfg["dropout"],
            n_patches=n_patches)
        init_note = "scratch"
    model = enc.to(device)
    print("arm=%s init=%s train=%d val=%d LOO=%d params=%d"
          % (args.name, init_note, len(tr_ds), len(val_idx), len(va_ds),
             sum(p.numel() for p in model.parameters())))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = len(tr_loader) * args.epochs
    warm = max(1, int(steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: ((s + 1) / warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    Path(args.out_dir, args.name).mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        hit = {1: 0, 5: 0, 10: 0}
        n = 0
        for batch in va_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(batch, compute_heads=True)
            m = batch["is_masked"]
            cl, ct = out["champ_logits"][m], batch["champ_labels"][m]
            r = (cl > cl.gather(1, ct[:, None])).sum(1)
            for k in hit:
                hit[k] += int((r < k).sum())
            n += ct.numel()
        model.train()
        return {("top%d" % k): hit[k] / n for k in hit} | {"n": n}

    best = (-1.0, -1)
    hist = []
    for epoch in range(1, args.epochs + 1):
        tr_ds.set_epoch(epoch)
        model.train()
        run, nb = 0.0, 0
        for batch in tr_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(batch, compute_heads=True)
                m = batch["is_masked"]
                lc = F.cross_entropy(out["champ_logits"][m],
                                     batch["champ_labels"][m])
                rm = batch["role_labels"] != IGNORE
                lr_ = F.cross_entropy(out["role_logits"][rm],
                                      batch["role_labels"][rm])
                loss = lc + args.lambda_role * lr_
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.detach())
            nb += 1
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        m = evaluate()
        hist.append({"epoch": epoch, **m})
        flag = ""
        if m["top5"] > best[0]:
            best = (m["top5"], epoch)
            flag = " *"
            torch.save({"model": model.state_dict(),
                        "config": {**cfg, "n_patches": n_patches,
                                   "with_value": False},
                        "args": vars(args), "epoch": epoch, "metrics": m},
                       Path(args.out_dir) / args.name / "best.pt")
        print("ep %2d loss %.4f | LOO top1 %.4f top5 %.4f top10 %.4f%s"
              % (epoch, run / max(1, nb), m["top1"], m["top5"], m["top10"],
                 flag))
    (Path(args.out_dir) / args.name / "history.json").write_text(
        json.dumps({"args": vars(args), "best": best, "history": hist},
                   indent=1), "utf-8")
    print("best LOO top5 %.4f at epoch %d (%.0fs)"
          % (best[0], best[1], time.time() - t0))
    return best


if __name__ == "__main__":
    main()
