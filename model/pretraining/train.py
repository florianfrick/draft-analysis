"""SoloQ masked-draft pretraining loop.

Usage:
    python pretraining/train.py --epochs 30                 # full run
    python pretraining/train.py --limit-drafts 4000 \\
        --epochs 2 --no-wandb                               # smoke test

Model (increased size, per 8GB plan): embed_dim=256, 4 intra-team
layers, 2 cross-team layers, 8 heads, dropout=0.0 -> ~6M params.
B=1024 + AMP fits comfortably in 8GB (RTX 4060).

Data: streams ../data/solo queue/process matchdata shards via
process matchdata/pipeline.py. First full run scans shards (~1 min) and
caches kept drafts to drafts_<patch>.pt; later runs load the cache
(--rescan forces a rebuild after new matches are collected).

Eval (as agreed): top-1/5/10 champ acc on masked slots, role-conditional
champ top-5 (role given - the 10% masks), hidden-role role acc, role|champ
acc on role-only slots (champ visible, role hidden), all vs popularity
baselines (per-role and global). Hard CE both heads:
loss = CE_champ(is_masked) + lambda_role * CE_role(role labels).

W&B: key read from pretraining/wandb_key.txt or WANDB_API_KEY env var;
--no-wandb disables. Checkpoints: pretraining/ckpt/{best,latest}.pt.
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

HERE = Path(__file__).resolve().parent           # .../model/pretraining
ROOT = HERE.parent.parent                        # .../lol draft analysis
PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"
sys.path.insert(0, str(PROC_DIR))
sys.path.insert(0, str(HERE.parent))             # for soloq_model

import pipeline as P                     # noqa: E402
from soloq_model import SoloQPretrainedModel  # noqa: E402

CKPT_ROOT = HERE / "ckpt"
KEY_FILE = HERE / "wandb_key.txt"


def make_run_dir(run_name=""):
    """Unique per-run checkpoint folder: ckpt/run_<stamp>[_<name>]/."""
    parts = ["run_" + time.strftime("%Y%m%d_%H%M%S")]
    if run_name.strip():
        safe = "".join(c if (c.isalnum() or c in "-_") else "-"
                       for c in run_name.strip())
        parts.append(safe)
    d = CKPT_ROOT / "_".join(parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SoloQ masked-draft pretraining")
    # data
    ap.add_argument("--patch", default="16.20")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--limit-drafts", type=int, default=0,
                    help="cap kept drafts (smoke); skips cache write")
    ap.add_argument("--rescan", action="store_true",
                    help="ignore drafts cache, re-scan shards")
    ap.add_argument("--data-seed", type=int, default=0)
    # optimisation
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-role", type=float, default=0.3)
    ap.add_argument("--role-only-prob", type=float, default=0.10,
                    help="P(hide role | champ visible) per slot; "
                    "teaches P(role|champ)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N optimizer steps (0 = no cap)")
    # model (increased size)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-intra", type=int, default=4)
    ap.add_argument("--num-cross", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    # bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate every N epochs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lol-draft-pretrain")
    ap.add_argument("--run-name", default="")
    return ap.parse_args(argv)

def load_mapping(patch):
    """Riot id -> contiguous idx 0..C-1 (cache on disk, rebuild if missing)."""
    try:
        riot_to_idx, num_champs, cached_patch = P.load_champ_mapping()
        if cached_patch == patch and num_champs:
            return riot_to_idx, num_champs
    except (FileNotFoundError, ValueError):
        pass
    print("building champion mapping (one shard scan)...")
    riot_to_idx = P.build_champ_mapping(P.list_shards(), patch)
    return riot_to_idx, len(riot_to_idx)


def load_rows(args):
    """Kept draft rows: cached .pt after first full scan, or limited scan."""
    cache = P.BASE_DIR / ("drafts_%s.pt" % args.patch)
    if args.limit_drafts:
        # quick path for smoke runs: scan only enough records, no cache
        mapping, _ = load_mapping(args.patch)
        rows, kept, scanned = [], 0, 0
        for _, _, rec in P.iter_records(P.list_shards()):
            scanned += 1
            out = P.record_to_slots(rec, mapping, args.patch)
            if out is not None:
                rows.append(out)
                kept += 1
                if kept >= args.limit_drafts:
                    break
            if scanned >= args.limit_drafts * 4:
                break
        print("limited scan: kept %d (scanned %d)" % (kept, scanned))
        return rows, mapping
    if cache.exists() and not args.rescan:
        payload = torch.load(str(cache), map_location="cpu")
        if payload.get("patch") == args.patch:
            mapping, _ = load_mapping(args.patch)
            print("loaded cache: %d drafts (%s)" %
                  (len(payload["rows"]), cache.name))
            return payload["rows"], mapping
    print("full shard scan (first run ~1 min, then cached)...")
    mapping, _ = load_mapping(args.patch)
    indexed = P.scan_and_index(P.list_shards(), mapping, args.patch)
    torch.save({"patch": args.patch, "rows": indexed["rows"],
                "kept": indexed["kept"], "skipped": indexed["skipped"]},
               str(cache))
    print("kept %d / skipped %d -> cached to %s" %
          (indexed["kept"], indexed["skipped"], cache.name))
    return indexed["rows"], mapping


def split_rows(rows, val_frac, seed):
    """Deterministic draft-level train/val split."""
    n = len(rows)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_frac))
    val_idx = set(perm[:n_val])
    train = [rows[i] for i in perm if i not in val_idx]
    val = [rows[i] for i in perm if i in val_idx]
    return train, val


def make_loader(rows, num_champs, batch_size, shuffle, mask_prob, seed,
                epoch=0, num_workers=0, role_only_prob=0.10):
    ds = P.SoloQDraftDataset(rows, num_champs=num_champs,
                             mask_prob=mask_prob, role_only_prob=role_only_prob,
                             seed=seed, epoch=epoch)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=P.soloq_collate, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def build_rank_tensors(base, num_champs):
    """Champ rank in popularity order: rank[r, c] / grank[c], 0 = most pick."""
    grank = torch.empty(num_champs, dtype=torch.long)
    order = base["global_top"]
    grank[torch.tensor(order)] = torch.arange(num_champs)
    rank = torch.empty(5, num_champs, dtype=torch.long)
    for r in range(5):
        o = base["role_top"][r]
        rank[r, torch.tensor(o)] = torch.arange(num_champs)
    return rank, grank


def masked_loss(out, batch, lambda_role):
    """Hard CE: champ head on champ-masked slots, role head on ALL role-label
    slots (champ-masked + role-only where champ stays visible).

    The role-only slots are what teach P(role | champ) - "x champion is most
    likely y role". Before they existed the role head only ever trained with
    the champ hidden, so champion->role affinities got zero supervision.
    """
    m = batch["is_masked"]
    loss_c = F.cross_entropy(out["champ_logits"][m],
                             batch["champ_labels"][m])
    rm = batch["role_labels"] != -100
    loss_r = F.cross_entropy(out["role_logits"][rm],
                             batch["role_labels"][rm])
    return loss_c + lambda_role * loss_r, loss_c, loss_r


def _hits(ranks, ks):
    return {k: float((ranks < k).float().mean()) for k in ks}


@torch.no_grad()
def evaluate(model, loader, base, lambda_role, device):
    """Val metrics: top-k champ acc (masked), role-cond top-5, hidden-role
    role acc, popularity baselines on the same masked slots."""
    model.eval()
    role_rank, grank = build_rank_tensors(
        base, model.champ_head.out_features)
    role_rank = role_rank.to(device)
    grank = grank.to(device)
    agg, n = {}, 0
    tot = tot_c = tot_r = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss, lc, lr = masked_loss(out, batch, lambda_role)
        m = batch["is_masked"]
        cnt = int(m.sum())
        if cnt == 0:
            continue
        cl, ct = out["champ_logits"][m], batch["champ_labels"][m]
        truth = cl.gather(1, ct[:, None]).squeeze(1)
        cranks = (cl > truth[:, None]).sum(1)  # rank of truth, 0 = top-1
        for k, v in _hits(cranks, (1, 5, 10)).items():
            agg["champ_top%d" % k] = agg.get("champ_top%d" % k, 0.0) + v * cnt
        # role-conditional: champ masks where the true role is given (10%)
        vis = ~batch["role_masked"][m]
        if bool(vis.any()):
            v5 = float((cranks[vis] < 5).float().mean())
            agg["role_cond_top5"] = agg.get("role_cond_top5", 0.0) + v5 * int(vis.sum())
            agg["_role_cond_n"] = agg.get("_role_cond_n", 0) + int(vis.sum())
        # role prediction split by whether the champ was visible:
        #  - champ-hidden (is_masked & role_masked): P(role | context)
        #  - champ-visible role-only (~is_masked & role_masked): P(role | champ)
        #    = the "x champion is most likely y role" metric.
        rl_all, rt_all = out["role_logits"], batch["role_labels"]
        rmask = rt_all != -100
        hid = batch["role_masked"][m]  # champ-masked subset (legacy key)
        rl, rt = out["role_logits"][m], batch["role_labels"][m]
        if bool(hid.any()):
            r1 = float((rl.argmax(1)[hid] == rt[hid]).float().mean())
            agg["role_hidden_top1"] = agg.get("role_hidden_top1", 0.0) + r1 * int(hid.sum())
            agg["_role_hidden_n"] = agg.get("_role_hidden_n", 0) + int(hid.sum())
        ro = batch["role_masked"] & (~m) & rmask  # role-only slots
        if bool(ro.any()):
            r1o = float((rl_all[ro].argmax(1) == rt_all[ro]).float().mean())
            agg["role_given_champ_top1"] = agg.get("role_given_champ_top1", 0.0) + r1o * int(ro.sum())
            agg["_role_given_champ_n"] = agg.get("_role_given_champ_n", 0) + int(ro.sum())
        # popularity baselines on identical slots (rank of truth in order)
        rr = role_rank[rt, ct]
        gr = grank[ct]
        for k in (1, 5):
            agg["base_role_top%d" % k] = agg.get("base_role_top%d" % k, 0.0) + float((rr < k).float().mean()) * cnt
            agg["base_global_top%d" % k] = agg.get("base_global_top%d" % k, 0.0) + float((gr < k).float().mean()) * cnt
        agg["loss"] = agg.get("loss", 0.0) + float(loss) * cnt
        agg["loss_champ"] = agg.get("loss_champ", 0.0) + float(lc) * cnt
        agg["loss_role"] = agg.get("loss_role", 0.0) + float(lr) * cnt
        n += cnt
    out_m = {k: v / n for k, v in agg.items() if not k.startswith("_")}
    if agg.get("_role_cond_n"):
        out_m["role_cond_top5"] = agg["role_cond_top5"] / agg["_role_cond_n"]
    if agg.get("_role_hidden_n"):
        out_m["role_hidden_top1"] = agg["role_hidden_top1"] / agg["_role_hidden_n"]
    if agg.get("_role_given_champ_n"):
        out_m["role_given_champ_top1"] = agg["role_given_champ_top1"] / agg["_role_given_champ_n"]
    return out_m


def setup_wandb(args):
    """Init W&B (key from key file or env). Returns run or None."""
    if args.no_wandb:
        return None
    try:
        import wandb
        if "WANDB_API_KEY" not in os.environ and KEY_FILE.exists():
            os.environ["WANDB_API_KEY"] = KEY_FILE.read_text().strip()
        run = wandb.init(project=args.wandb_project,
                         name=args.run_name or None,
                         config=vars(args), dir=str(HERE),
                         settings=wandb.Settings(init_timeout=60))
        return run
    except Exception as exc:  # noqa: BLE001 - logging must never kill training
        print("wandb disabled: %s" % exc)
        return None


def wb_log(run, metrics, step):
    if run is not None:
        import wandb
        wandb.log(metrics, step=step)


def make_lr_lambda(args, total_steps):
    """Linear warmup then cosine decay to 5% of peak."""
    warm = max(1, int(total_steps * args.warmup_frac))

    def fn(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))
    return fn


def save_ckpt(path, model, args, num_champs, metrics, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"num_champs": num_champs,
                           "embed_dim": args.embed_dim,
                           "num_intra": args.num_intra,
                           "num_cross": args.num_cross,
                           "num_heads": args.num_heads,
                           "dropout": args.dropout},
                "metrics": metrics, "step": step,
                "patch": args.patch}, str(path))


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"

    rows, riot_to_idx = load_rows(args)
    num_champs = len(riot_to_idx)
    if num_champs != 173:
        print("warning: %d champs on patch %s (contract example: 173)"
              % (num_champs, args.patch))
    train_rows, val_rows = split_rows(rows, args.val_frac, args.data_seed)
    print("train %d / val %d drafts, %d champs" %
          (len(train_rows), len(val_rows), num_champs))

    base = P.compute_popularity_baseline(train_rows, num_champs)
    tr_loader = make_loader(train_rows, num_champs, args.batch_size,
                            shuffle=True, mask_prob=0.15,
                            seed=args.seed,
                            role_only_prob=args.role_only_prob)
    va_loader = make_loader(val_rows, num_champs, args.batch_size,
                            shuffle=False, mask_prob=0.15,
                            seed=args.seed + 1,  # fixed val masks
                            role_only_prob=args.role_only_prob)

    model = SoloQPretrainedModel(
        num_champs=num_champs, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_intra_layers=args.num_intra,
        num_cross_layers=args.num_cross, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("model params: %.1fM, device: %s, amp: %s"
          % (n_params / 1e6, device, amp))

    steps_per_epoch = math.ceil(len(tr_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(args, total_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    run = setup_wandb(args)
    run_dir = make_run_dir(args.run_name)
    args.run_dir = str(run_dir)
    print("checkpoints ->", run_dir)
    best_top5, global_step, t0 = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loader.dataset.set_epoch(epoch)
        model.train()
        for i, batch in enumerate(tr_loader):
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(batch)
                loss, lc, lr = masked_loss(out, batch, args.lambda_role)
                scaler.scale(loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1
                if global_step % args.log_every == 0:
                    m = {"train/loss": float(loss), "train/loss_champ": float(lc),
                         "train/loss_role": float(lr),
                         "train/lr": sched.get_last_lr()[0]}
                    wb_log(run, m, global_step)
                    print("ep %d step %d loss %.4f (c %.4f r %.4f) lr %.2e"
                          % (epoch, global_step, float(loss), float(lc),
                             float(lr), sched.get_last_lr()[0]))
                if args.max_steps and global_step >= args.max_steps:
                    break
        metrics = evaluate(model, va_loader, base, args.lambda_role, device)
        metrics["epoch"] = epoch
        wb_log(run, {"val/" + k: v for k, v in metrics.items()}, global_step)
        print("val ep %d: loss %.4f top1 %.3f top5 %.3f top10 %.3f "
              "rolecond@5 %.3f rolehid@1 %.3f role|champ@1 %.3f | base role@1 %.3f "
              "global@1 %.3f (%.0fs)"
              % (epoch, metrics["loss"], metrics["champ_top1"],
                 metrics["champ_top5"], metrics["champ_top10"],
                 metrics.get("role_cond_top5", float("nan")),
                 metrics.get("role_hidden_top1", float("nan")),
                 metrics.get("role_given_champ_top1", float("nan")),
                 metrics["base_role_top1"], metrics["base_global_top1"],
                 time.time() - t0))
        if metrics["champ_top5"] > best_top5:
            best_top5 = metrics["champ_top5"]
            save_ckpt(run_dir / "best.pt", model, args, num_champs,
                      metrics, global_step)
        save_ckpt(run_dir / "latest.pt", model, args, num_champs,
                  metrics, global_step)
        if args.max_steps and global_step >= args.max_steps:
            break
    print("done: best val champ top-5 %.4f (%s)" % (best_top5,
                                                     run_dir / "best.pt"))
    if run is not None:
        import wandb
        wandb.finish()
    return best_top5


def main():
    args = parse_args()
    # fast contract sanity check before touching data (synthetic batch)
    try:
        sys.path.insert(0, str(HERE.parent / "tests"))
        from test_soloq_contract import (make_synthetic_batch,
                                         validate_batch_dict)
        validate_batch_dict(make_synthetic_batch())
    except Exception as exc:  # noqa: BLE001 - warn, do not block training
        print("contract sanity check skipped: %s" % exc)
    best = train(args)
    out = HERE / "last_run.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"best_val_champ_top5": best, "args": vars(args)},
                  f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

