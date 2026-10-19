"""Pro finetune Stage 1: masked draft heads + win/econ outcome heads.

Usage:
    python finetune/train.py --epochs0 3 --epochs1 6            # full run
    python finetune/train.py --limit-drafts 2000 --epochs0 1 \
        --epochs1 1 --no-wandb                                  # smoke

Init: encoder weights from a SoloQ-pretrained checkpoint (default: the
4.8M run1 best.pt - recommended trunk per the vague-totem-4 review;
--init-ckpt PATH or 'auto' for best val top5). Heads train from scratch.

Loss per batch (weighted by recency `weight`):
  total = masked_CE(champ + 0.3*role) + win_w*BCE(win) + econ_w*MSE(econ)
  + rehearse_frac-sampled SoloQ masked CE (encoder rehearsal, no win/econ).

Stage schedule: heads-only for epochs0, then +cross_team for epochs1,
then +intra encoder for epochs2. Embeddings stay frozen in this stage.
Best checkpoints: best_mask.pt (val masked top5) and best_win.pt (val
win AUC on full drafts) - saved separately since the two heads can peak
apart. Platt calibration (fit on val) + econ standardization stats ship
inside each checkpoint for the search stage.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent           # .../model/finetune
MODEL_DIR = HERE.parent                          # .../model
ROOT = MODEL_DIR.parent                          # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"
SOLOQ_PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"
PRETRAIN_CKPT_GLOB = str(MODEL_DIR / "pretraining" / "ckpt" /
                         "run_*" / "best.pt")
DEFAULT_INIT = str(MODEL_DIR / "pretraining" / "ckpt" /
                   "run_20261015_144100_run1-full-16-20" / "best.pt")
KEY_FILE = MODEL_DIR / "pretraining" / "wandb_key.txt"
IGNORE_INDEX = -100


def _load_module_by_path(name, path):
    spec = __import__("importlib").util.spec_from_file_location(
        name, str(path))
    mod = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(MODEL_DIR))
from soloq_model import SoloQPretrainedModel  # noqa: E402
from model import ProDraftModel  # noqa: E402

PRO_PIPE = _load_module_by_path("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load_module_by_path("pro_datasets", PRO_DIR / "datasets.py")
SQ_PIPE = _load_module_by_path("soloq_pipeline",
                               SOLOQ_PROC_DIR / "pipeline.py")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pro finetune Stage 1")
    # data
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--rehearse-frac", type=float, default=0.15,
                    help="share of batches drawn from soloQ rehearsal")
    ap.add_argument("--econ-mask-prob", type=float, default=0.5,
                    help="share of pro train examples as random subsets")
    # model/init
    ap.add_argument("--init-ckpt", default=DEFAULT_INIT,
                    help="soloQ checkpoint path or 'auto' (=best val top5)")
    ap.add_argument("--head-dropout", type=float, default=0.0)
    # losses
    ap.add_argument("--lambda-role", type=float, default=0.3)
    ap.add_argument("--win-weight", type=float, default=1.0)
    ap.add_argument("--econ-weight", type=float, default=0.3)
    # optimisation
    ap.add_argument("--epochs0", type=int, default=3,
                    help="heads-only epochs")
    ap.add_argument("--epochs1", type=int, default=6,
                    help="+cross_team epochs")
    ap.add_argument("--epochs2", type=int, default=6,
                    help="+intra encoder epochs")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lol-draft-finetune")
    ap.add_argument("--run-name", default="")
    return ap.parse_args(argv)


def pick_init_ckpt(spec):
    """Resolve --init-ckpt: explicit path, or 'auto' = best val top5."""
    if spec != "auto":
        payload = torch.load(str(spec), map_location="cpu",
                             weights_only=False)
        return str(spec), payload
    best, best_score = None, -1.0
    for path in sorted(glob.glob(PRETRAIN_CKPT_GLOB)):
        try:
            payload = torch.load(path, map_location="cpu",
                                 weights_only=False)
        except Exception:  # noqa: BLE001 - skip unreadable
            continue
        score = float(payload.get("metrics", {}).get("champ_top5", -1.0))
        if score > best_score:
            best, best_score = (path, payload), score
    if best is None:
        raise FileNotFoundError("no checkpoint under %s"
                                % PRETRAIN_CKPT_GLOB)
    return best


def build_encoder(payload):
    cfg = payload["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0))
    enc.load_state_dict(payload["model"])
    return enc, cfg


def load_pro_rows(args):
    rows, stats = PRO_PIPE.build_rows(limit=args.limit_drafts)
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_series_by_date(rows)
    return rows, splits


def load_soloq_rows():
    """Patch 16.20 soloQ rows for rehearsal (plain [champ],[role] lists)."""
    payload = torch.load(str(SOLOQ_PROC_DIR / "drafts_16.20.pt"),
                         map_location="cpu", weights_only=False)
    return payload["rows"], payload["patch"]


def fit_econ_stats(rows, indices, n_econ=6):
    """Mean/std of raw econ targets on train rows with has_econ (present
    values only). Returns (mean[6], std[6]) tensors for standardization."""
    acc, cnt = torch.zeros(n_econ), 0
    acc2 = torch.zeros(n_econ)
    for i in indices:
        row = rows[i]
        if not row.get("has_econ"):
            continue
        v = torch.tensor(PRO_DATA.econ_targets(row))
        acc += v
        acc2 += v * v
        cnt += 1
    cnt = max(1, cnt)
    mean = acc / cnt
    var = (acc2 / cnt - mean * mean).clamp_min(1e-6)
    return mean, var.sqrt()


def masked_losses(out, batch, lambda_role):
    """Guarded masked CE: batches without masks contribute 0 (no NaN).

    Champ head on is_masked; role head on ALL role-labelled slots
    (champ-masked + role-only) to match pretraining.
    """
    if "is_masked" not in batch:
        dev = batch["champ_ids"].device
        zero = torch.zeros((), device=dev)
        return zero, zero, zero
    m = batch["is_masked"]
    if not bool(m.any()):
        dev = batch["champ_ids"].device
        zero = torch.zeros((), device=dev)
        return zero, zero, zero
    lc = F.cross_entropy(out["champ_logits"][m], batch["champ_labels"][m])
    rm = batch["role_labels"] != IGNORE_INDEX
    lr = F.cross_entropy(out["role_logits"][rm], batch["role_labels"][rm])
    return lc + lambda_role * lr, lc, lr


def win_loss(out, batch):
    """BCE on blue-win logits (class 1 of win_logits), no NaN on empties."""
    if batch["win_label"].numel() == 0:
        return out["win_logits"].sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        out["win_logits"][:, 1], batch["win_label"].float())


def econ_loss(out, batch, econ_mean, econ_std):
    """MSE on standardized econ, masked to has_econ rows."""
    if int(batch["has_econ"].sum()) == 0:
        return out["econ_pred"].sum() * 0.0
    tgt = (batch["econ_target"] - econ_mean) / econ_std
    err = (out["econ_pred"] - tgt) ** 2
    w = batch["has_econ"].unsqueeze(1)
    return (err * w).sum() / (w.sum() * err.shape[1])


def batch_auc(logits, labels):
    """P(blue-win score) AUC via rank statistics; 0.5 on degenerate.

    NaN logits are dropped defensively (a poisoned model must degrade the
    metric, not corrupt it).
    """
    finite = ~torch.isnan(logits)
    logits, labels = logits[finite], labels[finite]
    pos = logits[labels == 1]
    neg = logits[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    wins = (pos[:, None] > neg[None, :]).float().sum()
    ties = (pos[:, None] == neg[None, :]).float().sum()
    return float((wins + 0.5 * ties) / (pos.numel() * neg.numel()))


@torch.no_grad()
def evaluate(model, loaders, base, args, device, econ_mean, econ_std):
    """Val: masked top-k on FULL drafts, win AUC on full+partial drafts,
    econ RMSE (standardized), popularity baselines on masked slots."""
    model.eval()
    out_m, n_mask = {}, 0
    agg_c = agg_r = 0.0
    top = {1: 0.0, 5: 0.0, 10: 0.0}
    cond_hits = cond_n = 0
    role_hits5 = role_hits1 = 0
    for batch in loaders["val_mask"]:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        loss, lc, lr = masked_losses(out, batch, args.lambda_role)
        m = batch["is_masked"]
        cnt = int(m.sum())
        if cnt == 0:
            continue
        agg_c += float(lc) * cnt
        agg_r += float(lr) * cnt
        cl, ct = out["champ_logits"][m], batch["champ_labels"][m]
        truth = cl.gather(1, ct[:, None]).squeeze(1)
        rk = (cl > truth[:, None]).sum(1)
        for k in top:
            top[k] += float((rk < k).sum())
        vis = ~batch["role_masked"][m]
        cond_n += int(vis.sum())
        if bool(vis.any()):
            role_hits5 += int((rk[vis] < 5).sum())
            role_hits1 += int((rk[vis] < 1).sum())
        n_mask += cnt
    out_m["mask_loss"] = (agg_c + args.lambda_role * agg_r) / max(1, n_mask)
    for k in top:
        out_m["champ_top%d" % k] = top[k] / max(1, n_mask)
    out_m["role_cond_top5"] = role_hits5 / max(1, cond_n)
    out_m["role_cond_top1"] = role_hits1 / max(1, cond_n)
    out_m["n_masked"] = n_mask
    # win + econ heads over full AND partial drafts
    logits_all, labels_all, econ_err, econ_n = [], [], 0.0, 0
    for batch in loaders["val_aux"]:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        logits_all.append(out["win_logits"][:, 1].float().cpu())
        labels_all.append(batch["win_label"].cpu())
        if int(batch["has_econ"].sum()):
            tgt = (batch["econ_target"] -
                   econ_mean) / econ_std.to(batch["econ_target"].device)
            err = (out["econ_pred"] - tgt) ** 2
            econ_err += float(err.sum())
            econ_n += int(batch["has_econ"].sum()) * err.shape[1]
    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
    out_m["win_auc"] = batch_auc(logits_all, labels_all)
    out_m["win_brier"] = float(
        ((torch.sigmoid(logits_all) - labels_all.float()) ** 2).mean())
    out_m["win_base_rate"] = float(labels_all.float().mean())
    out_m["econ_rmse"] = (econ_err / max(1, econ_n)) ** 0.5
    out_m["n_win"] = int(labels_all.numel())
    return out_m


def fit_platt(logits, labels):
    """1-D logistic calibration (Platt): returns (a, b) for a*x+b."""
    a = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=100,
                            line_search_fn="strong_wolfe")
    y = labels.float()

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(a * logits + b)
        loss = F.binary_cross_entropy(p, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:  # noqa: BLE001 - keep identity fallback
        return 1.0, 0.0
    return float(a.detach()), float(b.detach())


@torch.no_grad()
def collect_val_logits(model, loader, device):
    model.eval()
    outs, ys = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        outs.append(out["win_logits"][:, 1].float().cpu())
        ys.append(batch["win_label"].cpu())
    return torch.cat(outs), torch.cat(ys)


def run_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb
        if "WANDB_API_KEY" not in os.environ and KEY_FILE.exists():
            os.environ["WANDB_API_KEY"] = KEY_FILE.read_text().strip()
        return wandb.init(project=args.wandb_project,
                          name=args.run_name or None,
                          config=vars(args), dir=str(HERE),
                          settings=wandb.Settings(init_timeout=60))
    except Exception as exc:  # noqa: BLE001 - logging never kills training
        print("wandb disabled: %s" % exc)
        return None


def save_ckpt(path, model, args, cfg, metrics, step, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(),
               "config": {"num_champs": cfg["num_champs"],
                          "embed_dim": cfg["embed_dim"],
                          "num_heads": cfg["num_heads"],
                          "num_intra": cfg["num_intra"],
                          "num_cross": cfg["num_cross"],
                          "dropout": cfg.get("dropout", 0.0),
                          "n_econ": len(PRO_PIPE.ECON_KEYS)},
               "metrics": metrics, "step": step,
               "init_ckpt": extra["init_ckpt"],
               "stage": extra["stage"], "patch": extra.get("patch"),
               "econ_mean": extra["econ_mean"], "econ_std": extra["econ_std"],
               "platt": extra["platt"]}
    torch.save(payload, str(path))


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"

    ckpt_spec, payload = pick_init_ckpt(args.init_ckpt)
    print("init encoder <- %s (val top5 %.4f)" %
          (ckpt_spec, float(payload.get("metrics", {})
                            .get("champ_top5", -1))))
    encoder, cfg = build_encoder(payload)
    num_champs = cfg["num_champs"]
    model = ProDraftModel(encoder, n_econ=len(PRO_PIPE.ECON_KEYS),
                          head_dropout=args.head_dropout).to(device)

    rows, splits = load_pro_rows(args)
    print("pro rows %d (train %d / val %d / test %d)" %
          (len(rows), len(splits["train"]), len(splits["val"]),
           len(splits["test"])))
    econ_mean, econ_std = fit_econ_stats(rows, splits["train"],
                                         len(PRO_PIPE.ECON_KEYS))
    emu, esd = econ_mean.to(device), econ_std.to(device)

    def pro_loader(idxs, mask_prob, epoch, shuffle, econ_p):
        ds = PRO_DATA.ProMaskedDraftDataset(
            rows, idxs, num_champs=num_champs, mask_prob=mask_prob,
            seed=args.seed, epoch=epoch, econ_mask_prob=econ_p)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=PRO_DATA.pro_collate, num_workers=0,
                          pin_memory=torch.cuda.is_available())

    tr_ds = PRO_DATA.ProMaskedDraftDataset(
        rows, splits["train"], num_champs=num_champs, seed=args.seed,
        econ_mask_prob=args.econ_mask_prob)
    wts = [r.get("weight", 1.0) for r in rows]
    if max(wts) - min(wts) > 1e-6:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                               sampler=PRO_DATA.make_weighted_sampler(
                                   tr_ds, seed=args.seed),
                               collate_fn=PRO_DATA.pro_collate,
                               num_workers=0,
                               pin_memory=torch.cuda.is_available())
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                               shuffle=True, collate_fn=PRO_DATA.pro_collate,
                               num_workers=0,
                               pin_memory=torch.cuda.is_available())
    va_mask = pro_loader(splits["val"], 0.15, 0, False, 0.0)
    # aux val/test mix full + random-subset drafts (econ_mask_prob) so the
    # win metrics cover the partial-draft distribution MCTS will query
    va_aux = pro_loader(splits["val"], 0.0, 0, False, 0.5)
    te_mask = pro_loader(splits["test"], 0.15, 0, False, 0.0)
    te_aux = pro_loader(splits["test"], 0.0, 0, False, 0.5)

    # rehearsal: soloQ rows, train stream only (epoch-rolled masks)
    sq_rows, sq_patch = load_soloq_rows()
    print("rehearsal pool: %d soloQ drafts (%s)" % (len(sq_rows), sq_patch))

    def sq_loader(epoch):
        ds = SQ_PIPE.SoloQDraftDataset(sq_rows, num_champs=num_champs,
                                       seed=args.seed + 100, epoch=epoch)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=SQ_PIPE.soloq_collate, num_workers=0,
                          pin_memory=torch.cuda.is_available())

    sq_epoch, sq_it = 0, None
    run = run_wandb(args)
    run_dir = HERE / "ckpt" / ("run_" + time.strftime("%Y%m%d_%H%M%S") +
                               (("_" + args.run_name) if args.run_name else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    print("checkpoints ->", run_dir)

    stages = [(0, args.epochs0, args.lr_head), (1, args.epochs1, None),
              (2, args.epochs2, None)]
    global_step, best_top5, best_auc = 0, -1.0, -1.0
    t0 = time.time()
    # rehearsal share applied per batch with prob rehearse_frac
    rehearse_gen = torch.Generator().manual_seed(args.seed + 999)


    for stage, n_epochs, lr_head in stages:
        if n_epochs <= 0:
            continue
        model.set_stage(stage)
        head_p = [p for p in model.heads.parameters() if p.requires_grad]
        # encoder readout heads (champ/role) adapt with the head LR; the
        # attention trunk only joins at stage >= 1, at the low encoder LR
        readout_p, trunk_p = [], []
        for n_, p in model.encoder.named_parameters():
            if not p.requires_grad:
                continue
            (readout_p if "champ_head" in n_ or "role_head" in n_
             else trunk_p).append(p)
        groups = [{"params": head_p + readout_p,
                   "lr": lr_head or args.lr_head}]
        if trunk_p:
            groups.append({"params": trunk_p, "lr": args.lr_enc})
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(tr_loader))
        total_steps = steps_per_epoch * n_epochs
        warm = max(1, int(total_steps * args.warmup_frac))

        def lr_lambda(step, warm=warm, total_steps=total_steps):
            if step < warm:
                return (step + 1) / warm
            t = min(1.0, (step - warm) / max(1, total_steps - warm))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        for epoch in range(1, n_epochs + 1):
            tr_ds.set_epoch((global_step // max(1, len(tr_loader))) + epoch)
            model.train()
            if sq_it is None:
                sq_it = iter(sq_loader(sq_epoch))
            for i, batch in enumerate(tr_loader):
                batch = {k: v.to(device, non_blocking=True)
                         if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(batch)
                    m_loss, _, _ = masked_losses(
                        out, batch, args.lambda_role)
                    loss = m_loss + args.win_weight * win_loss(out, batch) \
                        + args.econ_weight * econ_loss(out, batch, emu, esd)
                # rehearsal: soloQ masked CE keeps old objective alive.
                # Stage 0 freezes the encoder entirely (head-only), so the
                # rehearsal gradient would be a no-op there; run it only
                # when encoder params actually train (stages 1-2).
                do_rehearse = args.rehearse_frac > 0 and stage >= 1 and \
                    (torch.rand((), generator=rehearse_gen).item() <
                     args.rehearse_frac)
                if do_rehearse:
                    try:
                        rb = next(sq_it)
                    except StopIteration:
                        sq_epoch += 1
                        sq_it = iter(sq_loader(sq_epoch))
                        rb = next(sq_it)
                    rb = {k: v.to(device, non_blocking=True)
                          for k, v in rb.items()}
                    with torch.cuda.amp.autocast(enabled=amp):
                        rout = model.encoder(rb)
                        r_loss, _, _ = masked_losses(
                            rout, rb, args.lambda_role)
                        loss = loss + r_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "non-finite loss at stage %d step %d"
                        % (stage, global_step))
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()  # scheduler after optimizer each step
                global_step += 1
                if global_step % 20 == 0:
                    msg = {"train/loss": float(loss),
                           "train/lr": sched.get_last_lr()[0]}
                    if run is not None:
                        import wandb
                        wandb.log(msg, step=global_step)
                    print("stage %d ep %d step %d loss %.4f lr %.2e" %
                          (stage, epoch, global_step, float(loss),
                           sched.get_last_lr()[0]))
            metrics = evaluate(model,
                               {"val_mask": va_mask, "val_aux": va_aux},
                               None, args, device, emu, esd)
            metrics["epoch"] = epoch
            metrics["stage"] = stage
            if run is not None:
                import wandb
                wandb.log({"val/" + k: v for k, v in metrics.items()},
                          step=global_step)
            print("val stage %d ep %d: mask_top5 %.4f win_auc %.4f "
                  "brier %.4f econ_rmse %.4f (%.0fs)" %
                  (stage, epoch, metrics["champ_top5"], metrics["win_auc"],
                   metrics["win_brier"], metrics["econ_rmse"],
                   time.time() - t0))
            extra = {"init_ckpt": ckpt_spec, "stage": stage,
                     "patch": payload.get("patch"),
                     "econ_mean": econ_mean, "econ_std": econ_std,
                     "platt": (1.0, 0.0)}
            if metrics["champ_top5"] > best_top5:
                best_top5 = metrics["champ_top5"]
                vl, vy = collect_val_logits(model, va_aux, device)
                extra["platt"] = fit_platt(vl, vy)
                save_ckpt(run_dir / "best_mask.pt", model, args, cfg,
                          metrics, global_step, extra)
            if metrics["win_auc"] > best_auc:
                best_auc = metrics["win_auc"]
                vl, vy = collect_val_logits(model, va_aux, device)
                extra["platt"] = fit_platt(vl, vy)
                save_ckpt(run_dir / "best_win.pt", model, args, cfg,
                          metrics, global_step, extra)
            save_ckpt(run_dir / "latest.pt", model, args, cfg,
                      dict(metrics, platt=extra["platt"]), global_step, extra)


    # held-out test report on both best checkpoints
    report = {}
    for name in ("best_mask.pt", "best_win.pt"):
        path = run_dir / name
        if not path.exists():
            continue
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        tm = evaluate(model, {"val_mask": te_mask, "val_aux": te_aux},
                      None, args, device, emu, esd)
        report[name] = tm
        print("TEST %s: mask_top5 %.4f win_auc %.4f brier %.4f "
              "econ_rmse %.4f" % (name, tm["champ_top5"], tm["win_auc"],
                                  tm["win_brier"], tm["econ_rmse"]))
    with open(run_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("done: best mask top5 %.4f, best win AUC %.4f (%s)"
          % (best_top5, best_auc, run_dir))
    if run is not None:
        import wandb
        wandb.finish()
    return best_top5, best_auc


def main():
    args = parse_args()
    best_top5, best_auc = train(args)
    out = HERE / "last_run.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"best_val_champ_top5": best_top5,
                   "best_val_win_auc": best_auc,
                   "args": vars(args)}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

