"""Train the win-prediction model on pro or soloQ drafts (CE or focal loss).

    # the first experiment: pro data, cross-entropy, the baseline's seed
    python model/winpred/train.py --data pro --loss ce --seed 20261105

    # smoke test
    python model/winpred/train.py --data pro --limit-drafts 2000 --epochs 2 \\
        --no-wandb

Selection is on **val log-loss** - the proper scoring rule the model is trained
on, and the only one where a better-calibrated model is unambiguously better.
Accuracy would prefer a model that hedges, AUC would ignore the side prior
entirely (it is the one metric a constant scores 0.5 on, by construction).

At the end the script scores the selected checkpoint on val and test, fits the
blue-win-rate baseline on the same train rows, and writes all three into
`<out-dir>/test_report.json` plus the raw probabilities to
`<out-dir>/test_preds.npz` (which `evaluate.py` can re-score or compare against
anything else).

W&B: on by default, key from `WANDB_API_KEY` (docker-compose loads `.env`);
`--no-wandb` disables, and a logging failure never kills a run.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent           # .../model/winpred
ROOT = HERE.parents[1]                           # .../lol draft analysis
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.winpred import data as D              # noqa: E402
from model.winpred import metrics as M           # noqa: E402
from model.winpred import model as W             # noqa: E402
from model.winpred.baseline import BlueWinRateBaseline  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Win prediction from a draft")
    # data
    ap.add_argument("--data", choices=("soloq", "pro", "both"), default="pro",
                    help="both = pro + soloQ in one corpus (89%% soloQ, "
                         "unweighted); each half keeps its own split rule")
    ap.add_argument("--corpus", choices=sorted(D.CORPORA), default="2022",
                    help="pro only: 2022 = the documented 46,879-draft corpus; "
                         "all = 2017-2026 (81,161)")
    ap.add_argument("--soloq-patch", default="16.20")
    ap.add_argument("--soloq-cache", default="")
    ap.add_argument("--limit-drafts", type=int, default=0,
                    help="smoke runs: evenly-strided subsample before splitting")
    # model
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-intra", type=int, default=2)
    ap.add_argument("--num-cross", type=int, default=1)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="0.2 (was 0.1 in run 1). The pro run overfitted "
                         "*confidence*: by epoch 40 val AUC had moved -0.013 "
                         "while log-loss went 0.688 -> 1.057 and ECE 0.01 -> "
                         "0.26, i.e. capacity is not the constraint. Dropout "
                         "is the cheapest lever on that, and the same value "
                         "is used for every arm of the sweep so the loss and "
                         "data comparisons are not confounded by it.")
    ap.add_argument("--head-dim", type=int, default=0, help="0 = embed_dim")
    ap.add_argument("--pool", choices=("mean", "pma"), default="mean")
    ap.add_argument("--patch-emb", dest="patch_emb", action="store_true",
                    default=True)
    ap.add_argument("--no-patch-emb", dest="patch_emb", action="store_false")
    ap.add_argument("--attr", dest="attr", action="store_true", default=False,
                    help="add the 18-axis kit-attribute adapter (STATUS.md §6 "
                         "measured it as noise; off by default)")
    # loss
    ap.add_argument("--loss", choices=("ce", "focal"), default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.5,
                    help="weight on the blue-win term; 0.5 is symmetric "
                         "(gamma=0 + alpha=0.5 is exactly 0.5 * CE)")
    # optimisation
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=D.SPLIT_SEED,
                    help="one seed for the split AND the init (default: the "
                         "data module's SPLIT_SEED, so the baseline and every "
                         "model see the same rows)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    # bookkeeping
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lol-draft-winpred")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out-dir", default="",
                    help="default: model/checkpoints/winpred-<data>-<loss>-"
                         "seed<seed>[-through-<date>|-p<patch>]")
    return ap.parse_args(argv)


# ------------------------------------------------------------------- losses
def focal_loss(logit, y, gamma=2.0, alpha=0.5, reduction="mean"):
    """Binary focal loss on logits: -a_t (1 - p_t)^gamma * log p_t.

    `alpha` weights the positive (blue-win) term; 0.5 is symmetric, and with
    gamma=0 the result is exactly `alpha * BCE`. Kept as a function rather than
    a class so a test can assert that identity.
    """
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    p_t = y * p + (1.0 - y) * (1.0 - p)
    a_t = y * alpha + (1.0 - y) * (1.0 - alpha)
    loss = a_t * (1.0 - p_t).pow(gamma) * bce
    return loss.mean() if reduction == "mean" else loss


def compute_loss(out, batch, args):
    return focal_loss(out["logit"], batch["win_label"], args.focal_gamma,
                      args.focal_alpha) if args.loss == "focal" \
        else F.binary_cross_entropy_with_logits(out["logit"],
                                                batch["win_label"])


# ------------------------------------------------------------------ helpers
def make_loader(table, idx, batch_size, shuffle, seed, num_workers=0):
    ds = D.WinDataset(table, idx)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=D.collate, num_workers=num_workers,
                      generator=gen if shuffle else None,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device):
    """-> (y [N], p [N], game_ids [N]) in loader order."""
    model.eval()
    ys, ps, ids = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        p = torch.sigmoid(model(batch)["logit"])
        ps.append(p.detach().float().cpu().numpy())
        ys.append(batch["win_label"].detach().float().cpu().numpy())
        ids.extend(m["game_id"] for m in batch["meta"])
    return (np.concatenate(ys), np.concatenate(ps), ids)


def summarise(y, p):
    return M.full_report(y, p, with_bins=False)


def _dom_masks(table, idx):
    """{domain: boolean mask over the split's rows}, in `idx` order."""
    dom = table.domain[np.asarray(idx)]
    return {d: (dom == d) for d in table.domains()}


def summarise_by_domain(table, idx, y, p):
    """Metrics per domain; empty for a single-dataset table.

    `idx` and `y`/`p` are in the same order (the loader preserves it), which is
    what makes the mask line up with the predictions.
    """
    if len(table.domains()) < 2:
        return {}
    return {d: summarise(y[m], p[m])
            for d, m in _dom_masks(table, idx).items() if m.any()}


def domain_deltas(table, idx, y, p, groups, base_p, seed=0, train_idx=None):
    """Per-domain model-vs-baseline deltas; `{}` for a single-dataset table.

    Kept as a function rather than inlined in `train()` because the "no
    domains" case is the common one and used to crash the whole run *after*
    training finished: a single-dataset table has one domain mask but no
    per-domain report, so the loop indexed an empty dict. Losing 14 minutes of
    training to a reporting KeyError is exactly the kind of thing a test should
    catch, and this is the smallest unit that can be tested.

    `train_idx` makes the per-domain reference the domain's **own** base rate.
    Without it the block was compared against the pooled prior, which is wrong
    whenever the domains disagree: on the joint corpus the pooled prior is
    0.4825 while pro's own rate is 0.5332, so the pro block read +0.019
    accuracy for a model that was 0.021 *below* pro's prior. Passing the train
    rows fixes the reference; `None` keeps the pooled behaviour for callers
    that have no split at hand.
    """
    rep = summarise_by_domain(table, idx, y, p)
    out = {}
    dom_train = table.domain[np.asarray(train_idx)] if train_idx is not None \
        else None
    y_train = np.asarray(table.y)[np.asarray(train_idx)] \
        if train_idx is not None else None
    for dom, mask in _dom_masks(table, idx).items() if rep else ():
        if not mask.any():
            continue
        ref, label = base_p[mask], "pooled prior"
        if dom_train is not None:
            tm = dom_train == dom
            if tm.any():
                rate = float(y_train[tm].mean())
                ref, label = np.full(int(mask.sum()), rate), \
                    "domain train rate (%.5f)" % rate
        base_dom = summarise(y[mask], ref)
        out[dom] = {
            "n": int(mask.sum()),
            "reference": label,
            "auc_delta": rep[dom]["roc_auc"] - base_dom["roc_auc"],
            "vs_baseline": {
                m: M.bootstrap_delta(y[mask], ref, p[mask],
                                     [g for g, k in zip(groups, mask) if k],
                                     metric=m, seed=seed)
                for m in ("log_loss", "brier", "accuracy")},
        }
    return out


def setup_wandb(args, cfg):
    if args.no_wandb:
        return None
    try:
        import wandb
        if not hasattr(wandb, "init"):
            print("wandb disabled: 'wandb' resolved to a namespace package")
            return None
        return wandb.init(project=args.wandb_project,
                          name=args.run_name or None, config=cfg,
                          dir=str(ROOT / "wandb_runs"),
                          settings=wandb.Settings(init_timeout=60))
    except Exception as exc:  # noqa: BLE001 - logging never kills training
        print("wandb disabled: %s" % exc)
        return None


def default_out_dir(args, table, meta):
    tag = "winpred-%s-%s-seed%d" % (args.data, args.loss, args.seed)
    if table.source in ("pro", "both"):
        newest = meta.get("newest_row_date") or meta.get("corpus_max_date", "")
        return ROOT / "model" / "checkpoints" / (
            tag + "-through-" + newest[:10].replace("-", ""))
    return ROOT / "model" / "checkpoints" / (
        tag + "-p" + str(meta.get("patch", "")).replace(".", "_"))


def save_ckpt(path, model, cfg, metrics, step, epoch, patch_vocab):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": cfg,
               "metrics": metrics, "step": step, "epoch": epoch,
               "patch_vocab": list(patch_vocab)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path))


# --------------------------------------------------------------------- main
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    device = torch.device(args.device)
    amp = device.type == "cuda"

    # ---- data
    if args.data == "pro":
        table, meta = D.load_pro(corpus=args.corpus)
    elif args.data == "both":
        table, meta = D.load_both(corpus=args.corpus,
                                  patch=args.soloq_patch)
    else:
        table, meta = D.load_soloq(patch=args.soloq_patch,
                                   cache_path=args.soloq_cache or None)
    if args.limit_drafts and args.limit_drafts < len(table):
        stride = max(1, len(table) // args.limit_drafts)
        keep = np.arange(0, len(table), stride)[:args.limit_drafts]
        table = table.subset(keep)
        print("limited run: %d drafts (stride %d)" % (len(table), stride))
    splits, split_rep = D.make_splits(table, seed=args.seed)

    n_patches = (meta["n_patches"] if args.patch_emb else 0)
    patch_vocab = meta["patch_vocab"] if args.patch_emb else []
    attr_matrix = W.attr_matrix_or_none(D.NUM_REAL_CHAMPS, args.attr)

    tr_loader = make_loader(table, splits["train"], args.batch_size, True,
                            args.seed, args.num_workers)
    va_loader = make_loader(table, splits["val"], args.batch_size, False,
                            args.seed, args.num_workers)
    te_loader = (make_loader(table, splits["test"], args.batch_size, False,
                             args.seed, args.num_workers)
                 if len(splits["test"]) else None)

    # ---- model
    model = W.WinPredictor(num_champs=D.NUM_REAL_CHAMPS,
                           embed_dim=args.embed_dim,
                           num_intra=args.num_intra,
                           num_cross=args.num_cross,
                           num_heads=args.num_heads, dropout=args.dropout,
                           n_patches=n_patches, attr_matrix=attr_matrix,
                           pool=args.pool,
                           head_dim=args.head_dim or None).to(device)
    cfg = W.config_of(model, data=args.data, loss=args.loss,
                      focal_gamma=args.focal_gamma,
                      focal_alpha=args.focal_alpha, seed=args.seed,
                      patch_emb=bool(args.patch_emb),
                      attrs=bool(args.attr), pool=args.pool)
    print("model: %.2fM params | patch emb %s (%d rows) | attrs %s | pool %s"
          % (W.param_count(model) / 1e6, "ON" if n_patches else "OFF",
             n_patches, "ON" if attr_matrix is not None else "OFF", args.pool))

    # ---- optimisation
    steps_per_epoch = math.ceil(len(tr_loader))
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    warm = max(1, int(total_steps * args.warmup_frac))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(
        args, table, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("checkpoints -> %s" % out_dir)
    run = setup_wandb(args, dict(cfg, **{
        "lr": args.lr, "batch_size": args.batch_size,
        "weight_decay": args.weight_decay, "epochs": args.epochs,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "n_test": len(splits["test"]), "data": args.data}))

    best = float("inf")
    global_step = 0
    t0 = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg, nb = 0.0, 0
        for batch in tr_loader:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = compute_loss(model(batch), batch, args)
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

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            yv, pv, _ = predict(model, va_loader, device)
            vm = summarise(yv, pv)
            print("ep %2d step %4d train %.4f | val ll %.4f brier %.4f "
                  "auc %.4f acc %.4f ece %.4f (%.0fs)"
                  % (epoch, global_step, agg / max(1, nb), vm["log_loss"],
                     vm["brier"], vm["roc_auc"], vm["accuracy"], vm["ece"],
                     time.time() - t0))
            history.append({"epoch": epoch, "step": global_step,
                            "train_loss": agg / max(1, nb),
                            **{("val_" + k): vm[k] for k in
                               ("log_loss", "brier", "roc_auc", "accuracy",
                                "ece")}})
            if run is not None:
                import wandb
                wandb.log({"train/loss": agg / max(1, nb),
                           "train/lr": sched.get_last_lr()[0],
                           "train/side_bias": float(model.side_bias),
                           **{"val/" + k: vm[k] for k in
                              ("log_loss", "brier", "roc_auc", "accuracy",
                               "ece")}}, step=global_step)
            if vm["log_loss"] < best:
                best = vm["log_loss"]
                save_ckpt(out_dir / "best.pt", model, cfg,
                          dict(vm, epoch=epoch), global_step, epoch,
                          patch_vocab)
        save_ckpt(out_dir / "latest.pt", model, cfg, {"epoch": epoch},
                  global_step, epoch, patch_vocab)

    # ---- final scoring of the selected checkpoint
    blob = torch.load(str(out_dir / "best.pt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()

    yv, pv, _ = predict(model, va_loader, device)
    val_rep = summarise(yv, pv)
    test_rep = None
    if te_loader is not None:
        yt, pt, ids = predict(model, te_loader, device)
        test_rep = summarise(yt, pt)
        D.save_predictions(out_dir / "test_preds.npz", yt, pt, table,
                           splits["test"])
        D.save_predictions(out_dir / "val_preds.npz", yv, pv, table,
                           splits["val"])

    # A joint run pools 38,436 soloQ and 4,762 pro test games, so the pooled
    # number is 89% a soloQ number. The per-domain blocks are the real output;
    # the pooled one is only there so the arm is comparable with itself.
    val_dom = summarise_by_domain(table, splits["val"], yv, pv)
    test_dom = summarise_by_domain(table, splits["test"], yt, pt) \
        if test_rep is not None else {}

    # ---- baseline on the same rows, same seed, same split
    base = BlueWinRateBaseline().fit(table.y[splits["train"]])
    base_val = summarise(yv, base.predict_proba(len(yv)))
    base_test = (summarise(yt, base.predict_proba(len(yt)))
                 if test_rep is not None else None)
    groups = ([table.series_id[i] for i in splits["test"]]
              if test_rep is not None else [])
    deltas = None
    if test_rep is not None:
        bp = base.predict_proba(len(yt))
        deltas = {
            "vs_baseline": {m: M.bootstrap_delta(yt, bp, pt, groups,
                                                 metric=m, seed=args.seed)
                            for m in ("log_loss", "brier", "accuracy")},
            "auc_delta": test_rep["roc_auc"] - base_test["roc_auc"],
            "by_domain": domain_deltas(table, splits["test"], yt, pt, groups,
                                       bp, args.seed,
                                       train_idx=splits["train"]),
        }

    report = {
        "protocol": {
            "data": args.data, "corpus": meta.get("corpus"),
            "seed": args.seed, "loss": args.loss,
            "focal": ({"gamma": args.focal_gamma, "alpha": args.focal_alpha}
                      if args.loss == "focal" else None),
            "selection": "val log_loss",
            "selected_epoch": blob.get("epoch"),
            "best_val_log_loss": best,
            "corpus_max_date": meta.get("corpus_max_date", ""),
            "newest_row_date": meta.get("newest_row_date", ""),
            "patch_vocab_size": len(patch_vocab),
            "split": split_rep,
            "data_info": table.info(),
            "dropped": meta.get("dropped", {}),
            "hyperparameters": {
                "embed_dim": args.embed_dim, "num_intra": args.num_intra,
                "num_cross": args.num_cross, "num_heads": args.num_heads,
                "dropout": args.dropout, "pool": args.pool,
                "patch_emb": bool(args.patch_emb), "attrs": bool(args.attr),
                "batch_size": args.batch_size, "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_frac": args.warmup_frac, "epochs": args.epochs,
                "grad_clip": args.grad_clip,
            },
            "params": W.param_count(model),
            "side_bias": float(model.side_bias),
            "checkpoint": str(out_dir / "best.pt"),
        },
        "val": val_rep,
        "test": test_rep,
        "val_by_domain": val_dom,
        "test_by_domain": test_dom,
        "baseline": {"prior": base.describe(), "val": base_val,
                     "test": base_test},
        "comparison": deltas,
        "history": history,
    }
    D.write_json(out_dir / "test_report.json", report)

    print("\n" + "=" * 72)
    print(M.format_report(val_rep, "MODEL  val"))
    if test_rep is not None:
        print(M.format_report(test_rep, "MODEL  test"))
        print(M.format_report(base_test, "BASE   test"))
        print("delta vs baseline (test, negative = model better):")
        for m, d in deltas["vs_baseline"].items():
            print("  %-10s %+.5f  95%% CI [%+.5f, %+.5f]  %s"
                  % (m, d["delta"], d["lo"], d["hi"],
                     "excludes 0" if d["excludes_zero"] else "straddles 0"))
        print("  roc_auc    %+.5f  (baseline is 0.5 by construction)"
              % deltas["auc_delta"])
        for dom, d in deltas["by_domain"].items():
            print("  -- %s only (n=%d) --" % (dom, d["n"]))
            print(M.format_report(test_dom[dom], "  %s" % dom, indent=2))
            for m, b in d["vs_baseline"].items():
                print("    d%-9s %+.5f  95%% CI [%+.5f, %+.5f]  %s"
                      % (m, b["delta"], b["lo"], b["hi"],
                         "excludes 0" if b["excludes_zero"]
                         else "straddles 0"))
            print("    d%-9s %+.5f" % ("roc_auc", d["auc_delta"]))
    print("side bias (learned): %+.4f  vs prior logit %+.4f"
          % (float(model.side_bias),
             math.log(base.prior / (1 - base.prior))))
    print("=" * 72)
    print("wrote %s" % (out_dir / "test_report.json"))

    if run is not None:
        import wandb
        final = {"test/log_loss": test_rep["log_loss"] if test_rep else None,
                 "test/brier": test_rep["brier"] if test_rep else None,
                 "test/roc_auc": test_rep["roc_auc"] if test_rep else None}
        for dom, rep in (test_dom or {}).items():
            final["test_%s/log_loss" % dom] = rep["log_loss"]
            final["test_%s/roc_auc" % dom] = rep["roc_auc"]
        wandb.log(final)
        wandb.finish()
    return report


def main():
    report = train(parse_args())
    print(json.dumps({"seed": report["protocol"]["seed"],
                      "data": report["protocol"]["data"],
                      "test": report["test"] and
                      {k: report["test"][k] for k in
                       ("n", "log_loss", "brier", "roc_auc", "accuracy")}},
                     indent=1))


if __name__ == "__main__":
    main()
