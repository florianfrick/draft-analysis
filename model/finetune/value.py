"""Prefix-sampled value stage: fix the value head on the PMA readout.

Why this file exists
--------------------
There are three value paths in the repo and only one of them works:

  encoder pma + win_head   test AUC 0.5597   <- the good one (soloQ pretrain)
  finetune PoolOutcomeHeads  test AUC 0.5172   <- weak, but complete drafts
  policy value_head          test AUC 0.4969   <- broken at every depth

A since-deleted diagnostic (`model/policy/diagnose_value.py`) showed the policy
head is not "diluted by early moves": it emits a constant for moves 0-6 (Brier
0.2483, exactly the base-rate figure) and then varies *anti*-correlated with
the label, ending at Brier 0.3527 on full drafts. It has no signal at any
depth, so a ban search cannot rank bans with it - which is what the depth-2
search found (human edge +0.00004, z +4.7, i.e. humans already sit at the
optimum of a V that is 0.9 winrate points wide).

The fix is not a new head shape. It is: use the readout that works (PMA, not
the 4D PAD-masked mean), train it on the states the search actually visits
(pro prefixes, not complete soloQ drafts), and let it see the ban list, which
the encoder cannot (`_encode` reads champ_ids/role_ids/pad_mask only).

What this trains
----------------
`PrefixValueModel` (see finetune/model.py): trunk frozen, `pma` + `win_head`
fine-tuned, plus a PMA ban pool and a zero-init residual block so the logit
starts exactly at the pretrained head and learns its ban correction on top.

Scoring is accuracy (AUC / Brier / skill vs a constant), because that is what
a value function should be scored on. The ban *decisions* derived from it are
a separate question, and were scored on regret by the ban-analysis thread that
was removed on 2026-11-03 (conclusions in STATUS.md §5). The two can disagree:
a global additive V predicted wins better (0.5508 vs 0.5456) while giving every
ban the same value.

Usage:
    python model/finetune/value.py --init-ckpt auto --epochs 12
    python model/finetune/value.py --limit-drafts 3000 --epochs 1 --no-wandb
    python model/finetune/value.py --resume model/finetune/ckpt/value_x/latest.pt
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
PRETRAIN_CKPT_GLOB = str(MODEL_DIR / "checkpoints" / "masking-soloqueue" / "*.pt")


def _load_module_by_path(name, path):
    spec = __import__("importlib").util.spec_from_file_location(
        name, str(path))
    mod = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(ROOT))
from model.pretraining.soloq_model import SoloQPretrainedModel  # noqa: E402
from model.pretraining.soloq_model import attr_matrix_for_config  # noqa: E402
from model import PrefixValueModel  # noqa: E402

PRO_PIPE = _load_module_by_path("pro_pipeline", PRO_DIR / "pipeline.py")
EP = _load_module_by_path("eval_protocol", PRO_DIR / "eval_protocol.py")
PRO_DATA = _load_module_by_path("pro_datasets", PRO_DIR / "datasets.py")

PHASE_NAMES = {0: "bans1", 1: "picks1", 2: "bans2", 3: "picks2"}


# --------------------------------------------------------------------- args
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pro prefix value stage")
    # data
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--swap-aug", dest="swap_aug", action="store_true",
                    default=True, help="mirror blue<->red with p=0.5")
    ap.add_argument("--no-swap-aug", dest="swap_aug", action="store_false")
    # model / init
    ap.add_argument("--init-ckpt", default="auto",
                    help="value-capable soloQ ckpt, or 'auto' = best "
                         "champ_top5 among checkpoints carrying pma.*")
    ap.add_argument("--no-bans", dest="use_bans", action="store_false",
                    default=True,
                    help="drop the ban pool + residual block (ablation: "
                         "pma+win_head fine-tuned alone)")
    ap.add_argument("--unfreeze", choices=("none", "cross", "intra"),
                    default="none",
                    help="how much of the frozen trunk to open up. 'none' "
                         "fine-tunes the value path only; 'cross' adds "
                         "cross_team; 'intra' adds the shared intra encoder. "
                         "Embeddings never unfreeze (they carry the soloQ "
                         "vocab semantics)")
    ap.add_argument("--ban-heads", type=int, default=4)
    ap.add_argument("--ban-seeds", type=int, default=4)
    ap.add_argument("--ban-dropout", type=float, default=0.1)
    # optimisation
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-pretrained", type=float, default=2e-4,
                    help="lr for the fine-tuned pma/win_head")
    ap.add_argument("--lr-new", type=float, default=1e-3,
                    help="lr for the ban pool + residual block")
    ap.add_argument("--lr-trunk", type=float, default=2e-5,
                    help="lr for any trunk module opened by --unfreeze")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=4,
                    help="epochs without val AUC improvement before stop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    # eval / io
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="cap val prefixes per epoch (0 = all)")
    ap.add_argument("--min-stratum", type=int, default=200,
                    help="min examples for a patch/phase row to be reported")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--score-ckpt", default="",
                    help="skip training: load this checkpoint, calibrate on "
                         "val, score val/test + complete drafts, and write "
                         "the report (re-scoring without retraining)")
    ap.add_argument("--also-full", dest="also_full", action="store_true",
                    default=True,
                    help="also score complete drafts (the apples-to-apples "
                         "comparison with the finetune win head and the "
                         "logistic additive V)")
    ap.add_argument("--no-also-full", dest="also_full", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lol-draft-value")
    ap.add_argument("--run-name", default="")
    return ap.parse_args(argv)


# ------------------------------------------------------------------- loading
def pick_value_ckpt(spec):
    """Resolve --init-ckpt. 'auto' = best champ_top5 among value-capable."""
    if spec != "auto":
        payload = torch.load(str(spec), map_location="cpu",
                             weights_only=False)
        return str(spec), payload
    best, best_score, n_skip = None, -1.0, 0
    for path in sorted(glob.glob(PRETRAIN_CKPT_GLOB)):
        try:
            payload = torch.load(path, map_location="cpu",
                                 weights_only=False)
        except Exception:  # noqa: BLE001 - skip unreadable
            n_skip += 1
            continue
        if not any(k.startswith("pma.") for k in payload.get("model", {})):
            n_skip += 1
            continue
        score = float(payload.get("metrics", {}).get("champ_top5", -1.0))
        if score > best_score:
            best, best_score = (path, payload), score
    if best is None:
        raise FileNotFoundError(
            "no value-capable checkpoint under %s - the value path needs a "
            "run trained with with_value=True (pma.* + win_head.* keys)"
            % PRETRAIN_CKPT_GLOB)
    print("  auto init: %d checkpoint(s) skipped (no pma.*), picked %s"
          % (n_skip, best[0]))
    return best


def build_value_encoder(payload):
    """Encoder WITH the value path, weights loaded verbatim."""
    cfg = payload["config"]
    if not cfg.get("with_value"):
        raise ValueError("checkpoint %s was trained with with_value=False; "
                         "it has no pma/win_head to fine-tune"
                         % cfg.get("_path", "?"))
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        with_value=True, value_seeds=cfg.get("value_seeds", 4),
        attr_matrix=attr_matrix_for_config(cfg))
    missing, unexpected = enc.load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "value checkpoint does not match the encoder: missing=%s "
            "unexpected=%s" % (list(missing)[:6], list(unexpected)[:6]))
    return enc, cfg


def load_pro_rows(args):
    rows, _stats = PRO_PIPE.build_rows(limit=args.limit_drafts)
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_series_by_date(rows)
    # Val must sit in the condition test is in (test is 100% unseen
    # patches; the raw val window is only 81.5% unseen and is easier).
    splits["val"], info = EP.patch_honest_val(rows, splits)
    args.patch_honest_val = info
    return rows, splits


# ------------------------------------------------------------------- metrics
def sigmoid(x):
    """Overflow-safe logistic for a Python float."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def summarize(logits, labels, platt=(1.0, 0.0)):
    """AUC / Brier / skill-vs-constant for one slice.

    `skill` is 1 - brier/brier_const, i.e. R^2 against always predicting the
    slice's base rate. Negative means the head is worse than a constant -
    the number that would have caught the policy value head immediately.
    """
    n = len(labels)
    if n == 0:
        return {"n": 0, "auc": float("nan"), "brier": float("nan"),
                "brier_const": float("nan"), "skill": float("nan"),
                "base_rate": float("nan"), "mean_p": float("nan")}
    a, b = platt
    ps = [sigmoid(a * x + b) for x in logits]
    ys = [int(y) for y in labels]
    base = sum(ys) / n
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    brier_const = sum((base - y) ** 2 for y in ys) / n
    skill = 1.0 - brier / brier_const if brier_const > 0 else float("nan")
    return {"n": n, "auc": float(EP.auc(logits, ys)), "brier": brier,
            "brier_const": brier_const, "skill": skill,
            "base_rate": base, "mean_p": sum(ps) / n}


def strata(logits, labels, keys, min_n=200):
    """Per-key summary, sorted by n descending, dropping tiny slices."""
    buckets = {}
    for lg, y, k in zip(logits, labels, keys):
        b = buckets.setdefault(k, ([], []))
        b[0].append(lg)
        b[1].append(y)
    out = {}
    for k, (ls, ys) in buckets.items():
        if len(ys) < min_n:
            continue
        out[k] = summarize(ls, ys)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


@torch.no_grad()
def collect(model, loader, device, use_bans=True):
    """Run the model over a loader; return flat arrays + per-example keys."""
    model.eval()
    lg, lgb, ys, mv, pat, ph = [], [], [], [], [], []
    for batch in loader:
        dev = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in batch.items()}
        out = model(dev, with_bans=use_bans)
        lg.append(out["win_logit"].squeeze(-1).float().cpu())
        lgb.append(out["win_logit_base"].squeeze(-1).float().cpu())
        ys.append(batch["win_label"].cpu())
        mv.append(batch["move_idx"].cpu())
        ph.append(batch["phase"].cpu())
        pat.extend([m.get("patch", "?") for m in batch["meta"]])
    return {"logit": torch.cat(lg).tolist(),
            "logit_base": torch.cat(lgb).tolist(),
            "y": torch.cat(ys).tolist(),
            "move_idx": torch.cat(mv).tolist(),
            "phase": torch.cat(ph).tolist(),
            "patch": pat}


def fit_platt(logits, labels):
    """1-D logistic calibration on val logits; identity on failure."""
    a = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=100,
                            line_search_fn="strong_wolfe")
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy(torch.sigmoid(a * x + b), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:  # noqa: BLE001 - keep identity fallback
        return 1.0, 0.0
    return float(a.detach()), float(b.detach())


def full_report(rec, platt, args, use_bans=True):
    """Overall + per-phase + per-depth + per-patch + ban ablation."""
    lg, ys = rec["logit"], rec["y"]
    rep = {"overall": summarize(lg, ys, platt)}
    rep["overall_raw"] = summarize(lg, ys, (1.0, 0.0))
    rep["by_phase"] = strata(lg, ys, [PHASE_NAMES.get(int(p), str(p))
                                      for p in rec["phase"]],
                             min_n=args.min_stratum)
    rep["by_move"] = strata(lg, ys, [int(m) for m in rec["move_idx"]],
                            min_n=args.min_stratum)
    rep["by_patch"] = strata(lg, ys, rec["patch"], min_n=args.min_stratum)
    rep["focus_patch"] = EP.FOCUS_PATCH
    rep["focus"] = rep["by_patch"].get(EP.FOCUS_PATCH)
    # The ablation that matters: how much does the ban path buy?
    rep["no_ban_residual"] = summarize(rec["logit_base"], ys, platt)
    if use_bans:
        d = [a - b for a, b in zip(lg, rec["logit_base"])]
        rep["ban_residual"] = {"mean_abs": sum(abs(x) for x in d) / len(d),
                               "max_abs": max(abs(x) for x in d)}
    rep["platt"] = {"a": platt[0], "b": platt[1]}
    rep["n_games_note"] = ("20 prefixes per game share one label, so n is "
                           "not 20 independent draws; by_move is the honest "
                           "per-depth view")
    return rep


@torch.no_grad()
def score_full_drafts(model, rows, idxs, num_champs, device, batch_size,
                      use_bans=False):
    """AUC/Brier on COMPLETE pro drafts, one example per game.

    The prefix dataset never contains the finished draft (move_idx 19 is
    the state *before* the last pick), so this is the only way to compare
    against the two other value functions on identical inputs:

        finetune PoolOutcomeHeads.win   test AUC 0.5172  (n=3144)
        logistic role x champ additive  test AUC 0.5456  (n=3189)

    `use_bans=False` is the point, not a limitation: ProMaskedDraftDataset
    emits no ban_ids, so this measures the PMA path alone - which is also
    the ban-blind condition the other two are in.
    """
    ds = PRO_DATA.ProMaskedDraftDataset(
        rows, idxs, num_champs=num_champs, mask_prob=0.0, role_keep_prob=0.0,
        econ_mask_prob=0.0)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=PRO_DATA.pro_collate)
    model.eval()
    lg, ys = [], []
    for batch in ld:
        dev = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in batch.items()}
        out = model(dev, with_bans=use_bans)
        lg.append(out["win_logit"].squeeze(-1).float().cpu())
        ys.append(batch["win_label"].cpu())
    return summarize(torch.cat(lg).tolist(), torch.cat(ys).tolist(),
                     (1.0, 0.0))


def format_report(rep, title):
    o = rep["overall"]
    ob = rep["no_ban_residual"]
    lines = ["", "=== %s ===" % title,
             "n=%-7d base=%.4f  AUC=%.4f  Brier=%.4f (const %.4f)  "
             "skill=%+.4f  mean_p=%.4f"
             % (o["n"], o["base_rate"], o["auc"], o["brier"],
                o["brier_const"], o["skill"], o["mean_p"]),
             "  ablation  no ban residual: AUC=%.4f  Brier=%.4f  skill=%+.4f"
             % (ob["auc"], ob["brier"], ob["skill"])]
    if "ban_residual" in rep:
        lines.append("  ban residual |delta logit| mean=%.4f max=%.4f"
                     % (rep["ban_residual"]["mean_abs"],
                        rep["ban_residual"]["max_abs"]))
    for name in ("by_phase", "by_move"):
        rows = rep[name]
        if not rows:
            continue
        lines.append("  %s:" % name)
        for k, v in rows.items():
            lines.append("    %-6s n=%-6d AUC=%.4f Brier=%.4f skill=%+.4f"
                         % (k, v["n"], v["auc"], v["brier"], v["skill"]))
    if rep.get("focus"):
        f = rep["focus"]
        lines.append("  patch %s (focus): n=%d AUC=%.4f skill=%+.4f"
                     % (rep["focus_patch"], f["n"], f["auc"], f["skill"]))
    return "\n".join(lines)


# ------------------------------------------------------------------- training
def run_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb
        return wandb.init(project=args.wandb_project,
                          name=args.run_name or None,
                          config=vars(args), dir=str(HERE),
                          settings=wandb.Settings(init_timeout=60))
    except Exception as exc:  # noqa: BLE001 - logging never kills training
        print("wandb disabled: %s" % exc)
        return None


def make_optimizer(model, args):
    """Three groups: pretrained value path, new ban modules, opened trunk."""
    pretrained = [p for m in (model.encoder.pma, model.encoder.win_head)
                  for p in m.parameters()]
    new = [p for m in (model.ban_pool, model.ban_block) if m is not None
           for p in m.parameters()]
    used = {id(p) for p in pretrained} | {id(p) for p in new}
    trunk = [p for p in model.parameters()
             if p.requires_grad and id(p) not in used]
    groups = [{"params": pretrained, "lr": args.lr_pretrained},
              {"params": new, "lr": args.lr_new}]
    if trunk:
        groups.append({"params": trunk, "lr": args.lr_trunk})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def open_trunk(model, how):
    """Unfreeze cross_team (and optionally intra_team). Embeddings stay put."""
    mods = [model.encoder.cross_team]
    if how == "intra":
        mods.append(model.encoder.intra_team)
    for m in mods:
        for p in m.parameters():
            p.requires_grad = True


def lr_lambda(step, total, warmup_frac):
    """Linear warmup then cosine to 5% of the peak."""
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))


def save_ckpt(path, model, args, cfg, metrics, step, platt, init_ckpt,
              optim_state=None, epoch=None):
    """Write a checkpoint. `optim_state` is only passed for latest.pt.

    best.pt stays small and loadable by anything that just wants weights;
    latest.pt carries optimizer + scheduler + epoch so --resume continues
    the LR schedule instead of restarting it at warmup.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(),
               "config": {"num_champs": cfg["num_champs"],
                          "embed_dim": cfg["embed_dim"],
                          "num_heads": cfg["num_heads"],
                          "num_intra": cfg["num_intra"],
                          "num_cross": cfg["num_cross"],
                          "dropout": cfg.get("dropout", 0.0),
                          "with_value": True,
                          "value_seeds": cfg.get("value_seeds", 4),
                          "use_bans": args.use_bans,
                          "ban_heads": args.ban_heads,
                          "ban_seeds": args.ban_seeds,
                          "ban_dropout": args.ban_dropout},
               "metrics": metrics, "step": step, "platt": platt,
               "init_ckpt": init_ckpt, "args": vars(args)}
    if optim_state is not None:
        payload["optim"] = optim_state["optim"]
        payload["sched"] = optim_state["sched"]
        payload["epoch"] = epoch
    torch.save(payload, str(path))


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print("device %s | torch %s" % (device, torch.__version__))

    init_path, payload = pick_value_ckpt(args.init_ckpt)
    print("init: %s" % init_path)
    encoder, cfg = build_value_encoder(payload)
    model = PrefixValueModel(encoder, ban_heads=args.ban_heads,
                             ban_seeds=args.ban_seeds,
                             ban_dropout=args.ban_dropout,
                             use_bans=args.use_bans).to(device)
    trainable = model.freeze_trunk()
    if args.unfreeze != "none":
        open_trunk(model, args.unfreeze)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print("trainable %s (%.1f%% of %s params)"
          % (n_tr, 100.0 * n_tr / max(1, n_all), n_all))
    print("  %s" % ", ".join(trainable[:8]) + (" ..." if len(trainable) > 8
                                               else ""))

    rows, splits = load_pro_rows(args)
    print("rows %d | train %d | val(honest) %d | test %d"
          % (len(rows), len(splits["train"]), len(splits["val"]),
             len(splits["test"])))
    num_champs = cfg["num_champs"]
    tr_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["train"], num_champs=num_champs,
        prefix_mode="random", swap_aug=args.swap_aug, seed=args.seed)
    va_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["val"], num_champs=num_champs, prefix_mode="all")
    te_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["test"], num_champs=num_champs, prefix_mode="all")
    if args.eval_limit:
        va_ds.rows = va_ds.rows[:args.eval_limit]
    sampler = PRO_DATA.make_weighted_sampler(tr_ds, seed=args.seed)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                       collate_fn=PRO_DATA.pro_collate,
                       num_workers=args.num_workers, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=PRO_DATA.pro_collate,
                       num_workers=args.num_workers)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=PRO_DATA.pro_collate,
                       num_workers=args.num_workers)

    opt = make_optimizer(model, args)
    total_steps = max(1, args.epochs * len(tr_ld))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, args.warmup_frac))

    start_epoch, best_auc, step = 0, -1.0, 0
    if args.resume:
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        if "optim" in blob:
            opt.load_state_dict(blob["optim"])
        if "sched" in blob:
            sched.load_state_dict(blob["sched"])
        start_epoch = int(blob.get("epoch", 0)) + 1
        best_auc = float(blob.get("metrics", {}).get("val_auc", -1.0))
        step = int(blob.get("step", 0))
        print("resumed %s at epoch %d (best val AUC %.4f)"
              % (args.resume, start_epoch, best_auc))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else \
        HERE / "ckpt" / ("value_%s" % stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("out: %s" % out_dir)
    wb = run_wandb(args)

    hist, bad_epochs = [], 0
    for ep in range(start_epoch, args.epochs):
        tr_ds.set_epoch(ep)
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in tr_ld:
            dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                   for k, v in batch.items()}
            out = model(dev)
            logit = out["win_logit"].squeeze(-1)
            y = dev["win_label"].float()
            w = dev["weight"].float()
            loss = F.binary_cross_entropy_with_logits(logit, y, weight=w,
                                                      reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            tot += float(loss.detach())
            nb += 1
        tr_loss = tot / max(1, nb)

        rec = collect(model, va_ld, device, use_bans=args.use_bans)
        va = summarize(rec["logit"], rec["y"], (1.0, 0.0))
        va["loss"] = tr_loss
        hist.append({"epoch": ep, "train_loss": tr_loss,
                     "val_auc": va["auc"], "val_brier": va["brier"],
                     "val_skill": va["skill"]})
        print("ep %2d  loss %.4f  val AUC %.4f  Brier %.4f  skill %+.4f  "
              "(%.1fs)" % (ep, tr_loss, va["auc"], va["brier"], va["skill"],
                           time.time() - t0))
        if wb is not None:
            wb.log({"epoch": ep, "train_loss": tr_loss,
                    "val_auc": va["auc"], "val_brier": va["brier"],
                    "val_skill": va["skill"]})
        save_ckpt(out_dir / "latest.pt", model, args, cfg, va, step,
                  (1.0, 0.0), init_path,
                  optim_state={"optim": opt.state_dict(),
                               "sched": sched.state_dict()}, epoch=ep)
        if va["auc"] > best_auc:
            best_auc, bad_epochs = va["auc"], 0
            save_ckpt(out_dir / "best.pt", model, args, cfg, va, step,
                      (1.0, 0.0), init_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stop: no val AUC gain for %d epochs"
                      % bad_epochs)
                break

    # ---- final: calibrate on val, score test with the best checkpoint
    best_blob = torch.load(out_dir / "best.pt", map_location="cpu",
                           weights_only=False)
    model.load_state_dict(best_blob["model"])
    model.to(device)
    rec_val = collect(model, va_ld, device, use_bans=args.use_bans)
    platt = fit_platt(rec_val["logit"], rec_val["y"])
    print("platt: a=%.4f b=%.4f" % platt)
    rec_te = collect(model, te_ld, device, use_bans=args.use_bans)

    report = {"init_ckpt": init_path, "out_dir": str(out_dir),
              "args": vars(args),
              "patch_honest_val": args.patch_honest_val,
              "history": hist,
              "splits": {"val_honest": full_report(rec_val, platt, args,
                                                   args.use_bans),
                         "test": full_report(rec_te, platt, args,
                                             args.use_bans)}}
    for name in ("val_honest", "test"):
        print(format_report(report["splits"][name], name))
    (out_dir / "value_report.json").write_text(json.dumps(report, indent=1))

    # re-save best with the calibrated platt so consumers get it for free
    save_ckpt(out_dir / "best.pt", model, args, cfg,
              {"val_auc": report["splits"]["val_honest"]["overall"]["auc"],
               "test_auc": report["splits"]["test"]["overall"]["auc"]},
              step, platt, init_path)
    if wb is not None:
        for name in ("val_honest", "test"):
            o = report["splits"][name]["overall"]
            wb.summary["%s_auc" % name] = o["auc"]
            wb.summary["%s_skill" % name] = o["skill"]
        wb.finish()
    print("\nwrote %s" % (out_dir / "value_report.json"))
    return report


def score_only(args):
    """Re-score a trained checkpoint without retraining."""
    device = torch.device(args.device)
    blob = torch.load(args.score_ckpt, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        with_value=True, value_seeds=cfg.get("value_seeds", 4),
        attr_matrix=attr_matrix_for_config(cfg))
    model = PrefixValueModel(
        enc, ban_heads=cfg["ban_heads"], ban_seeds=cfg["ban_seeds"],
        ban_dropout=cfg["ban_dropout"], use_bans=cfg["use_bans"]).to(device)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError("checkpoint mismatch: missing=%s unexpected=%s"
                           % (list(missing)[:6], list(unexpected)[:6]))
    model.eval()
    args.use_bans = cfg["use_bans"]
    print("scoring %s" % args.score_ckpt)

    rows, splits = load_pro_rows(args)
    num_champs = cfg["num_champs"]
    va_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["val"], num_champs=num_champs, prefix_mode="all")
    te_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["test"], num_champs=num_champs, prefix_mode="all")
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=PRO_DATA.pro_collate)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=PRO_DATA.pro_collate)
    rec_val = collect(model, va_ld, device, use_bans=args.use_bans)
    platt = fit_platt(rec_val["logit"], rec_val["y"])
    rec_te = collect(model, te_ld, device, use_bans=args.use_bans)
    report = {"score_ckpt": args.score_ckpt, "args": vars(args),
              "splits": {"val_honest": full_report(rec_val, platt, args,
                                                   args.use_bans),
                         "test": full_report(rec_te, platt, args,
                                             args.use_bans)}}
    if args.also_full:
        report["complete_drafts"] = {
            "val_honest": score_full_drafts(model, rows, splits["val"],
                                            num_champs, device,
                                            args.batch_size),
            "test": score_full_drafts(model, rows, splits["test"],
                                      num_champs, device, args.batch_size)}
    for name in ("val_honest", "test"):
        print(format_report(report["splits"][name], name))
    if args.also_full:
        for name, m in report["complete_drafts"].items():
            print("\n=== complete drafts (%s) ===\nn=%d  base=%.4f  "
                  "AUC=%.4f  Brier=%.4f (const %.4f)  skill=%+.4f"
                  % (name, m["n"], m["base_rate"], m["auc"], m["brier"],
                     m["brier_const"], m["skill"]))
    out = Path(args.out_dir) if args.out_dir else \
        Path(args.score_ckpt).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "score_report.json").write_text(json.dumps(report, indent=1))
    print("\nwrote %s" % (out / "score_report.json"))
    return report


def main():
    args = parse_args()
    if args.score_ckpt:
        score_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
