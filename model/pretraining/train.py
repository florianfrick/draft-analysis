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

Value head (opt-in, on by default; --no-win reverts): the shards always
carried teams.{blue,red}.win and pretraining never read it. A pooled win
head is now trained alongside masked modelling:
loss = CE_champ + lambda_role * CE_role + lambda_win * BCE(blue_win)
The point is not the head itself - the pro stage builds its own - but the
representation: masked-champ alone never has to know which comp wins, so the
pooled vector carried no outcome signal and the downstream value head sat at
the base rate for 8k steps (AUC 0.497, Brier 0.2510 vs 0.2482 for a constant).
Value metrics are measured on a CLEAN val view (no champion masked), because
a 15%-masked draft caps any readout and would flatter or punish the wrong
thing. `win_skill` = 1 - brier/brier_const; > 0 means it beats the constant.

Champ attributes (P3.1, on by default; --no-champ-attrs reverts): each
champion's embedding gets a residual from the 18-axis hand-tagged kit table
in data/champ_attrs.json (DraftSlotEmbedding.attr_proj). The projection is
zero-initialised, so with the same champ/role tables the model computes
exactly what it did before - "attributes on/off" is one flag, not two
initialisations. Two reasons it is worth 4.9k parameters: the axes are shared
across champions, so what the model learns about "engage" transfers to a
champion it has seen rarely; and a champion released after the training data
was collected has no useful embedding row but does have a tag row, which is
the difference between a cold start and a random vector. `attr_dim` goes into
the checkpoint config and every downstream stage rebuilds the adapter from it.

W&B: key read from pretraining/wandb_key.txt or WANDB_API_KEY env var;
--no-wandb disables. Checkpoints: ckpt/run_<stamp>/{
  best.pt        best val champ_top5 (weights + config only)
  best_value.pt  best val win AUC
  latest.pt      every epoch, + optimizer/scheduler/scaler/best counters
}
Resume with --resume auto (newest latest.pt) or an explicit run dir / .pt.
It restores model, AdamW moments, the cosine schedule position, the AMP
scaler scale, global_step and the best-so-far counters, and keeps writing
to the original run dir - so a killed 40-epoch run costs one epoch, not
the whole run. latest.pt is written atomically (tmp + rename), otherwise an
interrupted save would be the one file resume needs.
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
from soloq_model import (SoloQPretrainedModel,  # noqa: E402
                         load_champ_attr_matrix, load_encoder_state)

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
    ap.add_argument("--lambda-win", type=float, default=1.0,
                    help="weight on the blue-win BCE head. 1.0 = one global "
                    "label weighted like one masked slot (~1.5/draft), which "
                    "is the fair per-label default; 0.5 is the conservative "
                    "setting if champ_top5 regresses. Ignored with --no-win")
    ap.add_argument("--no-win", action="store_true",
                    help="masked-champ + role only (old behaviour); no value "
                    "path, no win labels needed")
    ap.add_argument("--win-warmup-epochs", type=float, default=5.0,
                    help="ramp lambda_win linearly from 0 to its full value "
                    "over this many epochs. Measured: the win head cannot "
                    "learn before the encoder knows champion identity - at 700 "
                    "steps the probe read 0.4996 and the head sat at the prior "
                    "(AUC 0.497), while the same head on a mature frozen "
                    "encoder reached 0.5396 in 700 steps. 0 disables the ramp")
    ap.add_argument("--win-patience", type=int, default=8,
                    help="freeze the value loss after this many epochs with no "
                    "val win-AUC gain (0 = never freeze). Measured: in the "
                    "D=512 run train win BCE fell 0.684->0.436 while val Brier "
                    "rose 0.2467->0.3857 and AUC fell 0.5601->0.5206, so the "
                    "last 25 epochs damaged both the head and the encoder")
    ap.add_argument("--win-freeze", action="store_true",
                    help="train the encoder with the value loss off from the "
                    "start epoch (the manual override of --win-patience; "
                    "useful when resuming a run whose value head already "
                    "overfit). The head still gets evaluated")
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
    ap.add_argument("--value-seeds", type=int, default=4,
                    help="PMA seed queries for the value readout")
    ap.add_argument("--no-champ-attrs", action="store_true",
                    help="disable the kit-attribute residual from "
                    "data/champ_attrs.json (P3.1). On by default: the "
                    "adapter is zero-initialised, so it costs nothing at "
                    "step 0 and gives every champion 18 tag dimensions "
                    "instead of one embedding row - the ablation is this "
                    "flag, not a rewrite")
    # bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate every N epochs")
    ap.add_argument("--resume", default="",
                    help="continue from a checkpoint: 'auto' (newest "
                    "ckpt/run_*/latest.pt), a run directory, or a .pt path. "
                    "Restores model/optimizer/scheduler/scaler/global_step/"
                    "best-so-far and continues writing to the SAME run dir. "
                    "lr schedule is rebuilt from --epochs, so pass the "
                    "original --epochs for an exact continuation")
    ap.add_argument("--start-epoch", type=int, default=0,
                    help="override the epoch to resume at (0 = take it from "
                    "the checkpoint). Use to rewind or to skip a bad stretch")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. Masking is per-item Python, so "
                    "this is the loader's only lever: measured 151 ms/batch at "
                    "0 vs 91 ms at 4 on the full 384k cache, against a ~640 ms "
                    "GPU step. Worth ~10%%. On Windows each worker pickles the "
                    "row list, so raise it only if RAM allows")
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
    """Kept draft rows (+ win labels) from cache, or a scan.

    Cache schema 2 adds `wins`, aligned to `rows`. A schema-1 cache (no wins)
    is rebuilt automatically when the win head is on, instead of silently
    training a value head against nothing - that exact failure mode already
    happened once with econ labels.

    Returns (rows, wins_or_None, riot_to_idx).
    """
    cache = P.BASE_DIR / ("drafts_%s.pt" % args.patch)
    want_win = not args.no_win
    if args.limit_drafts:
        # Smoke path. Subsample the cache when we have one: the old version
        # scanned a prefix of the shards, and because the corpus opens with
        # the previous patch it kept 0 drafts and died on a divide-by-zero.
        # An evenly-strided slice over the whole cache is representative
        # instead of patch-biased.
        if cache.exists() and not args.rescan:
            payload = torch.load(str(cache), map_location="cpu",
                                 weights_only=False)
            if (payload.get("patch") == args.patch
                    and (payload.get("wins") is not None or not want_win)):
                mapping, _ = load_mapping(args.patch)
                n = len(payload["rows"])
                stride = max(1, n // args.limit_drafts)
                idx = list(range(0, n, stride))[:args.limit_drafts]
                rows = [payload["rows"][i] for i in idx]
                wins = ([payload["wins"][i] for i in idx]
                        if want_win else None)
                print("limited run: %d drafts (even stride %d of %d cached)"
                      % (len(rows), stride, n))
                return rows, wins, mapping
        mapping, _ = load_mapping(args.patch)
        rows, wins, kept, scanned = [], [], 0, 0
        for _, _, rec in P.iter_records(P.list_shards()):
            scanned += 1
            out = P.record_to_slots(rec, mapping, args.patch,
                                    return_win=want_win)
            if out is not None:
                if want_win:
                    if out[2] is None:
                        if scanned >= args.limit_drafts * 40:
                            break
                        continue
                    rows.append((out[0], out[1]))
                    wins.append(out[2])
                else:
                    rows.append(out)
                kept += 1
                if kept >= args.limit_drafts:
                    break
            if scanned >= args.limit_drafts * 40:
                break
        print("limited scan: kept %d (scanned %d)" % (kept, scanned))
        return rows, (wins if want_win else None), mapping
    if cache.exists() and not args.rescan:
        payload = torch.load(str(cache), map_location="cpu")
        has_wins = payload.get("wins") is not None
        if payload.get("patch") == args.patch and (has_wins or not want_win):
            mapping, _ = load_mapping(args.patch)
            print("loaded cache: %d drafts (%s)%s" %
                  (len(payload["rows"]), cache.name,
                   "" if has_wins else ", no win labels"))
            return payload["rows"], payload.get("wins"), mapping
        if payload.get("patch") == args.patch and want_win and not has_wins:
            print("cache has no win labels (schema %s) -> rescanning"
                  % payload.get("schema", 1))
    print("full shard scan (first run ~1 min, then cached)...")
    mapping, _ = load_mapping(args.patch)
    indexed = P.scan_and_index(P.list_shards(), mapping, args.patch,
                               with_win=want_win)
    torch.save({"schema": 2, "patch": args.patch, "rows": indexed["rows"],
                "wins": indexed["wins"],
                "kept": indexed["kept"], "skipped": indexed["skipped"],
                "skipped_no_win": indexed["skipped_no_win"]},
               str(cache))
    print("kept %d / skipped %d / unlabelled %d -> cached to %s" %
          (indexed["kept"], indexed["skipped"], indexed["skipped_no_win"],
           cache.name))
    return indexed["rows"], indexed["wins"], mapping


def split_indices(n, val_frac, seed):
    """Deterministic draft-level train/val index split."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def make_loader(rows, num_champs, batch_size, shuffle, mask_prob, seed,
                epoch=0, num_workers=0, role_only_prob=0.10, wins=None,
                force_one=True):
    ds = P.SoloQDraftDataset(rows, num_champs=num_champs,
                             mask_prob=mask_prob, role_only_prob=role_only_prob,
                             seed=seed, epoch=epoch, wins=wins,
                             force_one=force_one)
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


def win_weight(args, epoch, stalled, frozen=False):
    """lambda_win for this epoch, or 0.0 once the value head stops helping.

    Two measured reasons, both from the D=512 40-epoch run:
    * Ramp in. At 700 steps the probe read 0.4996 and the head sat exactly at
      the prior, while the same head on a mature frozen encoder hit 0.5396 in
      700 steps - the head was starved, not broken.
    * Cut out. Once the masked task saturates the win head memorises outcome
      noise: train BCE fell 0.684 -> 0.436 while val Brier rose 0.2467 ->
      0.3857 and AUC fell 0.5601 -> 0.5206, i.e. the last 25 epochs made both
      the value head and the encoder worse. Freezing the loss at that point
      leaves the good head in place and lets the encoder finish on the masked
      task alone.

    `frozen` LATCHES the decision. Without it the guard oscillates: with the
    loss frozen the head's weights stop moving but the encoder keeps improving,
    so the frozen head's AUC drifts up by ~1e-4, which used to reset the stall
    counter and switch the loss back on - observed at epoch 26 of run
    v2-D512-p8, after which AUC fell 0.5597 -> 0.5467. A decision to stop is
    one-way; only an explicit --win-freeze/--win-patience override changes it.
    """
    if args.no_win or frozen:
        return 0.0
    if args.win_patience and stalled >= args.win_patience:
        return 0.0
    if args.win_warmup_epochs > 0:
        return args.lambda_win * min(1.0, epoch / args.win_warmup_epochs)
    return args.lambda_win


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


def win_loss(out, batch):
    """BCE on the pooled blue-win logit. Zero (graph-connected) if absent."""
    if "win_logit" not in out or "win_label" not in batch:
        ref = next(iter(out.values()))
        return ref.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        out["win_logit"].squeeze(-1), batch["win_label"].float())


def _auc(scores, labels):
    """Rank AUC (Mann-Whitney) with tie handling. 0.5 on a degenerate split."""
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    n_pos = float((labels == 1).sum())
    n_neg = float(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    # average ranks across ties, otherwise AUC is biased by champion order
    uniq, inv = torch.unique(scores, return_inverse=True)
    sums = torch.zeros(uniq.numel(), dtype=torch.float64).index_add_(0, inv, ranks)
    counts = torch.zeros(uniq.numel(), dtype=torch.float64).index_add_(
        0, inv, torch.ones_like(ranks))
    ranks = (sums / counts)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


@torch.no_grad()
def evaluate_value(model, loader, device):
    """Value metrics on CLEAN drafts (no champion masked).

    Reported separately from `evaluate` for two reasons. The masked val view
    deliberately hides ~15% of champions, which caps how well any readout can
    know the outcome - fine for training the representation, misleading as a
    measure of it. And the numbers this has to be judged against are absolute:
    AUC 0.5507 for a 173-parameter logistic regression on the same pro data,
    and Brier 0.2482 for a constant predictor at the base rate. `win_skill`
    is 1 - brier/brier_const, so > 0 means the head beats the constant at all,
    which the previous value head never did.
    """
    if not model.with_value:
        return {}
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if "win_label" not in batch:
            return {}
        logits.append(model.forward_value(batch).squeeze(-1).float().cpu())
        labels.append(batch["win_label"].float().cpu())
    if not logits:
        return {}
    z = torch.cat(logits).double()
    y = torch.cat(labels).double()
    p = torch.sigmoid(z)
    base = float(y.mean())
    brier = float(((p - y) ** 2).mean())
    brier_const = base * (1.0 - base)
    return {
        "win_auc": _auc(z, y),
        "win_brier": brier,
        "win_brier_const": brier_const,
        "win_skill": 1.0 - brier / brier_const if brier_const > 0 else 0.0,
        "win_logloss": float(F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y)),
        "win_base_rate": base,
        "win_n": int(y.numel()),
    }


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
    """Init W&B (key from key file or env). Returns run or None.

    `dir` must NOT be HERE: wandb creates its log directory there, and because
    this script's own folder is on sys.path, `pretraining/wandb/` then shadows
    the installed package as a namespace module - `import wandb` succeeds but
    `wandb.init` does not exist, so logging silently dies with a cryptic
    "module has no attribute init". Log to wandb_runs/ instead.
    """
    if args.no_wandb:
        return None
    try:
        import wandb
        if not hasattr(wandb, "init"):
            print("wandb disabled: resolved to %s (a namespace package - a "
                  "local 'wandb' folder is shadowing the install)"
                  % getattr(wandb, "__file__", "?"))
            return None
        if "WANDB_API_KEY" not in os.environ and KEY_FILE.exists():
            os.environ["WANDB_API_KEY"] = KEY_FILE.read_text().strip()
        run = wandb.init(project=args.wandb_project,
                         name=args.run_name or None,
                         config=vars(args),
                         dir=str(HERE / "wandb_runs"),
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


def save_ckpt(path, model, args, num_champs, metrics, step, epoch=None,
              resume=None):
    """Write a checkpoint atomically (tmp + rename).

    Atomic matters because a 40-epoch run writes latest.pt every epoch and an
    interrupted write leaves an unloadable file that also breaks --resume.
    `resume` carries the training-state dict (optimizer/scheduler/scaler/
    bests); only latest.pt gets it, so best.pt/best_value.pt stay small and
    loadable by downstream scripts that only want `model`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(),
               "config": {"num_champs": num_champs,
                          "embed_dim": args.embed_dim,
                          "num_intra": args.num_intra,
                          "num_cross": args.num_cross,
                          "num_heads": args.num_heads,
                          "dropout": args.dropout,
                          "with_value": not args.no_win,
                          "value_seeds": args.value_seeds,
                          # Written from the model, not from the flag: if the
                          # two ever disagree the model is the truth, and
                          # every downstream stage rebuilds from this number.
                          "attr_dim": int(getattr(model, "attr_dim", 0))},
               "metrics": metrics, "step": step, "epoch": epoch,
               "patch": args.patch}
    if resume is not None:
        payload["resume"] = resume
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path))
    return path


def find_resume_ckpt(spec):
    """Resolve --resume to a concrete .pt path."""
    if spec.strip().lower() in ("auto", "last", "latest", "1", "true"):
        cands = sorted(CKPT_ROOT.glob("run_*/latest.pt"),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            sys.exit("--resume auto: no run_*/latest.pt under %s" % CKPT_ROOT)
        return cands[-1]
    p = Path(spec.strip())
    if not p.is_absolute() and not p.exists():
        # friendly fallback: "run_20261029_131950" means a run under ckpt/
        p = CKPT_ROOT / p
    if p.is_dir():
        p = p / "latest.pt"
    if not p.exists():
        sys.exit("--resume: %s not found (tried as given and under %s)"
                 % (spec, CKPT_ROOT))
    return p


def _ckpt_load(path, device):
    """torch.load that tolerates optimizer/scaler state (our own files)."""
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:            # torch < 1.13 has no weights_only
        return torch.load(str(path), map_location=device)


def restore(resume_path, model, opt, sched, scaler, args, device):
    """Load training state. Returns dict(epoch, step, best_top5, best_win_auc,
    run_dir). Refuses a shape mismatch instead of raising a wall of keys.
    """
    payload = _ckpt_load(resume_path, device)
    cfg = payload.get("config") or {}
    want = {"num_champs": model.num_champs, "embed_dim": args.embed_dim,
            "num_intra": args.num_intra, "num_cross": args.num_cross,
            "num_heads": args.num_heads, "with_value": not args.no_win}
    bad = ["%s: ckpt %s vs now %s" % (k, cfg.get(k), v)
           for k, v in want.items() if cfg.get(k) != v]
    if bad:
        sys.exit("--resume: architecture mismatch with %s\n  %s\n"
                 "pass the original size flags (or drop --resume to start "
                 "fresh)" % (resume_path, "\n  ".join(bad)))
    # attr_dim is compared separately because the asymmetry is real:
    #   * ckpt has no attr_dim (trained before the table existed) -> resuming
    #     with attributes ON is fine, the adapter is zero-init so step 0 is
    #     byte-identical; say so rather than letting it pass silently.
    #   * ckpt HAS a trained adapter and we were asked to run without it ->
    #     refuse. Dropping it would resume from a model that has forgotten
    #     everything the adapter learned, and the loss curve would look like
    #     a lr spike rather than a config mistake.
    ck_attr, now_attr = int(cfg.get("attr_dim") or 0), int(model.attr_dim)
    if ck_attr and ck_attr != now_attr:
        sys.exit("--resume: %s was trained with attr_dim=%d, this run has %d"
                 " (%s)\n  drop --no-champ-attrs to continue it, or start a "
                 "fresh run to ablate attributes"
                 % (resume_path, ck_attr, now_attr,
                    "attributes off" if not now_attr else "table changed"))
    if now_attr and not ck_attr:
        print("resume: %s predates the attribute table -> the %d-axis "
              "adapter starts at zero (identical to the old model at step 0)"
              % (resume_path.name, now_attr))
    # Tolerates a missing attr_proj (zero-init) but still raises on any other
    # mismatch, including a checkpoint key the model cannot take.
    load_encoder_state(model, payload["model"])
    st = payload.get("resume") or {}
    if not st:
        print("resume: %s has no training state (weights only) -> epoch/step/"
              "best counters restart at 0" % resume_path.name)
    if st.get("opt"):
        opt.load_state_dict(st["opt"])
        # load_state_dict restores param_groups, i.e. the OLD lr and weight
        # decay, silently overriding this run's flags. Re-apply them so
        # "resume with a lower lr after a divergence" actually does something.
        prev_lr = opt.param_groups[0].get("lr")
        for g in opt.param_groups:
            g["lr"] = args.lr
            g["weight_decay"] = args.weight_decay
        if prev_lr is not None and abs(prev_lr - args.lr) > 1e-12:
            print("resume: peak lr %.2e -> %.2e" % (prev_lr, args.lr))
    if st.get("sched"):
        try:
            sched.load_state_dict(st["sched"])
            # LambdaLR multiplies base_lrs, which the state dict just reset to
            # the checkpoint's lr; rebase so the new --lr takes effect.
            sched.base_lrs = [args.lr for _ in sched.base_lrs]
        except (KeyError, ValueError) as exc:
            print("resume: scheduler state skipped (%s); lr continues from "
                  "the rebuilt schedule" % exc)
    if st.get("scaler") and scaler.is_enabled():
        scaler.load_state_dict(st["scaler"])
    prev = st.get("args") or {}
    if prev.get("epochs") and prev["epochs"] != args.epochs:
        print("resume: warning --epochs %s != original %s; the cosine "
              "schedule is rebuilt for the new horizon"
              % (args.epochs, prev["epochs"]))
    epoch = st.get("epoch") or payload.get("epoch") or 0
    if args.start_epoch:
        epoch = args.start_epoch
    return {"epoch": int(epoch), "step": int(st.get("step",
                                                    payload.get("step") or 0)),
            "best_top5": float(st.get("best_top5", -1.0)),
            "best_win_auc": float(st.get("best_win_auc", -1.0)),
            "win_stalled": int(st.get("win_stalled", 0)),
            "win_frozen": bool(st.get("win_frozen", False)),
            "run_dir": st.get("run_dir", "")}


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    if amp:
        # fixed shapes every step (6+6 slots, constant batch) -> let cuDNN
        # autotune once instead of re-picking per kernel launch
        torch.backends.cudnn.benchmark = True

    rows, wins, riot_to_idx = load_rows(args)
    if not rows:
        sys.exit("no drafts loaded for patch %s (check --patch / --rescan; a "
                 "schema-1 cache or a prefix-only scan both land here)"
                 % args.patch)
    num_champs = len(riot_to_idx)
    if num_champs != 173:
        print("warning: %d champs on patch %s (contract example: 173)"
              % (num_champs, args.patch))
    tr_idx, va_idx = split_indices(len(rows), args.val_frac, args.data_seed)
    train_rows = [rows[i] for i in tr_idx]
    val_rows = [rows[i] for i in va_idx]
    train_wins = [wins[i] for i in tr_idx] if wins is not None else None
    val_wins = [wins[i] for i in va_idx] if wins is not None else None
    if not args.no_win and train_wins is None:
        sys.exit("--lambda-win needs win labels but none were loaded; "
                 "rerun with --rescan")
    print("train %d / val %d drafts, %d champs, value head: %s"
          % (len(train_rows), len(val_rows), num_champs,
             "off" if args.no_win else "on (lambda %.2f)" % args.lambda_win))
    if train_wins is not None:
        print("blue-win base rate: train %.4f / val %.4f"
              % (sum(train_wins) / len(train_wins),
                 sum(val_wins) / len(val_wins)))

    base = P.compute_popularity_baseline(train_rows, num_champs)
    tr_loader = make_loader(train_rows, num_champs, args.batch_size,
                            shuffle=True, mask_prob=0.15,
                            seed=args.seed,
                            role_only_prob=args.role_only_prob,
                            wins=train_wins,
                            num_workers=args.num_workers)
    va_loader = make_loader(val_rows, num_champs, args.batch_size,
                            shuffle=False, mask_prob=0.15,
                            seed=args.seed + 1,  # fixed val masks
                            role_only_prob=args.role_only_prob,
                            wins=val_wins,
                            num_workers=args.num_workers)
    # Clean view: zero champions masked, used only for value metrics.
    clean_loader = (make_loader(val_rows, num_champs, args.batch_size,
                                shuffle=False, mask_prob=0.0,
                                seed=args.seed + 2,
                                role_only_prob=0.0, wins=val_wins,
                                force_one=False)
                    if not args.no_win else None)

    model = SoloQPretrainedModel(
        num_champs=num_champs, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_intra_layers=args.num_intra,
        num_cross_layers=args.num_cross, dropout=args.dropout,
        with_value=not args.no_win, value_seeds=args.value_seeds,
        attr_matrix=(None if args.no_champ_attrs
                     else load_champ_attr_matrix(num_champs))).to(device)
    if model.attr_dim:
        print("champ attributes: %d axes from data/champ_attrs.json "
              "(adapter zero-init; --resume restores trained weights)"
              % model.attr_dim)
    else:
        print("champ attributes: off (champion embedding only)")
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # Resume before W&B/run-dir so the new run logs into the original folder.
    resume_path, state = None, None
    if args.resume:
        resume_path = find_resume_ckpt(args.resume)
        state = restore(resume_path, model, opt, sched, scaler, args, device)
    if state and state["run_dir"]:
        run_dir = Path(state["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(args.run_name)

    run = setup_wandb(args)
    args.run_dir = str(run_dir)
    print("checkpoints ->", run_dir)
    best_top5 = state["best_top5"] if state else -1.0
    best_win_auc = state["best_win_auc"] if state else -1.0
    global_step = state["step"] if state else 0
    start_epoch = state["epoch"] if state else 0
    win_stalled = state.get("win_stalled", 0) if state else 0
    win_frozen = args.win_freeze or (state.get("win_frozen", False)
                                     if state else False)
    t0 = time.time()
    if state:
        print("resumed %s at epoch %d / step %d (best top5 %.4f, best win "
              "AUC %.4f)" % (resume_path, start_epoch, global_step,
                             best_top5, best_win_auc))
        if args.max_steps and global_step >= args.max_steps:
            print("--max-steps %d already reached; nothing to do"
                  % args.max_steps)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if args.max_steps and global_step >= args.max_steps:
            break
        tr_loader.dataset.set_epoch(epoch)
        model.train()
        if args.win_patience and win_stalled >= args.win_patience:
            win_frozen = True        # latch: never switch the value loss back on
        lam_win = win_weight(args, epoch, win_stalled, win_frozen)
        for i, batch in enumerate(tr_loader):
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(batch)
                loss, lc, lr = masked_loss(out, batch, args.lambda_role)
                lw = win_loss(out, batch)
                if not args.no_win:
                    loss = loss + lam_win * lw
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
                    if not args.no_win:
                        m["train/loss_win"] = float(lw)
                    wb_log(run, m, global_step)
                    print("ep %d step %d loss %.4f (c %.4f r %.4f%s) lr %.2e"
                          % (epoch, global_step, float(loss), float(lc),
                             float(lr),
                             "" if args.no_win else " w %.4f" % float(lw),
                             sched.get_last_lr()[0]))
                if args.max_steps and global_step >= args.max_steps:
                    break
        metrics = evaluate(model, va_loader, base, args.lambda_role, device)
        if clean_loader is not None:
            metrics.update(evaluate_value(model, clean_loader, device))
        metrics["epoch"] = epoch
        wb_log(run, {"val/" + k: v for k, v in metrics.items()}, global_step)
        print("val ep %d (lam_win %.2f): loss %.4f top1 %.3f top5 %.3f "
              "top10 %.3f rolecond@5 %.3f rolehid@1 %.3f role|champ@1 %.3f | "
              "base role@1 %.3f global@1 %.3f (%.0fs)"
              % (epoch, lam_win, metrics["loss"], metrics["champ_top1"],
                 metrics["champ_top5"], metrics["champ_top10"],
                 metrics.get("role_cond_top5", float("nan")),
                 metrics.get("role_hidden_top1", float("nan")),
                 metrics.get("role_given_champ_top1", float("nan")),
                 metrics["base_role_top1"], metrics["base_global_top1"],
                 time.time() - t0))
        if "win_auc" in metrics:
            print("        value: auc %.4f (const 0.5000) brier %.5f "
                  "(const %.5f) skill %+.4f logloss %.5f"
                  % (metrics["win_auc"], metrics["win_brier"],
                     metrics["win_brier_const"], metrics["win_skill"],
                     metrics["win_logloss"]))
        if metrics["champ_top5"] > best_top5:
            best_top5 = metrics["champ_top5"]
            save_ckpt(run_dir / "best.pt", model, args, num_champs,
                      metrics, global_step, epoch=epoch)
        if "win_auc" in metrics and metrics["win_auc"] > best_win_auc + 1e-6:
            best_win_auc = metrics["win_auc"]
            win_stalled = 0
            save_ckpt(run_dir / "best_value.pt", model, args, num_champs,
                      metrics, global_step, epoch=epoch)
        elif "win_auc" in metrics and not args.no_win and not win_frozen:
            win_stalled += 1
            if (args.win_patience and win_stalled == args.win_patience):
                print("value loss frozen: no val AUC gain for %d epochs "
                      "(best %.4f). The win head is memorising outcome noise -"
                      " train BCE keeps falling while val Brier rises - so the "
                      "encoder now trains on the masked task alone and "
                      "best_value.pt keeps the good head"
                      % (win_stalled, best_win_auc))
        save_ckpt(run_dir / "latest.pt", model, args, num_champs,
                  metrics, global_step, epoch=epoch,
                  resume={"opt": opt.state_dict(),
                          "sched": sched.state_dict(),
                          "scaler": scaler.state_dict(),
                          "step": global_step, "epoch": epoch,
                          "best_top5": best_top5,
                          "best_win_auc": best_win_auc,
                          "win_stalled": win_stalled,
                          "win_frozen": win_frozen,
                          "run_dir": str(run_dir),
                          "args": {"epochs": args.epochs,
                                   "batch_size": args.batch_size,
                                   "lr": args.lr,
                                   "lambda_win": args.lambda_win,
                                   "win_warmup_epochs": args.win_warmup_epochs,
                                   "embed_dim": args.embed_dim,
                                   "num_intra": args.num_intra,
                                   "num_cross": args.num_cross,
                                   "num_heads": args.num_heads,
                                   "patch": args.patch}})
        if args.max_steps and global_step >= args.max_steps:
            break
    print("done: best val champ top-5 %.4f (%s)" % (best_top5,
                                                     run_dir / "best.pt"))
    if best_win_auc >= 0:
        print("      best val win AUC %.4f (%s)" % (best_win_auc,
                                                    run_dir / "best_value.pt"))
    if run is not None:
        import wandb
        wandb.finish()
    return best_top5, best_win_auc


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
    best, best_win = train(args)
    out = HERE / "last_run.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"best_val_champ_top5": best,
                   "best_val_win_auc": (best_win if best_win >= 0 else None),
                   "args": vars(args)}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

