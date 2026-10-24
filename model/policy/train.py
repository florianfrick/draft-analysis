"""Policy-value training (Stage 2): next-move prediction + win value.

Trains the PolicyValueModel on ProNextMoveDataset (20 examples/game
from the 46.6k pro drafts). The SoloQ encoder is frozen; only the
ban-context module, state embedding, policy head, and value head
train from scratch.

Loss per example (weighted by recency `weight`):
  total = policy_CE(next_move) + value_w * BCE(blue_win) + econ_w * MSE(econ)

The policy CE is masked: logits of taken/banned/unavailable champs are
-inf (enforced by the model), so the softmax only normalizes over legal
moves. Value is blue-win BCE (side-neutral: red queries mirror slots).

Usage:
    python policy/train.py --epochs 15                      # full run
    python policy/train.py --limit-drafts 2000 --epochs 2 \\
        --no-wandb --run-name smoke                         # smoke

Stages:
  0 = heads + ban_context + state_embed only (encoder frozen)
  1 = + unfreeze encoder cross_team (optional, --unfreeze-stage 1)
  2 = + unfreeze encoder intra_team (optional, --unfreeze-stage 2)
  Embeddings stay frozen throughout (they carry soloQ vocab semantics).
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

HERE = Path(__file__).resolve().parent            # .../model/policy
MODEL_DIR = HERE.parent                           # .../model
ROOT = MODEL_DIR.parent                           # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"
SOLOQ_PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"
PRETRAIN_CKPT_GLOB = str(MODEL_DIR / "pretraining" / "ckpt" /
                         "run_*" / "best.pt")
DEFAULT_INIT = str(MODEL_DIR / "pretraining" / "ckpt" /
                   "run_20261019_141424_oct19-medium-with-new-role-given-champ-mask" / "best.pt")
KEY_FILE = MODEL_DIR / "pretraining" / "wandb_key.txt"
IGNORE_INDEX = -100
N_MOVES = 20


def _load_module_by_path(name, path):
    spec = __import__("importlib").util.spec_from_file_location(
        name, str(path))
    mod = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(MODEL_DIR))
from soloq_model import SoloQPretrainedModel  # noqa: E402
from model import PolicyValueModel  # noqa: E402

PRO_PIPE = _load_module_by_path("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load_module_by_path("pro_datasets", PRO_DIR / "datasets.py")
SQ_PIPE = _load_module_by_path("soloq_pipeline",
                               SOLOQ_PROC_DIR / "pipeline.py")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Policy-value Stage 2")
    # data
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--rehearse-frac", type=float, default=0.0,
                    help="share of soloQ masked batches (0=off; encoder "
                         "is frozen so rehearsal is optional)")
    # model/init
    ap.add_argument("--init-ckpt", default=DEFAULT_INIT,
                    help="soloQ checkpoint path or 'auto'")
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--head-dropout", type=float, default=0.0)
    ap.add_argument("--n-econ", type=int, default=6)
    ap.add_argument("--no-patch", action="store_true",
                    help="disable patch conditioning (ablation; default on)")
    # data sampling / augmentation
    ap.add_argument("--prefix-mode", choices=("random", "all"),
                    default="random",
                    help="random = 1 prefix per game per epoch (default, "
                         "20x cheaper epoch and no near-duplicate repeats); "
                         "all = 20 examples/game (old behaviour)")
    ap.add_argument("--swap-aug", dest="swap_aug", action="store_true",
                    default=True,
                    help="mirror blue<->red with p=0.5 (default on)")
    ap.add_argument("--no-swap-aug", dest="swap_aug", action="store_false",
                    help="disable blue/red mirror augmentation")
    # losses
    ap.add_argument("--value-weight", type=float, default=0.5,
                   help="weight on win BCE (lower = focus on policy)")
    ap.add_argument("--econ-weight", type=float, default=0.1)
    # optimisation
    ap.add_argument("--epochs", type=int, default=100,
                    help="passes over the game list. With --prefix-mode "
                         "random an epoch is 20x cheaper than the old "
                         "20-examples/game epoch, so 100 epochs is roughly "
                         "the optimizer-step budget of a 5-epoch run of "
                         "--prefix-mode all")
    ap.add_argument("--unfreeze-stage", type=int, default=0,
                    help="0=encoder frozen (default), 1=+cross, 2=+intra")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lol-draft-policy")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="epochs between val evals. Random prefix mode has "
                         "20x more epochs than the old mode, so evaluating "
                         "every epoch would cost more than training")
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
        except Exception:  # noqa: BLE001
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
    """Patch 16.20 soloQ rows for rehearsal (optional)."""
    payload = torch.load(str(SOLOQ_PROC_DIR / "drafts_16.20.pt"),
                         map_location="cpu", weights_only=False)
    return payload["rows"], payload["patch"]


def fit_econ_stats(rows, indices, n_econ=6):
    """Mean/std of raw econ targets on train rows with has_econ."""
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


def policy_loss(out, batch):
    """CE on next-move champion, with -inf legality masking already applied.

    Handles the edge case where the target champion was masked (shouldn't
    happen after the unavailable-on-bans fix, but guards against NaN).
    """
    cl = out["policy_logits"]
    tc = batch["target_champ"]
    # Safety: if any target logit is -inf/nan, replace with a large
    # negative so CE is finite (the model still gets penalized).
    tgt_logits = cl.gather(1, tc[:, None]).squeeze(1)
    bad = ~torch.isfinite(tgt_logits)
    if bool(bad.any()):
        cl = cl.clone()
        cl[bad.unsqueeze(1).expand_as(cl) & (cl < 0)] = -1e4
        cl[bad, tc[bad]] = 0.0  # give target a neutral logit
    return F.cross_entropy(cl, tc)


def value_loss(out, batch):
    """BCE on blue-win logit."""
    if batch["win_label"].numel() == 0:
        return out["value_logit"].sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        out["value_logit"].squeeze(-1), batch["win_label"].float())


def econ_loss(out, batch, econ_mean, econ_std):
    """MSE on standardized econ, masked to has_econ rows."""
    if "econ_target" not in batch or "has_econ" not in batch:
        return out.get("econ_pred", torch.zeros(1)).sum() * 0.0
    if int(batch["has_econ"].sum()) == 0:
        return out["econ_pred"].sum() * 0.0
    tgt = (batch["econ_target"] - econ_mean) / econ_std
    err = (out["econ_pred"] - tgt) ** 2
    w = batch["has_econ"].unsqueeze(1)
    return (err * w).sum() / (w.sum() * err.shape[1])


def batch_auc(logits, labels):
    """AUC via rank statistics; 0.5 on degenerate."""
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
def evaluate(model, loaders, args, device, econ_mean, econ_std):
    """Val: next-move top-k accuracy, win AUC, econ RMSE, popularity baselines."""
    model.eval()
    # Next-move metrics
    top = {1: 0.0, 5: 0.0, 10: 0.0}
    ban_top = {1: 0.0, 5: 0.0}
    pick_top = {1: 0.0, 5: 0.0}
    n_total = 0
    n_ban = 0
    n_pick = 0
    loss_sum = 0.0
    # Value metrics
    val_logits, val_labels = [], []
    econ_err, econ_n = 0.0, 0

    for batch in loaders["val"]:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        cl = out["policy_logits"]
        tc = batch["target_champ"]
        # Rank of truth
        truth_logit = cl.gather(1, tc[:, None]).squeeze(1)
        ranks = (cl > truth_logit[:, None]).sum(1)  # rank of truth, 0=top-1
        is_ban = batch["is_ban"].bool()

        n = tc.numel()
        for k in top:
            top[k] += float((ranks < k).sum())
        # Split by ban vs pick
        if bool(is_ban.any()):
            for k in ban_top:
                ban_top[k] += float((ranks[is_ban] < k).sum())
            n_ban += int(is_ban.sum())
        if bool((~is_ban).any()):
            for k in pick_top:
                pick_top[k] += float((ranks[~is_ban] < k).sum())
            n_pick += int((~is_ban).sum())

        loss_sum += float(policy_loss(out, batch)) * n
        n_total += n

        val_logits.append(out["value_logit"].squeeze(-1).float().cpu())
        val_labels.append(batch["win_label"].cpu())
        # build_prefix now always emits econ_target/has_econ (the econ
        # head used to have no labels at all), so gate on the head
        # existing too, or n_econ=0 runs would KeyError here.
        if "econ_pred" in out and "has_econ" in batch \
                and int(batch["has_econ"].sum()):
            tgt = (batch["econ_target"] - econ_mean) / \
                econ_std.to(batch["econ_target"].device)
            err = (out["econ_pred"] - tgt) ** 2
            econ_err += float(err.sum())
            econ_n += int(batch["has_econ"].sum()) * err.shape[1]

    out_m = {}
    out_m["policy_loss"] = loss_sum / max(1, n_total)
    for k in top:
        out_m["top%d" % k] = top[k] / max(1, n_total)
    out_m["ban_top1"] = ban_top[1] / max(1, n_ban)
    out_m["ban_top5"] = ban_top[5] / max(1, n_ban)
    out_m["pick_top1"] = pick_top[1] / max(1, n_pick)
    out_m["pick_top5"] = pick_top[5] / max(1, n_pick)
    out_m["n_total"] = n_total
    out_m["n_ban"] = n_ban
    out_m["n_pick"] = n_pick

    # Value metrics
    val_logits = torch.cat(val_logits)
    val_labels = torch.cat(val_labels)
    out_m["win_auc"] = batch_auc(val_logits, val_labels)
    out_m["win_brier"] = float(
        ((torch.sigmoid(val_logits) - val_labels.float()) ** 2).mean())
    out_m["win_base_rate"] = float(val_labels.float().mean())
    out_m["econ_rmse"] = (econ_err / max(1, econ_n)) ** 0.5
    out_m["n_win"] = int(val_labels.numel())

    return out_m


def compute_move_popularity(rows, indices, num_champs):
    """Popularity baseline: for each move position (0..19), the
    weighted-frequency rank of champions at that move position."""
    from collections import Counter
    move_counts = [Counter() for _ in range(N_MOVES)]
    for i in indices:
        row = rows[i]
        w = float(row.get("weight", 1.0))
        for m_idx, (kind, team, champ, role) in enumerate(row["moves"]):
            move_counts[m_idx][int(champ)] += w

    move_rank = []
    for m_idx in range(N_MOVES):
        c = move_counts[m_idx]
        order = sorted(range(num_champs),
                      key=lambda x: (-float(c.get(x, 0.0)), x))
        rank = torch.empty(num_champs, dtype=torch.long)
        rank[torch.tensor(order)] = torch.arange(num_champs)
        move_rank.append(rank)
    return move_rank, move_counts


@torch.no_grad()
def evaluate_with_baseline(model, loader, rows, train_indices,
                           num_champs, args, device):
    """Extended eval with popularity baseline comparison."""
    model.eval()
    move_rank, move_counts = compute_move_popularity(
        rows, train_indices, num_champs)
    move_rank = [r.to(device) for r in move_rank]

    top = {1: 0.0, 5: 0.0, 10: 0.0}
    base_top = {1: 0.0, 5: 0.0, 10: 0.0}
    n_total = 0

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        cl = out["policy_logits"]
        tc = batch["target_champ"]
        truth_logit = cl.gather(1, tc[:, None]).squeeze(1)
        ranks = (cl > truth_logit[:, None]).sum(1)

        for i in range(tc.numel()):
            m_idx = int(batch["move_idx"][i])
            br = move_rank[m_idx][tc[i]].item()
            for k in top:
                if int(ranks[i]) < k:
                    top[k] += 1
            for k in base_top:
                if br < k:
                    base_top[k] += 1
            n_total += 1

    out_m = {}
    for k in top:
        out_m["top%d" % k] = top[k] / max(1, n_total)
        out_m["base_top%d" % k] = base_top[k] / max(1, n_total)
    out_m["n_total"] = n_total
    return out_m


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
    except Exception as exc:  # noqa: BLE001
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
                          "state_dim": args.state_dim,
                          "n_econ": args.n_econ,
                          "head_dropout": args.head_dropout,
                          # Needed to rebuild the model at inference:
                          # sizes state_embed.patch_embed and maps a
                          # patch string to its row.
                          "n_patches": extra.get("n_patches", 0),
                          "patch_vocab": extra.get("patch_vocab")},
               "metrics": metrics, "step": step,
               "init_ckpt": extra["init_ckpt"],
               "unfreeze_stage": args.unfreeze_stage,
               "patch": extra.get("patch"),
               "econ_mean": extra.get("econ_mean"),
               "econ_std": extra.get("econ_std")}
    torch.save(payload, str(path))


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"

    # Load encoder
    ckpt_spec, payload = pick_init_ckpt(args.init_ckpt)
    print("init encoder <- %s (val top5 %.4f)" %
          (ckpt_spec, float(payload.get("metrics", {})
                            .get("champ_top5", -1))))
    encoder, cfg = build_encoder(payload)
    embed_dim = cfg["embed_dim"]

    # Data. Loaded before the model so the patch vocabulary can size the
    # patch embedding.
    rows, splits = load_pro_rows(args)
    print("pro rows %d (train %d / val %d / test %d)" %
          (len(rows), len(splits["train"]), len(splits["val"]),
           len(splits["test"])))

    # Patch vocabulary over all rows (metadata, not labels - see
    # build_patch_vocab). Index 0 is UNK so the next patch at inference
    # time maps to a learned fallback instead of crashing.
    patch_vocab, patch_index = PRO_DATA.build_patch_vocab(rows)
    if args.no_patch:
        n_patches = 0
        print("patch conditioning: OFF (ablation)")
    else:
        n_patches = len(patch_vocab) + 1
        print("patch conditioning: ON (%d patches %s..%s + UNK)"
              % (len(patch_vocab), patch_vocab[0], patch_vocab[-1]))

    model = PolicyValueModel(
        encoder, embed_dim=embed_dim, state_dim=args.state_dim,
        n_econ=args.n_econ, dropout=args.head_dropout,
        n_patches=n_patches,
    ).to(device)

    # Freeze encoder per --unfreeze-stage
    if args.unfreeze_stage == 0:
        model.set_freeze_encoder(True)
    else:
        model.set_freeze_encoder(False)
        # Partial unfreeze: freeze embeddings always
        for p in model.encoder.embed.parameters():
            p.requires_grad = False
        if args.unfreeze_stage == 1:
            for p in model.encoder.intra_team.parameters():
                p.requires_grad = False
        # stage 2: everything except embeddings trains

    econ_mean, econ_std = torch.zeros(args.n_econ), torch.ones(args.n_econ)
    if args.n_econ > 0:
        econ_mean, econ_std = fit_econ_stats(
            rows, splits["train"], args.n_econ)
    emu, esd = econ_mean.to(device), econ_std.to(device)

    # Train dataset: one prefix per game per epoch + blue/red mirror.
    # Val/test: every prefix, no augmentation, so metrics stay
    # comparable to earlier runs (62,880 test examples).
    tr_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["train"], num_champs=cfg["num_champs"],
        patch_index=patch_index, prefix_mode=args.prefix_mode,
        swap_aug=args.swap_aug, seed=args.seed)
    va_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["val"], num_champs=cfg["num_champs"],
        patch_index=patch_index, prefix_mode="all", swap_aug=False)
    te_ds = PRO_DATA.ProNextMoveDataset(
        rows, splits["test"], num_champs=cfg["num_champs"],
        patch_index=patch_index, prefix_mode="all", swap_aug=False)
    print("train examples/epoch %d (prefix_mode=%s, swap_aug=%s)"
          % (len(tr_ds), args.prefix_mode, args.swap_aug))

    # Weighted sampler for recency. `weights()` is aligned with
    # __len__, so in random-prefix mode it is one weight per game.
    train_weights = tr_ds.weights()
    if max(train_weights) - min(train_weights) > 1e-6:
        from torch.utils.data import WeightedRandomSampler
        g = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            train_weights, num_samples=len(tr_ds),
            replacement=True, generator=g)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                              sampler=sampler,
                              collate_fn=PRO_DATA.pro_collate,
                              num_workers=0,
                              pin_memory=torch.cuda.is_available())
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=PRO_DATA.pro_collate,
                              num_workers=0,
                              pin_memory=torch.cuda.is_available())

    va_loader = DataLoader(va_ds, batch_size=args.batch_size,
                           shuffle=False,
                           collate_fn=PRO_DATA.pro_collate,
                           num_workers=0,
                           pin_memory=torch.cuda.is_available())
    te_loader = DataLoader(te_ds, batch_size=args.batch_size,
                           shuffle=False,
                           collate_fn=PRO_DATA.pro_collate,
                           num_workers=0,
                           pin_memory=torch.cuda.is_available())

    # Optional soloQ rehearsal
    sq_epoch, sq_it = 0, None
    if args.rehearse_frac > 0:
        sq_rows, sq_patch = load_soloq_rows()
        print("rehearsal pool: %d soloQ drafts (%s)" % (len(sq_rows), sq_patch))

        def sq_loader(epoch):
            ds = SQ_PIPE.SoloQDraftDataset(sq_rows,
                                           num_champs=cfg["num_champs"],
                                           seed=args.seed + 200, epoch=epoch)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=SQ_PIPE.soloq_collate,
                              num_workers=0,
                              pin_memory=torch.cuda.is_available())
    else:
        sq_rows = None

    # Optimizer: two param groups -- heads at --lr, encoder at --lr-enc.
    #
    # Split by object identity. The previous filter used
    # `p is not model.encoder`, but a Parameter is never identical to
    # the encoder Module, so that test was always True and the encoder
    # landed in BOTH groups. It only stayed hidden at --unfreeze-stage 0
    # because frozen encoder params have requires_grad=False and were
    # filtered out; any stage > 0 raised "some parameters appear in more
    # than one parameter group".
    #
    # The encoder's own champ_head/role_head are dead in this model
    # (forward only calls encoder._encode), so keep them out of the
    # encoder group rather than feeding the optimizer unused weights.
    enc_owned = list(model.encoder.parameters())
    dead_ids = set()
    for _name in ("champ_head", "role_head"):
        _sub = getattr(model.encoder, _name, None)
        if _sub is not None:
            dead_ids.update(id(p) for p in _sub.parameters())
    enc_ids = {id(p) for p in enc_owned}

    head_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in enc_ids]
    enc_params = [p for p in enc_owned
                  if p.requires_grad and id(p) not in dead_ids]

    # Invariant: groups must be disjoint and cover every trainable
    # parameter (except the deliberately unused encoder heads). This is
    # the check that would have caught the duplicate-group bug.
    grouped = head_params + enc_params
    _ids = [id(p) for p in grouped]
    assert len(_ids) == len(set(_ids)), \
        "parameter appears in more than one optimizer group"
    _trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert not (_trainable - set(_ids) - dead_ids), \
        "trainable parameter missing from optimizer"

    groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_enc})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    print("optimizer: %d head params @ lr %.1e, %d encoder params @ lr %.1e"
          % (len(head_params), args.lr, len(enc_params), args.lr_enc))

    total_steps = len(tr_loader) * args.epochs
    warm = max(1, int(total_steps * args.warmup_frac))
    print("schedule: %d steps/epoch x %d epochs = %d steps (warmup %d)"
          % (len(tr_loader), args.epochs, total_steps, warm))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    run = run_wandb(args)
    run_dir = HERE / "ckpt" / ("run_" + time.strftime("%Y%m%d_%H%M%S") +
                               (("_" + args.run_name) if args.run_name else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    print("checkpoints ->", run_dir)

    global_step, best_top5, best_auc = 0, -1.0, -1.0
    t0 = time.time()
    rehearse_gen = torch.Generator().manual_seed(args.seed + 999)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Redraw which prefix each game contributes (random mode only).
        # Must happen before the loader is iterated.
        tr_ds.set_epoch(epoch)
        for i, batch in enumerate(tr_loader):
            batch = {k: v.to(device, non_blocking=True)
                     if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(batch)
                p_loss = policy_loss(out, batch)
                v_loss = value_loss(out, batch)
                loss = p_loss + args.value_weight * v_loss
                if args.n_econ > 0 and "econ_pred" in out:
                    e_loss = econ_loss(out, batch, emu, esd)
                    loss = loss + args.econ_weight * e_loss

            # Optional rehearsal (encoder must be unfrozen for it to matter)
            if args.rehearse_frac > 0 and not model._encoder_frozen and \
                    sq_rows is not None and \
                    (torch.rand((), generator=rehearse_gen).item()
                     < args.rehearse_frac):
                try:
                    rb = next(sq_it)
                except (StopIteration, TypeError):
                    sq_epoch += 1
                    sq_it = iter(sq_loader(sq_epoch))
                    rb = next(sq_it)
                rb = {k: v.to(device, non_blocking=True)
                      for k, v in rb.items()}
                with torch.cuda.amp.autocast(enabled=amp):
                    rout = model.encoder(rb)
                    from soloq_model import NUM_REAL_CHAMPS
                    m = rb["is_masked"]
                    if bool(m.any()):
                        cl = F.cross_entropy(
                            rout["champ_logits"][m],
                            rb["champ_labels"][m])
                        rm = rb["role_labels"] != IGNORE_INDEX
                        lr_ = F.cross_entropy(
                            rout["role_logits"][rm],
                            rb["role_labels"][rm])
                        loss = loss + cl + 0.3 * lr_

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "non-finite loss at epoch %d step %d"
                    % (epoch, global_step))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            if global_step % args.log_every == 0:
                msg = {"train/loss": float(loss),
                       "train/policy_loss": float(p_loss),
                       "train/value_loss": float(v_loss),
                       "train/lr": sched.get_last_lr()[0]}
                if run is not None:
                    import wandb
                    wandb.log(msg, step=global_step)
                print("ep %d step %d loss %.4f (p %.4f v %.4f) lr %.2e" %
                      (epoch, global_step, float(loss), float(p_loss),
                       float(v_loss), sched.get_last_lr()[0]))

        if epoch % args.eval_every == 0:
            metrics = evaluate(model, {"val": va_loader}, args,
                               device, emu, esd)
            metrics["epoch"] = epoch
            if run is not None:
                import wandb
                wandb.log({"val/" + k: v for k, v in metrics.items()},
                          step=global_step)
            print("val ep %d: top1 %.4f top5 %.4f top10 %.4f "
                  "ban@5 %.4f pick@5 %.4f win_auc %.4f brier %.4f "
                  "econ_rmse %.4f (%.0fs)" %
                  (epoch, metrics["top1"], metrics["top5"],
                   metrics["top10"], metrics["ban_top5"],
                   metrics["pick_top5"], metrics["win_auc"],
                   metrics["win_brier"], metrics["econ_rmse"],
                   time.time() - t0))

            extra = {"init_ckpt": ckpt_spec,
                     "patch": payload.get("patch"),
                     "econ_mean": econ_mean, "econ_std": econ_std,
                     "n_patches": n_patches, "patch_vocab": patch_vocab}
            if metrics["top5"] > best_top5:
                best_top5 = metrics["top5"]
                save_ckpt(run_dir / "best_policy.pt", model, args, cfg,
                          metrics, global_step, extra)
            if metrics["win_auc"] > best_auc:
                best_auc = metrics["win_auc"]
                save_ckpt(run_dir / "best_value.pt", model, args, cfg,
                          metrics, global_step, extra)
            save_ckpt(run_dir / "latest.pt", model, args, cfg,
                      dict(metrics), global_step, extra)

    # Test report
    report = {}
    for name in ("best_policy.pt", "best_value.pt"):
        path = run_dir / name
        if not path.exists():
            continue
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        tm = evaluate(model, {"val": te_loader}, args, device, emu, esd)
        report[name] = tm
        print("TEST %s: top1 %.4f top5 %.4f win_auc %.4f brier %.4f "
              "econ_rmse %.4f" % (name, tm["top1"], tm["top5"],
                                  tm["win_auc"], tm["win_brier"],
                                  tm["econ_rmse"]))
    with open(run_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("done: best policy top5 %.4f, best win AUC %.4f (%s)"
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
        json.dump({"best_val_top5": best_top5,
                   "best_val_win_auc": best_auc,
                   "args": vars(args)}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
