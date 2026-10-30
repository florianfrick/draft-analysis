"""P0 gate: does the encoder's pooled representation carry win signal at all?

A linear probe on frozen pooled features separates "the readout/head is wrong"
from "the representation is wrong", and it costs one pass over the data:

  probe AUC ~0.50  -> the pooled vector does not linearly separate winners.
                      No head or loss weight will fix that; the representation
                      has to change (which is what the win head in pretraining
                      is for).
  probe AUC >= 0.53 -> the signal is there and the head/training was at fault.

Reference points on this corpus: a 173-parameter logistic regression on
champion identity alone reaches 0.5507 on pro games and 0.5449 on soloQ, so a
probe below ~0.52 means the encoder is throwing away what a bag-of-champions
keeps. A constant predictor scores AUC 0.5000 / Brier 0.2482.

Two readouts are probed on the same checkpoint, because that is the one
comparison that says whether PMA was worth its parameters:
  mean  - the old readout, concat(vb, vr, vb-vr, vb*vr) with PAD-masked means
  pma   - the new readout, learned seed queries over each team's slots

Usage
-----
  python pretraining/probe_value.py                       # newest ckpt
  python pretraining/probe_value.py --ckpt ckpt/run_x/best.pt
  python pretraining/probe_value.py --ckpt ckpt/run_x/best.pt --fit-rows 80000

Works on old checkpoints too (no value path): only the mean probe is reported.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"
sys.path.insert(0, str(PROC_DIR))
sys.path.insert(0, str(HERE.parent))

import pipeline as P  # noqa: E402
from soloq_model import (SoloQPretrainedModel, mean_pool_vector,  # noqa: E402
                         load_encoder_state)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Linear probe of the value repr")
    ap.add_argument("--ckpt", default="",
                    help="checkpoint path (default: newest */best.pt)")
    ap.add_argument("--patch", default="16.20")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--fit-rows", type=int, default=60000,
                    help="train rows used to fit the probe (0 = all)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    return ap.parse_args(argv)


def newest_ckpt():
    cands = sorted(HERE.glob("ckpt/run_*/best*.pt"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        sys.exit("no checkpoint under pretraining/ckpt/run_*/best*.pt")
    return cands[-1]


def load_payload(path):
    # weights_only=False: latest.pt also carries optimizer/scheduler/scaler
    # state for --resume; these are our own files.
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model" not in payload:
        sys.exit("%s is not a pretraining checkpoint" % path)
    return payload


def build_encoder(payload, device):
    cfg = payload["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"],
        dropout=cfg.get("dropout", 0.0),
        with_value=bool(cfg.get("with_value", False)),
        value_seeds=cfg.get("value_seeds", 4))
    load_encoder_state(enc, payload["model"])
    return enc.to(device), cfg


def load_rows(patch):
    cache = PROC_DIR / ("drafts_%s.pt" % patch)
    if not cache.exists():
        sys.exit("no cache at %s - run pretraining/train.py once first" % cache)
    payload = torch.load(str(cache), map_location="cpu")
    if payload.get("wins") is None:
        sys.exit("cache has no win labels; rerun pretraining/train.py "
                 "(it rescans automatically) or pass --rescan there")
    return payload["rows"], payload["wins"]


def split(n, val_frac, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


@torch.no_grad()
def collect(enc, rows, wins, device, batch_size, want_pma):
    """Cache (z_mean, z_pma?, win_label) over a row subset."""
    ds = P.SoloQDraftDataset(rows, num_champs=enc.num_champs, mask_prob=0.0,
                             role_only_prob=0.0, seed=0, wins=wins,
                             force_one=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=P.soloq_collate)
    enc.eval()
    z_mean, z_pma, labels = [], [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        h = enc._encode(batch)
        pad_mask = batch["pad_mask"]
        z_mean.append(mean_pool_vector(h, pad_mask).float().cpu())
        if want_pma:
            z_pma.append(enc.value_vector(batch, h).float().cpu())
        labels.append(batch["win_label"].cpu())
    return (torch.cat(z_mean).numpy(),
            torch.cat(z_pma).numpy() if want_pma else None,
            torch.cat(labels).numpy())


def probe(z_fit, y_fit, z_ev, y_ev, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(z_fit)
    clf = LogisticRegression(C=0.1, max_iter=400)
    clf.fit(sc.transform(z_fit), y_fit)
    p = clf.predict_proba(sc.transform(z_ev))[:, 1]
    auc = float(roc_auc_score(y_ev, p))
    brier = float(np.mean((p - y_ev) ** 2))
    base = float(y_ev.mean())
    print("  %-6s dim=%-5d AUC %.4f  Brier %.5f (const %.5f)  logloss %.5f"
          % (tag, z_fit.shape[1], auc, brier, base * (1 - base),
             float(log_loss(y_ev, p))))
    return {"readout": tag, "dim": int(z_fit.shape[1]), "auc": auc,
            "brier": brier, "brier_const": base * (1 - base)}


def main():
    args = parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else newest_ckpt()
    print("checkpoint:", ckpt)
    payload = load_payload(ckpt)
    device = torch.device(args.device)
    enc, cfg = build_encoder(payload, device)
    print("with_value:", enc.with_value, " embed_dim:", cfg["embed_dim"])

    rows, wins = load_rows(args.patch)
    tr_idx, va_idx = split(len(rows), args.val_frac, args.data_seed)
    if args.fit_rows and len(tr_idx) > args.fit_rows:
        g = np.random.default_rng(args.seed)
        tr_idx = [tr_idx[i] for i in
                  g.choice(len(tr_idx), args.fit_rows, replace=False)]
    print("probe fit rows %d / eval rows %d" % (len(tr_idx), len(va_idx)))

    want_pma = enc.with_value
    zm_f, zp_f, y_f = collect(enc, [rows[i] for i in tr_idx],
                              [wins[i] for i in tr_idx], device,
                              args.batch_size, want_pma)
    zm_e, zp_e, y_e = collect(enc, [rows[i] for i in va_idx],
                              [wins[i] for i in va_idx], device,
                              args.batch_size, want_pma)
    print("eval base rate %.4f over %d drafts" % (y_e.mean(), len(y_e)))

    out = {"ckpt": str(ckpt), "eval_base_rate": float(y_e.mean()),
           "probes": [probe(zm_f, y_f, zm_e, y_e, "mean")]}
    if want_pma:
        out["probes"].append(probe(zp_f, y_f, zp_e, y_e, "pma"))

    # the trained head, for comparison with the probes on the same rows
    if enc.with_value:
        with torch.no_grad():
            ds = P.SoloQDraftDataset([rows[i] for i in va_idx],
                                     num_champs=enc.num_champs, mask_prob=0.0,
                                     role_only_prob=0.0, seed=0,
                                     wins=[wins[i] for i in va_idx],
                                     force_one=False)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=P.soloq_collate)
            zs, ys = [], []
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                zs.append(enc.forward_value(batch).squeeze(-1).float().cpu())
                ys.append(batch["win_label"].float().cpu())
        from sklearn.metrics import roc_auc_score
        z = torch.cat(zs).numpy()
        y = torch.cat(ys).numpy()
        out["trained_head_auc"] = float(roc_auc_score(y, z))
        print("  trained head AUC %.4f" % out["trained_head_auc"])

    best = max(p["auc"] for p in out["probes"])
    verdict = ("representation carries outcome signal - head/training was the "
               "bottleneck" if best >= 0.53 else
               "no outcome signal in the pooled vector - representation must "
               "change (win head / PMA), not the loss weight")
    out["verdict"] = verdict
    print("\nVERDICT:", verdict)
    dst = HERE / "probe_value.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
