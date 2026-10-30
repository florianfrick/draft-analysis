"""Is the value head broken, or is the encoder just not ready yet?

Symptom this exists for: `win_auc` sits at 0.49-0.50 for many epochs and it is
unclear whether the head is defective or the representation is immature. Those
look identical in the training log and have opposite fixes.

Method: freeze the checkpoint's encoder, cache its slot vectors once, then
train ONLY a fresh PMA + win head on top (no encoder forward per step, so
hundreds of steps take seconds).

  learns (>= ~0.53)  -> head/readout fine; the win loss just needs to wait for
                        the encoder (see --win-warmup-epochs)
  stuck at 0.50      -> the head or the readout has a real defect

Measured on the 40-epoch oct19 encoder: 0.5396 after 702 steps, against 0.5230
for a linear probe on the same cached features. On a 700-step encoder the same
head reached only 0.497 - the encoder's own probe was 0.4996.

Usage
-----
  python pretraining/diagnose_head.py
  python pretraining/diagnose_head.py --ckpt ckpt/run_x/best.pt --epochs 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"
sys.path.insert(0, str(PROC_DIR))
sys.path.insert(0, str(HERE.parent))

import pipeline as P  # noqa: E402
from soloq_model import (PMAPool, SoloQPretrainedModel,  # noqa: E402
                         load_encoder_state, mean_pool_vector)


class ValueHead(nn.Module):
    """PMAPool + the same MLP as SoloQPretrainedModel.win_head."""

    def __init__(self, dim=512, seeds=4, heads=8):
        super().__init__()
        self.pma = PMAPool(dim, heads, seeds, 0.0)
        self.head = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim),
                                  nn.GELU(), nn.Linear(dim, 1))

    def forward(self, h, pad):
        h = h.float()  # cached as fp16 to halve RAM
        vb = self.pma(h[:, :5], pad[:, :5])
        vr = self.pma(h[:, 5:], pad[:, 5:])
        return self.head(torch.cat([vb, vr, vb - vr, vb * vr],
                                   dim=1)).squeeze(-1)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--patch", default="16.20")
    ap.add_argument("--fit-rows", type=int, default=30000)
    ap.add_argument("--eval-rows", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    return ap.parse_args(argv)


def newest_ckpt():
    c = sorted(HERE.glob("ckpt/run_*/best*.pt"), key=lambda p: p.stat().st_mtime)
    if not c:
        sys.exit("no checkpoint under pretraining/ckpt/run_*/best*.pt")
    return c[-1]


@torch.no_grad()
def cache_h(enc, rows, wins, dev):
    ds = P.SoloQDraftDataset(rows, num_champs=enc.num_champs, mask_prob=0.0,
                             role_only_prob=0.0, seed=0, wins=wins,
                             force_one=False)
    dl = DataLoader(ds, batch_size=512, shuffle=False,
                    collate_fn=P.soloq_collate)
    hs, ys = [], []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        hs.append(enc._encode(b).half().cpu())
        ys.append(b["win_label"].cpu())
    return torch.cat(hs), torch.cat(ys)


def main():
    args = parse_args()
    dev = torch.device(args.device)
    ckpt = Path(args.ckpt) if args.ckpt else newest_ckpt()
    print("checkpoint:", ckpt)

    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=0.0).to(dev)
    load_encoder_state(enc, payload["model"])
    enc.eval()

    cache = PROC_DIR / ("drafts_%s.pt" % args.patch)
    p = torch.load(str(cache), map_location="cpu")
    if p.get("wins") is None:
        sys.exit("cache has no win labels; rerun pretraining/train.py")
    rows, wins = p["rows"], p["wins"]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(rows), generator=g).tolist()
    n_val = int(len(rows) * 0.05)
    tr = perm[n_val:][:args.fit_rows]
    va = perm[:n_val][:args.eval_rows]

    print("caching encoder outputs...", flush=True)
    htr, ytr = cache_h(enc, [rows[i] for i in tr], [wins[i] for i in tr], dev)
    hva, yva = cache_h(enc, [rows[i] for i in va], [wins[i] for i in va], dev)
    print("train %s  val %s  base rate %.4f"
          % (tuple(htr.shape), tuple(hva.shape), float(ytr.float().mean())))

    pad_tr = torch.zeros(htr.shape[0], 10, dtype=torch.bool)
    pad_va = torch.zeros(hva.shape[0], 10, dtype=torch.bool)
    zm = mean_pool_vector(htr.float(), pad_tr).numpy()
    zme = mean_pool_vector(hva.float(), pad_va).numpy()
    sc = StandardScaler().fit(zm)
    lr = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(zm), ytr.numpy())
    ref = roc_auc_score(yva.numpy(), lr.predict_proba(sc.transform(zme))[:, 1])
    print("reference: linear probe on cached h  AUC %.4f" % ref)

    torch.manual_seed(0)
    net = ValueHead(dim=cfg["embed_dim"], seeds=cfg.get("value_seeds", 4),
                    heads=cfg["num_heads"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    htr_d, hva_d = htr.to(dev), hva.to(dev)
    pad_tr_d, pad_va_d = pad_tr.to(dev), pad_va.to(dev)
    ytr_d, yva_f = ytr.float().to(dev), yva.float().numpy()
    B, n, step = args.batch_size, htr.shape[0], 0
    best = 0.5
    for epoch in range(1, args.epochs + 1):
        perm_e = torch.randperm(n)
        tot = 0.0
        for i in range(0, n - B + 1, B):
            idx = perm_e[i:i + B]
            z = net(htr_d[idx], pad_tr_d[idx])
            loss = F.binary_cross_entropy_with_logits(z, ytr_d[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            step += 1
        with torch.no_grad():
            zv = net(hva_d, pad_va_d).float().cpu().numpy()
        a = roc_auc_score(yva_f, zv)
        best = max(best, a)
        print("  epoch %d step %4d  train_bce %.4f  val_auc %.4f  brier %.5f"
              % (epoch, step, tot / max(1, n // B), a,
                 float(np.mean((1 / (1 + np.exp(-zv)) - yva_f) ** 2))),
              flush=True)

    print("\nbest val AUC %.4f (probe %.4f)" % (best, ref))
    # Three cases, not two: a head that fails on a representation that carries
    # no signal tells you nothing about the head.
    if ref < 0.515:
        print("VERDICT: the encoder itself carries no outcome signal (probe "
              "%.4f). Do not blame the head - this checkpoint is immature; "
              "train longer and re-probe." % ref)
    elif best >= 0.53:
        print("VERDICT: head learns from this encoder -> if win_auc is flat in "
              "training, the encoder was not mature yet (raise "
              "--win-warmup-epochs)")
    else:
        print("VERDICT: representation carries signal (probe %.4f) but the "
              "head still cannot learn -> the value readout/head has a "
              "defect, not a scheduling problem" % ref)


if __name__ == "__main__":
    main()
