"""Paired top-k comparison between two policy checkpoints.

Aggregate test top-5 is noisy because the 62,880 test examples are only
3,144 games (20 correlated moves each), so an unpaired +/-1pp difference
is inside the noise band. This computes the paired discordant counts
instead: on how many examples does A get the truth in top-5 while B
misses it, and vice versa. That is a McNemar test on the same examples.

Also prints the popularity baseline so the model can be read relative to
a lookup table rather than to zero.

Usage:
    python compare_policy.py --a <ckpt> --b <ckpt>
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PRO_DIR = HERE.parent.parent / "data" / "pro play"
sys.path.insert(0, str(HERE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")

import train as T  # noqa: E402
from model import PolicyValueModel  # noqa: E402

N_MOVES = PRO_DATA.N_MOVES
PHASE_OF = PRO_DATA.PHASE_OF
PHASE_NAMES = ["bans1", "picks1", "bans2", "picks2"]


def load_model(path, device):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    enc = T.SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=0.0)
    m = PolicyValueModel(enc, embed_dim=cfg["embed_dim"],
                         state_dim=cfg.get("state_dim", 64),
                         n_econ=cfg.get("n_econ", 0), dropout=0.0,
                         n_patches=cfg.get("n_patches", 0))
    m.load_state_dict(payload["model"])
    return m.to(device).eval(), payload


def build_popularity(rows, indices, num_champs):
    counts = [Counter() for _ in range(N_MOVES)]
    for i in indices:
        row = rows[i]
        w = float(row.get("weight", 1.0))
        for m_idx, (_k, _t, champ, _r) in enumerate(row["moves"]):
            counts[m_idx][int(champ)] += w
    ranks, tops = [], []
    for m_idx in range(N_MOVES):
        c = counts[m_idx]
        order = sorted(range(num_champs), key=lambda x: (-float(c.get(x, 0.0)), x))
        rank = torch.empty(num_champs, dtype=torch.long)
        rank[torch.tensor(order)] = torch.arange(num_champs)
        ranks.append(rank)
        tops.append(set(order[:5]))
    return ranks, tops


@torch.no_grad()
def hits(model, loader, device):
    """Per-example top-1 / top-5 hit flags."""
    h1, h5 = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        cl = out["policy_logits"].float().cpu()
        tc = batch["target_champ"].cpu()
        tl = cl.gather(1, tc[:, None]).squeeze(1)
        r = (cl > tl[:, None]).sum(1)
        h1.append(r < 1)
        h5.append(r < 5)
    return torch.cat(h1), torch.cat(h5)


def mcnemar(a, b):
    """Return (only_a, only_b, z, p_two_sided)."""
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 0.0, 1.0
    # normal approx with continuity correction
    z = (abs(only_a - only_b) - 1) / math.sqrt(n)
    p = math.erfc(abs(z) / math.sqrt(2))
    return only_a, only_b, z, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    rows, _ = PRO_PIPE.build_rows()
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_series_by_date(rows)
    num_champs = 173

    ma, pa = load_model(args.a, device)
    mb, pb = load_model(args.b, device)

    def loader_for(payload):
        # Patch ids must come from the checkpoint's own vocab: the patch
        # embedding rows are indexed by the training-time vocabulary, so
        # a checkpoint with a different vocab needs a different mapping.
        ds = PRO_DATA.ProNextMoveDataset(
            rows, splits["test"], num_champs=num_champs,
            patch_index=PRO_DATA.patch_index_from_vocab(
                payload["config"].get("patch_vocab")))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=PRO_DATA.pro_collate, num_workers=0)

    va = pa["config"].get("patch_vocab") or []
    vb = pb["config"].get("patch_vocab") or []
    if va and vb and va != vb:
        # Row order and (game, move) pairing are identical either way,
        # so the paired test stays valid; only patch_id differs.
        print("note: checkpoints use different patch vocabularies; "
              "each is evaluated with its own")
        la, lb = loader_for(pa), loader_for(pb)
    else:
        la = lb = loader_for(pa)

    a1, a5 = hits(ma, la, device)
    b1, b5 = hits(mb, lb, device)

    # per-move phase index for each example (dataset order = row*20+move)
    move_idx = torch.arange(len(a1)) % N_MOVES
    phase = torch.tensor([PHASE_OF[int(m)] for m in move_idx])
    is_ban = torch.tensor([PRO_DATA.PHASE_OF[int(m)] in (0, 2)
                           for m in move_idx])

    n = len(a1)
    print("n examples: %d" % n)
    print()
    print("%-24s %8s %8s %9s" % ("", "A", "B", "delta"))
    print("%-24s %8.4f %8.4f %+9.4f" % ("top1", a1.float().mean(),
                                        b1.float().mean(),
                                        float(b1.float().mean() - a1.float().mean())))
    print("%-24s %8.4f %8.4f %+9.4f" % ("top5", a5.float().mean(),
                                        b5.float().mean(),
                                        float(b5.float().mean() - a5.float().mean())))
    print()
    for name, mask in (("bans", is_ban), ("picks", ~is_ban)):
        print("%-10s top5  A %.4f  B %.4f  %+.4f" % (
            name, a5[mask].float().mean(), b5[mask].float().mean(),
            float(b5[mask].float().mean() - a5[mask].float().mean())))
    print()
    for ph in range(4):
        m = phase == ph
        print("%-10s top5  A %.4f  B %.4f  %+.4f  (n=%d)" % (
            PHASE_NAMES[ph], a5[m].float().mean(), b5[m].float().mean(),
            float(b5[m].float().mean() - a5[m].float().mean()), int(m.sum())))
    print()
    print("McNemar on top-5 (A = %s, B = %s)" % (
        Path(args.a).parent.name, Path(args.b).parent.name))
    for name, mask in (("all", torch.ones(n, dtype=torch.bool)),
                       ("bans", is_ban), ("picks", ~is_ban)):
        oa, ob, z, p = mcnemar(a5[mask], b5[mask])
        print("  %-6s onlyA=%-6d onlyB=%-6d z=%+6.2f p=%.4f %s" % (
            name, oa, ob, z, p,
            "(significant)" if p < 0.05 else "(not significant)"))
    print()
    oa, ob, z, p = mcnemar(a1, b1)
    print("  top1   onlyA=%-6d onlyB=%-6d z=%+6.2f p=%.4f %s" % (
        oa, ob, z, p, "(significant)" if p < 0.05 else "(not significant)"))

    # popularity baseline
    ranks, tops = build_popularity(rows, splits["train"], num_champs)
    targets = torch.tensor([int(ds[i]["target_champ"]) for i in range(len(ds))])
    pr = torch.stack([ranks[int(m)][targets[i]]
                      for i, m in enumerate(move_idx)])
    print()
    print("popularity baseline  top1 %.4f  top5 %.4f  top10 %.4f" % (
        (pr < 1).float().mean(), (pr < 5).float().mean(),
        (pr < 10).float().mean()))
    for name, mask in (("bans", is_ban), ("picks", ~is_ban)):
        print("  %-6s top5 %.4f" % (name, (pr[mask] < 5).float().mean()))


if __name__ == "__main__":
    main()
