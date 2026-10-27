"""Additive pairwise model of draft -> win probability.

Fits  logit P(blue wins) = sum_blue v  - sum_red v            (main effects)
                         + sum_{b in blue, r in red} W[b,r]    (counters)
                         + sum_{pairs in blue} S - ... in red   (synergies)

The point of this model class: it is *linear in log-odds*, so every
prediction decomposes exactly into named terms. And its sufficient
statistics are 1-way and 2-way cells -- the two densities the soloQ
corpus actually has (1-way: 100% of cells at n>=30, 2-way: 97.7%).
No 5-way cell is needed.

Reports test log-loss / accuracy for four nested models so the marginal
value of each structure is measurable, then decomposes one draft.

Usage
-----
  python "soloq data analysis/additive_model.py" fit
  python "soloq data analysis/additive_model.py" fit --epochs 40 --wd 2e-5
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "store")

# Riot champion ids are sparse (1..950), so index a dense 0..172 space.
_cb = pq.read_table(os.path.join(STORE, "champ_base.parquet"), columns=["champ"])
RIOT_IDS = sorted(set(_cb.column("champ").to_pylist()))
ID2IX = {c: i for i, c in enumerate(RIOT_IDS)}
NCH = len(RIOT_IDS)


def load():
    t = pq.read_table(os.path.join(STORE, "drafts.parquet"),
                      columns=["match_id", "blue_champs", "red_champs", "blue_win"])
    blue = t.column("blue_champs").to_pylist()
    red = t.column("red_champs").to_pylist()
    y = t.column("blue_win").to_pylist()
    keep = [i for i, v in enumerate(y) if v is not None and len(blue[i]) == 5
            and len(red[i]) == 5]
    b = torch.tensor([[ID2IX[c] for c in blue[i]] for i in keep], dtype=torch.long)
    r = torch.tensor([[ID2IX[c] for c in red[i]] for i in keep], dtype=torch.long)
    yy = torch.tensor([float(y[i]) for i in keep], dtype=torch.float32)
    return b, r, yy


def split(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    a, b = int(0.70 * n), int(0.85 * n)
    return perm[:a], perm[a:b], perm[b:]


class DraftModel(nn.Module):
    def __init__(self, use_counter=True, use_synergy=True):
        super().__init__()
        self.use_counter, self.use_synergy = use_counter, use_synergy
        self.v = nn.Parameter(torch.zeros(NCH))
        self.W = nn.Parameter(torch.zeros(NCH, NCH)) if use_counter else None
        self.S = nn.Parameter(torch.zeros(NCH, NCH)) if use_synergy else None

    def logit(self, b, r):
        z = self.v[b].sum(1) - self.v[r].sum(1)
        if self.W is not None:
            z = z + self.W[b[:, :, None], r[:, None, :]].sum((1, 2))
        if self.S is not None:
            m = 1.0 - torch.eye(5)
            sb = (self.S[b[:, :, None], b[:, None, :]] * m).sum((1, 2))
            sr = (self.S[r[:, :, None], r[:, None, :]] * m).sum((1, 2))
            z = z + 0.5 * (sb - sr)
        return z


def evaluate(model, b, r, y):
    model.eval()
    with torch.no_grad():
        z = model.logit(b, r)
        loss = nn.functional.binary_cross_entropy_with_logits(z, y).item()
        acc = ((z > 0).float() == y).float().mean().item()
    return loss, acc


def fit(kind, b, r, y, tr, va, epochs, wd, lr, l2_int=None, seed=0):
    torch.manual_seed(seed)
    model = DraftModel(use_counter=kind in ("counter", "both"),
                       use_synergy=kind in ("both",))
    # Main effects are 173 params on 290k games: they need almost no
    # shrinkage. The 29,929 interaction cells are ~10 games per parameter
    # once you count how little signal each carries, so they need a lot.
    # One global weight_decay cannot serve both -- that is why the first
    # fit lost to a constant.
    if l2_int is None:
        l2_int = wd
    groups = [{"params": [model.v], "weight_decay": wd}]
    for p in (model.W, model.S):
        if p is not None:
            groups.append({"params": [p], "weight_decay": l2_int})
    opt = torch.optim.Adam(groups, lr=lr)
    bt, rt, yt = b[tr], r[tr], y[tr]
    n = len(tr)
    bs = 8192
    best = (9e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                model.logit(bt[idx], rt[idx]), yt[idx])
            loss.backward()
            opt.step()
        vl, vacc = evaluate(model, b[va], r[va], y[va])
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2-int", type=float, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep interaction L2 and report the best on val")
    args = ap.parse_args()

    t0 = time.time()
    b, r, y = load()
    n = len(y)
    tr, va, te = split(n)
    print("games %d  (train %d / val %d / test %d)" % (n, len(tr), len(va), len(te)))
    base = float(y.mean())
    print("blue win base rate: %.4f" % base)
    const_loss = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    const_acc = max(base, 1 - base)
    print("%-22s %10s %8s %8s" % ("model", "test logloss", "bits/game", "acc"))
    print("%-22s %10.5f %8.4f %8.4f" % ("constant", const_loss,
                                        const_loss / np.log(2), const_acc))

    results, models = {}, {}
    if args.sweep:
        # The question is not "can pairwise fit the data" (it always can)
        # but "at the best shrinkage, does it beat main effects alone".
        print("\n--- interaction-L2 sweep (val logloss; main L2 = %g) ---" % args.wd)
        print("%10s %12s %12s" % ("l2_int", "counter", "counter+syn"))
        best = {}
        for l2 in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
            row = []
            for kind in ("counter", "both"):
                m, vl = fit(kind, b, r, y, tr, va, args.epochs, args.wd,
                            args.lr, l2_int=l2)
                row.append(vl)
                if kind not in best or vl < best[kind][0]:
                    best[kind] = (vl, l2, m)
            print("%10.4g %12.5f %12.5f" % (l2, row[0], row[1]))
        main_m, main_vl = fit("main", b, r, y, tr, va, args.epochs, args.wd, args.lr)
        print("\nmain effects only: val %.5f" % main_vl)
        for kind, (vl, l2, m) in best.items():
            tl, ta = evaluate(m, b[te], r[te], y[te])
            mv = evaluate(main_m, b[te], r[te], y[te])
            print("%-12s best l2=%g  val %.5f  test %.5f  acc %.4f   "
                  "vs main: %+.5f logloss %+.2f pp"
                  % (kind, l2, vl, tl, ta, mv[0] - tl, 100 * (ta - mv[1])))
            results[kind] = {"val_logloss": vl, "test_logloss": tl,
                             "test_acc": ta, "l2_int": l2}
            models[kind] = m
        results["main"] = {"val_logloss": main_vl}
        models["main"] = main_m
        tl, ta = evaluate(main_m, b[te], r[te], y[te])
        results["main"].update({"test_logloss": tl, "test_acc": ta})
    else:
        for kind, label in (("main", "main effects only"),
                            ("counter", "+ counters"),
                            ("both", "+ counters + synergies")):
            m, vl = fit(kind, b, r, y, tr, va, args.epochs, args.wd,
                        args.lr, l2_int=args.l2_int)
            tl, ta = evaluate(m, b[te], r[te], y[te])
            results[kind] = {"val_logloss": vl, "test_logloss": tl, "test_acc": ta}
            models[kind] = m
            print("%-22s %10.5f %8.4f %8.4f   (val %.5f)" % (
                label, tl, tl / np.log(2), ta, vl))
            print("    %-18s improvement over constant: %+.4f bits, %+.2f pp acc"
                  % ("", (const_loss - tl) / np.log(2), 100 * (ta - const_acc)))

    # ---- how much does the pairwise structure actually matter? ----
    print("\n--- marginal value of structure (test set) ---")
    main_only = results["main"]["test_logloss"]
    for kind in ("counter", "both"):
        d = main_only - results[kind]["test_logloss"]
        print("  %-22s %+.5f logloss (%+.4f bits/game)  %+.2f pp acc"
              % (kind, d, d / np.log(2),
                 100 * (results[kind]["test_acc"] - results["main"]["test_acc"])))

    # ---- spread of the pairwise term: is it decoration or does it move? ----
    m = models["both"]
    with torch.no_grad():
        vb = m.v[b[te]].sum(1) - m.v[r[te]].sum(1)
        mm = 1.0 - torch.eye(5)
        wb = m.W[b[te][:, :, None], r[te][:, None, :]].sum((1, 2))
        sb = 0.5 * ((m.S[b[te][:, :, None], b[te][:, None, :]] * mm).sum((1, 2))
                    - (m.S[r[te][:, :, None], r[te][:, None, :]] * mm).sum((1, 2)))
    tot = vb + wb + sb
    print("\n--- term magnitudes on test drafts (log-odds, sd across games) ---")
    for nm, x in (("main effects", vb), ("counters", wb), ("synergies", sb),
                  ("total", tot)):
        print("  %-14s sd %.3f  |mean| %.3f  p1 %.3f  p99 %.3f"
              % (nm, x.std().item(), x.abs().mean().item(),
                 x.kthvalue(int(0.01 * len(x))).values.item(),
                 x.kthvalue(int(0.99 * len(x))).values.item()))
    print("  corr(main, counters) = %.3f"
          % np.corrcoef(vb.numpy(), wb.numpy())[0, 1])

    # ---- how many counter cells are actually estimable? ----
    W = m.W.detach().numpy()
    print("\ncounter matrix: %d cells, sd of coefficients %.4f" % (W.size, W.std()))
    print("champion main-effect spread: sd %.4f  range %.3f..%.3f"
          % (m.v.detach().std().item(), m.v.min().item(), m.v.max().item()))

    torch.save({"v": m.v.detach(), "W": m.W.detach(), "S": m.S.detach(),
                "meta": {"n_games": n, "test_logloss": results["both"]["test_logloss"],
                         "const_logloss": const_loss}},
               os.path.join(HERE, "additive_model.pt"))
    with open(os.path.join(HERE, "additive_model.json"), "w") as f:
        json.dump({"base_rate": base, "const_logloss": const_loss,
                   "const_acc": const_acc, "results": results,
                   "epochs": args.epochs, "wd": args.wd}, f, indent=2)
    print("\nwrote additive_model.pt / .json   (%.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
