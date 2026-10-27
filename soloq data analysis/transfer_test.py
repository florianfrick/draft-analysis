"""Does a soloQ-measured lane fingerprint transfer to pro drafts?

The proposed architecture reads a draft as the sum of five lane matchups,
each described by a fingerprint measured on soloQ. That rests on two
assumptions nobody has tested:

  transfer   a lane measured in soloQ behaves the same in pro
  additivity the five lanes add up to a team-level early advantage

Pro drafts carry team gold@10 per side (not per lane), which is exactly the
sum the additive model predicts. So: build the predicted edge from soloQ
fingerprints, and regress actual pro gold@10 differential on it.

Reported against two baselines, because "champions that are strong win
early gold" would produce a positive R2 with no matchup information at all.

Usage:  python "soloq data analysis/transfer_test.py"
"""
import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "pt", os.path.join(HERE, "pro_theta.py"))
PT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PT)


def r2(y, pred):
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def lstsq_r2(X, y, tr, te):
    """Fit OLS on train, return test R2. X already includes an intercept col."""
    w = torch.linalg.lstsq(X[tr], y[tr].unsqueeze(1)).solution
    return r2(y[te], (X[te] @ w).squeeze(1))


def main():
    fp = torch.from_numpy(PT.build_fp())
    b, r, y, series = PT.load_pro()

    rows = torch.load(os.path.join(ROOT, "data", "pro play", "pro_drafts.pt"),
                      map_location="cpu", weights_only=False)["rows"]
    # econ is None for games the source left incomplete; has_econ gates it
    econ = torch.tensor([[v if v is not None else 0.0 for v in r_["econ"]]
                         for r_ in rows], dtype=torch.float32)
    has = torch.tensor([bool(r_["has_econ"]) for r_ in rows])
    gold10 = econ[:, 0] - econ[:, 1]          # blue minus red, at 10 min
    keep = has & (len(rows) == len(y))
    print("pro games: %d, with gold@10: %d" % (len(y), int(keep.sum())))
    print("gold@10 diff: mean %.0f  sd %.0f" % (
        float(gold10[keep].mean()), float(gold10[keep].std())))

    # predicted early edge: sum of the five same-role lane fingerprints
    ar = torch.arange(5)
    f = fp[ar, b, r, :]                       # (N, 5 roles, K)
    cs = f[:, :, 0].sum(1)                    # summed CS edge, standardised units
    lvl = f[:, :, 1].sum(1)                   # summed level edge
    dmg = f[:, :, 3].sum(1)                   # summed damage/min edge

    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    # split by series so a series never spans train and test
    uniq = sorted(set(series))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    trs = set(uniq[:int(.7 * len(uniq))])
    is_tr = torch.tensor([s in trs for s in series]) & keep
    tr = torch.where(is_tr)[0]
    te = torch.where(keep & ~is_tr)[0]
    print("train %d / test %d games" % (len(tr), len(te)))

    one = torch.ones(n, 1)
    # champion main effects as an indicator design: +1 for blue pick, -1 for red
    A = torch.zeros(n, PT.NCH)
    for j in range(5):
        A[torch.arange(n), b[:, j]] += 1.0
        A[torch.arange(n), r[:, j]] -= 1.0

    feats = {
        "intercept only": one,
        "champ main effects": torch.cat([one, A], 1),
        "fingerprint sums only": torch.cat([one, cs[:, None], lvl[:, None]], 1),
        "main + fingerprint": torch.cat([one, A, cs[:, None], lvl[:, None],
                                         dmg[:, None]], 1),
    }
    print("\n%-26s %9s  %s" % ("predictors", "test R2", "params"))
    for name, X in feats.items():
        print("%-26s %9.4f  %d" % (name, lstsq_r2(X, gold10, tr, te), X.shape[1]))

    # sign sanity: the summed CS edge should point the same way as gold
    c = float(torch.corrcoef(torch.stack([cs[keep], gold10[keep]]))[0, 1])
    print("\ncorr(summed CS edge, gold@10 diff) = %+.3f" % c)


if __name__ == "__main__":
    main()
