"""Score a set of checkpoints on BOTH test views, in one table.

The sweep's question is not "does arm X beat the prior on its own data" - each
`test_report.json` already answers that. It is the **cross** question: does a
model trained on soloQ hold up on pro games, and does adding pro data to soloQ
help or hurt the soloQ number? That needs every arm scored on the same two test
sets, which is what this script does.

    python model/winpred/sweep.py --seed 20261105 \\
        --ckpt soloq-ce=model/checkpoints/winpred-soloq-ce-seed20261105-p16_20/best.pt \\
        --ckpt pro-ce=model/checkpoints/winpred-pro-ce-seed20261105-through-20261104/best.pt \\
        --out model/winpred/reports/sweep_seed20261105.json

Both views use the same split seed as training, so the test rows are the ones
no arm saw. Patch vocabularies differ between arms (a soloQ-only checkpoint has
a one-row patch table), and the ids are translated by patch *label* - see
`evaluate.patch_remap`. A pro-trained arm therefore scores soloQ rows on the
16.20 row it actually learned, and a soloQ-only arm scores pro rows with
everything but 16.20 on its UNK row, which is the honest version of "this model
has never seen those patches".
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.winpred import data as D              # noqa: E402
from model.winpred import evaluate as E          # noqa: E402
from model.winpred import metrics as M           # noqa: E402
from model.winpred.baseline import BlueWinRateBaseline  # noqa: E402

VIEWS = ("soloq", "pro")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compare arms on both test views")
    ap.add_argument("--ckpt", action="append", default=[],
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--seed", type=int, default=D.SPLIT_SEED)
    ap.add_argument("--corpus", choices=sorted(D.CORPORA), default="2022")
    ap.add_argument("--soloq-patch", default="16.20")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default="")
    return ap.parse_args(argv)


def load_view(view, args):
    if view == "pro":
        return D.load_pro(corpus=args.corpus, verbose=False)
    return D.load_soloq(patch=args.soloq_patch, verbose=False)


def score_view(view, ckpts, args):
    """-> {source: {"report": ..., "delta": {...}}} for one test view."""
    table, meta = load_view(view, args)
    splits, _ = D.make_splits(table, seed=args.seed, verbose=False)
    idx = splits["test"]
    y = table.y[idx].astype(np.float64)
    groups = [table.series_id[i] for i in idx]

    base = BlueWinRateBaseline().fit(table.y[splits["train"]])
    sources = [("prior", y, base.predict_proba(len(idx)))]
    ns = SimpleNamespace(device=args.device, batch_size=args.batch_size,
                         allow_vocab_mismatch=False, data=view)
    for name, path in ckpts:
        _, p = E._predict_ckpt(path, table, idx, meta, ns)
        sources.append((name, y, p))

    ref = sources[0][2]
    out = {}
    prior_auc = None
    for name, yy, p in sources:
        rep = M.full_report(yy, p)
        if name == "prior":
            prior_auc = rep["roc_auc"]
        entry = {"report": rep}
        if name != "prior":
            entry["delta"] = {
                m: M.bootstrap_delta(yy, ref, p, groups, metric=m,
                                     n_boot=args.n_boot, seed=args.seed)
                for m in ("log_loss", "brier", "accuracy")}
            entry["auc_delta"] = rep["roc_auc"] - prior_auc
        out[name] = entry
    return out, len(idx), float(y.mean()), base.describe()


def main(argv=None):
    args = parse_args(argv)
    ckpts = []
    for item in args.ckpt:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit("--ckpt expects NAME=PATH, got %r" % item)
        ckpts.append((name.strip(), path.strip()))
    if not ckpts:
        raise SystemExit("no --ckpt given")

    payload = {"protocol": {"seed": args.seed, "corpus": args.corpus,
                            "views": args.views.split(","),
                            "n_boot": args.n_boot},
               "arms": [n for n, _ in ckpts], "views": {}}
    for view in args.views.split(","):
        scores, n, base_rate, prior = score_view(view, ckpts, args)
        payload["views"][view] = {"n": n, "blue_win_rate": base_rate,
                                  "prior": prior, "scores": scores}

        print("\n=== %s test (n=%d, blue win rate %.4f) ===" % (view, n,
                                                               base_rate))
        head = ("%-12s %9s %8s %8s %8s %8s %8s %7s"
                % ("arm", "log-loss", "Brier", "ECE", "AUC", "acc",
                   "prec@cf", "cover"))
        print(head)
        print("-" * len(head))
        for name in ["prior"] + [n for n, _ in ckpts]:
            r = scores[name]["report"]
            print("%-12s %9.4f %8.4f %8.4f %8.4f %8.4f %8.4f %6.1f%%"
                  % (name, r["log_loss"], r["brier"], r["ece"], r["roc_auc"],
                     r["accuracy"],
                     r["confident"]["precision_confident"],
                     100 * r["confident"]["coverage"]))
        print("delta vs prior (negative log-loss/Brier = better):")
        for name, _ in ckpts:
            d = scores[name]["delta"]
            line = "  %-12s dll %+.5f [%+.5f,%+.5f]  dBrier %+.5f  dAUC %+.5f" % (
                name, d["log_loss"]["delta"], d["log_loss"]["lo"],
                d["log_loss"]["hi"], d["brier"]["delta"],
                scores[name]["auc_delta"])
            line += "  (acc %+.4f)" % d["accuracy"]["delta"]
            print(line)

    if args.out:
        D.write_json(args.out, payload)
        print("\nwrote %s" % args.out)
    return payload


if __name__ == "__main__":
    main()
