"""Score and compare win-probability sources on one split.

This is the reusable half of the experiment: it knows nothing about how a
prediction was produced, only that a prediction is `(y, p)` over a known set of
games. Four kinds of source are supported, and they can be mixed freely:

    --baseline prior            the blue-win-rate prior fitted on train
    --ckpt name=path.pt         a checkpoint from train.py
    --preds  name=path.npz      a saved prediction set (anything, any model)
    --prior  name=0.53          a bare number

    python model/winpred/evaluate.py --data pro --split test \\
        --baseline prior --ckpt model=model/checkpoints/winpred-pro-ce-.../best.pt

Why it insists on game ids
--------------------------
Every source must have been scored on **exactly** the split's rows. The script
proves that by comparing id sets and refuses a subset or a superset, because
intersecting two predictions would silently score both models on the rows they
happen to share - the comparison that flatters whichever model is worse.

Output: a printed table (all five metrics plus the two confidence buckets), a
JSON report with every number, per-patch slices, and a **grouped** bootstrap of
the difference against the reference source (whole series resampled, not rows,
because two games of a series share teams and a fearless pool).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.winpred import data as D              # noqa: E402
from model.winpred import metrics as M           # noqa: E402
from model.winpred import model as W             # noqa: E402
from model.winpred.baseline import BlueWinRateBaseline  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compare win-probability sources")
    ap.add_argument("--data", choices=("soloq", "pro"), default="pro")
    ap.add_argument("--corpus", choices=sorted(D.CORPORA), default="2022")
    ap.add_argument("--soloq-patch", default="16.20")
    ap.add_argument("--soloq-cache", default="")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--seed", type=int, default=D.SPLIT_SEED)
    ap.add_argument("--limit-drafts", type=int, default=0)

    ap.add_argument("--baseline", default="",
                    help="name for the blue-win-rate prior, e.g. 'prior'. "
                         "Empty = do not include it.")
    ap.add_argument("--prior", action="append", default=[],
                    metavar="NAME=VALUE", help="a fixed probability (repeatable)")
    ap.add_argument("--ckpt", action="append", default=[],
                    metavar="NAME=PATH", help="a train.py checkpoint")
    ap.add_argument("--preds", action="append", default=[],
                    metavar="NAME=PATH", help="a saved .npz prediction set")
    ap.add_argument("--reference", default="",
                    help="name of the source every delta is measured against "
                         "(default: the first one given)")

    ap.add_argument("--conf-lo", type=float, default=M.CONF_LO)
    ap.add_argument("--conf-hi", type=float, default=M.CONF_HI)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--by-patch", action="store_true",
                    help="print the per-patch table")
    ap.add_argument("--allow-vocab-mismatch", action="store_true",
                    help="score a checkpoint whose patch vocabulary differs "
                         "from the data's (ids then mean different patches)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default="")
    return ap.parse_args(argv)


def _kv(items, flag):
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit("%s expects NAME=VALUE, got %r" % (flag, item))
        name, _, value = item.partition("=")
        out.append((name.strip(), value.strip()))
    return out


@torch.no_grad()
def _predict_ckpt(path, table, idx, meta, args):
    model, payload = W.load_checkpoint(path, args.device)
    ck_vocab = list(payload.get("patch_vocab") or [])
    data_vocab = list(meta.get("patch_vocab") or [])
    if ck_vocab != data_vocab and not args.allow_vocab_mismatch:
        raise SystemExit(
            "patch vocabulary mismatch for %s:\n  checkpoint: %s\n  data:       "
            "%s\nPatch ids are vocabulary positions, so the same integer means "
            "a different patch. Re-run with the corpus/patch the checkpoint "
            "was trained on, or pass --allow-vocab-mismatch to score it anyway."
            % (path, ck_vocab[:5], data_vocab[:5]))
    cfg_data = (payload.get("config") or {}).get("data")
    if cfg_data and cfg_data != args.data:
        print("WARNING: checkpoint was trained on %r data, scoring %r data"
              % (cfg_data, args.data))
    ds = D.WinDataset(table, idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=D.collate)
    ys, ps = [], []
    for batch in loader:
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        ps.append(torch.sigmoid(model(batch)["logit"]).float().cpu().numpy())
        ys.append(batch["win_label"].float().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def resolve_sources(args, table, splits, meta):
    """-> [(name, y, p)], all aligned to the split's row order."""
    idx = splits[args.split]
    ids = [table.game_id[i] for i in idx]
    y_split = table.y[idx].astype(np.float64)
    sources = []

    if args.baseline:
        base = BlueWinRateBaseline().fit(table.y[splits["train"]])
        sources.append((args.baseline, y_split,
                        base.predict_proba(len(idx))))
        print("baseline %r: prior %.5f fitted on %d train rows"
              % (args.baseline, base.prior, base.n_train))
    for name, value in _kv(args.prior, "--prior"):
        sources.append((name, y_split, np.full(len(idx), float(value))))
    for name, path in _kv(args.ckpt, "--ckpt"):
        y, p = _predict_ckpt(path, table, idx, meta, args)
        sources.append((name, y, p))
    for name, path in _kv(args.preds, "--preds"):
        pred = D.align_predictions(D.load_predictions(path), ids)
        sources.append((name, pred["y"].astype(np.float64),
                        pred["p"].astype(np.float64)))

    if not sources:
        raise SystemExit("no source given: use --baseline, --prior, --ckpt or "
                         "--preds")
    for name, y, _ in sources:
        if not np.array_equal(y, y_split):
            raise SystemExit(
                "source %r carries labels that differ from the split's (this "
                "is a row-alignment bug, not a model difference)" % name)
    return sources, idx


def main(argv=None):
    args = parse_args(argv)
    if args.data == "pro":
        table, meta = D.load_pro(corpus=args.corpus)
    else:
        table, meta = D.load_soloq(patch=args.soloq_patch,
                                   cache_path=args.soloq_cache or None)
    if args.limit_drafts and args.limit_drafts < len(table):
        stride = max(1, len(table) // args.limit_drafts)
        table = table.subset(np.arange(0, len(table), stride)[:args.limit_drafts])
    splits, split_rep = D.make_splits(table, seed=args.seed)

    sources, idx = resolve_sources(args, table, splits, meta)
    ref_name = args.reference or sources[0][0]
    if ref_name not in [s[0] for s in sources]:
        raise SystemExit("--reference %r is not one of the sources" % ref_name)
    ref = next(s for s in sources if s[0] == ref_name)

    groups = ([table.series_id[i] for i in idx] if table.source == "pro"
              else [table.game_id[i] for i in idx])

    reports = {}
    for name, y, p in sources:
        rep = M.full_report(y, p, lo=args.conf_lo, hi=args.conf_hi,
                            n_bins=args.n_bins, with_bins=True)
        # per-patch slice: the pro corpus spans 106 patches, so a pooled number
        # can hide a model that is only good on the newest one.
        by_patch = {}
        patches = np.asarray([table.patch[i] for i in idx])
        for patch in sorted(set(patches.tolist())):
            m = patches == patch
            if int(m.sum()) < 10:
                continue
            by_patch[patch] = {
                "n": int(m.sum()),
                "log_loss": M.log_loss(y[m], p[m]),
                "brier": M.brier(y[m], p[m]),
                "roc_auc": M.roc_auc(y[m], p[m]),
                "accuracy": M.accuracy(y[m], p[m]),
            }
        rep["by_patch"] = by_patch
        reports[name] = rep

    # ---- printed table
    print("\n%s split, seed %d: n=%d, blue win rate %.4f"
          % (args.split, args.seed, len(idx), float(table.y[idx].mean())))
    head = ("%-14s %7s %9s %8s %8s %8s %8s %8s %7s"
            % ("source", "n", "log-loss", "Brier", "ECE", "AUC", "acc",
               "prec@conf", "cover"))
    print(head)
    print("-" * len(head))
    for name, _, _ in sources:
        r = reports[name]
        print("%-14s %7d %9.4f %8.4f %8.4f %8.4f %8.4f %8.4f %6.1f%%"
              % (name, r["n"], r["log_loss"], r["brier"], r["ece"],
                 r["roc_auc"], r["accuracy"],
                 r["confident"]["precision_confident"],
                 100 * r["confident"]["coverage"]))
    print("\nhigh-confidence buckets (p<%.2f -> red, p>%.2f -> blue)"
          % (args.conf_lo, args.conf_hi))
    for name, _, _ in sources:
        c = reports[name]["confident"]
        print("  %-14s blue n=%5d prec %.4f recall %.4f | red n=%5d "
              "prec %.4f recall %.4f"
              % (name, c["n_blue_calls"], c["precision_blue"],
                 c["recall_blue"], c["n_red_calls"], c["precision_red"],
                 c["recall_red"]))

    # ---- paired deltas vs the reference
    deltas = {}
    for name, y, p in sources:
        if name == ref_name:
            continue
        entry = {"auc_delta": reports[name]["roc_auc"] - reports[ref_name]["roc_auc"],
                 "log_loss_delta": (reports[name]["log_loss"]
                                    - reports[ref_name]["log_loss"]),
                 "brier_delta": reports[name]["brier"] - reports[ref_name]["brier"]}
        if not args.no_bootstrap:
            for metric in ("log_loss", "brier", "accuracy"):
                entry[metric] = M.bootstrap_delta(
                    y, ref[2], p, groups, metric=metric, n_boot=args.n_boot,
                    seed=args.seed)
        deltas[name] = entry

    if deltas:
        print("\nvs %r (negative log-loss/Brier delta = the other model is "
              "better)" % ref_name)
        for name, d in deltas.items():
            print("  %-14s dlog-loss %+.5f  dBrier %+.5f  dAUC %+.5f"
                  % (name, d["log_loss_delta"], d["brier_delta"],
                     d["auc_delta"]))
            for metric in ("log_loss", "brier", "accuracy"):
                b = d.get(metric)
                if isinstance(b, dict):
                    print("      %-9s %+.5f  95%% CI [%+.5f, %+.5f]  %s"
                          % (metric, b["delta"], b["lo"], b["hi"],
                             "excludes 0" if b["excludes_zero"]
                             else "straddles 0"))

    if args.by_patch and reports:
        print("\nper-patch log-loss (newest 12 patches with n>=10)")
        patches = sorted(reports[ref_name]["by_patch"],
                         key=lambda p: tuple(int(x) for x in p.split(".")))
        print("  %-8s %6s " % ("patch", "n")
              + " ".join("%12s" % s[0] for s in sources))
        for patch in patches[-12:]:
            row = "  %-8s %6d " % (patch,
                                   reports[ref_name]["by_patch"][patch]["n"])
            for name, _, _ in sources:
                row += " %12.4f" % reports[name]["by_patch"][patch]["log_loss"]
            print(row)

    payload = {
        "protocol": {"data": args.data, "corpus": meta.get("corpus"),
                     "split": args.split, "seed": args.seed,
                     "conf_lo": args.conf_lo, "conf_hi": args.conf_hi,
                     "n_bins": args.n_bins, "reference": ref_name,
                     "corpus_max_date": meta.get("corpus_max_date", ""),
                     "newest_row_date": meta.get("newest_row_date", ""),
                     "split_counts": split_rep["counts"],
                     "data_info": table.info()},
        "sources": {name: {"path": dict(_kv(args.ckpt + args.preds,
                                            "--ckpt")).get(name, "")}
                    for name, _, _ in sources},
        "reports": reports,
        "deltas_vs_reference": deltas,
    }
    if args.out:
        D.write_json(args.out, payload)
        print("\nwrote %s" % args.out)
    return payload


if __name__ == "__main__":
    main()
