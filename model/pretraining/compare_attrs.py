"""Compare pretraining checkpoints on one fixed val split, split by frequency.

Why this exists. The kit-attribute adapter adds 18*D + D parameters to a
~5M-parameter encoder, so its effect on aggregate champ top-5 is small, and a
single pair of runs cannot tell a small real gain from seed noise. Two things
make the measurement usable:

* **One val view for every checkpoint.** Same rows, same masking seed, same
  batch size. The trainer's own val numbers use `seed + 1` for the mask
  stream, so two runs with different `--seed` are scored on different masks -
  comparable, but not identical.
* **Stratify by how often the true champion appeared in training.** The
  adapter's claim is not "every champion gets better", it is "a champion with
  few updates benefits from axes that are shared with champions that have
  many". Aggregate top-5 is dominated by the 30 champions that appear
  thousands of times, which is exactly where the adapter should matter least.
  So the number to read is the bottom tercile.

Usage (run from the repo root, inside the container):

    python model/pretraining/compare_attrs.py \
        --ckpt off=ckpt/run_x/best.pt --ckpt on=ckpt/run_y/best.pt \
        --limit-drafts 100000

Data flags must match the runs being compared, or the val split will not be
the one they were trained against.
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))             # soloq_model
sys.path.insert(0, str(HERE.parent.parent / "data" / "solo"
                      / "queue" / "process matchdata"))

import train as T                                        # noqa: E402
from soloq_model import (SoloQPretrainedModel,            # noqa: E402
                         attr_matrix_for_config, load_encoder_state)

VAL_MASK_SEED = 20261031   # fixed: the same masks for every checkpoint


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable, e.g. on=path.pt")
    ap.add_argument("--patch", default="16.20")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--limit-drafts", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="", help="optional report path")
    ap.add_argument("--holdout-champs", type=int, default=0,
                    help="must match the runs: adds a 'holdout' group using "
                         "the same fewest-appearances rule as the trainer")
    args = ap.parse_args(argv)
    # load_rows() reads these two; this script never wants win labels and
    # never rescans (it must see the same cache the runs were trained on).
    args.no_win = True
    args.rescan = False
    return args


def build_val(args):
    """-> (val_rows, num_champs, champ_count, holdout_ids)."""
    rows, _wins, mapping = T.load_rows(args)
    num_champs = len(mapping)      # load_rows returns riot_to_idx itself
    tr_idx, va_idx = T.split_indices(len(rows), args.val_frac, args.data_seed)
    train_rows = [rows[i] for i in tr_idx]
    val_rows = [rows[i] for i in va_idx]
    count = torch.zeros(num_champs)
    for champ_ids, _roles in train_rows:
        for c in champ_ids:
            count[int(c)] += 1
    holdout = (T.holdout_champions(train_rows, num_champs,
                                   args.holdout_champs)
               if args.holdout_champs else set())
    return val_rows, num_champs, count, holdout


def terciles(count):
    """Champion ids grouped by training frequency: (rare, mid, common)."""
    order = torch.argsort(count)
    n = len(order)
    third = n // 3
    return order[:third], order[third:2 * third], order[2 * third:]


def load_model(path, device):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        attr_matrix=attr_matrix_for_config(cfg)).to(device)
    load_encoder_state(model, payload["model"])
    return model.eval(), cfg


@torch.no_grad()
def measure(model, loader, groups, device):
    """-> dict with overall and per-tercile top-1 / top-5 on masked slots.

    Also keeps the per-slot hit vectors: every checkpoint is scored on the
    same val view in the same order, so a pair can be compared with a paired
    test instead of two independent percentages. That matters here because
    the differences are a fraction of a point, i.e. the same size as the
    binomial error of an unpaired comparison.
    """
    in_group = {k: torch.zeros(model.champ_head.out_features, dtype=torch.bool)
                .to(device) for k in groups}
    for k, ids in groups.items():
        in_group[k][ids.to(device)] = True
    agg, hits = {}, {}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        m = batch["is_masked"]
        if not bool(m.any()):
            continue
        logits, truth = out["champ_logits"][m], batch["champ_labels"][m]
        rank = (logits > logits.gather(1, truth[:, None])).sum(1)
        hit1, hit5 = (rank < 1), (rank < 5)
        for key in ["all"] + list(groups):
            sel = torch.ones_like(hit5) if key == "all" else in_group[key][truth]
            d = agg.setdefault(key, [0.0, 0.0, 0])
            d[0] += float(hit1[sel].sum())
            d[1] += float(hit5[sel].sum())
            d[2] += int(sel.sum())
            hits.setdefault(key, []).append(
                (hit1[sel].to("cpu"), hit5[sel].to("cpu")))
    out = {k: {"top1": v[0] / max(1, v[2]), "top5": v[1] / max(1, v[2]),
               "n": v[2]} for k, v in agg.items()}
    for k, pairs in hits.items():
        out[k]["_h1"] = torch.cat([p[0] for p in pairs])
        out[k]["_h5"] = torch.cat([p[1] for p in pairs])
    return out


def mcnemar(b, c):
    """Two-sided exact binomial p for the discordant pairs of a paired test.

    b = only the second model got it right, c = only the first did. Under the
    null both are equally likely, so X ~ Bin(b+c, 0.5); the p-value is the
    two-sided tail. Exact rather than chi-square because b+c is often small
    on the rare-champion groups.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    from math import exp, lgamma, log
    const = -n * log(2)
    tail = sum(exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + const)
               for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    val_rows, num_champs, count, holdout = build_val(args)
    rare, mid, common = terciles(count)
    groups = {"rare": rare, "mid": mid, "common": common}
    if holdout:
        groups["holdout"] = torch.tensor(sorted(holdout), dtype=torch.long)
    print("val %d drafts; terciles by train frequency: rare<=%d picks, "
          "common>=%d picks"
          % (len(val_rows), int(count[rare].max()), int(count[common].min())))
    if holdout:
        print("holdout: %d champions never trained (same rule as the runs)"
              % len(holdout))
    loader = T.make_loader(val_rows, num_champs, args.batch_size,
                           shuffle=False, mask_prob=0.15,
                           seed=VAL_MASK_SEED, role_only_prob=0.10,
                           force_one=True)

    report, order = {}, []
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        model, cfg = load_model(path, device)
        res = measure(model, loader, groups, device)
        res["attr_dim"] = int(cfg.get("attr_dim") or 0)
        if model.attr_dim:
            w = model.embed.attr_proj.weight
            res["adapter_norm"] = float(w.norm())
        report[label] = res
        order.append(label)
        del model

    cols = ["all", "rare", "mid", "common"] + (["holdout"] if holdout else [])
    print("\n%-8s %-6s " % ("arm", "axes")
          + " ".join("%-15s" % c for c in cols))
    print("%-8s %-6s " % ("", "")
          + " ".join("%-15s" % "top1 / top5" for _ in cols))
    for label in order:
        r = report[label]
        cells = ["%.4f / %.4f" % (r[c]["top1"], r[c]["top5"]) for c in cols]
        print("%-8s %-6d " % (label, r["attr_dim"]) + " ".join("%-15s" % c
                                                              for c in cells))

    if len(order) == 2:
        a, b = (report[order[0]], report[order[1]])
        print("\ndelta (%s - %s), percentage points, with a paired test on "
              "the same val slots:" % (order[1], order[0]))
        for c in cols:
            d5 = 100 * (b[c]["top5"] - a[c]["top5"])
            d1 = 100 * (b[c]["top1"] - a[c]["top1"])
            # McNemar on top-5: slots the two arms disagree about.
            h5a, h5b = a[c]["_h5"].bool(), b[c]["_h5"].bool()
            only_b = int((~h5a & h5b).sum())
            only_a = int((h5a & ~h5b).sum())
            p = mcnemar(only_b, only_a)
            print("  %-7s top5 %+6.2f pp   top1 %+6.2f pp   "
                  "(%s only %d vs %s only %d, p=%.2f, n=%d)"
                  % (c, d5, d1, order[1], only_b, order[0], only_a, p,
                     b[c]["n"]))
    for label in order:
        r = report[label]
        if "adapter_norm" in r:
            print("%s: adapter weight norm %.4f" % (label, r["adapter_norm"]))

    if args.json:
        import json
        slim = {}
        for label, r in report.items():
            slim[label] = {g: {"top1": v["top1"], "top5": v["top5"],
                               "n": v["n"]}
                           for g, v in r.items() if isinstance(v, dict)}
            for extra in ("attr_dim", "adapter_norm"):
                if extra in r:
                    slim[label][extra] = r[extra]
        Path(args.json).write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print("wrote", args.json)
    return report


if __name__ == "__main__":
    main()
