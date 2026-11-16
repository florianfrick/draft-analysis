"""A flat-feature MLP: the "no attention, no set structure" baseline.

The transformer families are given the board as a *set* - ten (champion, role)
tokens, a ban block, a fearless pool - and a permutation-invariant encoder with
cross-team attention. This is the cheapest honest way to ask whether that
matters: same board, same legality rule, same rows, same loss, but the board is
flattened into one 3,834-wide vector and pushed through two hidden layers.

It is not a strawman. The one-hot board is a *complete* encoding of the state
(every slot, every ban, every mask), so a big enough MLP could in principle
represent anything the transformer can. What it cannot do cheaply is share
weights across slots - a slot-position-specific fact has to be relearned ten
times - and that is exactly the hypothesis worth testing.

Scale
-----
`--hidden 192` over 3,834 inputs and two hidden layers is ~0.8 M parameters,
well inside the 3 M budget, and 20x smaller than the shipped policy model's
3.35 M. Both tasks are trained here:

    next_move   in 3,834 -> 173 logits, legality-masked, 20 states/game
    masked      in 1,974 -> 173 logits, no legality filter, 10 states/game

The masked arm deliberately applies *no* legality filter, because
`test_split.models.score_masked` does not: the encoder ranks over all 173
champions and so must this, or the two would be scored on different candidate
sets.

The split is not an argument
----------------------------
There is no `--val-frac` here. `test_split.split` is the project's one split and
this trainer builds it by calling `S.build`, which asserts the counts. A
baseline that could be trained on different rows would be a baseline nobody can
compare against, which is the entire reason the split module exists.

Usage:
    python -m pro.model.mlp.train                      # both tasks
    python -m pro.model.mlp.train --task next_move
    python -m pro.model.mlp.train --epochs 4 --states 40000   # smoke run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent          # .../pro/model/mlp
ROOT = HERE.parents[2]                          # .../lol draft analysis
sys.path.insert(0, str(ROOT))

from pro.model import baselines as B            # noqa: E402
from pro.model.mlp import model as MODEL        # noqa: E402
from pro.model.mlp.model import ILLEGAL, FlatMLP  # noqa: E402
from pro.model.test_split import models as M    # noqa: E402
from pro.model.test_split import split as S     # noqa: E402

CKPT_ROOT = ROOT / "pro" / "model" / "checkpoints"
NAMES = {"next_move": "nextpick-mlp-flat-v1", "masked": "masking-mlp-flat-v1"}
PER_GAME = {"next_move": B.N_MOVES, "masked": B.N_SLOTS}


def states_of(idx, per_row):
    return [(i, k) for i in idx for k in range(per_row)]


def resolve(rows, states):
    """(row_index, k) pairs -> (row, k) pairs, the shape `baselines` wants."""
    return [(rows[i], k) for i, k in states]


def build(task, rows, states, patch_index):
    if task == "next_move":
        return B.build_next_move(resolve(rows, states), patch_index)
    return B.build_masked(resolve(rows, states), patch_index)


def targets(task, rows, states):
    if task == "next_move":
        return np.array([int(rows[i]["moves"][k][2]) for i, k in states],
                        dtype=np.int64)
    return np.array([int(rows[i]["champ"][k]) for i, k in states],
                    dtype=np.int64)


def illegal_of(task, rows, states):
    """bool[B, C], True = not a legal answer. None on the masked task."""
    if task != "next_move":
        return None
    return B.illegal_batch(resolve(rows, states))


@torch.no_grad()
def evaluate(model, task, rows, idx, patch_index, batch=2048):
    """Top-1 / top-5 / mean rank on every state of the given rows."""
    model.eval()
    device = next(model.parameters()).device
    per_row = PER_GAME[task]
    states = states_of(idx, per_row)
    y = targets(task, rows, states)
    ranks = np.empty(len(states), dtype=np.int64)
    for b in range(0, len(states), batch):
        chunk = states[b:b + batch]
        X = torch.from_numpy(build(task, rows, chunk, patch_index)).to(device)
        logits = model(X)
        if task == "next_move":
            ill = illegal_of(task, rows, chunk)
            logits = logits.masked_fill(torch.from_numpy(ill).to(device),
                                        ILLEGAL)
        t = torch.from_numpy(y[b:b + batch]).unsqueeze(1).to(device)
        ranks[b:b + batch] = (logits > logits.gather(1, t)).sum(1).cpu().numpy()
    n = len(ranks)
    return {"n": n,
            "top1": float((ranks < 1).mean()),
            "top5": float((ranks < 5).mean()),
            "top10": float((ranks < 10).mean()),
            "mean_rank": float(ranks.mean())}


def train_task(task, rows, splits, patch_index, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    in_dim = B.dim(task, "mlp")
    hidden = (MODEL.largest_hidden(in_dim, depth=args.depth)
              if args.hidden in ("auto", "", None) else int(args.hidden))
    model = FlatMLP(in_dim, hidden=hidden, dropout=args.dropout,
                    depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    B.check_budget(NAMES[task], n_params)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * args.steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                       T_max=total_steps)

    per_row = PER_GAME[task]
    train_idx = np.array(splits["train"])
    print("\n== %s ==  in_dim %d  hidden %d  params %s"
          % (task, in_dim, hidden, format(n_params, ",")))
    print("   %d train games x %d states, %d states/epoch, %d epochs"
          % (len(train_idx), per_row, args.states, args.epochs))

    best, best_val, best_state, history = -1.0, None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        idx = rng.choice(train_idx, size=args.states, replace=True)
        kk = rng.integers(0, per_row, size=args.states)
        states = [(int(i), int(k)) for i, k in zip(idx, kk)]
        y_all = targets(task, rows, states)
        order = rng.permutation(args.states)
        run, seen = 0.0, 0
        for step, b in enumerate(range(0, args.states, args.batch)):
            sel = order[b:b + args.batch]
            chunk = [states[j] for j in sel]
            X = torch.from_numpy(build(task, rows, chunk, patch_index)).to(device)
            logits = model(X)
            if task == "next_move":
                ill = illegal_of(task, rows, chunk)
                logits = logits.masked_fill(torch.from_numpy(ill).to(device),
                                            ILLEGAL)
            y = torch.from_numpy(y_all[sel]).to(device)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(sel)
            seen += len(sel)
        val = evaluate(model, task, rows, splits["val"], patch_index)
        history.append({"epoch": epoch, "train_loss": run / seen, **val})
        print("   ep %2d  loss %.4f  val top1 %.4f top5 %.4f rank %.2f  (%.0fs)"
              % (epoch, run / seen, val["top1"], val["top5"], val["mean_rank"],
                 time.time() - t0))
        # Selected on val top-5, the criterion the policy trainer uses
        # (`rel_top5` is top-5 normalised by the frequency baseline; the
        # normalisation is constant across epochs, so the argmax is the same).
        if val["top5"] > best:
            best = val["top5"]
            best_val = dict(val, epoch=epoch)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(model, task, rows, splits["test"], patch_index)
    print("   TEST top1 %.4f top5 %.4f rank %.2f  (selected at epoch %d)"
          % (test["top1"], test["top5"], test["mean_rank"], best_val["epoch"]))

    run_dir = Path(args.out) / NAMES[task] if args.out else CKPT_ROOT / NAMES[task]
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "flat": {"kind": "mlp", "task": task, "in_dim": in_dim,
                         "hidden": hidden, "depth": args.depth,
                         "dropout": args.dropout,
                         "feature_version": B.FEATURE_VERSION},
                "patch_vocab": list(M.PRO_DATA.build_patch_vocab(rows)[0])},
               run_dir / "model.pt")
    B.write_flat_artifact(
        run_dir, kind="mlp", task=task, weights="model.pt", params=n_params,
        meta={"in_dim": in_dim, "hidden": hidden, "depth": args.depth,
              "dropout": args.dropout, "selected_epoch": best_val["epoch"],
              "selection": "val top-5", "states_per_epoch": args.states,
              "epochs": args.epochs, "seed": args.seed},
        val=best_val, test=test,
        split_counts={k: len(v) for k, v in splits.items()})
    try:
        shown = run_dir.relative_to(ROOT)
    except ValueError:
        shown = run_dir
    print("   wrote %s" % shown)
    return {"task": task, "params": n_params, "val": best_val, "test": test,
            "history": history}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=("next_move", "masked", "both"),
                    default="both")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--states", type=int, default=200_000,
                    help="states drawn per epoch (with replacement)")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--hidden", default="auto",
                    help="'auto' (default) = the widest layer that still fits "
                         "the 3 M budget, solved per task because the two "
                         "input widths differ; or an explicit width")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0,
                    help="scheduler steps per epoch (0 = infer from --states)")
    ap.add_argument("--out", default="",
                    help="checkpoint root (default pro/model/checkpoints)")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) or cuda. A 0.8 M-parameter MLP over "
                         "sparse one-hots does not need a GPU, and taking one "
                         "would evict whatever else is running on it.")
    args = ap.parse_args(argv)
    if not args.steps:
        args.steps = max(1, args.states // args.batch)

    device = torch.device(args.device)
    rows = S.load_rows(verbose=True)
    splits = S.build(rows)
    patch_index = M.PRO_DATA.build_patch_vocab(rows)[1]
    print("split: %d train / %d val / %d test"
          % (len(splits["train"]), len(splits["val"]), len(splits["test"])))

    tasks = ["next_move", "masked"] if args.task == "both" else [args.task]
    for task in tasks:
        train_task(task, rows, splits, patch_index, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
