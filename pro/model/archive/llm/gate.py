"""The adapter's job, done with five numbers and a logistic regression.

## The question

The 4B LoRA reranker scores 34.61 % exact on its eval slice against the head's
18.32 %; a hosted `deepseek-chat` reading the same shortlist and the same
percentages scores 22.68 % on 4,000 states. Is the remaining 12 pp *judgement*
that only a finetuned trunk has - or is it calibration that any learned rule can
recover from the head's own output?

This fits the rule. Nothing here is a language model: the input is the head's
own top-k probabilities plus what the board already says, and the output is
"keep the head's #1 or promote one of the others". If a handful of parameters
closes most of the gap to the LoRA, the adapter was doing calibration; if it
does not, the adapter knows something the head's posterior does not contain.

## What it fits

Nested scorers over the k candidates, `score_j = w . x_j`, picked by argmax (so
a row-level feature cancels - only per-candidate ones are used):

  rank       log p_j + a per-rank offset. Five parameters. This is the whole
             "flatten the head and re-rank it" hypothesis: it can learn that the
             head is overconfident, but not which candidate is right.
  +role      adds `role_open` (the candidate plays a lane this side has not
             filled) and `role_head1` (it shares the head's #1 role), because
             "first lane-legal in the top 20" was measured at +1.0 pp.
  +phase     adds the is_ban interactions, since every published arm has gained
             on picks and lost on bans, or the other way round.

Plus the one-parameter baseline the agent's own failures suggest: **switch to
#2 iff `p1 - p2 < tau`**, with `tau` tuned on the training games.

## Protocol

Fit on shared-val games, evaluate on 200 of them the fit never saw - the same
200 (`seed 0`) that `eval_api_rerank.py` scored, so head, API and gate are one
paired comparison on identical rows. The fit set is every other shared-val
game, so the gate's training data is drawn from the same distribution the LoRA
was evaluated on and neither side has seen the head's training rows.

    python -m pro.model.archive.llm.gate --train-games 1000 --val-games 200

    # the cache the fit reads: a couple of minutes on the 4060, reusable
    python -m pro.model.archive.llm.gate --collect-only

Report: `pro/model/archive/llm/gate_report.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.agent import draft_state as DS                 # noqa: E402
from pro.model.archive.llm import common as C                         # noqa: E402
from pro.model.archive.llm import data as D                           # noqa: E402
from pro.model.archive.llm import rerank as R                         # noqa: E402
from solo_queue.model.policy import lanes                     # noqa: E402

N_ROLES = 5
DEFAULT_ROWS = HERE / "gate_rows.npz"
DEFAULT_OUT = HERE / "gate_report.json"

# The published reference points, on rows that are NOT the ones scored here.
# Printed next to every run for the same reason `eval_api_rerank` prints them:
# a number quoted next to another number becomes a comparison whether or not
# the rows matched.
REFERENCE = {
    "head_95260": 0.1832,
    "lora_exact_9560": 0.3461,
    "api_chat_4000": 0.2268,
    "ceiling_4000": 0.5108,
}

PHASE_OF_MOVE = {**{m: "ban1" for m in range(0, 6)},
                 **{m: "pick1" for m in range(6, 12)},
                 **{m: "ban2" for m in range(12, 16)},
                 **{m: "pick2" for m in range(16, 20)}}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=5, help="shortlist size")
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--val-games", type=int, default=200,
                   help="eval games, seed 0 - the same sample eval_api_rerank "
                        "scored, so the comparison is paired")
    p.add_argument("--train-games", type=int, default=1000,
                   help="shared-val games to fit on, excluding the eval sample")
    p.add_argument("--head", default=str(R.HEAD_CKPT))
    p.add_argument("--device", default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--rows", default=str(DEFAULT_ROWS))
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args(argv)


# ------------------------------------------------------------------ features
def _open_roles(picks, side):
    """Roles `side` has not filled, from the 10-slot board.

    `lanes.assign_lanes` is the table's own assignment (96 % agreement with the
    role head), so this asks the question the frontend asks rather than a second
    opinion invented here.
    """
    assigned = lanes.assign_lanes(picks)
    filled = {assigned[side * 5 + r] for r in range(N_ROLES)
              if assigned[side * 5 + r] is not None}
    return [r for r in range(N_ROLES) if r not in filled]


def _legal_by_role():
    return {r: set(lanes.legal_champions(r)) for r in range(N_ROLES)}


def _compacted_picks(champ_ids, pad_mask):
    """Corpus lane slots -> the UI board (each team's picks in pick order)."""
    picks = []
    for team in (0, 1):
        lo, hi = team * 5, (team + 1) * 5
        filled = [int(c) for c, pad in zip(champ_ids[lo:hi], pad_mask[lo:hi])
                  if not pad]
        picks.extend(filled + [None] * (5 - len(filled)))
    return picks


F_NAMES = (["logp"] + ["rank%d" % j for j in range(1, 5)]
           + ["role_open", "role_head1", "logp_ban", "role_open_ban"])
# Which columns each nested model uses: a prefix, so the nesting is checkable.
MODEL_COLS = {"rank": 5, "rank+role": 7, "rank+role+phase": 9}


def row_features(cand_ids, probs, picks, side, is_ban, legal):
    """-> [k, F] float32 for one row, in F_NAMES order."""
    k = len(cand_ids)
    logp = np.log(np.clip(probs, 1e-12, None)).astype(np.float32)
    role_open = np.zeros(k, dtype=np.float32)
    role_head1 = np.zeros(k, dtype=np.float32)
    if not is_ban:
        open_roles = _open_roles(picks, side)
        head1_roles = [r for r in open_roles if cand_ids[0] in legal[r]]
        for j, c in enumerate(cand_ids):
            role_open[j] = float(any(c in legal[r] for r in open_roles))
            role_head1[j] = float(any(c in legal[r] for r in head1_roles))
    rank = np.zeros((k, k - 1), dtype=np.float32)
    for j in range(1, k):
        rank[j, j - 1] = 1.0
    return np.concatenate([
        logp[:, None], rank,
        role_open[:, None], role_head1[:, None],
        (logp * is_ban)[:, None], (role_open * is_ban)[:, None],
    ], axis=1).astype(np.float32)


# -------------------------------------------------------------------- collect
@torch.no_grad()
def collect(args, device):
    rows = C.load_rows(args.corpus)
    _, patch_index = D.make_patch_index(rows)
    _, val_idx = R.split_games(rows, "heldout")
    ds_eval = D.eval_dataset(rows, val_idx, patch_index, seed=0,
                             val_games=args.val_games)
    eval_games = {s["meta"]["gameid"] for s in ds_eval}
    train_idx = [i for i in val_idx if rows[i]["gameid"] not in eval_games]
    if args.train_games:
        train_idx = train_idx[:args.train_games]
    ds_fit = D.eval_dataset(rows, train_idx, patch_index, seed=1, val_games=0)
    print("shared val %d games | eval %d games / %d states | "
          "fit %d games / %d states"
          % (len({rows[i]["gameid"] for i in val_idx}), len(eval_games),
             len(ds_eval), len({rows[i]["gameid"] for i in train_idx}),
             len(ds_fit)), flush=True)

    head, cfg = R.load_head(args.head, device)
    C.assert_patch_compat(cfg, patch_index)
    print("head %s" % Path(args.head).parent.name, flush=True)

    legal = _legal_by_role()
    schedule = {0: DS.SCHEDULE_BLUE_FIRST, 1: DS.SCHEDULE_RED_FIRST}

    out = {}
    for tag, ds in (("fit", ds_fit), ("eval", ds_eval)):
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=D.collate, num_workers=0)
        acc = {key: [] for key in ("probs", "cand", "target", "feat", "is_ban",
                                   "move_idx", "gameid", "phase")}
        for batch, samples in loader:
            batch = D.to_device(batch, device)
            logits = head(batch)["policy_logits"]
            cid, cp = R.candidates(logits, args.k)
            for i, s in enumerate(samples):
                picks = _compacted_picks(s["champ_ids"].tolist(),
                                         s["pad_mask"].tolist())
                move = int(s["move_idx"])
                is_ban = int(s["is_ban"])
                _kind, side = schedule[int(s["first_pick"])][move]
                side = 0 if side == DS.BLUE else 1
                cands = [int(c) for c in cid[i].tolist()]
                probs = np.asarray(cp[i].tolist(), dtype=np.float32)
                acc["feat"].append(row_features(cands, probs, picks, side,
                                                float(is_ban), legal))
                acc["probs"].append(probs)
                acc["cand"].append(cands)
                acc["target"].append(int(s["target_champ"]))
                acc["is_ban"].append(is_ban)
                acc["move_idx"].append(move)
                acc["phase"].append(PHASE_OF_MOVE[move])
                acc["gameid"].append(s["meta"]["gameid"])
        out[tag] = acc
        n_open = float(np.mean([f[:, F_NAMES.index("role_open")].mean()
                                for f in acc["feat"]]))
        print("  %s: %d rows | mean role_open %.2f (~0.45 overall, ~0.9 on picks)"
              % (tag, len(acc["target"]), n_open), flush=True)
    return out


def save_rows(rows, path):
    flat = {}
    for tag, acc in rows.items():
        flat[tag + "_feat"] = np.stack(acc["feat"])
        flat[tag + "_probs"] = np.stack(acc["probs"])
        flat[tag + "_cand"] = np.stack(acc["cand"])
        for key in ("target", "is_ban", "move_idx"):
            flat[tag + "_" + key] = np.asarray(acc[key], dtype=np.int64)
        flat[tag + "_gameid"] = np.asarray(acc["gameid"])
        flat[tag + "_phase"] = np.asarray(acc["phase"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_rows_cached(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for tag in ("fit", "eval"):
        out[tag] = {
            "feat": list(z[tag + "_feat"]), "probs": list(z[tag + "_probs"]),
            "cand": list(z[tag + "_cand"]), "target": list(z[tag + "_target"]),
            "is_ban": list(z[tag + "_is_ban"]),
            "move_idx": list(z[tag + "_move_idx"]),
            "gameid": list(z[tag + "_gameid"]),
            "phase": list(z[tag + "_phase"])}
    return out


# ----------------------------------------------------------------------- fit
def _labels(acc):
    """-> (X [N,k,F], y [N]) where y = index of the truth, -1 when absent."""
    X = np.stack(acc["feat"]).astype(np.float32)
    cand = np.stack(acc["cand"])
    target = np.asarray(acc["target"])
    y = np.full(len(target), -1, dtype=np.int64)
    for i, (row, t) in enumerate(zip(cand, target)):
        hit = np.nonzero(row == t)[0]
        if len(hit):
            y[i] = int(hit[0])
    return X, y


def fit_softmax(X, y, cols, weight_decay=1e-3, iters=200):
    """-> w [cols] for `score_j = w . x_j`, multinomial NLL on rows that have
    the truth in the shortlist. Rows without a positive carry no gradient."""
    keep = y >= 0
    Xk = torch.from_numpy(X[keep][:, :, :cols])
    yk = torch.from_numpy(y[keep])
    w = torch.zeros(cols, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=iters,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xk @ w, yk)
        loss = loss + weight_decay * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def predict(X, w, cols):
    return (X[:, :, :cols] @ w).argmax(axis=1)


def margin_rule(probs, tau):
    """Switch to #2 when the head's own margin is under `tau`."""
    p = np.asarray(probs)
    out = np.zeros(len(p), dtype=np.int64)
    out[(p[:, 0] - p[:, 1]) < tau] = 1
    return out


def tune_margin(probs, y, grid=201):
    """Best `tau` on the fit rows. A miss (y < 0) is wrong for both answers, so
    the grid maximises the same quantity the eval reports."""
    best, best_tau = -1.0, 0.0
    for tau in np.linspace(0.0, 0.5, grid):
        acc = float((margin_rule(probs, tau) == y).mean())
        if acc > best:
            best, best_tau = acc, float(tau)
    return best_tau, best


def mcnemar(a_only, b_only):
    """Exact two-sided binomial, same rule as `eval_rank_api.mcnemar`."""
    n = a_only + b_only
    if not n:
        return 1.0
    m = min(a_only, b_only)
    lg = math.lgamma(n + 1)
    tail = sum(math.exp(lg - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        - n * math.log(2.0)) for i in range(m + 1))
    return min(1.0, 2.0 * tail)


def paired(pred, base, y):
    """-> both accuracies, the discordant counts and the McNemar p."""
    hit_a, hit_b = pred == y, base == y
    a_only = int((hit_a & ~hit_b).sum())
    b_only = int((hit_b & ~hit_a).sum())
    return {"acc": float(hit_a.mean()), "base_acc": float(hit_b.mean()),
            "a_only": a_only, "b_only": b_only,
            "p": mcnemar(a_only, b_only), "n": int(len(y))}


# ---------------------------------------------------------------------- run
def run(rows, args):
    Xf, yf = _labels(rows["fit"])
    Xe, ye = _labels(rows["eval"])
    head_pred = np.zeros(len(ye), dtype=np.int64)      # the head's #1 is rank 0
    rep = {"protocol": {
        "k": args.k, "fit_rows": int(len(yf)), "eval_rows": int(len(ye)),
        "fit_games": int(len(set(rows["fit"]["gameid"]))),
        "eval_games": int(len(set(rows["eval"]["gameid"]))),
        "head": Path(args.head).parent.name, "features": F_NAMES,
        "weight_decay": args.weight_decay,
        "reference_not_paired": REFERENCE}, "models": {}}

    rep["models"]["head"] = {"acc": float((head_pred == ye).mean()),
                             "n": int(len(ye))}
    tau, train_acc = tune_margin(np.stack(rows["fit"]["probs"]), yf)
    rep["models"]["margin_rule"] = dict(
        paired(margin_rule(np.stack(rows["eval"]["probs"]), tau), head_pred, ye),
        tau=tau, train_acc=float(train_acc), params=1)

    for name, cols in MODEL_COLS.items():
        w = fit_softmax(Xf, yf, cols, args.weight_decay)
        rep["models"][name] = dict(paired(predict(Xe, w, cols), head_pred, ye),
                                   cols=cols, params=int(cols),
                                   weights=w.tolist(),
                                   train_acc=float((predict(Xf, w, cols)
                                                    == yf).mean()))

    phase = np.asarray(rows["eval"]["phase"])
    rep["by_phase"] = {}
    for ph in sorted(set(phase.tolist())):
        m = phase == ph
        rep["by_phase"][ph] = {"n": int(m.sum()), **{
            name: float((predict(Xe[m], np.asarray(
                rep["models"][name]["weights"]), MODEL_COLS[name]) == ye[m]).mean())
            for name in ("rank", "rank+role", "rank+role+phase")},
            "head": float((head_pred[m] == ye[m]).mean())}
    rep["ceiling"] = float((ye >= 0).mean())
    return rep


def print_report(rep):
    print("\n%d eval rows, %d fit rows | ceiling (truth in top-%d) %.2f %%"
          % (rep["protocol"]["eval_rows"], rep["protocol"]["fit_rows"],
             rep["protocol"]["k"], 100 * rep["ceiling"]))
    print("  %-18s %8s %8s %6s %6s %6s" % ("model", "acc", "head", "+/-",
                                           "p", "params"))
    head = rep["models"]["head"]["acc"]
    print("  %-18s %7.2f%% %7.2f%%" % ("head argmax", 100 * head, 100 * head))
    for name in ("margin_rule", "rank", "rank+role", "rank+role+phase"):
        m = rep["models"][name]
        print("  %-18s %7.2f%% %7.2f%% %+5.2f %6.3f %6d"
              % (name, 100 * m["acc"], 100 * m["base_acc"],
                 100 * (m["acc"] - m["base_acc"]), m["p"], m["params"]))
    print("  reference (NOT paired): head %.2f | LoRA exact %.2f | "
          "API chat %.2f" % (100 * REFERENCE["head_95260"],
                             100 * REFERENCE["lora_exact_9560"],
                             100 * REFERENCE["api_chat_4000"]))
    for name in ("rank", "rank+role", "rank+role+phase"):
        w = rep["models"][name]["weights"]
        top = sorted(zip(F_NAMES, w), key=lambda t: -abs(t[1]))[:4]
        print("  %-18s %s" % (name, ", ".join("%s %+.2f" % t for t in top)))
    print("  by phase (top-1): " + " | ".join(
        "%s n=%d head %.1f rank %.1f full %.1f"
        % (ph, v["n"], 100 * v["head"], 100 * v["rank"],
           100 * v["rank+role+phase"]) for ph, v in rep["by_phase"].items()))


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    path = Path(args.rows)
    if path.is_file():
        rows = load_rows_cached(path)
        print("rows from %s" % path, flush=True)
    else:
        rows = collect(args, device)
        save_rows(rows, path)
        print("wrote %s" % path, flush=True)
    if args.collect_only:
        return 0
    rep = run(rows, args)
    print_report(rep)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print("wrote", args.out)
    return rep


if __name__ == "__main__":
    main()
