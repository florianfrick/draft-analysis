"""Does the reranker help the role board? The one question it was trained on.

## Why this exists

`eval_rerank5_mask4k.json` says the adapter picks the pro's champion 34.27 % of
the time against the head's 18.32 %. That number is **role-conditional**: the
v4 task line reads "Choose the best Blue Side pick for Top lane", and
`target_role` comes from the corpus, i.e. it is the true lane of the pro's own
pick. `eval_rerank5_nolane4k.json` deletes that clause and the same weights
score **20.85 %** - so 13.4 of the 16 points are the lane, not drafting skill.

That makes the unconstrained next-move panel a +2.5 pp place for this model and
a +16 pp place for nothing. But the *role board* asks the trained question:
`predict_in_lane` names a lane, and the lane is the query rather than oracle
information about the answer. So this harness measures the surface where the
model's advantage is in-distribution:

    for every held-out pick state, ask "who does <side> take for <the lane the
    pro's pick filled>?" and score the top-1 against the pro's actual champion

Both arms go through `model_tools.predict_in_lane`, which is the function the
frontend's `/api/suggest` calls, so the candidate set is lane-constrained by the
head's logits exactly as it is in the panel. That is *not* the candidate set
`eval_rerank5` used (unconstrained top-5), which is why this is a separate
number and not a re-slice of that one.

## What it is not

Not the next-move panel (no lane named -> 20.85 %, measured elsewhere), and not
the ban-phase role board: a ban-phase cell is a MASK query and needs the masking
checkpoint, which has no counterpart here (`can_answer_in_lane` excludes it).

## Cost

One reranker call per state (~2 s on an 8 GB card at batch 4) plus a head call.
`--limit 300` is about 12 minutes; the head-only column is milliseconds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.agent import model_tools                 # noqa: E402
from pro.model.archive.llm import common as C                   # noqa: E402
from pro.model.archive.llm import data as D                     # noqa: E402
from pro.model.archive.llm import rerank as R                   # noqa: E402
from pro.model.archive.llm.eval_api_rerank import (             # noqa: E402
    _state_text, mcnemar, wilson)
from solo_queue.model.policy import infer               # noqa: E402

DEFAULT_HEAD = str(ROOT / "pro/model/checkpoints"
                   "/nextpick-proonly-nov13-k5-through-lcp-df-sw/best.pt")
DEFAULT_RERANK = str(ROOT / "pro/model/archive/checkpoints/llm-lora-rerank5")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="llmval")
    p.add_argument("--corpus", default="2022")
    p.add_argument("--val-games", type=int, default=200)
    p.add_argument("--limit", type=int, default=300,
                   help="states to score; 0 = every eligible one")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--head", default=DEFAULT_HEAD)
    p.add_argument("--rerank", default=DEFAULT_RERANK)
    p.add_argument("--no-rerank", dest="rerank", action="store_const",
                   const="", help="head only - the cheap arm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def eligible_rows(args, names):
    """-> rows that the role board would actually ask the next-move model.

    Same 4,000-state slice as the rerank evals (`--split llmval`,
    `--val-games 200`), filtered to the rows `can_answer_in_lane` accepts: a
    pick on the clock with the ban block complete. A ban-phase row belongs to
    the masker and would be a different experiment.
    """
    corpus = C.load_rows(args.corpus)
    _, patch_index = D.make_patch_index(corpus)
    if args.split == "llmval":
        # The shared val, exactly as `evaluate_llm --split llmval` builds it -
        # `R.split_games("llmval")` refuses by design (it came from
        # reconstructing a split the shipped head never used), so the two
        # numbers are only comparable if this branch matches that one.
        eval_idx = C.split_rows(corpus)["val"]
    else:
        _, eval_idx = R.split_games(corpus, args.split)
    ds = D.eval_dataset(corpus, eval_idx, patch_index, seed=0,
                        val_games=args.val_games)
    rows = []
    for batch, samples in DataLoader(ds, batch_size=16, shuffle=False,
                                     collate_fn=D.collate):
        for s in samples:
            if int(s["is_ban"]):
                continue
            role = int(s["target_role"])
            if not 0 <= role < len(model_tools.ROLES):
                continue
            if int(s["n_bans"]) < 10:
                continue
            st, _ = _state_text(s, names, None)
            rows.append({"gameid": s["meta"]["gameid"],
                         "move_idx": int(s["move_idx"]),
                         "state": st, "role": role,
                         "target": names[int(s["target_champ"])]})
    if args.limit:
        rows = rows[:args.limit]
    return rows


def ask(pred, champs, row, k):
    """One role-board cell, through the function `/api/suggest` calls."""
    st = row["state"]
    board = {"blue_picks": st.picks_of("blue"), "red_picks": st.picks_of("red"),
             "bans": st.bans(), "first_pick": 1 if st.red_first else 0,
             "patch": st.patch, "unavailable": list(st.unavailable)}
    due = st.to_move()
    return model_tools.predict_in_lane(
        pred, champs, model_tools.ROLES[row["role"]], side=due["side"],
        k=k, due=due, masker=None, **board)


def score(rows, pred, champs, args, tag):
    t0 = time.time()
    hits1 = hitsk = 0
    per_row = []
    for i, row in enumerate(rows, 1):
        out = ask(pred, champs, row, args.k)
        names = [t["name"] for t in out["top"]]
        top1 = bool(names) and names[0] == row["target"]
        topk = row["target"] in names
        hits1 += int(top1)
        hitsk += int(topk)
        per_row.append({"gameid": row["gameid"], "move_idx": row["move_idx"],
                        "role": model_tools.ROLES[row["role"]],
                        "target": row["target"], "top": names,
                        "top1": top1, "topk": topk, "method": out["method"]})
        if i % 25 == 0:
            print("  %s %d/%d  top1 %.3f  top%d %.3f  %.0f s"
                  % (tag, i, len(rows), hits1 / i, args.k, hitsk / i,
                     time.time() - t0), flush=True)
    n = len(rows)
    lo, hi = wilson(hits1, n)
    return {"n": n, "top1": hits1 / max(1, n), "topk": hitsk / max(1, n),
            "k": args.k, "ci_top1": [lo, hi], "seconds": time.time() - t0,
            "methods": sorted({r["method"] for r in per_row}),
            "rows": per_row}


def main(argv=None):
    args = parse_args(argv)
    names = C.champ_names()
    champs = model_tools.ChampIndex(names)
    rows = eligible_rows(args, names)
    print("eligible pick states: %d (of the %d-state llmval slice)"
          % (len(rows), args.val_games * 20), flush=True)
    if args.dry_run:
        for row in rows[:3]:
            print("  %s m%-2d role %-8s target %s"
                  % (row["gameid"], row["move_idx"],
                     model_tools.ROLES[row["role"]], row["target"]))
        return 0

    head = infer.PolicyPredictor(args.head, device="cpu")
    head_res = score(rows, head, champs, args, "head")
    out = {"head": head_res, "rerank": None, "k": args.k,
           "split": args.split, "val_games": args.val_games,
           "note": ("top-1 of the role-board cell against the pro's actual "
                    "pick, candidates lane-constrained by the head's logits")}
    print("head   top1 %.4f  top%d %.4f  (n=%d)"
          % (head_res["top1"], args.k, head_res["topk"], head_res["n"]))

    if args.rerank:
        from pro.model.archive.llm.rerank_predictor import RerankPredictor
        rr = RerankPredictor(args.rerank, args.head, device=args.device)
        rr_res = score(rows, rr, champs, args, "rerank")
        # Paired: only rows both arms answered count, and the test runs on the
        # rows where they disagree - the same rule as every other arm here.
        b = sum(1 for h, r in zip(head_res["rows"], rr_res["rows"])
                if r["top1"] and not h["top1"])
        c = sum(1 for h, r in zip(head_res["rows"], rr_res["rows"])
                if h["top1"] and not r["top1"])
        rr_res["mcnemar_p"] = mcnemar(b, c)
        rr_res["discordant"] = [b, c]     # rerank-only wins, head-only wins
        rr_res["delta_pp"] = 100.0 * (rr_res["top1"] - head_res["top1"])
        out["rerank"] = rr_res
        print("rerank top1 %.4f  top%d %.4f  (n=%d)  %+.2f pp  p=%.3g  "
              "discordant %d/%d"
              % (rr_res["top1"], args.k, rr_res["topk"], rr_res["n"],
                 rr_res["delta_pp"], rr_res["mcnemar_p"], b, c))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
