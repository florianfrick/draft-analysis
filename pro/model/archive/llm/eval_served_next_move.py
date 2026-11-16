"""Served head vs served reranker, on one set of rows, through the real functions.

## Why this exists, and what it replaces

Every "the reranker is +16 pp" sentence in this repo compares
`eval_rerank5_mask4k.json` (34.27 %) against `C.BASELINE_VAL` (18.32 %). That
comparison is wrong twice over:

* `BASELINE_VAL` is `nextpick-proonly-**randomvalsplit**-.../test_report.json`,
  not the shipped `-nov13-k5-` head - a different checkpoint family on a
  different corpus;
* it is measured over **95,160** states, while the rerank arms were scored on the
  **4,000-row** `llmval` subsample.

Measured properly, the shipped head is **24.90 % top-1 / 58.03 % top-5** on
those 4,000 rows, not 18.32 / 51.18. A second difference follows from that: the
58.03 is higher than the `ceiling_recall` of 51.18 the rerank evals record,
because `predict_next_move` *filters* illegal champions out of the head's list
(banned, already picked, fearless pool) while `rerank.ranks` scores the raw
logits over all 173. So the reranker's 34.27 % was measured on a candidate set
that can contain illegal champions, and the head's 24.90 % on one that cannot.

This harness removes both problems by asking the question the frontend asks:
`model_tools.predict_next_move`, once with the head in the slot and once with
the reranker, on the same rows. Same candidates, same legality filter, same
call. What is left is the model.

## Arms

    --limit 0        the head alone, all 4,000 rows (CPU, ~1 min)
    --rerank DIR     adds the reranker over the same rows (~2 s each) and the
                     paired test on exactly those

## Reading it

`top1` is "the pro's next champion is the model's #1". `top5` is the shortlist
recall, and it **must be identical in both arms** - the reranker only re-orders
a list it did not build. If it moves, the arms saw different candidate sets and
the run is void; the harness prints a warning. `discordant` is [rerank-only
wins, head-only wins] and `mcnemar_p` is the exact two-sided test on them.
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
from pro.model.archive.llm.eval_api_rerank import (             # noqa: E402
    _state_text, mcnemar, wilson)
from solo_queue.model.policy import infer               # noqa: E402

DEFAULT_HEAD = str(ROOT / "pro/model/checkpoints"
                   "/nextpick-proonly-nov13-k5-through-lcp-df-sw/best.pt")
DEFAULT_RERANK = str(ROOT / "pro/model/archive/checkpoints/llm-lora-rerank5")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default="2022")
    p.add_argument("--val-games", type=int, default=200)
    p.add_argument("--limit", type=int, default=0,
                   help="rows to score; 0 = all 4,000")
    p.add_argument("--head", default=DEFAULT_HEAD)
    p.add_argument("--rerank", default=DEFAULT_RERANK)
    p.add_argument("--no-rerank", dest="rerank", action="store_const", const="",
                   help="head only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def load_rows(args, names, head=None):
    """Rows, plus the head's **raw** argmax when a head is given.

    The raw column exists because the LLM ladder in STATUS.md §9 is scored that
    way: `evaluate_llm` takes the head's logits over all 173 champions and asks
    whether the target is the argmax, with no legality filter. `exact` for the
    rerank arms is computed on the same construction, so the ladder's "head"
    row has to be the raw one - putting the *served* head there would compare a
    filtered number against unfiltered ones. `predict_next_move` filters, so the
    two differ by ~6 pp and both are printed.
    """
    corpus = C.load_rows(args.corpus)
    _, patch_index = D.make_patch_index(corpus)
    ds = D.eval_dataset(corpus, C.split_rows(corpus)["val"], patch_index,
                        seed=0, val_games=args.val_games)
    out = []
    for batch, samples in DataLoader(ds, batch_size=32, shuffle=False,
                                     collate_fn=D.collate):
        rank = None
        if head is not None:
            with torch.no_grad():
                logits = head.model(D.to_device(batch, "cpu"))["policy_logits"]
            rank = logits.argmax(dim=-1).tolist()
        for i, s in enumerate(samples):
            st, _ = _state_text(s, names, None)
            out.append({"gameid": s["meta"]["gameid"],
                        "move_idx": int(s["move_idx"]),
                        "board": {"blue_picks": st.picks_of("blue"),
                                  "red_picks": st.picks_of("red"),
                                  "bans": st.bans(),
                                  "first_pick": 1 if st.red_first else 0,
                                  "patch": st.patch,
                                  "unavailable": list(st.unavailable)},
                        "target": names[int(s["target_champ"])],
                        "target_idx": int(s["target_champ"]),
                        "raw_argmax": (None if rank is None
                                       else int(rank[i]))})
    return out


def score(rows, pred, champs, args, tag):
    t0 = time.time()
    hits1 = hits5 = 0
    per_row = []
    for i, row in enumerate(rows, 1):
        out = model_tools.predict_next_move(pred, champs, k=args.k,
                                            **row["board"])
        got = [t["name"] for t in out["top"]]
        top1 = bool(got) and got[0] == row["target"]
        hits1 += int(top1)
        hits5 += int(row["target"] in got[:5])
        per_row.append({"top1": top1, "top5": row["target"] in got[:5],
                        "top": got, "model": out["model"]})
        if i % 100 == 0:
            print("  %s %d/%d  top1 %.4f  top5 %.4f  %.0f s"
                  % (tag, i, len(rows), hits1 / i, hits5 / i,
                     time.time() - t0), flush=True)
    n = len(rows)
    lo, hi = wilson(hits1, n)
    return {"n": n, "top1": hits1 / max(1, n), "top5": hits5 / max(1, n),
            "ci_top1": [lo, hi], "k": args.k, "seconds": time.time() - t0,
            "model": per_row[0]["model"] if per_row else None,
            "rows": per_row}


def main(argv=None):
    args = parse_args(argv)
    names = C.champ_names()
    champs = model_tools.ChampIndex(names)
    head = infer.PolicyPredictor(args.head, device="cpu")
    all_rows = load_rows(args, names, head)
    print("rows: %d (val_games %d x %d moves)"
          % (len(all_rows), args.val_games, C.N_MOVES), flush=True)
    if args.limit:
        all_rows = all_rows[:args.limit]

    head_res = score(all_rows, head, champs, args, "head")
    out = {"head": {k: v for k, v in head_res.items() if k != "rows"},
           "rerank": None, "val_games": args.val_games, "k": args.k,
           "note": ("served head vs served reranker through "
                    "model_tools.predict_next_move - same rows, same legality "
                    "filter, same candidate construction")}
    print("head   served top1 %.4f [%.4f, %.4f]  top5 %.4f  (n=%d)"
          % (head_res["top1"], *head_res["ci_top1"], head_res["top5"],
             head_res["n"]))
    raw = [r for r in all_rows if r["raw_argmax"] is not None]
    if raw:
        hit = sum(1 for r in raw if r["raw_argmax"] == r["target_idx"])
        print("head   raw argmax top1 %.4f  (n=%d)  <- the column the LLM "
              "ladder's `exact` is comparable with" % (hit / len(raw), len(raw)))
        out["head"]["top1_raw"] = hit / len(raw)

    if args.rerank:
        from pro.model.archive.llm.rerank_predictor import RerankPredictor
        rr = RerankPredictor(args.rerank, args.head, device=args.device)
        rr_res = score(all_rows, rr, champs, args, "rerank")
        b = sum(1 for h, r in zip(head_res["rows"], rr_res["rows"])
                if r["top1"] and not h["top1"])
        c = sum(1 for h, r in zip(head_res["rows"], rr_res["rows"])
                if h["top1"] and not r["top1"])
        rr_res.update(mcnemar_p=mcnemar(b, c), discordant=[b, c],
                      delta_pp=100.0 * (rr_res["top1"] - head_res["top1"]))
        out["rerank"] = {k: v for k, v in rr_res.items() if k != "rows"}
        print("rerank top1 %.4f [%.4f, %.4f]  top5 %.4f  (n=%d)  %+.2f pp  "
              "p=%.3g  discordant %d/%d"
              % (rr_res["top1"], *rr_res["ci_top1"], rr_res["top5"],
                 rr_res["n"], rr_res["delta_pp"], rr_res["mcnemar_p"], b, c))
        if abs(rr_res["top5"] - head_res["top5"]) > 1e-9:
            print("WARNING: top-5 differs between arms (%.4f vs %.4f) - the "
                  "candidate sets are not the same, so the comparison is void"
                  % (rr_res["top5"], head_res["top5"]))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
