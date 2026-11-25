"""The agentic arm: state + next-pick + lanes, no SQL and no reranker.

The question this answers is narrower than "is the agent good". It is:

    Can an LLM with a *state tool*, a *next-move tool it can size itself*, and a
    *lane table* replace the local Qwen reranker as the middle-man between the
    head's distribution and the user's answer?

Three things are deliberately different from the single-turn arms in
`eval_api_rerank.py`, and each is the point:

1. **No fixed shortlist.** The model calls `predict_next_move` with the `k` it
   chooses. The head's top-1 is right 18.7 % of the time, its top-5 51.1 % and
   its top-20 85.2 %, so k is a real decision: k=5 throws away a third of the
   answers, k=20 buys them back at the cost of a longer list to read.
2. **No `sql`.** SQL makes the agent spend its budget building queries, and the
   aggregate tier (`webapp/tools.py`) already answers the questions a draft
   panel asks. It is ablated, not merely absent, so a model that invents the
   tool name gets a refusal it can read. `rank_candidates` needs no ablation:
   comparing a shortlist is the harnessed model's own job now, so the tool does
   not exist and the refusal is the generic unknown-tool one.
3. **The board is rendered, and also held in a state.** The render is
   `DraftState.render`, byte-identical to the `--render state` arm of
   `eval_api_rerank`, so the only difference from that arm is the tool tier.
   That is the clean comparison: same board text, same head, different
   affordances.

## The two baselines, and why there are two

The head can be asked two ways and they are not identical:

- **batch**: the encoder fed the dataset's tensors, including the corpus's
  *true* role ids. This is what the single-turn arms scored.
- **tool**: `pred.next_move` on a board rebuilt from names, with role ids
  inferred by the lane table (96 % agreement with the true roles). This is the
  shipped serving path and the one the agent actually sees.

They disagree on the top-1 for a few percent of rows, which is enough to move an
18 % metric by a point. Reporting only one would either flatter the agent
(against the weaker tool baseline) or flatter the reranker (against the batch
one). So both are computed, and their difference is itself a result: it is the
price of inferring lanes at serve time.

## What it does not measure

- **Fidelity to the head** is not the metric; accuracy against the truth is,
  because the question is whether the agent can beat the head. The paired
  McNemar against the head's own argmax is the statistic that matters.
- **Prose parsing.** The answer is prose, so it is parsed: first sentence
  first, whole answer as a fallback, and the mode is recorded per row. A run
  with a high `body` count is a run whose metric is partly a parser's.
- **The shipped chat surface.** This is the tool loop, not `webapp/chat.py`.

## Usage

    # smoke test: 4 games, prints every transcript
    python -m pro.model.archive.llm.eval_api_agent --val-games 4 --verbose

    # the real run: the same 200 games / 4,000 states as the other arms
    python -m pro.model.archive.llm.eval_api_agent --val-games 200 --workers 8 \\
        --out pro/model/archive/llm/eval_api_agent_heldout.json

    # no key needed: print the system prompt and one row's first message
    python -m pro.model.archive.llm.eval_api_agent --val-games 2 --dry-run
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.agent import draft_state as DS            # noqa: E402
from pro.model.agent import llm_client, loop            # noqa: E402
from pro.model.agent import model_tools                 # noqa: E402
from pro.model.agent import prompt_explained            # noqa: E402
from pro.model.agent import suggest                      # noqa: E402
from pro.model.agent.session import Session             # noqa: E402
from pro.model.archive.llm import common as C                   # noqa: E402
from pro.model.archive.llm import data as D                     # noqa: E402
from pro.model.archive.llm import rerank as R                   # noqa: E402
# One implementation of the slot compaction and of the answer parser, so this
# arm cannot drift from the arms it is compared against.
from pro.model.archive.llm.eval_api_rerank import (             # noqa: E402
    REFERENCE, _state_text, mcnemar, parse_answer, wilson)

SLICE_NAMES = {0: "bans1", 1: "picks1", 2: "bans2", 3: "picks2"}

# The one-click button in the frontend asks exactly this question, with the
# same prompt and the same ablation (`suggest.config()`). An edit here that does
# not reach the harness would make the measured arm and the served arm two
# different experiments with one name.
QUESTION = suggest.QUESTION


# ------------------------------------------------------------------ answer
_MARKER = re.compile(r"(?:PICK|BAN)\s*:\s*\**\s*([A-Za-z'.\- ]{2,24})\s*\**\s*$",
                     re.IGNORECASE)


def answer_of(text, legal, names):
    """-> (chosen_id, named_but_illegal_id, mode).

    The prompt asks for a final `PICK: <name>` line, and that line is read
    first. It is not fussiness: the transcripts contain answers like "the
    model's #1 is Corki at 0.28, but blue's bot lane is full, so Leona", where
    the first champion named is the one being argued *against*. A parser that
    reads position alone scores that answer wrong, and at n=4,000 a 3 % misparse
    is larger than the effect being measured.

    The prose fallback stays because the marker is a request, not a guarantee:
    `mode` records which path was taken, so a run with many `first-sentence` or
    `body` rows can be read as partly a parser's number.

    `legal` is the whole draftable pool (~160 ids), not a shortlist: the agent
    chose its own k, so there is no list to restrict to. A champion named
    outside the pool is counted separately - that is a legality error (a ban, a
    fearless pick) and a different failure from naming nothing.
    """
    if not text or not text.strip():
        return None, None, "empty"
    tail = text.strip().split("\n")[-1]
    m = _MARKER.search(tail)
    if m:
        named = m.group(1).strip(" *_")
        cid, outside = parse_answer(named, legal, names)
        if cid is not None:
            return cid, outside, "marker"
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    cid, outside = parse_answer(first, legal, names)
    if cid is not None:
        return cid, outside, "first-sentence"
    cid, outside2 = parse_answer(text, legal, names)
    if cid is not None:
        return cid, outside, "body"
    return None, outside or outside2, "none"


class LockedHead:
    """Serialise `next_move` across worker threads, under `no_grad`.

    Eight threads sharing one module is the difference between a 30-minute run
    and a crash: the forward pass is not reentrant and an unguarded call builds
    an autograd graph on every one of ~16,000 inferences. The lock costs
    nothing at this scale (a single-row forward is ~15 ms against a ~1 s API
    call).

    Everything except `next_move` delegates, so `patch_vocab`, `num_champs` and
    the `.head` lookup in `model_tools._head` behave as on the wrapped
    predictor.
    """

    def __init__(self, pred, lock=None):
        self._pred = pred
        self._lock = lock or threading.Lock()

    def __getattr__(self, name):
        return getattr(self._pred, name)

    def next_move(self, *args, **kwargs):
        with self._lock, torch.no_grad():
            return self._pred.next_move(*args, **kwargs)


# -------------------------------------------------------------------- rows
def _legal_ids(sample, n_champs):
    """Champion ids this row's move may legally name: not taken, not fearless."""
    taken = {int(c) for c in
             sample["champ_ids"][~sample["pad_mask"].bool()].tolist()}
    taken |= {int(b) for b in
              sample["ban_ids"][:int(sample["n_bans"])].tolist()}
    pool = {int(c) for c in
            sample["unavailable_mask"].nonzero().flatten().tolist()}
    return [c for c in range(n_champs) if c not in taken and c not in pool]


def build_rows(args, pred):
    """-> rows, names, champs. Both baselines are filled in here.

    The batch baseline is computed in the same pass as the states because that
    is the only place the stacked tensors exist; rebuilding them from the row
    dicts afterwards would mean re-deriving every field the encoder reads, and a
    baseline that quietly differs from the one the other arms scored is worse
    than no baseline.
    """
    corpus = C.load_rows(args.corpus)
    _, patch_index = D.make_patch_index(corpus)
    names = C.champ_names()
    _, eval_idx = R.split_games(corpus, args.split)
    ds = D.eval_dataset(corpus, eval_idx, patch_index, seed=0,
                        val_games=args.val_games)
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=D.collate)

    rows = []
    for batch, samples in loader:
        rank = None
        if args.batch_baseline:
            with torch.no_grad():
                # The *head's* logits, even when the slot holds the reranker:
                # this column is the paired "what the head alone would have
                # said", and it is the only way the report can print the
                # 18.32 -> 34.27 step on the same rows. `RerankPredictor` has no
                # `.model`; the head it wraps does.
                hlog = getattr(pred, "head", pred).model(
                    D.to_device(batch, "cpu"))["policy_logits"]
            rank = R.ranks(hlog, batch["target_champ"]).tolist()
        for i, s in enumerate(samples):
            if args.only and (("ban" if int(s["is_ban"]) else "pick")
                              != args.only):
                continue
            st, _text = _state_text(s, names, None)
            rows.append({
                "gameid": s["meta"]["gameid"],
                "move_idx": int(s["move_idx"]),
                "is_ban": int(s["is_ban"]), "phase": int(s["phase"]),
                "target": int(s["target_champ"]),
                "state": st,
                "legal": _legal_ids(s, len(names)),
                "rank_batch": None if rank is None else int(rank[i]),
            })
    return rows, names, model_tools.ChampIndex(names)


def tool_baseline(rows, pred, champs, names):
    """Fill `rank_tool` / `top1_tool` through the function the agent calls.

    `predict_next_move` with k=20, not a reimplementation: the agent's paired
    baseline has to be the same code path it sees. A target outside the top 20
    is recorded as rank 20 - the metric only needs "is it the argmax" and "is it
    inside the top k", and k=20 is the widest k the tool offers.
    """
    for r in rows:
        board = r["state"].board()
        out = model_tools.predict_next_move(
            pred, champs, blue_picks=board["blue_picks"],
            red_picks=board["red_picks"], bans=board["bans"],
            first_pick=board["first_pick"], patch=board["patch"], k=20,
            unavailable=board["unavailable"])
        hit = [t["rank"] for t in out["top"] if t["name"] == names[r["target"]]]
        r["rank_tool"] = int(hit[0]) - 1 if hit else 20
        r["top1_tool"] = out["top"][0]["name"] if out["top"] else None
        r["top1_id"] = (champs.resolve(r["top1_tool"]) if r["top1_tool"]
                        else None)
        r["top20"] = {champs.resolve(t["name"]) for t in out["top"]}
    return rows


# -------------------------------------------------------------------- loop
def row_key(row, args, system):
    """Cache identity for one row.

    The prompt hash is in the key on purpose: editing the system prompt changes
    every answer, so a cache that survived it would silently report the previous
    prompt's accuracy. Same argument as `max_tokens` in `eval_api_rerank` - a
    resume must never resume into the result it was trying to replace.
    """
    return "|".join([row["gameid"], str(row["move_idx"]), args.model,
                     str(args.max_steps),
                     hashlib.sha1(system.encode("utf-8")).hexdigest()[:8]])


def run_one(row, pred, champs, names, client, system, args, cache, lock):
    key = row_key(row, args, system)
    if key in cache:
        return dict(cache[key], cached=True)

    store = DS.Store(champs, real=row["state"])
    session = Session(predictor=pred, champs=champs, store=store,
                      ablate=suggest.ABLATE,
                      max_calls=args.max_calls)
    started = time.time()
    out = loop.run(session, QUESTION, client=client, system=system,
                   max_steps=args.max_steps, state_id="real")
    seconds = time.time() - started

    ks = []
    for call in out.get("calls") or []:
        if call.get("name") == "predict_next_move":
            try:
                ks.append(int(call["args"].get("k", 8)))
            except (TypeError, ValueError):
                pass
    chosen, illegal, mode = answer_of(out.get("answer"), row["legal"], names)
    rec = {
        "key": key, "gameid": row["gameid"], "move_idx": row["move_idx"],
        "answer": out.get("answer"), "chosen": chosen, "illegal": illegal,
        "parse": mode, "stop": out.get("stop"),
        "n_calls": len(out.get("calls") or []),
        "by_tool": (out.get("summary") or {}).get("by_tool") or {},
        "k_max": max(ks) if ks else None, "k_all": ks,
        "seconds": round(seconds, 2),
        "error": out.get("error"),
    }
    if args.trace:
        rec["messages"] = out.get("messages")
    # A failed row is not a result: caching it would make the retry a no-op and
    # the report would carry the failure as if it were an answer.
    if not rec["error"] and rec["stop"] != "loop error":
        with lock:
            cache[key] = rec
    return dict(rec, cached=False)


def _load_cache(path):
    cache = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[rec["key"]] = rec
    return cache


# ------------------------------------------------------------------ report
class Tally:
    def __init__(self):
        self.n = self.hit = self.head_hit = self.ceil = 0
        self.illegal = self.none = 0

    def add(self, rec, row):
        self.n += 1
        self.hit += int(rec["chosen"] == row["target"])
        self.head_hit += int(row["rank_tool"] == 0)
        self.ceil += int(row["rank_tool"] < 5)
        self.illegal += int(rec["chosen"] is None and rec["illegal"] is not None)
        self.none += int(rec["chosen"] is None)

    def out(self):
        n = max(1, self.n)
        return {"n": self.n, "acc": round(self.hit / n, 4),
                "acc_ci": wilson(self.hit, self.n),
                "head_top1": round(self.head_hit / n, 4),
                "ceiling_top5": round(self.ceil / n, 4),
                "named_but_illegal": self.illegal, "no_name": self.none}


def report(records, rows, elapsed, args, system):
    by_key = {(r["gameid"], r["move_idx"]): r for r in rows}
    overall = Tally()
    by_action = {0: Tally(), 1: Tally()}
    by_phase = {k: Tally() for k in SLICE_NAMES}
    # The mechanism, not just the score: an agent that agrees with the head's #1
    # on every row reproduces the head, and one that overrules on every row is a
    # different (worse) model. Splitting the rows says which of those happened.
    by_choice = {"deferred": Tally(), "overruled": Tally()}
    outside20 = 0
    pairs = {"both": 0, "api_only": 0, "head_only": 0, "neither": 0}
    parse, stops, kdist, tools = {}, {}, {}, {}
    for rec in records:
        row = by_key.get((rec["gameid"], rec["move_idx"]))
        if row is None:
            continue
        overall.add(rec, row)
        by_action[row["is_ban"]].add(rec, row)
        by_phase[row["phase"]].add(rec, row)
        if rec["chosen"] is not None:
            deferred = rec["chosen"] == row.get("top1_id")
            by_choice["deferred" if deferred else "overruled"].add(rec, row)
            if not deferred and rec["chosen"] not in row.get("top20", ()):
                outside20 += 1
        parse[rec["parse"]] = parse.get(rec["parse"], 0) + 1
        stops[rec["stop"]] = stops.get(rec["stop"], 0) + 1
        kdist[str(rec["k_max"])] = kdist.get(str(rec["k_max"]), 0) + 1
        for name, n in (rec["by_tool"] or {}).items():
            tools[name] = tools.get(name, 0) + int(n)
        api = rec["chosen"] == row["target"]
        head = row["rank_tool"] == 0
        pairs["both" if api and head else
              "api_only" if api else
              "head_only" if head else "neither"] += 1

    # The serving-path cost: the same head, asked with inferred lanes instead of
    # the corpus's true roles. Reported because the agent is scored against the
    # inferred one while the single-turn arms were scored against the true one.
    head_tool = sum(1 for r in rows if r["rank_tool"] == 0)
    head_batch = sum(1 for r in rows if r.get("rank_batch") == 0)
    both = sum(1 for r in rows if r["rank_tool"] == 0 and r.get("rank_batch") == 0)

    return {
        "checkpoint": Path(args.head).parent.name,
        "split": args.split, "val_games": args.val_games,
        "n_rows": len(records),
        "model": args.model, "max_steps": args.max_steps,
        "system_sha": hashlib.sha1(system.encode("utf-8")).hexdigest()[:8],
        "overall": overall.out(),
        "paired_vs_head": dict(pairs, p=mcnemar(pairs["api_only"],
                                                pairs["head_only"])),
        "by_action": {("ban" if k else "pick"): v.out()
                      for k, v in by_action.items()},
        "by_phase": {SLICE_NAMES[k]: v.out() for k, v in by_phase.items()},
        "by_choice": {k: v.out() for k, v in by_choice.items()},
        "overruled_outside_top20": outside20,
        "head_serving_cost": {
            "top1_batch": round(head_batch / max(1, len(rows)), 4),
            "top1_tool": round(head_tool / max(1, len(rows)), 4),
            # Rows where the *target* is the argmax under both paths - not a
            # path-agreement rate, and named so it cannot be read as one.
            "both_paths_hit_argmax": both,
        },
        "k_asked": dict(sorted(kdist.items(), key=lambda kv: -kv[1])),
        "tool_mix": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
        "stop": stops, "parse": parse,
        "mean_calls": round(sum(r["n_calls"] for r in records)
                            / max(1, len(records)), 2),
        "seconds": round(elapsed, 1),
        "reference": REFERENCE,
    }


def _print(rep):
    o = rep["overall"]
    print("\n%s  %d rows  (%s, %s)"
          % (rep["model"], rep["n_rows"], rep["split"], rep["checkpoint"]))
    print("  agent top-1      %6.2f %%  [%.2f, %.2f]"
          % (100 * o["acc"], 100 * o["acc_ci"][0], 100 * o["acc_ci"][1]))
    print("  head top-1       %6.2f %%   (tool path, same rows)"
          % (100 * o["head_top1"]))
    print("  head top-5 ceil  %6.2f %%" % (100 * o["ceiling_top5"]))
    p = rep["paired_vs_head"]
    print("  vs head  won %d / lost %d   exact McNemar p=%.2g"
          % (p["api_only"], p["head_only"], p["p"]))
    print("  picks %6.2f %%   bans %6.2f %%"
          % (100 * rep["by_action"]["pick"]["acc"],
             100 * rep["by_action"]["ban"]["acc"]))
    c = rep["by_choice"]
    print("  deferred to #1  %6.2f %% of rows, %6.2f %% right"
          % (100 * c["deferred"]["n"] / max(1, rep["n_rows"]),
             100 * c["deferred"]["acc"]))
    print("  overruled       %6.2f %% of rows, %6.2f %% right "
          "(%d of them outside the top 20)"
          % (100 * c["overruled"]["n"] / max(1, rep["n_rows"]),
             100 * c["overruled"]["acc"], rep["overruled_outside_top20"]))
    print("  k asked %s" % rep["k_asked"])
    print("  tools %s" % rep["tool_mix"])
    print("  stop %s" % rep["stop"])
    print("  parse %s   no name %d, named-but-illegal %d"
          % (rep["parse"], o["no_name"], o["named_but_illegal"]))
    print("  %.1f calls/row, %.0f s" % (rep["mean_calls"], rep["seconds"]))
    sc = rep["head_serving_cost"]
    print("  serving path: head top-1 batch %.2f %% -> tool %.2f %%"
          % (100 * sc["top1_batch"], 100 * sc["top1_tool"]))


# -------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="heldout", choices=["heldout", "rr"])
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--val-games", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--head", default=str(R.HEAD_CKPT))
    p.add_argument("--rerank", default="",
                   help="serve the 4B reranker in the next-move slot instead of "
                        "the head, so `predict_next_move` (the tool the agent "
                        "calls) returns its order. The head still proposes - "
                        "the reranker only re-orders the shortlist - and the "
                        "`--head` checkpoint is still what proposes, so the "
                        "batch column stays the head's own top-1 and the tool "
                        "column becomes the reranker's")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--model", default="", help="overrides LLM_MODEL")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--max-calls", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="prose answers; `deepseek-flash` needs 8192 because it "
                        "spends the budget on reasoning first")
    p.add_argument("--stats", default="",
                   help="override the accuracy the prompt claims, as "
                        "k=v[,k=v] (top1, top5, top20, games, patches). The "
                        "explained prompt teaches the agent how much to trust "
                        "the list, so serving a different model without "
                        "changing these numbers asks it to trust a 34 %% model "
                        "as if it were an 18 %% one. Part of the cache key.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--no-batch-baseline", dest="batch_baseline",
                   action="store_false", default=True)
    p.add_argument("--cache", default="")
    p.add_argument("--out", default="")
    p.add_argument("--only", default="", choices=["", "pick", "ban"],
                   help="restrict to one move kind - a ban has no lane, so a "
                        "mixed slice hides whether the lane tool did anything")
    p.add_argument("--trace", action="store_true",
                   help="keep the message transcript in the cache record")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from solo_queue.model.policy import infer

    if args.rerank:
        from pro.model.archive.llm.rerank_predictor import RerankPredictor
        pred = RerankPredictor(args.rerank, args.head, device=args.device)
        print("next-move:    %s (head %s proposes, adapter re-orders)"
              % (pred.label, Path(args.head).parent.name), flush=True)
    else:
        pred = infer.PolicyPredictor(args.head, device=args.device)
    rows, names, champs = build_rows(args, pred)
    print("split %s: %d games, %d states"
          % (args.split, args.val_games, len(rows)), flush=True)

    t0 = time.time()
    rows = tool_baseline(rows, pred, champs, names)
    print("baselines in %.1f s" % (time.time() - t0), flush=True)

    system = suggest.system()
    if args.stats:
        over = {}
        for part in args.stats.split(","):
            k, _, v = part.partition("=")
            over[k.strip()] = float(v) if "." in v else int(v)
        system = prompt_explained.system_prompt(over)
        print("prompt stats: %s" % ", ".join("%s=%.4g" % kv
                                             for kv in sorted(over.items())),
              flush=True)
    added = llm_client.load_dotenv()
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    info = llm_client.describe()
    args.model = info["model"]
    print("backend %s @ %s%s" % (info["model"], info["base_url"],
                                 "  (.env: %s)" % ", ".join(added) if added else ""),
          flush=True)

    if args.dry_run:
        print("\n=== SYSTEM (%d chars) ===\n%s" % (len(system), system))
        st = rows[0]["state"]
        display = DS.Store(champs, real=st).display
        print("\n=== FIRST USER MESSAGE ===\n%s\n\nQuestion: %s"
              % (st.render("real", display), QUESTION))
        print("\n=== TOOLS ===")
        for spec in Session(predictor=pred, champs=champs,
                            ablate=suggest.ABLATE).available():
            print(" -", spec["name"])
        return 0

    client = llm_client.client_from_env(temperature=args.temperature,
                                        max_tokens=args.max_tokens)
    if client is None:
        raise SystemExit("no LLM_API_KEY - set one in the environment or .env")

    locked = LockedHead(pred)
    cache_path = args.cache or (str(Path(args.out).with_suffix(""))
                                + ".responses.jsonl" if args.out else "")
    cache = _load_cache(cache_path)
    if cache:
        print("resuming: %d cached rows" % len(cache), flush=True)
    write_lock = threading.Lock()
    sink = open(cache_path, "a", encoding="utf-8") if cache_path else None

    def work(row):
        last = None
        for _ in range(max(1, args.retries)):
            rec = run_one(row, locked, champs, names, client, system, args,
                          cache, write_lock)
            if not rec.get("error") and rec["stop"] != "loop error":
                return rec
            last = rec
        return last

    records = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for rec in pool.map(work, rows):
            records.append(rec)
            if sink and not rec.get("cached"):
                with write_lock:
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sink.flush()
            if args.verbose:
                print("  %s m%-2d %-12s -> %s  [%s] %d calls"
                      % (rec["gameid"], rec["move_idx"],
                         (rec["stop"] or "")[:12],
                         (rec["answer"] or "").replace("\n", " ")[:70],
                         rec["parse"], rec["n_calls"]), flush=True)
    if sink:
        sink.close()

    rep = report(records, rows, time.time() - started, args, system)
    _print(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return rep


if __name__ == "__main__":
    main()
