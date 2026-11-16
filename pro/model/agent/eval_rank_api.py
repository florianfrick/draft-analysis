"""Score the SHIPPED rank arm (`/api/agent-rank`) on the shipped test split.

    python -m pro.model.agent.eval_rank_api --limit 500 --workers 8 \
        --out pro/model/agent/reports/rank_api_nov13_500.json

## Why this exists, next to `pro/model/archive/llm/eval_api_rerank.py`

That harness measures a *different* prompt: it predates `suggest.rank_question`
and asks the model to rank all candidates, with or without probabilities. The
column the frontend actually fills is `/api/agent-rank`, which asks
`suggest.rank_question(k)` - the head's legal shortlist with `(NN%)` - and
reads the answer back with `suggest.parse_ranking`. Both live in `suggest.py`,
and this harness imports all of them rather than restating them, so the button
and the measurement cannot drift (the same rule `eval_api_agent.py` follows).

It also scores on the **shipped split**, not the LLM track's `heldout`/`rr`:
`--corpus 2022-full` (46,939 drafts, freeze lifted) with `--no-pin` and the
fractional 85/10/5 within-patch tail. That is the only split the nov13
checkpoints have never trained on, and the only one whose numbers describe the
model that ships.

## What it measures

One prefix per game, drawn uniformly per game from the 20 (seeded, so the same
`--seed` scores the same rows). `--k 10` asks for ten names; top-1 and top-5
are read out of that answer, which is why a single run prices both.

The paired baseline is the **head's own order on the same rows** - not a
popularity table. `parse_ranking` fills any slot the model leaves unnamed from
that same shortlist, so a model that answers nothing still scores the head's
number; `parse` records which path produced the list, and the report counts
them. Without that count an empty answer would look like a reranker that ties.

## Cost

`--limit` examples, each a 5-step / 8-call loop. Measured at ~15.3k tokens and
~11 s per example on `deepseek-flash` (see `2026-11-13.md`), so 500 is ~7.7M
tokens and ~12 min at 8 workers. Every answer is cached in a JSONL next to
`--out`, keyed by game + move + model + prompt hash, so a re-run resumes and
never pays twice for the same row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pro" / "model" / "proonly"))

import common as PC                                        # noqa: E402
from pro.model.agent import draft_state as DS               # noqa: E402
from pro.model.agent import llm_client, loop, model_tools  # noqa: E402
from pro.model.agent import suggest                         # noqa: E402
from pro.model.agent.session import Session                 # noqa: E402

N_MOVES = 20
PHASES = {0: "ban1", 1: "pick1", 2: "ban2", 3: "pick2"}
DEFAULT_HEAD = str(ROOT / "pro/model/checkpoints"
                   "/nextpick-proonly-nov13-k5-through-lcp-df-sw/best.pt")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score the agent rank arm")
    p.add_argument("--limit", type=int, default=500,
                   help="examples (= games, one prefix each); 0 = the whole "
                        "test split")
    p.add_argument("--seed", type=int, default=0,
                   help="which of the 20 prefixes each game gets")
    p.add_argument("--k", type=int, default=10,
                   help="names to ask for; top-1/top-5 are read out of it")
    p.add_argument("--head", default=DEFAULT_HEAD)
    p.add_argument("--corpus", default="2022-full",
                   choices=sorted(PC.CORPORA))
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--min-patch-rows", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="", help="overrides LLM_MODEL")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--cache", default="")
    p.add_argument("--out", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="build the examples and print one prompt; no API call")
    return p.parse_args(argv)


# ------------------------------------------------------------------ examples
def build_examples(args):
    """-> (examples, names, champs). One (game, move) per test game."""
    class A:
        pass

    a = A()
    a.corpus = args.corpus
    a.val_frac = args.val_frac
    a.test_frac = args.test_frac
    a.min_patch_rows = args.min_patch_rows
    a.no_test = False
    a.no_pin = True
    a.val_scope = "all"
    a.val_split = "tail"
    a.limit_drafts = 0
    a.half_life = 0.0
    rows, splits, _patch_vocab, patch_index = PC.load_split(a, verbose=False)

    name_to_idx, _n = PC.PRO_PIPE.load_name_to_idx()
    names = [None] * len(name_to_idx)
    for name, i in name_to_idx.items():
        names[i] = name

    ds = PC.PRO_DATA.ProNextMoveDataset(
        rows, splits["test"], num_champs=173, patch_index=patch_index,
        prefix_mode="all")
    n_games = len(ds) // N_MOVES
    rng = random.Random(args.seed)
    moves = [rng.randrange(N_MOVES) for _ in range(n_games)]
    take = n_games if not args.limit else min(args.limit, n_games)

    examples = []
    for g in range(take):
        sample = ds[g * N_MOVES + moves[g]]
        examples.append({
            "gameid": sample["meta"]["gameid"],
            "move_idx": int(sample["move_idx"]),
            "phase": PHASES.get(int(sample["phase"]), str(sample["phase"])),
            "patch": sample["meta"]["patch"],
            "state": _state_of(sample, names),
            "truth": names[int(sample["target_champ"])],
        })
    return examples, names, model_tools.ChampIndex(names)


def _state_of(sample, names):
    """The row through the agent's own `DraftState` (same compaction the
    measured API arms use: the corpus is lane-indexed, the board is not)."""
    picks = []
    for team in (0, 1):
        lo, hi = team * 5, (team + 1) * 5
        filled = [names[int(c)] for c, p in
                  zip(sample["champ_ids"][lo:hi], sample["pad_mask"][lo:hi])
                  if not bool(p)]
        picks.extend(filled + [None] * (5 - len(filled)))
    return DS.log_from_board(
        picks,
        bans=[names[int(b)] for b in
              sample["ban_ids"][:int(sample["n_bans"])]],
        red_first=bool(int(sample["first_pick"])),
        patch=sample["meta"]["patch"],
        unavailable=[names[int(c)] for c in
                     sample["unavailable_mask"].nonzero().flatten().tolist()])


# ---------------------------------------------------------------------- run
class LockedHead:
    """Serialise `next_move` on one predictor across worker threads."""

    def __init__(self, pred):
        self._pred = pred
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._pred, name)

    def next_move(self, *a, **kw):
        with self._lock, torch.no_grad():
            return self._pred.next_move(*a, **kw)


def head_top(pred, champs, state, k=20):
    board = state.board()
    out = model_tools.predict_next_move(
        pred, champs, blue_picks=board["blue_picks"],
        red_picks=board["red_picks"], bans=board["bans"],
        first_pick=board["first_pick"], patch=board["patch"], k=k,
        unavailable=board["unavailable"])
    return [{"name": t["name"], "p_head": t["p_head"]} for t in out["top"]]


def key_of(ex, model, system, k):
    return "|".join([ex["gameid"], str(ex["move_idx"]), model, str(k),
                     hashlib.sha1(system.encode("utf-8")).hexdigest()[:8]])


def run_one(ex, pred, champs, client, system, args, head, usage_sink):
    store = DS.Store(champs, real=ex["state"])
    session = Session(predictor=pred, champs=champs, store=store,
                      ablate=suggest.RANK_ABLATE,
                      max_calls=suggest.RANK_CALLS)
    question = suggest.rank_question(args.k, "ban" if ex["phase"].startswith(
        "ban") else "pick")
    started = time.time()
    out = loop.run(session, question, client=client, system=system,
                   max_steps=suggest.RANK_STEPS, state_id="real")
    seconds = time.time() - started

    answer = out.get("answer") or ""
    parsed = suggest.parse_ranking(answer, champs, ex["state"], args.k, head)
    names10 = [s["name"] for s in parsed["suggestions"]]
    head_names = [h["name"] for h in head]
    truth = ex["truth"]

    def at(lst, n):
        return truth in lst[:n]

    usage = usage_sink
    return {
        "gameid": ex["gameid"], "move_idx": ex["move_idx"],
        "phase": ex["phase"], "patch": ex["patch"], "truth": truth,
        "top10": names10, "top1": names10[0] if names10 else None,
        "in1": bool(names10) and names10[0] == truth,
        "in5": at(names10, 5), "in10": at(names10, 10),
        "head_top1": head_names[0] if head_names else None,
        "head_in1": bool(head_names) and head_names[0] == truth,
        "head_in5": at(head_names, 5), "head_in10": at(head_names, 10),
        "truth_rank_head": next((i + 1 for i, n in enumerate(head_names)
                                 if n == truth), None),
        "parse": parsed["parse"], "stop": out.get("stop"),
        "model": out.get("model"), "seconds": round(seconds, 1),
        "calls": usage["calls"], "prompt": usage["prompt"],
        "completion": usage["completion"],
    }


# ------------------------------------------------------------------ metrics
def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def mcnemar(a_only, b_only):
    """Two-sided McNemar on paired outcomes -> p.

    Exact binomial rather than the chi-square approximation: the usual
    continuity-corrected form `(|a-b|-1)^2 / n` collapses to a non-1 value when
    the two arms tie (1-vs-1 comes out at 0.48), which reads as a win for
    neither arm yet still prints a "significant-ish" p. The exact tail is
    symmetric by construction, so a tie is exactly p = 1.
    """
    n = a_only + b_only
    if not n:
        return 1.0
    m = min(a_only, b_only)
    lg_n = math.lgamma(n + 1)
    tail = sum(math.exp(lg_n - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        - n * math.log(2.0)) for i in range(m + 1))
    return min(1.0, 2.0 * tail)


def summarise(recs):
    n = len(recs)
    out = {"n": n}
    for arm in ("", "head_"):
        for k in (1, 5, 10):
            key = "%sin%d" % (arm, k)
            hits = sum(1 for r in recs if r[key])
            lo, hi = wilson(hits, n)
            out[key] = {"acc": hits / n, "ci": [lo, hi], "hits": hits}
    for k in (1, 5, 10):
        a = sum(1 for r in recs if r["in%d" % k])
        b = sum(1 for r in recs if r["head_in%d" % k])
        a_only = sum(1 for r in recs if r["in%d" % k]
                     and not r["head_in%d" % k])
        b_only = sum(1 for r in recs if r["head_in%d" % k]
                     and not r["in%d" % k])
        out["delta_top%d" % k] = {
            "agent_minus_head": (a - b) / n, "agent_only": a_only,
            "head_only": b_only, "p": mcnemar(a_only, b_only)}
    out["parse"] = {}
    for r in recs:
        out["parse"][r["parse"]] = out["parse"].get(r["parse"], 0) + 1
    out["stop"] = {}
    for r in recs:
        out["stop"][str(r["stop"])] = out["stop"].get(str(r["stop"]), 0) + 1
    out["tokens"] = {"prompt": sum(r["prompt"] for r in recs),
                     "completion": sum(r["completion"] for r in recs),
                     "calls": sum(r["calls"] for r in recs)}
    out["seconds"] = round(sum(r["seconds"] for r in recs), 1)
    out["by_phase"] = {}
    for ph in sorted({r["phase"] for r in recs}):
        rows = [r for r in recs if r["phase"] == ph]
        out["by_phase"][ph] = {
            "n": len(rows),
            "top1": sum(1 for r in rows if r["in1"]) / len(rows),
            "top5": sum(1 for r in rows if r["in5"]) / len(rows),
            "head_top1": sum(1 for r in rows if r["head_in1"]) / len(rows),
            "head_top5": sum(1 for r in rows if r["head_in5"]) / len(rows)}
    return out


def print_report(rep):
    print("\n%d examples | agent %s | head-only baseline on the same rows"
          % (rep["n"], rep.get("model", "")))
    print("  %-6s %8s %8s %8s" % ("top", "agent", "head", "delta"))
    for k in (1, 5, 10):
        a = rep["in%d" % k]["acc"]
        h = rep["head_in%d" % k]["acc"]
        d = rep["delta_top%d" % k]
        print("  top%-3d %7.2f%% %7.2f%% %+7.2f pp  (agent-only %d, head-only "
              "%d, p %.3g)" % (k, 100 * a, 100 * h, 100 * (a - h),
                               d["agent_only"], d["head_only"], d["p"]))
    print("  parse %s | stop %s" % (rep["parse"], rep["stop"]))
    print("  tokens: %d prompt + %d completion over %d calls"
          % (rep["tokens"]["prompt"], rep["tokens"]["completion"],
             rep["tokens"]["calls"]))
    for ph, r in rep["by_phase"].items():
        print("  %-6s n=%-4d agent %5.1f/%5.1f  head %5.1f/%5.1f (top1/top5)"
              % (ph, r["n"], 100 * r["top1"], 100 * r["top5"],
                 100 * r["head_top1"], 100 * r["head_top5"]))


# --------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    from solo_queue.model.policy import infer

    examples, names, champs = build_examples(args)
    print("test split: %d games, scoring %d (seed %d, k=%d)"
          % (len(examples), len(examples), args.seed, args.k), flush=True)

    pred = LockedHead(infer.PolicyPredictor(args.head, device=args.device))
    heads = [head_top(pred, champs, ex["state"]) for ex in examples]

    system = suggest.system()
    added = llm_client.load_dotenv()
    if args.model:
        import os
        os.environ["LLM_MODEL"] = args.model
    info = llm_client.describe()
    print("backend %s @ %s%s" % (info["model"], info["base_url"],
                                 "  (.env: %s)" % ", ".join(added)
                                 if added else ""), flush=True)

    if args.dry_run:
        print("\n=== SYSTEM (%d chars) ===\n%s" % (len(system), system[:1200]))
        print("\n=== FIRST USER MESSAGE ===\n%s\n\nQuestion: %s"
              % (examples[0]["state"].render("real", None),
                 suggest.rank_question(
                     args.k, "ban" if examples[0]["phase"].startswith("ban")
                     else "pick")))
        return 0

    client = llm_client.client_from_env(temperature=args.temperature,
                                        max_tokens=args.max_tokens or None)
    if client is None:
        raise SystemExit("no LLM_API_KEY - set one in the environment or .env")

    cache_path = Path(args.cache or (str(Path(args.out).with_suffix(""))
                                     + ".responses.jsonl"))
    cache = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[rec["key"]] = rec
        print("resuming: %d cached answers" % len(cache), flush=True)

    todo = [(i, ex) for i, ex in enumerate(examples)
            if key_of(ex, info["model"], system, args.k) not in cache]
    print("%d to answer, %d cached" % (len(todo), len(examples) - len(todo)),
          flush=True)

    lock = threading.Lock()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sink = cache_path.open("a", encoding="utf-8")
    done = [0]
    t0 = time.time()

    def work(item):
        i, ex = item
        key = key_of(ex, info["model"], system, args.k)
        usage = {"calls": 0, "prompt": 0, "completion": 0}

        def counting(messages, tools=None):
            out = client(messages, tools)
            u = out.get("usage") or {}
            usage["calls"] += 1
            usage["prompt"] += int(u.get("prompt_tokens") or 0)
            usage["completion"] += int(u.get("completion_tokens") or 0)
            return out

        counting.model = getattr(client, "model", info["model"])
        last = None
        for attempt in range(args.retries + 1):
            try:
                rec = run_one(ex, pred, champs, counting, system, args,
                              heads[i], usage)
                break
            except Exception as exc:                       # noqa: BLE001
                last = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            rec = {"gameid": ex["gameid"], "move_idx": ex["move_idx"],
                   "phase": ex["phase"], "patch": ex["patch"],
                   "truth": ex["truth"], "top10": [], "top1": None,
                   "in1": False, "in5": False, "in10": False,
                   "head_top1": heads[i][0]["name"] if heads[i] else None,
                   "head_in1": False, "head_in5": False, "head_in10": False,
                   "truth_rank_head": None, "parse": "error",
                   "stop": "error: %s" % last, "model": info["model"],
                   "seconds": 0.0, "calls": 0, "prompt": 0, "completion": 0}
        rec["key"] = key
        with lock:
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(todo):
                print("  %d/%d  (%.0f s)" % (done[0], len(todo),
                                             time.time() - t0), flush=True)
        return rec

    fresh = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(work, todo):
            fresh.append(rec)
    sink.close()

    by_key = dict(cache)
    for rec in fresh:
        by_key[rec["key"]] = rec
    keys = [key_of(ex, info["model"], system, args.k) for ex in examples]
    recs = [by_key[k] for k in keys if k in by_key]
    ok = [r for r in recs if r.get("parse") != "error"]
    if ok:
        recs = ok

    rep = summarise(recs)
    rep.update(protocol={
        "corpus": args.corpus, "split": "test (within-patch 85/10/5, --no-pin)",
        "n_games": len(examples), "seed": args.seed, "k": args.k,
        "head": args.head, "model": info["model"],
        "config": suggest.rank_config(), "system": "explained",
        "prefix": "one per game, uniform 0..19"})
    print_report(rep)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"report": rep, "records": recs}, f, indent=1)
        print("wrote", args.out)
    return rep


if __name__ == "__main__":
    main()
