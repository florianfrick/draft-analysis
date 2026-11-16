"""Ask for a decision, not a ranking.

## Why

`eval_rank_api.py` measures the shipped `/api/agent-rank` arm: it asks a hosted
model to *order* the head's top-k and reads the list back. On 500 test games
`deepseek-flash` ties the head (top-1 -0.80 pp, top-5 +0.40) and on 100
`deepseek-chat` echoes the head's #1 on 99 of them. The prompt ends with "if you
trust the head, echo its order", so the measured behaviour may be the
instruction rather than the model's judgement.

This asks the same model the same material as a **binary decision**: keep the
head's #1, or name one of the others. One line, one of two shapes, no list to
produce. Two things are then read off:

  * `raw` - what the model answered, as answered.
  * `gated` - the answer with a confidence gate: a SWITCH is only taken when the
    head's own margin `p1 - p2` is under `tau`, with `tau` tuned on half the
    games and reported on the other half. That is the "one-parameter finetune"
    the adapter would otherwise supply, and it is the number that would ship.

## Protocol

200 test games of the shipped split (`2022-full`, `--no-pin`), one prefix each,
`seed 0` - the first 200 of the same rows `eval_rank_api --limit 200` scores, so
the rank arm, the decision arm and the head are one paired comparison. Games are
split 50/50 by game id: `A` tunes `tau`, `B` is reported. The head's baseline is
recomputed on the same rows; nothing is quoted across corpora.

    python -m pro.model.agent.eval_keep_switch --limit 200 --model deepseek-chat

Cost, measured: ~2.2k prompt tokens and one call per row (no loop, no tools), so
200 rows is ~0.5M tokens - a fifth of the rank arm's per-row price.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import eval_rank_api as E                                  # noqa: E402
from pro.model.agent import draft_state as DS               # noqa: E402
from pro.model.agent import llm_client                     # noqa: E402

SYSTEM = (
    "You are a League of Legends draft analyst working on professional drafts. "
    "You are given a board and a ranked shortlist produced by a model. Your job "
    "is one decision: does the model's first choice stand, or does another "
    "champion on its list do the job better? Answer with one line and nothing "
    "else.")

# Two answer shapes, both one line. `KEEP` may name the champion; the parser
# accepts a bare KEEP because a model that answers "KEEP" has answered.
_ANSWER = re.compile(r"\b(KEEP|SWITCH)\b\s*:?\s*([A-Za-z'.\- *]{2,24})?",
                     re.IGNORECASE)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=200, help="games (= rows)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5,
                   help="shortlist shown; 5 matches the LoRA and the API arm")
    p.add_argument("--head", default=E.DEFAULT_HEAD)
    p.add_argument("--corpus", default="2022-full")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--min-patch-rows", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--out", default="")
    p.add_argument("--cache", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def question(head_top, kind, p1_rate):
    """The binary question, with the head's own numbers as the evidence.

    `p1_rate` is the head's measured top-1 on rows like this one (by phase, from
    the shared-val gate table). It is stated because the shipped prompt states it
    too - a model told nothing about the head's accuracy will read 18 % as a
    strong lead.
    """
    lines = ["The head model ranks these %ss for the side on the clock:" % kind]
    for i, h in enumerate(head_top, 1):
        lines.append("  %d. %s - %.1f%%" % (i, h["name"], 100 * h["p_head"]))
    lines.append("")
    lines.append("On rows like this one its first choice is the actual next "
                 "move about %.0f %% of the time." % (100 * p1_rate))
    lines.append("Do the board and the numbers agree, or is one of the others "
                 "the better move?")
    lines.append("")
    lines.append("Answer with exactly one line:")
    lines.append("  KEEP %s" % head_top[0]["name"])
    lines.append("or")
    lines.append("  SWITCH <champion>")
    lines.append("Nothing else - no explanation, no list.")
    return "\n".join(lines)


def parse_answer(text, cands, champs):
    """-> (index into `cands` or None, mode). `mode` records how it was read."""
    if not text or not str(text).strip():
        return None, "empty"
    tail = [ln for ln in str(text).strip().splitlines() if ln.strip()][-1]
    m = _ANSWER.search(tail)
    if not m:
        # No marker: fall back to the first champion named anywhere, which is
        # what a model that ignored the format would have produced.
        idx = E.suggest._first_named(str(text), champs)
        if idx is None:
            return None, "unparsed"
        return (0 if idx == cands[0][0] else _position(cands, idx)), "body"
    verb = m.group(1).upper()
    name = (m.group(2) or "").strip(" *_.")
    idx = E.suggest._resolve(name, champs) if name else None
    if verb == "KEEP":
        return 0, "keep"
    if idx is None:
        return None, "switch-unnamed"
    return _position(cands, idx), "switch"


def _position(cands, idx):
    for j, (cid, _p) in enumerate(cands):
        if cid == idx:
            return j
    return None


def key_of(ex, model, system, k):
    return "|".join([ex["gameid"], str(ex["move_idx"]), model, str(k),
                     hashlib.sha1((system or "").encode()).hexdigest()[:8]])


def wilson(k, n, z=1.96):
    return E.wilson(k, n, z)


def summarise(recs, tau_a=None):
    """-> the raw arm, the gated arm, and the head's baseline on these rows."""
    n = len(recs)
    out = {"n": n, "head": sum(1 for r in recs if r["head_in1"]) / n,
           "raw": sum(1 for r in recs if r["in1"]) / n,
           "parse": {}, "stop": {}}
    for r in recs:
        out["parse"][r["parse"]] = out["parse"].get(r["parse"], 0) + 1
    for k in (1, 5):
        a = sum(1 for r in recs if r["in%d" % k])
        b = sum(1 for r in recs if r["head_in%d" % k])
        a_only = sum(1 for r in recs if r["in%d" % k] and not r["head_in%d" % k])
        b_only = sum(1 for r in recs if r["head_in%d" % k] and not r["in%d" % k])
        lo, hi = wilson(a, n)
        out["delta_top%d" % k] = {
            "agent_minus_head": (a - b) / n, "ci": [lo, hi],
            "agent_only": a_only, "head_only": b_only,
            "p": E.mcnemar(a_only, b_only)}
    out["tokens"] = {"prompt": sum(r["prompt"] for r in recs),
                     "completion": sum(r["completion"] for r in recs),
                     "calls": sum(r["calls"] for r in recs)}
    if tau_a is not None:
        # A switch the gate rejects falls back to the head's #1, so the gated
        # answer is scored per row: take `in1` when the switch survives, the
        # head's own hit when it does not. Scoring the record's `gated_in1`
        # instead would count every switch as rejected and quietly report the
        # head's accuracy under a different name.
        def take(r):
            return r["switched"] and (r["p1"] - r["p2"]) < tau_a

        hits = [r["in1"] if take(r) else r["head_in1"] for r in recs]
        kept = [r for r in recs if take(r)]
        out["tau"] = tau_a
        out["gated"] = sum(hits) / n
        out["switches"] = sum(1 for r in recs if r["switched"])
        out["gated_switches"] = len(kept)
        out["switch_precision"] = (sum(1 for r in kept if r["in1"]) / len(kept)
                                   if kept else None)
        a_only = sum(1 for r, h in zip(recs, hits) if h and not r["head_in1"])
        b_only = sum(1 for r, h in zip(recs, hits) if r["head_in1"] and not h)
        out["gated_paired"] = {"agent_only": a_only, "head_only": b_only,
                               "p": E.mcnemar(a_only, b_only)}
    return out


def main(argv=None):
    args = parse_args(argv)
    from solo_queue.model.policy import infer

    examples, names, champs = E.build_examples(args)
    print("scoring %d rows (seed %d, k=%d)" % (len(examples), args.seed, args.k),
          flush=True)
    pred = E.LockedHead(infer.PolicyPredictor(args.head, device=args.device))
    tops = [E.head_top(pred, champs, ex["state"], k=20) for ex in examples]

    added = llm_client.load_dotenv()
    if args.model:
        import os
        os.environ["LLM_MODEL"] = args.model
    info = llm_client.describe()
    print("backend %s @ %s%s" % (info["model"], info["base_url"],
                                 "  (.env: %s)" % ", ".join(added)
                                 if added else ""), flush=True)

    rows = []
    # The head's measured top-1 by phase, shared val, 4,000 rows: a prompt that
    # says nothing about the head's accuracy lets an 18 % lead read as strong.
    rate = {"ban1": 0.15, "ban2": 0.10, "pick1": 0.26, "pick2": 0.19}
    spelling = {DS.key(n): n for n in names if n}
    display = lambda k: spelling.get(k, k)          # noqa: E731
    for ex, top in zip(examples, tops):
        kind = "ban" if ex["phase"].startswith("ban") else "pick"
        board = ex["state"].render("real", display)
        body = "\n".join([board, "", question(top[:args.k], kind,
                                              rate.get(ex["phase"], 0.19))])
        rows.append((ex, top, kind, body))
    if args.dry_run:
        print("\n=== SYSTEM ===\n%s" % SYSTEM)
        print("\n=== ROW 1 ===\n%s" % rows[0][3])
        return 0

    client = llm_client.client_from_env(temperature=args.temperature,
                                        max_tokens=args.max_tokens or None)
    if client is None:
        raise SystemExit("no LLM_API_KEY - set one in the environment or .env")

    if args.cache:
        cache_path = Path(args.cache)
    elif args.out:
        cache_path = Path(str(Path(args.out).with_suffix(""))
                          + ".responses.jsonl")
    else:
        cache_path = None
    cache = {}
    if cache_path and cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[rec["key"]] = rec
        print("resuming: %d cached answers" % len(cache), flush=True)

    lock = threading.Lock()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    sink = cache_path.open("a", encoding="utf-8") if cache_path else None

    def work(item):
        ex, top, kind, body = item
        key = key_of(ex, info["model"], SYSTEM, args.k)
        if key in cache:                     # resume: the record is the answer
            return cache[key]
        cands = [(champs.resolve(h["name"]), h["p_head"]) for h in top[:args.k]]
        usage = {"calls": 0, "prompt": 0, "completion": 0}
        text, err = "", None
        for attempt in range(args.retries + 1):
            try:
                reply = client([{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": body}])
                u = reply.get("usage") or {}
                usage["calls"] += 1
                usage["prompt"] += int(u.get("prompt_tokens") or 0)
                usage["completion"] += int(u.get("completion_tokens") or 0)
                text = (reply.get("content") or "").strip()
                err = None
                break
            except Exception as exc:                       # noqa: BLE001
                err = str(exc)
                time.sleep(1.5 * (attempt + 1))

        pos, mode = parse_answer(text, cands, champs)
        truth = champs.resolve(ex["truth"])
        head_idx = cands[0][0]

        def hit_at(p, n):
            return p is not None and p < n and cands[p][0] == truth

        chosen = cands[pos][0] if pos is not None else head_idx
        rec = {
            "key": key, "gameid": ex["gameid"], "move_idx": ex["move_idx"],
            "phase": ex["phase"], "patch": ex["patch"], "truth": ex["truth"],
            "answer": text[:120], "parse": mode if pos is not None else (
                "error: %s" % err if err else mode),
            "pos": pos, "switched": bool(pos not in (None, 0)),
            "p1": float(cands[0][1]), "p2": float(cands[1][1]),
            "in1": chosen == truth, "in5": truth in [c for c, _ in cands],
            "head_in1": head_idx == truth,
            "head_in5": truth in [c for c, _ in cands],
            "model": info["model"], "calls": usage["calls"],
            "prompt": usage["prompt"], "completion": usage["completion"],
        }
        with lock:
            cache[key] = rec
            if sink:
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        recs = list(pool.map(work, rows))
    if sink:
        sink.close()
    print("%d rows in %.0f s" % (len(recs), time.time() - t0), flush=True)

    # Tune tau on half the games, report on the other half. A game's prefixes
    # are not split: one prefix per game here, but the rule is the same.
    games = sorted({r["gameid"] for r in recs})
    tune = set(games[0::2])
    a = [r for r in recs if r["gameid"] in tune]
    b = [r for r in recs if r["gameid"] not in tune]

    def score(rows_, tau):
        """The same rule `summarise` applies: a rejected switch is the head."""
        def take(r):
            return r["switched"] and (r["p1"] - r["p2"]) < tau

        return sum(1 for r in rows_ if (r["in1"] if take(r) else r["head_in1"])) \
            / len(rows_)

    grid = [i / 200 for i in range(0, 101)]
    tau = max(grid, key=lambda t: score(a, t))
    print("tau tuned on %d games: %.3f (A %.2f %%, B %.2f %%)"
          % (len(a), tau, 100 * score(a, tau), 100 * score(b, tau)), flush=True)

    rep = {"protocol": {"limit": args.limit, "seed": args.seed, "k": args.k,
                        "model": info["model"], "head": args.head,
                        "corpus": args.corpus, "tau": tau,
                        "tune_games": len(a), "eval_games": len(b)},
           "tune": summarise(a), "eval": summarise(b, tau_a=tau),
           "records": recs}
    print("\n%d rows | head %.2f %% | raw %.2f %% | gated %.2f %% (tau %.3f)"
          % (len(recs), 100 * rep["eval"]["head"], 100 * rep["eval"]["raw"],
             100 * rep["eval"]["gated"], tau))
    print("  parse %s" % rep["eval"]["parse"])
    print("  switches %d raw, %d kept by the gate"
          % (rep["eval"]["switches"], rep["eval"]["gated_switches"]))
    print("  tokens %d prompt + %d completion over %d calls"
          % (rep["eval"]["tokens"]["prompt"], rep["eval"]["tokens"]["completion"],
             rep["eval"]["tokens"]["calls"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        print("wrote", args.out)
    return rep


if __name__ == "__main__":
    main()
