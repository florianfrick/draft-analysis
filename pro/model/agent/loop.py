"""The bounded tool-calling loop, as a LangGraph state machine.

Two nodes, and the second one is the whole experiment:

    agent   ask the model, with the available tools advertised
    tools   run every call it asked for through `Session`

`Session` is not bypassed. The loop calls `session.call`, not the tool functions,
so every tool call the model makes lands in the same log as the scripted demo
and obeys the same ablation. That is the point of building the registry first:
`--ablate sql` here and `--ablate sql` in `demo.py` measure the same thing.

Three decisions worth naming:

**Omission is filled; disagreement is flagged.** Every model tool takes the whole
board, so the model has to re-type nine champion names on every call. If it omits
`blue_picks` it is being lazy and the turn's board is unambiguous, so the loop
fills it in. If it supplies a *different* board it has made a claim, and
silently correcting that claim would hide the most likely failure of this whole
architecture - so the mismatch is recorded in `out["transcription"]` and the
tool answers the board it was actually asked about.

**The step budget is a routing decision, not an exception.** A graph that runs
out of steps and raises `GraphRecursionError` loses the log. The router sends the
last assistant message to END and records why.

**No key is not a crash.** `client_from_env` returns None, `run` returns a
transcript with `stop="no client"` and the tools it would have offered. A run
without a key still checks the wiring.

    python pro/model/agent/loop.py --question "Kled or Olaf?" \
        --blue "Aatrox,Vi,Ahri" --red "Poppy,Lee Sin" --bans "Kled,Olaf,Vi"
    python pro/model/agent/loop.py --scripted          # no key needed
    python pro/model/agent/loop.py --scripted --ablate sql
"""
from __future__ import annotations

import argparse
import json
import operator
import sys
import warnings
from pathlib import Path
from typing import Annotated, TypedDict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import END, START, StateGraph          # noqa: E402

from pro.model.agent import llm_client, model_tools, schema  # noqa: E402
from pro.model.agent.session import Session                  # noqa: E402

MAX_STEPS = 8

SYSTEM = """\
You are a draft assistant for professional League of Legends. You answer one \
question about one board, using tools.

Use a tool before you assert a number. A claim with no tool result behind it is \
a bug, not an opinion.

Which tool, in order of preference:
1. `rank_candidates` when the question names champions ("Kled or Olaf?"). It was \
trained for exactly that question and it is the most reliable thing you have.
2. `predict_next_move` when the question is open ("what do we pick here?").
3. `sql` only when neither answers it: a rate over a patch window, a pairing's \
lane record, a count of games. Always select the count next to a rate, always \
filter `split = 'train'`, and treat n under 30 as an anecdote.

Reading the numbers:
- `p_head` is a share of the legal pool (about 160 champions). `p_rerank` is a \
share of the candidate set you named. They are different claims; never compare \
them, and always quote the rank when you quote a probability.
- `patch_known: false` means the model did not recognise the patch and answered \
as if patch did not matter. Say so if you rely on it.
- The model cannot see which side is blue. Do not ask it to tell you.

If a tool refuses, read the error and change the call. Do not retype it and \
hope. If it says a champion is already on the board, you misread the board.

The models rank better than you do. Overrule a distribution only when a tool \
result says why - a matchup count, a lane record - not because the pick feels \
wrong.

Answer in at most four sentences: name the champion, then the reason, citing \
the numbers the tools returned. No preamble, no summary of what you did.
"""


class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    pending: list
    steps: int
    answer: str
    stop: str
    model: str


# --------------------------------------------------------------------- board
def board_from_args(blue=(), red=(), bans=(), first_pick=0, patch=None):
    return {"blue_picks": list(blue or []), "red_picks": list(red or []),
            "bans": list(bans or []), "first_pick": int(first_pick or 0),
            "patch": patch}


def render_board(board):
    """The board as the model reads it. Mirrors what a user would type."""
    def names(xs):
        return ", ".join(xs) if xs else "(none yet)"
    return ("Board:\n"
            "  blue picks: %s\n"
            "  red picks:  %s\n"
            "  bans:       %s\n"
            "  first pick: %s\n"
            "  patch:      %s"
            % (names(board.get("blue_picks")), names(board.get("red_picks")),
               names(board.get("bans")),
               "blue" if board.get("first_pick") else "red",
               board.get("patch") or "not given"))


def _keys(xs):
    return [model_tools._norm(x) for x in (xs or [])]


def fill_board(board, args):
    """Fill omitted board fields from the turn's board; flag disagreements.

    -> (args, filled, mismatched). Both lists are field names. See the module
    docstring for why omission and disagreement are treated differently.
    """
    out = dict(args or {})
    filled, bad = [], []
    for field in ("blue_picks", "red_picks", "bans", "patch", "first_pick"):
        want = board.get(field)
        have = out.get(field)
        if have in (None, [], "") and want not in (None, [], ""):
            out[field] = want
            filled.append(field)
            continue
        if have in (None, [], "") or want in (None, [], ""):
            continue
        if field in ("blue_picks", "red_picks"):
            same = _keys(have) == _keys(want)          # slot order is semantic
        elif field == "bans":
            same = sorted(_keys(have)) == sorted(_keys(want))
        elif field == "patch":
            same = str(have).strip() == str(want).strip()
        else:
            same = int(have or 0) == int(want or 0)
        if not same:
            bad.append(field)
    return out, filled, bad


# --------------------------------------------------------------------- graph
def _openai_tools(session):
    return [{"type": "function",
             "function": {"name": s["name"], "description": s["description"],
                          "parameters": s["parameters"]}}
            for s in session.available()]


def _to_wire(reply):
    """Normalised assistant message -> the wire format the next request needs."""
    msg = {"role": "assistant", "content": reply.get("content") or ""}
    if reply.get("tool_calls"):
        msg["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c["args"], ensure_ascii=False)}}
            for c in reply["tool_calls"]]
    return msg


def build(session, client, system=SYSTEM, max_steps=MAX_STEPS, board=None):
    """-> compiled graph. `client` is callable(messages, tools=None) -> message."""
    tools = _openai_tools(session)
    board = board or {}
    seen = {"filled": [], "mismatched": []}

    def agent(state):
        msgs = [{"role": "system", "content": system}] + state["messages"]
        try:
            reply = client(msgs, tools=tools or None)
        except llm_client.LLMError as exc:
            # `stop` carries the reason because the state schema has no error
            # channel, and a state key the schema does not declare is a bug.
            return {"answer": "", "stop": "llm error: %s" % exc,
                    "pending": []}
        calls = reply.get("tool_calls") or []
        stop = "answered"
        if calls:
            stop = ("step budget" if state.get("steps", 0) >= max_steps
                    else "tool calls")
        return {"messages": [_to_wire(reply)], "pending": calls,
                "answer": reply.get("content") or "", "stop": stop,
                "model": reply.get("model")}

    def tools_node(state):
        out = []
        for call in state.get("pending") or []:
            args, filled, bad = fill_board(board, call.get("args"))
            if filled:
                seen["filled"] += ["%s.%s" % (call.get("name"), f)
                                   for f in filled]
            if bad:
                seen["mismatched"] += ["%s.%s" % (call.get("name"), f)
                                       for f in bad]
            res = session.call(call.get("name"), **args)
            out.append({"role": "tool", "tool_call_id": call.get("id"),
                        "name": call.get("name"),
                        "content": json.dumps(res, ensure_ascii=False,
                                              default=str)})
        return {"messages": out, "pending": [],
                "steps": state.get("steps", 0) + 1}

    def route(state):
        if not (state.get("pending") or []):
            return "done"
        if state.get("steps", 0) >= max_steps:
            return "done"
        return "tools"

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "done": END})
    g.add_edge("tools", "agent")
    compiled = g.compile()
    compiled.transcription = seen
    return compiled


# ----------------------------------------------------------------------- run
def run(session, question, client=None, board=None, system=SYSTEM,
        max_steps=MAX_STEPS, env=None, raise_errors=False):
    """One turn. -> a transcript dict, whether or not anything answered.

    `client=None` falls back to the environment, and no key is a recorded
    outcome rather than an exception.
    """
    board = board or {}
    out = {"question": question, "answer": None, "stop": "no client",
           "model": None, "tools_offered": session.names(),
           "transcription": {"filled": [], "mismatched": []}}
    client = client if client is not None else llm_client.client_from_env(env)
    if client is None:
        cfg = llm_client.describe(env)
        out["error"] = ("no LLM_API_KEY in the environment - this run would "
                        "have used %s at %s. Set LLM_API_KEY (and LLM_BASE_URL "
                        "/ LLM_MODEL if the provider is not the default), or "
                        "use --scripted to exercise the graph without one."
                        % (cfg["model"], cfg["base_url"]))
        out.update(summary=session.summary(), calls=list(session.log))
        return out

    graph = build(session, client, system=system, max_steps=max_steps,
                  board=board)
    init = {"messages": [{"role": "user",
                          "content": render_board(board) + "\n\nQuestion: "
                                     + question}],
            "pending": [], "steps": 0, "answer": "", "stop": ""}
    try:
        final = graph.invoke(init, {"recursion_limit": 2 * max_steps + 6})
    except Exception as exc:  # noqa: BLE001 - a batch run must not lose the log
        if raise_errors:
            raise
        out.update(stop="loop error", error="%s: %s" % (type(exc).__name__, exc),
                   summary=session.summary(), calls=list(session.log))
        return out

    stop = final.get("stop") or "answered"
    if stop == "answered" and (final.get("messages") or [{}])[-1].get("tool_calls"):
        stop = "no answer (ended on a tool call)"
    out.update(answer=final.get("answer") or None, stop=stop,
               model=final.get("model"),
               transcription=graph.transcription,
               summary=session.summary(), calls=list(session.log),
               messages=_trim_messages(final.get("messages") or []))
    return out


def _trim_messages(messages, limit=700):
    """Keep the transcript readable; a SQL result is not a log entry."""
    out = []
    for m in messages:
        text = m.get("content") or ""
        if not isinstance(text, str):
            text = json.dumps(text, default=str)
        row = {"role": m.get("role"),
               "content": text if len(text) <= limit else text[:limit] + "..."}
        if m.get("tool_calls"):
            row["tool_calls"] = ["%s(%s)" % (c["function"]["name"],
                                             c["function"]["arguments"])
                                 for c in m["tool_calls"]]
        if m.get("name"):
            row["name"] = m["name"]
        out.append(row)
    return out


def scripted_client(calls=("rank_candidates",), answer=None,
                    candidates=("Xayah", "Jinx", "Ashe")):
    """A fake client for wiring checks: call a tool, then answer.

    Not a stub of the model's judgement - it always asks for the same thing in
    the same order. It exists so the graph, the board filling and the ablation
    can be exercised without a key. `candidates` defaults to three champions
    that are not on the demo board; naming a banned champion is refused by the
    tool, which is correct but makes a poor smoke test.
    """
    queue = [{"id": "call_%d" % i, "name": n,
              "args": {"candidates": list(candidates)}
              if n == "rank_candidates" else {"query": "SELECT 1 AS one"}}
             for i, n in enumerate(calls)]

    def call(messages, tools=None):
        if queue:
            c = queue.pop(0)
            offered = [t["function"]["name"] for t in (tools or [])]
            if c["name"] not in offered:
                # What a real model does when a tool is missing: it invents one
                # or gives up. Giving up is the cleaner thing to test.
                return {"role": "assistant", "content": "no tool for that",
                        "tool_calls": [], "model": "scripted"}
            return {"role": "assistant", "content": "", "tool_calls": [c],
                    "model": "scripted"}
        text = answer or ("The tools returned %d result(s); no answer was "
                          "written." % sum(1 for m in messages
                                           if m.get("role") == "tool"))
        return {"role": "assistant", "content": text, "tool_calls": [],
                "model": "scripted"}

    return call


# ----------------------------------------------------------------------- cli
def _load(ckpt, db, ablate, max_calls):
    predictor = champs = con = None
    if Path(ckpt).exists():
        from solo_queue.model.policy.infer import PolicyPredictor
        predictor = PolicyPredictor(ckpt, device="cpu")
        champs = model_tools.ChampIndex(schema.names())
    else:
        print("no checkpoint at %s - model tools unavailable" % ckpt,
              file=sys.stderr)
    if Path(db).exists():
        con = schema.connect(db)
    else:
        print("no database at %s - run build_db.py first" % db, file=sys.stderr)
    return Session(con=con, predictor=predictor, champs=champs,
                   ablate=ablate, max_calls=max_calls)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Kled or Olaf here?")
    ap.add_argument("--blue", default="")
    ap.add_argument("--red", default="")
    ap.add_argument("--bans", default="")
    ap.add_argument("--first-pick", type=int, default=1)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--ablate", default="")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--max-calls", type=int, default=24)
    ap.add_argument("--scripted", action="store_true",
                    help="run a fake client, so no API key is needed")
    ap.add_argument("--json", action="store_true", help="print the transcript")
    ap.add_argument("--ckpt", default=str(ROOT / "pro" / "model" / "checkpoints"
                                          / "nextpick-sharedsplitthrough-las-dk-c-t1-r"
                                          / "best.pt"))
    ap.add_argument("--db", default=str(schema.DB_PATH))
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    def names(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    board = board_from_args(names(args.blue), names(args.red), names(args.bans),
                            args.first_pick, args.patch)
    # `.env` is where this repo keeps its keys; docker compose injects it for
    # the frontend, but a host-side script has to read it itself or the user
    # sets a key and is told there is none.
    from_env = llm_client.load_dotenv()
    session = _load(args.ckpt, args.db,
                    {x.strip() for x in args.ablate.split(",") if x.strip()},
                    args.max_calls)
    print("tools offered:", ", ".join(session.names()) or "none")
    cfg = llm_client.describe()
    note = "" if cfg["has_key"] else "   (no key)"
    llm_added = [k for k in from_env if k.startswith("LLM_")]
    if llm_added:
        note += "   [from .env: %s]" % ", ".join(llm_added)
    print("model: %s at %s%s" % (cfg["model"], cfg["base_url"], note))
    client = scripted_client() if args.scripted else None
    out = run(session, args.question, client=client, board=board,
              max_steps=args.max_steps)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    print("\nstop:", out["stop"], "| model:", out["model"] or "-")
    if out.get("error"):
        print("error:", out["error"])
    for i, c in enumerate(out["calls"]):
        print("  [%d] %s(%s) -> %s %s"
              % (i, c["name"], ", ".join("%s=%s" % (k, v[:40])
                                         for k, v in c["args"].items()),
                 "ok" if c["ok"] else "ERROR", c.get("n_rows", "")))
        if not c["ok"]:
            print("      %s" % (c.get("error") or "")[:200])
    if out["transcription"]["filled"] or out["transcription"]["mismatched"]:
        print("  board filled:", out["transcription"]["filled"] or "-")
        print("  board mismatched:", out["transcription"]["mismatched"] or "-")
    print("\nanswer:", out["answer"] or "(none)")
    print("\nsummary:", out["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
