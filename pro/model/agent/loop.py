"""The bounded tool-calling loop, as a LangGraph state machine.

Two nodes, and the second one is the whole experiment:

    agent   ask the model, with the available tools advertised
    tools   run every call it asked for through `Session`

`Session` is not bypassed. The loop calls `session.call`, not the tool functions,
so every tool call the model makes lands in the same log as the scripted demo
and obeys the same ablation. That is the point of building the registry first:
`--ablate sql` here and `--ablate sql` in `demo.py` measure the same thing.

Three decisions worth naming:

**The board is never transcribed.** Every tool takes a state id, so there is
nothing to retype and nothing to disagree about. An earlier version accepted a
board as well as a state, filled an omitted field from the turn's board and
flagged a mismatched one; that machinery existed only to detect a model
mis-typing nine champion names, and it is gone with the argument.

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
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Annotated, TypedDict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import END, START, StateGraph          # noqa: E402

from pro.model.agent import draft_state, llm_client, model_tools, schema  # noqa: E402
from pro.model.agent.engine import Engine                    # noqa: E402
from pro.model.agent.session import Session                  # noqa: E402

MAX_STEPS = 8

# The shipped head, shared with `demo.py` and `handrun.py` so there is one place
# to change when the checkpoint is retrained.
DEFAULT_CKPT = str(ROOT / "pro" / "model" / "checkpoints"
                   / "nextpick-sharedsplitthrough-las-dk-c-t1-r" / "best.pt")

SYSTEM = """\
You are a draft assistant for professional League of Legends. You answer one \
question about one board, using tools.

The board lives in a state, not in this message. Never retype a board into a \
tool call: read it with `get_state` and pass the state id. `real` is the live \
board and is read-only; the tools refuse to change it.

Use a tool before you assert a number. A claim with no tool result behind it is \
a bug, not an opinion.

Which tool, in order of preference:
1. `get_state` before anything about the board, and again after every change. \
It returns the board, whose move it is, which role every pick fills with the \
corpus's confidence, and what each side still has to fill.
2. `predict_roles` when the question is what a team needs or where a champion \
plays. It is a table lookup over 46,915 pro drafts, so it cannot be wrong about \
the patch and it cannot disagree with itself.
3. `predict_next_move` for an open question ("what do we pick here?"). This is \
the model's own ranking and the most reliable thing you have.
4. `predict_next_move` with `role=` when the question names a role - "what is \
red's jungle here?". Add `side=` when you mean the side that is not on the \
clock. The numbers then count only champions that play that role, so they are \
NOT comparable with a plain call, and the result says whether the head or a \
masking model answered.
5. `sql` only when none of the above answers it: a rate over a patch window, a \
pairing's role record, a count of games. Always select the count next to a rate, \
always filter `split = 'train'`, and treat n under 30 as an anecdote.
6. `soloq_matchup` when the question is a specific two-champion matchup - "does \
Ahri beat Zed in mid?", "does Alistar pair well with this ADC?" - that the head \
cannot answer because it scores champions independently. Pass both names (or Riot \
ids), `kind='vs'` (against) or `'with'` (alongside), and optionally a `patch` or a \
`role` to scope it to a lane. The result gives the matchup's win rate, the \
champion's own baseline over the same patch scope, and `delta_win` - the marginal \
effect. It is SOLO QUEUE data, a different population from the pro drafts the head \
learned on: a complementary signal, not a reason to override the head on its own.

Comparing named champions is your job, not a tool's. There is no shortlist tool: \
read the head's ranking and compare from it. If the champion you are weighing is \
not in the list, ask again with a wider `k` - do not compare a name you thought \
of against a list you did not choose. A shortlist you wrote yourself can omit \
the head's top pick, and then a careful comparison is a confident answer to the \
wrong question.

Exploring a hypothetical ("what if blue takes Alistar here?"):
  clone_state(from_state="real")        -> a sandbox id
  apply_move(state=<id>, champion="Alistar")
  predict_next_move(state=<id>)         -> what red does next
  undo_move(state=<id>)                 -> try the other branch
Say which state your answer is about. A recommendation about a sandbox is not a \
recommendation about the live board.

Reading the numbers:
- `p_head` is a share of the legal pool (about 160 champions). `p_lane` is that \
champion's own share of its picks in that role. `p_in_lane` is a share of the \
champions that play one role. They are three different claims; never compare \
them, and always quote the rank when you quote a probability.
- `patch_known: false` means the model did not recognise the patch and answered \
as if patch did not matter. Say so if you rely on it.
- `soloq_matchup` returns `delta_win` (the matchup win rate minus the champion's \
own, over the same patch scope) and a `reliable` flag on n. Treat \
`reliable=false` as noise - most pairwise deviations are the champion's own \
strength, not a real edge - and never cite a soloQ rate as if it were pro data.
- The model cannot see which side is blue. Do not ask it to tell you.

Fearless series: `get_state` may list `fearless` champions. Those were picked in \
earlier games of this series. They are not on the board, they are not available, \
and they must not appear in an answer as a suggestion - the tools already mask \
them, so a name from that list means you did not read the state.

If a tool refuses, read the error and change the call. Do not retype it and \
hope. If it says a champion is already on the board, you misread the board - \
call `get_state`.

The models rank better than you do. Overrule a distribution only when a tool \
result says why - a matchup count, a role record - not because the pick feels \
wrong.

Answer in at most four sentences: name the champion, then the reason, citing \
the numbers the tools returned. No preamble, no summary of what you did.
"""

# Sent as a final user turn when the step budget runs out. Without it the turn
# ends on a tool call and `answer` is empty, which the panel renders as nothing
# at all - a bug from the reader's side, not a decision.
BUDGET_NOTE = """\
You have used every tool call this turn allows. Answer now from the results you \
already have, in at most four sentences, and name anything you could not check. \
Do not ask for another tool."""


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


# The board renderer lives with the state, which is what owns a board now; it is
# re-exported here because every caller and test reaches for `loop.render_board`.
render_board = draft_state.render_board


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


def build(session, client, system=SYSTEM, max_steps=MAX_STEPS):
    """-> compiled graph. `client` is callable(messages, tools=None) -> message."""
    tools = _openai_tools(session)

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
        if calls and state.get("steps", 0) >= max_steps:
            # Out of budget. The results it already has are usually enough, so
            # the choice is an empty reply or one last call with no tools.
            try:
                reply = client(msgs + [{"role": "user", "content": BUDGET_NOTE}],
                               tools=None)
            except llm_client.LLMError as exc:
                return {"answer": "", "stop": "step budget: %s" % exc,
                        "pending": []}
            return {"messages": [_to_wire(reply)], "pending": [],
                    "answer": reply.get("content") or "",
                    "stop": "step budget", "model": reply.get("model")}
        stop = "answered"
        if calls:
            stop = "tool calls"
        return {"messages": [_to_wire(reply)], "pending": calls,
                "answer": reply.get("content") or "", "stop": stop,
                "model": reply.get("model")}

    def tools_node(state):
        out = []
        for call in state.get("pending") or []:
            res = session.call(call.get("name"), **(call.get("args") or {}))
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
    return g.compile()


# ----------------------------------------------------------------------- run
def run(session, question, client=None, system=SYSTEM, max_steps=MAX_STEPS,
        env=None, raise_errors=False, state_id=None, seed=None):
    """One turn. -> a transcript dict, whether or not anything answered.

    `client=None` falls back to the environment, and no key is a recorded
    outcome rather than an exception.

    The board is read from the session's store and the prompt names the state.
    The board is still *rendered* - the model has to see it to reason about
    comps - but no tool takes it as an argument, so there is nothing to
    transcribe and nothing to disagree about.
    """
    state_id = state_id or draft_state.REAL_ID
    out = {"question": question, "answer": None, "stop": "no client",
           "model": None, "state": state_id,
           "tools_offered": session.names()}
    if getattr(session, "store", None) is None:
        out["stop"] = "no state"
        out["error"] = ("this session has no draft state, so there is nothing "
                        "to ask about - build it with a Store")
        out.update(summary=session.summary(), calls=list(session.log))
        return out
    state = session.store.get(state_id)
    # `store.display`, not the default: without it the board reads
    # "missfortune, jarvan iv" - the canonical keys - while every tool result
    # renders the corpus spelling. One board, two spellings, and the model is
    # left to work out that they are the same champion.
    state_text = state.render(state_id, getattr(session.store, "display", None))

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

    graph = build(session, client, system=system, max_steps=max_steps)
    question_msg = {"role": "user",
                    "content": state_text + "\n\nQuestion: " + question}
    # `seed` carries prior conversation (user/assistant text turns) the rerank
    # should honour - e.g. the chat the user has been having. It is prepended,
    # not appended, so the model sees the context before the ranking question.
    # Only well-formed {role, content} turns are kept; a provider rejects a
    # leading assistant message, and a stray tool turn would confuse the rerank.
    seed_msgs = []
    if seed:
        for m in seed:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") \
                    and m.get("content"):
                seed_msgs.append({"role": m["role"],
                                  "content": str(m["content"])})
    init = {"messages": seed_msgs + [question_msg],
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


def scripted_client(calls=("predict_roles",), answer=None,
                    champion="Xayah"):
    """A fake client for wiring checks: call a tool, then answer.

    Not a stub of the model's judgement - it always asks for the same thing in
    the same order. It exists so the graph and the ablation can be exercised
    without a key. `champion` is only used by the apply_move script, and
    defaults to one that is not on the demo board: naming a banned champion is
    refused by the tool, which is correct but makes a poor smoke test.
    """
    args_for = {"predict_roles": {"state": "real"},
                "get_state": {"state": "real"},
                "clone_state": {"from_state": "real", "name": "scripted"},
                "apply_move": {"state": "real", "champion": champion},
                "undo_move": {"state": "real"},
                "sql": {"query": "SELECT 1 AS one"}}
    queue = [{"id": "call_%d" % i, "name": n, "args": dict(args_for.get(n, {}))}
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


# ----------------------------------------------------------------------- ping
PING_TOOL = {"type": "function",
             "function": {"name": "sql",
                          "description": "Run one read-only SQL query.",
                          "parameters": {"type": "object",
                                         "properties": {"query": {"type": "string"}},
                                         "required": ["query"]}}}


def ping(client=None, env=None):
    """One cheap round trip, plus one request that needs tool calling.

    Worth its 30 lines because the three ways a provider is wrong all look
    identical from inside `run()`: a bad base URL, a retired model id and a key
    without tool-calling access each produce an empty answer or an HTTP error
    deep inside a 200-state batch. This tells them apart in five seconds, and it
    is the same call path `run` uses, so a pass here means the wiring is good.
    """
    env = os.environ if env is None else env
    out = dict(llm_client.describe(env), ok=False)
    client = client if client is not None else llm_client.client_from_env(
        env, temperature=0)
    if client is None:
        out["error"] = "no LLM_API_KEY"
        return out
    started = time.monotonic()
    try:
        reply = client([{"role": "user",
                         "content": "Reply with the single word: ok"}])
    except llm_client.LLMError as exc:
        out["error"] = str(exc)
        return out
    out["latency_ms"] = round(1000 * (time.monotonic() - started))
    out["answer"] = (reply.get("content") or "").strip()[:60]
    out["ok"] = bool(out["answer"])
    try:
        reply = client([{"role": "user",
                         "content": "Call sql with the query: SELECT 1 AS one"}],
                       tools=[PING_TOOL])
        out["tool_calls"] = [c["name"] for c in reply.get("tool_calls") or []]
    except llm_client.LLMError as exc:
        out["tool_calls"] = "error: %s" % exc
    return out


# ----------------------------------------------------------------------- cli
def _load(ckpt, db, ablate, max_calls, board=None, masker_ckpt=None):
    # The state store needs champion *names*, not weights, so it is built even
    # when no checkpoint is loaded - exploring a board should not require a model.
    names = model_tools.ChampIndex(schema.names())
    predictor = con = masker = None
    if Path(ckpt).exists():
        from solo_queue.model.policy.infer import PolicyPredictor
        predictor = PolicyPredictor(ckpt, device="cpu")
    else:
        print("no checkpoint at %s - the next-move tool is unavailable" % ckpt,
              file=sys.stderr)
    if masker_ckpt and Path(masker_ckpt).exists():
        from solo_queue.model.policy.infer import PolicyPredictor
        masker = PolicyPredictor(masker_ckpt, device="cpu")
    if Path(db).exists():
        con = schema.connect(db)
    else:
        print("no database at %s - run build_db.py first" % db, file=sys.stderr)
    store = draft_state.Store(names)
    store.attach(Engine(names, predictor=predictor, masker=masker,
                        patch=(board or {}).get("patch")))
    if board:
        store.set_real(draft_state.log_from_board(
            board.get("blue_picks"), board.get("bans"),
            red_first=bool(board.get("first_pick")), patch=board.get("patch"),
            bans_blue=board.get("bans_blue"), bans_red=board.get("bans_red"),
            unavailable=board.get("unavailable") or (), engine=store.engine))
    return Session(con=con, store=store, ablate=ablate, max_calls=max_calls)


def _print_ping(rep, from_env=()):
    for key in ("model", "base_url", "ok", "latency_ms", "answer",
                "tool_calls", "error"):
        if key in rep:
            print("  %-11s %s" % (key, rep[key]))
    added = [k for k in from_env if k.startswith("LLM_")]
    if added:
        print("  %-11s %s" % ("from .env", ", ".join(added)))
    if not rep["ok"]:
        print("\n  not usable. DeepSeek:  LLM_BASE_URL=https://api.deepseek.com"
              "  LLM_MODEL=deepseek-flash")
        print("  Gemini:            LLM_BASE_URL=https://generativelanguage"
              ".googleapis.com/v1beta/openai  LLM_MODEL=<an id from /models>")
    elif not rep.get("tool_calls"):
        print("\n  answers, but did not call a tool - the loop cannot use it.")
    return 0 if rep["ok"] else 1


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
    ap.add_argument("--env-file", default=None,
                    help="dotenv file to read (default: .env). Use a separate "
                         "file for this experiment so it does not fight the "
                         "frontend's model settings.")
    ap.add_argument("--ping", action="store_true",
                    help="one round trip against the configured provider, then exit")
    ap.add_argument("--scripted", action="store_true",
                    help="run a fake client, so no API key is needed")
    ap.add_argument("--json", action="store_true", help="print the transcript")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--masker-ckpt", default="",
                    help="a masking checkpoint, for role questions during the "
                         "ban phase where the next-move head has no answer")
    ap.add_argument("--db", default=str(schema.DB_PATH))
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    def names(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    # `.env` is where this repo keeps its keys; docker compose injects it for
    # the frontend, but a host-side script has to read it itself or the user
    # sets a key and is told there is none.
    from_env = llm_client.load_dotenv(args.env_file)
    if args.ping:
        return _print_ping(ping(), from_env)

    board = board_from_args(names(args.blue), names(args.red), names(args.bans),
                            args.first_pick, args.patch)
    session = _load(args.ckpt, args.db,
                    {x.strip() for x in args.ablate.split(",") if x.strip()},
                    args.max_calls, board=board,
                    masker_ckpt=args.masker_ckpt)
    print("tools offered:", ", ".join(session.names()) or "none")
    cfg = llm_client.describe()
    note = "" if cfg["has_key"] else "   (no key)"
    llm_added = [k for k in from_env if k.startswith("LLM_")]
    if llm_added:
        note += "   [from .env: %s]" % ", ".join(llm_added)
    print("model: %s at %s%s" % (cfg["model"], cfg["base_url"], note))
    client = scripted_client() if args.scripted else None
    out = run(session, args.question, client=client, max_steps=args.max_steps)
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
    print("\nanswer:", out["answer"] or "(none)")
    print("\nsummary:", out["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
