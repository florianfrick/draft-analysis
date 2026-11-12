# The agent's tool layer

The tools were built first, on purpose: what a tool returns, what it refuses and
what it records is the part that has to be right, and the loop that picks a tool
is the part most likely to change.

```
schema.py       build the SQLite corpus + the schema text the model is shown
sql_tool.py     one read-only query tool with mechanical guards
model_tools.py  predict_next_move / rank_candidates over the trained heads
session.py      the registry, the call log, the ablation switch
demo.py         a scripted turn, so the wiring is checkable without an LLM
llm_client.py   an OpenAI-compatible chat client with tool calling (urllib)
loop.py         the bounded tool-calling graph, over `Session`
mcp_server.py   the same three tools over the MCP stdio wire protocol
```

`llm_client` and `loop` are deliberately **not** imported by `__init__.py`: the
tool layer stays usable without `langgraph` installed.

## Why a SQL fallback at all

`frontend/tools.py` precomputes the aggregates I guessed a coach would want, and
`_probe_tool_power.py` showed the guess was mostly wrong: the per-champion cell
is 0-2 of 120 boards at a given move. A query tool moves that choice to the
model, which is the right place for it - provided the schema makes the correct
question easier to write than the wrong one.

## The schema is the prompt

Three properties of this corpus were measured, not assumed, and each one is a
silent-wrong-answer trap for a model that does not know them:

| trap | what happens | what the schema does |
|---|---|---|
| **Patch is not a time partition** | patches overlap by weeks (16.17 runs into 16.18), 16.19 and 16.21 are absent, 16.22 is 5 games inside 16.20's window | `patch_rank` is a *dense* rank over observed patches; `date_ordinal` is the recency axis, and the doc says so |
| **`move_idx` is draft order, not a slot** | joining two champions on `move_idx` returns 0 rows, which reads as "they never met" | the doc says join on `gameid`; the 0-row hint repeats it |
| **`split` is not optional** | an agent allowed to query val contaminates the evaluation that decides whether the agent works | `split` on every table; the tool hints when a query omits it |

`kind` is `'ban'`/`'pick'` and `side` is `'blue'`/`'red'` rather than 0/1, so
`WHERE kind = 1` returns nothing instead of something plausible. Champions are
joined on `champ.key` (`kaisa`), never on `name`, because `'Kai'Sa'` is a SQL
syntax error on the most common apostrophe in the list.

## The guards are mechanical

A wrong model answer is legible - it arrives with a rank attached. A wrong SQL
answer is silent: a valid query, a small n, a confident sentence. So: read-only
connection (`query_only`), one statement, `SELECT`/`WITH` only, 200-row cap with
a `truncated` flag, a 2 s watchdog that aborts mid-query, and **the query is
returned with every result** - without it a wrong answer cannot be debugged and
the unsupported-claim metric cannot be computed.

`hint` is the part that pays for itself in a loop: a flash model writes
`WHERE kind = 1` or `patch > '16.20'` often enough that the error has to teach.

## The model tier carries the judgment

`predict_next_move` answers "what should we pick" (the head's distribution, 50.5 %
top-5 on the shared val). `rank_candidates` answers "Kled or Olaf" - a k-way
comparison of a *named* set, which is what a coach asks and what the reranker
was trained for.

Three refusals, each for a measured reason:

- **Never a bare probability.** `p_head` is a share of the legal pool (164),
  `p_rerank` is a share of the candidate set (5). Every entry carries `rank` and
  the result carries a `note` saying which is which.
- **The patch is echoed.** Patch ids are 1-based rows into `patch_vocab`, and a
  label the checkpoint does not know becomes UNK - the head then answers
  patch-agnostically without saying so. `patch_known: false` is the only signal.
- **The board is validated.** A champion picked twice, or picked and banned,
  comes back as a refusal. An agent that learns to trust a distribution computed
  for an illegal board is worse than one with no tools.

The board is passed as `blue_picks` / `red_picks` in slot order, not one 10-slot
array: a single sparse list cannot say which side the third name belongs to.

## The log and the ablation

Every call is recorded with its arguments, latency, success and row count.
`mix()["sql_share"]` is the number the design turns on: **if more than half the
calls are `sql`, the high-level tier is under-specified, not the SQL tool
over-powered.**

`ablate={"sql"}` removes a tool from the advertised specs *and* refuses it at
call time. Both halves matter: hiding the spec tests the agent as shipped,
refusing the call catches the model that invents a tool name it was never
offered. Refusals are logged as `ablated`, not as errors - "it tried and could
not" is a result.

## The loop

Two nodes: `agent` asks the model with the available tools advertised, `tools`
runs every call it asked for **through `Session`**. That last word is the point -
the loop does not call `predict_next_move` directly, so an MCP call, a scripted
demo turn and a model-driven turn all land in the same log and obey the same
ablation. `--ablate sql` means the same thing in all three.

Three rules, each for a failure that would otherwise be invisible:

- **Omission is filled; disagreement is flagged.** Every model tool takes the
  whole board, so the model re-types nine champion names on every call. If it
  omits `blue_picks` it is being lazy and the turn's board is unambiguous, so the
  loop fills it. If it supplies a *different* board it has made a claim, and
  correcting that claim silently would hide the most likely failure of the whole
  architecture. `transcription.mismatched` records it and the tool answers the
  board it was actually asked about.
- **The step budget is a routing decision.** A graph that runs out of steps and
  raises `GraphRecursionError` loses the log. The router sends the last assistant
  message to `END` and records `stop="step budget"`.
- **No key is not a crash.** `client_from_env` returns `None`, `run` returns
  `stop="no client"` with the tools it would have offered.

The prompt is part of the design, not boilerplate: it ranks the tools by
reliability (`rank_candidates` → `predict_next_move` → `sql`), forbids comparing
`p_head` with `p_rerank`, and says the head cannot see side. `test_loop.py` pins
those three sentences, because they are the instructions that decide what gets
measured.

## The MCP server

`mcp` 2.3.0 replaced the decorator API the docs show (`Server` has
`add_request_handler`, not `@server.list_tools`), so `mcp_server.py` speaks the
wire protocol directly rather than pinning a version that cannot be exercised
here. A tools-only server needs four methods: `initialize`, `tools/list`,
`tools/call`, `ping`.

Two inherited behaviours: the ablation applies (`--ablate sql` starts a server
that neither advertises nor accepts `sql`), and **a tool failure is a result,
not a protocol error** - `isError: true` with the text of the refusal, because
the agent has to read it and retry. JSON-RPC errors are reserved for malformed
requests, which a model never sees.

The one hazard worth naming: **stdout is the wire.** Torch's import banner, a
stray print, a warning - any of them corrupts the stream and the client drops the
connection with no useful message. `main` takes the real stdout, reassigns
`sys.stdout` to stderr, and only then loads the checkpoint. Verified by piping a
session through a real subprocess, not only through `StringIO`.

## Running it

```bash
# build the corpus (~1 min; 46,915 games, 938,300 moves, 107 patches)
python pro/model/agent/build_db.py \
  --known-patches-from pro/model/checkpoints/nextpick-sharedsplitthrough-las-dk-c-t1-r/best.pt

python pro/model/agent/demo.py                  # scripted turn, all tools
python pro/model/agent/demo.py --ablate sql     # the same turn without SQL

python pro/model/agent/loop.py --scripted \
  --blue "Aatrox,Vi,Ahri" --red "Poppy,Lee Sin" --bans "Kled,Olaf" --patch 16.20
python pro/model/agent/loop.py --question "Kled or Olaf?" --blue "..." --red "..."

python pro/model/agent/mcp_server.py --self-test   # the protocol, no client
python pro/model/agent/mcp_server.py               # serve on stdio

pytest pro/model/agent/ -q                         # 60 tests
```

The loop reads `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT` - the
same contract as `frontend/llm.py`, so a transcript that names a model names the
one that answered. `--scripted` replaces the model with a fake client that calls
one tool and stops, which is enough to exercise the graph, the board filling and
the ablation without a key.

## What this does not answer

Whether an LLM driving these tools beats the head. That needs the 200-state
evaluation on the shared val, split by whether the truth was inside the head's
top-20: the ~13.65 % of states where the head is confidently wrong are the only
ones where the agent can add anything the reranker cannot. Aggregate top-1 will
look like a wash and hide exactly that.

What the loop measures today is the **mix**, not the accuracy: how many calls,
which tool, how often it fell back to SQL, whether it retyped the board. That is
the diagnostic for the design; the pick-accuracy number needs the 200-state eval,
and that is not written yet.
