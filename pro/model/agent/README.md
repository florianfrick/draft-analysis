# The agent's tool layer

The tools were built first, on purpose: what a tool returns, what it refuses and
what it records is the part that has to be right, and the loop that picks a tool
is the part most likely to change.

```
schema.py       build the SQLite corpus + the schema text the model is shown
sql_tool.py     one read-only query tool with mechanical guards
model_tools.py  predict_next_move / rank_candidates over the trained heads
draft_state.py  the draft as an immutable move log, and a store of them
state_tools.py  get / clone / apply / undo / list, as tools
session.py      the registry, the call log, the ablation switch
demo.py         a scripted turn, so the wiring is checkable without an LLM
llm_client.py   an OpenAI-compatible chat client with tool calling (urllib)
loop.py         the bounded tool-calling graph, over `Session`
mcp_server.py   the same tools over the MCP stdio wire protocol
handrun.py      stratified val sample, answered blind, scored after
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

**The watchdog has to say it fired.** An aborted query is not an empty one, and
the two lead to opposite conclusions - "the corpus has nothing" versus "I could
not check". The first version let `interrupted` surface as an ordinary SQL error
and the reply said the evidence did not exist; it now returns `timed_out: true`
and a sentence naming the budget.

**The database is copied off the bind mount.** `schema.connect` copies the
corpus to `LOL_DB_CACHE` (docker-compose sets it to `/tmp`) and opens the copy,
because gRPC-FUSE charges about a millisecond per random 4 KB read and every
index probe is one. Six representative queries - a first-pick frequency, a role
win rate, a patch window, a self-join, a ban rate, a role split:

| where | total |
| --- | --- |
| bind mount, as shipped | 24.05 s |
| container-local disk, same file | 0.46 s |

52x, with identical SQL. The 2 s deadline was firing on queries the corpus
answers in milliseconds, so the tool looked broken for exactly the questions it
exists to answer. Adding indexes does **not** fix it - on the mount it makes
things worse, because a non-covering index turns one sequential scan into
hundreds of thousands of random reads (a covering index that made one query
instant made another 270x slower by attracting a plan it was not written for).
On local disk even a full scan is cheap enough that no index earns its place.
The copy is checked by size+mtime and replaced atomically, so a rebuilt corpus
is picked up on the next connection and two processes cannot read a half-written
file.

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
  message to `END` and records `stop="step budget"`. It also does not end on the
  refused tool call: that left `answer` empty and the panel showed an empty
  bubble. One last call is made with `tools=None` plus a note saying the budget
  is spent, so the model answers from the results it already has. (Measured on a
  real turn: 8 steps of SQL, no prose before this, a paragraph after.)
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

pytest pro/model/agent/ -q                         # 96 tests
```

The loop reads `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT` - the
same contract as `frontend/llm.py`, so a transcript that names a model names the
one that answered. It also reads `.env` from the repo root, without overriding
the shell, because that is where this repo keeps its keys and docker compose only
injects them for the frontend: a host-side script that ignored the file would
tell a user with a key that there is none. The banner prints the model, the base
URL and whether a key was found, so a misconfigured run is visible before it
spends a call.

`--scripted` replaces the model with a fake client that calls one tool and stops,
which is enough to exercise the graph, the board filling and the ablation without
a key.

## Which provider

The client is OpenAI-compatible and nothing else in the loop knows or cares, so
the provider is three environment variables. The defaults are DeepSeek
(`https://api.deepseek.com`, `deepseek-flash`) because that is the model this
experiment was specified against - `deepseek-flash` is served by
DeepSeek-V4.1-Flash-1017. **`deepseek-v4.1-flash` does not exist**; the obvious
guess 404s, and the legacy `deepseek-v4-flash` still resolves to the same model.

```bash
python pro/model/agent/loop.py --ping            # one round trip, then exit
python pro/model/agent/loop.py --ping --env-file .env.agent
```

`--ping` is not a nicety: a wrong base URL, a retired model id and a key without
tool-calling access all look the same from inside `run()` - an empty answer or an
HTTP error buried in a batch. The ping separates them, and it exercises the same
call path, so a pass means the wiring is good.

| option | cost | verdict |
|---|---|---|
| **DeepSeek direct** | ~¥1-2 for the whole 200-state eval (500 calls x ~1.3k in / 150 out, at peak rates; half off-peak) | the model under test, so results are comparable to this conversation |
| **Gemini free tier** | 0 | works, but the daily cap is the binding constraint; the eval has to be resumable per state |
| **Ollama, local 7-8B** | 0, unlimited | fine for measuring *tool selection*, not for measuring whether an agent beats the reranker |
| **WorkBuddy cloud LLM** | 0 | **not usable from here** - see below |

`--env-file` exists because `.env` belongs to the frontend: it points at
`gpt-4o-mini` for `/api/explain`, and changing it would change the frontend's
behaviour. Put the experiment's provider in `.env.agent` instead; `.env.*` is
git-ignored.

**Why the WorkBuddy keyless model is not an option for this loop.** The cloud
LLM is `/.cloud/llm/chat/completions` on a deployed app's origin, reached through
`@tencent-ai/workbuddy-cloud-sdk` from browser or mini-program code, authenticated
by a publishable key and an exact-Origin match. It is a client-side API scoped to
an application's quota, not an endpoint a local Python process can call. Reaching
it would mean deploying a relay app and posting through it - a proxy to another
product's quota, which is fragile and not something to build a measurement on.

## The board is held, not passed

The first version took the board as an argument: every model tool wanted nine
champion names and trusted the caller to retype them. That is a **silent**
failure - a legal but wrong board returns a confident distribution - and the
`transcription.mismatched` flag could only detect it after the fact. In the first
hand-run I transcribed six boards by hand and got them right only because I was
being careful; a flash model at temperature 0.2 across 200 states will not be.

So a draft is a state now (`draft_state.py`), and the tools take an id:

| tool | what it does |
|---|---|
| `get_state` | the board, whose move it is, how many champions are legal |
| `clone_state` | copy into a sandbox - the live board is read-only |
| `apply_move` | play a champion into the scheduled slot |
| `undo_move` | take it back and try the other branch |
| `list_states` | every live state, with depth and read-only flag |

`predict_next_move` and `rank_candidates` accept `state="<id>"` as well as the
board arguments, and the state path resolves to *the same call* - there is one
implementation, so the two forms cannot drift.

Four decisions, each of which is why this is a move log rather than a set of
board fields:

- **The schedule is a constant in the corpus.** All 46,915 games are exactly 20
  moves, and the (kind, side) at each index depends only on `first_pick`: six
  bans, six picks, four bans, four picks. `to_move()` is an index into that
  table, never an inference. The old tools inferred it from pick and ban counts,
  which is why a board with two bans and five picks got a plausible answer about
  a state that cannot occur.
- **`first_pick=1` means red picks first.** Measured, and the opposite of what
  the tool spec claimed until it was checked - a silent inversion a model
  following the spec would walk into. The field is `red_first` here so it cannot
  be repeated.
- **Undo is dropping the last move and clone is sharing the tuple**, both free
  because a state is immutable. A mutable board would need a diff log for either.
- **The real state is read-only to the model.** A stateless tool cannot be
  corrupted; a stateful one can, and the board the frontend is showing is the one
  thing an exploration must never damage. `Store.apply` refuses `real` by
  construction, and a test asserts a sandbox can never touch it.

**The frontend writes `real` on every request that carries a board.** That is the
integration, and it is one call: `_publish_state` in `frontend/server.py` maps the
UI's riot ids to indices, hands the board to `Store.set_board`, and nothing else
may write `real`. So the assistant reads the board the user is looking at without
being told what it is.

The two shapes are not the same question, which is why `log_from_board` exists:
the UI keeps **slots** (ten, two blocks of five in pick order) and **side-grouped
bans**; a state keeps **moves**, in the order they were made. Two rules fall out of
what the UI cannot say:

- **Bans are split by the schedule, not by halving.** A flat ban list is blue's
  bans then red's, and how many belong to each side is in the schedule rather than
  in the list. Halving is right only when both sides have made five; mid-phase-1
  with 2 blue and 1 red it puts every ban on the wrong side from then on.
- **The log stops at the first gap.** A pick slot is not a move index: a coach can
  fill blue's third slot and leave the second empty, and a state cannot have move
  7 without move 6. So the log is the longest prefix the board supports.
  (`draftscore.replay` skips the gap and keeps scoring, because it scores each
  placed pick independently - the two are right for their own question.)

`test_a_ui_board_rebuilds_every_prefix_of_a_real_game` converts every prefix of
eight real drafts and compares against the corpus, so the conversion is checked
against the data rather than against itself.

**The fearless pool is a third thing the board cannot say.** In a fearless series
the champions picked in earlier games are gone, and they are not on this board -
so a state that carries only moves looks legal and is not. The pool rides on
`DraftState.unavailable` (canonical keys, inherited by clones), `board()` passes
it to every model tool, the head masks it before its top-k, `apply_move` refuses
it with a sentence, and the rendered state lists it. Two guards, because the
failure was reported rather than caught: the head's mask stops the suggestion,
and `apply` stops the simulation.

The first version of this had the pool in the request and nothing reading it, so
the agent recommended a champion the series had already used. Three separate
holes: `DraftState` had no field for it, `rank_candidates` had no `unavailable`
parameter at all, and `_publish_state` dropped it. `StubHead` accepted
`unavailable` and ignored it, which is why the tests could not see any of this;
it now masks, as the real head does.

`fearless` absent from a publish means "unchanged", not "empty": the panel sends
it on every board, and a caller that forwards a board without the pool is saying
nothing about it. `[]` is the explicit "this series has no pool".

### Over HTTP

| route | what it is for |
|---|---|
| `GET /api/state` | the live state and every sandbox, each rendered |
| `POST /api/state` | `{op: clone\|apply\|undo\|drop\|reset, state, champion, name}` |
| `POST /api/agent` | one agent turn against `real`; returns the transcript |

`/api/agent` is the only route that calls a remote model. Three things have to be
true, and each missing one is a sentence rather than a traceback: `langgraph`
importable (503), a policy loaded (400), a key in the environment (200 with
`stop: "no client"` and the reason). The board comes from the store, not from the
payload, which is why the model is never asked to retype nine champion names. The
corpus database is optional - without it the SQL tool is simply not offered.

The frontend image does not ship `langgraph` (it lives in the venv inside the bind
mount, so it is not baked in). One command adds it, and it survives `--rm`:

    pro/model/llm/.venv/bin/pip install langgraph

Then put the provider in `.env`, which `docker compose` already passes to the
frontend container:

    LLM_API_KEY=...
    LLM_BASE_URL=https://api.deepseek.com
    LLM_MODEL=deepseek-flash

`python pro/model/agent/loop.py --ping` checks the pair before a single turn is
spent: it separates a bad base URL, a retired model id and a key without
tool-calling access, which all look identical from inside `run()`.

The chat panel has a `head` / `agent` toggle. `head` is the shipped path and needs
no key: the server computes the numbers and a template phrases them. `agent` hands
the same board to an LLM that calls the tools itself. Switching modes clears the
log, because a turn in the other mode answered a different question.

**The ranking is not this component's job.** The heads are well trained and are
not being improved on this branch; the goal is the user/chat interface. A state
handle removes a whole class of *input* error and leaves judgment to the head,
where it belongs - so expect fewer silent nonsense answers, and the same picks.

## The first hand-run (2026-11-11): fidelity 6/6, accuracy 2/6

Before any key existed, the tool layer was driven by hand: six val states,
stratified by where the head ranked the true next pick (2 at rank 1, 2 at 2-5, 2
at 6-20), boards printed without the truth, answers written blind, scored after.
`handrun.py` is that driver; it goes through `Session`, so the mix it reports is
the same number the real loop reports.

```
fidelity (answer == head's own #1): 6/6 = 100.0%   <- the interface
accuracy (answer == truth):         2/6 =  33.3%   <- the head's job
calls: 9  {'predict_next_move': 5, 'rank_candidates': 3, 'sql': 1}
```

**Fidelity is the requirement, and it was perfect.** The assistant's job is to
carry the model's ranking to a user without distorting it, so answering the
head's own #1 on every board is correct even when that pick is wrong - and on
four of six it was. The sample was stratified to be hard on purpose, so 2/6 is
not the head's rate either; the head's own val top-1 is 20.3 % and it is not
being improved here.

**The one real finding is about tool discipline, not accuracy.** In 2 of 3
`rank_candidates` calls, a shortlist written *before* seeing any distribution
omitted the head's actual #1 (Rumble in one board, Gwen in another). That breaks
fidelity: the comparison is confident and about the wrong set. So the prompt now
says to call `predict_next_move` first when the shortlist is yours.

Two things that follow from reading fidelity rather than accuracy:

- **Deferring to the head is not a cost.** One board had my own read right and
  the head's #1 wrong; answering the head's #1 anyway is the correct behaviour
  for this component, because it is a mouth, not a second opinion.
- **`handrun.py` recomputes the head's #1 at scoring time**, so fidelity works on
  any past run without re-answering, and it stays correct when the head changes.

**9 calls: 5 `predict_next_move`, 3 `rank_candidates`, 1 `sql`. `sql_share`
0.11, zero refusals.** The high-level tier carried the turn without the escape
hatch, which is what the two-tier design was betting on.

**Not measured**: whether a user finds the answers useful. That needs the
frontend and a person, which is the actual goal of this branch.

## What this does not answer

**Whether a user finds it useful.** That needs the frontend, a person and a
pay-per-token key, and it is the goal of this branch - not a top-1 number.

**Whether an LLM driving these tools beats the head.** Not the target any more:
the heads are trained and are not being improved here. The 200-state eval would
answer it, split by whether the truth was inside the head's top-20, because the
~13.65 % of states where the head is confidently wrong are the only ones where an
agent can add anything the reranker cannot - and aggregate top-1 would look like a
wash and hide exactly that.

What the loop measures today is the **mix**: how many calls, which tool, how often
it fell back to SQL, whether it retyped the board. The hand-run above is six
boards and cannot support a rate. `handrun.py` is the skeleton of that eval - the
sampling, the stratification and the blind scoring are the parts that are easy to
get wrong.
