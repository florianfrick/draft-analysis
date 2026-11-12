# The agent's tool layer

Not an agent. This is the part that has to be right *before* a LangGraph loop is
worth writing: what the tools return, what they refuse, and what they record.
The loop that picks a tool is the easy part and the part most likely to change.

```
schema.py       build the SQLite corpus + the schema text the model is shown
sql_tool.py     one read-only query tool with mechanical guards
model_tools.py  predict_next_move / rank_candidates over the trained heads
session.py      the registry, the call log, the ablation switch
demo.py         a scripted turn, so the wiring is checkable without an LLM
```

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

## Running it

```bash
# build the corpus (~1 min; 46,915 games, 938,300 moves, 107 patches)
python pro/model/agent/build_db.py \
  --known-patches-from pro/model/checkpoints/nextpick-sharedsplitthrough-las-dk-c-t1-r/best.pt

python pro/model/agent/demo.py                  # all tools
python pro/model/agent/demo.py --ablate sql     # the same turn without SQL
pytest pro/model/agent/test_agent.py -q
```

## What this does not answer

Whether an LLM driving these tools beats the head. That needs the 200-state
evaluation on the shared val, split by whether the truth was inside the head's
top-20: the ~13.65 % of states where the head is confidently wrong are the only
ones where the agent can add anything the reranker cannot. Aggregate top-1 will
look like a wash and hide exactly that.
