# Draft Assistant (local prototype)

A local-only web UI over two trained models and an LLM agent, all looking at one
draft board. Five things, and the board is the only one that is not a question:

| | what it is |
|---|---|
| **the board** | 5 ban + 5 pick slots per side, dDragon art, a fearless series bar, and a walkthrough of real pro games |
| **the next move** | `POST /api/next-move` - the head's ranked list for the side on the clock |
| **the role board** | `POST /api/suggest` - each side's five roles, filled and unfilled, with the model's answer for every open cell |
| **the agent** | `POST /api/agent` - one turn of an LLM that holds the board as an object, calls the models and the corpus through tools, and answers in prose |
| **the agent's suggestion** | `POST /api/agent-suggest` - one click: the same LLM, asked the fixed reranking question, commits to a single move |
| **the agent's suggestions** | `POST /api/agent-rank` - the LLM's reranked top-k, shown as a column beside the head's own list; reruns on every board change |

The agent replaced the breakdown page, the draft-score comparison and the
rule-based chat. That is deliberate: four panels each answered a fixed question,
and one surface that can be asked anything is what the models were for.
§"What was removed" says where each number went. The Qwen reranker came back
afterwards, in a narrower job than it had before: an opt-in next-move model
(§"What the reranker is worth"), not a second chat surface.

## Run

```bash
bash frontend/serve.sh                 # http://127.0.0.1:8777
bash frontend/serve.sh --port 8780     # or PORT=8780
```

Binds `127.0.0.1` only. The script runs the `dev` image with the repo mounted, so
there is no GPU requirement and no 4-bit load: the models in the loop are the
2.49 M Siamese heads, and the LLM is remote.

| flag | what it does |
|---|---|
| `--ckpt <path>` | pins the **role model** (a `masking-*` checkpoint) |
| `--policy-ckpt <path>` | pins the **next-move model** (a `nextpick-*` checkpoint, an `ensemble.json`, or an `llm-*` reranker directory) |
| `--rerank [dir]` | serves the 4B reranker as the next-move model instead of the head (needs `--device cuda`); a failure here is fatal |
| `--no-policy` | runs the role board alone; the agent still works, it just cannot rank a move |
| `--device cuda` | GPU inference (CPU default keeps the GPU free; the reranker needs cuda) |
| `--agent-calls N` | the agent's step budget per turn (default 24) |

Through Docker, pass `--host 0.0.0.0` - a published port arrives on the container's
`eth0`, not its loopback. See `DOCKER.md` §"The frontend".

## Where things are on the page

| control | where | why |
|---|---|---|
| **first pick**, **Reset draft** | inside `<main>`, above the Blue / Red side panels | both act on the game on the board, so they sit with it rather than in the header |
| **agent chat + suggestions** | directly below the next-move suggestion panel, which is directly below the head's suggestion list | the head's ranked list, the agent's rerank column, and the agent's prose are the three answers to "what next?" - adjacent, not split across the header. The agent panel and the rerank column are both **on by default**; the **agent features** checkbox in the header hides them (leaving only the base model's suggestions and board updates). |
| **Real draft ▾** | header, next to the patch dropdown and the agent-features checkbox (the walkthrough bar itself stays above the board) | the walkthrough is an optional comparison against pro games, not the main board |
| model names | **not on the page** | chosen with `--ckpt` / `--policy-ckpt` at startup, never inside a session. The header used to print both checkpoint names and a val top-5; that text is gone, so `--ckpt` is the only record of what is loaded. |

The role vocabulary is `TOP / JUNGLE / MIDDLE / BOTTOM / SUPPORT`. Role 4's
display name was `UTILITY`; `/api/champions` now publishes the list and
`app.js` adopts it, so the client and server cannot drift. `/api/suggest` keys
every role by that name - the cells, `open_roles`, `assignment`, `methods` and
`notes` all speak the same vocabulary - and the client normalises it through
`roleIndex()` instead of indexing `ROLES` with the raw value. Indexing a name
into an array printed `undefined` under each picked champion and left its role
cell empty. The corpus stores **role ids**, so the rename changes no count and
no assignment; the soloQ analysis store keeps Riot's raw `pos` string
(`UTILITY`) because that is a data value in a built DuckDB, not a name this app
chooses.

## The two slots are bound into one engine

Both slots are loaded at the same time and chosen independently, because they are
different models doing different jobs: the masking model can say "this team still
wants a MID", the policy ranks the next pick, and neither can do the other's.

```
APP["masked"], APP["policy"]  ->  engine.Engine  ->  draft_state.Store
```

`Engine` is the model binding: it converts a state into model arguments and owns
the *routing* (head vs masker vs the corpus role table). `Store` holds the board -
one read-only `real` plus any number of sandboxes - and every state in it carries
the engine, so a state can be asked a question without being handed a board.

**The server never passes a board to a model function.** It publishes the board
to the store (`_publish_state`), and the state carries the answers. Loading or
unloading a checkpoint rebinds the engine under every live state
(`Store.attach`), which is why swapping a model mid-draft changes what the next
question returns rather than leaving a stale binding behind.

## Choosing a model

**At startup, on the command line.** There is no model picker in the browser:
no dropdown, no Load button, no "From file…". The page does not name the loaded
checkpoints either - the startup flags are the only record of which two are
live. Swapping weights while a draft is open would leave every answer in flight
describing the previous checkpoint, so the choice belongs to the process that
owns the state:

```bash
bash frontend/serve.sh --ckpt pro/model/checkpoints/<masking-run>/best.pt \
                       --policy-ckpt pro/model/checkpoints/<nextpick-run>/best.pt
bash frontend/serve.sh --no-policy          # role board only
```

`GET /api/checkpoints`, `POST /api/load {path, role}` and `POST /api/upload`
still exist as a **local scripting surface** and are tested, but no page calls
them. Two different questions are answered there: which slot a checkpoint *can*
fill comes from its directory name (`_ckpt_roles`, so a masking model cannot be
filed as a policy), and which family it *is* comes from the state-dict keys at
load (`make_predictor`) - a renamed file still loads, a mis-filed one is refused.

Shipped defaults, both pro-track:

* **role** - `masking-proonly-nov13-through-lcp-df-sw/best.pt`.
* **next move** - `nextpick-proonly-nov13-k5-through-lcp-df-sw/best.pt`, the
  head. The 4B reranker (`llm-lora-rerank5/`) is one flag away (`--rerank`) and
  is **not** the default - §"What the reranker is worth" has the numbers and the
  reason.

That pair is a different **corpus**, not a newer run of the old one: the freeze
is lifted (`--corpus 2022-full`, 46,939 drafts through 2026-11-13), the pinned
test series are ordinary rows (`--no-pin`), the split is the fractional 85/10/5
within-patch tail, and the policy trains on `--prefix-k 5` (5 distinct prefixes
per game per epoch). So **none of its numbers belong next to the
`-wlds-ig-tes` rows** in `STATUS.md` §7. On its own 2,448-game fractional test
split it scores top-5 0.5083 (rel 3.36x); paired against the same recipe with
one prefix it is +2.55 pp (CI [+2.22, +2.87], p 2.7e-51). The
`-randomvalsplit-` pair is still on disk and still selectable.

### What the reranker is worth, and to which question

Four arms, one slice: 4,000 held-out states (200 games x 20 moves), the v4
prompt, k=5. `exact` is champion-level "did the pipeline name the pro's move".

| served prompt | exact | source |
|---|---|---|
| reranker, task line names the lane ("for Top lane") | **34.27 %** | `eval_rerank5_mask4k.json` |
| reranker, no lane - **this panel's question** | **20.85 %** | `eval_rerank5_nolane4k.json` |
| the head that proposed those candidates (`nextpick-bans-aware-*`) | 18.25 % raw | `eval_served_banaware4k.json` |
| the shipped head (`-nov13-k5-`), served, same rows | 24.90 % | `eval_served_head4k.json` |

Read it in this order:

1. **The 34.27 % is role-conditional, and it is not a next-move number.** The
   v4 task line reads "Choose the best Blue Side pick **for Top lane**", and
   `target_role` in the training rows is the *true* lane of the pro's own pick.
   That arm is handed part of the answer. Delete the clause and the same weights
   score 20.85 % - 13.4 of the 16 points were the lane.
2. **The panel cannot name a lane.** "What is the next move" has no lane to
   name; if it did, the question would be the role board. So the panel's arm is
   the 20.85 % one, at ~2 s per board instead of milliseconds.
3. **The baseline moved twice.** The 18.32 % this repo used to quote is
   `C.BASELINE_VAL`, which is `nextpick-proonly-**randomvalsplit**-.../test_report.json`
   over **95,160** states - a different checkpoint from the shipped `-nov13-k5-`
   head, on a different sample. The head that actually proposed the reranker's
   candidates is `nextpick-bans-aware-*` (`rerank.HEAD_CKPT`), and it scores
   18.25 % on these rows, which is what the 18.32 was standing in for.
4. **The 24.90 % is not a held-out number for the shipped head.** It is the
   `-nov13-` head on the *LLM track's* val split, and that head trained on a
   different corpus and a different split (fractional within-patch tail), so
   some of these rows may be its training data. Do not read 24.90 as "the head
   is better than the reranker"; read it as "no clean shared-split comparison of
   the shipped head against the reranker exists yet".

5. **The role-conditioned question already has an incumbent, and it is 2.5 M
   parameters.** `masking-proonly` scores 35.7 % top-1 on masked completion
   (STATUS §7, 70 slots, `--mask-mode slot`), i.e. the same number as the
   reranker's 34.27 % on a *harder* query - and the masker gets more
   information: `masked_ranks_slot` hides one slot and keeps the other nine,
   **including the picks after it**, and passes the full true `role_ids`
   vector. The reranker sees the prefix only. So on the one question where the
   4B looked strong, a CPU model of 2.5 M params ties it while seeing the
   future. That is the strongest single argument against shipping the reranker,
   and it is why 34.27 % should not be quoted as a capability.
   Caveat on the caveat: 35.7 % is 25 of 70 queries from 7 games, CI
   [25.5 %, 47.4 %], and it is the masker's *best-case* protocol. The same
   family on its own larger test reports 20.2 % (4,193 masked slots, random
   15 % masking) and 21.3 % on val (8,069 slots) - the 14 pp gap between those
   and the 35.7 % is protocol (one hole with full context vs ~1.5 holes), not
   model quality. So the honest reading is a **band, 20-36 %**, and the
   reranker's 34.27 % sits at the top of a band a 2.5 M-param CPU model also
   reaches. Neither number should be used to rank the two models.

What survives: the reranker's advantage over **its own proposer** on the panel's
question is ~2.6 pp on unfiltered candidates, at 2 s/board. That is not enough
to be the default, which is why `--rerank` is opt-in and a failure to build it is
fatal rather than a silent fallback. And on the role-conditioned question it
does not clearly beat the masker the repo already ships, so it is not a
capability gap being filled either.

**The role board is the other way round**, and it is why the flag exists. Each
cell *names* a lane, and there the lane is the query rather than oracle
information: `predict_in_lane` constrains the head's logits to that lane and the
reranker orders the shortlist, so the prompt matches the trained one and the
answer is "the best champion for this role". `pro/model/archive/llm/eval_role_cells.py`
scores that surface directly (head vs reranker, paired, on the 800 rows the role
board would actually ask - the pick phase after the ban block). **That run has
not been completed**; until it is, the role-board gain is unmeasured and the
honest statement is "the model is in-distribution there, which is not the same
as better".

Two arms that used to be the next-move default are still **CLI-selectable**:
* `nextpick-bans-aware-through-wlds-ig-tes/best.pt` - the pro-only policy with
  `ban_side`, `ban_fearless` and `counter_rank=32` on: +0.60 pp top-5 against a
  control that reproduces the previous model to 4e-5 (0.5056 vs 0.4995, paired
  bootstrap p = 4.3e-8). One seed, and a **val** number on the split its own
  control selected under - `pro/model/archive/bans/README.md` calls it "promising and
  unconfirmed".
* `nextpick-proto-ensemble/ensemble.json` - three pro-only arms whose policy
  logits are averaged before the softmax (+1.33 pp top-5 on 439 draft rows
  neither it nor its predecessor trained on). That is a different and harder set
  from the ban-aware val number; do not read the two as comparable and do not
  add them. An ensemble has no val run of its own, so `/api/champions` publishes
  "held-out top-5 (439 drafts · 3 models)" on the model payload (the page no
  longer prints it).

`llm-*` folders **are** offered, in the next-move slot (`roles: ["policy"]` in
`/api/checkpoints`), and the folder itself is the entry - `projector.pt` inside
it is skipped, because that file is one piece of the model rather than a model.
`resolve_ckpt_path` accepts a directory only when its name starts with `llm-`,
so the one artefact that is a folder cannot be confused with a typo. Loading it
needs the LLM stack, which is why `make_predictor` reports a missing package
instead of raising on import. `value-*` gets the other treatment: an outcome
head answers neither question, and it cannot rank picks at all (0.5002
[0.4974, 0.5028]).

## The board

Ten slots per side in pick order, plus bans. The next move to be made is
**highlighted, never enforced** - the schedule follows
`solo_queue/model/README.md`, mirrored when red has first pick, and it is a
constant: 20 moves, six bans, six picks, four bans, four picks.

Two things the board cannot say, both of which travel as data rather than as an
inference:

* **The fearless pool.** Champions picked in earlier games of the series are gone
  and are not on this board, so a board that carries only moves looks legal and is
  not. The pool is sent on every publish, masked by the head before its top-k, and
  refused by `apply`. Absent means "unchanged", `[]` means "no pool" - reading an
  absent field as empty would clear a real one.
* **Which side a flat ban list belongs to.** The UI keeps side-grouped bans; the
  model reads chronologically. `log_from_board` splits a flat list *by the
  schedule* (the first n bans of a draft are dealt to known sides), not by
  halving, which is only right once both sides have made five.
* **Which game of the series this is.** Ten picks and ten bans look the same in
  G1 and G5. The board publish carries the patch, `first_pick`, the fearless
  pool and a compact series summary (game number, number of games, per-game
  pick counts); the server turns the numbers into short lines on the state, so
  `get_state` and every clone describe the same game. An absent `series` field
  means *unchanged*, exactly like an absent pool - otherwise a board publish
  from the role board would silently reset a game-3 draft to game 1.

Champion art comes from dDragon (`16.20.1`), cached, with a deterministic name
fallback offline. `GET /api/drafts` + `GET /api/draft` walk through real pro games
from the cached corpus, showing what the pros actually moved. The dropdown is
pre-filtered to **WLDs** (`#wt-filter` ships with `value="WLDs"`; the 🎲 button
sends the same `q`, so a random game stays in the league on screen). It reads
`pro_drafts_full.pt` - the *unfrozen* superset, through 2026-11-13 - and falls
back to the frozen `pro_drafts.pt` (CORPUS_MAX_DATE, 2026-11-08) when that file
is absent. Browsing is not a measurement: the freeze that every recorded number
depends on is untouched, and the newest games are exactly the ones no shipped
checkpoint trained on.

## The next-move panel

`POST /api/next-move {picks, bans, first_pick, fearless, patch_id, k}` -> the
head's ranked list for the side on the clock, each champion with its share of the
legal pool (`p_head`), its index and its rank. `k` is clamped to 1-20 and is the
caller's, not the tool's: the head's top-1 is right 18.4 % of the time, its top-5
50.5 % and its top-20 85.2 %, so how wide to read is a decision.

The payload also carries the state it answered about (`move_idx`, `kind`, `side`,
`n_legal`, `n_unavailable`, `complete`), the patch label the head conditioned on,
and a `note` that is shown verbatim. A board the store refuses is a **400**, not a
list: a distribution computed for an illegal board is indistinguishable from a
good one. This is the one route that cannot fall back to the last good board -
`/api/suggest` may, and the panel shows `state_error` instead.

There is no separate ban ranking. A pro-like ban is the denial decision and the
same head makes it; the rollout that used to approximate it was removed
(`STATUS.md` §5/§8).

## The role board

`POST /api/suggest {picks, bans, first_pick, fearless, patch_id, k}` -> one
`predict_in_lane` question per open cell: for each side, for each role the
assignment says is still unfilled, "which champion would the model take for that
role".

* **The mask is applied to the logits, before the top-k**
  (`infer.next_move(allowed=...)`). Filtering the top-k afterwards empties the
  cell 13 % of the time, because the head's five favourites often do not play the
  role at all.
* **The assignment is not a model answer.** It comes from `data/champ_roles.json`
  - each champion's own pick history - maximising summed log-likelihood over at
  most 120 injective assignments per side, so it is **one-to-one**: five picks,
  five roles. 96 % agreement with the real roles, against 67 % for the role head's
  argmax; as serving input that is 30.2 % -> 37.7 % recall@5 (the corpus ceiling
  is 38.8 %). It travels on the state, so the panel reads it instead of computing
  a second, possibly different one.
* **Which model answered is per cell** (`methods`), because the head can only be
  asked on a pick move with the ban block complete. In the ban phase the masking
  checkpoint answers (`method: "masker"`); a cell neither can answer is left out
  with its reason in `notes` rather than failing the board - one refused role
  should not blank the other nine.

`conf` is `p_lane` per slot: a champion's share of its *own* picks in that role,
so 0.98 is a dedicated laner and 0.45 is a flex. Below 0.6 the panel draws the
slot as a flex rather than as a fact.

The head cannot tell blue from red - measured, `STATUS.md` §4 - so the board is
asked side by side but the model is not told which side it is looking at. Side
awareness needs an encoder change, not a serving one.

## The agent panel

Shown by default, directly below the head's suggestion panel (or opened by the
`#agent` deep link). A header checkbox, **agent
features**, hides both the chat and the rerank column so only the base model's
suggestions and board updates remain. The panel's header carries **suggest next
move**, the one-click reranking button described below.

`POST /api/agent {message, picks, bans, first_pick, fearless, patch_id, max_steps}`
-> one turn, answered in prose, with the tool calls it made and the sandboxes it
left behind.

The turn publishes the board first, so it always answers about what is on screen.
The agent reads `real` and **cannot edit it**: it clones into a sandbox to try a
hypothetical, and the panel shows the sandboxes, because "the model likes X" and
"X is now on the board" are different claims and the second is the user's.

Every turn shows three things, because each answers a different question:

* **the answer** - what the model and the corpus say, in sentences;
* **the tools** - what it actually called, with arguments and latency; the only
  audit trail there is, since nothing here is a template;
* **the mix** - calls, and what share were SQL. Above half means the high-level
  tools are under-specified, not that SQL is good.

The tools the panel can hand the model: `predict_next_move` (the **serving**
next-move model's ranked list for the side on the clock - the reranker's order
when the reranker is loaded, the head's otherwise, and `model` in the tool result
says which), `predict_roles` (the role board), the store
tools (`get_state` / clone / apply / undo / list), `sql` (one read-only query
over the pro corpus, returned with the query so a wrong answer is debuggable),
and `soloq_matchup` - a parameterised two-champion matchup over the soloQ
analytical store (win / gold / dmg / cs / kills / deaths / assists / vision,
plus a baseline, a `delta_win` and a `reliable` flag). `soloq_matchup` answers a
pairwise question the head structurally cannot ("does Ahri beat Zed in mid?"); its
rates are **soloQ**, a different population from the pro drafts the head learned
on. `soloq_matchup` and `sql` are both ablated from the one-click reranking
button (that turn is a rerank of the head), but both stay in the free-form panel
and the MCP server.

Two preconditions, each a sentence rather than a traceback: `langgraph` importable
(503 with the reason - it is in `requirements.txt`, so a rebuilt image has it, and
an image built before that answers 503 rather than pretending) and a key in the
environment (200 with `stop: "no client"`). No checkpoint is required; without one
the model tools are simply not offered.

`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT` are the same contract
the agent's own scripts read, so a transcript that names a model names the one
that answered. The server prints which at startup, or says there is no key.

## The agent's suggestion (the reranking button)

The header of the agent panel has a **suggest next move** button. It does not
call `/api/next-move`: that route asks the head for a ranked list. This one asks
the **agent** - the same tool loop, the same held state - to read the board,
choose its own `k` for `predict_next_move`, compare the list against the open
lanes and commit to one champion. That is the reranking job the retired 4B
adapter used to do, now done with tools and state management.

`POST /api/agent-suggest` takes the same board payload as `/api/agent` (picks,
bans, `first_pick`, fearless pool, patch and the series summary; no `message`).
It runs the configuration `pro/model/agent/suggest.py` shares with
`pro/model/archive/llm/eval_api_agent.py`, so the button and the 4,000-state measurement
cannot drift:

* the fixed question and the **explained** system prompt (where the head's
  numbers come from, and how often it is wrong);
* `sql` ablated, 5 graph steps, 8 tool calls;
* one turn on `real`, which stays read-only.

The answer is still prose and the tool log is still shown, but the panel also
parses the final `PICK:` / `BAN:` line into one move and shows it as a chip. A
champion already on the board, in the fearless pool, or of the wrong move kind
is shown but **cannot be played**; "play it" is the user's decision and calls
the same helper the head's own chips use. Next to it is the head's own #1 on
this board, recomputed at serving time: **fidelity** is this component's metric,
accuracy is the head's. The button never claims to be right.

A board the store refuses is a **400** (a recommendation is a claim about *this*
board), a completed draft returns `complete: true` without calling the LLM, and
a missing key returns `stop: "no client"` with the state and the head's #1
still attached. No checkpoint is required either: with no next-move model the
`head` block is simply `null`.

## The agent's suggestions (the top-k rerank column)

Below the head's `#nextmove` list sits a second column, **Agent suggestions**,
filled by `POST /api/agent-rank`. It is the *same* ablated, explained-prompt
agent and the *same* budget as the single-move button, but asked to **order**
the head's top-k rather than commit to one. Each chip is a champion with a
confidence and a `source` badge:

* **LLM** - the agent named this champion with its own `(NN%)` confidence;
* **head** - the agent left the slot unnamed (or named no champion at all), so it
  is filled from the head's own legal shortlist with the head's `p_head`.

The list is honest by construction: `suggest.parse_ranking` only accepts
champions the head listed (it matches names against the store and fills any
untouched slot from the head's own shortlist), so the panel always shows `k=8`
names and never invents one the head did not produce.

It reruns automatically on every board change (debounced) and after any chat
turn, so it mirrors the head's list but with the agent's reordering - and a
conversation that changes the agent's mind changes this list too, because the
chat is carried into the rerank as `seed` context. The chat log records each
board change ("Board changed: picked Ahri") and the agent's top picks, so the
panel shows *what* moved and *what* the agent thought. **Clear chat** drops the
conversation context and resets the column to the agent's fresh top-k on the
current board.

`POST /api/agent-rank {board..., messages?}` runs the configuration in
`suggest.py` (`rank_question`, `rank_config`) - `sql` and `soloq_matchup`
ablated, 5 steps / 8 calls - and returns `{suggestions: [{name, idx, conf,
source, legal, note}], head, complete, config, stop, state}`. A completed draft
returns `complete: true` with an empty list; a missing key degrades to the
head's own list with `source: "head"`.

## Routes

| route | what it is for |
|---|---|
| `GET /api/champions` | the champion list, art, and the move plan for `first_pick` |
| `GET /api/checkpoints` | both stores, each entry with its slot and val metric (no UI) |
| `POST /api/load` | `{path, role}` -> swap a model; a local scripting route, **no browser control** |
| `POST /api/upload` | a raw `.pt` body -> a checkpoint in `frontend/uploaded/`; same, **no browser control** |
| `GET /api/drafts`, `GET /api/draft` | the pro-game walkthrough |
| `POST /api/next-move` | the head's ranked list for the side on the clock |
| `POST /api/suggest` | the role board |
| `GET /api/state` | the live state and every sandbox, each rendered |
| `POST /api/state` | `{op: clone\|apply\|undo\|drop\|reset, state, champion}` |
| `POST /api/agent` | one agent turn against `real`; returns the transcript |
| `POST /api/agent-suggest` | one fixed agent turn: the reranking / next-move recommendation |
| `POST /api/agent-rank` | the agent's reranked top-k, as a column beside the head's list; reruns on board change |

## Tests

```bash
python -m pytest frontend/ -q          # 61 tests, ~40 s
python -m pytest pro/model/agent/ -q   # 164 tests
python -m pytest pro/model/archive/llm/test_eval_api_agent.py -q
```

`test_server.py` drives the real handler over a loopback HTTP server, with the
checkpoint slots saved and restored around every test - an earlier version
restored to whatever was loaded when the fixture started, which let a masked
checkpoint from one test make another test's answer say `masker`.

Two rules the suite pins because both have bitten: **a skip is not a pass** (globs
naming pre-split paths made 18 tests `pytest.skip` silently while the suite stayed
green), and **the frontend does not call a retired route** - the panel's JS is
checked against the route table. Static tests also pin the two UI contracts that
broke: `assignment` values are names the client can render (and `roleIndex()`
normalises them), and the page has no model picker while the CLI flags remain.

## What was removed, and where the numbers went

| removed | why | the numbers live in |
|---|---|---|
| `frontend/rerank.py`, the *chat* surface over the `llm-*` slot | a hosted API alone converts 44 % of the ceiling against the local reranker's 67 %, and a reranker can only reorder a shortlist it did not build. The **slot** came back (`--rerank`, `RerankPredictor`) - the chat surface did not, and the 67 % is now quoted as a lane-conditioned number (§"What the reranker is worth") | `pro/model/archive/llm/README.md`, `STATUS.md` §8-§9 |
| `frontend/chat.py` (rule-based chat) | it read names, a stance cue and a negation and never the reason; a true, a false and an irrelevant "because ..." bought the same weight | `STATUS.md` §8 |
| `frontend/draftscore.py`, the draft-score panel | it replayed each pick through the head to compare two drafts, which is a question the agent answers directly, and it was never a win probability | `STATUS.md` §8 |
| `/api/explain`, `/api/breakdown`, `/api/tool` | three ways to ask "why this pick"; the agent is the fourth and the only one that can answer something else too | `STATUS.md` §8 |

The agent's own measurement - whether an LLM holding these tools beats the head -
is **not** the target here: the heads are trained and are not being improved. The
metric for this component is **fidelity**: does the answer match the head's own
#1. Accuracy is the head's job. The suggestion button serves that measured arm
unchanged (`suggest.py`), so the number in `STATUS.md` §8 still describes what
the button does.

## Limits (prototype)

* **One board, one user, no persistence.** State lives in the process; a restart
  is a fresh draft.
* **The role table is counted, not predicted.** It is a corpus fact about
  champions, so it cannot know a series' plan - and 2.5 % of pro picks land
  outside their champion's legal roles.
* **Neither model sees side.** Counter-pick awareness is an encoder change.
* **The agent is as good as the model behind it.** Everything it says is either a
  tool result or prose about one; the mix panel exists to make that checkable.
