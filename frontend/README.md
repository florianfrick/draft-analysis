# Draft Assistant (local prototype)

Local-only web UI with **two model slots that are loaded at the same time**,
plus a **breakdown page** (`/breakdown.html`) that scores one real pro game
under several checkpoints and shows the real move next to each model's
prediction.

| slot | checkpoint family | what it answers |
|---|---|---|
| **role model** | masked (`pma.*`, `heads.*`) | the role board: each pick's role, and the target champion for every empty role (role-conditioned MASK queries) |
| **next move** | policy (`policy_head.*`) | one ranked list for the side on the clock (pick or ban) |

They are different models doing different jobs — the masked model that can say
"this team still wants a MID" is not the policy that ranks the next pick — so
they are two slots rather than one switch. Loading either leaves the other
alone, and the next-move slot can be empty (`— none —`), which runs the role
board alone. The family is read from the state-dict keys, not the filename, so
any kept checkpoint just works, and `--ckpt` / `--policy-ckpt` are validated
against the slot they are asked to fill.

## Run

```powershell
cd E:\VSCode\code\lol draft analysis
python frontend\server.py --port 8777     # http://127.0.0.1:8777
```

Binds `127.0.0.1` only. `--ckpt <path>` pins the **role model** (default:
`pro/model/checkpoints/masking-proonly-randomvalsplit-through-wlds-ig-tes/best.pt`;
if it is absent, the `best.pt` with the highest val champ-top5 across both
stores, `solo_queue/model/checkpoints/*/` and `pro/model/checkpoints/*/`),
`--policy-ckpt <path>` pins the **next-move model** (default: the LLM
reranker folder `pro/model/checkpoints/llm-lora-rerank5/` when this machine can
run it, otherwise the ban-aware head
`pro/model/checkpoints/nextpick-bans-aware-through-wlds-ig-tes/best.pt` — see
"The LLM reranker" below for why that is conditional),
`--no-policy` skips it entirely, and `--device cuda` for GPU inference (CPU
default keeps the GPU free).

The role default is the 2026-11-04 refresh (`-randomvalsplit-`: 42,121 train,
no test split). The `masking-proonly` originals are still listed in the dropdown
and are the only ones with a pinned-test score (`STATUS.md` §7); the refresh has
none by construction.

The next-move fallback — and the shortlist the reranker chooses from — is the
**ban-aware arm** (2026-11-07): the same pro-only recipe as the
`-randomvalsplit-` role model, with `ban_side`, `ban_fearless` and
`counter_rank=32` on, so the ban block — including which side made each ban and
which champions the fearless pool has already removed — is a model input rather
than something to infer from the absent champions. Against a control that
reproduces the previous model to 4e-5 it is +0.60 pp top-5 (0.5056 vs 0.4995,
paired bootstrap p = 4.3e-8, CI [+0.39, +0.82] pp). The flags are read from the
checkpoint's own `config`, not passed by the frontend, so an older checkpoint
still rebuilds unchanged.

`nextpick-proto-ensemble/ensemble.json` is still selectable but no longer the
default (nothing is, since the reranker took the slot): three pro-only policy
arms whose policy logits are averaged before the softmax. Its +1.33 pp top-5
(bootstrap [+0.64, +2.07], p = 2.6e-4; +0.94 pp
top-1) was measured on the 439 draft rows neither it nor the checkpoint it
replaced had trained on — a different set from the ban-aware arm's val number,
so the two are not comparable and must not be added. The members alone are
*not* better than the old default, so the manifest is the artefact:
`python pro/model/proto/compare_shipped.py` re-measures it. Because an ensemble
has no val run of its own, the header reports its number as
"held-out top-5 (439 drafts · 3 models)" rather than as a val score.

An `ensemble.json` is `{"members": ["a.pt", "b.pt"], "eval": {...}}`, members
resolved relative to the manifest and confined to the project. Any checkpoint
path in the dropdown can also be replaced by such a manifest, so the averaging
path is not special-cased to the default.

Through Docker, pass `--host 0.0.0.0` (a published port arrives on the
container's `eth0`, not its loopback) — see `DOCKER.md` §"The frontend".

### The LLM reranker (2026-11-09)

`--policy-ckpt pro/model/checkpoints/llm-lora-rerank5` (or the folder itself)
serves a different next-move model: the head still produces the shortlist, and a
LoRA-tuned Qwen3.5-4B chooses from it. It is **the default when it can run**, and
the head when it cannot — see below, because "can run" is not a preference.

```bash
bash frontend/serve.sh                 # http://127.0.0.1:8777
bash frontend/serve.sh --port 8780     # or PORT=8780
```

That script exists because three things must hold at once, and only one of them
is a preference: the image needs `transformers`/`peft`/`bitsandbytes` (the
derived `lol-draft:llm`, not the base `lol-draft:dev`), `python` must be that
image's venv interpreter, and the port must be **published**. `docker compose
run dev` (what `DOCKER.md` documents for the frontend) has none of the packages,
and `pro/model/llm/run.sh` — the training wrapper — publishes no port, so a
server started through it runs fine and is unreachable from the browser. Run the
frontend on the host interpreter and the model is still *listed*, disabled, with
the reason under the picker; `rerank.unavailable_reason()` decides that and
`/api/checkpoints` reports it as `rerank.available`. Everything else — the role
board, the head as next move — works exactly as before without any of this.

| | head alone | LLM rerank |
|---|---|---|
| top-1, 478 games held out from both | 19.14 % | **35.31 %** |
| ceiling (the head's own recall@5) | — | 51.85 % |
| parameters in the loop | 2.49 M | 4.2 B |
| one request, steady state | ms | **0.7 s** |
| startup | instant | ~107 s (27 s load + 58 s kernel warmup) |

Both numbers were measured on the 478-game `common` split, which has since been
retired: it turned out to be 92.3 % inside the shipped head's *train* (a
`seed=0` in `pro/model/llm/common.SPLIT` drew a second random 10 % instead of
reusing the head's). On the shared val the ceiling is **50.51 %** over 4,763
games, and the reranker needs re-scoring there — see `pro/model/llm/README.md`.

It also needs CUDA: 4.2 B in 4-bit is ~4.5 GB of VRAM, and a CPU forward pass
takes minutes — so `serve.sh` passes `--device cuda`, and the head fallback is
not something you can wait out.

The warmup is why the first click is not 53 s: Triton compiles the hybrid
attention kernels and `causal_conv1d` falls back to its reference implementation
on the first forward. It runs inside the constructor and prints its cost.

Two things it does that a head cannot, both of which change what the panel
shows:

* **It asks which lane the coming pick is for — and enforces the answer.** The
  panel grows a lane selector whenever the board does not decide the lane (four
  lanes filled ⇒ forced). Choosing one masks the head's pool to champions that
  play that lane (`data/champ_roles.json`, counted from the corpus), so "Bot"
  returns ADCs instead of the head's unconstrained favourites. Leaving it on
  *any* keeps the lane as a prompt hint and the list unfiltered, because
  filtering on a guess would make a wrong guess unrecoverable. `lane_source`
  says which case it is (`board` / `user` / `default`) and `lane_filter` says
  whether the pool was masked and how many champions were eligible.

  This is not cosmetic: over 1,200 pick moves the head's plain top-5 is only
  46 % lane-appropriate and 22 % of lists contain **no** champion that plays the
  lane at all. Masked to the lane it is 100 % appropriate, never empty, and
  recall of the actual pick rises from 31.0 % to 47.4 % — dropping competitors
  can only raise the answer's rank, so the constraint cannot cost a pick that
  belongs in the lane. The table behind it is counted from the corpus
  (`python data/build_champ_roles.py` → `data/champ_roles.json`): a champion is
  legal in a lane when it was picked there in ≥10 % of its own picks and ≥5
  times, which makes 113 of 173 champions single-lane and 54 flex. 2.5 % of pro
  picks land outside their champion's legal lanes, and those are exactly the
  picks a lane-constrained list cannot offer.
* **It can disagree with the head**, which is the point: on a Red-to-pick board
  asked for Bot it returns Ezreal 73.8 % where the head's top-1 was Nautilus
  10.8 % — two ADCs at the top of a list the head filled with a support, a
  toplaner and a support. Chips show the LLM's k-way probability and carry the
  head's own number in the tooltip, because 31 % of 173 champions and 31 % of
  five candidates are not the same claim. They are *ordered* by that
  probability too — the chips are read left to right as a ranking, so the
  answering model's order is the one on screen; the three candidates the LLM
  never scored keep the head's order behind it.

The same table fixed a second, larger thing on the way in. The encoder is
permutation-invariant within a team, so a placed pick's *role id* is the only
thing that says which lane it is in, and serving was filling it with the role
head's argmax — right 67 % of the time, against 96 % for the champion's own
history. `infer._fill_roles` now takes the lanes from `lanes.assign_lanes`, which
is worth **30.2 % → 37.7 % recall@5** on 1,500 held-out pick moves (the ceiling,
the corpus's own roles, is 38.8 %). The role head keeps only the per-slot
distribution the board displays.

The reranker is not a checkpoint *file* but a folder (`projector.pt` is its entry
point, `adapter/` its LoRA). Discovery offers a folder only when it can actually
answer — it needs `adapter/` and a `report.json` whose `protocol.prompt_format`
is `v4`, the format that carries the candidate list. That is what keeps a
checkout with a dozen warmup and probe runs from offering all of them; the
excluded ones are skipped with a reason (`server._llm_servable`).

## Choosing a model

The header shows both slots (label, kind, val metric each) with a `role` +
**Load** pair and a `next move` + **Load** pair. Three ways to switch:

1. **CLI** — `--ckpt` fills the role slot, `--policy-ckpt` the next-move slot:
   ```powershell
   python frontend\server.py --ckpt "solo_queue\model\checkpoints\masking-finetune\best_mask.pt"
   python frontend\server.py --policy-ckpt "solo_queue\model\checkpoints\nextpick-policy\best_policy.pt"
   ```
2. **Header dropdowns** — every `.pt` under `<track>/model/checkpoints/<name>/`
   (both tracks, one list) and `frontend/uploaded/`, **split by the slot each
   one can fill**. The kept artefacts are named for the inference they serve,
   and the name is the whole rule — the track does not affect it:

   | prefix | slot | what it answers |
   |---|---|---|
   | `masking-` | role model | masked completion + the role board |
   | `nextpick-` | next-move model | next-move ranking |
   | `llm-` | next-move model | a **folder**, not a file: the LLM reranker (above) |
   | `value-` | neither | outcome head only, not servable |

   Paths from before the rename (and uploads) fall back to the old family rules,
   and anything still unrecognised is offered in both slots and validated on
   load. Refreshed when a select gains focus (so freshly trained runs appear
   without a restart), then press **Load**.
3. **From file…** — opens the OS file picker; the file is POSTed to
   `/api/upload`, saved as `frontend/uploaded/<name>.pt` and filed into the slot
   its kind belongs to (a file that fails to load is deleted).

Finetune payloads store the shared encoder under `encoder.` plus the win/econ
outcome `heads.`; the server strips the prefix, drops the heads and serves
suggestions from the finetuned encoder's champ/role heads. `best_mask.pt` is
the right pick for suggestions (`best_win.pt` is calibrated for win
probability, not champ prediction). A policy checkpoint carries
`policy_head.*` and is routed to the next-move path instead.

Endpoints: `GET /api/checkpoints` (each entry carries `roles`, the slots it can
fill; `current` is per slot), `POST /api/load {"path": …, "role":
"masked"|"policy"|"auto"}` (paths must resolve inside the project; a null
`path` unloads that slot), `POST /api/upload?name=x.pt` (raw bytes, 512 MB
cap), `GET /api/drafts?q=&limit=&offset=&random=` and `GET /api/draft?i=<row>`
for the walkthrough, `/api/champions` (champions carry dDragon `img` URLs;
`model` and `policy_model` describe the two slots; `patches` is the loaded
policy checkpoint's own patch vocabulary, newest first, with `default_patch`
and `default_patch_id`), `POST /api/suggest`
(always the role model; response adds `roles` — the role distribution per
filled slot —, `assignment` — one role per pick, distinct within a team —,
per-candidate `flex`, and `fearless_filtered` — how many pool champions were
removed from the answer —), and `POST /api/next-move` (always the next-move
model):

```json
{"picks": ["Ezreal", null, "…"], "bans": ["Ashe", "…"],
 "first_pick": 0, "fearless": ["Vi", "…"],
 "patch_id": 106, "k": 8, "lane": 3}
```

`fearless` is the series pool in the same id space as `picks`/`bans`; it is
optional and defaults to empty, and it is excluded from the answer on both
picks and bans. `patch_id` is a row of the loaded checkpoint's patch table
(omitted → its `default_patch_id`, 16.20 on every kept checkpoint). `lane` is
only read when the loaded next-move model sets `accepts_lane` — the reranker
does, the head does not, and the server forwards the field conditionally rather
than letting a stale key reach a model that would ignore it. For the reranker it
is a *constraint*: the head's pool is masked to champions that play that lane.

→ `{model, state{move_idx, kind, side, phase, first_pick, warning, complete,
n_unavailable}, patch_id, patch, suggestions[{idx, prob, name}], roles,
rerank}`. With the reranker loaded, `rerank` carries `{k, chosen, chosen_idx,
lane, lane_name, lane_source, lane_assumed, lanes_open, lane_filter{applied,
n_legal, of}, head_top1_agrees, candidates[{idx, name, head_prob, prob, logp,
chosen}]}` and each suggestion in
the shortlist gains `llm_prob` + `llm_pick`. `prob` is always the answering
model's own number, so the head's probability and the LLM's are never mixed in
one field — a suggestion outside the shortlist simply has no `llm_prob`.
`rerank` is `null` for a head, and `{error: …}` if the LLM fails at request
time, in which case the head's ordering still stands.
`patch_id` is the row the model actually used: an unknown or out-of-range
request comes back as `0` (UNK) with `patch: null`.
`GET /api/draft?i=` also returns `unavailable` (the real fearless pool, in the
same id space) and `fearless` (whether the series used it).

### Breakdown page (`/breakdown.html`)

A separate page that answers "what would each model have picked here?" for one
real pro game. Pick a game (same 46.6k-draft list as the walkthrough), tick the
models (the checkpoint list plus an upload), and it scores the whole draft
under each one:

- **Policy** models score all 20 moves with the test-split methodology
  (`pro/model/test_split/eval_test_split.py`): the real patch id, exact role-keyed
  slots, and the real move's rank over the model's full next-move
  distribution. They cover bans and picks.
- **Masked** models hide one pick slot at a time and rank the real pick. The
  displayed top-5 is filtered to legal champions (not already taken / banned);
  the reported rank is over the full distribution. They show "—" on bans.
- The scoreboard reports top-1 / top-5 / mean rank per model. In the per-move
  table each cell is green when the real move was that model's #1, amber when
  it was in the top-5, red otherwise; click a row for every model's top-5 with
  champion icons.

`POST /api/breakdown {"game_i": <row>, "models": ["<repo-relative .pt>", …]}`
→ `{game, models:[{path, label, kind, val_top5, score{top1, top5, mean_rank,
n}, rows{<move>: {real_*, top1_*, top5[], target_rank, target_prob,
hit_top1, hit_top5}}}]}`. It does **not** touch the live slots — predictors
are cached per path, so the header models and the page stay independent.

## Behavior

- Blue/Red panels: 5 ban + 5 pick slots **in a horizontal strip, no roles
  attached** — click an empty slot for the searchable picker (champion art
  from dDragon), ✕ removes. `Reset draft` clears the board of the **game on
  screen**; the series and its fearless pool are left alone.
- **First pick** (header, `blue`/`red`): which side opens the draft. Mirrors
  the whole 20-move order; the next-move panel follows it. Stored **per
  game** — switching games restores that game's value. Only meaningful when
  no bans are placed yet (disabled during a walkthrough).
- **Next-move highlight**: the slot that would move next in a standard pro
  draft (bans B,R,B,R,B,R → picks B,R,R,B,B,R → bans R,B,R,B → picks
  R,B,B,R; mirrored wholesale for a red first-pick) is outlined and the team
  header shows "▶ to ban"/"▶ to pick". Highlight only — nothing is enforced,
  any slot can be filled any time. The order comes from `GET /api/champions`'s
  `move_plan` (built from `infer.move_plan`, checked against the pro-data
  pipeline), and `draft-order.js` resolves the next turn **synchronously**
  from the board's ban/pick counts. It never waits for `/api/next-move`: that
  answer belongs to the board it was requested for, so reading it here used
  to leave the highlight one move behind (blue banning twice at the start, a
  fourth ban slot appearing in ban phase 1). The next-move panel blanks itself
  while its answer is in flight for the same reason.

### Series and fearless

A **series** is a list of games that share a fearless pool. Two bars sit under
the header:

- **Series** — one chip per game (`G1 ✓ B 6/10`): game number, the side with
  first pick (`B`/`R`), and picks placed. Click a chip to switch games; each
  game keeps its own board and its own first pick. `＋ game` appends an empty
  game, `clear` forgets the series.
- **next game in series ▶** — closes the current game and opens the next one,
  whose pool is *this game's pool + everything picked in it*. The first-pick
  side is kept, not flipped: the teams swap sides between games but the side
  holding first pick stays the same in 94% of pro games (18,614 vs 1,139 in
  the corpus). The button is disabled on an already-closed game, since its
  successor is already in the series.
- **Fearless** — the pool itself: champions that cannot be picked again.
  `＋ fearless ban` adds one by hand, ✕ removes it.

**The pool is per game, not one running list.** Game *i*'s pool is the picks of
games `0..i-1`, which is exactly how the corpus defines it
(`pipeline.attach_fearless`; verified on all 10,021 fearless rows in
`pro_drafts.pt`, zero mismatches). A running list would make a *closed* game
illegal against itself — its own picks would be in its pool — so switching
back to G1 after starting G2 would show a board the server rejects.

**Which models can use it.** The two `nextpick-*` models take the pool as
`unavailable_mask` and mask it on **pick moves only**, so a champion picked in
an earlier game is still legal to *ban* — the rule the pro corpus follows.
The three `masking-*` models have no such input, so the server masks their
*output* instead: the pool is set to `-inf` and the softmax renormalised
(`_top5`), which makes the answer legal by construction. That turns out to be
enough for the role board — on the four fearless games of the test split the
real pick is still in the masked top-5 0.95 / 0.875 / 0.625 of the time
(`masking-proonly` / `masking-finetune` / `masking-soloqueue`), against 0.475
for a role-frequency prior. It is not fearless *understanding*: the order of
the legal champions is unchanged and the pool still holds 11–19% of the
probability mass before masking, so the bar says "role board filters them out
only" whenever a pool is active.
`pro/model/test_split/eval_fearless_mask.py` reproduces the numbers.

**Real games seed it.** Loading a walkthrough game sets the pool from that
game's own `unavailable`, so a G3 walkthrough starts with the 20 champions
that are actually gone; the info line shows `first pick red · fearless (20
gone)`. Leaving the walkthrough clears it.

State lives in `localStorage` (`draftSeries.v1`), so a refresh keeps the
series. `clear` is the way out.

### Next-move panel (policy slot)

A panel sits under the header whenever a policy checkpoint is loaded: **"Blue
to ban · move 1/20 · bans1 · patch 16.20"**, eight champion chips with the
policy's probability for the side on the clock. Clicking a chip places the
champion and the whole panel re-runs for the next move. It runs alongside the
role board, not instead of it.

- **Patch** (header selector, default 16.20) is a model input, not a label:
  the value is a row of the loaded checkpoint's patch embedding. The list
  comes from `GET /api/champions` (`patches`), so it can only offer patches
  that checkpoint was trained on; an `UNK (unseen patch)` entry covers the
  next patch, and the control is disabled when the checkpoint has no patch
  table. Loading a real game selects that game's patch (when the checkpoint
  knows it), so a 16.09 walkthrough is not conditioned on 16.20. The same
  selection is the default patch for the "why this move?" notes. It matters:
  on `nextpick-proonly` the opening ban is champion 51 at 11.3% on 16.20,
  versus a flat 4.1% for champion 6 on the UNK row.
- **One ranked list, no second opinion.** A ban suggestion is the same
  distribution as a pick suggestion — the policy's imitation of what a pro
  does on that move. There is deliberately no separate "denial" ranking: a
  pro-like ban already encodes denial, and the removed rollout ranked by a
  simulated objective the value head could not measure (`STATUS.md` §5).
- **Fearless champions never appear**, on a pick or a ban. The board accepts
  one as an already-placed ban (8 corpus rows do it — legal, never sensible),
  but the legality mask folds the pool into `taken`, so it cannot be
  suggested.
- The **role board** row belongs to the masked slot and is hidden only when
  *no* role model is loaded (`body.no-role-model`) — the masked model is the
  only family that can answer "what is this team still missing at MID". Every
  pick still carries its role badge (assigned role + confidence, amber when
  the runner-up role is ≥ 15%, red when it disagrees with the pro's actual role
  in a walkthrough).
- An illegal board (more bans than a phase allows, 6+ bans on one side, a
  champion twice) is rejected by the endpoint and shown as a board warning
  rather than silently repaired.

### Explaining a recommendation (`/api/explain`)

`POST /api/explain` takes the same body as `/api/next-move` and adds three
things on top of the top-k: **comparable boards from the pro corpus**, **the
patch notes for the candidates**, and **an explanation composed from the
model's own numbers**. No API key, no network.

```bash
curl -s localhost:8777/api/explain -H 'Content-Type: application/json' \
  -d '{"picks":[null,null,null,null,null,null,null,null,null,null],
       "bans":[1,2,3,4,5,6],"first_pick":0}'
```

- **Retrieval** (`frontend/llm.py: ProIndex`) scans all 46,875 pro drafts. Each
  row keeps its final draft as one 173-bit integer, so the coarse pass is a
  `bit_count()` sweep (~20 ms); the top 600 are then re-ranked on the champions
  that were on the board **at the query move**, not on the final draft. A row
  that shares champions with you *later* is not evidence about this decision.
  Rows are filtered to the same move index and first-pick side; if that leaves
  fewer than `k_games`, the side filter is dropped and `relaxed_side` says so.
- **Patch notes** come from `data/patch notes/clean/patch_<M>_<m>.md`, the
  draft-relevant subset. **That corpus is local only** — it is git-ignored and
  a fresh clone does not have it (see `.gitignore`; the evidence against it as
  a training signal is in `STATUS.md` §6). With the corpus present the
  explanation can cite patch changes; without it `patch.available` is `false`
  and nothing claims anything about patch changes.
  `patch` in the body overrides which patch is used; the default is the newest
  patch in the pro corpus. `patch.available` also separates "the notes were
  read and this champion was not changed" from "no notes for this patch", so
  `missing` is never ambiguous.
- **The explanation** (`llm.compose_local`, `explanation.local: true`) is built
  from four things, each of which the caller can check against the same
  response:
  - the answering model's probabilities (`suggestions`, `llm_prob` when the
    reranker ran) — the top entry, the runner-up, the spread;
  - whether the reranker **overrode** the head (`rerank.head_top1_agrees`), in
    which case the prose says "reordering", never "first choice at 12 %" while
    another entry has 30 %;
  - a count over the retrieved games ("3 of the 5 comparable drafts took X
    next"), which becomes "no consensus" when the modal count is 1;
  - the first line of the patch note for the recommended champion, when this
    patch touched it.

  It is deliberately **not** written by the local LLM. The adapter in
  `pro/model/llm/` is a reranker trained to emit one champion name from a
  shortlist; it has never written a sentence, and its generation does not even
  stop at the name (`loss_on_eos=False`). Prose from it would be invention
  dressed as analysis — the opposite of what this panel is for.
- **Every citation resolves by construction.** A `game` id is one of the
  retrieved games (the panel looks it up and would print "not in the retrieved
  set"), and a `patch` line is copied out of the notes that were parsed. The
  validator, the retry and the `unverified`/`problems` fields still exist for
  the hosted path; the local composer cannot produce an unresolvable one.
- **The hosted path is still there, unused.** `llm.explain(..., client=…)`
  with `client_from_env()` still does retrieve → prompt → validate → cache
  against any OpenAI-compatible endpoint. The button does not call it, so
  `LLM_API_KEY` is optional now (see `.env.example`).
- **Cache**: `frontend/.cache/explain/` (git-ignored) is only used by the
  hosted path. The local composer is a pure function of the response, so it
  needs none.

#### In the UI

**"why this move?"** in the next-move panel opens the explanation. It is a
button rather than automatic because the retrieval costs ~40 ms and the panel
is complete without it.

**"which draft is better?"** next to it opens `/api/draft-score`, which replays
every pick through the same head and compares how close each side came to the
model's own ranking *at the moment it picked* (`frontend/draftscore.py`). Per
side it reports the mean `share` (the pick's probability over the best legal
champion's, so a pick made with 160 champions available is comparable with one
made with 150), the mean rank, and how many times the side took the model's
first choice; the verdict is the difference in mean share, with a 2 pp dead
band. It is **not a win probability** — the policy has no value output worth
shipping — and the panel says so in the response's `note`.

The replay is also the only caller that re-interleaves the ban list: the UI
sends `bans.blue` then `bans.red`, while `build_board` stores bans
chronologically and derives each ban's side from its slot, so a grouped list
would tell the model that blue made the first three bans of an alternating
block. `draftscore.chronological_bans` rebuilds the interleaving from the two
lists; the other endpoints still take the flat list as they always have.

The panel shows, top to bottom: the numbered reasoning steps, the recommended
pick (click it to apply the move), `why` / `their answer` / `risk`, the cited
evidence resolved against the retrieval payload (teams, league, patch, what
they did next, overlap), anything that was **not** shown as evidence and why,
and the full comparable-drafts list so the answer can be audited against what
the model was given.

Three states matter:

- **no key** — the panel still opens and shows the comparable drafts and patch
  coverage, with `llm.reason` saying what to set. The retrieval is the part
  that does not need a model.
- **stale** — the panel hides as soon as the board changes. An explanation of
  the previous board is worse than no explanation.
- **unverified** — if the retry did not fix a citation, the header says
  `unverified citations` and the rejected items are listed in amber rather than
  hidden.

### Role board (masked slot)

- **Role board**: five slots per team, laid out TOP / JUNGLE / MID / BOTTOM /
  UTILITY left to right, **the same card size as the picks** (76 × 93 px) so
  the two strips read as a pair; the role label rides on the icon as a tag
  rather than taking a line of its own. Two things fill it:
  - a **picked** champion sits in its *most likely* role. The role head is
    per-slot (`Linear(D, 5)` over each pick's own embedding), so two picks can
    argmax the same role; the server therefore solves a **one-to-one assignment
    per team** (exhaustive over ≤ 120 permutations). That is why the badge on a
    pick and the slot it occupies always agree — and why a slot never appears
    twice.
  - an **empty** role shows the model's **target** for it: the top-1 champion
    of the role-conditioned MASK query for that role, gold-outlined, with the
    next two candidates as small clickable icons. Clicking the card (or a mini
    icon) adds that pick, which is the fastest way to try a suggestion.
- **The pick you just made wins its role.** `_assign_roles` takes the slot the
  client just filled (`last_slot`) and pins it to its own argmax *before*
  optimising the rest. A pure sum-of-probabilities optimum is wrong here: pick
  Vi (JUNGLE .97) onto a team that already shows Pantheon (JUNGLE .60, TOP .30)
  and the optimum keeps Pantheon in JUNGLE and pushes Vi to TOP — the board
  looks like it ignored the click. Pinning is lexicographic, so the newest pick
  always keeps the role the model likes best for it and the older ones move
  aside (Pantheon → TOP in that example).
- **A board is only ever drawn from an answer computed for it.** Responses
  carry the board signature they were computed for; `renderRoles`/`renderSlots`
  use a response only while it still matches, and a board change dims the
  cards (`…`, no targets, no confidences) until the reply lands. Rendering an
  old assignment against new picks is what made a fresh pick land in an old
  role while the role it belongs to kept its old target. A rejected board
  (duplicate champion, etc.) now shows the reason in a red bar instead of
  silently freezing the previous board.
- **Per-role MASK queries**: top-5 for all 5 roles × both teams (≤ 6 tiny
  forwards per board change); bans and already-picked champions are
  post-filtered out. Once a team has all five roles filled, every slot is
  filled and no target cards remain.
- Output contract (solo_queue/model/README.md §2): `[MASK]`/`[PAD]`/FLEX exist only on
  the **input** side — the champ head is `Linear(D, 173)` and the role head
  `Linear(D, 5)`, so a suggestion can never be a special token and a role
  prediction can never be FLEX (hard-filtered + regression-tested). The amber
  `flex` styling on a pick badge is a post-hoc annotation (runner-up role
  ≥ 15%), not a class the model predicts.

#### Why the role board keeps its own model (2026-11-10)

The obvious simplification — drop the masking checkpoint and let the reranker
fill the board — does not work. `python frontend/probe_roleboard.py 400` scores
it against the two alternatives on 400 held-out games; the target for every
cell is the champion that team actually fielded in that lane, and the number is
whether it is the top-1 / in the top-5:

| arm | 6 bans, no picks | 6 bans + 6 picks | 10 bans + 6 picks |
|---|---|---|---|
| masked board (shipped) | 6.4 % / 30.0 % | 7.8 % / **37.1 %** | 9.4 % / 40.6 % |
| per-lane popularity table | **8.9 % / 33.4 %** | 9.5 % / 34.6 % | 10.5 % / 37.4 % |
| next-move head, lane-masked | 3.6 % / 17.3 % | — | 12.1 % / 42.6 % |
| ↳ same head, lanes it is *not* picking | 1.9 % / 17.4 % | — | 7.8 % / 31.5 % |
| ↳ masked board, same cells | 5.8 % / 29.6 % | — | 7.5 % / 35.0 % |
| uniform over the lane's legal pool | 0 % / 11.5 % | 0 % / 11.3 % | 0 % / 11.8 % |

Three independent reasons the reranker cannot take the slot:

1. **A reranker ranks; it does not generate.** Every cell needs a shortlist
   first, and the only generator is the head — so "just the rerank model" is
   still two models, one of them the thing it was supposed to replace.
2. **The head only answers for the side on the clock.** `move_plan` returns one
   `(kind, team)` per move index, so five of the ten cells have no valid input
   at all. Its 42.6 % is not a counterfactual number either: on the lanes that
   side is *not* picking now it drops to 31.5 %, **below** the masked board's
   35.0 % on the same cells. It wins only on the single lane it is actually
   picking (53.7 %), which is the next-move panel's question, not this one.
   *(Amended later the same day: the side is not actually fixed —
   `_policy_batch` takes the team as an argument, so the off-turn cells can be
   posed as hypotheticals. Done that way the head does beat the masked board,
   +3.8 pp top-1 at 10 bans + 6 picks; see "The fabricated turn" below for the
   number and for the reason the two sides still come out the same.)*
3. **Latency.** Ten lane queries is ten LLM passes (~4.5 s at 0.45 s each), or a
   50-row batch on the card that had to train at batch 2. The board is
   recomputed on every click and takes 11 ms.

What the same run does say, and is worth knowing before trusting the panel: the
masking model is **+2.5 to +3.2 pp top-5 over a static table once picks are on
the board, and −3.4 pp at the empty board**, where the table's single
most-picked champion is a better first chip. So it does real but modest work —
it is a prior with a little context, not a decision model. And role
*inference* no longer needs it at all: `lanes.assign_lanes` decides `role_ids`
from the champion table (96 % agreement with real drafts, against the role
head's 67 %).

The conclusion is therefore **conditional on the ban block**, which is what the
fabricated-turn section below adds: once all ten bans are in, the lane-masked
head plus the reranker beats this board outright, and the masking checkpoint's
remaining value is the states where the ban block is still filling. Its slot is
not earned by "the reranker cannot generate" — the head generates — but by the
ban-phase boards the head cannot be honestly asked about.

#### What answers the board now (2026-11-10)

`/api/suggest` prefers the head and falls back to the masking checkpoint, and
says which one answered in the response's `method`:

| next move | `method` | why |
|---|---|---|
| a pick with all ten bans in (moves 16–19) | `head` | `next_move(..., allowed=legal_champions(lane), team=…)` — one lane-masked query per open cell, 15 ms each; 15.6 % top-1 / 49.5 % top-5 against this board's 10.5 / 44.2, p < 0.005 |
| anything else, including picks1 | `masked` | a pick-phase hypothetical on a ban-phase board is out of distribution (p = 0.81), and at 6 bans + 3 picks the head's top-5 is 39.9 % against 45.7 % — three picks is not enough board for it |

The first cut of this routed *every* pick move to the head. Picks1 says no: the
head needs six picks on the board before it beats a model built for exactly
this question, so the rule is "ban block complete", not "next move is a pick".

The head is reached through `pred.head` when the policy slot holds the LLM
reranker, because the reranker's own `next_move` would spend 0.45 s per cell on
a question the head answers in 15 ms. The response shape is unchanged, so the
panel needed no edit; the `model` block now names the model that actually
answered, not the policy slot's contents.

Two things this does not do. It does **not** differentiate the two sides (see
below — the head is nearly side-blind by construction), and it does **not**
remove the masking checkpoint from the install, because the ban-phase boards
still need it.

#### Zero-shot generation, tried and measured (2026-11-10)

The other way to drop a checkpoint is to let the LLM answer the cell directly,
with no shortlist. It is the **v2 prompt — the one the base model was trained
to answer** — with only the target cell changed, so nothing new has to be
rendered:

```
<|im_start|>system
You are an expert League of Legends draft engine.<|im_end|>
<|im_start|>user
<10 encoder soft tokens>
Patch: 16.20 | Turn: Blue 4 (Pick 7 of 10)
Blue Picks: Dr. Mundo (Top), Jarvan IV (Jungle), Aphelios (Bot)
Red Picks: Aurora (Top), Viego (Jungle), Yone (Mid)
Blue Bans: Renata Glasc, Lucian, Naafiri, Ziggs, Draven
Red Bans: Vladimir, Caitlyn, Jayce, Viktor, Lulu
Task: Recommend the optimal Blue Side pick for Mid lane. Answer with champion name only.
Answer:
```

The turn line is rebuilt so the asked-for side is the side on the clock: the
role board is a counterfactual, and the model has only ever seen a state where
the side being asked about is the one picking. `pro/model/llm/probe_roleboard_gen.py`
runs it two ways — greedy generation, and a readout of the model's own
next-token distribution at `Answer:` (the `ft_top*` mechanism, no decoding):

| arm | top-1 | top-5 | per board |
|---|---|---|---|
| masked board | 9.4 % | **40.6 %** | 11 ms |
| per-lane popularity table | **10.5 %** | 37.4 % | ~0 ms |
| LLM readout, `llm-lora-rerank5` | 9.4 % | 34.3 % | 0.9 s |
| LLM readout, `llm-lora-full2` | 8.7 % | 29.4 % | 1.3 s |
| LLM generation, `llm-lora-rerank5` | 7.2 % | one name | 0.6 s/cell |
| LLM generation, `llm-lora-full2` | 7.9 % | one name | 0.7 s/cell |

Same board (10 bans + 6 picks); 80 games for the readout rows, 40 for
generation, so cross-row differences under ~4 pp are not resolvable. It answers,
and every generation was a real champion name — it is just not better, and it is
60–120× slower. Two reasons that are structural rather than tuning:

* **Generation returns one name, not five.** A cell shows a top-1 plus two
  alternatives; five samples per cell is five times the cost, and greedy decode
  cannot produce a ranking at all.
* **The readout is limited by the tokenizer, not by the model.** Its top-5
  *first tokens* contain the answer 45–51 % of the time, but 24 collision groups
  cover 61 of the 173 names, so expanding five groups and keeping five names
  drops that to 29–34 %. The masked model's champion head is 173-way in one
  step and does not have this problem.

One artefact worth knowing if this is ever revisited: the rerank adapter does
not stop after the name — it never learned `<|im_end|>` (`loss_on_eos=False` in
training), so the raw output runs on into `Blue Picks:` and the name has to be
parsed off the front.

#### The fabricated turn, measured (2026-11-10)

The third way to fill a role-board cell is to keep the shortlist, but ask the
head a question it was never asked at serving time: *"suppose this side were
picking next, for this lane"*. `next_move` takes the side from `move_plan()`,
so the only way to pose it is to build the policy batch with that team —
`_policy_batch(board, move_idx, MOVE_PICK, team, …)` — mask the logits to
`lanes.legal_champions(lane)`, take five, and let the reranker choose. The
reranker's task line already says `Choose the best Red Side pick for Jungle
lane from the candidates`, so nothing in the prompt changes.

`python frontend/probe_roleboard_fab.py 200 10 6` scores all arms on the
**same cells** (this is the only safe
comparison — `data.eval_dataset` draws a fixed random subsample, so two probes
with different `N` do not share a sample):

| arm | top-1 | top-5 | 774 cells, 10 bans + 6 picks |
|---|---|---|---|
| masked board (shipped) | 10.5 % | 44.2 % | 12 ms |
| per-lane popularity table | 11.9 % | 41.1 % | ~0 ms |
| fabricated-turn head | 14.3 % | **49.5 %** | 15 ms |
| fabricated turn + rerank | **15.6 %** | (the head's five) | 594 ms |

Paired, on the same cells: masked right / rerank wrong 59, the reverse 99,
exact p < 0.005; against the bare head 57 / 87, p = 0.02. So the lane-masked
head is +3.8 pp top-1 on its own and the reranker adds +1.3 pp more, +5.3 pp
top-5 — the first variant measured that beats the masked board rather than
tying it. (A 60-game run of the same script showed 12.6 vs 13.9 % and looked
like a tie; at 231 cells the intervals are ±2.2 pp and that is what noise looks
like.)

It does **not** survive an incomplete ban block. At 6 bans + 6 picks the next
move is a ban, so a pick-phase hypothetical has to pretend the remaining bans
are in (`move_idx = 10 + n_picks` while `ban_ids` holds six):

| arm | top-1 | top-5 | 388 cells, 6 bans + 6 picks |
|---|---|---|---|
| masked board | 10.1 % | **46.1 %** | 12 ms |
| popularity table | 11.9 % | 41.8 % | ~0 ms |
| fabricated-turn head | 8.8 % | 38.7 % | 14 ms |
| fabricated turn + rerank | 10.8 % | (the head's five) | 565 ms |

p = 0.81 against the masked board: a tie, with the head's own top-5 four points
worse. The board the head is shown is not a board it ever saw — six bans with a
move index that says ten — so this is a fabrication cost, not a model limit.

Nor does it win in picks1, where nothing is fabricated. At 6 bans + 3 picks
(696 cells) the next move really is a pick and the head answers it in
distribution:

| arm | top-1 | top-5 | 696 cells, 6 bans + 3 picks |
|---|---|---|---|
| masked board | 10.2 % | **45.7 %** | 8 ms |
| popularity table | 10.8 % | 37.6 % | ~0 ms |
| fabricated-turn head | 9.1 % | 39.9 % | 13 ms |
| fabricated turn + rerank | **12.1 %** | (the head's five) | 487 ms |

p = 0.28 on top-1 and the head's recall is six points worse. So the win is not
"the head is better at lanes" — it is "with six picks on the board and all ten
bans in, the head has enough context to beat a model that answers each cell
from the champion alone". That is why `/api/suggest` routes on the ban block
rather than on the move kind (see "What answers the board now").

**The two sides never differentiate.** `python frontend/probe_roleboard_side.py`
asks both sides the same
board (52 shared cells, 10 bans + 6 picks): total-variation distance between
the Blue and Red lane distributions is **0.015**, against **0.147** for a
control that swaps one enemy pick — a real board change moves the answer ten
times more than the side does. 90 % of shared lanes get the identical five, and
the reranker picks a different winner in 4 % of them. The reason is structural:
`to_move` enters the model only as `side_embed(to_move)`, one 64-d vector added
into the state embedding, and the pick counts already determine whose turn it is
— so during training the flag was redundant and it carries almost no weight
(sd of its effect on the logits: 0.05, max 0.17). "Suppose Red were picking
Jungle" therefore does not produce a Red-specific answer; it produces the best
Jungle champion on this board. Making it counter-pick aware is a model change
(an explicit my-team/enemy-team split in the encoder), not a serving change.


### Both slots at once

- **Walkthrough of real games**: the "Real draft" bar lists the 46.6k pro
  drafts (filter by team/league/date/gameid, or 🎲 random). Picking one shows
  date, league, teams, game number, patch and winner; ◀ / ▶ (or the arrow
  keys) step through the 20 real moves, and the compare line shows what each
  model suggested at that point vs what the pros actually did — "policy ranked
  it #8 (2.8%)" from the policy slot, or "model ranked it #2 for TOP (17.4%) ·
  also MID #4" from the role slot. Each line is only shown while its answer
  still matches the current board.
- Champion art: `https://ddragon.leagueoflegends.com/cdn/16.20.1/img/champion/…`
  resolved through the dDragon manifest (cached in `frontend/.cache/`; a
  deterministic name-based fallback keeps URLs sane offline — the images
  themselves need internet).
- Inference: the role slot is 1 FLEX-role pass (role head over the picked
  champs → the one-to-one role assignment) plus role-conditioned MASK passes
  covering every role of both teams, with each picked champion held at its
  *assigned* role as context; bans/picks filtered from logits post-hoc. The
  policy slot is 1 role-head pass to infer the picked champions' roles, then
  1 forward for the move. The header shows each slot's checkpoint, its kind
  and its val metric.

## Limits (prototype)

- The UI board carries no roles, so the next-move path infers each picked
  champion's role from the encoder's role head before running the policy. A
  pro row is handed the real role, so a UI answer is not a like-for-like
  reproduction of the test-split number.
- Phase-1 bans are deliberately not a target: pros ban specialists and game
  plans there, which a symmetric model cannot represent. The ranked list is
  an imitation of what pros actually do, which is the honest answer available
  here — it does not claim a ban is *good*.
- A team's five role lists come from (5 − picks) MASKed slots per pass, so once
  several picks are filled, later roles are conditioned on slightly different
  empty-slot role inputs than earlier ones (≤ 6 tiny forwards).
- Which slot a checkpoint fills is guessed from its path (`_ckpt_roles`), not
  from the payload: reading forty checkpoints at startup to classify them is
  not worth it. A wrong guess is caught on load (`not a masked one`), never
  silently.
- The defaults (no `--ckpt` / `--policy-ckpt`) are resolved once at startup;
  freshly trained runs are reachable via the dropdowns without restarting, but
  the startup defaults only update when you relaunch.
- Walkthrough data comes from the cached `data/pro play/pro_drafts.pt` (46.6k
  games, loaded lazily on first use).

Tests: `python -m pytest frontend/test_server.py -q` and
`python -m pytest solo_queue/model/policy/test_infer.py -q`. The endpoint test pins the
published `move_plan` to `infer.move_plan`, and a source-level guard keeps
`renderTurn` board-derived rather than model-derived.
