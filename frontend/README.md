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
`--policy-ckpt <path>` pins the **next-move model**
(default: `pro/model/checkpoints/nextpick-bans-aware-through-wlds-ig-tes/best.pt`),
`--no-policy` skips it entirely, and `--device cuda` for GPU inference (CPU
default keeps the GPU free).

The role default is the 2026-11-04 refresh (`-randomvalsplit-`: 42,121 train,
no test split). The `masking-proonly` originals are still listed in the dropdown
and are the only ones with a pinned-test score (`STATUS.md` §7); the refresh has
none by construction.

The next-move default is the **ban-aware arm** (2026-11-07): the same pro-only
recipe as the `-randomvalsplit-` role model, with `ban_side`, `ban_fearless` and
`counter_rank=32` on, so the ban block — including which side made each ban and
which champions the fearless pool has already removed — is a model input rather
than something to infer from the absent champions. Against a control that
reproduces the previous model to 4e-5 it is +0.60 pp top-5 (0.5056 vs 0.4995,
paired bootstrap p = 4.3e-8, CI [+0.39, +0.82] pp). The flags are read from the
checkpoint's own `config`, not passed by the frontend, so an older checkpoint
still rebuilds unchanged.

`nextpick-proto-ensemble/ensemble.json` is still selectable but no longer the
default: three pro-only policy arms whose policy logits are averaged before the
softmax. Its +1.33 pp top-5 (bootstrap [+0.64, +2.07], p = 2.6e-4; +0.94 pp
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
 "patch_id": 106, "k": 8}
```

`fearless` is the series pool in the same id space as `picks`/`bans`; it is
optional and defaults to empty, and it is excluded from the answer on both
picks and bans. `patch_id` is a row of the loaded checkpoint's patch table
(omitted → its `default_patch_id`, 16.20 on every kept checkpoint).

→ `{model, state{move_idx, kind, side, phase, first_pick, warning, complete,
n_unavailable}, patch_id, patch, suggestions[{idx, prob, name}], roles}`.
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
patch notes for the candidates**, and — if `LLM_API_KEY` is set — **prose that
must cite one of those two**.

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
  explanation can cite patch changes; without it `patch.available` is `false`,
  the prompt tells the model not to claim anything about patch changes, and
  everything else — retrieval, validation, the endpoint — works unchanged.
  `patch` in the body overrides which patch is used; the default is the newest
  patch in the pro corpus. `patch.available` also separates "the notes were
  read and this champion was not changed" from "no notes for this patch", so
  `missing` is never ambiguous.
- **Reasoning** (`explanation.reasoning`) is asked for explicitly: 2–5 steps,
  in the order the model took them, capped at 6 × 240 chars. It is the model's
  own account — a summary written with the answer, **not** a transcript of
  hidden deliberation — so it is never merged into `evidence` and the UI
  labels it. Only the citations are checked.
- **Grounding** is enforced, not requested. Every `evidence` item must resolve:
  a `game` id must be one of the retrieved games, a `patch` line must appear
  verbatim in the notes that were handed to the model. Unresolvable items are
  dropped, the call is retried once with the failures fed back, and a second
  failure returns the answer with `unverified` populated, `problems` explaining
  each rejection, and `ok: false`. A fabricated citation never reaches the UI
  silently.
- **No key is not an error.** Without `LLM_API_KEY` the endpoint returns
  `explanation: null` plus `llm.reason` and the full retrieval — which is the
  part that has to be right anyway.
- **Cache**: answers are keyed on board + candidates + retrieved games + model
  + prompt version, written to `frontend/.cache/explain/` (git-ignored).
  `"force": true` bypasses it; `PROMPT_VERSION` in `llm.py` invalidates it when
  the answer schema changes.
- Env: `LLM_API_KEY`, `LLM_BASE_URL` (default `https://api.openai.com/v1`, any
  OpenAI-compatible endpoint works), `LLM_MODEL` (default `gpt-4o-mini`),
  `LLM_TIMEOUT` (default 45 s). See `.env.example`.

#### In the UI

**"why this move?"** in the next-move panel opens the explanation. It is a
button rather than automatic because a call costs seconds and, with a hosted
model, money — and the recommendation is already on screen without it.

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
