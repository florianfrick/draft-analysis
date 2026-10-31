# Draft Assistant (local prototype)

Local-only web UI with **two modes**, selected by whichever checkpoint you
load:

| loaded checkpoint | mode | what you get |
|---|---|---|
| pretraining / finetune (`pma.*`, `heads.*`) | **masked completion** | top-5 suggestions for every empty role slot on both teams |
| policy (`policy_head.*`) | **next move** | one ranked list for the side on the clock, plus denial-ranked ban candidates |

The family is read from the state-dict keys, not the filename, so any of the
four kept checkpoints just works.

## Run

```powershell
cd E:\VSCode\code\lol draft analysis
python frontend\server.py --port 8777     # http://127.0.0.1:8777
```

Binds `127.0.0.1` only. `--ckpt <path>` pins a checkpoint (default: the
`best.pt` with highest val champ-top5 across `model/pretraining/ckpt/run_*/`),
`--value <path>` pins the value model used by ban ranking (default: the newest
`model/finetune/ckpt/value_*/best.pt`), `--device cuda` for GPU inference (CPU
default keeps the GPU free).

Through Docker, pass `--host 0.0.0.0` (a published port arrives on the
container's `eth0`, not its loopback) — see `DOCKER.md` §"The frontend".

## Choosing a model

The header always shows the active model (label, kind, val metric). Three ways
to switch:

1. **CLI** — `--ckpt` accepts any family:
   ```powershell
   python frontend\server.py --ckpt "model\policy\ckpt\run_20261025_000606_ablF-stage1\best_policy.pt"
   python frontend\server.py --ckpt "model\finetune\ckpt\run_20261017_185016_stage1-full\best_mask.pt"
   ```
2. **Header dropdown** — every `best_*` file under `model/pretraining/ckpt/run_*/`,
   `model/finetune/ckpt/run_*/`, `model/policy/ckpt/run_*/` and
   `model/uploaded/`; refreshed when the select gains focus (so freshly trained
   runs appear without a restart), then press **Load**.
3. **From file…** — opens the OS file picker; the file is POSTed to
   `/api/upload`, saved as `model/uploaded/<name>.pt` and loaded (a file that
   fails to load is deleted).

Finetune payloads store the shared encoder under `encoder.` plus the win/econ
outcome `heads.`; the server strips the prefix, drops the heads and serves
suggestions from the finetuned encoder's champ/role heads. `best_mask.pt` is
the right pick for suggestions (`best_win.pt` is calibrated for win
probability, not champ prediction). A policy checkpoint carries
`policy_head.*` and is routed to the next-move path instead.

Endpoints: `GET /api/checkpoints`, `POST /api/load {"path": …}` (paths must
resolve inside the project), `POST /api/upload?name=x.pt` (raw bytes, 512 MB
cap), `GET /api/drafts?q=&limit=&offset=&random=` and `GET /api/draft?i=<row>`
for the walkthrough, `/api/champions` (champions carry dDragon `img` URLs),
`POST /api/suggest` (masked; response adds `roles` — likely role per filled
slot — and per-candidate `flex`), and `POST /api/next-move`:

```json
{"picks": ["Ezreal", null, "…"], "bans": ["Ashe", "…"],
 "first_pick": 0, "k": 8, "ban_k": 4, "rollout": true}
```

→ `{model, state{move_idx, kind, side, phase, first_pick, warning, complete},
suggestions[{idx, prob, name}], roles, bans{side, opponent, ranked_by,
win_prob_available, opponent_picks, candidates[{rank, idx, name, denial,
win_prob}]}}`. `bans` is `null` on pick moves and on a complete board.

## Behavior

- Blue/Red panels: 5 ban + 5 pick slots **in a horizontal strip, no roles
  attached** — click an empty slot for the searchable picker (champion art
  from dDragon), ✕ removes. `Reset draft` clears everything.
- **First pick** (header, `blue`/`red`): which side opens the draft. Mirrors
  the whole 20-move order; the next-move panel follows it. Only meaningful
  when no bans are placed yet (it is disabled during a walkthrough).
- **Next-move highlight**: the slot that would move next in a standard pro
  draft (bans B,R,B,R,B,R → picks B,R,R,B,B,R → bans R,B,R,B → picks
  R,B,B,R; mirrored wholesale for a red first-pick) is outlined and the team
  header shows "▶ to ban"/"▶ to pick". Highlight only — nothing is enforced,
  any slot can be filled any time.

### Next-move mode (policy checkpoint)

A panel sits under the header: **"Blue to ban · move 1/20 · bans1"**, eight
champion chips with the policy's probability for the side on the clock, and —
on ban moves — a ban-candidate list. Clicking either places the champion and
the whole panel re-runs for the next move.

- `ban rollout` (checkbox, default on) enables the ban-candidate list; it costs
  ~0.6 s per query because it plays the rest of the draft `ban_k` times.
- **Ban candidates** are ranked by **denial** = how likely the opponent was to
  take that champion, with `win%` = the value model's score of the finished
  simulated draft. Denial is the primary ordering on purpose: it is role-free
  and measured (humans' phase-2 bans land in the opponent's predicted top-5 at
  29.6% vs 8.3% for a frequency baseline). The simulated win% comes from a
  value model whose resolution is weak on this comparison, so treat it as a
  tiebreaker, not a decision.
- Without `--value` (or with a missing value checkpoint) the list still works,
  ranked by denial alone, and the subtitle says so.
- The per-role suggestion tables are hidden in this mode (`body.policy-mode`) —
  they belong to the masked model. The role head still labels the picks under
  "Likely roles".
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

- **Retrieval** (`frontend/llm.py: ProIndex`) scans all 46,813 pro drafts. Each
  row keeps its final draft as one 173-bit integer, so the coarse pass is a
  `bit_count()` sweep (~20 ms); the top 600 are then re-ranked on the champions
  that were on the board **at the query move**, not on the final draft. A row
  that shares champions with you *later* is not evidence about this decision.
  Rows are filtered to the same move index and first-pick side; if that leaves
  fewer than `k_games`, the side filter is dropped and `relaxed_side` says so.
- **Patch notes** come from `data/patch notes/clean/patch_<M>_<m>.md`, the
  draft-relevant subset. `patch` in the body overrides which patch is used;
  the default is the newest patch in the corpus. `patch.available` says whether
  that file exists at all, so `missing` is not ambiguous: with
  `available: true` a champion in `missing` was simply not changed that patch;
  with `available: false` nothing is known and the prompt tells the model not
  to claim anything about patch changes.
- **Grounding** is enforced, not requested. Every `evidence` item must resolve:
  a `game` id must be one of the retrieved games, a `patch` line must appear
  verbatim in the notes that were handed to the model. Unresolvable items are
  dropped, the call is retried once with the failures fed back, and a second
  failure returns the answer with `unverified` populated and `ok: false`.
  A fabricated citation never reaches the UI silently.
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

### Masked mode (pretraining / finetune checkpoint)

- **Likely roles**: below each strip, every picked champion gets its predicted
  role from the model's role head (FLEX-role query); a gold second badge
  appears when the runner-up role is ≥ 15% (flex), and a red badge when the
  model disagrees with the pro's actual role (walkthrough).
- **Suggestions per role**: top-5 for all 5 roles × both teams from
  role-conditioned MASK queries (≤ 6 tiny forwards per change); champs
  suggested under 2+ roles of the same team carry a `flex` badge; click a chip
  to add the pick. Bans/picked champs are post-filtered out.
- Output contract (model/README.md §2): `[MASK]`/`[PAD]`/FLEX exist only on
  the **input** side — the champ head is `Linear(D, 173)` and the role head
  `Linear(D, 5)`, so a suggestion can never be a special token and a role
  prediction can never be FLEX (hard-filtered + regression-tested). The UI's
  `flex` badge is a post-hoc annotation (same champ appearing under 2+ role
  lists), not a class the model predicts.

### Both modes

- **Walkthrough of real games**: the "Real draft" bar lists the 46.6k pro
  drafts (filter by team/league/date/gameid, or 🎲 random). Picking one shows
  date, league, teams, game number, patch and winner; ◀ / ▶ (or the arrow
  keys) step through the 20 real moves, and the compare line shows what the
  model suggested at each point vs what the pros actually did — e.g. "model
  ranked it #2 for TOP (17.4%) · also MID #4", or in next-move mode "top-5",
  "outside top-8".
- Champion art: `https://ddragon.leagueoflegends.com/cdn/16.20.1/img/champion/…`
  resolved through the dDragon manifest (cached in `frontend/.cache/`; a
  deterministic name-based fallback keeps URLs sane offline — the images
  themselves need internet).
- Inference: masked mode is 1 FLEX-role pass (role head over the picked
  champs, roles fed as context are the model's own predictions) plus
  role-conditioned MASK passes covering every role of both teams; bans/picks
  filtered from logits post-hoc. Next-move mode is 1 policy forward for the
  move, plus `1 + ban_k` short rollouts when the ban list is requested.
  Header shows the active checkpoint, its kind and its val metric.

## Limits (prototype)

- The UI board carries no roles, so the next-move rollout infers each
  simulated pick's role from the champion — the offline proposer is given the
  real role from the pro row. The UI's simulated win% is therefore a harder
  question than the offline number in `STATUS.md` §5, which is part of why it
  is only a tiebreaker here.
- Phase-1 bans are deliberately not a target: pros ban specialists and game
  plans there, which a symmetric model cannot represent. Denial ranking is
  honest about this — it ranks by "they want it", not "it is good".
- A team's five role lists come from (5 − picks) MASKed slots per pass, so once
  several picks are filled, later roles are conditioned on slightly different
  empty-slot role inputs than earlier ones (≤ 6 tiny forwards).
- The default (no `--ckpt`) is resolved once at startup; freshly trained runs
  are reachable via the dropdown without restarting, but the startup default
  only updates when you relaunch.
- Walkthrough data comes from the cached `data/pro play/pro_drafts.pt` (46.6k
  games, loaded lazily on first use).

Tests: `python -m pytest frontend/test_server.py -q` (19) and
`python -m pytest model/policy/test_infer.py -q` (20).
