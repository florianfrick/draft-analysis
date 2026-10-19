# Draft Assistant (local prototype)

Local-only web UI for the SoloQ model: pick/ban champions per
team and get **top-5 suggestions for every empty role slot on both teams**.
Loads **both** checkpoint families: SoloQ pretraining *and* the pro
finetune (`model/finetune/ckpt/…`).

## Run

```powershell
cd E:\VSCode\code\lol draft analysis
python frontend\server.py --port 8777     # http://127.0.0.1:8777
```

Binds `127.0.0.1` only. `--ckpt <path>` pins a checkpoint (default: the
`best.pt` with highest val champ-top5 across `model/pretraining/ckpt/run_*/`),
`--device cuda` for GPU inference (CPU default keeps the GPU free).

## Choosing a model

The header always shows the active model (label, kind, val top-5). Three
ways to switch:

1. **CLI** — `--ckpt` accepts either family:
   ```powershell
   python frontend\server.py --ckpt "model\finetune\ckpt\run_20261017_185016_stage1-full\best_mask.pt"
   ```
2. **Header dropdown** — every `*.pt` under `model/pretraining/ckpt/run_*/`,
   `model/finetune/ckpt/run_*/` and `model/uploaded/`; refreshed when the
   select gains focus (so freshly trained runs appear without a restart),
   then press **Load**.
3. **From file…** — opens the OS file picker; the file is POSTed to
   `/api/upload`, saved as `model/uploaded/<name>.pt` and loaded (a file
   that fails to load is deleted).

Finetune payloads store the shared encoder under `encoder.` plus the
win/econ outcome `heads.`; the server strips the prefix, drops the heads
(no MCTS value function here) and serves suggestions from the finetuned
encoder's champ/role heads. `best_mask.pt` is the right pick for
suggestions (`best_win.pt` is calibrated for win probability, not
champ prediction).

Endpoints: `GET /api/checkpoints`, `POST /api/load {"path": …}` (paths
must resolve inside the project), `POST /api/upload?name=x.pt` (raw
bytes, 512 MB cap), `GET /api/drafts?q=&limit=&offset=&random=` and
`GET /api/draft?i=<row>` for the walkthrough, plus `/api/champions`
(champions now carry dDragon `img` URLs) and `/api/suggest` (response
adds `roles` — likely role per filled slot — and per-candidate `flex`).

## Behavior

- Blue/Red panels: 5 ban + 5 pick slots **in a horizontal strip, no roles
  attached** — click an empty slot for the searchable picker (champion art
  from dDragon), ✕ removes. `Reset draft` clears everything.
- **Next-move highlight**: the slot that would move next in a standard pro
  draft (bans B,R,B,R,B,R → picks B,R,R,B,B,R → bans R,B,R,B → picks
  R,B,B,R; mirrored wholesale for a red first-pick) is outlined and the
  team header shows "▶ to ban"/"▶ to pick". Highlight only — nothing is
  enforced, any slot can be filled any time.
- **Likely roles**: below each strip, every picked champion gets its
  predicted role from the model's role head (FLEX-role query); a gold
  second badge appears when the runner-up role is ≥ 15% (flex), and a red
  badge when the model disagrees with the pro's actual role (walkthrough).
- **Suggestions per role**: top-5 for all 5 roles × both teams from
  role-conditioned MASK queries (≤ 6 tiny forwards per change); champs
  suggested under 2+ roles of the same team carry a `flex` badge; click a
  chip to add the pick. Bans/picked champs are post-filtered out.
- **Walkthrough of real games**: the "Real draft" bar lists the 46.6k pro
  drafts (filter by team/league/date/gameid, or 🎲 random). Picking one
  shows date, league, teams, game number, patch and winner; ◀ / ▶ (or the
  arrow keys) step through the 20 real moves, and the compare line shows
  what the model suggested at each point vs what the pros actually did —
  e.g. "model ranked it #2 for TOP (17.4%) · also MID #4".
- Champion art: `https://ddragon.leagueoflegends.com/cdn/16.20.1/img/champion/…`
  resolved through the dDragon manifest (cached in `frontend/.cache/`;
  a deterministic name-based fallback keeps URLs sane offline — the images
  themselves need internet).
- Inference: 1 FLEX-role pass (role head over the picked champs, roles fed
  as context are the model's own predictions) plus role-conditioned MASK
  passes covering every role of both teams; bans/picks filtered from
  logits post-hoc. Header shows the active checkpoint and its val top-5.
- Output contract (model/README.md §2): `[MASK]`/`[PAD]`/FLEX exist only
  on the **input** side — the champ head is `Linear(D, 173)` and the role
  head `Linear(D, 5)`, so a suggestion can never be a special token and a
  role prediction can never be FLEX (hard-filtered + regression-tested).
  The UI's `flex` badge is a post-hoc annotation (same champ appearing
  under 2+ role lists), not a class the model predicts.

## Limits (prototype)

- The model has no draft-order / ban-phase / fearless modeling and no ban
  predictor — the pro-order logic lives in the UI highlight only, and bans
  are a post-filter. "Likely roles" are the model's flex-query guesses,
  not ground truth.
- A team's five role lists come from (5 − picks) MASKed slots per pass, so
  once several picks are filled, later roles are conditioned on slightly
  different empty-slot role inputs than earlier ones (≤ 6 tiny forwards).
- The default (no `--ckpt`) is resolved once at startup; freshly trained
  runs are reachable via the dropdown without restarting, but the startup
  default only updates when you relaunch.
- Walkthrough data comes from the cached `data/pro play/pro_drafts.pt`
  (46.6k games, loaded lazily on first use).

Tests: `python -m pytest frontend/test_server.py -q`
