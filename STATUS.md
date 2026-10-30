# Project status

Last updated: 2026-10-30. Scope: what is trained, what is measured, what is kept.

Four trained artefacts remain, one per stage. Everything else (21 runs,
~7.6GB) was deleted; the conclusions those runs produced are recorded under
[Discarded ablations](#discarded-ablations-and-what-they-showed) so the
evidence survives without the weights.

| stage | kept checkpoint | purpose |
|---|---|---|
| Pretraining (soloQ) | `model/pretraining/ckpt/run_20261029_153912_v2-D512-p8/best.pt` | encoder + value head |
| Finetune (pro, stage-1) | `model/finetune/ckpt/run_20261017_185016_stage1-full/best_mask.pt` | masked draft completion |
| Policy (next-move) | `model/policy/ckpt/run_20261025_000606_ablF-stage1/best_policy.pt` | pick/ban recommendation |
| Prefix value (P4 input) | `model/finetune/ckpt/value_20261030_180102/best.pt` | `V(state)` for search |

Data: 46,813 pro drafts (173 champions, 5 roles, 20 moves), split
train 40,609 / val 2,458 / test 3,189, where val and test are
**patch-honest** (557 rows, 18.5%, dropped for sharing a patch with train).
SoloQ pretraining uses complete drafts only.

---

## 1. Pretraining — soloQ masked modelling

**Architecture.** Set Transformer over the 10 draft slots. Champion + role
embeddings summed, no positional encoding. 6 intra-team layers (Siamese: one
block, applied to each team) and 3 cross-team layers (symmetric), D=512, 8
heads, dropout 0. Opt-in value path: PMA pooling over each team's slot set
(4 seeds) feeding a win head, `with_value=True`.

**Training.** BERT-style masked champion modelling: corrupt 15% of slots,
predict champion (CE over 173) plus role (CE, weight 0.3). 90/10 split between
"role hidden" and "role visible" masks. Warmup + cosine, early stop on val
`champ_top5`, AMP.

**Result** (val, 19,217 complete drafts, epoch 33 / step 11,781):

| metric | model | baseline | ratio |
|---|---|---|---|
| champ top-1 | 0.0849 | 0.0182 (global freq) | 4.7x |
| champ top-5 | 0.2738 | 0.0848 | 3.2x |
| role-conditional top-5 | 0.3005 | 0.2531 (role freq) | 1.19x |
| role given champion, top-1 | 0.9568 | — | — |
| win AUC | 0.5508 | — | — |
| win Brier | 0.2519 | 0.2491 (constant) | skill -0.011 |

Kept evaluation: `model/pretraining/probe_value.json` — three readouts of the
same frozen representation on held-out data: mean-pool 0.5478, PMA 0.5521,
trained head **0.5597** (base rate 0.4706). Verdict recorded there: the
representation carries outcome signal, so the head is not the bottleneck.

## 2. Finetune — pro stage-1 (masked completion + outcome heads)

**Architecture.** Same encoder, initialised from the checkpoint above. Heads:
masked champion + role (as pretraining) and `PoolOutcomeHeads` — LayerNorm over
the 4xD pooled vector -> win logits (blue/red) and econ regression. The econ
head is **off by default** (`--econ-weight 0`): it scored R^2 -0.09 to -0.15 and
nothing downstream reads it.

**Training.** Pro complete drafts, masked champion modelling with a rehearsal
loss on a soloQ fraction to stop the encoder drifting. Two stages
(encoder frozen, then low-LR encoder), patch-honest val.

**Result** (test, 3,189 complete drafts, role-visible masking):

| metric | model | baseline (train freq table) | ratio |
|---|---|---|---|
| top-1 | 0.2372 | 0.0978 | 2.43x |
| top-5 | 0.5632 | 0.3733 | 1.51x |
| top-10 | 0.7511 | 0.6496 | 1.16x |
| mean rank | 7.45 | 11.56 | — |

Focus patch 16.20 (n=1,960): top-5 0.5934 vs 0.3577 table (1.66x), and 1.14x
against a **leave-one-out same-patch** reference (0.5219), so the gain is not
just "this patch is easy". Win head: AUC 0.5172, Brier 0.2904 vs 0.2483
constant — ranks slightly better than chance, calibrates worse than a constant.

Kept evaluations: `test_report.json` (both heads) and `eval_protocol.json`
(stratified: patch, tier, league, slot, role-visibility, Wilson intervals).

## 3. Policy — next-move recommendation

**Architecture.** Encoder + ban-context pool + state embedding -> policy head
over 173 champions, scored for whichever side is on the clock. Move kind, team
and role are inputs, so one head serves bans and picks. Value and econ heads
exist but are **off by default** (`--value-weight 0`, `--n-econ 0`): the value
head measured 0.4957 AUC, i.e. chance.

**Training.** Every one of the 20 prefixes per draft (`prefix_mode=random`),
legal moves masked with -inf, side-swap augmentation, no patch conditioning
(measured harmful, see below). 40,609 train drafts.

**Result** (test, 63,780 states from 3,189 games):

| slice | top-1 | top-5 | top-10 | baseline top-5 | ratio |
|---|---|---|---|---|---|
| all | 0.0861 | 0.2556 | 0.4068 | 0.1243 | 2.06x |
| picks | 0.1206 | 0.3532 | — | 0.1400 | 2.52x |
| bans | 0.0525 | 0.1579 | — | 0.1087 | 1.45x |
| patch 16.20 | 0.0793 | 0.2390 | — | 0.1209 | 1.98x |

Per move phase (`diagnose_policy.py`, same checkpoint): **ban1 0.98x, pick1
1.98x, ban2 2.38x, pick2 3.58x.** The pooled 1.45x ban number hides a 0.98/2.38
split. On patch 16.20 the model is 0.82x a leave-one-out same-patch popularity
table, i.e. the newest patch is the weakest slice.

Kept evaluations: `test_report.json`, `eval_protocol.json` (stratified, with the
16.20 focus block and both references).

## 4. Prefix value — the `V` that search consumes

**Architecture.** The pretrained value path (`pma` + `win_head`) fine-tuned on
pro **prefix** states, trunk frozen, plus a PMA pool over the 10 ban slots and a
**zero-initialised** residual block so `V` can see bans at all (the encoder
reads `champ_ids`/`role_ids`/`pad_mask` only). Zero init means the logit equals
the pretrained head bit-for-bit at step 0, which keeps the pretrained weights
usable and makes the ban ablation clean.

**Training.** All 20 prefixes per draft, 12 epochs max, warmup + cosine, early
stop on patch-honest val AUC, Platt calibration stored in the checkpoint.

**Result:**

| slice | n | AUC | Brier | const Brier | skill |
|---|---|---|---|---|---|
| complete drafts (test) | 3,189 | **0.5426** | 0.2514 | 0.2483 | -0.012 |
| all prefixes (test) | 63,780 | 0.5124 | 0.2482 | 0.2483 | +0.000 |
| complete drafts (val_honest) | 2,458 | 0.5286 | 0.2518 | 0.2485 | -0.013 |

Ablations: the ban residual **hurts** (0.5160 -> 0.5124 frozen, 0.5216 ->
0.5207 with cross-team unfrozen), and unfreezing `cross_team` overfits (train
loss 1.03 -> 0.79 while val AUC 0.515 -> 0.501). The cross-unfrozen variant was
deleted; it scored 0.5415 on complete drafts, no better.

Three value functions on **identical** complete drafts: logistic additive
role x champ 0.5456, this PMA model 0.5426, finetune `PoolOutcomeHeads` 0.5172.

## 5. Ban analyses (no weights, results kept)

| file | what it measures | headline |
|---|---|---|
| `ban_search_role.json` | depth-2 negamax over an additive `W[role, champ]` | phase-2 decision surface 0.0093 win-prob wide; humans sit at its optimum (edge +0.00004, z 4.7); human is argmax 1.5% vs 0.67% chance; frequency-table bans are *worse* than average (z -0.5) |
| `ban_intent.json` | does a ban hit what the opponent wanted | phase 2: banned champ in opponent's predicted top-5 29.6% vs 8.3% frequency (3.6x lift). Phase 1: no lift (10.4% vs 16.0%) |
| `ban_propose_bans2.json` | policy-rollout proposal vs the human ban | -0.00007 [-0.00055, +0.00039] under `V`; -0.00114 [-0.00412, +0.00197] under an independent verifier; the proposal changes the draft in 87.9% of decisions, so this is a real null |
| `ban_propose_bans1.json` | same, phase 1 | +0.00066 [-0.00011, +0.00143] |

Two structural results, both pinned by tests in `test_ban_search.py`:
a global additive `V` gives every ban the same value (spread 0.00000, measured
on 19,131 decisions) while predicting wins *better*; and phase 1 is flat under
any symmetric additive `V` because both teams need the same roles from the same
pool.

**Important caveat on the phase-2 null.** It is a property of the *model class*,
not of the game. An additive `W[role, champ]` cannot express "Karma is worth
more to the team that already has Ezreal than to the team facing Kaisa". Real
drafts have exactly that champion-specific complementarity — blue on Ezreal
wants Karma, red on Kaisa wants Nautilus — and the rollout does produce such
asymmetric drafts (87.9% of decisions change, mean 2.47 champion slots). What
the additive `W` cannot do is *score* the difference, which is why the proposal
reads as a null and why `--propose value` tops out at +0.0173: the effect is
below this value function's resolution, not absent from the game. The missing
ingredient is champion × champion interaction conditioned on what is already
locked in — see P3.

## 6. Discarded ablations (weights deleted, conclusions kept)

Policy top-5 on test, same protocol throughout:

| run | test top-5 | conclusion |
|---|---|---|
| policy-1 | 0.2401 | baseline for the earlier D256 encoder |
| policy-1-no-value | 0.2380 | value loss weight buys nothing |
| policy-1-no-value-or-overfit | 0.2503 | less head capacity is better |
| policy-1-...-unfreeze | 0.2551 | unfreezing the encoder helps |
| policy-2 (patch conditioning) | 0.2029 | **patch conditioning is harmful**, dropped |
| policy-3-nopatch | 0.2258 | confirms the above once patch is removed |
| ablB random-noswap | 0.2348 | side-swap augmentation ~neutral |
| ablD random-swap | 0.2339 | same |
| ablE stage0 encoder | 0.2547 | stage-1 encoder worth +0.1pp |
| **ablF stage1 (kept)** | **0.2558** | current policy |

Pretraining: four earlier runs (D256 `run1-full-16-20`, `oct19`, plus the
`gpu-timing` / `gpu-value-check` / `win-head` / `lr3e4` probes) were deleted;
the value-probe numbers they produced are preserved in `probe_value.json`.
Finetune: two pre-stage-1 runs whose three checkpoint files were identical, and
the cross-unfrozen value variant, were deleted.

## 7. Next: P3 and the frontend

The repo is now shaped for these four P3 items (`plan-model-v2.md` §P3):

1. **Kit-attribute embeddings.** New `data/champ_attrs.json` (`[173, K]`,
   LLM-tagged: engage, peel, poke, dive, scaling, lane priority, damage type,
   hard CC, mobility, range, objective speed). Hook: `DraftSlotEmbedding` in
   `model/soloq_model.py` — concat or initialise. Champion ids already align
   with `data/solo queue/process matchdata/champ_names.json`.
2. **Soft-label distillation.** Hook: `policy_loss` in
   `model/policy/train.py`. Target: ListNet / KL against LLM rankings on a few
   thousand states; `build_prefix` already produces the states.
3. **Retrieval + explanation.** `frontend/server.py` plus the bitmask
   nearest-neighbour store (`soloq data analysis/store/`): retrieve k similar
   historical drafts, let an LLM explain the recommendation. No training.
4. **Patch-day cold start.** `data/patch notes/` corpus -> champion priority
   prior, bridging the patch reset that `by_patch` currently exposes.

The frontend now serves both model families (done 2026-10-30):

- **Masked path** (`/api/suggest`): the `Predictor` class, role-conditioned
  MASK queries for every empty role slot on both teams. Accepts pretraining
  and finetune checkpoints only — a policy checkpoint is rejected with
  "not a next-move policy" because it has no champ head.
- **Next-move path** (`/api/next-move`): `model/policy/infer.py` owns it
  (`PolicyPredictor`), with no dataset or training imports, so the server
  never pulls in `datasets.py`. It resolves the canonical 20-move order
  (including a red first pick), builds the board from the UI's picks/bans,
  and returns the next move's top-k plus, on ban moves, denial-ranked ban
  candidates: roll the policy forward to the opponent's next pick, shortlist
  their top-k, ban each for real, score the finished drafts with the prefix
  value model, rank by denial. `--value` (default: the newest
  `ckpt/value_*/best.pt`) supplies the verifier; without it the panel ranks
  by denial alone and says so.
- Model family is dispatched from the state-dict keys, not the filename:
  `policy_head.*` → policy, `heads.*` → finetune, otherwise pretraining.
- `Predictor` loads through `load_encoder_state`, so a pretraining checkpoint
  that carries the value path (`pma.*`, `win_head.*`) loads cleanly while a
  real architecture mismatch still raises.
- The `--ckpt` dropdown lists the `best_*` files under `ckpt/run_*/` only
  (`latest.pt` is filtered out — it is a resume point, not a model), across
  all four kept checkpoints.
- `--host 0.0.0.0` is required when serving through the Docker port mapping.

Known gap: the offline proposer is told the role of each future pick from the
pro row; a UI board has no role labels, so the rollout infers each simulated
pick's role from the champion. The UI's simulated win% is therefore a harder
task than the offline null and is shown only as a tiebreaker behind denial.

## 8. Reproduce

```bash
docker compose run --rm dev python model/pretraining/train.py        # pretrain
docker compose run --rm dev python model/finetune/train.py           # stage-1
docker compose run --rm dev python model/finetune/value.py           # prefix V
docker compose run --rm dev python model/policy/train.py             # policy
docker compose run --rm dev python model/finetune/evaluate.py \
    --ckpt model/finetune/ckpt/run_20261017_185016_stage1-full/best_mask.pt
docker compose run --rm dev python model/policy/evaluate.py \
    --ckpt model/policy/ckpt/run_20261025_000606_ablF-stage1/best_policy.pt
```

Tests: run per file (12 suites, 168 tests + 1 skip). `model/policy/model.py`
claims the `model` module name and shadows `model/finetune/model.py` in a
combined run, so a single `pytest model/` will fail for reasons unrelated to
the code.
