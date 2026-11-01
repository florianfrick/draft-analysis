# Project status

Last updated: 2026-10-31. Scope: what is trained, what is measured, what is kept.

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

1. **Kit-attribute embeddings — WIRED (2026-10-31).** `data/champ_attrs.json`
   is now a real input: 173 champions x 18 axes (engage, peel, poke, dive,
   scaling, lane priority, hard CC, mobility, range, objective speed,
   tankiness, magic share, sustain, damage output, side laning, pick
   potential, kiting, healing), ordinal 0-3, tagged in
   `data/build_champ_attrs.py`. `DraftSlotEmbedding` adds a zero-initialised
   `Linear(18, D)` on the tag row, so an attribute-enabled encoder computes
   *bit-identical* outputs to the old one until the adapter learns, and
   `--no-champ-attrs` is the ablation. `attr_dim` lives in the checkpoint
   config and every stage rebuilds from it (`attr_matrix_for_config`), so no
   loader can forget the matrix and quietly serve a zero adapter. Matrix is a
   non-persistent buffer (checkpoints stay the same size); a re-tagged table
   with a different axis count refuses to load. Tests:
   `model/tests/test_champ_attrs_embed.py` (14) + `data/test_champ_attrs.py`
   (7). External checks on held-out pro data: `lane_priority` vs adjusted
   gold@15 rho +0.34 (3.9 sigma), `damage_output` vs damage/min +0.63
   (7.3 sigma), `scaling` vs long-short win rate +0.35 (4.0 sigma), and
   `magic_share` vs damage/min -0.02 (i.e. damage *type* is orthogonal to
   damage *volume*, which is what the axis split was for). **Measured
   2026-10-31:** no gain distinguishable from noise, and the cold-start
   claim does not hold as wired - see
   [P3.1 measured](#p31-measured-the-adapter-trains-the-cold-start-does-not).
2. **Soft-label distillation.** Hook: `policy_loss` in
   `model/policy/train.py`. Target: ListNet / KL against LLM rankings on a few
   thousand states; `build_prefix` already produces the states.
3. **Retrieval + explanation — BUILT (2026-10-31).** `/api/explain` retrieves
   the k most similar pro drafts (bitmask coarse pass + exact Jaccard on the
   board *at the query move*), pulls the patch bullets for the top candidates
   (when the local patch corpus is checked out),
   and asks an LLM for a grounded explanation. Every citation is checked
   mechanically before it is returned. No training involved; see
   [P3.3 built](#p33-built-retrieval-plus-a-validator-not-a-prompt).
4. **Patch-day cold start — DEPRIORITISED (2026-10-31).** The plan was
   `data/patch notes/` -> champion priority prior, bridging the patch reset
   that `by_patch` exposes. Two reasons it is not being built: the P3.1 result
   says the input-side adapter cannot produce a cold-start prior at all, so a
   patch-derived prior would have to enter through the output head (the same
   `W_a` term) before this idea can be tested; and the corpus is **local only**
   by owner decision — it was removed from the repo in `8a30a07` and is now
   git-ignored as a whole. The explainer uses it opportunistically when it is
   present on disk and says nothing about patch changes when it is not.

### P3.1 measured: the adapter trains, the cold start does not

Two experiments, both fixed-budget pairs that differ only in `--no-champ-attrs`.
Setup: 100k drafts (95k train / 5k val), D=256, 2 intra + 1 cross layer, batch
512, 2500 steps (14 epochs), value head **off** so the masked objective is the
only one moving, seeds 0 and 1. Every checkpoint scored on one val view with a
fixed mask seed (`model/pretraining/compare_attrs.py`), so the pair can be
compared with a paired test rather than two independent percentages.

**A. Plain ablation, all 173 champions.** `champ_top5`:

| arm | seed 0 | seed 1 |
|---|---|---|
| attributes off | 0.2600 | 0.2628 |
| attributes on | 0.2618 | 0.2662 |
| delta | +0.18 pp | +0.34 pp |

Paired McNemar on the same 8,464 masked slots: p = 0.49 and p = 0.34. The
seed-to-seed spread within an arm (0.28 and 0.44 pp) is larger than the
between-arm difference, so **the point estimate (+0.26 pp) is not
distinguishable from zero at this budget.** The adapter is not dead - its
weight norm reaches 3.3-3.5, so it is doing real work; it just does not buy
accuracy on champions that already have data. The only group with a
consistent sign is the *common* champions (+0.66 / +0.57 pp), which is the
opposite of where the mechanism is supposed to act.

**B. Cold start, 20 champions held out of training entirely** (`--holdout-champs
20`; they never appear in a train draft, so neither their embedding row nor
their head row is ever a target, but they stay in val):

| | off | on |
|---|---|---|
| held-out `champ_top5` (n=351) | 0.00% | 0.00% |
| mean rank of the held-out champion (of 173) | 155.2 | 154.8 |

Random would be rank 86. The model puts a champion it has never seen **below
random**, and the adapter moves that by 0.4 ranks. Two compounding reasons,
both structural rather than a training artefact:

1. **The output head is untied, and the softmax denominator trains it.** Every
   logit in the denominator receives gradient, so the row for a champion that
   is never a target is pushed *down*, not left at init. Rank 155 is the
   result.
2. **The adapter conditions the input, not the output.** The slot embedding of
   a masked slot is the MASK vector; the predicted champion's logit is
   `W_c · h`, and `W_c` is the one thing that never sees `attrs_c`. The
   attribute row of the champion being *predicted* never enters the
   computation.

So the table helps predict *other* champions from an attribute-rich context,
but it cannot give a new champion a row on patch day - which was the headline
justification. The fix is to let the output read the attributes as well:
`logits_c += (W_a @ attrs_c) · h` with `W_a` zero-initialised (additive, keeps
the untied head, degrades exactly to today's model at init), or tie the head to
the embedding table. The first is the smaller change and the one to try.

Verdict on the adapter: keep it (zero cost at init, no measurable harm, and it
is the only path to the cold-start prior), but do not expect it to move
`champ_top5` on an existing champion pool.

### P3.3 built: retrieval plus a validator, not a prompt

`frontend/llm.py` + `/api/explain`. The model returns a probability and cannot
say why; the explanation is generated at request time and never touches the
ranking. Three pieces carry the weight:

**Retrieval is over the board at the query move.** Each pro draft is one
173-bit mask (champions seen so far). A coarse pass ranks the 46,813 rows by
`int.bit_count()` overlap (~20 ms), then exact Jaccard on the *prefix* decides
the top k — ranking on the final draft would surface games that resemble the
finished board rather than the decision being asked about. Games are filtered
to the same first-pick side; if fewer than k survive the filter relaxes and
`relaxed_side` says so. Patch proximity ranks before date: on a real 16.20
board the date tie-break put an 8-patch-old game first. No fixture caught that
— the demo against the real corpus did.

**Grounding is mechanical.** Every `evidence` item must resolve: a game id must
be in the retrieved set, a patch line must be a verbatim substring of the notes
actually handed to the model. Failures are dropped and reported, the model gets
one retry with the problems fed back, and a second failure returns `ok: false`
instead of prose nobody checked. The alternative — instructing the model to
cite honestly and trusting it — is what produces confident citations of games
that do not exist.

**The reasoning is shown, and labelled for what it is.** The answer schema has
a `reasoning` field (2-5 steps, in order), rendered above the pick. It is the
model's own account written with the answer — not a transcript of hidden
deliberation, and not something the validator can check — so it is kept out of
`evidence`, capped at 6 x 240 chars, and the panel says plainly that only the
citations below it are verified. What *is* checked: every cited game and patch
line, plus anything the retry failed to fix, which is listed in amber rather
than dropped silently.

**No key is not an error.** Without `LLM_API_KEY` the endpoint still returns
the retrieved games and patch bullets with `explanation: null`, so the evidence
path can be inspected before anyone configures a provider. Cache is keyed on
prompt version + model + board + candidates + retrieved ids, so a re-ask after
a corpus change cannot serve a stale citation.

Known gap: prose quality is **unvalidated**. No `LLM_API_KEY` was configured,
so the endpoint has only been exercised through the no-key path and a stub
client (which did cite an invented game id on the first attempt and get
corrected by the retry — the validator works, the model was never asked
anything real).

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
- **Explainer** (`/api/explain`): `llm.ProIndex` is built once behind a lock
  on first request (46.8k drafts, 3.7 s) and then answers in ~300 ms. Needs a
  **next-move policy** checkpoint — it explains a single move, so the masked
  path (which fills every empty role slot at once) is rejected with a 400
  naming the model family it found.
- **Explanation panel** (`static/app.js: renderExplain`): "why this move?" in
  the next-move panel, on demand rather than automatic (a call costs seconds
  and money; the recommendation is already on screen). Shows the model's
  numbered reasoning steps, the pick (clickable to apply), why / their answer /
  risk, the citations resolved against the retrieved games, anything rejected
  by the validator, and the comparable-drafts list the answer was built from.
  Hides itself when the board changes — a stale explanation is worse than none.
- `--host 0.0.0.0` is required when serving through the Docker port mapping.

Known gap: the offline proposer is told the role of each future pick from the
pro row; a UI board has no role labels, so the rollout infers each simulated
pick's role from the champion. The UI's simulated win% is therefore a harder
task than the offline null and is shown only as a tiebreaker behind denial.

## 8. Pro-only training and the patch embedding (`exp/pro-only-patch-emb`)

Two questions, both from the same branch: **how far does pro data alone get
you** (no soloQ pretraining, no rehearsal), and **does a patch embedding help
once the patch is a *seen* input**?

The second one is only askable because the split changed. Under
`split_series_by_date` every test patch is unseen, so a patch id is UNK at
test time and the embedding can only memorise - which is exactly what
`policy-2` measured as "patch conditioning is harmful" (0.2029 vs 0.2558,
§6). `pipeline.split_within_patch` puts each patch on both sides: train = the
first 80% of its series, val/test = the last 10% / 10%, series never split.
46,813 drafts -> 37,313 / 4,737 / 4,763, 99 of 106 patches in test, 0 series
straddling. That is the deployment question: you always know today's patch,
and you have its first 80%.

**Patch embedding.** `DraftSlotEmbedding.patch_lookup`: one zero-initialised
vector per patch (row 0 = UNK), added to every slot before the LayerNorm, so
"patch on" and "patch off" are bit-identical at step 0. `n_patches` goes into
the checkpoint config; a checkpoint with a table cannot be loaded into an
encoder without one (and vice versa). Tests:
`model/tests/test_patch_embed.py` (10) + `model/proonly/test_proonly.py` (6).

Both stages: from scratch, D=256, 2 intra + 1 cross, dropout 0.1, uniform
recency weights (the test window is the tail of *every* patch, so down-
weighting old games would just make old patches unlearnable). The next-pick
stage initialises its encoder from the matching masked checkpoint.

**Masked completion** (test, 8,060 masked slots, baseline = per-slot
popularity fit on train, 0.4120):

| arm | seed | top-1 | top-5 | top-10 | rel@5 | role-cond@5 |
|---|---|---|---|---|---|---|
| patch | 0 / 1 | 0.2089 / 0.2133 | **0.5669 / 0.5712** | 0.7484 / 0.7509 | 1.376x / 1.386x | 0.651 / 0.632 |
| no patch | 0 / 1 | 0.1911 / 0.1872 | 0.5303 / 0.5163 | 0.7083 / 0.7061 | 1.287x / 1.253x | 0.581 / 0.558 |

Paired on identical masked slots: **+3.66 pp** top-5 (McNemar p=7.7e-14,
game-bootstrap CI [+2.71, +4.56]) and **+5.50 pp** (p=6.5e-29, [+4.56,
+6.48]). Seed spread *within* an arm is 0.4-1.4 pp, so the gap is 3-4x the
noise - the opposite of the P3.1 attribute result, where the gap was smaller
than the seed spread. Val peaks at **epoch 6** and decays to 0.487 / 0.496 by
epoch 40: 37k drafts against a 2.5M-parameter encoder overfits after ~440
steps, and selection on val is what keeps the reported number.

**More data.** The folder now also holds 2014-2021 OE exports; `CSV_GLOB` is
pinned to 2022+ so they cannot swap the corpus under existing scripts, and
`--corpus all` opts in (81,161 drafts, 2017-2026, 64,737 / 8,198 / 8,226, own
cache). Seed 0, same protocol: patch **1.493x** vs no patch **1.346x**,
paired **+5.51 pp** (p=1.1e-44). So the model is data-limited - 1.7x the
drafts buys +0.12 rel - and the patch effect does not shrink with more data.

**Next-pick** (test, 95,260 states; baseline = per-move popularity 0.1543;
same-patch leave-one-out reference 0.1576):

| slice | patch s0 / s1 | no patch s0 / s1 | rel@5 (patch / none) |
|---|---|---|---|
| all top-5 | **0.4839 / 0.4868** | 0.3894 / 0.3873 | 3.14x / 2.52x |
| ban1 | **0.4246 / 0.4339** | 0.1860 / 0.1860 | **2.43x / 1.07x** |
| pick1 | 0.5564 / 0.5552 | 0.4705 / 0.4647 | 3.64x / 3.08x |
| ban2 | 0.3656 / 0.3700 | 0.3594 / 0.3684 | 2.95x / 2.90x |
| pick2 | 0.5828 / 0.5801 | 0.6029 / 0.5920 | 3.72x / 3.85x |

Paired: **+9.45 pp** top-5 (p<1e-300, CI [+9.16, +9.74]) and **+9.95 pp**
(CI [+9.66, +10.24]). The per-phase split is the mechanism test, not a
headline: **ban1 goes 1.07x -> 2.43x**, i.e. with an empty board the model
stops sitting at the frequency table and starts predicting the patch's ban
priority - the same slice the soloQ-based policy lost at 0.98x, for a reason
that turns out to have been the protocol. And **pick2 is 1.2-2.0 pp better
*without* it** (both seeds): once the board is full, composition dominates,
and 107 x 256 patch parameters are a small net negative where they carry no
signal.

**Against the existing pipeline, same rows** (positive = pro-only better):

| rows | masked: `best_mask.pt` -> pro-only | next-pick: `ablF` -> pro-only |
|---|---|---|
| all | 0.3710 -> **0.5669** (+19.6 pp) | 0.2671 -> **0.4839** (+21.7 pp) |
| 2022 | 0.2721 -> 0.5839 (+31.2) | 0.1309 -> 0.4821 (+35.1) |
| 2024 | 0.3265 -> 0.5800 (+25.4) | 0.2350 -> 0.4650 (+23.0) |
| 2026 | 0.4939 -> 0.5431 (+4.9) | 0.4091 -> 0.4951 (+8.6) |

Read the rows: the kept stages use a 75-day recency half-life and their own
protocol only asks them about recent patches, so the pooled gap is mostly
"the old model was told to ignore 2022". On 2026 - the deployment target -
pro-only is +4.9 / +8.6 pp, and even that is confounded by the weighting
difference. The only clean cross-pipeline statement is the paired one above,
where nothing differs but the flag.

**Verdict.** Pro-only is viable: masked completion 1.38x and next-pick 3.14x
their same-patch baselines from 37k drafts, with no soloQ anywhere. The patch
embedding is a real gain under this protocol (+3.7 to +5.5 pp masked, +9.5 to
+10.0 pp next-pick), and its effect is concentrated in ban1 exactly as the
mechanism predicts. It does **not** contradict §6's "patch conditioning is
harmful": that result was measured with unseen test patches, where the input
was UNK. The two protocols answer different questions, and the product
question is this one.

**Does the masked stage earn its place?** Seed 0, patch arm: the policy
trained from scratch scores 0.4746 top-5 against 0.4839 for the
masked-initialised one, i.e. **+0.93 pp** for the masked stage (p=1.1e-19).
Real but an order of magnitude smaller than the patch embedding's +9.5 pp, and
ban1 is the same either way (0.4275 vs 0.4246) - 32 epochs of next-move CE on
37k games learns the patch prior by itself.

Not settled: one architecture size (the epoch curve says the fit is
data-limited); `rel@5` is not comparable across protocols (this window's
baseline is 0.412 vs 0.373 patch-honest, so pro-only 1.38x and finetune 1.51x
cannot be read as "finetune is better"); the wider corpus and the scratch
control are seed 0 only. Kept: 9 checkpoints (482MB) under
`model/proonly/ckpt/`, both arms and both seeds, because the paired
comparisons need the losing arm too; the numbers that survive them are in
`model/proonly/reports/`.

## 9. Reproduce

```bash
docker compose run --rm dev python model/pretraining/train.py        # pretrain
docker compose run --rm dev python model/finetune/train.py           # stage-1
docker compose run --rm dev python model/finetune/value.py           # prefix V
docker compose run --rm dev python model/policy/train.py             # policy
docker compose run --rm dev python model/finetune/evaluate.py \
    --ckpt model/finetune/ckpt/run_20261017_185016_stage1-full/best_mask.pt
docker compose run --rm dev python model/policy/evaluate.py \
    --ckpt model/policy/ckpt/run_20261025_000606_ablF-stage1/best_policy.pt
# pro-only branch (see model/proonly/README.md)
docker compose run --rm dev python model/proonly/train_masked.py --run-name m-patch-s0
docker compose run --rm dev python model/proonly/train_policy.py --init-ckpt <masked best.pt>
docker compose run --rm dev python model/proonly/compare.py --task masked --a A --b B
```

Tests: run per file (18 suites, 272 tests + 1 skip). `model/policy/model.py`
claims the `model` module name and shadows `model/finetune/model.py` in a
combined run, so a single `pytest model/` will fail for reasons unrelated to
the code. `soloq data analysis/explorer/test_explorer.py` is excluded from that
count: it opens an HTTP connection to a live explorer server at import time and
errors during collection when none is running.
