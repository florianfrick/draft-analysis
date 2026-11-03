# Project record — state, findings and plan

Last updated 2026-11-03. This is the single narrative document: what the models
are, what they measure, what failed and why, and what to do next. It replaces
`assumptions.md`, `plan-model-v2.md` and `model/proonly/FINDINGS.md`, whose
content is distilled here.

Operational docs stay separate and this file does not repeat them: `README.md`
to get running, `DOCKER.md` for the container, and each folder's `README.md` for
its own commands.

**How to read it.** §7 is the current measurement. §4 is the older protocol's
numbers, kept because they are the evidence behind the ablations in §6 — do not
quote §4 as today's scores. §5 and §6 are verdicts whose runs can no longer be
reproduced; they are here so nobody re-runs them.

## Contents

1. Where it stands
2. The model
3. The data and the split
4. Results, previous protocol
5. Ban analyses
6. Dead ends (measured, do not re-run)
7. The test split — current numbers
8. The frontend
9. Plan
10. Reproduce

---

## 1. Where it stands

Five checkpoints, one per model class, all under `model/checkpoints/`. The
directory name states the inference the checkpoint serves and the stage that
produced it, and `frontend/server._ckpt_roles` reads that prefix to decide which
of the two UI slots may load it.

| class | checkpoint | inference |
|---|---|---|
| pretraining (soloQ) | `masking-soloqueue/best.pt` | masked completion; the init for `masking-finetune` |
| finetune (soloQ → pro) | `masking-finetune/best_mask.pt` | masked completion, role board |
| pro-only, from scratch | `masking-proonly/best.pt` | masked completion, role board |
| policy (soloQ encoder) | `nextpick-policy/best_policy.pt` | next-move ranking |
| policy (pro-only) | `nextpick-proonly/best.pt` | next-move ranking |

Weights are git-ignored. 18 older checkpoints (1.9 GB) were deleted on
2026-11-03; their numbers live in `model/proonly/reports/` and §4/§6 here, but
their paired comparisons can no longer be re-run.

**The corpus is pinned** at `pipeline.CORPUS_MAX_DATE = "2026-11-01 23:59:59"`
— 46,875 drafts, patch 16.20 = 258 rows. The 2026 OE export keeps growing, so
without the pin a rebuild folds new games in and moves every recorded number.
This is not bookkeeping: during the 2026-11-01 merge an ordinary `pytest` run
called `build_rows()`, saw the 2026 CSV had changed, rescanned and overwrote
`pro_drafts.pt` — training the corpus on its own holdout. Rebuilding today
returns the same 46,875 drafts; raising the pin means re-measuring everything,
which is exactly what happened on 2026-11-03.

**The test split is pinned by series id**, not by a fraction: three Worlds 2026
final-day series, 7 games, all patch 16.20. Split: **42,100 train / 4,768 val /
7 test**. See §7.

---

## 2. The model

**Lineage.** Set Transformer (Lee et al. 2018) without ISAB, plus BERT-style
masked modelling. Sketch in the root README: embedding → Siamese intra-team
attention → symmetric inter-team cross-attention → slot-level champion and role
heads, plus pooling to auxiliary heads. Frozen as an I/O contract and enforced
by tests (`model/pretraining/soloq_model.py`).

**Why sets, not sequences.** A champion cannot repeat and only the current state
matters for comp value, so permutation invariance is wanted and there are no
positional embeddings; slot order is a loader convention. Within-team equivariance
and Siamese sharing are asserted by tests (one intra encoder over `[2B, 5, D]`,
one cross block both directions). No ISAB — 5+5 sets make full attention cheap.
The soloQ family is D=512, 6 intra + 3 cross, 8 heads, dropout 0; the pro-only
family is D=256, 2 intra + 1 cross, because 37k drafts do not support more (§6).

**Embeddings.** `DraftSlotEmbedding` = `LayerNorm(Embed(champ) + Embed(role))`.
Champion table has REAL + MASK + PAD rows (C = 173 on 16.20); role table has 5
roles + a FLEX input-only token. Output heads are `Linear(D, C)` and
`Linear(D, 5)`; MASK/PAD/FLEX are input-side only and never predicted. There is
no team embedding — team comes from the slot index (0–4 blue, 5–9 red); a team
vector was redundant and a risk to Siamese symmetry. The sum forces a shared
space and LayerNorm keeps scales comparable. Pooled `z` = concat(blue, red,
diff, product) with PAD-masked means, so asymmetry and interaction are explicit.

**Masked modelling.** Corrupt ~15% of slots (≥ 1 per draft), predict champion
(CE over 173) plus role (CE, weight 0.3). BERT 90/10 role-on-masked-slot split
(the 10% role-visible teaches P(champ | role), the "best X for MID" query) plus
~10% role-only masks (champion visible, role hidden, teaches P(role | champ) —
the flex query). Fresh train masks per epoch, fixed val masks. AdamW, warmup +
cosine, AMP, early stop on val `champ_top5`.

**Legality is a mask, not a learned preference.** A champion cannot be picked
twice; bans and fearless-unavailable are hard constraints applied as `-inf`
logits. This is why the next-move models break fearless zero times while the
masked models, which never receive the mask, break it constantly (§7).

**Patch embedding (pro-only family).** One zero-initialised D-vector per patch
(row 0 = UNK) added to every slot before the LayerNorm, so "patch on" and "patch
off" are bit-identical at step 0 and the model has to learn to use it. `n_patches`
lives in the checkpoint config; a table cannot be loaded into an encoder without
one. The kit-attribute adapter (`data/champ_attrs.json`, 18 axes, zero-init
`Linear(18, D)`) is the same trick for champion identity — see §6 for why it did
not buy what it was supposed to.

**Known gaps.** The trunk never sees bans, order or fearless — the policy and
finetune stages add them. Multi-MASK frontend inference differs from single-mask
training (measured by a K-curve). Hard CE gives no partial credit. The win head
is side-neutral ("blue wins"; red queries mirror slots). Econ is weak
supervision. No player or team identity: "Faker's Azir" is "Azir".

---

## 3. The data and the split

**SoloQ** (high-elo, single patch 16.20). Collected via the Riot API (puuids →
match ids → match details; RANKED_SOLO_5x5, Challenger/GM/Master, all platforms,
per-region limiters), ~3 kB per match, no timelines so no gold curves. Filtered
to queues 420/440, duration ≥ 300 s, Data Dragon 16.20.1, 173 champions. One
completed draft = 10 picks, **no ordering, no bans**; bans are excluded from
pretraining and PAD is reserved for partial states. 401,512 games.

It teaches champion strength, role affinity and generic synergy from volume —
solo-carry and lane-fit priors. It over-represents comfort picks and blinds and
under-represents pro coordination (swaps, dive comps, fearless depth). Its
embeddings do **not** transfer across patches: a new patch changes C and shifts
semantics, so anything patch-conditioned must be retrained. The outcome is used
only by the separate value head.

**Pro play** (Oracle's Elixir, 2022-2026). Raw CSVs → `pro_drafts.pt`, same
173-champion space, 46,875 drafts, skipping only for documented reasons (bad
rows, incomplete bans/picks, duplicate champion, pick mismatch). 12-row OE
blocks become a global 20-move sequence (6 bans1 + 6 picks1 + 4 bans2 + 4 picks2)
on the 2026 First-Selection order, pre-2026 blue-first; a move is
`(kind, team, champ, role)` with bans carrying IGNORE role. Era handling is
data-driven per series (2025+ mostly fearless, 2026 fully), and first-pick
decoupling is 2026 only.

Pro teaches what soloQ cannot: order/phase/side effects (ban priority, B1 blind
vs R5 counter, first-pick asymmetry), fearless depth — the main 2025+ shift —
and coordinated comp value. Limits: 46k games is small, no player identity, econ
labels are weak. Synergy/counter structure is assumed to drift slower than
popularity.

**The split is load-bearing, not bookkeeping.** `split_series_by_date` (the old
default) puts the whole patch on one side, so at test time the patch id is `UNK`
and any patch-conditioned input can only memorise — which is why the earlier
`policy-2` run measured patch conditioning as *harmful* (0.2029 vs 0.2558 top-5).
`split_within_patch` asks the deployment question instead: **you always know
today's patch and you have its first 80%.** Series are grouped first (a series
belongs to the patch of its first game, so it can never straddle), sorted by
`(date, gameid)`, and the tail windows are filled backwards so test is newest.

The test split is then **pinned**: `pipeline.TEST_SERIES` names three
`series_id` strings, and `split_within_patch(pinned_test=...)` holds them out of
train and val *by id*, so the set cannot move when the corpus is rebuilt. While
the pin is non-empty `test_frac` is forced to 0; a pinned id missing from the
corpus raises rather than quietly shrinking the test set. `strict_pin=False`
exists only for `--limit-drafts` smoke runs, which subsample.

---

## 4. Results, previous protocol

Measured on 46,813 drafts and the date split. Superseded by §7 for scores; kept
because §6's ablations are stated against these numbers.

**Pretraining** (val, 19,217 complete drafts, epoch 33 / step 11,781):

| metric | model | baseline | ratio |
|---|---|---|---|
| champ top-1 | 0.0849 | 0.0182 (global freq) | 4.7x |
| champ top-5 | 0.2738 | 0.0848 | 3.2x |
| role-conditional top-5 | 0.3005 | 0.2531 (role freq) | 1.19x |
| role given champion, top-1 | 0.9568 | — | — |
| win AUC | 0.5508 | — | — |

**Finetune** (test, 3,189 complete drafts, role-visible masking): top-1 0.2372,
top-5 0.5632, top-10 0.7511, mean rank 7.45 — against 0.0978 / 0.3733 / 0.6496 /
11.56 for a train frequency table. Focus patch 16.20 (n=1,960): top-5 0.5934 vs
0.3577, and 1.14x a leave-one-out same-patch reference (0.5219), so the gain is
not "this patch is easy". The econ head is **off by default**: R² −0.09 to −0.15
and nothing downstream reads it.

**Policy** (test, 63,780 states from 3,189 games): top-1 0.0861, top-5 0.2556,
top-10 0.4068, against 0.1243 baseline top-5 (2.06x). Per phase: **ban1 0.98x,
pick1 1.98x, ban2 2.38x, pick2 3.58x** — the pooled 1.45x ban number hides a
0.98/2.38 split, and ban1 is where the model is no better than a lookup table.
Value and econ heads exist but are off (`--value-weight 0`, `--n-econ 0`).

**Value** (the `V` that search would consume): fine-tuned pretrained value path
with a PMA pool over ban slots and a zero-init residual so `V` can see bans at
all. Complete drafts 0.5426 AUC, all prefixes 0.5124 (skill +0.000). The ban
residual *hurts* (0.5160 → 0.5124) and unfreezing `cross_team` overfits. Three
value functions on identical complete drafts: logistic additive role × champ
0.5456, this PMA model 0.5426, finetune `PoolOutcomeHeads` 0.5172. **No value
function in the repo beats a 173-parameter logistic regression.**

**Pro-only + patch embedding.** Two questions from one branch: how far does pro
data alone get you (no soloQ, no rehearsal), and does a patch embedding help once
the patch is a *seen* input?

| | masked top-5 | next-pick top-5 | rel@5 (next-pick) |
|---|---|---|---|
| patch, seed 0 / 1 | **0.5669 / 0.5712** | **0.4839 / 0.4868** | 3.14x / 2.52x |
| no patch, seed 0 / 1 | 0.5303 / 0.5163 | 0.3894 / 0.3873 | 2.52x (no-patch arm) |

Masked baseline is the per-slot popularity table (0.4120); next-pick baseline is
per-move popularity (0.1543), leave-one-out same-patch 0.1576. Paired on
identical rows: **+3.66 pp / +5.50 pp** masked (p = 7.7e-14 / 6.5e-29) and
**+9.45 pp / +9.95 pp** next-pick (p < 1e-300). Seed spread within an arm is
0.4–1.4 pp, so the gap is 3–4x the noise — the opposite of the attribute result
in §6, where the gap was smaller than the seed spread.

The mechanism is **ban1**: with an empty board there is nothing but the patch to
go on, so without it the model sits *at* the frequency table (1.07x) and with it
reaches 2.43x — the same slice the soloQ policy lost at 0.98x, for a reason that
turned out to be the protocol. Conversely **pick2 gets worse** (3.85x → 3.72x,
both seeds): once the board is full, composition dominates and 107 × 256 patch
parameters are a net negative. The patch vector is phase-specific — hence §9.1.

More data (`--corpus all`, 81,161 drafts) gives 1.493x vs 1.346x, +5.51 pp
(p=1.1e-44): 1.7x the drafts buys +0.12 rel and the patch effect does not
shrink, so the model is **data-limited** — which the epoch curve says
independently (val top-5 peaks at epoch 6, decays to 0.487 by 40). The masked
stage earns its place barely: next-pick from scratch scores 0.4746 vs 0.4839,
**+0.93 pp**, with ban1 unchanged either way.

The pooled cross-pipeline gaps (+19.6 / +21.7 pp) are mostly "the old model was
told to ignore 2022" — the kept stages use a 75-day half-life and are only asked
about recent patches. On 2026 it is +4.9 / +8.6 pp, still confounded. The paired
same-flag numbers above are the only clean statement.

---

## 5. Ban analyses

A research thread that ran to a conclusion and then became unreproducible:
every result recorded a checkpoint the 2026-11-03 reorganisation deleted. The
scripts, their five result JSONs and their two tests were removed with it. This
is the whole record; the only thing surviving in code is `infer.py`'s
`ban_candidates`, which reimplements the propose-and-verify rule for the UI.

| what it measured | headline |
|---|---|
| depth-2 negamax over an additive `W[role, champ]` | phase-2 decision surface 0.0093 win-prob wide; humans sit at its optimum (edge +0.00004, z 4.7); human is argmax 1.5% vs 0.67% chance; frequency-table bans are *worse* than average (z −0.5) |
| does a ban hit what the opponent wanted | phase 2: banned champion in the opponent's predicted top-5 29.6% vs 8.3% frequency (3.6x lift). Phase 1: no lift (10.4% vs 16.0%) |
| policy-rollout proposal vs the human ban, phase 2 | −0.00007 [−0.00055, +0.00039] under `V`; −0.00114 [−0.00412, +0.00197] under an independent verifier; the proposal changes the draft in 87.9% of decisions, so this is a real null |
| same, phase 1 | +0.00066 [−0.00011, +0.00143] |

Two structural results: a global additive `V` gives every ban the same value
(spread 0.00000 over 19,131 decisions) while predicting wins *better*; and phase
1 is flat under any symmetric additive `V` because both teams need the same
roles from the same pool.

**The null is a property of the model class, not the game.** An additive
`W[role, champ]` cannot express "Karma is worth more to the team that already has
Ezreal than to the team facing Kaisa". Real drafts have exactly that
complementarity, and the rollout does produce such asymmetric drafts (87.9% of
decisions change, mean 2.47 slots). What the additive `W` cannot do is *score*
the difference, which is why the proposal reads as a null and why `--propose
value` tops out at +0.0173 — below this value function's resolution, not absent
from the game. Champion × champion interaction has to enter `V` before the ban
search can be judged on its merits; until then the honest UI ordering is denial
("they want it"), not a simulated win rate.

Phase 1 stays a non-target for a different reason: pros ban specialists and game
plans there, which is opponent-specific information no symmetric model has.

---

## 6. Dead ends (measured, do not re-run)

**Pairwise win rates.** Logistic on P(blue wins), nested feature sets: soloQ
0.5449 (singles) → 0.5353 (+synergy) → 0.5436 (+counters); pro 0.5507 → 0.5346.
Pairwise structure adds variance, not signal. A split-half test shows the additive
baseline explains R²=0.37 of pair win-rate variance and the residual's sd equals
the binomial noise floor: synergy is ~100% noise, counters leave ~1.6 pp (2 sd) of
real structure — one tier of champion strength (sd 1.28 pp) — and are unestimable
at n≈300. A denoised low-rank attention bias is +0.002 AUC at rank 1 and negative
above rank 4; structurally, attention logits are already a learned low-rank
pairwise matrix, so a fixed bias can only help outside that span. In pro, **team
identity is worth +0.021 AUC**, more than the whole pairwise block. Keep the
tables for the explorer and as an evaluation target; do not feed them to the
encoder.

**RL as the training algorithm.** Draft is a finite, perfect-information, zero-sum
game with known legal moves, and a policy is already learned by imitation —
cheaply and well-conditioned. What is missing is a *value function*, not a
gradient method; that is the AlphaZero decomposition with value as the binding
constraint. "Self-play" here means modelling the opponent's policy, which is only
learnable from data, so it is offline RL over 46k×20 states that would largely
relearn the behaviour policy. Don't start it.

**Every value head.** Policy `value_head` 0.4969 AUC (chance, and anti-correlated
once the draft fills); finetune `PoolOutcomeHeads` 0.5172; the PMA prefix model
0.5426; a 173-parameter logistic 0.5456. The policy head is not "diluted by early
moves" — it emits a constant for moves 0–6 (Brier 0.2483, exactly the base rate)
then varies against the label. All RL/MCTS/win-aware work is blocked here; do not
surface a win probability until a value function beats Brier 0.2482 with monotone
calibration.

**Patch conditioning on the date split.** 0.2029 vs 0.2558 top-5, −5.3 pp. This is
the *same input* worth +9.5 pp under the within-patch split; the difference is
that the date split makes the patch id `UNK` at test. Never re-run it there.

**Kit-attribute adapter as a cold-start fix.** 173 champions × 18 ordinal axes,
zero-init `Linear(18, D)` on the tag row, `attr_dim` in the checkpoint config so no
loader can forget the matrix. On 100k drafts, paired McNemar on 8,464 identical
masked slots: **+0.18 / +0.34 pp, p = 0.49 / 0.34** — under the seed spread
(0.28/0.44 pp), so not distinguishable from zero. The adapter is not dead (weight
norm 3.3–3.5) and its only consistently positive group is *common* champions, the
opposite of where the mechanism should act. Cold start with 20 champions held out
entirely: **0.00% top-5, mean rank 155 of 173** — below random (86), and the
adapter moves it 0.4 ranks. Two structural reasons: the output head is untied and
the softmax denominator trains every row *down*, so a champion that is never a
target ends below init; and the adapter conditions the *input*, while the
predicted champion's own row (`W_c`) never sees `attrs_c`. Fix on the output side
— `logits_c += (W_a @ attrs_c) · h`, `W_a` zero-initialised — not more input
adapter. Keep the table (zero cost at init, no measurable harm, the only path to a
cold-start prior) but do not expect it to move `champ_top5` on an existing pool.

**Scaling the encoder up.** D512/6+3 from scratch is *worse* than D256/2+1 (masked
top-5 0.5139 vs 0.5669), and D512/6+3 *with* soloQ pretraining only ties it
(0.5648, p=0.674). 37k drafts do not support 28M parameters.

**soloQ pretraining as an init for pro-only.** A dead heat: with matched capacity
pretraining is +5.09 pp masked / +1.60 pp next-pick, but the shipped D256 pro-only
model is **−0.21 pp (p=0.674) / +0.02 pp (p=0.865)** against the pretrained D512
one, inside noise on top-1 and top-10 too. It settled two things: the patch
embedding's size is a property of the protocol, not the trunk (+3.75/+10.68 pp at
the pretrained trunk vs +3.66/+9.45 at the scratch one, ban1 1.07x → 2.46x both
times); and capacity was never the constraint, data is. Caveat stated rather than
hidden: at the recipe's trunk LR the wide scratch control trains badly, so
"+5.09 pp for pretraining" is really "pretraining plus a stable optimisation
regime". One earlier finding did not reproduce — "pick2 is worse with the patch
vector" flips sign at the pretrained trunk (3.61x → 3.68x), so it is
trunk-dependent, not a rule.

**The econ head.** R² −0.09 to −0.15; nothing downstream reads it. Off.

---

## 7. The test split — current numbers

Everything above is pooled over thousands of games, which answers "how good on
average" and never "what would it say here". This is the opposite, and it is the
only split nothing trains on.

**The split.** Three series, seven games, patch 16.20, all played 2026-10-31,
the last day of Worlds 2026:

| series | games | gameids |
|---|---|---|
| Bilibili Gaming vs Hanwha Life Esports | 2 | `LOLTMNT02_486564`, `_486571` |
| Karmine Corp vs Movistar KOI | 2 | `LOLTMNT02_485476`, `_486541` |
| LYON vs LØS | 3 | `LOLTMNT02_485482`, `_485488`, `_486555` |

These were a *holdout* until 2026-11-03 — outside the corpus entirely, because
`CORPUS_MAX_DATE` was `2026-10-24 23:19:46`. A series the corpus has never seen
cannot be a test split, only a holdout nothing can be trained against. So the
pin moved past them (46,813 → 46,875) and the split keeps them out of training
by id.

**Next move** — 140 states (20 per game); baseline = patch 16.20's per-move
frequency table with legality applied:

| model | training | top-1 | top-5 | mean rank | ban1 | pick1 | ban2 | pick2 |
|---|---|---|---|---|---|---|---|---|
| `nextpick-policy` | soloQ encoder + policy | 0.214 | 0.479 | 11.2 | 0.36 | 0.67 | 0.14 | 0.71 |
| `nextpick-proonly` | pro-only, from scratch | **0.229** | **0.564** | **9.1** | 0.38 | **0.74** | **0.36** | **0.79** |
| patch table | 16.20 frequency | 0.114 | 0.400 | 12.1 | **0.45** | 0.55 | 0.11 | 0.39 |

**Masked completion** — 70 slots (each pick hidden in turn):

| model | training | top-1 | top-5 | mean rank |
|---|---|---|---|---|
| `masking-soloqueue` | soloQ pretrain only | 0.329 | 0.529 | 8.7 |
| `masking-finetune` | soloQ → pro finetune | **0.357** | **0.743** | **3.10** |
| `masking-proonly` | pro-only, from scratch | **0.357** | 0.714 | 3.39 |

**What this supports.** Three claims survive the sample size; one does not.

1. **Training on pro drafts is what buys accuracy, on both tasks.** The soloQ-only
   encoder is the only model clearly behind its own family: 0.529 masked top-5
   against 0.714–0.743 for the two pro-trained encoders, and the intervals barely
   touch. The same ordering holds on next move, where the pro-only policy beats
   the soloQ-encoder policy on every aggregate.
2. **The patch-frequency table still owns ban1.** 0.452 against 0.381 and 0.357.
   With an empty board the best predictor of the first ban is 258 games of this
   patch's ban priority, and neither model has beaten it — consistent with §4's
   finding that ban1 is where the patch prior does all its work.
3. **The two pro recipes are tied, and this split cannot separate them.**
   `masking-finetune` 0.743 [0.63, 0.83] vs `masking-proonly` 0.714 [0.60, 0.81];
   `nextpick-proonly` 0.564 [0.48, 0.64] vs `nextpick-policy` 0.479 [0.40, 0.56].
   The intervals overlap in both pairs. Ranking them needs the 4,768-row
   validation split, not this.
4. **The masked models still cannot express fearless** — 85 of 120 masked slots
   on the four fearless games put an already-picked champion in their top-5. The
   next-move models get `unavailable_mask` and break it zero times. Scope limit
   of the masked task as trained, not a harness bug.

**Statistics discipline.** Seven games. One game is 14% of the sample and a 95%
Wilson interval on a rate over seven games is ~35 points wide, so every rate in
the report carries its interval. The aggregate tables are a smoke test on unseen
games — they catch a broken model and refuse to rank working ones. The per-state
tables carry the information, because every model is scored on the identical
board.

Full report: `model/test_split/REPORT_test_split_evaluation.{md,html}`; evidence:
`model/test_split/reports/test_split.json`.

---

## 8. The frontend

Two model slots, loaded at once and chosen independently: `masking-*` drives the
role board (`/api/suggest`), `nextpick-*` drives the next-move panel
(`/api/next-move`, `/api/explain`). Model family is dispatched from the state-dict
keys, not the filename. Loading a policy checkpoint into the role slot is
refused at load rather than failing later.

- **Role board** — role-conditioned MASK queries for every empty role slot on
  both teams. The slot the client just filled is pinned to its own argmax before
  the rest are optimised jointly (lexicographic), so the newest pick always wins
  a contested role and older picks move aside; without that, a fresh Vi was
  pushed to its runner-up while Pantheon kept jungle.
- **Next move** — `model/policy/infer.py:PolicyPredictor`, no dataset or training
  imports. Resolves the canonical 20-move order (including a red first pick) and,
  on ban moves, returns denial-ranked ban candidates: roll the policy forward to
  the opponent's next pick, shortlist their top-k, ban each for real, score the
  finished drafts, rank by denial. The UI infers each simulated pick's role from
  the champion (a pro row has the real role), so its simulated win% is a harder
  task than the offline number and is shown only as a tiebreaker.
- **Explanation** (`/api/explain`) — retrieves the k most similar pro drafts
  (bitmask coarse pass + exact Jaccard on the board *at the query move*; ranking
  on the final draft would surface the wrong games), pulls patch bullets when the
  local patch corpus is present, and asks an LLM for a grounded explanation.
  Every citation is checked mechanically before it is returned: a game id must be
  in the retrieved set, a patch line a verbatim substring of the notes handed to
  the model. Failures are dropped and reported, one retry is allowed, a second
  failure returns `ok: false` rather than unchecked prose. The `reasoning` field
  is the model's own account, not verified, and is kept out of `evidence`. No
  `LLM_API_KEY` is not an error — the endpoint still returns the evidence.
  **Gap: prose quality is unvalidated.** Only the no-key path and a stub client
  have been exercised.
- **Breakdown** (`/breakdown.html`) — scores one pro game under every selected
  checkpoint, real move next to each model's top-5. Mirrors the test-split
  methodology (real patch id, exact role-keyed slots) rather than the UI's
  inferred-role path.

Verified against a live server on the seven test games, not by reading the
loader: all five checkpoints load through `/api/load` and answer their own
endpoint; `/api/next-move` returns the real move's kind *and* side at 35/35
prefixes across all seven games and reports `complete` at 20 moves; `/api/suggest`
reproduces the G2 board exactly (red Syndra/Caitlyn/Karma → MIDDLE/BOTTOM/UTILITY,
blue Yunara/Lulu/Locke → BOTTOM/UTILITY/MIDDLE) and the pin holds end to end;
`/api/explain` and `/api/breakdown` answer on 16.20; both pages render with no
console errors.

---

## 9. Plan

**Done.** Kit-attribute embeddings wired (verdict in §6). Retrieval + explanation
built. The patch embedding measured and kept. Pro-only retrained and shipped.

**Open, in expected-gain order.** All of these reuse the within-patch split and
the paired protocol, so results stay comparable to §4.

1. **Gate the patch vector by phase.** §4 is unambiguous: ban1 +2.43x, ban2
   +0.05x, pick2 −0.13x. Multiply it by a learned scalar per phase (4 params) or
   per move (20), initialised to 1 so step 0 reproduces today's model:
   `x = x + g_φ · patch_lookup(patch_id)`. Success: pick2 recovers to ≥ 0.6029
   while ban1 holds ≥ 0.4246 — read the pick2 row, not the pooled top-5.
2. **Make the encoder smaller, not bigger.** D=128 (~0.6M), or D=256 with
   dropout 0.25–0.3 and weight decay 0.05. Read the whole val curve, not the
   selected point. Caveat: re-measure the patch-vs-no-patch gap at the new size —
   the patch vector is the only signal composition cannot memorise, so it could
   get *more* valuable.
3. **soloQ as rehearsal, not initialisation.** Composition (pick1/pick2) is
   exactly where pro-only is weakest, and pick2 is where the patch vector hurts.
   Replace 10–20% of each batch with soloQ drafts carrying `patch_id = UNK`, so
   the patch table is untouched. Two variants: soloQ in the masked stage only, or
   in both. Success: pick1/pick2 up on the same 95,260 states, ban1 unchanged —
   if ban1 moves, soloQ is leaking patch information it does not have.
4. **LLM → patch vector via the output side.** The costly failure is that a
   brand-new patch gets no prior at all (`patch_to_id` → UNK, so the model falls
   back to the frequency table on day one). Feeding patch-note *text* into the
   encoder is measured to fail (§6), so: an LLM reads the notes and emits a
   per-champion delta, a small learned `W_a` maps it into the patch-embedding
   space, and the predicted vector initialises the new row before the usual
   fine-tuning. Measure by holding out a patch **entirely** and comparing UNK vs
   text-predicted vs an oracle row trained on all of its drafts.
5. **Rebuild the value head as its own stage.** Mask-aware pool (mean over
   present slots plus a presence count), trained separately, scored on Brier and
   a calibration curve — not accuracy, and not on 40 states. Gate: Brier below
   the base rate's with monotone calibration across deciles.
6. **Search, once value works.** Depth-2/3 alpha-beta with the policy as move
   ordering and value at the leaves; branching ~150 makes depth 2 ~22k batched
   evaluations, milliseconds on a GPU. Report the value delta of the recommended
   move. Only then consider PUCT; full MCTS is unnecessary at depth 20.

**What not to do.** Don't re-run patch conditioning on the date split. Don't
scale the model up on 37k drafts (§6, measured). Don't add the patch vector to
pick2 without a gate. Don't quote the pooled +19.6/+21.7 pp cross-pipeline gap as
a quality result — quote the 2026 rows or the paired same-flag numbers. Don't feed
patch-note text into the encoder. Don't expect soloQ pretraining to carry a
pro-only model further than the pro-only model already is. Don't use an LLM as a
judge of ban quality without a calibration set — it is unmeasurable here and is
the easiest way to generate a confident number that means nothing. Don't surface
a win probability before a value head beats the constant Brier.

**Unsettled.** One architecture size, one corpus for the main grid; `rel@5` is
not comparable across protocols (this window's baseline is 0.412 vs 0.373
patch-honest, so pro-only 1.38x and finetune 1.51x cannot be read as "finetune is
better"); the wider corpus and the scratch control are seed 0 only.

---

## 10. Reproduce

```bash
# the five kept artefacts, one command per class
docker compose run --rm dev python model/proonly/train_masked.py \
    --out-dir model/checkpoints/masking-proonly
docker compose run --rm dev python model/proonly/train_policy.py \
    --init-ckpt model/checkpoints/masking-proonly/best.pt \
    --out-dir model/checkpoints/nextpick-proonly
docker compose run --rm dev python model/finetune/train.py \
    --out-dir model/checkpoints/masking-finetune
docker compose run --rm dev python model/policy/train.py \
    --out-dir model/checkpoints/nextpick-policy
# the pretrain is kept, not retrained: model/checkpoints/masking-soloqueue/

# the comparison
docker compose run --rm dev python model/test_split/eval_test_split.py
docker compose run --rm dev python model/test_split/render_test_split.py
```

`--out-dir` writes a run straight to its final name; without it a run lands in
`ckpt/run_<ts>[_<name>]/`.

**The recipe.** Defaults in `train_masked.py` / `train_policy.py`; both stages
are D=256 / 2 intra + 1 cross / 8 heads / dropout 0.1 / weight decay 0.01.

| | stage 1, masked | stage 2, next-pick |
|---|---|---|
| objective | CE champ + 0.3 · CE role, 15% masks, 10% role-visible | CE over 173, illegal moves at −inf |
| batch / LR | 512 / 1e-3 | 512 / 1e-3 head, 1e-4 encoder |
| epochs / warmup | 40 / 3% | 40 / 5% |
| selects on | val `rel_champ_top5` | val `rel_top5` |

Two choices there are not tuning knobs. **One random prefix per game per
epoch** (`--prefix-mode random`): taking all 20 would weight a game's
early-move states 20x over its last. **Uniform recency weights**, not the
75-day half-life the other stages use: the test window is the tail of *every*
patch, so recency weighting would make old patches unlearnable and turn the
metric into a statement about recency. Selection is on `rel_*` — the ratio to
the frequency table — not raw top-5, so beating the popular patch is the target
rather than memorising it.

**Tests.** Run per directory — the suite does not combine into one `pytest
model/`. `finetune/train.py` and `policy/train.py` both do `import train`, so
those two must be separate invocations. `model/policy` must go in as a
*directory*: `model/policy/model.py` shadows the `model` package when that folder
is first on `sys.path`, which is what `pytest model/policy/test_infer.py` does.
Green today: `pytest model/policy model/proonly model/pretraining model/tests
model/test_split` (127 passed, 1 skipped), `pytest model/finetune` (25),
`pytest "data/pro play" frontend` (105). `soloq data
analysis/explorer/test_explorer.py` is excluded: it opens an HTTP connection at
import time and errors during collection when no server is running.

**Two reproduction traps that look like regressions.**

- **`--epochs` sets the LR schedule, not just the duration.** `total_steps =
  steps_per_epoch × epochs`, warmup is 3% of it, then cosine to zero. At 40
  epochs warmup (87 steps) is longer than one epoch (73), so the whole first
  epoch is ramp-up; at 2 epochs it is 4 steps and the run is at full LR almost
  immediately — and the *shorter* run looks better (train 3.34 vs 3.52 at epoch
  1). A re-run at a different `--epochs` is a different experiment.
- **`DataLoader(shuffle=True)` draws its sampler seed from the global torch RNG
  at construction.** So "build the loaders" must stay before "build the model"
  (it does, in both trainers), or the shuffle and the whole trajectory change.
  `torch.load` consumes no global RNG, so reading a checkpoint earlier is safe.

**Patch ids are vocabulary positions.** A checkpoint's patch table can only be
loaded by a run that indexes it the checkpoint's way. When a stage initialises
from a checkpoint that was patch-conditioned differently — or not at all — the
run's own vocabulary can legitimately differ, and an unseen patch falls back to
UNK. `load_encoder_state(..., allow_missing_patch=True)` tolerates a missing
table; the evaluation path stays strict. Assert vocabulary agreement before
trusting a paired comparison.
