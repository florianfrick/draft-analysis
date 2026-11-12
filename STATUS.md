# Project record — state, findings and plan

Last updated 2026-11-03. This is the single narrative document: what the models
are, what they measure, what failed and why, and what to do next. It replaces
`assumptions.md`, `plan-model-v2.md` and `pro/model/proonly/FINDINGS.md`, whose
content is distilled here.

Operational docs stay separate and this file does not repeat them: `README.md`
to get running, `DOCKER.md` for the container, and each folder's `README.md` for
its own commands.

**How to read it.** §7 is the current measurement. §4 is the earlier grid, kept
because it is the evidence behind the ablations in §6 — do not quote §4 as
today's scores. §5 and §6 are verdicts whose runs can no longer be reproduced;
they are here so nobody re-runs them.

## Contents

1. Where it stands
2. The model
3. The data and the split
4. Results, earlier runs
5. Ban analyses
6. Dead ends (measured, do not re-run)
7. The test split — current numbers
8. The frontend
9. Plan
10. Reproduce

---

## 1. Where it stands

Nine checkpoints, each in its track's store: `solo_queue/model/checkpoints/`
for `masking-soloqueue`, `masking-finetune` and `nextpick-policy`; everything
else under `pro/model/checkpoints/`. The track says which data trained the
model; the directory name states the inference it serves and the stage that
produced it, and `frontend/server._ckpt_roles` reads that prefix to decide which
of the two UI slots may load it.

| class | checkpoint | inference |
|---|---|---|
| pretraining (soloQ) | `masking-soloqueue/best.pt` | masked completion; the init for `masking-finetune` |
| finetune (soloQ → pro) | `masking-finetune/best_mask.pt` | masked completion, role board |
| pro-only, from scratch | `masking-proonly/best.pt` | masked completion, role board |
| policy (soloQ encoder) | `nextpick-policy/best_policy.pt` | next-move ranking |
| policy (pro-only) | `nextpick-proonly/best.pt` | next-move ranking |
| pro-only, all data, 16.20 val | `masking-proonly-through-wlds-ig-tes/best.pt` | masked completion, role board |
| pro-only, all data, 16.20 val | `nextpick-proonly-through-wlds-ig-tes/best.pt` | next-move ranking |
| pro-only, all data, random val | `masking-proonly-randomvalsplit-through-wlds-ig-tes/best.pt` | masked completion, role board — **the shipped default** |
| pro-only, all data, random val | `nextpick-proonly-randomvalsplit-through-wlds-ig-tes/best.pt` | next-move ranking — the shipped default until 2026-11-05 |
| pro-only, seeded 80/10/10 split | `nextpick-proto-ensemble/ensemble.json` | next-move ranking — the shipped default until 2026-11-07 (mean of 3 members' logits; +1.33 pp top-5 over the row above on the rows neither trained on — `pro/model/proto/README.md` §6) |
| pro-only, ban-aware | `nextpick-bans-aware-through-wlds-ig-tes/best.pt` | next-move ranking — **the shipped default** (same corpus and split as the `-randomvalsplit-` row, plus `ban_side` / `ban_fearless` / `counter_rank`; +0.60 pp top-5 over its own control, p 4.3e-8, **one seed** — `pro/model/bans/README.md`) |

The `-through-` checkpoints were added on 2026-11-04: same architecture and seed,
retrained on the whole corpus with no test split (§7). The suffix states which
series the training data ends on, so a retrain cannot quietly land in an old
directory name. None of them is the one §7 scores, because a run with `--no-test`
has no held-out number by construction.

The `-randomvalsplit-` pair (2026-11-04, later that day) is the same corpus and
frontier as the `-through-` pair and differs only in the val split: 10% of *every*
patch drawn at random per patch, instead of each patch's tail (§3). That trains on
90% of every patch including its end — 42,121 train / 4,758 val, against
42,100 / 4,768 / 7 for the originals and 46,827 / 52 for the `-through-` pair —
so it gives back 4,706 rows of the earlier refresh and buys a val 90x larger than
the 16.20-only one. It is the default because it is the better model on the honest
rows: on the 458 games that are val for both it and the original, masked top-5 is
+0.005 (p 0.47, i.e. tied) and next-move top-1 is +0.007 (p 0.024, bootstrap CI
[+0.001, +0.013]) with top-10 +0.009 (p 0.004). The `-through-` pair cannot be
compared to anything this way — it trained on nearly every row, so no honest
common set exists with it — which is why it is kept as a recorded experiment
rather than as a candidate.

The ban-aware arm (2026-11-07) is the `-randomvalsplit-` recipe plus three
off-by-default flags on `PolicyValueModel` — `ban_side` (the ban block gets the
side it was made by, derived as `BAN_SIDE_BY_SLOT ^ first_pick`), `ban_fearless`
(the fearless pool is an input), `counter_rank=32` (a rank-limited
counter-pick interaction). Its control reproduces the shipped model to 4e-5, so
the flags are the only difference: **0.5056 vs 0.4995 top-5, +0.60 pp, paired
bootstrap p = 4.3e-8, CI [+0.39, +0.82] pp**. It is a val number on the split its
control trained and selected under, not a held-out one.

Two caveats belong with that number, and the frontend default rests on it anyway
because the model is both more correct and not worse. First, it is **one seed**:
each single switch is flat (counter-only +0.08 pp, fearless-only +0.11 pp,
p 0.485), so the interaction and seed luck are not separated until a second seed
runs. Second, its predecessor's claim is not comparable — the ensemble's +1.33 pp
was measured on the 439 draft rows neither it nor its predecessor trained on.
Do not add the two numbers, and do not quote the ban-aware arm as beating the
ensemble. Why it is not a 4th ensemble member: the manifest's members are fixed
at three, and evicting one on a single seed would void the number that describes
the manifest (`pro/model/bans/README.md`).

Weights are git-ignored. 18 older checkpoints (1.9 GB) were deleted on
2026-11-03; their numbers live in `pro/model/proonly/reports/` and §4/§6 here, but
their paired comparisons can no longer be re-run.

**The corpus is pinned** at `pipeline.CORPUS_MAX_DATE = "2026-11-04 23:59:59"`
— 46,879 drafts, patch 16.20 = 262 rows. The 2026 OE export keeps growing, so
without the pin a rebuild folds new games in and moves every recorded number.
This is not bookkeeping: during the 2026-11-01 merge an ordinary `pytest` run
called `build_rows()`, saw the 2026 CSV had changed, rescanned and overwrote
`pro_drafts.pt` — training the corpus on its own holdout. Raising the pin means
re-measuring everything, which is exactly what happened on 2026-11-03 and again
on 2026-11-04.

The 2026-11-04 raise exists because the export was refreshed and added the
Worlds final: Invictus Gaming vs Top Esports, 2 games on 2026-11-03 and 2 on
2026-11-04, 46,875 → 46,879. Every number in §4/§7 was measured on 46,875.

**The test split is pinned by series id**, not by a fraction: three Worlds 2026
final-day series, 7 games, all patch 16.20. Split: **42,100 train / 4,768 val /
7 test**. See §7.

---

## 2. The model

**Lineage.** Set Transformer (Lee et al. 2018) without ISAB, plus BERT-style
masked modelling. Sketch in the root README: embedding → Siamese intra-team
attention → symmetric inter-team cross-attention → slot-level champion and role
heads, plus pooling to auxiliary heads. Frozen as an I/O contract and enforced
by tests (`solo_queue/model/pretraining/base_model.py`).

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
173-champion space, 46,879 drafts, skipping only for documented reasons (bad
rows, incomplete bans/picks, duplicate champion, pick mismatch). 12-row OE
blocks become a global 20-move sequence (6 bans1 + 6 picks1 + 4 bans2 + 4 picks2)
on the 2026 First-Selection order, pre-2026 blue-first; a move is
`(kind, team, champ, role)` with bans carrying IGNORE role. Era handling is
data-driven per series (2025+ mostly fearless, 2026 fully), and first-pick
decoupling is 2026 only.

Pro teaches what soloQ cannot: order/phase/side effects (ban priority, B1 blind
vs R5 counter, first-pick asymmetry), fearless depth — the main 2025+ shift —
and coordinated comp value. Limits: 46.9k games is small, no player identity,
econ labels are weak. Synergy/counter structure is assumed to drift slower than
popularity.

**The split is load-bearing, not bookkeeping.** `split_series_by_date` (the old
default) puts the whole patch on one side, so at test time the patch id is `UNK`
and any patch-conditioned input can only memorise — which is why the earlier
`policy-2` run measured patch conditioning as *harmful* (0.2029 vs 0.2558 top-5).
`split_within_patch` asks the deployment question instead: **you always know
today's patch and you have most of its games.** Series are grouped first (a
series belongs to the patch of its first game, so it can never straddle), sorted
by `(date, gameid)`, and the tail windows are filled backwards so test is newest.

The test split is then **pinned**: `pipeline.TEST_SERIES` names three
`series_id` strings, and `split_within_patch(pinned_test=...)` holds them out of
train and val *by id*, so the set cannot move when the corpus is rebuilt. While
the pin is non-empty `test_frac` is forced to 0; a pinned id missing from the
corpus raises rather than quietly shrinking the test set. `strict_pin=False`
exists only for `--limit-drafts` smoke runs, which subsample.

**Val is drawn, not tailed, in the shipped protocol.** The tail split holds out
the *newest* series of every patch — the rows that carry the meta as it currently
stands — so every patch's frontier is never trained on, and selection happens on
the one slice the model cannot interpolate from. `--val-split random` (the
default is still `tail`) draws whole series per patch instead: 90% of every
patch's timeline *including its end* is trained on, and val is spread across the
patch rather than sitting on its edge. The draw is seeded per patch
(`Random("%s|%s" % (VAL_SPLIT_SEED, patch))`) so one patch's val does not depend
on which other patches exist or on iteration order, and `VAL_SPLIT_SEED` is a
module constant rather than a flag for the same reason `TEST_SERIES` is: the
masked stage and the policy stage must draw the *same* val series. On 46,879
drafts that is **42,121 train / 4,758 val / 0 test**, 99 patches contributing
val, patch 16.20 at 236 of its 262 rows in train. The trade is explicit — the
draw costs 4,758 val rows against the 52 of the 16.20-only scope, and buys a
selection signal large enough to be stable (8,053 masked slots instead of 78,
where the difference between the chosen epoch and the plateau was inside the
noise).

**One split, and the seed was the whole bug.** The LLM track meant to reuse
this draw and passed `seed=0` to `split_within_patch`, which overrides
`VAL_SPLIT_SEED` — so it drew a *second, independent* 10% per patch. Two
independent 10% draws of one corpus overlap on ~1% of it, which is 90% of a val:
**90.0% of the LLM's val was inside the head's train**, and that was recorded as
a difference in `val_random` (the head was assumed to use `val_random=False`;
it does not — the shipped checkpoints record `test: skipped (--no-test)` and
`init_ckpt = masking-proonly-randomvalsplit-*`). Dropping `seed=0` from
`pro/model/llm/common.SPLIT` takes the overlap to **0.3%** (14 of 4,763 games),
and that residue is only the 7 pinned test games the head's `--no-test` run did
not remove. With the pin kept on both sides the three lists are identical
(`test_rerank.py::test_the_three_stages_split_the_corpus_identically`).

Consequences, all measured on 46,915 drafts:

| | games | states | what it is |
|---|---|---|---|
| shared train | 42,145 | — | encoder, policy head, LLM stage 1 and 2 |
| shared val (`--split heldout`) | 4,763 | 95,260 | every cross-model number |
| pinned test | 7 | 140 | three Worlds series, held out of everything |
| reranker (`--split rr`) | 2,381 / 2,382 | — | half the val trains, half evaluates |

The head's recall@5 on that val is **50.51%** (18.44% top-1, 58.27% picks /
42.75% bans), which retires the 51.85% ceiling measured on the old 478-game
`common` — that set was 92.3% inside the shipped head's train. The correction is
small (+1.3 pp) because the head barely memorised its own train, but it moves
every ratio that quoted 51.85%, including the reranker's 35.31% (69.9% of the
ceiling, pending re-scoring on `heldout`). The reranker's old pool was worse
than its eval: `headheld - common` was 90.3% head-train, so it trained on
candidate lists the head had memorised and scored itself on clean ones. `rr`
puts both sides at serving difficulty.

---

## 4. Results, earlier runs

The first four blocks are the date split on 46,813 drafts. The last block
(pro-only + patch embedding) is the within-patch split — the protocol §7
reuses, but on the pre-freeze corpus, so its numbers are what §6's ablations
are anchored to. Both grids are superseded by §7 for scores.

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
is the whole record. The UI kept a `ban_candidates` rollout until 2026-11-04
and then dropped it too — see the next-move note in §8.

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
4. **The masked models cannot *condition* on fearless — but an output mask is
   enough for the role board.** Their raw top-5 puts a fearless-unavailable
   champion in it on 85 of 120 slots (three models × four fearless games); the
   next-move models get `unavailable_mask` as an input and break it zero times.
   Set the pool to `-inf` and renormalise — which `frontend/server._top5` already
   does — and the answer is legal by construction while the real pick stays in
   the top-5 **0.95 / 0.875 / 0.625** of the time (`masking-proonly` /
   `masking-finetune` / `masking-soloqueue`), against 0.475 for a role-frequency
   prior. It cannot be worse than the unmasked answer: the real pick is never in
   the pool and `-inf` does not reorder what remains, so the mask only removes
   competitors. What it cannot do is recover information the model never had —
   the pool still holds 11–19% of the probability mass before masking, and when
   the model's first choice is in the pool its fallback is right 6/8, 2/6 and
   0/8 (n too small to rank). `eval_fearless_mask.py` reproduces all of it.

**Statistics discipline.** Seven games. One game is 14% of the sample and a 95%
Wilson interval on a rate over seven games is ~35 points wide, so every rate in
the report carries its interval. The aggregate tables are a smoke test on unseen
games — they catch a broken model and refuse to rank working ones. The per-state
tables carry the information, because every model is scored on the identical
board.

Full report: `pro/model/test_split/REPORT_test_split_evaluation.{md,html}`; evidence:
`pro/model/test_split/reports/test_split.json`.

**The all-data refresh has no number here, by construction.** The two
`-through-wlds-ig-tes` checkpoints (2026-11-04) are trained with `--no-test` on
the whole 46,879-draft corpus, and their val is the tail of patch 16.20 only —
52 games, 78 masked slots, 1,040 next-move states. That is a *correct* val
(16.20 is the only patch the product is asked about) but a small one, so it
ranks nothing and the tables above still belong to the originals. What it does
establish, from `test_report.json`:

| | selected epoch | val top-5 | val rel | val n |
|---|---|---|---|---|
| `masking-proonly-through-wlds-ig-tes` | 9 | 0.641 | 2.17x | 52 games / 78 slots |
| `nextpick-proonly-through-wlds-ig-tes` | 20 | 0.500 | 4.03x | 52 games / 1,040 states |

Two things worth keeping. **The masked stage peaks early and then overfits**:
val `rel_champ_top5` is 2.17x at epoch 9 and 1.78x at epoch 40 while train loss
falls 4.33 → 0.73. The original `masking-proonly` picked epoch 5 on a 4,768-row
val, so this is the task, not the smaller val. **The policy stage is the
opposite** — it selected epoch 20 and was still flat-to-improving at 40, with
per-phase rel 3.29 ban1 / 4.20 pick1 / 6.00 ban2 / 4.10 pick2. `--val-scope
latest` is what makes the second row's 4.03x comparable to the 3.22x the
original reported on its 4,768-row val only in the loosest sense: different val
sets, so read them as "the refresh did not break anything", not as a delta.

### Is the 16.20-only val helping or hurting?

Asked on 2026-11-04, because a 52-game val is a noisy place to choose an epoch
and the whole refresh rests on it. Answered by scoring the checkpoints on games
**neither model trained on**, which needs a split that is val in both protocols:
`compare.py --split shared`. That is 27 games (all patch 16.20) — val for the
original every-patch split and val for the refresh's newest-patch split. Both
models selected their epoch on those rows; neither trained on them, so the
comparison is paired and symmetric.

Reproducing §7 exactly first (0.714 masked top-5 and 0.564 next-move top-5 for
the originals) is what makes the harness trustworthy; `--mask-mode slot` hides
each pick in turn, so a number here is the same protocol as the tables above.
`compare.py`'s defaults (`--new-scope latest --val-split tail`) are this
subsection's arm; the random-val arm below is always named explicitly, and the
harness prints which protocol it built.

| comparison | set | task | top-1 | top-5 | top-10 |
|---|---|---|---|---|---|
| refresh vs original | shared 27 | next move | +0.002 | +0.015 (p .43) | −0.022 (p .21) |
| refresh vs original | shared 27 | masked | **+0.052** (p .055) | +0.007 | −0.004 |
| refresh vs original | pinned 7 | next move | +0.014 | +0.014 | +0.036 |
| refresh vs original | pinned 7 | masked | +0.014 | −0.014 | −0.057 |
| 16.20 val vs all-patch val | shared 29 | next move | −0.002 | **+0.026** (p .12) | −0.021 |
| 16.20 val vs all-patch val | shared 29 | masked | 0.000 | −0.031 (p .31) | **−0.041** (p .031) |
| refresh ep 9 vs ep 40 | shared 27 | masked | +0.022 | **+0.056** | **+0.059** |

The middle two rows are the actual question: a third pair trained on the same
46,879 drafts with `--val-scope all --no-test` (42,107 train / 4,772 val,
selected epoch 6 and 38) isolates the val scope from the extra data. **The
16.20-only val is not hurting.** It is ahead on the product metric — next-move
top-5, the number of candidates the panel shows — by 2.6 pp, and behind on
masked top-10 by 4.1 pp, which is the one nominally significant result (p .031)
and rests on McNemar alone; the game-level bootstrap on the same rows straddles
zero, and McNemar treats the 10 slots of a draft as independent. The role board
renders top-5, where the two are tied. Keep the 16.20 val, and keep it for the
reason that does not depend on any of these p-values: it trains on 4,720 more
rows and stops on the distribution the model is served on.

The last row is the one that settles the *method*. The refresh's own val picked
epoch 9; on held-out 16.20 games epoch 9 beats epoch 40 by 5.6 pp top-5 (bootstrap
CI [−0.104, −0.007]) and 5.9 pp top-10. A "train 40 epochs, keep the last" run
would have shipped a measurably worse encoder, and the 52-game val is what
caught it — which is the argument for keeping a val at all, at whatever scope.

Evidence: `pro/model/proonly/reports/compare_{shared,pinned,allval}_{policy,masked}.json`
and `compare_epoch9_vs_40_masked.json`. The all-patch-val weights were deleted
after losing the comparison; reproduce with the two `--val-scope all --no-test`
commands in §10.

### The random-per-patch val (2026-11-04, later that day)

The 16.20-only val was then dropped by decision, not by a measurement: it holds
out the *newest* series of the one patch that matters, so 52 games — and the end
of 16.20, the games closest to what the product is asked about — are never
trained on. `--val-split random` draws 10% of *every* patch's series at random
instead (§3), which trains on 236 of 16.20's 262 rows and spreads val across each
patch rather than sitting on its edge. Split: **42,121 train / 4,758 val / 0
test**, 99 patches contributing val.

| | selected epoch | val top-5 | val rel | val n |
|---|---|---|---|---|
| `masking-proonly-randomvalsplit-through-wlds-ig-tes` | 6 | 0.560 | 1.39x | 4,758 games / 8,053 slots |
| `nextpick-proonly-randomvalsplit-through-wlds-ig-tes` | 40 | 0.500 | 3.28x | 4,758 games / 95,160 states |

**These val columns are not comparable to the table above** — a different val set,
with a much stronger frequency baseline (masked `base_champ_top5` 0.403 against
0.295 on 16.20 only, because the older patches are more predictable). The rel
column drops from 2.17x to 1.39x for that reason alone, not because the model got
worse. What the larger val does buy is a selection signal that is not noise: the
masked stage still peaks early (rel 1.39x at epoch 6, 1.24x at 40) but the chosen
epoch is now picked out of 8,053 slots instead of 78.

**The honest check is against the originals**, on rows that are val in both the
random-per-patch split and the original every-patch split — 458 games across all
five years, 4,580 masked slots, 9,160 next-move states, trained by neither model:

| comparison | set | task | top-1 | top-5 | top-10 |
|---|---|---|---|---|---|
| random val vs original | shared 458 | next move | **+0.007** (p .024) | +0.003 (p .49) | **+0.009** (p .004) |
| random val vs original | shared 458 | masked | −0.005 (p .51) | +0.005 (p .47) | −0.005 (p .34) |

So the policy model is slightly but consistently ahead (top-1's game bootstrap CI
is [+0.001, +0.013], the only interval here that excludes zero) and the masked
model is tied to within 0.5 pp. The masked per-year split is the more interesting
number, because it is monotone in recency — 2022 −0.002, 2023 −0.008, 2024 +0.003,
2025 +0.011, 2026 **+0.018** — which is what training on each patch's *end*
instead of only its beginning should do, and it is the effect the change was made
for. It is not significant per year, and it is not offered as a win: the claim is
that the split is no longer holding out the frontier, and the frontier is where
the gain is.

The `-through-` pair is **not** a usable baseline for this: it trained on 46,827
of 46,879 rows, so it has no honest common set with anything. That is why the
comparison above is against the originals, and why the `-through-` pair stays in
the record as a measured experiment rather than a candidate. Per instruction, the
three-arm control of the previous subsection was not repeated.

Evidence: `pro/model/proonly/reports/compare_shared_random_{masked,policy}.json`.

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
- **Next move** — `solo_queue/model/policy/infer.py:PolicyPredictor`, no dataset or training
  imports. Resolves the canonical 20-move order (including a red first pick) and
  runs one forward pass for the side on the clock. A ban suggestion is the same
  distribution as a pick suggestion: the policy's imitation of what a pro does
  on that move. The patch is a real input (`patch_id`, default 16.20), chosen
  from the checkpoint's own `patch_vocab`; before 2026-11-04 every request used
  the UNK row, which flattens the opening ban (top champion 4.1% instead of
  11.3%). The board carries no role labels, so picked champions' roles are
  inferred from the encoder's role head first.
  **The rollout is gone** — `ban_candidates`, `_rollout`, `_win_prob`, the
  `--value` flag and the ban panel were removed on 2026-11-04. A pro-like ban
  already encodes denial, and with no usable value head the simulation only
  compounded model error; §5 keeps the measurements that motivated it.
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
- **Next-move highlight** — the green outline is a property of the board, not
  of a model answer. `GET /api/champions` publishes `move_plan` (both first-pick
  sides, from `infer.move_plan`, itself checked against
  `pipeline.interleaved_moves`) and `draft-order.js` resolves the next turn
  synchronously from the board's ban/pick counts, exactly as the server's
  `resolve_position` does. It used to read `/api/next-move`'s `state`, which is
  an async answer for the *previous* board: stepping a real draft showed the
  same team banning twice at the start and a fourth ban slot in ban phase 1
  after all six bans (the whole highlight was one move behind). The next-move
  panel now blanks while its answer is in flight for the same reason.
  `test_server.py` pins the published plan to `infer.move_plan` and guards
  `renderTurn` against ever reading the async answer again.
- **Breakdown** (`/breakdown.html`) — scores one pro game under every selected
  checkpoint, real move next to each model's top-5. Mirrors the test-split
  methodology (real patch id, exact role-keyed slots) rather than the UI's
  inferred-role path.
- **Series and fearless** — a series is a list of games; each carries its own
  board, its own first-pick side and its own fearless pool (the picks of the
  games before it — the corpus's own definition, exact on all 10,021 fearless
  rows). "next game in series" closes a game and opens the next with that
  pool; the first-pick side is kept because it stays the same across 94% of
  pro games while the teams swap sides. **The pool is per game, not one
  running list**: a running list makes a closed game illegal against itself
  (its own picks are in the pool), so switching back to G1 would show a board
  the server rejects. Only the two `nextpick-*` models consume it. Serving
  folds the pool into `taken_mask`, so a fearless champion is excluded from
  every suggestion, pick **or** ban: pro teams never ban one. The board still
  *accepts* such a ban instead of returning 400 — 8 of 100,210 fearless-row
  ban moves in the corpus do exactly that, and the UI must be able to display
  a real game — it just never proposes one. Training keeps the looser rule
  (`_apply_legality` masks the pool on picks only) because masking those 8
  training targets would dominate the cross-entropy. The `masking-*` models
  have no such input, and their answers are filtered rather than conditioned —
  the bar says so.

- **Chat** (`/api/chat`, `frontend/chat.py` + `frontend/tools.py`) — the inverted
  loop. The adapter cannot generate (7-8 % top-1, one name, no stop token) but
  can select (33.85 %), so the server computes the legal pool, the head's whole
  distribution over it, and the corpus evidence, and the model only chooses
  among those. A user's insistence enters as a prior in log space
  (`log p' = log p + λ·1[c=named]`, renormalised, so the mass comes out of the
  other candidates) and the reply reports the λ it used, the rank the named
  champion moved from and to, and the λ it *would* have taken
  (`log p_top − log p_named`). Measured on the shipped head at move 9: Kled at
  rank 76 of 164 → rank 1 at +6.0 nats ("incredibly"), rank 37 at +1.5
  ("alright", pick unchanged, 5.0 nats needed), rank 164 at −6.0 ("bad").
  **The λ ladder is uncalibrated; the required-λ number is exact**, which is why
  both are printed. Tools return `n` and a `reliable` flag and refuse a rate
  below `MIN_CELL=30` — `_probe_tool_power.py` measured the same-lane pair a
  counter-pick claim needs at a median of 5 games. `next_move_counts` exists
  because the per-champion cell is 0-2 of 120 at a given move; aggregating the
  same 120 boards gives a leader of 8-15. The model does not call the tools —
  the server does. It answers from the head even when the reranker is loaded,
  because a reranker has no distribution to boost, and it does not answer for
  the off-turn side (§4: the two sides are not differentiated). The adjusted
  ranking is drawn a second time in the panel, **below** the head's own chips,
  with the named champion always present (after a `…` when it is outside the top
  eight) and `chatSig` — picks, bans, first pick, fearless, patch, *not* the
  lane — as its freshness key. One turn, not a running total: two messages do
  not stack. **The reason is not read**: a true, a false and an irrelevant
  "because …" all buy the same λ (measured on the real name table), and the
  reply says so. A champion named after `into/against/vs/facing/counters` is
  marked `opponent`, gets λ=0 and feeds `pair_evidence` only — before that rule
  "good into Olaf" boosted Olaf and "good against Olaf" scored Kled at −3, since
  `against` also sits in `NEGATIVE`.

Verified against a live server on the seven test games, not by reading the
loader: all five checkpoints load through `/api/load` and answer their own
endpoint; `/api/next-move` returns the real move's kind *and* side at 35/35
prefixes across all seven games and reports `complete` at 20 moves; `/api/suggest`
reproduces the G2 board exactly (red Syndra/Caitlyn/Karma → MIDDLE/BOTTOM/UTILITY,
blue Yunara/Lulu/Locke → BOTTOM/UTILITY/MIDDLE) and the pin holds end to end;
`/api/explain` and `/api/breakdown` answer on 16.20; both pages render with no
console errors. The fearless path was verified in a browser on the same build:
manual add, advance to G2 (pool 1 → 6, none of G1's picks offered), per-game
first pick, switching back to a closed game with no board error, and a real
G4 walkthrough seeding its 30 real fearless champions with no illegal pick
across 12 replayed moves.

### Two models, one board — an open observation

**Not acted on, and no conclusion drawn yet.** Stepping through IG vs TES game 2
(2026-11-03) in the UI, the role board and the next-move panel disagreed in a way
that looks systematic: the `masking-*` board assigned Lulu to UTILITY at 100%
confidence, and at move 12 the `nextpick-*` panel still put Thresh — a champion
that is a support in any draft — at the top of its suggestion. Utility was
already occupied by Lulu, so the two panels cannot both be right about the board
they are describing.

The two slots are independent by construction (§8), and the asymmetry is
structural, not a bug in either loader: the role board is *role-conditioned* —
it is asked "who fills UTILITY?" and answers per slot — while the next-move model
is asked "what does a pro do next?" over 173 champions and has to recover roles
from the encoder's role head, which is a soft input, not a constraint. So the
masking model can be confident about a role the policy model never conditions on.
That is also why the observation is worth recording rather than patching: the
cheap version of a fix (feeds the board's role assignment into the policy's
legality mask) would make the policy model inherit the masking model's errors,
and it would be scored by no existing number — the test split measures each model
separately (§7), never the pair.

What is known: one game, one move, one side. What is not: whether this repeats,
whether it is worse in the pick phases than the ban phases, and whether the
policy's own inferred roles agree with the board's on the same board. The
`/breakdown` page scores both checkpoints on one game and is the place to check
the third question without writing anything new. No change was made on the
strength of this.

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

**Items 5 and 6 are declined (2026-11-10).** The value head is off the table: it
did not work in the soloQ pipeline, it does not rank picks (§18: 0.5002
[0.4974, 0.5028]), and the win-prediction branch found team strength dominates
the draft. Nothing should be built to accommodate one — in particular, do not
reorder or duplicate the pooled team halves for a value branch's benefit. The
next-move head stays `with_value=False`.

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

### Blocked on an LLM key: the post-draft verdict

**Wanted, not built — there is no LLM API key yet**, so this is a note rather than
work in progress. After a draft is finished in the UI, a button that records who
won ("red won" / "blue won") should hand the completed 20-move board to an LLM,
which then explains two things:

1. **Was there a final team-comp advantage?** — did the composition that came out
   of the draft decide the game, and in which direction: a lane-by-lane and
   win-condition read of the two finished comps, not a restatement of the score.
2. **Which step of the draft should have changed to improve?** — name the move
   (ban or pick, phase and side, and which side it was on) and say what should
   have gone there instead. The interesting case is the losing side's earliest
   *reachable* mistake, not the last move before the loss.

The button is the only new UI input: a game loaded from the corpus already carries
`blue_win`, and for a hypothetical board the click is what supplies it — so the
outcome is data, not a model output. What the LLM adds is the reasoning, and the
risk is hindsight — with the result known, any explanation is unfalsifiable, and a
plausible one is not a correct one. `/api/explain` (§8) already sets the discipline
this should reuse: retrieve the k most similar pro drafts *at the query move* plus
patch bullets, and mechanically check every citation before returning it, with a
clean `ok: false` rather than unchecked prose. The same should hold here — the
verdict cites real games or says nothing — and the `reasoning` field stays the
model's own account, out of `evidence`.

**Why it might matter beyond the UI.** If the verdicts hold up, they are a weak
supervision source for the two things this project is missing: an **advantage /
value target** (the plan's item 5) and a **counterfactual preference** between a
real move and a plausible alternative, which is exactly what the policy head
cannot learn from imitation alone. That is the second-order use, and it is
downstream of the same gate as item 5: a verdict has to be scored against
something before it can be a label, or the model learns the LLM's taste. The
cheapest honest version is a small hand-labelled set of drafts — the reviewer says
which step actually mattered — and then the LLM is only used where it agrees with
that set, which also gives the calibration the "what not to do" list demands.

### Conversational steering — the long-run goal, not built

**Wanted, not built, and deliberately out of scope for the current work.** The
end state is that a user types "Kled looks good for Blue here, Red's jungler plus
that ban points them at Olaf next" and the model re-ranks with that in view and
answers in prose. The draft-score panel and the local explanation are the first
step in that direction: they give the LLM something to talk about on a board the
user did not ask it to change.

The reasoning is written up in `pro/model/llm/README.md` ("Where a coach-style
model would come from"), and it argues against the obvious plan. The three
results that decide it are all already in this file: the head does not overfit
its own train (50.51 % val top-5 against rows seen 40 times), so the constraint is
games rather than parameters; the win-prediction work found most of the
draft → outcome signal is champion strength (0.0031 nats, nearly all of it a
Bradley-Terry prior); and the reranker's gain is +13 pp from format against
+3.1 pp from training. Together they say the sentence in the example is not
recoverable from outcome labels at all — it is commentary, patch notes and coach
talk, which is a **post-training** corpus on an existing LLM, not a pretraining
corpus on draft tensors. The encoder stays: it is the only component that
*generates* a candidate list, and the LLM has been measured to lose at that.

The gate before any of it: a sentence about Kled has to move the ranking of
champions it does not mention. Movement on Kled alone is compliance, not
reasoning, and `/api/draft-score` already returns the per-pick ranks that would
show the difference.

### The agent experiment — `exp/langgraph`, built, not yet evaluated

**Built (2026-11-11).** The thesis being tested: a zero-shot end-to-end LLM draft
assistant is not reachable by fine-tuning here, so put the trained models *behind*
the LLM as tools and let it orchestrate. `pro/model/agent/` now has the tool layer
(`schema.py`, `sql_tool.py`, `model_tools.py`, `session.py`), a bounded
tool-calling graph (`loop.py`), an OpenAI-compatible client (`llm_client.py`) and
an MCP stdio server (`mcp_server.py`). 70 tests. Two tiers by design, because
neither works alone: **high-level** (`predict_next_move`, `rank_candidates` - the
heads that already rank well) and **very low level** (one read-only `sql` tool
over the whole corpus, with the schema in the prompt).

**The prediction, recorded before measuring.** MCP-as-interface works (high
confidence). The LLM as a full-pool ranker disappoints at **18-28 % top-1** -
against 18.44 % for the head's own argmax and 33.85 % for the fine-tuned k=5
reranker, i.e. no better than the model it is wrapping. End-to-end autonomy fails
unmeasurably. The one place an agent can add value is the **13.65 % of val truths
at ranks 6-20**, which the reranker structurally cannot reach; aggregate top-1
will hide that slice, so the evaluation has to be split by it.

**Why the low-level tier is a SQL tool and not ten more named tools.** The
`_probe_tool_power.py` result: a per-champion cell is 0-2 of 120 boards at a given
move, so the aggregate I guessed a coach wanted is usually empty. Guessing the
right aggregates is the error; a query tool moves the choice to the model. The
cost is that the schema becomes the prompt, and three properties of this corpus
are silent-wrong-answer traps (patches overlap in time, `move_idx` is draft order
not a slot, `split` is not optional) - each is handled in the schema and repeated
in the tool's `hint`, and each has a test.

**The measurement that motivated instrumentation.** The k=20 reranker experiment
(§6) converted only 29 % of a 86.35 % ceiling, so the shortlist was never the
bottleneck. If the agent's gain is also small, the useful question is not "does it
pick better" but "what did it do" - hence every call logged with arguments,
latency, row count and success, and `sql_share` as the number the design turns
on: **more than half the calls being `sql` means the high-level tier is
under-specified, not that SQL is over-powered.**

**Ablation is part of the instrument, not a variant.** `--ablate sql` hides the
spec *and* refuses the call. Hiding tests the shipped configuration; refusing
catches the model that invents a tool it was never offered. Both are logged as
`ablated`, not as errors.

**One structural finding already.** Every model tool takes the whole board, so the
model re-types nine champion names per call. The loop fills an *omitted* board
(the turn's board is unambiguous) but **flags** a board the model retyped
differently - `transcription.mismatched`. Silently correcting it would hide the
most likely failure of the architecture. If mismatches are common, the fix is a
stateful server that holds the board, not a better prompt.

**Not written: the 200-state eval.** Whether the agent picks better than the head
on the shared val, split by whether the truth was inside the head's top-20. Until
that exists, this branch measures tool mix and nothing else, and the branch does
not merge.

**Provider (2026-11-11).** `llm_client` is OpenAI-compatible, so the provider is
three env vars. Defaults are DeepSeek (`api.deepseek.com`, `deepseek-flash`)
because that is the model the experiment was specified against - and
`deepseek-flash` is served by **DeepSeek-V4.1-Flash-1017**, the same model behind
the WorkBuddy chat. Two corrections found while wiring it: the id
`deepseek-v4.1-flash` **does not exist** (404; the legacy `deepseek-v4-flash`
does resolve to the same model), and the WorkBuddy keyless cloud LLM is
**not usable here** - it is `/.cloud/llm/chat/completions` on a deployed app's
origin, reached through a browser/mini-program SDK with an exact-Origin match,
scoped to that app's quota rather than callable from a local process. Cost of the
eval at DeepSeek peak rates: ~500 calls x ~1.3k in / 150 out, so **~¥1-2**, half
that off-peak. `loop.py --ping` validates a provider in one round trip, including
whether it can call tools at all - the three failure modes (bad URL, retired
model, no tool support) are otherwise indistinguishable from inside a batch.

---

## 10. Reproduce

```bash
# the five kept artefacts, one command per class
docker compose run --rm dev python pro/model/proonly/train_masked.py \
    --out-dir pro/model/checkpoints/masking-proonly
docker compose run --rm dev python pro/model/proonly/train_policy.py \
    --init-ckpt pro/model/checkpoints/masking-proonly/best.pt \
    --out-dir pro/model/checkpoints/nextpick-proonly
docker compose run --rm dev python solo_queue/model/finetune/train.py \
    --out-dir solo_queue/model/checkpoints/masking-finetune
docker compose run --rm dev python solo_queue/model/policy/train.py \
    --out-dir solo_queue/model/checkpoints/nextpick-policy
# the pretrain is kept, not retrained: solo_queue/model/checkpoints/masking-soloqueue/

# the shipped defaults: all the data, no test split, val = a random 10% of every
# patch (stage 1 ~7 min, stage 2 ~14 min)
docker compose run --rm dev python pro/model/proonly/train_masked.py \
    --no-test --val-split random --seed 0 \
    --out-dir pro/model/checkpoints/masking-proonly-randomvalsplit-through-wlds-ig-tes
docker compose run --rm dev python pro/model/proonly/train_policy.py \
    --no-test --val-split random --seed 0 \
    --init-ckpt pro/model/checkpoints/masking-proonly-randomvalsplit-through-wlds-ig-tes/best.pt \
    --out-dir pro/model/checkpoints/nextpick-proonly-randomvalsplit-through-wlds-ig-tes

# the earlier refresh, kept as a recorded experiment: same data, val on 16.20 only
docker compose run --rm dev python pro/model/proonly/train_masked.py \
    --no-test --val-scope latest --val-frac 0.20 \
    --out-dir pro/model/checkpoints/masking-proonly-through-wlds-ig-tes
docker compose run --rm dev python pro/model/proonly/train_policy.py \
    --no-test --val-scope latest --val-frac 0.20 \
    --init-ckpt pro/model/checkpoints/masking-proonly-through-wlds-ig-tes/best.pt \
    --out-dir pro/model/checkpoints/nextpick-proonly-through-wlds-ig-tes

# the honest comparison for a no-test run: rows val for BOTH protocols
docker compose run --rm dev python pro/model/proonly/compare.py --task policy \
    --split shared --other old --new-scope all --val-split random \
    --a pro/model/checkpoints/nextpick-proonly/best.pt \
    --b pro/model/checkpoints/nextpick-proonly-randomvalsplit-through-wlds-ig-tes/best.pt
docker compose run --rm dev python pro/model/proonly/compare.py --task masked \
    --split shared --other old --new-scope all --val-split random --mask-mode slot \
    --a pro/model/checkpoints/masking-proonly/best.pt \
    --b pro/model/checkpoints/masking-proonly-randomvalsplit-through-wlds-ig-tes/best.pt

# the comparison
docker compose run --rm dev python pro/model/test_split/eval_test_split.py
docker compose run --rm dev python pro/model/test_split/render_test_split.py
```

`--out-dir` writes a run straight to its final name; without it a run lands in
`ckpt/run_<ts>[_<name>]/`. For the two `-through-` commands the name is not
free text: `--out-dir` is refused unless it ends with the `-through-<league>-<a>-<b>`
suffix derived from the corpus frontier, so a retrain cannot inherit a stale
name. A later export moves that suffix, which is the point.

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

**Tests.** Run per directory — the suite does not combine into one `pytest .`.
`solo_queue/model/finetune/train.py` and `solo_queue/model/policy/train.py` are
both `train`, and `test_datasets_aug.py` does `import train as T`; collecting
both folders in one session leaves `train` bound to whichever was imported first,
and three of that file's tests then call the wrong `evaluate` signature. The old
`model.py` shadowing is gone: the package is `solo_queue.model`, so
`pytest solo_queue/model/policy/test_infer.py` and the directory form are now
equivalent.
Green today (2026-11-07, after the `solo_queue/` + `pro/` split):
`pytest solo_queue/model/tests` 52 passed 1 skipped;
`pytest solo_queue/model/finetune` 25 passed;
`pytest solo_queue/model/policy` 59 passed, 1 failed
(`test_swap_sides_flips_econ_team_pairs`, pre-existing);
`pytest pro/model/proonly pro/model/bans pro/model/test_split` 47 passed 1 skipped;
`pytest "data/pro play"` 41 passed;
`pytest frontend` 76 passed 1 skipped (the ensemble-symlink test skips on
Windows, where `os.symlink` needs a privilege the shell does not have).

**`pytest frontend` read 58 passed 18 skipped for a while after the split, and
that was a lie.** Three globs in `test_server.py` still named pre-split paths
(`model/checkpoints/nextpick-policy/best_policy.pt`, `.../masking-finetune/*.pt`)
and the `policy_ckpt` fixture called `pytest.skip` when its glob came up empty,
so 18 next-move and finetune tests stopped running while the suite stayed green.
A skip that means "the fixture is broken" and a skip that means "this clone has
no weights" look identical; only the count gives it away. The globs are fixed and
the fixture now prefers `POLICY_DEFAULT`, so the endpoint tests exercise the model
that actually ships.
`solo_queue/analysis/explorer/test_explorer.py` is excluded: it opens an HTTP
connection at import time and errors during collection when no server is running.

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
