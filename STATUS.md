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

The kept checkpoints, each in its track's store: `solo_queue/model/checkpoints/`
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
| pro-only, all data, random val | `masking-proonly-randomvalsplit-through-wlds-ig-tes/best.pt` | masked completion, role board — the shipped default until 2026-11-14 |
| pro-only, all data, random val | `nextpick-proonly-randomvalsplit-through-wlds-ig-tes/best.pt` | next-move ranking — the shipped default until 2026-11-05 |
| pro-only, seeded 80/10/10 split | `nextpick-proto-ensemble/ensemble.json` | next-move ranking — the shipped default until 2026-11-07 (mean of 3 members' logits; +1.33 pp top-5 over the row above on the rows neither trained on — `pro/model/proto/README.md` §6) |
| pro-only, ban-aware | `nextpick-bans-aware-through-wlds-ig-tes/best.pt` | next-move ranking — the shipped default 2026-11-07..11-14 (same corpus and split as the `-randomvalsplit-` row, plus `ban_side` / `ban_fearless` / `counter_rank`; +0.60 pp top-5 over its own control, p 4.3e-8, **one seed** — `pro/model/bans/README.md`) |
| pro-only, 2026-11-13 corpus, 85/10/5 | `masking-proonly-nov13-through-lcp-df-sw/best.pt` | masked completion, role board — **the shipped default** |
| pro-only, 2026-11-13 corpus, 85/10/5 | `nextpick-proonly-nov13-k5-through-lcp-df-sw/best.pt` | next-move ranking — **the shipped default** (5 prefixes/game/epoch) |
| pro-only, same, k=10 | `nextpick-proonly-nov13-k10-through-lcp-df-sw/best.pt` | selectable; top-5 tied with k=5 (p 0.07), top-10 +1.0 pp, best epoch 14 |

The `-nov13-` pair (2026-11-14) is **not comparable with any row above** and
must not be put in the same table of numbers: it trains on an *unfrozen* corpus
(46,939 drafts through 2026-11-13, `--corpus 2022-full`), the three pinned test
series are ordinary rows (`--no-pin`), and the split is the fractional 85/10/5
within-patch tail — 39,720 / 4,771 / 2,448, no fixed holdout. It is the only
pair with a 2,448-game out-of-sample set instead of 7. Its own scores: masked
top-5 0.5638 (rel 1.407x), next-move top-5 0.5083 (rel 3.36x). The k axis was
measured on it: k=5 beats k=1 by +2.55 pp top-5 (paired, p 2.7e-51) and k=10 is
tied with k=5 (p 0.07), so 5 is shipped.

The four **flat baselines** (`nextpick-mlp-flat-v1`, `masking-mlp-flat-v1`,
`nextpick-forest-flat-v1`, `masking-forest-flat-v1`) are also directories in
`pro/model/checkpoints/`, but they are not checkpoints in the sense of the table:
each ships a `flat.json` plus a non-transformer estimator, and `registry.py`
scores them as baselines rather than as candidates for shipping. They exist to
size the architecture — how much of the result is attention and set structure —
and they train on the same canonical split as the `-nov13-` pair, so they are
comparable rows. Sources in `pro/model/mlp/` and `pro/model/forest/`; the numbers
are in `pro/model/test_split/REPORT_suite.md` and are deliberately not restated
here.

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

> **Superseded for new comparisons (2026-11-15).** This section measures a
> 7-game pinned test split: 140 next-move states and 70 masked slots, where a
> 95 % Wilson interval is ~22 points wide, so it can detect a broken model and
> rank nothing. It also is not the split the shipped `-nov13-` pair trained
> under. The canonical split is now `pro/model/test_split/split.py`
> (39,720 / 4,771 / 2,448 on the unfrozen 46,939-draft corpus, no pin) and the
> harness is `pro/model/test_split/suite.py`, with two task suites that score
> any checkpoint against the shipped pair. **Use that for a comparison; use
> this section for the per-state tables it is good at, and for the numbers
> recorded before the split moved.** The old harness is preserved at
> `pro/model/archive/test_split_legacy/`.

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

Full report: `pro/model/archive/test_split_legacy/REPORT_test_split_evaluation.{md,html}`; evidence:
`pro/model/archive/test_split_legacy/reports/test_split.json`.

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

> **Read this as a record.** As of 2026-11-13 the frontend is four things - the
> board, `/api/next-move`, `/api/suggest` and `/api/agent` - and `/api/explain`,
> `/api/breakdown`, `/api/draft-score`, `/api/tool`, `/api/chat` and the reranker
> slot below no longer exist (§9, "The frontend was replaced"). Their measured
> results are kept because they are why the agent exists; `frontend/README.md`
> describes what ships.

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

### A hosted API instead of the reranker — measured, and it is a third of the gain

The question was whether the 4 B Qwen could be dropped from the shipped image
and replaced by a pay-per-token API call, on the theory that the +13 pp the
reranker adds is mostly the shortlist rather than the model. `pro/model/llm/
eval_api_rerank.py` scores that directly: it sends the LoRA's own v4 text
(board + `Candidates:` line) to a hosted model, keeps the head's top-5, and
scores the answer against the head **on the same rows**, so nothing depends on
reproducing 95,260 states.

200 games of the shared val, 4,000 states, `deepseek-chat`, k=5:

| | top-1 (exact champion id) |
|---|---|
| head argmax | 18.65 % |
| ceiling (head recall@5) | 51.08 % |
| API + shortlist | **22.68 %** [21.40, 24.00] |
| API + shortlist, tool-shaped state | 22.25 % |
| API + shortlist, names only | 18.27 % |
| shipped reranker | 33.15 % `exact` |

+4.03 pp over the head's argmax, McNemar p=2.3e-14 (305/144 discordant), 44 %
of the ceiling converted against the reranker's 67 %. So: **the API recovers
about a third of the reranker's gain, and the adapter is the rest.**

Two details decide how to read it. **All of the gain is the head's own
numbers**: with the percentages stripped from the `Candidates:` line the API
scores 18.27 %, i.e. the head to within noise (p=0.63) — it is not exercising
independent draft judgement, it is re-reading a distribution and occasionally
overriding it. And the gain is entirely on **picks** (+8.90 pp) with bans flat
(-0.85), the reverse of where the trained reranker improved (+4.39 on bans,
+1.92 on picks), so it is not a cheaper version of the same thing.

**The tools do not substitute for the soft block.** Since `pad`/`mean` say the
block is only a "a real draft is in progress" key, the agent's own
`DraftState.render()` — which names the move, glosses the fearless pool and
gives the legal count — should have supplied the same thing and freed the model
to reason. It scores 22.25 %, 0.43 pp *below* the LoRA's board text and **more**
deferential to the head (196 rows won against the head, 52 lost, versus 305 and
144), and it does not shorten `deepseek-flash`'s reasoning (median 1,669 →
3,142 chars). Pre-digesting the state does not create judgement.

Operationally: `deepseek-flash` is unusable here — it is a reasoning model,
spends a median 1,739 characters thinking (tail 30,202), and fails specifically
in ban phase 2, where `Turn: Red 4 (Ban 7 of 10)` sends it reconstructing the
pro draft schedule; 17-33 % of rows come back with empty `content` and
`finish_reason: length`. `deepseek-chat` answers in 0.15 s/board with no
truncation, which is what the numbers above use. The full shared val would be
~4 h of API calls at 12 workers, against one 0.167 s GPU step.

Not tested, and the only thing left that could change the conclusion: k=20,
where the 4 B model converted 29 % of an 86 % ceiling and a larger model has
room the 4 B did not.

### The tool loop instead of the reranker — built, and it lands 3.6 pp short

The next question was the agent one: with state management, a next-move tool
whose `k` the model picks, and a lane table — no SQL, no `rank_candidates` —
does the model's reasoning do the middle-man's job? `pro/model/llm/
eval_api_agent.py` runs the real loop on the same 4,000 states;
`pro/model/agent/prompt_explained.py` gives it the head's provenance and its
measured accuracy (18.7 / 51.1 / 85.2 %); `predict_lanes` is new and exposes the
lane table with `p_lane` as the confidence.

| | top-1 |
|---|---|
| head argmax, tool path (inferred lanes) | 18.35 % |
| **agent + state + next-move + lanes** | **19.08 %** [17.89, 20.32] |
| single-turn, k=5, lane given | 22.68 % |
| shipped reranker | 33.85 % |

+0.73 pp over the head it can see (McNemar p=0.0038, 62 won / 33 lost), and
3.6 pp behind a single-turn call that was handed the target's lane. It defers to
the head's #1 on 90.0 % of rows and overrules on 9.9 %; on those it overrules the
head was right only 8.29 % of the time, so the *decision* is well-targeted and
the replacement is worth +7.3 pp there. That trade is the whole gain.

**The lane set is not the lane.** On the 2,000 pick rows, the head's #1 cannot
fill any open lane on 6.6 % of them, and "first lane-legal in the top 20" is
worth +1.0 pp (21.45 → 22.45). The agent's picks score 22.40 % — exactly the
mechanical rule. "Best champion in the *right* lane" is worth +12.3 pp (33.70),
and the right lane is in the label, not the board. The single-turn arm was told
it; the agent has to guess among two or three open lanes. That is the 3.6 pp, not
reasoning quality. It also reconciles `--no-probs`: numbers alone buy nothing
(18.27), numbers + a lane buy 4 pp, numbers + only the lane set buy 0.7 pp.

**It asks for the widest list every time**: k=20 on 3,979 of 4,000 rows (10 on
15, 12 on 6), converting 22 % of an 85.2 % ceiling against the single-turn arm's
44 % of a 51.1 % ceiling. That is the reranker's k=20 dead end reproduced with a
different model — the bottleneck is comparing twenty names, not seeing them.
Partly prompt-induced: the guidance says k=20 is the widest read that pays.

Loop cost: 2.9 calls/row, 34 min for 4,000 rows at 8 workers, `predict_lanes` on
3,190 rows (bans have no lane), exploration on ~4 %, 3,998/4,000 answers parsed
from a required `PICK:` line. Two silent defects fixed: `rerank.load_head` never
attached `patch_vocab` (every tool call answered as UNK while the batch path used
the real patch, worth 0.3 pp), and `model_tools` stringified
`PolicyPredictor.model`, dumping the module repr into every tool result.

Untested and worth one run each: the same loop *given* the lane (expected near
31 %); a forced k=5/8; `deepseek-flash` at `--max-tokens 8192`.

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
an MCP stdio server (`mcp_server.py`). 96 tests. Two tiers by design, because
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

**The SQL tool was measuring the filesystem, not the corpus.** The corpus DB
lives on the bind mount, and gRPC-FUSE charges ~1 ms per random 4 KB read. Six
representative queries took **24.05 s** over the mount and **0.46 s** on
container-local disk - 52x, identical SQL - so the 2 s watchdog was aborting
questions the corpus answers in milliseconds. Two fixes, both required:
`schema.connect` now copies the DB to `LOL_DB_CACHE` (compose sets `/tmp`) and
opens the copy, and an aborted query returns `timed_out: true` instead of an
ordinary SQL error, because `rows: 0` reads as "the corpus has nothing" and the
first real reply said exactly that. **Indexes are not the fix and are
counterproductive here**: a non-covering index converts one sequential scan into
hundreds of thousands of random reads, and a covering index that made one query
instant made another 270x slower by attracting a plan it was not written for.
On local disk no index earns its place - `schema.build` keeps only the four it
always had.


**One structural finding already.** Every model tool takes the whole board, so the
model re-types nine champion names per call. The loop fills an *omitted* board
(the turn's board is unambiguous) but **flags** a board the model retyped
differently - `transcription.mismatched`. Silently correcting it would hide the
most likely failure of the architecture. If mismatches are common, the fix is a
stateful server that holds the board, not a better prompt.

**The board is held, not passed (2026-11-11).** Every model tool used to take
nine champion names and trust the caller to retype them - a **silent** failure,
since a legal but wrong board returns a confident distribution, and the
`transcription.mismatched` flag could only detect it afterwards. A draft is now
an immutable move log (`draft_state.py`) with a store of them, and the tools take
an id: `get_state`, `clone_state`, `apply_move`, `undo_move`, `list_states`.
`predict_next_move`/`rank_candidates` accept `state="<id>"` or the board
arguments and resolve to the **same call**, so the two forms cannot drift.

Two measured facts made this cheap, and one was a bug:

- **The schedule is a constant in the corpus.** All 46,915 games are exactly 20
  moves, and the (kind, side) at each index depends only on `first_pick`: six
  bans, six picks, four bans, four picks. `to_move()` is now an index into that
  table instead of an inference from pick and ban counts - which is why a board
  with two bans and five picks used to get a plausible answer about a state that
  cannot occur. `test_the_schedule_matches_the_corpus` re-derives the table from
  the database.
- **`first_pick=1` means RED picks first**, the opposite of what the tool spec
  claimed. A model following the spec would have inverted it silently. The field
  is `red_first` now, and the spec says so.
- **The live board is read-only to the model.** A stateless tool cannot be
  corrupted; a stateful one can. `Store.apply` refuses `real` by construction and
  a test asserts a sandbox can never touch it. `Store.set_board` is the
  frontend's channel - it takes the UI's slots and side-grouped bans and builds
  the move log - and `_publish_state` calls it from every handler that carries a
  board, so `real` is never more than one request stale.
- **The UI's board and a state's log are different shapes**, and `log_from_board`
  is the conversion: bans are split by the schedule rather than halved (a flat
  list is blue's then red's, and mid-phase-1 halving puts every ban on the wrong
  side), and the log stops at the first empty pick slot because move 7 cannot
  exist without move 6. Checked by converting every prefix of eight real drafts
  back to the corpus.
- **The fearless pool is a third thing neither shape can say.** Champions picked
  in earlier games of the series are illegal and absent from this board, so a
  state of moves alone looks legal and is not - and the agent recommended one.
  `DraftState.unavailable` carries them, `board()` feeds every model tool,
  `head.next_move` masks before top-k, `apply_move` refuses with a sentence, and
  the rendered state lists them. Two guards because the failure was *reported*:
  the mask stops the suggestion, `apply` stops the simulation. Three holes made
  it possible - no field, no `unavailable` parameter on `rank_candidates` at all,
  and `_publish_state` dropped it - and `StubHead` accepting-and-ignoring the
  argument is why no test could see it. Measured after the fix: Lee Sin is the
  head's #1 at 15.4 % on a board where it is fearless, and 0 % of the answer.
  `fearless` absent from a publish means "unchanged", `[]` means "no pool".
- **`GET/POST /api/state` and `POST /api/agent`** expose the service to the
  panel; the panel's chat is the agent, with no toggle (§"The frontend was
  replaced", below). `/api/agent` needs `langgraph` (503 without it) and a key
  (200, `stop: "no client"`); no checkpoint, since the model tools are simply
  not offered without one. It reads the board from the store rather than the
  request.

**The ranking is not this component's job.** The heads are well trained and are
not being improved on this branch; the goal is the user/chat interface. A state
handle removes a class of *input* error and leaves judgment to the head, which is
where it belongs - so expect fewer silent nonsense answers, and the same picks.

**First hand-run (2026-11-11): fidelity 6/6, accuracy 2/6.** With no key
available, the tool layer was driven by hand (`handrun.py`): six val states,
stratified by where the head ranked the true pick (2 at rank 1, 2 at 2-5, 2 at
6-20), boards shown without the truth, answers written blind, scored after. **9
calls: 5 `predict_next_move`, 3 `rank_candidates`, 1 `sql`; `sql_share` 0.11,
zero refusals** - the high-level tier carried the turn, which is what the
two-tier design bet on.

Two numbers, and only one of them is this project's:

```
fidelity (answer == head's own #1): 6/6 = 100.0%   <- the interface
accuracy (answer == truth):         2/6 =  33.3%   <- the head's job
```

**Fidelity is the requirement and it was perfect.** Answering the head's own #1
on every board is correct behaviour even when that pick is wrong, and on four of
six it was. The sample is stratified to be hard, so 2/6 is not the head's rate
either (its own val top-1 is 20.3 %). One board had my own read right and the
head's wrong; answering the head anyway is right for a component that is a mouth,
not a second opinion. The one real lesson is tool discipline: a shortlist written
*before* seeing any distribution **omitted the head's actual #1** in 2 of 3
`rank_candidates` calls (Rumble, Gwen), which breaks fidelity - hence the prompt
now reads the head's ranking before comparing names. `handrun.py` recomputes the
head's #1 at scoring time, so the metric works on past runs and survives a
retrain.

**Not measured: whether a user finds it useful.** That needs the frontend and a
person, and it is the actual goal.

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

### The frontend was replaced, and the tool surface with it (2026-11-12/13)

The agent is now the *only* assistant surface. Removed from `frontend/`:
`chat.py` (rule-based chat), `rerank.py` and the `llm-*` slot, `draftscore.py`
and the draft-score panel, `tools.py` + `POST /api/tool`, `llm.py` and the
explain route, `roles.py`, `breakdown.html`/`breakdown.js`, and their JS and CSS.
What is left is four things: the board, `/api/next-move`, `/api/suggest` (the
role board) and `/api/agent`. **No numbers changed** - the deletions are the
surfaces, and the measurements that justified them are §8 above.

Three contract changes came with it:

- **`predict_lanes` → `predict_roles`, one-to-one per side.** Five picks, five
  roles: the assignment maximises summed log-likelihood over ≤120 injective
  assignments per side, from `data/champ_roles.json`. It **travels on the
  state** (`DraftState.roles`, with `p_lane` per slot), so `get_state` returns it
  without a second call and a sandbox carries the roles its moves imply.
- **`rank_candidates` is gone.** Comparing a named shortlist is the harnessed
  model's own job, so the tool that re-asked the head a question its top-k
  already answers was removed rather than kept for symmetry. The ablation in
  `eval_api_agent.py` is `{"sql", "soloq_matchup"}` (the free-form data tools,
  not just the pro-corpus query).
- **The state is engine-bound, and the server never hands a board to a model
  function.** `_publish_state` writes the board into the store; `Store.attach`
  rebinds the engine under every live state when a checkpoint loads. `/api/next-move`
  now **400s on a board the store refused** rather than answering from the last
  good one - a distribution for an illegal board is indistinguishable from a good
  one. `/api/suggest` may stay lenient and let the panel show `state_error`.

Tests: `frontend/` 47, `pro/model/agent/` 137. The panel's JS is checked against
the route table, so a retired route cannot be called from a stale panel.

### The reranking arm became a button (2026-11-13)

`POST /api/agent-suggest` + a **suggest next move** button in the agent panel.
It is the *agent* answering a next-move question, not `/api/next-move` with prose
attached: `suggest.py` shares the question, the explained prompt, the
`{"sql", "soloq_matchup"}` ablation and the 5-step / 8-call budget with
`eval_api_agent.py`, so the button
serves the arm §8 measured (19.08 % top-1, +0.73 pp over the head on 4,000
states) instead of a second experiment with the same name. **No new number was
measured** - no live provider was called for this change.

Three contract changes, all in the state rather than in a prompt:

- **The series travels on the draft state.** `picks`, `bans`, `first_pick`,
  patch and the fearless pool were already published; `series` (game n of m,
  per-game pick counts) is now rendered into `DraftState.render()` and reported
  by `get_state`, inherited by clones. Absent means *unchanged*, exactly like
  `fearless`: a role-board publish without the field must not turn game 3 into
  game 1. Every board route sends it, from one `seriesBody()` helper.
- **The answer is a move.** The final `PICK:` / `BAN:` line is parsed into a
  champion, whose legality (not on the board, not in the pool, right move kind)
  decides whether "play it" is offered. The panel never edits the board itself.
- **Fidelity is shown, not accuracy.** The head's own #1 on `real` is recomputed
  at serving time and shipped next to the parsed move (`matches`), because this
  component's job is the interface; accuracy belongs to the head (§8).

Degradation is explicit: no key -> `stop: "no client"` with the state and head
attached, no head -> `head: null`, complete draft -> no LLM call, refused board
-> 400. Tests: `frontend/` 53 (6 new), `pro/model/agent/` 149 (12 new), plus
`pro/model/llm/test_eval_api_agent.py`. One **pre-existing** failure remained in
`frontend/test_server.py::test_policy_default_is_the_ban_aware_arm`: the working
tree's `POLICY_DEFAULT` was already `nextpick-proonly-randomvalsplit-*` while
the server docstring, `frontend/README.md` and that test named
`nextpick-bans-aware-through-wlds-ig-tes`. The next change resolved it (below):
the randomvalsplit pair is the decision, and the docs and test now say so.

### Role names, model defaults and the page layout (2026-11-13)

**Role 4 is `SUPPORT`, not `UTILITY`.** The pro corpus stores role *ids*, and
`data/champ_roles.json` / `pipeline.ROLE_NAMES` only named them, so the rename
is display-only: no count and no assignment moved. `lanes.lane_name(4)`,
`ROLE_ROWS`, `app.js`'s `ROLES`, the CSS comment and `test_lanes.py` now agree.
The soloQ analysis store is deliberately **not** renamed - its `pos` column
holds Riot's raw strings in a built DuckDB, and changing the SQL literals would
silently drop every support row. Older entries in this file keep `UTILITY`
because they are records of measurements made under that spelling.

**The `undefined` badge was a type mismatch, not a missing role.** `/api/suggest`
keys roles by display name (`"SUPPORT"`) while `app.js` indexed its `ROLES`
array with that value as if it were a number. `ROLES["SUPPORT"]` printed
`undefined` under the champion, and `roleSlot["SUPPORT"]` was never found by
the loop that checked `roleSlot[4]`, so the correct role cell stayed empty.
`roleIndex()` now normalises either shape; `/api/champions` publishes the
server's own list and `init()` adopts it, so the two vocabularies cannot drift
again.

**Both model defaults are the `-randomvalsplit-` pair**
(`masking-` / `nextpick-proonly-randomvalsplit-through-wlds-ig-tes`). The
ban-aware arm and the 3-member ensemble remain selectable with
`--policy-ckpt`; their numbers are still not comparable. Browser model picking
is gone - no dropdowns, Load buttons or file upload - while `--ckpt`,
`--policy-ckpt` and `--no-policy` stay, and the `/api/load` + `/api/upload`
routes survive as a tested local scripting surface with no UI. The CLI help,
server docstring, `frontend/README.md`, `serve.sh` and the default test now
name the same two directories. **No new number was measured.**

**Layout.** `first pick` and `Reset draft` moved into `<main>` above the Blue /
Red panels; `ask the agent` sits directly below the head's suggestion panel;
the "Real draft" walkthrough starts collapsed behind a toggle row above where
it appears, and its comparison panel cannot reopen on its own. `.hidden` needed an explicit
`#wt-bar.hidden` rule because the ID selector otherwise wins.

### A soloQ matchup tool for the agent (2026-11-13)

A new tool, `soloq_matchup`, lets the agent answer two-champion pairwise
questions - *"does Ahri beat Zed in mid?"* - against the soloQ analytical store
(`solo_queue/analysis`, the DuckDB `cube2`), the same population the
`solo_queue/data analysis/exploration` folder works from. It is registered in
`session.py` alongside `sql` (the pro-corpus SQLite query) and dispatched through
the same `available()` / `_dispatch()` / ablation machinery, so it shows up in the
free-form "ask the agent" panel, the MCP server, and the tool log exactly like
every other tool.

Why a parameterised tool and not a second free-text SQL: the agent's escape
hatch is the *pro* corpus schema, not DuckDB, so reusing `sql` would force the
model to write DuckDB SQL against an unadvertised cube. `soloq_matchup` takes
`champion`, `partner`, `kind` (`vs` / `with`), optional `patch` and `role`, and
returns a precomputed 2-way aggregate plus the honesty the pairwise numbers need:

- **baseline** - the champion's own win rate with no partner filter over the
  same patch scope - so the *marginal* effect of the partner shows instead of
  being buried in champion strength;
- **`delta_win = observed.win − baseline.win`**, surfaced explicitly;
- **`reliable` flag on `n >= min_n`** (default 30) - the `pair_reliability`
  experiment showed pairwise deviations are mostly champion strength + noise, so
  a small cell is reported but flagged;
- the exact query is returned, like `sql`, so a wrong answer is debuggable;
- a `note` that the rates are **soloQ**, a different population from the pro
  drafts the head learned on - a complementary signal, never a substitute for
  `predict_next_move`.

Names and Riot ids both resolve (fuzzy, like `ChampIndex`); the connection is
opened lazily and cached (`_CON` / `_LOCK`), so importing or advertising the
tool opens no file; a missing or locked store is a returned `{"error": ...}`,
never a raised exception. It is offered only when the soloQ store is on disk
(`soloq_tool.available()`), and **ablated** from the one-click reranking button
(`suggest.ABLATE` now `{"sql", "soloq_matchup"}`) - that turn is a rerank of the
head, not a data dive - while the free-form panel and the MCP server keep both
data tools. `patch` is a legitimate scope filter here, so the
`test_no_tool_takes_a_board` guard now excludes it from the board-field set.

Tests: `frontend/` 61, `pro/model/agent/` 164 (incl. `test_soloq_tool.py`), plus
`pro/model/llm/test_eval_api_agent.py` and `test_eval_api_rerank.py`. **No new
model number was measured** - the tool reads a prebuilt cube and ablates from the
measured reranking arm.

### The agent suggestions column — a top-k rerank, always on (2026-11-13)

The agent chat now opens by **default** (the `#agent` panel is no longer `hidden`),
with an **agent features** header checkbox that hides both the chat and the new
rerank column, leaving only the base model's own suggestions and board updates.
Directly below the head's `#nextmove` list sits a second column,
**Agent suggestions**, filled by a new `POST /api/agent-rank` endpoint.

The endpoint reuses the *same* ablated, explained-prompt configuration as the
single-move button (`suggest.py`: `rank_question` / `rank_config` /
`parse_ranking`, `RANK_ABLATE = {"sql","soloq_matchup"}`, 5 steps / 8 calls) but
asks the agent to **order** the head's top-k rather than commit to one. The
server computes `head_top` (the head's own k=20 legal shortlist via
`_head_top_legal`), runs the agent, and `parse_ranking` reads back a list of
`NAME (NN%)` lines. The list is honest by construction: only champions the head
listed are accepted, any untouched slot is filled from `head_top`, each suggestion
carries `source` (`model` = the agent's own `(NN%)`, `head` = the head's
`p_head`), and `conf <= 0` is dropped. It always shows `k=8` names and never
invents one the head did not produce.

The column reruns (debounced) on every board change (`changed()` in `app.js` calls
`AgentPanel.scheduleAutoRank`) and after any chat turn - the chat is carried into
the rerank as `seed` context (a new optional `seed` arg on `loop.run`), so a
conversation that changes the agent's mind changes this list too. The chat log
records each board change ("Board changed: picked Ahri") and the agent's top
picks. **Clear chat** drops the conversation context and resets the column to the
agent's fresh top-k on the current board.

Frontend: `index.html` (panel shown by default, `#agentsuggest` below
`#nextmove`, toggle), `app.js` (`changed()` hook), `agent.js` (`scheduleAutoRank`,
`renderRank`, board-change logging, chat-driven rerank, clear reset). No key still
degrades cleanly to the head's own list with `source: "head"`.

Tests added: `test_suggest.py` (6 `parse_ranking` cases - order, Vi/Viktor
collision, head fallback, truncation, illegal, config) and `test_server.py` (5
`/api/agent-rank` HTTP cases - no-key outcome, live board wiring, conversation
seed, refused board, complete draft).

### The rerank arm measured against the head — a tie, at 500 games (2026-11-14)

`pro/model/agent/eval_rank_api.py` scores exactly what the column ships: the
`rank_question`/`rank_config`/`parse_ranking` arm with both data tools ablated,
on the `2022-full` split (`--no-pin`), one uniform prefix per game
(`prefix_mode="all"`, seeded), `deepseek-flash`, 8 workers, head = the shipped
`nextpick-proonly-nov13-k5` checkpoint. The only deviation from the browser
column is the list length: the eval asks for 10 names (the column asks for 8) so
that top-10 is measurable. The head-only baseline is the same `LockedHead` on
the same rows, so the comparison is paired.

| n = 500 | agent | head | delta | agent-only / head-only | p |
|---|---|---|---|---|---|
| top-1 | 19.80 % | 20.60 % | −0.80 pp | 3 / 7 | 0.344 |
| top-5 | 50.00 % | 49.60 % | +0.40 pp | 12 / 10 | 0.832 |
| top-10 | 68.40 % | 68.40 % | 0.00 pp | 6 / 6 | 1 |

Exact McNemar on the paired cells. By phase (top-5): ban1 32.7 vs 33.3 (2/3),
ban2 43.2 vs 36.8 (8/2, p 0.109), pick1 62.8 vs 65.5 (0/4, p 0.125), pick2 65.3
vs 64.4 (2/1). The one suggestive cell is ban2 and it is n = 95.

So the column re-orders the head's own shortlist and changes the answer in
~2 % of cases in either direction. It is honest and it is free of invention, but
it is not a lift — price it as a UI feature, not as accuracy. Report:
`pro/model/agent/reports/rank_api_nov13_500.json`.

**`deepseek-chat` on 100 of the same rows** (`--model deepseek-chat`, same seed,
so the first 100 games of the run above): top-1 16.00 vs 16.00 (0/0 discordant —
it echoed the head's #1 on **99 of 100** rows), top-5 42.0 vs 43.0 (1/2, p 1),
top-10 67.0 vs 63.0 (5/1, p 0.219). Paired against `deepseek-flash` on those
same 100 rows: top-1 +1/−0, top-5 +1/−5 (p 0.219), top-10 +3/−1 (p 0.625). It is
also *cheaper and faster*: 1.08M tokens over 270 calls, 3.9 s per example, no
step-budget hits — against flash's 19.6k tokens, 3.6 calls and 15.2 s. The
reasoning model spends its budget thinking and still does not move the answer.

So the arm is flat across two model families and two capability levels, which
locates the ceiling in the **prompt**, not the model: `rank_question` ends with
"if you trust the head, echo its order", and the stronger the instruction
follower, the more literally that is obeyed (chat 99/100 vs flash 92/100). A
binary keep-or-switch question with a calibrated threshold is the untested fix;
see the theory note in `.workbuddy-ai/memory/2026-11-14.md`.

Measured cost: 9.80M tokens (8.78M prompt / 1.01M completion) over 1,799 calls,
15.2 s per example, 3.6 calls each (histogram 2:124 3:211 4:20 5:56 6:65 7:24),
24/500 hit the 5-step budget, 499/500 answered in the agent's own words (1 fell
back to the head). 84 % of prompt tokens are cache hits, so at deepseek-flash's
¥0.02 / 1.00 / 4.00 per M (hit / miss / out) 500 examples is ≈ ¥5.6; the full
2,448-game test set is ≈ 48M tokens ≈ ¥28 (≈ ¥55 at peak rates) and ~78 min at
8 workers. `mcnemar` is the exact binomial, not the continuity-corrected
chi-square: the approximation returns p ≈ 0.48 for a 1-1 tie.

### Three ways to replace the adapter with a prompt — all three lose (2026-11-14)

The adapter is worth +14.5 pp over the head (34.61 vs 18.32 on its own slice)
and a prompted chat model is worth +4.0 (22.68 on 4,000 shared-val rows). Three
candidate explanations were tested; none closes the gap. **The ladder is the
result**: head 18.05 → a linear rule on the head's own numbers +0.5 → a chat
model that answers the original question +4.0 → the finetuned trunk +14.5.

**(a) A calibration gate, no LLM** (`pro/model/llm/gate.py`). Fit on 1,000
shared-val games (20,000 states), scored on the 200-game sample (4,000 states):

| scorer | params | top-1 | vs head | paired | p |
|---|---|---|---|---|---|
| head argmax | - | 18.05 % | - | - | - |
| switch-to-#2 iff p1−p2 < τ | 1 | 18.10 % | +0.05 | 40 / 38 | 0.910 |
| log p + rank offsets | 5 | 18.05 % | +0.00 | 0 / 0 | 1 |
| + lane legality | 7 | **18.55 %** | +0.50 | 26 / 6 | 0.0005 |
| + ban interactions | 9 | 18.55 % | +0.50 | 26 / 6 | 0.0005 |

The fitted weights say it plainly: `logp +0.86`, rank offsets −0.06 to −0.01,
`role_open +1.54`, `role_head1 0.00`. **The head's own numbers carry no
switching signal at all** — the five-parameter re-ranking changes zero of 4,000
answers — and the only thing that pays is whether a candidate fills a lane the
side has not filled (+0.6 on pick1, +1.4 on pick2, exactly 0 on both ban
phases). That is a real +0.5 pp and it is a rule, not a model.

Caveat on the baseline: the head reads 18.05 % here and 18.65 % in the API run
because the `2022` corpus cache has grown 39 drafts since (46,918 vs 46,879), so
the seeded 200-game sample is not literally the same games. The gap the gate
fails to close is unaffected.

**(b) A keep-or-switch question** (`pro/model/agent/eval_keep_switch.py`), same
model, one call per row, the board plus the head's top-5 and its measured
accuracy: "KEEP X or SWITCH <champion>". 200 test games, τ tuned on half and
reported on the other half (n = 100):

| arm | top-1 | vs head | paired | p |
|---|---|---|---|---|
| head argmax | 23.00 % | - | - | - |
| as answered | 16.00 % | −7.00 | 2 / 9 | 0.065 |
| with the gate (τ 0.135) | 18.00 % | −5.00 | 1 / 6 | 0.125 |

It switches on 50 of 100 rows, and of the 36 switches the gate lets through
**1 is right** (precision 2.8 %). Asked to adjudicate, the model overrides on
the rows where the head is least sure and is almost never right; asked to rank
(the shipped arm), it echoes. Neither behaviour is calibrated, and the two
failure modes are opposite.

**(c) The same question with twenty candidates instead of five** (`--k 20`):
raw 14.00 % (5 / 14, p 0.064), gated 21.00 % with τ 0.02 and 13 switches kept
(precision 7.7 %). Widening the list makes it override more (63 of 100) and
score worse; the gate's only job becomes suppression.

So: not calibration (0.5 pp), not the question format (−5 pp), not list width
(−2 pp). The adapter's advantage is a trunk fitted on 95k labelled states, and
the +4 pp a chat model does have comes from **answering the original question**
("name the best move", the v4 text) rather than from adjudicating the head —
which is why the shipped rank column, which asks it to reorder, measures a tie.
If the agent column stays, it should ask the single-move question; if accuracy
is the goal, it should not be sold as one.

Reports: `pro/model/llm/gate_report.json`,
`pro/model/agent/reports/keep_switch_chat_200{,_k20}.json`. Tests:
`pro/model/llm/test_gate.py`, `pro/model/agent/test_eval_keep_switch.py` (12).

### Can the local 4B do the agent's job? No — and the answer position is why (2026-11-14)

The hosted arms cap at +4 pp and the adapter is +16.0, which invites the obvious
product question: run the finetuned model locally, drop the API, and serve the
agent's panels from it. `pro/model/llm/eval_agent_local.py` answers it. Base and
adapter are the *same* trunk (21.2 M LoRA weights of difference), the same
shared-val rows, the same soft block spliced exactly as training does it,
greedy, 4-bit on the 4060.

**What shipping the model would buy, on one slice.** 4,000 held-out states (200
games x 20 moves), v4 prompt, k=5, the same head for every arm. `exact` is the
champion-level "did the pipeline name the pro's move", which is the end-to-end
number: out-of-shortlist states count as misses.

| arm | top-1 (group) | exact | conditional on being in the 5 | vs head |
|---|---|---|---|---|
| head only (ships today) | 18.32 % | 18.32 % | 35.8 % | - |
| base trunk + mask | 19.57 % | 18.70 % | 36.5 % | +0.4 |
| + full2 adapter | 31.90 % | 29.75 % | 58.1 % | +11.4 |
| **+ rerank5 fitting (the shipped reranker)** | **34.88 %** | **34.27 %** | **67.0 %** | **+16.0** |

Ceiling: the head's top-5 contains the pro's move **51.18 %** of the time, and
nothing downstream can be right on the other 48.82 %. So the end-to-end number
decomposes as 51.18 % x conversion, and conversion is the whole story: **the base
trunk converts 36.5 % of its shortlist - 0.7 pp above the head and 16 pp above
the 20 % a random pick among five gets, i.e. it does not choose at all - while
the shipped reranker converts 67.0 %.** Three readings of "35 %" are all correct
and only the third is interesting: the pipeline is right 35 % of the time, it is
offered an answer 51 % of the time, and it picks the offered answer 2 times in 3.
Quoting the first without the second is what made this look like a draft
predictor rather than a chooser with a recall ceiling.

**Correction (2026-11-15): the table's "head" row is a different checkpoint on a
different sample, and the 34.27 % is role-conditional.** Two independent
problems, both measured:

* **The baseline is not the shipped head.** The 18.32 % is `C.BASELINE_VAL`,
  which is `nextpick-proonly-**randomvalsplit**-…/test_report.json` over
  **95,160** states. The head that actually proposes the reranker's candidates
  is `nextpick-bans-aware-through-wlds-ig-tes` (`rerank.HEAD_CKPT`), and the
  shipped frontend head is `nextpick-proonly-nov13-k5-…`. Three checkpoints, and
  the table names one of them as "the head". Measured on the same 4,000 rows
  (`pro/model/llm/eval_served_next_move.py`, through `predict_next_move`):
  bans-aware **18.25 %** raw / 18.45 % served - so 18.32 was right for *that*
  checkpoint, on this split, and it is the correct row for the ladder. It is not
  the shipped head, whose served top-1 on these rows is **24.90 %** (raw 25.40)
  - and that number is **not held out**, because the `-nov13-` head trained on a
  different corpus and a different split (fractional within-patch tail), so some
  of these rows may be its training data. No clean shared-split comparison of
  the shipped head against the reranker exists yet.
* **The prompt names the true lane.** v4's task line is "Choose the best Blue
  Side pick **for Top lane**", and `target_role` in the training rows is the
  lane of the pro's own pick. Delete the clause and the same weights score
  **20.85 %** on the same 4,000 rows (`eval_rerank5_nolane4k.json`): 13.4 of the
  16 points are the lane, not drafting skill. The next-move panel asks the
  no-lane question - it cannot name a lane, that is what "the next move" means -
  so the panel's arm is the 20.85 % one, on candidates that were *not*
  legality-filtered. Against the 18.25 % of the proposer it is +2.6 pp; against
  the served shipped head it is below it. At ~2 s/board.

* **The role-conditioned question is not the reranker's to win.** The comparator
  for a lane-named query is not the head, it is §7's masker, which already
  answers that question at 2.5 M parameters on CPU. `masking-proonly` scores
  **35.7 %** top-1 on masked completion (70 slots, `--mask-mode slot`) against
  the reranker's **34.27 %** on 4,000 rows - and the masker is the one being
  handed more: `masked_ranks_slot` hides a single slot and keeps the other nine
  *including the picks after it*, plus the full true `role_ids` vector, while
  the reranker sees the prefix only. A tie under more information is a loss for
  the 4.2 B model. Two caveats, both against over-reading the 35.7 %: it is 25
  of 70 queries from 7 games (CI [25.5 %, 47.4 %]), and it is the best-case
  protocol - the same family reports **20.2 %** on 4,193 randomly-masked test
  slots and **21.3 %** on 8,069 val slots, the 14 pp gap being protocol (one
  hole with full context vs ~1.5). So the masker's number is a band, 20-36 %,
  and the reranker's 34.27 % lands at the top of a band a CPU model of 2.5 M
  params also reaches. Neither number ranks the two models; what they jointly
  establish is that the reranker's headline is **not a capability the repo
  lacked**.

What that leaves: the +16.0 pp headline is the gap between two different
checkpoints on two different questions, and it should not be quoted. What is
real is (a) the mask/adapter interaction below, (b) the +3.0 from rerank fitting
with the mask held fixed, and (c) the ordering gain on a **lane-conditioned**
question - which is the role board's question rather than the panel's, and on
which the masker is the incumbent rather than the head.
`pro/model/llm/eval_role_cells.py` measures that surface (800 eligible
pick-phase states, head vs reranker paired, both through `predict_in_lane`);
**it has not completed**, so the role-board gain is currently unmeasured and the
frontend serves the head by default with `--rerank` as an opt-in.

**First, where the 30 % comes from — the emphasis in §"Three ways" was wrong, and
so was the first version of this section.** The +13 pp is not the format. The
missing cell was *the base trunk with the same mask*, which nobody had run. Same
4,000 llmval rows, same prompt (v4), same k=5, same head, so the only variable
is the adapter:

| | no shortlist, free generation | masked to the head's 5 |
|---|---|---|
| base trunk + warmup projector | not run (head: 18.32) | **19.58 %** (exact 18.70) |
| + full2 adapter (46,915 drafts) | 19.13 % | **31.90 %** (exact 29.75) |
| `deepseek-chat`, shortlist in prose | 22.68 % (its own head: 18.65) | n/a - no logits access |

Read the columns: **the mask alone is worth +1.3 pp** (19.58 vs the head's
18.32), and **the adapter alone is worth +0.0** (19.13, i.e. the head's own
19.14). Together they are +13.6. This is an interaction, not a sum, and it kills
the reading that the format carries the gain - the format carries almost none of
it. `evaluate_llm --rerank 5` with and without `--adapter`, reports in
`pro/model/llm/eval_{base,full2}_mask4k.json`.

The mechanism is visible in the action split, and it is the cleanest single
number in this file: **with the mask, the base trunk scores 8.6 % on bans** -
below the 20 % a random pick among five would get. It is not choosing at all; it
is copying the head's order, whose ban #1 is right 8.6 % of the time. The
adapter scores 18.5 % on the same rows and 45.3 % on picks (base 30.5 %). The
adapter is the only arm in the project that *chooses* rather than echoes, and
choosing is worth 12 pp.

The one number that is purely the weights is the last step of the ladder:
31.90 -> 34.88 (+3.0) from 2 epochs of rerank fitting, with the mask held fixed.
(Earlier revisions of this section compared 31.90 against a 35.31 measured on a
*different* 478-game slice; the ceiling was 51.85 % there against 51.18 % here,
so the two are close but were not the same experiment. All four rows above are
one slice: `eval_{base,full2,rerank5}_mask4k.json`.)

**The adapter has not forgotten how to talk.** The first read of this experiment
said it had — every answer was "A pro team uses a lane swap to get a xayah
lane." — but that was the prompt: the v2/v4 training text has no assistant turn,
so a completion-format prompt makes the model continue a list instead of
answering a question. With `<|im_start|>assistant` in the prompt, 10 of 12
answers are coherent, board-reading prose ("A lane swap is when two champions
from different lanes trade positions..."), 7 of 12 open a thinking block, and 0
are empty. What the adapter loses is **task shape, not knowledge**: on the two
board-anchored questions it answered with a bare champion (`Zeri`) or a table
fragment (`blue: 100 % win rate`), i.e. it read "board + question" as "emit a
move". That is exactly what a loss on answer tokens only would do, and it is not
amnesia. The base trunk spends its whole budget inside `<think>` on all 12.

**The rank arm is the tie again, from a fourth direction.** n = 60, same rows,
head computed on the same rows so the comparison is paired:

| arm | format | top-1 | top-5 | vs head | paired | empty | invented | s/row |
|---|---|---|---|---|---|---|---|---|
| adapter | chat | 21.7 % | 60.0 % | ±0.0 | **0 / 0** | 0 | 0 | 20.6 |
| adapter | completion | 18.3 % | 56.7 % | −3.3 | 1 / 3 | 25 | 4 rows | 5.2 |
| head | - | 21.7 % | 60.0 % | - | - | - | - | - |

The chat-format arm reproduces the head's top-1 on **all 60 rows** (zero
discordant pairs) and its top-5 *order* on 58 of 60. So: head 21.7, base 25.0
(= head), adapter 21.7 (= head), `deepseek-chat` 22.68, all within noise of each
other, and the only arm that ever beats it is the masked 5-way one at 35.31.
Five prompt designs and four models now agree, which is the strongest form this
negative result can take.

The completion-format arm is the cautionary one: it *looks* like a model that
tried and failed (−3.3 pp, 4 rows naming champions the head never listed), but
42 % of its answers were empty because the prompt had no `Answer:` cue. An empty
answer is not a wrong answer, and averaging them together understates the model
and overstates the head.

**A parser bug this exposed.** `suggest.parse_ranking` matched `(\d+)%`, so the
adapter's `15.6%` was invisible: its whole list was discarded and the head's
order substituted, with `parse: "head"` as the only trace. The hosted arms write
integers, so no earlier number is affected. Fixed to accept one decimal, with a
test (`test_parse_ranking_accepts_a_decimal_confidence`).

**Verdict.** Do not replace the API with the 4B for the agent column. The rank
panel would ship the head's own list at 5-20 s per row on a 4060, which is the
head's answer with the head's latency budget destroyed. The chat panel is the
one place a local model is viable — the base trunk explains a board well and
costs nothing per token — but it has no tools, no multi-turn state and no
refusal, and those are the agent's actual value.

Reports: `pro/model/llm/eval_agent_local_{chat,rank}_*.json`. Harness:
`pro/model/llm/eval_agent_local.py`.

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
docker compose run --rm dev python pro/model/archive/test_split_legacy/eval_test_split.py
docker compose run --rm dev python pro/model/archive/test_split_legacy/render_test_split.py
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
Green today (2026-11-13, after the frontend replacement):
`pytest solo_queue/model/tests` 52 passed 1 skipped;
`pytest solo_queue/model/finetune` 25 passed;
`pytest solo_queue/model/policy` 69 passed, 1 failed
(`test_swap_sides_flips_econ_team_pairs`, pre-existing);
`pytest pro/model/proonly pro/model/bans pro/model/test_split` 47 passed 1 skipped;
`pytest "data/pro play"` 41 passed;
`pytest pro/model/agent` 137 passed;
`pytest frontend` 47 passed (the suite was rewritten with the surface: the 76
count included the chat, rerank and draft-score tests, and those modules are
gone). `pro/model/llm` still needs `bash pro/model/llm/run.sh python -m pytest
pro/model/llm/test_llm_prompt.py -q`, since it imports `transformers`.

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
