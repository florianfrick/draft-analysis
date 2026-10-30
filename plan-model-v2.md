# Model v2 — plan for better draft prediction & suggestions

Grounded in five measurements run against the current data (scripts in
`soloq data analysis/experiments/`). The short version: **the value head is
dead, bans are the weakest surface, and the pairwise win-rate matrix has no
signal left once champion strength is removed** — so the biggest wins are in
training/eval and features, not in a new encoder.

---

## TL;DR

| Idea | Verdict | Why |
|---|---|---|
| Pairwise win-rates → attention bias | **Don't build it** | Residual signal is ~all sampling noise (synergy) or ~0.8pp (counters). Measured AUC: 0.545 → 0.535 soloQ, 0.551 → 0.535 pro. Low-rank denoised version: +0.002 at rank 1, negative above. |
| RL / value from pairwise win-rates | **Blocked, and the wrong source** | Value head is at chance (AUC 0.497, Brier pinned at the constant). Build value from *game outcomes*; pairwise cannot supply it. |
| RL as policy-gradient / self-play | **Wrong framing** | Draft is a finite zero-sum game with known rules. You already have a policy by imitation. What's missing is a value + search, not a gradient method. |
| LLM as the predictor | **No** | 46k examples, uncalibrated, no legality guarantee, too slow for search. |
| LLM as prior / teacher / explainer | **Yes — highest leverage** | Kit-attribute embeddings fix new-champion and new-patch cold start; retrieval + explanation is the coach-facing win. |

Priority order: **P0 value head → P1 eval protocol → P2 ban features → P3 LLM
prior → P4 search → P5 (skip) pairwise bias.**

---

## 1. Evidence

### 1.1 Pairwise features do not beat champion strength

Logistic regression on P(blue wins), nested feature sets, held-out test.
Singles = one weight per champion (+1 blue, −1 red). Synergy = same-team pairs,
counter = cross-team pairs.

**SoloQ** (patch 16.20, queues 420/440, 401,512 games, 80/20 split, base 0.478):

| Features | AUC | log-loss | n feats |
|---|---|---|---|
| A singles | **0.5449** | 0.6891 | 173 |
| B + synergy | 0.5353 | 0.6966 | 15,051 |
| C + synergy + counter | 0.5436 | 0.7045 | 29,929 |

**Pro** (Oracle's Elixir 2022→2026, 47,657 games, time split: train ≤ 2026-01-15,
test 9,532, base 0.536):

| Features | AUC | log-loss | n feats |
|---|---|---|---|
| A singles | **0.5507** | 0.6883 | 173 |
| B + synergy + counter | 0.5346 | 0.7052 | 29,929 |
| C + synergy + counter + team identity | **0.5715** | 0.6907 | 31,309 |
| constant | 0.5000 | 0.6906 | — |

Two things to take from this. Pairwise structure *hurts* in both domains — it
adds variance, not signal. And in pro, **team identity is worth +0.021 AUC**,
more than the entire pairwise block. Who is playing matters more than how the
champions interact.

### 1.2 The pairwise signal is mostly sampling noise

Random half-split of soloQ games; per half compute pair win rate, subtract the
additive baseline (champ strength + partner strength), correlate the two halves'
residuals. If a residual is real it survives the split; if it is noise it does
not.

| Pair type | cells | median n | raw r | **residual r** | residual sd | binomial sd |
|---|---|---|---|---|---|---|
| synergy (with) | 27,778 | ~30 | +0.168 | **+0.035** | 0.0434 | 0.0390 |
| synergy (with) | 8,486 | ~300 | +0.404 | **+0.090** | 0.0228 | 0.0220 |
| counter (vs) | 14,807 | ~30 | +0.240 | **+0.087** | 0.0381 | 0.0346 |
| counter (vs) | 5,580 | ~300 | +0.454 | **+0.178** | 0.0233 | 0.0218 |

The additive baseline explains **R² = 0.37** of pair win-rate variance. What is
left over has a standard deviation almost exactly equal to the binomial noise
floor — for synergy at n≥300, 0.0228 observed vs 0.0220 expected from pure
sampling. **Synergy is 100% noise.** Counters leave ~0.008 win-rate sd of real
structure, i.e. a genuine strong counter is worth about **1.6 percentage points
(2 sd)**.

For scale: champion strength itself has sd **1.28pp** across the patch
(p5–p95 = 47.9%–51.9%). So a real counter is worth roughly one tier of champion
strength, and it cannot be estimated at n≈300.

### 1.3 A denoised low-rank bias adds nothing

The fair version of "put the pairwise matrix in the attention": fit the
residual matrix on train, truncated SVD to rank k, use the reconstruction as a
fixed per-game bias, then logistic on [singles, bias].

| Bias | none | rank 1 | rank 4 | rank 8 | rank 16 | rank 32 |
|---|---|---|---|---|---|---|
| AUC | 0.5483 | **0.5506** | 0.5491 | 0.5469 | 0.5432 | 0.5397 |

Rank 1 is +0.002 AUC — indistinguishable from noise. Everything above rank 4 is
worse than no bias at all.

### 1.4 The value head never learns

From `ckpt/scheduled_policy-3.log`, 100 epochs × 80 steps = 8,000 optimizer steps:

```
val ep   5: win_auc 0.5008  brier 0.2507
val ep  50: win_auc 0.4938  brier 0.2513
val ep 100: win_auc 0.4939  brier 0.2513
TEST       win_auc 0.5016  brier 0.2506
```

A constant predictor at the base rate 0.542 scores Brier **0.24824**. The head
sits at 0.2506–0.2514 for the entire run — *slightly worse than predicting the
prior and refusing to move*. This is not "weak signal": the same pro data gives
AUC 0.5507 with 173 parameters. The head is structurally disconnected from the
signal. **Every RL / MCTS / win-aware-suggestion plan is blocked on this.**

### 1.5 Ban prediction is the weakest surface

| Metric | val (ep 100) | test |
|---|---|---|
| top5 (all moves) | 0.3366 | 0.2258 |
| pick_top5 | 0.3870 | — |
| ban_top5 | 0.2862 | — |

Bans are 12 of 20 moves. Your own README records a patch lookup table at
**37.4% top-5 on move 0** against **12.3%** for the model — a frequency table
beating the network on 60% of all moves. Separately, val→test collapses from
0.337 to 0.226, so the checkpoint selection is optimizing a val that does not
represent deployment.

---

## 2. Verdicts on your three ideas

### 2.1 Pairwise win-rates as an attention mechanism

The mechanism is easy and I would have liked it to work: attention logits are
`(W_q h_i)·(W_k h_j)/√d`, so adding `λ·W[champ_i, champ_j]` inside the
intra-team block is a one-line change and the cross-team block gives you the
counter term for free.

But the premise fails, and there is a structural reason on top of the numbers:
**attention logits are already a learned bilinear form over pairs, i.e. a
low-rank pairwise matrix with rank ≤ d (256).** The architecture already spans
the representable part of W. A fixed bias can only help if W holds reliable
structure outside that span — and §1.2/§1.3 show the reliable part is rank 1–2
and worth 0.002 AUC.

Keep the pairwise tables for what they are good at: the explorer UI, diagnostics,
and as an **evaluation target** ("does the model reproduce the known counters?").
Do not feed them to the encoder.

### 2.2 RL

Split the idea in two.

**Value function from pairwise win-rates — no.** §1.2 settles it: the residual
is noise. Build value from game outcomes, which you have for 47.7k pro games.

**RL as the training algorithm — wrong framing.** Draft is a finite,
perfect-information, zero-sum game with known legal moves. You do not need
policy gradients to learn a policy; you already learn one by imitation, which is
cheap and well-conditioned. What you lack is a *value function* to evaluate
leaves. That is the AlphaZero decomposition — imitation policy + value +
search — and the binding constraint is value accuracy, not the RL algorithm.

Also note what "self-play" would mean here: the environment is the opponent's
drafting policy, which you can only learn from data. So this is offline RL at
best, over 46k×20 states, and it would largely relearn the behavior policy. Not
worth it yet.

**What to do instead:** once value beats 0.55, run depth-2/3 alpha-beta with the
policy as move ordering prior and value at the leaves. Branching is ~150 legal
champions, so depth 2 is ~22k batched evaluations — milliseconds on a GPU. That
is where the benefit people expect from "RL" actually lives.

### 2.3 LLM

Not as the predictor. 46k training examples, no calibration, no legality
guarantee, and far too slow to sit inside a search loop.

As a **prior, teacher, and explainer** it is the highest-leverage item on this
list, because it attacks the two weaknesses your own `assumptions.md` already
names: embeddings do not transfer across patches, and new champions have no
data.

---

## 3. Work plan

### P0 — Prove the value head can learn (blocking, ~half a day)

Do this before anything else. Diagnostic ladder, cheapest first:

1. **Linear probe.** Freeze the encoder, cache `pooled z` + `state` for every
   val/test prefix, fit logistic regression on `z → win`. Probe > 0.53 ⇒ the
   head or its training is at fault. Probe ≈ 0.50 ⇒ the pooled representation is
   at fault. This one number decides which of steps 2–3 you do.
2. **If the representation is at fault.** The value head sees
   `mean(5 slot vectors)` per team, which averages away champion identity. Try
   attention pooling (PMA) or `concat(max, mean)`, and verify `pad_mask` on
   partial drafts — an all-PAD team yields a zero vector by construction.
3. **If training is at fault.** `--value-weight 0.5` against a policy CE of
   ~4.4 gives value a small share of the gradient; `weight_decay 0.05` is also
   heavy for a 2-layer head. Run value-only for 5 epochs with weight decay off
   and confirm it can fit at all.
4. **Shuffle-label control.** Train with labels permuted. If Brier still lands
   at 0.2510, the head is emitting the prior regardless of input — that is a
   wiring bug, not a capacity problem.
5. **Add the strength skip.** Give the value head an explicit additive feature:
   `Σ logit_blue − Σ logit_red` from a learned 173-vector. That guarantees it
   matches the linear baseline (0.55) instead of sitting at 0.50, and gives the
   rest of the network something to improve on.

**Gate:** test AUC ≥ 0.55 and Brier < 0.2482 (the constant) before P4.

**RESULT (done) — P0 passes; the gate was met at epoch 15 of 40.**

Ladder outcome: step 1 (linear probe) localised the fault to the *head and its
training*, not the representation, then step 2 (PMA) + a warmup ramp fixed it.
`ckpt/run_20261029_133104_v2-D512-lr3e4/best_value.pt`, 19,217 held-out
soloQ drafts:

| measure | AUC | note |
|---|---|---|
| old encoder (run1, D=256), mean-pool probe | 0.5329 | the number to beat |
| new encoder, mean-pool probe | 0.5511 | representation improved |
| new encoder, PMA probe | 0.5529 | |
| new encoder, trained head | **0.5601** | Brier 0.24669, skill +0.0098 |
| reference: logistic on soloQ pairwise singles | 0.5449 | hand-built baseline |
| reference: logistic on pro singles | 0.5507 | pro ceiling |

So the pooled representation now beats the hand-built logistic model on pairwise
features, which is the property P4 search needs.

Three findings that matter for the next run:

1. **The value loss must be ramped in.** At 700 steps the probe read 0.4996 and
   the head sat exactly at the prior; the same head on the mature frozen
   encoder hit 0.5396 in 700 steps. `--win-warmup-epochs 5` now does this.
2. **And cut out again.** After ~epoch 19 the head memorises outcome noise:
   train win BCE 0.6842 → 0.4359 while val Brier 0.2467 → 0.3857 and AUC
   0.5601 → 0.5206. `--win-patience 8` now freezes the value loss there and
   leaves the encoder to finish on the masked task. `best_value.pt` keeps the
   good head; **`best.pt`'s value head is the overfitted one — do not use it.**
3. **LR does not transfer across width.** lr 1e-3 (fine for the 6M D=256 run1)
   diverged the 31.7M D=512 model at step ~660: role CE 0.30 → 1.71, champ CE
   4.06 → 4.97. The block is Post-LN (`x = norm1(q + a)`), which needs both a
   lower peak and a longer warmup. The working setting is `--lr 3e-4
   --warmup-frac 0.05`.

**Rerun with `--win-patience 8`** (`ckpt/run_20261029_153912_v2-D512-p8`): the
guard fired at epoch 19 and held to 40. Value result unchanged (head 0.5597,
PMA probe 0.5521 vs 0.5601/0.5529), masked result slightly better (`best.pt`
top-5 0.2738 vs 0.2721) **and** its value head is now usable (Brier 0.2519 vs
0.3727). Two things it taught:

- **The freeze must latch.** The frozen head's AUC drifts up ~1e-4 as the
  encoder moves; letting that reset the stall counter switched the loss back on
  at epoch 26 and AUC fell 0.5597 → 0.5467. Now latched, with `--win-freeze` as
  the manual override.
- **The objectives trade off, so pick the checkpoint per purpose.** Cutting the
  value loss stops pushing outcome information into the pooled vector and it
  decays: a fresh probe on the epoch-33 encoder reads 0.5444 against 0.5521 at
  epoch 26. Late checkpoint = best masked encoder; early = best value encoder.
  **Load `best_value.pt` for P4.**

### P1 — Fix the eval protocol (~half a day)

Val 0.337 → test 0.226 means checkpoint selection is misaligned. Decide which
condition you ship under:

- If deployment is *next patch*, test must be the future and val must be a
  held-out **time window inside train** (e.g. last 8 weeks), never a random
  slice of the same era.
- Report metrics **per patch** and per move index, not pooled. Pooled numbers
  hide exactly the drift that matters.
- Keep the popularity baseline in every table. `ban_top5` vs a patch frequency
  table is the number that tells you whether the model is adding anything.

### P2 — Ban features (~1–2 days, largest immediate metric gain)

Bans are 60% of moves and the weakest predictions.

1. **Patch×champ frequency as an explicit input.** A `[n_patches, C]`
   log-frequency tensor from train counts, gathered per batch and concatenated
   into the policy head. The patch embedding currently has to *infer* the meta
   from 8–64 dims; give it the counts directly.
2. **Series history.** You have `unavailable_mask`, but not *which side* picked
   what. Add `unavailable_by_side [B, 2, C]` so the model can represent "they
   first-picked X, so X is gone for us, and we banned Y last game". Fearless
   depth is the documented main 2025+ shift and this is the feature that
   expresses it.
3. **Recent-move context.** You deliberately dropped order for comp value —
   correct — but *next-move prediction* is order-sensitive (a ban after a
   first-pick answers a threat). Add the last 2–4 moves as a small sequence
   context alongside the state embedding. Cheap, and it targets the
   ban-after-pick case directly.

### P3 — LLM as prior and teacher (~3–5 days)

1. **Kit-attribute embeddings (do this first).** Have an LLM tag all 173
   champions on axes that matter for drafting — engage, peel, poke, dive,
   scaling, lane priority, damage type, hard CC, mobility, range, objective
   speed. You get a `[173, K]` matrix. Use it three ways: initialize or
   concatenate into `DraftSlotEmbedding`; give a **new champion** a row from its
   kit instead of a random embedding; and give the model a compositional prior
   that transfers across patches. This is the single best answer to "embeddings
   do not transfer across patches".
2. **Soft-label distillation.** Take a few thousand draft states, have an LLM
   rank candidates with rationale, use the ranking as a KL / ListNet target
   alongside hard CE. This directly fixes the "hard CE gives no partial credit"
   limitation already noted in `assumptions.md`. A few thousand states is
   enough; do not try to label all 46k×20.
3. **Retrieval + explanation in the frontend.** You already have bitmask
   nearest-neighbour over 46k pro drafts in `store.py`. Retrieve the k most
   similar historical drafts plus the patch notes, and have the LLM explain the
   recommendation and the opponent's likely idea. Zero training, immediately
   useful to a coach, and it is the honest way to "learn from the few pro
   examples" — retrieval, not weights.
4. **Patch-day cold start.** LLM reads the patch notes → champion priority
   prior. Bridges the reset between patches, where you currently have nothing.

### P4 — Search, once value works (~2–3 days)

- Depth-2/3 alpha-beta, policy for move ordering, value at leaves, legality from
  the existing `-inf` masks. Report the *value delta* of the recommended move
  ("+1.8pp win prob") — this is the product change from "what pros pick" to
  "what wins".
- Only then consider PUCT with a small simulation budget. Full MCTS is
  unnecessary at depth 20 with branching 150.
- Do not start offline RL until value clears 0.55 and you have either more data
  or a credible opponent model.

### P5 — Pairwise attention bias: skip

If you want the negative result in the repo for future-you, the mechanism is:

```python
# inside the intra-team block: synergy bias
bias = W_syn[champ_i, champ_j]        # [B, 5, 5], additive on attn logits
attn = softmax(q @ k.T / sqrt(d) + lam * bias)
# cross-team block: same with W_ctr
```

Ship it behind a flag only if §1.3 changes. It will not.

---

## 4. Keep unchanged

The core encoder is sound and I would not touch it: Siamese set attention with
no positional embeddings (correct — order is irrelevant to comp value),
permutation invariance, masked-draft pretraining, `-inf` legality masking,
side-swap augmentation. `assumptions.md` §3–4 is right about all of this.

The one place the "no order" choice is wrong is next-move prediction, not comp
value — hence the small sequence context in P2.3, not a positional encoding on
the slot set.

---

## 5. Verification protocol

Every change reports, on a fixed time-split test set:

| Target | Metric | Baseline to beat |
|---|---|---|
| next move | top1/5/10, per move index, per patch | patch frequency table (37.4% top-5 at move 0) |
| bans | ban_top5 | same table restricted to ban moves |
| value | AUC, Brier, log-loss, calibration curve | 173-param logistic (0.5507) and constant Brier (0.2482) |
| econ | `econ_skill` (R²) | 0 — currently −0.15, also broken |

A change that does not beat the relevant baseline on the test split does not
ship, regardless of val.

---

## 6. Files added

- `experiments/pairwise_signal.py` — nested logistic A/B/C, soloQ.
- `experiments/pair_reliability.py` — split-half residual reliability.
- `experiments/pair_bias_rank.py` — denoised low-rank bias sweep.
- `experiments/pro_ceiling.py` — pro time-split ceiling + team-identity control.

Run any of them with `python "soloq data analysis/experiments/<name>.py"`.
Each writes a JSON next to itself.
