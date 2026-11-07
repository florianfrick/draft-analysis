# Win prediction from a completed draft — handoff to `main`

> **Where this came from.** Written on the `exp/win-prediction` branch, which is
> where the code, the 19-section README and the report JSONs referenced below
> live. Nothing in this file describes `main`; it is here so the conclusion
> survives if that branch is dropped.

**Bottom line.** The question is answerable and the code is sound, but the answer
is not worth building on. The best model reaches **0.59624 test log-loss against
a 0.69273 constant**, and almost all of that is *who is playing*, not the draft:
player-strength ratings alone get 0.60010. The draft itself is worth **0.0039
nats**, and when asked directly whether it can rank one champion above another it
returns **0.5002 [0.4974, 0.5028]** — chance. So: **do not add a win-probability
stage to the suggestion pipeline.** The existing `masking-*` policy models answer
a better question ("what would a pro pick here") and this branch has nothing to
add to them.

What is reusable: the strength-feature machinery in `data.py`, the split and
evaluation protocol, and the negative results below.

---

## 1. What was built

`pro/model/winpred/`, self-contained (nothing outside the folder changed,
`STATUS.md` deliberately untouched so the branch merges or drops as a unit).
Input is a **finished 10-pick board** — no bans, no prefixes, no masking, no
partial drafts. Output is one logit for blue.

- Data: 46,879 pro drafts (2022→2026-11, 106 patches) and 384,356 soloQ drafts.
  Series-grouped 80/10/10, seed 20261105 for the split *and* the init.
- Model: per-side 5-token transformer (intra-team blocks, cross-team attention,
  mean pool) → MLP head over `(h_blue, h_red, h_blue − h_red)` + a scalar side
  bias. 2.5–5.9 M parameters.
- Training: cross-entropy, Adam, early stopping on **val log-loss**. 50 tests.

## 2. The best model, and why it wins

| model | test log-loss | AUC | acc |
|---|---|---|---|
| side prior (always guess blue, 53.05 %) | 0.69273 | 0.5000 | 0.5197 |
| draft only, d384 encoder | 0.68775 | 0.5552 | 0.5412 |
| team win-rate, smoothed | 0.64260 | 0.6774 | 0.6264 |
| **player ratings + encoder, end to end** | **0.60323** | **0.7376** | 0.6716 |
| **player ratings + draft logit, 2 coefficients on val** | **0.59624** | **0.7410** | 0.6751 |

Paired on the same 4,762 test rows: **−0.0845 [−0.0977, −0.0719]** against
draft-only, **−0.0394 [−0.0502, −0.0283]** against the team-rate model.

**Why it wins.** The outcome is dominated by execution and by the ten players, so
the model that knows the players wins. Three decisions mattered:

1. **Rate players, not teams.** A team tag is a roster that changes — the median
   team in the corpus fields 3 different rosters (max 33). A Bradley–Terry fit
   over `player × role` columns is a strictly finer description of "who is
   playing", worth **−0.033 nats and +0.04 AUC** over team ratings.
2. **Hand the head the linear part.** The strength estimate is already the
   answer; the encoder's job is one logit of interaction. A **24-unit** head
   beats the full-width one by **−0.0128 nats**.
3. **Fit the combination, don't learn it.** Stacking the player ratings and the
   draft logit with two coefficients fitted on val beats asking the encoder to
   consume the feature end to end by 0.007 — the price of learning a scale that
   could have been handed over.

Recipe, if it is ever wanted: d256 encoder, dropout 0.2, `--head-dim 24`,
`--team-mode player --team-player-k 10`, CE, select on val log-loss.
Checkpoint `winpred-pro-ce-players-seed20261105-through-20261104`.

**Caveat that must travel with the number:** rosters are announced before the
draft, so this is legitimate pre-game information — but a UI showing 0.60 would
be showing a *lineup* number, not a draft number. It is also the worst-calibrated
arm (ECE 0.0286 vs 0.0093 for draft-only), which matters the moment a probability
is displayed rather than scored.

## 3. Why the experiment failed

Not a bug and not a lack of data — the quantity is small and misaligned with the
product.

1. **The draft is a minor part of the outcome.** 0.005 nats for the draft against
   0.044 for who plays — a factor of eight. Even a perfect draft model could only
   move a win probability by a few points.
2. **What draft signal exists is linear in champion identity.** A champion × role
   Bradley–Terry prior, 865 numbers and no network, scores **0.68879** against
   the 2.6 M-parameter encoder's 0.68884. The encoder's marginal contribution is
   **0.0003 nats**. Every attempt to give it more room was neutral or worse.
3. **The signal describes outcomes, not choices.** `choice_test.py` swaps one
   slot for each of 24 role-legal alternatives, 47,620 picks: the chosen champion
   outranks **0.5002** of them, and the champion prior alone manages **0.5000**.
   Champion win-rate has no relationship to which champion a pro team picks —
   pros draft for composition, counters and role fit.
4. **It cannot score a live draft.** The input is a complete board with no bans.
   Mid-draft you would have to imagine the remaining picks; the spread across
   imagined endings is far wider than the 0.003-nat signal being read.
5. **As a displayed number it is nearly constant.** 90 % of pro draft-only
   predictions fall in **45.1–58.1 %**, soloQ in **40.2–55.2 %**. Well
   calibrated, and uninformative — a panel that says 51 % for every draft.

## 4. What not to try

Each of these is closed by a measurement, not an opinion.

**Architecture and capacity**

1. **Don't add encoder capacity** — wider encoder, wider head, more intra
   blocks, a `h_blue*h_red` product term: all neutral or worse once a strength
   feature is present. The bottleneck is the signal.
2. **Don't reason about capacity without rows per parameter.** Head 24 helps pro
   (−0.0128, 69 parameters per training row) and *hurts* soloQ by +0.00068
   (8.5 per row). One rule does not transfer across those regimes.
3. **Don't add hand-set priors without measuring them.** The `[5,5]` role matrix
   drifts 1 % of its own scale and zeroing it at inference moves log-loss by
   4e-6 — it is inert, because the role is already an input.
4. **Don't ensemble.** Five members of the best recipe: −0.00027
   [−0.00223, +0.00160], AUC *worse*. There is nothing there to average away.
5. **Don't add blue/red swap augmentation** while `side_bias` exists — mirroring
   makes the task exactly symmetric and zeroes the parameter by construction.

**Features and data**

6. **Don't add recency weighting.** Monotone harmful at both half-lives
   (hl365 0.66211 vs 0.64747 with none): the median row is 865 days old.
7. **Don't fit team and player blocks jointly.** They are near-collinear and the
   split is decided by the penalty, not the data — the `kp` grid is
   non-monotone, which is the signature of reporting the optimiser.
8. **Don't expect more data to help.** The full 2017–2026 corpus (81,227 drafts)
   does not beat 2022-onward: 0.68605 vs 0.68578.
9. **Don't read `side_bias` as the learned side advantage.** LayerNorm→GELU→
   Linear synthesises its own constant; setting `bias=False` only half-fixes it.

**Cross-domain**

10. **Don't transfer across domains.** Pro-fitted champion ratings on soloQ:
    AUC **0.4994**, refit coefficient **0.001**. A soloQ model on pro costs
    **+0.0046 nats over the pro prior**. Champion strength is a property of the
    league.
11. **Don't train `--data both` without a domain input** or per-domain loss
    normalisation. It collapses to the constant; the signature is **train loss
    rising**, i.e. representational collapse, not overfitting.

**Evaluation (these cost the most time when ignored)**

12. **Don't select on anything but log-loss.** Accuracy is winnable by drifting
    to the majority side (+0.019 accuracy with log-loss *worse* than the prior),
    and ECE does not track log-loss (best arm 0.0286 vs 0.0093).
13. **Don't compare architectures on pro at one seed.** At n = 4,762 one seed
    moves AUC by ±0.007 — the size of the effect one arm "gained" from a
    parameter that provably does nothing.
14. **Don't trust a val gap below ~0.001** when the signal is 0.003 nats. A
    soloQ arm that won the val grid at 0.68817 lost on test by **+0.00068**.
15. **Don't compare arms across protocols** with per-arm reports. Use the shared
    rows only (`compare.py --split shared`, `sweep.py`): each model trained on
    most of the other's validation set.

**Product**

16. **Don't build a win-probability panel.** See §3.5 — the number barely moves.
17. **Don't re-rank suggestions with a win head.** Measured at 0.5002.
18. **Don't ship the champion tier list as a recommendation.** It is a
    description of outcomes; it also fails the choice test at 0.5000.
19. **Don't make a value head a stage of the suggestion pipeline.** A value model
    answers "how good is this position"; a suggestion needs "what is the next
    move". They are different functions and this branch is the demonstration.

## 5. What might still be worth one attempt

- **Bans.** Untested here by design, and the largest untested input in the
  project — half of a pro draft's denial happens in the ban phase, and it is the
  main branch's own domain. Do it as a fresh experiment with its own ablation,
  and hold it to the choice-test bar, not to log-loss.
- **The choice objective itself.** "Which of these candidates did the team take,
  given the board" is a different function, and `masking-*` already implements
  it. If a value signal is ever wanted, it should be trained against that
  objective, not against outcomes.

Nothing else from this branch is worth a second attempt.

## 6. Where the evidence is

- `pro/model/winpred/README.md` §1–§19 — every number above, with its interval, in
  the section that measured it.
- `pro/model/winpred/reports/*.json` — the raw output of every run, including the
  ones that lost.
- `python pro/model/winpred/choice_test.py` — the decisive test, 10 min on CPU.
- `pytest pro/model/winpred` — 50 tests, including the ones that pin the traps in §4.

The whole branch runs on CPU; a full sweep is minutes and a training run is
under an hour. Nothing here is expensive to re-check, which is the main argument
for checking rather than trusting any of the above.
