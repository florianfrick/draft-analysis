# Win prediction from a completed draft — `exp/win-prediction`

A separate question from the rest of the repo. Everything here asks one thing:
**given the finished 10-pick board, who wins?** Nothing else — no next-move
ranking, no masked completion, no bans, no partial drafts.

The folder is self-contained on purpose: it reads the two data pipelines and
reuses the encoder blocks, and changes nothing outside `model/winpred/`
(`STATUS.md` is deliberately untouched — this is an exploration, not a shipped
stage). If it works, the folder moves to `main` as a whole.

```
data.py       full-draft tables, the 80/10/10 split, prediction-set I/O
metrics.py    log-loss, Brier, ECE, ROC-AUC, precision/recall at high confidence
baseline.py   the blue-side win-rate prior (one scalar)
model.py      Siamese draft encoder -> MLP(h_blue, h_red, h_blue - h_red) + side bias
train.py      training, CE or focal, W&B, checkpoint + saved test predictions
evaluate.py   score/compare any mix of checkpoints, .npz prediction sets, priors
test_winpred.py  25 tests
reports/      the JSON evidence behind every number below
```

```bash
# baseline only (step 5)
python model/winpred/evaluate.py --data pro   --split test --baseline prior \
    --out model/winpred/reports/baseline_pro_seed20261105.json
python model/winpred/evaluate.py --data soloq --split test --baseline prior \
    --out model/winpred/reports/baseline_soloq_seed20261105.json

# the experiment (step 8)
python model/winpred/train.py --data pro --loss ce --seed 20261105 --epochs 40

# compare it to the baseline on the same rows (step 2's script)
python model/winpred/evaluate.py --data pro --split test --seed 20261105 \
    --baseline prior \
    --ckpt winpred-ce=model/checkpoints/winpred-pro-ce-seed20261105-through-20261104/best.pt \
    --by-patch --out model/winpred/reports/compare_pro_ce_vs_baseline_seed20261105.json

pytest model/winpred
```

---

## 1. Data

| | soloQ | pro |
|---|---|---|
| source | `drafts_16.20.pt` (schema 2, with `wins`) | `pro_drafts.pt` (Oracle's Elixir) |
| games | 384,356 | 46,879 |
| patch | 16.20 only | 106 patches, 12.01 → 16.20 |
| dates | — | 2022-01-10 → **2026-11-04 02:20** |
| blue win rate | **47.64%** | **53.05%** |
| dropped by the loader | 0 | 0 |
| split | 307,484 / 38,436 / 38,436 | 37,343 / 4,774 / 4,762 |

A row is a **completed draft**: 10 picks in the canonical slot order
`[B_TOP..B_SUP, R_TOP..R_SUP]` with exact roles, the patch, and `blue_win`.
**No bans are read** — not embedded, not attended to, not in the loss. **No
masking and no prefixes**: one row per game, the finished board.

**Remakes and unlabelled rows** are already excluded upstream and the loaders
assert it rather than filter it again:

* soloQ — `pipeline.MIN_DURATION_S = 300` at scan time, and
  `scan_and_index(with_win=True)` counts rows whose two sides disagree about the
  result as `skipped_no_win` instead of caching them.
* pro — `game_to_draft` drops any game without a complete 20-move draft
  (`incomplete_picks` 4,008 / `incomplete_bans` 2,731 / `duplicate_champ` 30 /
  `pick_mismatch` 7 / `bad_result` 2 / `bad_firstpick` 69). The 2022-2026 CSVs
  contain exactly **one** team-row block under 300 s
  (`ESPORTSTMNT01_3408461`, patch 13.14) and it is not in the corpus for an
  unrelated reason, so a duration filter would match zero rows.

### The split (seed **20261105**)

Random 80/10/10, and two structural rules:

1. **Pro series are drawn whole.** A series shares both teams and — from 2025 —
   a fearless pool, so splitting its games across the boundary would put the
   same players and the same removed champions on both sides. 0 series
   straddle a split (asserted in `test_pro_split_keeps_series_whole_...`).
   Series are assigned to the patch of their first game.
2. **The draw is per patch** — `Random("<seed>|<patch>")` — so one patch's
   allocation does not depend on which other patches exist, and every patch
   above `min_patch_rows=10` appears in all three splits. This is also what
   makes the patch embedding legitimate: the same patch is on both sides, so it
   interpolates a seen patch instead of memorising an unseen id (the failure
   mode STATUS.md §6 records for the date split). No patch fell below the
   threshold.

SoloQ has one patch and no series structure, so it is a plain seeded
permutation. One seed drives both the split and the model init, because the
brief asks for the model to be trained "using the same seed as the baseline" —
and a baseline fits its prior on `train`, so two seeds would mean two datasets
wearing the same label.

---

## 2. Metrics

`evaluate.py` scores any source with the same code: a checkpoint, a saved
`.npz` prediction set, the fitted prior, or a bare number. Sources are aligned
by **game id** and a subset/superset is refused, so two models can never be
compared on the rows they happen to share.

| metric | the question it answers |
|---|---|
| **log-loss** | the primary: a proper scoring rule, and what CE optimises. Selection is on val log-loss. |
| **Brier** | proper, bounded; the gate STATUS.md §6/§9 already uses ("beat the constant's Brier before surfacing a probability"). |
| **ECE / MCE** | is "71%" actually 71%? 15 equal-width bins, weighted by bin mass. |
| **ROC-AUC** | rank only, so it is invariant to the side prior. **A constant scores exactly 0.5**, which makes it the cleanest statement of what the draft adds. |
| **precision/recall at p<0.35 / p>0.65** | the operational question: does a confident call mean anything, and how often is the model confident at all. 0.35–0.65 counts as an abstention. |

Rates carry Wilson intervals; model-vs-model differences carry a **grouped**
bootstrap (whole series resampled, 2,000 draws) — only for log-loss / Brier /
accuracy, because AUC is not a per-row sum and an interval on it would be
wrong rather than merely noisy.

---

## 3. Baseline (step 5) — `p = mean(blue_win | train)`

| data | n test | test blue rate | prior | log-loss | Brier | ECE | AUC | accuracy |
|---|---|---|---|---|---|---|---|---|
| soloQ 16.20 | 38,436 | 0.4754 | 0.47662 | 0.6919 | 0.2494 | 0.0012 | 0.5000 | **0.5246** |
| pro 2022-2026 | 4,762 | 0.5197 | 0.53322 | 0.6927 | 0.2498 | 0.0135 | 0.5000 | **0.5197** |

**~53% confirmed, with a twist.** Pro blue wins 53.05% of 46,879 games, so
"always call blue" is 53.05% over the corpus — the expected number. SoloQ blue
wins **47.64%** of 384,356 games, so the best constant there is **red**, at
52.36%. The guess that pro carries a bigger blue-side bias is right (+5.4 pp
between the two), but the soloQ bias points the other way: at high elo, blue's
first pick is worth less than red's last pick. That is why the prior is fitted
per dataset and never assumed to be 0.5.

The accuracy column above is the *test split's* own rate (pro 51.97%, soloQ
52.46%), which is what a model has to beat on those rows; the corpus rates
(53.05% / 47.64%) are the population the split was drawn from. On pro the
draw came out 1.1 pp redder than the corpus, which is inside sampling noise
(±0.72 pp at n = 4,762) but is why the two numbers differ.

Both baselines make **zero** high-confidence calls (0.47–0.53 is never outside
0.35/0.65), so the confidence buckets have nothing to score against until a
model produces probabilities that are actually spread out. That is a real
finding, not a missing row.

---

## 4. The model

```
slot = LayerNorm(champ[173] + role[5] + patch[106+UNK] [+ attrs])
  -> Siamese intra-team self-attention   (ONE encoder over [2B,5,D])
  -> symmetric inter-team cross-attention (ONE block, both directions)
  -> h_blue, h_red = mean over the team's 5 slots
  -> logit = MLP([h_blue, h_red, h_blue - h_red]) + b_side
```

* **Role is the only positional signal.** There are no positional embeddings,
  so the role embedding is what tells the set which pick is the support.
* **`h_blue - h_red` is an explicit head input**, so the head's job is "how
  much does the difference matter", not "compare two vectors".
* **The patch vector is a segment embedding** (zero-initialised, added to every
  slot), so patch-on and patch-off are bit-identical at step 0.
* **`b_side` is one scalar**, initialised at 0 — the baseline's whole degree of
  freedom, available to the network.
* **No blue/red swap augmentation**, and this is a design constraint rather
  than an omission: mirroring makes the task exactly symmetric
  (`p(mirror) = 1 - p(orig)`), which forces `b_side = 0` and deletes the
  parameter the experiment is about. The architecture stays Siamese; only the
  augmentation is incompatible.
* Attributes (18-axis kit table) are implemented and **off** — §6 measured them
  as noise and the brief says no extra experiments yet.

---

## 5. Run 1 — pro, cross-entropy, seed 20261105

`model/checkpoints/winpred-pro-ce-seed20261105-through-20261104/best.pt`

| | |
|---|---|
| data | pro, corpus 2022-2026, 46,879 drafts, newest row **2026-11-04** |
| split | seed **20261105**, 37,343 / 4,774 / 4,762 |
| loss | cross-entropy (`--loss ce`) |
| model | D=256, 2 intra + 1 cross, 8 heads, dropout 0.1, mean pool, patch emb ON, attrs OFF → **2,642,178 params** |
| optim | AdamW, lr 1e-3, wd 0.01, warmup 5 %, cosine, batch 512, 40 epochs, clip 1.0 |
| selected | **epoch 5** on val log-loss 0.687903 |

W&B: `lol-draft-winpred` → run `m8f67o9h` (2 min on the 4060; the whole 40-epoch
run is one curve because each epoch is 73 steps).

**Test (n = 4,762), against the baseline on the identical rows:**

| | model | baseline | Δ | 95 % CI (series bootstrap) |
|---|---|---|---|---|
| log-loss | **0.6868** | 0.6927 | **−0.00595** | [−0.00943, −0.00265] |
| Brier | **0.2468** | 0.2498 | **−0.00296** | [−0.00469, −0.00133] |
| accuracy | **0.5487** | 0.5197 | **+0.02898** | [+0.01063, +0.04711] |
| ROC-AUC | **0.5605** | 0.5000 | **+0.06051** | (AUC is not a per-row sum) |
| ECE | **0.0090** | 0.0135 | — | — |

High confidence: **47 blue calls** (0.99 % coverage) at precision **0.723**
[0.582, 0.831], recall 0.014; **zero red calls**. The model is better than the
prior on all three decomposable metrics with intervals that exclude zero, and
its ranking gain (+0.06 AUC) is entirely the draft's, since a constant is 0.5
by construction.

### The learning curve is the interesting part

| epoch | train | val log-loss | val AUC | val ECE |
|---|---|---|---|---|
| 1 | 0.6905 | 0.6913 | 0.5506 | 0.037 |
| **5** | 0.6801 | **0.6879** | 0.5555 | 0.012 |
| 10 | 0.6648 | 0.6940 | 0.5572 | 0.045 |
| 20 | 0.6059 | 0.7339 | 0.5556 | 0.113 |
| 40 | 0.3950 | **1.0572** | 0.5426 | 0.262 |

The model overfits **confidence, not ranking**. By epoch 40 the val AUC has
moved only −0.013 while log-loss has risen 0.37 and ECE 0.26 — the ranking is
stable and the probabilities have become unusable. "Train 40 epochs and keep
the last" would have shipped a model *worse than the prior* on log-loss and
Brier while looking almost as good on AUC, which is exactly why selection is on
val log-loss and not on accuracy or AUC. Early stopping at epoch 5 is doing
real work here, and it matches the repo's own finding that pro models peak
early (STATUS.md §7).

---

## 6. What these numbers do and do not say

**Do:** the finished draft carries real, measurable win information — +0.06 AUC
and −0.006 log-loss over the side prior, on 4,762 held-out pro games, with
intervals that exclude zero under a series-grouped bootstrap. The gain is small
in absolute terms and it is not a tool yet: the model almost never reaches 0.65,
so 99 % of games get a prediction inside the abstention band.

**Do not:** read +0.06 AUC as "draft decides 6 % of games". AUC is a ranking
statistic; the honest effect size is the log-loss delta, and by that measure
the draft is worth about 0.6 % of a nat over the base rate. Compare it to the
repo's earlier value heads (STATUS.md §6): policy `value_head` 0.4969 AUC,
finetune `PoolOutcomeHeads` 0.5172, PMA prefix model 0.5426, and a
173-parameter logistic 0.5456. This model is at **0.5605** with no bans, no
partial drafts and no player identity — better than all of them, which is
consistent with "the trunk never had to know which comp wins" being the
bottleneck rather than capacity.

**Do not** compare the per-patch table to a pooled number. On the newest 12
patches with n ≥ 10 the model is better on 6 and worse on 6 (16.20: 0.6978 →
0.6953). The pooled gain is not coming from the newest patch.

### One issue to fix before the next run

The learned `b_side` ended at **+0.0040**, while the prior's logit is +0.1331 —
the side prior was absorbed by the **last layer's bias inside the head**, which
is a second, redundant constant. So `b_side` is currently unidentifiable: the
model has the right degree of freedom, twice, and the split between them is
arbitrary. The comparison against the baseline is unaffected (the model has the
same one degree of freedom available, wherever it lives), but the parameter the
architecture claims to expose is not doing the work. Next run: drop the head's
final bias so `b_side` is the only path.

---

## 7. Next steps (not run — the brief says stop here)

1. **Fix `b_side` identifiability** (above), then re-run the same seed. This is
   a correctness fix, not a new experiment.
2. **SoloQ, same seed, same model** — the 384 k-game set is 8× the pro corpus
   and the prior points the other way, so it is the cheapest test of whether
   the +0.06 AUC is a pro-specific effect or just "more data".
3. **Bans as an input** — the biggest untested signal, and the one this model
   was deliberately built without. It needs its own ablation: ban-only, picks +
   bans, and the fearless pool as a *context* rather than a constraint.
4. **Regularise before scaling.** Val log-loss turns up after 5 epochs on 37 k
   rows with 2.6 M parameters. Dropout 0.2–0.3 or D=128 first (STATUS.md §9.2
   says the same for the encoder), otherwise every later comparison is a
   comparison of early-stopping luck.
5. **Per-patch calibration.** The pooled ECE is 0.009, but the newest patches
   are where a shipped number would be read; check whether the calibration
   holds on 16.18–16.20 alone before quoting a probability in the UI.

**What not to do:** don't quote accuracy as the headline (the prior alone gets
52 %, and a model can win accuracy by drifting to the majority side). Don't
compare a focal-loss run to this one without re-running the baseline on the same
rows. Don't add a blue/red swap augmentation while `b_side` exists — it zeroes
it by construction.
