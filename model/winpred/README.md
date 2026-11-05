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
models/
  baseline.py        the blue-side win-rate prior (one scalar)
  siamese_model.py   Siamese encoder -> MLP(h_blue, h_red, h_blue-h_red[, h_blue*h_red])
                     + side bias, with the optional [5,5] role prior on cross attention
train.py      training, CE or focal, soloq/pro/both, W&B, checkpoint + predictions
evaluate.py   score/compare any mix of checkpoints, .npz prediction sets, priors
sweep.py      all arms on BOTH test views in one table (§7)
compare_arms.py  paired model-vs-model test on the same rows, and ensembling (§8)
search.py     val-only hyperparameter search (`--no-test`, early stopping) (§9)
team_check.py how much of the outcome is team/player strength rather than draft
              (§10, §13) - `--team-mode {rate,bt,player,roster}`
test_winpred.py  43 tests
reports/      the JSON evidence behind every number below
```

```bash
# baseline only (step 5)
python model/winpred/evaluate.py --data pro   --split test --baseline prior \
    --out model/winpred/reports/baseline_pro_seed20261105.json
python model/winpred/evaluate.py --data soloq --split test --baseline prior \
    --out model/winpred/reports/baseline_soloq_seed20261105.json

# the experiment (step 8), and the six-arm sweep (§7)
python model/winpred/train.py --data pro --loss ce --seed 20261105 --epochs 40
for d in soloq pro both; do for l in ce focal; do
  python model/winpred/train.py --data $d --loss $l --seed 20261105 --epochs 40
done; done

# compare it to the baseline on the same rows (step 2's script)
python model/winpred/evaluate.py --data pro --split test --seed 20261105 \
    --baseline prior \
    --ckpt winpred-ce=model/checkpoints/winpred-pro-ce-seed20261105-through-20261104/best.pt \
    --by-patch --out model/winpred/reports/compare_pro_ce_vs_baseline_seed20261105.json

# all six arms on both test views (the §7 tables)
python model/winpred/sweep.py --seed 20261105 --ckpt pro-ce=... --ckpt pro-focal=... \
    --ckpt soloq-ce=... --ckpt soloq-focal=... --ckpt both-ce=... --ckpt both-focal=... \
    --out model/winpred/reports/sweep_seed20261105.json

# the v2 architecture (§8): product term in the head + role prior on cross attention
# --tag keeps it from overwriting the v1 checkpoints the tables above cite
python model/winpred/train.py --data pro --loss ce --seed 20261105 --epochs 40 \
    --head-product --role-bias --tag v2

# the search (§9-§13). Every grid is val-only: --no-test drops the held-out
# rows before the loaders are built, so a config cannot be chosen on test.
python model/winpred/search.py --grid team_bt      # which strength estimator
python model/winpred/search.py --grid head_small   # how wide the head should be
python model/winpred/search.py --grid player       # player ratings instead of teams

# how much of the outcome is who is playing, rather than what was picked (§10, §13)
# --team-mode player also prints the draft+strength stack fitted on val
python model/winpred/team_check.py --team-mode player \
    --ckpt model/checkpoints/winpred-pro-ce-d384-seed20261105-through-20261104 \
    --out model/winpred/reports/player_strength_pro_seed20261105.json

# the best arm (§13): 20 epochs is plenty, it selects at epoch 1-6
python model/winpred/train.py --data pro --loss ce --seed 20261105 --epochs 20 \
    --patience 4 --tag players --dropout 0.2 --head-dim 24 \
    --team-feat --team-mode player --team-player-k 10

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

**The confidence row turns out to be almost empty**, and that is a result rather
than a bug in the metric: across all six arms of §7 and **172,792** test
predictions, the model leaves the 0.35/0.65 band **5 times** — every one of them
in `pro`/focal. The two CE arms stop just short on opposite sides: pro/CE peaks
at 0.6463 against a 0.65 gate, soloQ/CE bottoms at 0.3508 against 0.35. So
precision/recall at high confidence is `nan` for five of six arms, and the honest
reading is "the model never commits". The calibration bins carry the information
a fixed band throws away — see §15 item 5.

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
* **Two optional parts exist behind flags** (`--head-product`, `--role-bias`;
  both default off, so §5–§7 describe exactly the model that produced them):
  - `h_blue * h_red`, appended to the head input (3D → 4D). The difference is
    linear in each team vector; the product is the elementwise interaction, the
    shape a *match-up* takes, and it separates cases the difference cannot
    (`a - b` and `a' - b` coincide whenever `a - a'` is orthogonal to the
    weight row). It is symmetric under a blue/red swap, so it carries no side
    information. +66,073 parameters, +2.50 %.
  - a `[5,5]` additive bias on the cross-attention logits, indexed by *role id*
    (not slot position, which keeps the block permutation-invariant), with a
    hand-set start: 0.5 same-role, +0.2 jungle row/column, 0.3 bot↔sup. That is
    the only prior knowledge in the model and all of it is about the map. It is
    a strict superset of `SymmetricCrossAttention` — zero it and the two blocks
    are numerically identical (`test_zeroed_role_bias_reproduces_the_plain_block`).
  §8 measures whether either helps.
* **The patch vector is a segment embedding** (zero-initialised, added to every
  slot), so patch-on and patch-off are bit-identical at step 0.
* **`b_side` is one scalar and the head's last layer has `bias=False`**, so it
  is the only *constant* the logit can contain — the baseline's whole degree of
  freedom, available to the network and nothing else. Run 1 had both, and
  measured the consequence: `b_side` ended at +0.004 while the prior's logit is
  +0.133, because the head's bias had absorbed the prior and the split between
  two redundant constants is arbitrary. `test_only_one_constant_parameter_exists`
  pins this down: zero every parameter and the logit is exactly `b_side`.
* **Dropout 0.2** (was 0.1 in run 1), on the transformer layers, the cross
  block and the head. Run 1's problem was not capacity — it overfitted
  *confidence*, with val AUC moving −0.013 while ECE went 0.01 → 0.26 — so
  dropout is the right lever and the same value is used for every arm of the
  sweep, otherwise the loss and data comparisons would be confounded by it.
* **No blue/red swap augmentation**, and this is a design constraint rather
  than an omission: mirroring makes the task exactly symmetric
  (`p(mirror) = 1 - p(orig)`), which forces `b_side = 0` and deletes the
  parameter the experiment is about. The architecture stays Siamese; only the
  augmentation is incompatible.
* Attributes (18-axis kit table) are implemented and **off** — §6 measured them
  as noise and the brief says no extra experiments yet.

---

## 5. Run 1 — pro, cross-entropy, seed 20261105

> **Superseded.** This was the first run, with dropout **0.1** and a head bias.
> The §7 sweep re-ran the same arm at dropout 0.2 with `bias=False` and wrote to
> the **same checkpoint path**, so `best.pt` no longer holds these weights. The
> numbers below survive in `reports/compare_pro_ce_vs_baseline_seed20261105.json`
> and are not re-runnable from disk — the same convention STATUS.md §4/§6 uses
> for the deleted pro-only checkpoints. Read §7 for the arm that is on disk.

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
patches with n ≥ 10, run 1 was better on 6 and worse on 6 (16.20: 0.6978 →
0.6953). The pooled gain is not coming from the newest patch. (Those are run-1
numbers, from the weights that were overwritten; the §7 arm's `--by-patch` puts
16.20 at 0.7156 on **n = 27**, which is too small to read either way.)

### The `b_side` issue — fixed, and only half fixed

Run 1's `b_side` ended at **+0.0040** while the prior's logit is +0.1331: the
head's own bias was a second, redundant constant, so the split between the two
was arbitrary. The sweep drops it (`nn.Linear(hidden, 1, bias=False)`), and the
result splits cleanly **by domain**:

| arm | learned `b_side` | prior logit | learned it? |
|---|---|---|---|
| soloq / CE | **−0.0824** | −0.0936 | essentially yes |
| soloq / focal | −0.0374 | −0.0936 | partly |
| pro / CE | **+0.0037** | +0.1331 | no |
| pro / focal | +0.0112 | +0.1331 | no |

So `bias=False` removed the *explicit* duplicate but not the *implicit* one.
`LayerNorm → GELU → Linear` can still synthesise a constant: the layer-normed
features have zero mean *across features*, not across samples, so a fixed weight
vector still carries a sample-independent component. On soloQ the optimiser puts
the side prior in `b_side`; on pro it puts it in the features and leaves
`b_side` at zero. The comparison against the baseline is unaffected either way —
the model has the same degree of freedom, wherever it lives — but the parameter
the architecture claims to expose still cannot be *read* as "the learned side
advantage", which is what §7's joint arms would need it to be.

---

## 7. The six-arm sweep — CE vs focal × soloQ / pro / both

Six runs: `--loss {ce,focal}` × `--data {soloq,pro,both}`, seed 20261105, one
split, one architecture, one optimiser. `protocol.hyperparameters` is identical
in all six (checked, not assumed); the parameter count moves only with the patch
vocabulary — **2,615,297** for soloQ's one-row table, **2,642,177** for pro's
107 — so the sweep varies exactly two things. Selection is val log-loss
everywhere, so no arm gets a better stopping rule than another. Focal is
`alpha = 0.5, gamma = 2.0`, which is a reweighting of BCE by `(1 - p_t)^2`, not
a rescaling.

### Own-domain test

Each arm on its own test view, against **that view's** prior. All intervals in
this section come from one run of `sweep.py` (`reports/sweep_seed20261105.json`,
1,000 grouped bootstraps), so they are comparable with each other:

| arm | n | log-loss | Brier | ECE | AUC | acc | Δ log-loss vs prior [95 % CI] | sel. epoch |
|---|---|---|---|---|---|---|---|---|
| prior (pro) | 4,762 | 0.6927 | 0.2498 | 0.0135 | 0.5000 | 0.5197 | — | — |
| **pro / CE** | 4,762 | 0.6883 | 0.2476 | 0.0156 | **0.5521** | 0.5378 | **−0.00440** [−0.00726, −0.00149] | 6 |
| **pro / focal** | 4,762 | 0.6885 | 0.2477 | 0.0116 | **0.5537** | 0.5447 | **−0.00422** [−0.00702, −0.00152] | 12 |
| prior (soloQ) | 38,436 | 0.6919 | 0.2494 | 0.0012 | 0.5000 | 0.5246 | — | — |
| **soloQ / CE** | 38,436 | 0.6888 | 0.2479 | 0.0049 | **0.5438** | 0.5371 | **−0.00310** [−0.00394, −0.00232] | 31 |
| **soloQ / focal** | 38,436 | 0.6904 | 0.2486 | 0.0136 | **0.5417** | 0.5362 | **−0.00156** [−0.00198, −0.00116] | 27 |

Every arm improves on its prior on **all three** decomposable metrics, with the
log-loss interval excluding zero in every case, and every arm ranks above 0.5
AUC — the draft is worth about **0.3–0.4 % of a nat** and **+0.044 to +0.052
AUC**, on both corpora, with no bans and no partial drafts. (Accuracy is the exception to
the interval claim: pro/CE's is +0.018 [−0.002, +0.037], so it does not
distinguish anything.) The four arms are within 0.0021 log-loss of each other,
which is the first thing to notice: **the loss function is not the lever.**

Two differences that are large enough to be structural rather than noise:

* **soloQ needs 27–31 epochs to peak; pro needs 6–12.** 307 k training rows at
  one patch barely overfit at all (val log-loss 0.6896 → 0.6885 across 40
  epochs), while 37 k pro rows overfit hard (0.6957 → 0.8592 at dropout 0.2, an
  improvement on run 1's 1.0572 but still a collapse of the probabilities).
* **Focal is worse on soloQ and a wash on pro.** It gains 0.00156 log-loss on
  soloQ where CE gains 0.00310, and 0.0016 AUC on pro where it gains 0.0000 on
  soloQ. Neither gap survives a seed repeat (§15).

### The cross view — the sweep's real question

The same six checkpoints, scored on **both** test views. Patch ids are
translated by patch *label* (`evaluate.patch_remap`), so a pro-trained arm scores
soloQ rows on the 16.20 row it actually learned. Bold = the arm's own view.

| arm | soloQ (n = 38,436) log-loss / AUC / Δ | pro (n = 4,762) log-loss / AUC / Δ |
|---|---|---|
| prior | 0.6919 / 0.5000 / — | 0.6927 / 0.5000 / — |
| **pro / CE** | 0.6976 / **0.5002** / **+0.00569** [+0.00465, +0.00684] | **0.6883** / **0.5521** / **−0.00440** [−0.00726, −0.00149] |
| **pro / focal** | 0.7013 / **0.4999** / **+0.00940** [+0.00809, +0.01083] | **0.6885** / **0.5537** / **−0.00422** [−0.00702, −0.00152] |
| **soloQ / CE** | **0.6888** / **0.5438** / **−0.00310** [−0.00394, −0.00232] | 0.6999 / **0.5070** / **+0.00720** [+0.00312, +0.01159] |
| **soloQ / focal** | **0.6904** / **0.5417** / **−0.00156** [−0.00198, −0.00116] | 0.6944 / **0.5085** / +0.00170 [−0.00100, +0.00442] |
| `both` / CE | 0.6900 / 0.5363 / −0.00194 [−0.00274, −0.00117] | 0.6964 / 0.5180 / +0.00369 [−0.00010, +0.00742] |
| `both` / focal | 0.6908 / 0.5389 / −0.00111 [−0.00153, −0.00069] | 0.6916 / 0.5356 / −0.00118 [−0.00322, +0.00080] |

Every arm improves on the prior on its **own** view and fails to on the other,
and the cross-view AUCs sit on 0.500. The pro → soloQ direction is the clean one:
**0 of 38,436 rows** fall on the checkpoint's UNK patch row, so the model sees a
patch embedding it was trained on (pro's table contains 16.20) and still ranks at
**0.5002** — chance, to four decimals, on 38 k games. The soloQ → pro direction
is a lower bound rather than a measurement, because 4,735 of 4,762 rows hit the
UNK patch row, so it is "never seen those patches" as much as "never seen pro";
it points the same way and is even worse.

So the draft's win information is **domain-specific**: the ordering a pro draft
induces does not order soloQ games, and vice versa, even though both are clearly
above chance at home. That is the sweep's most useful result and it was not the
expected one — it says the two corpora do not share the thing the encoder is
learning, so joint training has nothing to pool.

### The `both` arms: a shared head on an 8:1 mixture collapses

The joint corpus is 46,879 pro + 384,356 soloQ = 431,235 drafts, **89.1 %
soloQ**, split 344,827 / 43,210 / 43,198. Neither joint arm produces a usable
model, and the way they fail is informative:

| | `both` / CE | `both` / focal |
|---|---|---|
| best val log-loss | 0.69049 (ep **2**) | 0.69091 (ep **5**) |
| train loss, minimum → end | 0.69037 (ep 3) → **0.69255** | 0.08627 (ep 4) → 0.08658 |
| first epoch with val AUC exactly 0.5000 | **9** | **16** |
| epochs at exactly 0.5000 AUC | 31 of 40 | 25 of 40 |

The model does not overfit — **train loss rises** from 0.69037 (ep 3) to 0.69255
(ep 5) and stays there, so the optimiser left a strictly better solution for the
constant and never came back. Overfitting would show train ↓ / val ↑. This is a
representational collapse: with 89 % of the rows in a domain where the draft
barely helps and whose base rate points the *other* way (soloQ blue 47.6 %, pro
53.1 %), the shared optimum is very close to the constant, so nothing punishes
the champion embeddings for collapsing and the model cannot climb back out.
Focal only delays it — 7 epochs, then the same constant — which fits: it
down-weights the examples the model already handles, so the majority mass
dominates the gradient a little later.

Early stopping catches it (`best.pt` is epoch 2 and 5 respectively), which is why
the joint arms still report numbers rather than NaNs. Scored per domain against
**that domain's own** prior (`sweep.py`, not `test_by_domain`, which used the
pooled prior before the fix):

| joint arm | view | n | log-loss | AUC | Δ vs that domain's prior [95 % CI] |
|---|---|---|---|---|---|
| `both` / CE | soloQ | 38,436 | 0.6900 | 0.5363 | **−0.00194** [−0.00274, −0.00117] |
| `both` / CE | pro | 4,762 | 0.6964 | 0.5180 | **+0.00369** [−0.00010, +0.00742] |
| `both` / focal | soloQ | 38,436 | 0.6908 | 0.5389 | **−0.00111** [−0.00153, −0.00069] |
| `both` / focal | pro | 4,762 | 0.6916 | 0.5356 | −0.00118 [−0.00322, +0.00080] |

Focal's joint arm is the better of the two, and it is the only **joint** arm
whose pro number is not worse than the prior — but it is still 0.003 log-loss
behind pro/CE on pro and 0.002 behind soloQ/CE on soloQ, i.e. the joint corpus
buys nothing on either half while costing the majority half real accuracy.
`both`/CE's pro number is the one clear failure: **+0.0037 against pro's own
prior** (interval touching zero from above).

Adding pro data also makes the **soloQ** number worse, and that is measurable on
the identical rows: `both`/CE vs `soloQ`/CE on the same 38,436 soloQ test games
is **+0.00117 log-loss [+0.00063, +0.00171]**, and `both`/focal vs `soloQ`/focal
is **+0.00044 [+0.00022, +0.00066]** — smaller under focal, but the same sign and
also excluding zero. The pro half is not merely unhelpful to the soloQ half; it
is harmful, because it drags the shared model toward a different base rate and
toward the collapse. Under both losses, the honest summary of `--data both` at
this ratio is: **it costs the majority half and does not pay for it.**

Note the accuracy trap this exposed. The first version of `test_by_domain`
compared every block against the **pooled** prior, so `both`/CE's pro subset read
accuracy **+0.019** — while the model was actually 0.021 *below* pro's own prior
(0.4990 vs 0.5197). Accuracy moved only because the pooled prior predicts red and
pro's rate is above 0.5. `domain_deltas(..., train_idx=...)` now fits each
domain's own rate, and `test_domain_deltas_reference_is_the_domains_own_rate`
pins it; the joint reports on disk predate the fix, which is why the table above
quotes `sweep.py`.

### What the sweep does and does not say

**Does:** within a domain, the finished draft carries a small but real signal,
0.003–0.004 nats and +0.044 to +0.052 AUC over the side prior, on both corpora,
with log-loss intervals excluding zero. Across domains it carries **nothing**
(0.5002 / 0.5070 AUC), so "train on soloQ, deploy on pro" is measured and does
not work. Joint training at this ratio is a negative result.

**Does not:** rank CE against focal. Every within-domain gap here is smaller
than the early-stopping noise (pro selects epoch 6 vs 12 out of 40), and the
joint comparison is one collapse against a slower collapse. It is also one seed
throughout; §15 lists the repeat as the cheapest way to settle it.

**Does not** say the joint corpus is hopeless. It says *this* joint setup is:
one shared head, no domain input, no reweighting. A domain flag or per-domain
loss normalisation attacks exactly the failure mode observed, and §15 puts it
first for that reason.

---

## 8. Architecture v2 — the product term is small, the role prior is a no-op

Two changes, both behind flags (§4). Everything else is held fixed: same seed,
same split, same hyperparameters (`protocol.hyperparameters` is identical across
the four arms below except the two new booleans), so the delta is the
architecture. `--tag` keeps the new checkpoints out of the v1 paths, because
§7's tables cite those.

A 2×2 on **pro** (n = 4,762, 2,737 series), each arm differenced against v1 on
the same rows (`compare_arms.py`, 2,000 grouped bootstraps):

| arm | sel. ep | params | log-loss | Brier | ECE | AUC | acc | Δ log-loss vs v1 [95 % CI] | ΔAUC |
|---|---|---|---|---|---|---|---|---|---|
| v1 (neither) | 6 | 2,642,177 | 0.68833 | 0.24761 | 0.0156 | 0.5521 | 0.5378 | — | — |
| + product | 7 | 2,708,225 | 0.68780 | 0.24733 | 0.0205 | 0.5596 | 0.5435 | −0.00053 [−0.00381, +0.00270] | +0.0075 |
| + role bias | 5 | 2,642,202 | 0.68800 | 0.24742 | 0.0193 | 0.5584 | 0.5475 | −0.00033 [−0.00277, +0.00215] | +0.0063 |
| + both (v2) | 6 | 2,708,250 | 0.68728 | 0.24708 | 0.0144 | 0.5600 | 0.5428 | −0.00105 [−0.00334, +0.00133] | +0.0078 |

Every variant is ahead of v1 on every metric and **no interval excludes zero**.
The three variants are also indistinguishable from each other (0.0007 nats,
0.0016 AUC apart) — which is not a lack of power in the usual sense: repeating
the identical v2 config twice moves log-loss by 0.00006 and AUC by 0.0002. The
arms are tied because pro's test set is small, not because the runs are noisy.

### The role prior is inert — measured, not assumed

The `[5,5]` matrix barely moves in training, and deleting it from a trained
checkpoint changes nothing:

| | v2 (both) | role-bias only |
|---|---|---|
| mean abs drift from the hand-set start | 0.0103 | 0.0056 |
| zero it at inference → Δ log-loss | **+0.000004** | **+0.000028** |
| zero it at inference → Δ AUC | −0.00004 | −0.00007 |
| max change to any single prediction | 0.0017 | — |

So the prior contributes nothing to the served model: 25 parameters whose entries
moved by ~0.01 on a scale of 0.2–0.9 (about 5 % of the matrix's mean magnitude),
removable without a measurable effect. The
role-only arm's +0.006 AUC over v1 therefore **cannot be the prior itself** —
25 extra parameters changed the gradient path, the model converged somewhere
else, and that somewhere else scored higher. That is the useful warning: on this
test set **±0.007 AUC is what a trajectory perturbation buys**, so a single-seed
architecture comparison of that size proves nothing (§15 item 2 is still open).

Why inert: the slot embedding already contains the role, so cross-attention can
learn role weighting from content, and an additive ±0.2–0.9 on a softmax over
logits the network already controls is redundant. The prior would matter for a
smaller model or scarcer data; here it is a no-op with a plausible story.

### The product term: a real but small gain on pro, nothing on soloQ

Same seed, the large corpus (n = 38,436, CI ±0.0004):

| | log-loss | Brier | ECE | AUC | acc |
|---|---|---|---|---|---|
| v1 | 0.68884 | 0.24786 | 0.0049 | 0.5438 | 0.5371 |
| v2 | 0.68877 | 0.24782 | 0.0071 | 0.5449 | 0.5365 |
| Δ | −0.00007 [−0.00042, +0.00028] | −0.00003 | +0.0022 | +0.0010 | −0.0006 |

Indistinguishable, and the interval rules out any gain larger than 0.0003 nats —
a tenth of the entire draft signal v1 has on this corpus (0.0031). On the corpus
that can actually resolve a small effect, the extra term buys nothing.

What it does change is the *shape* of the output, which is the problem §7 flagged:

* std of `p`: 0.041 → 0.047 (soloQ), 0.045 → 0.052 (pro)
* calls outside 0.35/0.65: **0 → 69** (soloQ), **0 → 32** (pro)
* the 69 soloQ calls hit **71.0 % [59.4, 80.4]** against a 47.6 % base rate; the
  32 pro calls hit 53.1 % [36.5, 69.1] against a 53.3 % prior, i.e. nothing
* ECE gets *worse* on soloQ (0.0049 → 0.0071): the wider tails are not calibrated

So v2 commits where v1 refused to, and on soloQ those commitments beat the base
rate — the first non-degenerate reading of §7's confidence metric — but the
calibration loss cancels the accuracy gain and log-loss does not move. It is a
diagnostic, not yet a reason to prefer the model.

### A second seed, because one seed cannot carry this

The pro arms above are one split. Re-running **both** configs at seed 20261106
(different split, so the pair is compared against itself):

| seed 20261106, n = 4,785 | v1 | v2 | Δ [95 % CI] |
|---|---|---|---|
| log-loss | 0.69015 | 0.68782 | **−0.00233 [−0.00416, −0.00053]** |
| Brier | 0.24847 | 0.24736 | **−0.00111 [−0.00201, −0.00025]** |
| AUC | 0.55256 | 0.55637 | +0.00381 |
| ECE | 0.03303 | 0.02312 | −0.00991 |
| selected epoch | 4 | 4 | — |

Both intervals exclude zero here, so on pro the product term survives a change
of split. Two facts keep this from being a strong claim:

* the two arms selected the **same** epoch (4) with val log-loss 0.68361 vs
  0.68366 — the selection is not biased, but it also means both are early-stopped
  by the same noisy val set;
* v1's own AUC is the stable quantity across seeds (0.5521 → 0.5526) while its
  log-loss moved 0.0018 — *more* than the v1→v2 gap at seed 1. AUC is the metric
  to trust on pro at this n; log-loss and ECE are not.

**Verdict.** The product term is worth +0.004 to +0.008 AUC on pro across two
seeds and 0.000 on soloQ, where the interval is tight enough to say so. The role
prior is measurably a no-op. Keeping v2 is defensible if pro is the target
(§1's brief is pro drafts, and +2.5 % parameters is cheap); it is not established
on the corpus that can measure it, and neither change touches the two results
that matter — the within-domain signal size (§7) and the absence of cross-domain
transfer.

---

## 9. The search protocol — val only, and a 20-epoch budget

The brief that produced §10–§14 changed two rules: **pro data only**, and
**log-loss only**. The first rule is what makes §13 possible at all - soloQ has
no rosters, so the feature that carries the outcome there does not exist - and
it is the question §1 asked. It also keeps every grid in this section on the
same 37 k train rows, so a config's val number is comparable to the next one's
without a per-domain block.

The second rule is not a restriction in practice. Log-loss is a strictly proper
scoring rule, so every arm in §10–§14 that wins on log-loss also wins on Brier
and AUC by construction of the comparison — §12's table shows the three moving
together, which is the evidence for the brief's premise that they are
correlated. ECE is the one that does *not* follow (§13's best arm is the worst
calibrated), so calibration is discussed rather than searched.

**Nothing in §10–§14 is chosen on test.** `search.py` runs `train.py` with
`--no-test`, which drops the held-out rows *before the loaders are built*, so a
config cannot score them even by accident; the winner then gets one run with
the test set attached, and that number is the result. Forty configs on 4,762
games will always produce a "best test score", and it will be a lie.

The budget is `--epochs 20 --patience 4 --min-epochs 3`, down from 40. Every
pro arm so far peaks in the first 5–12 epochs and then overfits confidence
(§5), so the remaining 30 epochs were only ever training the val curve's tail.
With early stopping the selected epoch is reported in every table: the §13 arm
selects at **epoch 1**, and a 40-epoch budget would have cost 40 epochs to
learn that.

---

## 10. Who is playing, not what was picked

`team_check.py` puts four predictors on the same 4,762 test rows. Team strength
is each team's smoothed win rate (shrinkage `k = 10` games), fitted on **train
rows only** and converted to a logit; the "draft + team" row adds the d384
model's own logit and fits its two coefficients on **val**, never test.

| predictor | log-loss | Brier | AUC | acc |
|---|---|---|---|---|
| side prior (the null) | 0.69273 | 0.24979 | 0.5000 | 0.5197 |
| draft only (d384) | 0.68775 | 0.24731 | 0.5552 | 0.5412 |
| team only | 0.64842 | 0.22829 | 0.6698 | 0.6298 |
| draft + team | 0.64226 | 0.22580 | 0.6758 | 0.6228 |

Paired, series-grouped bootstrap on the same rows:

* team only vs draft only: **−0.03933** [−0.05053, −0.02838]
* draft + team vs draft only: **−0.04549** [−0.05369, −0.03747]

So the draft is worth **−0.005 nats** and knowing who is playing is worth
**−0.044**, about eight times as much. The brief's suspicion that 0.69 means
"learned nothing" was right about the *number* and wrong about the *cause*: the
model had learned the draft signal (0.68775 beats 0.69273, interval excludes 0)
and was missing the feature that carries the outcome. Everything after this
section is about improving that feature.

---

## 11. Bradley-Terry beats the win rate; recency weighting does not

A smoothed win rate treats every game as equally informative. It is not: it
cannot tell a 60 % team in a strong league from a 60 % team in a minor one, and
this corpus spans several. `--team-mode bt` fits `logit P(blue) = r_blue -
r_red + h` by L2 logistic regression (sparse design matrix, LBFGS), so the
ratings condition on the opponent. `k` keeps its pseudo-games meaning by
dividing out the curvature of the log-loss (`lambda = 0.25 v^2 k`).

The ratings differ where the marginal rate is blind: Gen.G is 0.95 in logit
terms under the rate and **+2.26** under BT, T1 is 0.65 and **+1.96** — T1
played the harder schedule.

All numbers are val log-loss with the same encoder and `--head-dim 256`:

| arm | val log-loss |
|---|---|
| rate, `k = 10` (the §10 recipe) | 0.65115 |
| **bt, `k = 10`** | **0.64747** |
| bt, `k = 4` | 0.64889 |
| bt, `k = 20` | 0.65596 |
| bt, half-life 730 d | 0.65652 |
| bt, half-life 365 d | 0.66211 |
| rate, half-life 365 d | 0.66709 |

−0.0037 nats for conditioning on the opponent, and `k = 10` is the right
amount of shrinkage (4 is too little, 20 too much).

**Recency weighting fails, monotonically.** Rosters change, so a team's 2022
record should be weak evidence about its 2026 form — but the median row in this
corpus is **865 days** old and the oldest is 1,759, so a 365-day half-life keeps
19 % of the data and a 730-day one keeps 48 %. Whatever the decay buys in
currency it loses in sample size, and the loss dominates at every half-life
tried. The honest reading is not "recency is irrelevant" but "this corpus is too
sparse to pay for it": `--team-half-life` is left in, off by default.

---

## 12. The head was the bottleneck, not the encoder

§5–§8 spent four sections on encoder capacity and architecture. The actual
problem was the last layer. `head_in` is `3 * 256 + 3 = 771` wide, so a
full-width head is `Linear(771, 256)` — 197 k parameters trained on 37 k rows,
to combine two numbers (a matchup vector and its difference).

Val log-loss, bt team strength, dropout 0.3, everything else fixed:

| `--head-dim` | val log-loss | val AUC | sel. ep |
|---|---|---|---|
| full width (256) | 0.64747 | 0.6697 | 1 |
| 128 | 0.64973 | 0.6839 | 6 |
| 64 | 0.65411 | 0.6833 | 6 |
| 32 | 0.64065 | 0.6825 | 4 |
| **24** | **0.63465** | **0.6874** | 5 |
| 16 | 0.63562 | 0.6868 | 5 |
| 8 | 0.64250 | 0.6820 | 11 |

−0.0128 nats from shrinking one layer, three times the gain that
Bradley-Terry itself bought. The curve is U-shaped with the bottom at 24 — h64
being worse than both h32 and h128 is inside the selection noise, but the two
ends are not. Regularising instead of shrinking is worse than either
(`--dropout 0.2` 0.63662, `0.4` 0.63686) and slowing the schedule is worse
still (`--lr 5e-4` 0.65174, which cannot reach the useful region in 5 epochs).

This is the §8 conclusion arriving from the other side. There, a `[5,5]` role
prior was inert because *the encoder could already see the role*. Here, the
head does not need capacity because *the feature it is handed is already the
answer* — the encoder's contribution is one logit, and a 24-unit MLP is enough
to combine it with a strength estimate. Bigger heads memorise the training
roster.

---

## 13. Player identity beats team identity

The CSVs carry `playerid`, which §4's row schema drops. A team tag is a roster
that changes — the median team in this corpus fields **3 different rosters**
(max 33) — so a per-player rating is a strictly finer description of "who is
playing". `--team-mode player` fits `logit P(blue) = mean(r_5 blue) -
mean(r_5 red) + h` with the same L2/LBFGS machinery, players indexed by role
from the CSVs (18 s scan, memoised; `data.py` reads them directly rather than
extending `pipeline.build_rows`, whose cached rows the policy work also uses).

Ratings fitted on train rows only, then scored directly (`sigmoid(blue - red)`,
no fitted scale):

| strength estimate | train | val | test | test AUC |
|---|---|---|---|---|
| team, BT | 0.61720 | 0.63954 | 0.63150 | 0.6950 |
| **player, `kp = 10`** | 0.53471 | **0.60617** | **0.60152** | **0.7355** |

−0.033 nats and +0.04 AUC on test, and it is not a tuning artefact: the player
`kp` grid is flat (val 0.60903 / 0.60617 / 0.60706 / 0.61149 for kp = 5 / 10 /
20 / 40), unlike §11's, which has a real optimum. The train/val gap is large
(0.535 vs 0.606) because 6,784 ratings are fitted on 37 k games, and the val
number is the honest one: the fit is train-only and the split is series-grouped,
so a player's test games are never in his own history.
`test_strength_features_never_read_val_or_test_labels` asserts exactly that by
flipping every val and test label and requiring the feature not to move.

**The joint fit is not identified, and that is a negative result.** Fitting both
blocks (`--team-mode roster`) is the obvious thing to want — "T1 is good"
*plus* "T1 still has Faker" — but a roster is what a team tag *is*, so the two
blocks are near-collinear and the split between them is decided by the prior,
not by the data. Its `kp` grid is non-monotone (val 0.60545 / 0.64668 / 0.67129
/ 0.62914 for kp = 10 / 50 / 200 / 1000) — the signature of an objective that
LBFGS stops somewhere different in for each penalty. Reporting the best of those
would be reporting the optimiser. `--team-mode player` is the identified
version of the same question, and it is the one in the table.

### The final arm

Two ways to use the feature, both with a fair test split:

| model | test log-loss | Brier | AUC | acc | ECE |
|---|---|---|---|---|---|
| side prior | 0.69273 | 0.24979 | 0.5000 | 0.5197 | — |
| d384, draft only (§5) | 0.68775 | 0.24731 | 0.5552 | 0.5412 | 0.0093 |
| team2, team rate (§11) | 0.64260 | 0.22582 | 0.6774 | 0.6264 | 0.0185 |
| **players, end to end** | **0.60323** | **0.20787** | **0.7376** | 0.6716 | 0.0286 |
| player ratings + draft logit, 2 coefficients on val | **0.59624** | 0.20569 | **0.7410** | 0.6751 | — |

Paired against team2 on the same rows: **−0.03937** [−0.05022, −0.02833]. Against
the d384 draft-only model: −0.08452 [−0.09767, −0.07190].

The end-to-end arm is the d256 encoder, dropout 0.2, `--head-dim 24`,
`--team-mode player --team-player-k 10`, selected at epoch 1 (2.46 M params,
checkpoint `winpred-pro-ce-players-seed20261105-through-20261104`). The stack
below it is the same information fitted as two coefficients on val — it wins by
0.007 nats, which is the price of asking an encoder to learn a scale it could
have been handed.

**What the draft is worth, like for like.** Both rows of this comparison are
fitted on val, so the difference is the draft column and nothing else:

| val-fitted | test log-loss | AUC | weight |
|---|---|---|---|
| player strength only | 0.60010 | 0.7355 | 0.851 |
| player strength + draft logit | 0.59624 | 0.7410 | 0.846 / 1.130 |

**−0.0039 nats.** That is the honest size of the draft signal once the roster is
known, and it is the same ~0.004 the draft was worth over the prior in §7 — the
signal is small, it is real, and it does not grow when the baseline improves.
The player weight is below 1 (0.851) because the ratings are overconfident on
their own scale: the fit overfits train (0.535), so val shrinks it.

**Caveat, stated once:** this is a "who wins" number, not a "is this draft good"
number. Rosters are announced before the draft, so player identity is legitimate
pre-game information, but a UI that shows 0.60 must say that most of it is the
lineup. And the best arm is the worst-calibrated one (ECE 0.0286 against the
draft-only model's 0.0093), which matters the moment a probability is displayed
rather than scored.

---

## 14. What did not work

Collected so it does not get re-tried:

* **Recency weighting** — §11, monotone in the wrong direction at both
  half-lives.
* **Ensembling** — five members of the §11 recipe (same split, `--init-seed`
  77101–77105, so `compare_arms.py` can average them) score 0.64232 against
  0.64260 for one member: **−0.00027 [−0.00223, +0.00160]**, and AUC is
  *worse* (−0.00032). With early stopping already selecting a flat region of the
  val curve, the members are too correlated for averaging to help.
* **The joint team + player fit** — §13, not identified.
* **More data** — `--corpus all` (81,227 drafts, 219 patches) does not beat
  2022-onward for the draft-only model: all-d256 0.68605 < all-d384 0.68698,
  against d384-on-2022 0.68578 (§§6–7).
* **More capacity, again** — §12. Every attempt to give the model more room
  (wider encoder, wider head, more intra blocks, product term) is neutral or
  worse once a strength feature is present.
* **Player ratings as a *joint* block with teams** — the same collinearity.

---

## 15. Next steps

The brief's "don't train more experiments yet" expired with the six-arm sweep, so
items 1, 2 and 4 of the old list are **done** (§7): `b_side` is bias-free, soloQ
has been run, and dropout is 0.2 everywhere. §8's architecture variant is also
done, and §10–§14 answered the "improve the pro log-loss" brief: −0.045 nats from
the strength feature, −0.039 more from player ratings, and −0.013 from shrinking
the head (each measured on its own base, so they do not simply add). What is
left:

0. **Player strength is a feature, not a trick.** It lives in `data.py` as a
   `[N,2]` column so the encoder can consume it; it should be a first-class
   input with its own ablation (announced lineup only, per-role ratings, a
   roster-change indicator) and its own provenance in any UI. Everything else on
   this list is smaller than the 0.033 nats it is already worth.
1. **Domain conditioning, before any further joint training.** §7 shows a shared
   head on an 8:1 mixture drifting to the constant under both losses. Two cheap
   fixes: feed a domain flag/embedding in, or normalise the loss per domain so
   each half contributes equally. Until one of those exists, `--data both` is a
   negative result to reproduce, not a training recipe.
2. **Repeat the sweep on ≥3 seeds before quoting CE-vs-focal.** Both pro arms
   select an epoch out of 40 (6 and 12), so their 0.0016 AUC gap is well inside
   early-stopping noise; the only robust statements in §7 are the within-arm
   ones against the prior and the cross-view collapse to ≈0.50 AUC.
3. **Bans as an input** — the biggest untested signal, and the one this model
   was deliberately built without. It needs its own ablation: ban-only, picks +
   bans, and the fearless pool as a *context* rather than a constraint.
4. **Calibration of the arm that wins.** §13's best model is the worst
   calibrated (ECE 0.0286 vs 0.0093 for draft-only): the ratings are
   overconfident on their own scale and the stack's 0.851 weight is the
   correction. Refit a temperature on val before any number is shown as a
   probability, and check per-patch — the newest patches are where a shipped
   number would be read.
5. **Give the confidence gate a rank readout.** In v1 the band is crossed 5 times
   in 172,792 own-view predictions (all in `pro`/focal, 0.6 precision on n = 5),
   so precision/recall-at-high-confidence is `nan` everywhere else. §8's product
   term fixes the symptom — 69 soloQ calls at 71 % — but the *metric* is still
   the wrong one: on 4,762 pro games a 0.35/0.65 band yields 32 calls, and a
   32-call precision has a Wilson interval 0.33 wide. Report the top decile of
   confidence (n = 477 on pro) instead of a fixed band, and quote the interval.
6. **Don't add more hand-set priors before measuring the one that exists.** §8's
   `[5,5]` role prior was inert: it moved 1 % of its scale and can be deleted
   from the checkpoint with no measurable change, because the role is already an
   input and the attention can learn the same weighting from content. If a role
   prior is wanted, it has to go where the role is *not* already available (ban
   slots, a fearless pool) or be strong enough to change the softmax — but
   measure it the same way (drift + zero-it-at-inference) before crediting it
   with any part of a delta.

**What not to do:** don't quote accuracy as the headline (the prior alone gets
52 %, and a model can win accuracy by drifting to the majority side — §7's
`both/ce` pro subset does exactly that: accuracy +0.019 with log-loss *worse*
than the prior). Don't read a joint-corpus number without the per-domain block.
Don't add a blue/red swap augmentation while `b_side` exists — it zeroes it by
construction. Don't compare architectures on pro alone: at n = 4,762 a single
seed moves AUC by ±0.007, which is the size of the effect §8 measured on the
second seed (§8's role-only arm is the demonstration — it "gained" 0.006 AUC
with a parameter that provably does nothing).
