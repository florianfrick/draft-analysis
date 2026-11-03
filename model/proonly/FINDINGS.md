# Findings — pro-only training and the patch embedding

Branch `exp/pro-only-patch-emb`, merged into `main`. Everything below is
measured on this repo's own data, in Docker, with the commands in
`model/proonly/README.md` and `model/test_split/README.md`.

> **Re-run 2026-11-03.** The checkpoints these numbers were measured on were
> deleted and both pro-only stages were retrained once under new names —
> `masking-proonly` and `nextpick-proonly`, in `model/checkpoints/` — on a
> corpus whose freeze moved to 2026-11-01 (46,875 drafts) and whose test split
> is now three pinned Worlds series rather than a fractional patch tail.
> Everything below this line is the record of the *previous* runs and their
> ablations; the current numbers are in `STATUS.md` §9 and
> `model/test_split/REPORT_test_split_evaluation.md`. Read §1–§4 as the
> evidence for the patch-embedding claim (which stands — the arm was not
> re-abated), not as the current checkpoint scores.

**The short version.** A 3.35M-parameter model trained on 37,313 pro drafts
alone — no solo queue, no rehearsal, random init — predicts the next draft move
at **48.4% top-5** and **3.14x** the per-move frequency table. The single
decisive ingredient is a **patch embedding**, which is worth **+9.5 to +10.0 pp
top-5** paired (p < 1e-300), and its whole effect sits in the first ban phase,
where the model goes from **1.07x to 2.43x** the frequency table. The masked
pre-training stage that precedes it is worth **+0.93 pp** — real, but an order
of magnitude smaller, and it should not be the focus of future work.

---

## 1. What the branch set out to do

Four questions, in the order they have to be answered:

1. **Can pro data alone train this?** Drop the soloQ pre-training and the
   rehearsal schedule; start from random weights on pro drafts only.
2. **Does a patch embedding help?** One learned vector per patch, added to
   every draft slot.
3. **Does masked completion work first**, as a pre-training stage for next-pick?
4. **Train on the head of each patch, test on its tail** — because that is the
   only split on which a patch embedding can mean anything (see §2).

Everything was run as a paired ablation: for each question, two arms that
differ by exactly one flag, scored on identical rows, with McNemar on
discordant pairs plus a game-level bootstrap.

---

## 2. The one change that made the question askable

The repo's default split is `split_series_by_date`: sort by date, take the last
10% as test. Under it **every test patch is unseen**, so the patch id is `UNK`
at test time and a patch embedding can only memorise. That is why the earlier
`policy-2` run measured patch conditioning as *harmful* (0.2029 vs 0.2558
top-5, `STATUS.md` §6) — the model was handed an input that was always blank
when it mattered.

`pipeline.split_within_patch` changes the question to the one deployment asks:
**you always know today's patch, and you have its first 80%.**

| | `split_series_by_date` | `split_within_patch` |
|---|---|---|
| per patch | entire patch on one side | head trains, tail validates/tests |
| test patch seen in train | never | yes, for 99 of 106 patches |
| patch id at test time | `UNK` | real row |

Series are grouped first (a series is assigned to the patch of its *first*
game, so it can never straddle), sorted by `(date, gameid)`, and the test then
val windows are filled backwards from the end so test is the newest games.
46,813 drafts → **37,313 / 4,737 / 4,763**; 99 of 106 patches in test; 0 series
straddling; 7 patches (194 rows) below the 40-row floor stay whole in train.

The repo default is untouched — existing scripts still get the old split.

---

## 3. What was trained

Two stages per arm, from scratch, D=256 / 2 intra + 1 cross / 8 heads /
dropout 0.1. Uniform recency weights everywhere (the 75-day half-life the other
stages use would make old patches unlearnable when the test window is the tail
of *every* patch).

| run | stage | patch emb | seed | corpus |
|---|---|---|---|---|
| `m-patch-s0` / `-s1` | masked | yes | 0 / 1 | 2022-2026 |
| `m-nopatch-s0` / `-s1` | masked | no | 0 / 1 | 2022-2026 |
| `p-patch-s0` / `-s1` | next-pick | yes | 0 / 1 | 2022-2026 |
| `p-nopatch-s0` / `-s1` | next-pick | no | 0 / 1 | 2022-2026 |
| `m-all-patch-s0` / `-nopatch` | masked | yes / no | 0 | 2017-2026 |
| `p-scratch-patch-s0` | next-pick | yes | 0 | 2022-2026, no masked init |

Every pair differs by exactly one flag. The patch table is zero-initialised, so
patch-on and patch-off are **bit-identical at step 0** — the ablation is not
"two models that happened to diverge".

---

## 4. Results

### 4.1 Masked completion

Test: 4,763 games, **8,060 masked slots**. Baseline = per-slot popularity fit
on train (0.4120) — the strong null, because in a pro draft a slot maps to a
fixed role.

| arm | seed | top-1 | top-5 | top-10 | rel@5 |
|---|---|---|---|---|---|
| patch | 0 | 0.2089 | **0.5669** | 0.7484 | 1.376x |
| patch | 1 | 0.2133 | **0.5712** | 0.7509 | 1.386x |
| no patch | 0 | 0.1911 | 0.5303 | 0.7083 | 1.287x |
| no patch | 1 | 0.1872 | 0.5163 | 0.7061 | 1.253x |

Paired on identical slots: **+3.66 pp** (p=7.7e-14, game-bootstrap CI
[+2.71, +4.56]) and **+5.50 pp** (p=6.5e-29, [+4.56, +6.48]). Seed spread
*within* an arm is 0.4-1.4 pp, so the gap is 3-4x the noise it has to beat.

### 4.2 Next-pick

Test: **95,260 states** (20 prefixes × 4,763 games). Baseline = per-move
popularity (0.1543). Same-patch leave-one-out reference = 0.1576.

| slice | patch s0 | patch s1 | no patch s0 | no patch s1 | rel@5 (patch / none) |
|---|---|---|---|---|---|
| top-1 | **0.1763** | 0.1757 | 0.1360 | 0.1372 | — |
| top-5 | **0.4839** | 0.4868 | 0.3894 | 0.3873 | 3.14x / 2.52x |
| top-10 | 0.6676 | 0.6679 | 0.5496 | 0.5488 | — |
| ban1 | **0.4246** | 0.4339 | 0.1860 | 0.1860 | **2.43x / 1.07x** |
| pick1 | 0.5564 | 0.5552 | 0.4705 | 0.4647 | 3.64x / 3.08x |
| ban2 | 0.3656 | 0.3700 | 0.3594 | 0.3684 | 2.95x / 2.90x |
| pick2 | 0.5828 | 0.5801 | 0.6029 | 0.5920 | 3.72x / 3.85x |

Paired: **+9.45 pp** (p<1e-300, CI [+9.16, +9.74]) and **+9.95 pp**
(CI [+9.66, +10.24]). Per-year deltas are uniform (-6.8 to -12.6 pp), so the
pooled number is not a year-mix artefact.

### 4.3 The mechanism is ban1, and it is the whole story

This is the part worth remembering, because it is a mechanism test rather than
a headline:

* **ban1: 1.07x → 2.43x.** With an empty board there is nothing but the patch
  to go on. Without a patch input the model sits *at* the frequency table
  (1.07x); with one it is 2.4x. This is the same slice the soloQ-based policy
  lost at 0.98x — and the reason it lost was the protocol, not the idea.
* **pick2: 3.85x → 3.72x — the patch embedding makes it slightly *worse*.**
  Both seeds (-2.0 and -1.2 pp). Once the board is full, composition dominates
  and 107 × 256 patch parameters are a small net negative where they carry no
  signal.

So the patch vector is **phase-specific**: indispensable in ban1, neutral in
ban2, mildly harmful in pick2. That single observation drives the first item of
the improvement plan (§6.1).

### 4.4 More data

The folder also holds 2014-2021 exports. `CSV_GLOB` is pinned to `202[2-9]_*`
so they cannot silently swap the corpus; `--corpus all` opts in (81,161 drafts,
2017-2026, 64,737 / 8,198 / 8,226, own cache file).

| corpus | patch rel@5 | no patch rel@5 | Δ top-5 (paired) |
|---|---|---|---|
| 2022-2026 (37,313 train) | 1.376x | 1.287x | +3.66 pp (p=7.7e-14) |
| 2017-2026 (64,737 train) | **1.493x** | 1.346x | **+5.51 pp** (p=1.1e-44) |

1.7x the drafts buys +0.12 rel (patch) and +0.06 rel (no patch), and the patch
effect does not shrink. The model is **data-limited**, which the epoch curve
says independently: val top-5 peaks at **epoch 6** and decays to 0.487 by
epoch 40 — 37k drafts against a 2.5M-parameter encoder overfits after ~440
steps. Selection on val is what keeps the reported checkpoints honest.

Absolute top-5 *falls* slightly on the wider corpus (0.5669 → 0.5565) only
because 2017-2021 is harder; the `rel` column is the comparable one. Seed 0
only — a direction, not a measurement.

### 4.5 Against the kept pipeline, on the same rows

Same within-patch test split, positive = pro-only better:

| rows | masked: `best_mask.pt` → pro-only | next-pick: `ablF` → pro-only |
|---|---|---|
| all | 0.3710 → **0.5669** (+19.6 pp) | 0.2671 → **0.4839** (+21.7 pp) |
| 2022 | 0.2721 → 0.5839 (+31.2) | 0.1309 → 0.4821 (+35.1) |
| 2024 | 0.3265 → 0.5800 (+25.4) | 0.2350 → 0.4650 (+23.0) |
| 2026 | 0.4939 → 0.5431 (+4.9) | 0.4091 → 0.4951 (+8.6) |

**Read the rows, not the total.** The kept stages are trained with a 75-day
recency half-life and their protocol only ever asks them about recent patches,
so the pooled gap mostly says "the old model was deliberately told to ignore
2022". On 2026 — the deployment target — pro-only is +4.9 / +8.6 pp, and even
that is confounded by the weighting difference. The only clean cross-pipeline
statement is the paired one above, where nothing differs but the flag.

### 4.6 The held-out series

The 2026 Worlds final, **BLG vs HLE** (`LOLTMNT02_486564` / `_486571`, patch
16.20, fearless on), scored by the series walkthrough that is now
`model/test_split/`. Both caches stopped at 2026-10-24 23:19:46 and contained
neither gameid; the 2026 CSV was appended to afterwards, so the series was a
real holdout for every checkpoint. (As of 2026-11-03 the corpus pin moved past
these games and they are the pinned test split instead — see the re-run callout
above.)

| model | top-1 | top-5 | mean rank | ban1 top-5 |
|---|---|---|---|---|
| `pro-patch-s0` | **0.225** | **0.675** | 7.3 | **0.58** |
| `pro-scratch-s0` | 0.200 | 0.675 | 7.8 | 0.58 |
| `pro-patch-s1` | 0.150 | 0.600 | 5.9 | 0.33 |
| `pro-nopatch-s0` | 0.100 | 0.550 | 13.2 | 0.17 |
| `policy-ablF` | 0.100 | 0.450 | 17.1 | 0.17 |
| patch-16.20 frequency table | 0.050 | 0.350 | 13.6 | 0.33 |

40 states, so this is an index, not a measurement — but the ban1 column
reproduces the ablation on data no model had seen, which is the useful part.

One confound is unavoidable and is stated first in the report: patch 16.20 has
196 corpus drafts, and under the **date** split all 196 are test rows, so
`policy-ablF` trained on **zero** of them (and has no patch input), while the
pro-only models trained on **155**. The head-to-head is a *deployment*
comparison — a patch ships, you have its first 80%, predict the rest — not a
like-for-like one.

---

## 5. `pro-patch-s0` is the model we ship

| | |
|---|---|
| checkpoint | `model/proonly/ckpt/run_20261101_183519_p-patch-s0/best.pt` |
| params | 3,350,559 (2.5M encoder + policy head) |
| test top-1 / top-5 / top-10 | 0.1763 / 0.4839 / 0.6676 |
| rel@5 | 3.14x the per-move frequency table |
| val selection | epoch 32, `rel_top5` 3.1555 |
| train time | ~6 min masked + ~9 min policy on one 8 GB card |

### 5.1 What "best" does and does not mean

`pro-patch-s0` is the best model **on the held-out series** and the anchor of
every ablation. It is **not** distinguishable from `pro-patch-s1` on the test
split: 0.4839 vs 0.4868 top-5, a 0.29 pp gap, inside the seed spread. So the
choice of seed is not a quality claim — do not read it as one. s0 is shipped
because it is the arm whose no-patch and scratch controls exist, so every claim
in this document is anchored to it.

What *is* a real quality claim: the **configuration** (pro-only, within-patch
split, patch embedding, masked init) beats every alternative measured here.

### 5.2 The recipe, step by step

1. **Corpus** — 46,813 pro drafts, 2022-2026, built once into `pro_drafts.pt`.
   `CSV_GLOB` is pinned so newly-arrived CSVs cannot swap the corpus out from
   under existing numbers.
2. **Split** — `split_within_patch`, 37,313 / 4,737 / 4,763. This is load-
   bearing: on the date split the patch id is `UNK` at test and the whole
   result inverts.
3. **Stage 1, masked** — 15% champ masks + 0.3 · role CE on complete drafts.
   Batch 512, lr 1e-3, warmup + cosine, 40 epochs, select on val
   `rel_champ_top5`. It peaks at epoch 6.
4. **Stage 2, next-pick** — encoder initialised from stage 1's `best.pt`;
   lr 1e-3 on the head and 1e-4 on the encoder; 32 epochs; CE over 173 with
   illegal moves at −inf; select on val `rel_top5`.
5. **Patch embedding** — one zero-initialised 256-d vector per patch (107 rows,
   row 0 = `UNK`), added to every slot before the LayerNorm. Zero init means
   step 0 is exactly the no-patch model, so the model has to *learn* to use the
   patch rather than start from noise.
6. **Uniform recency weights.** The other stages use a 75-day half-life. Here
   that would make old patches unlearnable while the test window is their tail.

### 5.3 Which of those decisions actually carry the result

| decision | evidence |
|---|---|
| within-patch split | on the date split, the same input measured **-5.3 pp** (`policy-2`); here it is **+9.5 pp** |
| patch embedding | +9.45 / +9.95 pp paired, p<1e-300 |
| one prefix per game per epoch | prevents 20x weighting of early-move states |
| legality as −inf | never proposes an illegal champion |
| selection on `rel_top5`, not raw top-5 | rewards beating the frequency table, not memorising the popular patch |
| masked pre-training | **+0.93 pp** — keep, but stop investing here |
| uniform weights | required by the split, not a tuning choice |

---

## 6. Where it fails

1. **pick2 is worse than the no-patch model** (§4.3). The patch vector carries
   no signal once the board is full, and it costs 1.2-2.0 pp.
2. **It overfits after ~440 steps.** 2.5M parameters on 37k drafts; val peaks
   at epoch 6 of 40. Absolute top-5 is well below what the data supports.
3. **A brand-new patch gets no prior at all.** `patch_to_id` maps an unseen
   patch to `UNK`, so on the day a patch ships the model falls back to the
   frequency table — precisely the ban1 1.07x failure mode this branch fixed
   for *seen* patches. This is the biggest deployment gap.
4. **Masked completion cannot express fearless.** `ProMaskedDraftDataset` never
   receives `unavailable`, so on the held-out game 2 it named an already-picked
   champion in 56 of 80 slots. The next-move models, which do get the mask,
   break fearless zero times.
5. **No usable value head.** All four in the repo are flat or noisy on the
   series; `policy-value` swings 0.55 → 0.01 → 0.99 inside one draft, matching
   `STATUS.md` §3.
6. **One architecture size, one corpus for the main grid.** Everything is
   D=256; the wider corpus and the scratch control are seed 0 only.

---

## 7. Preliminary plan to improve performance

Ordered by expected gain per unit of work, with the measurement that would
settle each one. All of them reuse the same within-patch split and the same
paired protocol, so results stay comparable to the tables above.

### 7.1 Gate the patch vector by phase — do this first

**Why.** §4.3 is unambiguous: ban1 +2.43x, ban2 +0.05x, pick2 −0.13x. The
model is being forced to use one vector where it needs it and ignore it where
it does not.

**What.** Multiply the patch vector by a learned scalar per phase (4 params) or
per move (20 params), initialised to 1 so step 0 reproduces today's model:

```
x = x + g_φ · patch_lookup(patch_id)
```

**Cost.** Four parameters, one training run, same schedule.
**Success test.** pick2 recovers to ≥ 0.6029 (the no-patch number) while ban1
holds ≥ 0.4246. Read the pick2 row specifically, not the pooled top-5.

### 7.2 Make the encoder smaller, not bigger

**Why.** Val peaks at epoch 6 and decays 0.558 → 0.487 by epoch 40. The model
is not capacity-starved; it is capacity-rich and data-poor.

**What.** D=128 (≈0.6M params), or D=256 with dropout 0.25-0.3 and weight
decay 0.05, or 1 intra + 1 cross layer. Train the same 40 epochs and read the
*whole* val curve, not the selected point.

**Success test.** Absolute top-5 above 0.4839 at the same or lower parameter
count. Caveat: the patch-vs-no-patch gap must be **re-measured** at the new
size — the patch vector is the only signal that cannot be memorised from
composition, so it could get relatively *more* valuable, not less.

### 7.3 soloQ as rehearsal, not initialisation

**Why.** The kept pipeline used soloQ as an *init* and lost ban1 (0.98x), but
that run was confounded by the recency protocol and by an unseen test patch.
The clean version of the question has never been asked: does soloQ add
composition knowledge *on top of* the pro-only model?

The evidence says composition is exactly where pro-only is weakest — pick1 and
pick2, and pick2 is where the patch vector hurts. That is the complement this
model is missing.

**What.** Same recipe, but replace 10-20% of each batch with soloQ drafts
carrying `patch_id = UNK` (so the patch table is untouched by them). Two
variants worth separating:
* soloQ in the **masked** stage only, then the usual pro-only policy stage;
* soloQ in **both** stages.

**Success test.** pick1/pick2 up on the same 95,260 pro test states, ban1
unchanged. If ban1 moves, the soloQ data is leaking patch information it does
not have and the result is suspect.

**Risk.** SoloQ champ select has a different structure (no ban phase, different
pick order, different state distribution). That argues for using it in the
masked objective — where the task is "complete a partial draft" and the format
difference is smaller — rather than in the policy objective.

### 7.4 LLM: predict the patch vector from patch notes

**Why.** Failure #3 is the one that costs money: on a new patch the model has
no prior at all. And `STATUS.md` §P3.1 already established what does *not*
work — feeding patch-note text into the encoder through an input-side adapter
cannot produce a cold-start prior. The fix is on the **output side**.

**What.** Don't give the encoder text. Instead:
1. An LLM reads the patch notes and emits a structured per-champion delta
   (buffed / nerfed / mechanic change / unchanged, with magnitude);
2. a small learned projection `W_a` maps that delta vector into the 256-d
   patch-embedding space;
3. the predicted vector initialises the new patch's row, which is then
   fine-tuned on the patch's first games as usual.

**How to measure it honestly.** Hold out a patch **entirely**: train on
2022-2026 minus patch X, then compare three cold-start rows for X on its first
N drafts — (a) `UNK`, today's behaviour; (b) text-predicted via `W_a`;
(c) an oracle row trained on all of X's drafts, as the ceiling. Patch 16.20 has
196 drafts, so this is a real experiment, not a toy.

**Second, separate use.** An explanation layer — turn the top-5 plus ranks into
a rationale. `frontend/llm.py` already does some of this. That is product
value, not accuracy; do not conflate the two when reporting.

**Do not** use an LLM as a judge of ban quality without a calibration set. It
is unmeasurable against this test split and would be the easiest way to
generate a confident-looking number that means nothing.

### 7.5 Rebuild the value head as its own stage

**Why.** All four value heads in the repo are unusable (§6.5), and the failure
mode is diagnostic: `pretrain-win` collapses to 0.05 at move 7 because PMA
pools over a near-empty team. The policy's pooled 4D vector is the same
problem one layer down.

**What.** A prefix-state encoder with a **mask-aware pool** (mean over present
slots, plus a presence count), trained separately, evaluated with Brier score
and a calibration curve on the within-patch test split — not accuracy, and not
on 40 states.

**Success test.** Brier below the base rate's, and monotone calibration across
deciles. Until then, don't surface a win probability in the frontend.

### 7.6 What not to do

* Don't re-run patch conditioning on the global date split — it makes the
  input `UNK` at test and the answer is meaningless (this was already measured
  once, at −5.3 pp).
* Don't scale the model up on 37k drafts; the epoch curve says that is the
  wrong direction. **Measured:** D512/6/3 from scratch is *worse* than
  D256/2/1 (masked top-5 0.5139 vs 0.5669), and D512/6/3 *with* soloQ
  pretraining only ties it (0.5648, p=0.674).
* Don't add the patch vector to pick2 without a gate.
* Don't quote the pooled +19.6 / +21.7 pp cross-pipeline gap as a quality
  result; quote the 2026 rows or, better, the paired same-flag numbers.
* Don't feed patch-note text directly into the encoder — measured to fail.
* Don't expect soloQ pretraining to carry a pro-only model further than the
  pro-only model already is — **measured:** a dead heat, both stages.

### 7.7 Reproducibility traps that cost time

Two training facts here bite reproduction, and both look like a regression
when they are not:

* **`--epochs` sets the LR schedule, not just the duration.** `total_steps =
  steps_per_epoch × epochs`, warmup is 3% of it, then cosine to zero. At 40
  epochs the warmup (87) is longer than one epoch (73 steps), so the whole
  first epoch is ramp-up; at 2 epochs it is 4 steps and the run is at full LR
  almost immediately — and the *shorter* run looks better (train 3.34 vs 3.52
  at epoch 1). A re-run at a different `--epochs` is a different experiment,
  not a reproduction; repeat it exactly.
* **`DataLoader(shuffle=True)` draws its sampler seed from the global torch RNG
  at construction.** So "build the loaders" must stay before "build the model"
  (it does, in both trainers), or the shuffle — and the whole trajectory —
  changes. `torch.load` consumes no global RNG, so reading a checkpoint earlier
  is safe.


---

## 8. What is in the repo after the merge

| path | what it is |
|---|---|
| `model/proonly/` | the winning recipe: `train_masked.py`, `train_policy.py`, `common.py`, `compare.py`, tests |
| `model/checkpoints/` | the kept artefacts (git-ignored): `masking-proonly`, `nextpick-proonly`, `masking-soloqueue` as the init, plus the finetune and soloQ-policy classes |
| `model/proonly/reports/` | every run's `test_report.json` and every paired comparison, including the pruned arms |
| `model/test_split/` | the per-class evaluation on the three pinned Worlds series: `eval_test_split.py`, `render_test_split.py`, the report, tests |
| `STATUS.md` §8-§9 | the condensed record, with the same numbers |

Nine further runs (seed 1, the wider 2017-2026 corpus pair, the scratch
control, and the six pretrained-initialisation arms) were removed as losing or
relocated arms. Their numbers are preserved in `model/proonly/reports/` and in
this document; only their weights are gone. The three that remain are the
shipped model, the control that proves the ablation, and the init.

The six pretrained-initialisation arms that this folder used to hold were moved
to `model/finetune/` and `model/policy/`: the pretrain → finetune → policy
pipeline is now the place that studies whether soloQ initialisation helps, and
this folder is from-scratch only by design.

### The corpus is frozen, and that is load-bearing

`pipeline.CORPUS_MAX_DATE = "2026-10-24 23:19:46"` pins the corpus to the date
every number above was measured on. This is not bookkeeping: during the merge
an ordinary test run called `build_rows()`, saw that the 2026 CSV had changed,
rescanned, and overwrote `pro_drafts.pt` with 46,875 drafts — the two Worlds
final games included, i.e. it trained the corpus on its own holdout and moved
every test number at the same time.

With the pin, a rescan is idempotent: games after the cutoff are counted as
`after_corpus_freeze` and dropped, and rebuilding returns exactly the 46,813
drafts the models saw. Raising the pin is a deliberate act that requires
re-measuring everything, which is the correct amount of friction.
