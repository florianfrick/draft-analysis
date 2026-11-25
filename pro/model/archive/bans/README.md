# Ban-aware next-move policy

**Shipped on 2026-11-07.** `nextpick-bans-aware-through-wlds-ig-tes` is the
frontend's next-move default (`webapp/server.py`, `POLICY_DEFAULT`), and the
three flags are off by default on `PolicyValueModel` so every older checkpoint
still rebuilds unchanged. Read the caveat at the end of "Where the gain comes
from" before quoting the number: it is **one seed**, and the single-switch arms
are flat. The honest summary is "more correct, and measured not worse".

The question: the next-move model is told *that* ten champions are gone, but is
it built to *reason* about which ones? A counterpick being unavailable should
raise the value of the champion it counters. If the model only learns "these
are illegal", it is enforcing a constraint, not making a read.

This branch changes the architecture so bans are an input to the decision, and
then measures the difference against the shipped model on the same split.

Terminology, because "pool" is overloaded: **fearless pool** means the
champions already picked in earlier games of the series; **pooling / pooled**
means the set reduction (PAD-masked mean or PMA), never a champion set.

## What was actually missing

Three gaps, all of them in `solo_queue/model/policy/model.py`:

1. **The candidate's own row never sees the ban set.** The head is
   `Linear(pooled 4D + state → 173)`: one fixed learned row per champion. So
   "X is banned, therefore Y is better" has to be expressed as a shift in a
   single pooled vector — a rank-limited path that must average over every
   ban. `STATUS.md` §6 had already diagnosed this exact defect for champion
   attributes ("the adapter conditions the *input*, while the predicted
   champion's own row never sees `attrs_c`. Fix on the output side") and never
   applied it to bans.
2. **Bans were a team-agnostic unordered set.** `build_prefix` discarded
   `team` for ban moves, and `swap_sides` documented that it deliberately does
   not permute `ban_ids` *because* the model treats bans as team-agnostic. Your
   own ban and the enemy's ban are not the same fact.
3. **Fearless was an output mask, never an input.** 10,023 of 46,879 rows
   (21%) carry a non-empty fearless pool, mean 15.6 champions, sizes exactly
   10 / 20 / 30 / 40 (games 2–5 of a series). The model saw none of it:
   `unavailable_mask` only set logits to `-inf`.

## What was added

Three flags on `PolicyValueModel`, all off by default so every pre-existing
checkpoint rebuilds the module set it was trained with. Cost: **3.35M → 3.37M
parameters (+16k)**.

| flag | what it does | effect at step 0 |
|---|---|---|
| `ban_side` | side embedding per ban token, derived from `(slot, first_pick)` | exactly zero |
| `ban_fearless` | fearless-pool champions join the ban path as tokens; fearless-pool size enters the state vector | **not** zero — the tokens are real champion embeddings |
| `counter_rank` | output-side `logits_c += e_cᵀ M (mean gone champion)`, `M = ABᵀ` at rank r | exactly zero |

`ban_side` and `counter_rank` are zero-initialised, which is what makes
"ban-aware on/off" a single-variable ablation rather than two different
networks. `ban_fearless` cannot be, and the README says so instead of pretending
otherwise: adding fearless-pool tokens changes the forward pass by construction,
because the tokens carry champion identity. `test_bans.py` pins both facts.

The forward pass with all three switches is diagrammed in
`pro/model/proonly/README.md` ("Stage 2, ban-aware variant"), including which
tensors the counter term reads.

### Why the side is *derived* and not carried

`interleaved_moves` fixes the ban order as a function of `first_pick`: bans1
alternates first-pick team first (slots 0–5), bans2 second-pick team first
(slots 6–9). So which side made ban slot *i* is a pure function of
`(slot, first_pick)`, and `first_pick` is already swapped by `swap_sides`.

Deriving it means the mirror augmentation flips every ban's side for free, with
no ban block to permute and get wrong. Storing the side in the batch would have
required swapping the block too, and a missed swap would mirror every ban onto
the wrong team with nothing to catch it. Verified against the corpus:
**468,790 ban slots, 0 mismatches**.

### What was deliberately *not* done

- **No new loss.** Cross-entropy already supervises "pick Y given X banned"
  about 460k times (10 pick moves × 46.9k drafts, and bans are always on the
  board before the first pick). The bottleneck was capacity to represent the
  pairwise interaction, not supervision.
- **No fearless-pool roles in the model.** Every one of the 156,370
  fearless-pool champions has a recoverable role (`build_fearless_roles`,
  0 fallbacks), and the role is the interesting half of that set. The serving
  path knows *which* champions are gone but not what role they were played in —
  `infer.build_board` takes a bare list of champion ids. A role-tagged
  fearless-pool token would therefore be real value in training and FLEX at
  serving: a silent train/serve skew in the exact channel this branch exists to
  make correct. Left as a follow-up that needs a client change first.
- **No bans as encoder slots.** Adding tokens to the 10-slot Siamese trunk
  would break the pretraining contract and the frozen-trunk story for a channel
  that cross-attention already covers.

## Result

Same split, same init (`masking-proonly-randomvalsplit-…`), same seed (0),
40 epochs — the only variable is ban-awareness. `control` is a fresh run with
all three flags off.

| model | top-1 | top-5 | top-10 | rel top-5 |
|---|---|---|---|---|
| shipped `nextpick-proonly-randomvalsplit` | 0.1832 | 0.4995 | 0.6801 | 3.284 |
| `nextpick-bans-control` (this branch, flags off) | 0.1831 | 0.4995 | 0.6801 | 3.285 |
| `nextpick-bans-aware` (all three on) | **0.1845** | **0.5056** | **0.6838** | **3.324** |

The control reproducing the shipped model to 4e-5 is the check that the split,
init and seed are matched — without it the comparison would be two runs of the
same script, not two arms of one experiment.

Paired on the same 95,160 states / 4,758 games (`compare.py --split latest-val`):

| | delta | McNemar p | game bootstrap 95% CI |
|---|---|---|---|
| top-1 | +0.0014 | 0.161 | [−0.0005, +0.0031] |
| top-5 | **+0.0060** | **4.3e-08** | **[+0.0039, +0.0082]** |
| top-10 | +0.0037 | 1.6e-04 | — |

So performance is not merely "similar" — it is better on top-5 and top-10 by
more than the seed spread on these rows. Read the attribution section before
believing the size of it: the four arms do not agree that any single switch
produces the gain.

By year (top-5 delta, control → aware): 2012 +0.47, 2013 +0.32, 2014 +0.11,
**2015 +1.06, 2016 +1.00 pp**. The gain is 3–10× larger in the two years
Fearless Draft exists, which is where the fearless-pool channel can contribute.

## The probe — why the accuracy number is not the evidence

A better top-5 does not prove ban-awareness. The legality mask guarantees no
banned champion is ever suggested, so a ban-blind model looks correct on every
ban metric the repo has had. `probe.py` separates the information channel from
the constraint channel: it replaces `ban_ids` with PAD while leaving
`taken_mask` intact, so the model still cannot pick those champions — it just
no longer knows which ones they are.

On the same 47,580 pick states:

| | control | aware |
|---|---|---|
| pick top-5 | 0.5801 | 0.5831 |
| top-5 with ban content hidden | 0.5529 | 0.5494 |
| **cost of hiding the bans** | **−2.72 pp** | **−3.37 pp** |
| top-5 with the fearless pool hidden | 0.5801 (±0.0000) | 0.5809 (−0.22 pp) |
| one-ban counterfactuals: top-1 changes | 7.08% | **9.06%** |
| logit-change spread across candidates | 1.172 | 1.197 |

- The fearless-pool blindfold on the control is **exactly** +0.0000 — proof the
  channel did not exist before, and that the test can detect its absence.
- Removing one ban now flips the model's top-1 recommendation 28% more often
  (n = 361,608 counterfactuals). That is the question the branch is about: *if
  that champion were not gone, would you pick differently?*
- The cost of hiding bans grows with how many bans are on the board: +0.45 pp
  more than control at 6 bans, +0.94 pp more at 10. Monotone in ban
  information, which is what a model that uses bans should look like.

Honest caveat on `spread` (1.172 → 1.197): the increase is small. The
per-candidate *structure* of a ban's influence did not change much; what
changed is how often and how strongly the model responds at all. Do not read
that row as "the counter matrix produced a sparse candidate-specific map" — it
did not, measurably.

### Where the gain comes from — and no single switch earns it

Each arm run alone, same split / init / seed:

| arm | top-1 | top-5 | top-10 | delta top-5 vs control | p |
|---|---|---|---|---|---|
| control (nothing) | 0.1831 | 0.4995 | 0.6801 | — | — |
| `--counter-rank 32` only | 0.1837 | 0.5003 | 0.6796 | +0.0008 [−0.0013, +0.0029] | 0.485 |
| `--ban-fearless` only | 0.1835 | 0.5006 | 0.6811 | +0.0011 | — |
| all three | **0.1845** | **0.5056** | **0.6838** | **+0.0060 [+0.0039, +0.0082]** | 4.3e-08 |

**This is the uncomfortable result and it should not be smoothed over.** The
two mechanisms the branch was built around each add about a tenth of a point
on their own — the counter matrix +0.08 pp with a CI straddling zero and
p = 0.485, the fearless pool +0.11 pp — and the three together add +0.60 pp.
Nothing here supports "the counter matrix works"; equally, nothing supports
"the fearless pool works". The combination does, and a single seed cannot tell
an interaction apart from a lucky draw.

Two readings, and the data does not choose between them:

1. **The switches interact.** Side attribution separates "banned this game"
   from "gone this series", which is what makes the fearless-pool tokens mean
   something different from the ban tokens. Neither channel alone gives the
   model that distinction, so neither alone moves the number.
2. **The aware run drew a favourable seed.** The repo's own history
   (`STATUS.md` §6) is that 0.3–0.4 pp can be seed-to-seed spread. +0.60 pp
   with a paired CI of [+0.39, +0.82] clears that bar *on these rows* — a
   paired test measures "B beats A here", not "this replicates".

**A second seed for `control` and `aware` is the next experiment, not an
optional one.** Until it runs, treat +0.60 pp as promising and unconfirmed,
and treat the two flat single-switch arms as the strongest available evidence
that the mechanism story in this branch is not yet demonstrated.

Why the counter matrix plausibly does nothing, rather than being broken:
- The head already has 173 output rows fed by a pooled vector, so it can
  express "the pooled ban summary looks like this → boost Y". The counter term
  adds a *specific* functional form — bilinear in the encoder's champion
  embeddings — and that form only helps if those embeddings are aligned with
  counter semantics. Nothing in the pretraining objective asks them to be: the
  masked-completion task learns "what is this champion", not "what does this
  champion beat".
- `M = ABᵀ` at rank 32 is ~16k parameters trained by the same CE on 42k
  drafts. If pairwise structure is there to be found, this is a thin budget —
  but raising the rank is a hypothesis, not a fix, and the flat result at rank
  32 does not predict rank 128.
- It is wired and it is not inert: one-ban counterfactuals flip the top-1 more
  often (7.08% → 9.06%), so per-ban sensitivity did increase. It simply did not
  convert into accuracy.

### The fearless pool slice

Split by whether a fearless pool exists (n = 37,440 pick states without,
10,140 with):

| | control | aware | delta |
|---|---|---|---|
| no fearless pool | 0.5804 | 0.5826 | +0.22 pp |
| fearless pool present | 0.5792 | 0.5848 | **+0.56 pp** |

The gain is 2.5× larger where a fearless pool exists, but not exclusive to
those rows — the ban-side and counter channels act everywhere. Note also that
on rows with a fearless pool the aware model loses *less* when bans are hidden
(−1.70 pp vs −3.68 pp): with the fearless pool visible it has a second source
of "what is gone", so removing one source hurts less. That is a consequence of the design, not a defect, and it is
the opposite of what a naive "it uses bans more" story would predict — which is
why the aggregate blindfold number, not this slice, is the headline.

## Reproduce

```bash
# both arms (~22 min each on a 4060)
docker compose run --rm dev bash pro/model/archive/bans/run_arms.sh

# paired comparison on the shared val split
docker compose run --rm dev python pro/model/proonly/compare.py --task policy \
    --a pro/model/archive/checkpoints/nextpick-bans-control-through-wlds-ig-tes/best.pt \
    --b pro/model/archive/checkpoints/nextpick-bans-aware-through-wlds-ig-tes/best.pt \
    --split latest-val --new-scope all --val-split random --other all --other-random

# the probe (add --no-ablation to skip the ~10x counterfactual pass)
docker compose run --rm dev python pro/model/archive/bans/probe.py \
    --ckpt pro/model/archive/checkpoints/nextpick-bans-control-through-wlds-ig-tes/best.pt \
    --ckpt pro/model/archive/checkpoints/nextpick-bans-aware-through-wlds-ig-tes/best.pt \
    --split val --val-split random --no-test
```

Evidence lives in `pro/model/archive/bans/reports/`; weights are git-ignored, so the
numbers here are the record. Tests: `pytest pro/model/archive/bans -q`.

## Caveats

- **The probe runs on the val split**, which is also what selected `best.pt`
  for both arms. The comparison is paired and same-protocol, so the delta is
  meaningful; the absolute numbers are mildly optimistic for both. This corpus
  has no honest test split left under `--no-test` (that is what `--no-test`
  means), so a held-out number needs a retrain with the pinned series held out.
- **One seed per arm, and the arms disagree.** This is the biggest caveat in
  the document. The repo's history is that 0.3–0.4 pp differences can be seed
  noise; `control` and `aware` differ by +0.60 pp with a game-level CI of
  [+0.39, +0.82], but the two single-switch arms sit at +0.08 and +0.11 pp.
  Until a second seed is run for `control` and `aware`, +0.60 pp is promising
  and unconfirmed, and the mechanism is not demonstrated.
- **The side channel cannot be used by a ban-agnostic client.** `infer.py` now
  derives sides from `(slot, first_pick)` internally, so serving gets it for
  free — but only if the client sends bans in chronological order. A UI that
  sends bans as an unordered set will get sides that are internally consistent
  and factually wrong.
- **The counter matrix is not interpretable on its own.** `M = ABᵀ` at rank 32
  is a learned bilinear form over the encoder's champion embeddings; reading
  "X counters Y" off it would need the embedding basis, which is not
  orthonormal. Nothing here claims the learned pairs match human counter logic.
