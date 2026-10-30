# Policy-value model (Stage 2): next-move prediction + win value

Decoupled policy + value architecture on top of the frozen SoloQ
pretrained encoder. Trains on `ProNextMoveDataset` (20 examples per
game from the 46.6k pro drafts) instead of masked-draft finetuning.

## Why a different architecture?

The finetune approach (`model/finetune/`) trains the encoder on
masked champion completion of full 10-pick pro drafts (the same task
as soloQ pretraining) plus a bolt-on win head. Results were poor:
win AUC ~0.53 (base rate 0.54), and the masked-CE task doesn't teach
next-pick prediction with bans, order, or fearless context.

This stage decouples the two objectives:

- **Policy head**: learns P(next pick/ban | draft prefix, bans,
  fearless, turn, phase, side). Trained directly on the next-move
  task — the thing a coach actually queries.
- **Value head**: learns V(partial draft state) = P(blue wins).
  Trained on the same prefix states, so the value function sees the
  distribution it will encounter at inference (MCTS search).

The encoder is **frozen** (its champion/role/synergy knowledge from
soloQ is preserved). Only the ban-context module, state embedding,
policy head, and value head train from scratch.

## Architecture

```
SoloQ pretrained encoder (frozen)
  -> slot vectors h [B,10,D]
  -> BanContext (cross-attention: slots query ban embeddings)
  -> StateEmbedding (phase + side + move_idx + first_pick + patch)
  -> pool(h) = concat(v_blue, v_red, diff, product) [B, 4D]
  -> Policy head: MLP([4D + state_dim] -> C) with legality -inf
  -> Value head: MLP([4D + state_dim] -> 1) blue-win logit
  -> Econ head (optional): MLP([4D + state_dim] -> n_econ)
```

Key design choices:

- **Ban context**: banned champions are embedded via the encoder's
  frozen lookup table, then cross-attended into the slot vectors. This
  lets each slot "see" bans when computing its representation — the
  encoder alone can't do this (soloQ pretraining excludes bans).
- **State embedding**: phase (0=bans1, 1=picks1, 2=bans2, 3=picks2),
  side to move (0=blue, 1=red), first-pick side, move index (0..19),
  and **patch id** are all learned embeddings summed together. This
  tells the policy head what kind of move to predict (ban vs pick, which
  side) and which patch's meta applies.
- **Patch conditioning** (`--no-patch` to disable): the patch string was
  previously only present in `meta`, never as a tensor, so the model
  could not represent "which champs are high-priority on this patch".
  That is fatal at move 0, where there are no picks on the board and the
  patch's ban list is the only signal — measured at 12.3% top-5 for the
  model vs 37.4% for a lookup table fit on the test patches. The vocab
  is built over all rows (`build_patch_vocab`), index 0 is UNK for
  patches unseen at training time, and the vocab list is saved in the
  checkpoint config so inference maps patches the same way.
- **Legality masking**: taken (banned|picked) and unavailable
  (fearless-picked) champions get -inf logits in the policy head. The
  target champion is never in the taken set by construction.
- **Pooling**: same concat(v_blue, v_red, diff, product) as the
  finetune model. PAD-masked means per team. The pool captures comp
  asymmetry; the state embedding adds draft-position context on top.

## Sampling and augmentation

The 20 prefixes of one game are near-duplicates: same draft, same
next-move label, different depth. Unrolling all 20 every epoch inflates
the dataset to 810k examples while the effective sample size stays at
46.6k games — which is why val loss climbed from epoch 1 onward.

- `--prefix-mode random` (default): each game contributes **one**
  example per epoch, at a move index redrawn by `set_epoch()`
  (deterministic in `seed`+`epoch`). An epoch is 80 steps instead of
  1582, so the default is 100 epochs ≈ 8k optimizer steps, roughly the
  budget of a 5-epoch run of the old mode — but each (game, prefix)
  pair is seen once instead of six times.
- `--prefix-mode all`: the old behaviour. Used for val/test always, so
  metrics stay comparable across runs (62,880 test examples).
- `--swap-aug` (default on): mirror blue↔red with p=0.5
  (`datasets.swap_sides`). Valid because the encoder is Siamese with no
  team embedding, so the mirrored draft is the same draft from the other
  side with the win label flipped. Doubles effective data and enforces
  the symmetry the architecture claims. Bans are left untouched on
  purpose: `BanContext` embeds them with no team or position signal, so
  the ban list is already team-agnostic, and a team swap doesn't change
  the chronological ban order.

## Run

```text
python policy/train.py --epochs 100                       # full run (new defaults)
python policy/train.py --epochs 15 --prefix-mode all \
    --no-swap-aug --no-patch                              # old behaviour
python policy/train.py --epochs 100 --no-patch \
    --run-name ablation-nopatch                           # patch ablation
python policy/train.py --limit-drafts 2000 --epochs 2 \
    --no-wandb --run-name smoke                           # smoke
pytest policy/test_policy.py policy/test_datasets_aug.py -q  # 30 tests ~7s
```

### Init

`--init-ckpt` (default = run1 best.pt, the recommended trunk;
`--init-ckpt auto` picks the pretraining checkpoint with the highest
val champ_top5). The encoder is loaded and frozen.

### Stages

- `--unfreeze-stage 0` (default): encoder fully frozen. Only the
  ban-context, state-embedding, policy head, value head, and econ head
  train. This is the recommended starting point.
- `--unfreeze-stage 1`: additionally unfreeze the encoder's
  `cross_team` blocks (at `--lr-enc`). Use if stage 0 plateaus and
  the cross-attention could benefit from pro-specific adaptation.
- `--unfreeze-stage 2`: additionally unfreeze the encoder's
  `intra_team` blocks. Embedding tables stay frozen throughout.

### Loss

```
total = policy_CE(next_move) + value_w * BCE(blue_win) + econ_w * MSE(econ)
```

- Policy CE: cross-entropy over C champions, with -inf masking on
  taken/banned/unavailable champs (the softmax normalizes over legal
  moves only). The target is the actual next move's champion.
- Value BCE: blue-win logit vs binary label. Side-neutral (red
  queries mirror slots). Default `--value-weight 0.5` (lower than
  policy, since policy is the primary objective for the coach UI).
- Econ MSE: standardized gold/xp at 10/15 min per side. Optional
  (`--econ-weight 0.1`); set `--n-econ 0` to disable.

### Data

`ProNextMoveDataset` (from `data/pro play/datasets.py`): turns each
20-move draft into prefix-to-next-move examples (20 per game in `all`
mode, 1 per game per epoch in `random` mode — see Sampling above). Each
example carries: partial draft (champ_ids/role_ids with PAD for
unpicked), ban list, taken_mask (legality), unavailable_mask
(fearless), turn info (move_idx, phase, side, first_pick), `patch_id`,
win_label, and the econ targets.

Note: `econ_target`/`has_econ` used to be emitted only by
`ProMaskedDraftDataset`, so the policy model's econ head was trained
against nothing at all. `build_prefix` now emits them (90.7% of rows
have econ data), and `evaluate` gates the econ metric on the head
existing, so `--n-econ 0` no longer raises.

Recency-weighted sampling (`--half-life 75d`): recent meta is
upweighted. Series-grouped train/val/test split (no leakage).

### Eval

| Metric | Meaning |
|---|---|
| `top1/5/10` | next-move champion top-k hit rate |
| `ban_top5` | top-5 on ban moves only |
| `pick_top5` | top-5 on pick moves only |
| `win_auc` | blue-win AUC on prefix states |
| `win_brier` | Brier score on win prediction |
| `econ_rmse` | standardized econ RMSE (includes the patch-drift offset) |
| `econ_rmse_centered` | econ RMSE after removing the split's mean residual |
| `econ_rmse_const` | what the best constant predictor scores on this split |
| `econ_skill` | `1 - centered/const` = R²; **> 0 means the head beats a constant** |
| `policy_loss` | mean policy CE |

Reading the econ numbers: gold/XP at 10/15 min inflate patch over patch, so
the val/test targets sit +0.8 to +1.5σ above the train-fitted
standardization. `econ_rmse` alone therefore cannot fall below
`econ_rmse_const` even for a perfect model, because the drift is in the
level and no draft-only model can see it. Judge the head by `econ_skill`.

## Checkpoints (ckpt/run_<ts>[_<name>]/)

- `best_policy.pt` — best val next-move top5
- `best_value.pt` — best val win AUC
- `latest.pt` — last epoch
- Payload: model + config + metrics + init provenance + econ stats.
  The model state dict includes the encoder under `encoder.` plus all
  new heads/modules — loadable into `PolicyValueModel(encoder, ...)`
  after building the encoder from the init checkpoint.

## Connection to frontend and MCTS

The policy head outputs a distribution over champions for the next
move given the current draft state — exactly what the frontend UI
needs for "suggest next pick/ban." The value head outputs a win
probability for any partial draft state, enabling MCTS search over
draft continuations (expand via policy, evaluate via value).

To use in the frontend: the `Predictor` class in `frontend/server.py`
needs a new code path that builds a `build_prefix`-style batch from
the UI's draft state and calls the policy head. The existing masked-
draft suggestion path can remain for comp-completion queries.

## Ban decisions: `ban_search.py` vs `ban_propose.py`

Two different framings of "which champion should I ban", and the second
one supersedes the first.

**`ban_search.py`** fits an additive `W[role, champ]` and scores each
legal ban by its negamax value. It is structurally incapable of ranking
phase-1 bans (a symmetric additive V gives every ban the same value -
proved, and observed as `spread 0.00000` over 19,131 decisions), and on
phase 2 it found the human ban within 0.0046 of optimal with a 0.9
winrate-point spread. Useful as a diagnostic; not a decision tool.

**`ban_propose.py`** implements the rule "ban what they are about to
take", with the policy proposing and a value function verifying:

1. roll the policy forward from my ban to the opponent's first pick,
   with *both* teams playing the policy (the stated premise);
2. shortlist = their most likely picks;
3. ban the top one, then let the rollout finish the draft;
4. score the completed drafts (V), and compare against the human ban
   and a random legal ban.

The rollout is what makes this work at all: `V` reads the *picked*
slots, so a ban on a fixed board changes nothing. Here the arms are
different completed drafts, which is the regime V was trained for
(0.5426 AUC on complete drafts).

Measured on the test split (12,756 phase-2 decisions, policy
`ablF-stage1`, value `value_20261030_180102`):

| arm | V | verifier |
|---|---|---|
| proposed | 0.4759 | 0.6136 |
| observed (human) | 0.4760 | 0.6148 |
| random legal | 0.4762 | 0.6126 |

`proposed - observed = -0.00007 [-0.00055, +0.00039]` under V and
`-0.00114 [-0.00412, +0.00197]` under an independently trained second
value model. The proposal changes the simulated draft in 87.9% of
decisions (mean 2.47 champion slots), so this is a real null, not a
no-op. Selecting on V instead (`--propose value`) yields +0.0173, which
bounds V's resolution well below that - the effect is genuinely near
zero, not unmeasurable.

The reason is structural: **under the premise "the opponent drafts like
me", both teams want the same champions**, so denying one to them
denies it to me. A symmetric self-play rollout cannot produce a ban
advantage. The asymmetry has to come from somewhere - opponent-specific
champion pools (player specialties), not from a better search over the
same symmetric model. That is also why phase 1 is not a target: pros
ban specialists and game plans there, which is exactly the
opponent-specific information a symmetric model lacks.

`ban_intent.py` measures the premise directly: observed phase-2 bans
land in the opponent's predicted top-5 at 29.6% vs 8.3% for a frequency
baseline (3.6x), so humans do follow this rule - but only for a
minority of bans, and phase-1 bans show no lift at all.

