# Flat-feature MLP (`pro/model/mlp/`)

The "no attention, no set structure" baseline for both evaluation tasks.

The shipped models see the board as a **set**: ten (champion, role) tokens, a
ban block and a fearless pool, through a Siamese encoder with cross-team
attention. This sees the same board **flattened into one fixed-length vector**
and pushed through two hidden layers. Same rows, same legality rule, same loss,
same 173-way output.

| | |
|---|---|
| `next_move` | in 3,834 -> 644 -> 644 -> 173, legality-masked, 20 states/game |
| `masked` | in 1,974 -> 963 -> 963 -> 173, no legality filter, 10 states/game |
| params | **2,996,705** (`next_move`), **2,997,029** (`masked`), against a 3 M budget |
| weights | `pro/model/checkpoints/{nextpick,masking}-mlp-flat-v1/model.pt` |

The shipped width is not hand-picked. `--hidden auto` (the default) solves for the
widest two-layer trunk that still fits under `baselines.PARAM_BUDGET` by
bisection, so the row is "the largest flat MLP 3 M parameters can buy" rather
than "an MLP someone guessed at". The two tasks have different input widths
(3,834 vs 1,974), so one shared `hidden` cannot fill the same budget twice: 644
for `next_move`, 963 for `masked`, and one more hidden unit in either overruns.
Pass `--hidden N` to pin a width; the budget check still refuses an overrun.

An earlier pair at `hidden=192` (~0.81 M / ~0.45 M) was trained on identical
settings and is the capacity ablation - same split, same features, same loss,
6.7x fewer parameters on the masked arm. Re-running it needs no code change:
`--hidden 192`.

## Why it is worth a row

The one-hot board is a *complete* encoding of the state - every slot, every ban,
every mask is in the vector - so a large enough MLP could in principle represent
anything the transformer can. What it cannot do cheaply is **share weights
across slots**: a fact about slot 3 has to be relearned for slot 7, and a fact
about "enemy mid" has to be relearned as "enemy top", "enemy jungle" and so on.
That is the specific thing attention buys, and this baseline is how much it is
worth on this data.

Two caveats on reading the row, because both are easy to get wrong:

* **It is not short of next-move training examples.** Each epoch draws
  `--states` (game, prefix-position) pairs uniformly, so every one of the 20
  prefixes per game is available every epoch - more prefix diversity than the
  k=5 arm of the shipped transformer gets. A gap against the transformer is
  therefore not a gap in what it was shown.
* **`masked` applies no legality filter**, because `test_split.models.score_masked`
  does not: the encoder ranks over all 173 champions, and a baseline scored
  against a smaller candidate set would look better than it is for a reason that
  has nothing to do with the model.

## Training

```bash
# both tasks, ~15 min each on CPU at the solved 3 M width
docker compose run --rm -T dev python -m pro.model.mlp.train

# capacity ablation: pin the width instead of solving for the budget
docker compose run --rm -T dev python -m pro.model.mlp.train \
    --hidden 192 --out pro/model/checkpoints/_mlp-192

# smoke run (keep --out inside the repo: on Git Bash for Windows an absolute
# /tmp path is rewritten into the working directory)
docker compose run --rm -T dev python -m pro.model.mlp.train \
    --epochs 2 --states 20000 --out pro/model/checkpoints/_smoke
```

There is no `--val-frac` and no `--corpus`. The split is
`pro/model/test_split/split.py`, built by calling `S.build`, which asserts the
counts. A baseline trainable on different rows would be a baseline nobody can
compare against, which is the reason that module exists.

Selection is on **val top-5**, the criterion the policy trainer uses
(`rel_top5` is top-5 normalised by the frequency baseline; the normalisation is
constant across epochs, so the argmax is the same). `--device` defaults to
`cpu`: three million parameters over sparse one-hots does not need a GPU, and
taking one would evict whatever else is running.

The two arms behave differently under that criterion, and the difference is
informative rather than a tuning artefact. `next_move` keeps improving to its
last epoch. `masked` peaks early and then memorises: at 3 M its train loss falls
to 1.83 while val top-5 stops improving after epoch 6. Best-val state is deep
copied at selection, so the shipped weights are the epoch-6 ones and not the
overfit tail - but the gap is the honest reading that the masked question has
little left for capacity to buy.

## Files

| file | what it is |
|---|---|
| `model.py` | `FlatMLP`, the `MLPPredictor` wrapper, the loader, and the capacity solver (`params_for` / `largest_hidden`) |
| `train.py` | the training loop, artifact writer |

`model.py` exists so the trainer and the evaluation suite construct the *same*
module from the same `flat.json`. Two constructions of one architecture is how a
loader ends up with a differently-shaped first layer and `load_state_dict`
starts raising on a model that trained fine.

The features are not defined here - they are in `pro/model/baselines.py`,
shared with `pro/model/forest/`, so the two baselines differ only in how they
model the board and not in what they are shown.
