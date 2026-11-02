# SoloQueue Pretrained Model — I/O Contract

> Pretraining stage: learn champion / role / synergy / counter
> representations from **completed high-elo SoloQ drafts on a single
> patch**, via **masked-draft modeling (champion + role heads only)**.
> Pro-draft order / fearless / next-pick is a *separate finetune stage*.
> Game-outcome / multitask heads (win prob, duration, gold, kills) are
> **deferred** and NOT part of this contract.

Source of truth for architecture: `../README.md` (DraftSlotEmbedding
`champ + role` sum -> LayerNorm, intra-team self-attention -> inter-team
cross-attention -> slot-level champion + role heads;
set/permutation-invariant; Siamese blue/red). Decided revisions: **no team
embedding** (team from slot index only), vocab `REAL + MASK + PAD = 175`,
role head outputs exactly the 5 real roles, BERT-style 10% role-visible
masks, hard CE losses, >= 1 mask per draft, plus an opt-in value path
(PMA pooling + blue-win head, Sec 4.3) that is off by default.

This file freezes the **tensor shapes, dtypes, vocabs, and invariants**
the future data pipeline/loader and `pretraining/soloq_model.py` MUST satisfy.
`tests/test_soloq_contract.py` encodes the same contract as checks.

## 1. Draft representation

A SoloQ example is **one completed 5v5 draft** (10 picks, no ordering).

- `N_SLOTS = 10` = 5 Blue + 5 Red.
- Canonical flat order: `[B_TOP,B_JGL,B_MID,B_ADC,B_SUP,R_TOP,R_JGL,R_MID,R_ADC,R_SUP]`, idx `0..9`.
- Team-split view: flat `[B,10,...]` reshapes to `[B,2,5,...]` with dim=1 team (0=Blue,1=Red), dim=2 role order TOP,JGL,MID,ADC,SUP.
- Positional embeddings: **NONE**. Permutation-invariant within each team (plain Set Attention, no ISAB, small set). Slot order is loader convention only.
- Bans: **excluded** from SoloQ pretraining input. Only final 10 picks.
- Patch: single patch (e.g. 26.20). No patch-id input for now. If added later: `patch_ids: Long[B]`.
- PAD: never occurs in full SoloQ drafts, but contract reserves it for partial drafts (pro finetune). Loader/model must handle it.

## 2. Vocabularies

All IDs are `torch.long`. Counts are **parameterizable by patch**; numbers below are the example from `../README.md`. Code must expose them as constants.

### 2.1 Champions

- `NUM_REAL_CHAMPS = 173` (playable champs on this patch).
- `CHAMP_MASK_ID = NUM_REAL_CHAMPS` (=173) = `[MASK]`.
- `CHAMP_PAD_ID = NUM_REAL_CHAMPS + 1` (=174) = `[PAD]` (partial drafts only).
- `CHAMP_VOCAB_SIZE = NUM_REAL_CHAMPS + 2` (=175) = embedding rows.

Valid `champ_ids`: `0 .. CHAMP_VOCAB_SIZE-1`. Unmasked full drafts use only `0 .. NUM_REAL_CHAMPS-1`.

NOTE: sketch in `../README.md` shows `nn.Embedding(num_champs+1)` with `num_champs=173`. That is superseded: this contract **requires** `CHAMP_VOCAB_SIZE = NUM_REAL_CHAMPS + 2` (REAL + MASK + PAD = 175). `+1` implementations are rejected.

### 2.2 Roles

- `0`=TOP, `1`=JUNGLE, `2`=MID, `3`=ADC/BOT, `4`=SUPPORT, `5`=FLEX/UNKNOWN = **input-side masked-role token only**.
- `NUM_ROLES=5`, `ROLE_VOCAB_SIZE=6`, `ROLE_FLEX_ID=5`. Input embedding `nn.Embedding(6, D)`.
- **Output role head is exactly 5-way** (`Linear(D, 5)` over `0..4`). FLEX is never a label and never a prediction class.
- Masking follows BERT-style 90/10 (see 3.1): 90% of champ-masked slots also mask the role (role input = 5); 10% keep the true role visible (role input = truth) so the model learns `P(champ | role)` for the coach UI. PLUS role-only masks (see 3.1): ~10% of champ-visible slots hide the role so the role head learns `P(role | champ)` ("x champion is most likely y role"). `role_labels` hold truth on champ-masked AND role-only slots.

### 2.3 Teams

- `0`=Blue (slots 0-4), `1`=Red (slots 5-9). Team is derived from slot index.
- **No team embedding.** `DraftSlotEmbedding` is `LayerNorm(champ_lookup(champ_ids) + role_lookup(role_ids))` with `nn.Embedding(175, D)` + `nn.Embedding(6, D)`. `team_ids` is NOT a model input; the loader still records team implicitly via the slot convention (no separate tensor required).

## 3. Loader -> Model INPUT batch (training)

Loader yields a **dict of tensors** (first dim = batch `B`).
`C = NUM_REAL_CHAMPS`. Shapes:

| Key | Shape | Dtype | Values / rule |
|---|---|---|---|
| `champ_ids` | `[B,10]` | `long` | `0..C-1` visible, `CHAMP_MASK_ID` if masked, `CHAMP_PAD_ID` only if partial draft |
| `role_ids` | `[B,10]` | `long` | `0..4` visible; `5` on ~90% of champ-masked slots and ~10% of champ-visible slots (role-only); true role kept on ~10% of champ-masked slots (see 3.1) |
| `role_masked` | `[B,10]` | `bool` | `True` <=> this slot's role input was replaced with FLEX (champ-masked-hidden + role-only). Lets tests distinguish "role hidden" from "role given" |
| `is_masked` | `[B,10]` | `bool` | `True` <=> champ masked for prediction (~15% of slots, **>= 1 per draft**, see 3.1) |
| `champ_labels` | `[B,10]` | `long` | real champ `0..C-1` where `is_masked`, `-100` elsewhere (ignore index) |
| `role_labels` | `[B,10]` | `long` | real role `0..4` where `is_masked` OR role-only (`role_masked & ~is_masked`), `-100` elsewhere (always truth, even when role input kept visible) |
| `pad_mask` | `[B,10]` | `bool` | `True` <=> `champ_ids==PAD` (all-False for full SoloQ drafts) |
| `win_label` | `[B]` | `long` | **optional**, `1` if blue won. Present only when the loader is given `wins`; masked modelling ignores it |

No `team_ids` tensor: team follows slot index (0-4 Blue, 5-9 Red).
No `aux_targets`: auxiliary heads deferred (see Sec 6). `win_label` exists only
to supervise the opt-in value path (Sec 4.3).

### 3.1 Masking protocol (train)

- Champion: independently per slot with `p ~= 0.15` (~1-2 slots/draft), then **force at least one masked slot in every draft** (if a draft drew zero masks, mask one uniformly random slot). No wasted forward passes.
- If slot `m` champ-masked: `champ_ids[:,m]=CHAMP_MASK_ID`, `is_masked[:,m]=True`, labels hold ground truth.
- Role (conditional on champ mask, BERT-style 90/10): with prob 0.9 also mask the role (`role_ids[:,m]=ROLE_FLEX_ID`, `role_masked[:,m]=True`); with prob 0.1 keep the true role visible (`role_ids[:,m]=truth`, `role_masked[:,m]=False`). This teaches both joint prediction and `P(champ | known role)` for the coach UI.
- Role-only (independent, teaches `P(role | champ)`): on each champ-VISIBLE slot, with prob `ROLE_ONLY_PROB ~= 0.10` hide the role (`role_ids=FLEX`, `role_masked=True`) while the champ input stays truth. These slots carry a role label but NO champ label, forcing the role head to learn "x champion is most likely y role" from the champ embedding + context. Without this the role head only ever trains with the champ hidden and champion->role affinities get zero supervision.
- Other slots keep truth, labels = `-100`, `role_masked=False`.
- **No duplicate real champions** within one draft (visible + hidden truth combined = 10 unique).
- Eval/inference: `is_masked` all-False except slots being queried; single-slot query sets that slot to MASK (role MASK or given depending on whether the role is known).

### 3.2 Example (B=32, C=173, D=128)

```text
champ_ids     Long  [32,10]  e.g. [[12,45,3,...,101],...] (173 where masked)
role_ids      Long  [32,10]  e.g. [[0,1,2,3,4,0,1,2,3,4],...] (5 on ~90% of champ-masked + ~10% role-only)
role_masked   Bool  [32,10]  champ-masked-hidden + role-only (~90% of champ-masked plus ~1/draft role-only)
is_masked     Bool  [32,10]  ~15% True, >= 1 True per row
champ_labels  Long  [32,10]  -100 / 0..172
role_labels   Long  [32,10]  -100 / 0..4 (champ-masked + role-only)
pad_mask      Bool  [32,10]  all False (full drafts)
```

## 4. Model OUTPUT (forward)

`forward(batch_dict) -> dict` with exactly:

| Key | Shape | Dtype | Rule |
|---|---|---|---|
| `champ_logits` | `[B,10,C]` | `float` | `C`=NUM_REAL_CHAMPS (real champs only, exactly 173 cols). No MASK/PAD columns |
| `role_logits` | `[B,10,5]` | `float` | exactly 5-way over `0..4`. No FLEX column |
| `win_logit` | `[B,1]` | `float` | **only when built with `with_value=True`**; blue-win logit. Forbidden otherwise |

Heads: `champ: Linear(D,C)`, `role: Linear(D,5)`. No auxiliary heads beyond the
opt-in value path in Sec 4.3.

### 4.1 Duplicate-logit masking (required)

Before softmax/CE, for each masked position `m` in draft `b`, logits for champs **already visible elsewhere in that draft** must be suppressed to large negative (`-inf` or `<= -1e9`):

```text
visible = {champ_ids[b,j] for j in 0..9 if j != m and not pad and not masked}
champ_logits[b,m,c] = -inf  for c in visible
```

Ground-truth champ (hidden) is by construction not in `visible`, so it stays legal. Tests check `isneginf` or `< -1e8`.

### 4.2 Losses (reference, not part of forward contract)

```text
Loss = CE(champ_logits[is_masked], champ_labels) + lam * CE(role_logits[role labelled], role_labels)
```

- Both heads use **hard Cross-Entropy** (baseline).
- KNOWN LIMITATION (hard CE treats all wrong champions as equally wrong): a functionally similar-but-wrong champion (e.g. another engage support) is penalized the same as a nonsense pick, which may make learning slower/less graded than a similarity-weighted target. Options considered: soft targets from cosine similarity of a frozen prior (NOT the embedding being trained — that is circular), co-pick statistics, or role-conditioned priors. Deferred: implement hard CE first, revisit if confusion patterns show the model needs graded credit.
- Loss: champ head on `is_masked` slots; role head on ALL role-labelled slots (champ-masked + role-only); PAD slots never contribute (key_padding_mask).

### 4.3 Opt-in value path (`with_value=True`)

Off by default, so the masked-modelling contract and every pre-existing
checkpoint are unchanged. SoloQ pretraining turns it on.

```text
h [B,10,D]  ->  PMA(seeds=4) over blue slots  -> v_blue [B,D]
            ->  PMA(seeds=4) over red slots   -> v_red  [B,D]   (Siamese: one instance)
            ->  z = concat(v_blue, v_red, v_blue-v_red, v_blue*v_red)  [B,4D]
            ->  win_head: LN -> Linear(4D,D) -> GELU -> Linear(D,1)
```

- **Why it exists.** Every soloQ shard already carried `teams.{blue,red}.win`;
  pretraining never read it. Masked-champ alone never has to know which comp
  wins, so the pooled vector carried no *trained* outcome signal, and the pro
  value head sat at the base rate for 8,000 steps (AUC 0.497, Brier 0.2510
  against 0.2482 for a constant). A linear probe on the old checkpoint's
  pooled vector reads **AUC 0.5329**, i.e. the information was present but
  nothing downstream was trained to use it.
- **PMA, not mean pooling.** A PAD-masked mean over 5 slots divides per-slot
  differences by 5 and is dominated by whichever slots are present. PMA's
  learned queries stay permutation-invariant (fixed queries, set keys) and
  Siamese (shared instance), so invariants 8 and 9 still hold.
- **All-PAD team** (partial drafts): one key is kept attendable to avoid a NaN
  softmax, and the pooled vector is zeroed by mask, matching mean-pool.
- **Loss:** `+ lambda_win * BCE(win_logit, win_label)`; `lambda_win=1.0` by
  default (one global label weighted like one masked slot), `--no-win` reverts
  to the old objective.
- **Ramp, don't switch on.** `lambda_win` is ramped linearly from 0 to its full
  value over `--win-warmup-epochs` (default 5). Measured reason: at 700 steps
  the encoder probe read 0.4996 and the win head sat exactly at the prior
  (AUC 0.4970, BCE 0.6904 vs ln2 0.6931), while a fresh PMA+win head on the
  *mature* frozen encoder reached AUC 0.5396 in the same 700 steps. The head
  design is right; it was starved by an immature representation, so the value
  loss must not compete with champion-identity learning early.
- **And cut out.** `--win-patience` (default 8) freezes `lambda_win` at 0 once
  val win AUC has not improved for that many epochs. Measured: in the D=512
  40-epoch run train win BCE fell 0.6842 -> 0.4359 while val Brier rose
  0.2467 -> 0.3857 and AUC fell 0.5601 -> 0.5206, i.e. everything after
  ~epoch 19 made the head *and* the encoder worse. `best_value.pt` already
  holds the good head (epoch 15); freezing lets the encoder finish on the
  masked task alone instead of chasing outcome noise.
- **Which checkpoint to load.** `best.pt` = best `champ_top5` (encoder for
  policy/finetune), `best_value.pt` = best val win AUC (the value head). In the
  D=512 run these are epoch 38 and epoch 15, and the value head inside
  `best.pt` is the overfitted one (Brier 0.373) - do not use it for value.
- **The two objectives trade off; the freeze decides which way.** Freezing
  `lambda_win` stops the head overfitting but also stops pushing outcome
  information into the pooled vector, and that information then decays:
  a fresh PMA probe on the epoch-33 encoder reads **0.5444**, against
  **0.5521-0.5529** at the epoch-15/26 encoder. So a late checkpoint is the
  better *masked* encoder and an early one is the better *value* encoder - they
  are not interchangeable. Concretely: load `best_value.pt` for P4 search,
  `best.pt` for policy/finetune.
- **The freeze latches.** Once `--win-patience` fires it stays off; the frozen
  head's AUC drifts up ~1e-4 as the encoder moves, and letting that reset the
  counter resumed the overfitting (observed at epoch 26, AUC 0.5597 -> 0.5467).
  `--win-freeze` is the manual override, e.g. when resuming a run whose head
  already overfit.
- **Measured on a clean view.** Value metrics are computed with
  `mask_prob=0, force_one=False`, because a 15%-masked draft caps any readout.
  `win_skill = 1 - brier/brier_const`; > 0 means it beats the constant.
- **Label rule.** `blue_win = teams.blue.win` only when it disagrees with
  `teams.red.win`; agreeing rows are dropped (72 of 384,428 on 16.20). Verified
  identical to `store.py`'s `blue_win` on 37,820 overlapping games.

### 4.4 Kit-attribute residual (`attr_matrix`, P3.1)

Off unless a matrix is passed; on by default in pretraining (`--no-champ-attrs`
reverts). It changes the *input* to the encoder, so it is documented here.

```text
x = champ_lookup(champ_ids) + role_lookup(role_ids)
    + Linear_K->D(champ_attrs[champ_ids])      # zero-initialised
-> LayerNorm(D)
```

- **The table.** `data/champ_attrs.json`, built by `data/build_champ_attrs.py`:
  173 champions x 18 axes (engage, peel, poke, dive, scaling, lane_priority,
  hard_cc, mobility, range, objective_speed, tankiness, magic_share, sustain,
  damage_output, side_laning, pick_potential, kiting, healing), ordinal 0-3,
  divided by 3 so every axis is 0-1. The ordinal scale is deliberate: a
  hand-tagged column has about half a level of real resolution, and
  standardising 18 three-valued columns would invent precision.
- **Why it is a residual, not a concatenation.** Concatenating would change
  `embed_dim` and therefore every downstream weight; a zero-init `Linear(K,D)`
  added to the summed embedding keeps the shape, so an old checkpoint loads
  into an attribute-enabled encoder with the adapter at zero and computes
  *bit-identical* outputs (`tests/test_champ_attrs_embed.py` asserts equality,
  not closeness). "Attributes on/off" is then one flag, and a difference
  between two runs is the adapter learning rather than a different
  initialisation.
- **What it buys.** The axes are shared across champions, so a tag learned on
  one champion transfers to another with little data; and a champion released
  after the training data was collected has no useful embedding row but does
  have a tag row, which is the cold-start difference between a prior and a
  random vector. 18 x D + D parameters (~4.9k at D=256).
- **MASK and PAD are zero rows**, so an empty or masked slot contributes
  nothing to the residual - invariant 5 (PAD discipline) still holds.
- **The matrix is a non-persistent buffer**, i.e. data, not a parameter: it is
  absent from the state dict and the checkpoint stores only `attr_dim`. That
  keeps checkpoints small and portable, and makes a table revision loud: a
  checkpoint whose `attr_dim` disagrees with the JSON refuses to load rather
  than silently reinterpreting its own inputs.
- **Loading asymmetry.** `load_encoder_state()` tolerates *missing*
  `embed.attr_proj.*` (absent == zero-initialised, so a pre-P3.1 checkpoint
  resumes into an attribute model unchanged) but refuses to *drop* them (a
  trained adapter disappearing is a different model, not a config choice).
  Every stage rebuilds the encoder through `attr_matrix_for_config(cfg)`, so
  `attr_dim` in the checkpoint config decides, and a stage cannot forget it.

## 5. Invariants checklist (what tests enforce)

1. Shapes/dtypes/ranges: every key in Sec 3 matches table (incl. `role_masked Bool [B,10]`, no `team_ids`/`win_label`).
2. Slot convention: slots 0-4 Blue, 5-9 Red; `[B,2,5]` split consistent. No team tensor, no team embedding.
3. Mask consistency: `is_masked <=> champ_ids==MASK` (full drafts); `role_masked <=> role_ids==FLEX` on labelled slots; champ-masked role-hidden slots are a subset of `is_masked`; role-only slots (`role_masked & ~is_masked`) show a real champ in `champ_ids` with `champ_labels==-100`; role-visible masks (`is_masked & ~role_masked`) keep true role in `role_ids`; champ labels `-100` exactly where not `is_masked`; role labels `-100` exactly where neither `is_masked` nor `role_masked`; labelled roles in range; **>= 1 champ-masked slot per draft**.
4. No duplicates: no repeated real champ id within a draft (visible + hidden truth).
5. PAD discipline: full drafts have no PAD; if PAD present that slot's labels are `-100`, not masked, and model must ignore via `pad_mask`/key_padding_mask.
6. Forward shapes: outputs exactly `champ_logits [B,10,173]` + `role_logits [B,10,5]`; no NaN/+inf (`-inf` only via duplicate mask); no aux outputs; grad flows. `win_logit [B,1]` present **iff** `with_value=True`, finite, and absent otherwise.
7. Duplicate-logit mask: Sec 4.1 holds on every masked position.
8. Permutation equivariance: permuting slots *within* a team (same perm on all slot-aligned tensors) permutes that team's slot outputs identically (no positional embedding allowed; red-team and cross-team outputs unaffected beyond the permuted team's content).
9. Siamese teams: intra-team attention weights shared across Blue/Red (param sharing / identical output under team swap when champs/roles swapped accordingly).

## 6. What is NOT in this contract

- Draft **order**, pick/ban sequence, fearless bans, turn structure: pro finetune only.
- Frontend/coach UI (top-N per remaining role, flex handling): consumes this model's embeddings/logits later.
- Auxiliary / multitask heads (duration, gold, kills, objectives): deferred, not
  implemented, not tested. The one exception is the opt-in value path in
  Sec 4.3 (win probability only), which pretraining uses and which pro stages
  may either keep or replace with their own head - `load_encoder_state()`
  drops `pma.*` / `win_head.*` when the target has no value path, and raises on
  any other unexpected key so real shape mismatches stay loud.
- Exact `D`, `lam`, heads depth: hyperparameters, not contract. Defaults: `D=128`, `dropout=0.0` (underfitting risk dominates at this size; raise only if train/val gap shows overfitting).

## 7. How to satisfy this contract

Loader builder: return dict in Sec 3 with exact keys/shapes/dtypes (incl. `role_masked`, no `team_ids`). Reuse `tests/test_soloq_contract.py::validate_batch_dict` locally.

Model builder (`pretraining/soloq_model.py`): implement e.g.

```python
class DraftSlotEmbedding(nn.Module):  # LayerNorm(Embedding(175,D) + Embedding(6,D)), no team embed
    ...
class SoloQPretrainedModel(nn.Module):
    def forward(self, batch: dict) -> dict:
        # batch: champ_ids/role_ids [B,10], is_masked/role_masked/pad_mask [B,10]
        # returns: champ_logits [B,10,173], role_logits [B,10,5]
```

Run `pytest tests/test_soloq_contract.py -v`. Pure-contract tests (synthetic batch, masking logic) must pass; loader/model-gated tests `skip` until implemented, then must pass unmodified.

## 8. Constants quick-reference

```text
N_SLOTS=10, NUM_REAL_CHAMPS=173 (patch-specific), CHAMP_MASK_ID=173, CHAMP_PAD_ID=174,
CHAMP_VOCAB_SIZE=175 (required), NUM_ROLES=5, ROLE_VOCAB_SIZE=6 (input only), ROLE_FLEX_ID=5 (input only),
ROLE_HEAD_DIM=5 (output, no FLEX), MASK_PROB~=0.15, ROLE_KEEP_PROB=0.1, ROLE_ONLY_PROB~=0.10, IGNORE_INDEX=-100, EMBED_DIM example 128, DROPOUT default 0.0
No team_ids tensor. No win/aux outputs unless with_value=True (Sec 4.3).
```


