# Assumptions — LoL Draft Analysis

Living doc: what we believe about the data, the embeddings, and why the
current SoloQ-pretrain to pro-finetune + set-attention strategy was chosen.

## 1. Problem framing

- Goal is a draft assistant for pro play: given a partial draft (picks, bans,
  fearless-unavailable, side to move, phase), suggest top viable options for
  every remaining role, predict likely roles and flexes, and eventually
  estimate win value for search (MCTS or lookahead).
- Draft is treated as set-completion plus sequential decision: pretraining
  learns P(missing champ | rest of draft), P(role | champ) and P(champ | role);
  finetuning adds P(next move | prefix, legality) and P(blue wins | state).
- Core simplification: only the current set of picks and bans (+ roles, side)
  matters for the next pick. Order of earlier moves is assumed irrelevant to
  comp value (permutation invariance). Roughly true for comp quality, false
  for intent signalling (B1 blind vs R5 counter differ).
- Same champ cannot be picked twice; bans and fearless-unavailable are hard
  legality constraints applied as -inf logit masks, not learned.

## 2. Data — what each source can and cannot teach

### 2.1 SoloQ (high-elo, single patch 16.20)

- Collection: puuids to match ids to match details via Riot API
  (RANKED_SOLO_5x5, Challenger/GM/Master, all platform regions, per-region
  limiters). Compact ~3 kB record per match; raw DTO not kept. No timeline
  calls, so no gold/XP-at-minute curves — peak lane-lead stand-ins instead.
- Processing: process matchdata pipeline streams shards to champ mapping
  (Riot id to 0..C-1) to canonical slots (B_TOP..R_SUP) to masking. Filters:
  ranked queues 420/440, duration >= 300s, Data Dragon 16.20.1, 173 champs.
- What it is: one completed 5v5 draft = 10 picks, no ordering, no bans. Bans
  excluded from pretraining. PAD reserved for partial/pro states.
- Assumes high-elo SoloQ on one patch is a good proxy for champ strength,
  role affinity, synergy/counter structure. Volume compensates for noise.
- Meta is patch-pinned. Embeddings do NOT transfer across patches without
  retraining — new/reworked champs change C and shift semantics.
- Teaches solo-carry and lane-fit priors, not coordinated pro comps. It
  over-represents comfort/blinds and under-represents pro coordination
  (swaps, dive comps, fearless depth). Pretrain-only output is SoloQ-biased.
- No outcome weighting: losers count the same as winners. Down-weighting
  losing drafts as mistakes is NOT implemented.
- Riot position is assumed truth for role labels. Off-role games are noise.

### 2.2 Pro play (Oracle's Elixir, 2022-2026)

- Source: raw OE CSVs (~350 MB), cached to pro_drafts.pt. Same Data Dragon
  champ space as SoloQ — 173/173 coverage every year. 46,662 kept drafts;
  skips only for documented reasons (bad rows, incomplete bans/picks,
  duplicate champ, pick mismatch).
- Representation: 12-row OE blocks to a global 20-move sequence (6 bans1 +
  6 picks1 + 4 bans2 + 4 picks2) on the 2026 First-Selection order; pre-2026
  blue-first. Move = (kind, team, champ, role); bans carry IGNORE role.
- Era handling: fearless unavailable/prior_picks are data-driven per series
  (2025+ mostly fearless, 2026 fully, pre-2025 priors only). First-pick
  decoupling in 2026 only. Rules changed, so encoding is era-aware.
- Splits: series-grouped and time-ordered (train ~40.5k, val ~3k, test ~3k).
  A series never straddles splits. Recency weighting w = 2^(-age/75d).
- ProNextMoveDataset: each draft to 20 prefix-to-next-move examples (partial
  encoding: unpicked = PAD + FLEX + pad_mask, plus bans, turn info, fearless
  mask, legality taken-mask for -inf suppression, win label).
- ProMaskedDraftDataset: full 10-pick pro drafts through verbatim SoloQ
  masking (single source of truth), plus win label and econ targets
  (gold/xp at 10/15; missing to zeros with has_econ = 0).
- What pro teaches that SoloQ cannot: order/phase/side effects (ban priority,
  B1 blind vs R5 counter, first-pick asymmetry), fearless depth (pool
  exhaustion across a series — the main 2025+ shift), and coordinated comp
  value (win + econ to a value function for search).
- Limits: 46k games over 5 patches is small for a 4-6M model on the current
  patch — hence recency weighting + SoloQ rehearsal (0.15) instead of
  pro-only training. Assumes synergy/counter structure drifts slower than
  popularity. No player/team identity ("Faker's Azir" = "Azir") — team pools
  intentionally not modelled. Econ labels are weak auxiliary supervision.

## 3. Champion (and role) embeddings — what they mean

- DraftSlotEmbedding: LayerNorm(Embedding(175, D) + Embedding(6, D)).
  Champ table = REAL + MASK + PAD = 175 rows (C = 173 on 16.20). Role table =
  5 roles + FLEX input-only token. Output heads are Linear(D, C) and
  Linear(D, 5) — MASK/PAD/FLEX are input-side only, never predicted.
- No team embedding (decided revision): team comes from slot index alone
  (0..4 Blue, 5..9 Red). A team vector was redundant and a risk to Siamese
  symmetry.
- Intended semantics: champ row = static prior (base strength + role affinity
  + generic synergy/counter bias from SoloQ co-occurrence, like a word
  embedding in BERT). Adding the role row overlays role context (ADC Vayne vs
  TOP Vayne). Sum forces a shared space; LayerNorm keeps scales comparable.
  After intra-team + inter-team attention, slot vector h = champion-in-draft:
  prior contextualised by 4 allies (synergy) and 5 enemies (counters). h is
  shared by both heads and later by pro value heads via _encode() reuse.
- Pooled z = concat(v_blue, v_red, diff, product) with PAD-masked means =
  draft-level comp vector for win/econ heads. Diff + product expose asymmetry
  and interaction explicitly.
- Embeddings do NOT contain: patch id (single-patch), pick order, player
  skill, spells/items, in-game state. They are draft-only priors.

## 4. Current strategy — architecture lineage and why it fits

- Lineage: Set Transformer (Lee et al. 2018) without ISAB + BERT-style masked
  modelling. Sketch in root README: embedding to Siamese intra-team attention
  to symmetric inter-team cross-attention to slot-level champ + role heads
  (plus pooling to aux heads). Frozen as I/O contract, enforced by tests,
  implemented in soloq_model.py.
- Why sets, not sequences: same champ cannot repeat and only the current
  state matters, so permutation invariance is wanted and there are no
  positional embeddings. Slot order is loader convention only; tests assert
  within-team equivariance + Siamese sharing (one intra encoder over
  [2B, 5, D], one cross block both directions). No ISAB: 5+5 sets make full
  attention cheap. Scale D = 256, 4 intra + 2 cross layers, 8 heads,
  ~4.8-6M params, B = 1024 + AMP on RTX 4060; dropout 0.0 (underfitting
  dominates at this size).
- Why masked modelling (BERT for drafts): loss = CE_champ + 0.3 * CE_role,
  p ~ 0.15 champ masks, >= 1 mask/draft, BERT 90/10 role on masked slots (10%
  role-visible teaches P(champ | role), the "best X for MID" query) plus
  p ~ 0.10 role-only masks (champ visible, role hidden teaches P(role |
  champ), the flex query). Duplicate -inf in the model; hard CE (soft targets
  dropped; graded-credit limitation noted). Fresh train masks per epoch, fixed
  val masks. AdamW 1e-3, warmup + cosine, AMP, one folder per run (best.pt on
  val champ_top5 + latest.pt).
- Eval: champ_top1/5/10, role_cond_top5 (the coach metric), role_hidden_top1,
  role_given_champ_top1, MRR, per-role splits, all vs popularity baselines.
  Gate: beat base_role_top5, then push role_cond_top5. evaluate.py adds
  train/val gap + K-mask robustness curve (K = 10 is the empty draft).
- Why SoloQ-pretrain to pro-finetune: Stage 0 learns comp grammar from
  abundant SoloQ; Stage 1 loads the encoder, adds pooled outcome heads (win
  logits + econ regressions), finetunes masked CE + win BCE + econ MSE with
  staged unfreezing (0 = heads+readouts, 1 = +cross, 2 = +intra; embeddings
  frozen; readouts at head LR, trunk at 1e-5). econ-mask-prob 0.5 mixes full
  drafts with 1..9-pick subsets so the win head values partial states for
  MCTS (all-PAD teams kept attendable to avoid NaN). Checkpoints split
  best_mask.pt (suggestions) vs best_win.pt (value) + Platt calibration +
  econ stats. Stage 2 (policy + MCTS) is planned, not built; frontend serves
  today's UI (role-conditioned MASK queries, flex as post-hoc overlap,
  walkthrough of the 46.6k pro drafts).
- Fit vs gaps: set completion matches the coach query; Siamese teams encode
  Blue/Red symmetry; cross-attention separates synergy from counters;
  role-conditioning gives per-role top-5 + flex for free; -inf enforces
  legality. Gaps: trunk never sees bans/order/fearless (finetune/policy must
  add it); multi-MASK frontend inference differs from single-mask training
  (K-curve measures it); hard CE gives no partial credit; win head is
  side-neutral ("blue wins" — red queries mirror slots); econ is weak
  supervision.

