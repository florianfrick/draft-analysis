"""Policy-value model for pro draft prediction (Stage 2).

Architecture:
  SoloQ pretrained encoder (frozen) -> slot vectors h [B,10,D]
    + ban context cross-attention (bans attend to slots)
    + turn/phase/side state embedding
    -> Policy head: Linear(D+state, C) with legality -inf masking
    -> Value head: MLP on pooled vector + state

The encoder is the same SiameseBase from
solo_queue/model/pretraining/base_model.py.
We reuse its _encode() to get slot vectors for the partial draft (PAD
slots are handled by the encoder's key_padding_mask logic). On top of
that, a lightweight context module adds ban information and draft
state (move index, phase, side to move, first-pick side) via:
  1. Ban embeddings: champion embeddings of the banned champs, pooled
     and injected into each slot via cross-attention.
  2. State embedding: small learned embeddings for (phase, to_move,
     first_pick, move_idx, patch_id), summed. Patch id is what lets the
     model represent "which champs are high-priority on this patch" -
     without it the first ban of a game is unlearnable.

Policy head output: [B, C] logits for the next move (champion id).
Legality is enforced by setting logits of taken/fearless champions to
-inf (the taken_mask from the dataset unions bans+picks; unavailable_mask
adds fearless-picked champs). This is the TRAINING contract: fearless
champions stay bannable because 8 corpus rows ban one, and masking a
training target out of the cross-entropy would dominate the loss. The
serving path (`solo_queue/model/policy/infer.py:_policy_batch`) folds the fearless
pool into `taken_mask` too, so the UI never suggests one on a pick OR a
ban. Do not "simplify" the split below without handling those rows.

Value head output: [B, 1] win logit for the side to move (or blue-win
logit — configurable). Default: blue-win logit (matches finetune).

Ban-awareness (branch exp/bans). The description above is the original
"ban-blind" shape: bans reach the logits only through one cross-attention
block and a single pooled vector, and the fearless pool is a legality mask
rather than an input. Three switches make the model intrinsically ban-aware,
all off by default so every existing checkpoint rebuilds unchanged:
  * `ban_side` - a side embedding per ban token, derived from
    (slot, first_pick). Your ban and the enemy's ban stop being one fact.
  * `ban_fearless` - the fearless pool joins the ban path as tokens, and a
    pool-size summary enters the state vector.
  * `counter_rank` - an output-side low-rank champion x gone-champion term
    (see `BanInteraction`). This is the mechanism for "X is banned, so Y is
    worth more"; without it the candidate's own row never sees the ban set.
`ban_side` and `counter_rank` are zero-initialised: at step 0 the model is
bit-identical to the ban-blind one, which is what makes them single-variable
ablations. `ban_fearless` is not, and cannot be: its *parameters* are
zero-initialised, but the pool tokens are real champion embeddings that join
the attention key set, so the forward pass differs from step 0 by
construction. The honest statement for that switch is "the pool becomes an
input", not "nothing changes until it is learned".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SLOTS = 10
TEAM_SIZE = 5
NUM_REAL_CHAMPS = 173
CHAMP_PAD_ID = NUM_REAL_CHAMPS + 1
ROLE_FLEX_ID = 5
IGNORE_INDEX = -100
N_PHASES = 4  # 0=bans1, 1=picks1, 2=bans2, 3=picks2
N_SIDES = 2   # 0=blue, 1=red
N_BAN_SLOTS = 10  # max 5 bans per side = 10 total
DUPLICATE_LOGIT_NEG = -1e4  # large negative, fp16-safe (max ~65504)

# Fearless pool width. A series is at most 5 games of 10 picks, so the pool
# entering game 5 holds at most 40 champions - a bound, not a guess. Kept here
# as well as in the dataset because the model has to size the token axis even
# for a batch that arrives without the key.
FEARLESS_SLOTS = 40
# Ban tokens that are not bans (fearless-pool tokens) get kind 1.
KIND_BAN, KIND_POOL = 0, 1
# Side ids for the side embedding: 0 blue, 1 red, 2 "no side" (pool tokens,
# and any batch that does not carry sides).
SIDE_NONE = 2

# Which side made ban slot i when first_pick is 0. The interleave is fixed by
# `pipeline.interleaved_moves`: bans1 alternates first-pick team first (slots
# 0-5), bans2 alternates second-pick team first (slots 6-9). So the side is a
# pure function of (slot, first_pick) and is DERIVED, not carried in the batch.
# That is what keeps `swap_sides` honest: it flips `first_pick`, and every
# derived side flips with it, with no ban block to permute and get wrong.
BAN_SIDE_BY_SLOT = (0, 1, 0, 1, 0, 1, 1, 0, 1, 0)


class StateEmbedding(nn.Module):
    """Learned embedding for (phase, to_move, first_pick, patch) +
    learned move_idx.

    Output dim = state_dim. The move index (0..19) is encoded with a
    small learned embedding (20 is small enough for a lookup table).
    Phase (0-3), side-to-move (0-1), first-pick (0-1), and patch id use
    separate learned embeddings. All are summed.

    `n_patches` = vocabulary size including the UNK row at index 0
    (see data/pro play/datasets.build_patch_vocab). n_patches=0 disables
    the patch embedding, which keeps pre-patch checkpoints loadable.

    Why patch is part of the state: with no picks on the board (move 0,
    the first ban) the only thing that determines the answer is the
    current patch's ban priority list. Without this input the model
    cannot represent it at all, which is why bans1 scored 12.3% against
    37.4% for a static lookup table fit on the same patches.
    """

    def __init__(self, state_dim=64, n_patches=0, n_pool_feats=0):
        super().__init__()
        self.move_embed = nn.Embedding(20, state_dim)
        self.phase_embed = nn.Embedding(N_PHASES, state_dim)
        self.side_embed = nn.Embedding(N_SIDES, state_dim)
        self.first_pick_embed = nn.Embedding(N_SIDES, state_dim)
        self.patch_embed = (nn.Embedding(n_patches, state_dim)
                            if n_patches > 0 else None)
        # Fearless-pool summary: per-role counts of the champions already gone
        # plus a total. Zero-initialised, so a pool-aware model is bit-identical
        # to a pool-blind one at step 0 - the same one-flag rule the patch
        # embedding and the attribute adapter follow.
        self.pool_proj = (nn.Linear(n_pool_feats, state_dim)
                          if n_pool_feats else None)
        if self.pool_proj is not None:
            nn.init.zeros_(self.pool_proj.weight)
            nn.init.zeros_(self.pool_proj.bias)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, move_idx, phase, to_move, first_pick, patch_id=None,
                pool_feats=None):
        """All inputs [B] long -> [B, state_dim]."""
        x = (self.move_embed(move_idx)
             + self.phase_embed(phase)
             + self.side_embed(to_move)
             + self.first_pick_embed(first_pick))
        if self.patch_embed is not None and patch_id is not None:
            x = x + self.patch_embed(patch_id)
        if self.pool_proj is not None and pool_feats is not None:
            x = x + self.pool_proj(pool_feats)
        return self.norm(x)


class BanContext(nn.Module):
    """Cross-attention: slot vectors query ban embeddings.

    Banned champions are embedded using the encoder's champion lookup
    (frozen) and then attended to by the slot vectors. This lets each
    slot "see" what's been banned when computing its representation.

    PAD ban slots are masked out in the attention.

    `n_kinds` > 1 turns on a kind embedding (this-game ban vs fearless-pool
    token), `n_sides` > 0 a side embedding (blue / red / no side). Both are
    zero-initialised: with them present but untrained the block computes
    exactly what it did before, so "side on" and "fearless on" are single
    flags rather than two different initialisations.

    Why a side embedding at all, given the side is derivable from (slot,
    first_pick) and `first_pick` is already in the state vector: the model
    would have to learn the XOR of a 10-slot position and a 2-way bit, per
    ban, before it could use the distinction. Handing it the resolved side is
    one lookup. It matters because your own ban and the enemy's ban are not
    the same fact - one states a plan, the other removes a threat you no
    longer have to answer.
    """

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.0,
                 n_kinds=1, n_sides=0):
        super().__init__()
        self.ban_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.kind_embed = (nn.Embedding(n_kinds, embed_dim)
                           if n_kinds > 1 else None)
        self.side_embed = (nn.Embedding(n_sides, embed_dim)
                           if n_sides > 0 else None)
        for m in (self.kind_embed, self.side_embed):
            if m is not None:
                nn.init.zeros_(m.weight)

    def forward(self, slot_vecs, ban_embeds, ban_pad_mask, kind_ids=None,
                side_ids=None):
        """slot_vecs [B,10,D], ban_embeds [B,T,D], ban_pad_mask [B,T] bool.

        ban_pad_mask: True = PAD token (not a real ban/pool entry). Used as
        key_padding_mask in the attention (True = ignore).

        kind_ids / side_ids [B,T] long, optional; only read when the
        corresponding embedding exists.
        """
        if self.kind_embed is not None and kind_ids is not None:
            ban_embeds = ban_embeds + self.kind_embed(kind_ids)
        if self.side_embed is not None and side_ids is not None:
            ban_embeds = ban_embeds + self.side_embed(side_ids)
        # If all bans are PAD (early draft, no bans yet), skip attention
        # to avoid NaN from all-masked softmax.
        all_pad = ban_pad_mask.all(dim=1, keepdim=True)  # [B,1]
        safe_mask = ban_pad_mask & ~all_pad  # ensure at least 1 key attendable

        attn_out, _ = self.ban_attn(slot_vecs, ban_embeds, ban_embeds,
                                    key_padding_mask=safe_mask,
                                    need_weights=False)
        x = self.norm(slot_vecs + attn_out)
        return self.norm2(x + self.ffn(x))


class BanInteraction(nn.Module):
    """Output-side term: logits_c += e_c^T M (mean over gone champions).

    The point of the whole branch. The policy head gives each champion one
    fixed row (`Linear(D, 173)`), so "X is banned, therefore Y is better" has
    to be expressed as a shift in the single pooled vector - a rank-limited,
    sample-hungry path that has to average over every ban. Here the term is
    per-candidate and pairwise: with M = A B^T of rank r,

        logits_c += e_c^T A B^T g ,  g = mean_{b in gone} e_b

    which is exactly sum_b e_c^T M e_b / |gone|. A is [D,r], B is [D,r], so
    ~2Dr parameters (11k at D=256, r=32). It shares the encoder's champion
    embedding table, so a pair is represented in the space the trunk already
    learned rather than as a free 173x173 table.

    `b` is zero-initialised and `a` is small random: the term is exactly 0 at
    step 0, but d(loss)/db is non-zero there, so b moves first and a follows.
    Zero-initialising both would make the product dead.

    This is the same fix `STATUS.md` §6 prescribes for champion attributes -
    "the adapter conditions the input, while the predicted champion's own row
    never sees attrs_c. Fix on the output side" - applied to bans, which have
    the same defect and more data behind them.
    """

    def __init__(self, embed_dim=256, rank=32, num_champs=NUM_REAL_CHAMPS):
        super().__init__()
        self.rank = rank
        self.a = nn.Parameter(torch.randn(embed_dim, rank) * 0.02)
        self.b = nn.Parameter(torch.zeros(embed_dim, rank))
        self.num_champs = num_champs

    def summary(self, gone, gone_pad):
        """Mean gone-champion vector [B,D]; all-pad rows give a zero vector."""
        keep = (~gone_pad).float().unsqueeze(-1)  # [B,T,1]
        return (gone * keep).sum(1) / keep.sum(1).clamp_min(1.0)

    def forward(self, champ_w, g):
        """champ_w [C',D] (champion embedding table), g [B,D] -> [B,C]."""
        u = champ_w[:self.num_champs] @ self.a   # [C,r]
        v = g @ self.b                           # [B,r]
        return v @ u.t()                         # [B,C]


class PolicyValueModel(nn.Module):
    """Frozen encoder + ban context + state embedding -> policy + value.

    Parameters:
      encoder: SiameseBase (frozen by default, can be unfrozen
        for staged training).
      embed_dim: must match encoder's embed_dim.
      state_dim: dimension of the state embedding.
      n_econ: number of economy regression targets (0 = skip econ head).

    Forward batch keys (from ProNextMoveDataset / pro_collate):
      champ_ids [B,10], role_ids [B,10], pad_mask [B,10],
      ban_ids [B,10] (PAD-padded),
      taken_mask [B,C] (banned|picked, for legality),
      unavailable_mask [B,C] (fearless-picked),
      fearless_ids [B,40], fearless_role_ids [B,40] (optional; only read
        when ban_fearless=True),
      move_idx [B], phase [B], to_move [B], first_pick [B],
      patch_id [B] (optional; only used when n_patches > 0),
      target_champ [B], target_role [B], is_ban [B],
      win_label [B], weight [B], econ_target [B,6], has_econ [B]

    Ban-awareness switches (all off by default, so every pre-existing
    checkpoint rebuilds the exact module set it was trained with):
      ban_side:      per-ban side embedding, derived from (slot, first_pick).
      ban_fearless:  fearless pool as extra ban-path tokens + a pool-size
                     summary in the state vector. Unlike the other two this is
                     NOT a no-op at initialisation - the pool tokens are real
                     champion embeddings, so they change the forward pass from
                     step 0 even with every new parameter at zero.
      counter_rank:  rank of the output-side champion x gone-champion
                     interaction (0 = off).

    Outputs:
      policy_logits [B, C] (legality-masked, -inf on taken/unavailable)
      value_logit [B, 1] (blue-win logit; absent when with_value=False)
      econ_pred [B, n_econ] (if n_econ > 0)
    """

    def __init__(self, encoder, embed_dim=256, state_dim=64,
                 n_econ=6, dropout=0.0, n_patches=0, with_value=True,
                 ban_side=False, ban_fearless=False, counter_rank=0):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.n_econ = n_econ
        self.n_patches = n_patches
        self.with_value = with_value
        self.ban_side = bool(ban_side)
        self.ban_fearless = bool(ban_fearless)
        self.counter_rank = int(counter_rank or 0)

        # Ban context: banned champs attend into slot vectors. `n_kinds` /
        # `n_sides` size the optional embeddings; 1 / 0 reproduces the
        # original block exactly.
        self.ban_context = BanContext(
            embed_dim, num_heads=4, dropout=dropout,
            n_kinds=(2 if self.ban_fearless else 1),
            n_sides=(3 if self.ban_side else 0))
        # State embedding
        self.state_embed = StateEmbedding(
            state_dim, n_patches=n_patches,
            n_pool_feats=(1 if self.ban_fearless else 0))
        # Output-side champion x gone-champion interaction
        self.counter = (BanInteraction(embed_dim, self.counter_rank)
                        if self.counter_rank > 0 else None)

        # Policy head: pool all slot vectors (4D) + state -> C logits.
        # The policy is a distribution over all champions for the next
        # move. We don't condition on a specific slot — the state
        # (phase, side, move_idx) tells the head whether it's a ban or
        # pick and which side is acting.
        pooled_dim = embed_dim * 4
        policy_in_dim = pooled_dim + state_dim
        self.policy_head = nn.Sequential(
            nn.LayerNorm(policy_in_dim),
            nn.Linear(policy_in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, NUM_REAL_CHAMPS),
        )

        # Value head: pooled + state -> blue-win logit.
        # OFF by default in new runs (--value-weight 0): measured test
        # win_auc 0.4969 across all 20 prefixes, flat 0.5015 on moves 0-6
        # (Brier exactly at the base rate) and anti-correlated once the
        # draft fills - measured by the (since deleted) diagnose_value.py.
        # The usable
        # value function is solo_queue/model/finetune/value.py on the encoder's PMA
        # path. Kept constructible so old checkpoints still load.
        value_in_dim = pooled_dim + state_dim
        if with_value:
            self.value_head = nn.Sequential(
                nn.LayerNorm(value_in_dim),
                nn.Linear(value_in_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, 1),
            )
        else:
            self.value_head = None

        # Optional econ head
        if n_econ > 0:
            self.econ_head = nn.Sequential(
                nn.LayerNorm(value_in_dim),
                nn.Linear(value_in_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, n_econ),
            )
        else:
            self.econ_head = None

        self.set_freeze_encoder(True)

    def set_freeze_encoder(self, freeze):
        """Freeze/unfreeze the pretrained encoder."""
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
        self._encoder_frozen = freeze

    def _pool(self, h, pad_mask):
        """Set-pooled vector z [B, 4D] (PAD-masked means per team).

        Same pooling as finetune/model.py: concat(v_blue, v_red,
        diff, product). PAD slots never contribute.
        """
        keep = (~pad_mask).float().unsqueeze(-1)  # [B,10,1]
        blue, red = h[:, :TEAM_SIZE], h[:, TEAM_SIZE:]
        kb, kr = keep[:, :TEAM_SIZE], keep[:, TEAM_SIZE:]
        vb = (blue * kb).sum(1) / kb.sum(1).clamp_min(1.0)
        vr = (red * kr).sum(1) / kr.sum(1).clamp_min(1.0)
        return torch.cat([vb, vr, vb - vr, vb * vr], dim=1)

    def _apply_legality(self, policy_logits, batch):
        """Mask taken + unavailable champions to -inf (TRAINING rule).

        taken_mask: banned|picked (hard legality — you can't ban or pick
        a champ that's already been banned or picked THIS game).
        unavailable_mask: fearless-picked (champs picked in earlier
        series games), masked on PICK moves only. The corpus has 8 ban
        moves whose target is such a champion; masking those targets
        here would put a 1e4-scale CE term into every epoch.

        Inference is stricter without changing this function:
        `infer._policy_batch` passes `taken | unavailable` as
        `taken_mask`, which makes the pool illegal on bans too.

        Both masks are [B, C] bool, True = illegal.
        """
        illegal = batch["taken_mask"].clone()
        if "unavailable_mask" in batch:
            # Only mask unavailable (fearless-picked) champs on PICK
            # moves; bans can target any champ not already taken.
            is_pick = ~batch["is_ban"].bool()  # [B] True = pick move
            unavail_for_picks = batch["unavailable_mask"] & is_pick.unsqueeze(1)
            illegal = illegal | unavail_for_picks
        policy_logits = policy_logits.masked_fill(illegal,
                                                    DUPLICATE_LOGIT_NEG)
        return policy_logits

    def _gone_tokens(self, batch):
        """The ban path's token set: this game's bans [+ the fearless pool].

        Returns (embeds, pad_mask, kind_ids, side_ids, raw_champ_embeds).
        `raw_champ_embeds` is the un-normed champion lookup of the same tokens,
        which is what the output-side interaction reads: the formula it
        implements is sum_b e_c^T M e_b over *champion* embeddings, not over
        the ban block's normalised inputs. `pad_mask` is shared by the
        attention and the interaction - the same tokens are PAD for both.

        Bans have no role of their own (role is IGNORE_INDEX on a ban move), so
        they take the FLEX token - unchanged from before. Fearless-pool
        champions *could* be role-tagged: they were picked in an earlier game
        and `datasets.build_fearless_roles` recovers a role for every one of
        them (measured: 156,370 / 156,370, zero fallbacks). The model does not
        read it yet, on purpose. The serving path knows which champions are in
        the pool but not what role they were played in - `infer.build_board`
        takes a bare list of champion ids - so a role-tagged pool token would
        be a real value during training and FLEX at serving, i.e. a silent
        train/serve skew in the exact channel this branch exists to make
        correct. Pool tokens therefore use FLEX like bans, and the role channel
        is left as a documented follow-up that needs a client change first.

        A batch without `fearless_ids` (the serving path before the UI carries
        a pool) is treated as an empty pool rather than raising, so the model
        stays loadable and usable from an old client.
        """
        ban_ids = batch["ban_ids"]                      # [B,10]
        B, dev = ban_ids.shape[0], ban_ids.device
        ban_pad = ban_ids == CHAMP_PAD_ID
        ban_role_ids = torch.full_like(ban_ids, ROLE_FLEX_ID)
        embeds = self.encoder.embed.champ_lookup(ban_ids)
        raw = embeds
        embeds = self.encoder.embed.layer_norm(
            embeds + self.encoder.embed.role_lookup(ban_role_ids))
        pad = ban_pad
        kinds = torch.zeros_like(ban_ids)
        side_ids = None
        if self.ban_context.side_embed is not None:
            table = torch.tensor(BAN_SIDE_BY_SLOT, dtype=torch.long, device=dev)
            sides = table.unsqueeze(0) ^ batch["first_pick"].long().unsqueeze(1)
            side_ids = torch.where(ban_pad,
                                   torch.full_like(sides, SIDE_NONE), sides)
        if self.ban_fearless:
            fids = batch.get("fearless_ids")
            if fids is None:
                fids = torch.full((B, FEARLESS_SLOTS), CHAMP_PAD_ID,
                                  dtype=torch.long, device=dev)
            fpad = fids == CHAMP_PAD_ID
            fraw = self.encoder.embed.champ_lookup(fids)
            fem = self.encoder.embed.layer_norm(
                fraw + self.encoder.embed.role_lookup(
                    torch.full_like(fids, ROLE_FLEX_ID)))
            embeds = torch.cat([embeds, fem], dim=1)
            raw = torch.cat([raw, fraw], dim=1)
            pad = torch.cat([pad, fpad], dim=1)
            kinds = torch.cat([kinds, torch.full_like(fids, KIND_POOL)], dim=1)
            if side_ids is not None:
                side_ids = torch.cat(
                    [side_ids, torch.full_like(fids, SIDE_NONE)], dim=1)
        return embeds, pad, kinds, side_ids, raw

    def _pool_feats(self, batch):
        """Fearless-pool size, [B,1] float (or None).

        Deliberately just the count, not a per-role breakdown: the serving path
        carries champion ids for the pool but no roles (see `_gone_tokens`), so
        a per-role summary would be populated in training and empty at serving.
        Normalised by 10 so it is O(1) - a full pool is 40 champions.

        This is the "the pool is thin" signal, separate from the pool *tokens*
        (which say which champions are gone). The tokens are mean-pooled for
        the interaction term, and a mean cannot express size.
        """
        if self.state_embed.pool_proj is None:
            return None
        ban_ids = batch["ban_ids"]
        fids = batch.get("fearless_ids")
        if fids is None:
            return torch.zeros((ban_ids.shape[0], 1), dtype=torch.float32,
                               device=ban_ids.device)
        total = (fids != CHAMP_PAD_ID).float().sum(1, keepdim=True)
        return total / 10.0

    def _logits(self, batch):
        """Encode -> ban path -> state -> head -> interaction -> legality."""
        # 1. Encode the partial draft through the frozen encoder
        h = self.encoder._encode(batch)  # [B,10,D]

        # 2. Ban path: this game's bans (and the fearless pool) cross-attend
        #    into the slot vectors
        (ban_embeds, ban_pad_mask, kind_ids, side_ids,
         gone_raw) = self._gone_tokens(batch)
        h = self.ban_context(h, ban_embeds, ban_pad_mask,
                             kind_ids=kind_ids, side_ids=side_ids)

        # 3. State embedding (patch id is optional: absent for batches
        # built without a patch vocab, and ignored when n_patches == 0)
        state = self.state_embed(
            batch["move_idx"], batch["phase"],
            batch["to_move"], batch["first_pick"],
            batch.get("patch_id"), self._pool_feats(batch),
        )  # [B, state_dim]

        # 4. Policy head: pool all slot vectors + state -> C logits
        pooled = self._pool(h, batch["pad_mask"])  # [B, 4D]
        # The policy is a distribution over all champions for the next
        # move. We don't condition on a specific slot — the state
        # (phase, side, move_idx) tells the head whether it's a ban or
        # pick and which side is acting.
        policy_input = torch.cat([pooled, state], dim=-1)  # [B, 4D+state]
        policy_logits = self.policy_head(policy_input)  # [B, C]

        # 4b. Output-side interaction: logits_c += e_c^T M (mean gone champ).
        # This is the only path by which a *specific* banned champion can move
        # a *specific* candidate's logit. Added before legality so the term is
        # a property of the model rather than of the mask.
        if self.counter is not None:
            g = self.counter.summary(gone_raw, ban_pad_mask)   # [B,D]
            policy_logits = policy_logits + self.counter(
                self.encoder.embed.champ_lookup.weight, g)

        policy_logits = self._apply_legality(policy_logits, batch)
        return policy_logits, pooled, state, policy_input

    def forward(self, batch):
        """Forward pass: encode partial draft, add context, predict."""
        policy_logits, pooled, state, policy_input = self._logits(batch)

        # 5. Value head: pooled + state -> blue-win logit
        out = {
            "policy_logits": policy_logits,  # [B, C]
            # Exposed so an auxiliary head (pro/model/proto) can reuse the same
            # representation without a second encoder pass. Additive: nothing
            # in the serving path reads it.
            "policy_input": policy_input,     # [B, 4D+state]
        }
        if self.value_head is not None:
            out["value_logit"] = self.value_head(
                torch.cat([pooled, state], dim=-1))  # [B, 1]

        # 6. Optional econ head
        if self.econ_head is not None:
            out["econ_pred"] = self.econ_head(
                torch.cat([pooled, state], dim=-1))  # [B, n_econ]

        return out

    def forward_policy(self, batch):
        """Forward only the policy head (for inference)."""
        with torch.no_grad():
            return self._logits(batch)[0]
