"""Policy-value model for pro draft prediction (Stage 2).

Architecture:
  SoloQ pretrained encoder (frozen) -> slot vectors h [B,10,D]
    + ban context cross-attention (bans attend to slots)
    + turn/phase/side state embedding
    -> Policy head: Linear(D+state, C) with legality -inf masking
    -> Value head: MLP on pooled vector + state

The encoder is the same SoloQPretrainedModel from model/soloq_model.py.
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
Legality is enforced by setting logits of taken/banned/unavailable
champions to -inf (the taken_mask from the dataset unions bans+picks;
unavailable_mask adds fearless-picked champs).

Value head output: [B, 1] win logit for the side to move (or blue-win
logit — configurable). Default: blue-win logit (matches finetune).
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

    def __init__(self, state_dim=64, n_patches=0):
        super().__init__()
        self.move_embed = nn.Embedding(20, state_dim)
        self.phase_embed = nn.Embedding(N_PHASES, state_dim)
        self.side_embed = nn.Embedding(N_SIDES, state_dim)
        self.first_pick_embed = nn.Embedding(N_SIDES, state_dim)
        self.patch_embed = (nn.Embedding(n_patches, state_dim)
                            if n_patches > 0 else None)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, move_idx, phase, to_move, first_pick, patch_id=None):
        """All inputs [B] long -> [B, state_dim]."""
        x = (self.move_embed(move_idx)
             + self.phase_embed(phase)
             + self.side_embed(to_move)
             + self.first_pick_embed(first_pick))
        if self.patch_embed is not None and patch_id is not None:
            x = x + self.patch_embed(patch_id)
        return self.norm(x)


class BanContext(nn.Module):
    """Cross-attention: slot vectors query ban embeddings.

    Banned champions are embedded using the encoder's champion lookup
    (frozen) and then attended to by the slot vectors. This lets each
slot "see" what's been banned when computing its representation.

    PAD ban slots are masked out in the attention.
    """

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.0):
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

    def forward(self, slot_vecs, ban_embeds, ban_pad_mask):
        """slot_vecs [B,10,D], ban_embeds [B,10,D], ban_pad_mask [B,10] bool.

        ban_pad_mask: True = PAD ban slot (not a real ban). Used as
        key_padding_mask in the attention (True = ignore).
        """
        # If all bans are PAD (early draft, no bans yet), skip attention
        # to avoid NaN from all-masked softmax.
        all_pad = ban_pad_mask.all(dim=1, keepdim=True)  # [B,1]
        safe_mask = ban_pad_mask & ~all_pad  # ensure at least 1 key attendable

        attn_out, _ = self.ban_attn(slot_vecs, ban_embeds, ban_embeds,
                                    key_padding_mask=safe_mask,
                                    need_weights=False)
        x = self.norm(slot_vecs + attn_out)
        return self.norm2(x + self.ffn(x))


class PolicyValueModel(nn.Module):
    """Frozen encoder + ban context + state embedding -> policy + value.

    Parameters:
      encoder: SoloQPretrainedModel (frozen by default, can be unfrozen
        for staged training).
      embed_dim: must match encoder's embed_dim.
      state_dim: dimension of the state embedding.
      n_econ: number of economy regression targets (0 = skip econ head).

    Forward batch keys (from ProNextMoveDataset / pro_collate):
      champ_ids [B,10], role_ids [B,10], pad_mask [B,10],
      ban_ids [B,10] (PAD-padded),
      taken_mask [B,C] (banned|picked, for legality),
      unavailable_mask [B,C] (fearless-picked),
      move_idx [B], phase [B], to_move [B], first_pick [B],
      patch_id [B] (optional; only used when n_patches > 0),
      target_champ [B], target_role [B], is_ban [B],
      win_label [B], weight [B], econ_target [B,6], has_econ [B]

    Outputs:
      policy_logits [B, C] (legality-masked, -inf on taken/unavailable)
      value_logit [B, 1] (blue-win logit)
      econ_pred [B, n_econ] (if n_econ > 0)
    """

    def __init__(self, encoder, embed_dim=256, state_dim=64,
                 n_econ=6, dropout=0.0, n_patches=0):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.n_econ = n_econ
        self.n_patches = n_patches

        # Ban context: banned champs attend into slot vectors
        self.ban_context = BanContext(embed_dim, num_heads=4,
                                      dropout=dropout)
        # State embedding
        self.state_embed = StateEmbedding(state_dim, n_patches=n_patches)

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

        # Value head: pooled + state -> blue-win logit
        value_in_dim = pooled_dim + state_dim
        self.value_head = nn.Sequential(
            nn.LayerNorm(value_in_dim),
            nn.Linear(value_in_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, 1),
        )

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
        """Mask taken + unavailable champions to -inf.

        taken_mask: banned|picked (hard legality — you can't ban or pick
        a champ that's already been banned or picked THIS game).
        unavailable_mask: fearless-picked (champs picked in earlier
        series games). These are illegal to PICK but legal to BAN —
        teams routinely ban champs the opponent picked earlier.

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

    def forward(self, batch):
        """Forward pass: encode partial draft, add context, predict."""
        # 1. Encode the partial draft through the frozen encoder
        h = self.encoder._encode(batch)  # [B,10,D]

        # 2. Ban context: embed banned champs and cross-attend into slots
        ban_ids = batch["ban_ids"]  # [B,10] long (PAD = CHAMP_PAD_ID)
        ban_pad_mask = ban_ids == CHAMP_PAD_ID  # [B,10] bool
        # Use the encoder's champion embedding table (frozen) for bans
        ban_embeds = self.encoder.embed.champ_lookup(ban_ids)  # [B,10,D]
        # Add role info for bans (bans have IGNORE_INDEX role, so use FLEX)
        ban_role_ids = torch.full_like(ban_ids, ROLE_FLEX_ID)
        ban_role_embeds = self.encoder.embed.role_lookup(ban_role_ids)
        ban_embeds = self.encoder.embed.layer_norm(
            ban_embeds + ban_role_embeds)
        h = self.ban_context(h, ban_embeds, ban_pad_mask)  # [B,10,D]

        # 3. State embedding (patch id is optional: absent for batches
        # built without a patch vocab, and ignored when n_patches == 0)
        state = self.state_embed(
            batch["move_idx"], batch["phase"],
            batch["to_move"], batch["first_pick"],
            batch.get("patch_id"),
        )  # [B, state_dim]

        # 4. Policy head: pool all slot vectors + state -> C logits
        pooled = self._pool(h, batch["pad_mask"])  # [B, 4D]
        # The policy is a distribution over all champions for the next
        # move. We don't condition on a specific slot — the state
        # (phase, side, move_idx) tells the head whether it's a ban or
        # pick and which side is acting.
        policy_input = torch.cat([pooled, state], dim=-1)  # [B, 4D+state]
        policy_logits = self.policy_head(policy_input)  # [B, C]
        policy_logits = self._apply_legality(policy_logits, batch)

        # 5. Value head: pooled + state -> blue-win logit
        value_input = torch.cat([pooled, state], dim=-1)
        value_logit = self.value_head(value_input)  # [B, 1]

        out = {
            "policy_logits": policy_logits,  # [B, C]
            "value_logit": value_logit,     # [B, 1]
        }

        # 6. Optional econ head
        if self.econ_head is not None:
            out["econ_pred"] = self.econ_head(value_input)  # [B, n_econ]

        return out

    def forward_policy(self, batch):
        """Forward only the policy head (for inference)."""
        with torch.no_grad():
            h = self.encoder._encode(batch)
            ban_ids = batch["ban_ids"]
            ban_pad_mask = ban_ids == CHAMP_PAD_ID
            ban_embeds = self.encoder.embed.champ_lookup(ban_ids)
            ban_role_ids = torch.full_like(ban_ids, ROLE_FLEX_ID)
            ban_role_embeds = self.encoder.embed.role_lookup(ban_role_ids)
            ban_embeds = self.encoder.embed.layer_norm(
                ban_embeds + ban_role_embeds)
            h = self.ban_context(h, ban_embeds, ban_pad_mask)
            state = self.state_embed(
                batch["move_idx"], batch["phase"],
                batch["to_move"], batch["first_pick"],
                batch.get("patch_id"),
            )
            pooled = self._pool(h, batch["pad_mask"])
            policy_input = torch.cat([pooled, state], dim=-1)
            logits = self.policy_head(policy_input)
            logits = self._apply_legality(logits, batch)
        return logits
