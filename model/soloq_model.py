"""SoloQ pretrained draft model: masked champion + role prediction.

Architecture (see model/README.md):
  DraftSlotEmbedding (champ + role sum -> LayerNorm, NO team embedding)
    -> Siamese intra-team self-attention (shared weights blue/red)
    -> symmetric inter-team cross-attention (shared weights both directions)
    -> slot heads: champ Linear(D, C) + role Linear(D, 5)

Set/permutation-invariant within each team: no positional embeddings.
Forward I/O exactly matches model/README.md Sec.3-4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SLOTS = 10
TEAM_SIZE = 5
NUM_REAL_CHAMPS = 173  # patch-specific; constructor `num_champs` overrides
CHAMP_MASK_ID = NUM_REAL_CHAMPS
CHAMP_PAD_ID = NUM_REAL_CHAMPS + 1
CHAMP_VOCAB_SIZE = NUM_REAL_CHAMPS + 2  # REAL + MASK + PAD = 175 (required)
NUM_ROLES = 5
ROLE_VOCAB_SIZE = 6  # input only: 0..4 + FLEX/masked-role token
ROLE_FLEX_ID = 5
ROLE_HEAD_DIM = 5  # output: exactly the 5 real roles, no FLEX column
IGNORE_INDEX = -100
DUPLICATE_LOGIT_NEG = float("-inf")  # suppress visible champs (Sec 4.1)


class DraftSlotEmbedding(nn.Module):
    """LayerNorm(champ_lookup(champ_ids) + role_lookup(role_ids))."""

    def __init__(self, num_champs=NUM_REAL_CHAMPS, num_roles=ROLE_VOCAB_SIZE,
                 embed_dim=128):
        super().__init__()
        self.champ_lookup = nn.Embedding(num_champs + 2, embed_dim)  # +M+P
        self.role_lookup = nn.Embedding(num_roles, embed_dim)  # 5 + FLEX
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, champ_ids, role_ids):
        x = self.champ_lookup(champ_ids) + self.role_lookup(role_ids)
        return self.layer_norm(x)


class SymmetricCrossAttention(nn.Module):
    """Shared cross-attention applied in both directions (Siamese teams)."""

    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def _one_way(self, query, kv, kv_pad_mask):
        a, _ = self.attn(query, kv, kv, key_padding_mask=kv_pad_mask,
                         need_weights=False)
        query = self.norm1(query + a)
        return self.norm2(query + self.ffn(query))

    def forward(self, blue, red, blue_pad_mask=None, red_pad_mask=None):
        return (self._one_way(blue, red, red_pad_mask),
                self._one_way(red, blue, blue_pad_mask))



class SoloQPretrainedModel(nn.Module):
    """Masked-draft encoder -> champ_logits [B,10,C] + role_logits [B,10,5]."""

    def __init__(self, num_champs=NUM_REAL_CHAMPS, embed_dim=128, num_heads=8,
                 num_intra_layers=2, num_cross_layers=1, dropout=0.0):
        super().__init__()
        self.num_champs = num_champs
        self.embed = DraftSlotEmbedding(num_champs=num_champs,
                                        embed_dim=embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                           nhead=num_heads,
                                           dim_feedforward=embed_dim * 4,
                                           dropout=dropout,
                                           activation="gelu",
                                           batch_first=True)
        # ONE instance applied to both teams -> Siamese intra-team encoder.
        self.intra_team = nn.TransformerEncoder(layer,
                                                num_layers=num_intra_layers)
        self.cross_team = nn.ModuleList(
            [SymmetricCrossAttention(embed_dim, num_heads, dropout)
             for _ in range(num_cross_layers)])
        self.champ_head = nn.Linear(embed_dim, num_champs)
        self.role_head = nn.Linear(embed_dim, NUM_ROLES)

    def _apply_duplicate_mask(self, champ_logits, batch):
        """Set logits of already-visible champs to -inf at masked positions."""
        """Set logits of already-visible champs to -inf at masked positions."""
        champ_ids = batch["champ_ids"]
        is_m = batch["is_masked"]
        is_pad = batch.get("pad_mask")
        if is_pad is None:
            is_pad = champ_ids == (self.num_champs + 1)
        visible = (~is_m) & (~is_pad)  # slots showing a real champion
        safe_ids = champ_ids.clamp(max=self.num_champs - 1)
        one_hot = F.one_hot(safe_ids, num_classes=self.num_champs)
        one_hot = one_hot * visible.unsqueeze(-1)
        taken = one_hot.sum(dim=1, keepdim=True) > 0  # [B,1,C] visible set
        suppress = taken.expand(-1, N_SLOTS, -1) & is_m.unsqueeze(-1)
        return champ_logits.masked_fill(suppress, DUPLICATE_LOGIT_NEG)

    def _encode(self, batch):
        """Slot vectors h [B,10,D] after intra + cross attention.

        Shared by forward() and the pro finetune stage (pooling/value
        heads). No positional embeddings: permuting slots within a team
        permutes that team's outputs identically.
        """
        champ_ids = batch["champ_ids"]  # [B,10] long
        role_ids = batch["role_ids"]  # [B,10] long
        pad_mask = batch.get("pad_mask")  # [B,10] bool or None
        B = champ_ids.shape[0]
        if pad_mask is None:
            pad_mask = champ_ids == (self.num_champs + 1)

        x = self.embed(champ_ids, role_ids)  # [B,10,D], no positions
        blue, red = x[:, :TEAM_SIZE], x[:, TEAM_SIZE:]  # [B,5,D] each
        blue_pad, red_pad = pad_mask[:, :TEAM_SIZE], pad_mask[:, TEAM_SIZE:]
        # PyTorch MHA NaNs when EVERY key is masked (softmax over all -inf).
        # Partial drafts can legitimately show zero picks for one team
        # (win-head value on early states); treat such a team's slots as
        # attendable. Pooled team vectors still zero out via keep-mask, and
        # PAD slots never carry labels/mask targets, so no contract change.
        blue_pad = blue_pad & ~blue_pad.all(dim=1, keepdim=True)
        red_pad = red_pad & ~red_pad.all(dim=1, keepdim=True)

        both = torch.cat([blue, red], dim=0)  # [2B,5,D], shared encoder
        both_pad = torch.cat([blue_pad, red_pad], dim=0)  # [2B,5]
        both = self.intra_team(both, src_key_padding_mask=both_pad)
        blue, red = both[:B], both[B:]

        for block in self.cross_team:  # symmetric exchange, shared weights
            blue, red = block(blue, red, blue_pad, red_pad)

        return torch.cat([blue, red], dim=1)  # [B,10,D]

    def forward(self, batch, compute_heads=True):
        h = self._encode(batch)
        if not compute_heads:
            return {"h": h}
        champ_logits = self._apply_duplicate_mask(self.champ_head(h), batch)
        role_logits = self.role_head(h)
        return {"champ_logits": champ_logits, "role_logits": role_logits}
