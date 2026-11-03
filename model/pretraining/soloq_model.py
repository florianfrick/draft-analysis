"""SoloQ pretrained draft model: masked champion + role prediction.

Architecture (see model/README.md):
  DraftSlotEmbedding (champ + role sum -> LayerNorm, NO team embedding)
    -> Siamese intra-team self-attention (shared weights blue/red)
    -> symmetric inter-team cross-attention (shared weights both directions)
    -> slot heads: champ Linear(D, C) + role Linear(D, 5)
    -> [opt-in] PMA team pooling -> value vector -> win head (blue wins)

Set/permutation-invariant within each team: no positional embeddings.
Forward I/O exactly matches model/README.md Sec.3-4.

The value path is OFF by default (`with_value=False`): the masked-modelling
contract and every pre-existing checkpoint stay exactly as they were. SoloQ
pretraining turns it on so the pooled representation is trained against the
outcome label that has been sitting unused in every shard.
"""
import json
from pathlib import Path

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

CHAMP_ATTRS_PATH = (Path(__file__).resolve().parent.parent.parent
                    / "data" / "champ_attrs.json")


def load_champ_attr_matrix(num_champs=NUM_REAL_CHAMPS, path=None,
                           expect_dim=None):
    """-> FloatTensor [num_champs + 2, K] scaled to 0..1, or None.

    The hand-tagged kit table (data/champ_attrs.json) as a fixed matrix. The
    ordinal 0..3 values are divided by 3 so all axes share a scale; anything
    fancier (standardising, whitening) is left to the adapter, which can learn
    it from 18 inputs far more cheaply than it can from 173 embeddings.

    The two extra rows are MASK and PAD: zero vectors, so a masked or empty
    slot contributes nothing. A champion with no tagged row (none at present)
    gets the mean of the tagged ones - least informative, but it keeps the
    matrix rectangular and the loader honest.
    """
    path = Path(path) if path is not None else CHAMP_ATTRS_PATH
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    tagged = [e["attrs"] for e in payload["champions"].values()
              if e.get("attrs") is not None]
    if not tagged:
        return None
    k = len(payload["axes"])
    if expect_dim is not None and k != expect_dim:
        raise RuntimeError(
            "%s has %d axes but the checkpoint was trained with %d - the "
            "table changed since that run, so it cannot be rebuilt"
            % (path.name, k, expect_dim))
    mean = [sum(row[j] for row in tagged) / len(tagged) for j in range(k)]
    mat = torch.tensor([mean] * num_champs, dtype=torch.float32)
    for entry in payload["champions"].values():
        idx = int(entry["idx"])
        if not 0 <= idx < num_champs or entry.get("attrs") is None:
            continue
        mat[idx] = torch.tensor(entry["attrs"], dtype=torch.float32)
    mat = mat / 3.0
    return torch.cat([mat, torch.zeros(2, k)], dim=0)


def attr_matrix_for_config(cfg, num_champs=None):
    """The kit-attribute matrix a checkpoint was trained with, or None.

    Every stage rebuilds the encoder from the checkpoint's `config`, so
    `attr_dim` living there is the one thing that decides whether the adapter
    exists. That matters more than it looks: a stage that forgot to pass the
    matrix would still *load* the checkpoint without complaint (missing
    `attr_proj.*` is tolerated, because zero-init means absent == correctly
    initialised) and would then serve a model whose adapter is frozen at zero -
    silently different from the one that was trained. Reading the flag from
    the checkpoint makes that mistake impossible to make.
    """
    cfg = cfg or {}
    k = int(cfg.get("attr_dim") or 0)
    if not k:
        return None
    return load_champ_attr_matrix(
        num_champs=int(num_champs or cfg.get("num_champs", NUM_REAL_CHAMPS)),
        expect_dim=k)


class DraftSlotEmbedding(nn.Module):
    """LayerNorm(champ_lookup + role_lookup [+ attr_proj(champ_attrs)] [+ patch]).

    `attr_matrix` turns on the kit-attribute residual (STATUS.md §6).
    The projection is zero-initialised, so with the same champ/role tables this
    module computes exactly what it did before the adapter existed. That is the
    whole point: "attributes on/off" becomes a single-variable comparison
    instead of two different initialisations, and an old checkpoint loads
    straight into an attribute-enabled encoder with the adapter still at zero.

    `n_patches` turns on the patch embedding: one learned vector per patch
    (index 0 = UNK), added to *every* slot, i.e. a segment embedding. It is
    zero-initialised for the same reason as the attribute adapter - at step 0
    the model is bit-identical to one without it, so "patch on/off" is one
    flag and one seed, not two different networks. What it buys, when the
    patch is in the training data at all, is a per-patch prior shift on top of
    the champion table: the champion table says what a champion is, the patch
    vector says how much it is worth right now. See
    data/pro play/pipeline.split_within_patch for the protocol that makes the
    patch a *seen* input at test time (the global date split does not).
    """

    def __init__(self, num_champs=NUM_REAL_CHAMPS, num_roles=ROLE_VOCAB_SIZE,
                 embed_dim=128, attr_matrix=None, n_patches=0):
        super().__init__()
        self.champ_lookup = nn.Embedding(num_champs + 2, embed_dim)  # +M+P
        self.role_lookup = nn.Embedding(num_roles, embed_dim)  # 5 + FLEX
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attr_dim = 0 if attr_matrix is None else int(attr_matrix.shape[-1])
        self.n_patches = int(n_patches or 0)
        if self.n_patches:
            self.patch_lookup = nn.Embedding(self.n_patches, embed_dim)
            nn.init.zeros_(self.patch_lookup.weight)
        else:
            self.patch_lookup = None
        if self.attr_dim:
            # Non-persistent: the matrix is data, not a learned parameter, and
            # keeping it out of the state dict keeps checkpoints small and
            # portable across a table revision (the shape check above catches
            # a revision that would silently change meaning).
            self.register_buffer("champ_attrs", attr_matrix.float(),
                                 persistent=False)
            self.attr_proj = nn.Linear(self.attr_dim, embed_dim)
            nn.init.zeros_(self.attr_proj.weight)
            nn.init.zeros_(self.attr_proj.bias)
        else:
            self.attr_proj = None

    def forward(self, champ_ids, role_ids, patch_id=None):
        x = self.champ_lookup(champ_ids) + self.role_lookup(role_ids)
        if self.attr_proj is not None:
            x = x + self.attr_proj(self.champ_attrs[champ_ids])
        if self.patch_lookup is not None:
            # Missing patch_id means "unknown patch" (index 0), not "no patch
            # signal": a batch assembled without a patch vocabulary - a soloQ
            # rehearsal batch, or an inference call from an old client - lands
            # on the UNK row, which is exactly the row a *new* patch would use.
            if patch_id is None:
                patch_id = champ_ids.new_zeros(champ_ids.shape[0])
            x = x + self.patch_lookup(patch_id).unsqueeze(1)
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



class PMAPool(nn.Module):
    """Set Transformer pooling by multihead attention (PMA), one team at a time.

    `num_seeds` learned query vectors attend over a team's 5 slot vectors and
    the result is averaged over seeds. This replaces the PAD-masked mean that
    the value head used to read.

    Why: a mean over 5 slots averages champion identity away - the pooled
    vector is dominated by whichever slots happen to be present, and the
    per-slot differences are divided by 5. That is the most likely reason the
    value head sat at the base rate for 8k steps. PMA instead learns *what to
    attend to* for the outcome, while staying permutation-invariant (queries
    are fixed, keys are a set) and Siamese (one instance serves both teams).

    An all-PAD team would softmax over nothing and produce NaN, so one key is
    kept attendable and the output is zeroed by `empty` instead - matching the
    mean-pool behaviour for partial drafts where a team has no picks yet.
    """

    def __init__(self, embed_dim=128, num_heads=8, num_seeds=4, dropout=0.0):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, slots, pad_mask):
        """slots [B,5,D], pad_mask [B,5] (True = PAD) -> [B,D]."""
        B = slots.shape[0]
        empty = pad_mask.all(dim=1, keepdim=True)      # [B,1] no picks on this team
        safe = pad_mask & ~empty                       # keep >= 1 attendable key
        q = self.seeds.unsqueeze(0).expand(B, -1, -1)  # [B,S,D]
        a, _ = self.attn(q, slots, slots, key_padding_mask=safe,
                         need_weights=False)
        x = self.norm1(q + a)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1) * (~empty).to(x.dtype)


def mean_pool_vector(h, pad_mask):
    """The readout the value head used before PMA: concat(vb, vr, vb-vr, vb*vr).

    Kept as a module-level function so `probe_value.py` can measure the mean
    readout and the PMA readout on the same checkpoint - that comparison is
    the evidence for whether PMA is worth its parameters.
    """
    keep = (~pad_mask).float().unsqueeze(-1)           # [B,10,1]
    blue, red = h[:, :TEAM_SIZE], h[:, TEAM_SIZE:]
    kb, kr = keep[:, :TEAM_SIZE], keep[:, TEAM_SIZE:]
    vb = (blue * kb).sum(1) / kb.sum(1).clamp_min(1.0)
    vr = (red * kr).sum(1) / kr.sum(1).clamp_min(1.0)
    return torch.cat([vb, vr, vb - vr, vb * vr], dim=1)


def load_encoder_state(module, state_dict, allow_missing_value=True,
                       allow_missing_patch=False):
    """Load a pretraining state dict into an encoder, tolerating optional paths.

    Two asymmetries, on purpose:

    * **Dropped** keys are only allowed for the value path (`pma.*`,
      `win_head.*`). Downstream stages build an encoder without them, so
      filtering is correct. An `embed.attr_proj.*` key that the target does
      not have is a real mismatch - dropping it would silently turn the
      attribute adapter off and quietly evaluate a different model. The
      same holds for `embed.patch_lookup.*`: a checkpoint that *has* patch
      rows must never be loaded into an encoder without them.
    * **Missing** keys are allowed for the value path and for
      `embed.attr_proj.*`: the adapter is zero-initialised, so "absent from
      the checkpoint" and "correctly initialised" are the same tensor. That is
      what lets a pre-attribute checkpoint load into an attribute-enabled
      encoder unchanged.

    `embed.patch_lookup.*` is missing-tolerant **only** when the caller sets
    `allow_missing_patch=True`, because the patch table is zero-initialised
    for exactly the same reason as the attribute adapter - absent and
    correctly-initialised are the same tensor there too. It stays opt-in
    because the default path is the *evaluation* path: there, a checkpoint
    with no patch rows entering a patch-enabled encoder must be loud rather
    than silently scoring a model nobody validated. The opt-in is for the
    *initialisation* path - seeding a patch-enabled encoder from a trunk
    that predates patch conditioning - where a zero table is the intended
    starting point, not a silent substitution.

    Any other mismatch (wrong embed_dim, wrong layer count) still raises.
    """
    own = set(module.state_dict().keys())
    optional = ("pma.", "win_head.", "embed.attr_proj.")
    if allow_missing_patch:
        optional = optional + ("embed.patch_lookup.",)
    keep = {k: v for k, v in state_dict.items() if k in own}
    dropped = sorted(set(state_dict) - set(keep))
    if dropped and not (allow_missing_value and
                        all(k.startswith(("pma.", "win_head."))
                            for k in dropped)):
        raise RuntimeError("checkpoint has keys the encoder does not know: %s"
                           % dropped[:8])
    missing = sorted(own - set(keep))
    bad = [k for k in missing if not k.startswith(optional)]
    if bad:
        raise RuntimeError("encoder needs weights the checkpoint lacks: %s"
                           % bad[:8])
    module.load_state_dict(keep, strict=False)
    return {"dropped": dropped, "missing": missing}



class SoloQPretrainedModel(nn.Module):
    """Masked-draft encoder -> champ_logits [B,10,C] + role_logits [B,10,5]."""

    def __init__(self, num_champs=NUM_REAL_CHAMPS, embed_dim=128, num_heads=8,
                 num_intra_layers=2, num_cross_layers=1, dropout=0.0,
                 with_value=False, value_seeds=4, attr_matrix=None,
                 n_patches=0):
        super().__init__()
        self.num_champs = num_champs
        self.n_patches = int(n_patches or 0)
        self.embed = DraftSlotEmbedding(num_champs=num_champs,
                                        embed_dim=embed_dim,
                                        attr_matrix=attr_matrix,
                                        n_patches=self.n_patches)
        self.attr_dim = self.embed.attr_dim
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

        # Optional value path. Off by default so the masked-modelling contract
        # (and every existing checkpoint) is unchanged; pretraining turns it
        # on to give the encoder an outcome-aware pooled representation.
        self.with_value = with_value
        if with_value:
            self.pma = PMAPool(embed_dim, num_heads, value_seeds, dropout)
            self.win_head = nn.Sequential(
                nn.LayerNorm(embed_dim * 4),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.pma = None
            self.win_head = None

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

        x = self.embed(champ_ids, role_ids, batch.get("patch_id"))  # [B,10,D]
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

    def value_vector(self, batch, h=None):
        """Draft-level value vector z [B, 4D] from PMA-pooled team vectors.

        Same layout as the old mean pool (vb, vr, vb-vr, vb*vr) so the value
        head and the probe are a drop-in swap of the readout: the diff and
        product terms still expose asymmetry and interaction explicitly.
        """
        if not self.with_value:
            raise RuntimeError("value_vector needs with_value=True")
        if h is None:
            h = self._encode(batch)
        pad_mask = batch.get("pad_mask")
        if pad_mask is None:
            pad_mask = batch["champ_ids"] == CHAMP_PAD_ID
        vb = self.pma(h[:, :TEAM_SIZE], pad_mask[:, :TEAM_SIZE])
        vr = self.pma(h[:, TEAM_SIZE:], pad_mask[:, TEAM_SIZE:])
        return torch.cat([vb, vr, vb - vr, vb * vr], dim=1)

    def forward(self, batch, compute_heads=True, compute_value=None):
        h = self._encode(batch)
        if not compute_heads:
            return {"h": h}
        champ_logits = self._apply_duplicate_mask(self.champ_head(h), batch)
        role_logits = self.role_head(h)
        out = {"champ_logits": champ_logits, "role_logits": role_logits}
        if compute_value is None:
            compute_value = self.with_value
        if compute_value:
            out["win_logit"] = self.win_head(self.value_vector(batch, h))
        return out

    @torch.no_grad()
    def forward_value(self, batch):
        """Value only (inference): blue-win logit [B,1]."""
        return self.win_head(self.value_vector(batch))
