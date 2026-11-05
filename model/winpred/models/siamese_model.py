"""Siamese draft encoder -> blue-win logit.

Adapted from `model/pretraining/soloq_model.py` (same blocks, same embedding
contract), with the masked-modelling heads replaced by one outcome head:

    slot embedding = LayerNorm(champ + role [+ attrs] [+ patch])
      -> Siamese intra-team self-attention   (ONE encoder, applied to [2B,5,D])
      -> symmetric inter-team cross-attention (ONE block, both directions)
         + an additive, learned role x role bias on the cross attention logits
      -> per-team pooled vector h_blue, h_red
      -> logit = MLP([h_blue, h_red, h_blue - h_red, h_blue * h_red]) + b_side

Design notes, each of which is a decision rather than a default
---------------------------------------------------------------
* **No bans, no prefix, no masking.** The input is the finished 10-pick board.
  `champ_ids` is always full, so `pad_mask` is all-False and no mask token is
  ever used. That also means the model cannot be confused by a PAD slot, which
  is the failure mode a partial-draft input would introduce here.
* **The embedding carries four things**, all of them inputs that exist before
  the game starts:
  - *champion* - the 173-row table shared with every other model in the repo;
  - *role* - TOP/JGL/MID/BOT/SUP. Because there are no positional embeddings,
    this is also the only thing that tells the set "who is the support";
  - *patch* - one learned vector added to every slot (a segment embedding),
    zero-initialised so "patch on/off" is one flag and one seed, not two
    networks. It is legitimate here because the split is random *within* each
    patch: the same patch is on both sides, so this interpolates a seen patch
    instead of memorising an unseen id (the failure mode STATUS.md §6 records
    for the date split);
  - *attributes* (opt-in) - the 18-axis kit table via a zero-init
    `Linear(18, D)`, the same adapter as the encoder. Off for the first run:
    §6 measured it as indistinguishable from zero, and the brief says do not
    run extra experiments yet.
* **`h_blue - h_red` is an explicit input to the head**, not something the MLP
  has to discover. `[h_blue, h_red, h_blue - h_red]` is the same asymmetry
  recipe the repo's value heads use, and it makes the head's job "how much does
  the difference matter" rather than "compare two vectors".
* **`h_blue * h_red` is the second-order term** (`head_product=True`). The
  difference is linear in each team vector; the product is the elementwise
  *interaction* - "these two team vectors point the same way along dimension d",
  which is the shape of a match-up (a pick that is good into the other pick) and
  is what a pure difference cannot express: `(a-b)` and `(a'-b)` are
  indistinguishable whenever `a-a'` is orthogonal to the weight row, and the
  product separates exactly those cases. It is also **symmetric** under a
  blue/red swap, so it cannot carry side information - all of the asymmetry
  still has to come from `h_blue - h_red` and `b_side`, which is the split we
  want. Cost is one extra `D`-wide input to the head: +0.066 M parameters at
  D=256 (a 2.5% increase).
* **A role x role bias on the cross attention** (`role_bias=True`). Cross-team
  attention starts with no notion that a TOP query should weight the enemy TOP
  more than the enemy SUP; the slot embeddings carry the role, so it *can* learn
  this, but from 37 k pro games it has no reason to spend capacity on it. A
  5x5 additive matrix on the attention logits states the prior once: same-role
  pairs, jungle-to-everything, and the bot/sup pair. One instance is applied in
  both directions (Siamese), and it is indexed by *role id*, not by slot
  position, so permuting the five slots still cannot change the output.
  Not applied to the intra-team encoder: inside one team each role appears
  exactly once, so a "same-role" prior there has no pair to act on.
* **A scalar `side_bias`** is added to the logit. It is the whole blue-side
  prior: one parameter, initialised at 0, so the model has to learn it. This is
  what makes the baseline comparison meaningful - the network has exactly the
  baseline's one degree of freedom available, plus the draft.
* **No blue/red swap augmentation.** It would make the task exactly symmetric,
  which forces `side_bias` to 0 (a mirrored draft must satisfy
  `p(mirror) = 1 - p(orig)`), i.e. it would delete the parameter the experiment
  is about. The architecture is still Siamese - one set of weights serves both
  teams - it is the *augmentation* that is incompatible, not the symmetry.
* **No value head, no econ head, no role/champ head.** One logit out.
"""
import torch
import torch.nn as nn

from model.pretraining.soloq_model import (CHAMP_PAD_ID, PMAPool,
                                           SymmetricCrossAttention,
                                           DraftSlotEmbedding,
                                           attr_matrix_for_config,
                                           load_champ_attr_matrix)

N_SLOTS = 10
TEAM_SIZE = 5
NUM_REAL_CHAMPS = 173

# Where the role prior starts. Kept as module-level constants rather than
# literals inside the initialiser so a test can assert the numbers the design
# claims, and so the README has one place to point at.
ROLE_PRIOR = {
    "same_role": 0.5,     # TOP vs TOP, ... SUP vs SUP
    "jungle": 0.2,        # row 1 and column 1: jungle touches every lane
    "bot_sup": 0.3,       # 3 <-> 4: the one lane that is a pair
}


def init_role_bias_(role_bias, team_size=TEAM_SIZE):
    """Fill a [5,5] cross-attention role prior in place.

    Rows are the *query* slot's role, columns the *key* slot's role, so
    `role_bias[i, j]` is added to the attention logit of "role i looks at role
    j". Positive = attend more.

    These three entries are the only prior knowledge in the whole model, and all
    three are about the map rather than about any champion: a lane opponent is
    the natural reference class, the jungler is the only role not tied to one
    lane, and bot/sup are the only two roles that share a lane. Everything else
    starts at zero and has to be learned.

    Note `role_bias[1, 1]` ends at 0.9, not 0.5: it is on the diagonal *and* on
    the jungle row and column. That is the intended reading (the jungle mirror
    is the strongest single pair on the board), not an oversight.
    """
    if tuple(role_bias.shape) != (team_size, team_size):
        raise ValueError("role_bias must be [%d,%d], got %s"
                         % (team_size, team_size, tuple(role_bias.shape)))
    with torch.no_grad():
        role_bias.zero_()
        for i in range(team_size):
            role_bias[i, i] = ROLE_PRIOR["same_role"]
        role_bias[1, :] += ROLE_PRIOR["jungle"]          # jungle row
        role_bias[:, 1] += ROLE_PRIOR["jungle"]          # jungle column
        role_bias[3, 4] = ROLE_PRIOR["bot_sup"]
        role_bias[4, 3] = ROLE_PRIOR["bot_sup"]
    return role_bias


class RoleAwareCrossAttention(SymmetricCrossAttention):
    """`SymmetricCrossAttention` + a learned additive role x role bias.

    A strict superset of the parent: same projections, same norms, same FFN,
    same state-dict key names, plus one [5,5] parameter. With `role_bias`
    zeroed the two blocks are numerically identical, which is what
    `test_zeroed_role_bias_reproduces_the_plain_block` asserts - so a
    difference in the experiment is attributable to the bias and nothing else.

    The bias is gathered by *role id*, not by slot index. In this dataset the
    two coincide (slot i of a team is always role i), but keying on the id is
    what keeps the block honest: the test that shuffles slots within a team
    still passes, because the bias follows the role rather than the position.
    """

    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1,
                 team_size=TEAM_SIZE):
        super().__init__(embed_dim, num_heads, dropout)
        self.team_size = int(team_size)
        self.role_bias = init_role_bias_(
            nn.Parameter(torch.zeros(team_size, team_size)), self.team_size)

    def _additive_mask(self, role_q, role_k, dtype):
        """[B,5] x [B,5] role ids -> [B*num_heads,5,5] additive attn mask."""
        bias = self.role_bias[role_q.unsqueeze(-1), role_k.unsqueeze(-2)]
        return bias.to(dtype).repeat_interleave(self.attn.num_heads, dim=0)

    def _one_way(self, query, kv, kv_pad_mask, role_q, role_k):
        a, _ = self.attn(query, kv, kv, key_padding_mask=kv_pad_mask,
                         attn_mask=self._additive_mask(role_q, role_k,
                                                       query.dtype),
                         need_weights=False)
        query = self.norm1(query + a)
        return self.norm2(query + self.ffn(query))

    def forward(self, blue, red, blue_pad_mask=None, red_pad_mask=None,
                blue_roles=None, red_roles=None):
        """Roles default to canonical order, for standalone use in tests."""
        if blue_roles is None or red_roles is None:
            ar = torch.arange(blue.shape[1], device=blue.device)
            blue_roles = ar.unsqueeze(0).expand(blue.shape[0], -1)
            red_roles = ar.unsqueeze(0).expand(red.shape[0], -1)
        return (self._one_way(blue, red, red_pad_mask, blue_roles, red_roles),
                self._one_way(red, blue, blue_pad_mask, red_roles, blue_roles))


class WinPredictor(nn.Module):
    """Full 10-pick draft -> P(blue wins) as a single logit."""

    def __init__(self, num_champs=NUM_REAL_CHAMPS, embed_dim=256,
                 num_intra=2, num_cross=1, num_heads=8, dropout=0.1,
                 n_patches=0, attr_matrix=None, pool="mean", head_dim=None,
                 head_product=False, role_bias=False, team_feat=False):
        super().__init__()
        if pool not in ("mean", "pma"):
            raise ValueError("pool must be 'mean' or 'pma', got %r" % (pool,))
        self.num_champs = int(num_champs)
        self.embed_dim = int(embed_dim)
        self.n_patches = int(n_patches or 0)
        self.pool = pool
        # Kept as flags on the module so `config_of` can round-trip them: a
        # checkpoint that does not record which blocks it was built from cannot
        # be rebuilt, and `load_checkpoint` is strict on purpose.
        self.head_product = bool(head_product)
        self.use_role_bias = bool(role_bias)
        # Team strength is NOT a draft input. It is off by default and, when on,
        # it changes the question from "is this draft good" to "who wins this
        # game" - README §10 measures the difference (0.044 nats vs 0.005).
        self.team_feat = bool(team_feat)
        self.head_in = (self.embed_dim * (4 if self.head_product else 3)
                        + (3 if self.team_feat else 0))

        self.embed = DraftSlotEmbedding(num_champs=self.num_champs,
                                        embed_dim=self.embed_dim,
                                        attr_matrix=attr_matrix,
                                        n_patches=self.n_patches)
        self.attr_dim = self.embed.attr_dim

        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                           nhead=num_heads,
                                           dim_feedforward=self.embed_dim * 4,
                                           dropout=dropout,
                                           activation="gelu",
                                           batch_first=True)
        # ONE instance applied to both teams -> Siamese intra-team encoder.
        self.intra_team = nn.TransformerEncoder(layer,
                                                num_layers=num_intra)
        block_cls = (RoleAwareCrossAttention if self.use_role_bias
                     else SymmetricCrossAttention)
        self.cross_team = nn.ModuleList(
            [block_cls(self.embed_dim, num_heads, dropout)
             for _ in range(num_cross)])
        self.pma = PMAPool(self.embed_dim, num_heads) if pool == "pma" else None

        hidden = int(head_dim or self.embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.head_in),
            nn.Linear(self.head_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            # bias=False on purpose: a bias here is a second constant added to
            # the logit, i.e. a second, redundant side prior. With both, the
            # learned split between them is arbitrary - measured on the first
            # run, `side_bias` ended at +0.004 while the prior's logit is
            # +0.133, because the head's bias had absorbed the rest.
            #
            # Dropping it removes the EXPLICIT duplicate, and that is all: on
            # the six-arm sweep (README §6) soloQ learns the prior into
            # `side_bias` (-0.082 vs a -0.094 target) while pro still does not
            # (+0.004 vs +0.133), because LayerNorm -> GELU -> Linear can
            # synthesise a constant of its own along the features' mean
            # direction. So read `side_bias` as "the side prior, if the model
            # chose to put it here" - never as "the learned side advantage".
            nn.Linear(hidden, 1, bias=False),
        )
        # The entire blue-side prior, in one parameter. Init 0 = no prior.
        self.side_bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------- encode
    def encode(self, batch):
        """-> (h_blue [B,D], h_red [B,D]) after intra + cross attention."""
        champ_ids = batch["champ_ids"]
        role_ids = batch["role_ids"]
        B = champ_ids.shape[0]
        pad = batch.get("pad_mask")
        if pad is None:
            pad = champ_ids == CHAMP_PAD_ID          # full drafts: all False
        pad = pad.to(torch.bool)

        x = self.embed(champ_ids, role_ids, batch.get("patch_id"))
        blue, red = x[:, :TEAM_SIZE], x[:, TEAM_SIZE:]
        blue_pad, red_pad = pad[:, :TEAM_SIZE], pad[:, TEAM_SIZE:]
        # PyTorch MHA NaNs when every key is masked; a full draft never hits
        # this, but a one-team batch (a unit test) would.
        blue_pad = blue_pad & ~blue_pad.all(dim=1, keepdim=True)
        red_pad = red_pad & ~red_pad.all(dim=1, keepdim=True)

        both = torch.cat([blue, red], dim=0)         # [2B,5,D], shared weights
        both_pad = torch.cat([blue_pad, red_pad], dim=0)
        # Full drafts have nothing padded, and passing an all-False mask would
        # route MHA through the (slower, prototype) nested-tensor path for no
        # reason. Only a caller with a genuinely partial draft needs it.
        both = self.intra_team(both, src_key_padding_mask=both_pad
                               if bool(both_pad.any()) else None)
        blue, red = both[:B], both[B:]
        pad_args = (blue_pad, red_pad) if bool(both_pad.any()) else (None, None)
        # Only the role-aware block takes role ids; `SymmetricCrossAttention`
        # does not, and every checkpoint written before it existed must still
        # load (README §7's six arms).
        role_args = ((role_ids[:, :TEAM_SIZE], role_ids[:, TEAM_SIZE:])
                     if self.use_role_bias else ())
        for block in self.cross_team:
            blue, red = block(blue, red, *pad_args, *role_args)
        return blue, red

    def team_vectors(self, batch, h=None):
        """Pooled per-team vectors. Full drafts -> a plain mean over 5 slots."""
        if h is None:
            h = self.encode(batch)
        blue, red = h
        pad = batch.get("pad_mask")
        if pad is None:
            pad = batch["champ_ids"] == CHAMP_PAD_ID
        pad = pad.to(torch.bool)
        blue_pad, red_pad = pad[:, :TEAM_SIZE], pad[:, TEAM_SIZE:]
        if self.pma is not None:
            return self.pma(blue, blue_pad), self.pma(red, red_pad)
        kb = (~blue_pad).to(blue.dtype).unsqueeze(-1)
        kr = (~red_pad).to(red.dtype).unsqueeze(-1)
        vb = (blue * kb).sum(1) / kb.sum(1).clamp_min(1.0)
        vr = (red * kr).sum(1) / kr.sum(1).clamp_min(1.0)
        return vb, vr

    # ------------------------------------------------------------- forward
    def head_input(self, vb, vr, team_feat=None):
        """The head's view of the matchup: both vectors, their difference,
        (opt-in) their elementwise product, and (opt-in) the team term."""
        parts = [vb, vr, vb - vr]
        if self.head_product:
            parts.append(vb * vr)
        if self.team_feat:
            if team_feat is None:
                raise ValueError("model built with team_feat=True but the batch "
                                 "has no 'team_feat'")
            # [logit(sb), logit(sr), logit(sb)-logit(sr)]: the same
            # absolute/difference recipe the draft vectors get, so the head does
            # not have to invent the comparison for this one input.
            parts.append(torch.cat([team_feat, team_feat[:, :1] - team_feat[:, 1:]],
                                   dim=1))
        return torch.cat(parts, dim=1)               # [B, 3D..4D+3]

    def forward(self, batch):
        vb, vr = self.team_vectors(batch)
        z = self.head_input(vb, vr, batch.get("team_feat"))
        logit = self.head(z).squeeze(-1) + self.side_bias
        return {"logit": logit, "h_blue": vb, "h_red": vr}

    @torch.no_grad()
    def predict_proba(self, batch):
        return torch.sigmoid(self.forward(batch)["logit"])


# --------------------------------------------------------------- checkpoint
def config_of(model, **extra):
    """Everything needed to rebuild the model, plus how it was trained."""
    cfg = {"num_champs": model.num_champs, "embed_dim": model.embed_dim,
           "num_intra": sum(1 for _ in model.intra_team.layers),
           "num_cross": len(model.cross_team),
           "num_heads": model.intra_team.layers[0].self_attn.num_heads,
           "dropout": model.intra_team.layers[0].dropout.p,
           "n_patches": model.n_patches, "attr_dim": model.attr_dim,
           "pool": model.pool,
           "head_dim": model.head[1].out_features,
           # The two architecture flags are part of the config, not a default:
           # a checkpoint that cannot say how it was built is a checkpoint
           # `load_checkpoint` (strict) will refuse.
           "head_product": model.head_product,
           "role_bias": model.use_role_bias,
           "team_feat": model.team_feat}
    cfg.update(extra)
    return cfg


def build_from_config(cfg, attr_matrix=None):
    """Rebuild a model from a checkpoint's `config` dict.

    `attr_dim` is read from the config, not passed in: a stage that forgot the
    matrix would still *load* the checkpoint (the adapter is zero-init) and
    would then serve a model whose attribute path is frozen at zero - silently
    different from the one that was trained. Same reasoning as
    `soloq_model.attr_matrix_for_config`.

    The two architecture flags default to False, so every checkpoint written
    before they existed (the six arms of README §7) still rebuilds exactly.
    """
    if attr_matrix is None:
        attr_matrix = attr_matrix_for_config(cfg)
    return WinPredictor(num_champs=cfg["num_champs"],
                        embed_dim=cfg["embed_dim"],
                        num_intra=cfg["num_intra"],
                        num_cross=cfg["num_cross"],
                        num_heads=cfg["num_heads"],
                        dropout=cfg.get("dropout", 0.0),
                        n_patches=cfg.get("n_patches", 0),
                        attr_matrix=attr_matrix,
                        pool=cfg.get("pool", "mean"),
                        head_dim=cfg.get("head_dim"),
                        head_product=cfg.get("head_product", False),
                        role_bias=cfg.get("role_bias", False),
                        team_feat=cfg.get("team_feat", False))


def load_checkpoint(path, device="cpu"):
    """-> (model, payload). Strict: a config that does not rebuild is an error."""
    payload = torch.load(str(path), map_location=device, weights_only=False)
    model = build_from_config(payload["config"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def attr_matrix_or_none(num_champs=NUM_REAL_CHAMPS, use_attr=False):
    """The kit-attribute matrix, or None. Kept here so both trainers agree."""
    return load_champ_attr_matrix(num_champs) if use_attr else None
