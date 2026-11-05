"""Siamese draft encoder -> blue-win logit.

Adapted from `model/pretraining/soloq_model.py` (same blocks, same embedding
contract), with the masked-modelling heads replaced by one outcome head:

    slot embedding = LayerNorm(champ + role [+ attrs] [+ patch])
      -> Siamese intra-team self-attention   (ONE encoder, applied to [2B,5,D])
      -> symmetric inter-team cross-attention (ONE block, both directions)
      -> per-team pooled vector h_blue, h_red
      -> logit = MLP([h_blue, h_red, h_blue - h_red]) + b_side

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


class WinPredictor(nn.Module):
    """Full 10-pick draft -> P(blue wins) as a single logit."""

    def __init__(self, num_champs=NUM_REAL_CHAMPS, embed_dim=256,
                 num_intra=2, num_cross=1, num_heads=8, dropout=0.1,
                 n_patches=0, attr_matrix=None, pool="mean", head_dim=None):
        super().__init__()
        if pool not in ("mean", "pma"):
            raise ValueError("pool must be 'mean' or 'pma', got %r" % (pool,))
        self.num_champs = int(num_champs)
        self.embed_dim = int(embed_dim)
        self.n_patches = int(n_patches or 0)
        self.pool = pool

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
        self.cross_team = nn.ModuleList(
            [SymmetricCrossAttention(self.embed_dim, num_heads, dropout)
             for _ in range(num_cross)])
        self.pma = PMAPool(self.embed_dim, num_heads) if pool == "pma" else None

        hidden = int(head_dim or self.embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 3),
            nn.Linear(self.embed_dim * 3, hidden),
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
        if bool(both_pad.any()):
            for block in self.cross_team:
                blue, red = block(blue, red, blue_pad, red_pad)
        else:
            for block in self.cross_team:
                blue, red = block(blue, red)
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
    def forward(self, batch):
        vb, vr = self.team_vectors(batch)
        z = torch.cat([vb, vr, vb - vr], dim=1)      # [B, 3D]
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
           "head_dim": model.head[1].out_features}
    cfg.update(extra)
    return cfg


def build_from_config(cfg, attr_matrix=None):
    """Rebuild a model from a checkpoint's `config` dict.

    `attr_dim` is read from the config, not passed in: a stage that forgot the
    matrix would still *load* the checkpoint (the adapter is zero-init) and
    would then serve a model whose attribute path is frozen at zero - silently
    different from the one that was trained. Same reasoning as
    `soloq_model.attr_matrix_for_config`.
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
                        head_dim=cfg.get("head_dim"))


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
