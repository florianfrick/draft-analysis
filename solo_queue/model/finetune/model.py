"""Pro finetune model: pretrained draft encoder + outcome heads.

Loads a SoloQ-pretrained `SiameseBase` (default: the current best
run_*/best.pt, i.e. vague-totem-4 @ ep 48 unless a better checkpoint
lands) and adds one pooled outcome head computing per-side win logits
plus auxiliary economy regressions. Stage flag freezes everything except
the heads; unfreezing reaches cross_team first, then the shared intra
encoder, never the embedding tables in this stage.

Head input follows the set-pooling recipe from ../README.md:
z = concat(v_blue, v_red, v_blue - v_red, v_blue * v_red), where
v_blue/v_red are key_padding-masked means of the final slot vectors
per team. PAD slots never contribute to the pooled vector.
"""
import torch
import torch.nn as nn

from solo_queue.model.pretraining.base_model import CHAMP_PAD_ID, ROLE_FLEX_ID, PMAPool


class PoolOutcomeHeads(nn.Module):
    """Per-side win logits + economy regressions on the pooled vector."""

    def __init__(self, embed_dim=128, n_econ=3, dropout=0.0):
        super().__init__()
        self.win = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, 2),  # [logit blue wins, logit red wins]
        )
        self.econ = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_econ),  # standardized econ targets
        )

    def forward(self, z):
        return self.win(z), self.econ(z)


class ProDraftModel(nn.Module):
    """Pretrained encoder + fresh outcome heads; staged unfreezing."""

    def __init__(self, encoder, n_econ=3, head_dropout=0.0):
        super().__init__()
        self.encoder = encoder
        d = encoder.embed.champ_lookup.embedding_dim
        self.heads = PoolOutcomeHeads(d, n_econ=n_econ,
                                      dropout=head_dropout)
        self.set_stage(0)

    def set_stage(self, stage):
        """0 = outcome heads + encoder readouts only (attention trunk
        frozen); 1 = + cross_team; 2 = + intra_team. Embedding lookups
        stay frozen in this whole stage (they carry soloQ vocab semantics).
        The encoder's champ_head/role_head remain trainable throughout -
        the pro masked CE drives them directly."""
        freeze = [self.encoder.embed]
        partial = [self.encoder.cross_team]
        if stage <= 0:
            for mod in partial + freeze:
                for p in mod.parameters():
                    p.requires_grad = False
            for p in self.encoder.intra_team.parameters():
                p.requires_grad = False
        elif stage == 1:
            for mod in partial:
                for p in mod.parameters():
                    p.requires_grad = True
            for p in self.encoder.intra_team.parameters():
                p.requires_grad = False
            for mod in freeze:
                for p in mod.parameters():
                    p.requires_grad = False
        else:
            for mod in partial + [self.encoder.intra_team]:
                for p in mod.parameters():
                    p.requires_grad = True
            for mod in freeze:
                for p in mod.parameters():
                    p.requires_grad = False
        for p in self.heads.parameters():
            p.requires_grad = True
        self.stage = stage
        return self

    def pool(self, batch):
        """Set-pooled vector z [B, 4D] (PAD-masked means per team)."""
        from solo_queue.model.pretraining.base_model import TEAM_SIZE
        h = self.encoder._encode(batch)  # [B,10,D], no positions
        pad = batch.get("pad_mask")
        if pad is None:
            pad = batch["champ_ids"] == self.encoder.embed \
                .champ_lookup.num_embeddings - 1
        keep = (~pad).float().unsqueeze(-1)  # [B,10,1]
        blue, red = h[:, :TEAM_SIZE], h[:, TEAM_SIZE:]
        kb, kr = keep[:, :TEAM_SIZE], keep[:, TEAM_SIZE:]
        vb = (blue * kb).sum(1) / kb.sum(1).clamp_min(1.0)
        vr = (red * kr).sum(1) / kr.sum(1).clamp_min(1.0)
        return torch.cat([vb, vr, vb - vr, vb * vr], dim=1)

    def forward(self, batch):
        # keep the pretrained heads live for the rehearsal loss when the
        # batch carries masking (pro masked batches + soloQ rehearsal do;
        # full drafts for outcome-only eval do not - skip dead heads then)
        has_masks = "is_masked" in batch
        enc = self.encoder(batch, compute_heads=has_masks)
        z = self.pool(batch)
        win_logits, econ = self.heads(z)
        enc["win_logits"] = win_logits
        enc["econ_pred"] = econ
        return enc


class PrefixValueModel(nn.Module):
    """Draft-state value on the pretrained PMA path, conditioned on bans.

        logit(state) = win_head(z4)  +  ban_block([z4, ban_vec])

    `z4` is the encoder's own `value_vector` - PMA-pooled per team, then
    concat(vb, vr, vb-vr, vb*vr) - i.e. the readout that scored 0.5597 AUC
    in the pretraining probe, not the 4D PAD-masked mean that the policy
    value head used. The trunk is frozen; only `pma`, `win_head`,
    `ban_pool` and `ban_block` train.

    Why a ban path at all: the encoder cannot see bans (`_encode` reads
    champ_ids/role_ids/pad_mask only), so a head on `z4` alone is blind to
    the ban list. That is fatal for the thing this value function exists
    for - a ban is a counterfactual "remove c from the pool", and V has to
    move when the pool moves. `ban_vec` is a PMA pool over the 10 ban
    slots, permutation-invariant and zero when no ban has happened yet.

    Why a *residual* block rather than a wider first layer: `ban_block`'s
    last layer is zero-initialised, so at init the logit is exactly the
    pretrained `win_head` output. That keeps the pretrained value weights
    usable verbatim and makes "how much do bans buy us" a single ablation
    (score `win_logit_base` vs `win_logit`). It is not additive-only: the
    block also reads `z4`, so it can learn ban x pick interactions; it
    just starts at zero.

    No patch input, deliberately. The eval protocol is patch-honest (test
    patches never appear in train), so a patch embedding could only
    memorise - and on an unseen patch it would collapse to UNK anyway.
    """

    def __init__(self, encoder, ban_heads=4, ban_seeds=4, ban_dropout=0.0,
                 use_bans=True):
        super().__init__()
        if not getattr(encoder, "with_value", False):
            raise ValueError("encoder must be built with with_value=True "
                             "(it needs pma + win_head)")
        self.encoder = encoder
        self.use_bans = use_bans
        d = encoder.embed.champ_lookup.embedding_dim
        self.embed_dim = d
        if use_bans:
            self.ban_pool = PMAPool(d, ban_heads, ban_seeds, ban_dropout)
            self.ban_block = nn.Sequential(
                nn.LayerNorm(d * 5),
                nn.Linear(d * 5, d),
                nn.GELU(),
                nn.Dropout(ban_dropout),
                nn.Linear(d, 1),
            )
            # Exact-equivalence-at-init: no ban signal until it is learned.
            nn.init.zeros_(self.ban_block[-1].weight)
            nn.init.zeros_(self.ban_block[-1].bias)
        else:
            self.ban_pool = None
            self.ban_block = None

    def freeze_trunk(self):
        """Freeze everything except the value path. Returns trainable names."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for mod in (self.encoder.pma, self.encoder.win_head,
                    self.ban_pool, self.ban_block):
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = True
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def ban_vector(self, batch):
        """[B, D] PMA pool over the ban set; zero vector when empty.

        Same embedding the encoder uses for picks (champ + FLEX role, then
        the shared LayerNorm), so a banned champion sits in the same space
        a picked one would. Bans carry no role and no team, and the pool
        has no positional signal, so the vector is invariant to ban order.
        """
        ban_ids = batch["ban_ids"]                       # [B,10]
        pad = ban_ids == CHAMP_PAD_ID                    # [B,10] True=PAD
        emb = self.encoder.embed.champ_lookup(ban_ids)
        role = self.encoder.embed.role_lookup(
            torch.full_like(ban_ids, ROLE_FLEX_ID))
        emb = self.encoder.embed.layer_norm(emb + role)
        return self.ban_pool(emb, pad)

    def forward(self, batch, with_bans=True):
        h = self.encoder._encode(batch)                  # [B,10,D]
        z4 = self.encoder.value_vector(batch, h)         # [B,4D]
        base = self.encoder.win_head(z4)                 # [B,1]
        out = {"win_logit_base": base, "z4": z4}
        if self.use_bans and with_bans:
            bv = self.ban_vector(batch)
            corr = self.ban_block(torch.cat([z4, bv], dim=1))
            out["ban_vector"] = bv
            out["win_corr"] = corr
            out["win_logit"] = base + corr
        else:
            out["win_logit"] = base
        return out

    @torch.no_grad()
    def forward_value(self, batch):
        """Blue-win logit [B,1], inference only."""
        return self.forward(batch)["win_logit"]

