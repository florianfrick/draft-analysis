"""Pro finetune model: pretrained draft encoder + outcome heads.

Loads a SoloQ-pretrained `SoloQPretrainedModel` (default: the current best
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
        from soloq_model import TEAM_SIZE
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
        z = self.pool(batch)
        win_logits, econ = self.heads(z)
        enc["win_logits"] = win_logits
        enc["econ_pred"] = econ
        return enc
