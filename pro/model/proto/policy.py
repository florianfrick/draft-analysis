"""The prototype policy: PolicyValueModel plus the four ideas under test.

Kept in its own module so the trainer and the evaluator rebuild the *same*
class from a checkpoint. A checkpoint whose config says `slot_mask: true` has a
wider first layer than the production head, so `PolicyValueModel` cannot load
it - and silently loading it with strict=False is exactly the failure mode that
produces a plausible-looking wrong number.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from solo_queue.model import PolicyValueModel
from solo_queue.model.policy.model import CHAMP_PAD_ID, ROLE_FLEX_ID

NUM_CHAMPS = 173


class ProtoPolicy(PolicyValueModel):
    """Four knobs, one forward.

    `slot_mask` - `_pool` averages the non-PAD slot vectors per team, so the
    head sees how strong each team's picks are but never *which roles are
    filled*. A blue team with top+jungle and one with adc+support pool to
    similar statistics at the same move index, and the correct next pick is
    completely different; move_idx gives the count, not the set. Twenty binary
    features for ~2k parameters is the cheapest hypothesis here.

    `block_head` - predicts the partner of a double-pick block at the first
    move of the pair. Only useful with an unfrozen encoder: a frozen trunk
    cannot be shaped by an auxiliary loss, so running this arm frozen would
    "show" the idea fails while nothing was learning.

    `prior_bias` - a learned multiple of the train-only empirical
    log P(champion | patch, move index), i.e. the lookup table the model has
    always lost ban1 to, handed over as a feature. Weight starts at zero, so
    the arm begins at the baseline. On a random split the test patches are in
    train, so this prior is at its most useful here and its least useful on a
    genuinely new patch - the number must travel with that caveat.
    """

    def __init__(self, encoder, block_head=False, slot_mask=False,
                 prior_bias=False, **kw):
        super().__init__(encoder, **kw)
        d, s = kw.get("embed_dim", 256), kw.get("state_dim", 64)
        self.slot_mask = slot_mask
        self.prior_bias = prior_bias
        base = d * 4 + s
        if slot_mask:
            self.policy_head = nn.Sequential(
                nn.LayerNorm(base + 20), nn.Linear(base + 20, d), nn.GELU(),
                nn.Dropout(kw.get("dropout", 0.0)), nn.Linear(d, NUM_CHAMPS))
        width = base + (20 if slot_mask else 0)
        self.block_head = (nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, d), nn.GELU(),
            nn.Linear(d, NUM_CHAMPS)) if block_head else None)
        if prior_bias:
            self.register_buffer("log_prior", torch.zeros(1, 1, NUM_CHAMPS))
            self.prior_w = nn.Parameter(torch.zeros(1))

    def set_prior(self, log_prior):
        self.log_prior = log_prior.to(self.prior_w.device)

    def forward(self, batch):
        h = self.encoder._encode(batch)
        ban_ids = batch["ban_ids"]
        ban_pad = ban_ids == CHAMP_PAD_ID
        ban_embeds = self.encoder.embed.champ_lookup(ban_ids)
        ban_role = self.encoder.embed.role_lookup(
            torch.full_like(ban_ids, ROLE_FLEX_ID))
        ban_embeds = self.encoder.embed.layer_norm(ban_embeds + ban_role)
        h = self.ban_context(h, ban_embeds, ban_pad)
        state = self.state_embed(batch["move_idx"], batch["phase"],
                                 batch["to_move"], batch["first_pick"],
                                 batch.get("patch_id"))
        pooled = self._pool(h, batch["pad_mask"])
        x = torch.cat([pooled, state], dim=-1)
        if self.slot_mask:
            x = torch.cat([x, (~batch["pad_mask"]).float(),
                           (~ban_pad).float()], dim=-1)
        logits = self.policy_head(x)
        if self.prior_bias:
            logits = logits + self.prior_w * self.log_prior[
                batch["patch_id"], batch["move_idx"]]
        out = {"policy_input": x,
               "policy_logits": self._apply_legality(logits, batch)}
        if self.block_head is not None:
            blog = self.block_head(x)
            illegal = batch["taken_mask"].clone()
            illegal[torch.arange(illegal.shape[0]),
                    batch["target_champ"]] = True
            out["block_logits"] = blog.masked_fill(illegal, -1e4)
        return out


def build_prior(rows, indices, patch_index, n_patches, n_moves,
                patch_to_id, alpha=0.5):
    """log P(champion | patch, move index) on TRAIN rows only -> [P,20,C].

    Row 0 (unknown patch) gets the all-patch prior, which is what a genuinely
    new patch falls back to at serving time.
    """
    cnt = torch.zeros(n_patches, n_moves, NUM_CHAMPS, dtype=torch.float64)
    for i in indices:
        row = rows[i]
        pid = patch_to_id(row.get("patch"), patch_index)
        for m, mv in enumerate(row["moves"]):
            cnt[pid, m, int(mv[2])] += 1
    logp = torch.log((cnt + alpha) /
                     (cnt.sum(-1, keepdim=True) + alpha * NUM_CHAMPS))
    glob = cnt.sum(0)
    logp[0] = torch.log((glob + alpha) /
                        (glob.sum(-1, keepdim=True) + alpha * NUM_CHAMPS))
    return logp.float()
