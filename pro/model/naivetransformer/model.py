"""A deliberately naive, single-stack transformer for next-move prediction.

The design question is *not* accuracy per se: it is whether one uniform
transformer over a flat chronological record of the prefix can recover the
structure that the repo's two-tower encoder + separate policy head encodes by
hand. Everything the model sees is prefix-only, so there is no masked-token
pretraining stage and no causal attention mask to maintain: the sequence itself
is already causal by construction.

Sequence (61 tokens, always this shape)::

    [STATE] | 10 pick slots | 10 ban slots | 40 fearless slots

* pick token  = champ emb + role emb + "pick" type emb + slot pos emb
* ban token   = champ emb + "ban" type emb + ban-index pos emb
* fearless    = champ emb + role emb + "fearless" type emb + pool pos emb
* [STATE]     = sum of move / phase / side / first-pick / game-num / patch
                embeddings, LayerNorm'd

`src_key_padding_mask` removes unpicked slots, absent bans and pool padding.
The [STATE] token is never masked, so an all-padding row cannot produce NaN.

The [STATE] output drives a Linear(D -> 173) for the next move. The *same*
head is also applied to the ten pick-slot positions: for an empty slot this is
a role-conditioned pick suggestion (blue TOP .. red SUP) that works even when
the actual next move is a ban. Legality is applied to the logits *after* the
head: champions already picked/banned are always illegal, fearless-picked
champions are illegal on pick moves only (they may still be banned). We use
-1e4 rather than -inf so fp16 autocast keeps the row finite.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CHAMPS = 173
CHAMP_PAD_ID = 174
CHAMP_VOCAB = 175
# 0..4 = the five roles, 5 = FLEX (also the corpus's "unknown role" id),
# 6 = NO_ROLE, used for bans, which have no role in the corpus.
ROLE_FLEX_ID = 5
NO_ROLE_ID = 6
ROLE_VOCAB = 7

N_PICKS = 10
N_BANS = 10
FEARLESS_SLOTS = 40
SEQ_LEN = 1 + N_PICKS + N_BANS + FEARLESS_SLOTS  # 61

STATE_POS = 0
PICK_START = 1
BAN_START = PICK_START + N_PICKS            # 11
FEARLESS_START = BAN_START + N_BANS          # 21

TYPE_PICK, TYPE_BAN, TYPE_FEARLESS = 0, 1, 2
N_TYPES = 3

N_MOVES = 20
N_PHASES = 4
MAX_GAME_NUM = 6  # 0 = unknown, 1..5 = game within series

ILLEGAL_LOGIT = -1e4


class NaiveTransformer(nn.Module):
    """Flat 61-token encoder + one policy head on the [STATE] position."""

    def __init__(self, embed_dim=768, num_layers=12, num_heads=12,
                 ffn_dim=3072, dropout=0.1, n_patches=108,
                 num_champs=NUM_CHAMPS, use_game_num=True):
        super().__init__()
        d = int(embed_dim)
        self.embed_dim = d
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.n_patches = int(n_patches)
        self.use_game_num = bool(use_game_num)
        self.num_champs = int(num_champs)

        self.champ_emb = nn.Embedding(CHAMP_VOCAB, d, padding_idx=CHAMP_PAD_ID)
        self.role_emb = nn.Embedding(ROLE_VOCAB, d)
        self.type_emb = nn.Embedding(N_TYPES, d)
        self.pick_pos = nn.Embedding(N_PICKS, d)
        self.ban_pos = nn.Embedding(N_BANS, d)
        self.fearless_pos = nn.Embedding(FEARLESS_SLOTS, d)

        dims = [N_MOVES, N_PHASES, 2, 2, MAX_GAME_NUM]
        self.state_names = ("move_idx", "phase", "to_move", "first_pick",
                            "game_num")
        self.state_embs = nn.ModuleList(
            [nn.Embedding(n, d) for n in dims] + [nn.Embedding(max(1, n_patches), d)])
        self.state_norm = nn.LayerNorm(d)
        self.tok_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.num_heads, dim_feedforward=self.ffn_dim,
            dropout=float(dropout), activation=F.gelu, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.num_layers, enable_nested_tensor=False)
        self.head_norm = nn.LayerNorm(d)
        self.policy_head = nn.Linear(d, self.num_champs)
        nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.policy_head.bias)

        self.register_buffer("pick_positions", torch.arange(N_PICKS),
                             persistent=False)
        self.register_buffer("ban_positions", torch.arange(N_BANS),
                             persistent=False)
        self.register_buffer("fearless_positions",
                             torch.arange(FEARLESS_SLOTS), persistent=False)
        self.register_buffer("pick_type", torch.tensor(TYPE_PICK),
                             persistent=False)
        self.register_buffer("ban_type", torch.tensor(TYPE_BAN),
                             persistent=False)
        self.register_buffer("fearless_type", torch.tensor(TYPE_FEARLESS),
                             persistent=False)

    def encode(self, batch):
        """Prefix batch -> hidden states [B,61,D]. Shared by both read-outs."""
        champ_ids = batch["champ_ids"].long()
        ban_ids = batch["ban_ids"].long()
        fearless_ids = batch["fearless_ids"].long()
        bsz = champ_ids.size(0)
        device = champ_ids.device

        # ---- [STATE] token
        state = None
        for i, name in enumerate(self.state_names):
            if name == "game_num" and not self.use_game_num:
                continue
            # Older checkpoints/boards may lack game_num; fall back to a
            # harmless in-range id rather than raising at serve time.
            key = name if name in batch else ("move_idx" if name == "game_num"
                                              else name)
            val = batch[key].long().to(device)
            if name == "game_num":
                val = val.clamp(0, MAX_GAME_NUM - 1)
            else:
                val = val.clamp(0, self.state_embs[i].num_embeddings - 1)
            part = self.state_embs[i](val)
            state = part if state is None else state + part
        patch_ids = batch.get("patch_id")
        if patch_ids is None:
            patch_ids = torch.zeros(bsz, dtype=torch.long, device=device)
        patch_ids = patch_ids.long().clamp(0, self.state_embs[-1].num_embeddings - 1)
        state = state + self.state_embs[-1](patch_ids)
        state = self.state_norm(state).unsqueeze(1)          # [B,1,D]

        # ---- picks: champ + role + type + slot position
        role_ids = batch["role_ids"].long().clamp(0, ROLE_FLEX_ID)
        picks = (self.champ_emb(champ_ids)
                 + self.role_emb(role_ids)
                 + self.type_emb(self.pick_type)
                 + self.pick_pos(self.pick_positions))
        # ---- bans: no role token (NO_ROLE), champ + type + chronological index
        bans = (self.champ_emb(ban_ids)
                + self.type_emb(self.ban_type)
                + self.ban_pos(self.ban_positions))
        # ---- fearless pool: champ + prior role + type + pool position
        fr = batch["fearless_role_ids"].long().clamp(0, ROLE_FLEX_ID)
        fearless = (self.champ_emb(fearless_ids)
                    + self.role_emb(fr)
                    + self.type_emb(self.fearless_type)
                    + self.fearless_pos(self.fearless_positions))

        seq = torch.cat([state, picks, bans, fearless], dim=1)  # [B,61,D]
        key_padding_mask = torch.zeros(bsz, SEQ_LEN, dtype=torch.bool,
                                       device=device)
        key_padding_mask[:, PICK_START:BAN_START] = batch["pad_mask"].bool()
        key_padding_mask[:, BAN_START:FEARLESS_START] = (ban_ids == CHAMP_PAD_ID)
        key_padding_mask[:, FEARLESS_START:] = (fearless_ids == CHAMP_PAD_ID)
        # STATE is always attendable, so no row can be fully masked -> no NaN.
        return self.encoder(self.tok_norm(seq),
                            src_key_padding_mask=key_padding_mask)

    def _apply_legality(self, logits, batch, picks_only=False):
        """Mask illegal champions. Works on [B,173] and [B,10,173]."""
        taken = batch["taken_mask"].bool()
        unavail = batch["unavailable_mask"].bool()
        if logits.ndim == 3:  # [B,10,173] vs the batch's [B,173]
            taken, unavail = taken.unsqueeze(1), unavail.unsqueeze(1)
        logits = logits.masked_fill(taken, ILLEGAL_LOGIT)
        if picks_only:
            # A role-slot query is always a pick, even when the actual next
            # move is a ban. The fearless pool is illegal here unconditionally.
            logits = logits.masked_fill(unavail, ILLEGAL_LOGIT)
        elif "is_ban" in batch:
            is_pick = batch["is_ban"].reshape(-1) == 0
            if bool(is_pick.any()):
                pool_illegal = (batch["unavailable_mask"].bool()
                                & is_pick[:, None])
                logits = logits.masked_fill(pool_illegal, ILLEGAL_LOGIT)
        return logits

    def forward(self, batch):
        hidden = self.encode(batch)
        next_logits = self.policy_head(self.head_norm(hidden[:, STATE_POS, :]))
        # Unpicked slots are key-masked but still attend as queries, so these
        # positions carry a real "who belongs here" representation. The head
        # is SHARED with next-move: no new parameters, no checkpoint change.
        slot_logits = self.policy_head(
            self.head_norm(hidden[:, PICK_START:PICK_START + N_PICKS, :]))
        return {
            "policy_logits": self._apply_legality(next_logits, batch),
            "slot_logits": self._apply_legality(slot_logits, batch,
                                                picks_only=True),
        }

    @torch.no_grad()
    def role_suggestions(self, batch, team, role, k=5):
        """Top-k for `team` (0 blue / 1 red) at `role` (0..4), any draft state.

        Returns None when that slot is already filled. `batch` is the ordinary
        prefix batch (build_prefix + pro_collate), so this works on ban states:
        the read-out ignores `is_ban` entirely.
        """
        if int(team) not in (0, 1) or not 0 <= int(role) < 5:
            raise ValueError("team must be 0/1 and role 0..4")
        slot = int(team) * 5 + int(role)
        bsz = batch["champ_ids"].size(0)
        if bsz != 1:
            raise ValueError("role_suggestions expects a single board")
        if not bool(batch["pad_mask"][0, slot]):
            return None
        hidden = self.encode(batch)
        logits = self.policy_head(
            self.head_norm(hidden[:, PICK_START + slot, :]))
        logits = self._apply_legality(logits, batch, picks_only=True)
        vals, idx = torch.topk(logits, min(int(k), self.num_champs))
        return [{"champ": int(c), "logit": float(v)}
                for c, v in zip(idx[0].tolist(), vals[0].tolist())]


def model_kwargs_from_args(args, n_patches):
    """The exact dict stored in every checkpoint and replayed by every loader."""
    return {
        "embed_dim": int(args.embed_dim),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "ffn_dim": int(args.ffn_dim),
        "dropout": float(args.dropout),
        "n_patches": int(n_patches),
        "num_champs": 173,
        "use_game_num": bool(args.use_game_num),
    }


def build_from_config(cfg):
    """Rebuild the architecture from a checkpoint config (no weights loaded)."""
    kw = dict(cfg["model"])
    return NaiveTransformer(**kw)


def load_from_checkpoint(path, device="cpu"):
    """(model, patch_index, cfg) for the eval suite and for inference."""
    blob = torch.load(str(path), map_location=device, weights_only=False)
    cfg = blob["config"]
    model = build_from_config(cfg)
    model.load_state_dict(blob["model"], strict=True)
    model.to(device).eval()
    vocab = blob.get("patch_vocab") or []
    patch_index = {str(k): i + 1 for i, k in enumerate(vocab)}
    return model, patch_index, cfg


def param_count(model):
    return sum(p.numel() for p in model.parameters())
