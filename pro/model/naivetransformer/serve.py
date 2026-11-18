"""Serve a naive-transformer checkpoint through the next-move predictor API.

The offline suit scores these checkpoints with `data/pro play/datasets.py::
build_prefix`, which is *not* the board the frontend passes around: the corpus
row carries exact roles and a series game number, while a live board has only
slots, bans and a fearless pool. This adapter is the one place that translates
the live board into the 61-token prefix batch, so the served numbers describe
the model that was measured rather than a second, quietly re-derived model.

Duck-types `solo_queue.model.policy.infer.PolicyPredictor`: the engine, the
agent tools and `/api/next-move` ask only for `next_move`, the patch helpers
and the metadata attributes. Everything except `next_move` is inherited.
"""
from __future__ import annotations

import torch

from pro.model.naivetransformer import model as NT
from solo_queue.model.policy import infer

# The frontend's board -> role mapping. Roles are per side, so slot 0 is blue
# TOP and slot 5 is red TOP - the same convention build_prefix uses for picks.
FLEX = NT.ROLE_FLEX_ID


class NaiveTransformerPolicy(infer.PolicyPredictor):
    """A `PolicyPredictor` whose batch is the transformer's prefix batch."""

    kind = "policy"   # the engine files this in the next-move slot

    def __init__(self, ckpt_path, device="cpu", payload=None, label=None):
        blob = payload if payload is not None else torch.load(
            str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = blob["config"]
        if str(cfg.get("arch", "")) != "naive_transformer":
            raise ValueError("%s is not a naive_transformer checkpoint"
                             % ckpt_path)
        model, patch_index, cfg = NT.load_from_checkpoint(
            ckpt_path, device=device) if payload is None else self._load_blob(
                blob, cfg, device)
        # PolicyPredictor's inherited helpers read these, not a torch module.
        self.device = torch.device(device)
        self.models = [model]
        self.cfgs = [cfg]
        self.cfg = cfg
        self.num_champs = int(model.num_champs)
        self.n_patches = int(cfg.get("n_patches") or 0)
        self.patch_vocab = list(blob.get("patch_vocab") or [])
        self.patch_index = patch_index
        self.ckpt_path = str(label or ckpt_path)
        self.members = [self.ckpt_path]
        self.metric = "next-move top5"
        metrics = blob.get("metrics") or {}
        self.val_top5 = float(metrics["top5"]) if "top5" in metrics else None
        self.epoch = metrics.get("epoch")
        sh = cfg.get("slot_head")
        self.slot_head = sh if isinstance(sh, str) else bool((sh or {}).get("enabled"))

    @staticmethod
    def _load_blob(blob, cfg, device):
        model = NT.build_from_config(cfg)
        model.load_state_dict(blob["model"], strict=True)
        model.to(device).eval()
        vocab = blob.get("patch_vocab") or []
        return model, {str(k): i + 1 for i, k in enumerate(vocab)}, cfg

    @property
    def model(self):
        return self.models[0]

    def _batch(self, board, move_idx, kind, team, patch_id, game_num=0):
        """Live board -> the exact keys NaiveTransformer.encode reads."""
        champ_ids = board["champ_ids"]
        dev = champ_ids.device
        role_ids = torch.full_like(board["role_ids"], FLEX)
        for slot in range(NT.N_PICKS):
            if not bool(board["pad_mask"][0, slot]):
                role_ids[0, slot] = slot % 5       # exact role = slot in team

        pool = [i for i, flag in enumerate(board["unavailable"][0].tolist())
                if flag]
        fearless_ids = torch.full((1, NT.FEARLESS_SLOTS), NT.CHAMP_PAD_ID,
                                  dtype=torch.long, device=dev)
        fearless_roles = torch.full((1, NT.FEARLESS_SLOTS), FLEX,
                                    dtype=torch.long, device=dev)
        for i, champ in enumerate(pool[:NT.FEARLESS_SLOTS]):
            fearless_ids[0, i] = champ
            # The live board has no prior roles for pool champions; the corpus
            # supplies them offline, so FLEX is the honest serving value.

        def full(value):
            return torch.full((1,), int(value), dtype=torch.long, device=dev)

        return {
            "champ_ids": champ_ids,
            "role_ids": role_ids,
            "pad_mask": board["pad_mask"],
            "ban_ids": board["ban_ids"],
            "fearless_ids": fearless_ids,
            "fearless_role_ids": fearless_roles,
            # Training's legality rule: taken (picks + bans) always; the pool
            # only on picks. A fearless champion may still be banned.
            "taken_mask": board["taken"],
            "unavailable_mask": board["unavailable"],
            "move_idx": full(move_idx),
            "phase": full(infer.PHASE_OF[int(move_idx)]),
            "to_move": full(team),
            "first_pick": full(board["first_pick"]),
            "game_num": full(max(0, min(NT.MAX_GAME_NUM - 1, int(game_num)))),
            "is_ban": full(1 if kind == infer.MOVE_BAN else 0),
            "patch_id": full(patch_id),
        }

    @torch.no_grad()
    def next_move(self, picks, bans, first_pick=0, k=8, unavailable=None,
                  patch_id=infer.PATCH_UNK_ID, allowed=None, team=None,
                  game_num=0):
        """Top-k for the move on the clock; same return shape as the old head."""
        pid = self.patch_id_for(patch_id)
        board = infer.build_board(picks, bans, self.num_champs, first_pick,
                                  self.device, unavailable=unavailable)
        n_unavail = int(board["unavailable"][0].sum())
        move_idx, left_b, left_p = infer.resolve_position(
            int(board["n_bans"][0]), int(board["n_picks"][0].sum()),
            first_pick)
        warnings = []
        if left_b:
            warnings.append("%d ban(s) do not fit the draft order" % left_b)
        if left_p:
            warnings.append("%d pick(s) do not fit the draft order" % left_p)
        if move_idx >= infer.N_MOVES:
            return {"complete": True, "warning": "; ".join(warnings) or None,
                    "first_pick": int(first_pick), "roles": None,
                    "suggestions": [], "move_idx": infer.N_MOVES,
                    "kind": None, "side": None, "phase": None,
                    "patch_id": pid, "n_unavailable": n_unavail}
        kind, planned = infer.move_plan(first_pick)[move_idx]
        team = planned if team is None else int(team)
        if team not in (0, 1):
            raise ValueError("team must be 0 (blue) or 1 (red), got %r" % team)
        batch = self._batch(board, move_idx, kind, team, pid, game_num)
        logits = self.policy_logits(batch)[0]
        if allowed is not None:
            idx = torch.as_tensor(list(allowed), dtype=torch.long,
                                  device=logits.device)
            if idx.numel() == 0:
                raise ValueError("allowed is empty: nothing can answer this "
                                 "move")
            keep = torch.zeros(logits.shape, dtype=torch.bool,
                               device=logits.device)
            keep[idx] = True
            logits = logits.masked_fill(~keep, float("-inf"))
        p = torch.softmax(logits.float(), dim=-1)
        top = p.topk(min(k, self.num_champs))
        suggestions = [{"idx": int(i), "prob": float(v)}
                       for i, v in zip(top.indices, top.values)
                       if float(v) > 0 and int(i) < self.num_champs]
        return {
            "complete": False,
            "move_idx": move_idx,
            "kind": "ban" if kind == infer.MOVE_BAN else "pick",
            "side": int(team),
            "phase": infer.PHASE_NAMES[infer.PHASE_OF[move_idx]],
            "first_pick": int(first_pick),
            "warning": "; ".join(warnings) or None,
            "suggestions": suggestions,
            # No role head in this architecture: role placement comes from
            # data/champ_roles.json, not from a model query.
            "roles": None,
            "patch_id": pid,
            "n_unavailable": n_unavail,
        }
