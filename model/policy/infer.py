"""Inference API for the next-move policy, plus optional value scoring.

Board in, recommendation out. No CLI, no dataset import, no training code:
`frontend/server.py` imports this, and it is the only path that serves the
policy. The offline statistics (bootstrap CIs, paired tests, regret) lived in
`ban_propose.py`, which was removed on 2026-11-03 along with the rest of the
ban-analysis thread; STATUS.md §5 keeps its conclusions.

Three pieces:

    move_plan(first_pick)     the canonical 20-move order
    resolve_position(...)     how far along a hand-built board is
    PolicyPredictor           next-move distribution + ban ranking

Board convention (the UI's, not the dataset's)
----------------------------------------------
`picks` is a 10-slot list - blue 0-4, red 5-9, in PICK ORDER, None = empty.
`bans` is the list of champions banned so far, any order. The slot index
within a team carries no meaning to the model: the encoder has no
positional embedding and both the policy and the value pool each team with
a permutation-invariant reduction, so only the (champion, role) pair is
reconstructed. Roles are not part of the UI board, so they are *inferred*
once per state from the encoder's role head (the same FLEX-role query the
masked frontend uses) and then fed back in.

Why this can differ from the offline proposer
---------------------------------------------
The offline proposer (deleted with the ban thread) simulated future picks
with the roles recorded in the pro row, i.e. it was told which role each
future pick would fill. A UI board has no such label, so the rollout here
infers the role of each simulated pick from the champion. That is a strictly
harder task, and it is why the
numbers this produces are a UI estimate, not a reproduction of the offline
result.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
for _p in (str(HERE), str(MODEL_DIR)):        # `model` -> policy/model.py
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import PolicyValueModel                             # noqa: E402
from model.pretraining.soloq_model import (SoloQPretrainedModel, CHAMP_PAD_ID,   # noqa: E402
                         ROLE_FLEX_ID, NUM_ROLES, load_encoder_state,
                         attr_matrix_for_config)

N_MOVES = 20
TEAM_SIZE = 5
N_SLOTS = 10
BAN_SLOTS = 10
MOVE_BAN, MOVE_PICK = 0, 1
# 6 bans, 6 picks, 4 bans, 4 picks
PHASE_OF = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4
PHASE_NAMES = {0: "bans1", 1: "picks1", 2: "bans2", 3: "picks2"}
PATCH_UNK_ID = 0            # index 0 of the patch vocab is reserved for UNK


def move_plan(first_pick):
    """The canonical 20-move order -> [(kind, team)] (see pipeline.py).

    Bans phase 1 start with the first-pick team, picks phase 1 also start
    with it, and phase 2 is the mirror image. first_pick=0 recovers the
    classic blue-first order.
    """
    fp = 1 if int(first_pick) else 0
    sp = 1 - fp
    seq = []
    seq += [(MOVE_BAN, t) for t in (fp, sp, fp, sp, fp, sp)]
    seq += [(MOVE_PICK, t) for t in (fp, sp, sp, fp, fp, sp)]
    seq += [(MOVE_BAN, t) for t in (sp, fp, sp, fp)]
    seq += [(MOVE_PICK, t) for t in (sp, fp, fp, sp)]
    assert len(seq) == N_MOVES
    return seq


def phase_of(move_idx):
    return PHASE_OF[int(move_idx)]


def resolve_position(n_bans, n_picks, first_pick):
    """How many moves a board has consumed -> (move_idx, leftover_bans,
    leftover_picks).

    A board is a legal prefix if its bans and picks can be laid onto the
    move order from the front. Anything that does not fit is returned as
    leftover rather than silently reordered: the caller shows a warning and
    the model is asked about the position the board actually supports.
    """
    plan = move_plan(first_pick)
    left = {MOVE_BAN: int(n_bans), MOVE_PICK: int(n_picks)}
    k = 0
    for kind, _team in plan:
        if left[kind] <= 0:
            break
        left[kind] -= 1
        k += 1
    return k, left[MOVE_BAN], left[MOVE_PICK]


# ------------------------------------------------------------------- boards
def empty_board(num_champs, first_pick=0, device="cpu"):
    """Batch-1 board: nothing banned, nothing picked.

    PAD is `num_champs + 1` *per model*, not the module-level constant:
    a checkpoint with a smaller vocabulary sizes its embedding table to
    match, so the global 174 would index past it.
    """
    pad = int(num_champs) + 1
    return {
        "champ_ids": torch.full((1, N_SLOTS), pad, dtype=torch.long,
                                device=device),
        "role_ids": torch.full((1, N_SLOTS), ROLE_FLEX_ID, dtype=torch.long,
                               device=device),
        "pad_mask": torch.ones(1, N_SLOTS, dtype=torch.bool, device=device),
        "ban_ids": torch.full((1, BAN_SLOTS), pad, dtype=torch.long,
                              device=device),
        "taken": torch.zeros(1, num_champs, dtype=torch.bool, device=device),
        "n_bans": torch.zeros(1, dtype=torch.long, device=device),
        "n_picks": torch.zeros(1, 2, dtype=torch.long, device=device),
        "first_pick": int(first_pick),
        "num_champs": int(num_champs),
    }


def build_board(picks, bans, num_champs, first_pick=0, device="cpu"):
    """UI board -> batch-1 tensors. Raises ValueError on an illegal board.

    picks: 10 entries, champion index or None (blue 0-4 / red 5-9, pick
    order). bans: champion indices banned so far, any order.
    """
    if len(picks) != N_SLOTS:
        raise ValueError("picks must have 10 entries")
    board = empty_board(num_champs, first_pick, device)
    seen = {}
    for slot, champ in enumerate(picks):
        if champ is None:
            continue
        champ = int(champ)
        if not 0 <= champ < num_champs:
            raise ValueError("champion index %d out of range" % champ)
        if champ in seen:
            raise ValueError("champion %d appears twice in the draft" % champ)
        seen[champ] = ("pick", slot)
        board["champ_ids"][0, slot] = champ
        board["pad_mask"][0, slot] = False
        board["n_picks"][0, slot // TEAM_SIZE] += 1
        board["taken"][0, champ] = True
    for champ in bans:
        champ = int(champ)
        if not 0 <= champ < num_champs:
            raise ValueError("ban index %d out of range" % champ)
        if champ in seen:
            raise ValueError("champion %d is both picked and banned" % champ)
        if board["taken"][0, champ]:
            raise ValueError("champion %d banned twice" % champ)
        if int(board["n_bans"][0]) >= BAN_SLOTS:
            raise ValueError("more than %d bans" % BAN_SLOTS)
        board["ban_ids"][0, int(board["n_bans"][0])] = champ
        board["n_bans"][0] += 1
        board["taken"][0, champ] = True
    return board


def repeat_board(board, n):
    """Batch n copies (same board) - used to score candidate arms together."""
    out = dict(board)
    for key in ("champ_ids", "role_ids", "pad_mask", "ban_ids", "taken"):
        out[key] = board[key].repeat(n, *([1] * (board[key].dim() - 1)))
    out["n_bans"] = board["n_bans"].repeat(n)
    out["n_picks"] = board["n_picks"].repeat(n, 1)
    return out


def _policy_batch(board, move_idx, kind, team):
    """Tensors PolicyValueModel.forward expects, for one move index."""
    B = board["champ_ids"].shape[0]
    dev = board["champ_ids"].device

    def full(value, dtype=torch.long):
        return torch.full((B,), value, dtype=dtype, device=dev)

    return {
        "champ_ids": board["champ_ids"],
        "role_ids": board["role_ids"],
        "pad_mask": board["pad_mask"],
        "ban_ids": board["ban_ids"],
        "taken_mask": board["taken"],
        "unavailable_mask": torch.zeros_like(board["taken"]),
        "move_idx": full(move_idx),
        "phase": full(PHASE_OF[int(move_idx)]),
        "to_move": full(team),
        "first_pick": full(board["first_pick"]),
        "is_ban": full(1 if kind == MOVE_BAN else 0),
        "patch_id": full(PATCH_UNK_ID),
    }


# ------------------------------------------------------------------- models
def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# `from model import X` resolves to policy/model.py here, so the finetune
# models are loaded under a private module name. Imported lazily: the
# frontend must work with no value checkpoint present.
_FIN_MODEL = None


def finetune_model_module():
    global _FIN_MODEL
    if _FIN_MODEL is None:
        _FIN_MODEL = _load_by_path("finetune_model",
                                   MODEL_DIR / "finetune" / "model.py")
    return _FIN_MODEL


def load_policy_model(path, device="cpu", payload=None):
    """Policy checkpoint -> (PolicyValueModel in eval mode, config).

    `payload` lets a caller that already loaded the file (the frontend has
    to peek at the state dict to tell the model families apart) avoid
    reading a 400MB checkpoint twice.
    """
    blob = payload if payload is not None else torch.load(
        str(path), map_location="cpu", weights_only=False)
    cfg = blob["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        # Patch-conditioned checkpoints (the pro-only `*-patch-*` runs) carry
        # `encoder.embed.patch_lookup.*`; the encoder must be built with the
        # same `n_patches` or load_state_dict reports it as unexpected.
        n_patches=cfg.get("n_patches", 0),
        attr_matrix=attr_matrix_for_config(cfg))
    # `with_value` is not recorded by every training entry point (the
    # policy pipeline omits it for a value-free run). When it is absent,
    # infer it from the state dict instead of assuming a value head: a
    # missing `value_head.*` would otherwise fail the strict load.
    with_value = cfg.get("with_value")
    if with_value is None:
        with_value = any(k.startswith("value_head.") for k in blob["model"])
    model = PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg["state_dim"],
        n_econ=cfg.get("n_econ", 0), dropout=cfg.get("head_dropout", 0.0),
        n_patches=cfg.get("n_patches", 0),
        with_value=with_value)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError("policy ckpt mismatch: %s / %s"
                           % (list(missing)[:4], list(unexpected)[:4]))
    model.eval()
    model.to(device)
    return model, cfg


def load_value_model(path, device="cpu"):
    """PrefixValueModel checkpoint -> (model in eval mode, config)."""
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = blob["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        with_value=True, value_seeds=cfg.get("value_seeds", 4),
        attr_matrix=attr_matrix_for_config(cfg))
    model = finetune_model_module().PrefixValueModel(
        enc, ban_heads=cfg["ban_heads"], ban_seeds=cfg["ban_seeds"],
        ban_dropout=cfg["ban_dropout"], use_bans=cfg["use_bans"])
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError("value ckpt mismatch: %s / %s"
                           % (list(missing)[:4], list(unexpected)[:4]))
    model.eval()
    model.to(device)
    return model, cfg


# ---------------------------------------------------------------- predictor
class PolicyPredictor:
    """Next-move recommendations from a policy checkpoint.

    Two entry points:
      next_move(...)      what the side to move should take
      ban_candidates(...) what to ban, by simulating the draft after each
                          candidate and asking who the opponent would take
    """

    kind = "policy"

    def __init__(self, ckpt_path, device="cpu", value_path=None, payload=None):
        self.device = torch.device(device)
        self.model, self.cfg = load_policy_model(ckpt_path, self.device,
                                                 payload=payload)
        self.num_champs = int(self.cfg["num_champs"])
        self.ckpt_path = str(ckpt_path)
        self.metric = "next-move top5"
        metrics = (payload or {}).get("metrics", {})
        top5 = metrics.get("top5")
        self.val_top5 = float(top5) if top5 is not None else None
        self.epoch = metrics.get("epoch")
        self.value = None
        if value_path:
            try:
                self.value, _ = load_value_model(value_path, self.device)
                self.value_path = str(value_path)
            except Exception as exc:     # noqa: BLE001 - optional component
                print("value model unavailable (%s): %s" % (value_path, exc))
                self.value_path = None
        else:
            self.value_path = None

    # -- internals ----------------------------------------------------
    @torch.no_grad()
    def _fill_roles(self, board):
        """Set role_ids from the encoder's role head; -> per-slot probs.

        Runs with the current role inputs (FLEX for a just-placed pick),
        which is the same "role unknown" query the masked frontend uses,
        so the answer is conditioned on the champion and its team context
        only.
        """
        batch = {
            "champ_ids": board["champ_ids"],
            "role_ids": board["role_ids"],
            "pad_mask": board["pad_mask"],
            "is_masked": torch.zeros_like(board["champ_ids"], dtype=torch.bool),
        }
        role_logits = self.model.encoder(batch, compute_heads=True)["role_logits"]
        probs = torch.softmax(role_logits.float(), dim=-1)
        best = probs.argmax(dim=-1)
        board["role_ids"] = torch.where(~board["pad_mask"], best,
                                        torch.full_like(best, ROLE_FLEX_ID))
        return probs

    def _place(self, board, team, kind, champs):
        """Write the chosen champions into the board and update legality."""
        idx = torch.arange(board["champ_ids"].shape[0], device=self.device)
        if kind == MOVE_BAN:
            pos = board["n_bans"].clamp(max=BAN_SLOTS - 1)
            board["ban_ids"][idx, pos] = champs
            board["n_bans"] = (board["n_bans"] + 1).clamp(max=BAN_SLOTS)
        else:
            slot = (team * TEAM_SIZE + board["n_picks"][:, team]).clamp(
                max=team * TEAM_SIZE + TEAM_SIZE - 1)
            board["champ_ids"][idx, slot] = champs
            board["pad_mask"][idx, slot] = False
            board["role_ids"][idx, slot] = ROLE_FLEX_ID
            board["n_picks"][:, team] += 1
        board["taken"][idx, champs] = True

    @torch.no_grad()
    def _rollout(self, board, move_idx, forced=None, watch=None):
        """Play moves move_idx..19 with the policy on both sides.

        `forced` overrides the champion at the first move (the ban under
        evaluation). `watch=(kind, team)` captures that side's first
        matching move distribution - used to read "what would they pick".
        """
        plan = move_plan(board["first_pick"])
        seen = None
        for t in range(int(move_idx), N_MOVES):
            kind, team = plan[t]
            logits = self.model(_policy_batch(board, t, kind, team))[
                "policy_logits"]
            champs = forced if (t == move_idx and forced is not None) \
                else logits.argmax(dim=1)
            if watch is not None and seen is None and (kind, team) == watch:
                seen = torch.softmax(logits.float(), dim=1)
            self._place(board, team, kind, champs)
            if kind == MOVE_PICK:
                self._fill_roles(board)
        return board, seen

    def _win_prob(self, board):
        """P(blue wins) for a finished board, or None without a value model."""
        if self.value is None:
            return None
        batch = {k: board[k] for k in ("champ_ids", "role_ids", "pad_mask",
                                       "ban_ids")}
        with torch.no_grad():
            logit = self.value(batch, with_bans=False)["win_logit"]
        return torch.sigmoid(logit.float().squeeze(-1))

    def _roles_json(self, probs, board):
        """Per-slot role distribution, same shape as the masked frontend.

        Empty slots report an empty list (the masked panel only renders
        roles for picked champions), even though the role head always
        emits a distribution.
        """
        out = []
        for slot in range(N_SLOTS):
            if bool(board["pad_mask"][0, slot]):
                out.append([])
                continue
            order = torch.argsort(probs[0, slot], descending=True)
            out.append([{"role": int(r), "prob": float(probs[0, slot, r])}
                        for r in order[:NUM_ROLES]])
        return out

    # -- public -------------------------------------------------------
    @torch.no_grad()
    def next_move(self, picks, bans, first_pick=0, k=8):
        """Top-k champions for the move that is next, plus the state."""
        board = build_board(picks, bans, self.num_champs, first_pick,
                            self.device)
        n_picks = int((~board["pad_mask"][0]).sum())
        move_idx, left_b, left_p = resolve_position(
            int(board["n_bans"][0]), n_picks, first_pick)
        warnings = []
        if left_b:
            warnings.append("%d ban(s) do not fit the draft order" % left_b)
        if left_p:
            warnings.append("%d pick(s) do not fit the draft order" % left_p)
        if move_idx >= N_MOVES:
            return {"complete": True, "warning": "; ".join(warnings) or None,
                    "first_pick": int(first_pick), "roles": None,
                    "suggestions": [], "move_idx": N_MOVES,
                    "kind": None, "side": None, "phase": None}
        probs = self._fill_roles(board)
        kind, team = move_plan(first_pick)[move_idx]
        logits = self.model(_policy_batch(board, move_idx, kind, team))[
            "policy_logits"][0]
        p = torch.softmax(logits.float(), dim=-1)
        top = p.topk(min(k, self.num_champs))
        suggestions = [{"idx": int(i), "prob": float(v)}
                       for i, v in zip(top.indices, top.values)
                       if float(v) > 0 and int(i) < self.num_champs]
        return {
            "complete": False,
            "move_idx": move_idx,
            "kind": "ban" if kind == MOVE_BAN else "pick",
            "side": int(team),
            "phase": PHASE_NAMES[PHASE_OF[move_idx]],
            "first_pick": int(first_pick),
            "warning": "; ".join(warnings) or None,
            "suggestions": suggestions,
            "roles": self._roles_json(probs, board),
        }

    @torch.no_grad()
    def ban_candidates(self, picks, bans, first_pick=0, k=4, top_picks=3):
        """Rank ban candidates by how likely the opponent was to take them.

        The reference rollout lets the policy play the current ban itself,
        then records the opponent's first pick. Each shortlisted candidate
        is then banned for real and the draft is played out, so the win
        probabilities are for genuinely different finished drafts (the
        regime the value function was trained on) rather than different
        bans on one unchanged board, where it is structurally flat.

        Returns None when the next move is not a ban.
        """
        board = build_board(picks, bans, self.num_champs, first_pick,
                            self.device)
        n_picks = int((~board["pad_mask"][0]).sum())
        move_idx, _lb, _lp = resolve_position(
            int(board["n_bans"][0]), n_picks, first_pick)
        plan = move_plan(first_pick)
        if move_idx >= N_MOVES or plan[move_idx][0] != MOVE_BAN:
            return None
        my_team = plan[move_idx][1]
        opp = 1 - my_team
        self._fill_roles(board)

        # 1. reference: the policy bans its own choice, we watch their pick
        ref = repeat_board(board, 1)
        _ref, seen = self._rollout(ref, move_idx, watch=(MOVE_PICK, opp))
        if seen is None:
            return None
        ref_probs = seen[0]

        # 2. shortlist = the champions they were most likely to take
        taken = board["taken"][0]
        order = torch.argsort(ref_probs, descending=True)
        short = [int(c) for c in order[:k + 8]
                 if not bool(taken[int(c)]) and int(c) < self.num_champs][:k]
        if not short:
            return None

        # 3. ban each candidate for real, finish the draft
        arms = repeat_board(board, len(short))
        sims, _ = self._rollout(arms, move_idx,
                                forced=torch.tensor(short, device=self.device))
        win = self._win_prob(sims)
        if win is not None:
            # report the acting side's win probability, not blue's
            win = win if my_team == 0 else 1.0 - win

        opp_top = []
        for c in order[:top_picks + 4]:
            c = int(c)
            if taken[c] or c >= self.num_champs:
                continue
            opp_top.append({"idx": c, "prob": float(ref_probs[c])})
            if len(opp_top) == top_picks:
                break

        cands = []
        for j, champ in enumerate(short):
            cands.append({
                "idx": int(champ),
                "denial": float(ref_probs[champ]),
                "win_prob": (float(win[j]) if win is not None else None),
            })
        cands.sort(key=lambda c: (-c["denial"],
                                  -(c["win_prob"] if c["win_prob"] is not None
                                    else 0.0)))
        for rank, c in enumerate(cands, 1):
            c["rank"] = rank
        return {
            "move_idx": int(move_idx),
            "side": int(my_team),
            "opponent": int(opp),
            "opponent_picks": opp_top,
            "candidates": cands,
            "ranked_by": "denial",
            "win_prob_available": win is not None,
        }
