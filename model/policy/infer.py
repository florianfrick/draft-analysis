"""Inference API for the next-move policy.

Board in, recommendation out. No CLI, no dataset import, no training code:
`frontend/server.py` imports this, and it is the only path that serves the
policy. The offline ban-search statistics (bootstrap CIs, paired tests,
regret) lived in `ban_propose.py`, removed on 2026-11-03 with the rest of the
ban-analysis thread; `STATUS.md` §5 keeps its conclusions. The rollout-based
`ban_candidates` was removed on 2026-11-04: a pro-like ban already encodes
denial, so greedy self-play on top of a value head that measures nothing only
compounds model error.

Three pieces:

    move_plan(first_pick)     the canonical 20-move order
    resolve_position(...)     how far along a hand-built board is
    PolicyPredictor           the next-move distribution

Board convention (the UI's, not the dataset's)
----------------------------------------------
`picks` is a 10-slot list - blue 0-4, red 5-9, in PICK ORDER, None = empty.
`bans` is the list of champions banned so far, any order. The slot index
within a team carries no meaning to the model: the encoder has no
positional embedding and the policy pools each team with a
permutation-invariant reduction, so only the (champion, role) pair is
reconstructed. Roles are not part of the UI board, so they are *inferred*
once per state from the encoder's role head (the same FLEX-role query the
masked frontend uses) and then fed back in.

`unavailable` is the fearless pool: champions picked in earlier games of
the same series. The BOARD keeps it separate from `taken` so a board that
already bans one of them still loads: 8 of 100,210 ban moves in fearless
corpus rows do exactly that (technically legal, never sensible). But
serving folds the pool into `taken_mask`, so no suggestion - pick OR ban -
can ever name a fearless champion. Training keeps the looser rule, because
masking those 8 targets out of the cross-entropy would dominate the loss.
This is a policy-only input; the masked models have no equivalent.

Patch conditioning
------------------
`patch_id` is a row in the checkpoint's OWN patch vocabulary, not a patch
string. `GET /api/champions` publishes that vocabulary; the UI sends the
selected id on `/api/next-move` and `/api/explain`. Unknown or out-of-range
ids - and every id on a checkpoint with no patch table - fall back to
`PATCH_UNK_ID`. Breakdown scoring is the exception: it uses each real pro
game's own patch, not the UI's selection.
"""
from __future__ import annotations

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
DEFAULT_PATCH = "16.20"     # the patch the frozen corpus ends on


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
        "unavailable": torch.zeros(1, num_champs, dtype=torch.bool,
                                   device=device),
        "n_bans": torch.zeros(1, dtype=torch.long, device=device),
        "n_picks": torch.zeros(1, 2, dtype=torch.long, device=device),
        "first_pick": int(first_pick),
        "num_champs": int(num_champs),
    }


def build_board(picks, bans, num_champs, first_pick=0, device="cpu",
                unavailable=None):
    """UI board -> batch-1 tensors. Raises ValueError on an illegal board.

    picks: 10 entries, champion index or None (blue 0-4 / red 5-9, pick
    order). bans: champion indices banned so far, any order. `unavailable`:
    champions picked in *earlier games of a fearless series*.

    `unavailable` stays its own mask rather than being folded into `taken`
    here, so that a board which already banned one of those champions still
    loads - the corpus contains 8 such rows and the UI must not 400 on them.
    It is folded into `taken_mask` at the inference boundary
    (`_policy_batch`), where the pool becomes unconditionally illegal to
    suggest. A champion both *picked* and unavailable THIS game is still
    rejected.
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
    for champ in (unavailable or ()):
        champ = int(champ)
        if not 0 <= champ < num_champs:
            raise ValueError("fearless index %d out of range" % champ)
        if champ in seen:
            raise ValueError(
                "champion %d was picked earlier in the series and cannot be "
                "picked again" % champ)
        board["unavailable"][0, champ] = True
    return board


def _policy_batch(board, move_idx, kind, team, patch_id=PATCH_UNK_ID):
    """Tensors PolicyValueModel.forward expects, for one move index.

    `taken_mask` is `taken | unavailable` on purpose: the UI must never
    offer a fearless champion, on a pick OR a ban. Keeping the two apart
    in the board is what lets a technically-legal board load; folding them
    HERE is what makes the suggestion safe. `PolicyValueModel`'s own
    `_apply_legality` keeps the looser TRAINING rule - see the module
    docstring for the 8 corpus rows that require it.
    """
    B = board["champ_ids"].shape[0]
    dev = board["champ_ids"].device

    def full(value, dtype=torch.long):
        return torch.full((B,), value, dtype=dtype, device=dev)

    return {
        "champ_ids": board["champ_ids"],
        "role_ids": board["role_ids"],
        "pad_mask": board["pad_mask"],
        "ban_ids": board["ban_ids"],
        "taken_mask": board["taken"] | board["unavailable"],
        "unavailable_mask": board["unavailable"],
        "move_idx": full(move_idx),
        "phase": full(PHASE_OF[int(move_idx)]),
        "to_move": full(team),
        "first_pick": full(board["first_pick"]),
        "is_ban": full(1 if kind == MOVE_BAN else 0),
        "patch_id": full(int(patch_id)),
    }


# ------------------------------------------------------------------- models
def load_policy_model(path, device="cpu", payload=None):
    """Policy checkpoint -> (PolicyValueModel in eval mode, config).

    `payload` lets a caller that already loaded the file (the frontend has
    to peek at the state dict to tell the model families apart) avoid
    reading a 400MB checkpoint twice. `patch_vocab` is copied into the
    returned config when it only exists at the checkpoint top level, so
    the predictor can resolve patch ids without a second torch.load.
    """
    blob = payload if payload is not None else torch.load(
        str(path), map_location="cpu", weights_only=False)
    cfg = dict(blob["config"])
    if not cfg.get("patch_vocab"):
        cfg["patch_vocab"] = blob.get("patch_vocab") or []
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        # Patch-conditioned checkpoints carry `encoder.embed.patch_lookup.*`;
        # the encoder must be built with the same `n_patches` or
        # load_state_dict reports it as unexpected.
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


# ---------------------------------------------------------------- predictor
class PolicyPredictor:
    """Next-move recommendations from a policy checkpoint.

    One entry point: `next_move(...)` returns the policy's distribution for
    the move the board is actually on, ban or pick.
    """

    kind = "policy"

    def __init__(self, ckpt_path, device="cpu", payload=None):
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
        self.n_patches = int(self.cfg.get("n_patches", 0))
        self.patch_vocab = list(self.cfg.get("patch_vocab") or [])

    def patch_id_for(self, patch_id):
        """Client-supplied patch id -> a row this checkpoint actually has.

        Anything malformed, out of range, or on a checkpoint without a
        patch table (`n_patches == 0`) becomes UNK rather than an embedding
        lookup error.
        """
        try:
            pid = int(patch_id)
        except (TypeError, ValueError):
            return PATCH_UNK_ID
        if not 0 <= pid < self.n_patches:
            return PATCH_UNK_ID
        return pid

    def patch_key_for(self, patch_id):
        """The patch label for a resolved id, or None for UNK."""
        pid = self.patch_id_for(patch_id)
        if pid <= 0 or pid - 1 >= len(self.patch_vocab):
            return None
        return str(self.patch_vocab[pid - 1])

    def patch_options(self):
        """[{id, key}] for the UI selector, newest patch first."""
        out = [{"id": i + 1, "key": str(k)}
               for i, k in enumerate(self.patch_vocab)]
        out.reverse()
        return out

    def default_patch_id(self):
        """Id of `DEFAULT_PATCH`, else the newest known patch, else UNK."""
        opts = self.patch_options()
        for o in opts:
            if o["key"] == DEFAULT_PATCH:
                return o["id"]
        return opts[0]["id"] if opts else PATCH_UNK_ID

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
    def next_move(self, picks, bans, first_pick=0, k=8, unavailable=None,
                  patch_id=PATCH_UNK_ID):
        """Top-k champions for the move that is next, plus the state.

        `unavailable` is the fearless pool (champions picked in earlier
        games of the series); it is folded into the legality mask on BOTH
        move kinds, so it can never appear among the suggestions. The
        board itself still accepts one as an already-placed ban.

        `patch_id` selects a row of this checkpoint's patch embedding;
        `patch_id_for` clamps anything it cannot resolve to UNK.
        """
        pid = self.patch_id_for(patch_id)
        board = build_board(picks, bans, self.num_champs, first_pick,
                            self.device, unavailable=unavailable)
        n_unavail = int(board["unavailable"][0].sum())
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
                    "kind": None, "side": None, "phase": None,
                    "patch_id": pid, "n_unavailable": n_unavail}
        probs = self._fill_roles(board)
        kind, team = move_plan(first_pick)[move_idx]
        logits = self.model(_policy_batch(board, move_idx, kind, team,
                                          patch_id=pid))["policy_logits"][0]
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
            "patch_id": pid,
            "n_unavailable": n_unavail,
        }
