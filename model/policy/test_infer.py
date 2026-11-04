"""Tests for the policy inference layer (model/policy/infer.py).

The two claims that matter:

1. The UI's move order is the *data's* move order. `move_plan` is compared
   against `pipeline.interleaved_moves` directly, so the frontend cannot
   drift from what the policy was trained on.
2. Recommendations respect the board: never a champion that is already
   picked or banned, and the move index/side the board actually supports.

A tiny randomly-initialised PolicyValueModel is used as the stub - these
are contract tests, not accuracy tests.
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
PRO_DIR = MODEL_DIR.parent / "data" / "pro play"
for p in (str(HERE), str(MODEL_DIR), str(PRO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import infer  # noqa: E402
import pipeline as P  # noqa: E402
from model import PolicyValueModel  # noqa: E402
from model.pretraining.soloq_model import SoloQPretrainedModel  # noqa: E402

# The kept soloQ-encoder policy. It used to be
# `ckpt/run_20261025_000606_ablF-stage1/best_policy.pt`, which the 2026-11-03
# reorganisation deleted, so this test had been skipping silently ever since -
# a skipped test is worse than no test, because the suite still reads green.
CKPT = HERE.parent / "checkpoints" / "nextpick-policy" / "best_policy.pt"


def stub(num_champs=173, seed=0):
    """A PolicyPredictor with a tiny random policy (no checkpoint).

    num_champs stays 173: PolicyValueModel's policy head is hard-coded to
    NUM_REAL_CHAMPS, so only the encoder width can be shrunk here.
    """
    torch.manual_seed(seed)
    enc = SoloQPretrainedModel(num_champs=num_champs, embed_dim=32,
                               num_heads=4, num_intra_layers=1,
                               num_cross_layers=1)
    model = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                             n_patches=0, with_value=False)
    model.eval()
    pred = object.__new__(infer.PolicyPredictor)
    pred.device = torch.device("cpu")
    pred.model = model
    pred.num_champs = num_champs
    pred.cfg = {"num_champs": num_champs}
    pred.ckpt_path = "stub"
    pred.val_top5 = float("nan")
    pred.epoch = None
    pred.value = None
    pred.value_path = None
    return pred


# ------------------------------------------------------------ move order
@pytest.mark.parametrize("first_pick", [0, 1])
def test_move_plan_matches_the_data_pipeline(first_pick):
    """UI order == training order, for both first-pick sides."""
    bans = {0: [1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10]}
    picks = {0: [(11 + i, i) for i in range(5)],
             1: [(20 + i, i) for i in range(5)]}
    rows = P.interleaved_moves(first_pick, bans, picks)
    want = [("ban" if k == P.MOVE_BAN else "pick", team) for k, team, _, _ in rows]
    got = [("ban" if k == infer.MOVE_BAN else "pick", team)
           for k, team in infer.move_plan(first_pick)]
    assert got == want


def test_move_plan_shape():
    for fp in (0, 1):
        plan = infer.move_plan(fp)
        assert len(plan) == 20
        assert sum(1 for k, _ in plan if k == infer.MOVE_BAN) == 10
        assert sum(1 for k, _ in plan if k == infer.MOVE_PICK) == 10
        # each side picks exactly 5
        assert [sum(1 for k, t in plan if k == infer.MOVE_PICK and t == side)
                for side in (0, 1)] == [5, 5]
        # phases: 6 bans, 6 picks, 4 bans, 4 picks
        assert [k for k, _ in plan[:6]] == [infer.MOVE_BAN] * 6
        assert [k for k, _ in plan[6:12]] == [infer.MOVE_PICK] * 6
        assert [k for k, _ in plan[12:16]] == [infer.MOVE_BAN] * 4
        assert [k for k, _ in plan[16:]] == [infer.MOVE_PICK] * 4


# -------------------------------------------------------------- position
def test_resolve_position_counts_a_prefix():
    assert infer.resolve_position(0, 0, 0) == (0, 0, 0)
    assert infer.resolve_position(6, 0, 0) == (6, 0, 0)
    assert infer.resolve_position(6, 6, 0) == (12, 0, 0)
    assert infer.resolve_position(10, 10, 0) == (20, 0, 0)
    # one ban short of the pick phase: the walk stops at the gap
    assert infer.resolve_position(5, 1, 0) == (5, 0, 1)


def test_resolve_position_reports_leftovers():
    """A board that is not a legal prefix is reported, not reordered."""
    k, left_b, left_p = infer.resolve_position(3, 1, 0)
    assert (k, left_b, left_p) == (3, 0, 1)
    k, left_b, left_p = infer.resolve_position(9, 0, 0)
    assert left_b == 3 and k == 6      # bans1 can only absorb 6


# ----------------------------------------------------------------- board
def test_build_board_rejects_illegal_boards():
    with pytest.raises(ValueError):
        infer.build_board([1] * 10, [], 40)
    with pytest.raises(ValueError):
        infer.build_board([1, None, None, None, None, None, None, None,
                           None, None], [1], 40)
    with pytest.raises(ValueError):
        infer.build_board([99] + [None] * 9, [], 40)
    with pytest.raises(ValueError):
        infer.build_board([None] * 10, [0, 0], 40)
    with pytest.raises(ValueError):
        infer.build_board([None] * 10, list(range(11)), 40)
    with pytest.raises(ValueError):
        infer.build_board([None] * 9, [], 40)


def test_build_board_tracks_counts_and_taken():
    picks = [1, 2, 3, None, None, 4, 5, None, None, None]
    board = infer.build_board(picks, [6, 7], 40)
    assert board["champ_ids"][0, 0].item() == 1
    assert board["pad_mask"][0, 3].item() is True
    assert int(board["n_bans"][0]) == 2
    assert board["n_picks"][0].tolist() == [3, 2]
    for c in (1, 2, 3, 4, 5, 6, 7):
        assert board["taken"][0, c].item() is True
    assert board["taken"][0, 8].item() is False
    # no pool passed -> nothing unavailable
    assert not board["unavailable"].any().item()


def test_repeat_board_shapes():
    board = infer.build_board([1, None, None, None, None, None, None, None,
                               None, None], [2], 40)
    rep = infer.repeat_board(board, 3)
    for key in ("champ_ids", "role_ids", "pad_mask", "ban_ids", "taken",
                "unavailable"):
        assert rep[key].shape[0] == 3
    assert rep["n_picks"].shape == (3, 2)
    assert torch.equal(rep["champ_ids"][0], rep["champ_ids"][2])


# ------------------------------------------------------------- fearless
# A fearless pool is the champions picked in earlier games of the series.
# The contract that matters: illegal to PICK, legal to BAN. The model
# enforces that split itself (`_apply_legality`), so the board must keep the
# two masks apart rather than folding the pool into `taken`.
def test_build_board_keeps_the_fearless_pool_out_of_taken():
    board = infer.build_board([None] * 10, [20], 40, unavailable=[20, 21])
    assert board["taken"][0, 20].item() is True       # banned this game
    assert board["unavailable"][0, 20].item() is True  # and fearless
    assert board["unavailable"][0, 21].item() is True
    # 21 is fearless only: banning it is still legal, so it is not "taken"
    assert board["taken"][0, 21].item() is False
    assert board["unavailable"][0, 22].item() is False


def test_build_board_rejects_a_fearless_pick():
    with pytest.raises(ValueError):
        infer.build_board([20] + [None] * 9, [], 40, unavailable=[20])
    with pytest.raises(ValueError):
        infer.build_board([None] * 10, [], 40, unavailable=[99])


def test_repeat_board_carries_the_fearless_pool():
    board = infer.build_board([None] * 10, [2], 40, unavailable=[20, 21])
    rep = infer.repeat_board(board, 4)
    assert rep["unavailable"].shape == (4, 40)
    assert rep["unavailable"][:, 20].all().item()
    assert rep["unavailable"][:, 21].all().item()


def test_next_move_masks_fearless_on_picks_only():
    """The whole point of the split, at the inference boundary.

    k = num_champs asks for every champion, so a masked one is simply
    absent from the answer (softmax of -inf is 0) and the test reads the
    legality rule directly instead of sampling a top-8.
    """
    pred = stub()
    pool = [20, 21, 22]

    # move 0 is a ban: a champion picked last game can still be banned
    out = pred.next_move([None] * 10, [], first_pick=0, k=173,
                         unavailable=pool)
    assert out["kind"] == "ban"
    assert out["n_unavailable"] == 3
    assert set(pool) <= {c["idx"] for c in out["suggestions"]}, \
        "fearless champions must stay bannable"

    # six bans -> move 6, the first pick: now they cannot be taken
    bans = [10, 11, 12, 13, 14, 15]
    out = pred.next_move([None] * 10, bans, first_pick=0, k=173,
                         unavailable=pool)
    assert out["kind"] == "pick"
    got = {c["idx"] for c in out["suggestions"]}
    assert not (got & set(pool)), "fearless champions must not be pickable"
    assert not (got & set(bans))


def test_ban_candidates_carry_the_fearless_pool_through_the_rollout():
    """A simulated pick must not land on a champion that is unavailable.

    The rollout plays the rest of the draft with the policy; if the pool did
    not survive `repeat_board`/`_place`, an arm would be free to "answer" a
    ban with a champion that is already gone - which is what made the
    pre-fearless UI propose illegal continuations.
    """
    pred = stub()
    pool = [20, 21, 22]
    res = pred.ban_candidates([None] * 10, [], first_pick=0, k=3,
                              unavailable=pool)
    assert res is not None
    taken = {c["idx"] for c in res["candidates"]}
    assert not (taken & set(pool)), "cannot ban-propose a champion twice"
    for c in res["opponent_picks"]:
        assert c["idx"] not in pool


# ------------------------------------------------------------- next move
def test_next_move_follows_the_plan():
    pred = stub()
    picks = [None] * 10
    bans = []
    plan = infer.move_plan(0)
    # walk the first 8 moves, filling in exactly what the plan says
    for t in range(8):
        kind, team = plan[t]
        out = pred.next_move(picks, bans, first_pick=0, k=5)
        assert out["move_idx"] == t
        assert out["kind"] == ("ban" if kind == infer.MOVE_BAN else "pick")
        assert out["side"] == team
        assert out["phase"] == infer.PHASE_NAMES[infer.PHASE_OF[t]]
        champ = out["suggestions"][0]["idx"]
        if kind == infer.MOVE_BAN:
            bans.append(champ)
        else:
            picks[team * 5 + sum(1 for s in range(team * 5, team * 5 + 5)
                                 if picks[s] is not None)] = champ


def test_next_move_suggestions_are_legal():
    pred = stub()
    picks = [3, 4, None, None, None, 5, None, None, None, None]
    bans = [0, 1, 2, 6, 7, 8]
    out = pred.next_move(picks, bans, first_pick=0, k=8)
    got = {c["idx"] for c in out["suggestions"]}
    assert got, "expected suggestions"
    assert not (got & set(picks + bans))
    probs = [c["prob"] for c in out["suggestions"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0 + 1e-6


def test_next_move_on_a_complete_board():
    pred = stub()
    picks = list(range(10))
    bans = list(range(10, 20))
    out = pred.next_move(picks, bans, first_pick=0)
    assert out["complete"] is True
    assert out["suggestions"] == []


def test_next_move_warns_about_an_out_of_order_board():
    """Bans carry no team in the model, so a per-team mismatch is harmless
    - only counts that cannot fit the phase order are flagged."""
    pred = stub()
    out = pred.next_move([None] * 10, [1, 2, 3], first_pick=0)
    assert out["warning"] is None and out["move_idx"] == 3
    out = pred.next_move([None] * 10, list(range(1, 10)), first_pick=0)
    assert out["warning"] and "do not fit" in out["warning"]
    assert out["move_idx"] == 6          # bans phase 1 can only absorb 6


def test_next_move_returns_roles_for_picked_slots():
    pred = stub()
    picks = [11, 12, None, None, None, None, None, None, None, None]
    out = pred.next_move(picks, [], first_pick=0)
    roles = out["roles"]
    assert len(roles) == 10
    assert roles[0] and 0 <= roles[0][0]["role"] < 5
    assert roles[2] == [] or all(r["role"] < 5 for r in roles[2])


# ----------------------------------------------------------- ban ranking
def test_ban_candidates_none_when_the_next_move_is_a_pick():
    pred = stub()
    picks = [None] * 10
    bans = [1, 2, 3, 4, 5, 6]           # bans1 done -> pick
    assert pred.ban_candidates(picks, bans, first_pick=0) is None


def test_ban_candidates_shortlist_is_legal_and_sorted():
    pred = stub()
    picks = [10, 11, 12, None, None, 13, 14, 15, None, None]
    bans = [1, 2, 3, 4, 5, 6]
    out = pred.ban_candidates(picks, bans, first_pick=0, k=4)
    assert out is not None
    assert out["move_idx"] == 12          # bans phase 2 starts at 12
    assert out["side"] == 1               # red bans first in phase 2
    taken = set(picks) | set(bans)
    cands = out["candidates"]
    assert 1 <= len(cands) <= 4
    assert not ({c["idx"] for c in cands} & taken)
    denials = [c["denial"] for c in cands]
    assert denials == sorted(denials, reverse=True)
    assert [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
    assert all(c["win_prob"] is None for c in cands)
    assert out["win_prob_available"] is False


def test_ban_candidates_opponent_list_is_legal():
    pred = stub()
    picks = [10, 11, 12, None, None, 13, 14, 15, None, None]
    bans = [1, 2, 3, 4, 5, 6]
    out = pred.ban_candidates(picks, bans, first_pick=0, k=3, top_picks=3)
    taken = set(picks) | set(bans)
    assert out["opponent_picks"]
    assert not ({c["idx"] for c in out["opponent_picks"]} & taken)
    probs = [c["prob"] for c in out["opponent_picks"]]
    assert probs == sorted(probs, reverse=True)


def test_ban_candidates_on_a_red_first_pick_board():
    """first_pick flips who bans first in phase 2, not just phase 1."""
    pred = stub()
    picks = [10, 11, 12, None, None, 13, 14, 15, None, None]
    bans = [1, 2, 3, 4, 5, 6]
    out = pred.ban_candidates(picks, bans, first_pick=1)
    assert out["side"] == infer.move_plan(1)[12][1] == 0


# ------------------------------------------------------------ placement
def test_place_never_overflows_the_ban_slots():
    pred = stub()
    board = infer.build_board([None] * 10, [], 40)
    kind, team = infer.MOVE_BAN, 0
    for c in range(12):
        pred._place(board, team, kind, torch.tensor([c]))
    assert int(board["n_bans"][0]) == infer.BAN_SLOTS
    assert board["ban_ids"].shape == (1, infer.BAN_SLOTS)
    assert all(int(v) < 40 for v in board["ban_ids"][0])


def test_place_fills_pick_slots_in_order_and_marks_taken():
    pred = stub()
    board = infer.build_board([None] * 10, [], 40)
    for c in (5, 6):
        pred._place(board, 1, infer.MOVE_PICK, torch.tensor([c]))
    assert board["champ_ids"][0, 5].item() == 5
    assert board["champ_ids"][0, 6].item() == 6
    assert board["pad_mask"][0, 5].item() is False
    assert board["taken"][0, 6].item() is True
    assert board["n_picks"][0].tolist() == [0, 2]


# ------------------------------------------------------- real checkpoint
@pytest.mark.skipif(not CKPT.is_file(), reason="kept policy checkpoint absent")
def test_real_policy_checkpoint_loads_and_predicts():
    pred = infer.PolicyPredictor(CKPT, device="cpu")
    assert pred.kind == "policy"
    assert pred.num_champs == 173
    out = pred.next_move([None] * 10, [], first_pick=0, k=5)
    assert out["move_idx"] == 0 and out["kind"] == "ban" and out["side"] == 0
    assert len(out["suggestions"]) == 5
    assert all(0 <= c["idx"] < 173 for c in out["suggestions"])
