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


def stub(num_champs=173, seed=0, n_patches=0):
    """A PolicyPredictor with a tiny random policy (no checkpoint).

    num_champs stays 173: PolicyValueModel's policy head is hard-coded to
    NUM_REAL_CHAMPS, so only the encoder width can be shrunk here.
    """
    torch.manual_seed(seed)
    enc = SoloQPretrainedModel(num_champs=num_champs, embed_dim=32,
                               num_heads=4, num_intra_layers=1,
                               num_cross_layers=1, n_patches=n_patches)
    model = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                             n_patches=n_patches, with_value=False)
    model.eval()
    pred = object.__new__(infer.PolicyPredictor)
    pred.device = torch.device("cpu")
    pred.model = model
    pred.num_champs = num_champs
    pred.cfg = {"num_champs": num_champs}
    pred.ckpt_path = "stub"
    pred.val_top5 = float("nan")
    pred.epoch = None
    pred.n_patches = int(n_patches)
    pred.patch_vocab = []
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


# ------------------------------------------------------------- fearless
# A fearless pool is the champions picked in earlier games of the series.
# The board keeps it separate from `taken` so a technically-legal board (one
# that already bans a pool champion) still loads. The INFERENCE batch folds it
# into `taken_mask`, so no suggestion - pick OR ban - can name one. The corpus
# has 8 such bans across 100,210 fearless-row ban moves, which is exactly why
# the board is tolerant and the suggestion layer is strict.
def test_build_board_keeps_the_fearless_pool_out_of_taken():
    board = infer.build_board([None] * 10, [20], 40, unavailable=[20, 21])
    assert board["taken"][0, 20].item() is True       # banned this game
    assert board["unavailable"][0, 20].item() is True  # and fearless
    assert board["unavailable"][0, 21].item() is True
    # 21 is fearless only: the board loads, `_policy_batch` excludes it
    assert board["taken"][0, 21].item() is False
    assert board["unavailable"][0, 22].item() is False


def test_build_board_accepts_a_board_that_bans_a_fearless_champion():
    """Technically legal, never done in practice: must not raise.

    The UI has to be able to display such a board (8 corpus rows do it);
    it must simply never propose one.
    """
    board = infer.build_board([None] * 10, [20, 21], 40, unavailable=[20, 21])
    assert board["taken"][0, 20].item() is True
    assert board["unavailable"][0, 20].item() is True


def test_build_board_rejects_a_fearless_pick():
    with pytest.raises(ValueError):
        infer.build_board([20] + [None] * 9, [], 40, unavailable=[20])
    with pytest.raises(ValueError):
        infer.build_board([None] * 10, [], 40, unavailable=[99])


def test_policy_batch_folds_the_pool_into_taken():
    """The one place the two masks are combined."""
    board = infer.build_board([None] * 10, [3], 40, unavailable=[20, 21])
    batch = infer._policy_batch(board, 0, infer.MOVE_BAN, 0)
    for champ in (3, 20, 21):
        assert batch["taken_mask"][0, champ].item() is True
    # the separate mask is still supplied, so the batch keeps the keys the
    # training path uses
    assert batch["unavailable_mask"][0, 20].item() is True


def test_next_move_masks_fearless_on_picks_and_bans():
    """The whole point: a fearless champion is never suggested.

    k = num_champs asks for every champion, so a masked one is simply
    absent from the answer (softmax of -inf is 0) and the test reads the
    legality rule directly instead of sampling a top-8.
    """
    pred = stub()
    pool = [20, 21, 22]

    # move 0 is a ban: the pool is excluded here too, not only on picks
    out = pred.next_move([None] * 10, [], first_pick=0, k=173,
                         unavailable=pool)
    assert out["kind"] == "ban"
    assert out["n_unavailable"] == 3
    assert not ({c["idx"] for c in out["suggestions"]} & set(pool)), \
        "fearless champions must never be suggested for a ban"

    # six bans -> move 6, the first pick: same rule
    bans = [10, 11, 12, 13, 14, 15]
    out = pred.next_move([None] * 10, bans, first_pick=0, k=173,
                         unavailable=pool)
    assert out["kind"] == "pick"
    got = {c["idx"] for c in out["suggestions"]}
    assert not (got & set(pool)), "fearless champions must not be pickable"
    assert not (got & set(bans))


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


# --------------------------------------------------------------- patch id
def test_patch_id_is_unknown_without_a_patch_table():
    pred = stub()                       # n_patches = 0
    assert pred.patch_id_for(0) == infer.PATCH_UNK_ID
    assert pred.patch_id_for(7) == infer.PATCH_UNK_ID
    assert pred.patch_key_for(7) is None
    assert pred.patch_options() == []
    assert pred.default_patch_id() == infer.PATCH_UNK_ID


def test_patch_id_is_clamped_to_the_checkpoint_vocabulary():
    """An id the checkpoint cannot embed must become UNK, not raise."""
    pred = stub(n_patches=4)            # UNK + 3 patches
    pred.patch_vocab = ["16.18", "16.19", "16.20"]
    assert pred.patch_id_for(3) == 3
    assert pred.patch_key_for(3) == "16.20"
    assert pred.patch_id_for(4) == infer.PATCH_UNK_ID      # past the table
    assert pred.patch_id_for(-1) == infer.PATCH_UNK_ID
    assert pred.patch_id_for("nonsense") == infer.PATCH_UNK_ID
    assert pred.patch_id_for(None) == infer.PATCH_UNK_ID
    assert pred.patch_key_for(0) is None


def test_patch_options_are_newest_first_and_default_to_16_20():
    pred = stub(n_patches=4)
    pred.patch_vocab = ["16.18", "16.19", "16.20"]
    assert [o["key"] for o in pred.patch_options()] == \
        ["16.20", "16.19", "16.18"]
    assert pred.default_patch_id() == 3
    # no 16.20 in this table -> the newest patch it does know
    pred.patch_vocab = ["16.18", "16.19"]
    pred.n_patches = 3
    assert pred.default_patch_id() == 2


def test_next_move_feeds_the_selected_patch_row():
    """A patch id must reach the model, not be silently dropped."""
    pred = stub(n_patches=4)
    pred.patch_vocab = ["16.18", "16.19", "16.20"]
    seen = {}
    inner = pred.model

    class Spy:
        encoder = inner.encoder

        def __call__(self, batch):
            seen["patch_id"] = int(batch["patch_id"][0])
            return inner(batch)

    pred.model = Spy()
    out = pred.next_move([None] * 10, [], first_pick=0, k=3, patch_id=3)
    assert seen["patch_id"] == 3
    assert out["patch_id"] == 3


def test_next_move_reports_the_unk_row_it_fell_back_to():
    pred = stub()
    out = pred.next_move([None] * 10, [], first_pick=0, k=3, patch_id=99)
    assert out["patch_id"] == infer.PATCH_UNK_ID


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
