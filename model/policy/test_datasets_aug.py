"""Tests for the pro next-move dataset: patch ids, prefix modes, and
blue/red swap augmentation.

Run: pytest model/policy/test_datasets_aug.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import train as T  # noqa: E402  (gives us T.PRO_DATA / T.PRO_PIPE)
from model import PolicyValueModel  # noqa: E402
from soloq_model import SoloQPretrainedModel, NUM_REAL_CHAMPS  # noqa: E402

PRO_DATA = T.PRO_DATA
PRO_PIPE = T.PRO_PIPE
C = NUM_REAL_CHAMPS
N_MOVES = 20
N_SLOTS = 10


def make_row(first_pick=0, seed=0, patch="16.03", blue_win=1):
    """A valid 20-move pro draft row, built through the real pipeline."""
    g = torch.Generator().manual_seed(seed)
    champs = torch.randperm(C, generator=g)[:20].tolist()
    bans_by_team = {0: champs[0:5], 1: champs[5:10]}
    picks_by_team = {0: [(champs[10 + i], i) for i in range(5)],
                     1: [(champs[15 + i], i) for i in range(5)]}
    moves = PRO_PIPE.interleaved_moves(first_pick, bans_by_team,
                                       picks_by_team)
    assert len(moves) == N_MOVES
    return {
        "moves": moves, "first_pick": first_pick, "unavailable": [],
        "blue_win": blue_win, "weight": 1.0, "patch": patch,
        "econ": [1500.0, 1400.0, 2400.0, 2300.0, 9000.0, 8800.0],
        "has_econ": True, "gameid": "g%d" % seed, "series_id": "s%d" % seed,
        "league": "LCK", "date": "2026-03-01",
        "blue_team": "T1", "red_team": "GEN",
    }


# --------------------------------------------------------------- patches

def test_patch_key_normalizes():
    assert PRO_DATA.patch_key("16.3") == "16.03"
    assert PRO_DATA.patch_key("16.03") == "16.03"
    assert PRO_DATA.patch_key(16.3) == "16.03"
    assert PRO_DATA.patch_key("") == ""
    assert PRO_DATA.patch_key(None) == ""


def test_patch_vocab_reserves_unk_and_is_sorted():
    rows = [make_row(patch="16.03", seed=1),
            make_row(patch="16.2", seed=2),
            make_row(patch="15.09", seed=3)]
    vocab, index = PRO_DATA.build_patch_vocab(rows)
    assert vocab == ["15.09", "16.02", "16.03"]
    # index 0 is reserved for UNK, real patches start at 1
    assert index["15.09"] == 1 and index["16.03"] == 3
    assert PRO_DATA.PATCH_UNK_ID == 0


def test_patch_id_in_prefix_and_unk_fallback():
    rows = [make_row(patch="16.03", seed=1)]
    _, index = PRO_DATA.build_patch_vocab(rows)
    s = PRO_DATA.build_prefix(rows[0], 0, patch_index=index)
    assert int(s["patch_id"]) == index["16.03"]
    # a patch never seen in the vocab -> UNK, not a crash
    other = make_row(patch="17.01", seed=2)
    s2 = PRO_DATA.build_prefix(other, 0, patch_index=index)
    assert int(s2["patch_id"]) == PRO_DATA.PATCH_UNK_ID
    # no vocab at all -> UNK
    s3 = PRO_DATA.build_prefix(rows[0], 0)
    assert int(s3["patch_id"]) == PRO_DATA.PATCH_UNK_ID


def test_patch_index_from_vocab_roundtrip():
    rows = [make_row(patch="16.03", seed=1), make_row(patch="15.09", seed=2)]
    vocab, index = PRO_DATA.build_patch_vocab(rows)
    rebuilt = PRO_DATA.patch_index_from_vocab(vocab)
    assert rebuilt == index


# ------------------------------------------------------- swap augmentation

def test_swap_sides_is_involution():
    rows = [make_row(first_pick=0, seed=7)]
    s = PRO_DATA.build_prefix(rows[0], 12)
    back = PRO_DATA.swap_sides(PRO_DATA.swap_sides(s))
    for key in ("champ_ids", "role_ids", "pad_mask", "ban_ids",
                "to_move", "first_pick", "win_label", "move_idx",
                "phase", "target_champ", "patch_id"):
        assert torch.equal(back[key], s[key]), key


def test_swap_sides_swaps_team_blocks_and_flips_labels():
    rows = [make_row(first_pick=0, seed=7, blue_win=1)]
    s = PRO_DATA.build_prefix(rows[0], 12)  # 6 bans + 6 picks
    w = PRO_DATA.swap_sides(s)
    half = 5
    assert torch.equal(w["champ_ids"][:half], s["champ_ids"][half:])
    assert torch.equal(w["champ_ids"][half:], s["champ_ids"][:half])
    assert torch.equal(w["role_ids"][:half], s["role_ids"][half:])
    assert torch.equal(w["pad_mask"][:half], s["pad_mask"][half:])
    assert int(w["to_move"]) == 1 - int(s["to_move"])
    assert int(w["first_pick"]) == 1 - int(s["first_pick"])
    assert int(w["win_label"]) == 1 - int(s["win_label"])
    # the move being predicted does not change
    assert int(w["target_champ"]) == int(s["target_champ"])
    assert int(w["is_ban"]) == int(s["is_ban"])


def test_swap_sides_leaves_team_agnostic_fields_alone():
    rows = [make_row(first_pick=0, seed=7)]
    s = PRO_DATA.build_prefix(rows[0], 12)
    w = PRO_DATA.swap_sides(s)
    # BanContext embeds bans with no team or position signal, so the ban
    # list is already the correct mirrored representation.
    assert torch.equal(w["ban_ids"], s["ban_ids"])
    assert torch.equal(w["taken_mask"], s["taken_mask"])
    assert torch.equal(w["unavailable_mask"], s["unavailable_mask"])
    assert int(w["move_idx"]) == int(s["move_idx"])


def test_swap_sides_flips_econ_team_pairs():
    """econ_target is (g10_b, g10_r, g15_b, g15_r, xp10_b, xp10_r)."""
    rows = [make_row(seed=7)]
    s = PRO_DATA.build_prefix(rows[0], 12)
    w = PRO_DATA.swap_sides(s)
    assert w["econ_target"].tolist() == pytest.approx(
        [s["econ_target"][1], s["econ_target"][0],
         s["econ_target"][3], s["econ_target"][2],
         s["econ_target"][5], s["econ_target"][4]])
    assert torch.equal(PRO_DATA.swap_sides(w)["econ_target"],
                       s["econ_target"])


def test_swap_preserves_draft_and_legality():
    """Same champions on the board, and the target is still legal."""
    rows = [make_row(first_pick=1, seed=11)]
    s = PRO_DATA.build_prefix(rows[0], 9)
    w = PRO_DATA.swap_sides(s)
    picked = lambda d: set(d["champ_ids"][~d["pad_mask"]].tolist())
    banned = lambda d: set(d["ban_ids"][d["ban_ids"] < C].tolist())
    assert picked(w) == picked(s)
    assert banned(w) == banned(s)
    assert not bool(w["taken_mask"][w["target_champ"]])


def test_swap_batch_forward_is_finite():
    """A mirrored batch must produce finite policy/value output."""
    rows = [make_row(first_pick=0, seed=3)]
    samples = [PRO_DATA.swap_sides(PRO_DATA.build_prefix(rows[0], m))
               for m in (0, 6, 12, 19)]
    batch = PRO_DATA.pro_collate(samples)
    enc = SoloQPretrainedModel(num_champs=C, embed_dim=32, num_heads=4,
                               num_intra_layers=1, num_cross_layers=1)
    m = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=6)
    out = m(batch)
    assert torch.isfinite(out["policy_logits"]).all()
    assert torch.isfinite(out["value_logit"]).all()


# ----------------------------------------------------------- prefix modes

def test_prefix_all_mode_matches_old_layout():
    rows = [make_row(seed=i) for i in range(4)]
    ds = PRO_DATA.ProNextMoveDataset(rows, prefix_mode="all")
    assert len(ds) == 4 * N_MOVES
    # deterministic (game, move) mapping
    for idx in (0, 1, 19, 20, 79):
        assert int(ds[idx]["move_idx"]) == idx % N_MOVES
    assert len(ds.weights()) == len(ds)


def test_random_prefix_one_example_per_game():
    rows = [make_row(seed=i) for i in range(16)]
    ds = PRO_DATA.ProNextMoveDataset(rows, prefix_mode="random")
    assert len(ds) == len(rows)
    assert len(ds.weights()) == len(rows)
    # default epoch assignment is already populated
    assert ds._move_of is not None and ds._move_of.numel() == len(rows)


def test_random_prefix_is_deterministic_per_epoch():
    rows = [make_row(seed=i) for i in range(32)]
    a = PRO_DATA.ProNextMoveDataset(rows, prefix_mode="random", seed=5)
    b = PRO_DATA.ProNextMoveDataset(rows, prefix_mode="random", seed=5)
    a.set_epoch(3)
    b.set_epoch(3)
    assert torch.equal(a._move_of, b._move_of)
    idx_a = [int(a[i]["move_idx"]) for i in range(len(rows))]
    idx_b = [int(b[i]["move_idx"]) for i in range(len(rows))]
    assert idx_a == idx_b


def test_random_prefix_changes_across_epochs_and_covers_all_moves():
    rows = [make_row(seed=i) for i in range(200)]
    ds = PRO_DATA.ProNextMoveDataset(rows, prefix_mode="random", seed=0)
    seen = set()
    per_epoch = []
    for epoch in range(6):
        ds.set_epoch(epoch)
        moves = [int(ds[i]["move_idx"]) for i in range(len(rows))]
        per_epoch.append(moves)
        seen.update(moves)
    assert seen == set(range(N_MOVES)), "not every move index was sampled"
    # consecutive epochs must not be identical assignments
    assert per_epoch[0] != per_epoch[1]


def test_bad_prefix_mode_rejected():
    with pytest.raises(ValueError):
        PRO_DATA.ProNextMoveDataset([make_row()], prefix_mode="nope")


def test_swap_aug_only_in_train_mode():
    rows = [make_row(seed=i) for i in range(40)]
    ds = PRO_DATA.ProNextMoveDataset(rows, prefix_mode="random",
                                     swap_aug=True, seed=0)
    flipped = 0
    for i in range(len(rows)):
        s = ds[i]
        # base row has first_pick=0, so a swap shows up as first_pick=1
        flipped += int(s["first_pick"]) == 1
    assert 0 < flipped < len(rows), "swap augmentation not applied ~50/50"


# ------------------------------------------------------ model integration

def test_patch_embedding_changes_logits():
    """Patch conditioning must actually reach the head."""
    rows = [make_row(patch="16.03", seed=1)]
    _, index = PRO_DATA.build_patch_vocab(rows)
    samples = [PRO_DATA.build_prefix(rows[0], 0, patch_index=index)]
    batch = PRO_DATA.pro_collate(samples)
    enc = SoloQPretrainedModel(num_champs=C, embed_dim=32, num_heads=4,
                               num_intra_layers=1, num_cross_layers=1)
    m = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                         n_patches=4).eval()
    with torch.no_grad():
        a = m(batch)["policy_logits"]
        batch2 = dict(batch)
        batch2["patch_id"] = torch.tensor([3], dtype=torch.long)
        b = m(batch2)["policy_logits"]
    assert not torch.allclose(a, b), "patch id had no effect on logits"


def test_model_without_patch_ignores_patch_id():
    """n_patches=0 keeps old checkpoints loadable and patch-free."""
    rows = [make_row(seed=1)]
    samples = [PRO_DATA.build_prefix(rows[0], 0)]
    batch = PRO_DATA.pro_collate(samples)
    enc = SoloQPretrainedModel(num_champs=C, embed_dim=32, num_heads=4,
                               num_intra_layers=1, num_cross_layers=1)
    m = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                         n_patches=0).eval()
    assert m.state_embed.patch_embed is None
    with torch.no_grad():
        a = m(batch)["policy_logits"]
        batch["patch_id"] = torch.tensor([7], dtype=torch.long)
        b = m(batch)["policy_logits"]
    assert torch.allclose(a, b)
