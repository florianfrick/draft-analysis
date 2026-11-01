"""Tests for the pro-only experiment plumbing (branch exp/pro-only-patch-emb).

What these pin, and why each one would otherwise fail silently:

1. **The masked dataset actually carries the patch id.** A patch embedding fed
   a constant (or nothing) trains fine and shows up as "patch embedding: no
   effect" - which is the null result this branch is trying to measure, so a
   wiring bug is indistinguishable from the finding.
2. **A checkpoint rebuilds the encoder it was trained with.** The config in the
   checkpoint is the only record of `n_patches`; if it is not written, the
   policy stage silently builds a 0-row table and `load_encoder_state` raises
   (good) or the table is missing (bad).
3. **The split is the one both stages use.** The masked stage and the policy
   stage must not disagree about which rows are test, or the two halves of the
   experiment are measured on different data.
4. **The baseline-relative score is the number selection uses**, so it must be
   model_top5 / base_top5 and not something that silently becomes 0/0.
"""
import importlib.util
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402


def _rows(n=40, patches=("16.01", "16.02")):
    """Synthetic pro rows: just enough for ProMaskedDraftDataset."""
    rows = []
    for i in range(n):
        champ = [(i * 3 + j) % 173 for j in range(10)]
        rows.append({
            "champ": champ, "role": list(range(5)) * 2,
            "blue_win": i % 2, "weight": 1.0, "has_econ": False,
            "econ": None, "gameid": "g%d" % i, "series_id": "s%d" % (i // 2),
            "league": "LPL", "date": "2026-01-0%d" % (1 + i % 9),
            "date_ordinal": 738000 + i, "game_number": 1,
            "patch": patches[i % len(patches)],
            "moves": [(1, 0, champ[j], j) for j in range(10)],
            "unavailable": [], "first_pick": 0,
        })
    return rows


def test_masked_dataset_carries_the_patch_id():
    rows = _rows()
    vocab, index = C.PRO_DATA.build_patch_vocab(rows)
    ds = C.PRO_DATA.ProMaskedDraftDataset(
        rows, num_champs=173, seed=0, epoch=0, patch_index=index)
    seen = {int(ds[i]["patch_id"]) for i in range(len(rows))}
    assert seen == {index[p] for p in vocab}
    assert 0 not in seen, "no synthetic row should fall back to UNK"
    # without a vocabulary every row is UNK, not a crash
    ds_none = C.PRO_DATA.ProMaskedDraftDataset(rows, num_champs=173, seed=0)
    assert {int(ds_none[i]["patch_id"]) for i in range(len(rows))} == {0}


def test_patch_id_reaches_the_encoder():
    """The batch key the dataset emits must be the key `_encode` reads."""
    rows = _rows()
    vocab, index = C.PRO_DATA.build_patch_vocab(rows)
    ds = C.PRO_DATA.ProMaskedDraftDataset(rows, num_champs=173, seed=0,
                                          patch_index=index)
    batch = C.PRO_DATA.pro_collate([ds[0], ds[1]])
    assert "patch_id" in batch and batch["patch_id"].dtype == torch.long
    enc = C.build_encoder(173, 32, 1, 1, 4, 0.0, len(vocab) + 1).eval()
    torch.nn.init.normal_(enc.embed.patch_lookup.weight, std=0.3)
    out = enc(batch)
    # -inf is the intentional duplicate mask; NaN is never acceptable
    assert not torch.isnan(out["champ_logits"]).any()
    batch2 = dict(batch, patch_id=(batch["patch_id"] + 1) % (len(vocab) + 1))
    assert not torch.allclose(out["champ_logits"], enc(batch2)["champ_logits"])


def test_encoder_round_trips_through_a_checkpoint(tmp_path):
    cfg = {"num_champs": 173, "embed_dim": 32, "num_intra": 1, "num_cross": 1,
           "num_heads": 4, "dropout": 0.0, "n_patches": 5}
    enc = C.build_encoder(173, 32, 1, 1, 4, 0.0, 5)
    torch.nn.init.normal_(enc.embed.patch_lookup.weight, std=0.2)
    path = tmp_path / "ck.pt"
    torch.save({"model": enc.state_dict(), "config": cfg}, str(path))
    rebuilt, got_cfg, info = C.encoder_from_checkpoint(path, n_patches=None)
    assert got_cfg["n_patches"] == 5
    assert info["missing"] == [] and info["dropped"] == []
    batch = {"champ_ids": torch.zeros(2, 10, dtype=torch.long),
             "role_ids": torch.zeros(2, 10, dtype=torch.long),
             "patch_id": torch.tensor([1, 3]),
             "is_masked": torch.zeros(2, 10, dtype=torch.bool),
             "pad_mask": torch.zeros(2, 10, dtype=torch.bool)}
    assert torch.allclose(enc(batch)["champ_logits"],
                          rebuilt(batch)["champ_logits"])


def test_policy_model_accepts_a_patch_encoder():
    enc = C.build_encoder(173, 32, 1, 1, 4, 0.0, 5)
    model = C.PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                               n_patches=5, with_value=False)
    b = 3
    batch = {
        "champ_ids": torch.zeros(b, 10, dtype=torch.long),
        "role_ids": torch.full((b, 10), 5, dtype=torch.long),
        "pad_mask": torch.ones(b, 10, dtype=torch.bool),
        "ban_ids": torch.full((b, 10), 174, dtype=torch.long),
        "taken_mask": torch.zeros(b, 173, dtype=torch.bool),
        "unavailable_mask": torch.zeros(b, 173, dtype=torch.bool),
        "is_ban": torch.zeros(b, dtype=torch.long),
        "move_idx": torch.zeros(b, dtype=torch.long),
        "phase": torch.zeros(b, dtype=torch.long),
        "to_move": torch.zeros(b, dtype=torch.long),
        "first_pick": torch.zeros(b, dtype=torch.long),
        "patch_id": torch.tensor([0, 1, 4]),
    }
    out = model(batch)
    assert out["policy_logits"].shape == (b, 173)
    assert torch.isfinite(out["policy_logits"]).all()


def test_split_is_the_shared_one():
    """Both trainers must resolve the split through the same function."""
    import train_masked
    import train_policy
    for mod in (train_masked, train_policy):
        assert mod.C is C
        assert mod.PRO_PIPE.split_within_patch is C.PRO_PIPE.split_within_patch


def test_slice_accumulator_reports_relative_scores():
    acc = C.SliceAccumulator()
    acc.add([0, 3], ["16.01", "16.01"], [0, 0])
    acc.add([7], ["16.02"], [0])
    rep = acc.report()
    p1 = rep["by_patch"]["16.01"]
    assert p1["n"] == 2 and p1["top5"] == 1.0 and p1["base_top5"] == 1.0
    assert abs(p1["rel_top5"] - 1.0) < 1e-9
    p2 = rep["by_patch"]["16.02"]
    assert p2["top5"] == 0.0 and p2["base_top5"] == 1.0
    assert p2["rel_top5"] == 0.0
    assert set(rep["by_year"]) == {"16"}
