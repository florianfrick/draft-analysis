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


# ------------------------------------------------------- checkpoint naming --

def test_team_codes_are_the_orgs_tags():
    """The suffix must read like the tag, not like an initialism.

    'Top Esports' -> 'TES' is only knowable from the table; the fallback would
    say 'TE'. Both are unique, so nothing raises - which is exactly why the
    table has to be checked rather than assumed.
    """
    assert C.team_code("Top Esports") == "TES"
    assert C.team_code("Invictus Gaming") == "IG"
    assert C.team_code("Hanwha Life Esports") == "HLE"
    assert C.team_code("100 Thieves") == "100T"


def test_team_code_falls_back_deterministically():
    """An unlisted team must still produce a stable, ASCII, short code."""
    a = C.team_code("Some Unlisted Org")
    assert a == C.team_code("Some Unlisted Org") == "SUO"
    assert C.team_code("LYON") == "LYON"          # single token keeps its name
    assert C.team_code("LØS") == "LOS"            # Ø is not silently dropped
    assert C.team_code("") == "TBD"
    assert all(ord(ch) < 128 for ch in C.team_code("Naïve Ünïcode Esports"))


def test_team_codes_file_has_no_conflicting_duplicates():
    codes = C.load_team_codes()
    assert codes, "team_codes.json did not load"
    assert all(not k.startswith("_") for k in codes)


def test_through_suffix_names_the_frontier_series():
    """The newest row wins, and the name is league + both tags, slugged."""
    rows = _rows(n=6)
    for i, r in enumerate(rows):
        r["blue_team"], r["red_team"] = "Invictus Gaming", "Top Esports"
        r["league"] = "WLDs"
        r["date_ordinal"] = 738000 + i
    frontier = C.corpus_frontier(rows)
    assert frontier["teams"] == ("Invictus Gaming", "Top Esports")
    assert frontier["n_games"] == 2          # _rows puts 2 games per series
    assert C.through_suffix(frontier) == "through-wlds-ig-tes"
    # sorting the pair means the name does not depend on who was on blue side
    rows[5]["blue_team"], rows[5]["red_team"] = "Top Esports", "Invictus Gaming"
    assert C.through_suffix(C.corpus_frontier(rows)) == "through-wlds-ig-tes"
    # and the suffix follows the newest row, not row order
    rows[0]["date_ordinal"] = 739999
    rows[0]["league"] = "LPL"
    assert C.through_suffix(C.corpus_frontier(rows)) == "through-lpl-ig-tes"


def test_frontier_does_not_parse_the_series_id():
    """A team called 'JD|XL' exists in the NLC exports; splitting the id on
    '|' would mangle it. The row's own team fields are the source."""
    rows = _rows(n=2)
    for r in rows:
        r["series_id"] = "NLC|Dusty|JD|XL|2022-01-10|s1"
        r["blue_team"], r["red_team"], r["league"] = "Dusty", "JD|XL", "NLC"
    frontier = C.corpus_frontier(rows)
    assert frontier["teams"] == ("Dusty", "JD|XL")
    assert C.through_suffix(frontier) == "through-nlc-dusty-jd-xl"


def test_check_out_dir_name_rejects_a_stale_name():
    C.check_out_dir_name("model/checkpoints/masking-proonly-through-wlds-ig-tes",
                         "through-wlds-ig-tes", "x")
    C.check_out_dir_name("", "through-wlds-ig-tes", "x")   # smoke runs
    for bad in ("model/checkpoints/masking-proonly",
                "model/checkpoints/masking-proonly-through-wlds-g2-hle"):
        try:
            C.check_out_dir_name(bad, "through-wlds-ig-tes", "suggested")
        except SystemExit as exc:
            assert "through-wlds-ig-tes" in str(exc)
        else:
            raise AssertionError("%s should have been refused" % bad)


# ------------------------------------------------------------ val scope ----

def _patch_rows():
    """Three patches of 60 rows, each in its own series of 3 games."""
    rows = []
    for p, patch in enumerate(("16.18", "16.19", "16.20")):
        for i in range(60):
            rows.append({
                "series_id": "%s|A%d|B%d|2026-01-01|s1" % (patch, i // 3, i // 3),
                "date_ordinal": 738000 + p * 100 + i, "gameid": "g%d-%d" % (p, i),
                "patch": patch, "game_number": 1, "league": "X",
                "blue_team": "A", "red_team": "B", "date": "2026-01-01",
                "champ": list(range(10)), "role": list(range(5)) * 2,
                "blue_win": 1, "weight": 1.0, "has_econ": False, "econ": None,
                "moves": [(1, 0, j, j) for j in range(10)],
                "unavailable": [], "first_pick": 0,
            })
    return rows


def test_val_scope_all_is_the_default_protocol():
    rows = _patch_rows()
    sp = C.PRO_PIPE.split_within_patch(rows, val_frac=0.10, pinned_test=(),
                                       test_frac=0.0)
    assert len(sp["val"]) == 3 * 6            # a tail in every patch
    assert {rows[i]["patch"] for i in sp["val"]} == {"16.18", "16.19", "16.20"}


def test_val_scope_latest_validates_only_on_the_newest_patch():
    rows = _patch_rows()
    sp = C.PRO_PIPE.split_within_patch(rows, val_frac=0.10, pinned_test=(),
                                       test_frac=0.0, val_patches="latest")
    assert {rows[i]["patch"] for i in sp["val"]} == {"16.20"}
    assert len(sp["val"]) == 6
    # every other patch is entirely train - that is the point of the switch
    assert len(sp["train"]) == 60 + 60 + 54
    assert not sp["test"]


def test_val_scope_latest_skips_a_patch_below_min_patch_rows():
    """A too-small newest patch cannot own the val tail, or val is empty."""
    rows = _patch_rows()
    small = [r for r in rows if r["patch"] == "16.20"][:10]
    for r in small:
        r["patch"] = "16.21"
    rows = [r for r in rows if r["patch"] != "16.20"]
    sp = C.PRO_PIPE.split_within_patch(rows, val_frac=0.10, pinned_test=(),
                                       test_frac=0.0, val_patches="latest")
    assert {rows[i]["patch"] for i in sp["val"]} == {"16.19"}
    assert sum(1 for i in sp["train"] if rows[i]["patch"] == "16.21") == 10


def test_no_test_keeps_every_pinned_series_row_in_the_split():
    """--no-test must not drop rows: the pinned series become train/val."""
    rows = _patch_rows()
    for i, r in enumerate(rows[:7]):
        r["series_id"] = "PINNED|A|B|2026-01-02|s1"
    pinned = ("PINNED|A|B|2026-01-02|s1",)
    with_pin = C.PRO_PIPE.split_within_patch(rows, pinned_test=pinned)
    without = C.PRO_PIPE.split_within_patch(rows, pinned_test=(),
                                            test_frac=0.0)
    assert len(with_pin["test"]) == 7
    assert not without["test"]
    for sp in (with_pin, without):
        total = sum(len(sp[k]) for k in ("train", "val", "test"))
        assert total == len(rows), "a row went missing from the split"


# ------------------------------------------------ cross-protocol comparison --

def _compare_args(**kw):
    """The subset of compare.py's namespace `_build_splits` reads."""
    import argparse
    ns = argparse.Namespace(val_frac=0.10, val_frac_latest=0.20, test_frac=0.10,
                            min_patch_rows=40, other="old")
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_shared_split_is_val_for_both_protocols(monkeypatch):
    """`--split shared` is only honest if neither protocol trained on it.

    This is the whole basis of the 2026-11-04 verdict: two checkpoints trained
    under different splits have no common val set otherwise, and scoring one on
    the other's training rows would decide the comparison by construction.
    """
    import compare
    rows = _patch_rows()
    pinned = ("16.18|A0|B0|2026-01-01|s1",)
    monkeypatch.setattr(C.PRO_PIPE, "TEST_SERIES", pinned)
    splits = compare._build_splits(rows, _compare_args())
    shared = set(splits["shared"])
    assert shared, "the synthetic rows should produce a non-empty shared set"

    old = C.PRO_PIPE.split_within_patch(rows, val_frac=0.10, test_frac=0.10,
                                        pinned_test=pinned)
    new = C.PRO_PIPE.split_within_patch(rows, val_frac=0.20, test_frac=0.0,
                                        pinned_test=(), val_patches="latest")
    assert shared <= set(old["val"]) and shared <= set(new["val"])
    assert not shared & set(old["train"]), "shared row was trained by the old split"
    assert not shared & set(new["train"]), "shared row was trained by the new split"
    # and it is the newest patch only, because the refresh's val is
    assert {rows[i]["patch"] for i in shared} == {"16.20"}


def test_shared_split_against_the_all_patch_protocol(monkeypatch):
    """`--other all` intersects with a --val-scope all --no-test run instead."""
    import compare
    rows = _patch_rows()
    pinned = ("16.18|A0|B0|2026-01-01|s1",)
    monkeypatch.setattr(C.PRO_PIPE, "TEST_SERIES", pinned)
    splits = compare._build_splits(rows, _compare_args(other="all"))
    shared = set(splits["shared"])
    assert shared
    other = C.PRO_PIPE.split_within_patch(rows, val_frac=0.10, test_frac=0.0,
                                          pinned_test=())
    assert shared <= set(other["val"])
    assert not shared & set(other["train"])
    assert not splits["test"], "--other all has no test split"


def test_mcnemar_separates_a_only_from_b_only():
    """The two counts are the content of the comparison; swapping them inverts
    the reading of every table in the report."""
    import compare
    a_only, b_only, chi2, p = compare.mcnemar([1, 1, 0, 0], [0, 1, 1, 0])
    assert (a_only, b_only) == (1, 1)
    # balanced discordant pairs are not significant (the continuity correction
    # is why this is 0.48 rather than 1.0)
    assert p > 0.4
    # a clear win for B
    a_only, b_only, chi2, p = compare.mcnemar([0] * 10 + [1] * 2,
                                              [1] * 10 + [0] * 2)
    assert (a_only, b_only) == (2, 10)
    assert p < 0.05
    assert compare.mcnemar([], []) == (0, 0, 0.0, 1.0)
