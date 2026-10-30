"""Tests for the shared eval protocol.

The protocol exists because a pooled number hid two different effects (window
difficulty and real generalisation loss) plus two broken heads. So the tests
pin down exactly the pieces that make it honest: the patch-honest filter, the
confidence intervals, and the slicing.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_protocol as E  # noqa: E402


# ------------------------------------------------------------------- leagues
def test_league_tier_is_explicit_and_defaults_to_other():
    assert E.league_tier("LPL") == "major"
    assert E.league_tier("LCK") == "major"
    assert E.league_tier("WLDs") == "international"
    assert E.league_tier("MSI") == "international"
    assert E.league_tier("LAS") == "academy"
    assert E.league_tier("LCKC") == "academy"
    # unlisted labels must NOT be guessed into a tier
    assert E.league_tier("SomeNewLeague") == "other"
    assert E.league_tier(None) == "other"
    assert E.league_tier("  ") == "other"


# --------------------------------------------------------------- patch-honest
def _rows(patches):
    return [{"patch": p} for p in patches]


def test_patch_honest_val_drops_patches_seen_in_train():
    rows = _rows(["16.07"] * 2 + ["16.09"] * 300 + ["16.10"] * 300)
    splits = {"train": [0, 1], "val": [2, 3, 4], "test": []}
    kept, info = E.patch_honest_val(rows, splits, min_rows=1, verbose=False)
    assert kept == [2, 3, 4]                 # 16.09/16.10 are novel
    assert info == {"honest": True, "kept": 3, "dropped": 0}

    splits2 = {"train": [0, 1], "val": [0, 1, 2, 3], "test": []}
    kept, info = E.patch_honest_val(rows, splits2, min_rows=1, verbose=False)
    assert kept == [2, 3]                    # the 16.07 rows are dropped
    assert info["dropped"] == 2 and info["honest"] is True


def test_patch_honest_val_falls_back_rather_than_emptying_the_window():
    """An empty val is worse than a slightly optimistic one, so the filter
    must degrade loudly instead of silently returning nothing."""
    rows = _rows(["16.07"] * 4)
    splits = {"train": [0], "val": [0, 1, 2, 3], "test": []}
    kept, info = E.patch_honest_val(rows, splits, min_rows=10, verbose=False)
    assert kept == [0, 1, 2, 3]              # full window kept
    assert info["honest"] is False
    assert info["dropped"] == 0


# ---------------------------------------------------------------- statistics
def test_wilson_matches_known_values():
    lo, hi = E.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)
    # symmetric at p=0.5, and wider for smaller n
    lo2, hi2 = E.wilson(50, 100)
    assert (hi2 - lo2) < (hi - lo)
    assert lo2 == pytest.approx(1 - hi2, abs=1e-9)
    # degenerate inputs must not raise or produce nan
    assert E.wilson(0, 0) == (0.0, 1.0)
    lo0, hi0 = E.wilson(0, 20)
    assert lo0 == 0.0 and 0 < hi0 < 0.2


def test_auc_handles_ties_and_degenerate_splits():
    assert E.auc([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert E.auc([1, 2, 3, 4], [1, 1, 0, 0]) == pytest.approx(0.0)
    # all scores tied -> exactly chance, not 0 or 1
    assert E.auc([5, 5, 5, 5], [0, 1, 0, 1]) == pytest.approx(0.5)
    # one class only -> chance by definition
    assert E.auc([1, 2, 3], [1, 1, 1]) == pytest.approx(0.5)
    # a single swapped pair out of 4*4=16
    assert E.auc([1, 2, 3, 4], [0, 1, 0, 1]) == pytest.approx(0.75)


def test_baseline_relative():
    assert E.baseline_relative(0.30, 0.10) == pytest.approx(3.0)
    assert math.isnan(E.baseline_relative(0.30, 0.0))


def test_move_popularity_ranks_are_per_move_index():
    """Move 0 and move 1 have different popular champions, and the baseline
    must reflect that - a single global pick list would be a weaker (and
    wrong) denominator for baseline-relative scoring."""
    rows = [
        {"weight": 1.0, "moves": [(0, 0, 5, -1), (1, 0, 7, 0)]},
        {"weight": 1.0, "moves": [(0, 1, 5, -1), (1, 1, 8, 0)]},
        {"weight": 1.0, "moves": [(0, 1, 6, -1), (1, 1, 8, 1)]},
    ]
    ranks = E.move_popularity_ranks(rows, [0, 1, 2], num_champs=9, n_moves=2)
    # move 0: champ 5 seen twice, 6 once -> 5 is rank 0, 6 is rank 1
    assert ranks[0][5] == 0
    assert ranks[0][6] == 1
    # move 1: champ 8 seen twice, 7 once
    assert ranks[1][8] == 0
    assert ranks[1][7] == 1
    # unseen champions sort after seen ones, by id
    assert ranks[0][0] == 2


# ------------------------------------------------------------------- reports
def test_self_popularity_topk_is_a_same_rows_reference():
    """The same-patch table is the bar a model must clear on a NEW patch,
    because the train-era table is stale there."""
    rows = [
        {"champ": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"champ": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"champ": [11, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    ]
    out = E.self_popularity_topk(rows, [0, 1, 2], num_champs=12, k=1)
    # slot 0: champ 1 twice, 11 once -> champ 1 is rank 0 -> 2 of 3 hits there
    # every other slot has one champion repeated 3x -> always rank 0
    assert out["n"] == 30
    assert out["top1"] == pytest.approx(29 / 30)

    # next-move style: index by move position
    def get(row):
        return list(enumerate(row["champ"]))
    out2 = E.self_popularity_topk(rows, [0], num_champs=12, k=5, get=get)
    assert out2["n"] == 10 and out2["top5"] == pytest.approx(1.0)


def test_rank_report_counts_and_slices():
    rep = E.RankReport()
    # batch 1: patch 16.20, major league, move 0, pick, ranks 0 and 4
    rep.add([0, 4], "16.20", ["LPL", "LPL"], [0, 0], ["pick", "pick"])
    # batch 2: patch 16.19, academy, move 1, ban, ranks 2 and 9
    rep.add([2, 9], "16.19", ["LAS", "LAS"], [1, 1], ["ban", "ban"])
    assert len(rep) == 4

    allm = rep.metrics()
    assert allm["n"] == 4
    assert allm["top1"] == pytest.approx(0.25)   # only rank 0
    assert allm["top5"] == pytest.approx(0.75)   # ranks 0,4,2
    assert allm["mean_rank"] == pytest.approx(15 / 4)

    by_patch = rep.by("patch", order="natural")
    assert list(by_patch) == ["16.19", "16.20"]
    assert by_patch["16.20"]["top5"] == pytest.approx(1.0)   # ranks 0,4
    assert by_patch["16.19"]["top5"] == pytest.approx(0.5)   # rank 2 only

    by_tier = rep.by("tier")
    assert by_tier["major"]["n"] == 2
    assert by_tier["academy"]["n"] == 2

    by_kind = rep.by("kind")
    assert by_kind["pick"]["top5"] == pytest.approx(1.0)
    assert by_kind["ban"]["top5"] == pytest.approx(0.5)

    # min_n filters thin slices out instead of reporting them as if solid
    assert rep.by("patch", min_n=3) == {}


def test_rank_report_broadcasts_scalars_and_validates_length():
    rep = E.RankReport(ks=(1, 5, 10))
    rep.add([3, 7], patch="16.20", league="LPL", idx=2, kind="mask")
    m = rep.metrics()
    assert m["n"] == 2 and m["top5"] == pytest.approx(0.5)
    assert m["top10"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rep.add([1, 2], patch=["a"], league="x", idx=0, kind="k")


def test_rank_report_ci_shrinks_with_n():
    small = E.RankReport()
    small.add([0] * 10, "16.20", "LPL", 0, "pick")
    big = E.RankReport()
    big.add([0] * 1000, "16.20", "LPL", 0, "pick")
    lo_s, hi_s = small.metrics()["ci1"]
    lo_b, hi_b = big.metrics()["ci1"]
    assert (hi_s - lo_s) > (hi_b - lo_b)


def test_format_table_renders_every_slice():
    rep = E.RankReport()
    rep.add([0, 1], "16.20", ["LPL", "LPL"], [0, 1], ["pick", "pick"])
    rep.add([3, 6], "16.19", ["LAS", "LAS"], [0, 1], ["pick", "pick"])
    text = E.format_table(rep.by("patch", order="natural"), title="by patch")
    assert "by patch" in text
    assert "16.19" in text and "16.20" in text
    assert "95% CI" in text
    assert text.count("\n") >= 3


# ------------------------------------------------------- stratified report
def test_self_popularity_topk_accepts_several_ks_in_one_pass():
    """One pass must serve both the top-1 and top-5 reference, so the focus
    block does not have to rebuild the same table twice."""
    rows = [{"champ": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            {"champ": [11, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    out = E.self_popularity_topk(rows, [0, 1], num_champs=12, k=(1, 5))
    assert out["n"] == 20
    # slot 0 is a 2-way tie, so after leave-one-out the other champion is
    # strictly more frequent -> rank 1 (not 0) for both of its examples.
    # Every other slot repeats 2x -> rank 0.
    assert out["top1"] == pytest.approx(18 / 20)
    assert out["top5"] == pytest.approx(1.0)


def test_rank_report_where_restricts_to_one_value():
    rep = E.RankReport()
    rep.add([0, 4], "16.20", ["LPL", "LPL"], [0, 0], ["pick", "pick"])
    rep.add([2, 9], "16.19", ["LAS", "LAS"], [1, 1], ["ban", "ban"])
    m = rep.where("patch", "16.20")
    assert m["n"] == 2 and m["top5"] == pytest.approx(1.0)
    assert rep.where("patch", "16.99")["n"] == 0


def test_focus_block_reports_both_references_separately():
    """The two denominators answer different questions and must not be
    conflated: `rel` (train table) flatters a model on a new patch, `rel_self`
    (same-patch table) does not."""
    model = E.RankReport()
    model.add([0, 1, 2, 3], "16.20", "LPL", 0, "pick")
    train_base = E.RankReport()
    train_base.add([9, 9, 9, 9], "16.20", "LPL", 0, "pick")   # top5 = 0
    f = E.focus_block(model, "16.20", base_rep=train_base,
                      self_ref={"top5": 0.5})
    assert f["n"] == 4
    assert f["top5"] == pytest.approx(1.0)
    assert f["base"] == pytest.approx(0.0)
    assert f["rel"] is None            # 1.0 / 0.0 is undefined, not inf
    assert f["rel_self"] == pytest.approx(2.0)
    # absent patch degrades to n=0 instead of raising
    assert E.focus_block(model, "16.99") == {"patch": "16.99", "n": 0}


def test_stratified_is_json_serialisable_and_orders_patches_numerically():
    import json
    rep = E.RankReport()
    rep.add([0, 1], "16.9", ["LPL", "LPL"], [0, 1], ["pick", "pick"])
    rep.add([2, 3], "16.10", ["LAS", "LAS"], [0, 1], ["ban", "ban"])
    rep.add([0, 4], "16.20", ["WLDs", "WLDs"], [2, 3], ["pick", "pick"])
    out = E.stratified(rep, focus="16.20")
    json.dumps(out)                      # must not raise
    assert out["overall"]["n"] == 6
    # 16.9 sorts before 16.10 numerically, which string sort would invert
    assert list(out["by_patch"]) == ["16.9", "16.10", "16.20"]
    assert list(out["by_tier"]) == ["major", "international", "academy"]
    assert out["focus"]["patch"] == "16.20"
    assert out["focus"]["n"] == 2


def test_stratified_keeps_move_index_in_numeric_order():
    """`idx` is a move/slot index. Insertion order is not reliable (the
    composition evaluator sees masked slots in draft order, e.g. 2,1,3,6,...)
    and string-sorting the keys would put move 10 between 1 and 2."""
    rep = E.RankReport()
    for m in (2, 0, 10, 1, 11, 3, 4, 5, 6, 7, 8, 9):
        rep.add([m % 7], "16.20", "LPL", m, "pick")
    out = E.stratified(rep, slices=("idx",))
    assert list(out["by_idx"]) == [str(m) for m in range(12)]



def test_format_stratified_prints_the_focus_line():
    rep = E.RankReport()
    rep.add([0, 1, 2, 3], "16.20", "LPL", 0, "pick")
    text = E.format_stratified(E.stratified(rep, focus="16.20"))
    assert "overall n=4" in text
    assert "focus patch 16.20" in text
    assert "same-patch rel" in text

