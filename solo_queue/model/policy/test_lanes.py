"""Tests for the champion -> lane table and the assignment built on it.

The table is a *measured* artefact (`data/build_champ_roles.py`), so the tests
that matter are the ones that would fail if it were rebuilt wrong or the
assignment stopped using it: a champion's known lanes, injectivity, and the
agreement rate with real pro drafts.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lanes as L  # noqa: E402


def _corpus():
    """Pro draft rows, or None when the corpus is not on this machine."""
    path = ROOT / "data" / "pro play" / "pipeline.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("propipeline_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["propipeline_test"] = mod
    spec.loader.exec_module(mod)
    rows, _stats = mod.build_rows()
    return mod, rows


def test_the_table_covers_every_champion():
    t = L.table()
    assert t["roles"] == ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    assert len(t["counts"]) == 173
    for champ, counts in t["counts"].items():
        assert len(counts) == 5, champ
        assert all(c >= 0 for c in counts), champ
        assert sum(counts) > 0, "champion %d has no picks at all" % champ


def test_a_single_lane_champion_is_only_legal_there():
    """The claim the constraint rests on, checked by name.

    Nautilus is a support in 99 % of his pro games and K'Sante a top in 98 %,
    so a lane-filtered list must not offer either anywhere else.
    """
    corpus = _corpus()
    if corpus is None:
        pytest.skip("no corpus on this machine")
    mod, _rows = corpus
    name_to_idx, _ = mod.load_name_to_idx()

    support = L.legal_champions(4)
    top = L.legal_champions(0)
    for name, lane, pool in (("Nautilus", 4, support), ("K'Sante", 0, top),
                             ("Ezreal", 3, L.legal_champions(3)),
                             ("Ahri", 2, L.legal_champions(2))):
        idx = name_to_idx[name]
        assert idx in pool, "%s should be legal in its lane" % name
        for other in range(5):
            if other == lane:
                continue
            assert idx not in L.legal_champions(other), \
                "%s should not be legal in %s" % (name, L.lane_name(other))


def test_a_flex_champion_is_legal_in_both_lanes():
    corpus = _corpus()
    if corpus is None:
        pytest.skip("no corpus on this machine")
    mod, _rows = corpus
    idx = mod.load_name_to_idx()[0]["Tristana"]   # 68 % mid, 30 % bot
    assert idx in L.legal_champions(2)
    assert idx in L.legal_champions(3)


def test_empty_slots_are_flex_and_filled_ones_are_distinct():
    picks = [None] * 10
    assert L.assign_lanes(picks) == [None] * 10

    # A real board: five blue picks and five red, all distinct champions.
    corpus = _corpus()
    if corpus is None:
        pytest.skip("no corpus on this machine")
    _mod, rows = corpus
    picks = list(rows[-1]["champ"])
    lanes = L.assign_lanes(picks)
    assert len(lanes) == 10
    for team in (0, 1):
        got = lanes[team * 5:(team + 1) * 5]
        assert sorted(got) == [0, 1, 2, 3, 4], \
            "a team's five lanes must be a permutation, got %s" % got


def test_two_champions_that_share_a_lane_are_split_up():
    """Injective, not per-champion argmax: the second one moves.

    Two junglers cannot both be the jungler. Which one moves is decided by the
    summed likelihood, so the single-lane champion keeps its lane and the flex
    one is displaced - that is the whole reason for the exhaustive search.
    """
    corpus = _corpus()
    if corpus is None:
        pytest.skip("no corpus on this machine")
    mod, _rows = corpus
    n2i = mod.load_name_to_idx()[0]
    picks = [None] * 10
    picks[0] = n2i["Vi"]          # jungle only
    picks[1] = n2i["Lee Sin"]     # jungle only
    lanes = L.assign_lanes(picks)
    assert lanes[0] != lanes[1]
    assert lanes[0] == 1 or lanes[1] == 1, "neither landed in the jungle"


def test_the_assignment_agrees_with_real_drafts():
    """The number the change was made on: >=90 % against 67 % for the role head.

    This is the guard that would catch a rebuild of `champ_roles.json` from the
    wrong columns or the wrong id space - the assignment would still run, and
    still be injective, and simply be worse.
    """
    corpus = _corpus()
    if corpus is None:
        pytest.skip("no corpus on this machine")
    _mod, rows = corpus

    agree = total = 0
    for r in rows[-40:]:
        picks = [None] * 10
        true_roles = [None] * 10
        per_team = [0, 0]
        for kind, team, champ, role in r["moves"]:
            if kind != 1:
                continue
            lanes = L.assign_lanes(picks)
            for slot in range(10):
                if true_roles[slot] is not None:
                    total += 1
                    agree += 1 if lanes[slot] == true_roles[slot] else 0
            picks[team * 5 + per_team[team]] = champ
            true_roles[team * 5 + per_team[team]] = role
            per_team[team] += 1

    assert total > 1000
    rate = agree / total
    assert rate >= 0.90, "assignment agrees only %.1f %% with real drafts" % (
        100 * rate)
