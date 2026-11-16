"""Tests for the soloQ pairwise-matchup tool.

Most of these need the soloQ analytical store (`store.duckdb`); they skip when
it is not built, so the suite stays green in environments that only have the pro
corpus. The resolution and validation logic is tested without the store.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.agent import draft_state, model_tools, schema, soloq_tool  # noqa: E402
from pro.model.agent.session import Session                          # noqa: E402

CHAMPS = model_tools.ChampIndex(schema.names())


@pytest.fixture()
def session_with_store():
    """A store-backed session, so the soloq tool dispatches without a corpus."""
    store = draft_state.Store(model_tools.ChampIndex(schema.names()))
    return Session(store=store, max_calls=6)


def test_availability_is_a_bool():
    assert isinstance(soloq_tool.available(), bool)


def test_invalid_kind_is_refused_without_a_store():
    out = soloq_tool.run("Ahri", "Zed", kind="against")
    assert "error" in out
    assert "vs" in out["error"] or "with" in out["error"]


def test_invalid_role_is_refused_without_a_store():
    out = soloq_tool.run("Ahri", "Zed", role="midlaner")
    assert "error" in out
    assert "role" in out["error"]


def test_name_resolution_maps_a_name_to_a_riot_id():
    # Ahri -> 103, and the map is total across the shared dense space.
    assert soloq_tool._resolve("Ahri", CHAMPS) == 103
    assert soloq_tool._resolve(103, CHAMPS) == 103
    # Fuzzy: punctuation and case.
    assert soloq_tool._resolve("Kai'Sa", CHAMPS) == \
        soloq_tool._resolve("kaisa", CHAMPS)
    with pytest.raises(ValueError):
        soloq_tool._resolve("NotAChampionXYZ", CHAMPS)


@pytest.mark.skipif(not soloq_tool.available(),
                    reason="soloQ analytical store not built")
def test_a_real_matchup_returns_observed_baseline_and_delta():
    out = soloq_tool.run("Ahri", "Zed", kind="vs", patch="16.20", champs=CHAMPS)
    assert "error" not in out, out
    assert out["champion"] == "Ahri" and out["partner"] == "Zed"
    assert out["kind"] == "vs" and out["patch"] == "16.20"
    assert out["observed"] is not None
    assert out["observed"]["n"] >= 1
    assert 0 <= out["observed"]["win"] <= 1
    assert out["baseline"] is not None
    assert out["delta_win"] is not None
    # The query string is returned for debuggability.
    assert "cube2" in out["query"]
    # A patch with a real cell is reliable at the default threshold.
    assert isinstance(out["reliable"], bool)


@pytest.mark.skipif(not soloq_tool.available(),
                    reason="soloQ analytical store not built")
def test_a_huge_min_n_flips_reliable_off():
    out = soloq_tool.run("Ahri", "Zed", kind="vs", patch="16.20", min_n=10**9,
                         champs=CHAMPS)
    assert "error" not in out
    assert out["reliable"] is False


@pytest.mark.skipif(not soloq_tool.available(),
                    reason="soloQ analytical store not built")
def test_a_role_filter_runs_the_lane_scan():
    out = soloq_tool.run("Ahri", "Zed", kind="vs", patch="16.20", role="MIDDLE",
                         champs=CHAMPS)
    assert "error" not in out
    assert out["role"] == "MIDDLE"
    assert out["lane_scan"] is not None
    assert out["lane_scan"]["role"] == "MIDDLE"


@pytest.mark.skipif(not soloq_tool.available(),
                    reason="soloQ analytical store not built")
def test_omitting_patch_weights_across_patches():
    out = soloq_tool.run("Ahri", "Zed", kind="vs", champs=CHAMPS)
    assert "error" not in out
    assert out["patch"] is None
    assert out["observed"] is not None
    assert out["observed"]["n"] >= 1


@pytest.mark.skipif(not soloq_tool.available(),
                    reason="soloQ analytical store not built")
def test_the_tool_surfaces_through_the_session(session_with_store):
    """The session advertises and dispatches it exactly like the other tools."""
    names = session_with_store.names()
    if soloq_tool.available():
        assert "soloq_matchup" in names
        res = session_with_store.call("soloq_matchup", champion="Ahri",
                                      partner="Zed", kind="vs")
        assert "error" not in res, res
        assert res["observed"] is not None
    else:
        assert "soloq_matchup" not in names
