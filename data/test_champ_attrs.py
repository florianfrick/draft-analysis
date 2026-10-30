"""Contract tests for data/champ_attrs.json.

The attribute matrix is an *input* to the model, so the failure mode that
matters is silent misalignment: a row at the wrong index, a champion missing,
an axis reordered. All of those would train fine and just quietly be wrong.
These tests pin the alignment against the same mapping the datasets use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import build_champ_attrs as B  # noqa: E402

ATTRS = json.loads(B.OUT_PATH.read_text(encoding="utf-8"))
MAPPING = json.loads(B.MAPPING_PATH.read_text(encoding="utf-8"))
NAMES = {int(k): v for k, v in
         json.loads(B.NAMES_PATH.read_text(encoding="utf-8"))["names"].items()}
RIOT_TO_IDX = {int(k): int(v) for k, v in MAPPING["riot_to_idx"].items()}


def test_every_mapped_champion_has_a_row():
    covered = {int(k) for k in ATTRS["champions"]}
    assert covered == set(RIOT_TO_IDX), (
        "missing: %s / extra: %s"
        % (sorted(set(RIOT_TO_IDX) - covered), sorted(covered - set(RIOT_TO_IDX))))


def test_idx_matches_the_pipeline_mapping():
    for riot, entry in ATTRS["champions"].items():
        assert entry["idx"] == RIOT_TO_IDX[int(riot)], entry["name"]
        assert entry["name"] == NAMES[int(riot)], entry["name"]


def test_axes_are_the_documented_ones():
    assert ATTRS["axes"] == [a[0] for a in B.AXES]
    assert set(ATTRS["anchors"]) == set(ATTRS["axes"])
    for axis in ATTRS["axes"]:
        assert set(ATTRS["anchors"][axis]) == {"0", "1", "2", "3"}


def test_values_are_ordinal_zero_to_three():
    k = len(ATTRS["axes"])
    for entry in ATTRS["champions"].values():
        if entry["attrs"] is None:
            assert entry["confidence"] == "unknown"
            continue
        assert len(entry["attrs"]) == k, entry["name"]
        for v in entry["attrs"]:
            assert isinstance(v, int) and 0 <= v <= 3, (entry["name"], v)


def test_unknown_champions_are_declared_not_silent():
    unknown = {e["name"] for e in ATTRS["champions"].values()
               if e["attrs"] is None}
    assert unknown == set(ATTRS["unknown"])
    # every one of them must explain itself
    for entry in ATTRS["champions"].values():
        if entry["attrs"] is None:
            assert entry.get("note"), entry["name"]


def test_matrix_is_rectangular_and_indexed():
    names, rows = B.matrix(ATTRS)
    assert len(names) == len(rows) == MAPPING["num_champs"]
    assert all(n is not None for n in names), "a champion has no name at its idx"
    missing = [i for i, r in enumerate(rows) if r is None]
    assert [names[i] for i in missing] == sorted(ATTRS["unknown"])


def test_json_matches_a_fresh_build(tmp_path):
    """Editing the JSON by hand is a mistake: the builder is the source."""
    fresh = B.build(tmp_path / "champ_attrs.json")
    assert fresh == ATTRS, "champ_attrs.json is stale - re-run build_champ_attrs.py"
