"""The two pure helpers on `RerankPredictor` that the role board depends on.

Neither needs a GPU or the 4B trunk: `lane_of` is set arithmetic and
`patch_str` is a lookup. They are tested separately because both fail *silently*
- a wrong `lane_of` does not raise, it sends the no-lane prompt to a model that
was trained with a lane in it, and the only symptom is a lower score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pro.model.archive.llm.rerank_predictor import RerankPredictor  # noqa: E402
from solo_queue.model.policy import lanes as L              # noqa: E402


class _Pure:
    """`lane_of` / `patch_str` without __init__ (no trunk, no encoder)."""

    lane_of = RerankPredictor.lane_of
    patch_str = RerankPredictor.patch_str

    def __init__(self, vocab=()):
        self._patch_vocab = list(vocab)


def test_lane_of_recovers_every_lane_the_role_board_asks_about():
    """The role board passes `lanes.legal_champions(lane)` verbatim.

    The first version compared against `lanes.table().items()`, which is a dict
    of dicts (`roles`, `counts`, `share`, `floor`) - so `int(c)` ran over role
    *names*, the comparison never matched, and every role-board cell silently
    got the no-lane prompt. A wrong answer here is invisible: the score just
    comes out lower than the model can do.
    """
    p = _Pure()
    for lane in range(5):
        assert p.lane_of(L.legal_champions(lane)) == lane


def test_lane_of_says_unknown_rather_than_guessing():
    p = _Pure()
    assert p.lane_of(None) == -100
    assert p.lane_of([]) == -100
    assert p.lane_of([1, 2, 3]) == -100
    # a lane set with one champion removed is a different set, not that lane
    full = list(L.legal_champions(2))
    assert p.lane_of(full[1:]) == -100
    # the sets are disjoint, so a union is not a lane either
    assert p.lane_of(list(L.legal_champions(0)) + list(L.legal_champions(1))) == -100


def test_lane_of_accepts_string_ids():
    """`allowed` comes from JSON in some paths; ids arrive as strings."""
    p = _Pure()
    assert p.lane_of([str(c) for c in L.legal_champions(3)]) == 3


def test_patch_str_prefers_the_name_and_refuses_to_invent_one():
    p = _Pure(vocab=["16.19", "16.20"])
    assert p.patch_str(1) == "16.19"
    assert p.patch_str(2) == "16.20"
    # out of range / unparseable -> "", so the prompt says nothing rather than
    # the wrong patch
    assert p.patch_str(99) == ""
    assert p.patch_str(None) == ""
    # a name is passed through: the agent's tool sends the name, not the row
    assert p.patch_str("16.21") == "16.21"
