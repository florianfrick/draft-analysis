"""Tests for `eval_keep_switch`'s parsing and its gate, not for the model.

The measured claim is "asked as a keep-or-switch decision the model loses 7 pp
against the head". That only means something if a KEEP is scored as the head, a
SWITCH as the named candidate, and a switch the confidence gate rejects as the
head again - three ways to silently score the wrong arm.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import eval_keep_switch as K                                # noqa: E402
from pro.model.agent import model_tools                     # noqa: E402

NAMES = ["Lulu", "Leona", "Gwen", "Vex", "Nautilus"]
CHAMPS = model_tools.ChampIndex(NAMES)
CANDS = [(0, 0.18), (1, 0.10), (2, 0.09), (3, 0.06), (4, 0.05)]


def _rec(switched, p1, p2, in1, head_in1):
    return {"switched": switched, "p1": p1, "p2": p2,
            "in1": in1, "head_in1": head_in1,
            "in5": head_in1, "head_in5": head_in1, "parse": "keep",
            "prompt": 10, "completion": 1, "calls": 1}


def test_parse_keep_and_switch_name_the_right_candidate():
    assert K.parse_answer("KEEP Lulu", CANDS, CHAMPS) == (0, "keep")
    assert K.parse_answer("SWITCH Leona", CANDS, CHAMPS) == (1, "switch")
    assert K.parse_answer("  switch: **Gwen**  ", CANDS, CHAMPS) == (2, "switch")
    # a bare KEEP has still answered - the parser must not call it unparsed
    assert K.parse_answer("KEEP", CANDS, CHAMPS) == (0, "keep")


def test_parse_refuses_a_switch_it_cannot_place():
    # named a champion that is not on the list, and one that does not exist
    assert K.parse_answer("SWITCH Jinx", CANDS, CHAMPS) == (None, "switch-unnamed")
    assert K.parse_answer("SWITCH", CANDS, CHAMPS) == (None, "switch-unnamed")
    assert K.parse_answer("", CANDS, CHAMPS) == (None, "empty")


def test_parse_falls_back_to_the_first_name_in_prose():
    pos, mode = K.parse_answer("Leona is the better ban here.", CANDS, CHAMPS)
    assert (pos, mode) == (1, "body")
    # `Vi` must not match inside `Viktor`: longest name wins in `_first_named`
    assert K.parse_answer("Vex", CANDS, CHAMPS) == (3, "body")


def test_summarise_scores_a_rejected_switch_as_the_head():
    # Four rows. The switch is right once (row 0) and wrong once (row 1); rows
    # 2-3 keep. tau = 0.05 accepts the switch on rows 0 and 1 only.
    recs = [
        _rec(True, 0.18, 0.15, in1=True, head_in1=False),
        _rec(True, 0.18, 0.16, in1=False, head_in1=True),
        _rec(False, 0.30, 0.10, in1=True, head_in1=True),
        _rec(False, 0.12, 0.11, in1=False, head_in1=False),
    ]
    raw = K.summarise(recs)
    assert raw["raw"] == pytest.approx(0.5)        # 2 of 4 answered right
    assert raw["head"] == pytest.approx(0.5)
    gated = K.summarise(recs, tau_a=0.05)
    # rows 0 and 1 switch (accepted), rows 2-3 keep: 1 + 0 + 1 + 0
    assert gated["gated"] == pytest.approx(0.5)
    assert gated["gated_switches"] == 2
    assert gated["switch_precision"] == pytest.approx(0.5)


def test_summarise_rejects_a_switch_the_gate_would_not_take():
    # Same rows, tau = 0: both switches are above the margin, so both fall back
    # to the head - which is right on row 1 and wrong on row 0.
    recs = [
        _rec(True, 0.18, 0.15, in1=True, head_in1=False),
        _rec(True, 0.18, 0.16, in1=False, head_in1=True),
    ]
    gated = K.summarise(recs, tau_a=0.0)
    assert gated["gated_switches"] == 0
    assert gated["gated"] == pytest.approx(0.5)
    assert gated["gated_paired"] == {"agent_only": 0, "head_only": 0, "p": 1.0}
