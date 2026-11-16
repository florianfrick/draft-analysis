"""The agent arm's measurement layer: the answer parser and the row contract.

The accuracy this harness reports is `parsed champion == truth`, so the parser
is part of the metric. A parser that silently reads "unlike Corki, Leona" as
Corki does not crash - it reports a number two points low and the run looks like
the model's fault. These tests pin the cases that actually occur in the
transcripts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import eval_api_agent as E  # noqa: E402

NAMES = ["Leona", "Corki", "Lucian", "Rell", "Kai'Sa", "Vi", "Viktor",
         "Miss Fortune", "Dr. Mundo"]
LEONA, CORKI, LUCIAN, RELL, KAISA, VI, VIKTOR, MF, MUNDO = range(9)


def test_the_marker_line_wins_over_the_prose():
    """The case the marker exists for: the first champion named is the one the
    answer argues against."""
    cid, _illegal, mode = E.answer_of(
        "The model's #1 is Corki at 0.28, but blue's bot lane is full. "
        "Leona is the best lane-legal pick.\nPICK: Leona",
        [LEONA, CORKI], NAMES)
    assert cid == LEONA and mode == "marker"


def test_the_marker_tolerates_bold_and_a_ban_verb():
    for text in ("Leona.\n**PICK: Leona**", "Leona.\nBAN: Leona",
                 "Leona.\nPICK:  **Leona**  "):
        cid, _illegal, mode = E.answer_of(text, [LEONA], NAMES)
        assert (cid, mode) == (LEONA, "marker"), text


def test_a_contrast_opening_is_parsed_from_the_prose_when_no_marker_is_given():
    """Documented limitation, not an endorsement: without the marker the first
    champion named wins, which is the wrong one here. The mode says so."""
    cid, _illegal, mode = E.answer_of(
        "The model's #1 is Corki at 0.28, but blue's bot lane is full. "
        "Leona is the best lane-legal pick.",
        [LEONA, CORKI], NAMES)
    assert cid == CORKI and mode == "first-sentence"


def test_the_first_sentence_is_parsed_before_the_body():
    cid, _illegal, mode = E.answer_of(
        "**Leona** - the model's top two cannot fill an open lane.",
        [LEONA, CORKI], NAMES)
    assert cid == LEONA and mode == "first-sentence"


def test_the_body_is_the_fallback_when_the_first_sentence_names_nothing():
    cid, _illegal, mode = E.answer_of(
        "Reading the board first. Leona fills the open support slot.",
        [LEONA, CORKI], NAMES)
    assert cid == LEONA and mode == "body"


def test_a_champion_outside_the_legal_pool_is_counted_separately():
    """Naming a banned champion is a legality error, not a non-answer."""
    cid, illegal, mode = E.answer_of("Viktor is the pick.", [LEONA], NAMES)
    assert cid is None and illegal == VIKTOR and mode == "none"


def test_an_empty_answer_is_its_own_mode():
    assert E.answer_of(None, [LEONA], NAMES) == (None, None, "empty")
    assert E.answer_of("   ", [LEONA], NAMES) == (None, None, "empty")


def test_vi_is_not_read_out_of_viktor():
    """Longest match wins, or a one-name answer reads as two champions."""
    cid, _illegal, _mode = E.answer_of("Viktor.", [VIKTOR], NAMES)
    assert cid == VIKTOR


def test_a_multi_word_name_survives_punctuation():
    cid, _illegal, _mode = E.answer_of("**Miss Fortune** - the model's #1.",
                                       [MF], NAMES)
    assert cid == MF


def _sample():
    champ_ids = torch.full((10,), 0, dtype=torch.long)
    pad = torch.ones(10, dtype=torch.bool)
    champ_ids[0], pad[0] = 3, False          # Rell picked by blue
    champ_ids[5], pad[5] = 4, False          # Kai'Sa picked by red
    ban_ids = torch.zeros(10, dtype=torch.long)
    ban_ids[0], ban_ids[1] = 1, 2            # Corki, Lucian banned
    pool = torch.zeros(9, dtype=torch.bool)
    pool[8] = True                           # Dr. Mundo fearless
    return {"champ_ids": champ_ids, "pad_mask": pad, "ban_ids": ban_ids,
            "n_bans": 2, "unavailable_mask": pool}


def test_the_legal_pool_excludes_picks_bans_and_the_fearless_pool():
    legal = E._legal_ids(_sample(), len(NAMES))
    assert 3 not in legal and 4 not in legal          # picked
    assert 1 not in legal and 2 not in legal          # banned
    assert 8 not in legal                             # fearless
    assert 0 in legal and 5 in legal                  # untouched


def test_the_cache_key_changes_when_the_prompt_changes():
    """A prompt edit invalidates every cached answer, or the resume lies."""
    row = {"gameid": "g1", "move_idx": 7}
    args = SimpleNamespace(model="deepseek-chat", max_steps=5)
    a = E.row_key(row, args, "prompt one")
    b = E.row_key(row, args, "prompt one")
    c = E.row_key(row, args, "prompt two")
    assert a == b and a != c


def test_the_cache_key_changes_with_the_step_budget():
    row = {"gameid": "g1", "move_idx": 7}
    a = E.row_key(row, SimpleNamespace(model="m", max_steps=5), "p")
    b = E.row_key(row, SimpleNamespace(model="m", max_steps=8), "p")
    assert a != b
