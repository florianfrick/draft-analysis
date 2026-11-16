"""The one-click suggestion's parser and its shared configuration.

`parse_answer` is the difference between a prose answer and a move the panel
can offer to play. It is also the thing that can silently turn a refusal into a
name: an answer that never names a champion, names one already on the board, or
names a fake `PICK:` on a ban clock is not a suggestion. These tests pin those
cases, plus the rule the measured harness uses (marker first, longest name
wins), so the served arm and `eval_api_agent` cannot drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.agent import draft_state as D          # noqa: E402
from pro.model.agent import prompt_explained, suggest  # noqa: E402
from pro.model.agent.model_tools import ChampIndex    # noqa: E402

NAMES = ["Leona", "Corki", "Lucian", "Rell", "Kai'Sa", "Vi", "Viktor",
         "Miss Fortune", "Dr. Mundo"]
CHAMPS = ChampIndex(NAMES)
LEONA, CORKI, LUCIAN, RELL, KAISA, VI, VIKTOR, MF, MUNDO = range(9)


def state(*moves, unavailable=()):
    return D.DraftState(moves=tuple(moves), unavailable=tuple(unavailable))


def test_the_marker_line_wins_over_the_prose():
    """The marker exists for answers that name the champion they argue against."""
    out = suggest.parse_answer(
        "The model's #1 is Corki at 0.28, but blue's bot lane is full. "
        "Leona is the best lane-legal pick.\nPICK: Leona", CHAMPS, state())
    assert (out["champion"], out["idx"], out["parse"]) == ("Leona", LEONA,
                                                           "marker")
    assert out["legal"] is True


def test_the_marker_tolerates_bold_and_a_ban_verb():
    for text in ("Leona.\n**PICK: Leona**", "Leona.\nBAN: Leona",
                 "Leona.\nPICK:  **Leona**  "):
        out = suggest.parse_answer(text, CHAMPS, state())
        assert out["champion"] == "Leona" and out["parse"] == "marker", text
    assert suggest.parse_answer("BAN: Leona", CHAMPS, state())["kind"] == "ban"


def test_a_marker_that_disagrees_with_the_clock_is_flagged():
    """Move 0 is a ban; `PICK: Leona` is not a playable recommendation."""
    out = suggest.parse_answer("PICK: Leona", CHAMPS, state())
    assert out["kind"] == "pick" and out["kind_matches_clock"] is False
    assert suggest.parse_answer("BAN: Leona", CHAMPS,
                                state())["kind_matches_clock"] is True


def test_the_first_sentence_and_then_the_body_are_the_fallbacks():
    first = suggest.parse_answer("**Leona** - the head's top two cannot fill "
                                 "an open lane.", CHAMPS, state())
    assert (first["champion"], first["parse"]) == ("Leona", "first-sentence")
    body = suggest.parse_answer("Reading the board first. Leona fills the "
                                "open support slot.", CHAMPS, state())
    assert (body["champion"], body["parse"]) == ("Leona", "body")


def test_an_answer_with_no_name_is_not_a_suggestion():
    assert suggest.parse_answer(None, CHAMPS, state())["parse"] == "empty"
    assert suggest.parse_answer("  ", CHAMPS, state())["parse"] == "empty"
    out = suggest.parse_answer("the list is flat, so defer to the model.",
                               CHAMPS, state())
    assert out["parse"] == "none" and out["champion"] is None
    assert out["legal"] is False


def test_vi_is_not_read_out_of_viktor_and_long_names_survive_punctuation():
    assert suggest.parse_answer("Viktor.", CHAMPS, state())["idx"] == VIKTOR
    assert suggest.parse_answer("Kai'Sa.", CHAMPS, state())["idx"] == KAISA
    assert suggest.parse_answer("**Miss Fortune** - the model's #1.",
                                CHAMPS, state())["idx"] == MF


def test_an_on_board_or_fearless_champion_is_reported_illegal():
    picked_or_pooled = state(("ban", "blue", "corki"),
                             unavailable=("rell",))
    for text in ("BAN: Corki", "PICK: Rell"):
        out = suggest.parse_answer(text, CHAMPS, picked_or_pooled)
        assert out["legal"] is False and "board" in out["note"], text
    assert suggest.parse_answer("BAN: Leona", CHAMPS,
                                picked_or_pooled)["legal"] is True


class _Head:
    def predict_next_move(self, k=1):
        return {"top": [{"name": "Corki", "rank": 1, "p_head": 0.28}],
                "patch_resolved": "16.20", "patch_known": True}


class _NoHead:
    def predict_next_move(self, k=1):
        raise ValueError("no next-move model is loaded")


def test_fidelity_is_the_heads_own_number_one_on_this_board():
    defer = suggest.head_fidelity(_Head(), CHAMPS, {"champion": "Corki"})
    assert defer["matches"] is True and defer["idx"] == CORKI
    assert defer["p_head"] == 0.28 and defer["patch"] == "16.20"
    overrule = suggest.head_fidelity(_Head(), CHAMPS, {"champion": "Leona"})
    assert overrule["matches"] is False
    assert suggest.head_fidelity(_Head(), CHAMPS,
                                 {"champion": None})["matches"] is False
    assert suggest.head_fidelity(_NoHead(), CHAMPS, {}) is None


def test_the_served_arm_is_the_measured_configuration():
    c = suggest.config()
    assert c["ablate"] == ["soloq_matchup", "sql"] and c["max_steps"] == 5
    assert c["max_calls"] == 8 and c["system"] == "explained"
    assert "single best move" in c["question"]
    assert suggest.system() == prompt_explained.system_prompt()
    assert "sql" not in suggest.system()
    assert "soloq_matchup" not in suggest.system()


# ---- ranking arm --------------------------------------------------------
HEAD_TOP = [
    {"name": "Corki", "p_head": 0.28},
    {"name": "Lucian", "p_head": 0.20},
    {"name": "Kai'Sa", "p_head": 0.15},
    {"name": "Viktor", "p_head": 0.11},
    {"name": "Rell", "p_head": 0.09},
    {"name": "Leona", "p_head": 0.07},
    {"name": "Miss Fortune", "p_head": 0.05},
    {"name": "Dr. Mundo", "p_head": 0.03},
    {"name": "Vi", "p_head": 0.02},
]


def test_parse_ranking_keeps_agent_order_and_fills_from_head():
    """The agent's own (NN%) lines win and keep their order; the untouched
    slots are back-filled from the head, head-sourced, up to k."""
    text = "1. Corki (80%)\n2. Lucian (70%)\n3. Viktor (60%)"
    out = suggest.parse_ranking(text, CHAMPS, state(), k=8, head_top=HEAD_TOP)
    sugs = out["suggestions"]
    assert len(sugs) == 8
    assert [s["name"] for s in sugs[:3]] == ["Corki", "Lucian", "Viktor"]
    assert [s["source"] for s in sugs[:3]] == ["model", "model", "model"]
    assert [round(s["conf"], 2) for s in sugs[:3]] == [0.80, 0.70, 0.60]
    # The rest are the head's own, in head order, head-sourced.
    assert [s["name"] for s in sugs[3:]] == [
        "Kai'Sa", "Rell", "Leona", "Miss Fortune", "Dr. Mundo"]
    assert all(s["source"] == "head" for s in sugs[3:])
    assert all(s["legal"] for s in sugs)


def test_parse_ranking_accepts_a_decimal_confidence():
    """A local 4B asked for `NN%` answers `15.6%`.

    An integer-only pattern threw the whole list away and fell back to the head
    without saying so, which reads as "the model had nothing to say" - the most
    expensive kind of parser bug, because it looks like a model result.
    """
    out = suggest.parse_ranking("1. Corki (15.6%)\n2. Lucian (7.5%)", CHAMPS,
                               state(), k=8, head_top=HEAD_TOP)
    sugs = out["suggestions"]
    assert out["parse"] == "model"
    assert [s["name"] for s in sugs[:2]] == ["Corki", "Lucian"]
    assert [s["source"] for s in sugs[:2]] == ["model", "model"]
    assert round(sugs[0]["conf"], 3) == 0.156
    # and the head still fills the rest, so k is unchanged
    assert len(sugs) == 8


def test_parse_ranking_vi_is_not_read_out_of_viktor():
    """Same collision guard as the single-move parser: `Vi` must not claim a
    hit inside `Viktor`, and both keep their text order."""
    out = suggest.parse_ranking("Vi (50%) Viktor (40%)", CHAMPS, k=8,
                               head_top=HEAD_TOP)
    sugs = out["suggestions"]
    assert sugs[0]["name"] == "Vi" and sugs[1]["name"] == "Viktor"
    assert sugs[0]["source"] == "model" and sugs[1]["source"] == "model"
    assert round(sugs[0]["conf"], 2) == 0.50


def test_parse_ranking_empty_answer_falls_back_to_head():
    """A turn with no key (or no text) degrades to the head's own list."""
    out = suggest.parse_ranking("", CHAMPS, k=8, head_top=HEAD_TOP)
    sugs = out["suggestions"]
    assert len(sugs) == 8 and out["parse"] == "head"
    assert [s["name"] for s in sugs] == [h["name"] for h in HEAD_TOP[:8]]
    assert all(s["source"] == "head" for s in sugs)


def test_parse_ranking_truncates_to_k_and_omits_zero_confidence():
    """More than k named champions is truncated; a name with NN%=0 is dropped
    (and, being seen, is not re-added from the head)."""
    long_text = ("Corki (90%)\nLucian (80%)\nKai'Sa (70%)\nViktor (60%)\n"
                 "Rell (50%)\nLeona (40%)\nMiss Fortune (30%)\n"
                 "Dr. Mundo (20%)\nVi (10%)")
    out = suggest.parse_ranking(long_text, CHAMPS, k=8, head_top=HEAD_TOP)
    sugs = out["suggestions"]
    assert len(sugs) == 8                      # truncated, Vi dropped
    assert sugs[-1]["name"] == "Dr. Mundo"
    assert "Vi" not in [s["name"] for s in sugs]

    zero = suggest.parse_ranking("Corki (0%) Lucian (80%)", CHAMPS, k=8,
                                 head_top=HEAD_TOP)
    names = [s["name"] for s in zero["suggestions"]]
    assert "Corki" not in names                 # zero-confidence, not filled
    assert names[0] == "Lucian"


def test_parse_ranking_reports_an_illegal_champion_from_state():
    """A champion the agent names that is already on the board is shown but
    flagged, exactly like the single-move parser."""
    taken = state(("pick", "blue", "corki"))
    out = suggest.parse_ranking("Corki (90%) Lucian (80%)", CHAMPS, taken,
                                k=8, head_top=HEAD_TOP)
    sugs = out["suggestions"]
    assert sugs[0]["name"] == "Corki" and sugs[0]["legal"] is False
    assert "board" in sugs[0]["note"]
    assert sugs[1]["name"] == "Lucian" and sugs[1]["legal"] is True


def test_rank_question_and_config_shape():
    q = suggest.rank_question(k=8, kind="ban")
    assert "8" in q and "ban" in q and "predict_next_move" in q
    c = suggest.rank_config()
    assert c["ablate"] == ["soloq_matchup", "sql"]
    assert c["max_steps"] == 5 and c["max_calls"] == 8 and c["k"] == 8
    assert c["system"] == "explained"
