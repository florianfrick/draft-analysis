"""Tests for the chat surface: the parser, the inverted loop, the tools.

Run:  pytest frontend/test_chat.py -q

Three things are being pinned, in order of how expensive they are to discover
later:

  * the ladder is monotone and the sign flips on negation, because a chat that
    answers "incredibly good" and "alright" identically is the failure this
    whole design exists to avoid;
  * `invert` is a renormalised prior, not a rewrite - the mass has to come out
    of the other candidates, and a taken champion has to stay untakeable;
  * a thin cell reports a count and no rate, because the alternative (a rate
    over five games) is the thing `_probe_tool_power.py` was written to prevent.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chat  # noqa: E402
import llm  # noqa: E402
import tools  # noqa: E402

PAD_FROM = 160
# Two name tables, because the two halves of this file need different things:
# the parser has to be tested against the real dDragon names (that is where the
# apostrophes and the "Dr. Mundo" full stop live), and the tools against a table
# whose indices the test controls.
NAMES = ["Champ%d" % i for i in range(173)]
REAL = tools._load_names(173)


def make_row(gameid, blue, red, blue_win=1, first_pick=0, patch="16.20"):
    """A row with both shapes the code needs: `moves` (20 long) and `champ`."""
    moves = ([(1, 0, c, i) for i, c in enumerate(blue)]
             + [(1, 1, c, i) for i, c in enumerate(red)])
    filler = PAD_FROM
    while len(moves) < llm.N_MOVES:
        moves.append((0, len(moves) % 2, filler, -100))
        filler -= 1
    return {"gameid": gameid, "patch": patch, "first_pick": first_pick,
            "blue_win": blue_win, "date_ordinal": 1, "league": "LCK",
            "date": "2026-09-01 00:00:00", "blue_team": "T1",
            "red_team": "GEN", "moves": moves,
            "champ": list(blue) + list(red)}


def corpus(rows):
    return tools.Corpus(llm.ProIndex(rows))


# ------------------------------------------------------------------ parsing
def test_the_ladder_is_monotone():
    """The four rungs must be strictly ordered, or intensity is decorative."""
    lam = {label: v for label, v, _cues in chat.LADDER}
    assert lam["pin"] > lam["strong"] > lam["normal"] > lam["weak"]
    assert lam["weak"] > chat.MENTION_LAM


@pytest.mark.parametrize("message,expected", [
    ("Kled must be the pick", "pin"),
    ("Kled is incredibly good here", "strong"),
    ("I think Kled is good", "normal"),
    ("Kled is alright I guess", "weak"),
    ("Kled", "mention"),
])
def test_stance_is_read_from_the_clause(message, expected):
    got = chat.parse(message, REAL)
    assert got["mentions"], message
    assert got["mentions"][0]["stance"] == expected


def test_a_stronger_word_is_worth_more_nats():
    weak = chat.parse("Kled is alright", REAL)["mentions"][0]["lam"]
    strong = chat.parse("Kled is incredibly good", REAL)["mentions"][0]["lam"]
    assert strong > weak


def test_negation_flips_the_sign():
    pos = chat.parse("Kled is good", REAL)["mentions"][0]
    neg = chat.parse("Kled is bad", REAL)["mentions"][0]
    assert pos["lam"] > 0 > neg["lam"]
    assert neg["stance"].startswith("against")


def test_negation_in_a_later_clause_still_counts():
    """'don't pick Kled, he is bad' puts the qualifier in the second clause."""
    m = chat.parse("dont pick Kled, he is bad", REAL)["mentions"][0]
    assert m["lam"] < 0


def test_mentions_keep_message_order():
    """The first name the user said is the one that steers the ranking."""
    got = chat.parse("what about Vi and Orianna", REAL)
    assert [m["name"] for m in got["mentions"]] == ["Vi", "Orianna"]


def test_a_name_inside_another_name_is_not_a_mention():
    """"Viego" must not also register as "Vi"."""
    got = chat.parse("Viego is fine", REAL)
    assert [m["name"] for m in got["mentions"]] == ["Viego"]


def test_apostrophes_and_abbreviations_still_resolve():
    assert chat.parse("kaisa is fine", REAL)["mentions"][0]["name"] == "Kai'Sa"
    assert chat.parse("mundo is fine", REAL)["mentions"][0]["name"] == "Dr. Mundo"
    assert chat.parse("Dr. Mundo must be the ban", REAL)["mentions"][0]["name"] \
        == "Dr. Mundo"


def test_side_is_read_from_the_message():
    assert chat.parse("Kled is good for Blue", REAL)["side"] == 0
    assert chat.parse("Kled is good for red side", REAL)["side"] == 1
    assert chat.parse("Kled is good", REAL)["side"] is None


# ------------------------------------------------------------- the loop
def _flat_probs(top=0.2):
    """A distribution with one clear leader and a long, equal tail."""
    p = {i: (1.0 - top) / 172 for i in range(173)}
    p[0] = top
    return p


def test_invert_renormalises():
    board = tools.Board([None] * 10, [], first_pick=0)
    inv = chat.invert(_flat_probs(), board, 100, 4.0)
    assert abs(sum(inv["p"].values()) - 1.0) < 1e-9


def test_invert_is_monotone_in_lambda():
    """More nats never makes the named champion worse."""
    board = tools.Board([None] * 10, [], first_pick=0)
    ranks = [chat.invert(_flat_probs(), board, 100, lam)["after_rank"]
             for lam in (0.5, 1.5, 3.0, 6.0, 15.0)]
    assert ranks == sorted(ranks, reverse=True), ranks


def test_a_pin_reaches_the_top_of_a_normal_distribution():
    board = tools.Board([None] * 10, [], first_pick=0)
    inv = chat.invert(_flat_probs(), board, 100, 15.0)
    assert inv["after_rank"] == 1
    assert inv["lam_needed"] < 15.0


def test_lam_needed_is_the_gap_it_says_it_is():
    """The reported number must actually be enough - otherwise the reply lies."""
    board = tools.Board([None] * 10, [], first_pick=0)
    probs = _flat_probs()
    inv = chat.invert(probs, board, 100, 0.0)
    just_under = chat.invert(probs, board, 100, inv["lam_needed"] - 0.01)
    just_over = chat.invert(probs, board, 100, inv["lam_needed"] + 0.01)
    assert just_under["after_rank"] > 1
    assert just_over["after_rank"] == 1


def test_a_taken_champion_cannot_be_boosted():
    board = tools.Board([None] * 10, [100, 101], first_pick=0)
    inv = chat.invert(_flat_probs(), board, 100, 15.0)
    assert 100 not in inv["p"]
    assert inv["before_rank"] is None and inv["after_rank"] is None


def test_invert_refuses_a_distribution_with_no_legal_mass():
    """The one impossible case: the head puts nothing on any legal champion."""
    board = tools.Board([None] * 10, [], first_pick=0)
    with pytest.raises(ValueError):
        chat.invert({}, board, 0, 1.0)


def test_illegal_mass_is_dropped_before_ranking():
    """The head does not hard-mask legality, so `invert` has to."""
    board = tools.Board([None] * 10, [0], first_pick=0)
    inv = chat.invert(_flat_probs(), board, 1, 0.0)
    assert 0 not in inv["p"]
    assert inv["ranking"][0][0] != 0


# ----------------------------------------------------------------- tools
def test_legal_pool_excludes_bans_picks_and_the_fearless_pool():
    board = tools.Board([1, 2, None, None, None, None, None, None, None, None],
                        [3, 4], first_pick=0, unavailable=[5])
    out = tools.legal_pool(None, board, {}, NAMES)
    assert out["result"]["n_legal"] == 173 - 5
    names = [c["idx"] for c in out["result"]["top"]]
    assert not set(names) & {1, 2, 3, 4, 5}


def test_a_thin_cell_reports_a_count_and_no_rate():
    """The whole point of the tool layer: no rate over five games."""
    rows = [make_row("g%d" % i, [1, 2, 3, 50, 60], [70, 71, 72, 73, 74])
            for i in range(3)]
    c = corpus(rows)
    board = tools.Board([None] * 10, [1, 2, 3], first_pick=0, patch="16.20")
    out = tools.champ_evidence(c, board, 50, NAMES)
    cond = out["result"]["conditional"]
    assert cond["n_next"] == 3 and cond["n_decided"] == 3
    assert cond["rate"] is None          # 3 of 3 is not a rate, it is a count
    assert cond["share"] == 1.0          # ...but the count is still reported
    assert out["reliable"] is False
    assert "played it next" in (out["note"] or "")


def test_a_powered_cell_reports_a_rate():
    rows = [make_row("g%d" % i, [1, 2, 3, 50, 60], [70, 71, 72, 73, 74],
                     blue_win=1) for i in range(40)]
    c = corpus(rows)
    board = tools.Board([None] * 10, [1, 2, 3], first_pick=0, patch="16.20")
    out = tools.champ_evidence(c, board, 50, NAMES)
    cond = out["result"]["conditional"]
    assert cond["n_next"] == 40 and out["reliable"] is True
    assert cond["rate"] == 1.0          # blue took it and blue won every game


def test_next_move_counts_aggregates_the_whole_choice():
    """The distribution is the powered version of the per-champion cell."""
    rows = ([make_row("a%d" % i, [1, 2, 3, 50, 60], [70, 71, 72, 73, 74])
             for i in range(30)]
            + [make_row("b%d" % i, [1, 2, 3, 51, 61], [70, 71, 72, 73, 74])
               for i in range(10)])
    c = corpus(rows)
    board = tools.Board([None] * 10, [1, 2, 3], first_pick=0, patch="16.20")
    out = tools.next_move_counts(c, board, NAMES, top=5)
    assert out["n"] == 40
    assert out["result"]["top"][0]["idx"] == 50
    assert out["result"]["top"][0]["n"] == 30


def test_pair_evidence_separates_the_three_senses():
    rows = [make_row("same", [50, 51, 60, 61, 62], [70, 71, 72, 73, 74]),
            make_row("opp", [50, 80, 81, 82, 83], [51, 71, 72, 73, 74])]
    c = corpus(rows)
    out = tools.pair_evidence(c, 50, 51, NAMES)
    counts = out["result"]["counts"]
    assert counts["same_team"] == 1
    assert counts["opposing"] == 1
    assert counts["same_role"] == 1     # both at index 0 of their team
    assert counts["same_role"] <= counts["opposing"]


def test_run_rejects_an_unknown_tool():
    with pytest.raises(ValueError):
        tools.run(corpus([]), tools.Board([None] * 10, []), "nope", NAMES)


# ------------------------------------------------------------- one turn
def test_a_complete_board_answers_without_ranking():
    board = tools.Board(list(range(10)), list(range(10, 20)), first_pick=0)
    out = chat.respond(None, board, {}, NAMES, "who wins?")
    assert "complete" in out["reply"]
    assert out["ranking"] == [] and out["pick"] is None


def test_the_reply_reports_where_the_named_champion_moved():
    rows = [make_row("g%d" % i, [1, 2, 3, 50, 60], [70, 71, 72, 73, 74])
            for i in range(40)]
    board = tools.Board([None] * 10, [1, 2, 3], first_pick=0, patch="16.20")
    out = chat.respond(corpus(rows), board, _flat_probs(), NAMES,
                       "Champ100 is incredibly good here")
    assert out["moved"]["name"] == "Champ100"
    assert out["moved"]["before_rank"] > out["moved"]["after_rank"]
    assert "nats" in out["reply"]


def test_a_mention_of_a_taken_champion_is_refused_not_boosted():
    board = tools.Board([None] * 10, [100], first_pick=0, patch="16.20")
    out = chat.respond(corpus([]), board, _flat_probs(), NAMES,
                       "Champ100 is the pick")
    assert "already off the board" in out["reply"]
    assert out["pick"]["idx"] != 100


def test_the_answer_carries_every_field_the_panel_row_draws():
    """The row under the header draws `lambda`, the named champion's two ranks
    and its boosted probability, and an adjusted top-k with `idx` so a chip can
    be placed on the board. All of it has to come out of `respond`: the panel
    is not allowed to re-derive a ranking of its own, or the two rows would be
    two different claims wearing the same numbers.
    """
    rows = [make_row("g%d" % i, [1, 2, 3, 50, 60], [70, 71, 72, 73, 74])
            for i in range(40)]
    board = tools.Board([None] * 10, [1, 2, 3], first_pick=0, patch="16.20")
    out = chat.respond(corpus(rows), board, _flat_probs(), NAMES,
                       "Champ100 is incredibly good here")
    assert out["lambda"] == 6.0
    moved = out["moved"]
    assert {"name", "before_rank", "after_rank", "before_p",
            "after_p"} <= set(moved)
    assert moved["after_p"] > moved["before_p"]
    assert out["ranking"] and all(
        {"idx", "name", "prob"} <= set(c) for c in out["ranking"])
    # `pick` is the first chip, not a separate opinion about the board
    assert out["pick"]["idx"] == out["ranking"][0]["idx"]
    # and the named champion has to be findable in that list by index, because
    # that is how the chip learns which one the message was about
    assert out["mentions"][0]["idx"] in [c["idx"] for c in out["ranking"]]


def test_a_weak_mention_still_reports_its_probability_from_outside_the_list():
    """The row draws the named champion even when the boost did not lift it into
    the top eight - that is the whole point of a weak stance. So `after_p` has
    to be there for a champion the ranking does not contain.
    """
    probs = {i: 0.25 * (0.97 ** i) for i in range(173)}
    board = tools.Board([None] * 10, [1, 2, 3], first_pick=0, patch="16.20")
    out = chat.respond(corpus([]), board, probs, NAMES, "Champ100")
    assert out["lambda"] == chat.MENTION_LAM
    assert out["moved"]["after_rank"] > len(out["ranking"])
    assert 0 < out["moved"]["before_p"] < out["moved"]["after_p"]
