"""Tests for the hosted-API reranker harness: the parser, the prompt, the stats.

The failure mode this guards against is the one that makes an eval worthless
rather than wrong: an answer parsed as the wrong champion (`Vi` read out of
`Viktor`), a `--no-probs` arm that also quietly stripped a percentage out of
the task line, or a paired test whose two counts came from different rows.
None of those raise; they just move the number.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import common as C            # noqa: E402
from pro.model.archive.llm import eval_api_rerank as E   # noqa: E402
from pro.model.archive.llm.prompt import build_text      # noqa: E402

NAMES = ["Vi", "Viktor", "Kai'Sa", "Dr. Mundo", "Jax", "Malphite",
         "Ahri", "Zed", "Leona", "Nautilus", "Zeri", "Corki"]


def test_extract_does_not_read_vi_out_of_viktor():
    """`Vi` is a prefix of `Viktor` once punctuation is gone. Longest match at
    a position wins, and the shorter overlap is dropped - otherwise every
    answer mentioning Viktor also names Vi, and a one-name answer reads as two
    with Vi first."""
    assert E._extract("Viktor", [0, 1], NAMES) == [1]
    assert E._extract("viktor", [0, 1], NAMES) == [1]


def test_extract_matches_through_punctuation_and_case():
    assert E._extract("KAI'SA", [2], NAMES) == [2]
    assert E._extract("kaisa.", [2], NAMES) == [2]
    assert E._extract("Dr Mundo", [3], NAMES) == [3]


def test_extract_keeps_the_order_the_model_wrote():
    """`--rank-all` reads a permutation off this, so position has to survive."""
    got = E._extract("Malphite, then Vi, then Jax", [0, 4, 5], NAMES)
    assert got == [5, 0, 4]


def test_extract_ignores_a_candidate_that_is_not_there():
    assert E._extract("I would pick Malphite", [0, 4], NAMES) == []


def test_a_champion_outside_the_list_is_counted_separately():
    """Naming a real champion the head did not offer is the model refusing the
    constraint. That is a different failure from a refusal to answer, and it
    is the cost the decoder mask hides."""
    chosen, outside = E.parse_answer("Malphite", [0, 1], NAMES)
    assert chosen is None and outside == 5


def test_a_refusal_is_neither_a_choice_nor_an_outside_name():
    assert E.parse_answer("I can't help with that.", [0, 1], NAMES) == (None,
                                                                      None)
    assert E.parse_answer("", [0, 1], NAMES) == (None, None)


def test_parse_answer_takes_the_first_name_it_finds():
    chosen, outside = E.parse_answer("Jax is the pick, not Vi", [0, 4], NAMES)
    assert chosen == 4 and outside is None


def test_without_probs_only_rewrites_the_candidate_line():
    state = ("Patch: 16.20 | Turn: Blue 1 (Ban 1 of 10)\n"
             "Blue Bans: Jinx, Yuumi\n"
             "Candidates: Jax 31%, Renekton 22%, Vi 0.4%\n"
             "Task: Choose the best Blue Side ban from the candidates. "
             "Answer with champion name only.")
    out = E._without_probs(state)
    assert "Candidates: Jax, Renekton, Vi" in out
    assert "Blue Bans: Jinx, Yuumi" in out
    assert "31%" not in out and "0.4%" not in out
    # every other line is byte-identical
    assert [a for a in state.split("\n") if not a.startswith("Candidates:")] \
        == [b for b in out.split("\n") if not b.startswith("Candidates:")]


def test_without_probs_refuses_text_that_is_not_v4():
    """Silently returning the input would make `--no-probs` a no-op that still
    reports an arm."""
    with pytest.raises(ValueError):
        E._without_probs("Patch: 16.20 | Turn: Blue 1 (Ban 1 of 10)")


def _sample(move_idx=0, is_ban=1, to_move=0, n_bans=0, first_pick=0):
    """The smallest sample `render_state` accepts, built by hand.

    `test_board.py` pins the real corpus rendering; this is for the harness'
    own text surgery, so a synthetic board is enough and needs no DB.
    """
    champ = torch.full((10,), 5, dtype=torch.long)
    role = torch.zeros(10, dtype=torch.long)
    pad = torch.ones(10, dtype=torch.bool)
    return {"champ_ids": champ, "role_ids": role, "pad_mask": pad,
            "ban_ids": torch.zeros(10, dtype=torch.long),
            "n_bans": torch.tensor(n_bans), "first_pick": torch.tensor(first_pick),
            "move_idx": torch.tensor(move_idx), "is_ban": torch.tensor(is_ban),
            "to_move": torch.tensor(to_move), "target_role": torch.tensor(-100),
            "unavailable_mask": torch.zeros(173, dtype=torch.bool),
            "meta": {"patch": "16.20", "gameid": "G", "series_id": "S"}}


def test_the_no_probs_arm_changes_only_the_candidate_line_of_a_real_prompt():
    """End to end on the actual builder: the two arms must differ by the
    percentages and nothing else, or the ablation is measuring two changes."""
    s = _sample()
    cands = [(4, 0.31), (0, 0.22), (1, 0.004)]
    _, body, _ = build_text(s, NAMES, context="soft+text", prompt_format="v4",
                            candidates=cands)
    stripped = E._without_probs(body)
    assert body.endswith("Answer:")
    assert stripped.endswith("Answer:")
    assert "Jax 31%" in body and "Jax" in stripped and "31%" not in stripped
    assert body.count("\n") == stripped.count("\n")


def _display():
    """The harness' key -> corpus-spelling map, so the test covers it too."""
    table = {E.DS.key(n): n for n in NAMES}
    return lambda k: table.get(k, k)


def _lane_indexed_sample():
    """Six bans, blue's *first* pick in its Bot slot, red's in its Top slot.

    This is the corpus layout: a pick sits at `team * 5 + role`, so a team's
    first pick is routinely not at index 0. The UI's strips are in pick order,
    which is why `log_from_board` reads slot n as that side's n-th pick.
    """
    s = _sample(move_idx=8, is_ban=0, to_move=1, n_bans=6)
    s["champ_ids"] = torch.tensor([0, 0, 0, 4, 0, 3, 0, 0, 0, 0])
    s["pad_mask"] = torch.tensor(
        [True, True, True, False, True, False, True, True, True, True])
    # bans are a separate, disjoint set: a champion on both lists is a board
    # `log_from_board` refuses, which is not what this test is about
    s["ban_ids"] = torch.tensor([6, 7, 8, 9, 10, 11, 0, 0, 0, 0])
    return s


def test_the_state_arm_compacts_lane_slots_into_pick_order():
    """Without compaction the log stops at blue's empty slot 0 and the agent is
    told it is move 6 while the user is on move 8 - the same class of silent
    staleness the fearless bug was."""
    s = _lane_indexed_sample()
    st, text = E._state_text(s, NAMES, _display())
    assert st.move_idx == 8
    due = st.to_move()
    assert due["move_idx"] == 8 and due["kind"] == "pick" and due["side"] == "red"
    assert "Jax" in text and "Dr. Mundo" in text
    # the uncompacted list is what the old code passed, and it truncates
    raw = [NAMES[int(c)] if not bool(p) else None
           for c, p in zip(s["champ_ids"], s["pad_mask"])]
    assert E.DS.log_from_board(
        raw, bans=[NAMES[int(b)] for b in s["ban_ids"][:6]]).move_idx == 6


def test_the_state_arm_states_the_fearless_pool_and_the_legal_count():
    s = _lane_indexed_sample()
    s["unavailable_mask"] = torch.zeros(173, dtype=torch.bool)
    s["unavailable_mask"][[0, 1]] = True          # Vi and Viktor are gone
    # 173 - 6 bans - 2 picks - 2 fearless = 163: `legal_count` subtracts the
    # pool as well, which is the point of carrying it on the state
    _, text = E._state_text(s, NAMES, _display())
    assert "picked earlier in this series" in text
    assert "legal champions left: 163" in text


def test_the_rank_all_arm_changes_only_the_answer_instruction():
    """Two conflicting asks in one prompt ("rank all five" above "answer with
    champion name only") would make the arm unreadable - the model picks one
    and the number means nothing."""
    s = _sample()
    _, body, _ = build_text(s, NAMES, context="soft+text", prompt_format="v4",
                            candidates=[(4, 0.31), (0, 0.22), (1, 0.004)])
    ranked = E._rank_all_task(body, 3)
    assert "Answer with champion name only." not in ranked
    assert "all 3 names in order" in ranked
    assert body.split("\n")[:-2] == ranked.split("\n")[:-2]


def test_the_rank_all_arm_refuses_a_prompt_it_does_not_recognise():
    with pytest.raises(ValueError):
        E._rank_all_task("Task: pick something.", 5)


def test_wilson_brackets_the_estimate_and_narrows_with_n():
    lo, hi = E.wilson(60, 200)
    assert lo < 0.30 < hi
    lo2, hi2 = E.wilson(600, 2000)
    assert (hi2 - lo2) < (hi - lo)


def test_wilson_is_defined_at_the_extremes():
    assert E.wilson(0, 0) == (0.0, 0.0)
    lo, hi = E.wilson(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.1
    lo, hi = E.wilson(50, 50)
    assert hi == 1.0 and 0.9 < lo < 1.0


def test_mcnemar_needs_no_discordant_pairs():
    """Two identical columns are not evidence of a difference, and a chi-square
    approximation divides by zero here."""
    assert E.mcnemar(0, 0) == 1.0


def test_mcnemar_is_symmetric_and_small_when_one_side_wins():
    assert E.mcnemar(30, 5) == E.mcnemar(5, 30)
    assert E.mcnemar(30, 5) < 0.001
    assert E.mcnemar(12, 9) > 0.5
    assert math.isclose(E.mcnemar(1, 1), 1.0, abs_tol=1e-9)


def test_mean_rank_is_one_based_and_counts_an_unnamed_candidate_as_last():
    """`evaluate_llm` reports a 1-based mean_rank (1.62 at k=5), so a top-1
    answer has to score 1.0 here, not 0.0, or the two reports cannot be
    compared at all."""
    t = E.Tally()
    for pos in (0, 0, 2, 4):
        t.add({"hit": pos == 0, "head_hit": False, "ceiling_hit": True,
               "chosen": 1, "outside": None, "error": None, "pos": pos})
    out = t.out()
    assert out["n"] == 4
    assert out["mean_rank"] == pytest.approx((0 + 0 + 2 + 4) / 4 + 1.0)
    assert out["ft_top5"] == pytest.approx(1.0)


def test_an_offlist_answer_scores_zero_not_an_error():
    t = E.Tally()
    t.add({"hit": False, "head_hit": True, "ceiling_hit": True,
           "chosen": None, "outside": 5, "error": None, "pos": None})
    out = t.out()
    assert out["api_top1"] == 0.0
    assert out["offlist"] == 1.0
    assert out["named_outside_list"] == 1.0
    assert "mean_rank" not in out
