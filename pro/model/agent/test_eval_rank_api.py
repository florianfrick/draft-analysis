"""Tests for `eval_rank_api`'s accounting, not for the model.

The harness itself costs money, so what is tested here is everything that could
be wrong *silently*: the paired tally (an arm that ties is not the same as an
arm that never answered), the confidence interval, the cache key, and the
lane-compaction that turns a corpus row into a board. A bug in any of those
produces a plausible percentage for the wrong question.
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_rank_api as E  # noqa: E402


def _rec(truth, top10, head, phase="pick1"):
    """One record with the fields `summarise` reads."""
    def at(lst, n):
        return truth in lst[:n]
    head = list(head)
    return {
        "gameid": "g", "move_idx": 10, "phase": phase, "patch": "16.20",
        "truth": truth, "top10": top10,
        "top1": top10[0] if top10 else None,
        "in1": bool(top10) and top10[0] == truth,
        "in5": at(top10, 5), "in10": at(top10, 10),
        "head_top1": head[0] if head else None,
        "head_in1": bool(head) and head[0] == truth,
        "head_in5": at(head, 5), "head_in10": at(head, 10),
        "truth_rank_head": None, "parse": "model", "stop": "answered",
        "model": "m", "seconds": 1.0, "calls": 3,
        "prompt": 100, "completion": 10,
    }


def test_summarise_counts_paired_wins_separately_from_ties():
    recs = [
        # agent finds it at 1, head misses the top-5 entirely -> agent_only
        _rec("A", ["A", "B", "C", "D", "E"], ["X", "Y", "Z", "W", "V"]),
        # head finds it, agent's list lost it -> head_only
        _rec("F", ["G", "H", "I", "J", "K"], ["F", "Y", "Z", "W", "V"]),
        # both find it -> neither column counts it
        _rec("L", ["L", "M", "N", "O", "P"], ["L", "M", "N", "O", "P"]),
        # neither -> neither
        _rec("Q", ["R", "S", "T", "U", "W"], ["R", "S", "T", "U", "W"]),
    ]
    rep = E.summarise(recs)
    assert rep["n"] == 4
    # recs 1 and 3 put the truth first in BOTH lists, rec 2 first in the head's
    # only, rec 4 in neither - so top-1 ties 2-2 while top-5 splits 1-1.
    assert rep["in1"]["hits"] == 2 and rep["head_in1"]["hits"] == 2
    d5 = rep["delta_top5"]
    assert d5["agent_only"] == 1 and d5["head_only"] == 1
    assert d5["agent_minus_head"] == pytest.approx(0.0)
    # a tie is p=1, not a small number
    assert d5["p"] == pytest.approx(1.0)
    assert rep["tokens"] == {"prompt": 400, "completion": 40, "calls": 12}


def test_summarise_reports_a_real_win_as_significant():
    # 40 rows the agent wins alone, 4 the head wins alone: p must be tiny
    recs = [_rec("A", ["A"] + ["b"] * 9, ["z"] * 10) for _ in range(40)]
    recs += [_rec("A", ["z"] * 10, ["A"] + ["b"] * 9) for _ in range(4)]
    rep = E.summarise(recs)
    assert rep["delta_top1"]["agent_only"] == 40
    assert rep["delta_top1"]["head_only"] == 4
    assert rep["delta_top1"]["p"] < 1e-6
    assert rep["in1"]["acc"] == pytest.approx(40 / 44)


def test_wilson_brackets_the_estimate_and_shrinks_with_n():
    lo, hi = E.wilson(250, 500)
    assert lo < 0.5 < hi
    lo2, hi2 = E.wilson(500, 1000)
    assert (hi2 - lo2) < (hi - lo)
    assert E.wilson(0, 0) == (0.0, 0.0)


def test_cache_key_separates_prompts_and_list_lengths():
    ex = {"gameid": "g1", "move_idx": 7}
    a = E.key_of(ex, "m", "system-a", 10)
    assert a == E.key_of(ex, "m", "system-a", 10)
    assert a != E.key_of(ex, "m", "system-b", 10)     # prompt edit
    assert a != E.key_of(ex, "m", "system-a", 5)      # different list length
    assert a != E.key_of(ex, "m", "system-a", 10) + "x"


def test_state_of_compacts_lane_indexed_picks_into_a_board():
    names = ["champ%d" % i for i in range(173)]
    champ_ids = torch.zeros(10, dtype=torch.long)
    pad = torch.ones(10, dtype=torch.bool)
    # blue's three picks sit at lane slots 0, 2, 4 - the corpus stores
    # `team * 5 + role`, so a pick can leave a hole in the middle
    for slot, idx in ((0, 1), (2, 3), (4, 5)):
        champ_ids[slot] = idx
        pad[slot] = False
    for slot, idx in ((6, 7), (8, 9)):
        champ_ids[slot] = idx
        pad[slot] = False
    sample = {
        "champ_ids": champ_ids, "pad_mask": pad,
        "ban_ids": torch.tensor([10, 11, 12, 13, 14, 15, 0, 0, 0, 0]),
        "n_bans": 6, "first_pick": torch.tensor(0),
        "meta": {"patch": "16.20"},
        "unavailable_mask": torch.zeros(173, dtype=torch.bool),
    }
    st = E._state_of(sample, names)
    board = st.board()
    assert board["blue_picks"] == ["champ1", "champ3", "champ5"]
    assert board["red_picks"] == ["champ7", "champ9"]
    assert len(board["bans"]) == 6
    assert board["first_pick"] == 0
    # `log_from_board` replays the schedule, so 6 bans + 5 picks is 11 moves -
    # the state is a log, not a 0-move board. The real harness feeds it the
    # board of a row at move m, so `move_idx` comes back as m.
    assert st.move_idx == 11
