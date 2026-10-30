"""Tests for the ban-decision scorer.

The load-bearing claims in `ban_search.py` are not empirical, they are
structural, so they can be pinned without any data:

  * under a symmetric additive V, a ban has *identical* value for every
    candidate whenever both teams need the same roles - the score is flat, and
    "which ban is best" is not a question the value function can answer;
  * a global (role-free) additive V is flat in *both* phases, because every role
    sees the same champion ordering;
  * role structure is what creates a decision at all, and it does so exactly
    when the two teams' remaining roles differ.

If these tests pass, the flat numbers in the report are a result, not a bug.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ban_search as B  # noqa: E402


NUM_CHAMPS = 8


def _W():
    """Role strength where champion i is best in role i (i < 5).

    Role 0: c0 0.10, c5 0.05
    Role 1: c1 0.08, c6 0.04
    Role 2: c2 0.06, c7 0.03
    Role 3: c3 0.04, c0 0.01
    Role 4: c4 0.02, c1 0.01
    """
    W = np.zeros((B.NUM_ROLES, NUM_CHAMPS))
    W[0, 0], W[0, 5] = 0.10, 0.05
    W[1, 1], W[1, 6] = 0.08, 0.04
    W[2, 2], W[2, 7] = 0.06, 0.03
    W[3, 3], W[3, 0] = 0.04, 0.01
    W[4, 4], W[4, 1] = 0.02, 0.01
    return W


def _state(filled_blue=(), filled_red=(), taken=()):
    taken = np.zeros(NUM_CHAMPS, dtype=bool)
    for c in taken:
        taken[c] = True
    return B.DraftState(taken, (frozenset(filled_blue), frozenset(filled_red)),
                        0.0, 0.0)


# ---------------------------------------------------------------- completion
def test_completion_picks_best_per_role_and_falls_back_to_second():
    W = _W()
    st = _state()
    base_b, base_r, loss_b, loss_r = B.completion_values(W, st)
    # every role takes its best champion
    assert base_b == pytest.approx(0.10 + 0.08 + 0.06 + 0.04 + 0.02)
    assert base_r == pytest.approx(base_b)      # same pool, same needs
    # banning c0 costs exactly the gap to role 0's second best
    assert loss_b[0] == pytest.approx(0.05)
    assert loss_r[0] == pytest.approx(0.05)
    # a champion that is nobody's best costs nothing
    assert loss_b[6] == pytest.approx(0.0)


def test_completion_only_counts_unfilled_roles():
    W = _W()
    st = _state(filled_blue=(0, 1), filled_red=(0, 1))
    base_b, _, _, _ = B.completion_values(W, st)
    assert base_b == pytest.approx(0.06 + 0.04 + 0.02)


def test_flex_conflict_is_reported_not_hidden():
    """c0 is the best for role 0 and the second best for role 3, so the
    per-role greedy can count it in two places. The rate is reported rather
    than silently absorbed."""
    W = _W()
    W[3, 0] = 0.09          # now c0 is the best for role 3 as well
    st = _state()
    assert B.double_claim_rate(W, st) > 0.0


# ------------------------------------------------------- the flat theorem
def test_score_is_flat_when_both_teams_need_the_same_roles():
    """The ban-phase-1 result: no picks on the board means both teams want the
    same five roles out of the same pool, so every ban removes the same value
    from each side and the decision is unidentifiable."""
    W = _W()
    st = _state()
    legal = np.arange(NUM_CHAMPS)
    p = B.ban_probabilities(W, 0.0, st, legal, side=0)
    assert p.max() - p.min() == pytest.approx(0.0, abs=1e-12)


def test_global_weights_are_flat_even_with_different_role_needs():
    """The control: a role-free V orders champions identically everywhere, so it
    stays flat even in ban phase 2.

    The qualification matters: this holds when both teams have the *same number*
    of roles left, which a pro draft guarantees at every ban (both sides have
    made the same number of picks). With unequal counts even a global V moves,
    because a ban removes k times the same gap for k remaining roles - that is a
    property of the draft, not a decision.
    """
    W = np.zeros((B.NUM_ROLES, NUM_CHAMPS))
    w_global = np.array([0.10, 0.08, 0.06, 0.04, 0.02, 0.0, 0.0, 0.0])
    for r in range(B.NUM_ROLES):
        W[r] = w_global
    # blue needs {3,4}, red needs {0,1}: two roles each, different roles
    st = _state(filled_blue=(0, 1, 2), filled_red=(2, 3, 4))
    legal = np.arange(NUM_CHAMPS)
    p = B.ban_probabilities(W, 0.0, st, legal, side=0)
    assert p.max() - p.min() == pytest.approx(0.0, abs=1e-12)


def test_role_weights_are_not_flat_when_role_needs_differ():
    """Role structure is what creates the decision: here blue still needs the
    cheap roles while red needs the expensive ones, so denying red's best
    champion is worth more than losing blue's own."""
    W = _W()
    # blue needs {3,4}, red needs {0,1}
    st = _state(filled_blue=(0, 1, 2), filled_red=(2, 3, 4))
    legal = np.arange(NUM_CHAMPS)
    p = B.ban_probabilities(W, 0.0, st, legal, side=0)
    assert p.max() - p.min() > 1e-6
    # blue's best ban is red's most valuable needed champion, c0
    assert int(np.argmax(p)) == 0


def test_ban_probability_is_reported_from_the_banning_side():
    W = _W()
    st = _state(filled_blue=(0, 1, 2), filled_red=(2, 3, 4))
    legal = np.arange(NUM_CHAMPS)
    pb = B.ban_probabilities(W, 0.0, st, legal, side=0)
    pr = B.ban_probabilities(W, 0.0, st, legal, side=1)
    assert np.allclose(pb + pr, 1.0)
    # the same ban cannot be best for both sides
    assert int(np.argmax(pb)) != int(np.argmax(pr))


# ------------------------------------------------------------- state replay
def _row():
    # 1 ban (c7), then blue picks c0 top, red picks c1 jng
    return {"moves": [(B.MOVE_BAN, 0, 7, -100),
                      (B.MOVE_PICK, 0, 0, 0),
                      (B.MOVE_PICK, 1, 1, 1)],
            "unavailable": [5]}


def test_state_at_replays_bans_picks_and_fearless():
    W = _W()
    st = B.state_at(_row(), 1, W)
    assert st.taken[7] and not st.taken[0]
    assert st.filled[0] == frozenset() and st.filled[1] == frozenset()

    st2 = B.state_at(_row(), 3, W)
    assert st2.taken[7] and st2.taken[0] and st2.taken[1]
    assert st2.taken[5]                       # fearless-unavailable
    assert st2.filled[0] == frozenset({0})
    assert st2.filled[1] == frozenset({1})
    assert st2.blue_sum == pytest.approx(0.10)
    assert st2.red_sum == pytest.approx(0.08)


# --------------------------------------------------------------- statistics
def _rec(p_obs, p_freq, p_random, p_best, spread):
    return {"p_obs": p_obs, "p_freq": p_freq, "p_random": p_random,
            "p_best": p_best, "regret_obs": p_best - p_obs,
            "regret_freq": p_best - p_freq, "regret_random": p_best - p_random,
            "spread": spread, "rank_obs": 0, "rank_freq": 0,
            "obs_is_best": p_obs >= p_best, "freq_is_best": p_freq >= p_best,
            "obs_eq_freq": p_obs == p_freq}


def test_summarise_flags_a_flat_phase():
    flat = [_rec(0.5, 0.5, 0.5, 0.5, 0.0) for _ in range(10)]
    s = B.summarise(flat)
    assert s["flat"] is True and s["spread"] == 0.0


def test_summarise_paired_edge_detects_a_real_advantage():
    """edge_obs is a per-decision difference, so it must not be computed as a
    difference of two means - it needs its own SE."""
    good = [_rec(0.52, 0.50, 0.50, 0.52, 0.04) for _ in range(200)]
    s = B.summarise(good)
    assert s["flat"] is False
    assert s["edge_obs"] == pytest.approx(0.02)
    assert s["edge_obs_z"] > 10          # n=200 with zero variance

    avg = [_rec(0.50, 0.50, 0.50, 0.52, 0.04) for _ in range(200)]
    assert B.summarise(avg)["edge_obs"] == pytest.approx(0.0)
