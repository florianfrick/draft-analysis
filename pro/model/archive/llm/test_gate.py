"""Tests for `gate`'s arithmetic, not for the head.

The measured claim is "a linear rule on the head's own output buys +0.5 pp".
That number is only worth anything if the rule is fitted and scored correctly,
so what is pinned here is the machinery: the label extraction, the argmax over
candidates, the margin rule's tuning, and the paired statistic. A bug in any of
them produces a plausible percentage for the wrong question.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

import gate as G  # noqa: E402


def _rows(probs, targets, cands=None):
    """The minimal `acc` dict `_labels` reads."""
    probs = np.asarray(probs, dtype=np.float32)
    cands = np.asarray(cands if cands is not None
                       else [list(range(probs.shape[1]))] * len(probs))
    return {"feat": [np.zeros((probs.shape[1], len(G.F_NAMES)),
                              dtype=np.float32) for _ in probs],
            "probs": list(probs), "cand": list(cands),
            "target": list(targets), "is_ban": [0] * len(probs),
            "move_idx": [7] * len(probs), "gameid": ["g"] * len(probs),
            "phase": ["pick1"] * len(probs)}


def test_labels_mark_a_row_whose_truth_is_off_the_shortlist():
    acc = _rows([[0.3, 0.2, 0.1], [0.4, 0.3, 0.2]],
                targets=[7, 99], cands=[[5, 6, 7], [5, 6, 7]])
    _, y = G._labels(acc)
    assert y.tolist() == [2, -1]        # 7 is the third candidate; 99 is absent


def test_fit_learns_to_promote_a_rank_the_logp_disagrees_with():
    # Flat probabilities, truth always at index 1: only the rank offset can
    # explain it, so a correct fit must beat the head's own #1 on every row.
    n, k = 200, 5
    probs = np.full((n, k), 1.0 / k, dtype=np.float32)
    acc = _rows(probs, targets=[1] * n)
    for i in range(n):
        acc["feat"][i] = G.row_features(list(range(k)), probs[i],
                                        [None] * 10, 0, 0.0, {r: set() for r in range(5)})
    X, y = G._labels(acc)
    # the features must actually vary by rank, or the test proves nothing
    assert X[0, 1, G.F_NAMES.index("rank1")] == 1.0
    w = G.fit_softmax(X, y, G.MODEL_COLS["rank"], weight_decay=0.0)
    pred = G.predict(X, w, G.MODEL_COLS["rank"])
    assert (pred == 1).mean() == 1.0
    assert w[0] == pytest.approx(0.0, abs=0.2)     # logp carries no signal


def test_margin_rule_switches_only_under_the_threshold():
    probs = np.asarray([[0.30, 0.10], [0.20, 0.19], [0.20, 0.05]])
    # margins 0.20 / 0.01 / 0.15: the rule is strict, so tau=0 switches nothing
    assert G.margin_rule(probs, 0.0).tolist() == [0, 0, 0]
    assert G.margin_rule(probs, 0.10).tolist() == [0, 1, 0]
    assert G.margin_rule(probs, 0.50).tolist() == [1, 1, 1]


def test_tune_margin_finds_the_threshold_the_labels_imply():
    # rows 0-49: big margin, keep is right. rows 50-99: tiny margin, switch is
    # right. Anything under 0.2 keeps the first group and switches the second.
    probs = np.asarray([[0.4, 0.1]] * 50 + [[0.2, 0.19]] * 50)
    y = np.asarray([0] * 50 + [1] * 50)
    tau, acc = G.tune_margin(probs, y)
    assert 0.0 <= tau <= 0.2
    assert acc == pytest.approx(1.0)
    assert (G.margin_rule(probs, tau) == y).all()


def test_mcnemar_calls_a_tie_a_tie():
    assert G.mcnemar(1, 1) == pytest.approx(1.0)
    assert G.mcnemar(0, 0) == pytest.approx(1.0)
    assert G.mcnemar(40, 4) < 1e-6
    assert G.mcnemar(3, 7) == pytest.approx(0.34375, abs=1e-6)


def test_paired_reports_the_discordant_counts_not_just_the_gap():
    y = np.asarray([0, 0, 0, 1, 1])
    pred = np.asarray([0, 0, 1, 1, 0])
    base = np.asarray([0, 1, 0, 0, 0])
    out = G.paired(pred, base, y)
    assert out["acc"] == pytest.approx(0.6)
    assert out["base_acc"] == pytest.approx(0.4)
    assert out["a_only"] == 2 and out["b_only"] == 1


def test_open_roles_shrinks_as_a_side_picks():
    legal = G._legal_by_role()
    assert len(legal) == 5 and all(legal[r] for r in range(5))
    empty = [None] * 10
    assert G._open_roles(empty, 0) == [0, 1, 2, 3, 4]
    # one pick on blue: whatever lane it claims is no longer open, and the
    # other four are - the count is the assertion, not the identity, because
    # the table decides the lane and a test should not restate it.
    picks = [1, None, None, None, None, None, None, None, None, None]
    assert len(G._open_roles(picks, 0)) == 4
    assert len(G._open_roles(picks, 1)) == 5      # red has picked nothing
