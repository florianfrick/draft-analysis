"""The forest as a candidate scorer.

Why this is not a 173-class classifier
-------------------------------------
`RandomForestClassifier` over 173 labels is not merely slow here, it is
infeasible: scikit-learn's split criterion carries a class histogram and pays
`O(n_samples_at_node x n_classes)` *per feature tried*. At the canonical split's
794k next-move training states that is ~1.4e8 operations for a single level of
a single tree at the root alone, and a forest is hundreds of trees x ~15 levels.
Measured in the same order of magnitude as hours per tree, not per forest.

So the forest answers the same question the transformer answers, but in the
form trees are actually good at: score one (state, candidate) pair, then rank.
That is a standard tree-based recommender and it produces exactly the ranking
the suite measures - top-1 / top-5 / mean rank over the legal candidates - so
the number is comparable even though the architecture is not.

The cost of that choice, stated plainly: the forest's ranking is over
`baselines.candidates_for`, the legal set, which is the same set the policy
model's `-inf` masking leaves. The masked task applies no filter at all, in
both families, because `score_masked` does not.

`params`
--------
A tree ensemble has no weight vector, so `params` is the total node count
across all trees. It is the closest honest analogue: the thing that grows when
the model gets more capacity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pro.model import baselines as B


class ForestScorer:
    """`scores(states, candidate_lists)` - one float array per state.

    Chunked by state, not by row, so the per-state board read happens once and
    the (state, candidate) rows are built in one vectorised pass per chunk. A
    whole test set at once would be 7.8 M rows x 548 floats = 17 GB.
    """

    def __init__(self, estimator, task, patch_index, prior=None, chunk=96):
        self.estimator = estimator
        self.task = task
        self.patch_index = patch_index
        self.prior = B.role_prior() if prior is None else prior
        self.chunk = chunk
        classes = list(getattr(estimator, "classes_", [0, 1]))
        self.pos = classes.index(1) if 1 in classes else 0

    def _build(self, states, candidate_lists):
        if self.task == "next_move":
            return B.build_forest_next_move(states, candidate_lists,
                                            self.patch_index, prior=self.prior)
        return B.build_forest_masked(states, candidate_lists, self.patch_index,
                                     prior=self.prior)

    def scores(self, states, candidate_lists):
        out = []
        for b in range(0, len(states), self.chunk):
            st = states[b:b + self.chunk]
            cl = candidate_lists[b:b + self.chunk]
            X = self._build(st, cl)
            p = self.estimator.predict_proba(X)[:, self.pos]
            i = 0
            for cands in cl:
                n = len(cands)
                out.append(p[i:i + n])
                i += n
        return out

    @property
    def params(self):
        return int(sum(t.tree_.node_count for t in self.estimator.estimators_))


def n_nodes(estimator):
    return int(sum(t.tree_.node_count for t in estimator.estimators_))


def load(run_dir, flat):
    """Rebuild a scorer from a checkpoint directory + its `flat.json`."""
    import joblib

    run_dir = Path(run_dir)
    blob = joblib.load(str(run_dir / flat["weights"]))
    patch_index = {k: i + 1 for i, k in enumerate(blob.get("patch_vocab") or [])}
    scorer = ForestScorer(blob["estimator"], flat["task"], patch_index)
    return scorer, scorer.params
