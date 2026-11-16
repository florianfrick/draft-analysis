"""The flat-feature MLP, in one place so the trainer and the scorer agree.

Both the trainer (`train.py`) and the evaluation suite need to construct this
exact module from a checkpoint's `flat.json`. Two constructions of one
architecture is how a loader silently ends up with a differently-shaped first
layer and `load_state_dict` starts raising on a model that trained fine.

The predictor wrapper at the bottom is the interface `test_split.models` scores
through: `scores(states, candidate_lists)` returns one float array per state,
aligned with that state's candidates. That is the same interface the forest
exposes, which is what lets one ranking function serve both families.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pro.model import baselines as B

# Mirrors `policy/model.DUPLICATE_LOGIT_NEG`: the transformer masks illegal
# champions with -1e4 rather than -inf, and so does this, so "illegal" means
# the same thing numerically in both.
ILLEGAL = -1e4


class FlatMLP(nn.Module):
    """in_dim -> hidden (x depth) -> 173 logits."""

    def __init__(self, in_dim, hidden=192, n_out=B.NUM_CHAMPS, dropout=0.1,
                 depth=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build(flat):
    """`flat.json`'s arch dict -> a module with no weights loaded."""
    return FlatMLP(flat["in_dim"], hidden=flat.get("hidden", 192),
                   dropout=flat.get("dropout", 0.0),
                   depth=flat.get("depth", 2))


def params_for(in_dim, hidden, n_out=B.NUM_CHAMPS, depth=2):
    """Exact parameter count for `FlatMLP(in_dim, hidden, n_out, depth)`.

    Written out rather than measured off an instance so `largest_hidden` can
    solve for a width without building one per candidate, and so the number the
    budget check uses is the same one the artifact records.
    """
    p = in_dim * hidden + hidden                       # first layer + bias
    p += (depth - 1) * (hidden * hidden + hidden)      # hidden -> hidden
    p += hidden * n_out + n_out                        # output layer + bias
    return int(p)


def largest_hidden(in_dim, n_out=B.NUM_CHAMPS, depth=2, budget=B.PARAM_BUDGET):
    """Widest `hidden` that still fits the budget, by bisection.

    The two tasks have different input widths (3,834 vs 1,974), so one width
    cannot fill the same budget twice: `hidden` has to be solved per task or one
    arm is quietly given less capacity than the other. Bisection because
    `params_for` is monotone in `hidden` and the answer only needs to be exact
    at the boundary.
    """
    lo, hi, best = 1, 8192, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if params_for(in_dim, mid, n_out, depth) <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


class MLPPredictor:
    """Scores candidates for a state with one forward pass per state.

    The candidate axis is the model's *output*, so the candidate list only
    selects columns - it never changes the input. That is why this is one
    forward per state rather than one per (state, candidate).
    """

    def __init__(self, model, task, patch_index):
        self.model = model.eval()
        self.task = task
        self.patch_index = patch_index
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def scores(self, states, candidate_lists):
        if self.task == "next_move":
            X = B.build_next_move(states, self.patch_index)
            illegal = B.illegal_batch(states)
        else:
            X = B.build_masked(states, self.patch_index)
            illegal = None
        logits = self.model(torch.from_numpy(X).to(self.device))
        if illegal is not None:
            logits = logits.masked_fill(
                torch.from_numpy(illegal).to(self.device), ILLEGAL)
        logits = logits.cpu().numpy()
        return [logits[i, np.asarray(c, dtype=np.int64)]
                for i, c in enumerate(candidate_lists)]


def load(run_dir, flat):
    """Rebuild a predictor from a checkpoint directory + its `flat.json`."""
    run_dir = Path(run_dir)
    blob = torch.load(str(run_dir / flat["weights"]), map_location="cpu",
                      weights_only=False)
    model = build(blob["flat"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    patch_index = {k: i + 1 for i, k in enumerate(blob.get("patch_vocab") or [])}
    return MLPPredictor(model, flat["task"], patch_index), \
        sum(p.numel() for p in model.parameters())
