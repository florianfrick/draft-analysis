"""`pool_order="to_move"` is a reordering, and nothing else.

The switch exists because `to_move` is redundant with the board and
`swap_aug` turns it into a flag rather than a position, so the head never
learned it (a side flip moves the lane distribution by TV 0.015 against 0.147
for one enemy pick changing). Putting "my team" first in the pooled vector
removes the XOR the head would otherwise have to learn.

What has to be true, and is pinned here:

1. The default is `"absolute"`, so every checkpoint written before the flag
   existed rebuilds the function it was trained as.
2. `"absolute"` does not depend on `to_move` at all.
3. `"to_move"` only moves blocks: the two team halves swap, the difference
   flips sign, the product is unchanged. No shape change, no parameters.
4. The point of the exercise - a board and its side-swapped twin produce the
   same "me" vector when each is pooled with its own `to_move`. If that fails,
   the flag buys nothing.
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
PRO_DIR = MODEL_DIR.parents[1] / "data" / "pro play"
for p in (str(HERE), str(MODEL_DIR), str(PRO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from solo_queue.model import PolicyValueModel  # noqa: E402
from solo_queue.model.pretraining.base_model import SiameseBase  # noqa: E402

D = 32
B = 3


def _model(pool_order):
    torch.manual_seed(0)
    enc = SiameseBase(num_champs=173, embed_dim=D, num_heads=4,
                      num_intra_layers=1, num_cross_layers=1)
    m = PolicyValueModel(enc, embed_dim=D, state_dim=16, n_econ=0,
                         with_value=False, pool_order=pool_order)
    return m.eval()


def _h(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 10, D, generator=g)


PAD = torch.zeros(B, 10, dtype=torch.bool)


def test_default_is_absolute_and_unknown_names_raise():
    """A checkpoint with no `pool_order` key must rebuild the old function."""
    assert _model("absolute").pool_order == "absolute"
    torch.manual_seed(0)
    enc = SiameseBase(num_champs=173, embed_dim=D, num_heads=4,
                      num_intra_layers=1, num_cross_layers=1)
    assert PolicyValueModel(enc, embed_dim=D, state_dim=16,
                            with_value=False).pool_order == "absolute"
    with pytest.raises(ValueError):
        PolicyValueModel(enc, embed_dim=D, state_dim=16, with_value=False,
                         pool_order="me-first")


def test_absolute_ignores_to_move():
    """The old pooling must be bit-identical however the batch is labelled."""
    m = _model("absolute")
    h = _h()
    with torch.no_grad():
        a = m._pool(h, PAD, torch.zeros(B, dtype=torch.long))
        b = m._pool(h, PAD, torch.ones(B, dtype=torch.long))
        c = m._pool(h, PAD)                       # no to_move at all
    assert torch.equal(a, b) and torch.equal(a, c)


def test_to_move_only_moves_blocks():
    """Halves swap, the difference flips sign, the product is unchanged."""
    m = _model("to_move")
    h = _h()
    with torch.no_grad():
        base = m._pool(h, PAD, torch.zeros(B, dtype=torch.long))
        flip = m._pool(h, PAD, torch.ones(B, dtype=torch.long))
    assert base.shape == flip.shape == (B, 4 * D)
    assert torch.allclose(flip[:, :D], base[:, D:2 * D])
    assert torch.allclose(flip[:, D:2 * D], base[:, :D])
    assert torch.allclose(flip[:, 2 * D:3 * D], -base[:, 2 * D:3 * D])
    assert torch.allclose(flip[:, 3 * D:], base[:, 3 * D:])


def test_a_side_swap_with_its_own_to_move_pools_the_same_me_vector():
    """The reason the flag exists: 'my team' is a position, not a label.

    `swap_sides` swaps the two halves of the board and flips `to_move`. If the
    pooling is me-first, the swapped board must produce exactly the vector the
    original did - otherwise the head still has to learn the XOR.
    """
    m = _model("to_move")
    h = _h()
    h_swapped = torch.cat([h[:, 5:], h[:, :5]], dim=1)
    with torch.no_grad():
        a = m._pool(h, PAD, torch.zeros(B, dtype=torch.long))
        b = m._pool(h_swapped, PAD, torch.ones(B, dtype=torch.long))
    assert torch.allclose(a, b, atol=1e-6)

    # ...and the same pair under "absolute" does NOT, which is the bug.
    old = _model("absolute")
    with torch.no_grad():
        a = old._pool(h, PAD, torch.zeros(B, dtype=torch.long))
        b = old._pool(h_swapped, PAD, torch.ones(B, dtype=torch.long))
    assert not torch.allclose(a, b, atol=1e-6)
