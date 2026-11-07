"""Tests for the ban-aware policy (branch exp/bans).

What each one pins, and what would fail silently without it:

1. **The derived ban side is the real side.** `_gone_tokens` derives which team
   made ban slot i from (slot, first_pick) instead of carrying it in the batch.
   That is what lets `swap_sides` stay untouched. If `BAN_SIDE_BY_SLOT` were
   wrong the model would be told a ban belongs to the wrong team, which is
   indistinguishable from "the side embedding does not help".
2. **Mirroring flips the sides.** The same derivation has to flip when
   `first_pick` flips, or the augmentation teaches the model that a blue-side
   ban is a red-side ban.
3. **Ban-awareness is exactly zero at initialisation.** All three switches are
   zero-initialised so that "ban-aware on/off" is one flag on one seed rather
   than two different networks. A non-zero difference here means the ablation
   is measuring an initialisation, not a mechanism.
4. **The counter matrix can actually leave zero.** Both factors zero-initialised
   is a dead product; the scheme is `a` small-random, `b` zero, so `b` receives
   gradient at step 0 and `a` does not. If that ever inverts, the term stays at
   zero forever and the branch's core change silently does nothing.
5. **The fearless pool is an input only when asked for.** With `ban_fearless`
   off, changing the pool must not move a single logit; with it on it must.
   Otherwise "the pool is visible" is a claim with no test behind it.
6. **A ban-blind checkpoint still loads.** Every pre-existing checkpoint lacks
   the new config keys; `infer.load_policy_model` must rebuild the original
   module set for them, not raise and not silently leave modules uninitialised.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
ROOT = MODEL_DIR.parents[1]
# ROOT only, and first: `model` has to resolve to the package. Putting
# solo_queue/model/policy on the path would make `model` resolve to solo_queue/model/policy/model.py,
# a module, and every `model.pretraining` import would fail.
sys.path.insert(0, str(ROOT))

from solo_queue.model.policy.model import (                                   # noqa: E402
    BAN_SIDE_BY_SLOT, FEARLESS_SLOTS, SIDE_NONE, BanInteraction,
    PolicyValueModel)
from solo_queue.model.pretraining.base_model import SiameseBase     # noqa: E402

C = 173
PAD = C + 1
N_SLOTS = 10
N_MOVES = 20
CORPUS = ROOT / "data" / "pro play" / "pro_drafts.pt"


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DATASETS = _load_by_path("ban_datasets", ROOT / "data" / "pro play"
                         / "datasets.py")


def tiny_encoder(d=32):
    return SiameseBase(num_champs=C, embed_dim=d, num_heads=4,
                                num_intra_layers=1, num_cross_layers=1)


def tiny_model(d=32, **kw):
    return PolicyValueModel(tiny_encoder(d), embed_dim=d, state_dim=16,
                            n_econ=0, with_value=False, **kw)


def make_batch(B=4, seed=0, n_bans=3, pool=0):
    """A ProNextMoveDataset-shaped batch."""
    g = torch.Generator().manual_seed(seed)
    champ_ids = torch.full((B, N_SLOTS), PAD, dtype=torch.long)
    role_ids = torch.full((B, N_SLOTS), 5, dtype=torch.long)
    pad_mask = torch.ones(B, N_SLOTS, dtype=torch.bool)
    ban_ids = torch.full((B, 10), PAD, dtype=torch.long)
    for b in range(B):
        champ_ids[b, 0] = int(torch.randint(0, C, (1,), generator=g).item())
        role_ids[b, 0] = 0
        pad_mask[b, 0] = False
        taken = set()
        for k in range(n_bans):
            v = int(torch.randint(0, C, (1,), generator=g).item())
            while v in taken:
                v = int(torch.randint(0, C, (1,), generator=g).item())
            taken.add(v)
            ban_ids[b, k] = v
    fids = torch.full((B, FEARLESS_SLOTS), PAD, dtype=torch.long)
    for b in range(B):
        if pool:
            fids[b, :pool] = torch.arange(pool) + 100
    taken_mask = torch.zeros(B, C, dtype=torch.bool)
    for b in range(B):
        for v in ban_ids[b].tolist():
            if v != PAD:
                taken_mask[b, v] = True
        taken_mask[b, champ_ids[b, 0]] = True
    return {
        "champ_ids": champ_ids, "role_ids": role_ids, "pad_mask": pad_mask,
        "ban_ids": ban_ids, "taken_mask": taken_mask,
        "unavailable_mask": torch.zeros(B, C, dtype=torch.bool),
        "fearless_ids": fids,
        "fearless_role_ids": torch.full_like(fids, 5),
        "move_idx": torch.full((B,), 2, dtype=torch.long),
        "phase": torch.zeros(B, dtype=torch.long),
        "to_move": torch.zeros(B, dtype=torch.long),
        "first_pick": torch.zeros(B, dtype=torch.long),
        "is_ban": torch.ones(B, dtype=torch.long),
        "patch_id": torch.zeros(B, dtype=torch.long),
    }


# ------------------------------------------------------------------ side ----
def test_ban_side_matches_the_real_corpus():
    """The derived side must be the side that actually made the ban.

    Checked on the whole corpus rather than a sample: `build_prefix` fills
    `ban_ids` in chronological order and the interleave is fixed, so this is
    an exact identity and there is no reason to accept a sampling error.
    """
    if not CORPUS.exists():
        pytest.skip("pro corpus cache not present")
    rows = torch.load(str(CORPUS), map_location="cpu",
                      weights_only=False)["rows"]
    mismatches = 0
    checked = 0
    for row in rows:
        sides = DATASETS.ban_sides(row["first_pick"]).tolist()
        n = 0
        for kind, team, _champ, _role in row["moves"]:
            if kind != DATASETS.MOVE_BAN:
                continue
            checked += 1
            if sides[n] != team:
                mismatches += 1
            n += 1
    assert checked == 10 * len(rows)
    assert mismatches == 0, "%d of %d ban slots landed on the wrong team" % (
        mismatches, checked)


def test_ban_side_table_and_dataset_helper_agree():
    """The model's table and the data layer's helper must not drift."""
    assert tuple(DATASETS.ban_sides(0).tolist()) == tuple(BAN_SIDE_BY_SLOT)
    assert tuple(DATASETS.ban_sides(1).tolist()) == \
        tuple(1 - s for s in BAN_SIDE_BY_SLOT)


def test_mirroring_flips_the_ban_sides():
    """swap_sides must flip every ban's side, via first_pick alone.

    `swap_sides` deliberately does not permute `ban_ids`; the sides flip
    because they are derived from `first_pick`. This is the test that keeps
    that argument honest - if it ever fails, the mirror augmentation is
    teaching the model that the enemy's ban is its own.
    """
    model = tiny_model(d=32, ban_side=True)
    batch = make_batch(B=2, n_bans=6)
    batch["first_pick"] = torch.tensor([0, 1])

    def sides_of(b):
        _e, _p, _k, side_ids, _r = model._gone_tokens(b)
        return side_ids

    before = sides_of(batch)
    mirrored = dict(batch)
    mirrored["first_pick"] = 1 - batch["first_pick"]
    after = sides_of(mirrored)

    real = before != SIDE_NONE
    assert real.any()
    assert torch.equal(after[real], 1 - before[real])
    # PAD slots stay "no side" in both directions.
    assert torch.equal(before[~real], after[~real])


# ----------------------------------------------------------- zero-init ------
def test_side_and_counter_are_exactly_zero_at_init():
    """`ban_side` and `counter_rank` must leave the logits where they were.

    Measured with an EMPTY pool on purpose. `ban_fearless` is not in this
    claim and could not be: its parameters are zero-initialised, but the pool
    tokens are real champion embeddings, so they join the attention key set and
    change the forward pass at step 0 regardless of the kind embedding. That is
    the intended behaviour (the pool *is* an input), not a bug - see
    `test_pool_is_visible_when_the_switch_is_on`.
    """
    torch.manual_seed(0)
    enc = tiny_encoder(32)
    blind = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                             with_value=False)
    aware = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                             with_value=False, ban_side=True,
                             counter_rank=8)
    own = set(aware.state_dict())
    aware.load_state_dict({k: v for k, v in blind.state_dict().items()
                           if k in own}, strict=False)
    blind.eval()
    aware.eval()
    batch = make_batch(B=3, n_bans=4, pool=0)
    with torch.no_grad():
        a = blind(batch)["policy_logits"]
        b = aware(batch)["policy_logits"]
    # ~1e-7 of float32 noise from the extra zero additions; not a signal.
    assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())


def test_pool_changes_the_forward_pass_at_init_by_construction():
    """`ban_fearless` is an input, so it moves the logits before any training.

    Pinned because it is the one switch whose effect is *not* deferred: if
    adding pool tokens left the logits untouched, the tokens would be reaching
    nothing and the whole fearless arm would be measuring a no-op.
    """
    torch.manual_seed(0)
    enc = tiny_encoder(32)
    model = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                             with_value=False, ban_fearless=True)
    model.eval()
    empty = make_batch(B=2, n_bans=2, pool=0)
    with_pool = make_batch(B=2, n_bans=2, pool=0)
    with_pool["fearless_ids"] = torch.full((2, FEARLESS_SLOTS), 100,
                                           dtype=torch.long)
    with torch.no_grad():
        a = model(empty)["policy_logits"]
        b = model(with_pool)["policy_logits"]
    assert not torch.allclose(a, b, atol=1e-4)


def test_counter_gets_gradient_in_b_only_at_first_step():
    """`a` small-random + `b` zero: the product is zero but b moves first.

    Zero-initialising both factors would be a dead product - the term would
    stay at exactly 0 for every step and the branch's core change would be a
    no-op that still "runs".
    """
    torch.manual_seed(0)
    inter = BanInteraction(embed_dim=16, rank=4)
    gone = torch.randn(2, 3, 16)
    pad = torch.zeros(2, 3, dtype=torch.bool)
    g = inter.summary(gone, pad)
    champ_w = torch.randn(C, 16, requires_grad=False)
    out = inter(champ_w, g)
    assert torch.equal(out, torch.zeros_like(out))
    out.sum().backward()
    assert inter.b.grad is not None and inter.b.grad.abs().sum() > 0
    assert inter.a.grad is None or inter.a.grad.abs().sum() == 0


def test_counter_becomes_nonzero_after_a_step():
    """One SGD step on `b` must make the term non-zero and candidate-specific."""
    torch.manual_seed(0)
    inter = BanInteraction(embed_dim=16, rank=4)
    gone = torch.randn(2, 3, 16)
    pad = torch.zeros(2, 3, dtype=torch.bool)
    champ_w = torch.randn(C, 16)
    with torch.no_grad():
        inter.b.add_(0.1)      # stand in for the first optimiser step
    out = inter(champ_w, inter.summary(gone, pad))
    assert out.abs().max() > 0
    # Different candidates move by different amounts - that is the whole
    # point (a per-candidate term, not a pooled shift).
    assert out[0].std() > 1e-4


# ------------------------------------------------------------ fearless ------
def test_pool_is_invisible_when_the_switch_is_off():
    """With ban_fearless off the pool must not move a single logit."""
    torch.manual_seed(0)
    model = tiny_model(d=32)
    model.eval()
    empty = make_batch(B=2, n_bans=2, pool=0)
    full = make_batch(B=2, n_bans=2, pool=0)
    full["fearless_ids"] = torch.full((2, FEARLESS_SLOTS), 100,
                                      dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(model(empty)["policy_logits"],
                           model(full)["policy_logits"])


def test_pool_is_visible_when_the_switch_is_on():
    """With ban_fearless on, the pool must change the logits."""
    torch.manual_seed(0)
    model = tiny_model(d=32, ban_fearless=True)
    # A trained-looking kind embedding: zero-init would make this pass
    # trivially. Perturb so the test measures the wiring, not the init.
    with torch.no_grad():
        model.ban_context.kind_embed.weight.normal_(0, 0.1)
        model.state_embed.pool_proj.weight.normal_(0, 0.1)
    model.eval()
    empty = make_batch(B=2, n_bans=2, pool=0)
    full = make_batch(B=2, n_bans=2, pool=0)
    full["fearless_ids"] = torch.full((2, FEARLESS_SLOTS), 100,
                                      dtype=torch.long)
    with torch.no_grad():
        a = model(empty)["policy_logits"]
        b = model(full)["policy_logits"]
    assert not torch.allclose(a, b), "the pool reached nothing"


def test_pool_ids_are_sorted_and_padded():
    """The dataset's pool must be ascending ids then PAD, or the kind
    embedding sees an order-dependent token sequence for no reason."""
    rows = [{
        "champ": list(range(10)), "role": list(range(5)) * 2,
        "blue_win": 1, "weight": 1.0, "has_econ": False, "econ": None,
        "gameid": "g2", "series_id": "s1", "game_number": 2,
        "league": "LCK", "patch": "16.01", "date": "2025-01-01",
        "blue_team": "A", "red_team": "B", "first_pick": 0,
        "unavailable": [3, 7, 1],
        "moves": [(0, 0, 50 + i, -100) for i in range(6)]
                 + [(1, 0, 60 + i, i) for i in range(6)]
                 + [(0, 1, 70 + i, -100) for i in range(4)]
                 + [(1, 1, 80 + i, i) for i in range(4)],
    }]
    s = DATASETS.build_prefix(rows[0], 0, C, None, {})
    ids = s["fearless_ids"]
    assert ids.shape[0] == FEARLESS_SLOTS
    real = ids[ids != PAD].tolist()
    assert real == [1, 3, 7]
    assert (ids[len(real):] == PAD).all()


# -------------------------------------------------------------- serving -----
def test_serving_batch_carries_the_pool():
    """`_policy_batch` must turn the board's pool mask into tokens.

    Without this the model would be ban-aware in training and pool-blind at
    serving - the exact skew the `ban_fearless` docstring warns about.
    """
    from solo_queue.model.policy import infer

    board = infer.build_board([None] * 10, [5, 6], C, first_pick=0,
                              unavailable=[11, 12, 13])
    batch = infer._policy_batch(board, 2, infer.MOVE_BAN, 0)
    ids = batch["fearless_ids"]
    assert ids.shape == (1, FEARLESS_SLOTS)
    assert ids[0, ids[0] != PAD].tolist() == [11, 12, 13]
    assert (ids[0, 3:] == PAD).all()


def test_ban_blind_checkpoint_still_loads(tmp_path):
    """A checkpoint with none of the new config keys must rebuild unchanged."""
    from solo_queue.model.policy import infer

    torch.manual_seed(0)
    enc = tiny_encoder(32)
    model = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                             with_value=False)
    cfg = {"num_champs": C, "embed_dim": 32, "num_intra": 1, "num_cross": 1,
           "num_heads": 4, "dropout": 0.0, "state_dim": 16, "n_econ": 0,
           "with_value": False, "n_patches": 0}
    path = tmp_path / "blind.pt"
    torch.save({"model": model.state_dict(), "config": cfg,
                "patch_vocab": []}, str(path))
    loaded, out_cfg = infer.load_policy_model(str(path))
    assert loaded.counter is None
    assert loaded.ban_context.side_embed is None
    assert loaded.state_embed.pool_proj is None
    assert out_cfg["embed_dim"] == 32


def test_ban_aware_checkpoint_round_trips(tmp_path):
    """The flags must come back out of the checkpoint, not be defaulted."""
    from solo_queue.model.policy import infer

    torch.manual_seed(0)
    enc = tiny_encoder(32)
    model = PolicyValueModel(enc, embed_dim=32, state_dim=16, n_econ=0,
                             with_value=False, ban_side=True,
                             ban_fearless=True, counter_rank=4)
    cfg = {"num_champs": C, "embed_dim": 32, "num_intra": 1, "num_cross": 1,
           "num_heads": 4, "dropout": 0.0, "state_dim": 16, "n_econ": 0,
           "with_value": False, "n_patches": 0, "ban_side": True,
           "ban_fearless": True, "counter_rank": 4}
    path = tmp_path / "aware.pt"
    torch.save({"model": model.state_dict(), "config": cfg,
                "patch_vocab": []}, str(path))
    loaded, _ = infer.load_policy_model(str(path))
    assert loaded.counter is not None
    assert loaded.ban_context.side_embed is not None
    assert loaded.state_embed.pool_proj is not None
