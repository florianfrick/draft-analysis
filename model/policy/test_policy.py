"""Tests for the policy-value model (Stage 2).

Run: pytest model/policy/test_policy.py -q

Uses a tiny synthetic encoder (no checkpoint IO) for speed.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from soloq_model import SoloQPretrainedModel, NUM_REAL_CHAMPS  # noqa: E402
from model import PolicyValueModel  # noqa: E402

C = NUM_REAL_CHAMPS  # 173
PAD = C + 1
MASK = C
N_MOVES = 20
N_SLOTS = 10


def tiny_encoder(d=32):
    return SoloQPretrainedModel(
        num_champs=C, embed_dim=d, num_heads=4,
        num_intra_layers=1, num_cross_layers=1)


def tiny_model(d=32, n_econ=6):
    enc = tiny_encoder(d)
    return PolicyValueModel(enc, embed_dim=d, state_dim=16,
                            n_econ=n_econ)


def make_next_move_batch(B=4, seed=0):
    """Synthetic ProNextMoveDataset-style batch."""
    g = torch.Generator().manual_seed(seed)
    # Partial draft: some slots filled, some PAD
    champ_ids = torch.full((B, N_SLOTS), PAD, dtype=torch.long)
    role_ids = torch.full((B, N_SLOTS), 5, dtype=torch.long)  # FLEX
    pad_mask = torch.ones(B, N_SLOTS, dtype=torch.bool)

    # Fill 0-4 picks per draft (random)
    for b in range(B):
        n_picks = int(torch.randint(0, N_SLOTS, (1,), generator=g).item())
        perm = torch.randperm(N_SLOTS, generator=g)[:n_picks]
        champs = torch.randperm(C, generator=g)[:n_picks]
        champ_ids[b, perm] = champs
        role_ids[b, perm] = torch.randint(0, 5, (n_picks,), generator=g)
        pad_mask[b, perm] = False

    # Bans: 0-10 bans, PAD-padded
    ban_ids = torch.full((B, 10), PAD, dtype=torch.long)
    n_bans = int(torch.randint(0, 6, (1,), generator=g).item())
    ban_champs = torch.randperm(C, generator=g)[:n_bans]
    ban_ids[:, :n_bans] = ban_champs[:B].unsqueeze(1)  # same bans for simplicity

    # Target: a champion NOT in the taken set
    taken_mask = torch.zeros(B, C, dtype=torch.bool)
    unavailable_mask = torch.zeros(B, C, dtype=torch.bool)
    for b in range(B):
        for j in range(N_SLOTS):
            if not pad_mask[b, j]:
                taken_mask[b, champ_ids[b, j]] = True
        for j in range(10):
            if ban_ids[b, j] < C:
                taken_mask[b, ban_ids[b, j]] = True

    target_champ = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        for c in range(C):
            if not taken_mask[b, c]:
                target_champ[b] = c
                break

    return {
        "champ_ids": champ_ids,
        "role_ids": role_ids,
        "pad_mask": pad_mask,
        "ban_ids": ban_ids,
        "taken_mask": taken_mask,
        "unavailable_mask": unavailable_mask,
        "move_idx": torch.randint(0, N_MOVES, (B,), generator=g),
        "phase": torch.randint(0, 4, (B,), generator=g),
        "to_move": torch.randint(0, 2, (B,), generator=g),
        "first_pick": torch.randint(0, 2, (B,), generator=g),
        "is_ban": torch.randint(0, 2, (B,), generator=g),
        "target_champ": target_champ,
        "target_role": torch.full((B,), -100, dtype=torch.long),
        "win_label": torch.randint(0, 2, (B,), generator=g).long(),
        "weight": torch.ones(B),
    }


def test_forward_shapes_and_finite():
    """Policy logits [B, C], value logit [B, 1], econ [B, n_econ]."""
    m = tiny_model()
    b = make_next_move_batch(B=4)
    out = m(b)
    assert out["policy_logits"].shape == (4, C)
    assert out["value_logit"].shape == (4, 1)
    assert "econ_pred" in out
    assert out["econ_pred"].shape == (4, 6)
    assert torch.isfinite(out["policy_logits"]).any()  # at least some finite
    assert torch.isfinite(out["value_logit"]).all()
    assert torch.isfinite(out["econ_pred"]).all()


def test_legality_masking():
    """Taken champions must be masked; unavailable only on pick moves."""
    m = tiny_model()
    b = make_next_move_batch(B=2, seed=42)
    # Set some unavailable champs and make some examples bans, some picks
    b["unavailable_mask"][0, 0] = True
    b["unavailable_mask"][1, 5] = True
    b["is_ban"] = torch.tensor([1, 0])  # row 0 = ban, row 1 = pick
    out = m(b)
    logits = out["policy_logits"]
    # Taken champs are always masked (both rows)
    for i in range(2):
        for c in range(C):
            if b["taken_mask"][i, c]:
                assert float(logits[i, c]) <= -1e3, \
                    "taken champ %d in draft %d not masked" % (c, i)
    # Unavailable: masked on pick moves (row 1), NOT on ban moves (row 0)
    assert float(logits[0, 0]) > -1e3, \
        "unavailable champ 0 masked on BAN move (should be legal)"
    assert float(logits[1, 5]) <= -1e3, \
        "unavailable champ 5 not masked on PICK move"


def test_target_champ_not_masked():
    """The target champion must have a finite (non -inf) logit."""
    m = tiny_model()
    b = make_next_move_batch(B=4, seed=7)
    out = m(b)
    for i in range(4):
        tc = int(b["target_champ"][i])
        assert float(out["policy_logits"][i, tc]) > -1e3, \
            "target champ %d in draft %d is masked" % (tc, i)


def test_fearless_ban_target_not_masked():
    """A ban move targeting a fearless-picked champ must NOT be masked.

    Unavailable (fearless-picked) champs are only illegal to PICK, not
    BAN. The target of a ban move that's in unavailable_mask must still
    have a finite logit (otherwise CE -> NaN).
    """
    m = tiny_model()
    b = make_next_move_batch(B=1, seed=10)
    # Make this a ban move
    b["is_ban"] = torch.tensor([1])
    # Set target champ to 3, ensure 3 is NOT in taken_mask but IS in unavailable
    b["taken_mask"][0, 3] = False
    b["unavailable_mask"][0, 3] = True
    b["target_champ"] = torch.tensor([3])
    out = m(b)
    tc = int(b["target_champ"][0])
    assert float(out["policy_logits"][0, tc]) > -1e3, \
        "ban target champ %d is masked by unavailable (should be legal for bans)" % tc
    # CE should be finite
    loss = F.cross_entropy(out["policy_logits"], b["target_champ"])
    assert torch.isfinite(loss), "policy CE is NaN on fearless ban target"


def test_encoder_frozen_by_default():
    """Encoder parameters must be frozen by default."""
    m = tiny_model()
    for p in m.encoder.parameters():
        assert not p.requires_grad, "encoder param not frozen"
    # Heads and context modules must be trainable
    for p in m.policy_head.parameters():
        assert p.requires_grad
    for p in m.value_head.parameters():
        assert p.requires_grad
    for p in m.ban_context.parameters():
        assert p.requires_grad
    for p in m.state_embed.parameters():
        assert p.requires_grad


def test_grad_flows_to_heads_only():
    """Gradient flows to heads + context, not to encoder."""
    m = tiny_model()
    b = make_next_move_batch(B=2)
    out = m(b)
    loss = (out["policy_logits"].sum() + out["value_logit"].sum())
    loss.backward()
    for n, p in m.encoder.named_parameters():
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0, \
            "encoder param %s got gradient (should be frozen)" % n
    for n, p in m.policy_head.named_parameters():
        assert p.grad is not None, "policy_head param %s has no grad" % n
    for n, p in m.value_head.named_parameters():
        assert p.grad is not None, "value_head param %s has no grad" % n


def test_partial_draft_zero_picks_finite():
    """A draft with 0 picks (all PAD) must not produce NaN."""
    m = tiny_model()
    b = make_next_move_batch(B=2, seed=99)
    # Make all slots PAD
    b["champ_ids"][:] = PAD
    b["role_ids"][:] = 5
    b["pad_mask"][:] = True
    # Clear taken_mask of picks (bans still there)
    b["taken_mask"][:] = False
    for j in range(10):
        if b["ban_ids"][0, j] < C:
            b["taken_mask"][0, b["ban_ids"][0, j]] = True
    out = m(b)
    assert torch.isfinite(out["value_logit"]).all(), "value_logit NaN on empty draft"
    assert not torch.isnan(out["policy_logits"]).any(), "policy NaN on empty draft"
    # Pooled vector for empty teams should be zero
    h = m.encoder._encode(b)
    pooled = m._pool(h, b["pad_mask"])
    d = pooled.shape[1] // 4
    assert float(pooled[0, :d].abs().sum()) == 0.0  # blue zero
    assert float(pooled[0, d:2*d].abs().sum()) == 0.0  # red zero


def test_policy_ce_finite():
    """Policy CE loss must be finite and not NaN."""
    m = tiny_model()
    b = make_next_move_batch(B=4)
    out = m(b)
    loss = F.cross_entropy(out["policy_logits"], b["target_champ"])
    assert torch.isfinite(loss)
    assert float(loss) > 0


def test_value_bce_finite():
    """Value BCE loss must be finite."""
    m = tiny_model()
    b = make_next_move_batch(B=4)
    out = m(b)
    loss = F.binary_cross_entropy_with_logits(
        out["value_logit"].squeeze(-1), b["win_label"].float())
    assert torch.isfinite(loss)


def test_no_econ_head():
    """n_econ=0 should produce no econ head and no econ_pred."""
    m = tiny_model(n_econ=0)
    b = make_next_move_batch(B=2)
    out = m(b)
    assert "econ_pred" not in out
    assert m.econ_head is None


def test_unfreeze_encoder():
    """set_freeze_encoder(False) should make encoder params trainable."""
    m = tiny_model()
    m.set_freeze_encoder(False)
    for p in m.encoder.parameters():
        assert p.requires_grad


def test_all_bans_pad_finite():
    """When all ban slots are PAD (no bans yet), must not NaN."""
    m = tiny_model()
    b = make_next_move_batch(B=2, seed=3)
    b["ban_ids"][:] = PAD  # all bans are PAD
    out = m(b)
    assert torch.isfinite(out["policy_logits"]).any()
    assert torch.isfinite(out["value_logit"]).all()
    assert not torch.isnan(out["policy_logits"]).any()


def test_real_init_checkpoint_if_present():
    """If a pretraining checkpoint exists, load it and verify forward."""
    try:
        import train as T
        # train.py loads pro datasets at import time; if pipeline.py is
        # broken/empty, this raises. That's a data issue, not a model bug.
        _ = T.PRO_DATA
    except (FileNotFoundError, AttributeError, ImportError):
        pytest.skip("pro pipeline/datasets not loadable (pipeline.py empty?)")
    try:
        ckpt_spec, payload = T.pick_init_ckpt("auto")
    except FileNotFoundError:
        pytest.skip("no pretraining checkpoints on disk")
    enc, cfg = T.build_encoder(payload)
    m = PolicyValueModel(enc, embed_dim=cfg["embed_dim"],
                          state_dim=64, n_econ=6)
    b = make_next_move_batch(B=3)
    out = m(b)
    assert out["policy_logits"].shape == (3, C)
    assert torch.isfinite(out["value_logit"]).all()
    assert torch.isfinite(out["econ_pred"]).all()
    # Encoder frozen
    for p in m.encoder.parameters():
        assert not p.requires_grad
