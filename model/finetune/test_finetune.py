"""Tests for the pro finetune stage (model + losses + helpers).

Run: pytest model/finetune/test_finetune.py -q
Uses a tiny synthetic encoder (no checkpoint IO) for speed; one test
loads the real default init checkpoint if present (skips otherwise).
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from soloq_model import SoloQPretrainedModel  # noqa: E402
from model import ProDraftModel  # noqa: E402
import train as T  # noqa: E402

C = 173
PAD = C + 1


def tiny_model(d=32):
    enc = SoloQPretrainedModel(num_champs=C, embed_dim=d, num_heads=4,
                               num_intra_layers=1, num_cross_layers=1)
    return ProDraftModel(enc, n_econ=6)


def full_batch(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    champs = torch.stack([torch.randperm(C, generator=g)[:10]
                          for _ in range(B)])
    roles = torch.zeros(B, 10, dtype=torch.long)
    roles[:, 5:] = 1  # fake distinct red roles for sanity only
    roles = roles % 5
    return {
        "champ_ids": champs, "role_ids": roles,
        "pad_mask": torch.zeros(B, 10, dtype=torch.bool),
        "is_masked": torch.zeros(B, 10, dtype=torch.bool),
        "champ_labels": torch.full((B, 10), T.IGNORE_INDEX,
                                   dtype=torch.long),
        "role_labels": torch.full((B, 10), T.IGNORE_INDEX,
                                  dtype=torch.long),
        "win_label": torch.tensor([0, 1, 1, 0]),
        "has_econ": torch.ones(B),
        "econ_target": torch.randn(B, 6) * 3000 + 18000,
        "weight": torch.ones(B),
    }


def test_forward_shapes_and_finite():
    m = tiny_model()
    out = m(full_batch())
    assert out["win_logits"].shape == (4, 2)
    assert out["econ_pred"].shape == (4, 6)
    assert out["champ_logits"].shape == (4, 10, C)
    assert torch.isfinite(out["win_logits"]).all()
    assert torch.isfinite(out["econ_pred"]).all()
    assert not torch.isnan(out["champ_logits"]).any()


def test_empty_team_partial_draft_finite():
    """Regression: one team fully PAD must not NaN (attention all-masked)."""
    m = tiny_model()
    b = full_batch(B=2)
    b["pad_mask"][0, 5:] = True          # red team entirely PAD
    b["pad_mask"][1, :5] = True          # blue team entirely PAD
    b["champ_ids"][b["pad_mask"]] = PAD
    out = m(b)
    for k in ("win_logits", "econ_pred", "champ_logits", "role_logits"):
        assert torch.isfinite(out[k]).all(), k
    # pooled vector for the empty team must be exactly zero
    z = m.pool(b)
    d = z.shape[1] // 4
    assert float(z[0, d:2 * d].abs().sum()) == 0.0  # red side zero
    assert float(z[1, :d].abs().sum()) == 0.0       # blue side zero


def test_stage_freezing_schedule():
    m = tiny_model()
    trunk = (list(m.encoder.intra_team.parameters())
             + list(m.encoder.cross_team.parameters()))
    trunk_n = sum(p.numel() for p in trunk)
    assert trunk_n > 0
    m.set_stage(0)
    # stage 0: attention trunk + embeddings frozen; outcome heads and the
    # encoder readout heads (champ_head/role_head) train
    assert all(not p.requires_grad for p in trunk)
    assert all(not p.requires_grad for p in m.encoder.embed.parameters())
    assert all(p.requires_grad for p in m.heads.parameters())
    assert all(p.requires_grad for p in m.encoder.champ_head.parameters())
    # stage 1: cross_team joins; intra still frozen
    m.set_stage(1)
    assert all(p.requires_grad for p in m.encoder.cross_team.parameters())
    assert all(not p.requires_grad for p in m.encoder.intra_team.parameters())
    # stage 2: full trunk trains, embeddings stay frozen throughout
    for stage in (1, 2):
        m.set_stage(stage)
        assert all(not p.requires_grad
                   for p in m.encoder.embed.parameters())
    m.set_stage(2)
    assert all(p.requires_grad for p in trunk)


def test_losses_guards_no_nan():
    m = tiny_model()
    b = full_batch()
    # give the batch one real mask: slot 0 masked, truth not visible
    b["is_masked"][0, 0] = True
    b["champ_ids"][0, 0] = C  # CHAMP_MASK_ID
    visible = set(b["champ_ids"][0, 1:].tolist())
    b["champ_labels"][0, 0] = next(c for c in range(C) if c not in visible)
    b["role_labels"][0, 0] = 2
    out = m(b)
    l1, lc, lr = T.masked_losses(out, b, 0.3)
    assert torch.isfinite(l1) and torch.isfinite(lc) and torch.isfinite(lr)
    # batch with NO masked slots -> guarded zeros (no CE over empty set)
    b2 = full_batch()
    l0, _, _ = T.masked_losses(out, b2, 0.3)
    assert float(l0) == 0.0
    # batch lacking is_masked entirely -> guarded zeros (outcome-only)
    l2, _, _ = T.masked_losses(out, {"champ_ids": b2["champ_ids"]}, 0.3)
    assert float(l2) == 0.0
    # win + econ finite
    wl = T.win_loss(out, b)
    assert torch.isfinite(wl)
    mean = b["econ_target"].mean(0)
    std = b["econ_target"].std(0, unbiased=False).clamp_min(1e-3)
    assert torch.isfinite(T.econ_loss(out, b, mean, std))


def test_batch_auc_rank_stats():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    labels = torch.tensor([1, 1, 0, 0])
    assert abs(T.batch_auc(logits, labels) - 1.0) < 1e-6
    assert abs(T.batch_auc(-logits, labels) - 0.0) < 1e-6
    assert T.batch_auc(logits, torch.zeros(4, dtype=torch.long)) == 0.5
    # NaN defensive: drop poisoned entries, keep the rest meaningful
    nan_logits = torch.tensor([float("nan"), 3.0, 1.0])
    nan_labels = torch.tensor([1, 1, 0])
    assert abs(T.batch_auc(nan_logits, nan_labels) - 1.0) < 1e-6


def test_fit_platt_and_econ_stats():
    x = torch.linspace(-4, 4, 512)
    y = (torch.sigmoid(2.5 * x + 0.3) > 0.5).float()
    a, b = T.fit_platt(x, y)
    assert a > 0 and torch.isfinite(torch.tensor([a, b])).all()
    # econ stats: constant series -> std floored, mean preserved
    rows = [{"has_econ": True, "econ": [1.0] * 6},
            {"has_econ": False, "econ": [None] * 6},
            {"has_econ": True, "econ": [3.0] * 6}]
    mean, std = T.fit_econ_stats(rows, [0, 1, 2], 6)
    assert float(mean[0]) == pytest.approx(2.0)
    assert float(std[0]) >= 1e-3


def test_real_init_checkpoint_if_present():
    path, payload = None, None
    try:
        path, payload = T.pick_init_ckpt("auto")
    except FileNotFoundError:
        pytest.skip("no pretraining checkpoints on disk")
    enc, cfg = T.build_encoder(payload)
    m = ProDraftModel(enc, n_econ=6)
    b = full_batch(B=3)
    out = m(b)
    assert torch.isfinite(out["win_logits"]).all()
    assert torch.isfinite(out["champ_logits"]).all()
    # stage 0: attention trunk + embeddings frozen; readouts live
    m.set_stage(0)
    assert all(not p.requires_grad
               for p in m.encoder.intra_team.parameters())
    assert all(not p.requires_grad for p in m.encoder.cross_team.parameters())
    assert all(not p.requires_grad for p in m.encoder.embed.parameters())
    assert all(p.requires_grad for p in m.heads.parameters())
    assert all(p.requires_grad for p in m.encoder.champ_head.parameters())

