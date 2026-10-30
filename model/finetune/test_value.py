"""Tests for the prefix value stage (model + metrics + report).

Run: pytest model/finetune/test_value.py -q

No checkpoint IO: a tiny encoder with the value path enabled, d=32.
"""
import argparse
import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from soloq_model import SoloQPretrainedModel, CHAMP_PAD_ID, ROLE_FLEX_ID  # noqa: E402
from model import PrefixValueModel  # noqa: E402
import value as V  # noqa: E402

C = 173
D = 32


def tiny_encoder(with_value=True, d=D):
    return SoloQPretrainedModel(num_champs=C, embed_dim=d, num_heads=4,
                                num_intra_layers=1, num_cross_layers=1,
                                with_value=with_value)


def tiny_value(use_bans=True):
    m = PrefixValueModel(tiny_encoder(), ban_heads=4, ban_seeds=2,
                         ban_dropout=0.0, use_bans=use_bans)
    m.freeze_trunk()
    m.eval()
    return m


def batch(B=3, n_picks=4, n_bans=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    champ = torch.full((B, 10), CHAMP_PAD_ID, dtype=torch.long)
    role = torch.full((B, 10), ROLE_FLEX_ID, dtype=torch.long)
    pad = torch.ones(B, 10, dtype=torch.bool)
    for b in range(B):
        picks = torch.randperm(C, generator=g)[:n_picks]
        champ[b, :n_picks] = picks
        role[b, :n_picks] = torch.arange(n_picks) % 5
        pad[b, :n_picks] = False
    ban = torch.full((B, 10), CHAMP_PAD_ID, dtype=torch.long)
    if n_bans:
        ban[:, :n_bans] = torch.randperm(C, generator=g)[:n_bans]
    return {"champ_ids": champ, "role_ids": role, "pad_mask": pad,
            "ban_ids": ban}


# ------------------------------------------------------------------ freezing
def test_trunk_is_frozen_and_value_path_is_not():
    m = tiny_value()
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable, "nothing is trainable"
    for n, p in m.named_parameters():
        allowed = n.startswith(("encoder.pma.", "encoder.win_head.",
                                "ban_pool.", "ban_block."))
        assert p.requires_grad == allowed, "%s requires_grad=%s" % (
            n, p.requires_grad)
    # the shared trunk must not train
    assert not any(p.requires_grad for p in m.encoder.intra_team.parameters())
    assert not any(p.requires_grad for p in m.encoder.cross_team.parameters())
    assert not any(p.requires_grad
                   for p in m.encoder.embed.champ_lookup.parameters())


def test_encoder_without_value_path_is_rejected():
    with pytest.raises(ValueError, match="with_value"):
        PrefixValueModel(tiny_encoder(with_value=False))


# ------------------------------------------------------- exactness at init
def test_ban_residual_is_exactly_zero_at_init():
    """At init the logit must equal the pretrained head, bit for bit.

    This is what makes the pretrained value weights usable verbatim and
    makes 'what do bans buy us' a clean ablation.
    """
    m = tiny_value()
    b = batch()
    out = m(b)
    assert torch.equal(out["win_logit"], out["win_logit_base"])
    assert torch.all(out["win_corr"] == 0)
    with torch.no_grad():
        z4 = m.encoder.value_vector(b)
        assert torch.equal(out["win_logit_base"], m.encoder.win_head(z4))


def test_no_ban_path_matches_base():
    m = tiny_value(use_bans=False)
    out = m(batch())
    assert torch.equal(out["win_logit"], out["win_logit_base"])
    assert "ban_vector" not in out
    assert m.ban_pool is None and m.ban_block is None


# ------------------------------------------------------------- ban pooling
def test_ban_pool_is_permutation_invariant():
    """Bans carry no order: shuffling ban_ids must not move the vector."""
    m = tiny_value()
    b = batch()
    bv = m.ban_vector(b)
    perm = b["ban_ids"][:, torch.randperm(10)]
    b2 = dict(b, ban_ids=perm)
    assert torch.allclose(bv, m.ban_vector(b2), atol=1e-6)


def test_ban_pool_is_zero_and_finite_with_no_bans():
    m = tiny_value()
    b = batch(n_bans=0)
    bv = m.ban_vector(b)
    assert torch.isfinite(bv).all()
    assert torch.all(bv == 0)


def test_bans_change_the_logit_once_weights_are_nonzero():
    m = tiny_value()
    b = batch()
    base = m(b)["win_logit"].clone()
    # zero-init means no gradient reaches the block's input side until its
    # last layer moves, so drive it explicitly here
    with torch.no_grad():
        m.ban_block[-1].weight.normal_(0.0, 0.5)
    moved = m(b)["win_logit"]
    assert not torch.allclose(base, moved)
    # swapping the *set* (not the order - the pool is order-invariant) must
    # move the value: this is the whole point of giving V a ban path
    other = dict(b)
    ban = b["ban_ids"].clone()
    ban[:, 0] = 7  # a champion that is neither banned nor picked above
    other["ban_ids"] = ban
    assert not torch.allclose(m(other)["win_logit"], moved, atol=1e-5)


def test_zero_init_block_still_learns():
    """A zero-init last layer must not be a dead end: one step moves it."""
    m = tiny_value()
    m.train()
    b = batch()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                          lr=0.1)
    before = m.ban_block[-1].weight.detach().clone()
    logit = m(b)["win_logit"].squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, torch.tensor([1.0, 0.0, 1.0]))
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = m.ban_block[-1].weight.detach()
    assert not torch.allclose(before, after), "zero-init block never moved"


# ------------------------------------------------------------------ metrics
def test_summarize_perfect_separation():
    logits = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    ys = [0, 0, 0, 1, 1, 1]
    s = V.summarize(logits, ys)
    assert s["auc"] == 1.0
    assert s["skill"] > 0
    assert s["n"] == 6
    assert s["base_rate"] == 0.5


def test_summarize_constant_is_zero_skill():
    """A constant predictor must score skill ~0, never positive."""
    s = V.summarize([0.0] * 10, [1, 0] * 5)
    assert abs(s["skill"]) < 1e-9
    assert abs(s["auc"] - 0.5) < 1e-9


def test_summarize_anticorrelation_is_negative_skill():
    """The policy value head's signature: confident and wrong."""
    s = V.summarize([2.0, 2.0, -2.0, -2.0], [0, 0, 1, 1])
    assert s["auc"] == 0.0
    assert s["skill"] < 0


def test_summarize_empty_slice_is_nan_not_crash():
    s = V.summarize([], [])
    assert s["n"] == 0
    assert math.isnan(s["auc"])


def test_summarize_auc_matches_eval_protocol():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(60, generator=g).tolist()
    ys = (torch.rand(60, generator=g) < 0.4).long().tolist()
    assert V.summarize(logits, ys)["auc"] == pytest.approx(
        V.EP.auc(logits, ys))


def test_sigmoid_survives_large_inputs():
    assert V.sigmoid(1e4) == pytest.approx(1.0)
    assert V.sigmoid(-1e4) == pytest.approx(0.0)
    assert V.sigmoid(0.0) == pytest.approx(0.5)


def test_strata_drops_small_slices_and_sorts_by_n():
    logits = list(range(10)) + [0.0, 1.0]
    ys = [0, 1] * 5 + [0, 1]
    keys = ["big"] * 10 + ["small"] * 2
    out = V.strata(logits, ys, keys, min_n=5)
    assert list(out) == ["big"]
    assert out["big"]["n"] == 10


def test_platt_recovers_an_overconfident_scale():
    g = torch.Generator().manual_seed(5)
    raw = torch.randn(400, generator=g) * 4.0
    y = (torch.rand(400, generator=g) < torch.sigmoid(raw / 4.0)).float()
    a, b = V.fit_platt(raw.tolist(), y.long().tolist())
    # logits were 4x too wide -> calibration should shrink them
    assert 0.05 < a < 1.0


# ------------------------------------------------------------------- report
def _rec(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    logit = torch.randn(n, generator=g).tolist()
    y = (torch.rand(n, generator=g) < 0.5).long().tolist()
    return {"logit": logit, "logit_base": logit, "y": y,
            "move_idx": [i % 20 for i in range(n)],
            "phase": [V.PRO_DATA.PHASE_OF[i % 20] for i in range(n)],
            "patch": ["16.20" if i % 3 else "16.19" for i in range(n)]}


def test_full_report_carries_the_ban_ablation_and_phase_table():
    args = argparse.Namespace(min_stratum=50)
    rep = V.full_report(_rec(), (1.0, 0.0), args, use_bans=True)
    assert set(rep["by_phase"]) <= set(V.PHASE_NAMES.values())
    assert rep["focus_patch"] == V.EP.FOCUS_PATCH
    assert "no_ban_residual" in rep
    assert "ban_residual" in rep
    # identical logits -> zero residual, as the ablation should report
    assert rep["ban_residual"]["max_abs"] == 0.0


def test_full_report_omits_residual_when_bans_are_off():
    args = argparse.Namespace(min_stratum=50)
    rep = V.full_report(_rec(), (1.0, 0.0), args, use_bans=False)
    assert "ban_residual" not in rep
