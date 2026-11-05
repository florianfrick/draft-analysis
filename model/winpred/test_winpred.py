"""Tests for the win-prediction experiment (run: `pytest model/winpred`).

Deliberately cheap on data: the pro and soloQ caches are loaded once per
session and every test slices them. The point of the suite is the contract -
that a metric cannot be quietly wrong, that the split cannot leak a series,
that the model cannot see a ban, and that a checkpoint round-trips.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pretraining.soloq_model import SymmetricCrossAttention  # noqa: E402
from model.winpred import data as D              # noqa: E402
from model.winpred import metrics as M           # noqa: E402
from model.winpred.models import siamese_model as W     # noqa: E402
from model.winpred.models.baseline import BlueWinRateBaseline  # noqa: E402
from model.winpred.train import focal_loss        # noqa: E402

EP = D._load("winpred_test_eval_protocol", D.PRO_DIR / "eval_protocol.py")


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def pro_table():
    table, meta = D.load_pro(corpus="2022", verbose=False)
    return table, meta


@pytest.fixture(scope="session")
def soloq_table():
    table, meta = D.load_soloq(verbose=False)
    return table, meta


@pytest.fixture(scope="session")
def both_table():
    table, meta = D.load_both(verbose=False)
    return table, meta


def _batch(n=4, seed=0, patch_id=1):
    g = torch.Generator().manual_seed(seed)
    champ = torch.stack([torch.randperm(173, generator=g)[:10]
                         for _ in range(n)])
    role = torch.arange(10).remainder(5).repeat(n, 1)
    return {"champ_ids": champ, "role_ids": role,
            "patch_id": torch.full((n,), patch_id, dtype=torch.long)}


# ----------------------------------------------------------------- metrics
def test_perfect_predictions_score_perfectly():
    y = np.array([1, 0, 1, 0, 1], dtype=float)
    p = np.where(y == 1, 1.0, 0.0)
    rep = M.full_report(y, p)
    assert rep["log_loss"] < 1e-5
    assert rep["brier"] < 1e-10
    assert rep["roc_auc"] == 1.0
    assert rep["accuracy"] == 1.0
    assert rep["ece"] < 1e-5


def test_constant_half_is_the_reference_point():
    y = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    p = np.full(6, 0.5)
    assert M.log_loss(y, p) == pytest.approx(math.log(2), abs=1e-9)
    assert M.brier(y, p) == pytest.approx(0.25, abs=1e-12)
    assert M.roc_auc(y, p) == 0.5          # all ties
    assert M.accuracy(y, p) == 0.5


def test_auc_matches_the_repo_protocol():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(float)
    p = np.clip(0.35 * y + rng.random(500), 1e-4, 1 - 1e-4)
    assert M.roc_auc(y, p) == pytest.approx(
        EP.auc(p.tolist(), y.tolist()), abs=1e-9)


def test_auc_ignores_a_constant_shift():
    """The whole point of AUC here: a side prior cannot move it."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400).astype(float)
    p = np.clip(0.3 + 0.4 * y + 0.2 * rng.random(400), 1e-4, 1 - 1e-4)
    shifted = np.clip(p + 0.05, 1e-4, 1 - 1e-4)
    assert M.roc_auc(y, p) == pytest.approx(M.roc_auc(y, shifted), abs=1e-9)


def test_ece_is_zero_for_a_calibrated_bin():
    # 60 games at p=0.6, 36 win -> the single occupied bin is exactly calibrated
    y = np.array([1] * 36 + [0] * 24, dtype=float)
    p = np.full(60, 0.6)
    cal = M.ece(y, p, n_bins=15)
    assert cal["ece"] == pytest.approx(0.0, abs=1e-12)
    # ... and 0.1 when the bin is 10 points off (42/60 = 0.70 against p=0.60)
    y2 = np.array([1] * 42 + [0] * 18, dtype=float)
    assert M.ece(y2, p, n_bins=15)["ece"] == pytest.approx(0.1, abs=1e-12)


def test_confident_buckets_are_side_specific():
    y = np.array([1, 1, 0, 0, 1, 0], dtype=float)
    p = np.array([0.9, 0.8, 0.1, 0.7, 0.5, 0.4])
    c = M.confident_metrics(y, p, lo=0.35, hi=0.65)
    # blue calls: p>0.65 at idx 0,1,3 -> y = 1,1,0 -> 2/3 correct
    assert c["n_blue_calls"] == 3
    assert c["precision_blue"] == pytest.approx(2 / 3)
    # recall over the 3 real blue wins: idx 0,1 are covered -> 2/3
    assert c["recall_blue"] == pytest.approx(2 / 3)
    # red call: only idx 2 (0.1) -> red won -> precision 1, recall 1 of 3 reds
    assert c["n_red_calls"] == 1
    assert c["precision_red"] == pytest.approx(1.0)
    assert c["recall_red"] == pytest.approx(1 / 3)
    # 0.5 and 0.4 are abstentions
    assert c["n_confident"] == 4
    assert c["coverage"] == pytest.approx(4 / 6)


def test_skill_scores_are_zero_at_the_base_rate():
    y = np.array([1, 0, 1, 1, 0], dtype=float)
    p = np.full(5, y.mean())
    s = M.skill_scores(y, p)
    assert s["brier_skill"] == pytest.approx(0.0, abs=1e-12)
    assert s["log_loss_skill"] == pytest.approx(0.0, abs=1e-12)


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        M.log_loss([1, 0], [0.5])
    with pytest.raises(ValueError):
        M.log_loss([1, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        M.log_loss([1, 0], [0.5, float("nan")])


def test_bootstrap_is_zero_for_identical_predictions():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300).astype(float)
    p = np.clip(0.3 + 0.4 * y + 0.2 * rng.random(300), 1e-4, 1 - 1e-4)
    groups = ["s%d" % (i // 3) for i in range(300)]
    out = M.bootstrap_delta(y, p, p.copy(), groups, metric="log_loss",
                            n_boot=200)
    assert out["delta"] == pytest.approx(0.0, abs=1e-12)
    assert out["lo"] <= 0 <= out["hi"]
    assert not out["excludes_zero"]


def test_bootstrap_rejects_auc():
    with pytest.raises(ValueError):
        M.bootstrap_delta([1, 0], [0.5, 0.5], [0.4, 0.6], ["a", "b"],
                          metric="roc_auc")


# -------------------------------------------------------------------- loss
def test_focal_with_gamma_zero_is_scaled_bce():
    torch.manual_seed(0)
    logit = torch.randn(64)
    y = torch.randint(0, 2, (64,)).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    assert float(focal_loss(logit, y, gamma=0.0, alpha=0.5)) == \
        pytest.approx(0.5 * float(bce), abs=1e-6)


def test_focal_downweights_easy_examples():
    logit = torch.tensor([4.0, 0.1])       # confident-correct, uncertain
    y = torch.tensor([1.0, 1.0])
    ce = float(torch.nn.functional.binary_cross_entropy_with_logits(logit, y))
    fl = float(focal_loss(logit, y, gamma=2.0, alpha=1.0))
    assert fl < ce


# -------------------------------------------------------------------- data
def test_pro_table_is_full_drafts_with_labels(pro_table):
    table, meta = pro_table
    assert table.champ.shape[1] == 10 and table.role.shape[1] == 10
    assert set(np.unique(table.y).tolist()) <= {0, 1}
    assert table.champ.min() >= 0 and table.champ.max() < D.NUM_REAL_CHAMPS
    assert set(np.unique(table.role).tolist()) == {0, 1, 2, 3, 4}
    # no duplicate champion within a draft
    for i in range(0, len(table), 997):
        assert len(set(table.champ[i].tolist())) == 10
    assert meta["n_patches"] == len(meta["patch_vocab"]) + 1


def test_soloq_table_is_full_drafts_with_labels(soloq_table):
    table, meta = soloq_table
    assert len(table) > 100000
    assert set(np.unique(table.y).tolist()) == {0, 1}
    assert meta["patch_vocab"] == ["16.20"]
    assert set(table.patch_id.tolist()) == {1}


def test_dataset_has_no_bans_and_no_padding(pro_table):
    table, _ = pro_table
    sample = D.WinDataset(table, np.arange(8))[0]
    assert "ban_ids" not in sample and "unavailable_mask" not in sample
    assert sample["champ_ids"].shape == (10,)
    assert sample["win_label"].ndim == 0
    batch = D.collate([D.WinDataset(table, np.arange(8))[i] for i in range(4)])
    assert batch["champ_ids"].shape == (4, 10)
    assert "pad_mask" not in batch        # full drafts: nothing to pad


def test_soloq_split_is_seeded_and_disjoint():
    a = D.split_soloq(1000, seed=7)
    b = D.split_soloq(1000, seed=7)
    c = D.split_soloq(1000, seed=8)
    for k in ("train", "val", "test"):
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a["train"], c["train"])
    allidx = np.concatenate([a["train"], a["val"], a["test"]])
    assert len(allidx) == 1000 and len(set(allidx.tolist())) == 1000
    assert len(a["test"]) == 100 and len(a["val"]) == 100


def test_pro_split_keeps_series_whole_and_covers_every_patch(pro_table):
    table, _ = pro_table
    splits, rep = D.make_splits(table, verbose=False)
    assert rep["n_series_straddling"] == 0
    total = sum(rep["counts"].values())
    assert total == len(table)
    # 80/10/10 of games, allowing for whole-series rounding
    assert 0.78 <= rep["shares"]["train"] <= 0.82
    assert 0.08 <= rep["shares"]["val"] <= 0.12
    assert 0.08 <= rep["shares"]["test"] <= 0.12
    # every patch above min_patch_rows appears in all three splits
    for patch, v in rep["by_patch"].items():
        if v["held_out"]:
            continue
        assert v["train"] > 0 and v["val"] > 0 and v["test"] > 0, patch


def test_split_is_reproducible_across_calls(pro_table):
    table, _ = pro_table
    a, _ = D.make_splits(table, verbose=False)
    b, _ = D.make_splits(table, verbose=False)
    for k in ("train", "val", "test"):
        assert np.array_equal(a[k], b[k])


def test_both_table_reproduces_each_single_dataset_split(pro_table,
                                                         soloq_table,
                                                         both_table):
    """The joint corpus must not move the test rows of either half.

    Every cross-arm comparison in the sweep depends on this: a `both` arm and a
    `soloq` arm have to be scored on the same soloQ games.
    """
    pro, pro_meta = pro_table
    sq, _ = soloq_table
    both, meta = both_table
    assert both.source == "both"
    assert len(both) == len(pro) + len(sq)
    assert both.domains() == ["pro", "soloq"]
    # one patch space: 16.20 is indexed with the PRO vocabulary on both halves
    assert meta["patch_vocab"] == pro_meta["patch_vocab"]
    pid = meta["patch_vocab"].index("16.20") + 1
    assert set(both.patch_id[both.domain == "soloq"].tolist()) == {pid}

    splits, rep = D.make_splits(both, verbose=False)
    sq_splits, _ = D.make_splits(sq, verbose=False)
    pro_splits, _ = D.make_splits(pro, verbose=False)
    off = len(pro)
    for name in ("train", "val", "test"):
        joint_pro = splits[name][splits[name] < off]
        joint_sq = splits[name][splits[name] >= off] - off
        # the joint table's indices are sorted (it concatenates per domain);
        # the invariant is the row *set*, not the order
        assert np.array_equal(np.sort(joint_pro), np.sort(pro_splits[name])), name
        assert np.array_equal(np.sort(joint_sq), np.sort(sq_splits[name])), name
    assert set(rep["by_domain"]) == {"pro", "soloq"}
    assert rep["by_domain"]["pro"]["test"] == len(pro_splits["test"])


def test_patch_remap_translates_by_label():
    from model.winpred.evaluate import patch_remap
    remap = patch_remap(["16.01", "16.02", "16.20"], ["16.20", "16.05"])
    assert remap[0] == 0                  # UNK stays UNK
    assert remap[1] == 3                  # "16.20" is id 3 in the checkpoint
    assert remap[2] == 0                  # "16.05" was never seen -> UNK row


def test_domain_deltas_is_empty_for_a_single_domain(pro_table):
    """A single-dataset run must not crash its report on the domain block.

    This used to raise KeyError *after* training finished, which cost a whole
    run; the pooled number already is the only domain's number.
    """
    from model.winpred.train import domain_deltas
    table, _ = pro_table
    splits, _ = D.make_splits(table, verbose=False)
    idx = splits["test"][:200]
    y = table.y[idx].astype(float)
    p = np.full(len(idx), 0.55)
    base = np.full(len(idx), 0.53)
    groups = [table.series_id[i] for i in idx]
    assert domain_deltas(table, idx, y, p, groups, base) == {}


def test_domain_deltas_breaks_out_a_joint_table(both_table):
    from model.winpred.train import domain_deltas
    table, _ = both_table
    splits, _ = D.make_splits(table, verbose=False)
    idx = splits["test"]
    # 120 rows from each domain, so the bootstrap has groups to resample
    pro_idx = idx[table.domain[idx] == "pro"][:120]
    sq_idx = idx[table.domain[idx] == "soloq"][:120]
    idx = np.concatenate([pro_idx, sq_idx])
    y = table.y[idx].astype(float)
    p = np.full(len(idx), 0.55)
    base = np.full(len(idx), 0.53)
    groups = [table.series_id[i] for i in idx]
    out = domain_deltas(table, idx, y, p, groups, base)
    assert set(out) == {"pro", "soloq"}
    assert out["pro"]["n"] == 120 and out["soloq"]["n"] == 120


def test_domain_deltas_reference_is_the_domains_own_rate(both_table):
    """The per-domain block must beat the domain's own prior, not the pooled one.

    The joint corpus's pooled prior is ~0.48 while pro's own rate is above 0.5,
    so a pooled reference credited the pro block with accuracy the model did not
    have (`both/ce` read +0.019 accuracy on pro while sitting 0.021 *below*
    pro's prior). Pinned on log-loss, which is reference-sensitive even when
    both references predict the same class.
    """
    from model.winpred.train import domain_deltas
    table, _ = both_table
    splits, _ = D.make_splits(table, verbose=False)
    idx = splits["test"]
    idx = np.concatenate([idx[table.domain[idx] == "pro"][:120],
                          idx[table.domain[idx] == "soloq"][:120]])
    y = table.y[idx].astype(float)
    p = np.full(len(idx), 0.55)
    groups = [table.series_id[i] for i in idx]
    pooled = np.full(len(idx), table.y[splits["train"]].mean())

    out = domain_deltas(table, idx, y, p, groups, pooled,
                        train_idx=splits["train"])

    pro = table.domain[idx] == "pro"
    train_dom = table.domain[splits["train"]]
    rate = table.y[splits["train"]][train_dom == "pro"].mean()
    expected = (M.log_loss(y[pro], p[pro])
                - M.log_loss(y[pro], np.full(int(pro.sum()), rate)))
    assert out["pro"]["vs_baseline"]["log_loss"]["delta"] == pytest.approx(
        expected)
    assert "domain train rate" in out["pro"]["reference"]
    # and without train rows the old pooled behaviour is still available
    assert domain_deltas(table, idx, y, p, groups, pooled)["pro"]["reference"] \
        == "pooled prior"


def test_prediction_roundtrip_and_alignment(tmp_path, pro_table):
    table, _ = pro_table
    idx = np.arange(50)
    y = table.y[idx].astype(np.float64)
    p = np.linspace(0.05, 0.95, 50)
    path = D.save_predictions(tmp_path / "p.npz", y, p, table, idx)
    pred = D.load_predictions(path)
    aligned = D.align_predictions(pred, [table.game_id[i] for i in idx])
    assert np.allclose(aligned["p"], p, atol=1e-6)
    # a different row set is an error, not an intersection
    with pytest.raises(ValueError):
        D.align_predictions(pred, [table.game_id[i] for i in idx[1:]])


# ---------------------------------------------------------------- baseline
def test_baseline_is_the_train_blue_win_rate(pro_table):
    table, _ = pro_table
    splits, _ = D.make_splits(table, verbose=False)
    base = BlueWinRateBaseline().fit(table.y[splits["train"]])
    assert base.prior == pytest.approx(float(table.y[splits["train"]].mean()))
    p = base.predict_proba(10)
    assert np.all(p == base.prior)
    d = base.describe()
    assert d["expected_accuracy"] == pytest.approx(max(base.prior, 1 - base.prior))
    assert d["predicted_side"] == ("blue" if base.prior > 0.5 else "red")


def test_baseline_auc_is_exactly_half(pro_table):
    table, _ = pro_table
    splits, _ = D.make_splits(table, verbose=False)
    base = BlueWinRateBaseline().fit(table.y[splits["train"]])
    idx = splits["test"]
    p = base.predict_proba(len(idx))
    assert M.roc_auc(table.y[idx], p) == 0.5


# ------------------------------------------------------------------- model
def test_model_shapes_and_parameter_contract():
    m = W.WinPredictor(embed_dim=32, num_intra=1, num_cross=1, num_heads=4,
                       n_patches=4)
    out = m(_batch(4))
    assert out["logit"].shape == (4,)
    assert out["h_blue"].shape == (4, 32)
    assert torch.isfinite(out["logit"]).all()
    assert m.side_bias.numel() == 1
    assert float(m.side_bias.detach()) == 0.0
    # no value/aux heads: logit, two team vectors, nothing else
    assert set(out) == {"logit", "h_blue", "h_red"}


def test_model_is_permutation_invariant_within_a_team():
    m = W.WinPredictor(embed_dim=32, num_intra=1, num_cross=1, num_heads=4,
                       dropout=0.0).eval()
    batch = _batch(3, seed=1)
    perm = torch.tensor([3, 0, 4, 1, 2])
    shuffled = dict(batch)
    shuffled["champ_ids"] = batch["champ_ids"].clone()
    shuffled["role_ids"] = batch["role_ids"].clone()
    for half in (slice(0, 5), slice(5, 10)):
        shuffled["champ_ids"][:, half] = batch["champ_ids"][:, half][:, perm]
        shuffled["role_ids"][:, half] = batch["role_ids"][:, half][:, perm]
    with torch.no_grad():
        a = m(batch)["logit"]
        b = m(shuffled)["logit"]
    assert torch.allclose(a, b, atol=1e-5)


def test_side_bias_is_the_only_path_to_the_prior():
    m = W.WinPredictor(embed_dim=32, num_intra=1, num_cross=1, num_heads=4)
    # The head's last layer has NO bias: otherwise it is a second constant and
    # the learned split between it and `side_bias` is arbitrary.
    assert m.head[-1].bias is None
    with torch.no_grad():
        m.head[-1].weight.zero_()
        m.side_bias.fill_(0.7)
    out = m(_batch(2))
    assert torch.allclose(out["logit"], torch.full((2,), 0.7), atol=1e-6)


def test_only_one_constant_parameter_exists():
    """`side_bias` must be the only bias-free path to a constant logit."""
    m = W.WinPredictor(embed_dim=32, num_intra=1, num_cross=1, num_heads=4)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.side_bias.fill_(-1.3)
    out = m(_batch(3))
    assert torch.allclose(out["logit"], torch.full((3,), -1.3), atol=1e-6)


def test_config_roundtrips_into_a_loadable_model(tmp_path):
    m = W.WinPredictor(embed_dim=32, num_intra=2, num_cross=1, num_heads=4,
                       dropout=0.1, n_patches=5, pool="pma")
    cfg = W.config_of(m, data="pro", seed=1)
    rebuilt = W.build_from_config(cfg)
    rebuilt.load_state_dict(m.state_dict())
    assert W.param_count(rebuilt) == W.param_count(m)
    path = tmp_path / "ck.pt"
    torch.save({"model": m.state_dict(), "config": cfg,
                "patch_vocab": ["16.01", "16.02"]}, str(path))
    loaded, payload = W.load_checkpoint(path)
    assert loaded.num_champs == m.num_champs
    assert payload["config"]["seed"] == 1


# --------------------------------------------------- architecture: the head
def test_head_product_appends_the_elementwise_term():
    m = W.WinPredictor(embed_dim=32, num_intra=1, num_cross=1, num_heads=4,
                       head_product=True)
    assert m.head_in == 32 * 4
    assert tuple(m.head[0].normalized_shape) == (32 * 4,)
    vb, vr = torch.randn(3, 32), torch.randn(3, 32)
    z = m.head_input(vb, vr)
    assert z.shape == (3, 32 * 4)
    assert torch.allclose(z[:, :32], vb)
    assert torch.allclose(z[:, 32:64], vr)
    assert torch.allclose(z[:, 64:96], vb - vr)
    assert torch.allclose(z[:, 96:], vb * vr)


def test_head_without_the_flag_is_byte_for_byte_the_old_head():
    m = W.WinPredictor(embed_dim=32, num_intra=1, num_cross=1, num_heads=4)
    assert m.head_in == 96
    assert m.head_input(torch.randn(2, 32), torch.randn(2, 32)).shape == (2, 96)


def test_product_term_is_symmetric_under_a_team_swap():
    """The product cannot carry side information, so `h_b - h_r` and
    `side_bias` remain the only asymmetric paths to the logit."""
    m = W.WinPredictor(embed_dim=16, num_intra=1, num_cross=1, num_heads=4,
                       head_product=True)
    vb, vr = torch.randn(4, 16), torch.randn(4, 16)
    a = m.head_input(vb, vr)
    b = m.head_input(vr, vb)
    assert torch.allclose(a[:, 48:], b[:, 48:])       # product: unchanged
    assert torch.allclose(a[:, 32:48], -b[:, 32:48])  # difference: negated


# ---------------------------------------- architecture: the role-aware bias
def test_role_bias_starts_at_the_stated_prior():
    m = W.WinPredictor(embed_dim=16, num_intra=1, num_cross=1, num_heads=4,
                       role_bias=True)
    rb = m.cross_team[0].role_bias.detach()
    assert tuple(rb.shape) == (5, 5)
    assert m.cross_team[0].role_bias.requires_grad
    for i in range(5):
        # the diagonal is 0.5, except jungle, which also collects its own
        # row/column boost: 0.5 + 0.2 + 0.2
        assert float(rb[i, i]) == pytest.approx(0.9 if i == 1 else 0.5)
    assert float(rb[3, 4]) == pytest.approx(0.3)
    assert float(rb[4, 3]) == pytest.approx(0.3)
    assert float(rb[0, 1]) == pytest.approx(0.2)      # jungle column
    assert float(rb[1, 0]) == pytest.approx(0.2)      # jungle row
    assert float(rb[0, 2]) == pytest.approx(0.0)      # top vs mid: no prior
    assert float(rb[2, 4]) == pytest.approx(0.0)


def test_role_bias_is_indexed_by_role_not_by_slot():
    blk = W.RoleAwareCrossAttention(embed_dim=16, num_heads=4)
    roles = torch.tensor([[2, 0, 4, 1, 3]])           # deliberately permuted
    mask = blk._additive_mask(roles, roles, torch.float32).detach()
    assert tuple(mask.shape) == (4, 5, 5)             # [B*heads, 5, 5]
    assert float(mask[0, 0, 4]) == pytest.approx(0.0)   # role 2 -> role 3
    assert float(mask[0, 2, 3]) == pytest.approx(0.2)   # role 4 -> role 1
    assert torch.allclose(mask[0], mask[3])            # shared across heads


def test_role_bias_changes_the_logit_and_receives_gradient():
    m = W.WinPredictor(embed_dim=16, num_intra=1, num_cross=1, num_heads=4,
                       dropout=0.0, role_bias=True).eval()
    batch = _batch(3, seed=2)
    with torch.no_grad():
        before = m(batch)["logit"].clone()
        m.cross_team[0].role_bias.zero_()
        after = m(batch)["logit"]
    assert not torch.allclose(before, after, atol=1e-5)
    m.train()
    m(batch)["logit"].sum().backward()
    g = m.cross_team[0].role_bias.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_zeroed_role_bias_reproduces_the_plain_block():
    """A strict superset: at zero bias the new block IS the block the six arms
    were trained with, so a difference can only come from the bias."""
    torch.manual_seed(0)
    plain = SymmetricCrossAttention(16, 4, 0.0).eval()
    aware = W.RoleAwareCrossAttention(16, 4, 0.0).eval()
    missing, unexpected = aware.load_state_dict(plain.state_dict(), strict=False)
    assert unexpected == [] and set(missing) == {"role_bias"}
    with torch.no_grad():
        aware.role_bias.zero_()
        blue, red = torch.randn(3, 5, 16), torch.randn(3, 5, 16)
        pb, pr = plain(blue, red)
        ab, ar = aware(blue, red)
    assert torch.allclose(pb, ab, atol=1e-6)
    assert torch.allclose(pr, ar, atol=1e-6)


def test_role_aware_model_is_still_permutation_invariant():
    """Gathering the bias by role is what keeps slot order irrelevant."""
    m = W.WinPredictor(embed_dim=16, num_intra=1, num_cross=1, num_heads=4,
                       dropout=0.0, role_bias=True, head_product=True).eval()
    batch = _batch(3, seed=1)
    perm = torch.tensor([3, 0, 4, 1, 2])
    shuffled = dict(batch)
    shuffled["champ_ids"] = batch["champ_ids"].clone()
    shuffled["role_ids"] = batch["role_ids"].clone()
    for half in (slice(0, 5), slice(5, 10)):
        shuffled["champ_ids"][:, half] = batch["champ_ids"][:, half][:, perm]
        shuffled["role_ids"][:, half] = batch["role_ids"][:, half][:, perm]
    with torch.no_grad():
        assert torch.allclose(m(batch)["logit"], m(shuffled)["logit"], atol=1e-5)


def test_architecture_flags_roundtrip_and_old_configs_still_build():
    m = W.WinPredictor(embed_dim=16, num_intra=1, num_cross=1, num_heads=4,
                       head_product=True, role_bias=True)
    cfg = W.config_of(m)
    assert cfg["head_product"] is True and cfg["role_bias"] is True
    rebuilt = W.build_from_config(cfg)
    rebuilt.load_state_dict(m.state_dict())
    assert W.param_count(rebuilt) == W.param_count(m)
    assert isinstance(rebuilt.cross_team[0], W.RoleAwareCrossAttention)

    # A config written before the flags existed (README 7's six arms).
    old = {k: v for k, v in cfg.items()
           if k not in ("head_product", "role_bias")}
    old_model = W.build_from_config(old)
    assert old_model.head_in == 16 * 3
    assert not isinstance(old_model.cross_team[0], W.RoleAwareCrossAttention)
    assert old_model.head[1].in_features == 48
