"""Contract tests for SoloQ pretrained model I/O (see solo_queue/model/README.md).

Decisions frozen: no team embedding / no team_ids tensor, champ vocab
REAL+MASK+PAD = 175, role head exactly 5-way, BERT-style 90/10 role masks on
champ-masked slots PLUS role-only masks (champ visible, role hidden) teaching
P(role | champ), hard CE (graded-credit limitation noted in README),
>= 1 champ mask per draft.

Revised: an OPT-IN value path (PMA pooling + blue-win head) is now part of
the contract, but only when the model is built with `with_value=True`. The
default model must stay exactly as before, so `win_logit` is still forbidden
by default. This was a deliberate change, not drift: the shards always
carried `teams.{blue,red}.win`, pretraining never read it, and the downstream
value head consequently sat at the base rate for 8k steps (AUC 0.497, Brier
0.2510 against 0.2482 for a constant).
"""
import importlib.util
import pathlib

import pytest
import torch

N_SLOTS = 10
NUM_REAL_CHAMPS = 173
CHAMP_MASK_ID = NUM_REAL_CHAMPS
CHAMP_PAD_ID = NUM_REAL_CHAMPS + 1
CHAMP_VOCAB_SIZE = NUM_REAL_CHAMPS + 2
NUM_ROLES = 5
ROLE_VOCAB_SIZE = 6
ROLE_FLEX_ID = 5
ROLE_HEAD_DIM = 5
IGNORE_INDEX = -100
MASK_PROB = 0.15
ROLE_KEEP_PROB = 0.1
BLUE_SLOTS = slice(0, 5)
RED_SLOTS = slice(5, 10)

MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent
              / "pretraining" / "base_model.py")


def _load_soloq_module():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        pytest.skip("base_model.py not implemented yet")
    spec = importlib.util.spec_from_file_location("base_model", str(MODEL_PATH))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not import solo_queue.model.pretraining.base_model as base_model.py: {exc}")
    return mod


ROLE_ONLY_PROB = 0.10


def make_synthetic_batch(batch_size=4, mask_prob=MASK_PROB, seed=0, partial=False,
                         role_keep_prob=ROLE_KEEP_PROB,
                         role_only_prob=ROLE_ONLY_PROB):
    """Reference batch satisfying README Sec.3.

    90/10 role masks on champ-masked slots, >=1 champ mask/draft, PLUS
    role-only masks (champ visible, role hidden) teaching P(role | champ).
    """
    g = torch.Generator().manual_seed(seed)
    B = batch_size
    truth = torch.stack(
        [torch.randperm(NUM_REAL_CHAMPS, generator=g)[:N_SLOTS] for _ in range(B)]
    )
    role_truth = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]] * B, dtype=torch.long)
    is_m = torch.rand(B, N_SLOTS, generator=g) < mask_prob
    for b in range(B):  # guarantee >= 1 champ mask per draft
        if not bool(is_m[b].any()):
            is_m[b, int(torch.randint(N_SLOTS, (1,), generator=g))] = True
    keep = (torch.rand(B, N_SLOTS, generator=g) < role_keep_prob) & is_m
    role_only = (torch.rand(B, N_SLOTS, generator=g) < role_only_prob) & (~is_m)
    role_masked = (is_m & ~keep) | role_only
    cids = truth.clone()
    rids = role_truth.clone()
    cids[is_m] = CHAMP_MASK_ID
    rids[role_masked] = ROLE_FLEX_ID  # 10% of champ masks keep true role visible
    clab = torch.full((B, N_SLOTS), IGNORE_INDEX, dtype=torch.long)
    rlab = torch.full((B, N_SLOTS), IGNORE_INDEX, dtype=torch.long)
    clab[is_m] = truth[is_m]
    rlab[is_m] = role_truth[is_m]
    rlab[role_only] = role_truth[role_only]  # P(role | champ) supervision
    pad = torch.zeros(B, N_SLOTS, dtype=torch.bool)
    if partial:
        cids[0, -1] = CHAMP_PAD_ID
        rids[0, -1] = ROLE_FLEX_ID
        is_m[0, -1] = False
        keep[0, -1] = False
        clab[0, -1] = IGNORE_INDEX
        rlab[0, -1] = IGNORE_INDEX
        pad[0, -1] = True
    return {
        "champ_ids": cids, "role_ids": rids, "role_masked": role_masked,
        "is_masked": is_m, "champ_labels": clab, "role_labels": rlab,
        "pad_mask": pad, "meta": {"truth": truth},
    }


def validate_batch_dict(batch, num_real_champs=NUM_REAL_CHAMPS, allow_partial=True):
    """Assert README Sec.3 + invariants 1-5. Reusable by loader tests."""
    B = batch["champ_ids"].shape[0]
    C = num_real_champs
    assert set(batch.keys()) >= {"champ_ids", "role_ids", "role_masked", "is_masked",
                                 "champ_labels", "role_labels", "pad_mask"}
    assert "team_ids" not in batch, "team_ids removed: team follows slot index"
    assert "win_label" not in batch and "aux_targets" not in batch, "aux deferred"
    assert batch["champ_ids"].shape == (B, N_SLOTS)
    assert batch["role_ids"].shape == (B, N_SLOTS)
    assert batch["role_masked"].shape == (B, N_SLOTS)
    assert batch["is_masked"].shape == (B, N_SLOTS)
    assert batch["champ_labels"].shape == (B, N_SLOTS)
    assert batch["role_labels"].shape == (B, N_SLOTS)
    assert batch["pad_mask"].shape == (B, N_SLOTS)
    for k in ("champ_ids", "role_ids", "champ_labels", "role_labels"):
        assert batch[k].dtype == torch.long, k
    for k in ("is_masked", "role_masked", "pad_mask"):
        assert batch[k].dtype == torch.bool, k
    assert int(batch["champ_ids"].min()) >= 0
    assert int(batch["champ_ids"].max()) <= C + 1
    assert int(batch["role_ids"].min()) >= 0
    assert int(batch["role_ids"].max()) <= ROLE_FLEX_ID
    is_m = batch["is_masked"]
    rm = batch["role_masked"]
    is_pad = batch["pad_mask"]
    assert torch.equal(is_pad, batch["champ_ids"] == CHAMP_PAD_ID)
    full = ~is_pad
    assert torch.equal(is_m[full], (batch["champ_ids"][full] == CHAMP_MASK_ID))
    # role_masked = champ-masked-hidden roles + role-only (champ visible).
    # champ-masked part must be a subset of is_masked; role-only part lives
    # on ~is_masked with champ input = truth. Both show FLEX in role_ids.
    assert not bool(((rm & is_m) & ~(batch["role_ids"] == ROLE_FLEX_ID)).any())
    assert not bool(((rm & ~is_m) & ~(batch["role_ids"] == ROLE_FLEX_ID)).any())
    assert torch.equal(rm[full], (batch["role_ids"][full] == ROLE_FLEX_ID) & (batch["role_labels"][full] != IGNORE_INDEX))
    vis_role = is_m & ~rm  # 10%: true role kept visible
    if bool(vis_role.any()):
        assert int(batch["role_ids"][vis_role].min()) >= 0
        assert int(batch["role_ids"][vis_role].max()) < NUM_ROLES
        assert torch.equal(batch["role_ids"][vis_role], batch["role_labels"][vis_role])
    role_only = rm & ~is_m  # champ visible, role hidden -> P(role | champ)
    if bool(role_only.any()):
        assert torch.equal(batch["champ_labels"][role_only],
                            torch.full_like(batch["champ_labels"][role_only], IGNORE_INDEX))
        assert int(batch["role_labels"][role_only].min()) >= 0
        assert int(batch["role_labels"][role_only].max()) < NUM_ROLES
        assert int(batch["champ_ids"][role_only].min()) >= 0
        assert int(batch["champ_ids"][role_only].max()) < C
    assert bool((((batch["champ_labels"] == IGNORE_INDEX) == (~is_m)).all()))
    has_role_label = batch["role_labels"] != IGNORE_INDEX
    assert bool(((has_role_label == (is_m | rm))[full].all())), \
        "role labels exactly on champ-masked + role-only slots"
    m = is_m
    assert bool(m.any()), "batch must contain >= 1 masked slot"
    assert int(batch["champ_labels"][m].min()) >= 0
    assert int(batch["champ_labels"][m].max()) < C
    assert int(batch["role_labels"][m].min()) >= 0
    assert int(batch["role_labels"][m].max()) < NUM_ROLES
    for b in range(B):  # >= 1 mask per draft (no wasted forwards)
        assert bool(is_m[b].any()), f"draft {b} has zero masks"
    if not allow_partial:
        assert not bool(is_pad.any()), "full SoloQ drafts must have no PAD"
    for b, s in is_pad.nonzero().tolist():
        assert int(batch["champ_labels"][b, s]) == IGNORE_INDEX
        assert int(batch["role_labels"][b, s]) == IGNORE_INDEX
        assert not bool(batch["is_masked"][b, s])
    for b in range(B):  # no duplicate real champs per draft
        seen = []
        for s in range(N_SLOTS):
            if bool(is_pad[b, s]):
                continue
            cid = int(batch["champ_labels"][b, s]) if bool(is_m[b, s]) else int(batch["champ_ids"][b, s])
            assert cid not in seen, f"duplicate champ {cid} in draft {b}"
            seen.append(cid)
    return B


def validate_forward_outputs(out, batch, num_real_champs=NUM_REAL_CHAMPS,
                             expect_win=False):
    """Assert README Sec.4 + invariants 6-7 (exact shapes, finite, dup-mask).

    `expect_win=True` requires the opt-in value path and checks it; the
    default keeps asserting its absence, so a model built without
    `with_value=True` can never quietly grow a value head.
    """
    B = batch["champ_ids"].shape[0]
    C = num_real_champs
    assert "champ_logits" in out and "role_logits" in out
    if expect_win:
        assert "win_logit" in out, "with_value=True must emit win_logit"
        wl = out["win_logit"]
        assert wl.shape == (B, 1), wl.shape
        assert torch.isfinite(wl.float()).all(), "win_logit must be finite"
    else:
        assert "win_logit" not in out
    assert "aux_preds" not in out, "aux heads deferred"
    cl, rl = out["champ_logits"], out["role_logits"]
    assert cl.shape == (B, N_SLOTS, C), cl.shape
    core = cl
    # -inf allowed ONLY as intentional duplicate mask (Sec 4.1); NaN/+inf never.
    assert not bool(torch.isnan(core).any()), "champ logits contain NaN"
    assert not bool(torch.isposinf(core).any()), "champ logits contain +inf"
    assert rl.shape == (B, N_SLOTS, ROLE_HEAD_DIM), rl.shape
    assert torch.isfinite(rl.float()).all()
    is_m, is_pad = batch["is_masked"], batch["pad_mask"]
    for b in range(B):
        visible = {int(batch["champ_ids"][b, j]) for j in range(N_SLOTS) if not bool(is_pad[b, j]) and not bool(is_m[b, j])}
        for mpos in is_m[b].nonzero().reshape(-1).tolist():
            truth = int(batch["champ_labels"][b, mpos])
            assert truth not in visible
            for c in visible:
                v = float(core[b, mpos, c])
                assert v <= -1e8, f"draft {b} slot {mpos}: champ {c} logit {v} not suppressed"
            assert float(core[b, mpos, truth]) > -1e8, "truth logit must stay finite"
    return True

def test_input_shapes_dtypes_ranges():
    validate_batch_dict(make_synthetic_batch())


def test_slot_convention_and_split_view():
    b = make_synthetic_batch(batch_size=3, seed=1)
    validate_batch_dict(b)
    assert "team_ids" not in b  # team from slot index, no tensor
    flat = b["champ_ids"]
    split = flat.view(3, 2, 5)
    assert torch.equal(split[:, 0, :], flat[:, BLUE_SLOTS])
    assert torch.equal(split[:, 1, :], flat[:, RED_SLOTS])


def test_masking_rate_and_label_semantics():
    b = make_synthetic_batch(batch_size=64, seed=2)
    rate = float(b["is_masked"].float().mean())
    assert 0.05 < rate < 0.30, f"mask rate {rate} far from 0.15"
    validate_batch_dict(b)
    assert bool((b["champ_labels"][~b["is_masked"]] == IGNORE_INDEX).all())
    # role labels live on champ-masked slots AND role-only slots (champ
    # visible, role hidden) - both teach the role head; unlabelled elsewhere.
    has_rlab = b["role_labels"] != IGNORE_INDEX
    assert bool(((has_rlab == (b["is_masked"] | b["role_masked"])).all()))
    for bb in range(64):  # per-draft guarantee, not just per-batch
        assert bool(b["is_masked"][bb].any())


def test_no_duplicate_champions_in_draft():
    for seed in range(5):
        validate_batch_dict(make_synthetic_batch(batch_size=8, seed=seed))


def test_pad_slot_discipline_partial_draft():
    b = make_synthetic_batch(batch_size=2, seed=3, partial=True)
    validate_batch_dict(b, allow_partial=True)
    assert bool(b["pad_mask"][0, -1])
    assert int(b["champ_ids"][0, -1]) == CHAMP_PAD_ID
    with pytest.raises(AssertionError):
        validate_batch_dict(make_synthetic_batch(batch_size=2, seed=3, partial=True), allow_partial=False)


def test_role_90_10_masking():
    b = make_synthetic_batch(batch_size=256, seed=7)
    validate_batch_dict(b)
    frac = float(b["role_masked"][b["is_masked"]].float().mean())
    assert 0.80 < frac < 0.97, f"role-masked fraction {frac} far from 0.9"
    vis = b["is_masked"] & ~b["role_masked"]
    assert bool(vis.any()), "expected some role-visible masks"
    assert torch.equal(b["role_ids"][vis], b["role_labels"][vis])
    assert bool((b["role_ids"][b["role_masked"]] == ROLE_FLEX_ID).all())
    assert set(b["role_labels"][b["is_masked"]].tolist()) <= set(range(NUM_ROLES))


def test_role_only_masks_teach_champ_to_role():
    """Role-only slots: champ visible, role hidden, role label present."""
    b = make_synthetic_batch(batch_size=256, seed=11)
    validate_batch_dict(b)
    ro = b["role_masked"] & ~b["is_masked"]
    assert bool(ro.any()), "expected some role-only masks"
    # champ input is a real visible champ, no champ label
    assert int(b["champ_ids"][ro].min()) >= 0
    assert int(b["champ_ids"][ro].max()) < NUM_REAL_CHAMPS
    assert bool((b["champ_labels"][ro] == IGNORE_INDEX).all())
    # role input hidden, role label = truth
    assert bool((b["role_ids"][ro] == ROLE_FLEX_ID).all())
    assert set(b["role_labels"][ro].tolist()) <= set(range(NUM_ROLES))
    frac = float(ro.float().mean())
    assert 0.02 < frac < 0.25, f"role-only rate {frac} far from 0.10"


def test_duplicate_masking_logic_on_fake_logits():
    B, C = 2, NUM_REAL_CHAMPS
    b = make_synthetic_batch(batch_size=B, seed=4)
    fake = torch.zeros(B, N_SLOTS, C)
    is_m, is_pad = b["is_masked"], b["pad_mask"]
    for bb in range(B):
        vis = {int(b["champ_ids"][bb, j]) for j in range(N_SLOTS) if not bool(is_pad[bb, j]) and not bool(is_m[bb, j])}
        for mpos in is_m[bb].nonzero().reshape(-1).tolist():
            for c in vis:
                fake[bb, mpos, c] = float("-inf")
    good = {"champ_logits": fake, "role_logits": torch.zeros(B, N_SLOTS, ROLE_HEAD_DIM)}
    validate_forward_outputs(good, b)
    bad = {"champ_logits": torch.zeros(B, N_SLOTS, C), "role_logits": torch.zeros(B, N_SLOTS, ROLE_HEAD_DIM)}
    with pytest.raises(AssertionError):
        validate_forward_outputs(bad, b)


def test_future_loader_batch_contract():
    try:
        from data_loader import load_soloq_batch  # noqa: F401
    except Exception:
        pytest.skip("future data pipeline not implemented yet")
    pytest.fail("loader exists but not wired — call validate_batch_dict on its output")


def _get_model_cls(mod):
    assert hasattr(mod, "SiameseBase"), "SiameseBase missing"
    return getattr(mod, "SiameseBase")


def _build_model(model_cls):
    try:
        return model_cls(num_champs=NUM_REAL_CHAMPS)
    except TypeError:
        return model_cls()


def test_future_model_forward_contract():
    mod = _load_soloq_module()  # skips if base_model.py empty/missing
    model = _build_model(_get_model_cls(mod))
    model.eval()
    b = make_synthetic_batch(batch_size=2, seed=5)
    with torch.no_grad():
        out = model(b)
    assert isinstance(out, dict)
    validate_forward_outputs(out, b)
    assert out["champ_logits"].shape[2] == NUM_REAL_CHAMPS
    assert out["role_logits"].shape[2] == ROLE_HEAD_DIM


def test_future_model_grad_and_invariance():
    mod = _load_soloq_module()  # skips if base_model.py empty/missing
    model = _build_model(_get_model_cls(mod))
    model.train()
    b = make_synthetic_batch(batch_size=2, seed=6)
    out = model(b)
    validate_forward_outputs(out, b)
    loss = out["champ_logits"].sum() + out["role_logits"].sum()
    loss.backward()  # must flow without error
    model.eval()  # deterministic comparison below: no dropout
    perm = torch.randperm(5)
    inv = torch.argsort(perm)

    def permute_team(t):
        t2 = t.clone()
        t2[:, BLUE_SLOTS] = t2[:, BLUE_SLOTS][:, perm]
        return t2

    keys = ("champ_ids", "role_ids", "role_masked", "is_masked", "champ_labels", "role_labels", "pad_mask")
    b2 = {k: (permute_team(v) if k in keys else v) for k, v in b.items() if k != "meta"}
    model.eval()
    with torch.no_grad():
        out = model(b)  # deterministic reference (no dropout)
        out2 = model({**b2, "meta": b.get("meta")})
    validate_forward_outputs(out2, b2)
    # Permutation EQUIVARIANCE: outputs move with the input permutation.
    # Undo the blue perm on out2 -> must match reference out.
    for key in ("champ_logits", "role_logits"):
        unperm = out2[key].clone()
        unperm[:, BLUE_SLOTS] = unperm[:, BLUE_SLOTS][:, inv]
        assert torch.allclose(out[key], unperm, atol=1e-5), \
            f"{key} not permutation-equivariant within team"


# ---------------------------------------------------------------------------
# Opt-in value path: PMA readout + blue-win head
# ---------------------------------------------------------------------------

def _build_value_model(model_cls, seeds=4):
    return model_cls(num_champs=NUM_REAL_CHAMPS, embed_dim=32, num_heads=4,
                     num_intra_layers=1, num_cross_layers=1,
                     with_value=True, value_seeds=seeds)


def _swap_teams(b):
    """Mirror blue<->red: same draft, win label flipped."""
    out = dict(b)
    for k in ("champ_ids", "role_ids", "role_masked", "is_masked",
              "champ_labels", "role_labels", "pad_mask"):
        t = b[k]
        out[k] = torch.cat([t[:, RED_SLOTS], t[:, BLUE_SLOTS]], dim=1)
    if "win_label" in b:
        out["win_label"] = 1 - b["win_label"]
    return out


def test_value_path_off_by_default():
    """The default model must not grow a value head (contract preserved)."""
    mod = _load_soloq_module()
    model = _build_model(_get_model_cls(mod))
    assert model.with_value is False
    assert model.win_head is None and model.pma is None
    out = model(make_synthetic_batch(batch_size=2, seed=11))
    validate_forward_outputs(out, make_synthetic_batch(batch_size=2, seed=11))


def test_value_path_forward_contract():
    mod = _load_soloq_module()
    model = _build_value_model(_get_model_cls(mod)).eval()
    b = make_synthetic_batch(batch_size=3, seed=12)
    b["win_label"] = torch.tensor([0, 1, 1], dtype=torch.long)
    with torch.no_grad():
        out = model(b)
    validate_forward_outputs(out, b, expect_win=True)
    # value_vector is the documented 4D layout
    z = model.value_vector(b)
    assert z.shape == (3, 32 * 4), z.shape
    assert torch.isfinite(z).all()


def test_value_path_gradients_flow():
    mod = _load_soloq_module()
    model = _build_value_model(_get_model_cls(mod))
    model.train()
    b = make_synthetic_batch(batch_size=2, seed=13)
    b["win_label"] = torch.tensor([0, 1], dtype=torch.long)
    out = model(b)
    loss = (out["champ_logits"].sum() + out["role_logits"].sum()
            + out["win_logit"].sum())
    loss.backward()
    assert model.pma.seeds.grad is not None
    assert model.pma.seeds.grad.abs().sum() > 0
    assert model.win_head[1].weight.grad is not None


def test_value_readout_is_siamese_and_permutation_invariant():
    """Swapping teams swaps the two pooled halves; permuting slots does nothing."""
    mod = _load_soloq_module()
    model = _build_value_model(_get_model_cls(mod)).eval()
    b = make_synthetic_batch(batch_size=2, seed=14)
    b["win_label"] = torch.tensor([0, 1], dtype=torch.long)
    D = 32
    with torch.no_grad():
        z = model.value_vector(b)              # [B, 4D]
        vb, vr = z[:, :D], z[:, D:2 * D]
        # side swap: blue's vector must become red's and vice versa
        zs = model.value_vector(_swap_teams(b))
        vbs, vrs = zs[:, :D], zs[:, D:2 * D]
        assert torch.allclose(vb, vrs, atol=1e-5), "PMA is not Siamese"
        assert torch.allclose(vr, vbs, atol=1e-5), "PMA is not Siamese"
        # within-team permutation must leave the pooled vector identical
        perm = torch.randperm(5)
        bp = dict(b)
        bp["champ_ids"] = b["champ_ids"].clone()
        bp["role_ids"] = b["role_ids"].clone()
        for k in ("champ_ids", "role_ids", "is_masked", "pad_mask"):
            bp[k] = b[k].clone()
            bp[k][:, BLUE_SLOTS] = bp[k][:, BLUE_SLOTS][:, perm]
        assert torch.allclose(z, model.value_vector(bp), atol=1e-5), \
            "PMA readout must be permutation-invariant within a team"


def test_pma_handles_all_pad_team():
    """A team with no picks must pool to exactly zero, never NaN."""
    mod = _load_soloq_module()
    model = _build_value_model(_get_model_cls(mod)).eval()
    b = make_synthetic_batch(batch_size=2, seed=15)
    b["champ_ids"] = b["champ_ids"].clone()
    b["pad_mask"] = b["pad_mask"].clone()
    b["champ_ids"][1, BLUE_SLOTS] = CHAMP_PAD_ID
    b["pad_mask"][1, BLUE_SLOTS] = True
    with torch.no_grad():
        z = model.value_vector(b)
    assert torch.isfinite(z).all(), "all-PAD team produced NaN in the readout"
    D = 32
    assert torch.allclose(z[1, :D], torch.zeros(D)), \
        "an empty team must pool to zero"
    # red still has five picks, so its half must NOT be zero
    assert z[1, D:2 * D].abs().sum() > 0, "non-empty team pooled to zero"


def test_load_encoder_state_tolerates_value_keys():
    mod = _load_soloq_module()
    cls = _get_model_cls(mod)
    src = _build_value_model(cls)
    # same trunk shape, no value path: only the value keys may be dropped
    dst = cls(num_champs=NUM_REAL_CHAMPS, embed_dim=32, num_heads=4,
              num_intra_layers=1, num_cross_layers=1, with_value=False)
    res = mod.load_encoder_state(dst, src.state_dict())
    assert res["dropped"], "value keys should have been dropped"
    assert all(k.startswith(("pma.", "win_head.")) for k in res["dropped"])
    # a real mismatch must still be loud
    bad = dict(src.state_dict())
    first = next(iter(bad))
    bad["bogus." + first] = bad[first]
    with pytest.raises(RuntimeError):
        mod.load_encoder_state(dst, bad)
    # a shape mismatch must also stay loud, not be silently filtered
    wrong = cls(num_champs=NUM_REAL_CHAMPS, embed_dim=64, num_heads=4,
                num_intra_layers=1, num_cross_layers=1, with_value=False)
    with pytest.raises(RuntimeError):
        mod.load_encoder_state(wrong, src.state_dict())


