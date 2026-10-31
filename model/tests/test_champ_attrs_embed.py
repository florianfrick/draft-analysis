"""Contract tests for the kit-attribute adapter in DraftSlotEmbedding (P3.1).

Two failure modes this file exists to prevent, both of which train fine and
are only visible as a worse number months later:

1. **The adapter is not actually at zero at init.** Then "attributes on" and
   "attributes off" are two different models, and the ablation the flag is
   supposed to give you measures initialisation noise instead.
2. **A checkpoint and an encoder disagree about the adapter.** Loading an old
   checkpoint into an attribute-enabled encoder is legal (missing == zero, so
   step 0 is byte-identical). Loading a *trained* adapter into a model that
   cannot take it is not: it would be dropped silently, and the model would
   serve different numbers than the one that was validated.

Everything is asserted on a 20-champion toy model - the adapter only needs a
rectangular matrix and an embedding table to exercise the plumbing.
"""
import importlib.util
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(MODEL))

from soloq_model import (DraftSlotEmbedding, NUM_REAL_CHAMPS,  # noqa: E402
                         SoloQPretrainedModel, attr_matrix_for_config,
                         load_champ_attr_matrix, load_encoder_state)

TRAIN_PY = MODEL / "pretraining" / "train.py"
C = 20            # toy champion pool
PAD, MASK = C + 1, C
D = 32


def _load_train():
    spec = importlib.util.spec_from_file_location("pretrain_train_attrs",
                                                  TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pretrain_train_attrs"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the table
def test_matrix_is_rectangular_scaled_and_padded():
    mat = load_champ_attr_matrix()
    k = mat.shape[1]
    assert mat.shape == (NUM_REAL_CHAMPS + 2, k)
    assert k == 18, "axis count changed; the anchors/notes live in the JSON"
    assert float(mat.min()) >= 0.0 and float(mat.max()) <= 1.0
    # MASK and PAD must contribute nothing to the residual.
    assert torch.equal(mat[NUM_REAL_CHAMPS:], torch.zeros(2, k))


def test_matrix_rows_follow_the_champion_idx():
    """A row written to the wrong index would still be rectangular."""
    import json
    payload = json.loads(
        (MODEL.parent / "data" / "champ_attrs.json").read_text(encoding="utf-8"))
    mat = load_champ_attr_matrix()
    for entry in payload["champions"].values():
        row = mat[int(entry["idx"])]
        want = torch.tensor(entry["attrs"], dtype=torch.float32) / 3.0
        assert torch.allclose(row, want), entry["name"]


def test_toy_matrix_indexes_by_the_smaller_pool():
    """num_champs is honoured: a 20-champion model gets a 22-row matrix."""
    mat = load_champ_attr_matrix(C)
    assert mat.shape[0] == C + 2
    assert torch.equal(mat[C:], torch.zeros(2, mat.shape[1]))


# ------------------------------------------------------- zero-init equivalence
def _pair(num_champs=C, seed=0):
    """Two encoders, identical weights, one with a zero-init adapter."""
    torch.manual_seed(seed)
    with_attr = SoloQPretrainedModel(
        num_champs=num_champs, embed_dim=D, num_heads=4, num_intra_layers=1,
        num_cross_layers=1, dropout=0.0,
        attr_matrix=load_champ_attr_matrix(num_champs))
    plain = SoloQPretrainedModel(
        num_champs=num_champs, embed_dim=D, num_heads=4, num_intra_layers=1,
        num_cross_layers=1, dropout=0.0)
    plain.load_state_dict(with_attr.state_dict(), strict=False)
    return with_attr.eval(), plain.eval()


def _batch(b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    champ = torch.randint(0, C, (b, 10), generator=g)
    role = torch.randint(0, 5, (b, 10), generator=g)
    pad = torch.zeros(b, 10, dtype=torch.bool)
    pad[0, 5:] = True                       # one team still picking
    champ[0, 5:] = PAD
    return {"champ_ids": champ, "role_ids": role,
            "is_masked": champ == MASK, "pad_mask": pad}


def test_attr_proj_is_zero_at_init():
    torch.manual_seed(0)
    emb = DraftSlotEmbedding(num_champs=C, embed_dim=D,
                             attr_matrix=load_champ_attr_matrix(C))
    assert emb.attr_dim == 18
    assert torch.count_nonzero(emb.attr_proj.weight) == 0
    assert torch.count_nonzero(emb.attr_proj.bias) == 0


def test_embedding_output_is_identical_at_zero_init():
    torch.manual_seed(0)
    a = DraftSlotEmbedding(num_champs=C, embed_dim=D,
                           attr_matrix=load_champ_attr_matrix(C)).eval()
    torch.manual_seed(0)
    b = DraftSlotEmbedding(num_champs=C, embed_dim=D).eval()
    b.load_state_dict(a.state_dict(), strict=False)
    champ = torch.randint(0, C + 2, (6, 10))
    role = torch.randint(0, 6, (6, 10))
    assert torch.equal(a(champ, role), b(champ, role))


def test_encoder_output_is_identical_at_zero_init():
    """The whole point of zero-init: 'attributes on' starts where the old
    model was, so a run difference is the adapter learning, not a reseed."""
    a, b = _pair()
    batch = _batch()
    out_a, out_b = a(batch), b(batch)
    for key in ("champ_logits", "role_logits"):
        assert torch.equal(out_a[key], out_b[key]), key
    assert torch.equal(a._encode(batch), b._encode(batch))


def test_adapter_actually_moves_the_output_once_trained():
    """Guard against a no-op wiring: if attr_proj could not affect the
    embedding, every test above would pass and the feature would do nothing."""
    torch.manual_seed(0)
    emb = DraftSlotEmbedding(num_champs=C, embed_dim=D,
                             attr_matrix=load_champ_attr_matrix(C)).eval()
    champ = torch.randint(0, C, (4, 10))
    role = torch.randint(0, 5, (4, 10))
    before = emb(champ, role)
    torch.nn.init.normal_(emb.attr_proj.weight, std=0.5)
    torch.nn.init.normal_(emb.attr_proj.bias, std=0.5)
    assert not torch.allclose(before, emb(champ, role))


# ------------------------------------------------------- checkpoint asymmetry
def test_old_checkpoint_loads_into_an_attribute_encoder():
    """`b` is the pre-P3.1 checkpoint, `a` the attribute-enabled encoder."""
    a, b = _pair()
    info = load_encoder_state(a, b.state_dict())
    assert info["missing"] == ["embed.attr_proj.bias", "embed.attr_proj.weight"]
    assert info["dropped"] == []
    batch = _batch()
    assert torch.equal(a(batch)["champ_logits"], b(batch)["champ_logits"])


def test_dropping_a_trained_adapter_raises():
    """`a` carries the adapter; loading it into `b` (no attr_proj) would
    silently discard it, which is the failure this asymmetry exists to stop."""
    a, b = _pair()
    with pytest.raises(RuntimeError) as exc:
        load_encoder_state(b, a.state_dict())
    assert "does not know" in str(exc.value)


def test_mismatched_axis_count_raises():
    """A re-tagged table would silently change the meaning of the inputs."""
    with pytest.raises(RuntimeError) as exc:
        attr_matrix_for_config({"attr_dim": 5, "num_champs": C})
    assert "5" in str(exc.value)


def test_config_without_attr_dim_means_no_adapter():
    assert attr_matrix_for_config({}) is None
    assert attr_matrix_for_config({"attr_dim": 0}) is None
    assert attr_matrix_for_config(None) is None
    mat = attr_matrix_for_config({"attr_dim": 18, "num_champs": C})
    assert mat.shape == (C + 2, 18)


# ------------------------------------------------------------- the ckpt config
def _args(T, extra=()):
    return T.parse_args(["--no-wandb", "--embed-dim", str(D), "--num-intra",
                         "1", "--num-cross", "1", "--num-heads", "4"] + list(extra))


def _save(T, path, model, args):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    T.save_ckpt(path, model, args, C, {"champ_top5": 0.1}, 5, epoch=1)
    return opt, sched, torch.amp.GradScaler("cuda", enabled=False)


def test_checkpoint_records_attr_dim(tmp_path):
    T = _load_train()
    args = _args(T)
    torch.manual_seed(0)
    model = T.SoloQPretrainedModel(
        num_champs=C, embed_dim=D, num_heads=4, num_intra_layers=1,
        num_cross_layers=1, attr_matrix=load_champ_attr_matrix(C))
    _save(T, tmp_path / "a.pt", model, args)
    payload = torch.load(str(tmp_path / "a.pt"), weights_only=False)
    assert payload["config"]["attr_dim"] == 18
    # The matrix is data, not a parameter: it must not bloat the checkpoint.
    assert not any("champ_attrs" in k for k in payload["model"])


def test_resume_accepts_a_checkpoint_from_before_the_table(tmp_path):
    """Old ckpt (no attr_dim key) -> attribute model is a legal resume."""
    T = _load_train()
    args = _args(T)
    torch.manual_seed(0)
    old = T.SoloQPretrainedModel(num_champs=C, embed_dim=D, num_heads=4,
                                 num_intra_layers=1, num_cross_layers=1)
    _save(T, tmp_path / "old.pt", old, args)
    payload = torch.load(str(tmp_path / "old.pt"), weights_only=False)
    payload["config"].pop("attr_dim", None)      # emulate a pre-P3.1 file
    torch.save(payload, str(tmp_path / "old.pt"))

    torch.manual_seed(1)
    new = T.SoloQPretrainedModel(
        num_champs=C, embed_dim=D, num_heads=4, num_intra_layers=1,
        num_cross_layers=1, attr_matrix=load_champ_attr_matrix(C))
    opt = torch.optim.AdamW(new.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = T.restore(tmp_path / "old.pt", new, opt, sched, scaler, args,
                      torch.device("cpu"))
    assert state["step"] == 5
    # Weights came across and the adapter is still the zero it started as.
    assert torch.count_nonzero(new.embed.attr_proj.weight) == 0
    assert torch.equal(new.embed.champ_lookup.weight, old.embed.champ_lookup.weight)


def test_resume_refuses_to_drop_a_trained_adapter(tmp_path):
    T = _load_train()
    args = _args(T)
    torch.manual_seed(0)
    trained = T.SoloQPretrainedModel(
        num_champs=C, embed_dim=D, num_heads=4, num_intra_layers=1,
        num_cross_layers=1, attr_matrix=load_champ_attr_matrix(C))
    _save(T, tmp_path / "trained.pt", trained, args)

    plain = T.SoloQPretrainedModel(num_champs=C, embed_dim=D, num_heads=4,
                                   num_intra_layers=1, num_cross_layers=1)
    opt = torch.optim.AdamW(plain.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(SystemExit) as exc:
        T.restore(tmp_path / "trained.pt", plain, opt, sched, scaler, args,
                  torch.device("cpu"))
    assert "attr_dim" in str(exc.value)
