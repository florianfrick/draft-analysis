"""Contract tests for the patch embedding (pro-only experiment, 2026-11-01).

Same two failure modes as the kit-attribute adapter, plus one that is specific
to a patch input:

1. **The embedding is not at zero at init.** Then "patch on" and "patch off"
   are two different networks and the ablation measures initialisation noise.
2. **A checkpoint and an encoder disagree about the vocabulary.** A checkpoint
   trained with 107 patch rows loaded into an encoder sized for 1 silently
   re-uses row 0 for everything; a checkpoint without patch rows loaded into a
   patch-enabled encoder silently starts from zero, which is legal but must be
   visible (missing keys are reported, not swallowed).
3. **The patch id reaches the model but not the gradient.** A patch embedding
   that is fed the wrong key (or a constant) trains fine and is dead weight.
   `test_patch_embedding_receives_gradient` and
   `test_patch_changes_the_encoder_output` pin the wiring from both ends.
"""
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(MODEL))

from model.pretraining.soloq_model import (DraftSlotEmbedding, SoloQPretrainedModel,  # noqa: E402
                         load_encoder_state)

C = 20            # toy champion pool
PAD, MASK = C + 1, C
D = 32
N_PATCHES = 7     # 6 real patches + UNK


def _emb(n_patches=N_PATCHES, seed=0):
    torch.manual_seed(seed)
    return DraftSlotEmbedding(num_champs=C, embed_dim=D,
                              n_patches=n_patches).eval()


def _ids(b=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    champ = torch.randint(0, C, (b, 10), generator=g)
    role = torch.randint(0, 5, (b, 10), generator=g)
    patch = torch.randint(0, N_PATCHES, (b,), generator=g)
    return champ, role, patch


def _model(n_patches=N_PATCHES, seed=0):
    torch.manual_seed(seed)
    return SoloQPretrainedModel(num_champs=C, embed_dim=D, num_heads=4,
                                num_intra_layers=1, num_cross_layers=1,
                                dropout=0.0, n_patches=n_patches).eval()


def _batch(seed=0):
    champ, role, patch = _ids(seed=seed)
    return {"champ_ids": champ, "role_ids": role, "patch_id": patch,
            "is_masked": torch.zeros_like(champ, dtype=torch.bool),
            "pad_mask": torch.zeros_like(champ, dtype=torch.bool)}


# ------------------------------------------------------------------ zero init
def test_patch_embedding_is_zero_at_init():
    emb = _emb()
    assert emb.n_patches == N_PATCHES
    assert torch.count_nonzero(emb.patch_lookup.weight) == 0


def test_embedding_output_is_identical_at_zero_init():
    a, b = _emb(), _emb(n_patches=0)
    b.load_state_dict(a.state_dict(), strict=False)
    champ, role, patch = _ids()
    assert torch.equal(a(champ, role, patch), b(champ, role))


def test_encoder_output_is_identical_at_zero_init():
    a, b = _model(), _model(n_patches=0)
    b.load_state_dict(a.state_dict(), strict=False)
    batch = _batch()
    out_a, out_b = a(batch), b(batch)
    for key in ("champ_logits", "role_logits"):
        assert torch.equal(out_a[key], out_b[key]), key


# ------------------------------------------------------------- it is not dead
def test_patch_changes_the_encoder_output():
    """Guard against a no-op wiring: with a non-zero table the patch must move
    the logits, otherwise every other test here passes and the input is dead."""
    model = _model()
    torch.nn.init.normal_(model.embed.patch_lookup.weight, std=0.5)
    batch = _batch()
    before = model(batch)["champ_logits"]
    batch2 = dict(batch, patch_id=(batch["patch_id"] + 1) % N_PATCHES)
    assert not torch.allclose(before, model(batch2)["champ_logits"])


def test_patch_embedding_receives_gradient():
    model = _model()
    out = model(_batch())
    loss = out["champ_logits"].sum() + out["role_logits"].sum()
    loss.backward()
    grad = model.embed.patch_lookup.weight.grad
    assert grad is not None and torch.count_nonzero(grad) > 0


def test_missing_patch_id_falls_back_to_unk_not_to_nothing():
    """A batch without patch_id (a soloQ rehearsal batch, an old client) must
    use the UNK row, not silently skip the embedding - the two differ once the
    UNK row has been trained."""
    model = _model()
    torch.nn.init.normal_(model.embed.patch_lookup.weight, std=0.5)
    batch = _batch()
    unk = dict(batch, patch_id=torch.zeros_like(batch["patch_id"]))
    no_key = {k: v for k, v in batch.items() if k != "patch_id"}
    assert torch.equal(model(no_key)["champ_logits"],
                       model(unk)["champ_logits"])


# ------------------------------------------------------- checkpoint asymmetry
def test_checkpoint_without_patch_rows_cannot_enter_a_patch_encoder():
    """Silently starting the table at zero would evaluate a different model
    than the one that was validated, so this must raise."""
    a, b = _model(), _model(n_patches=0)
    with pytest.raises(RuntimeError) as exc:
        load_encoder_state(a, b.state_dict())
    assert "lacks" in str(exc.value)


def test_patch_checkpoint_cannot_enter_a_plain_encoder():
    a, b = _model(), _model(n_patches=0)
    with pytest.raises(RuntimeError) as exc:
        load_encoder_state(b, a.state_dict())
    assert "does not know" in str(exc.value)


def test_same_vocabulary_round_trips():
    a, b = _model(), _model()
    info = load_encoder_state(b, a.state_dict())
    assert info["missing"] == [] and info["dropped"] == []
    batch = _batch()
    assert torch.equal(a(batch)["champ_logits"], b(batch)["champ_logits"])


def test_vocab_size_lives_in_the_config():
    """Every stage rebuilds the encoder from the checkpoint config, so the
    number that sizes the table has to be written there (and read back)."""
    model = _model()
    assert model.n_patches == N_PATCHES
    assert model.embed.patch_lookup.num_embeddings == N_PATCHES
    cfg = {"num_champs": C, "embed_dim": D, "n_patches": N_PATCHES}
    rebuilt = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=4, num_intra_layers=1, num_cross_layers=1,
        n_patches=cfg["n_patches"]).eval()
    load_encoder_state(rebuilt, model.state_dict())
    batch = _batch()
    assert torch.equal(rebuilt(batch)["champ_logits"],
                       model(batch)["champ_logits"])
