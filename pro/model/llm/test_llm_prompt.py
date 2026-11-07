"""Contract tests for the soft-prompt splice and the answer scoring.

Every test here exists because the corresponding bug was actually hit while
building this stage, and each was silent rather than loud - the wrong number
came out looking like a result:

* the placeholder landing in the middle of a word (the BPE split a marker that
  was not in the vocabulary), so the projector was trained on noise;
* the answer tokenized as `"Vi"` in the candidate table and `" Vi"` in the
  prompt, which made every target "not a candidate";
* comparing a candidate *index* against a vocabulary *id*, which scored every
  row as rank 0 - a perfect-looking top-1 of 1.000;
* the soft block shifting the answer positions by `n_soft - 1`, which the mask
  and the gather both have to agree on.

These run on CPU with the tokenizer only; no GPU, no checkpoint.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.llm import common as C          # noqa: E402
from pro.model.llm import data as D            # noqa: E402
from pro.model.llm.prompt import (PREFIX, assemble, build_text,  # noqa: E402
                              splice_positions, tokenize_answer)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    t = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    return t


@pytest.fixture(scope="module")
def names():
    return C.champ_names()


@pytest.fixture(scope="module")
def samples(names):
    rows = C.load_rows()
    split = C.split_rows(rows)
    _, patch_index = D.make_patch_index(rows)
    ds = D.eval_dataset(rows, split["val"], patch_index, val_games=1)
    return [ds[i] for i in range(C.N_MOVES)]


def test_placeholder_is_at_the_end_of_the_prefix(tok, names, samples):
    batch = assemble(tok, samples, names)
    expect = len(tok(PREFIX, add_special_tokens=False)["input_ids"])
    assert batch["ph_idx"] == expect


def test_scored_tokens_are_exactly_the_answer(tok, names, samples):
    batch = assemble(tok, samples, names)
    for i, s in enumerate(samples):
        want = tokenize_answer(tok, " " + names[int(s["target_champ"])])
        got = [t for t in batch["ans_ids"][i].tolist() if t >= 0]
        assert got == want


def test_loss_on_eos_only_appends_the_stop_token(tok, names, samples):
    plain = assemble(tok, samples[:2], names)
    with_eos = assemble(tok, samples[:2], names, loss_on_eos=True)
    eos = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
    for i in range(2):
        assert with_eos["n_ans"][i] == plain["n_ans"][i] + len(eos)
        got = [t for t in with_eos["ans_ids"][i].tolist() if t >= 0]
        assert got[-len(eos):] == eos


def test_answer_positions_shift_by_the_soft_block(tok, names, samples):
    batch = assemble(tok, samples, names)
    n_soft = 10
    shifted = splice_positions(batch, n_soft)
    live = batch["ans_pos_pre"] >= 0
    assert torch_equal(shifted[live], batch["ans_pos_pre"][live] + n_soft - 1)


def test_spliced_mask_has_the_spliced_width(tok, names, samples):
    batch = assemble(tok, samples, names)
    n_soft = 10
    width = batch["ids"].shape[1] - 1 + n_soft
    # mirrors DraftLLM.spliced_mask without building a model
    attn, p = batch["attn"], batch["ph_idx"]
    mask = torch_cat([attn[:, :p], attn.new_ones((attn.shape[0], n_soft)),
                      attn[:, p + 1:]], dim=1)
    assert mask.shape[1] == width
    # every answer position must land on a real (unmasked) token
    pos = splice_positions(batch, n_soft)
    for i in range(pos.shape[0]):
        for p_i in pos[i].tolist():
            if p_i >= 0:
                assert mask[i, p_i] == 1


def test_the_answer_is_not_already_in_the_prompt(tok, names, samples):
    """The target is the *next* move, so its tokens must not be readable off
    the board - otherwise the task is copy-paste and a high score means
    nothing."""
    for s in samples:
        name = names[int(s["target_champ"])]
        _, body, _ = build_text(s, names, answer=name)
        body_ids = tok(body, add_special_tokens=False)["input_ids"]
        ans_ids = tokenize_answer(tok, " " + name)
        assert not _contains(body_ids, ans_ids), (name, body)


def test_scored_positions_predict_rather_than_read_the_answer(tok, names, samples):
    """The tensor-level invariant the copy bug violated.

    Teacher forcing necessarily puts the answer in the input - position p
    predicts answer token p+1, so the earlier answer tokens have to be there.
    What must never happen is scoring *at* an answer token, because a causal
    model can then read it. `ids` used to be scored at `len(prompt)+j`, one
    token too late: the model saw the answer and scored 98.6 % first-token
    accuracy by copying it. Corrupting the target and watching the prediction
    follow the corruption is how that was found; this is the cheap version.
    """
    batch = assemble(tok, samples, names)
    for i in range(batch["ids"].shape[0]):
        n = int(batch["n_ans"][i])
        live = int(batch["attn"][i].sum())
        prompt_len = live - n
        for j in range(n):
            pos = int(batch["ans_pos_pre"][i, j])
            # position j sits exactly one token before answer token j
            assert pos == prompt_len - 1 + j, (i, j, pos, prompt_len)
        # the first scored position is the last prompt token, so it cannot
        # have seen any answer token
        assert int(batch["ans_pos_pre"][i, 0]) == prompt_len - 1
        assert int(batch["ans_pos_pre"][i, 0]) < prompt_len


def test_candidate_tables_cover_every_champion(tok, names):
    first, owner = C.candidate_tables(tok, names)
    assert owner.shape == (first.numel(), len(names))
    assert int(owner.sum()) == len(names)
    assert bool((owner.sum(dim=1) >= 1).all())
    # and every name's first token really is in the table
    for n in names:
        t = tok(" " + n, add_special_tokens=False)["input_ids"][0]
        assert t in set(first.tolist())


def test_swap_sides_mirrors_the_rendered_board(names, samples):
    """A side swap must move the picks in the prose, or the text and the soft
    tokens describe different drafts."""
    s = samples[19]                       # a late pick: both teams have picks
    swapped = D.DS.swap_sides(s)
    a = C.render_state(s, names).splitlines()
    b = C.render_state(swapped, names).splitlines()
    blue_a = next(x for x in a if x.startswith("Blue:"))
    red_b = next(x for x in b if x.startswith("Red:"))
    assert blue_a.split(":", 1)[1] == red_b.split(":", 1)[1]


# ------------------------------------------------------------------ helpers
def torch_equal(a, b):
    import torch
    return bool(torch.equal(a, b))


def torch_cat(tensors, dim):
    import torch
    return torch.cat(tensors, dim=dim)


def _contains(haystack, needle):
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))
