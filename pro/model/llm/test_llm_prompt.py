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
    nothing. Checked in both renderings: `v2` adds the turn label and the lane,
    which is exactly the kind of edit that could splice the answer in."""
    for fmt in ("v1", "v2"):
        for s in samples:
            name = names[int(s["target_champ"])]
            _, body, _ = build_text(s, names, answer=name, prompt_format=fmt)
            body_ids = tok(body, add_special_tokens=False)["input_ids"]
            ans_ids = tokenize_answer(tok, " " + name)
            assert not _contains(body_ids, ans_ids), (fmt, name, body)


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
    blue_a = next(x for x in a if x.startswith("Blue Picks:"))
    red_b = next(x for x in b if x.startswith("Red Picks:"))
    assert blue_a.split(":", 1)[1] == red_b.split(":", 1)[1]
    # and the turn label has to swap with it, or the model is told the wrong
    # side is acting on the board it is reading
    turn_a = _turn_of(a[0])
    turn_b = _turn_of(b[0])
    assert turn_a.replace("Blue", "X").replace("Red", "Blue") == \
        turn_b.replace("Red", "X")


def test_v2_bans_are_split_by_side_and_nothing_is_lost(names, samples):
    """The side of a ban is derived (`BAN_SIDE_BY_SLOT ^ first_pick`). If the
    derivation were wrong the two lists would still look plausible, so the
    check is a multiset identity plus a slot-order check."""
    for s in samples:
        n = int(s["n_bans"])
        if not n:
            continue
        lines = C.render_state(s, names).splitlines()
        sides = C.DS.ban_sides(int(s["first_pick"]))
        got = {}
        for team in (0, 1):
            want = [names[int(s["ban_ids"][i])] for i in range(n)
                    if int(sides[i]) == team]
            got[team] = want
            line = [x for x in lines
                    if x.startswith(C.SIDE_NAMES[team] + " Bans:")]
            if not want:
                # omitted rather than written as "none" - that is the terse
                # rule the renderer documents
                assert not line, (team, line)
                continue
            listed = [x.strip() for x in line[0].split(":", 1)[1].split(",")]
            # slot order is chronological and the split must preserve it
            assert listed == want, (team, listed, want)


def test_v2_turn_line_counts_the_board(names, samples):
    """'Turn: Red 2 (Pick 4 of 10)' must be read off the board, not tracked.

    Both numbers are recomputed here from the picks and bans themselves, so a
    label that drifts from the board (the failure that makes the lane hint
    actively misleading) fails this test instead of silently training.
    """
    for i, s in enumerate(samples):
        line = _turn_of(C.render_state(s, names).splitlines()[0])
        to_move = int(s["to_move"])
        n_b = int((~s["pad_mask"][:5]).sum())
        n_r = int((~s["pad_mask"][5:]).sum())
        if bool(int(s["is_ban"])):
            sides = C.DS.ban_sides(int(s["first_pick"]))
            made = sum(1 for k in range(int(s["n_bans"]))
                       if int(sides[k]) == to_move)
            want = "Turn: %s %d (Ban %d of 10)" % (
                C.SIDE_NAMES[to_move], made + 1, int(s["n_bans"]) + 1)
        else:
            made = n_b if to_move == 0 else n_r
            want = "Turn: %s %d (Pick %d of 10)" % (
                C.SIDE_NAMES[to_move], made + 1, n_b + n_r + 1)
        assert line == want, (i, line, want)


def test_v2_task_line_names_the_lane_of_the_pick(names, samples):
    """The point of the format change: the question says which lane to fill.

    A ban has no lane, so it must not be asked for one - a ban asked "for Top
    lane" would teach the model that the label means something it does not.
    """
    for s in samples:
        body = C.task_line(s)
        if bool(int(s["is_ban"])):
            assert " ban." in body and "lane" not in body, body
            assert body.startswith("Task: Recommend the optimal %s Side ban."
                                   % C.SIDE_NAMES[int(s["to_move"])]), body
        else:
            lane = C.ROLE_DISPLAY[int(s["target_role"])]
            assert body == ("Task: Recommend the optimal %s Side pick for %s "
                            "lane. Answer with champion name only."
                            % (C.SIDE_NAMES[int(s["to_move"])], lane)), body


def test_v1_still_renders_the_old_board(names, samples):
    """`v1` is frozen so the pre-2026-11-08 numbers stay reproducible: its
    first line is the old `Patch ... | move n/20 ...` header."""
    for s in samples[:3]:
        first = C.render_state(s, names, prompt_format="v1").splitlines()[0]
        assert first.startswith("Patch ")
        assert " | move " in first and "first pick" in first
        assert "Turn:" in C.render_state(s, names).splitlines()[0]


# ------------------------------------------------------------------ helpers
def _turn_of(first_line):
    """The `Turn: ...` half of the `Patch: ... | Turn: ...` header."""
    head, turn = first_line.split(" | ", 1)
    assert head.startswith("Patch: ") and turn.startswith("Turn: ")
    return turn


def torch_equal(a, b):
    import torch
    return bool(torch.equal(a, b))


def torch_cat(tensors, dim):
    import torch
    return torch.cat(tensors, dim=dim)


def _contains(haystack, needle):
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


# ------------------------------------------------------------------ v4 / reranker
def test_v4_is_v2_plus_one_candidate_line(names, samples):
    """`v4` must not disturb the board: the reranker's board has to be the
    exact text the projector was trained on, or the comparison between a
    reranked answer and a free-generation answer is also a prompt change."""
    cands = [(3, 0.31), (7, 0.22), (11, 0.14)]
    for s in samples[:4]:
        v2 = C.render_state(s, names, prompt_format="v2").splitlines()
        v4 = C.render_state(s, names, prompt_format="v4",
                            candidates=cands).splitlines()
        assert v4[:-1] == v2, "v4 changed a board line"
        assert v4[-1].startswith("Candidates: ")


def test_v4_lists_the_candidates_in_the_order_given(names, samples):
    """The order is the head's ranking, so it must not be sorted or deduped."""
    ids = [9, 2, 40]
    cands = [(i, 0.5 - 0.1 * j) for j, i in enumerate(ids)]
    line = C.render_state(samples[6], names, prompt_format="v4",
                          candidates=cands).splitlines()[-1]
    body = line[len("Candidates: "):]
    shown = [part.rsplit(" ", 1)[0] for part in body.split(", ")]
    assert shown == [names[i] for i in ids]


def test_v4_refuses_to_render_without_candidates(names, samples):
    """A v4 prompt with no list would ask for a choice from nothing while the
    decode is masked to nothing - fail at render time, not at scoring time."""
    with pytest.raises(ValueError):
        C.render_state(samples[0], names, prompt_format="v4")


def test_v4_task_line_asks_for_a_choice_from_the_list(names, samples):
    picks = [s for s in samples if not bool(int(s["is_ban"]))]
    bans = [s for s in samples if bool(int(s["is_ban"]))]
    assert "from the candidates" in C.task_line(picks[0], prompt_format="v4")
    assert "from the candidates" in C.task_line(bans[0], prompt_format="v4")
    assert "from the candidates" not in C.task_line(picks[0])


def test_candidate_percentages_stay_readable():
    """Integer percent, with a decimal only when the value would round to 0 -
    a 5-way list of "0%" is worse than useless to the model."""
    assert C._fmt_pct(0.31) == "31%"
    assert C._fmt_pct(0.999) == "100%"
    assert C._fmt_pct(0.004) == "0.4%"
    assert C._fmt_pct(0.0) == "0.0%"


def test_candidate_first_ids_are_the_names_first_tokens(tok, names, samples):
    """The decode mask is over first tokens, so this table is the mask."""
    from pro.model.llm.prompt import candidate_first_ids
    cands = [[(3, 0.5), (7, 0.3)], [(9, 0.6), (11, 0.2)]]
    table = candidate_first_ids(tok, names, cands)
    assert table.shape == (2, 2)
    for row, cand in zip(table.tolist(), cands):
        want = [tok(" " + names[c], add_special_tokens=False)["input_ids"][0]
                for c, _p in cand]
        assert row == want


def test_patch_index_is_bounded_by_the_encoder_table():
    """A patch newer than the encoder's table must fall to UNK, not shift ids.

    This is the failure that raising the corpus freeze would otherwise cause:
    rebuilding the vocabulary from the corpus gives 16.22 a new id, and the
    encoder then reads the wrong embedding row for *every* row in the run - or
    indexes past its table. The index has to come from the checkpoint.
    """
    vocab = C.encoder_patch_vocab()
    assert vocab, "the encoder checkpoint carries no patch_vocab"
    keys, index = D.make_patch_index([], vocab=vocab)
    assert keys == vocab
    assert max(index.values()) == len(vocab), "ids must be 1..len(vocab)"
    assert D.DS.PATCH_UNK_ID == 0
    # a patch that is not in the table maps to the reserved row
    assert D.DS.patch_to_id("99.99", index) == D.DS.PATCH_UNK_ID
    # ... and the newest one in the corpus is either known or UNK, never out of
    # range - that is the invariant the encoder depends on
    for k in keys:
        assert 0 <= D.DS.patch_to_id(k, index) <= len(vocab)
