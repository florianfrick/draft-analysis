"""Tests for the reranker's plumbing: the mask, the dataset, the splits.

The failure these are guarding against is silent: a candidate list that does
not match the row it is shown for, or a training pool that overlaps the eval
set, produces numbers that look fine and mean nothing.
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.llm import common as C          # noqa: E402
from pro.model.llm import data as D            # noqa: E402
from pro.model.llm import rerank as R          # noqa: E402


def test_keep_mask_allows_the_candidates_and_the_target():
    """The target must survive the mask: a row whose answer is not in the list
    is scored as wrong, not as +inf loss."""
    cand = torch.tensor([[5, 9, -1], [7, -1, -1]])
    target = torch.tensor([3, 7])
    keep = R.keep_mask(cand, target, n_vocab=16)
    assert keep.shape == (2, 16)
    assert keep[0, [5, 9, 3]].all() and keep[0].sum() == 3
    assert keep[1, [7]].all() and keep[1].sum() == 1


def test_candidates_are_top_k_of_the_masked_logits():
    """The list is the head's ranking, and its probabilities come from the
    same masked logits - so an illegal champion can never appear in it.

    The k probabilities are the head's raw softmax over the *legal* set, not
    renormalised over k: that is what lets the model see a confident head
    (near 1.0 across the list) versus a flat one (0.2 across five).
    """
    logits = torch.full((1, 8), -1e4)
    logits[0, [1, 4, 6]] = torch.tensor([2.0, 5.0, 1.0])
    ids, probs = R.candidates(logits, 2)
    assert ids.tolist() == [[4, 1]]
    assert probs[0, 0] > probs[0, 1]
    # the two listed carry all but the third legal champion's share
    sm = torch.softmax(torch.tensor([2.0, 5.0, 1.0]), dim=0)
    assert pytest.approx(float(probs.sum()), abs=1e-5) == float(sm[:2].sum())
    assert float(probs.sum()) < 1.0


def test_pair_dataset_only_yields_usable_pairs():
    """A pair list is the *filtered* set; the dataset must not invent rows."""
    rows = C.load_rows()
    patch_index = D.make_patch_index(rows)[1]
    pairs = [(0, 3), (0, 7), (1, 0), (1, 11), (1, 12)]
    ds = D.PairDataset(rows, pairs, patch_index, per_game=1, seed=0,
                       swap_aug=False)
    allowed = {(0, 3), (0, 7), (1, 0), (1, 11), (1, 12)}
    seen = set()
    for epoch in (0, 1, 2):
        ds.set_epoch(epoch)
        # one per game, and never a move that is not in the list
        assert len(ds) == 2
        for i in range(len(ds)):
            r, m = ds._pick[i]
            assert (r, m) in allowed
            seen.add((r, m))
        sample = ds[0]
        assert int(sample["target_champ"]) >= 0
    # the redraw covers different moves across epochs, not the same one always
    assert len(seen) >= 2


def test_build_pairs_maps_every_batch_to_its_own_games():
    """The batch offset is the bug this test exists for.

    `build_pairs` reads `pos` out of a per-batch `nonzero()`; using it as a
    dataset position maps every batch onto the first `batch_size // 20` games.
    That produced 45,407 pairs over **13** games and still looked plausible in
    the log line - the run trained on 13 drafts in 2 steps.
    """
    rows = C.load_rows()
    patch_index = D.make_patch_index(rows)[1]
    games = [0, 1, 2, 3, 4, 5, 6, 7]
    pairs, n = R.build_pairs(rows, games, patch_index, _TargetHead(), "cpu", 5,
                             batch_size=8)
    assert n == len(games) * C.N_MOVES
    # the fake head always ranks the target first, so every row is usable
    assert len(pairs) == n
    assert len({g for g, _m in pairs}) == len(games)
    assert all(0 <= m < C.N_MOVES for _g, m in pairs)
    assert {g for g, _m in pairs} == set(games)


class _TargetHead:
    """A head that always puts the true answer first - 100 % recall@1."""

    def __call__(self, batch):
        tgt = batch["target_champ"]
        logits = torch.full((tgt.shape[0], C.NUM_REAL_CHAMPS), -1e4)
        logits[torch.arange(tgt.shape[0]), tgt] = 1.0
        return {"policy_logits": logits}


def test_the_three_stages_split_the_corpus_identically():
    """One split, so a head number and an LLM number are about the same games.

    This is the invariant the whole cross-model record rests on, and it was
    broken for months by a single `seed=0` in `common.SPLIT`: the LLM drew its
    own random 10% per patch instead of reusing `VAL_SPLIT_SEED`, so 90% of its
    val sat inside the head's train and every comparison shrank to a 478-game
    intersection. The head's trainer and the LLM must therefore produce the
    same three lists from the same rows, bit for bit.
    """
    rows = C.load_rows()
    llm = C.split_rows(rows)
    head = R.head_split(rows)
    for part in ("train", "val", "test"):
        assert llm[part] == head[part], "the %s lists differ" % part
    assert head["test"], "the pinned test series must survive the split"


def test_rr_split_keeps_games_whole_and_disjoint():
    rows = C.load_rows()
    train_idx, eval_idx = R.split_games(rows, "rr")
    assert train_idx and eval_idx
    assert not (set(train_idx) & set(eval_idx))
    val = {rows[i]["gameid"] for i in R.head_split(rows)["val"]}
    assert {rows[i]["gameid"] for i in train_idx} | \
        {rows[i]["gameid"] for i in eval_idx} == val
    assert abs(len(train_idx) - len(eval_idx)) <= 1


def test_heldout_split_has_no_training_rows():
    rows = C.load_rows()
    train_idx, eval_idx = R.split_games(rows, "heldout")
    assert train_idx == []
    assert len(eval_idx) == len(R.head_split(rows)["val"])


def test_the_retired_split_names_say_why_they_are_gone():
    rows = C.load_rows()
    for name in ("common", "headheld", "llmval"):
        with pytest.raises(SystemExit):
            R.split_games(rows, name)
