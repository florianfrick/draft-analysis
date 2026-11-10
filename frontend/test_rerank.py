"""Tests for the LLM reranker's serving path that do not need a GPU.

The heavy half - loading Qwen, building the prompt, scoring the candidates -
needs CUDA and the LLM image, so it is skipped here rather than faked. What is
left is the part that can be wrong without any of that: the decision rule, the
capability report, and the dropdown's honesty about what this machine can load.

The decision rule is the one that matters. It has to be the argmax the reported
`ft_top1` measured, not a differently-shaped quantity that happens to score
better on some other set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import rerank as R  # noqa: E402


def test_winner_is_the_masked_first_token_argmax():
    """The served pick is the argmax over candidates of the first-token logp."""
    # Candidate 2 is the most likely at the decision point.
    first = [-2.1, -0.4, -3.0, -1.2]
    total = [-4.0, -5.5, -7.0, -6.0]
    toks = [101, 202, 303, 404]
    assert R.pick_winner(first, total, toks) == 1


def test_a_first_token_collision_is_broken_by_the_whole_name():
    """Two candidates sharing a first token: the first position cannot split
    them, so the longer-horizon score does. 61 of 173 champions share a first
    token with another, so this is not a corner case."""
    first = [-1.0, -1.0, -2.5]      # 0 and 1 tie at the decision point
    total = [-6.0, -3.2, -8.0]      # candidate 1 spells out more plausibly
    toks = [500, 500, 600]
    assert R.pick_winner(first, total, toks) == 1
    # The same numbers with distinct first tokens must not consult `total`.
    assert R.pick_winner(first, [-6.0, -9.9, -8.0], [500, 501, 600]) == 0


def test_the_decode_mask_is_the_same_set_for_every_row():
    """The bug this exists to catch, which shipped once.

    Giving each row a mask containing only *its own* answer is satisfiable by
    construction: every candidate scores log P = 0, the softmax comes out
    uniform at 1/k, and it reads as a perfectly confident model rather than a
    broken constraint. The mask has to be one set shared by the batch.
    """
    import torch
    tok = torch.tensor([11, 22, 33, 44, 55])
    keep = R.candidate_mask(tok, 100)
    assert tuple(keep.shape) == (5, 100)
    assert [int(row.sum()) for row in keep] == [5] * 5
    assert keep.all(dim=0).sum() == 5          # same five columns, every row
    # A first-token collision shrinks the mask rather than duplicating a column.
    dup = R.candidate_mask(torch.tensor([11, 11, 33, 44, 55]), 100)
    assert int(dup[0].sum()) == 4


def test_a_single_candidate_is_chosen():
    assert R.pick_winner([-0.7], [-1.0], [42]) == 0


def test_capability_report_is_a_string_or_none():
    """`unavailable_reason` must explain, not return a bare bool.

    On a CPU-only host it is the string that the dropdown shows; on a GPU host
    with the LLM image it is None. Either way it must not raise.
    """
    reason = R.unavailable_reason()
    assert reason is None or (isinstance(reason, str) and reason.strip())


def test_the_unavailable_reason_names_a_command_that_works():
    """The message is read by someone who cannot fix it from that screen.

    Its first version said `bash pro/model/llm/run.sh python frontend/server.py`
    - which runs, and publishes no port, so the browser could never reach it.
    A remedy that cannot work is worse than no remedy, so the string has to
    name the wrapper that does both jobs.
    """
    reason = R.unavailable_reason()
    if reason is None:
        pytest.skip("this machine can serve the reranker")
    if "not installed" not in reason:
        pytest.skip("a different failure mode: %s" % reason)
    assert "frontend/serve.sh" in reason


def test_serve_script_publishes_the_port_in_the_llm_image():
    """`serve.sh` exists because the two obvious commands each miss one half.

    `docker compose run dev` has no transformers; `pro/model/llm/run.sh` has
    them but no `-p`. This asserts the script keeps doing both, and that the
    compose service it targets still exists.
    """
    sh = (HERE / "serve.sh").read_text("utf-8")
    assert "docker compose run" in sh
    assert "-p \"127.0.0.1:${PORT}:${PORT}\"" in sh, "no published port"
    assert "llm python frontend/server.py" in sh, "not the llm service"
    assert "--host 0.0.0.0" in sh, "a published port lands on eth0, not loopback"
    compose = (ROOT / "docker-compose.yml").read_text("utf-8")
    assert "\n  llm:" in compose
    assert "pro/model/llm/Dockerfile" in compose


def test_the_llm_checkpoint_is_offered_in_the_policy_slot():
    """Discovery: the folder is a model, and it answers the next move."""
    import server as S
    cps = {Path(c["path"]).name: c for c in S.list_checkpoints()}
    entry = cps.get("projector.pt")
    if entry is None:
        pytest.skip("no llm-* checkpoint in this checkout")
    assert entry["roles"] == ["policy"]
    assert entry.get("family") == "llm-rerank"
    assert entry["available"] == (R.unavailable_reason() is None)
    if not entry["available"]:
        assert entry["reason"]


def test_the_reranker_is_the_default_only_when_it_can_run():
    """The default is conditional by design.


    Defaulting to a 4.2B model on a host that cannot run it would break the
    next-move panel on the first click, so the head must be what is used when
    `unavailable_reason()` is not None.
    """
    import server as S
    if not S.RERANK_DEFAULT.is_dir():
        pytest.skip("no llm-* checkpoint in this checkout")
    reason = R.unavailable_reason()
    if reason is None:
        assert (S.RERANK_DEFAULT / "projector.pt").is_file()
    else:
        # Nothing is loaded at import time, so all this can assert is that the
        # head is still a valid fallback and the folder really is the one the
        # default would pick.
        assert S.POLICY_DEFAULT.is_file()
        assert (S.RERANK_DEFAULT / "projector.pt").is_file()


def test_resolving_the_checkpoint_folder_finds_its_projector():
    import server as S
    if not S.RERANK_DEFAULT.is_dir():
        pytest.skip("no llm-* checkpoint in this checkout")
    got = S.resolve_ckpt_path(S.RERANK_DEFAULT, confine=False)
    assert got.name == "projector.pt"


# ---------------------------------------------------------------- lane legality
#
# The lane selector used to be a wording change: the prompt said "for Bot lane"
# and the head's lane-blind top-5 came back regardless, so 22 % of lists had no
# Bot champion in them at all. These pin the fix - the lane now masks the pool -
# and the rule that decides when it may.

def test_the_lane_table_is_well_formed():
    """Every lane has a real pool, and every id is a champion index.

    A lane with a handful of champions would make the masked top-5 repeat
    itself; an id outside the 173 would be a silent -inf or a crash.
    """
    import roles as ROLES
    pools = [ROLES.legal_champions(l) for l in range(5)]
    for lane, pool in enumerate(pools):
        assert len(pool) >= 20, "%s has only %d champions" % (
            ROLES.lane_name(lane), len(pool))
        assert all(0 <= c < 173 for c in pool)
        assert len(set(pool)) == len(pool), "duplicate champion"
    # Flex exists: the lanes are not a partition, and the single-lane
    # champions are the majority (113 of 173 when the table was built).
    total = sum(len(p) for p in pools)
    assert total > 173, "no champion plays two lanes - the table is wrong"
    assert total < 2 * 173, "every champion plays every lane - the table is wrong"


def test_a_lane_outside_the_five_is_refused():
    import roles as ROLES
    for bad in (-1, 5, 99):
        with pytest.raises(ValueError):
            ROLES.legal_champions(bad)


def test_a_chosen_lane_constrains_the_pool_but_a_guess_does_not():
    """`_resolve_lane` decides which of the three it is, and only two constrain.

    "user" (the drafter picked it) and "board" (four lanes filled, so the last
    one is forced) are decisions the answer must respect. "default" is the
    first open lane, which is only a guess: filtering on it would make a wrong
    guess unrecoverable, because the right champion would not be in the pool.
    """
    empty = {"kind": "pick", "side": 0, "roles": [[] for _ in range(10)]}
    picks = [None] * 10

    role, source, open_lanes = R.RerankPredictor._resolve_lane(
        None, empty, picks, 3)
    assert (role, source) == (3, "user")
    assert open_lanes == [0, 1, 2, 3, 4]

    role, source, _open = R.RerankPredictor._resolve_lane(None, empty, picks,
                                                          None)
    assert source == "default", "with five lanes open nothing is decided"
    assert role == 0

    # Four of blue's five lanes taken -> the fifth is forced, not guessed.
    four = {"kind": "pick", "side": 0, "roles": [[] for _ in range(10)]}
    filled = [None] * 10
    for slot, lane in zip(range(4), (0, 1, 2, 4)):
        four["roles"][slot] = [{"role": lane, "prob": 1.0}]
        filled[slot] = 100 + slot
    role, source, open_lanes = R.RerankPredictor._resolve_lane(
        None, four, filled, None)
    assert open_lanes == [3]
    assert (role, source) == (3, "board")

    # A ban has no lane at all, whatever the caller sent.
    ban = {"kind": "ban", "side": 0, "roles": [[] for _ in range(10)]}
    assert R.RerankPredictor._resolve_lane(None, ban, picks, 3) == (None, None, [])
