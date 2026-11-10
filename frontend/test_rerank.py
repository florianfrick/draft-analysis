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
