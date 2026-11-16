"""The explained prompt is a contract: the numbers and the tool list are claims.

The reason to pin these in a test rather than in a comment is that both halves
drift silently. A prompt that says "top-5 is 51 %" while the run measures 44 %
does not crash - it teaches the model a miscalibration and the run reports it as
the model's fault. And a prompt that still tells the agent to reach for `sql`
after the tool was ablated sends it at a tool it cannot call, which looks like a
tool-calling failure and is a prompt bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.agent import model_tools, prompt_explained as P  # noqa: E402


def test_the_prompt_states_the_measured_accuracy():
    s = P.SYSTEM_EXPLAINED
    assert "18.6%" in s and "51.1%" in s and "85.2%" in s
    assert "wrong about 8 times in 10" in s


def test_a_rerun_can_restate_the_numbers_without_rewriting_the_prompt():
    """The harness measures its own slice; the prompt has to be able to follow."""
    s = P.system_prompt({"top5": 0.4412})
    assert "44.1%" in s and "51.1%" not in s
    assert "46,915" in s                      # untouched defaults survive


def test_the_prompt_names_only_the_tools_this_arm_has():
    s = P.SYSTEM_EXPLAINED
    for name in ("get_state", "predict_roles", "predict_next_move"):
        assert name in s, name
    for name in ("rank_candidates", "sql"):
        assert name not in s, name


def test_the_k_guidance_matches_the_tool_schema():
    """A prompt that says k up to 20 while the schema caps at 8 is a dead end."""
    props = model_tools.PREDICT_SPEC["parameters"]["properties"]["k"]
    assert props["maximum"] == 20 and props["minimum"] == 1
    assert "k=20" in P.SYSTEM_EXPLAINED and "k=1" in P.SYSTEM_EXPLAINED


def test_the_prompt_says_the_model_cannot_tell_blue_from_red():
    """Measured: neither side is ever differentiated. An agent that believes
    otherwise will ask the head a question it has no answer to."""
    assert "cannot tell blue from red" in P.SYSTEM_EXPLAINED


def test_the_prompt_forbids_an_unbacked_number():
    assert "no tool returned" in P.SYSTEM_EXPLAINED
