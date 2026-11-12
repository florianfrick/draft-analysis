"""Tests for the loop and the MCP server, without a model and without a key.

Both files are about *wiring*, so everything here drives them with a fake client
and a synthetic corpus. The two things that cannot be tested this way - whether
the real model picks good tools, and whether its answers are right - are what the
loop exists to measure, not what these tests can assert.

The corpus helpers are imported from `test_agent` on purpose: a second, slightly
different fixture would let the two files disagree about what a row looks like.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.agent import llm_client, loop, mcp_server, schema  # noqa: E402
from pro.model.agent.session import Session                        # noqa: E402
from pro.model.agent.test_agent import SPLIT, _rows                # noqa: E402

BOARD = {"blue_picks": ["Aatrox", "Vi", "Ahri"], "red_picks": ["Poppy", "Lee Sin"],
         "bans": ["Kled", "Olaf"], "first_pick": 1, "patch": "16.20"}


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("loop") / "query.sqlite"
    schema.build(_rows(), path, known_patches={"16.20"}, split=SPLIT)
    return schema.connect(path)


@pytest.fixture()
def session(con):
    return Session(con=con, max_calls=6)


# ---------------------------------------------------------------- fake model
def reply(content="", calls=(), model="fake"):
    """-> an assistant message in the shape `llm_client` normalises to."""
    return {"role": "assistant", "content": content, "model": model,
            "tool_calls": [{"id": "call_%d" % i, "name": n, "args": a}
                           for i, (n, a) in enumerate(calls)]}


def client_of(*replies):
    """A client that returns `replies` in order, then a plain answer."""
    queue = list(replies)

    def call(messages, tools=None):
        if queue:
            return queue.pop(0)
        return reply("final answer")

    call.seen = []
    return call


def client_that_records():
    """A client that always asks for the same tool, and remembers what it saw."""
    seen = []

    def call(messages, tools=None):
        seen.append([t["function"]["name"] for t in (tools or [])])
        return reply("", [("sql", {"query": "SELECT 1 AS one"})])

    call.seen = seen
    return call


# ---------------------------------------------------------------------- run
def test_the_board_is_filled_when_the_model_omits_it(session):
    """Every model tool takes the whole board; re-typing nine names is the tax."""
    out = loop.run(session, "Kled or Olaf?",
                   client=client_of(reply("", [("rank_candidates",
                                                {"candidates": ["Xayah", "Jinx"]})])),
                   board=BOARD)
    assert out["stop"] == "answered"
    assert set(out["transcription"]["filled"]) == {
        "rank_candidates.blue_picks", "rank_candidates.red_picks",
        "rank_candidates.bans", "rank_candidates.patch",
        "rank_candidates.first_pick"}
    assert out["transcription"]["mismatched"] == []
    # And the tool really was called with the board, not with an empty one.
    args = session.log[0]["args"]
    assert "Aatrox" in args["blue_picks"] and "16.20" in args["patch"]


def test_a_board_the_model_retypes_differently_is_flagged_not_corrected(session):
    """Silently fixing the model's board would hide the likeliest failure here."""
    out = loop.run(session, "Kled or Olaf?",
                   client=client_of(reply("", [("rank_candidates",
                                                {"candidates": ["Xayah", "Jinx"],
                                                 "blue_picks": ["Vayne"],
                                                 "red_picks": ["Poppy", "Lee Sin"],
                                                 "bans": ["Kled", "Olaf"],
                                                 "patch": "16.20",
                                                 "first_pick": 1})])),
                   board=BOARD)
    assert out["transcription"]["mismatched"] == ["rank_candidates.blue_picks"]
    assert out["transcription"]["filled"] == []
    # Not corrected: the tool answered about the board it was asked about.
    assert "Vayne" in session.log[0]["args"]["blue_picks"]


def test_an_omitted_field_is_filled_but_a_contradicted_one_is_not(session):
    """The two rules have to coexist in one call, not just in isolation."""
    out = loop.run(session, "x",
                   client=client_of(reply("", [("rank_candidates",
                                                {"candidates": ["Xayah", "Jinx"],
                                                 "blue_picks": ["Vayne"]})])),
                   board=BOARD)
    assert out["transcription"]["filled"] == [
        "rank_candidates.red_picks", "rank_candidates.bans",
        "rank_candidates.patch", "rank_candidates.first_pick"]
    assert out["transcription"]["mismatched"] == ["rank_candidates.blue_picks"]


def test_no_key_is_a_recorded_outcome_not_a_crash(session):
    out = loop.run(session, "Kled or Olaf?", env={})
    assert out["stop"] == "no client"
    assert "LLM_API_KEY" in out["error"]
    # Whatever the session would have offered, not a hard-coded list: the
    # fixture has no predictor, so only `sql` is available here.
    assert out["tools_offered"] == session.names()
    assert out["summary"]["calls"] == 0


def test_the_answer_is_the_last_assistant_message(session):
    out = loop.run(session, "Kled or Olaf?",
                   client=client_of(reply("", [("sql", {"query":
                                                        "SELECT 1 AS one"})]),
                                    reply("Pick Xayah.")),
                   board=BOARD)
    assert out["answer"] == "Pick Xayah."
    assert out["stop"] == "answered"
    assert [c["name"] for c in out["calls"]] == ["sql"]


def test_the_loop_stops_at_the_step_budget_and_keeps_the_log(session):
    """A graph that runs out of steps must return, not raise."""
    out = loop.run(session, "x", client=client_that_records(), board=BOARD,
                   max_steps=2)
    assert out["stop"] == "step budget"
    assert out["answer"] is None
    assert len(out["calls"]) == 2          # two tool rounds, then the router
    assert out["summary"]["by_tool"] == {"sql": 2}


def test_a_tool_error_does_not_end_the_loop(session):
    out = loop.run(session, "x",
                   client=client_of(reply("", [("sql", {"query": "DELETE FROM move"})]),
                                    reply("I could not query that.")),
                   board=BOARD)
    assert out["stop"] == "answered"
    assert out["summary"]["errors"] == 1
    assert out["answer"] == "I could not query that."


def test_an_llm_error_is_reported_in_the_stop_reason(session):
    def boom(messages, tools=None):
        raise llm_client.LLMError("HTTP 401 from x: bad key")

    out = loop.run(session, "x", client=boom, board=BOARD)
    assert out["stop"].startswith("llm error")
    assert "401" in out["stop"]


def test_a_crashing_client_does_not_lose_the_log(session):
    def boom(messages, tools=None):
        raise RuntimeError("kaboom")

    out = loop.run(session, "x", client=boom, board=BOARD)
    assert out["stop"] == "loop error"
    assert "kaboom" in out["error"]
    assert out["summary"]["calls"] == 0
    with pytest.raises(RuntimeError):
        loop.run(session, "x", client=boom, board=BOARD, raise_errors=True)


# ------------------------------------------------------------------ ablation
def test_an_ablated_tool_is_not_advertised(con):
    """Hiding the spec is the shipped configuration: the model never sees it."""
    # Sentinels, not a real head: `available()` only asks whether a predictor
    # and a name index were loaded, and this test is about the listing.
    session = Session(con=con, predictor=object(), champs=object(),
                      ablate={"sql"})
    client = client_of(reply("no tools needed"))
    out = loop.run(session, "x", client=client, board=BOARD)
    assert out["tools_offered"] == ["predict_next_move", "rank_candidates"]
    assert out["calls"] == []
    assert out["answer"] == "no tools needed"


def test_an_ablated_tool_is_also_refused_at_call_time(con):
    """A model that invents a tool name gets a refusal it can read, not a crash."""
    session = Session(con=con, ablate={"sql"})
    out = loop.run(session, "x",
                   client=client_of(reply("", [("sql", {"query": "SELECT 1"})]),
                                    reply("no sql")),
                   board=BOARD)
    assert out["summary"]["ablated"] == 1
    assert out["summary"]["errors"] == 1
    assert out["stop"] == "answered"
    tool_msg = [m for m in out["messages"] if m["role"] == "tool"][0]
    assert "not available" in tool_msg["content"]


def test_every_call_goes_through_the_session(con):
    """The log is the measurement; a direct call would bypass it."""
    session = Session(con=con)
    loop.run(session, "x", client=client_of(reply("", [("sql", {"query":
                                                                "SELECT 1 AS one"})]),
                                            reply("ok")), board=BOARD)
    assert len(session.log) == 1
    assert session.log[0]["name"] == "sql"
    assert session.log[0]["ok"] is True
    assert session.log[0]["ms"] >= 0


# -------------------------------------------------------------- the prompt
def test_the_prompt_ranks_the_tools_by_reliability():
    """The order in the prompt is the design decision; pin it."""
    i_rank = loop.SYSTEM.index("rank_candidates")
    i_pred = loop.SYSTEM.index("predict_next_move")
    i_sql = loop.SYSTEM.index("`sql`")
    assert i_rank < i_pred < i_sql


def test_the_prompt_says_the_two_probabilities_are_not_comparable():
    assert "p_head" in loop.SYSTEM and "p_rerank" in loop.SYSTEM
    assert "different claims" in loop.SYSTEM


def test_the_prompt_tells_the_model_it_cannot_see_side():
    assert "cannot see which side is blue" in loop.SYSTEM


def test_the_tools_are_offered_in_the_openai_shape(session):
    tools = loop._openai_tools(session)
    assert [t["function"]["name"] for t in tools] == session.names()
    for t in tools:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


def test_the_board_renders_the_way_a_user_would_type_it():
    text = loop.render_board(BOARD)
    assert "Aatrox, Vi, Ahri" in text
    assert "first pick: blue" in text
    assert "16.20" in text
    assert "not given" in loop.render_board({})


# ------------------------------------------------------------- the env contract
def test_dotenv_is_read_because_the_repo_keeps_its_keys_there(tmp_path):
    """docker compose injects `.env` for the frontend; a host script must too."""
    p = tmp_path / ".env"
    p.write_text("# comment\n\nLLM_API_KEY='sk-test'\nLLM_MODEL=deepseek-v4.1-flash\n",
                 encoding="utf-8")
    env = {}
    added = llm_client.load_dotenv(p, env)
    assert added == ["LLM_API_KEY", "LLM_MODEL"]
    assert env["LLM_API_KEY"] == "sk-test"          # quotes stripped
    assert env["LLM_MODEL"] == "deepseek-v4.1-flash"


def test_dotenv_never_overrides_the_shell(tmp_path):
    """A key exported for one run must not be silently replaced by the file."""
    p = tmp_path / ".env"
    p.write_text("LLM_API_KEY=from-file\n", encoding="utf-8")
    env = {"LLM_API_KEY": "from-shell"}
    assert llm_client.load_dotenv(p, env) == []
    assert env["LLM_API_KEY"] == "from-shell"


def test_an_empty_value_in_the_file_does_not_count_as_set(tmp_path):
    p = tmp_path / ".env"
    p.write_text("LLM_API_KEY=\n", encoding="utf-8")
    env = {}
    assert llm_client.load_dotenv(p, env) == []
    assert "LLM_API_KEY" not in env


def test_a_missing_dotenv_is_not_an_error(tmp_path):
    assert llm_client.load_dotenv(tmp_path / "nope", {}) == []


def test_describe_says_what_a_run_would_use():
    cfg = llm_client.describe({})
    assert cfg["model"] == llm_client.DEFAULT_MODEL
    assert cfg["has_key"] is False
    cfg = llm_client.describe({"LLM_API_KEY": "k", "LLM_MODEL": "m",
                               "LLM_BASE_URL": "https://x/v1/"})
    assert cfg == {"model": "m", "base_url": "https://x/v1", "has_key": True}


def test_the_no_key_message_names_the_model_it_would_have_used(session):
    """The dead end this closes: a key set in `.env` and a loop that ignored it."""
    out = loop.run(session, "x", env={"LLM_MODEL": "deepseek-flash"})
    assert "deepseek-flash" in out["error"]
    assert "--scripted" in out["error"]


def test_ping_without_a_key_says_so():
    rep = loop.ping(env={})
    assert rep["ok"] is False
    assert rep["error"] == "no LLM_API_KEY"
    assert rep["model"] == llm_client.DEFAULT_MODEL


def test_ping_reports_latency_and_whether_tools_work():
    """The three ways a provider is wrong look identical from inside `run`."""
    def client(messages, tools=None):
        return reply("", [("sql", {"query": "SELECT 1"})]) if tools \
            else reply("ok")

    rep = loop.ping(client=client, env={"LLM_API_KEY": "k", "LLM_MODEL": "m"})
    assert rep["ok"] is True
    assert rep["answer"] == "ok"
    assert rep["tool_calls"] == ["sql"]
    assert rep["latency_ms"] >= 0
    assert rep["model"] == "m"


def test_ping_separates_a_key_that_cannot_call_tools():
    def client(messages, tools=None):
        return reply("ok" if not tools else "I have no tools")

    rep = loop.ping(client=client, env={"LLM_API_KEY": "k"})
    assert rep["ok"] is True
    assert rep["tool_calls"] == []      # answers, but the loop cannot use it


def test_ping_reports_an_http_error_instead_of_raising():
    def client(messages, tools=None):
        raise llm_client.LLMError("HTTP 404 from https://x: model not found")

    rep = loop.ping(client=client, env={"LLM_API_KEY": "k"})
    assert rep["ok"] is False
    assert "404" in rep["error"]


# --------------------------------------------------------------------- mcp
def _serve(session, messages):
    buf = io.StringIO()
    n = mcp_server.serve(session, stdin=io.StringIO(
        "\n".join(json.dumps(m) for m in messages) + "\n"),
        stdout=buf, log=io.StringIO())
    return n, [json.loads(line) for line in buf.getvalue().splitlines()]


def test_initialize_echoes_a_version_the_client_asked_for(session):
    _n, out = _serve(session, [{"jsonrpc": "2.0", "id": 1,
                                "method": "initialize",
                                "params": {"protocolVersion": "2024-11-05"}}])
    assert out[0]["result"]["protocolVersion"] == "2024-11-05"
    assert out[0]["result"]["capabilities"]["tools"]["listChanged"] is False
    assert out[0]["result"]["serverInfo"]["name"] == "lol-draft-tools"


def test_an_unknown_protocol_version_falls_back_to_ours(session):
    _n, out = _serve(session, [{"jsonrpc": "2.0", "id": 1,
                                "method": "initialize",
                                "params": {"protocolVersion": "1999-01-01"}}])
    assert out[0]["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION


def test_tools_list_matches_the_session(session):
    _n, out = _serve(session, [{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    tools = out[0]["result"]["tools"]
    assert [t["name"] for t in tools] == session.names()
    for t in tools:
        assert t["inputSchema"]["type"] == "object"
        assert t["description"]


def test_tools_list_hides_an_ablated_tool(con):
    session = Session(con=con, ablate={"sql"})
    _n, out = _serve(session, [{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    assert "sql" not in [t["name"] for t in out[0]["result"]["tools"]]


def test_a_tool_result_is_json_text_in_a_content_block(session):
    _n, out = _serve(session, [{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                "params": {"name": "sql",
                                           "arguments": {"query":
                                                         "SELECT COUNT(*) AS n FROM move"}}}])
    res = out[0]["result"]
    assert res["isError"] is False
    assert res["content"][0]["type"] == "text"
    assert json.loads(res["content"][0]["text"])["rows"] == [{"n": 120}]


def test_a_refused_call_is_a_result_not_a_protocol_error(session):
    """The agent has to be able to read the refusal and retry."""
    _n, out = _serve(session, [{"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                "params": {"name": "sql",
                                           "arguments": {"query": "DELETE FROM move"}}}])
    assert "error" not in out[0]              # no JSON-RPC error object
    assert out[0]["result"]["isError"] is True
    assert "SELECT" in json.loads(out[0]["result"]["content"][0]["text"])["error"]


def test_an_unknown_tool_is_refused_by_the_session(session):
    _n, out = _serve(session, [{"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                "params": {"name": "nope", "arguments": {}}}])
    assert out[0]["result"]["isError"] is True
    body = json.loads(out[0]["result"]["content"][0]["text"])
    assert "unknown tool" in body["error"]


def test_an_unknown_method_is_a_protocol_error(session):
    _n, out = _serve(session, [{"jsonrpc": "2.0", "id": 6, "method": "no/such"}])
    assert out[0]["error"]["code"] == -32601


def test_a_notification_gets_no_response(session):
    n, out = _serve(session, [
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 1, "method": "ping"}])
    assert n == 1
    assert len(out) == 1 and out[0]["id"] == 1


def test_a_malformed_line_does_not_kill_the_server(session):
    buf = io.StringIO()
    n = mcp_server.serve(session, stdin=io.StringIO(
        "{not json\n" + json.dumps({"jsonrpc": "2.0", "id": 1,
                                    "method": "ping"}) + "\n"),
        stdout=buf, log=io.StringIO())
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert n == 2
    assert lines[0]["error"]["code"] == -32700
    assert lines[1]["result"] == {}


def test_the_wire_is_one_object_per_line(session):
    """A newline inside the payload would split one message into two."""
    buf = io.StringIO()
    mcp_server.serve(session, stdin=io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"),
        stdout=buf, log=io.StringIO())
    raw = buf.getvalue()
    assert raw.endswith("\n") and raw.count("\n") == 1
    json.loads(raw)                          # and it is still one document


def test_handling_writes_nothing_to_stdout(session):
    """stdout is the wire; a stray print corrupts the stream."""
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        for req in ({"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    {"jsonrpc": "2.0", "id": 3, "method": "ping"},
                    {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                     "params": {"name": "sql", "arguments": {"query": "SELECT 1"}}}):
            mcp_server.handle(session, req)
    assert capture.getvalue() == ""


def test_a_bad_params_shape_is_a_protocol_error(session):
    _n, out = _serve(session, [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "nope"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "sql", "arguments": [1, 2]}}])
    assert [o["error"]["code"] for o in out] == [-32602, -32602]


def test_the_mcp_layer_and_the_loop_share_one_log(session):
    """Same registry, same ablation, same numbers - that is why it is one Session."""
    _serve(session, [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": "sql",
                                 "arguments": {"query": "SELECT 1 AS one"}}}])
    assert session.summary()["by_tool"] == {"sql": 1}
    assert session.summary()["sql_share"] == 1.0
