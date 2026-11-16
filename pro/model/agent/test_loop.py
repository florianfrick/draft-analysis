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
import itertools
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.agent import (draft_state, handrun, llm_client, loop,  # noqa: E402
                             mcp_server, model_tools, schema, soloq_tool)
from pro.model.agent.session import Session                        # noqa: E402
from pro.model.agent.test_agent import NAMES, SPLIT, _rows         # noqa: E402

BOARD = {"blue_picks": ["Aatrox", "Vi", "Ahri"], "red_picks": ["Poppy", "Lee Sin"],
         "bans": ["Kled", "Olaf"], "first_pick": 1, "patch": "16.20"}


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("loop") / "query.sqlite"
    schema.build(_rows(), path, known_patches={"16.20"}, split=SPLIT)
    return schema.connect(path)


@pytest.fixture()
def session(con):
    """A session with a live board: no tool takes a board any more, so a run
    without a state has nothing to answer about."""
    store = draft_state.Store(model_tools.ChampIndex(NAMES))
    store.set_real(draft_state.log_from_board(
        BOARD["blue_picks"], BOARD["bans"], red_first=True, patch=BOARD["patch"]))
    return Session(con=con, store=store, max_calls=6)


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
def test_no_tool_takes_a_board(session):
    """The transcription failure is removed by construction, not flagged.

    An earlier version accepted a board as well as a state, filled an omitted
    field from the turn's board and flagged a contradicted one. Every one of
    those branches existed to detect a model re-typing nine champion names.
    """
    # The transcription failure was nine champion names (plus first_pick and
    # the ban/fearless sets). `patch` is a legitimate query filter on some tools
    # (it is a scope, not a board to transcribe), so it is not in this set.
    board_fields = {"blue_picks", "red_picks", "bans", "first_pick",
                    "unavailable"}
    for spec in session.available():
        props = set((spec["parameters"].get("properties") or {}))
        assert not (props & board_fields), spec["name"]
    # `state` is required by the two model tools, not by the state verbs.
    assert model_tools.PREDICT_SPEC["parameters"]["required"] == ["state"]
    assert model_tools.ROLES_SPEC["parameters"]["required"] == ["state"]


def test_a_run_without_a_state_says_so(con):
    """A session with no store has no board, and guessing one is the bug."""
    out = loop.run(Session(con=con), "Kled or Olaf?")
    assert out["stop"] == "no state"
    assert "Store" in out["error"]


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
                   )
    assert out["answer"] == "Pick Xayah."
    assert out["stop"] == "answered"
    assert [c["name"] for c in out["calls"]] == ["sql"]


def test_the_loop_stops_at_the_step_budget_and_still_answers(session):
    """A graph that runs out of steps must return, not raise.

    And it must not return nothing. The first version ended on the tool call the
    budget refused, so `answer` was empty and the panel rendered an empty bubble
    - a bug from the reader's side. The last call is now made without tools, so
    the model answers from the results it already has.
    """
    seen = []

    def client(messages, tools=None):
        seen.append((tools, messages))
        if tools:
            return reply("", [("sql", {"query": "SELECT 1 AS one"})])
        return reply("One, from the query I already ran.")

    out = loop.run(session, "x", client=client, max_steps=2)
    assert out["stop"] == "step budget"
    assert out["answer"] == "One, from the query I already ran."
    assert len(out["calls"]) == 2          # two tool rounds, then the router
    assert out["summary"]["by_tool"] == {"sql": 2}
    assert seen[-1][0] is None             # the last call offered no tools
    assert "every tool call" in seen[-1][1][-1]["content"]


def test_a_model_that_ignores_the_budget_note_gets_no_answer(session):
    """The residual case: it keeps asking even when nothing is offered.

    Nothing in the loop can invent an answer, so this pins the limit rather than
    a behaviour - and it is why the budget path drops the tools as well as
    sending the note, instead of relying on the model to notice.
    """
    out = loop.run(session, "x", client=client_that_records(), max_steps=2)
    assert out["stop"] == "step budget"
    assert len(out["calls"]) == 2
    assert out["answer"] is None


def test_a_tool_error_does_not_end_the_loop(session):
    out = loop.run(session, "x",
                   client=client_of(reply("", [("sql", {"query": "DELETE FROM move"})]),
                                    reply("I could not query that.")),
                   )
    assert out["stop"] == "answered"
    assert out["summary"]["errors"] == 1
    assert out["answer"] == "I could not query that."


def test_an_llm_error_is_reported_in_the_stop_reason(session):
    def boom(messages, tools=None):
        raise llm_client.LLMError("HTTP 401 from x: bad key")

    out = loop.run(session, "x", client=boom)
    assert out["stop"].startswith("llm error")
    assert "401" in out["stop"]


def test_a_crashing_client_does_not_lose_the_log(session):
    def boom(messages, tools=None):
        raise RuntimeError("kaboom")

    out = loop.run(session, "x", client=boom)
    assert out["stop"] == "loop error"
    assert "kaboom" in out["error"]
    assert out["summary"]["calls"] == 0
    with pytest.raises(RuntimeError):
        loop.run(session, "x", client=boom, raise_errors=True)


# ------------------------------------------------------------------ ablation
def test_an_ablated_tool_is_not_advertised(con):
    """Hiding the spec is the shipped configuration: the model never sees it."""
    # Sentinels, not a real head: `available()` only asks whether a predictor
    # and a name index were loaded, and this test is about the listing.
    session = Session(con=con, predictor=object(), champs=object(),
                      ablate={"sql"})
    client = client_of(reply("no tools needed"))
    out = loop.run(session, "x", client=client)
    # The base set, plus `soloq_matchup` when its store is on disk - the
    # assertion is store-independent so it passes whether or not the soloQ
    # analytical store was built in this environment. `sql` is ablated.
    base = {"get_state", "clone_state", "apply_move", "undo_move",
            "list_states", "predict_next_move", "predict_roles"}
    if soloq_tool.available():
        base.add("soloq_matchup")
    assert set(out["tools_offered"]) == base
    assert "sql" not in out["tools_offered"]
    assert out["calls"] == []
    assert out["answer"] == "no tools needed"


def test_an_ablated_tool_is_also_refused_at_call_time(session):
    """A model that invents a tool name gets a refusal it can read, not a crash."""
    session.ablate = {"sql"}
    out = loop.run(session, "x",
                   client=client_of(reply("", [("sql", {"query": "SELECT 1"})]),
                                    reply("no sql")),
                   )
    assert out["summary"]["ablated"] == 1
    assert out["summary"]["errors"] == 1
    assert out["stop"] == "answered"
    tool_msg = [m for m in out["messages"] if m["role"] == "tool"][0]
    assert "not available" in tool_msg["content"]


def test_every_call_goes_through_the_session(session):
    """The log is the measurement; a direct call would bypass it."""
    loop.run(session, "x", client=client_of(reply("", [("sql", {"query":
                                                                "SELECT 1 AS one"})]),
                                            reply("ok")))
    assert len(session.log) == 1
    assert session.log[0]["name"] == "sql"
    assert session.log[0]["ok"] is True
    assert session.log[0]["ms"] >= 0


# -------------------------------------------------------------- the prompt
def test_the_prompt_reads_the_board_before_it_ranks_anything():
    """The order in the prompt is the design decision; pin it."""
    i_state = loop.SYSTEM.index("get_state")
    i_roles = loop.SYSTEM.index("predict_roles")
    i_pred = loop.SYSTEM.index("predict_next_move")
    i_sql = loop.SYSTEM.index("`sql`")
    assert i_state < i_roles < i_pred < i_sql


def test_the_prompt_forbids_retyping_a_board():
    """The transcription failure the state service exists to remove."""
    text = loop.SYSTEM.lower()
    assert "never retype a board" in text
    assert "read-only" in text


def test_the_prompt_warns_that_a_shortlist_can_omit_the_heads_top_pick():
    """Measured in the first hand-run: 2 of 3 hand-made shortlists missed it."""
    text = loop.SYSTEM.lower()
    assert "shortlist you wrote" in text
    assert "omit the head's top pick" in text


def test_the_prompt_says_the_probabilities_are_not_comparable():
    for key in ("p_head", "p_lane", "p_in_lane"):
        assert key in loop.SYSTEM
    assert "different claims" in loop.SYSTEM


def test_the_prompt_sends_a_named_comparison_back_to_the_ranking():
    """There is no shortlist tool; the model has to widen k instead."""
    text = loop.SYSTEM.lower()
    assert "your job, not a tool's" in text
    assert "wider `k`" in text


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
    # BOARD has first_pick=1, which is RED first - the corpus meaning, and the
    # opposite of what the tool spec claimed before it was measured.
    assert "first pick: red" in text
    assert "16.20" in text
    assert "not given" in loop.render_board({})


def test_scoring_separates_fidelity_from_accuracy(tmp_path, monkeypatch, capsys):
    """Two numbers, two owners: the interface carries the head, the head is judged."""
    monkeypatch.setattr(handrun, "STATE", tmp_path / "s.json")
    board = {"blue_picks": [], "red_picks": [], "bans": ["olaf"],
             "first_pick": 0, "patch": "16.20"}
    (tmp_path / "s.json").write_text(json.dumps({
        "states": {
            "A": {"board": board, "truth": "vi", "bucket": "2-5",
                  "head_rank": 3},
            "B": {"board": board, "truth": "kled", "bucket": "1", "head_rank": 1},
        }, "calls": [], "answers": {}}), encoding="utf-8")
    from pro.model.agent.test_agent import StubPredictor
    monkeypatch.setattr(handrun, "_session", lambda: types.SimpleNamespace(
        predictor=StubPredictor(), champs=model_tools.ChampIndex(NAMES)))
    handrun.cmd_score(types.SimpleNamespace(
        answers='{"A": "Kled", "B": "Kled"}'))
    out = capsys.readouterr().out
    # Both answers are the head's #1 (Kled); only one is the truth.
    assert "fidelity" in out and "2/2 = 100.0%" in out
    assert "accuracy" in out and "1/2 = 50.0%" in out


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


def test_tools_list_hides_an_ablated_tool(session):
    session.ablate = {"sql"}
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


# ------------------------------------------------------------- the hand-run
def test_the_board_is_everything_strictly_before_the_move(con):
    """One move too many and the answer is in the board; one too few and it is not."""
    board = handrun._board(con, "g0", 12)
    assert len(board["bans"]) == 10              # five each, in move order
    assert board["blue_picks"] == [schema.key(schema.names()[0])]
    assert board["red_picks"] == [schema.key(schema.names()[5])]
    assert board["first_pick"] == 1
    assert board["patch"] == "16.20"


def test_the_sample_hides_the_bucket_while_answering(con, tmp_path, monkeypatch,
                                                     capsys):
    """The bucket says which boards to distrust - a real agent never has it."""
    monkeypatch.setattr(handrun, "STATE", tmp_path / "state.json")
    # cycle, not iter: `sample` asks for two per bucket by default and would
    # exhaust a three-item iterator before it filled them.
    ranks = itertools.cycle([1, 3, 8])
    monkeypatch.setattr(handrun, "_head_rank",
                        lambda *a, **k: (next(ranks), None))
    monkeypatch.setattr(handrun, "_session",
                        lambda: types.SimpleNamespace(
                            con=con, predictor=object(), champs=object()))
    handrun.cmd_sample(types.SimpleNamespace(n=3, seed=0))
    out = capsys.readouterr().out
    assert "bucket" not in out.lower()
    assert "=== A ===" in out and "to move:" in out
    blob = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    buckets = [s["bucket"] for s in blob["states"].values()]
    assert len(buckets) == 3
    assert set(buckets) <= set(handrun.BUCKETS)
    assert all(s["truth"] for s in blob["states"].values())


def test_the_sample_is_reproducible_from_its_seed(con, tmp_path, monkeypatch):
    """A run nobody can repeat is an anecdote, not a measurement."""
    monkeypatch.setattr(handrun, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(handrun, "_head_rank", lambda *a, **k: (1, None))
    first, _ = handrun.sample(con, object(), object(), n=2, seed=7)
    second, _ = handrun.sample(con, object(), object(), n=2, seed=7)
    assert [s["gameid"] for s in first.values()] == \
           [s["gameid"] for s in second.values()]
