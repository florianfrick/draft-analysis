"""Tests for the agent tool layer.

Everything here runs against a synthetic corpus and a stub head, because the
questions are about *contracts*: what a tool refuses, what it reports, and what
the session records. The one thing that needs the real data - whether the schema
answers a real question correctly - is `build_db.py --query`.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.agent import model_tools, schema, sql_tool  # noqa: E402
from pro.model.agent.session import Session  # noqa: E402

PAD = 160


def _moves(blue, red, first_pick=0):
    """20 moves: 5 bans each, then 5 picks each, in the canonical order."""
    out = []
    for i in range(5):
        out.append((0, 0, 100 + i, -100))          # blue bans
        out.append((0, 1, 110 + i, -100))          # red bans
    for i in range(5):
        out.append((1, 0, blue[i], i))
        out.append((1, 1, red[i], i))
    return out


def _rows():
    """Two patches, six games, deterministic ids so assertions can be exact."""
    rows = []
    for g in range(4):
        rows.append({
            "gameid": "g%d" % g, "patch": "16.20",
            "date_ordinal": 739000 + g, "league": "LCK",
            "blue_team": "T1", "red_team": "GEN", "blue_win": g % 2,
            "first_pick": 1, "fearless": False,
            "moves": _moves([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
        })
    for g in range(2):
        rows.append({
            "gameid": "h%d" % g, "patch": "16.22",
            "date_ordinal": 739100 + g, "league": "LPL",
            "blue_team": "JDG", "red_team": "BLG", "blue_win": 1,
            "first_pick": 0, "fearless": True,
            "moves": _moves([10, 11, 12, 13, 14], [15, 16, 17, 18, 19]),
        })
    return rows


SPLIT = {"train": [0, 1, 2], "val": [3, 4], "test": [5]}


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("agent") / "query.sqlite"
    stats = schema.build(_rows(), path, known_patches={"16.20"}, split=SPLIT)
    return stats, path


@pytest.fixture(scope="module")
def con(db):
    return schema.connect(db[1])


# --------------------------------------------------------------------- build
def test_the_build_encodes_kind_and_side_as_words(db):
    """0/1 columns invite `WHERE kind = 1`, which silently matches nothing."""
    _stats, path = db
    con = sqlite3.connect(str(path))
    kinds = {r[0] for r in con.execute("SELECT DISTINCT kind FROM move")}
    sides = {r[0] for r in con.execute("SELECT DISTINCT side FROM move")}
    assert kinds == {"ban", "pick"} and sides == {"blue", "red"}
    con.close()


def test_a_ban_has_no_role_and_a_pick_has_one(con):
    n_null = con.execute(
        "SELECT COUNT(*) FROM move WHERE kind='ban' AND role IS NULL").fetchone()[0]
    n_pick = con.execute(
        "SELECT COUNT(*) FROM move WHERE kind='pick' AND role IS NULL").fetchone()[0]
    assert n_null == 6 * 10 and n_pick == 0


def test_patch_rank_is_dense_and_not_the_patch_id(con):
    """16.19 and 16.21 are absent from the real corpus, so patch arithmetic is
    wrong by construction. A dense rank is the only safe "last N patches"."""
    got = con.execute(
        "SELECT patch, patch_rank FROM patch ORDER BY patch_rank").fetchall()
    assert [r[1] for r in got] == [0, 1]
    assert [r[0] for r in got] == ["16.20", "16.22"]


def test_is_scored_comes_from_the_checkpoint_not_from_a_guess(con):
    got = dict(con.execute("SELECT patch, is_scored FROM patch").fetchall())
    assert got == {"16.20": 1, "16.22": 0}


def test_every_row_carries_its_split(con):
    """An agent allowed to query val contaminates the evaluation that decides
    whether the agent works."""
    for table in ("game", "move"):
        nulls = con.execute("SELECT COUNT(*) FROM %s WHERE split IS NULL"
                            % table).fetchone()[0]
        assert nulls == 0, table
    assert con.execute("SELECT COUNT(*) FROM game WHERE split='train'"
                       ).fetchone()[0] == 3


# ----------------------------------------------------------------------- sql
def test_a_write_is_refused(con):
    out = sql_tool.run(con, "DELETE FROM move")
    assert "error" in out and "SELECT" in out["error"]


def test_two_statements_are_refused(con):
    out = sql_tool.run(con, "SELECT 1; SELECT 2")
    assert "error" in out and "one statement" in out["error"]


def test_a_pragma_is_refused(con):
    assert "error" in sql_tool.run(con, "PRAGMA table_info(move)")


def test_the_connection_itself_is_read_only(db):
    con = sqlite3.connect("file:%s?mode=ro" % Path(db[1]).as_posix(), uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM move")


def test_the_row_cap_is_enforced_and_reported(con):
    out = sql_tool.run(con, "SELECT * FROM move", max_rows=5)
    assert out["n_rows"] == 5 and out["truncated"] is True
    assert len(out["rows"]) == 5


def test_the_query_is_returned_with_the_rows(con):
    """Without the SQL in the transcript a wrong answer cannot be debugged."""
    out = sql_tool.run(con, "SELECT COUNT(*) AS n FROM move")
    assert out["query"].lower().startswith("select")
    assert out["rows"][0]["n"] == 120


def test_a_syntax_error_is_a_result_not_an_exception(con):
    out = sql_tool.run(con, "SELECT FROM")
    assert "error" in out and out["query"]


def test_a_query_the_watchdog_kills_is_not_reported_as_no_rows(con):
    """The failure this tool exists to avoid: a confident empty answer.

    `rows: 0` is a claim about the corpus, so an aborted query must not look
    like one. It cost a real answer: on the bind mount the deadline fired on
    queries the corpus answers in milliseconds, and the model reported that
    the evidence did not exist.
    """
    out = sql_tool.run(con, "SELECT COUNT(*) FROM move a, move b, move c",
                       timeout=0.001)
    assert out.get("timed_out") is True
    assert "aborted" in out["error"]
    assert "rows" not in out


# ---------------------------------------------------------------- local copy
def test_the_cache_opens_a_copy_of_the_mount(db, tmp_path, monkeypatch):
    """52x, measured: see `schema._local_copy` for the six queries."""
    monkeypatch.setenv("LOL_DB_CACHE", str(tmp_path))
    con = schema.connect(db[1])
    opened = Path(con.execute("PRAGMA database_list").fetchone()[2])
    assert opened == tmp_path / Path(db[1]).name
    assert opened.exists() and opened != Path(db[1])


def test_a_rebuilt_corpus_is_copied_again(db, tmp_path, monkeypatch):
    """Size+mtime is the staleness test, so a rebuild cannot be missed."""
    monkeypatch.setenv("LOL_DB_CACHE", str(tmp_path))
    src = Path(db[1])
    schema.connect(src).close()
    copy = tmp_path / src.name
    before = copy.stat().st_mtime
    st = src.stat()
    os.utime(src, (st.st_atime, st.st_mtime + 10))
    schema.connect(src).close()
    assert copy.stat().st_mtime > before


def test_without_the_variable_the_database_is_opened_in_place(db, monkeypatch):
    monkeypatch.delenv("LOL_DB_CACHE", raising=False)
    con = schema.connect(db[1])
    assert Path(con.execute("PRAGMA database_list").fetchone()[2]) == Path(db[1])


def test_the_hint_teaches_the_mistake_it_caught(con):
    assert "kind is text" in (sql_tool.run(con, "SELECT 1 FROM move WHERE kind=1")
                              ["hint"] or "")
    assert "split" in (sql_tool.run(con, "SELECT COUNT(*) FROM move")["hint"] or "")
    assert "patch_rank" in (sql_tool.run(
        con, "SELECT 1 FROM move WHERE patch > '16.20'")["hint"] or "")


# ---------------------------------------------------------------- champions
def test_names_resolve_through_apostrophes_and_spacing():
    champs = model_tools.ChampIndex(["Kled", "Kai'Sa", "Dr. Mundo", "Vi"])
    assert champs.resolve("Kai'Sa") == 1
    assert champs.resolve("kaisa") == 1
    assert champs.resolve("kai sa") == 1
    assert champs.resolve("mundo") == 2       # unique prefix, not 'Dr'
    assert champs.resolve(0) == 0
    with pytest.raises(ValueError):
        champs.resolve("Notachampion")


# -------------------------------------------------------------- model tools
class StubHead:
    """Mirrors the real `next_move`: state at the TOP level, 1-based patch ids.

    Both of those were wrong in the first version of the tool and neither was
    caught by a stub that nested the state and took patch labels as ids - which
    is the argument for pinning the shape here rather than only on the real
    checkpoint.
    """
    num_champs = 8
    n_patches = 3
    patch_vocab = ["16.18", "16.20"]

    def next_move(self, picks, bans, first_pick=0, k=8, unavailable=(),
                  patch_id=None):
        probs = {0: 0.30, 1: 0.20, 2: 0.10, 3: 0.05, 4: 0.04, 5: 0.03,
                 6: 0.02, 7: 0.01}
        for c in list(picks) + list(bans):
            probs.pop(c, None)
        sug = [{"idx": i, "prob": p}
               for i, p in sorted(probs.items(), key=lambda kv: -kv[1])][:k]
        return {"suggestions": sug, "patch_id": patch_id,
                "kind": "ban", "side": "blue", "move_idx": len(bans),
                "phase": "ban phase 1"}

    def patch_id_for(self, pid):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return 0
        return pid if 0 <= pid < self.n_patches else 0

    def patch_key_for(self, pid):
        pid = self.patch_id_for(pid)
        if pid <= 0 or pid - 1 >= len(self.patch_vocab):
            return None
        return self.patch_vocab[pid - 1]


class StubPredictor:
    model = "stub-head"
    head = StubHead()


NAMES = ["Kled", "Olaf", "Nocturne", "Vi", "Poppy", "Lee Sin", "Locke",
         "Jarvan IV", "Aatrox"]


@pytest.fixture()
def champs():
    return model_tools.ChampIndex(NAMES)


def test_predict_next_move_attaches_rank_and_pool_share(champs):
    out = model_tools.predict_next_move(StubPredictor(), champs,
                                        blue_picks=["Kled"], red_picks=[],
                                        patch="16.20")
    assert out["top"][0]["rank"] == 1 and out["top"][0]["p_head"] > 0
    assert out["patch_known"] is True and out["patch_resolved"] == "16.20"
    assert "legal champions" in out["note"]


def test_the_sides_are_placed_in_their_own_slots(champs):
    """Blue 0-4, red 5-9. A single sparse list cannot say which side the third
    name belongs to, which is why the tool takes two."""
    assert model_tools._slots([1, 2], [3]) == [1, 2, None, None, None, 3,
                                               None, None, None, None]
    out = model_tools.predict_next_move(StubPredictor(), champs,
                                        blue_picks=["Kled", "Vi"],
                                        red_picks=["Olaf"])
    # Kled and Olaf are both gone from the stub's distribution
    names = [c["name"] for c in out["top"]]
    assert "Kled" not in names and "Olaf" not in names


def test_a_patch_label_is_looked_up_in_the_vocabulary_not_parsed(champs):
    """Patch ids are 1-based rows into `patch_vocab`, and `patch_key_for` does
    `vocab[pid - 1]`. Parsing '16.20' as 1620 lands out of range and silently
    becomes UNK."""
    out = model_tools.predict_next_move(StubPredictor(), champs,
                                        blue_picks=[], red_picks=[],
                                        patch="16.20")
    assert out["patch_known"] is True and out["patch_resolved"] == "16.20"
    # the first vocabulary row is id 1, not id 0 (0 is UNK)
    assert model_tools._patch_state(StubPredictor(), "16.18")[0] == 1
    assert model_tools._patch_state(StubPredictor(), "16.20")[0] == 2


def test_the_move_state_is_read_from_the_top_level(champs):
    out = model_tools.predict_next_move(StubPredictor(), champs,
                                        blue_picks=["Kled"], bans=["Olaf"])
    assert out["kind"] == "ban" and out["side"] == "blue"
    assert out["move_idx"] == 1


def test_a_numeric_side_is_reported_in_words():
    """The predictor returns side 0/1; a bare 0 in a transcript reads as
    'no side'."""
    assert model_tools._state({"side": 0})["side"] == "blue"
    assert model_tools._state({"side": 1})["side"] == "red"
    assert model_tools._state({"side": "red"})["side"] == "red"


def test_an_unknown_patch_is_reported_not_hidden(champs):
    """A patch the checkpoint never saw becomes UNK and the head answers
    patch-agnostically. Silence here is how an agent attributes a
    patch-agnostic answer to a specific patch."""
    out = model_tools.predict_next_move(StubPredictor(), champs,
                                        blue_picks=[], red_picks=[],
                                        patch="16.22")
    assert out["patch_known"] is False and out["patch_resolved"] is None
    assert "patch-agnostic" in out["note"]


def test_an_illegal_board_is_refused(champs):
    with pytest.raises(ValueError):
        model_tools.predict_next_move(StubPredictor(), champs,
                                      blue_picks=["Kled", "Kled"])
    with pytest.raises(ValueError):
        model_tools.predict_next_move(StubPredictor(), champs,
                                      blue_picks=["Kled"], bans=["Kled"])
    with pytest.raises(ValueError):
        model_tools.predict_next_move(StubPredictor(), champs,
                                      blue_picks=["Kled", "Vi", "Olaf",
                                                  "Poppy", "Nocturne", "Locke"])


def test_rank_candidates_renormalises_over_the_candidates(champs):
    out = model_tools.rank_candidates(StubPredictor(), champs,
                                      candidates=["Kled", "Vi", "Poppy"])
    shares = [c["share_of_candidates"] for c in out["candidates"]]
    assert abs(sum(shares) - 1.0) < 1e-3
    assert out["candidates"][0]["name"] == "Kled"
    assert out["candidates"][0]["rank_in_candidates"] == 1
    # the pool rank and the candidate share are different claims
    assert out["candidates"][0]["rank_in_pool"] == 1
    assert out["candidates"][-1]["rank_in_pool"] > 1


def test_rank_candidates_refuses_a_champion_already_on_the_board(champs):
    with pytest.raises(ValueError) as exc:
        model_tools.rank_candidates(StubPredictor(), champs,
                                    blue_picks=["Kled"],
                                    candidates=["Kled", "Vi"])
    assert "already on the board" in str(exc.value)


# ------------------------------------------------------------------- session
def test_an_ablated_tool_is_hidden_and_refused(db):
    s = Session(con=schema.connect(db[1]), ablate={"sql"})
    assert "sql" not in s.names()
    out = s.call("sql", query="SELECT 1")
    assert out["ablated"] is True and "not available" in out["error"]
    assert s.summary()["ablated"] == 1


def test_the_log_records_every_call(db):
    s = Session(con=schema.connect(db[1]))
    s.call("sql", query="SELECT COUNT(*) AS n FROM move")
    s.call("sql", query="DELETE FROM move")
    assert len(s.log) == 2
    assert s.log[0]["ok"] is True and s.log[0]["n_rows"] == 1
    assert s.log[1]["ok"] is False and "SELECT" in s.log[1]["error"]
    assert s.log[0]["args"]["query"].startswith("SELECT")


def test_sql_share_is_the_number_the_design_is_judged_on(db):
    """If more than half the calls are SQL, the high-level tier is
    under-specified - that is the reading, not 'the SQL tool is too good'."""
    s = Session(con=schema.connect(db[1]))
    s.call("sql", query="SELECT 1 AS n")
    s.call("sql", query="SELECT 2 AS n")
    s.call("sql", query="SELECT 3 AS n")
    mix = s.mix()
    assert mix["calls"] == 3 and mix["sql_share"] == 1.0


def test_the_budget_stops_the_loop_without_raising(db):
    s = Session(con=schema.connect(db[1]), max_calls=2)
    for _ in range(3):
        out = s.call("sql", query="SELECT 1 AS n")
    assert out.get("budget") is True
    assert s.summary()["budget_refusals"] == 1


def test_an_unknown_tool_is_an_error_not_a_crash(db):
    s = Session(con=schema.connect(db[1]))
    out = s.call("query_matchup_stats", champ_a="Kled", champ_b="Olaf")
    assert "unknown tool" in out["error"] and "sql" in out["error"]


def test_a_model_tool_without_a_model_says_so(db):
    s = Session(con=schema.connect(db[1]))
    assert "predict_next_move" not in s.names()
    out = s.call("predict_next_move", picks=[], bans=[])
    assert "unknown tool" in out["error"]
