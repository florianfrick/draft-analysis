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

from pro.model.agent import draft_state, model_tools, schema, sql_tool  # noqa: E402
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
                  patch_id=None, allowed=None, team=None):
        probs = {0: 0.30, 1: 0.20, 2: 0.10, 3: 0.05, 4: 0.04, 5: 0.03,
                 6: 0.02, 7: 0.01}
        # The real head masks the fearless pool before its top-k; a stub that
        # accepts `unavailable` and ignores it hid exactly that bug once.
        for c in list(picks) + list(bans) + list(unavailable):
            probs.pop(c, None)
        # `allowed` masks the LOGITS, before the top-k, so a role question is a
        # real ranking inside the role rather than a filtered top-k.
        if allowed is not None:
            keep = set(allowed)
            probs = {i: p for i, p in probs.items() if i in keep}
            # The real head softmaxes AFTER the mask, so a role answer is a
            # distribution over the role. Without this the stub's numbers would
            # still sum to 0.3 and a unit assertion would be meaningless.
            total = sum(probs.values()) or 1.0
            probs = {i: p / total for i, p in probs.items()}
        sug = [{"idx": i, "prob": p}
               for i, p in sorted(probs.items(), key=lambda kv: -kv[1])][:k]
        return {"suggestions": sug, "patch_id": patch_id,
                "kind": "ban", "side": "blue" if team is None else team,
                "move_idx": len(bans), "phase": "ban phase 1"}

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


def test_the_model_field_is_a_label_not_the_module():
    """`PolicyPredictor.model` is a property returning the torch module.

    Stringifying it put the whole `PolicyValueModel(...)` repr - 2,000
    characters of architecture - into every tool result. The model reads that
    JSON, so it is not cosmetic: it is a third of the payload spent on a
    repr, repeated on every call.
    """
    class Pred:
        kind = "policy"
        head = StubHead()

        @property
        def model(self):                    # mimics the shipped predictor
            return object()

    out = model_tools.predict_next_move(Pred(), model_tools.ChampIndex(NAMES))
    assert out["model"] == "policy"
    assert "\n" not in out["model"] and len(out["model"]) < 40


def test_predict_next_move_attaches_rank_and_pool_share(champs):
    out = model_tools.predict_next_move(StubPredictor(), champs,
                                        blue_picks=["Kled"], red_picks=[],
                                        patch="16.20")
    assert out["top"][0]["rank"] == 1 and out["top"][0]["p_head"] > 0
    assert out["patch_known"] is True and out["patch_resolved"] == "16.20"
    assert "p_head" not in out["note"]


def test_the_fearless_pool_is_masked_out_of_the_distribution(champs):
    """The pool is not on the board, so only the head's own mask can remove it.

    `unavailable` is fearless: champions picked in an earlier game of the same
    series. A tool that subtracts picks and bans from its pool and nothing else
    ranks a champion that cannot be drafted, at a plausible probability.
    """
    first = model_tools.predict_next_move(StubPredictor(), champs)
    top = first["top"][0]["name"]
    out = model_tools.predict_next_move(StubPredictor(), champs,
                                        unavailable=[top])
    assert top not in [r["name"] for r in out["top"]]
    assert out["n_legal"] == first["n_legal"] - 1


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
    out = s.call("predict_next_move", state="real")
    assert "unknown tool" in out["error"]


# ----------------------------------------------------------------- role tool
# These run against the committed `data/champ_roles.json` with the real
# champion ids, not a stub: the tool *is* a table lookup, so a stub would only
# test the arithmetic and none of the data. The import is inside the fixture so
# the rest of this file keeps running without the name tables.
@pytest.fixture(scope="module")
def real():
    from pro.model.archive.llm.common import champ_names
    return model_tools.ChampIndex(champ_names())


def test_predict_roles_places_a_real_board(real):
    out = model_tools.predict_roles(
        real, ["Jayce", "Trundle", "Karma"], ["Irelia", "Xin Zhao", "Viktor"])
    blue = {r["champion"]: r["role"] for r in out["blue"]}
    assert blue == {"Jayce": "top", "Trundle": "jungle", "Karma": "support"}
    red = {r["champion"]: r["role"] for r in out["red"]}
    assert red == {"Irelia": "top", "Xin Zhao": "jungle", "Viktor": "mid"}
    # This is the question the v4 prompt left to the reader ("Karma (Bot)"):
    # blue still needs a mid and a bot, red still needs a bot and a support.
    assert out["open_roles"] == {"blue": ["mid", "bot"],
                                 "red": ["bot", "support"]}


def test_the_assignment_is_one_to_one_per_side(real):
    """Two flex champions that both list jungle are split, not doubled.

    A per-slot argmax would put both in the jungle and leave the role board
    claiming a team has two junglers and no mid laner.
    """
    out = model_tools.predict_roles(real, ["Vi", "Pantheon"], [])
    roles = [r["role"] for r in out["blue"]]
    assert len(roles) == len(set(roles))
    assert "jungle" in roles


def test_p_lane_separates_a_dedicated_laner_from_a_flex(real):
    out = model_tools.predict_roles(real, ["Jayce", "Trundle"], [])
    by = {r["champion"]: r["p_lane"] for r in out["blue"]}
    assert by["Trundle"] > 0.9          # 8002 of 8017 jungle
    assert 0.1 < by["Jayce"] < 0.9      # genuinely top/mid


def test_a_forced_assignment_reports_a_low_confidence(real):
    """Vi owns jungle, so Pantheon has to go somewhere else.

    The table's best joint fit for the pair is Pantheon support at 4 %, which is
    not a claim that Pantheon is a support - it is the arithmetic being honest.
    The number is the signal, and the note says so; hiding it would make a
    forced placement look like a confident call.
    """
    out = model_tools.predict_roles(real, ["Vi", "Pantheon"], [])
    by = {r["champion"]: r for r in out["blue"]}
    assert by["Vi"]["role"] == "jungle" and by["Vi"]["p_lane"] > 0.9
    assert by["Pantheon"]["role"] != "jungle"
    assert by["Pantheon"]["p_lane"] < 0.1
    assert "best joint fit" in out["note"]


def test_legal_in_lane_counts_only_draftable_champions(real):
    base = model_tools.predict_roles(real)
    banned = model_tools.predict_roles(real, bans=["Jayce"])
    assert banned["legal_in_lane"]["top"] == base["legal_in_lane"]["top"] - 1


def test_the_role_tool_needs_no_checkpoint(real):
    """A run with no head still knows which role a pick fills.

    Gating it on the predictor would make the `--no-head` ablation look like the
    role knowledge had disappeared with the model.
    """
    s = Session(champs=real)
    assert "predict_roles" in s.names()
    assert "predict_next_move" not in s.names()
    out = s.call("predict_roles", state="real")
    assert out["open_roles"]["blue"] == ["top", "jungle", "mid", "bot",
                                         "support"]
    s.store.set_real(draft_state.log_from_board(
        ["Jayce"] + [None] * 9,
        bans=["Rumble", "Ivern", "Renekton", "Tristana", "Ashe", "Zeri"]))
    out = s.call("predict_roles", state="real")
    assert out["blue"][0]["role"] == "top" and "error" not in out


def test_predict_roles_accepts_a_state_id(real):
    from pro.model.agent import draft_state as DS
    # Both sides need three picks: the log stops at the first empty pick slot,
    # so a blue-only board is a one-pick draft, not a three-pick one.
    st = DS.log_from_board(["Jayce", "Trundle", "Karma", None, None]
                           + ["Irelia", "Xin Zhao", "Viktor", None, None],
                           bans=["Rumble", "Ivern", "Renekton", "Tristana",
                                 "Ashe", "Zeri"])
    assert st.move_idx == 12
    s = Session(champs=real, store=DS.Store(real, real=st))
    out = s.call("predict_roles", state="real")
    assert out["blue"][2]["role"] == "support"
    assert out["state"]["move_idx"] == 12


# ------------------------------------------------------- one role's next pick
class StubMasker:
    """A masking checkpoint: it answers a role-conditioned completion."""
    kind = "masker"

    def complete_slot(self, picks, role_ids, slot):
        assert role_ids[slot] == 2          # the lane it was asked about
        probs = [0.0] * 8
        probs[2], probs[3] = 0.6, 0.3
        return [0.0] * 8, probs


# The stub head has eight champions, so a 10-ban board cannot be built from
# its own index: the extra names exist to be banned and never appear in a
# suggestion.
WIDE = NAMES + ["Filler%d" % i for i in range(12)]


@pytest.fixture()
def wide():
    return model_tools.ChampIndex(WIDE)


TEN_BANS = WIDE[9:19]


def _lane_pool(monkeypatch, pool=(1, 2)):
    monkeypatch.setattr(model_tools._lanes, "legal_champions",
                        lambda lane: list(pool))


def test_a_role_question_masks_the_logits_before_the_top_k(wide, monkeypatch):
    """Filtering a top-k instead empties the list 13 % of the time."""
    _lane_pool(monkeypatch, pool=(1, 2))
    out = model_tools.predict_in_lane(
        StubPredictor(), wide, "jungle",
        due={"kind": "pick", "side": "blue"}, bans=TEN_BANS)
    assert [c["name"] for c in out["top"]] == ["Olaf", "Nocturne"]
    assert out["method"] == "head" and out["role"] == "jungle"
    # a share of the role, not of the pool
    assert abs(sum(c["p_in_lane"] for c in out["top"]) - 1.0) < 1e-6
    assert "not of the whole pool" in out["note"]


def test_a_role_question_can_name_the_side_that_is_not_on_the_clock(
        wide, monkeypatch):
    _lane_pool(monkeypatch)
    out = model_tools.predict_in_lane(
        StubPredictor(), wide, "jungle", side="red",
        due={"kind": "pick", "side": "blue"}, bans=TEN_BANS)
    assert out["side"] == "red" and out["hypothetical"] is True
    assert "NOT on the clock" in out["note"]
    same = model_tools.predict_in_lane(
        StubPredictor(), wide, "jungle", side="blue",
        due={"kind": "pick", "side": "blue"}, bans=TEN_BANS)
    assert same["hypothetical"] is False


def test_the_ban_phase_has_no_head_answer_so_it_refuses(champs, monkeypatch):
    """A pick-phase hypothetical on a 2-ban board is not a question."""
    _lane_pool(monkeypatch)
    with pytest.raises(ValueError, match="ban phase"):
        model_tools.predict_in_lane(StubPredictor(), champs, "jungle",
                                    due={"kind": "ban", "side": "blue"},
                                    bans=[0, 1])


def test_a_masking_checkpoint_answers_the_ban_phase(champs, monkeypatch):
    out = model_tools.predict_in_lane(StubPredictor(), champs, "mid",
                                      due={"kind": "ban", "side": "blue"},
                                      bans=[0, 1], masker=StubMasker())
    assert out["method"] == "masker"
    assert [c["name"] for c in out["top"]][:2] == ["Nocturne", "Vi"]
    assert "masking checkpoint" in out["note"]
    # and it masks what is already gone
    gone = model_tools.predict_in_lane(StubPredictor(), champs, "mid",
                                       due={"kind": "ban", "side": "blue"},
                                       bans=[0, 1], unavailable=["Nocturne"],
                                       masker=StubMasker())
    assert "Nocturne" not in [c["name"] for c in gone["top"]]


def test_an_unknown_role_is_refused_not_read_as_top(wide):
    """A role masks logits; an unrecognised one silently read as 0 would
    answer a question about the top lane that nobody asked."""
    with pytest.raises(ValueError, match="unknown role"):
        model_tools.predict_in_lane(StubPredictor(), wide, "jungel",
                                    due={"kind": "pick", "side": "blue"},
                                    bans=TEN_BANS)
    assert model_tools.role_index("JNG") == 1
    assert model_tools.role_index("MIDDLE") == 2
    assert model_tools.role_index(4) == 4
