"""Tests for the draft-state service.

The two that matter most are not about the API surface:

`test_the_schedule_matches_the_corpus` re-derives the 20-move table from the
database, so the constant cannot drift from the data and `first_pick` cannot be
silently re-interpreted.

`test_a_sandbox_never_touches_the_live_board` is the safety property the whole
design rests on: the state the frontend is showing is the one thing an
exploration must never damage.

The rest is the refusal surface, because a stateful tool's failure mode is a
state the model did not intend to create.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.agent import draft_state, model_tools, schema  # noqa: E402
from pro.model.agent.engine import Engine                     # noqa: E402
from pro.model.agent.session import Session                   # noqa: E402
from pro.model.agent.test_agent import NAMES, StubHead        # noqa: E402

D = draft_state


# ------------------------------------------------------------------ fixtures
class FixedHead(StubHead):
    """A head that reports the move it is told to, so agreement can be tested."""

    def __init__(self, kind="pick", side="blue", move_idx=6):
        self.kind, self.side, self.move_idx = kind, side, move_idx

    def next_move(self, picks, bans, **kw):
        out = super().next_move(picks, bans, **kw)
        out.update(kind=self.kind, side=self.side, move_idx=self.move_idx)
        return out


class Predictor:
    model = "fixed-head"

    def __init__(self, head=None):
        self.head = head or FixedHead()


def store(red_first=False, patch="16.20"):
    return D.Store(model_tools.ChampIndex(NAMES), patch=patch,
                   red_first=red_first)


def session(store, head=None):
    return Session(predictor=Predictor(head),
                   champs=model_tools.ChampIndex(NAMES), store=store)


# ------------------------------------------------------------------ schedule
def test_the_schedule_matches_the_corpus():
    """The table is a constant; this is what makes it a fact rather than a guess."""
    if not Path(schema.DB_PATH).exists():
        pytest.skip("corpus not built - run build_db.py")
    con = schema.connect(str(schema.DB_PATH))
    for red_first, table in ((True, D.SCHEDULE_RED_FIRST),
                             (False, D.SCHEDULE_BLUE_FIRST)):
        rows = con.execute(
            "SELECT m.move_idx, m.kind, m.side FROM move m JOIN game g "
            "USING(gameid) WHERE g.first_pick = ? GROUP BY 1,2,3 ORDER BY 1",
            (1 if red_first else 0,)).fetchall()
        assert tuple((k, s) for _i, k, s in rows) == table


def test_first_pick_one_means_red_picks_first():
    """Measured, and the opposite of what the tool spec used to claim."""
    assert D.SCHEDULE_RED_FIRST[6] == (D.PICK, D.RED)
    assert D.SCHEDULE_BLUE_FIRST[6] == (D.PICK, D.BLUE)
    assert D.DraftState(red_first=True).board()["first_pick"] == 1


def test_the_ban_phase_is_split_in_two():
    """Six bans, six picks, four bans, four picks - not ten bans then ten picks."""
    kinds = [k for k, _s in D.SCHEDULE_BLUE_FIRST]
    assert kinds.index(D.PICK) == 6
    assert kinds[12:16] == [D.BAN] * 4
    assert kinds[16:] == [D.PICK] * 4


# --------------------------------------------------------------------- state
def test_apply_follows_the_schedule():
    st = D.DraftState(red_first=False)
    for champ in ("kled", "olaf", "vi", "poppy", "lee sin", "locke"):
        st = st.apply(champ)
    assert [k for k, _s, _c in st.moves] == [D.BAN] * 6
    assert st.to_move() == {"kind": D.PICK, "side": D.BLUE, "move_idx": 6}
    assert st.board()["bans"] == ["kled", "olaf", "vi", "poppy", "lee sin",
                                  "locke"]
    assert st.board()["blue_picks"] == []


def test_a_champion_already_on_the_board_is_refused():
    st = D.DraftState().apply("kled")
    with pytest.raises(D.IllegalMove) as exc:
        st.apply("Kled")
    assert "already on the board" in str(exc.value)
    assert "ban blue" in str(exc.value)          # says where it is


def test_a_move_on_the_wrong_side_is_refused():
    st = D.DraftState(red_first=False)           # move 0 is blue's ban
    with pytest.raises(D.IllegalMove) as exc:
        st.apply("kled", side="red")
    assert "move 0 is blue's, not red's" in str(exc.value)


def test_a_move_of_the_wrong_kind_is_refused():
    st = D.DraftState(red_first=False)
    with pytest.raises(D.IllegalMove) as exc:
        st.apply("kled", kind="pick")
    assert "move 0 is a ban, not a pick" in str(exc.value)


def test_a_complete_draft_refuses_further_moves():
    st = D.DraftState(red_first=False)
    for name in schema.names()[:20]:              # 20 distinct legal champions
        st = st.apply(name)
    assert st.to_move() is None
    with pytest.raises(D.IllegalMove) as exc:
        st.apply("kled")
    assert "complete" in str(exc.value)


def test_undo_drops_the_last_move():
    st = D.DraftState().apply("kled").apply("olaf")
    assert st.move_idx == 2
    assert st.undo().move_idx == 1
    with pytest.raises(D.IllegalMove):
        D.DraftState().undo()


def test_the_store_resolves_names_the_way_the_models_do():
    s = store()
    assert s.resolve("lee sin") == "lee sin"
    assert s.resolve("Lee Sin") == "lee sin"
    with pytest.raises(D.IllegalMove) as exc:
        s.resolve("notachamp")
    assert "unknown champion" in str(exc.value)


# --------------------------------------------------------------------- store
def test_the_live_board_is_read_only():
    s = store()
    with pytest.raises(D.IllegalMove) as exc:
        s.apply(D.REAL_ID, "kled")
    assert "read-only" in str(exc.value)
    with pytest.raises(D.IllegalMove):
        s.undo(D.REAL_ID)


def test_a_sandbox_never_touches_the_live_board():
    """The safety property: exploring must not edit what the frontend shows."""
    s = store()
    sandbox = s.clone(name="alistar-line")
    s.apply(sandbox, "kled")
    assert s.get(sandbox).move_idx == 1
    assert s.get(D.REAL_ID).move_idx == 0
    assert s.get(D.REAL_ID).moves == ()


def test_a_clone_starts_a_fresh_history_at_the_clone_point():
    """Undo in a sandbox takes back *your* moves, not the parent's."""
    s = store()
    sandbox = s.clone()
    s.apply(sandbox, "kled")
    branch = s.clone(sandbox, name="branch")
    assert s.history(branch) == 1           # nothing of its own to undo yet
    with pytest.raises(D.IllegalMove):
        s.undo(branch)
    s.apply(branch, "olaf")
    s.undo(branch)
    assert s.get(branch).move_idx == 1      # back to the clone point
    assert s.get(sandbox).move_idx == 1     # the original kept its move


def test_an_unknown_state_is_refused_with_the_live_ids():
    s = store()
    s.clone(name="a")
    with pytest.raises(D.UnknownState) as exc:
        s.get("nope")
    assert "a" in str(exc.value) and D.REAL_ID in str(exc.value)


def test_listing_reports_depth_and_read_only():
    s = store()
    s.clone(name="a")
    rows = {r["state"]: r for r in s.list()}
    assert rows[D.REAL_ID]["readonly"] is True
    assert rows["a"]["readonly"] is False
    assert rows["a"]["depth"] == 1


# ------------------------------------------------------- board -> move log
def _ui_board(moves, red_first):
    """A chronological move list -> the board shape the frontend sends.

    Slots are two blocks of five in pick order; the flat ban list is the UI's
    `allBansFlat`, i.e. blue's bans then red's. Built here rather than in the
    module so the test asserts against the UI's shape, not against a helper
    that could agree with the code it is checking.
    """
    blue = [c for k, s, c in moves if k == D.PICK and s == D.BLUE]
    red = [c for k, s, c in moves if k == D.PICK and s == D.RED]
    picks = blue + [None] * (5 - len(blue)) + red + [None] * (5 - len(red))
    bans = ([c for k, s, c in moves if k == D.BAN and s == D.BLUE]
            + [c for k, s, c in moves if k == D.BAN and s == D.RED])
    return picks, bans


def test_a_ui_board_rebuilds_every_prefix_of_a_real_game():
    """The conversion, checked against the corpus rather than against itself.

    Every prefix of a real draft is a board the frontend could be showing, so
    every prefix has to come back as the moves the corpus recorded - including
    the ones where one side has picked three times and the other twice.
    """
    if not Path(schema.DB_PATH).exists():
        pytest.skip("corpus not built - run build_db.py")
    con = schema.connect(str(schema.DB_PATH))
    games = con.execute(
        "SELECT gameid, first_pick FROM game g WHERE split = 'val'"
        " AND (SELECT COUNT(*) FROM move WHERE gameid = g.gameid) = 20"
        " ORDER BY first_pick, gameid LIMIT 8").fetchall()
    checked = 0
    for gameid, first_pick in games:
        rows = con.execute(
            "SELECT kind, side, champ_key FROM move WHERE gameid = ?"
            " ORDER BY move_idx", (gameid,)).fetchall()
        assert len(rows) == 20
        for n in range(1, 21):
            picks, bans = _ui_board(rows[:n], red_first=bool(first_pick))
            got = D.log_from_board(picks, bans, red_first=bool(first_pick))
            assert got.moves == tuple(rows[:n]), (gameid, n)
            assert got.move_idx == n
            checked += 1
    assert checked >= 100


def test_the_ban_split_is_dealt_by_the_schedule_not_halved():
    """2 blue + 1 red mid-phase-1 is the case halving gets wrong.

    The flat list is side-grouped (`allBansFlat`: blue's bans, then red's), so a
    list of three is blue's first two and red's first one - and halving it reads
    as 1 blue + 2 red, which puts every ban on the wrong side from then on.
    """
    # Blue-first: ban order is BLUE, RED, BLUE, RED, ...
    st = D.log_from_board([], ["kled", "olaf", "vi"], red_first=False)
    assert st.moves == ((D.BAN, D.BLUE, "kled"), (D.BAN, D.RED, "vi"),
                        (D.BAN, D.BLUE, "olaf"))


def test_the_side_grouped_bans_are_accepted_directly():
    """The frontend sends `bans_blue`/`bans_red` too; that path must agree."""
    flat = D.log_from_board([], ["kled", "olaf", "vi"], red_first=False)
    split = D.log_from_board([], red_first=False,
                             bans_blue=["kled", "olaf"], bans_red=["vi"])
    assert flat.moves == split.moves


def test_a_gap_in_the_pick_slots_truncates_the_log():
    """A slot is not a move index, so a hole means the log stops there.

    `draftscore.replay` skips the hole and keeps scoring the picks after it.
    A state cannot: move 7 cannot exist without move 6.
    """
    # Blue-first, six bans done, then blue picks slots 0 and 2 but not 1.
    bans = ["kled", "olaf", "vi", "poppy", "lee sin", "locke"]
    picks = ["ahri", None, "vi2"] + [None] * 7
    st = D.log_from_board(picks, bans, red_first=False)
    assert st.move_idx == 7                       # six bans + blue's first pick
    assert st.moves[-1] == (D.PICK, D.BLUE, "ahri")
    assert st.board()["blue_picks"] == ["ahri"]


def test_red_first_mirrors_which_slot_a_pick_comes_from():
    """`red_first` swaps the schedule, not the slot layout."""
    st = D.log_from_board([], ["kled"], red_first=True)
    assert st.moves == ((D.BAN, D.RED, "kled"),)
    assert st.board()["first_pick"] == 1


def test_a_duplicated_champion_is_refused():
    with pytest.raises(D.IllegalMove) as exc:
        D.log_from_board([], ["kled", "kled"], red_first=False)
    assert "twice" in str(exc.value)


def test_the_store_converts_indices_and_keeps_the_name_table():
    """The frontend speaks indices; the log speaks canonical keys."""
    s = store()
    names = NAMES
    state, old = s.set_board([None] * 10, [names.index("Kled")], red_first=False)
    assert state.moves == ((D.BAN, D.BLUE, "kled"),)
    assert old.move_idx == 0                      # what it replaced
    assert s.get(D.REAL_ID).moves == state.moves
    with pytest.raises(D.IllegalMove) as exc:
        s.set_board([999] + [None] * 9, [], red_first=False)
    assert "outside the name table" in str(exc.value)


# ------------------------------------------------------------------ fearless
# A fearless series removes every champion picked in an earlier game. Nothing on
# this board produced that list, so a state without it is legal-looking and
# wrong - which is how the agent came to recommend a champion the series had
# already used.
def test_a_fearless_champion_cannot_be_played_into_a_state():
    st = D.DraftState(unavailable=("kled",))
    with pytest.raises(D.IllegalMove) as exc:
        st.apply("kled")
    assert "fearless" in str(exc.value) and "kled" in str(exc.value)
    assert st.apply("olaf").move_idx == 1        # a legal one still works


def test_the_pool_is_rendered_and_counted_out_of_the_legal_pool():
    plain, pooled = D.DraftState(), D.DraftState(unavailable=("kled", "olaf"))
    assert "fearless" not in plain.render()
    assert "kled, olaf" in pooled.render()
    assert pooled.legal_count() == plain.legal_count() - 2
    assert pooled.summary()["fearless"] == ["kled", "olaf"]


def test_a_pool_entry_already_on_the_board_is_not_counted_twice():
    """The UI seeds the pool from the series; overlap cannot change legality."""
    st = D.log_from_board([None] * 10, bans=["kled"],
                          unavailable=["kled", "olaf"])
    assert st.unavailable == ("olaf",)
    assert st.legal_count() == D.N_CHAMPS - 2


def test_a_clone_inherits_the_pool():
    s = store()
    s.set_real(D.DraftState(unavailable=("kled",)))
    assert s.get(s.clone()).unavailable == ("kled",)


def test_the_store_takes_the_pool_in_the_board_index_space():
    s = store()
    idx = model_tools.ChampIndex(NAMES).resolve("Kled")
    state, _old = s.set_board([None] * 10, [], unavailable=[idx])
    assert state.unavailable == ("kled",)


# ---- series context -----------------------------------------------------
# The pool says which champions are gone; the series says which game this is.
# Both ride on the state, so `get_state` and a clone describe the same game the
# first message did. They are prose for the reader, never a model input.
def test_the_series_is_rendered_and_reported_by_the_state():
    lines = ("series: game 3 of 5", "series games: G1 10/10, G2 10/10")
    st = D.DraftState(series=lines)
    assert "game 3 of 5" in st.render()
    assert "G2 10/10" in st.render()
    assert st.summary()["series"] == list(lines)
    assert "series" not in D.DraftState().render()
    assert D.DraftState().summary()["series"] == []


def test_the_board_constructor_carries_the_series_and_every_clone_keeps_it():
    s = store()
    st, _old = s.set_board([None] * 10, [], series=["series: game 2 of 3"])
    assert st.series == ("series: game 2 of 3",)
    assert s.get(s.clone()).series == st.series
    sandbox = s.clone()
    assert s.apply(sandbox, "kled").series == st.series
    assert s.undo(sandbox).series == st.series


def test_the_series_does_not_change_the_board_or_the_legal_pool():
    plain = D.DraftState()
    with_series = D.DraftState(series=("series: game 3 of 5",))
    assert with_series.board() == plain.board()
    assert with_series.legal_count() == plain.legal_count()
    assert with_series.to_move() == plain.to_move()


def test_predicting_on_a_state_never_offers_a_fearless_champion():
    """The reported bug, at the level it happened: a state-based prediction."""
    s = session(store())
    s.store.set_real(D.DraftState(unavailable=("kled",)))
    out = s.call("predict_next_move", state=D.REAL_ID, k=8)
    assert "kled" not in [r["name"].lower() for r in out["top"]]
    assert out["n_legal"] == 7                   # the stub head has 8 champions


def test_the_state_carries_the_role_assignment_it_implies():
    """`apply` recomputes it, so `get_state` never needs a second call."""
    s = session(store())
    sandbox = s.store.clone()
    st = s.store.apply(sandbox, "kled")
    # A ban fills no role: the assignment is about picks, not moves.
    assert st.roles == ()
    # moves 1-5 are the rest of the ban block; move 6 is blue's first pick
    for champ in ("olaf", "nocturne", "vi", "poppy", "lee sin", "locke"):
        st = s.store.apply(sandbox, champ)
    assert len(st.roles) == 1
    out = s.call("get_state", state=sandbox)
    assert out["roles"] == [{"side": "blue", "pick": 1,
                             "champion": "locke", "role": st.roles[0].role,
                             "p_lane": st.roles[0].p_lane}]
    assert len(out["open_roles"]["blue"]) == 4


BANS6 = ["olaf", "nocturne", "vi", "poppy", "lee sin", "locke"]


def test_a_clone_inherits_the_assignment():
    s = store()
    s.set_real(D.log_from_board(["kled"] + [None] * 9, bans=BANS6))
    clone = s.clone()
    assert s.get(clone).roles == s.get(D.REAL_ID).roles


def test_a_store_binds_an_engine_that_arrives_later():
    """The frontend loads a checkpoint while the board is open."""
    s = store()
    s.set_real(D.log_from_board(["kled"] + [None] * 9, bans=BANS6))
    assert s.get(D.REAL_ID).roles          # roles exist with no weights
    before = s.get(D.REAL_ID).roles
    s.attach(Engine(s.champs, predictor=Predictor()))
    assert s.get(D.REAL_ID).roles == before
    assert s.get(D.REAL_ID).engine.predictor is not None


def test_a_state_without_an_engine_says_so_rather_than_guessing():
    st = D.DraftState()
    with pytest.raises(ValueError, match="no models attached"):
        st.predict_next_move()
    assert st.open_roles() == {"blue": [], "red": []}


def test_state_tools_are_offered_only_with_a_store():
    # A session with no store has no state, so it offers no state or model
    # tools. The soloQ matchup data tool is independent of the draft store and
    # is offered whenever its database is on disk, so it is not part of this
    # assertion.
    without = Session()
    board_tools = {"get_state", "clone_state", "apply_move", "undo_move",
                   "list_states", "predict_next_move", "predict_roles"}
    assert not (set(without.names()) & board_tools)
    with_store = session(store())
    assert with_store.names()[:5] == ["get_state", "clone_state", "apply_move",
                                      "undo_move", "list_states"]


def test_the_model_tools_are_methods_on_the_state():
    """One board, one object: there is no second way to name it."""
    s = session(store())
    sandbox = s.store.clone()
    s.store.apply(sandbox, "kled")
    st = s.store.get(sandbox)
    assert st.predict_next_move(k=3)["top"] == \
        s.call("predict_next_move", state=sandbox, k=3)["top"]
    # and the tool answer carries the state it answered about
    assert s.call("predict_next_move", state=sandbox, k=3)["state"]["move_idx"] == 1


def test_a_role_question_goes_through_the_state(monkeypatch):
    """`predict_next_move(role=..., side=...)` is the state's own question."""
    monkeypatch.setattr(model_tools._lanes, "legal_champions", lambda lane: [1, 2])
    s = session(store())
    st = s.store.get(D.REAL_ID)
    # move 0 is a ban, so the head cannot answer a pick-phase hypothetical and
    # there is no masker to fall back on: the state refuses instead of guessing.
    with pytest.raises(ValueError, match="ban phase"):
        st.predict_next_move(role="jungle", side="red", k=2)


def test_a_hypothetical_line_can_be_explored_and_undone():
    """The workflow the prompt describes, end to end."""
    s = session(store())
    live = s.call("get_state")
    assert live["state"] == D.REAL_ID and live["readonly"] is True

    sandbox = s.call("clone_state", name="alistar-line")["state"]
    after = s.call("apply_move", state=sandbox, champion="Kled")
    assert after["move_idx"] == 1 and after["kind"] == "ban"
    assert "Kled" in after["rendered"] or "kled" in after["rendered"]

    pred = s.call("predict_next_move", state=sandbox, k=3)
    assert pred["state"]["move_idx"] == 1

    back = s.call("undo_move", state=sandbox)
    assert back["move_idx"] == 0
    assert s.store.get(D.REAL_ID).move_idx == 0


def test_a_refused_move_comes_back_as_a_readable_error():
    """`Session.call` never raises; the model has to be able to read and retry."""
    s = session(store())
    out = s.call("apply_move", state=D.REAL_ID, champion="Kled")
    assert "error" in out
    assert "read-only" in out["error"]
    out = s.call("apply_move", state="nope", champion="Kled")
    assert "unknown state" in out["error"]


def test_a_state_tool_without_a_store_is_refused():
    out = Session().call("get_state")
    assert "unknown tool" in out["error"]
