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


# ------------------------------------------------------------------- session
def test_state_tools_are_offered_only_with_a_store():
    without = Session(predictor=Predictor(),
                      champs=model_tools.ChampIndex(NAMES))
    assert "get_state" not in without.names()
    with_store = session(store())
    assert with_store.names()[:5] == ["get_state", "clone_state", "apply_move",
                                      "undo_move", "list_states"]


def test_the_state_form_and_the_board_form_give_the_same_answer():
    """A state id resolves to the board a caller would have typed; one path."""
    s = session(store())
    sandbox = s.store.clone()
    s.store.apply(sandbox, "kled")
    board = s.store.get(sandbox).board()
    by_state = s.call("predict_next_move", state=sandbox, k=3)
    by_board = s.call("predict_next_move", blue_picks=board["blue_picks"],
                      red_picks=board["red_picks"], bans=board["bans"],
                      first_pick=board["first_pick"], patch=board["patch"], k=3)
    assert by_state["top"] == by_board["top"]
    assert by_state["state"]["move_idx"] == 1
    assert "state" not in by_board


def test_the_result_says_whether_the_head_agrees_with_the_schedule():
    """A board form call can be a state that cannot occur; this is the tell."""
    # Blue-first: move 0 is blue's ban, so move 1 is red's.
    s = session(store(), head=FixedHead(kind="ban", side="red", move_idx=1))
    sandbox = s.store.clone()
    s.store.apply(sandbox, "kled")                  # move 0 done -> move 1
    agree = s.call("predict_next_move", state=sandbox, k=3)
    assert agree["move_matches_state"] is True
    s2 = session(store(), head=FixedHead(kind="pick", side="red", move_idx=6))
    sandbox2 = s2.store.clone()
    s2.store.apply(sandbox2, "kled")
    assert s2.call("predict_next_move", state=sandbox2, k=3)[
        "move_matches_state"] is False


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
    without = Session(predictor=Predictor(),
                      champs=model_tools.ChampIndex(NAMES))
    out = without.call("get_state")
    assert "unknown tool" in out["error"]
