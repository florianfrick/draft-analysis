"""Draft state as an immutable move log, and a store of them.

The board used to be an argument: every model tool took nine champion names and
hoped the caller retyped them correctly. That is a silent failure - a legal but
wrong board returns a confident distribution - and the `transcription.mismatched`
flag could only *detect* it. Here the board is held instead, and tools take a
state id.

Four decisions, each of which is the reason this is a move log rather than a set
of board fields:

**The schedule is a constant in the corpus.** All 46,915 games are exactly 20
moves, and the (kind, side) at each index depends only on `first_pick` - two
mirrored sequences, verified in `test_state.py::test_the_schedule_matches_the_corpus`.
So `to_move()` is an index into a table, never an inference. The old tools had to
guess whose turn it was from pick and ban counts, which is why a board with two
bans and five picks got a plausible-looking answer about a state that cannot
occur.

**`first_pick=1` means red picks first.** Measured, and the opposite of what
`model_tools` used to tell the model. `red_first` is the field name here so the
inversion cannot be repeated.

**Undo is dropping the last move, and clone is sharing the tuple.** Both are free
because a state is immutable; a mutable board would need a diff log to support
either.

**The real state is read-only to the model.** A stateless tool cannot be
corrupted; a stateful one can, and the state the frontend is showing is the one
thing an exploration must never damage. Sandboxes are explicit clones, and
`Store.put` refuses `REAL_ID` by construction rather than by convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

BAN, PICK = "ban", "pick"
BLUE, RED = "blue", "red"
REAL_ID = "real"
N_CHAMPS = 173

# Verified against the corpus; `test_the_schedule_matches_the_corpus` re-checks
# it against the database so this table cannot drift from the data.
SCHEDULE_RED_FIRST = (
    (BAN, RED), (BAN, BLUE), (BAN, RED), (BAN, BLUE), (BAN, RED), (BAN, BLUE),
    (PICK, RED), (PICK, BLUE), (PICK, BLUE), (PICK, RED), (PICK, RED), (PICK, BLUE),
    (BAN, BLUE), (BAN, RED), (BAN, BLUE), (BAN, RED),
    (PICK, BLUE), (PICK, RED), (PICK, RED), (PICK, BLUE),
)
SCHEDULE_BLUE_FIRST = (
    (BAN, BLUE), (BAN, RED), (BAN, BLUE), (BAN, RED), (BAN, BLUE), (BAN, RED),
    (PICK, BLUE), (PICK, RED), (PICK, RED), (PICK, BLUE), (PICK, BLUE), (PICK, RED),
    (BAN, RED), (BAN, BLUE), (BAN, RED), (BAN, BLUE),
    (PICK, RED), (PICK, BLUE), (PICK, BLUE), (PICK, RED),
)
N_MOVES = len(SCHEDULE_RED_FIRST)


class IllegalMove(ValueError):
    """The message is written for the model to read and retry."""


class UnknownState(KeyError):
    pass


def key(text):
    """Canonical champion key; same rule as `schema.key` and `ChampIndex`."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def render_board(board, display=None):
    """The board as a reader sees it. Moved here so state owns its own view.

    `display` maps a canonical key back to the corpus spelling, so a reader gets
    "Jarvan IV" while the tools keep working on "jarvan iv".
    """
    def names(xs):
        xs = [display(x) if display else x for x in (xs or [])]
        return ", ".join(xs) if xs else "(none yet)"
    return ("Board:\n"
            "  blue picks: %s\n"
            "  red picks:  %s\n"
            "  bans:       %s\n"
            "  first pick: %s\n"
            "  patch:      %s"
            % (names(board.get("blue_picks")), names(board.get("red_picks")),
               names(board.get("bans")),
               "red" if board.get("first_pick") else "blue",
               board.get("patch") or "not given"))


@dataclass(frozen=True)
class DraftState:
    """One draft. Immutable: `apply` returns a new state, it does not edit this."""
    moves: tuple = ()          # ((kind, side, champ_key), ...)
    patch: str = None
    red_first: bool = False

    @property
    def schedule(self):
        return SCHEDULE_RED_FIRST if self.red_first else SCHEDULE_BLUE_FIRST

    @property
    def move_idx(self):
        return len(self.moves)

    def to_move(self):
        """-> {kind, side, move_idx}, or None when the draft is complete."""
        if self.move_idx >= N_MOVES:
            return None
        kind, side = self.schedule[self.move_idx]
        return {"kind": kind, "side": side, "move_idx": self.move_idx}

    def taken(self):
        return {champ for _k, _s, champ in self.moves}

    def board(self):
        blue = [c for k, s, c in self.moves if k == PICK and s == BLUE]
        red = [c for k, s, c in self.moves if k == PICK and s == RED]
        bans = [c for k, _s, c in self.moves if k == BAN]
        return {"blue_picks": blue, "red_picks": red, "bans": bans,
                "first_pick": 1 if self.red_first else 0, "patch": self.patch}

    def apply(self, champ, kind=None, side=None):
        """Play `champ` into the scheduled slot. -> a new DraftState.

        `kind`/`side` are optional and, when given, must match the schedule: the
        schedule is the authority, so a disagreement is a misunderstanding worth
        reporting rather than silently resolving.
        """
        champ = key(champ)
        if not champ:
            raise IllegalMove("no champion given")
        due = self.to_move()
        if due is None:
            raise IllegalMove("the draft is complete (20 moves); use undo_move "
                              "or start another sandbox")
        if kind and kind != due["kind"]:
            raise IllegalMove("move %d is a %s, not a %s"
                              % (due["move_idx"], due["kind"], kind))
        if side and side != due["side"]:
            raise IllegalMove("move %d is %s's, not %s's"
                              % (due["move_idx"], due["side"], side))
        if champ in self.taken():
            where = next((("%s %s" % (k, s)) for k, s, c in self.moves
                          if c == champ), "already")
            raise IllegalMove("%s is already on the board (%s)" % (champ, where))
        return replace(self, moves=self.moves + ((due["kind"], due["side"], champ),))

    def undo(self):
        if not self.moves:
            raise IllegalMove("nothing to undo - this state has no moves")
        return replace(self, moves=self.moves[:-1])

    def render(self, label=None, display=None):
        head = "State %s" % (label or "") if label else "State"
        due = self.to_move()
        turn = ("%s %s (move %d of %d)" % (due["side"], due["kind"],
                                          due["move_idx"], N_MOVES)
                if due else "complete")
        return "%s - to move: %s\n%s\n  legal champions left: %d" % (
            head, turn, render_board(self.board(), display),
            N_CHAMPS - len(self.taken()))

    def summary(self, label=None):
        due = self.to_move()
        return {"state": label, "move_idx": self.move_idx,
                "kind": (due or {}).get("kind"), "side": (due or {}).get("side"),
                "n_legal": N_CHAMPS - len(self.taken()),
                "red_first": self.red_first, "patch": self.patch,
                "complete": due is None}


class Store:
    """Named draft states. One real (read-only), any number of sandboxes."""

    def __init__(self, champs, real=None, patch=None, red_first=False):
        # `champs` is a `model_tools.ChampIndex`; the store needs names, not
        # weights, so this works even when no checkpoint is loaded.
        self.champs = champs
        self._states = {REAL_ID: [real or DraftState(patch=patch,
                                                     red_first=red_first)]}
        self._labels = {REAL_ID: "the live board"}
        self._next = 1

    def resolve(self, name):
        """Name -> canonical key, or a refusal that names the problem.

        `ChampIndex.resolve` raises on an unknown name; this translates it so
        every refusal from the store is an `IllegalMove` with the same shape,
        which is what the model reads.
        """
        try:
            idx = self.champs.resolve(name) if self.champs is not None else None
        except ValueError:
            idx = None
        if idx is None:
            raise IllegalMove("unknown champion %r - use the corpus spelling"
                              % (name,))
        return key(self.champs.names[idx])

    # ---------------------------------------------------------- reading
    def display(self, champ_key):
        """Canonical key -> the spelling the corpus and the models use."""
        try:
            return self.champs.names[self.champs.resolve(champ_key)]
        except (ValueError, TypeError, IndexError):
            return champ_key

    def ids(self):
        return list(self._states)

    def get(self, state_id=REAL_ID):
        try:
            return self._states[state_id][-1]
        except KeyError:
            raise UnknownState("unknown state %r; live states: %s"
                               % (state_id, ", ".join(self.ids()))) from None

    def history(self, state_id):
        return len(self._states[state_id])

    def label(self, state_id):
        return self._labels.get(state_id, "")

    def list(self):
        return [dict(self.get(s).summary(s), label=self.label(s),
                     depth=self.history(s), readonly=(s == REAL_ID))
                for s in self.ids()]

    # ---------------------------------------------------------- writing
    def set_real(self, state, label="the live board"):
        """The frontend's channel. Not reachable from the tool layer."""
        self._states[REAL_ID] = [state]
        self._labels[REAL_ID] = label
        return REAL_ID

    def clone(self, from_id=REAL_ID, name=None):
        """-> a new sandbox id holding a copy of `from_id`'s current state."""
        base = self.get(from_id)
        state_id = _slug(name) if name else "s%d" % self._next
        if state_id in self._states:
            state_id = "%s-%d" % (state_id, self._next)
        self._next += 1
        self._states[state_id] = [base]
        self._labels[state_id] = name or ("copy of %s" % from_id)
        return state_id

    def apply(self, state_id, champ, kind=None, side=None):
        if state_id == REAL_ID:
            raise IllegalMove("the live board is read-only; clone it first "
                              "(clone_state) and explore on the copy")
        new = self.get(state_id).apply(self.resolve(champ), kind=kind, side=side)
        self._states[state_id].append(new)
        return new

    def undo(self, state_id):
        if state_id == REAL_ID:
            raise IllegalMove("the live board is read-only")
        if len(self._states[state_id]) <= 1:
            raise IllegalMove("nothing to undo in %r" % state_id)
        self._states[state_id].pop()
        return self.get(state_id)

    def drop(self, state_id):
        if state_id == REAL_ID:
            raise IllegalMove("the live board cannot be dropped")
        self._states.pop(state_id, None)
        self._labels.pop(state_id, None)


def _slug(name):
    out = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return out[:32] or "s"
