"""A draft as an object: an immutable move log that can answer questions.

The board used to be an argument: every model tool took nine champion names and
hoped the caller retyped them correctly. That is a silent failure - a legal but
wrong board returns a confident distribution - and the `transcription.mismatched`
flag could only *detect* it. Here the board is held instead, and the tools are
methods on the object that holds it:

    state = store.get("real")            # the live board
    state.predict_next_move(k=5)         # what the head wants played
    state.predict_roles()                # what each side still needs
    state.predict_next_move(role="jungle", side="red")
    sandbox = store.clone()              # explore on a copy
    store.apply(sandbox, "Alistar")

That is the whole interface an agent needs. It never sees a champion list it has
to reproduce, and no tool takes a board.

Five decisions, each of which is the reason this is a move log rather than a set
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

**The role assignment is carried, not re-derived.** `apply` recomputes it, so
`get_state` can return which role every pick fills and how confident that is
without a second call. It is the same table the encoder's `role_ids` come from,
so the object and the model never disagree about where a champion is playing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .model_tools import ROLES

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
    out = ("Board:\n"
           "  blue picks: %s\n"
           "  red picks:  %s\n"
           "  bans:       %s\n"
           "  first pick: %s\n"
           "  patch:      %s"
           % (names(board.get("blue_picks")), names(board.get("red_picks")),
              names(board.get("bans")),
              "red" if board.get("first_pick") else "blue",
              board.get("patch") or "not given"))
    # The fearless pool is not on this board and is not derivable from it: those
    # champions were picked in *earlier games of the same series*. A reader that
    # cannot see the pool will happily propose one of them.
    pool = board.get("unavailable") or []
    if pool:
        out += ("\n  fearless:   %s (picked earlier in this series, so not "
                "available)" % names(pool))
    return out


@dataclass(frozen=True)
class Role:
    """One pick's role, and how sure the corpus is about it.

    `p_lane` is the champion's own share of its picks in that role: 0.98 is a
    dedicated laner, 0.45 a genuine flex. It is not a probability that the team
    wants the role, and it is not a prediction about this game.
    """
    slot: int
    side: str
    pick: int
    champion: str
    role: str
    p_lane: float

    def as_dict(self):
        return {"side": self.side, "pick": self.pick, "champion": self.champion,
                "role": self.role, "p_lane": self.p_lane}


@dataclass(frozen=True)
class DraftState:
    """One draft. Immutable: `apply` returns a new state, it does not edit this.

    `engine` is the model binding (`engine.Engine`). It is excluded from
    equality and hashing because two states with the same moves are the same
    draft whether or not a checkpoint is loaded; it is carried so the state can
    answer questions about itself.
    """
    moves: tuple = ()          # ((kind, side, champ_key), ...)
    patch: str = None
    red_first: bool = False
    # Fearless: champions picked in *earlier games of this series*. Not part of
    # the move log - nothing in this draft produced them - but they are illegal
    # here, and the head masks them before its top-k. Carried on the state
    # because a state that cannot say "these ten are gone" produces a legal,
    # confident, wrong answer, which is the failure this module exists to stop.
    unavailable: tuple = ()
    # Series context the board cannot say: which game of a series this is, and
    # what the earlier games looked like. The fearless pool is the *effect* of
    # the series on legality; these lines are the rest of the story (game 3 of
    # 5, how many picks each earlier game used). They are prose, not model
    # inputs, so they are carried as short lines for `render()` and `summary()`
    # rather than as a second board. Clones inherit them, and `get_state`
    # publishes them, so the agent reads the same series it was asked about.
    series: tuple = ()
    # The role assignment, recomputed by `apply`/`undo` and by the store when an
    # engine arrives. Empty when no engine is attached - not "unknown roles",
    # which is what `open_roles()` reports.
    roles: tuple = ()
    engine: object = field(default=None, compare=False, repr=False)

    # ------------------------------------------------------------- reading
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

    def blocked(self):
        """Champions this draft cannot use but that are not on its board."""
        return set(self.unavailable) - self.taken()

    def legal_count(self):
        return N_CHAMPS - len(self.taken() | set(self.unavailable))

    def picks_of(self, side):
        """One side's picks, in pick order, unpadded. The head's own layout."""
        return [c for k, s, c in self.moves if k == PICK and s == side]

    def picks(self):
        """The 10-slot board the encoder reads: blue 0-4, red 5-9, None empty."""
        blue, red = self.picks_of(BLUE), self.picks_of(RED)
        return (blue + [None] * (5 - len(blue))
                + red + [None] * (5 - len(red)))

    def bans(self):
        return [c for k, _s, c in self.moves if k == BAN]

    def board(self):
        return {"blue_picks": self.picks_of(BLUE), "red_picks": self.picks_of(RED),
                "bans": self.bans(),
                "first_pick": 1 if self.red_first else 0, "patch": self.patch,
                "unavailable": list(self.unavailable)}

    def open_roles(self):
        """{side: [role, ...]} still to be filled.

        Empty for a side with no engine: an unknown assignment must not read as
        "all five roles are open", which is what a caller would act on.
        """
        if self.engine is None:
            return {BLUE: [], RED: []}
        out = {}
        for side in (BLUE, RED):
            taken = {r.role for r in self.roles if r.side == side}
            out[side] = [r for r in ROLES if r not in taken]
        return out

    def roles_for(self, engine):
        """The assignment this state would have under `engine`, as Role rows."""
        if engine is None:
            return ()
        return tuple(Role(slot=r["slot"], side=r["side"], pick=r["pick"],
                          champion=r["champion"], role=r["role"],
                          p_lane=r["p_lane"])
                     for r in engine.assignment(self))

    # ----------------------------------------------------------- questions
    def _need_engine(self):
        if self.engine is None:
            raise IllegalMove("this state has no models attached, so it cannot "
                              "answer questions about itself - build it through "
                              "a Store, which attaches one")
        return self.engine

    def predict_next_move(self, k=8, role=None, side=None):
        """What the head wants played, or what it wants in one role.

        `role` narrows the question to the champions that play it - the query
        behind "what does the model think red's jungle is" and behind the role
        board. `side` names the side to answer for; naming the side that is not
        on the clock is a hypothetical, and the answer says so.
        """
        return self._need_engine().next_move(self, k=k, role=role, side=side)

    def predict_roles(self):
        """Which role each pick fills, with confidence, and what is still open."""
        return self._need_engine().predict_roles(self)

    # ------------------------------------------------------------- writing
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
        if champ in self.unavailable:
            raise IllegalMove(
                "%s was picked in an earlier game of this series, so it is "
                "fearless-unavailable in this one - pick someone else" % champ)
        new = replace(self, moves=self.moves + ((due["kind"], due["side"], champ),))
        # The assignment moves with the board: one role can change hands when a
        # flex champion arrives, and a state whose roles lag its moves is the
        # kind of quiet disagreement this module exists to prevent.
        return replace(new, roles=new.roles_for(new.engine))

    def undo(self):
        if not self.moves:
            raise IllegalMove("nothing to undo - this state has no moves")
        new = replace(self, moves=self.moves[:-1])
        return replace(new, roles=new.roles_for(new.engine))

    # ----------------------------------------------------------- rendering
    def render_roles(self, display=None):
        """The assignment as two lines, plus what each side still needs."""
        if not self.roles:
            return ""
        lines = []
        for side in (BLUE, RED):
            rows = [r for r in self.roles if r.side == side]
            if not rows:
                continue
            cells = ", ".join(
                "%s (%s %.2f)" % (display(r.champion) if display else r.champion,
                                  r.role, r.p_lane)
                for r in rows)
            lines.append("  %s roles: %s" % (side, cells))
            opened = self.open_roles().get(side) or []
            if opened:
                lines.append("  %s still needs: %s" % (side, ", ".join(opened)))
        return "\n".join(lines)

    def render(self, label=None, display=None):
        head = "State %s" % (label or "") if label else "State"
        due = self.to_move()
        turn = ("%s %s (move %d of %d)" % (due["side"], due["kind"],
                                          due["move_idx"], N_MOVES)
                if due else "complete")
        series = ("\n" + "\n".join("  " + line for line in self.series)
                  if self.series else "")
        out = "%s - to move: %s\n%s%s\n  legal champions left: %d" % (
            head, turn, render_board(self.board(), display), series,
            self.legal_count())
        roles = self.render_roles(display)
        return out + ("\n" + roles if roles else "")

    def summary(self, label=None):
        due = self.to_move()
        return {"state": label, "move_idx": self.move_idx,
                "kind": (due or {}).get("kind"), "side": (due or {}).get("side"),
                "n_legal": self.legal_count(),
                "red_first": self.red_first, "patch": self.patch,
                "fearless": list(self.unavailable),
                "series": list(self.series),
                "roles": [r.as_dict() for r in self.roles],
                "open_roles": self.open_roles(),
                "complete": due is None}


def log_from_board(picks, bans=(), red_first=False, patch=None,
                   bans_blue=None, bans_red=None, unavailable=(), engine=None,
                   series=()):
    """The frontend's board -> the chronological move log. -> DraftState.

    The two shapes are not the same question. The UI keeps **slots** (ten, two
    blocks of five in pick order) and **side-grouped bans**; a state keeps
    **moves**, in the order they were made, because that is what the model reads
    and what `to_move()` indexes. Nothing else in the codebase converts one to
    the other, so a board reaching the store any other way would be a guess.

    Two rules, both of which are about what the UI cannot tell us:

    **Bans are split by the schedule, not by halving.** A flat ban list is the
    UI's `allBansFlat`: blue's bans in order, then red's. How many belong to
    each side is not in the list, but it *is* in the schedule - the first n bans
    of the draft are dealt to known sides - so the split is exact for any legal
    prefix. Halving is only right when both sides have made five, which is why
    `draftscore.score`'s fallback is wrong mid-phase-1 (2 blue + 1 red reads as
    1 blue + 2 red). Pass `bans_blue`/`bans_red` and this is moot.

    **The log stops at the first gap.** A pick slot is not a move index: a coach
    can fill blue's third slot and leave the second empty, and there is no
    chronological claim to make about that board. `draftscore.replay` skips the
    gap and keeps going, because it scores each placed pick independently; a
    state cannot, because move 7 cannot exist without move 6. So the log is the
    longest prefix the board supports, and `move_idx` is its length.

    `unavailable` is the fearless pool - champions picked in earlier games of
    the series. It is not in the log and cannot be: this draft did not produce
    them. It rides on the state so the head can mask it and `apply` can refuse
    it; a pool entry that is also on this board is dropped, which is not a
    silent repair - `taken()` already blocks it, so legality is unchanged.

    `series` is the accompanying context (lines like "series: game 3 of 5").
    It is not an input to any model: the head already has the pool, and what is
    left is for the reader. It is carried on the state so `get_state` reports
    the same series the first message described.
    """
    schedule = SCHEDULE_RED_FIRST if red_first else SCHEDULE_BLUE_FIRST
    picks = list(picks or [])
    if len(picks) < 10:
        picks = picks + [None] * (10 - len(picks))

    if bans_blue is not None or bans_red is not None:
        by_side = {BLUE: [key(b) for b in (bans_blue or [])],
                   RED: [key(b) for b in (bans_red or [])]}
    else:
        flat = [key(b) for b in (bans or [])]
        # Deal the flat list onto the schedule to learn each side's count.
        ban_sides = [side for kind, side in schedule if kind == BAN]
        n_blue = sum(1 for side in ban_sides[:len(flat)] if side == BLUE)
        by_side = {BLUE: flat[:n_blue], RED: flat[n_blue:]}

    used = {BLUE: 0, RED: 0}          # picks consumed, per side
    moves, seen = [], set()
    for kind, side in schedule:
        if kind == BAN:
            queue = by_side[side]
            if not queue:
                break
            champ = queue.pop(0)
        else:
            slot = used[side] if side == BLUE else 5 + used[side]
            champ = picks[slot] if slot < len(picks) else None
            if champ is None:
                break
            used[side] += 1
            champ = key(champ)
        if not champ:
            break
        if champ in seen:
            raise IllegalMove("%s appears twice on this board - the UI and the "
                              "state disagree" % champ)
        seen.add(champ)
        moves.append((kind, side, champ))
    pool = []
    for champ in (key(c) for c in (unavailable or [])):
        if champ and champ not in seen and champ not in pool:
            pool.append(champ)
    state = DraftState(moves=tuple(moves), patch=patch,
                       red_first=bool(red_first), unavailable=tuple(pool),
                       series=tuple(str(s) for s in (series or ())),
                       engine=engine)
    return replace(state, roles=state.roles_for(engine))


class Store:
    """Named draft states. One real (read-only), any number of sandboxes."""

    def __init__(self, champs, real=None, patch=None, red_first=False,
                 engine=None):
        # `champs` is a `model_tools.ChampIndex`; the store needs names, not
        # weights, so this works even when no checkpoint is loaded.
        self.champs = champs
        # An engine with no predictor is still worth having: it assigns roles,
        # so a board can be read before any weights are loaded. A `real` state
        # that already carries models keeps them - otherwise seeding a store
        # with a board would silently drop the checkpoint it was built with.
        self.engine = engine
        if self.engine is None and getattr(real, "engine", None) is not None:
            self.engine = real.engine
        if self.engine is None and champs is not None:
            from .engine import Engine
            self.engine = Engine(champs=champs)
        self._states = {REAL_ID: [self._bind(real or DraftState(
            patch=patch, red_first=red_first))]}
        self._labels = {REAL_ID: "the live board"}
        self._next = 1

    def _bind(self, state):
        """Attach this store's engine and recompute the roles it implies.

        Every state in the store is bound, including one a caller constructed
        directly: a state with moves and no assignment would answer
        `get_state` with an empty role list, which reads as "nothing placed".
        """
        return replace(state, engine=self.engine,
                       roles=state.roles_for(self.engine))

    def attach(self, engine):
        """Swap the model binding under every state, live ones included.

        The frontend loads and swaps checkpoints while the board is open, so the
        binding cannot be fixed at construction. Roles are recomputed because
        the table is part of the engine and a different engine can assign
        differently (a different champion index, a different table).
        """
        self.engine = engine
        for state_id, history in list(self._states.items()):
            self._states[state_id] = [self._bind(s) for s in history]
        return engine

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
        self._states[REAL_ID] = [self._bind(state)]
        self._labels[REAL_ID] = label
        return REAL_ID

    def set_board(self, picks, bans=(), red_first=False, patch=None,
                  bans_blue=None, bans_red=None, unavailable=(), series=()):
        """A UI board -> the live state. -> (state, the state it replaced).

        The translation from indices to canonical keys lives here rather than in
        the server because the store is what holds the name table. An index the
        table cannot spell is refused: silently dropping it would shorten the
        log and hand the model a board that is not the one on screen.

        `unavailable` is the UI's fearless pool, in the same index space.
        """
        names = list(getattr(self.champs, "names", None) or [])

        def nm(x):
            if x is None or isinstance(x, str):
                return x
            i = int(x)
            if not 0 <= i < len(names):
                raise IllegalMove("champion index %d is outside the name table "
                                  "(%d entries)" % (i, len(names)))
            return names[i]

        state = log_from_board([nm(p) for p in (picks or [])],
                               bans=[nm(b) for b in (bans or [])],
                               red_first=red_first, patch=patch,
                               bans_blue=None if bans_blue is None
                               else [nm(b) for b in bans_blue],
                               bans_red=None if bans_red is None
                               else [nm(b) for b in bans_red],
                               unavailable=[nm(u) for u in (unavailable or [])],
                               series=series, engine=self.engine)
        old = self.get(REAL_ID)
        self.set_real(state)
        return state, old

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
