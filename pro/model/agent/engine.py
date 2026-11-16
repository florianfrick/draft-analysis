"""The models a draft state can ask: the binding between a state and weights.

A state is data - an immutable move log, a fearless pool, a patch. It knows
nothing about torch, and `draft_state` has to stay importable in a process with
no checkpoint (the store needs the name table and nothing else). This module is
the other half. It holds the predictors, turns a state into the arguments
`model_tools` wants, and owns the **routing**: which model answers which
question, and when the honest answer is a refusal rather than an approximation.

Routing is the reason this is a class and not a bag of arguments:

  * the next-move head answers a board question, always;
  * it also answers a role question, but only on a pick move with the ban block
    complete - during the ban phase that question has no in-distribution form,
    and a masking checkpoint answers instead (`model_tools.can_answer_in_lane`);
  * the role table answers "which role is this" with no weights at all.

An engine with no predictor is legal and useful: it still assigns roles, so a
board can be published and read before any model is loaded. An engine with no
champion index is not - every question it answers is about a champion.
"""
from __future__ import annotations

from . import model_tools


class Engine:
    """The models one state service can reach. Shared by every clone of it."""

    def __init__(self, champs, predictor=None, masker=None, patch=None):
        if champs is None:
            raise ValueError("an engine needs a champion index (names, not "
                             "weights) - every question it answers is about a "
                             "champion")
        self.champs = champs
        self.predictor = predictor
        self.masker = masker
        self.patch = patch

    # ------------------------------------------------------------ describe
    def describe(self):
        """What is loaded, for a transcript or a header. Never the module."""
        return {"next_move": (model_tools.label(self.predictor)
                              if self.predictor is not None else None),
                "masker": (model_tools.label(self.masker)
                           if self.masker is not None else None),
                "roles": "champ-role table",
                "patch": self.patch}

    def can_predict(self):
        return self.predictor is not None

    # --------------------------------------------------------------- board
    def board(self, state):
        """A state -> the keyword arguments every model tool takes.

        The one place that converts the move log into the head's two side
        lists. A tool therefore cannot be handed a board that is not the
        state's, which is the failure the state service exists to remove.
        """
        return {"blue_picks": state.picks_of("blue"),
                "red_picks": state.picks_of("red"),
                "bans": state.bans(),
                "first_pick": 1 if state.red_first else 0,
                "patch": state.patch or self.patch,
                "unavailable": list(state.unavailable)}

    # ------------------------------------------------------------ questions
    def next_move(self, state, k=8, role=None, side=None):
        """The head's answer for the move on the clock, or for one role.

        With no `role` this is the board question and needs a checkpoint. With
        a `role` it is the role question, which the head can answer on a pick
        move and the masker has to answer otherwise - so it is allowed here
        even when no policy is loaded, as long as a masker is.
        """
        board = self.board(state)
        if role is None:
            if self.predictor is None:
                raise ValueError(
                    "no next-move model is loaded, so there is nothing to rank "
                    "the move on the clock. predict_roles still works - it is "
                    "a table lookup - and so does everything about the board.")
            return model_tools.predict_next_move(self.predictor, self.champs,
                                                 k=k, **board)
        return model_tools.predict_in_lane(self.predictor, self.champs, role,
                                           side=side, k=k, due=state.to_move(),
                                           masker=self.masker, **board)

    def predict_roles(self, state):
        board = self.board(state)
        return model_tools.predict_roles(self.champs, board["blue_picks"],
                                         board["red_picks"], board["bans"],
                                         board["unavailable"])

    # --------------------------------------------------------- assignment
    def assignment(self, state):
        """The per-slot role rows for a state, as `model_tools.assignment`
        returns them (a dict per filled slot).

        The champion is reported as the state's own canonical key, not the
        corpus spelling: `board()` says 'locke' and a role row that said 'Locke'
        would be a second spelling of one champion in one payload.
        """
        picks = [None if c is None else self.champs.resolve(c)
                 for c in state.picks()]
        rows = model_tools.assignment(self.champs, picks)
        for row in rows:
            row["champion"] = model_tools._norm(row["champion"])
        return rows
