"""Rebuild the prompt's board from a live draft, not from a corpus row.

`common.render_state` reads a sample dict that `datasets.build_prefix` produces
from a 20-move corpus row. At serving time there is no row: the frontend has a
board (picks placed on their lanes, bans in the order they were made, a fearless
pool, a patch string). This module synthesises the row and hands it to the same
`build_prefix`, so the text the frontend sends is produced by the code that
produced every training example - not by a second renderer that would have to be
kept in step with the first.

Two layout facts make that possible, and both are easy to get wrong:

* **A pick's slot is `team * 5 + role`**, i.e. the board is indexed by *lane*,
  not by pick order (`datasets.build_prefix`). The frontend's 10 slots are pick
  order, so every placed pick has to arrive with its lane, and two picks on one
  team cannot share one.
* **`ban_ids` is chronological and its side is derived**, never stored
  (`datasets.ban_sides(first_pick)`). So the ban list must be in the order the
  bans were made - reordering it silently relabels which side banned what.

Because picks are lane-indexed, the *order* of the picks in the input does not
matter; only the bans are order-sensitive. `test_board.py` pins both.

What this module cannot know: the lane of the pick being asked for. In the corpus
that is the role the pro actually took; mid-draft it is a decision, not a fact.
`target_role` is therefore an argument, and `lane_is_assumed()` exists so a
caller can say so instead of pretending the lane was known.
"""
from __future__ import annotations

from pro.model.llm import common as C

DS = C.DS

# Filler for the moves after the target. `build_prefix` only looks at
# `moves[move_idx + 1]` (for the same-team double-pick partner), and a ban there
# is the one kind that cannot look like a partner, so padding with bans keeps
# that field at IGNORE no matter how short the real draft is.
_FILLER = (DS.MOVE_BAN, 0, 0, 0)


def _check_board(picks, lanes):
    if len(picks) != C.N_SLOTS or len(lanes) != C.N_SLOTS:
        raise ValueError("picks and lanes must both have %d slots"
                         % C.N_SLOTS)
    for team in (0, 1):
        seen = {}
        for slot in range(team * C.TEAM_SIZE, (team + 1) * C.TEAM_SIZE):
            if picks[slot] is None:
                if lanes[slot] is not None:
                    raise ValueError(
                        "slot %d is empty but carries lane %r - an empty slot "
                        "has no lane" % (slot, lanes[slot]))
                continue
            lane = lanes[slot]
            if lane is None or not 0 <= int(lane) < C.TEAM_SIZE:
                raise ValueError(
                    "slot %d is filled but its lane is %r; every pick needs a "
                    "lane 0-4 because the board is indexed by lane, not by "
                    "pick order" % (slot, lane))
            lane = int(lane)
            if lane in seen:
                raise ValueError(
                    "slots %d and %d are both on lane %s - a team cannot have "
                    "two champions in one lane" % (seen[lane], slot,
                                                   C.ROLE_DISPLAY[lane]))
            seen[lane] = slot


def row_from_board(picks, lanes, bans, first_pick, move_idx, patch,
                   target_kind, target_team, target_role=None,
                   target_champ=None, fearless=(), meta=None):
    """-> a 20-move corpus row that describes exactly this board.

    `picks`  [10] champion idx or None, the frontend's pick-order slots
    `lanes`  [10] role id (0-4) or None, one per filled slot
    `bans`   chronological champion idx, length = bans already made
    `target_kind`  DS.MOVE_BAN or DS.MOVE_PICK
    """
    _check_board(picks, lanes)
    n_picks = sum(1 for p in picks if p is not None)
    if len(bans) + n_picks != move_idx:
        raise ValueError(
            "the board has %d bans and %d picks but the move index is %d - the "
            "board and the move disagree about how much draft has happened"
            % (len(bans), n_picks, move_idx))

    sides = DS.ban_sides(int(first_pick))
    moves = [(DS.MOVE_BAN, int(sides[i]), int(c), 0)
             for i, c in enumerate(bans)]
    for slot in range(C.N_SLOTS):
        if picks[slot] is not None:
            moves.append((DS.MOVE_PICK, slot // C.TEAM_SIZE,
                          int(picks[slot]), int(lanes[slot])))
    if len(moves) != move_idx:                 # pragma: no cover - guarded above
        raise AssertionError("move list length %d != move_idx %d"
                             % (len(moves), move_idx))

    role = (int(target_role) if target_kind == DS.MOVE_PICK
            and target_role is not None else DS.IGNORE_INDEX)
    moves.append((int(target_kind), int(target_team),
                  int(target_champ if target_champ is not None else 0), role))
    while len(moves) < C.N_MOVES:
        moves.append(_FILLER)

    m = dict(meta or {})
    return {
        "moves": moves,
        "first_pick": int(first_pick),
        "unavailable": [int(c) for c in fearless],
        "patch": m.get("patch", patch),
        "gameid": m.get("gameid", "live"),
        "series_id": m.get("series_id", "live"),
        "league": m.get("league"),
        "date": m.get("date"),
        "blue_team": m.get("blue_team"),
        "red_team": m.get("red_team"),
        "blue_win": m.get("blue_win", 0),
        "has_econ": False,
        "econ": None,
    }


def sample_from_board(picks, lanes, bans, first_pick, move_idx, patch,
                      target_kind, target_team, target_role=None,
                      target_champ=None, fearless=(), meta=None,
                      num_champs=C.NUM_REAL_CHAMPS, patch_index=None):
    """-> the sample dict `common.render_state` and `prompt.assemble` read.

    `target_champ` is irrelevant to the *prompt* (the answer is not in it) and
    only matters if the caller scores a specific answer, which is how the
    reranker evaluates each candidate: one sample per candidate, batched.
    """
    row = row_from_board(picks, lanes, bans, first_pick, move_idx, patch,
                         target_kind, target_team, target_role=target_role,
                         target_champ=target_champ, fearless=fearless,
                         meta=meta)
    return DS.build_prefix(row, int(move_idx), num_champs=num_champs,
                           patch_index=patch_index)


def empty_lanes():
    """A fresh `lanes` list: no slot placed."""
    return [None] * C.N_SLOTS


def remaining_lanes(lanes, team):
    """Lanes that side has not placed a champion in yet, ascending.

    One remaining lane means the coming pick is *forced*; more than one means
    the lane is a decision the drafter has not made yet, so whatever is passed
    as `target_role` is a choice and the caller should label it as one.
    """
    filled = {int(lanes[slot]) for slot in
              range(team * C.TEAM_SIZE, (team + 1) * C.TEAM_SIZE)
              if lanes[slot] is not None}
    return [r for r in range(C.TEAM_SIZE) if r not in filled]
