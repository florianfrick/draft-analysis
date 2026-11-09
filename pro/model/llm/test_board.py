"""Contract tests for rebuilding the prompt's board from a live draft.

The whole point of `board.py` is that the frontend does not get a second
renderer: it synthesises a corpus row and reuses `datasets.build_prefix`, so the
text it sends is the text the model was trained on. These tests are what makes
that claim checkable rather than asserted - a real row's board is extracted,
rebuilt from the board alone, and the two renderings must be byte-identical.

CPU only, no tokenizer, no model.
"""
from __future__ import annotations

import pytest
import torch

from pro.model.llm import board as BD
from pro.model.llm import common as C

DS = C.DS

PRO_DRAFTS = C.PRO_DIR / "pro_drafts.pt"
CANDIDATES = [(3, 0.31), (11, 0.22), (42, 0.18), (7, 0.16), (90, 0.13)]


@pytest.fixture(scope="module")
def rows():
    if not PRO_DRAFTS.exists():
        pytest.skip("no cached pro drafts at %s" % PRO_DRAFTS)
    payload = torch.load(str(PRO_DRAFTS), map_location="cpu",
                         weights_only=False)
    rows = payload["rows"] if isinstance(payload, dict) else payload
    usable = [r for r in rows if len(r["moves"]) == C.N_MOVES]
    assert usable, "no full 20-move draft in the cache"
    picked = usable[:4]
    # One row with a fearless pool and one where red picks first: the pool line
    # and the ban-side derivation are the two parts a naive rebuild gets wrong.
    for want in (lambda r: bool(r.get("unavailable")),
                 lambda r: int(r["first_pick"]) == 1):
        hit = next((r for r in usable if want(r)), None)
        if hit is not None and hit not in picked:
            picked.append(hit)
    return picked


def board_of(sample):
    """The board state a frontend would hold, read off a real sample.

    `picks`/`lanes` are the frontend's pick-order slots, so the extraction is a
    plain read - if the layout were pick-ordered this would be a shuffle.
    """
    n_bans = int(sample["n_bans"])
    bans = [int(c) for c in sample["ban_ids"][:n_bans]]
    picks, lanes = [None] * C.N_SLOTS, [None] * C.N_SLOTS
    for slot in range(C.N_SLOTS):
        if not bool(sample["pad_mask"][slot]):
            picks[slot] = int(sample["champ_ids"][slot])
            lanes[slot] = int(sample["role_ids"][slot])
    return picks, lanes, bans


def rebuild(row, move_idx, picks, lanes, bans):
    kind, team, champ, role = row["moves"][move_idx]
    return BD.sample_from_board(
        picks, lanes, bans, row["first_pick"], move_idx, row["patch"],
        kind, team, target_role=role, target_champ=champ,
        fearless=row.get("unavailable") or ())


def test_rebuilt_board_renders_identically(rows):
    """The only test that matters: same text, from the board alone."""
    names = C.champ_names()
    checked = 0
    for row in rows:
        for move_idx in range(C.N_MOVES):
            real = DS.build_prefix(row, move_idx, C.NUM_REAL_CHAMPS)
            picks, lanes, bans = board_of(real)
            built = rebuild(row, move_idx, picks, lanes, bans)
            for fmt in ("v2", "v4"):
                want = C.render_state(real, names, prompt_format=fmt,
                                      candidates=CANDIDATES)
                got = C.render_state(built, names, prompt_format=fmt,
                                     candidates=CANDIDATES)
                assert got == want, (
                    "move %d of %s rendered differently\n--- real ---\n%s\n"
                    "--- rebuilt ---\n%s" % (move_idx, row["gameid"], want, got))
            checked += 1
    assert checked == C.N_MOVES * len(rows)


def test_pick_order_does_not_matter_but_ban_order_does(rows):
    """Picks are lane-indexed, so their order is free; bans are chronological.

    This is the trap in the layout: a frontend that hands over picks in pick
    order is fine, one that hands over bans out of order silently relabels which
    side banned what.
    """
    names = C.champ_names()
    row = next(r for r in rows if sum(1 for m in r["moves"][:10]
                                      if m[0] == DS.MOVE_BAN) >= 4)
    move_idx = 6
    real = DS.build_prefix(row, move_idx, C.NUM_REAL_CHAMPS)
    picks, lanes, bans = board_of(real)
    assert len(bans) >= 4

    base = C.render_state(rebuild(row, move_idx, picks, lanes, bans), names,
                          prompt_format="v2")

    # Picks: reorder the filled slots among themselves, keeping lane pairing.
    filled = [s for s in range(C.N_SLOTS) if picks[s] is not None]
    shuffled_picks, shuffled_lanes = [None] * C.N_SLOTS, [None] * C.N_SLOTS
    for dst, src in zip(filled, reversed(filled)):
        shuffled_picks[dst], shuffled_lanes[dst] = picks[src], lanes[src]
    assert C.render_state(
        rebuild(row, move_idx, shuffled_picks, shuffled_lanes, bans), names,
        prompt_format="v2") == base

    # Bans: swapping two of them moves the champions between the side lines.
    swapped = list(bans)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert C.render_state(rebuild(row, move_idx, picks, lanes, swapped), names,
                          prompt_format="v2") != base


def test_a_pick_without_a_lane_is_refused(rows):
    row = rows[0]
    real = DS.build_prefix(row, 8, C.NUM_REAL_CHAMPS)
    picks, lanes, bans = board_of(real)
    slot = next(s for s in range(C.N_SLOTS) if picks[s] is not None)
    lanes[slot] = None
    with pytest.raises(ValueError, match="needs a lane"):
        rebuild(row, 8, picks, lanes, bans)


def test_two_picks_on_one_lane_are_refused(rows):
    row = rows[0]
    # move 11 is deep enough that one side has three picks on the board.
    real = DS.build_prefix(row, 11, C.NUM_REAL_CHAMPS)
    picks, lanes, bans = board_of(real)
    team = next(t for t in (0, 1)
                if picks[t * 5:(t + 1) * 5].count(None) <= 3)
    slots = [s for s in range(team * 5, team * 5 + 5) if picks[s] is not None]
    lanes[slots[1]] = lanes[slots[0]]
    with pytest.raises(ValueError, match="two champions in one lane"):
        rebuild(row, 11, picks, lanes, bans)


def test_a_board_that_disagrees_with_the_move_index_is_refused(rows):
    row = rows[0]
    real = DS.build_prefix(row, 8, C.NUM_REAL_CHAMPS)
    picks, lanes, bans = board_of(real)
    with pytest.raises(ValueError, match="move disagree"):
        rebuild(row, 9, picks, lanes, bans)


def test_remaining_lanes_is_what_makes_a_lane_forced(rows):
    """The lane of a pick is forced only when one lane is left.

    `remaining_lanes` is what the reranker uses to decide whether it is
    choosing a lane or reporting one, so the count is the contract: 1 means the
    prompt states a fact, more than 1 means it states a decision.
    """
    row = rows[0]
    real = DS.build_prefix(row, 8, C.NUM_REAL_CHAMPS)
    picks, lanes, bans = board_of(real)
    for team in (0, 1):
        placed = [lanes[s] for s in range(team * 5, team * 5 + 5)
                  if lanes[s] is not None]
        remaining = BD.remaining_lanes(lanes, team)
        assert len(remaining) == C.TEAM_SIZE - len(placed)
        assert not (set(remaining) & set(placed))
    # An empty board has every lane open, so nothing is forced.
    empty = BD.empty_lanes()
    assert BD.remaining_lanes(empty, 0) == list(range(C.TEAM_SIZE))
    assert len(BD.remaining_lanes(empty, 0)) == C.TEAM_SIZE


def test_the_pool_reaches_the_prompt(rows):
    """A fearless pool must render as `Fearless Bans:` - 21 % of the corpus."""
    names = C.champ_names()
    row = next((r for r in rows if r.get("unavailable")), None)
    if row is None:
        pytest.skip("no pooled row in the sample")
    real = DS.build_prefix(row, 12, C.NUM_REAL_CHAMPS)
    picks, lanes, bans = board_of(real)
    text = C.render_state(rebuild(row, 12, picks, lanes, bans), names,
                          prompt_format="v2")
    assert "Fearless Bans: " in text
    assert names[int(row["unavailable"][0])] in text
