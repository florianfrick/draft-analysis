"""Slot -> lane assignment for a draft board.

Lives on its own because two callers need it: `/api/suggest` (the role board,
where the slot the user just clicked is pinned) and the reranker (where the
lanes are needed to build the prompt, and no slot is "newest"). It used to sit
in `server.py`, which the reranker cannot import - `server` imports the
reranker, so reaching back for one function would be a cycle.
"""
from __future__ import annotations

import json
from pathlib import Path


def assign_roles(role_dists, filled, newest=None):
    """One-to-one slot -> role assignment per team.

    The role head is per-slot, so two picks on one team can argmax the same
    role. A team's five roles are distinct, so collisions are resolved with
    an exhaustive search over injective assignments (<=5 picks -> <=120
    leaves).

    `newest` (the slot just filled) is pinned to its own argmax first and
    the rest are then optimised jointly. That ordering matters: a pure
    sum-of-probabilities optimum happily leaves an older pick in a
    contested role - pick Vi (JUNGLE .97) while Pantheon (JUNGLE .60, TOP
    .30) is already there and the optimum keeps Pantheon in JUNGLE and
    pushes Vi to TOP, so the board looks like it ignored the click. Pinning
    the newest pick is lexicographic: the pick the user just made keeps the
    role the model likes best for it, and the older ones move aside.

    Returns a length-10 list of role ids (None for empty slots).
    """
    out = [None] * len(role_dists)
    for team in (0, 1):
        slots = [s for s in range(team * 5, team * 5 + 5)
                 if filled[s] and role_dists[s]]
        if not slots:
            continue
        if newest in slots:
            first, rest = newest, [s for s in slots if s != newest]
        else:
            first, rest = None, slots
        used = [False] * 5
        if first is not None:
            for cand in role_dists[first]:        # already prob-descending
                r = cand["role"]
                if 0 <= r < 5 and not used[r]:
                    out[first] = r
                    used[r] = True
                    break
        cur = {}
        best = {"score": -1.0, "assign": {}}

        def rec(i, score):
            if i == len(rest):
                if score > best["score"]:
                    best["score"] = score
                    best["assign"] = dict(cur)
                return
            for cand in role_dists[rest[i]]:
                r = cand["role"]
                if not (0 <= r < 5) or used[r]:
                    continue
                used[r] = True
                cur[rest[i]] = r
                rec(i + 1, score + cand["prob"])
                used[r] = False
                del cur[rest[i]]

        rec(0, 0.0)
        for s, r in best["assign"].items():
            out[s] = r
    return out


# ---------------------------------------------------------------- lane legality
#
# `assign_roles` above answers "which lane is this pick". This answers the other
# direction, "which champions play this lane", which is what a lane-constrained
# shortlist needs. The table is counted from the pro corpus by
# `data/build_champ_roles.py`; the rule (a share of the champion's own picks,
# plus a floor) is applied here at read time so the thresholds can be tuned
# without a rebuild.

TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "champ_roles.json"
_TABLE = None


def _table():
    """Parsed table: {"roles": [...], "legal": [[champ ids] per lane], "source": {}}.

    Built once. A missing file is an error, not an empty pool: silently serving
    an unconstrained list is the bug this table exists to fix, so the caller
    gets an exception it can report next to the answer.
    """
    global _TABLE
    if _TABLE is None:
        data = json.loads(TABLE_PATH.read_text("utf-8"))
        share = float(data["policy"]["share"])
        floor = int(data["policy"]["floor"])
        legal = [[] for _ in data["roles"]]
        for champ, counts in data["counts"].items():
            n = sum(counts)
            if not n:
                continue
            for lane, v in enumerate(counts):
                if v >= floor and v / n >= share:
                    legal[lane].append(int(champ))
        _TABLE = {"roles": list(data["roles"]),
                  "legal": [sorted(x) for x in legal],
                  "source": data.get("source") or {}}
    return _TABLE


def legal_champions(lane):
    """Champion indices that play `lane`, most-picked first is NOT implied.

    Sorted by index, because the caller ranks them itself (the head's
    distribution decides the order). The point of the table is the *set*.
    """
    lane = int(lane)
    if not 0 <= lane < 5:
        raise ValueError("lane %d is not one of the five" % lane)
    return list(_table()["legal"][lane])


def lane_name(lane):
    return _table()["roles"][int(lane)]


def table_source():
    """Provenance of the table, for the UI/logs (corpus size and build date)."""
    return dict(_table()["source"])
