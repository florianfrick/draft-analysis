"""Slot -> lane assignment for a draft board.

Lives on its own because two callers need it: `/api/suggest` (the role board,
where the slot the user just clicked is pinned) and the reranker (where the
lanes are needed to build the prompt, and no slot is "newest"). It used to sit
in `server.py`, which the reranker cannot import - `server` imports the
reranker, so reaching back for one function would be a cycle.
"""
from __future__ import annotations


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
# `assign_roles` above answers "which lane is this pick", from the role head's
# per-slot distribution. The other direction - "which champions play this lane",
# and "which lane is each champion actually in" - is answered by the corpus
# table, and it lives in the model package because `infer._fill_roles` needs it
# too (the policy's `role_ids` input). Re-exported here so the frontend has one
# import for both and there is no second copy of the table to drift.

from solo_queue.model.policy.lanes import (  # noqa: E402,F401
    assign_lanes,
    lane_name,
    legal_champions,
    source as table_source,
)
