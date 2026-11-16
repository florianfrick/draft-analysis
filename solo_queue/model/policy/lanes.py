"""Champion -> lane: which lanes a champion actually plays, and one assignment.

`data/champ_roles.json` is counted from the pro corpus by
`data/build_champ_roles.py`. This module is the reader plus the assignment the
serving path needs, and it lives in the model package because `infer._fill_roles`
needs it to decide each placed pick's lane - that value is the encoder's
`role_ids` input. Reaching into `frontend/` for it would invert the dependency:
the frontend imports this package, not the other way round.

Why it exists at all: the encoder is permutation-invariant within a team
(`base_model.py`), so the *only* thing a pick's slot position carries is its
role id. Serving used to fill that in with the role head's argmax, which agrees
with the corpus's real roles 67 % of the time. The champion's own history agrees
96 % of the time, and recall@5 follows - 30.2 % -> 36.7 % over 1,500 pick moves,
against 38.8 % for the true roles. The role head adds nothing on top (36.7 %),
so it is kept only for the per-slot distribution the role board displays, which
is a different question ("given this champion and team, where could it go").

Cost at serve time: a dict lookup and an exhaustive search over <=120 injective
assignments per team, which is what `roles.assign_roles` already did for the UI.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                  # solo_queue/model/policy -> repo root
TABLE_PATH = ROOT / "data" / "champ_roles.json"

# Same guard as infer.py: this module is imported both as `lanes` (bare, with
# this directory on sys.path) and as `solo_queue.model.policy.lanes`.
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Smoothing for P(lane | champion). Deliberately light: it only has to stop a
# zero count from being -inf, and to make an unseen champion a tie rather than a
# silent favourite. These are the values the 96 % agreement was measured with.
_ALPHA = 0.02

_CACHE = None


def table():
    """Parsed table, read once.

    Missing file is an error rather than a silent fallback to the role head:
    that fallback is 6.5 pp worse and nothing would say so. The file is
    committed; `python data/build_champ_roles.py` regenerates it.
    """
    global _CACHE
    if _CACHE is None:
        data = json.loads(TABLE_PATH.read_text("utf-8"))
        counts = {int(c): [int(v) for v in row]
                  for c, row in data["counts"].items()}
        _CACHE = {
            "roles": list(data["roles"]),
            "counts": counts,
            "share": float(data["policy"]["share"]),
            "floor": int(data["policy"]["floor"]),
            "source": dict(data.get("source") or {}),
        }
    return _CACHE


def lane_name(lane):
    """Display name for a role id.

    The table's `roles` are names, not data values: the corpus cache stores ids
    and the counts are positional, so renaming role 4 from UTILITY to SUPPORT
    changes no assignment. The soloQ analysis store is different - its `pos`
    column holds Riot's own strings and is left alone.
    """
    return table()["roles"][int(lane)]


def source():
    """Provenance (corpus size, build date) for logs and the UI."""
    return dict(table()["source"])


def legal_champions(lane):
    """Champion indices that play `lane`, under the table's own rule.

    A champion is legal when it was picked in that lane at least `floor` times
    and in at least `share` of its own picks. Sorted by index: the caller ranks
    them (the head's distribution decides the order), so only the set matters.
    """
    lane = int(lane)
    t = table()
    if not 0 <= lane < len(t["roles"]):
        raise ValueError("lane %d is not one of the five" % lane)
    out = []
    for champ, counts in t["counts"].items():
        n = sum(counts)
        if n and counts[lane] >= t["floor"] and counts[lane] / n >= t["share"]:
            out.append(champ)
    return sorted(out)


def lane_logp(champ, lane):
    """log P(lane | champion), smoothed. -inf never; unseen champion ties."""
    counts = table()["counts"].get(int(champ))
    if not counts:
        return math.log(1.0 / len(table()["roles"]))
    n = sum(counts)
    return math.log((counts[int(lane)] + _ALPHA) / (n + _ALPHA * len(counts)))


def assign_lanes(picks):
    """Lane per slot for a 10-slot board; None for empty slots.

    `picks` is the UI's board (team blocks, pick order, None for empty). A team's
    five lanes are distinct, so this is the same exhaustive search
    `roles.assign_roles` runs - <=120 leaves - but scored by the champions'
    own history instead of the role head's output.

    Maximising the summed log-likelihood is what resolves flex picks: two
    champions that both list JUNGLE get split by whichever second choice is
    likelier, and a single-lane champion (113 of 173) effectively claims its
    lane and pushes the flex one elsewhere.
    """
    t = table()
    n_roles = len(t["roles"])
    out = [None] * len(picks)
    for team in (0, 1):
        slots = [(s, int(picks[s])) for s in range(team * 5, team * 5 + 5)
                 if s < len(picks) and picks[s] is not None]
        if not slots:
            continue
        best = {"score": float("-inf"), "assign": {}}
        cur = {}
        used = [False] * n_roles

        def rec(i, score):
            if i == len(slots):
                if score > best["score"]:
                    best["score"] = score
                    best["assign"] = dict(cur)
                return
            slot, champ = slots[i]
            for lane in range(n_roles):
                if used[lane]:
                    continue
                used[lane] = True
                cur[slot] = lane
                rec(i + 1, score + lane_logp(champ, lane))
                used[lane] = False
                del cur[slot]

        rec(0, 0.0)
        for slot, lane in best["assign"].items():
            out[slot] = lane
    return out
