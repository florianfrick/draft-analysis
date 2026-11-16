"""Champion -> lane counts from the pro corpus -> ``data/champ_roles.json``.

Why this exists: the next-move head has no lane input. It answers "who is the
best next champion", so its top-k is a mix of lanes - measured over 1,200 pick
moves, an 8-long list covers 3.2 distinct lanes and only 3.5 of its 8 entries
are legal for the lane the pick actually went to. Asking the LLM reranker for
"the best pick for Bot lane" cannot fix that, because the LLM only chooses
among the five the head hands it.

So the lane has to constrain the *pool*. That needs one fact the models do not
carry: which champions are played in which lane. The corpus already knows - every
pick in `pro_drafts.pt` records its role - so this script counts it.

Legality is a share of the champion's own history, not a share of the lane's:
"Qiyana was picked in the jungle in 93 % of her games" is the claim we want, and
it is the one that stays stable as pick rates move. A champion with fewer than
`floor` picks in a lane is not legal there whatever the share says, so a single
off-role appearance does not open a lane.

Measured on the current corpus (46,915 drafts, 469,150 pick-role pairs): 113
champions are legal in exactly one lane, 54 in two, 6 in three, and only 2.5 %
of pro picks land outside their champion's legal lanes. That last number is the
price of the constraint: a lane-filtered list cannot contain those picks.

Run: ``python data/build_champ_roles.py`` (writes the JSON, prints the summary).
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "champ_roles.json"
# Display names for the five role ids. The corpus stores ids, never names, so
# this rename is display-only: the counts below are positional and do not move.
# (The soloQ analysis store keeps Riot's own `pos` string, "UTILITY", which is
# a raw data value in a built DuckDB - not a name this table chooses.)
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]

# A champion is legal in a lane when it was picked there at least SHARE of its
# own picks and at least FLOOR times. 0.10/5 is the knee: 0.25 makes Tristana
# (68 % mid, 30 % bot) a mid-only champion and 0.05 admits Yasuo to every lane
# he has ever been trolled into. Stored in the JSON so the reader sees the
# policy the numbers were cut with.
SHARE = 0.10
FLOOR = 5


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "propipeline", HERE / "pro play" / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["propipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    P = load_pipeline()
    rows, _stats = P.build_rows()
    counts = Counter()
    for r in rows:
        for champ, role in zip(r["champ"], r["role"]):
            counts[(int(champ), int(role))] += 1

    champs = sorted({c for c, _ in counts})
    table = {str(c): [counts[(c, r)] for r in range(len(ROLES))] for c in champs}

    per_champ = {c: sum(v) for c, v in
                 ((c, [counts[(c, r)] for r in range(len(ROLES))]) for c in champs)}

    def legal(c):
        n = per_champ[c]
        return [r for r in range(len(ROLES))
                if table[str(c)][r] >= FLOOR and table[str(c)][r] / n >= SHARE]

    n_roles = Counter(len(legal(c)) for c in champs)
    odd = sum(v for (c, r), v in counts.items()
              if per_champ[c] >= FLOOR and v / per_champ[c] < SHARE)
    total = sum(counts.values())

    payload = {
        "source": {
            "corpus": "pro_drafts.pt",
            "drafts": len(rows),
            "pick_role_pairs": total,
            "champions": len(champs),
            "built": "2026-11-09",
        },
        "policy": {
            "share": SHARE,
            "floor": FLOOR,
            "rule": ("a champion is legal in a lane when it was picked there "
                     "at least `share` of its own picks and at least `floor` "
                     "times"),
            "note": ("counts are raw pick counts per lane; the rule is applied "
                     "at read time by frontend/roles.py, so changing a "
                     "threshold does not need a rebuild"),
        },
        "roles": ROLES,
        "counts": table,
    }
    # One line per champion: 173 entries in a diffable shape, instead of the
    # six lines per list json.dumps(indent=1) would produce.
    body = ",\n".join(
        '  "%d": [%s]' % (c, ", ".join(str(v) for v in table[str(c)]))
        for c in champs)
    OUT.write_text(
        '{\n "source": %s,\n "policy": %s,\n "roles": %s,\n "counts": {\n%s\n }\n}\n'
        % (json.dumps(payload["source"], sort_keys=True),
           json.dumps(payload["policy"], sort_keys=True),
           json.dumps(ROLES), body),
        encoding="utf-8")

    print("wrote %s: %d champions, %d pick-role pairs from %d drafts"
          % (OUT.relative_to(HERE.parent), len(champs), total, len(rows)))
    print("legal lanes per champion: %s" % dict(sorted(n_roles.items())))
    print("picks outside their champion's legal lanes: %d (%.2f %%)"
          % (odd, 100 * odd / total))
    for lane, name in enumerate(ROLES):
        pool = [c for c in champs if lane in legal(c)]
        print("  %-8s %3d champions" % (name, len(pool)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
