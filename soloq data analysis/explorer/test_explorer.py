"""End-to-end checks against a running explorer server.

    python explorer/server.py --port 8778
    python explorer/test_explorer.py

Covers the query semantics, not just the shapes: that "either side" really is
orientation-free, that the team win rate is read for the selection rather than
for the literal blue side, and that the pair/suggest numbers agree with the
cube they came from.
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8778"


def get(path):
    t0 = time.perf_counter()
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        body = json.loads(r.read().decode())
    return body, (time.perf_counter() - t0) * 1000


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("  " + detail if detail else ""))
    return ok


fails = 0

print("meta")
m, ms = get("/api/meta")
fails += not check("corpus size", m["games"] == 423494, f"{m['games']} games, {ms:.0f} ms")
fails += not check("ten players per game", m["players"] == 10 * m["games"],
                   f"{m['players']} rows")
fails += not check("champions", len(m["champions"]) == 173, f"{len(m['champions'])}")
fails += not check("icons resolved", all(c["img"].startswith("http") for c in m["champions"]))
fails += not check("patches", len(m["patches"]) == 2, str(m["patches"]))
visible = [x for x in m["metrics"] if not x.get("hidden")]
fails += not check("every field is selectable", len(visible) == len(m["metrics"]),
                   f"{len(visible)} of {len(m['metrics'])}")
fails += not check("win rate is a metric", any(x["key"] == "win" for x in visible))
fails += not check("every metric has a direction",
                   all(x.get("better") in (1, -1) for x in m["metrics"]))
fails += not check("every metric reports its coverage",
                   all(isinstance(x.get("cov"), float) for x in m["metrics"]))

# The registry has to keep up with the collector. Import both sides rather than
# trusting a hand-written list: the collector's whitelist is the definition of
# "collected", the store's CH_KEYS is what survives to disk, and METRICS is what
# the UI can show. A new key added to only one of the three fails here.
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import store  # noqa: E402  (needs the analysis dir on the path)

COLLECTOR = os.path.join(store.ROOT, "data", "solo queue", "collect_match_data.py")
tree = ast.parse(open(COLLECTOR, encoding="utf-8").read())
collected = None
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "CHALLENGE_KEYS" for t in node.targets):
        collected = [e.value for e in node.value.elts]
fails += not check("collector whitelist found", collected is not None,
                   f"{len(collected or [])} keys")
fails += not check("store keeps every collected key",
                   set(collected) == set(store.CH_KEYS),
                   "missing: %s" % sorted(set(collected) - set(store.CH_KEYS)))
ui = {x["key"] for x in m["metrics"]}
fails += not check("UI exposes every collected key", set(collected) <= ui,
                   "missing: %s" % sorted(set(collected) - ui))
fails += not check("UI exposes the box score and flags",
                   {"win", "kills", "deaths", "assists", "gold", "dmg", "cs",
                    "vision", "fb_kill", "fb_assist", "ft_kill",
                    "ft_assist"} <= ui)
fails += not check("no metric the store cannot answer",
                   ui <= set(store.AGG_STATS), sorted(ui - set(store.AGG_STATS)))

print("\nmatchup (capability 1, precomputed)")
# one throwaway call so the timed one is not measuring a cold page cache
get("/api/matchup?champ=103&partner=64&mode=with&patch=16.20")
d, ms = get("/api/matchup?champ=103&partner=64&mode=with&patch=16.20")
fails += not check("n > 3000", d["n"] > 3000, f"n={d['n']}, {ms:.0f} ms")
fails += not check("stats present", d["stats"]["win"] is not None)
fails += not check("delta present", "delta" in d)
fails += not check("baseline present", d["baseline"]["n"] > 30000)
fails += not check("fast", ms < 60, f"{ms:.0f} ms")

d2, _ = get("/api/matchup?champ=103&partner=64&mode=vs&patch=16.20")
fails += not check("vs differs from with", d2["n"] != d["n"], f"vs n={d2['n']}")

d3, _ = get("/api/matchup?champ=103&partner=64&mode=with&patch=16.20&role=MIDDLE")
fails += not check("role breakdown", "role" in d3 and d3["role"]["pos"] == "MIDDLE")

print("\ndraft (capability 2, live scan)")
# one throwaway call so the timed one is not measuring a cold page cache
get("/api/draft?blue=8,53,64,84,523&red=113,126,145,518,711&patch=16.20")
d, ms = get("/api/draft?blue=8,53,64,84,523&red=113,126,145,518,711&patch=16.20")
fails += not check("full draft has a count", isinstance(d["n"], int), f"n={d['n']}, {ms:.0f} ms")
fails += not check("ladder length 10", len(d["ladder"]) == 10)
fails += not check("ladder monotone", all(d["ladder"][i]["n"] >= d["ladder"][i + 1]["n"]
                                          for i in range(9)),
                   " -> ".join(str(x["n"]) for x in d["ladder"]))
fails += not check("ladder last == n", d["ladder"][-1]["n"] == d["n"])
fails += not check("drop_one length 10", len(d["drop_one"]) == 10)
fails += not check("drop_one >= n", all(x["n_without"] >= d["n"] for x in d["drop_one"]))
fails += not check("pair rows = 2*n*(n-1)", len(d["pairs"]) == 2 * 10 * 9, f"{len(d['pairs'])}")
fails += not check("estimate present", len(d["estimate"]) == 10)
fails += not check("under 400 ms", ms < 400, f"{ms:.0f} ms")

print("\nteam block")
t = d["team"]
fails += not check("team present", t is not None)
fails += not check("subject is the blue selection", t["subject"] == "blue")
fails += not check("wins <= n", 0 <= t["sel_wins"] <= t["n"], f"{t['sel_wins']}/{t['n']}")
fails += not check("rate consistent", abs(t["sel_win_rate"] - t["sel_wins"] / t["n"]) < 1e-9)
fails += not check("per-side samples present",
                   d["side_stats"]["blue"]["n"] >= d["n"]
                   and d["side_stats"]["red"]["n"] >= d["n"])
fails += not check("objectives carried per game",
                   set(t["obj_diff"]) == {"dragon", "baron", "tower", "inhib",
                                          "herald", "horde"},
                   str(sorted(t["obj_diff"])))

# The objective columns come from `teams[side].obj`, which the old build threw
# away. Winners must out-tower and out-dragon losers by a wide margin, or the
# columns are misaligned with the side they belong to.
wt, _ = get("/api/draft?blue=8,53,64&red=113,126,145&patch=16.20")
wr = wt["team"]["sel_win_rate"]
td = wt["team"]["obj_diff"]["tower"]
fails += not check("objectives agree with the result",
                   abs(wr - 0.5) < 0.03 or (td > 0) == (wr > 0.5),
                   f"towers {td:+.2f} at {wr:.1%} over {wt['team']['n']} games")

# The orientation correction has one exact invariant: a single champion's team
# win rate must be the same number of wins as that champion's own win rate,
# whichever side it held. The denominators can differ by up to the 81 games in
# the corpus that record no result for either team - those are excluded rather
# than guessed at.
one, _ = get("/api/draft?blue=8&patch=16.20")
base, _ = get("/api/matchup?champ=8&partner=64&patch=16.20")
bw = round(base["baseline"]["win"] * base["baseline"]["n"])
fails += not check("single champ team WR == champ WR",
                   one["team"]["sel_wins"] == bw
                   and 0 <= base["baseline"]["n"] - one["team"]["n"] <= 81,
                   f"{one['team']['sel_wins']} of {one['team']['n']} vs "
                   f"{bw} of {base['baseline']['n']}")

print("\neither-side matching")
a, _ = get("/api/draft?blue=8&patch=16.20")
b, _ = get("/api/draft?red=8&patch=16.20")
c, _ = get("/api/draft?blue=8,53,64&patch=16.20")
fails += not check("blue-only == red-only", a["n"] == b["n"], f"{a['n']} vs {b['n']}")
# a lone champion must match every game it appears in, on either side - which
# is exactly its own row count inside the patch
fails += not check("lone champ matches both sides", a["n"] == base["baseline"]["n"],
                   f"n={a['n']} vs {base['baseline']['n']} games played")
fails += not check("3 champs either side", c["n"] == 61, f"n={c['n']}")
both, _ = get("/api/draft?blue=8&red=8&patch=16.20")
fails += not check("same champ on both sides is impossible", both["n"] == 0, f"n={both['n']}")

print("\ndraft, partial build")
d, ms = get("/api/draft?blue=8,53,64&patch=16.20")
fails += not check("3 champs n=61", d["n"] == 61, f"n={d['n']}, {ms:.0f} ms")
fails += not check("champ_stats for 3", len(d["champ_stats"]) == 3)
fails += not check("games capped at 12", len(d["games"]) == 12)
fails += not check("pair rows = 2*n*(n-1)", len(d["pairs"]) == 2 * 3 * 2, f"{len(d['pairs'])}")
fails += not check("pairs carry every core metric",
                   all(k in d["pairs"][0]["deltas"] for k in ("win", "gold", "cs", "vision")))
# whatever the metric selector offers must be answerable by the pair matrix
sel = [x["key"] for x in m["metrics"] if not x.get("hidden")]
fails += not check("pairs carry every selectable metric",
                   all(k in d["pairs"][0]["deltas"] for k in sel),
                   "%d metrics" % len(sel))
fails += not check("champ_stats carry every selectable metric",
                   all(k in d["champ_stats"][0] for k in sel))
fails += not check("champ_stats carry the whole registry",
                   len([k for k in d["champ_stats"][0] if k in sel]) == len(sel),
                   f"{len(sel)} metrics")
fails += not check("no team card without a count",
                   d["team"]["n"] == d["n"])
# the blue/red views need each side's own sample, and an empty side must say so
fails += not check("blue side sample present",
                   d["side_stats"]["blue"]["n"] >= d["n"])
fails += not check("empty side has no sample", d["side_stats"]["red"] is None)

print("\ndraft, empty (random games for the UI button)")
d, ms = get("/api/draft?patch=16.20")
fails += not check("empty returns corpus size", d["n"] == 412953, f"n={d['n']}")
fails += not check("empty returns 12 games", len(d["games"]) == 12)
fails += not check("games have 5+5 champs", all(len(g["blue"]) == 5 and len(g["red"]) == 5
                                               for g in d["games"]))
fails += not check("empty has no team block", d["team"] is None)
fails += not check("empty has no pairs", d["pairs"] == [])

print("\nsuggest (synergies / counters)")
s, ms = get("/api/suggest?side=blue&blue=8,53,64&red=113,126,145&metric=win&patch=16.20")
fails += not check("synergy list", len(s["synergy"]) > 100, f"{len(s['synergy'])}")
fails += not check("counter list", len(s["counter"]) > 100, f"{len(s['counter'])}")
fails += not check("fast", ms < 60, f"{ms:.0f} ms")
excl = set(s["mine"]) | set(s["theirs"])
fails += not check("selected champions excluded",
                   not any(x["riot"] in excl for x in s["synergy"] + s["counter"]))
fails += not check("ranked by the metric",
                   all(s["synergy"][i]["specific"] >= s["synergy"][i + 1]["specific"]
                       for i in range(len(s["synergy"]) - 1)))
fails += not check("delta == specific - general",
                   all(abs(x["delta"] - (x["specific"] - x["general"])) < 1e-9
                       for x in s["synergy"]))
fails += not check("terms <= my champions", all(x["terms"] <= len(s["mine"]) for x in s["synergy"]))
fails += not check("n >= terms", all(x["n_specific"] >= x["terms"] for x in s["synergy"]))

sg, _ = get("/api/suggest?side=blue&blue=8,53,64&red=113,126,145"
            "&metric=goldPerMinute&patch=16.20")
fails += not check("metric switch changes the ranking",
                   sg["synergy"][0]["riot"] != s["synergy"][0]["riot"],
                   f"{sg['synergy'][0]['name']} vs {s['synergy'][0]['name']}")
fails += not check("rate metric has a different unit",
                   sg["synergy"][0]["specific"] > 100, f"{sg['synergy'][0]['specific']:.0f}")

# Every number in a suggest row must describe the CANDIDATE. With one champion
# on my side each row is a single cube cell, so it has to equal that cell
# exactly - the check that catches reading the row off my own champion instead.
one, _ = get("/api/suggest?side=blue&blue=8&metric=goldPerMinute&patch=16.20")
top = one["synergy"][0]
cell, _ = get("/api/matchup?champ=%d&partner=8&mode=with&patch=16.20" % top["riot"])
fails += not check("synergy measures the candidate, not my champion",
                   abs(top["specific"] - cell["stats"]["goldPerMinute"]) < 1e-6,
                   f"{top['name']} {top['specific']:.1f} vs cell "
                   f"{cell['stats']['goldPerMinute']:.1f}")
topc = one["counter"][0]
ccell, _ = get("/api/matchup?champ=%d&partner=8&mode=vs&patch=16.20" % topc["riot"])
fails += not check("counter measures the candidate",
                   abs(topc["specific"] - ccell["stats"]["goldPerMinute"]) < 1e-6,
                   f"{topc['name']} {topc['specific']:.1f} vs cell "
                   f"{ccell['stats']['goldPerMinute']:.1f}")
fails += not check("terms counts my champions, not rows",
                   top["terms"] == 1 and top["n_specific"] == cell["n"],
                   f"terms={top['terms']} n={top['n_specific']} cell n={cell['n']}")

sh, _ = get("/api/suggest?side=blue&blue=8,53,64&metric=not_a_stat&patch=16.20")
fails += not check("unknown metric falls back to win rate", sh["metric"] == "win", sh["metric"])
# the 0/1 laning flag is a legitimate metric now, just labelled as a flag
lf, _ = get("/api/suggest?side=blue&blue=8,53,64"
            "&metric=laningPhaseGoldExpAdvantage&patch=16.20")
fails += not check("0/1 flag is rankable",
                   lf["metric"] == "laningPhaseGoldExpAdvantage"
                   and 0 <= lf["synergy"][0]["specific"] <= 1,
                   f"{lf['synergy'][0]['name']} {lf['synergy'][0]['specific']:.3f}")

# Across patches each (champ, partner) pair appears twice, so "how many of my
# champions have a measured cell" must count distinct champions, not rows.
sa, _ = get("/api/suggest?side=blue&blue=8,53,64&metric=win&patch=")
fails += not check("all-patch terms still <= my champions",
                   all(x["terms"] <= 3 for x in sa["synergy"] + sa["counter"]),
                   f"max {max(x['terms'] for x in sa['synergy'])}")
s20, _ = get("/api/suggest?side=blue&blue=8,53,64&metric=win&patch=16.20")
s19, _ = get("/api/suggest?side=blue&blue=8,53,64&metric=win&patch=16.19")
g20 = {x["riot"]: x["general"] for x in s20["synergy"]}
g19 = {x["riot"]: x["general"] for x in s19["synergy"]}
inside = [x for x in sa["synergy"] if x["riot"] in g19 and x["riot"] in g20]
fails += not check("all-patch general sits between the patches",
                   all(min(g19[x["riot"]], g20[x["riot"]]) - 1e-9 <= x["general"]
                       <= max(g19[x["riot"]], g20[x["riot"]]) + 1e-9 for x in inside),
                   f"{len(inside)} candidates compared")

sr, _ = get("/api/suggest?side=red&blue=8&red=113,126,145&metric=win&patch=16.20")
fails += not check("red side works", sr["side"] == "red" and len(sr["synergy"]) > 100)
fails += not check("empty side returns nothing",
                   get("/api/suggest?side=red&blue=8&patch=16.20")[0]["synergy"] == [])

print("\nchampion detail")
c, ms = get("/api/champion?champ=8&blue=8,53,64&patch=16.20")
fails += not check("in-games n matches", c["in_games"]["n"] == 61, f"n={c['in_games']['n']}")
fails += not check("baseline is the whole patch", c["baseline"]["n"] > 24000, f"{c['baseline']['n']}")
fails += not check("delta consistent",
                   abs(c["delta"]["win"] - (c["in_games"]["win"] - c["baseline"]["win"])) < 1e-9)
fails += not check("partial stats report their own n",
                   "n_earliestBaron" in c["in_games"]
                   and c["in_games"]["n_earliestBaron"] <= c["in_games"]["n"],
                   f"{c['in_games'].get('n_earliestBaron')} of {c['in_games']['n']}")
fails += not check("full-line stats need no n",
                   "n_gold" not in c["in_games"] and "n_soloKills" not in c["in_games"])
fails += not check("detail covers the whole registry",
                   all(k in c["in_games"] for k in sel), f"{len(sel)} metrics")
fails += not check("under 200 ms", ms < 200, f"{ms:.0f} ms")

print("\nall patches / bad input")
d, _ = get("/api/draft?blue=8&patch=")
fails += not check("patch='' means all", d["n"] > 24000, f"n={d['n']}")
try:
    d, _ = get("/api/draft?blue=99999&patch=16.20")
    fails += not check("unknown champ tolerated", d["n"] == 0, f"n={d['n']}")
except Exception as exc:
    fails += not check("unknown champ tolerated", False, str(exc))

# The cubes are keyed by patch. Asking for every patch has to union the cells
# with n-weights; a plain mean (or just taking one row) would let the 60-game
# 16.19 cell weigh the same as the 1813-game 16.20 one.
p20, _ = get("/api/matchup?champ=8&partner=64&patch=16.20")
p19, _ = get("/api/matchup?champ=8&partner=64&patch=16.19")
ap, _ = get("/api/matchup?champ=8&partner=64&patch=")
exp = ((p20["n"] * p20["stats"]["win"] + p19["n"] * p19["stats"]["win"])
       / (p20["n"] + p19["n"]))
fails += not check("all-patch n is the sum of the cells",
                   ap["n"] == p20["n"] + p19["n"], f"{ap['n']} vs {p20['n'] + p19['n']}")
fails += not check("all-patch rate is n-weighted",
                   abs(ap["stats"]["win"] - exp) < 1e-9,
                   f"{ap['stats']['win']:.6f} vs {exp:.6f}")
fails += not check("all-patch baseline is the sum too",
                   ap["baseline"]["n"] == p20["baseline"]["n"] + p19["baseline"]["n"],
                   f"{ap['baseline']['n']}")

print("\n%s (%d failures)" % ("ALL CHECKS PASSED" if not fails else "FAILURES", fails))
