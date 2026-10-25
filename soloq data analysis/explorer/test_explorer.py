"""End-to-end checks against a running explorer server."""
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
fails += not check("corpus size", m["games"] == 415000, f"{m['games']} games, {ms:.0f} ms")
fails += not check("champions", len(m["champions"]) == 173, f"{len(m['champions'])}")
fails += not check("icons resolved", all(c["img"].startswith("http") for c in m["champions"]))
fails += not check("patches", len(m["patches"]) == 2, str(m["patches"]))

print("\nmatchup (capability 1, precomputed)")
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
fails += not check("under 250 ms", ms < 250, f"{ms:.0f} ms")

print("\ndraft, partial build")
d, ms = get("/api/draft?blue=8,53,64&patch=16.20")
fails += not check("3 champs n=32", d["n"] == 32, f"n={d['n']}, {ms:.0f} ms")
fails += not check("champ_stats for 3", len(d["champ_stats"]) == 3)
fails += not check("games capped at 12", len(d["games"]) == 12)
fails += not check("pair rows = 2*n*(n-1)", len(d["pairs"]) == 2 * 3 * 2, f"{len(d['pairs'])}")

print("\ndraft, sides")
a, _ = get("/api/draft?blue=8,53,64,84,523&red=113,126,145,518,711&patch=16.20&sides=any")
b, _ = get("/api/draft?blue=8,53,64,84,523&red=113,126,145,518,711&patch=16.20&sides=fixed")
fails += not check("any >= fixed", a["n"] >= b["n"], f"any={a['n']} fixed={b['n']}")

print("\ndraft, empty (random games for the UI button)")
d, ms = get("/api/draft?patch=16.20")
fails += not check("empty returns corpus size", d["n"] == 404459, f"n={d['n']}")
fails += not check("empty returns 12 games", len(d["games"]) == 12)
fails += not check("games have 5+5 champs", all(len(g["blue"]) == 5 and len(g["red"]) == 5
                                               for g in d["games"]))

print("\nall patches")
d, _ = get("/api/draft?blue=8&patch=")
fails += not check("patch='' means all", d["n"] > 12000, f"n={d['n']}")

print("\nno-store / bad input handling")
try:
    d, _ = get("/api/draft?blue=99999&patch=16.20")
    fails += not check("unknown champ tolerated", d["n"] == 0, f"n={d['n']}")
except Exception as exc:
    fails += not check("unknown champ tolerated", False, str(exc))

print("\n%s (%d failures)" % ("ALL CHECKS PASSED" if not fails else "FAILURES", fails))
