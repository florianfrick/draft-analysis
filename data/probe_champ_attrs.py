"""Audit ``data/champ_attrs.json``: nearest neighbours + external checks.

Two kinds of check, because "does this table look right" has two different
answers:

1. **Internal coherence** - nearest neighbours in attribute space and the
   top-5 of each axis. A mis-tagged row shows up as a champion sitting next
   to champions it has nothing in common with. This is the check a human can
   do by eye, and it is what the table is shown to the user for.

2. **External validation** - the tags predict something measured in the pro
   data that they were never fit to:
     * ``scaling`` -> win rate in long games minus win rate in short games;
     * ``lane_priority`` -> mean gold difference at 15 minutes.
   Neither number went into the tagging, so a positive rank correlation is
   evidence the axis is real rather than a restatement of my priors. A null
   here would mean the axis is decorative and should be dropped before it is
   wired into an embedding.

Run: ``python data/probe_champ_attrs.py``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ATTRS_PATH = BASE_DIR / "champ_attrs.json"
PRO_DIR = BASE_DIR / "pro play"
CSVS = sorted(PRO_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))

SHORT_GAME_S = 28 * 60
LONG_GAME_S = 34 * 60
MIN_GAMES = 150          # per bucket, per champion


def load_attrs():
    payload = json.loads(ATTRS_PATH.read_text(encoding="utf-8"))
    axes = payload["axes"]
    rows = {}
    for entry in payload["champions"].values():
        if entry["attrs"] is not None:
            rows[entry["name"]] = entry["attrs"]
    return payload, axes, rows


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest(rows, name, k=3):
    me = rows[name]
    others = sorted(((dist(me, v), n) for n, v in rows.items() if n != name))
    return others[:k]


def axis_tops(rows, axes, k=5):
    out = {}
    for j, axis in enumerate(axes):
        ranked = sorted(rows.items(), key=lambda kv: (-kv[1][j], kv[0]))
        out[axis] = [n for n, _ in ranked[:k]]
    return out


def read_pro():
    """-> per-champion stats from the Oracle's Elixir player rows.

    Gold at 15 is accumulated per (champion, position) as well, because it is
    strongly role-confounded: a support is gold-negative at 15 by definition,
    so a raw champion mean mostly measures "is this champion played support".
    The position-adjusted mean subtracts the position's own mean first.
    """
    stats = defaultdict(lambda: defaultdict(float))
    pos_gold = defaultdict(lambda: defaultdict(float))   # pos -> {sum, n}
    seen_rows = 0
    for path in CSVS:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for rec in csv.DictReader(f):
                champ = (rec.get("champion") or "").strip()
                if not champ:
                    continue
                try:
                    gl = float(rec["gamelength"])
                    win = int(float(rec["result"]))
                except (TypeError, ValueError):
                    continue
                seen_rows += 1
                s = stats[champ]
                s["games"] += 1
                s["wins"] += win
                if gl < SHORT_GAME_S:
                    s["short_games"] += 1
                    s["short_wins"] += win
                elif gl > LONG_GAME_S:
                    s["long_games"] += 1
                    s["long_wins"] += win
                try:
                    gold = float(rec["golddiffat15"])
                except (TypeError, ValueError):
                    gold = None
                pos = (rec.get("position") or "").strip().lower() or "?"
                if gold is not None:
                    s["gold15_sum"] += gold
                    s["gold15_n"] += 1
                    pos_gold[pos]["sum"] += gold
                    pos_gold[pos]["n"] += 1
                    s["gold15_pos_%s_sum" % pos] += gold
                    s["gold15_pos_%s_n" % pos] += 1
                try:
                    dmg = float(rec["damagetochampions"])
                except (TypeError, ValueError):
                    continue
                if gl > 0:
                    s["dmg_sum"] += dmg
                    s["min_sum"] += gl / 60.0
                    s["dmg_n"] += 1
    pos_mean = {p: v["sum"] / v["n"] for p, v in pos_gold.items() if v["n"]}
    return stats, seen_rows, pos_mean


def adjusted_gold15(stats, pos_mean):
    """Per-champion mean gold@15 with each position's mean removed."""
    out = {}
    for champ, s in stats.items():
        num = den = 0.0
        for pos, mean in pos_mean.items():
            n = s.get("gold15_pos_%s_n" % pos, 0)
            if not n:
                continue
            num += s["gold15_pos_%s_sum" % pos] - mean * n
            den += n
        if den:
            out[champ] = num / den
    return out



def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def query_nearest(args, axes, rows):
    """Compare an externally supplied vector against the table.

    Used to sanity-check a proposed row for a champion I cannot tag myself
    (or one whose kit the reviewer knows better): if the closest existing
    champions are the ones you would expect, the row is consistent with the
    rest of the table.
    """
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    pool = {n: v for n, v in rows.items() if n not in exclude}
    k = len(axes)
    for spec in args.nearest:
        label, _, body = spec.partition(":")
        vec = [int(x) for x in body.replace(" ", "").split(",") if x != ""]
        if len(vec) != k:
            raise SystemExit("%s: expected %d values, got %d"
                             % (label or "vector", k, len(vec)))
        if any(not 0 <= v <= 3 for v in vec):
            raise SystemExit("%s: values must be 0..3" % label)
        ranked = sorted((dist(vec, v), n) for n, v in pool.items())[:args.k]
        print("%s  %s" % (label or "vector", " ".join(str(x) for x in vec)))
        for rank, (d, name) in enumerate(ranked):
            print("  %2d. %-12s %.2f" % (rank + 1, name, d))
            if rank < 3:
                diff = ["%s %d/%d" % (axes[i], vec[i], pool[name][i])
                        for i in range(k) if vec[i] != pool[name][i]]
                print("      differs on: %s" % (", ".join(diff) or "nothing"))
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="audit data/champ_attrs.json",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nearest", action="append", default=[],
                    metavar="LABEL:v0,v1,...",
                    help="query a vector against the table and exit "
                         "(repeatable; 16 values in axis order)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated champions to drop from the pool")
    ap.add_argument("-k", type=int, default=5, help="how many neighbours")
    args = ap.parse_args(argv)

    payload, axes, rows = load_attrs()
    if args.nearest:
        print("== %d axes: %s\n" % (len(axes), ", ".join(axes)))
        return query_nearest(args, axes, rows)

    print("== %d tagged champions, %d axes, patch %s"
          % (len(rows), len(axes), payload.get("patch")))
    print("unknown: %s\n" % ", ".join(payload.get("unknown") or []))

    print("== nearest neighbours (euclidean on the 0..3 vector)")
    for name in ["Malphite", "Janna", "Xerath", "Zed", "Ornn", "Sivir",
                 "Nautilus", "Yuumi", "Kassadin", "Thresh"]:
        if name not in rows:
            continue
        nn = nearest(rows, name, 3)
        print("%-12s %s" % (name, "  ".join("%s(%.1f)" % (n, d) for d, n in nn)))

    print("\n== top of each axis")
    for axis, names in axis_tops(rows, axes).items():
        print("%-15s %s" % (axis, ", ".join(names)))

    print("\n== per-axis level coverage (0/1/2/3)")
    for j, axis in enumerate(axes):
        counts = [sum(1 for v in rows.values() if v[j] == lvl) for lvl in range(4)]
        flag = "  <- degenerate" if max(counts) > 0.85 * len(rows) else ""
        print("%-15s %s%s" % (axis, counts, flag))

    print("\n== identical rows (the axes cannot tell these apart)")
    by_vec = defaultdict(list)
    for name, vec in rows.items():
        by_vec[tuple(vec)].append(name)
    dupes = sorted((sorted(v) for v in by_vec.values() if len(v) > 1),
                   key=lambda g: -len(g))
    for group in dupes:
        print("  " + " = ".join(group))
    print("  %d champions collapse into %d groups"
          % (sum(len(g) for g in dupes), len(dupes)))

    print("\n== external check against pro data")
    stats, n_rows, pos_mean = read_pro()
    print("read %d player rows from %d CSVs" % (n_rows, len(CSVS)))
    print("position means for gold@15: %s"
          % ", ".join("%s %+.0f" % (p, v) for p, v in sorted(pos_mean.items())))

    # --- lane priority: position-adjusted gold diff at 15
    adj = adjusted_gold15(stats, pos_mean)
    lp_x, lp_y, lp_n = [], [], 0
    for name, g in adj.items():
        if name not in rows or stats[name]["gold15_n"] < 3 * MIN_GAMES:
            continue
        lp_x.append(rows[name][axes.index("lane_priority")])
        lp_y.append(g)
        lp_n += 1
    lp_rho = spearman(lp_x, lp_y)
    print("lane_priority vs gold@15 (position-adjusted): rho=%+.3f  n=%d  "
          "(SE %.3f, so %+.1f sigma)"
          % (lp_rho, lp_n, 1 / math.sqrt(lp_n - 1), lp_rho * math.sqrt(lp_n - 1)))

    # control: axes with no business predicting gold@15
    for ctl in ("tankiness", "healing"):
        cx, cy = [], []
        for name, g in adj.items():
            if name not in rows or stats[name]["gold15_n"] < 3 * MIN_GAMES:
                continue
            cx.append(rows[name][axes.index(ctl)])
            cy.append(g)
        print("%-14s vs gold@15 (control):            rho=%+.3f"
              % (ctl, spearman(cx, cy)))

    # --- damage output: damage to champions per minute.
    # Role-confounded by design - the axis is "damage in this champion's role",
    # so a support scoring low is the axis working, not a defect.
    dmg_rows = [(name, s["dmg_sum"] / s["min_sum"]) for name, s in stats.items()
                if name in rows and s["dmg_n"] >= 3 * MIN_GAMES
                and s["min_sum"] > 0]
    for axis in ("damage_output", "magic_share"):
        xs = [rows[n][axes.index(axis)] for n, _ in dmg_rows]
        ys = [v for _, v in dmg_rows]
        rho = spearman(xs, ys)
        print("%-14s vs damage/min:                rho=%+.3f  n=%d  (%+.1f sigma)"
              % (axis, rho, len(xs), rho * math.sqrt(len(xs) - 1)))

    # --- scaling: win rate in long games minus short games.
    # This one is confounded and the report says so: the set of games that
    # reach 34 minutes is not a random sample, it is the games where the
    # champion failed to end early, so a strong-early champion is pushed
    # negative regardless of its actual late-game power.
    sc_x, sc_y, sc_n, sc_names = [], [], 0, []
    for name, s in stats.items():
        if name not in rows:
            continue
        if s["short_games"] < MIN_GAMES or s["long_games"] < MIN_GAMES:
            continue
        delta = s["long_wins"] / s["long_games"] - s["short_wins"] / s["short_games"]
        sc_x.append(rows[name][axes.index("scaling")])
        sc_y.append(delta)
        sc_names.append(name)
        sc_n += 1
    sc_rho = spearman(sc_x, sc_y)
    print("scaling vs (win%% long - win%% short):        rho=%+.3f  n=%d  "
          "(SE %.3f, so %+.1f sigma)"
          % (sc_rho, sc_n, 1 / math.sqrt(sc_n - 1), sc_rho * math.sqrt(sc_n - 1)))

    # show the extremes so the number is interpretable
    print("\nmost late-game-tilted champions in pro (long - short win%):")
    ranked = []
    for name, s in stats.items():
        if name not in rows:
            continue
        if s["short_games"] < MIN_GAMES or s["long_games"] < MIN_GAMES:
            continue
        delta = s["long_wins"] / s["long_games"] - s["short_wins"] / s["short_games"]
        ranked.append((delta, name, rows[name][axes.index("scaling")]))
    ranked.sort(reverse=True)
    for delta, name, tag in ranked[:8]:
        print("  %+.1fpp  %-14s scaling=%d" % (100 * delta, name, tag))
    print("  ...")
    for delta, name, tag in ranked[-8:]:
        print("  %+.1fpp  %-14s scaling=%d" % (100 * delta, name, tag))

    # champions where the data and the tag disagree hardest - the useful part
    print("\nbiggest disagreements (tag says late, data says early, or back):")
    disagreements = []
    for delta, name, tag in ranked:
        # expected sign from the tag: scaling 3 should be positive
        pred = (tag - 1.5)
        if pred * delta < 0 and abs(delta) > 0.06:
            disagreements.append((abs(delta), name, tag, delta))
    disagreements.sort(reverse=True)
    for _, name, tag, delta in disagreements[:10]:
        print("  %+.1fpp  %-14s scaling=%d" % (100 * delta, name, tag))


if __name__ == "__main__":
    main()
