"""Empirical stats explorer over the soloQ store (DuckDB, read-only).

Answers the questions the draft UI needs, from measured data only:

  matchup   avg stats of champ X when paired with / against champ Y   (2-way)
  draft     given the picks built so far, how many games match, how the two
            teams fared, and what each champion actually did             (N-way)
  suggest   which champions would best complete / counter a team        (2-way)
  champion  one champion's full stat line inside the matching games     (N-way)

Design notes
------------
- 2-way questions are served from the precomputed `cube2` table (~1 ms).
- The N-way question is a LIVE SCAN of `drafts` (415k rows, one per game),
  not a precomputed cube. A 5-champion cube would be 240M cells / 1.1 GB and
  is 99.99% empty (the densest 5-champ cell in 415k games has 21 games), so
  there is nothing to precompute.
- Containment is tested with bitmasks, not `list_has_all`: three integer ANDs
  per row instead of rebuilding a list per row. Measured 40 ms -> 3 ms.
- `sides=any` (the default and the only mode the UI uses) treats a draft as an
  unordered pair of teams: the selection may have been blue or red in the real
  game. Counts are therefore orientation-free, and any per-side statistic is
  computed relative to the *selection*, not to the literal blue side.
- Every metric lives in METRICS. Adding a stat means adding one entry there
  plus the column in store.py - the cube, the API and the UI all read it.

Run:  python explorer/server.py --port 8778
Open: http://127.0.0.1:8778
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
STORE_DIR = HERE.parent / "store"
STORE_DB = STORE_DIR / "store.duckdb"
PROC = PROJ / "data" / "solo queue" / "process matchdata"
STATIC = HERE / "static"
ICON_CACHE = PROJ / "frontend" / ".cache" / "champion_16.20.1.json"
CDN = "https://ddragon.leagueoflegends.com/cdn/"
VER = "16.20.1"

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
# team-level objective columns carried by `drafts`, as (key, blue_col, red_col)
TEAM_OBJ = [("dragon", "blue_dragon", "red_dragon"),
            ("baron", "blue_baron", "red_baron"),
            ("tower", "blue_tower", "red_tower"),
            ("inhib", "blue_inhib", "red_inhib"),
            ("herald", "blue_herald", "red_herald"),
            ("horde", "blue_horde", "red_horde")]

# ---------------------------------------------------------------- metric registry
# Every per-player field the collector writes is registered here exactly once:
# the eight box-score columns, the four first-blood/first-tower flags and all 32
# challenge keys. The registry is the only place that knows about a stat - the
# cubes aggregate whatever it lists, the matrix/champion table/detail panel all
# iterate it, and /api/suggest ranks by it - so adding a field to the collector
# plus one entry here is enough to make it selectable everywhere.
#
# better: +1 = higher is better, -1 = lower is better. The UI colours deltas and
#         sorts by it, so a metric only has to be declared once.
# fmt:    pct | int | num1 | num2 | sec (seconds shown as m:ss)
# cov:    filled in at startup from the store (share of rows that carry the
#         field). A metric below 99% is flagged `partial` and the panel shows
#         the real n instead of pretending the average covers every game.
#
# Notes from profiling the corpus: `laningPhaseGoldExpAdvantage` and
# `earlyLaningPhaseGoldExpAdvantage` are 0/1 flags in this data, not lead
# magnitudes, so they are registered as rates and labelled as flags. The
# `earliestBaron` / `firstTurretKilledTime` keys are timestamps in seconds and
# lower is better. Everything else is a count, a rate or a lead.
METRICS = [
    # --- box score (present on every row, archive included)
    {"key": "win", "label": "win rate", "better": 1, "fmt": "pct", "group": "core"},
    {"key": "kills", "label": "kills", "better": 1, "fmt": "num2", "group": "core"},
    {"key": "deaths", "label": "deaths", "better": -1, "fmt": "num2", "group": "core"},
    {"key": "assists", "label": "assists", "better": 1, "fmt": "num2", "group": "core"},
    {"key": "gold", "label": "gold", "better": 1, "fmt": "int", "group": "core"},
    {"key": "dmg", "label": "damage to champions", "better": 1, "fmt": "int", "group": "core"},
    {"key": "cs", "label": "CS", "better": 1, "fmt": "num1", "group": "core"},
    {"key": "vision", "label": "vision score", "better": 1, "fmt": "num1", "group": "core"},
    # --- per-minute rates
    {"key": "goldPerMinute", "label": "gold / min", "better": 1, "fmt": "num1", "group": "rate"},
    {"key": "damagePerMinute", "label": "damage / min", "better": 1, "fmt": "num1", "group": "rate"},
    {"key": "visionScorePerMinute", "label": "vision / min", "better": 1, "fmt": "num2", "group": "rate"},
    # --- lane phase
    {"key": "maxCsAdvantageOnLaneOpponent", "label": "peak CS lead vs laner",
     "better": 1, "fmt": "num1", "group": "lane"},
    {"key": "maxLevelLeadLaneOpponent", "label": "peak level lead vs laner",
     "better": 1, "fmt": "num1", "group": "lane"},
    {"key": "visionScoreAdvantageLaneOpponent", "label": "vision lead vs laner",
     "better": 1, "fmt": "num1", "group": "lane"},
    {"key": "laningPhaseGoldExpAdvantage", "label": "laning gold/XP lead (0/1 flag)",
     "better": 1, "fmt": "pct", "group": "lane"},
    {"key": "earlyLaningPhaseGoldExpAdvantage", "label": "early laning lead (0/1 flag)",
     "better": 1, "fmt": "pct", "group": "lane"},
    {"key": "laneMinionsFirst10Minutes", "label": "lane minions @10",
     "better": 1, "fmt": "num1", "group": "lane"},
    {"key": "jungleCsBefore10Minutes", "label": "jungle camps @10",
     "better": 1, "fmt": "num1", "group": "lane"},
    # --- tempo and firsts
    {"key": "takedownsFirstXMinutes", "label": "takedowns @10",
     "better": 1, "fmt": "num1", "group": "tempo"},
    {"key": "earliestBaron", "label": "first baron (time)",
     "better": -1, "fmt": "sec", "group": "tempo"},
    {"key": "firstTurretKilledTime", "label": "first turret (time)",
     "better": -1, "fmt": "sec", "group": "tempo"},
    {"key": "fb_kill", "label": "first blood (kill)", "better": 1, "fmt": "pct",
     "group": "tempo"},
    {"key": "fb_assist", "label": "first blood (assist)", "better": 1, "fmt": "pct",
     "group": "tempo"},
    {"key": "ft_kill", "label": "first turret (kill)", "better": 1, "fmt": "pct",
     "group": "tempo"},
    {"key": "ft_assist", "label": "first turret (assist)", "better": 1, "fmt": "pct",
     "group": "tempo"},
    # --- team and objectives
    {"key": "killParticipation", "label": "kill participation",
     "better": 1, "fmt": "pct", "group": "team"},
    {"key": "teamDamagePercentage", "label": "share of team damage",
     "better": 1, "fmt": "pct", "group": "team"},
    {"key": "damageTakenOnTeamPercentage", "label": "share of damage taken",
     "better": 1, "fmt": "pct", "group": "team"},
    {"key": "turretPlatesTaken", "label": "plates taken",
     "better": 1, "fmt": "num1", "group": "team"},
    {"key": "turretTakedowns", "label": "turrets",
     "better": 1, "fmt": "num1", "group": "team"},
    {"key": "dragonTakedowns", "label": "dragons",
     "better": 1, "fmt": "num1", "group": "team"},
    {"key": "baronTakedowns", "label": "barons",
     "better": 1, "fmt": "num1", "group": "team"},
    {"key": "riftHeraldTakedowns", "label": "heralds",
     "better": 1, "fmt": "num1", "group": "team"},
    {"key": "teamBaronKills", "label": "team barons",
     "better": 1, "fmt": "num1", "group": "team"},
    {"key": "teamRiftHeraldKills", "label": "team heralds",
     "better": 1, "fmt": "num1", "group": "team"},
    {"key": "teamElderDragonKills", "label": "team elder dragons",
     "better": 1, "fmt": "num1", "group": "team"},
    # --- combat profile
    {"key": "soloKills", "label": "solo kills", "better": 1, "fmt": "num1",
     "group": "combat"},
    {"key": "deathsByEnemyChamps", "label": "deaths to champions", "better": -1,
     "fmt": "num1", "group": "combat"},
    {"key": "enemyChampionImmobilizations", "label": "CC applied (s)",
     "better": 1, "fmt": "num1", "group": "combat"},
    {"key": "effectiveHealAndShielding", "label": "healing + shielding",
     "better": 1, "fmt": "int", "group": "combat"},
    {"key": "killsNearEnemyTurret", "label": "kills near enemy turret",
     "better": 1, "fmt": "num1", "group": "combat"},
    {"key": "maxKillDeficit", "label": "worst kill deficit", "better": -1,
     "fmt": "num1", "group": "combat"},
    {"key": "bountyGold", "label": "bounty gold", "better": 1, "fmt": "int",
     "group": "combat"},
    {"key": "outnumberedKills", "label": "outnumbered kills", "better": 1,
     "fmt": "num1", "group": "combat"},
]
METRIC_BY_KEY = {m["key"]: m for m in METRICS}
# every registered metric, in registry order. The cubes carry all of them, so the
# pair matrix, the per-champion table and the matchup card all iterate this.
STATS = [m["key"] for m in METRICS]
DETAIL = STATS
# filled by probe_coverage() at startup; a metric below FULL_COVER is `partial`
PARTIAL = []
FULL_COVER = 0.99

_lock = threading.Lock()
APP = {}


# ---------------------------------------------------------------- champion data

def load_champions(con):
    names, idx, icons = {}, {}, {}
    p = PROC / "champ_names.json"
    if p.is_file():
        names = {int(k): v for k, v in json.loads(p.read_text("utf-8"))["names"].items()}
    p = PROC / "champ_mapping.json"
    if p.is_file():
        idx = {int(k): int(v) for k, v in
               json.loads(p.read_text("utf-8"))["riot_to_idx"].items()}
    if ICON_CACHE.is_file():
        try:
            data = json.loads(ICON_CACHE.read_text("utf-8"))
            for c in data["data"].values():
                icons[int(c["key"])] = "%s%s/img/champion/%s" % (
                    CDN, VER, c["image"]["full"])
        except (KeyError, ValueError, TypeError):
            icons = {}
    games = dict(con.execute(
        "SELECT champ, count(*) FROM wide GROUP BY 1").fetchall())
    champs = []
    for riot, name in names.items():
        champs.append({
            "riot": riot, "idx": idx.get(riot), "name": name,
            "img": icons.get(riot) or "%s%s/img/champion/%s.png" % (
                CDN, VER, "".join(ch for ch in name if ch.isalpha())),
            "games": int(games.get(riot, 0)),
        })
    champs.sort(key=lambda c: -c["games"])
    return champs


def mask_of(riots):
    """Champion set -> 3 x 64-bit words over the dense 0..172 index space."""
    m = [0, 0, 0]
    idx = APP["idx"]
    for r in riots:
        i = idx.get(int(r))
        if i is None:
            continue
        m[i // 64] |= 1 << (i % 64)
    return m


def _contains(side, riots):
    m = mask_of(riots)
    sql = " AND ".join("(%s_m%d & ?) = ?" % (side, w) for w in range(3))
    return sql, [m[0], m[0], m[1], m[1], m[2], m[2]]


def draft_filter(blue, red, patch=None, sides="any"):
    """WHERE clause over `drafts` for 'blue contains X and red contains Y'.

    sides='any' treats the draft as an unordered pair of teams: the selection
    may have been blue or red in the real game. That also applies when only one
    team is filled in - a lone champion set matches whichever side held it.
    """
    parts, params = [], []
    if blue or red:
        if sides == "any" and blue and red:
            b1, p1 = _contains("blue", blue)
            r1, p2 = _contains("red", red)
            b2, p3 = _contains("blue", red)
            r2, p4 = _contains("red", blue)
            parts.append("((%s AND %s) OR (%s AND %s))" % (b1, r1, b2, r2))
            params += p1 + p2 + p3 + p4
        elif sides == "any":
            only = blue or red
            b1, p1 = _contains("blue", only)
            r1, p2 = _contains("red", only)
            parts.append("(%s OR %s)" % (b1, r1))
            params += p1 + p2
        else:
            if blue:
                s, p = _contains("blue", blue)
                parts.append(s)
                params += p
            if red:
                s, p = _contains("red", red)
                parts.append(s)
                params += p
    if patch:
        parts.append("patch = ?")
        params.append(patch)
    return (" AND ".join(parts) if parts else "TRUE"), params


def count_drafts(con, blue, red, patch, sides):
    where, params = draft_filter(blue, red, patch, sides)
    return con.execute("SELECT count(*) FROM drafts WHERE " + where, params).fetchone()[0]


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------- endpoints

def api_meta(con):
    patches = [{"patch": p, "games": int(g)} for p, g in con.execute(
        "SELECT patch, count(*) FROM drafts GROUP BY 1 ORDER BY 2 DESC").fetchall()]
    queues = [{"queue_id": int(q), "games": int(g)} for q, g in con.execute(
        "SELECT queue_id, count(*) FROM drafts GROUP BY 1 ORDER BY 2 DESC").fetchall()]
    return {
        "games": int(con.execute("SELECT count(*) FROM drafts").fetchone()[0]),
        "players": int(con.execute("SELECT count(*) FROM wide").fetchone()[0]),
        "patches": patches,
        "queues": queues,
        "roles": ROLES,
        "metrics": METRICS,
        "champions": APP["champions"],
        "thresholds": {"strong": 200, "ok": 50, "weak": 20},
    }


def _statrow(row, cols):
    return {c: (float(v) if isinstance(v, float) else v)
            for c, v in zip(cols, row) if v is not None}


def cube_cell(con, table, keys, cols, patch=None):
    """One cube cell, or the patch-weighted union of them when no patch is pinned.

    The cubes are keyed by patch so that two metas are never averaged together
    by accident. When the user does ask for "all patches" the cells still have
    to be combined, and the only correct way to do that with per-cell averages
    is to weight by the cell's own n - a plain mean would let a 3-game cell
    outvote a 900-game one. `nullif(...)` keeps a stat whose cells are all NULL
    reporting NULL instead of 0.
    """
    where = " AND ".join("%s = ?" % k for k in keys)
    params = list(keys.values())
    if patch:
        row = con.execute(
            "SELECT n, %s FROM %s WHERE %s AND patch = ?"
            % (", ".join(cols), table, where), params + [patch]).fetchone()
    else:
        aggs = ["sum(n) AS n"] + [
            "sum(%s * n) / nullif(sum(CASE WHEN %s IS NOT NULL THEN n END), 0) AS %s"
            % (c, c, c) for c in cols]
        row = con.execute(
            "SELECT %s FROM %s WHERE %s" % (", ".join(aggs), table, where),
            params).fetchone()
    return row


def api_matchup(con, champ, partner, mode, patch, role):
    """2-way: served from the precomputed cube2 (no scan)."""
    kind = "vs" if mode == "vs" else "with"
    cols = ["n"] + STATS
    row = cube_cell(con, "cube2", {"champ": champ, "partner": partner,
                                   "kind": kind}, STATS, patch)
    out = {"mode": kind, "n": 0, "stats": None}
    brow = cube_cell(con, "champ_base", {"champ": champ}, STATS, patch)

    if row:
        out["n"] = int(row[0])
        out["stats"] = _statrow(row, cols)
    if brow:
        out["baseline"] = _statrow(brow, cols)
        if out["stats"]:
            out["delta"] = {k: out["stats"][k] - out["baseline"][k]
                            for k in STATS if out["stats"].get(k) is not None
                            and out["baseline"].get(k) is not None}
    if role:
        rrow = cube_cell(con, "champ_pos_base", {"champ": champ, "pos": role},
                         ["win", "gold"], patch)
        if rrow:
            out["role"] = {"pos": role, "n": int(rrow[0]),
                           "win": float(rrow[1]) if rrow[1] is not None else None,
                           "gold": float(rrow[2]) if rrow[2] is not None else None}
    return out


def presence_histogram(con, selected, patch):
    """One scan -> [(presence_code, games)] for the selected champions.

    Bit 2j of the code = "champion j is on the blue side of this game",
    bit 2j+1 = "on the red side". Every question the UI asks (the exact
    count, the cumulative ladder, leave-one-out) is then arithmetic over at
    most a few hundred histogram buckets instead of another scan.

    Measured over 415k drafts with 10 champions selected: 20 separate scans
    185 ms, 20 count(*) FILTER in one scan 342 ms, this histogram 33 ms.
    """
    parts, params = [], []
    for j, (_, riot) in enumerate(selected):
        i = APP["idx"].get(riot)
        if i is None:
            parts.append("0")
            parts.append("0")
            continue
        bit = 1 << (i % 64)
        word = i // 64
        parts.append("CASE WHEN (blue_m%d & ?) <> 0 THEN %d ELSE 0 END"
                     % (word, 1 << (2 * j)))
        parts.append("CASE WHEN (red_m%d & ?) <> 0 THEN %d ELSE 0 END"
                     % (word, 1 << (2 * j + 1)))
        params += [bit, bit]
    if not parts:
        return []
    where = "patch = ?" if patch else "TRUE"
    if patch:
        params.append(patch)
    rows = con.execute(
        "SELECT code, count(*) FROM (SELECT %s AS code FROM drafts WHERE %s) "
        "GROUP BY 1" % (" + ".join(parts), where), params).fetchall()
    return [(int(c), int(n)) for c, n in rows]


def _holds(code, blue_pos, red_pos):
    return (all((code >> (2 * j)) & 1 for j in blue_pos) and
            all((code >> (2 * j + 1)) & 1 for j in red_pos))


def hist_count(hist, blue_pos, red_pos, sides):
    """Games whose blue side holds every blue_pos champion and red every red_pos.

    With sides='any' the same set is also accepted on the opposite side, which
    is what makes a half-built draft count every game that contained those
    champions rather than only the ones drawn the way the UI shows them.
    """
    total = 0
    for code, n in hist:
        ok = _holds(code, blue_pos, red_pos)
        if not ok and sides == "any":
            ok = _holds(code, red_pos, blue_pos)
        if ok:
            total += n
    return total


def team_block(con, blue, red, patch):
    """Win rate *between the two selections*, orientation-corrected.

    With sides=any, the blue selection was on the red side in roughly half the
    matching games, so 'blue won' is not the same as 'our team won'. The
    subject is the blue selection (or red if blue is empty) and its win is
    read from whichever side it actually held.
    """
    subject = blue or red
    if not subject:
        return None
    other = red if blue else []
    s_sql, s_p = _contains("blue", subject)
    if other:
        o_sql, o_p = _contains("red", other)
        drawn, dp = "(%s AND %s)" % (s_sql, o_sql), s_p + o_p
    else:
        drawn, dp = s_sql, s_p
    any_where, any_p = draft_filter(blue, red, patch, "any")
    obj_sql = "".join(
        ", sum(CASE WHEN drawn THEN %s - %s ELSE %s - %s END) d_%s"
        % (bc, rc, rc, bc, k) for k, bc, rc in TEAM_OBJ)
    sql = """
      SELECT count(blue_win) n, count(*) games,
             sum(CASE WHEN drawn THEN blue_win ELSE 1 - blue_win END) sel_wins,
             sum(CASE WHEN drawn THEN blue_gold - red_gold
                      ELSE red_gold - blue_gold END) gold_diff,
             sum(CASE WHEN drawn THEN blue_kills - red_kills
                      ELSE red_kills - blue_kills END) kill_diff%s
      FROM (SELECT blue_win, blue_gold, red_gold, blue_kills, red_kills,
                   %s AS drawn, %s
            FROM drafts WHERE %s) t
    """ % (obj_sql, drawn,
           ", ".join("%s, %s" % (bc, rc) for _, bc, rc in TEAM_OBJ),
           any_where)
    row = con.execute(sql, dp + any_p).fetchone()
    n, games, wins, gd, kd = row[:5]
    n = int(n or 0)
    wins = int(wins or 0)
    return {
        "subject": "blue" if blue else "red",
        "n": n, "games": int(games or 0), "sel_wins": wins,
        "sel_win_rate": (wins / n) if n else None,
        "gold_diff": (float(gd) / n) if n and gd is not None else None,
        "kill_diff": (float(kd) / n) if n and kd is not None else None,
        "obj_diff": {k: (float(v) / n) if n and v is not None else None
                     for (k, _, _), v in zip(TEAM_OBJ, row[5:])},
    }


def api_draft(con, blue, red, patch, sides):
    """N-way: live scan of `drafts` (bitmask containment)."""
    selected = [("blue", r) for r in blue] + [("red", r) for r in red]
    names = APP["names"]
    out = {"blue": blue, "red": red, "selected": len(selected), "sides": sides,
           "pairs": [], "estimate": [], "baseline": {}, "champ_stats": []}

    if not selected:
        out["n"] = int(con.execute(
            "SELECT count(*) FROM drafts WHERE " +
            ("patch = ?" if patch else "TRUE"),
            [patch] if patch else []).fetchone()[0])
        out["ladder"], out["drop_one"] = [], []
        out["champ_stats"], out["pairs"] = [], []
        out["team"] = None
        out["games"] = _sample_games(con, names, patch)
        return out

    hist = presence_histogram(con, selected, patch)

    def split(idxs):
        return ([i for i in idxs if selected[i][0] == "blue"],
                [i for i in idxs if selected[i][0] == "red"])

    every = list(range(len(selected)))
    b, r = split(every)
    out["n"] = hist_count(hist, b, r, sides)

    out["ladder"] = []
    for k in range(1, len(selected) + 1):
        bk, rk = split(list(range(k)))
        out["ladder"].append({
            "step": k, "side": selected[k - 1][0], "riot": selected[k - 1][1],
            "name": names.get(selected[k - 1][1], str(selected[k - 1][1])),
            "n": hist_count(hist, bk, rk, sides),
        })

    out["drop_one"] = []
    for i in range(len(selected)):
        bi, ri = split([j for j in every if j != i])
        out["drop_one"].append({
            "riot": selected[i][1],
            "name": names.get(selected[i][1], str(selected[i][1])),
            "n_without": hist_count(hist, bi, ri, sides),
        })

    out["team"] = team_block(con, blue, red, patch)
    # each side on its own, so the side views can say "these five champions
    # together" without borrowing the (much smaller) whole-draft sample
    out["side_stats"] = {
        "blue": team_block(con, blue, [], patch) if blue else None,
        "red": team_block(con, [], red, patch) if red else None,
    }

    # --- what the selected champions actually did in the matching games
    where, params = draft_filter(blue, red, patch, sides)
    riots = [r for _, r in selected]
    ph = ",".join("?" * len(riots))
    saggs = ", ".join("avg(w.%s) AS %s" % (k, k) for k in STATS)
    rows = con.execute(f"""
      SELECT w.champ, count(*) n, {saggs}
      FROM wide w
      WHERE w.champ IN ({ph})
        AND w.match_id IN (SELECT match_id FROM drafts WHERE {where})
      GROUP BY 1
    """, riots + params).fetchall()
    out["champ_stats"] = [
        dict(riot=int(r[0]), name=names.get(int(r[0]), str(r[0])),
             n=int(r[1]), **{c: _num(v) for c, v in zip(STATS, r[2:])})
        for r in rows]

    # --- pairwise values among the selection (the dense, measurable part)
    if len(selected) >= 2:
        ph = ",".join("?" * len(riots))
        mcols = ", ".join(STATS)
        psql = (f"SELECT champ, partner, kind, n, {mcols} FROM cube2 "
                f"WHERE champ IN ({ph}) AND partner IN ({ph})")
        pp = riots + riots
        if patch:
            psql += " AND patch = ?"
            pp.append(patch)
        bsql = f"SELECT champ, n, {mcols} FROM champ_base WHERE champ IN ({ph})"
        bp = riots
        if patch:
            bsql += " AND patch = ?"
            bp.append(patch)
        base = {}
        for row in con.execute(bsql, bp).fetchall():
            base[int(row[0])] = dict(n=int(row[1]),
                                     **{k: _num(v) for k, v in zip(STATS, row[2:])})
        pairs = []
        for row in con.execute(psql, pp).fetchall():
            c, p = int(row[0]), int(row[1])
            bm = base.get(c)
            if not bm:
                continue
            vals = {k: _num(v) for k, v in zip(STATS, row[4:])}
            deltas = {k: vals[k] - bm[k] for k in STATS
                      if vals.get(k) is not None and bm.get(k) is not None}
            pairs.append({"champ": c, "partner": p, "kind": row[2],
                          "n": int(row[3]), "vals": vals, "deltas": deltas})
        out["pairs"] = pairs
        out["baseline"] = {str(k): v for k, v in base.items()}

        # --- partner-effect summary: mean of the measured pair deviations.
        # Summing them would double-count, because the nine pairs share games
        # (Vladimir-with-Blitzcrank and Vladimir-with-LeeSin overlap whenever
        # those two are on the same team). The mean is the average conditional
        # shift and does not compound.
        by = {(x["champ"], x["partner"], x["kind"]): x for x in pairs}
        est = []
        for side, rr in selected:
            bm = base.get(rr)
            if not bm or bm.get("win") is None:
                continue
            allies = [x for s, x in selected if s == side and x != rr]
            enemies = [x for s, x in selected if s != side]
            terms = []
            for a in allies:
                e = by.get((rr, a, "with"))
                if e and "win" in e["deltas"]:
                    terms.append(e["deltas"]["win"])
            for e_ in enemies:
                e = by.get((rr, e_, "vs"))
                if e and "win" in e["deltas"]:
                    terms.append(e["deltas"]["win"])
            if not terms:
                continue
            mean_d = sum(terms) / len(terms)
            est.append({"riot": rr, "name": names.get(rr, str(rr)),
                        "base": bm["win"], "n_base": bm["n"],
                        "mean_delta": mean_d,
                        "estimate": max(0.0, min(1.0, bm["win"] + mean_d)),
                        "terms": len(terms),
                        "best": max(terms), "worst": min(terms)})
        out["estimate"] = est

    # --- the matching games themselves (a few, for eyeballing)
    where, params = draft_filter(blue, red, patch, sides)
    subject = set(blue or red)
    games = con.execute(f"""
      SELECT match_id, blue_champs, red_champs, blue_win, duration
      FROM drafts WHERE {where} LIMIT 12
    """, params).fetchall()
    out["games"] = [{
        "match_id": g[0],
        "blue": [names.get(int(c), str(c)) for c in g[1]],
        "red": [names.get(int(c), str(c)) for c in g[2]],
        "blue_win": int(g[3]) if g[3] is not None else None,
        "duration": int(g[4]) if g[4] is not None else None,
        "mine": "blue" if subject and subject <= set(int(c) for c in g[1])
                else ("red" if subject else None),
    } for g in games]
    return out


def _sample_games(con, names, patch, limit=12):
    rows = con.execute(
        "SELECT match_id, blue_champs, red_champs, blue_win, duration "
        "FROM drafts" + (" WHERE patch = ?" if patch else "") +
        " USING SAMPLE %d ROWS (reservoir, 42)" % limit,
        [patch] if patch else []).fetchall()
    return [{"match_id": g[0],
             "blue": [names.get(int(c), str(c)) for c in g[1]],
             "red": [names.get(int(c), str(c)) for c in g[2]],
             "blue_win": int(g[3]) if g[3] is not None else None,
             "duration": int(g[4]) if g[4] is not None else None,
             "mine": None} for g in rows]


def api_suggest(con, blue, red, metric, patch, top=5, pool=14):
    """Best allies to add to each side, and best answers to each side.

    The two lists answer different questions and are read from different
    places on purpose:

      pool    every candidate is ranked cheaply from the 2-way cube, which is
              dense (median 370-495 games per cell) and covers all 173 champs
      final   the top `pool` are then re-measured on `wide`, where the number
              is a count of *distinct games*: "champion C in a game where at
              least one of my champions was on C's team". Summing cube cells
              would count a game twice whenever two of my champions were in it
              (2,685 vs 2,632 real games for one candidate here), so the cube
              is used to choose the shortlist and the table to report the
              number that is actually shown.

    `delta` compares the candidate with its own average over the patch, which
    is what separates "a strong champion" from "a strong champion into this
    draft". A side with no champions gets empty lists - there is nothing to
    synergise with or to answer.
    """
    m = metric if metric in METRIC_BY_KEY and not METRIC_BY_KEY[metric].get("hidden") else "win"
    sign = METRIC_BY_KEY[m]["better"]
    names = APP["names"]
    out = {"metric": m, "sides": {}}

    # the candidate's own average over the patch, for the delta column
    if patch:
        grow = con.execute(
            "SELECT champ, n, %s FROM champ_base WHERE patch = ?" % m, [patch]).fetchall()
    else:
        grow = con.execute("""
          SELECT champ, sum(n), sum(%s * n) / nullif(sum(CASE WHEN %s IS NOT NULL THEN n END), 0)
          FROM champ_base GROUP BY 1""" % (m, m)).fetchall()
    general = {int(c): (int(n), _num(v)) for c, n, v in grow}

    pat = " AND patch = ?" if patch else ""
    pp = [patch] if patch else []

    def shortlist(kind, mine, excl):
        """Cube-ranked candidates for one question (cheap, all 173 champs)."""
        ph = ",".join("?" * len(mine))
        wavg = ("sum(%s * n) / nullif(sum(CASE WHEN %s IS NOT NULL THEN n END), 0)"
                % (m, m))
        rows = con.execute(f"""
          SELECT champ, sum(n), {wavg}
          FROM cube2 WHERE kind = '{kind}' AND partner IN ({ph}){pat}
          GROUP BY 1""", mine + pp).fetchall()
        items = [(int(c), _num(v)) for c, n, v in rows
                 if int(c) not in excl and _num(v) is not None]
        items.sort(key=lambda x: sign * x[1], reverse=True)
        return [c for c, _ in items[:pool]]

    def measure(ids, mine):
        """Exact per-game numbers for a shortlist, one scan of `wide`.

        The two membership tests are computed once per row in the inner select
        and then filtered twice, so each `?` appears exactly once and one
        champion's games are never counted twice.
        """
        if not ids:
            return {}
        ally = " OR ".join("list_contains(ally_champs, ?)" for _ in mine)
        enemy = " OR ".join("list_contains(enemy_champs, ?)" for _ in mine)
        rows = con.execute(f"""
          WITH t AS (
            SELECT champ, {m} AS v, ({ally}) AS on_my_team, ({enemy}) AS vs_my_team
            FROM wide
            WHERE champ IN ({",".join("?" * len(ids))}){pat}
          )
          SELECT champ,
                 count(*) FILTER (WHERE on_my_team) n_syn,
                 avg(v) FILTER (WHERE on_my_team) v_syn,
                 count(*) FILTER (WHERE vs_my_team) n_ctr,
                 avg(v) FILTER (WHERE vs_my_team) v_ctr
          FROM t GROUP BY 1""", mine + mine + ids + pp).fetchall()
        return {int(r[0]): (int(r[1]), _num(r[2]), int(r[3]), _num(r[4]))
                for r in rows}

    for side, mine, other in (("blue", blue, red), ("red", red, blue)):
        side_out = {"mine": mine, "synergy": [], "counter": []}
        out["sides"][side] = side_out
        if not mine:
            continue
        excl = set(mine) | set(other)
        syn_ids = shortlist("with", mine, excl)
        ctr_ids = shortlist("vs", mine, excl)
        exact = measure(sorted(set(syn_ids) | set(ctr_ids)), mine)

        def build(ids, idx):
            items = []
            for cand in ids:
                row = exact.get(cand)
                if not row:
                    continue
                n, val = row[idx], row[idx + 1]
                g = general.get(cand)
                if not n or val is None or not g or g[1] is None:
                    continue
                items.append({
                    "riot": cand, "name": names.get(cand, str(cand)),
                    "specific": val, "n": int(n),
                    "general": g[1], "n_general": g[0],
                    "delta": val - g[1],
                })
            items.sort(key=lambda x: (sign * x["specific"], -x["n"]), reverse=True)
            return items[:top]

        side_out["synergy"] = build(syn_ids, 0)
        side_out["counter"] = build(ctr_ids, 2)

    out["partial"] = bool(METRIC_BY_KEY[m].get("partial"))
    out["cov"] = METRIC_BY_KEY[m].get("cov")
    return out


def api_champion(con, champ, blue, red, patch):
    """One champion's full stat line, inside the matching games and overall.

    Partial stats are averaged over the rows that carry them, so the response
    also carries `n_<key>` counts - otherwise a 40-game average of a stat that
    only exists in the newer schema would be indistinguishable from a 12k one.
    """
    where, params = draft_filter(blue, red, patch, "any")
    aggs = ", ".join("avg(%s) AS %s" % (c, c) for c in DETAIL)
    counts = ", ".join("count(%s) AS n_%s" % (c, c) for c in PARTIAL)
    row = con.execute(f"""
      SELECT count(*) n, {aggs}, {counts}
      FROM wide
      WHERE champ = ? AND match_id IN (SELECT match_id FROM drafts WHERE {where})
    """, [champ] + params).fetchone()
    cols = ["n"] + DETAIL + ["n_" + c for c in PARTIAL]
    in_games = _statrow(row, cols)
    in_games["n"] = int(row[0])

    bsql = f"SELECT count(*) n, {aggs} FROM wide WHERE champ = ?"
    bp = [champ]
    if patch:
        bsql += " AND patch = ?"
        bp.append(patch)
    brow = con.execute(bsql, bp).fetchone()
    baseline = _statrow(brow, ["n"] + DETAIL)
    baseline["n"] = int(brow[0])

    delta = {k: in_games[k] - baseline[k] for k in DETAIL
             if in_games.get(k) is not None and baseline.get(k) is not None}
    return {"riot": champ, "name": APP["names"].get(champ, str(champ)),
            "in_games": in_games, "baseline": baseline, "delta": delta}


# ----------------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _ids(self, qs, key):
        raw = qs.get(key, [""])[0]
        out = []
        for tok in raw.split(","):
            tok = tok.strip()
            if tok.isdigit():
                out.append(int(tok))
        return out

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        qs = parse_qs(urlparse(self.path).query)
        con = APP["con"]
        patch = qs.get("patch", [""])[0] or None
        try:
            if path == "/api/meta":
                with _lock:
                    self._send(200, api_meta(con))
                return
            if path == "/api/matchup":
                champ = self._ids(qs, "champ")
                partner = self._ids(qs, "partner")
                if not champ or not partner:
                    self._send(400, {"error": "champ and partner are required"})
                    return
                with _lock:
                    out = api_matchup(con, champ[0], partner[0],
                                      qs.get("mode", ["with"])[0], patch,
                                      qs.get("role", [""])[0] or None)
                self._send(200, out)
                return
            if path == "/api/draft":
                with _lock:
                    out = api_draft(con, self._ids(qs, "blue"),
                                    self._ids(qs, "red"), patch,
                                    qs.get("sides", ["any"])[0])
                self._send(200, out)
                return
            if path == "/api/suggest":
                with _lock:
                    out = api_suggest(con, self._ids(qs, "blue"),
                                      self._ids(qs, "red"),
                                      qs.get("metric", ["win"])[0], patch)
                self._send(200, out)
                return
            if path == "/api/champion":
                champ = self._ids(qs, "champ")
                if not champ:
                    self._send(400, {"error": "champ is required"})
                    return
                with _lock:
                    out = api_champion(con, champ[0], self._ids(qs, "blue"),
                                       self._ids(qs, "red"), patch)
                self._send(200, out)
                return
        except Exception as exc:  # noqa: BLE001 - surface the real message
            self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc)})
            return
        if path.startswith("/api/"):
            self._send(404, {"error": "unknown endpoint"})
            return
        rel = "index.html" if path == "/" else path.lstrip("/")
        fpath = (STATIC / rel).resolve()
        if not str(fpath).startswith(str(STATIC.resolve())) or not fpath.is_file():
            self._send(404, {"error": "not found"}, "text/plain")
            return
        ctype = {".html": "text/html", ".js": "text/javascript",
                 ".css": "text/css"}.get(fpath.suffix, "application/octet-stream")
        self._send(200, fpath.read_bytes(), ctype)


def probe_coverage(con):
    """One scan: what share of rows actually carries each registered field.

    The archive snapshot predates the expanded challenge whitelist, so a stat
    can be perfectly aggregated in the cube and still describe a fraction of
    the corpus. Measuring it beats hardcoding a list: when the next backfill
    lands the flags clear themselves.
    """
    cols = ", ".join(
        "avg(CASE WHEN %s IS NOT NULL THEN 1 ELSE 0 END) c%d" % (k, i)
        for i, k in enumerate(STATS))
    row = con.execute("SELECT %s FROM wide" % cols).fetchone()
    global PARTIAL
    PARTIAL = []
    for i, m in enumerate(METRICS):
        cov = float(row[i] or 0.0)
        m["cov"] = round(cov, 4)
        m.pop("partial", None)
        if cov < FULL_COVER:
            m["partial"] = 1
            PARTIAL.append(m["key"])
    thin = [(m["key"], m["cov"]) for m in METRICS if m.get("partial")]
    print("coverage: %d/%d fields on %.0f%%+ of rows%s"
          % (len(METRICS) - len(thin), len(METRICS), FULL_COVER * 100,
             ("; thin: " + ", ".join("%s %.0f%%" % (k, c * 100)
                                     for k, c in thin[:6])) if thin else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--db", default=str(STORE_DB))
    a = ap.parse_args()
    if not os.path.exists(a.db):
        raise SystemExit("no store at %s - run `python store.py build` first" % a.db)
    con = duckdb.connect(a.db, read_only=True)
    con.execute("SET threads=%d" % (os.cpu_count() or 4))
    champs = load_champions(con)
    APP.update({
        "con": con, "champions": champs,
        "names": {c["riot"]: c["name"] for c in champs},
        "idx": {c["riot"]: c["idx"] for c in champs if c["idx"] is not None},
    })
    print("store   : %s" % a.db)
    print("games   : %d" % con.execute("SELECT count(*) FROM drafts").fetchone()[0])
    print("champions: %d" % len(champs))
    probe_coverage(con)
    print("serving : http://127.0.0.1:%d  (local only)" % a.port)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
