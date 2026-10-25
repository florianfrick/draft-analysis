"""Empirical stats explorer over the soloQ store (DuckDB, read-only).

Answers the two questions the draft UI needs, from measured data only:

  matchup   avg stats of champ X when paired with / against champ Y   (2-way)
  draft     given the picks built so far, how many games match, and what
            did each champion actually do in those games              (N-way)

Design notes
------------
- The 2-way question is served from the precomputed `cube2` table (~1 ms).
- The N-way question is a LIVE SCAN of `drafts` (415k rows, one per game),
  not a precomputed cube. A 5-champion cube would be 240M cells / 1.1 GB and
  is 99.99% empty (the densest 5-champ cell in 415k games has 21 games), so
  there is nothing to precompute.
- Containment is tested with bitmasks, not `list_has_all`: three integer ANDs
  per row instead of rebuilding a list per row. Measured 40 ms -> 3 ms.
- Everything the UI shows is returned with its `n`. When n is small the
  response also carries `estimate`, a first-order additive prediction built
  from the (dense) 2-way tables, so a 0-game draft still says something
  honest instead of nothing.

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

STATS = ["win", "kills", "deaths", "assists", "gold", "dmg", "cs", "vision"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

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

    sides='any' accepts either orientation (a draft is a set of two teams, so
    blue-in-my-UI may have been red in the actual game).
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
        "champions": APP["champions"],
        "thresholds": {"strong": 200, "ok": 50, "weak": 20},
    }


def _statrow(row, cols):
    return {c: (float(v) if isinstance(v, float) else v)
            for c, v in zip(cols, row) if v is not None}


def api_matchup(con, champ, partner, mode, patch, role):
    """2-way: served from the precomputed cube2 (no scan)."""
    kind = "vs" if mode == "vs" else "with"
    sql = ("SELECT n, win, kills, deaths, assists, gold, dmg, cs, vision "
           "FROM cube2 WHERE champ = ? AND partner = ? AND kind = ?")
    params = [champ, partner, kind]
    if patch:
        sql += " AND patch = ?"
        params.append(patch)
    cols = ["n"] + STATS
    row = con.execute(sql, params).fetchone()
    out = {"mode": kind, "n": 0, "stats": None}

    base_sql = "SELECT n, win, kills, deaths, assists, gold, dmg, cs, vision " \
               "FROM champ_base WHERE champ = ?"
    bp = [champ]
    if patch:
        base_sql += " AND patch = ?"
        bp.append(patch)
    brow = con.execute(base_sql, bp).fetchone()

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
        rsql = ("SELECT n, win, gold, dmg FROM champ_pos_base "
                "WHERE champ = ? AND pos = ?")
        rp = [champ, role]
        if patch:
            rsql += " AND patch = ?"
            rp.append(patch)
        rrow = con.execute(rsql, rp).fetchone()
        if rrow:
            out["role"] = {"pos": role, "n": int(rrow[0]),
                           "win": float(rrow[1]) if rrow[1] is not None else None,
                           "gold": float(rrow[2]) if rrow[2] is not None else None}
    return out


def presence_histogram(con, selected, patch):
    """One scan -> {(presence_code, games)} for the selected champions.

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


def hist_count(hist, blue_pos, red_pos, sides):
    """Games whose blue side holds every blue_pos champion and red every red_pos."""
    total = 0
    for code, n in hist:
        ok = all((code >> (2 * j)) & 1 for j in blue_pos) and \
             all((code >> (2 * j + 1)) & 1 for j in red_pos)
        if not ok and sides == "any" and blue_pos and red_pos:
            ok = all((code >> (2 * j)) & 1 for j in red_pos) and \
                 all((code >> (2 * j + 1)) & 1 for j in blue_pos)
        if ok:
            total += n
    return total


def api_draft(con, blue, red, patch, sides):
    """N-way: live scan of `drafts` (bitmask containment)."""
    selected = [("blue", r) for r in blue] + [("red", r) for r in red]
    names = APP["names"]
    out = {"blue": blue, "red": red, "selected": len(selected), "sides": sides}

    if not selected:
        out["n"] = int(con.execute(
            "SELECT count(*) FROM drafts WHERE " +
            ("patch = ?" if patch else "TRUE"),
            [patch] if patch else []).fetchone()[0])
        out["ladder"], out["drop_one"] = [], []
        out["champ_stats"] = []
        # with nothing selected, hand back a few random real drafts to load
        rows = con.execute(
            "SELECT match_id, blue_champs, red_champs, blue_win, duration "
            "FROM drafts" + (" WHERE patch = ?" if patch else "") +
            " USING SAMPLE 12 ROWS (reservoir, 42)",
            [patch] if patch else []).fetchall()
        out["games"] = [{"match_id": g[0],
                         "blue": [names.get(int(c), str(c)) for c in g[1]],
                         "red": [names.get(int(c), str(c)) for c in g[2]],
                         "blue_win": int(g[3]) if g[3] is not None else None,
                         "duration": int(g[4]) if g[4] is not None else None}
                        for g in rows]
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

    # --- what the selected champions actually did in the matching games
    if selected:
        where, params = draft_filter(blue, red, patch, sides)
        riots = [r for _, r in selected]
        ph = ",".join("?" * len(riots))
        rows = con.execute(f"""
          SELECT w.champ, count(*) n, avg(w.win) win, avg(w.kills) kills,
                 avg(w.deaths) deaths, avg(w.assists) assists, avg(w.gold) gold,
                 avg(w.dmg) dmg, avg(w.cs) cs, avg(w.vision) vision
          FROM wide w
          WHERE w.champ IN ({ph})
            AND w.match_id IN (SELECT match_id FROM drafts WHERE {where})
          GROUP BY 1
        """, riots + params).fetchall()
        out["champ_stats"] = [
            dict(riot=int(r[0]), name=names.get(int(r[0]), str(r[0])),
                 **{c: (float(v) if isinstance(v, float) else int(v))
                    for c, v in zip(["n"] + STATS, r[1:])})
            for r in rows]

    # --- pairwise deltas among the selection (the dense, measurable part)
    if len(selected) >= 2:
        riots = [r for _, r in selected]
        ph = ",".join("?" * len(riots))
        psql = (f"SELECT champ, partner, kind, n, win FROM cube2 "
                f"WHERE champ IN ({ph}) AND partner IN ({ph})")
        pp = riots + riots
        if patch:
            psql += " AND patch = ?"
            pp.append(patch)
        base = {}
        bsql = f"SELECT champ, n, win FROM champ_base WHERE champ IN ({ph})"
        bp = riots
        if patch:
            bsql += " AND patch = ?"
            bp.append(patch)
        for c, bn, bw in con.execute(bsql, bp).fetchall():
            base[int(c)] = (int(bn), float(bw) if bw is not None else None)
        pairs = []
        for c, p, k, pn, pw in con.execute(psql, pp).fetchall():
            c, p = int(c), int(p)
            b = base.get(c)
            if not b or b[1] is None or pw is None:
                continue
            pairs.append({"champ": c, "partner": p, "kind": k, "n": int(pn),
                          "win": float(pw), "delta": float(pw) - b[1]})
        out["pairs"] = pairs
        out["baseline"] = {str(k): {"n": v[0], "win": v[1]} for k, v in base.items()}

        # --- partner-effect summary: mean of the measured pair deviations.
        # Summing them would double-count, because the nine pairs share games
        # (Vladimir-with-Blitzcrank and Vladimir-with-LeeSin overlap whenever
        # those two are on the same team). The mean is the average conditional
        # shift and does not compound.
        by = {(x["champ"], x["partner"], x["kind"]): x for x in pairs}
        est = []
        for side, r in selected:
            b = base.get(r)
            if not b or b[1] is None:
                continue
            allies = [x for s, x in selected if s == side and x != r]
            enemies = [x for s, x in selected if s != side]
            terms = []
            for a in allies:
                e = by.get((r, a, "with"))
                if e:
                    terms.append(e["delta"])
            for e_ in enemies:
                e = by.get((r, e_, "vs"))
                if e:
                    terms.append(e["delta"])
            if not terms:
                continue
            mean_d = sum(terms) / len(terms)
            est.append({"riot": r, "name": names.get(r, str(r)),
                        "base": b[1], "n_base": b[0],
                        "mean_delta": mean_d,
                        "estimate": max(0.0, min(1.0, b[1] + mean_d)),
                        "terms": len(terms),
                        "best": max(terms), "worst": min(terms)})
        out["estimate"] = est

    # --- the matching games themselves (a few, for eyeballing)
    where, params = draft_filter(blue, red, patch, sides)
    games = con.execute(f"""
      SELECT match_id, blue_champs, red_champs, blue_win, duration
      FROM drafts WHERE {where} LIMIT 12
    """, params).fetchall()
    out["games"] = [{"match_id": g[0],
                     "blue": [names.get(int(c), str(c)) for c in g[1]],
                     "red": [names.get(int(c), str(c)) for c in g[2]],
                     "blue_win": int(g[3]) if g[3] is not None else None,
                     "duration": int(g[4]) if g[4] is not None else None}
                    for g in games]
    return out


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
                                      qs.get("mode", ["with"])[0],
                                      qs.get("patch", [""])[0] or None,
                                      qs.get("role", [""])[0] or None)
                self._send(200, out)
                return
            if path == "/api/draft":
                with _lock:
                    out = api_draft(con, self._ids(qs, "blue"),
                                    self._ids(qs, "red"),
                                    qs.get("patch", [""])[0] or None,
                                    qs.get("sides", ["any"])[0])
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
    print("serving : http://127.0.0.1:%d  (local only)" % a.port)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
