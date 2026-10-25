"""Analytical store for soloQ champ stats (DuckDB + Parquet).

Implements the two query shapes the frontend needs, on the layout that measured
fastest on this corpus:

  q1  avg stats of champ X when paired with / against champ Y      (2-way)
  q2  avg stats of champ X paired with Y and Z, against A and B    (5-way)

Design (measured, see notes at the bottom):
  - ONE subject-centric "wide" row per player-game:
        champ, pos, win, stats..., ally_champs[4], enemy_champs[5]
    2-way  -> list_contains(ally_champs, Y)        ~14 ms over 4.15M rows
    5-way  -> 4x list_contains                    ~28 ms over 4.15M rows
    No joins, no cube, no server.
  - A materialized 2-way cube (29.7k rows x 2 directions) for sub-ms lookups.
  - The 5-way cube is deliberately NOT built: 240M cells, 1.1 GB, and slower
    to query (~195 ms) than scanning the wide table (~28 ms). It is also 99.99%
    empty - the densest 5-champ cell in 415k games has 21 games, and no cell
    reaches n=30.

Usage
-----
  python store.py build [--limit N] [--out DIR]
  python store.py q1 --champ 103 --with 64 [--against] [--role MIDDLE]
  python store.py q2 --champ 103 --with 64,1 --against 2,3

Champ ids are Riot ids (same space as champ_mapping.json). Names accepted too
via champ_names.json.
"""
import argparse
import gzip
import glob
import json
import os
import sys
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATCHDATA = os.path.join(ROOT, "data", "solo queue", "matchdata")
DEFAULT_OUT = os.path.join(HERE, "store")

# challenge whitelist - MUST stay in sync with collect_match_data.py.
# Older archives carry only the first 7; missing keys stay NULL, never 0.
CH_KEYS = [
    "goldPerMinute", "damagePerMinute", "visionScorePerMinute",
    "killParticipation", "teamDamagePercentage", "damageTakenOnTeamPercentage",
    "maxCsAdvantageOnLaneOpponent", "maxLevelLeadLaneOpponent",
    "laningPhaseGoldExpAdvantage", "earlyLaningPhaseGoldExpAdvantage",
    "visionScoreAdvantageLaneOpponent", "jungleCsBefore10Minutes",
    "laneMinionsFirst10Minutes", "takedownsFirstXMinutes",
    "earliestBaron", "firstTurretKilledTime", "turretPlatesTaken",
    "turretTakedowns", "dragonTakedowns", "baronTakedowns",
    "riftHeraldTakedowns", "teamBaronKills", "teamRiftHeraldKills",
    "teamElderDragonKills", "soloKills", "deathsByEnemyChamps",
    "enemyChampionImmobilizations", "effectiveHealAndShielding",
    "killsNearEnemyTurret", "maxKillDeficit", "bountyGold", "outnumberedKills",
]

BASE_STATS = ["kills", "deaths", "assists", "gold", "dmg", "cs", "vision"]
AGG_STATS = ["win"] + BASE_STATS + CH_KEYS   # every column the cubes aggregate

SCHEMA = pa.schema(
    [("match_id", pa.string()), ("src", pa.string()), ("schema_ver", pa.int8()),
     ("patch", pa.string()), ("queue_id", pa.int16()), ("start", pa.int64()),
     ("duration", pa.int32()), ("side", pa.string()), ("win", pa.int8()),
     ("pid", pa.int8()), ("champ", pa.int16()), ("pos", pa.string())]
    + [(k, pa.int32()) for k in ["kills", "deaths", "assists", "gold", "dmg",
                                 "cs", "vision"]]
    + [(k, pa.int8()) for k in ["fb_kill", "fb_assist", "ft_kill", "ft_assist"]]
    + [(k, pa.float32()) for k in CH_KEYS]
)


def patch_of(ver):
    if not ver:
        return None
    p = str(ver).split(".")
    return ".".join(p[:2]) if len(p) >= 2 else str(ver)


def iter_records(limit=None):
    """Stream every shard, archive first, then the live crawl."""
    files = []
    for f in sorted(glob.glob(os.path.join(MATCHDATA, "archive_pre_snapshots", "*", "*.jsonl.gz"))):
        files.append((f, "archive", True))
    for f in sorted(glob.glob(os.path.join(MATCHDATA, "*", "shard_*.jsonl"))):
        files.append((f, "live", False))
    n = 0
    for path, src, gz in files:
        op = gzip.open if gz else open
        with op(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # truncated tail of a shard being written
                if not rec.get("players"):
                    continue
                yield rec, src
                n += 1
                if limit and n >= limit:
                    return


def rows_of(rec, src):
    dur = rec.get("duration") or 0
    teams = rec.get("teams") or {}
    for p in rec["players"]:
        side = "blue" if p.get("team") == 100 else "red"
        tw = (teams.get(side) or {}).get("win")
        row = {
            "match_id": rec.get("match_id"), "src": src,
            "schema_ver": 2 if src == "live" else 1,
            "patch": patch_of(rec.get("version")),
            "queue_id": rec.get("queue_id"), "start": rec.get("start"),
            "duration": dur, "side": side, "win": tw,
            "pid": p.get("pid"), "champ": p.get("champ"), "pos": p.get("pos"),
        }
        for k in ["kills", "deaths", "assists", "gold", "dmg", "cs", "vision"]:
            row[k] = p.get(k)
        for k in ["fb_kill", "fb_assist", "ft_kill", "ft_assist"]:
            row[k] = p.get(k)
        ch = p.get("ch") or {}
        for k in CH_KEYS:
            row[k] = ch.get(k)
        yield row


def build(outdir, limit=None):
    os.makedirs(outdir, exist_ok=True)
    long_path = os.path.join(outdir, "players.parquet")
    db_path = os.path.join(outdir, "store.duckdb")
    t0 = time.time()
    w = pq.ParquetWriter(long_path, SCHEMA, compression="zstd")
    buf, games = [], 0
    for rec, src in iter_records(limit):
        buf.extend(rows_of(rec, src))
        games += 1
        if len(buf) >= 200_000:
            w.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
            buf = []
            print("  %5.0fs  %d games  %d rows" % (time.time() - t0, games, games * 10), flush=True)
    if buf:
        w.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
    w.close()
    print("read %d games in %.0fs" % (games, time.time() - t0))

    con = duckdb.connect(db_path)
    con.execute("SET threads=%d" % (os.cpu_count() or 4))
    print("deduping (archive snapshot vs live re-fetch)...")
    con.execute(f"""
      CREATE OR REPLACE TABLE players AS
      SELECT * EXCLUDE (rn) FROM (
        SELECT *, row_number() OVER (PARTITION BY match_id, pid ORDER BY schema_ver DESC) rn
        FROM read_parquet('{long_path.replace(chr(92), "/")}')
      ) WHERE rn = 1
    """)
    print("deduped:", con.execute("SELECT count(*), count(DISTINCT match_id) FROM players").fetchall())

    print("building subject-centric wide table...")
    con.execute("""
      CREATE OR REPLACE TABLE wide AS
      WITH s AS (
        SELECT match_id, side, list(champ ORDER BY champ) champs FROM players GROUP BY 1,2
      )
      SELECT p.*, list_filter(sf.champs, lambda c: c <> p.champ) ally_champs,
             se.champs enemy_champs
      FROM players p
      JOIN s sf ON sf.match_id = p.match_id AND sf.side = p.side
      JOIN s se ON se.match_id = p.match_id AND se.side <> p.side
    """)

    print("building 2-way cube...")
    stat_aggs = "".join(", avg(%s) %s" % (k, k) for k in AGG_STATS)
    con.execute(f"""
      CREATE OR REPLACE TABLE cube2 AS
      SELECT patch, champ, partner, kind, count(*) n{stat_aggs}
      FROM (
        SELECT patch, champ, unnest(ally_champs) partner, 'with' kind,
               * EXCLUDE (ally_champs, enemy_champs) FROM wide
        UNION ALL
        SELECT patch, champ, unnest(enemy_champs) partner, 'vs' kind,
               * EXCLUDE (ally_champs, enemy_champs) FROM wide
      ) GROUP BY 1,2,3,4
    """)
    build_derived(con)
    for t, f in [("wide", "wide.parquet"), ("cube2", "cube2.parquet"),
                 ("drafts", "drafts.parquet"), ("champ_base", "champ_base.parquet")]:
        con.execute(f"COPY {t} TO '{os.path.join(outdir, f).replace(chr(92), '/')}' "
                    "(FORMAT parquet, COMPRESSION zstd)")
    con.execute("CHECKPOINT")
    print(con.execute("""
      SELECT kind, count(*) cells, median(n) med, max(n) mx FROM cube2 GROUP BY 1
    """).fetchall())
    con.close()
    for f in ["players.parquet", "wide.parquet", "cube2.parquet", "drafts.parquet",
              "champ_base.parquet", "store.duckdb"]:
        p = os.path.join(outdir, f)
        if os.path.exists(p):
            print("  %-20s %7.1f MB" % (f, os.path.getsize(p) / 1e6))
    print("done in %.0fs" % (time.time() - t0))


def build_derived(con):
    """Everything that can be rebuilt from `players` without re-reading shards."""
    build_drafts(con)
    stat_aggs = "".join(", avg(%s) %s" % (k, k) for k in AGG_STATS)
    con.execute(f"""
      CREATE OR REPLACE TABLE champ_base AS
      SELECT patch, champ, count(*) n{stat_aggs} FROM wide GROUP BY 1,2
    """)
    con.execute(f"""
      CREATE OR REPLACE TABLE champ_pos_base AS
      SELECT patch, champ, pos, count(*) n{stat_aggs}
      FROM wide WHERE pos IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
      GROUP BY 1,2,3
    """)
    print("champ_base:", con.execute("SELECT count(*) FROM champ_base").fetchone()[0])


def build_drafts(con):
    """One row per game with both team champion sets, as lists AND bitmasks.

    A full-draft question ("has this exact 10-champ game been played?") then
    becomes a single-table scan. The bitmask columns are what make it fast:
    `(blue_m0 & ?) = ?` is three integer ANDs per row, versus list_has_all
    rebuilding a list per row (measured 40 ms -> 5 ms over 415k drafts).
    173 champions need 3 x 64-bit words.
    """
    con.execute("CREATE OR REPLACE TABLE champ_idx (riot INTEGER, idx INTEGER, "
                "part INTEGER, bit UBIGINT)")
    rows = []
    for riot, idx in sorted(CHAMP_IDX.items()):
        rows.append((riot, idx, idx // 64, 1 << (idx % 64)))
    con.executemany("INSERT INTO champ_idx VALUES (?,?,?,?)", rows)
    con.execute("""
      CREATE OR REPLACE TABLE drafts AS
      WITH t AS (
        SELECT p.match_id,
               any_value(p.patch) patch, any_value(p.queue_id) queue_id,
               any_value(p.duration) duration, p.side,
               list(p.champ ORDER BY p.champ) champs,
               any_value(p.win) win,
               sum(p.kills) kills, sum(p.deaths) deaths, sum(p.gold) gold,
               sum(CASE WHEN m.part = 0 THEN m.bit ELSE 0 END)::UBIGINT m0,
               sum(CASE WHEN m.part = 1 THEN m.bit ELSE 0 END)::UBIGINT m1,
               sum(CASE WHEN m.part = 2 THEN m.bit ELSE 0 END)::UBIGINT m2
        FROM players p
        LEFT JOIN champ_idx m ON m.riot = p.champ
        GROUP BY p.match_id, p.side
      )
      SELECT b.match_id, b.patch, b.queue_id, b.duration,
             b.champs blue_champs, r.champs red_champs, b.win blue_win,
             b.m0 blue_m0, b.m1 blue_m1, b.m2 blue_m2,
             r.m0 red_m0, r.m1 red_m1, r.m2 red_m2,
             b.kills blue_kills, r.kills red_kills,
             b.gold blue_gold, r.gold red_gold
      FROM t b JOIN t r ON b.match_id = r.match_id
      WHERE b.side = 'blue' AND r.side = 'red'
    """)
    print("drafts:", con.execute("SELECT count(*) FROM drafts").fetchone()[0])


def connect(outdir, read_only=True):
    db = os.path.join(outdir, "store.duckdb")
    if not os.path.exists(db):
        sys.exit("no store at %s - run `build` first" % db)
    try:
        con = duckdb.connect(db, read_only=read_only)
    except Exception as exc:  # noqa: BLE001 - almost always the explorer holding the lock
        if "being used by another process" in str(exc) or "Cannot open file" in str(exc):
            sys.exit("the store is locked by a running process (probably\n"
                     "  python explorer/server.py\n"
                     "Stop the explorer, rebuild, then start it again. On Windows a\n"
                     "read-only DuckDB connection still holds the file.")
        raise
    con.execute("SET threads=%d" % (os.cpu_count() or 4))
    return con


def champ_names():
    """riot id (int) -> display name, from the patch-pinned names file."""
    path = os.path.join(ROOT, "data", "solo queue", "process matchdata",
                        "champ_names.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    names = data.get("names", data) if isinstance(data, dict) else {}
    return {int(k): v for k, v in names.items() if str(k).isdigit()}


def champ_idx_map():
    """riot id (int) -> dense 0..C-1 index (the model's champion space)."""
    path = os.path.join(ROOT, "data", "solo queue", "process matchdata",
                        "champ_mapping.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data["riot_to_idx"].items()}


CHAMP_IDX = champ_idx_map()


def resolve(outdir, token):
    """Accept a champion name or a numeric Riot id."""
    try:
        return int(token)
    except ValueError:
        pass
    for riot, name in champ_names().items():
        if str(name).lower() == str(token).lower():
            return riot
    sys.exit("unknown champion %r" % token)


def show(cur):
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if row is None:
        print("  no rows")
        return
    w = max(len(c) for c in cols)
    for c, v in zip(cols, row):
        print("  %-*s  %s" % (w, c, ("%.3f" % v) if isinstance(v, float) else v))


def q1(outdir, champ, partner, against=False, role=None, patch=None):
    con = connect(outdir)
    x = resolve(outdir, champ)
    y = resolve(outdir, partner)
    kind = "vs" if against else "with"
    sql = "SELECT * FROM cube2 WHERE champ=? AND partner=? AND kind=?"
    params = [x, y, kind]
    if patch:
        sql += " AND patch=?"
        params.append(patch)
    row = con.execute(sql, params).fetchone()
    print("\nQ1  champ %s %s %s" % (x, "against" if against else "with", y))
    if row is None:
        print("  n = 0 (cell never observed)")
    else:
        cols = [d[0] for d in con.description]
        d = dict(zip(cols, row))
        for k in ["n", "win"] + BASE_STATS:
            if k in d and d[k] is not None:
                print("  %-8s %s" % (k, ("%.3f" % d[k]) if isinstance(d[k], float) else d[k]))
        print("  (source: precomputed cube2, ~2 ms)")
    if role:
        r = con.execute("""
          SELECT count(*) n, avg(win) wr, avg(gold) gold, avg(dmg) dmg
          FROM wide WHERE champ=? AND pos=?
            AND (CASE WHEN ? THEN list_contains(enemy_champs,?) ELSE list_contains(ally_champs,?) END)
        """, [x, role, against, y, y]).fetchone()
        print("  role=%s -> n=%d wr=%.3f" % (role, r[0], r[1] if r[1] else 0))
    con.close()


def q2(outdir, champ, with_, against, min_n=30):
    con = connect(outdir)
    x = resolve(outdir, champ)
    ys = [resolve(outdir, t) for t in with_.split(",")]
    as_ = [resolve(outdir, t) for t in against.split(",")]
    pred = ["champ = ?"]
    params = [x]
    for y in ys:
        pred.append("list_contains(ally_champs, ?)")
        params.append(y)
    for a in as_:
        pred.append("list_contains(enemy_champs, ?)")
        params.append(a)
    sql = ("SELECT count(*) n, avg(win) wr, avg(gold) gold, avg(dmg) dmg, "
           "avg(kills) kills, avg(deaths) deaths, avg(assists) assists "
           "FROM wide WHERE " + " AND ".join(pred))
    t0 = time.perf_counter()
    r = con.execute(sql, params).fetchone()
    ms = (time.perf_counter() - t0) * 1000
    print("\nQ2  champ %s with %s against %s" % (x, ys, as_))
    print("  n = %d   (%.1f ms, single-table scan of %d rows)"
          % (r[0], ms, con.execute("SELECT count(*) FROM wide").fetchone()[0]))
    if r[0]:
        print("  wr=%.3f  gold=%.0f  dmg=%.0f  kda=%.2f/%.2f/%.2f"
              % (r[1], r[2], r[3], r[4], r[5], r[6]))
    if r[0] < min_n:
        print("  WARNING: n=%d < %d - this average is noise, not a matchup estimate."
              % (r[0], min_n))
        print("           Fall back to the 2-way marginals (q1) or a shrinkage model.")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "derive", "q1", "q2"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--champ", default=None)
    ap.add_argument("--with", dest="with_", default=None)
    ap.add_argument("--against", default=None)
    ap.add_argument("--role", default=None)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.limit)
    elif a.cmd == "derive":
        con = connect(a.out, read_only=False)
        build_derived(con)
        for t in ["drafts", "champ_base"]:
            con.execute(f"COPY {t} TO "
                        f"'{os.path.join(a.out, t + '.parquet').replace(chr(92), '/')}' "
                        "(FORMAT parquet, COMPRESSION zstd)")
        con.execute("CHECKPOINT")
        con.close()
        for t in ["drafts", "champ_base"]:
            p = os.path.join(a.out, t + ".parquet")
            print("  %-20s %6.1f MB" % (t + ".parquet", os.path.getsize(p) / 1e6))
    elif a.cmd == "q1":
        if not a.champ or not a.with_:
            sys.exit("q1 needs --champ and --with")
        q1(a.out, a.champ, a.with_, against=bool(a.against), role=a.role,
           patch=a.patch)
    else:
        if not a.champ or not a.with_ or not a.against:
            sys.exit("q2 needs --champ, --with Y,Z and --against A,B")
        q2(a.out, a.champ, a.with_, a.against, a.min_n)


if __name__ == "__main__":
    main()
