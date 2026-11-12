"""The queryable view of the pro corpus, and the schema the agent is shown.

`frontend/tools.py` precomputes the aggregates *I* guessed a coach would want,
and `_probe_tool_power.py` showed the guess was mostly wrong: the per-champion
cell is 0-2 of 120 boards at a given move. A SQL fallback moves that choice to
the model, which is the right place for it - but only if the schema makes the
correct question easier to write than the wrong one.

Three properties of this corpus shape the schema, and all three were measured
rather than assumed:

**Patch is not a time partition.** Patches overlap by weeks because leagues play
on different schedules (16.17 runs 08-27..10-13, 16.18 runs 09-19..10-17), 16.19
and 16.21 are absent entirely, and 16.22 is 5 games on a single day inside
16.20's window. So `patch_rank` is a *dense* rank over observed patches, not
`major * 100 + minor`, and `date_ordinal` is the only honest recency axis. The
schema doc says so in as many words, because `ORDER BY patch DESC LIMIT 1` is
the query a good model writes first and it returns five games.

**`is_scored` is not derivable.** The head maps a patch it never saw to UNK and
answers patch-agnostically without saying so, so the `patch` table carries
`is_scored` from the checkpoint's own `patch_vocab` rather than guessing. For
the shipped shared-split head all 107 corpus patches are scored - the flag earns
its place the first time the corpus outruns a checkpoint, and until then it is
a truthful all-1 column rather than an assumption. 16.22 is still a trap, but
for a different reason: 5 games on a single day, inside 16.20's window.

**`split` is not optional.** The one shared split is 42,145 / 4,763 / 7 games.
An agent allowed to query val games contaminates the evaluation that decides
whether the agent works at all, so every table carries `split` and the tool
defaults to `train`.

Encodings are chosen to be hard to misuse: `kind` is 'ban'/'pick' rather than
0/1, `side` is 'blue'/'red' rather than 0/1, and `role` is a word or NULL. A
model that writes `WHERE kind = 1` gets zero rows and a hint, not a plausible
wrong answer.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRO_DIR = ROOT / "data" / "pro play"
DB_PATH = PRO_DIR / "query.sqlite"
NAMES_DIR = ROOT / "data" / "solo queue" / "process matchdata"

ROLES = ("top", "jungle", "mid", "bot", "support")
KINDS = ("ban", "pick")
SIDES = ("blue", "red")

# The floor below which a rate is not reported. Same number as
# `frontend/tools.MIN_CELL`, and it is a documented part of the schema rather
# than a hidden guard: the SQL tool cannot know what an arbitrary aggregate is
# counting, so the *prompt* has to carry the rule.
MIN_CELL = 30

# The one shared split, exactly as `pro/model/llm/common.py: SPLIT` defines it.
# The defaults of `split_within_patch` are NOT these - they draw a fractional
# test tail and take each patch's newest series, which gives 42,134 / 4,774 / 7
# instead of 42,145 / 4,763 / 7. Eleven games is a rounding error and the wrong
# principle: a second split means the agent's `train` rows are not the rows the
# head was trained on, and every comparison between them is quietly off.
SHARED_SPLIT = dict(val_frac=0.10, test_frac=0.0, val_random=True)


# ------------------------------------------------------------------ helpers
def names(n=173):
    """dense champion index -> dDragon name, same table `frontend/tools.py` uses.

    Duplicated rather than imported: `frontend/` is an application, `pro/` is
    the model tree, and an experiment must not make the model tree depend on the
    UI. The two readers are four lines each and the JSON is the contract.
    """
    with open(NAMES_DIR / "champ_mapping.json", encoding="utf-8") as f:
        mapping = {int(k): int(v) for k, v in json.load(f)["riot_to_idx"].items()}
    with open(NAMES_DIR / "champ_names.json", encoding="utf-8") as f:
        riot_to_name = {int(k): v for k, v in json.load(f)["names"].items()}
    out = ["Riot%d" % i for i in range(n)]
    for riot, i in mapping.items():
        if riot in riot_to_name and i < n:
            out[i] = riot_to_name[riot]
    return out


def key(text):
    """Normalised champion key: 'Kai'Sa', 'kaisa' and 'kai sa' all -> 'kaisa'.

    Mirrors `frontend/chat._norm`. The apostrophe is why this exists at all: a
    model writing `WHERE champ = 'Kai'Sa'` produces a SQL syntax error, so the
    schema tells it to join on `champ.key` and this is the key it gets.
    """
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def load_pipeline():
    """The pro pipeline module, which owns the one shared split."""
    path = PRO_DIR / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pro_pipeline_agent", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_rows():
    import torch
    blob = torch.load(PRO_DIR / "pro_drafts.pt", weights_only=False,
                      map_location="cpu")
    return blob["rows"] if isinstance(blob, dict) else blob


# -------------------------------------------------------------------- build
def build(rows, out_path=DB_PATH, known_patches=None, split=None,
          split_kwargs=None):
    """Write the queryable corpus. Returns a stats dict.

    `known_patches` is the checkpoint's `patch_vocab`: anything absent is
    `is_scored = 0`, which is how the agent learns that 16.22 cannot be
    conditioned on. `split` defaults to the shared split, which is
    deterministic in (rows, seed, fractions) - never invent a second one.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    champ_names = names()
    if split is None:
        kw = dict(SHARED_SPLIT if split_kwargs is None else split_kwargs)
        split = load_pipeline().split_within_patch(rows, **kw)
    split_of = {}
    for name, idxs in split.items():
        for i in idxs:
            split_of[i] = name

    # patch_rank is a dense rank over observed patches ordered by their first
    # game, because patch ids are neither contiguous (16.19, 16.21 missing) nor
    # disjoint in time (16.22 sits inside 16.20).
    first_ord, last_ord, n_games = {}, {}, {}
    for r in rows:
        p, o = str(r["patch"]), int(r["date_ordinal"])
        n_games[p] = n_games.get(p, 0) + 1
        first_ord[p] = min(o, first_ord.get(p, o))
        last_ord[p] = max(o, last_ord.get(p, o))
    order = sorted(n_games, key=lambda p: (first_ord[p], p))
    patch_rank = {p: i for i, p in enumerate(order)}
    known = None if known_patches is None else {str(p) for p in known_patches}

    con = sqlite3.connect(str(out_path))
    con.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE champ (
            idx INTEGER PRIMARY KEY, key TEXT NOT NULL, name TEXT NOT NULL);
        CREATE TABLE patch (
            patch TEXT PRIMARY KEY, patch_rank INTEGER NOT NULL,
            n_games INTEGER NOT NULL, first_ord INTEGER NOT NULL,
            last_ord INTEGER NOT NULL, is_scored INTEGER NOT NULL);
        CREATE TABLE game (
            gameid TEXT PRIMARY KEY, patch TEXT NOT NULL,
            patch_rank INTEGER NOT NULL, date_ordinal INTEGER NOT NULL,
            league TEXT, blue_team TEXT, red_team TEXT, blue_win INTEGER,
            first_pick INTEGER, fearless INTEGER, split TEXT NOT NULL);
        CREATE TABLE move (
            gameid TEXT NOT NULL, move_idx INTEGER NOT NULL, kind TEXT NOT NULL,
            side TEXT NOT NULL, champ_idx INTEGER NOT NULL, champ_key TEXT,
            role TEXT, patch TEXT NOT NULL, patch_rank INTEGER NOT NULL,
            date_ordinal INTEGER NOT NULL, league TEXT, blue_win INTEGER,
            split TEXT NOT NULL);
    """)
    con.executemany(
        "INSERT INTO champ (idx, key, name) VALUES (?,?,?)",
        [(i, key(nm), nm) for i, nm in enumerate(champ_names)])
    con.executemany(
        "INSERT INTO patch (patch, patch_rank, n_games, first_ord, last_ord,"
        " is_scored) VALUES (?,?,?,?,?,?)",
        [(p, patch_rank[p], n_games[p], first_ord[p], last_ord[p],
          0 if (known is not None and p not in known) else 1) for p in order])

    games, moves = [], []
    for i, r in enumerate(rows):
        p = str(r["patch"])
        pr, o = patch_rank[p], int(r["date_ordinal"])
        sp = split_of.get(i, "train")
        games.append((str(r["gameid"]), p, pr, o, r.get("league"),
                      r.get("blue_team"), r.get("red_team"),
                      int(r.get("blue_win") or 0), int(r.get("first_pick") or 0),
                      1 if r.get("fearless") else 0, sp))
        for j, mv in enumerate(r["moves"]):
            kind, team, champ, role = mv
            moves.append((str(r["gameid"]), j, KINDS[1 if kind else 0],
                          SIDES[1 if team else 0], int(champ),
                          key(champ_names[int(champ)]),
                          None if role is None or role < 0 else ROLES[role],
                          p, pr, o, r.get("league"), int(r.get("blue_win") or 0),
                          sp))
    con.executemany("INSERT INTO game VALUES (%s)" % ",".join("?" * 11), games)
    con.executemany("INSERT INTO move VALUES (%s)" % ",".join("?" * 13), moves)
    con.executescript("""
        CREATE INDEX move_champ ON move (champ_idx, split);
        CREATE INDEX move_game  ON move (gameid, move_idx);
        CREATE INDEX move_patch ON move (patch_rank, split);
        CREATE INDEX game_patch ON game (patch_rank, split);
    """)
    con.commit()
    con.close()
    return {"db": str(out_path), "games": len(games), "moves": len(moves),
            "patches": len(order), "split": {k: len(v) for k, v in split.items()}}


def _local_copy(src, cache_dir):
    """Copy the corpus next to the process, once per build. Returns the path.

    Not an optimisation in the ordinary sense. The database is on a bind mount,
    and Docker Desktop's gRPC-FUSE charges about a millisecond per random 4 KB
    read - which is every index probe and every table lookup. Six
    representative queries (a first-pick frequency, a role win rate, a
    patch window, a self-join, a ban rate, a role split) took:

        bind mount, as shipped            24.05 s
        container-local disk, same file    0.46 s     52x

    and the SQL is identical, so the 2 s deadline in `sql_tool` was firing on
    queries the corpus answers in milliseconds. Indexes do not fix it: they
    make it worse on the mount, because each hit is a random read (the covering
    index that made one query instant made another 270x slower by attracting a
    plan it was not designed for). On local disk even a full scan is cheap
    enough that no index is worth adding.

    `copy2` preserves the mtime, so the size+mtime check is the staleness test
    and a rebuilt corpus is picked up on the next connection. The copy goes
    through a `.part` file and `os.replace` so two processes starting at once
    cannot read a half-written database.
    """
    dst = Path(cache_dir) / src.name
    try:
        st = src.stat()
        if dst.exists():
            d = dst.stat()
            if d.st_size == st.st_size and d.st_mtime >= st.st_mtime:
                return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return dst
    except OSError:
        # A cache that cannot be written is not a reason to fail a query; the
        # mount is just slower.
        return src


def connect(path=DB_PATH, cache_dir=None):
    """Read-only connection. Writes are impossible, not merely discouraged.

    `cache_dir` (or the `LOL_DB_CACHE` environment variable, which
    docker-compose sets to `/tmp`) opens a local copy - see `_local_copy` for
    why that is 52x on the six queries measured. Unset means open in place,
    which is what the tests and the host-side CLIs do.
    """
    src = Path(path)
    where = os.environ.get("LOL_DB_CACHE") if cache_dir is None else cache_dir
    if where and src.exists():
        src = _local_copy(src, where)
    con = sqlite3.connect("file:%s?mode=ro" % src.as_posix(), uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


# ------------------------------------------------------------------- schema
SCHEMA_DOC = """\
Tables (SQLite, read-only). `split` is 'train' | 'val' | 'test' on every table:
query split='train' unless the question is about evaluation. `date_ordinal` is
the only honest recency axis - patch ids overlap in time and are not contiguous.

champ(idx INT, key TEXT, name TEXT)
    one row per champion; `key` is the normalised name ('kaisa', 'drmundo').
    Join on key, never on name: 'Kai'Sa' breaks a SQL string literal.

patch(patch TEXT, patch_rank INT, n_games INT, first_ord INT, last_ord INT,
      is_scored INT)
    patch_rank is a DENSE rank over observed patches (0 = oldest), so
    patch_rank >= (SELECT max(patch_rank) - 2 FROM patch) is "the last 3
    patches". is_scored = 0 means the model has no embedding for that patch
    and will answer patch-agnostically. All 107 are currently 1 for the
    shipped head; 16.22 is still the one to avoid - it has 5 games, all on
    one day, inside 16.20's window.

game(gameid TEXT, patch, patch_rank, date_ordinal, league, blue_team,
     red_team, blue_win INT 0/1, first_pick INT 0/1, fearless INT 0/1, split)
    one row per draft.

move(gameid, move_idx INT 0..19, kind 'ban'|'pick', side 'blue'|'red',
     champ_idx INT, champ_key TEXT, role TEXT, patch, patch_rank,
     date_ordinal, league, blue_win INT, split)
    one row per ban or pick, in draft order. `role` is
    'top'|'jungle'|'mid'|'bot'|'support' for picks and NULL for bans.
    move() is denormalised: patch, league, date and split are already on the
    row, so most questions need no join at all.
    move_idx is draft order 0..19, NOT a slot: two picks in the same game have
    different move_idx. To relate two champions in one game, join on gameid -
    joining on move_idx returns 0 rows and looks like "they never met".

Rules that make an answer honest:
- Any rate needs its denominator. Select the count next to the rate, and if
  the count is under 30, report it as a count, not a rate.
- A champion's "role" is where it was PLAYED, not where it belongs.
- Win rates here are draft-composition effects in pro play, not causal.

Example: how often does a champion get banned in the last 3 patches?
  SELECT m.champ_key, COUNT(*) n,
         ROUND(AVG(CASE WHEN g.blue_win = 1 THEN 1.0 ELSE 0 END), 3) blue_wr
  FROM move m JOIN game g USING (gameid)
  WHERE m.kind = 'ban' AND m.split = 'train'
    AND m.patch_rank >= (SELECT MAX(patch_rank) - 2 FROM patch)
  GROUP BY m.champ_key ORDER BY n DESC LIMIT 20
"""
