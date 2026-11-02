"""Pro drafts pipeline: OraclesElixir CSVs -> cached, model-ready rows.

Input: `2026_LoL_esports_match_data_from_OraclesElixir.csv` and the 2022-
2025 year-slice siblings in this folder (12 rows per game: 10 player rows
with position+champion, 2 team rows with ban1-5/pick1-5/firstPick/
result; no gameid overlap between files, verified). Champion names reuse
the SoloQ mapping (`../solo queue/process matchdata/champ_mapping.json` +
`champ_names.json`) so pro and soloQ share one champion index space
(verified 173/173 coverage in every file, incl. 2022's smaller pool).

Era rules (data-verified, see test_pipeline.py):

- 2026 "First Selection" (all regions): the team with side-selection
  privilege chooses EITHER side OR first pick; the opponent gets the
  other. First pick is decoupled from side (`firstPick` column: 1 for
  the first-picking team's rows). Pre-2026, blue side always
  first-picked, so empty firstPick defaults to blue for <2026 games
  (~20% of 2022-24 rows lack the column value). The global 20-move
  draft sequence is RECONSTRUCTED from per-side lists:

    bans  phase 1:  FP SP FP SP FP SP
    picks phase 1:  FP SP SP FP FP SP
    bans  phase 2:  SP FP SP FP
    picks phase 2:  SP FP FP SP

  Bans start with the first-pick team (FP) in phase 1 and the
  second-pick team (SP) in phase 2. Picks start with FP in phase 1
  and SP in phase 2. For pre-2026 drafts (first_pick==0), blue-side
  is always FP so blue bans/picks first in phase 1 and red first in
  phase 2, recovering the classic order. For 2026 red-first-pick
  games, red bans/picks first in phase 1 and blue in phase 2.

- Fearless Draft (re-picks forbidden within a series) started 2025:
  later-game re-pick rate ~85-97% (2022-24), ~1.4% (2025), 0% (2026).
  `unavailable` (champs picked in earlier games of the series) is
  attached only for 2025+ rows; rows carry a `fearless` flag.

Series: gameids lack `_game_N` for most leagues, so series are
reconstructed from (league, sorted(teams), day), sorted by datetime,
split whenever the `game` column restarts at 1. Validated on 2026 data:
4,499 series of size 1-5 with 0 fearless violations.

A draft row carries: 10 slot champs/roles in SoloQ contract order
[B_TOP..B_SUP, R_TOP..R_SUP] (roles from player rows - exact, no flex
guessing), the interleaved 20 moves, fearless `unavailable` champs,
side/first-pick/result metadata, and a recency weight.

Caveats: incomplete drafts (missing picks/bans) are skipped with a
reason code; series spanning >1 day or with dropped games lose fearless
context; global move order is reconstructed, not observed.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import os


import torch

csv.field_size_limit(10 ** 7)  # OE rows are wide (165 cols)

N_SLOTS = 10
TEAM_SIZE = 5
N_MOVES = 20  # 10 bans + 10 picks
NUM_ROLES = 5
MOVE_BAN = 0
MOVE_PICK = 1
IGNORE_INDEX = -100

# position (OE lowercase) -> role id, matching SoloQ ROLE_TO_ID
POS_TO_ROLE = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
ROLE_NAMES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

BASE_DIR = Path(__file__).resolve().parent          # .../data/pro play
SOLOQ_PROC_DIR = BASE_DIR.parent / "solo queue" / "process matchdata"
MAPPING_PATH = SOLOQ_PROC_DIR / "champ_mapping.json"
CHAMP_NAMES_PATH = SOLOQ_PROC_DIR / "champ_names.json"
CSV_GLOB_ALL = str(BASE_DIR / "*OraclesElixir*.csv")
# The corpus every recorded number is measured on (STATUS.md: 46,813 drafts).
# Pinned to 2022+ on purpose: the folder also holds 2014-2021 exports, and a
# bare "*" glob would silently swap the corpus out from under every existing
# script the moment those files appear. The wider set is opt-in via
# CSV_GLOB_ALL (81,161 drafts, 2017-2026).
CSV_GLOB = str(BASE_DIR / "202[2-9]_*OraclesElixir*.csv")
CACHE_PATH = BASE_DIR / "pro_drafts.pt"
# Cache for the opt-in wider corpus (CSV_GLOB_ALL). Separate file, because a
# single cache keyed by CSV signature would rescan back and forth every time a
# script alternates between the two corpora.
CACHE_PATH_ALL = BASE_DIR / "pro_drafts_all.pt"
CACHE_VERSION = 3  # v3: prior_picks + data-driven fearless flag

# Corpus freeze. Every number in STATUS.md and FINDINGS.md is measured on the
# drafts that existed on 2026-10-24 23:19:46, which is also the last date in
# both caches. The OE exports keep growing, and a rebuild would otherwise fold
# the 2026 Worlds knockout stage - including the BLG vs HLE final that
# model/holdout/ uses as a holdout - into the training corpus. That invalidates
# every recorded number and destroys the holdout at the same time, silently.
# Moving this date is a deliberate act: raise it, then re-measure everything.
CORPUS_MAX_DATE = "2026-10-24 23:19:46"

# Fearless Draft era (re-picks forbidden across a series): 2025 season on.
# Real data: later-series-game re-pick rate ~95% (2022), ~97% (2023),
# ~85% (2024), ~1.4% (2025), 0% (2026) -> gate on calendar date.
FEARLESS_MIN_ORDINAL = datetime(2025, 1, 1).toordinal()
# First Selection (first pick decoupled from side) started 2026; before
# that, blue side always first-picked, so empty firstPick defaults to blue.
FIRST_SELECTION_MIN_YEAR = 2026

# aux economy columns (per-side values; red side read via opp_ mirror)
AUX_BASE_COLS = ("goldat10", "goldat15", "xpat10", "killsat10")
# econ regression targets, in head order: per-side raw values

# aux economy columns (per-side values; red side read via opp_ mirror)
AUX_BASE_COLS = ("goldat10", "goldat15", "xpat10", "killsat10")
# econ regression targets, in head order: per-side raw values
ECON_KEYS = ("gold10_blue", "gold10_red", "gold15_blue", "gold15_red",
             "xp10_blue", "xp10_red")
DEFAULT_HALF_LIFE_DAYS = 75.0  # recency weight: 2^(-age/half_life)


def patch_key(patch):
    """Normalize a patch label to 'major.mm' ('16.3' -> '16.03').

    Lives here rather than in datasets.py because the split needs the same
    notion of "one patch" that the patch vocabulary uses: two spellings of
    the same patch must not end up on opposite sides of a within-patch
    boundary. datasets.py re-exports this (P.patch_key) so there is one
    definition.
    """
    s = str(patch or "").strip()
    if not s:
        return ""
    maj, _, mnr = s.partition(".")
    if not mnr:
        mnr = "0"
    try:
        return "%d.%02d" % (int(maj), int(mnr))
    except ValueError:
        return s


def patch_sort_key(key):
    """Chronological sort key for a normalised patch label."""
    maj, _, mnr = key.partition(".")
    try:
        return (int(maj), int(mnr))
    except ValueError:
        return (10 ** 6, 0)


def load_name_to_idx(mapping_path=MAPPING_PATH):
    """Champion display name -> contiguous idx via the SoloQ mapping.

    champ_names.json maps riot_id -> Data Dragon display name (unique per
    champ), champ_mapping.json maps riot_id -> idx. Returns
    (name_to_idx {str: int}, num_champs).
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    riot_to_idx = {int(k): int(v) for k, v in payload["riot_to_idx"].items()}
    with open(CHAMP_NAMES_PATH, "r", encoding="utf-8") as f:
        riot_to_name = {int(k): v for k, v in json.load(f)["names"].items()}
    name_to_idx = {}
    for riot, idx in riot_to_idx.items():
        name = riot_to_name.get(riot)
        if name is not None:
            name_to_idx[name] = idx
    return name_to_idx, len(riot_to_idx)


def list_csvs(csv_glob=CSV_GLOB):
    """All OraclesElixir CSVs (2022-2026), sorted by filename."""
    paths = sorted(glob(csv_glob))
    if not paths:
        raise FileNotFoundError("no OraclesElixir CSV under %s" % csv_glob)
    return [Path(p) for p in paths]


def iter_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            yield row


def group_games(csv_paths):
    """One pass over CSV(s) -> list of raw game dicts (12-row blocks).

    `csv_paths` is a single path or a list. Files are scanned in order;
    a gameid seen in an earlier file is skipped (files are year-sliced
    OE exports with no observed overlap, but the guard is cheap).
    """
    if not isinstance(csv_paths, (list, tuple)):
        csv_paths = [csv_paths]
    games, seen = {}, set()
    for csv_path in csv_paths:
        for row in iter_csv_rows(csv_path):
            gid = row.get("gameid") or ""
            if not gid:
                continue
            g = games.get(gid)
            if g is None:
                if gid in seen:
                    continue
                g = games[gid] = {"gameid": gid, "players": [], "blue": None,
                                  "red": None}
            pid = row.get("participantid")
            if pid == "100":
                g["blue"] = row
            elif pid == "200":
                g["red"] = row
            else:
                g["players"].append(row)
        seen.update(games)
    return list(games.values())


def _parse_date(s):
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def assign_series(games):
    """Attach (series_id, game_number) to each game dict in place.

    Series are grouped by (league, sorted(teams), day); within a group,
    games sorted by datetime are split into series whenever the `game`
    column restarts at 1. Games missing a team row get series None.
    """
    groups = defaultdict(list)
    for g in games:
        blue, red = g["blue"], g["red"]
        if blue is None or red is None:
            g["series_id"], g["game_number"] = None, 1
            continue
        dt = _parse_date(blue.get("date") or "")
        if dt is None:
            g["series_id"], g["game_number"] = None, 1
            continue
        g["_dt"] = dt
        try:
            gnum = int(blue.get("game") or 1)
        except ValueError:
            gnum = 1
        g["game_number"] = gnum
        teams = tuple(sorted([blue.get("teamname") or "?",
                              red.get("teamname") or "?"]))
        groups[(blue.get("league") or "?", teams, dt.strftime("%Y-%m-%d"))] \
            .append(g)
    for (league, teams, day), gl in groups.items():
        gl.sort(key=lambda g: (g["_dt"], g["game_number"]))
        open_sid, k = None, 0
        for g in gl:
            if g["game_number"] == 1 or open_sid is None:
                k += 1
                open_sid = "%s|%s|%s|%s|s%d" % (
                    league, teams[0], teams[1], day, k)
            g["series_id"] = open_sid
    for g in games:
        g.pop("_dt", None)
    return games


def interleaved_moves(first_pick, bans_by_team, picks_by_team):
    """Reconstruct the global 20-move order (see module docstring).

    first_pick: 0 (blue) or 1 (red). bans_by_team: {0: [5 idx], 1: [5 idx]}
    in the team's own ban order (ban1..ban5). picks_by_team:
    {0: [(champ, role) x5], 1: [...]} in the team's own pick order
    (pick1..pick5). Returns [(kind, team, champ, role) x20] with
    kind in {MOVE_BAN, MOVE_PICK}; role is IGNORE_INDEX for bans.
    """
    fp = int(first_pick)
    sp = 1 - fp
    ban_i = {0: 0, 1: 0}
    pick_i = {0: 0, 1: 0}
    moves = []

    def take_ban(team):
        c = bans_by_team[team][ban_i[team]]
        ban_i[team] += 1
        moves.append((MOVE_BAN, team, c, IGNORE_INDEX))

    def take_pick(team):
        c, r = picks_by_team[team][pick_i[team]]
        pick_i[team] += 1
        moves.append((MOVE_PICK, team, c, r))

    # bans phase 1: first-pick team bans first
    for team in (fp, sp, fp, sp, fp, sp):
        take_ban(team)
    # picks phase 1: first-pick team picks first
    for team in (fp, sp, sp, fp, fp, sp):
        take_pick(team)
    # bans phase 2: second-pick team bans first
    for team in (sp, fp, sp, fp):
        take_ban(team)
    # picks phase 2: second-pick team starts
    for team in (sp, fp, fp, sp):
        take_pick(team)
    assert len(moves) == N_MOVES
    return moves



def _num(s):
    try:
        return float((s or "").strip())
    except (ValueError, AttributeError):
        return None


def game_aux_targets(game, champ_to_team):
    """Per-side gold/xp at 10/15 min + gamelength from the two team rows.

    The OE columns are written per side with the opponent's numbers in
    `opp_*`; kills are stored as the side's own kill total.
    champ_to_team maps pick champion name -> side (0/1) from player rows.
    Returns ((gold10_blue, gold10_red, gold15_blue, gold15_red,
              xp10_blue, xp10_red), kills10_blue, kills10_red, has_all).
    """
    side_rows = {0: game["blue"], 1: game["red"]}
    gold10, gold15, xp10, kills = {}, {}, {}, {}
    for side, trow in side_rows.items():
        gold10[side] = _num(trow.get("goldat10"))
        gold15[side] = _num(trow.get("goldat15"))
        xp10[side] = _num(trow.get("xpat10"))
        kills[side] = _num(trow.get("killsat10"))
    # mirror check: opp_goldat10 on one side must equal goldat10 opposite
    opp = {0: _num(side_rows[1].get("opp_goldat10")),
           1: _num(side_rows[0].get("opp_goldat10"))}
    for side in (0, 1):
        if opp[side] is not None and gold10[side] is not None and \
                abs(opp[side] - gold10[side]) > 1.0:
            return None, None, False
    econ = (gold10[0], gold10[1], gold15[0], gold15[1], xp10[0], xp10[1])
    has_all = all(v is not None for v in econ)
    return econ, kills, has_all


def game_to_draft(game, name_to_idx):
    """Filter + convert one raw game dict -> (draft dict | None, reason)."""
    blue, red, players = game["blue"], game["red"], game["players"]
    if blue is None or red is None:
        return None, "missing_team_row"
    dt = _parse_date(blue.get("date") or "")
    if dt is None:
        return None, "bad_date"
    fp_blue, fp_red = blue.get("firstPick"), red.get("firstPick")
    if {fp_blue, fp_red} == {"0", "1"}:
        first_pick = 0 if fp_blue == "1" else 1
    elif fp_blue == "" and fp_red == "" and \
            dt.year < FIRST_SELECTION_MIN_YEAR:
        first_pick = 0  # pre-2026: blue side always first-picked
    else:
        return None, "bad_firstpick"
    if {blue.get("result"), red.get("result")} != {"0", "1"}:
        return None, "bad_result"
    blue_win = 1 if blue.get("result") == "1" else 0
    if len(players) != N_SLOTS:
        return None, "missing_players"
    econ, _kills, has_econ = game_aux_targets(game, {})
    # per-side (champ_name -> role) from player rows (exact roles)
    role_of = {0: {}, 1: {}}
    side_of_row = {"Blue": 0, "Red": 1}
    for pr in players:
        side = side_of_row.get(pr.get("side"))
        role = POS_TO_ROLE.get(pr.get("position"))
        champ = pr.get("champion")
        if side is None or role is None or not champ:
            return None, "bad_position"
        role_of[side][champ] = role
    if any(len(role_of[s]) != TEAM_SIZE for s in (0, 1)):
        return None, "bad_position"
    # team rows: bans + picks in the team's own order
    bans_by_team, picks_by_team, all_champs = {0: [], 1: []}, {0: [], 1: []}, []
    for side, trow in ((0, blue), (1, red)):
        for i in range(1, 6):
            bname = (trow.get("ban%d" % i) or "").strip()
            if not bname:
                return None, "incomplete_bans"
            if bname not in name_to_idx:
                return None, "unknown_champ"
            bans_by_team[side].append(name_to_idx[bname])
            all_champs.append(name_to_idx[bname])
        for i in range(1, 6):
            pname = (trow.get("pick%d" % i) or "").strip()
            if not pname:
                return None, "incomplete_picks"
            if pname not in name_to_idx:
                return None, "unknown_champ"
            role = role_of[side].get(pname)
            if role is None:  # pick must appear on a player row of that side
                return None, "pick_mismatch"
            picks_by_team[side].append((name_to_idx[pname], role))
            all_champs.append(name_to_idx[pname])
    if len(set(all_champs)) != N_MOVES:
        return None, "duplicate_champ"
    # canonical slots [B_TOP..B_SUP, R_TOP..R_SUP]
    champ_slot = [None] * N_SLOTS
    role_slot = [None] * N_SLOTS
    for side in (0, 1):
        for cname, role in role_of[side].items():
            slot = side * TEAM_SIZE + role
            champ_slot[slot] = name_to_idx[cname]
            role_slot[slot] = role
    if any(c is None for c in champ_slot):
        return None, "pick_mismatch"
    return {
        "gameid": game["gameid"],
        "series_id": game.get("series_id"),
        "game_number": int(game.get("game_number") or 1),
        "league": blue.get("league") or "?",
        "patch": blue.get("patch") or "?",
        "date": blue.get("date") or "",
        "date_ordinal": dt.toordinal(),
        "blue_team": blue.get("teamname") or "?",
        "red_team": red.get("teamname") or "?",
        "first_pick": first_pick,
        "blue_win": blue_win,
        "champ": champ_slot,
        "role": role_slot,
        "moves": interleaved_moves(first_pick, bans_by_team, picks_by_team),
        "unavailable": [],  # filled by attach_fearless()
        "prior_picks": [],
        "fearless": False,
        "econ": list(econ) if econ is not None else [None] * 6,
        "has_econ": bool(has_econ),
    }, "ok"


def attach_fearless(rows):
    """Fill row['unavailable'] = champs picked in earlier series games.

    Fearless Draft (re-picks forbidden) started with the 2025 season, but
    adoption is per-league (2025: HW/NEXO/LJL mostly non-fearless) and
    series reconstruction is heuristic, so the flag is data-driven:
    a series is fearless iff it is dated >= 2025-01-01 AND contains no
    internal re-pick (any game_number>1 re-picking an earlier series
    champ disqualifies the whole series). Rows keep both the factual
    `prior_picks` (always) and `unavailable` (= prior picks iff
    fearless, else []). Caveat: dropped/incomplete games lose fearless
    context for their series mates.
    """
    by_series = defaultdict(list)
    for row in rows:
        if row["series_id"] is not None:
            by_series[row["series_id"]].append(row)
        else:
            row["fearless"] = False
            row["prior_picks"] = []
            row["unavailable"] = []
    for series_rows in by_series.values():
        series_rows.sort(key=lambda r: r["game_number"])
        seen = set()
        violation = False
        for row in series_rows:
            if row["game_number"] > 1 and (set(row["champ"]) & seen):
                violation = True
            seen.update(row["champ"])
        modern = max(r["date_ordinal"] for r in series_rows) >= \
            FEARLESS_MIN_ORDINAL
        fearless = modern and not violation
        seen = set()
        for row in series_rows:
            row["fearless"] = fearless
            row["prior_picks"] = sorted(seen)
            row["unavailable"] = sorted(seen) if fearless else []
            seen.update(row["champ"])
    return rows


def _game_date(g):
    """The date of a raw game dict, parsed the same way assign_series does."""
    blue = g.get("blue") or {}
    return _parse_date(blue.get("date") or "")


def scan_and_index(csv_paths, name_to_idx, limit=0):
    """One full pass over CSV(s) -> {'rows', 'stats', 'csvs'}."""
    if not isinstance(csv_paths, (list, tuple)):
        csv_paths = [csv_paths]
    games = assign_series(group_games(csv_paths))
    total_games = len(games)
    if CORPUS_MAX_DATE:
        # Freeze the corpus: anything newer than the pin is held out, not
        # trained on. Counted as a skip reason so the stats identity holds.
        cutoff = _parse_date(CORPUS_MAX_DATE)
        games = [g for g in games
                 if (_game_date(g) is None or _game_date(g) <= cutoff)]
    rows, reasons = [], Counter()
    for g in games:
        out, reason = game_to_draft(g, name_to_idx)
        if out is None:
            reasons[reason] += 1
        else:
            out.setdefault("fearless", False)
            rows.append(out)
        if limit and (len(rows) + sum(reasons.values())) >= limit:
            break
    if total_games != len(games):
        reasons["after_corpus_freeze"] = total_games - len(games)
    attach_fearless(rows)
    stats = {"total_games": total_games, "kept": len(rows),
             "skipped": sum(reasons.values()),
             "skip_reasons": dict(reasons)}
    return {"rows": rows, "stats": stats,
            "csvs": [Path(p).name for p in csv_paths]}


def _csv_signature(csv_paths):
    """Cheap content signature for cache validation.

    Names alone are not sufficient: the current-season export is appended to
    during the split, so a cache built mid-season keeps serving a truncated
    corpus. That is exactly what happened -- the cache held games to
    2026-10-12 while the 2026 CSV had games to 2026-10-24, and nothing
    noticed because the filename never changed.
    """
    sig = []
    for p in csv_paths:
        st = p.stat()
        sig.append([p.name, int(st.st_size), int(st.st_mtime)])
    return sig


def build_rows(csv_glob=CSV_GLOB, rescan=False, limit=0, cache_path=None):
    """Cached scan: loads pro_drafts.pt unless rescan/limit/missing."""
    cache_path = Path(cache_path) if cache_path is not None else CACHE_PATH
    csv_paths = list_csvs(csv_glob)
    names = [p.name for p in csv_paths]
    sig = _csv_signature(csv_paths)
    if not rescan and not limit and cache_path.exists():
        payload = torch.load(str(cache_path), map_location="cpu",
                             weights_only=False)  # our own cache
        if payload.get("version") == CACHE_VERSION and \
                payload.get("csv_sig") == sig:
            print("loaded cache: %d drafts (%s)" %
                  (len(payload["rows"]), cache_path.name))
            return payload["rows"], payload["stats"]
        if payload.get("csv_sig") is None:
            print("cache predates content signatures; rescanning")
        else:
            print("csvs changed since the cache was built; rescanning")
    print("scanning %d CSVs: %s ..." % (len(csv_paths),
                                        ", ".join(names)))
    payload = scan_and_index(csv_paths, load_name_to_idx()[0], limit=limit)
    if not limit:
        torch.save(dict(payload, version=CACHE_VERSION, csv_sig=sig),
                   str(cache_path))
        print("cached %d drafts -> %s" % (len(payload["rows"]),
                                          cache_path.name))
    print("kept %d / %d games (skipped %d: %s)" % (
        payload["stats"]["kept"], payload["stats"]["total_games"],
        payload["stats"]["skipped"], payload["stats"]["skip_reasons"]))
    return payload["rows"], payload["stats"]


def _merge_patch_weights():
    """Return {patch: factor} map; empty unless a weights file exists.

    Looks for `patch_weights.json` next to the CSVs. If absent, returns
    {} so all patches get the default factor of 1.0.
    """
    wp = BASE_DIR / "patch_weights.json"
    if wp.exists():
        with open(wp, "r", encoding="utf-8") as f:
            return {k: float(v) for k, v in json.load(f).items()}
    return {}


def assign_weights(rows, half_life_days=DEFAULT_HALF_LIFE_DAYS, use_recency=True, patch_weights=None):
    """Set row['weight'] = recency x patch_factor for training sampling.

    Patches not listed get 1.0. `half_life_days <= 0` disables recency.
    """
    if patch_weights is None:
        patch_weights = _merge_patch_weights()
    if half_life_days <= 0 or not rows:
        for row in rows:
            row["weight"] = patch_weights.get(row.get("patch", "?"), 1.0)
        return rows
    newest = max(row["date_ordinal"] for row in rows)
    raw = []
    for row in rows:
        pw = patch_weights.get(row.get("patch", "?"), 1.0)
        if use_recency:
            w = pw * 2.0 ** (-(newest - row["date_ordinal"]) / half_life_days)
        else:
            w = pw
        raw.append(w)
    mean = sum(raw) / len(raw)
    for row, w in zip(rows, raw):
        row["weight"] = w / mean
    return rows


def split_series_by_date(rows, val_frac=0.05, test_frac=0.05):
    """Train/val/test index split by SERIES in date order (no leakage).

    Series (rows sharing series_id; null-series rows each form a singleton)
    are sorted by their earliest game date; the last `test_frac` of series
    go to test, the preceding `val_frac` to val. Whole series always stay
    together so fearless context and team meta never cross splits.
    """
    series = defaultdict(list)
    for i, row in enumerate(rows):
        key = row["series_id"] if row["series_id"] is not None else \
            "__null_%d" % i
        series[key].append(i)
    ordered = sorted(series.values(),
                     key=lambda idx: (rows[idx[0]]["date_ordinal"],
                                      rows[idx[0]]["gameid"]))
    n = len(ordered)
    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
    test_series = ordered[n - n_test:] if n_test else []
    val_series = ordered[n - n_test - n_val:n - n_test] if n_val else []
    train_series = ordered[:n - n_test - n_val]
    return {"train": [i for s in train_series for i in s],
            "val": [i for s in val_series for i in s],
            "test": [i for s in test_series for i in s]}


def split_within_patch(rows, val_frac=0.10, test_frac=0.10,
                       min_patch_rows=40, verbose=False):
    """Train/val/test by SERIES *inside* each patch: head -> train, tail -> val/test.

    Why this split exists, next to `split_series_by_date`. The global date
    split puts whole patches on one side of the boundary, so a test row's
    patch never appears in train. That is the honest protocol for "will this
    work on the next patch", but it makes patch conditioning untestable by
    construction: every test row's patch id is UNK, so a patch embedding can
    only memorise. Here each patch is on both sides - the first
    (1 - val - test) of its series train, the last val/test - so the patch id
    is a *seen* input everywhere and the question becomes whether the prior
    learned from the head of a patch transfers to its tail.

    Ordering is by series, not row: series (rows sharing series_id;
    null-series rows are singletons) are sorted by earliest
    (date_ordinal, gameid) and assigned whole, so fearless context never
    straddles a split. The tail is filled from the end backwards, so test is
    always the last games of the patch, val the games just before it, and a
    series is never split even when that overshoots the target fraction.

    A series is assigned to the patch of its *first* game before any of this.
    Without that step a series played across a patch boundary (game 1 on
    16.20, game 2 on 16.21) would be the head of one patch and the tail of
    another - i.e. a fearless context split across train and test, which is
    exactly the leakage the series grouping exists to prevent.

    `min_patch_rows`: patches smaller than this stay entirely in train. Their
    tail would be 1-2 games, which is noise rather than a measurement; on
    this corpus that is 7 patches / 194 rows (0.4%) at the default.

    Returns {"train": [...], "val": [...], "test": [...]} index lists. A
    patch below `min_patch_rows` contributes all of its rows to train and
    nothing to val/test.
    """
    series = defaultdict(list)
    for i, row in enumerate(rows):
        key = row["series_id"] if row["series_id"] is not None \
            else "__null_%d" % i
        series[key].append(i)

    by_patch = defaultdict(list)     # patch -> [series (list of row indices)]
    moved = 0
    for idxs in series.values():
        idxs = sorted(idxs, key=lambda i: (rows[i]["date_ordinal"],
                                           rows[i].get("game_number") or 0,
                                           rows[i]["gameid"]))
        first_patch = patch_key(rows[idxs[0]].get("patch"))
        by_patch[first_patch].append(idxs)
        moved += sum(1 for i in idxs
                     if patch_key(rows[i].get("patch")) != first_patch)

    out = {"train": [], "val": [], "test": []}
    report = []
    for patch in sorted(by_patch, key=patch_sort_key):
        ordered = sorted(by_patch[patch],
                         key=lambda s: (rows[s[0]]["date_ordinal"],
                                        rows[s[0]]["gameid"]))
        idxs = [i for s in ordered for i in s]
        n = len(idxs)
        if n < min_patch_rows or test_frac <= 0:
            out["train"].extend(idxs)
            report.append((patch, n, n, 0, 0))
            continue
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test_series, val_series, acc = [], [], 0
        cut = len(ordered)
        for j in range(len(ordered) - 1, -1, -1):
            if acc >= n_test:
                break
            test_series.insert(0, ordered[j])
            acc += len(ordered[j])
            cut = j
        acc = 0
        cut_val = cut
        for j in range(cut - 1, -1, -1):
            if acc >= n_val:
                break
            val_series.insert(0, ordered[j])
            acc += len(ordered[j])
            cut_val = j
        train_series = ordered[:cut_val]
        tr = [i for s in train_series for i in s]
        va = [i for s in val_series for i in s]
        te = [i for s in test_series for i in s]
        out["train"].extend(tr)
        out["val"].extend(va)
        out["test"].extend(te)
        report.append((patch, n, len(tr), len(va), len(te)))

    if verbose:
        small = [r for r in report if r[3] == 0 and r[4] == 0]
        print("within-patch split: %d patches, %d train / %d val / %d test"
              % (len(report), len(out["train"]), len(out["val"]),
                 len(out["test"])))
        if moved:
            print("  %d rows follow their series into an earlier patch "
                  "(series spanning a patch boundary)" % moved)
        if small:
            print("  %d patches below min_patch_rows stay in train: %s"
                  % (len(small), ", ".join("%s(%d)" % (p, n)
                                           for p, n, *_ in small[:8])))
    return out


def compute_popularity_baseline(rows, indices, num_champs):
    """Weighted popularity orderings fit on `indices` only (train split).

    Returns pick global top-k, per-role pick top-k, and ban top-k, plus
    the raw weighted counts. Weights come from row['weight'] (recency).
    """
    pick_glob = torch.zeros(num_champs)
    pick_role = torch.zeros(NUM_ROLES, num_champs)
    ban_glob = torch.zeros(num_champs)
    for i in indices:
        row = rows[i]
        w = float(row.get("weight", 1.0))
        for slot in range(N_SLOTS):
            pick_glob[row["champ"][slot]] += w
            pick_role[row["role"][slot], row["champ"][slot]] += w
        for kind, _team, champ, _role in row["moves"]:
            if kind == MOVE_BAN:
                ban_glob[champ] += w

    def order(counts):
        return sorted(range(num_champs),
                      key=lambda c: (-float(counts[c]), c))

    return {"pick_global_counts": pick_glob,
            "pick_role_counts": pick_role,
            "ban_counts": ban_glob,
            "pick_global_top": order(pick_glob),
            "pick_role_top": [order(pick_role[r]) for r in range(NUM_ROLES)],
            "ban_top": order(ban_glob)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pro drafts pipeline (OE CSV)")
    ap.add_argument("--rescan", action="store_true",
                    help="ignore cache, re-scan the CSV")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap games (smoke; skips cache write)")
    ap.add_argument("--patch-weight", nargs=2, metavar=("PATCH", "FACTOR"),
                    action="append", default=[],
                    help="e.g. --patch-weight 16.20 5 (repeatable): extra "
                         "training weight for a patch")
    ap.add_argument("--no-recency", action="store_true",
                    help="disable recency decay in weights")
    ap.add_argument("--half-life", type=float,
                    default=DEFAULT_HALF_LIFE_DAYS)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    args = ap.parse_args(argv)

    rows, stats = build_rows(rescan=args.rescan, limit=args.limit)
    assign_weights(rows, args.half_life, not args.no_recency,
                   {p: float(f) for p, f in args.patch_weight})
    splits = split_series_by_date(rows, args.val_frac, args.test_frac)
    patches = Counter(row["patch"] for row in rows)
    print("weights: patch factors %s, recency %s" %
          (args.patch_weight, not args.no_recency))
    print("splits: train=%d val=%d test=%d (games)" % tuple(
        len(splits[k]) for k in ("train", "val", "test")))
    print("fearless rows (game_number>1 with prior picks): %d" %
          sum(1 for r in rows if r["game_number"] > 1 and r["unavailable"]))
    print("patches:", dict(sorted(patches.items())))


if __name__ == "__main__":
    main()

