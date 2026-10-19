"""Collect the full match details for every match id gathered in stage 2.

This is stage 3 of the data pipeline:
    stage 1: puuids          (collect_puuids.py)
    stage 2: match ids       (collect_matchids.py)
    stage 3: match details   <- this script

Design notes
------------
- ONE request per match (GET /lol/match/v5/matches/{id}); no timeline calls,
  so the total API cost equals the size of the stage-2 worklist.
- Compact records: each match is reduced AT DOWNLOAD TIME to a ~3 kB record
  of just the fields the auxiliary heads need (the raw DTO is ~50 kB and is
  NOT kept). Per match: match_id, queue_id, gameVersion, start, duration,
  surrender flags, per-team win / kills / deaths / gold / objectives
  (kills + first flags, incl. team dragon kills) / bans, and per participant:
  champion, position, K/D/A, gold, damage, CS, vision, first blood/tower
  flags, plus a `challenges` whitelist (goldPerMinute, damagePerMinute,
  visionScorePerMinute, killParticipation, teamDamagePercentage,
  maxCsAdvantageOnLaneOpponent, maxLevelLeadLaneOpponent,
  earliestTowerTakedown/Dragon/Herald/Baron). NOT collected (would need the
  timeline endpoint, i.e. a second request per match): gold/XP leads at fixed
  minutes - maxCsAdvantageOnLaneOpponent / maxLevelLeadLaneOpponent are the
  peak lane-lead stand-ins. Quality gates (queueId, patch, remakes) remain
  derivable from these fields offline.
- Resumability: on startup every shard is scanned for the match ids already
  stored and only new ids are queued, so re-running after stage 2 has grown
  (e.g. the americas crawl still in flight) automatically picks up the new
  ids. Ctrl+C at any time; a truncated last line in the active shard is
  repaired on the next run. The startup scan re-parses the shards (the shards
  are the single source of truth) and takes a while once the corpus is large.
- Failures: HTTP 404 (match aged out of Riot's storage) is permanent and is
  logged to matchdata/{routing}/gone.txt, skipped on every future run;
  anything else is logged to matchdata/{routing}/failed.txt and retried on
  the next run.
- Routing: match ids are grouped by the routing region of the stage-2 file
  they came from ({routing}_matchids.json) rather than by platform prefix,
  because some prefixes (e.g. ME1) are not in PLATFORM_TO_ROUTING.
- Rate limits: same per-region limiter infrastructure as stage 2
  (HostRateLimiters, auto-tuned from the X-*-Rate-Limit response headers).
"""

import argparse
import asyncio
import gzip
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from riot_api import HostRateLimiters, fetch_json, new_session

SHARD_SIZE = 5000  # matches per shard file
WORKERS_PER_REGION = 16  # concurrent in-flight requests per routing region
PROGRESS_INTERVAL = 30  # seconds between progress lines per region
ROUTINGS = ("americas", "asia", "europe", "sea")

BASE_DIR = Path(__file__).resolve().parent
MATCHIDS_DIR = BASE_DIR / "matchids"
OUT_DIR = BASE_DIR / "matchdata"

SHARD_NAME_RE = re.compile(r"shard_(\d{5})\.jsonl(\.gz)?$")


def load_worklist(routing_filter=None):
    """routing region -> [match ids] from the stage-2 per-routing summaries."""
    workload = {}
    for routing in ROUTINGS:
        if routing_filter and routing not in routing_filter:
            continue
        path = MATCHIDS_DIR / ("%s_matchids.json" % routing)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        workload[routing] = data["match_ids"]
    return workload


def load_gone(region_dir):
    """Match ids whose fetch failed permanently (HTTP 404) in a past run."""
    gone = set()
    path = region_dir / "gone.txt"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    gone.add(line.split("\t")[0])
    return gone


def scan_shard_lines(lines):
    """match ids in shard, number of valid records, end offset after last
    valid line. `lines` yields raw byte lines; parsing stops at the first
    invalid line (the truncated tail a killed run can leave behind).
    """
    ids, valid, end = set(), 0, 0
    offset = 0
    for raw in lines:
        offset += len(raw)
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except (json.JSONDecodeError, UnicodeDecodeError):
            break  # truncated last line from a killed run; drop it
        match_id = record.get("match_id") or (record.get("metadata") or {}).get("matchId")
        if match_id:
            ids.add(match_id)
        valid += 1
        end = offset
    return ids, valid, end


def compress_shard(plain_path):
    """Gzip a full plain shard (atomic: write .tmp, replace, drop the plain)."""
    gz_path = plain_path.with_name(plain_path.name + ".gz")
    tmp_path = plain_path.with_name(plain_path.name + ".gz.tmp")
    with open(plain_path, "rb") as src, gzip.open(tmp_path, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp_path, gz_path)
    plain_path.unlink()


def load_stored(region_dir):
    """Scan existing shards: repair, finish interrupted rotations, index them.

    Returns (stored match ids, active plain shard path or None, matches
    already in the active shard, index for the next shard file).
    """
    stored = set()
    active_path, active_count = None, 0
    max_index = -1
    candidates = []  # plain shards with room left
    if region_dir.exists():
        for path in sorted(region_dir.iterdir()):
            m = SHARD_NAME_RE.match(path.name)
            if not m:
                continue
            index = int(m.group(1))
            max_index = max(max_index, index)
            if m.group(2):  # .jsonl.gz - complete shard
                try:
                    with gzip.open(path, "rb") as f:
                        ids, _, _ = scan_shard_lines(f)
                    stored |= ids
                except (OSError, EOFError):
                    print(
                        "  WARNING: ignoring unreadable shard %s; its matches "
                        "will be re-fetched into new shards" % path
                    )
                continue
            with open(path, "rb") as f:
                ids, valid, end = scan_shard_lines(f)
            stored |= ids
            size = path.stat().st_size
            if end < size:  # repair truncated tail from a killed run
                with open(path, "r+b") as f:
                    f.truncate(end)
            if valid >= SHARD_SIZE:  # finish a rotation interrupted mid-run
                compress_shard(path)
            else:
                candidates.append((index, path, valid))
    if candidates:
        # Normally there is exactly one plain shard (the active one); any
        # older ones with room left get promoted to complete first.
        for _, path, _ in sorted(candidates)[:-1]:
            compress_shard(path)
        active_path, active_count = candidates[-1][1], candidates[-1][2]
    return stored, active_path, active_count, max_index + 1


class RegionState:
    """Shard bookkeeping for one routing region (append-only on disk)."""

    def __init__(self, routing):
        self.routing = routing
        self.region_dir = OUT_DIR / routing
        (
            self.stored,
            self.active_path,
            self.active_count,
            self.next_index,
        ) = load_stored(self.region_dir)
        self.fetched = 0
        self.failed = 0
        self.gone = 0
        self.lock = asyncio.Lock()

    async def record(self, match_id, match):
        async with self.lock:
            if self.active_path is None or self.active_count >= SHARD_SIZE:
                self._rotate()
            with open(self.active_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(match, separators=(",", ":")) + "\n")
            self.active_count += 1
            self.stored.add(match_id)
            self.fetched += 1

    def _rotate(self):
        self.region_dir.mkdir(parents=True, exist_ok=True)
        if self.active_path is not None and self.active_path.exists():
            compress_shard(self.active_path)
        self.active_path = self.region_dir / ("shard_%05d.jsonl" % self.next_index)
        self.next_index += 1
        self.active_count = 0

    async def record_failure(self, match_id, error, permanent):
        async with self.lock:
            self.region_dir.mkdir(parents=True, exist_ok=True)
            name = "gone.txt" if permanent else "failed.txt"
            with open(self.region_dir / name, "a", encoding="utf-8") as f:
                f.write("%s\t%s\n" % (match_id, error))
            if permanent:
                self.gone += 1
            else:
                self.failed += 1


# `challenges` whitelist kept per participant - the full dict has ~300 keys
# of mostly noise; missing keys are simply omitted from the record.
CHALLENGE_KEYS = (
    # rate-normalized performance
    "goldPerMinute",
    "damagePerMinute",
    "visionScorePerMinute",
    "killParticipation",
    "teamDamagePercentage",
    # lane dominance proxies (peak lead over the direct lane opponent; the
    # single-endpoint stand-in for a gold/XP lead at a fixed minute)
    "maxCsAdvantageOnLaneOpponent",
    "maxLevelLeadLaneOpponent",
    # tempo proxies (when the objective fell)
    "earliestTowerTakedown",
    "earliestDragon",
    "earliestHerald",
    "earliestBaron",
)


def extract_player(p):
    """Compact per-participant slice: identity, KDA, economy, tempo flags."""
    ch = p.get("challenges") or {}
    return {
        "pid": p.get("participantId"),
        "team": p.get("teamId"),
        "champ": p.get("championId"),
        "pos": p.get("individualPosition"),
        "kills": int(p.get("kills") or 0),
        "deaths": int(p.get("deaths") or 0),
        "assists": int(p.get("assists") or 0),
        "gold": int(p.get("goldEarned") or 0),
        "dmg": int(p.get("totalDamageDealtToChampions") or 0),
        "cs": int(p.get("totalMinionsKilled") or 0),
        "vision": int(p.get("visionScore") or 0),
        "fb_kill": 1 if p.get("firstBloodKill") else 0,
        "fb_assist": 1 if p.get("firstBloodAssist") else 0,
        "ft_kill": 1 if p.get("firstTowerKill") else 0,
        "ft_assist": 1 if p.get("firstTowerAssist") else 0,
        "ch": {k: ch[k] for k in CHALLENGE_KEYS if k in ch},
    }


def extract(match):
    """Reduce a full Match-V5 DTO to the compact record written to the shards.

    Outcome, duration, team K/D + objectives/bans, per-champion economy,
    lane-dominance and rate-normalized proxies, tempo flags - see the module
    docstring for what is deliberately NOT kept. Raises if the DTO has no
    participants (recorded as a failed fetch and retried next run).
    """
    info = match.get("info") or {}
    parts = info.get("participants") or []
    if not parts:
        raise ValueError("match DTO has no info.participants")
    start = int(info.get("gameStartTimestamp") or 0)
    end = int(info.get("gameEndTimestamp") or 0)
    duration = (end - start) // 1000 if start and end else int(info.get("gameDuration") or 0)

    teams = {}
    for team in info.get("teams") or []:
        side = "blue" if team.get("teamId") == 100 else "red"
        bans = team.get("bans") or []
        teams[side] = {
            "win": 1 if team.get("win") else 0,
            "obj": {
                key: [int(o.get("kills") or 0), 1 if o.get("first") else 0]
                for key, o in (team.get("objectives") or {}).items()
            },
            "bans": [
                b.get("championId")
                for b in sorted(bans, key=lambda b: b.get("pickTurn") or 0)
            ],
        }
    for side, team_id in (("blue", 100), ("red", 200)):
        t = teams.setdefault(side, {"win": 0, "obj": {}, "bans": []})
        mine = [p for p in parts if p.get("teamId") == team_id]
        t["kills"] = sum(int(p.get("kills") or 0) for p in mine)
        t["deaths"] = sum(int(p.get("deaths") or 0) for p in mine)
        t["gold"] = sum(int(p.get("goldEarned") or 0) for p in mine)

    return {
        "match_id": None,  # set by the worker to the requested id
        "queue_id": info.get("queueId"),
        "version": info.get("gameVersion"),
        "start": start // 1000 if start else 0,
        "duration": int(duration),
        "surrender": any(bool(p.get("gameEndedInSurrender")) for p in parts),
        "early_surrender": any(bool(p.get("gameEndedInEarlySurrender")) for p in parts),
        "teams": teams,
        "players": [extract_player(p) for p in parts],
    }


async def fetch_match(session, limiters, routing, match_id):
    """One full Match-V5 DTO."""
    url = "https://%s.api.riotgames.com/lol/match/v5/matches/%s" % (routing, match_id)
    return await fetch_json(session, limiters, routing, url)


async def worker(session, limiters, state, queue):
    while True:
        try:
            match_id = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            match = await fetch_match(session, limiters, state.routing, match_id)
            payload = extract(match)
            payload["match_id"] = match_id
            await state.record(match_id, payload)
        except Exception as exc:  # noqa: BLE001 - keep crawling, retry next run
            permanent = "HTTP 404" in str(exc)
            await state.record_failure(match_id, str(exc), permanent)


async def progress_loop(state, total, stop, start):
    while True:
        try:
            await asyncio.wait_for(stop.wait(), PROGRESS_INTERVAL)
        except asyncio.TimeoutError:
            pass
        else:
            return  # stop was set; exit immediately
        elapsed = time.monotonic() - start
        rate = state.fetched / elapsed if elapsed > 0 else 0.0
        remaining = total - state.fetched
        eta_h = remaining / rate / 3600.0 if rate > 0 else float("inf")
        print(
            "  %-9s %d/%d queued matches stored (%.1f/s, eta %.1fh), "
            "%d failed, %d gone"
            % (state.routing, state.fetched, total, rate, eta_h,
               state.failed, state.gone)
        )


async def run_region(session, limiters, routing, match_ids, workers, limit):
    state = RegionState(routing)
    gone = load_gone(state.region_dir)

    queue = asyncio.Queue()
    queued = 0
    for match_id in match_ids:
        if match_id in state.stored or match_id in gone:
            continue
        queue.put_nowait(match_id)
        queued += 1
        if limit and queued >= limit:
            break

    print(
        "%-9s %d matches (%d already stored, %d gone, %d queued%s) - %d workers"
        % (routing, len(match_ids), len(state.stored), len(gone), queued,
           ", limited" if limit else "", workers)
    )
    if queued == 0:
        return state

    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(worker(session, limiters, state, queue))
        for _ in range(workers)
    ]
    progress = asyncio.create_task(progress_loop(state, queued, stop, time.monotonic()))
    try:
        await asyncio.gather(*tasks)
    finally:
        stop.set()
        try:
            await progress
        except asyncio.CancelledError:
            pass
    print(
        "  %-9s finished: %d fetched, %d stored total, %d failed, %d gone"
        % (routing, state.fetched, len(state.stored), state.failed, state.gone)
    )
    return state


def write_manifest(states):
    """Best-effort summary of what is on disk (rewritten after every run)."""
    OUT_DIR.mkdir(exist_ok=True)
    data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "shard_size": SHARD_SIZE,
        "routing": {
            routing: {
                "stored_total": len(state.stored),
                "fetched_this_run": state.fetched,
                "failed_this_run": state.failed,
                "gone_this_run": state.gone,
            }
            for routing, state in sorted(states.items())
        },
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def main(routing_filter, workers, limit):
    OUT_DIR.mkdir(exist_ok=True)
    workload = load_worklist(routing_filter)
    if not workload:
        print("Nothing to do (no stage-2 matchids files matched).")
        return

    limiters = HostRateLimiters()
    states = {}
    async with new_session() as session:
        region_tasks = {
            routing: asyncio.create_task(
                run_region(session, limiters, routing, ids, workers, limit)
            )
            for routing, ids in workload.items()
        }
        try:
            for routing, task in region_tasks.items():
                states[routing] = await task
        finally:
            # Persist progress for every region that completed, even on Ctrl+C;
            # regions still running keep their progress in the shard files.
            for routing, task in region_tasks.items():
                if task.done() and not task.cancelled() and task.exception() is None:
                    states[routing] = task.result()
            if states:
                write_manifest(states)

    print("-" * 55)
    for routing in sorted(states):
        state = states[routing]
        print(
            "  %-9s %d stored (%d new this run, %d failed, %d gone)"
            % (routing, len(state.stored), state.fetched, state.failed, state.gone)
        )
    print("-" * 55)
    print(
        "total: %d matches stored under %s"
        % (sum(len(s.stored) for s in states.values()), OUT_DIR)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        default="",
        help="comma-separated routing regions to collect (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="max NEW matches to fetch per region (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS_PER_REGION,
        help="concurrent requests per routing region (default %(default)s)",
    )
    args = parser.parse_args()
    regions = None
    if args.regions:
        regions = {r.strip().lower() for r in args.regions.split(",") if r.strip()}
        unknown = regions - set(ROUTINGS)
        if unknown:
            parser.error("unknown routing regions: %s" % ", ".join(sorted(unknown)))
    asyncio.run(main(regions, args.workers, args.limit or None))

