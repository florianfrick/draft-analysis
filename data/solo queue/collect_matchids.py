"""Collect ranked solo-queue match IDs played since 2026-10-07 for every
high-elo puuid gathered in stage 1.

This is stage 2 of the data pipeline:
    stage 1: puuids          (collect_puuids.py)
    stage 2: match ids       <- this script
    stage 3: match details   (collect_match_data.py)

Design notes
------------
- Riot's application rate limit is per REGIONAL routing host (americas,
  asia, europe, sea), so all four routing regions crawl IN PARALLEL, each
  with its own independent limiter. Observed live limits for a dev key:
  20 req/1s + 100 req/2min (the matchlist method limit, 2000/10min, never
  binds). The limiters auto-tune from the X-*-Rate-Limit response headers,
  so this also works unchanged with a production key.
- Pagination: /matches/by-puuid/{puuid}/ids returns at most 100 ids per
  page; we page with start/count until a short page comes back.
- Resumability: each finished puuid is appended as one JSON line to
  matchids/raw/{routing}.jsonl. On restart, finished puuids are skipped
  and their ids still count toward the totals; puuids that failed are
  retried. You can Ctrl+C at any time and just re-run.
- Output: matchids/{routing}_matchids.json - sorted unique match ids per
  routing region (rewritten at the end of every run). Failed puuids are
  logged to matchids/{routing}_failed.txt.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from riot_api import PLATFORM_TO_ROUTING, HostRateLimiters, fetch_json, new_session

# Matches played on/after this timestamp are collected.
START_TIME = int(datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp())
QUEUE_TYPE = "ranked"
PAGE_SIZE = 100  # max ids returned per matchlist request
MAX_PAGES = 100  # safety cap (10k games per player; far beyond reality)
WORKERS_PER_REGION = 16  # concurrent in-flight requests per routing region
PROGRESS_INTERVAL = 30  # seconds between progress lines per region

BASE_DIR = Path(__file__).resolve().parent
PUUID_DIR = BASE_DIR / "puuids"
OUT_DIR = BASE_DIR / "matchids"
RAW_DIR = OUT_DIR / "raw"


def load_workload(routing_filter=None, limit_per_region=None):
    """routing region -> [(leaguePoints, tier, puuid), ...] sorted by LP desc.

    Reads the stage-1 per-platform files; a puuid seen in several platform
    regions is kept once with its highest league points. Processing the
    highest-LP players first means an interrupted run still leaves the most
    valuable matches collected.
    """
    buckets = {}
    for path in sorted(PUUID_DIR.glob("*.json")):
        if path.name == "index.json":
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        routing = data["routing_region"]
        if routing_filter and routing not in routing_filter:
            continue
        bucket = buckets.setdefault(routing, {})
        for entry in data["entries"]:
            puuid = entry["puuid"]
            lp = entry.get("leaguePoints", 0)
            if puuid not in bucket or lp > bucket[puuid][0]:
                bucket[puuid] = (lp, entry["tier"], puuid)
    workload = {}
    for routing, bucket in buckets.items():
        entries = sorted(bucket.values(), reverse=True)
        if limit_per_region:
            entries = entries[:limit_per_region]
        workload[routing] = entries
    return workload


def load_existing(routing):
    """(set of finished puuids, set of match ids) from the raw jsonl log."""
    done, ids = set(), set()
    raw_path = RAW_DIR / ("%s.jsonl" % routing)
    if not raw_path.exists():
        return done, ids
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from a killed run; re-fetch it
            done.add(record["puuid"])
            ids.update(record["ids"])
    return done, ids



class RegionState:
    """Accumulated progress for one routing region (append-only on disk)."""

    def __init__(self, routing):
        self.routing = routing
        self.done = set()
        self.ids = set()
        self.lock = asyncio.Lock()
        self.raw_path = RAW_DIR / ("%s.jsonl" % routing)

    async def record(self, puuid, match_ids):
        async with self.lock:
            with open(self.raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"puuid": puuid, "ids": match_ids}) + "\n")
                f.flush()
            self.done.add(puuid)
            self.ids.update(match_ids)

    async def record_failure(self, puuid, error):
        async with self.lock:
            failed_path = OUT_DIR / ("%s_failed.txt" % self.routing)
            with open(failed_path, "a", encoding="utf-8") as f:
                f.write("%s\t%s\n" % (puuid, error))

    def write_final(self):
        OUT_DIR.mkdir(exist_ok=True)
        final_path = OUT_DIR / ("%s_matchids.json" % self.routing)
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "start_time": START_TIME,
                    "queue_type": QUEUE_TYPE,
                    "match_ids": sorted(self.ids),
                },
                f,
            )


async def fetch_all_match_ids(session, limiters, routing, puuid):
    """All ranked match ids for one puuid since START_TIME (paginated)."""
    url = (
        "https://%s.api.riotgames.com/lol/match/v5/matches/by-puuid/%s/ids"
        % (routing, puuid)
    )
    match_ids = []
    start = 0
    for _ in range(MAX_PAGES):
        page = await fetch_json(
            session,
            limiters,
            routing,
            url,
            params={
                "startTime": START_TIME,
                "type": QUEUE_TYPE,
                "start": start,
                "count": PAGE_SIZE,
            },
        )
        match_ids.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return match_ids


async def worker(session, limiters, state, queue):
    while True:
        try:
            puuid = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            match_ids = await fetch_all_match_ids(session, limiters, state.routing, puuid)
            await state.record(puuid, match_ids)
        except Exception as exc:  # noqa: BLE001 - keep crawling, retry next run
            await state.record_failure(puuid, repr(exc))


async def progress_loop(state, total, stop):
    while True:
        try:
            await asyncio.wait_for(stop.wait(), PROGRESS_INTERVAL)
        except asyncio.TimeoutError:
            pass
        else:
            return  # stop was set; exit immediately
        print(
            "  %-9s %d/%d puuids, %d unique match ids"
            % (state.routing, len(state.done), total, len(state.ids))
        )


async def run_region(session, limiters, routing, entries, workers):
    state = RegionState(routing)
    state.done, state.ids = load_existing(routing)

    queue = asyncio.Queue()
    queued = 0
    for _, _, puuid in entries:
        if puuid not in state.done:
            queue.put_nowait(puuid)
            queued += 1

    total = len(entries)
    print(
        "%-9s %d puuids (%d already done, %d unique ids on disk) - %d workers"
        % (routing, total, total - queued, len(state.ids), workers)
    )
    if queued == 0:
        state.write_final()
        return state

    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(worker(session, limiters, state, queue))
        for _ in range(workers)
    ]
    progress = asyncio.create_task(progress_loop(state, total, stop))
    try:
        await asyncio.gather(*tasks)
    finally:
        stop.set()
        try:
            await progress
        except asyncio.CancelledError:
            pass
    state.write_final()
    print(
        "  %-9s finished: %d puuids, %d unique match ids"
        % (routing, len(state.done), len(state.ids))
    )
    return state


async def main(routing_filter, limit_per_region, workers):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    workload = load_workload(routing_filter, limit_per_region)
    if not workload:
        print("Nothing to do (no stage-1 puuid files matched).")
        return

    limiters = HostRateLimiters()
    states = {}
    async with new_session() as session:
        region_tasks = {
            routing: asyncio.create_task(
                run_region(session, limiters, routing, entries, workers)
            )
            for routing, entries in workload.items()
        }
        try:
            for routing, task in region_tasks.items():
                states[routing] = await task
        finally:
            # Persist finals for every region that completed, even on Ctrl+C;
            # regions still running keep their progress in the raw jsonl log.
            for routing, task in region_tasks.items():
                if task.done() and not task.cancelled() and task.exception() is None:
                    states[routing] = task.result()

    all_ids = set()
    print("-" * 55)
    for routing in sorted(states):
        state = states[routing]
        all_ids.update(state.ids)
        print(
            "  %-9s %d puuids, %d unique match ids"
            % (routing, len(state.done), len(state.ids))
        )
    print("-" * 55)
    print(
        "total: %d puuids, %d unique match ids (since %d)"
        % (sum(len(s.done) for s in states.values()), len(all_ids), START_TIME)
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
        help="only the top-N highest-LP puuids per region (for testing)",
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
        unknown = regions - set(PLATFORM_TO_ROUTING.values())
        if unknown:
            parser.error("unknown routing regions: %s" % ", ".join(sorted(unknown)))
    asyncio.run(main(regions, args.limit or None, args.workers))
