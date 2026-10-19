"""Collect the PUUIDs of every Challenger/Grandmaster/Master soloQ player.

This is stage 1 of the data pipeline:
    stage 1: puuids          <- this script
    stage 2: match ids       (collect_matchids.py, per-puuid match-v5 lists)
    stage 3: match details   (collect_match_data.py)

All platform regions are crawled IN PARALLEL. Riot's application rate limit
is per region, so each region gets its own independent rate limiter and the
regions never contend with each other.

Output (in ./puuids/):
    puuids/{region}.json  - entries per region: puuid, tier, leaguePoints
    puuids/index.json     - {puuid: [platform region, ...]} for stage 2
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from riot_api import (
    PLATFORM_TO_ROUTING,
    HostRateLimiters,
    fetch_json,
    new_session,
)

QUEUE = "RANKED_SOLO_5x5"
TIERS = [
    ("challengerleagues", "CHALLENGER"),
    ("grandmasterleagues", "GRANDMASTER"),
    ("masterleagues", "MASTER"),
]
OUT_DIR = Path(__file__).resolve().parent / "puuids"


async def fetch_tier(session, limiters, platform, tier_path, tier_name):
    """Fetch one league tier list for one platform region."""
    url = (
        "https://%s.api.riotgames.com/lol/league/v4/%s/by-queue/%s"
        % (platform, tier_path, QUEUE)
    )
    data = await fetch_json(session, limiters, platform, url)
    entries = []
    for entry in data.get("entries", []):
        entries.append(
            {
                "puuid": entry.get("puuid"),
                "tier": tier_name,
                "leaguePoints": entry.get("leaguePoints", 0),
            }
        )
    return entries


async def collect_region(session, platform, limiters):
    """Collect all three tiers for one platform region (tiers in parallel)."""
    tier_lists = await asyncio.gather(
        *(
            fetch_tier(session, limiters, platform, tier_path, tier_name)
            for tier_path, tier_name in TIERS
        )
    )

    # Dedupe by puuid (a player should only be in one tier, but be safe),
    # keeping the entry with the highest league points.
    by_puuid = {}
    for entries in tier_lists:
        for entry in entries:
            puuid = entry["puuid"]
            if not puuid:
                continue
            existing = by_puuid.get(puuid)
            if existing is None or entry["leaguePoints"] > existing["leaguePoints"]:
                by_puuid[puuid] = entry

    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "platform_region": platform,
        "routing_region": PLATFORM_TO_ROUTING[platform],
        "queue": QUEUE,
        "entries": sorted(
            by_puuid.values(),
            key=lambda e: e["leaguePoints"],
            reverse=True,
        ),
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / ("%s.json" % platform)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return platform, len(by_puuid)


async def main(regions):
    OUT_DIR.mkdir(exist_ok=True)
    # One independent limiter per platform region: regions run in parallel,
    # each within its own rate-limit budget.
    limiters = HostRateLimiters()

    async with new_session() as session:
        results = await asyncio.gather(
            *(collect_region(session, p, limiters) for p in regions),
            return_exceptions=True,
        )

    # Build the global index: puuid -> [platform regions it appeared in].
    index = {}
    total = 0
    for result in results:
        if isinstance(result, Exception):
            continue
        platform, count = result
        total += count
        with open(OUT_DIR / ("%s.json" % platform), "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data["entries"]:
            index.setdefault(entry["puuid"], []).append(platform)
    with open(OUT_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    print("-" * 50)
    for result in results:
        if isinstance(result, Exception):
            print("  failed: %r" % result)
        else:
            print("  %-5s %d players" % result)
    print("-" * 50)
    print("total players: %d, unique puuids: %d" % (total, len(index)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        default=",".join(PLATFORM_TO_ROUTING),
        help="comma-separated platform regions to collect (default: all)",
    )
    args = parser.parse_args()
    regions = [r.strip().lower() for r in args.regions.split(",") if r.strip()]
    unknown = [r for r in regions if r not in PLATFORM_TO_ROUTING]
    if unknown:
        parser.error("unknown regions: %s" % ", ".join(unknown))
    asyncio.run(main(regions))
