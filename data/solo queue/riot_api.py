"""Shared Riot Games API helpers.

Key facts that shape this module:

- Riot enforces application rate limits PER REGION (per API host).
  For a dev/personal key the LoL application limit is 20 requests / 1 s and
  100 requests / 2 min, per region. Requests to different regions do NOT
  count against each other, so regions can be crawled fully in parallel as
  long as each region stays under its own budget.
- league-v4 endpoints are addressed by platform host
  (e.g. https://na1.api.riotgames.com), while match-v5 endpoints are
  addressed by a regional routing host (americas/asia/europe/sea).

So this module keeps one independent rate limiter per platform region,
which lets every region run at its own full budget concurrently.
"""

import asyncio
import os
import random
import time
from collections import deque

import aiohttp

# Riot API key. Provide via the RIOT_API_KEY environment variable
# (dev keys expire after 24h, so never commit a key to the repo).
# On Windows PowerShell:  $env:RIOT_API_KEY = "RGAPI-..."
# On Linux/macOS:         export RIOT_API_KEY="RGAPI-..."
API_KEY = os.environ.get("RIOT_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "RIOT_API_KEY is not set. Export RIOT_API_KEY with your Riot dev key "
        "(see https://developer.riotgames.com) before collecting data.")

# Application rate limits for dev/personal keys, per region:
#   20 requests per 1 second, 100 requests per 2 minutes.
# If you get a production key (higher limits), bump these.
APP_RATE_LIMITS = [
    (20, 1.0),
    (100, 120.0),
]

# Platform routing region (league-v4 host prefix) -> match-v5 routing region.
PLATFORM_TO_ROUTING = {
    # americas
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "na1": "americas",
    # asia
    "jp1": "asia",
    "kr": "asia",
    # europe
    "eun1": "europe",
    "euw1": "europe",
    "ru": "europe",
    "tr1": "europe",
    # sea (PH/TH/MY/ID/SG merged into the sg2 server with patch 15.1, Jan 2025;
    # the old ph2/th2 hosts no longer exist)
    "oc1": "sea",
    "sg2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}


class SlidingWindowLimiter:
    """Sliding-window rate limiter enforcing several (limit, period) windows.

    A request may proceed only when every window still has room. The caller
    awaits acquire(), which sleeps just long enough to free a slot.
    """

    def __init__(self, windows, seed_counts=None):
        # Each window: [limit, period_seconds, deque of recent acquire timestamps]
        self._windows = [
            [limit, period, deque()] for (limit, period) in windows
        ]
        if seed_counts:
            # Pre-fill window(s) with the usage the server already counted,
            # so a re-tuned limiter starts from the true current state.
            now = time.monotonic()
            for _, period, stamps in self._windows:
                count = seed_counts.get(period, 0)
                stamps.extend([now] * count)
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                wait = 0.0
                for limit, period, stamps in self._windows:
                    while stamps and now - stamps[0] >= period:
                        stamps.popleft()
                    if len(stamps) >= limit:
                        wait = max(wait, period - (now - stamps[0]))
                if wait <= 0:
                    for _, _, stamps in self._windows:
                        stamps.append(now)
                    return
            await asyncio.sleep(wait)


_retry_logged = {}  # host -> monotonic time of last printed retry message


async def fetch_json(session, limiters, host, url, *, params=None, max_retries=8):
    """GET url, respecting the per-host rate limiter; retry on 429/5xx/network errors.

    `limiters` is a HostRateLimiters; `host` is the API host key (e.g. "na1"
    or "europe"). The limiter is re-tuned from the rate-limit response headers
    of every response.

    Returns the parsed JSON body on HTTP 200, raises RuntimeError otherwise
    (after retries are exhausted).
    """
    headers = {"X-Riot-Token": API_KEY}

    # Retry messages are throttled per host (see _retry_logged) so a burst of
    # 429s can't flood the log; sleeps are jittered so retried requests don't
    # sync up again.
    async def retry_wait(seconds, reason):
        now = time.monotonic()
        if now - _retry_logged.get(host, -1e9) > 10.0:
            _retry_logged[host] = now
            print("    retrying in %.1fs (%s): %s" % (seconds, reason, url))
        await asyncio.sleep(seconds * random.uniform(1.0, 1.5))

    for attempt in range(max_retries):
        await limiters.get(host).acquire()
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                limiters.observe(host, resp.headers)
                if resp.status == 200:
                    return await resp.json(content_type=None)

                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", 5))
                    counts = resp.headers.get("X-App-Rate-Limit-Count", "?")
                    await retry_wait(
                        retry_after, "429 rate limited [app limits: %s]" % counts
                    )
                    continue

                if resp.status >= 500:
                    await retry_wait(2 ** attempt, "HTTP %d" % resp.status)
                    continue

                body = await resp.text()
                raise RuntimeError(
                    "HTTP %d for %s: %s" % (resp.status, url, body[:300])
                )
        except aiohttp.ClientError as exc:
            await retry_wait(2 ** attempt, "network error: %r" % exc)

    raise RuntimeError("Giving up after %d attempts: %s" % (max_retries, url))


def parse_limit_header(value):
    """Parse a Riot rate-limit header, e.g. '100:120,20:1' -> [(100, 120.0), (20, 1.0)]."""
    windows = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        limit, period = part.split(":")
        windows.append((int(limit), float(period)))
    return windows


def parse_count_header(value):
    """Parse a Riot rate-limit-count header, e.g. '3:120,1:1' -> {120.0: 3, 1.0: 1}."""
    counts = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        count, period = part.split(":")
        counts[float(period)] = int(count)
    return counts


class HostRateLimiters:
    """One sliding-window limiter per API host (Riot limits are per region).

    Limiters start from conservative default windows (APP_RATE_LIMITS) and
    are re-tuned once per host from the actual X-App-Rate-Limit and
    X-Method-Rate-Limit response headers, seeded with the usage reported in
    the X-*-Rate-Limit-Count headers. This keeps us exactly under whatever
    the real application and method limits are, for any key type.
    """

    def __init__(self, default_windows=None):
        self._default_windows = default_windows or APP_RATE_LIMITS
        self._limiters = {}
        self._configured = set()

    def get(self, host):
        limiter = self._limiters.get(host)
        if limiter is None:
            limiter = SlidingWindowLimiter(self._default_windows)
            self._limiters[host] = limiter
        return limiter

    def observe(self, host, headers):
        """Re-tune the host's limiter from rate-limit response headers."""
        if host in self._configured:
            return
        windows = []
        for key in ("X-App-Rate-Limit", "X-Method-Rate-Limit"):
            value = headers.get(key)
            if value:
                windows.extend(parse_limit_header(value))
        if not windows:
            return
        # One window per period; keep the strictest limit for each period.
        by_period = {}
        for limit, period in windows:
            if period not in by_period or limit < by_period[period]:
                by_period[period] = limit
        counts = {}
        for key in ("X-App-Rate-Limit-Count", "X-Method-Rate-Limit-Count"):
            value = headers.get(key)
            if value:
                for period, count in parse_count_header(value).items():
                    counts[period] = max(counts.get(period, 0), count)
        self._limiters[host] = SlidingWindowLimiter(
            [(limit, period) for period, limit in by_period.items()],
            seed_counts=counts,
        )
        self._configured.add(host)


def new_session():
    """Shared aiohttp session with sane timeouts."""
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    return aiohttp.ClientSession(timeout=timeout)
