"""Shared Riot Games API helpers.

Key facts that shape this module:

- Riot enforces application rate limits PER REGION (per API host).
  For a dev/personal key the LoL application limit is 20 requests / 1 s and
  100 requests / 2 min, per region. Requests to different regions do NOT
  count against each other, so regions can be crawled fully in parallel as
  long as each region stays under its own budget. The 2-minute window is the
  binding one: 100/2min = ~0.83 req/s, far below the 20 req/s burst.
- Multiple keys do NOT multiply the budget in practice. Measured with two
  dev keys: both reported the same X-App-Rate-Limit-Count and the second key
  429'd as soon as the first pushed the shared counter past 100. So the pool
  defaults to one shared per-host limiter with round-robin key use (see
  HostRateLimiters). Only keys from genuinely separate Riot applications get
  independent budgets (RIOT_KEYS_SHARED_BUDGET=0).
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


class RiotAuthError(RuntimeError):
    """Key is missing/expired/forbidden (HTTP 401/403). Retrying cannot help."""

# Riot API key. Provide via RIOT_API_KEY (dev keys expire after 24h, so never
# commit a key to the repo):
#   RIOT_API_KEY="RGAPI-a..."
# On Windows PowerShell:  $env:RIOT_API_KEY = "RGAPI-a..."
# On Linux/macOS:         export RIOT_API_KEY="RGAPI-a..."
#
# ONE KEY BY DEFAULT. Extra keys may be supplied (RIOT_API_KEYS comma-separated,
# or RIOT_API_KEY_2 .. _9) but they are IGNORED unless RIOT_USE_ALL_KEYS=1.
def _load_api_keys():
    """Ordered, de-duplicated list of keys from the environment."""
    keys = []

    def add(value):
        value = (value or "").strip()
        if value and value not in keys:
            keys.append(value)

    for part in (os.environ.get("RIOT_API_KEYS") or "").split(","):
        add(part)
    add(os.environ.get("RIOT_API_KEY"))
    for i in range(2, 10):
        add(os.environ.get("RIOT_API_KEY_%d" % i))
    return keys


def _use_all_keys():
    env = os.environ.get("RIOT_USE_ALL_KEYS", "0").strip().lower()
    return env in ("1", "true", "yes", "on")


_ALL_KEYS = _load_api_keys()
if not _ALL_KEYS:
    raise RuntimeError(
        "No Riot API key found. Set RIOT_API_KEY before collecting data "
        "(see https://developer.riotgames.com).")
# Default to exactly ONE key; any extras are opt-in via RIOT_USE_ALL_KEYS=1.
# Measured: Riot shares one application budget across a developer's keys, so
# extra keys add no throughput and only risk 429s.
API_KEYS = _ALL_KEYS if _use_all_keys() else _ALL_KEYS[:1]
API_KEY = API_KEYS[0]

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

    def wait_time(self):
        """Seconds until a slot frees (0 if a request can go right now).

        Non-blocking peek used by HostRateLimiters to choose which key to
        spend. Prunes expired stamps in place; asyncio is cooperative, so
        this cannot interleave with a half-finished acquire().
        """
        now = time.monotonic()
        wait = 0.0
        for limit, period, stamps in self._windows:
            while stamps and now - stamps[0] >= period:
                stamps.popleft()
            if len(stamps) >= limit:
                wait = max(wait, period - (now - stamps[0]))
        return wait

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
    """GET url, respecting the per-(host,key) rate limiter; retry on 429/5xx.

    `limiters` is a HostRateLimiters; `host` is the API host key (e.g. "na1"
    or "europe"). Each attempt spends the least-loaded key for that host, so
    with several keys the region's effective budget is multiplied. A 429
    cools down only the key that received it; the retry then prefers a
    different key. Limiters are re-tuned from the rate-limit headers.

    Returns the parsed JSON body on HTTP 200, raises RuntimeError otherwise
    (after retries are exhausted).
    """

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
        key_idx = await limiters.acquire(host)
        headers = {"X-Riot-Token": limiters.key_for(key_idx)}
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                limiters.observe(host, key_idx, resp.headers)
                if resp.status == 200:
                    return await resp.json(content_type=None)

                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", 5))
                    counts = resp.headers.get("X-App-Rate-Limit-Count", "?")
                    # Cool this key down; the next attempt prefers another.
                    limiters.penalize(host, key_idx, retry_after)
                    await retry_wait(
                        retry_after,
                        "429 rate limited on key #%d [app limits: %s]"
                        % (key_idx + 1, counts),
                    )
                    continue

                if resp.status >= 500:
                    await retry_wait(2 ** attempt, "HTTP %d" % resp.status)
                    continue

                body = await resp.text()
                if resp.status in (401, 403):
                    # Dev keys expire after 24h. Retrying is pointless and
                    # would burn the whole worklist into failed.txt, so fail
                    # fast and let the caller abort the run cleanly.
                    raise RiotAuthError(
                        "HTTP %d (key expired or invalid) for %s: %s"
                        % (resp.status, url, body[:200]))
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
    """Rate limiters per API host, with an optional key pool.

    MEASURED BEHAVIOUR (dev keys, Oct 2026): Riot shares ONE application
    budget across all of a developer's keys. Two keys on the same account/IP
    report the same X-App-Rate-Limit-Count (observed: key A at 84:120 while
    key B hit 429 at 101:120), so extra keys do NOT add throughput - they
    only push the shared counter over the limit and earn 69-second
    Retry-After penalties. Hence `keys_share_budget` defaults to True: one
    limiter per host and only the FIRST key is used, with extras ignored
    entirely. Set RIOT_KEYS_SHARED_BUDGET=0 (or keys_share_budget=False)
    only if the keys come from genuinely separate Riot applications, in
    which case each key gets its own limiter and the budget multiplies.

    Limiters start from conservative defaults (APP_RATE_LIMITS) and are
    re-tuned once per budget from the actual X-App-Rate-Limit /
    X-Method-Rate-Limit headers, seeded with the usage reported in the
    X-*-Rate-Limit-Count headers.
    """

    def __init__(self, default_windows=None, keys=None, keys_share_budget=None):
        self._default_windows = default_windows or APP_RATE_LIMITS
        self._keys = list(keys) if keys else list(API_KEYS)
        if keys_share_budget is None:
            env = os.environ.get("RIOT_KEYS_SHARED_BUDGET", "1").strip().lower()
            keys_share_budget = env not in ("0", "false", "no", "off")
        self._shared = keys_share_budget
        self._limiters = {}       # budget key -> SlidingWindowLimiter
        self._configured = set()  # budget keys already re-tuned
        self._cooldown = {}       # budget key -> monotonic time usable again

    @property
    def key_count(self):
        return len(self._keys)

    @property
    def shared_budget(self):
        return self._shared

    def key_for(self, key_idx):
        return self._keys[key_idx]

    def _budget_key(self, host, key_idx):
        """Limiters are per host when the budget is shared, per pair otherwise."""
        return host if self._shared else (host, key_idx)

    def get(self, host, key_idx=0):
        budget = self._budget_key(host, key_idx)
        limiter = self._limiters.get(budget)
        if limiter is None:
            limiter = SlidingWindowLimiter(self._default_windows)
            self._limiters[budget] = limiter
        return limiter

    def penalize(self, host, key_idx, seconds):
        """Cool the budget down for `seconds` (used after a 429)."""
        budget = self._budget_key(host, key_idx)
        until = time.monotonic() + max(0.0, seconds)
        self._cooldown[budget] = max(self._cooldown.get(budget, 0.0), until)

    def _pick(self, host):
        """Index of the key to spend next.

        Shared budget: always the first key. Rotating keys cannot raise
        throughput (one shared counter) and only adds 429/ToS risk, so it is
        deliberately not done. Independent budgets: the key with the most
        headroom.
        """
        if self._shared:
            return 0
        now = time.monotonic()
        best, best_wait = 0, float("inf")
        for i in range(len(self._keys)):
            wait = max(
                self.get(host, i).wait_time(),
                self._cooldown.get((host, i), 0.0) - now,
            )
            if wait < best_wait:
                best, best_wait = i, wait
        return best

    async def acquire(self, host):
        """Wait for room, then return the key index to use for this request."""
        key_idx = self._pick(host)
        await self.get(host, key_idx).acquire()
        cooldown = self._cooldown.get(self._budget_key(host, key_idx), 0.0)
        now = time.monotonic()
        if cooldown > now:
            await asyncio.sleep(cooldown - now)
        return key_idx

    def observe(self, host, key_idx, headers):
        """Re-tune a budget's limiter from rate-limit response headers."""
        budget = self._budget_key(host, key_idx)
        if budget in self._configured:
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
        self._limiters[budget] = SlidingWindowLimiter(
            [(limit, period) for period, limit in by_period.items()],
            seed_counts=counts,
        )
        self._configured.add(budget)


def new_session():
    """Shared aiohttp session with sane timeouts."""
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    return aiohttp.ClientSession(timeout=timeout)
