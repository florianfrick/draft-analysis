# Stats Explorer

Local web UI for the empirical side of the draft project: what actually
happened in the soloQ corpus, as opposed to what the model predicts.

```powershell
cd "E:\VSCode\code\lol draft analysis\soloq data analysis"
python explorer\server.py --port 8778      # http://127.0.0.1:8778
```

Requires `duckdb` and a built store (`python store.py build`). Binds
`127.0.0.1` only. Start-up is instant — no torch, no checkpoint loading.

## Two capabilities, two very different costs

| | Query | Served by | Measured |
|---|---|---|---|
| Capability 1 | champ X with / against champ Y | precomputed `cube2` table | ~5 ms |
| Capability 2 | a whole draft, N champions | live scan of `drafts` | 90–210 ms |
| Both | synergies / counters for one side | `cube2` grouped by candidate | ~15 ms |

Capability 1 is a 29.7k-cell lookup per patch. Capability 2 is **not**
precomputed, because there is nothing to precompute: a 5-champion cube is
240.7M cells / 1.08 GB, is 99.99% empty, and is *slower* to query (195 ms)
than scanning the 415k-row game table (28 ms).

## Why the UI looks the way it does

The corpus cannot answer most full-draft questions by counting. Measured
over 415,000 games:

| Champions named | Cells | Median n | Max n | Cells reaching n≥30 |
|---|---|---|---|---|
| 1 | 173 | 20,553 | 75,656 | 100% |
| 2 | 29,752 | 370 | 12,976 | 97.7% |
| 3 | 2.12M | 6 | 1,114 | 8.9% |
| 4 | 78.4M | 1 | 128 | 0.0125% |
| 5 | 240.7M | 1 | 21 | 0% |

So the UI never leads with an average from a 3-game sample. It leads with
`n` and a confidence badge, and then splits into three views.

## The three views

**Overall** answers "how does this draft play out". The win rate is read
between the two selections (orientation-corrected: in half the matching
games your team was the red one), each champion's line inside those games is
clickable for its full stat breakdown, clicking a column header ranks the
champions by that stat, and the whole selection gets a pairwise delta matrix
against each champion's own patch average.

**Blue** and **Red** answer "what should I pick next, and what beats me".
Both lists come from the dense 2-way cube rather than the sparse draft
sample, and each candidate carries two numbers:

| | what it is | typical n |
|---|---|---|
| vs this draft | the candidate's 2-way cells against only *your* champions | 1–7k |
| overall | the candidate's whole patch | 10–75k |

The gap between them is the difference between "a strong champion" and "a
strong champion into this draft". A `rank by` toggle sorts on either, and
the metric selector can sort on any registered stat — win rate, gold, gold
per minute, CS lead over the laner, kill participation.

## Metrics

Every stat lives in one `METRICS` list in `server.py` (key, label, direction,
format, group). The cubes already carry all 32 challenge keys, so exposing a
new stat is one entry there plus, if it is a brand-new column, one line in
`store.py`. `laningPhaseGoldExpAdvantage` is deliberately *not* exposed: in
this corpus it is a 0/1 flag rather than a lead magnitude. When the real
gold-lead-at-10 column lands it slots into the `lane` group and becomes
selectable everywhere without touching the UI.

## Endpoints

- `GET /api/meta` — corpus size, patches, queues, champions with icons, metrics
- `GET /api/matchup?champ=&partner=&mode=with|vs&patch=&role=`
- `GET /api/draft?blue=1,2&red=6,7&patch=&sides=any|fixed`
- `GET /api/suggest?side=blue&blue=&red=&metric=&patch=` — synergies + counters
- `GET /api/champion?champ=&blue=&red=&patch=` — one champion's full line

`sides=any` treats a draft as two unordered teams (blue in the UI may have
been red in the actual game); `fixed` requires the orientation as drawn.
`suggest` and `champion` are always orientation-free.

## Sharing a state

The URL carries the state, so a draft or matchup can be linked:

```
#8,53,64,84,523/113,126,145,518,711     blue/red picks
?v=blue&est=1&games=1                   view + revealed cards
#m=103,64,vs                            Ahri against Lee Sin
```

## Notes

- Champion ids are Riot ids, same space as `champ_mapping.json`.
- The store is patch-aware; the patch selector defaults to the largest
  patch in the corpus. Mixing patches is available but not recommended.
- Queue 710 (11k games) is included in the counts; filter it out in the
  query if you want ranked solo/duo + flex only.
- Rebuilding the store requires stopping this server first — on Windows a
  read-only DuckDB connection still holds the file lock.
