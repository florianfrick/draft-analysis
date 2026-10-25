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
| Capability 1 | champ X with / against champ Y | precomputed `cube2` table | ~2 ms |
| Capability 2 | a whole draft, N champions | live scan of `drafts` | ~120 ms |

Capability 1 is a 29.7k-cell lookup. Capability 2 is **not** precomputed,
because there is nothing to precompute: a 5-champion cube is 240.7M cells /
1.08 GB, is 99.99% empty, and is *slower* to query (195 ms) than scanning
the 415k-row game table (28 ms).

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

So the panel leads with `n` and a confidence badge, then shows *how* the
sample collapsed as you added champions (the ladder), then — when `n` is
too small to read — falls back to the part that is well measured: the
pairwise win-rate deltas between the ten selected champions, and the
average pair effect on each champion's own win rate.

The "recover games by dropping one champion" strip exists for the same
reason: it tells you which single pick is costing you your sample.

## Endpoints

- `GET /api/meta` — corpus size, patches, queues, champions with icons
- `GET /api/matchup?champ=&partner=&mode=with|vs&patch=&role=`
- `GET /api/draft?blue=1,2&red=6,7&patch=&sides=any|fixed`

`sides=any` treats a draft as two unordered teams (blue in the UI may have
been red in the actual game); `fixed` requires the orientation as drawn.

## Sharing a state

The URL carries the state, so a draft or matchup can be linked:

```
#8,53,64,84,523/113,126,145,518,711     blue/red picks
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
