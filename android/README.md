# Draft Assistant — Android (MVP)

Native Kotlin client for the draft-analysis backend in `frontend/`. This is a
**thin client**: it holds the board, renders it, and asks the Python service two
questions per board (what to take next, and what fills each open role). No model
weights, no corpus, no inference on the device.

## Status

Scaffolded on branch `android`. **Not yet compiled** — this machine has a JDK but
no Android SDK or Gradle, so the first `./gradlew assembleDebug` in Android
Studio is the real verification. Treat the first build as a debugging session,
not a formality.

## Running it

1. Start the backend. The default bind is loopback-only, which is exactly what
   the emulator needs:

   ```bash
   bash frontend/serve.sh          # http://127.0.0.1:8777
   ```

2. Open `android/` in Android Studio (JDK 17 or 21 — AGP 8.7 does not support
   JDK 25, and `java -version` on this machine reports 25; Studio's bundled JBR
   is fine).

3. Run on an **emulator**. `BuildConfig.API_BASE` is `http://10.0.2.2:8777`, and
   `10.0.2.2` is the emulator's alias for the *host's* loopback interface, so no
   server flag needs changing.

### On a physical device

The phone cannot see the host's loopback. You need two edits:

- start the server with `--host 0.0.0.0` (and be aware that `docker-compose.yml`
  publishes the port to `127.0.0.1` only — that mapping has to change too);
- put the machine's LAN address in `API_BASE` (`app/build.gradle.kts`) **and** in
  `res/xml/network_security_config.xml`, or cleartext HTTP is blocked.

There is no auth and no TLS. Do not expose this beyond your LAN.

## What the MVP does

- Fetches `/api/champions`: champions, the role vocabulary, the canonical
  `move_plan`, the patch list.
- Renders both sides: 5 bans and 5 picks each, plus a role badge per pick
  (`assignment` + `conf` from `/api/suggest`).
- Computes whose turn it is locally, from the published plan.
- Taps an empty slot to open a searchable champion grid; taps a ban slot to add
  a ban; the small `×` removes a pick.
- Asks `/api/next-move` (k=8) and `/api/suggest` (k=5) after every change and
  lists the head's ranked suggestions; tapping one plays it into the slot the
  current turn implies.
- Patch selector and first-pick selector.

## What it deliberately does not do yet

- **No series / fearless pool.** Requests send `fearless: []` explicitly, which
  the server reads as "this series has no pool". It is *not* the same as
  omitting the field, which means "keep the existing pool".
- No real-draft walkthrough (`/api/drafts`, `/api/draft`).
- No agent chat (`/api/agent`, `/api/agent-suggest`) — that needs an
  `LLM_API_KEY` server-side, and shipping it means paying for other people's
  tokens.
- No checkpoint picker (`/api/load`). The server's startup defaults are used.

## Contract notes worth keeping

These are the places where a plausible-looking reimplementation is wrong:

- **`move_plan`'s second element is the SIDE (0 blue, 1 red), not a slot index.**
  A pick's slot is not in the plan at all. Position is found by *consuming
  counts* — n bans and n picks against the kinds in order — not by testing which
  slots look filled. See `data/DraftOrder.kt`, ported from
  `frontend/static/draft-order.js`.
- **Role keys are names** (`TOP..SUPPORT`) on every route. Never index a role
  list numerically.
- **A reply belongs to the board it was asked about.** `BoardViewModel` stamps
  each request with a board signature and drops a reply whose signature no
  longer matches; role badges are cleared on every board change so a badge is
  never shown for a board it never saw. The web client had exactly this bug —
  a fresh pick looking ignored.
- **`picks` is indexed by slot**, 10 entries, null for empty. Slots 0-4 are blue,
  5-9 red.
- Champion icons are dDragon CDN URLs from the API — nothing is bundled, so the
  first load needs network.

## Layout

```
app/src/main/java/com/draftassistant/
  MainActivity.kt            entry point
  data/Models.kt             wire DTOs, field-for-field with server.py
  data/DraftOrder.kt         board -> next turn (ported, not reinvented)
  data/DraftApi.kt           OkHttp + kotlinx.serialization, 3 routes
  ui/BoardViewModel.kt       board state, debounce, stale-reply guard
  ui/BoardScreen.kt          board, turn bar, suggestions
  ui/ChampionPicker.kt       bottom-sheet champion grid
  ui/Theme.kt                palette lifted from style.css
```

## Next steps, in order

1. Get it compiling and rendering against a live backend.
2. Confirm the turn computation against the web UI on the same boards — if the
   two disagree, `DraftOrder` is wrong, not the server.
3. Add series + fearless (the state plumbing already accepts a pool).
4. Then decide on hosting: auth, TLS, and a real base URL — before any of the
   agent features ship to anyone else.
