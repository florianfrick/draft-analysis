# Draft Assistant — Android (MVP)

Native Kotlin client for the draft-analysis backend in `frontend/`. This is a
**thin client**: it holds the board, renders it, and asks the Python service two
questions per board (what to take next, and what fills each open role). No model
weights, no corpus, no inference on the device.

## Status

On branch `android`. **Compiles and tests green** — `:app:assembleDebug` produces
a 9.6 MB APK with no warnings, and `:app:testDebugUnitTest` passes 6/6 tests
covering the ported draft order. Not yet run on a device or emulator, so the
Compose layout is verified by the compiler only, not by eye.

The build used a self-contained toolchain (JDK 21, Android SDK 34/35, Gradle 8.9)
kept outside the repo, because `sdkmanager.bat` cannot handle the space in the
default Windows home path:

```bash
TC=/e/android-toolchain
export JAVA_HOME="$TC/jdk/jdk-21.0.13+9"
export ANDROID_HOME="$TC/sdk"
"$TC/gradle/gradle-8.9/bin/gradle" --no-daemon :app:assembleDebug
"$TC/gradle/gradle-8.9/bin/gradle" --no-daemon :app:testDebugUnitTest
```

`local.properties` points at `E:/android-toolchain/sdk` and is git-ignored, so
Android Studio will regenerate it on first open.

## Running it

1. Start the backend. The default bind is loopback-only, which is exactly what
   the emulator needs:

   ```bash
   bash frontend/serve.sh          # http://127.0.0.1:8777
   ```

2. Open `android/` in Android Studio. AGP 8.7 needs JDK 17-21; the system JDK on
   this machine is 25, which AGP rejects — Studio's bundled JBR 21 is fine, and
   the command-line build above pins JDK 21 explicitly.

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
- **Series and fearless pool.** Game chips per series, `＋ game`, `next game`,
  `clear`, a fearless strip you can edit by hand, and per-game first-pick side.

## Series and fearless

The pool is **stored per game, not derived from earlier games**. That mirrors the
reference client: each game slot owns an explicit, editable `fearless` list, and
"next game" folds the closed game's picks into a fresh copy. Deriving it from the
picks alone would make the manual add/remove unrepresentable.

Transitions, matching `app.js`:

- `＋ game` appends a slot that **inherits the current pool unchanged** — nothing
  was closed, so it is the same point in the series.
- `next game` sets `done` on the current slot, computes
  `pool + picks` of this game, and inserts a successor at `current + 1`. Bans
  never enter the pool: a champion banned in game 1 is legal again in game 2.
- **First pick is kept, not flipped.** The teams swap sides between games but the
  side holding first pick stays the same in 94% of pro games.
- A closed game that already has a successor cannot be closed again — that would
  splice a duplicate in between. Use `＋ game` to insert one.
- The pool blocks **picks but never bans**. It is not a "taken" list.

Every request sends `fearless` (this game's pool) and `series`
(`{game, of, games:[{game, picks, fearless, closed}]}`). The server renders the
series into lines itself; no client text reaches a model prompt.

One caveat carried over from the web UI that this client does **not** surface:
if the answering model is a masking checkpoint rather than the next-move head,
the pool was applied as a post-filter on an answer computed as if those
champions were available (`method == "masker"` with `fearless_filtered > 0`). The
head takes the pool as an input and re-scores. The default checkpoints are heads,
so this is dormant — but it is why `SuggestResponse.fearlessFiltered` is parsed.

## What it deliberately does not do yet

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
  ui/BoardViewModel.kt       board + series state, debounce, stale-reply guard
  ui/BoardScreen.kt          board, series bar, fearless strip, suggestions
  ui/ChampionPicker.kt       bottom-sheet champion grid
  ui/Theme.kt                palette lifted from style.css

app/src/test/java/com/draftassistant/data/
  DraftOrderTest.kt          pins the plan to infer.move_plan's four sequences
```

## Next steps, in order

1. Run it on an emulator against a live backend. The layout is compiler-checked
   only — this is the first time anyone will have *seen* it.
2. Compare the turn highlight and the suggestion list against the web UI on the
   same boards. If they disagree, `DraftOrder` is wrong, not the server.
3. Persist the series across process death (the web client uses `localStorage`;
   this holds it in memory only, so a restart loses the series).
4. Then decide on hosting: auth, TLS, and a real base URL — before any of the
   agent features ship to anyone else.
