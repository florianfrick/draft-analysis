# Draft Assistant — Android (MVP)

Native Kotlin client for the draft-analysis backend in `frontend/`. This is a
**thin client**: it holds the board, renders it, and asks the Python service two
questions per board (what to take next, and what fills each open role). No model
weights, no corpus, no inference on the device.

## Status

On branch `android`. **Compiles, tests green, and verified running on an
emulator** against a live backend. `:app:assembleDebug` produces a 9.6 MB APK
with no warnings; `:app:testDebugUnitTest` passes 6/6; `:app:lintDebug` reports
0 errors.

The first emulator run (Pixel 5, API 34, 1080×2340) found three real bugs that
neither the compiler nor the tests could have caught, all now fixed:

1. **`NetworkOnMainThreadException`** — `reload()` called the API from
   `viewModelScope`, which is `Dispatchers.Main`. `refresh()` was already
   correct; the initial load was not.
2. **`patches` was the wrong type.** It is `[{"id": 106, "key": "16.20"}]`, not
   a list of strings. The `id` is a row of the checkpoint's patch vocabulary and
   is *not* an index into the list — the list arrives newest-first, 107 down
   to 1.
3. **The fresh board read "draft complete".** `turn` was only computed inside
   `mutate()`, so the initial load left it null. Both paths now share
   `withTurn()`.

Confirmed on device: patch defaults to 16.20, the turn line and the "on the
clock" highlight track the plan, the champion grid loads all 173 portraits from
the dDragon CDN, `/api/next-move` returns a ranked list (Nocturne 13 %, Vi 12 %,
Poppy 8 % …), "next game" closes G1 into G2 and carries the picks into the
fearless pool, and a pool champion is dimmed and unselectable in the picker.

### Build

The toolchain is self-contained (JDK 21, Android SDK 34/35, Gradle 8.9) and kept
outside the repo, because `sdkmanager.bat` cannot handle the space in the default
Windows home path — it dies with `'C:\Users\Florian' is not recognized`.

```bash
TC=/e/android-toolchain
export JAVA_HOME="$TC/jdk/jdk-21.0.13+9"
export ANDROID_HOME="$TC/sdk"
"$TC/gradle/gradle-8.9/bin/gradle" --no-daemon :app:assembleDebug
"$TC/gradle/gradle-8.9/bin/gradle" --no-daemon :app:testDebugUnitTest
```

`local.properties` points at `E:/android-toolchain/sdk` and is git-ignored, so
Android Studio will regenerate it on first open.

### Pointing the app at a backend

`API_BASE` is a build property, so no source edit is needed:

```bash
gradle -PapiBase=http://10.0.2.2:8777   assembleDebug   # emulator (default)
gradle -PapiBase=http://127.0.0.1:8777  assembleDebug   # via `adb reverse`
gradle -PapiBase=http://192.168.1.20:8777 assembleDebug # LAN device
```

You usually do **not** need to set it. `DraftApi` tries the configured address,
then `10.0.2.2`, then `127.0.0.1`, and caches whichever answers — so the same
APK works on an emulator and on a phone with `adb reverse`. Only a transport
failure triggers the next candidate; an HTTP error is surfaced as-is. The final
error message names every address tried.

**Over adb (emulator or phone), prefer `adb reverse`** if `10.0.2.2` fails:

```bash
adb reverse tcp:8777 tcp:8777
```

This tunnels over the adb transport, so it works even when the guest has no
route to the host. That happened in this environment: the AVD booted with only a
link-local route (`10.0.2.0/24 dev wlan0`) and **no default gateway**, so
`10.0.2.2` was unreachable *and* the champion portraits could not load. Adding
the route by hand fixed both:

```bash
adb root
adb shell ip route add default via 10.0.2.2 dev eth0
```

If `10.0.2.2` times out and the images are blank, that is the cause — it is an
emulator networking problem, not an app bug.

## Running it

1. Start the backend. The default bind is loopback-only, which is exactly what
   the emulator needs:

   ```bash
   bash frontend/serve.sh          # http://127.0.0.1:8777
   ```

2. Open `android/` in Android Studio. AGP 8.7 needs JDK 17-21; the system JDK on
   this machine is 25, which AGP rejects — Studio's bundled JBR 21 is fine, and
   the command-line build above pins JDK 21 explicitly.

3. Run on an **emulator**. `BuildConfig.API_BASE` defaults to
   `http://10.0.2.2:8777`, the emulator's alias for the *host's* loopback, so no
   server flag needs changing — but see "Pointing the app at a backend" above if
   the guest has no route to the host.

### On a physical device

`10.0.2.2` is an emulator-only alias — on a phone it is just some address on
the carrier network. The simple route is **adb reverse**, which needs no change
to `serve.sh` or to the compose port binding:

```bash
adb reverse tcp:8777 tcp:8777
```

The phone's own `127.0.0.1:8777` is then tunnelled to the machine's loopback,
where the backend already listens. `127.0.0.1` is already permitted for
cleartext in `network_security_config.xml`, and the app's fallback will find it
even if you just press Run with the default build. Re-run `adb reverse` after
reconnecting the cable or rebooting the device — the mapping does not survive a
new adb transport.

Verified on a Pixel 7 Pro over Android Studio's adb connection: patch 16.20,
turn line, all champion portraits and the ranked suggestions load normally.

If you would rather run untethered, use the LAN instead: change the compose
publish from `127.0.0.1:8777` to `0.0.0.0:8777`, build with
`-PapiBase=http://<LAN-IP>:8777`, and add that same IP to
`res/xml/network_security_config.xml` (cleartext is blocked by default). There
is no auth and no TLS — keep it on the LAN.

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
head takes the pool as an input and re-scores. The default next-move model is
now the role-conditioned naive transformer
(`nextpick-naivetransformer-role-d512-l8-h8-k5-through-lcp-df-sw`), so this is
dormant — but it is why `SuggestResponse.fearlessFiltered` is parsed. The
transformer sees the pool as 40 tokens and masks it on picks only; a champion in
the pool can still be banned. Its slot head is **not** wired into this client's
role board: ban-phase cells still come from the masking checkpoint, and pick-phase
cells use the role-filtered next-move head. The header's `model` label reports
whichever checkpoint the server actually loaded at startup.

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

1. Compare the turn highlight and the suggestion list against the web UI on the
   same boards. If they disagree, `DraftOrder` is wrong, not the server.
2. Persist the series across process death (the web client uses `localStorage`;
   this holds it in memory only, so a restart loses the series).
3. Reach a real game end-to-end: 6 bans, 6 picks, 4 bans, 4 picks, and confirm
   the board reads "draft complete" at move 21 rather than earlier.
4. Then decide on hosting: auth, TLS, and a real base URL — before any of the
   agent features ship to anyone else.
