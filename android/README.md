# Draft Assistant — Android

Native Kotlin/Compose app for LoL draft analysis. **The models run on the
phone.** There is no backend, no network dependency, no auth, no API key. Turn
on airplane mode after installing and the app behaves identically.

## Status

On branch `android`. `:app:assembleDebug` produces a **75 MB** debug APK
(36 MB of ExecuTorch weights, 5 MB of portraits); `:app:testDebugUnitTest`
passes 6/6; `:app:lintDebug` is clean. Verified on a Pixel 5 / API 34 emulator
and a Pixel 7 Pro, with the backend stopped.

On-device output was checked against the Python serving path on two boards:

| board | device (top 5) | Python reference (top 5) |
| --- | --- | --- |
| empty, blue to ban, patch 16.20 | Vi 10.39 %, Poppy 6.59 %, Nocturne 6.39 %, Renata Glasc 5.87 %, Olaf 5.22 % | Vi 10.37 %, Poppy 6.58 %, Nocturne 6.49 %, Renata Glasc 5.85 %, Lee Sin 5.17 % |
| after banning Poppy | Nocturne 21.42 %, Vi 18.53 %, Cassiopeia 13.64 %, Lee Sin 9.70 %, Qiyana 6.17 % | Nocturne 21.62 %, Vi 18.41 %, Cassiopeia 13.68 %, Lee Sin 9.58 %, Qiyana 6.20 % |

The ranking agrees; the sub-percent deltas are dynamic-int8 noise. The masker's
blue-SUPPORT cell matches to three decimals (Nautilus 22.7 %, Alistar 13.1 %,
Rell 9.0 %). A near-tie at ranks 5–6 can flip between the quantized and fp32
models — that is expected, not a contract failure.

## How it works

`LocalEngine` (`data/LocalEngine.kt`) is the whole offline backend. It loads two
ExecuTorch programs from `assets/`, builds the same tensors the Python serving
path builds, applies the same legality rules to the logits, and returns the same
DTOs `DraftApi` used to get over HTTP. `BoardViewModel` and every UI component
were left untouched at the DTO boundary.

| asset | source checkpoint | size | output |
| --- | --- | --- | --- |
| `policy.pte` | `nextpick-naivetransformer-role-d512-l8-h8-k5-through-lcp-df-sw` | 26.6 MB | 173 next-move logits + 10×173 slot logits |
| `masker.pte` | `masking-proonly-nov13-through-lcp-df-sw` | 10.0 MB | 10×173 role-conditioned completion logits |
| `offline.json` | both checkpoints + `champ_roles.json` | 40 KB | champions, roles, move plans, patches, lane table |
| `icons/*.png` | dDragon 16.20.1 | 5 MB | 173 bundled portraits |

The policy is **dynamically quantized int8** (XNNPACK); the masker ships **fp32**
because ExecuTorch 1.5's out-variant pass cannot lower its quantized ops, and
10 MB fp32 is a smaller risk than a hand-patched graph.

Both graphs take **int64 tensors only**, including the 0/1 masks. The AAR has
`Tensor.fromBlob(LongArray, shape)` but no `BooleanArray` overload, so the
exported graphs cast masks to `bool` internally. Changing that boundary means
re-exporting both `.pte` files, not patching Kotlin.

First launch copies the weights and portraits into `filesDir` once (a size
match short-circuits later launches). `Module.load` needs a real path, and Coil
needs real files. That copy is the only noticeable startup cost.

### Regenerating the assets

Run inside the project's dev container, where the ExecuTorch tooling is cached:

```bash
docker compose run --rm -T dev python android/tools/export_offline.py
```

The script reloads both checkpoints, re-exports the graphs, writes
`offline.json`, and downloads any missing portraits. The export is measured
once per checkpoint change; do not re-run it as a test.

### Build and verify

The toolchain is outside the repo (the default Windows home contains a space,
which breaks `sdkmanager.bat`):

```bash
TC=/e/android-toolchain
export JAVA_HOME="$TC/jdk/jdk-21.0.13+9"
export ANDROID_HOME="$TC/sdk"
"$TC/gradle/gradle-8.9/bin/gradle" --no-daemon :app:assembleDebug
"$TC/gradle/gradle-8.9/bin/gradle" --no-daemon :app:testDebugUnitTest :app:lintDebug
```

`local.properties` points at `E:/android-toolchain/sdk` and is git-ignored.

Diagnostics are on logcat under `LocalEngine` (module load, top-8 suggestions,
role-board method) and `BoardViewModel` (failures with stack traces):

```bash
adb logcat -v brief | grep -E 'LocalEngine|BoardViewModel'
```

## What the app does

- Empty-to-complete draft board: 5 bans and 5 picks per side, role badge and
  confidence per placed pick.
- Ranked next move (k=8), tapped to play into the slot the current turn implies.
- Patch selector (1-based vocabulary rows) and a first-pick toggle switch whose
  label sits inside the switch and whose thumb marks blue (left) or red (right).
- **Series and fearless pool**, entirely on-device.

`DraftApi` is retained in the tree as a reference implementation of the HTTP
contract but is no longer called. `BuildConfig.API_BASE` and
`network_security_config.xml` are likewise inert; they are left so the wire
format stays inspectable against `frontend/server.py`.

## Series and fearless

The pool is **stored per game, not derived from earlier games**. That mirrors
the reference client: each game slot owns an explicit, editable `fearless`
list, and "next game" folds the closed game's picks into a fresh copy.
Deriving it from the picks alone would make the manual edits unrepresentable.

Transitions, matching `app.js`:

- `next game` sets `done` on the current slot, computes
  `pool + picks` of this game, and inserts a successor at `current + 1`. Bans
  never enter the pool: a champion banned in game 1 is legal again in game 2.
  `addGame()` still exists in the ViewModel (it appends a slot inheriting the
  current pool unchanged) but the UI no longer exposes a `＋ game` button; game
  chips show only the side and pick count, not a `G1`/`G2` label.
- **First pick is kept, not flipped.** The teams swap sides between games but the
  side holding first pick stays the same in 94% of pro games.
- A closed game that already has a successor cannot be closed again — that would
  splice a duplicate in between. `clear` is the way to start over.
- The pool blocks **picks but never bans**. The serving head keeps that
  training-time rule; a fearless champion can still be suggested as a ban.

The pool is sent to the policy as 40 ascending champion-id tokens, PAD-filled,
all with FLEX roles — the same input `NaiveTransformerPolicy._batch` builds. It
is also applied as a logit mask before the top-k on picks.

Role-board routing matches `model_tools.predict_in_lane`: when the next move is
a pick and all ten bans are placed, the head answers each open cell with a
lane-constrained logit mask; during the ban phase the masking checkpoint
answers instead. The route is reported in logcat (`method=head|masker|mixed`).

## Walkthrough video

`video/blg_geng_worlds_2023.mp4` is a 59-second recording of the app replaying
all five games of the 2023 Worlds BLG vs Gen.G series from
`assets/demo/blg_geng_worlds2023.json`. Every move goes through the same
`placePick`/`placeBan` path a tap uses, so the ranked suggestions in the video
are the live ExecuTorch answers, not canned rows.

To replay it yourself:

```bash
adb shell am start -n com.draftassistant/.MainActivity --es demo blg_geng_worlds2023
```

The demo is opt-in via that intent extra; a normal launch is unaffected.

### Rank bar animation

`video/blg_geng_rank_bars.mp4` is a 25-second bar-chart animation built from
the same series. For each of the 50 real picks it asks where the picked champion
ranked in the served model's legal next-move list *before* that pick was
placed, then eases the cumulative bars forward as the games progress. There is
no fearless pool between games, matching the 2023 series and the app demo.

Final counts: `#1` 6, `#2` 2, `#3` 0, `#4` 3, `#5` 4, `6–10` 7, `11–20` 4,
`21+` 24 — 15/50 top-five hits (30 %). Top-five hits rise from 1/10 in game 1
to 7/10 in game 5.

The rank data is checked in as `video/blg_geng_rank_data.json`. Rebuild both
steps with:

```bash
docker compose run --rm -T dev python android/tools/rank_walkthrough.py
python android/tools/animate_rank_bars.py   # needs Pillow + ffmpeg
```

The rank step uses `NaiveTransformerPolicy.next_move`, the same path the
ExecuTorch export mirrors; the renderer only draws the resulting counts.

## Contract notes worth keeping

These are the places where a plausible-looking reimplementation is wrong:

- **`move_plan`'s second element is the SIDE (0 blue, 1 red), not a slot index.**
  A pick's slot is not in the plan at all. Position is found by *consuming
  counts* — n bans and n picks against the kinds in order — not by testing which
  slots look filled. See `data/DraftOrder.kt`.
- **Champion ids on the wire are Riot ids; tensors use 0..172 model indices.**
  `offline.json` carries both. Mixing them produces a confident, wrong answer.
- **`picks` is indexed by slot**, 10 entries, null for empty. Slots 0-4 are blue,
  5-9 red.
- **`fearless: []` differs from omitting `fearless`** in the old HTTP contract.
  The DTOs still set `encodeDefaults` for that reason.
- **Legality is a logit mask before the top-k**, value `-1e4`, never a filter
  afterwards. Filtering the top-k empties a lane cell 13 % of the time.
- **Patch ids are 1-based vocabulary rows**; 0 is UNK. 107 is valid, 108 is not.
- **Role keys are names** (`TOP..SUPPORT`) on every route. Never index a role
  list numerically.
- **A reply belongs to the board it was asked about.** `BoardViewModel` stamps
  each request with a board signature and drops a reply whose signature no
  longer matches; role badges are cleared on every board change.
- **Per-slot role assignment is a joint search**, not a per-champion argmax:
  120 injective placements per team. A plain argmax would put two flex champions
  in the same lane and disagree with the server's 96 %-agreement table.

## Layout

```
android/tools/export_offline.py      export .pte + metadata + icons
app/src/main/assets/
  policy.pte  masker.pte  offline.json  icons/*.png
app/src/main/java/com/draftassistant/
  MainActivity.kt            initialises LocalEngine with app context
  data/Models.kt             wire DTOs, field-for-field with server.py
  data/DraftOrder.kt         board -> next turn (ported, not reinvented)
  data/LocalEngine.kt        ExecuTorch runtime, tensor builder, legality, role table
  data/DraftApi.kt           legacy HTTP reference, no longer called
  ui/BoardViewModel.kt       board + series state, debounce, stale-reply guard
  ui/BoardScreen.kt          board, series bar, fearless strip, suggestions
  ui/ChampionPicker.kt       bundled-portrait champion grid
  ui/Theme.kt                palette lifted from style.css
app/src/test/java/com/draftassistant/data/
  DraftOrderTest.kt          pins the plan to infer.move_plan's four sequences
```

## Deliberately absent

- No real-draft walkthrough, agent chat, or LLM features — those need data or
  tokens that do not belong on a phone.
- No checkpoint picker. The two shipped defaults are compiled into the assets.
- No series persistence across process death. The web client uses
  `localStorage`; this holds the series in memory only.

## Next steps, in order

1. Reach a full 20-move draft end-to-end and confirm the header reads "draft
   complete" at move 21, not earlier.
2. Exercise the head path of the role board (ban phase complete) on device and
   compare a lane cell against `model_tools.predict_in_lane`.
3. Persist the series across process death.
4. Consider shipping the masker quantized if ExecuTorch gains the missing
   out-variants — it would save ~6 MB.
