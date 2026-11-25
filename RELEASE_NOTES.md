# Release notes — v0.1.0

First GitHub source release for the LoL Draft Assistant.

## Highlights

- Native Kotlin/Compose Android app with a custom launcher icon. The offline
  core runs both models on the phone with ExecuTorch and needs no backend,
  login, network access, or API key. The old HTTP fallback was removed.
- Series and fearless support, including `nextGame()` carryover and alternating
  first-pick side.
- Bundled 2026 Worlds final BLG 3–1 Gen.G walkthrough and rank-bar animation.
- Opt-in Agent chat for explanations. It is off by default, never fires on a
  board update, and uses a Cloudflare Worker so the DeepSeek key stays
  server-side.
- The research pipeline, evaluation split (including the `role_next` suite),
  baselines, and negative-result history remain in the repository as the
  provenance of the shipped models.

## What is in the repository

| Path | Role |
|---|---|
| `android/` | The shipped app, ExecuTorch assets, portraits, demo data, and optional Worker proxy. |
| `pro/`, `solo_queue/`, `data/`, `webapp/` | Reproduction and evaluation tooling; not required for the Android app. |
| `STATUS.md`, `FINDINGS.md`, `PROJECT_REPORT.md` | Measurements, archived experiment index, and project summary. |
| `LICENSE`, `ASSET_NOTICE.md` | Code license and third-party asset notice. |

## Verification

- `:app:assembleDebug`, `:app:testDebugUnitTest` (11 tests), and
  `:app:lintDebug` pass.
- On-device next-move output matches the Python serving path on representative
  boards; sub-percent differences are dynamic-int8 noise. See
  `android/README.md` for the parity table.
- No secrets, `.env`, checkpoints, raw data, build outputs, or APKs are tracked.

## Install and try

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.draftassistant/.MainActivity \
  --es demo blg_geng_worlds2026
```

The attached debug APK is for evaluation. A public GitHub Release should also
attach a signed APK/AAB built with your own keystore.

## Known limitations

- `targetSdk = 34`; a real edge-to-edge / insets pass is still needed before a
  Play Store target-SDK bump.
- Series state does not survive process death.
- The masker is FP32 because ExecuTorch 1.5 lacks the quantized out-variants.
- The optional Agent chat needs the deployed Worker and its Cloudflare secret;
  the offline app does not.
- Champion names and portraits are Riot-owned assets. See `ASSET_NOTICE.md`.
- The `.pte` assets are shipped in the repo, but regenerating them requires the
  git-ignored source checkpoints and `android/tools/export_offline.py`.
