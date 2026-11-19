# Third-party asset notice

The MIT license in `LICENSE` applies to the original code in this repository.
It does **not** apply to the third-party material listed below.

## League of Legends / Riot Games

League of Legends and all associated names, champion names, logos, artwork,
and game imagery are trademarks or copyrighted works of Riot Games, Inc.
This project is an unofficial, non-commercial fan project and is not
endorsed by, sponsored by, or affiliated with Riot Games.

The Android app bundles 173 champion portraits under
`android/app/src/main/assets/icons/`. They were downloaded from Riot's Data
Dragon (version 16.20.1) and are included only so the app can identify picks
and bans without a network connection. Champion names also appear in
`data/champ_roles.json`, `android/app/src/main/assets/offline.json`, chat
output, and the demo videos.

If Riot Games asks for these assets to be removed, they should be removed from
a future build and replaced with neutral placeholder art. Do not represent
this project as an official Riot product.

## Demo data and video

`android/app/src/main/assets/demo/blg_geng_worlds2026.json` and the files under
`android/video/` were derived from public match records (Oracle's Elixir) and
the app's own replay output. The raw CSVs are intentionally not committed.
Check the source provider's current terms before redistributing derived data
or footage in a public release.

## Model assets

`android/app/src/main/assets/policy.pte` and `masker.pte` are derived from this
project's own checkpoints and source data. The original `.pt` checkpoints are
git-ignored; see `STATUS.md` for provenance and `android/tools/export_offline.py`
for the export path.
