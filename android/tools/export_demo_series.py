#!/usr/bin/env python3
"""Export one Oracle's Elixir series into the Android demo replay asset.

Uses the project pipeline's draft reconstruction so bans, pick slots,
first-pick side and fearless context match the corpus exactly.  The app's
demo then advances with `nextGame()`, which carries the previous game's picks
into the pool; this script asserts that this is exactly what OE's fearless
`unavailable` list records.

Example (from the repo root, in the dev container):

    python android/tools/export_demo_series.py \
      --csv data/pro play/.downloaded_2026_oe.csv \
      --out android/app/src/main/assets/demo/blg_geng_worlds2026.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/pro play"))

import pipeline  # noqa: E402  (the data dir has a space, so no dotted import)

OFFLINE = ROOT / "android/app/src/main/assets/offline.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.csv = args.csv.resolve()
    args.out = args.out.resolve()
    meta = json.loads(OFFLINE.read_text(encoding="utf-8"))
    idx_to_riot = {int(c["idx"]): int(c["riot"]) for c in meta["champions"]}
    idx_to_name = {int(c["idx"]): str(c["name"]) for c in meta["champions"]}

    name_to_idx, _num = pipeline.load_name_to_idx()
    games = pipeline.group_games([args.csv])
    target_ids = {
        "LOLTMNT02_487405", "LOLTMNT02_487406",
        "LOLTMNT02_487408", "LOLTMNT02_488489",
    }
    picked = [g for g in games if g["gameid"] in target_ids]
    if len(picked) != len(target_ids):
        raise SystemExit(f"expected {len(target_ids)} games, found {len(picked)}")
    pipeline.assign_series(picked)

    rows = []
    for game in picked:
        row, reason = pipeline.game_to_draft(game, name_to_idx)
        if row is None:
            raise SystemExit(f"{game['gameid']}: {reason}")
        rows.append(row)
    pipeline.attach_fearless(rows)
    rows.sort(key=lambda r: r["game_number"])

    # nextGame() = current pool + this game's PICKS. Bans never enter it.
    seen: set[int] = set()
    games_out = []
    for row in rows:
        expected = set(row["unavailable"])
        if expected != seen:
            raise SystemExit(
                "fearless pool mismatch at game %d: OE %s, nextGame %s"
                % (row["game_number"], sorted(expected), sorted(seen)))
        moves = []
        for kind, side, champ_idx, _role in row["moves"]:
            moves.append({
                "k": "ban" if int(kind) == pipeline.MOVE_BAN else "pick",
                "s": int(side),
                "c": idx_to_riot[int(champ_idx)],
                "name": idx_to_name[int(champ_idx)],
            })
        games_out.append({
            "blue": row["blue_team"],
            "red": row["red_team"],
            "firstPick": int(row["first_pick"]),
            "gameNumber": int(row["game_number"]),
            "patch": row["patch"],
            "date": row["date"],
            "fearless": row["fearless"],
            "unavailable": [idx_to_riot[i] for i in row["unavailable"]],
            "moves": moves,
        })
        seen.update(int(c) for c in row["champ"])

    payload = {
        "title": "BLG vs Gen.G — Worlds 2026 Final",
        "series": rows[0]["series_id"],
        "source": str(args.csv),
        "games": games_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("wrote", args.out.relative_to(ROOT))
    print("series", payload["series"])
    print("firstPick", [g["firstPick"] for g in games_out],
          "fearless", rows[0]["fearless"])
    for g in games_out:
        print(" G%d %s vs %s patch %s unavailable=%d moves=%d"
              % (g["gameNumber"], g["blue"], g["red"], g["patch"],
                 len(g["unavailable"]), len(g["moves"])))


if __name__ == "__main__":
    main()
