#!/usr/bin/env python3
"""Rank every real pick in the BLG/Gen.G walkthrough against the served model.

The animation this feeds asks a simple question: as the 2026 Worlds final
unfolded, how often was the champion a team actually picked the model's 1st,
2nd, 3rd ... suggestion?  Ranking is done on the board *before* the pick is
placed, through the exact `NaiveTransformerPolicy.next_move` path the Android
app's ExecuTorch model mirrors.  The fearless pool carries picks from earlier
games in the series, exactly like the app demo's `nextGame()` path.

Run from the repo root in the dev container:

    docker compose run --rm -T dev python android/tools/rank_walkthrough.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch  # noqa: F401  (the policy import needs the torch runtime)

ROOT = Path(__file__).resolve().parents[2]
sys_path = str(ROOT)
import sys  # noqa: E402
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from pro.model.naivetransformer.serve import NaiveTransformerPolicy  # noqa: E402

DEMO = ROOT / "android/app/src/main/assets/demo/blg_geng_worlds2026.json"
OFFLINE = ROOT / "android/app/src/main/assets/offline.json"
OUT = ROOT / "android/video/blg_geng_rank_data.json"


def main() -> None:
    series = json.loads(DEMO.read_text(encoding="utf-8"))
    meta = json.loads(OFFLINE.read_text(encoding="utf-8"))
    by_riot = {int(c["riot"]): int(c["idx"]) for c in meta["champions"]}
    names = {int(c["idx"]): str(c["name"]) for c in meta["champions"]}
    patch_id = int(meta["default_patch_id"])

    ckpt = ROOT / meta["models"]["policy"]["ckpt"]
    print(f"loading {ckpt.name} ...", flush=True)
    policy = NaiveTransformerPolicy(str(ckpt), device="cpu")

    records = []
    pool: set[int] = set()
    for game_no, game in enumerate(series["games"], 1):
        picks: list[int | None] = [None] * 10
        bans: list[int] = []
        first_pick = int(game.get("firstPick", 0))
        expected = {by_riot[int(r)] for r in game.get("unavailable", [])}
        if expected != pool:
            raise ValueError(f"fearless pool mismatch at game {game_no}")
        for move_no, move in enumerate(game["moves"], 1):
            champ = by_riot[int(move["c"])]
            if move["k"] == "ban":
                bans.append(champ)
                continue
            if move["k"] != "pick":
                raise ValueError(f"unknown move kind {move['k']!r}")

            result = policy.next_move(
                picks, bans, first_pick=first_pick,
                k=policy.num_champs, patch_id=patch_id, game_num=0,
                unavailable=sorted(pool),
            )
            if result["complete"] or result["kind"] != "pick":
                raise ValueError(f"board out of sync at G{game_no} M{move_no}")
            side = int(move["s"])
            if int(result["side"]) != side:
                raise ValueError(
                    f"side mismatch at G{game_no} M{move_no}: "
                    f"dataset {side}, model {result['side']}")

            order = [int(s["idx"]) for s in result["suggestions"]]
            rank = order.index(champ) + 1 if champ in order else None
            prob = next(
                (float(s["prob"]) for s in result["suggestions"]
                 if int(s["idx"]) == champ), None,
            )
            records.append({
                "game": game_no,
                "move": move_no,
                "side": side,
                "champion": str(move["name"]),
                "riot": int(move["c"]),
                "idx": champ,
                "rank": rank,
                "prob": prob,
                "top5": [names[i] for i in order[:5]],
                "top1": names[order[0]] if order else None,
            })

            start = 0 if side == 0 else 5
            for slot in range(start, start + 5):
                if picks[slot] is None:
                    picks[slot] = champ
                    break
            else:
                raise ValueError(f"no free slot for side {side}")
        # The next game's pool is this pool plus this game's picks. Bans do
        # not carry over, exactly like BoardViewModel.nextGame().
        pool.update(c for c in picks if c is not None)

    buckets = {str(i): 0 for i in range(1, 6)}
    buckets["6+"] = 0
    for rec in records:
        key = str(rec["rank"]) if rec["rank"] is not None and rec["rank"] <= 5 else "6+"
        buckets[key] += 1

    payload = {
        "title": "BLG vs Gen.G — Worlds 2026 Final",
        "series": series["title"],
        "model": meta["models"]["policy"]["ckpt"],
        "patch": meta["default_patch"],
        "rank_definition": (
            "Position of the actually picked champion in the model's legal "
            "next-move ranking, evaluated before that pick was placed. "
            "Fearless pool = picks from earlier games in this series; bans "
            "do not carry over."
        ),
        "records": records,
        "final_counts": buckets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print("final counts:", buckets)
    print("ranks:", [r["rank"] for r in records])


if __name__ == "__main__":
    main()
