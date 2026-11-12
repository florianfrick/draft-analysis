"""A scripted turn through the tool layer, with the log and the ablation.

Not an agent - it calls the tools in a fixed order so that the wiring, the
instrumentation and the ablation can be checked without an LLM in the loop:

    python pro/model/agent/demo.py                 # all tools
    python pro/model/agent/demo.py --ablate sql    # the same turn, no SQL
    python pro/model/agent/demo.py --ablate predict_next_move,rank_candidates

The interesting output is the last block. `sql_share` is the number the design
turns on: if the agent reaches for SQL more than half the time, the high-level
tools are under-specified.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.agent import draft_state, schema, model_tools  # noqa: E402
from pro.model.agent.session import Session                   # noqa: E402

BOARD = dict(
    blue_picks=["Aatrox", "Vi", "Ahri"],
    red_picks=["Poppy", "Lee Sin"],
    bans=["Kled", "Olaf", "Jarvan IV", "Nocturne", "Locke", "Rumble"],
    first_pick=0,          # blue first - `red_first=False` in MOVES below
    patch="16.20",
)

# The same board as a chronological move log, which is what a state is. With
# `red_first=False` the schedule puts blue's bans at 0, 2, 4 and blue's picks at
# 6, 9, 10 - so this list reproduces BOARD exactly, and `main` asserts it does.
MOVES = ["Kled", "Olaf", "Jarvan IV", "Nocturne", "Locke", "Rumble",
         "Aatrox", "Poppy", "Lee Sin", "Vi", "Ahri"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "pro" / "model" / "checkpoints"
                                          / "nextpick-sharedsplitthrough-las-dk-c-t1-r"
                                          / "best.pt"))
    ap.add_argument("--db", default=str(schema.DB_PATH))
    ap.add_argument("--ablate", default="")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    ablate = {s.strip() for s in args.ablate.split(",") if s.strip()}
    predictor = None
    if Path(args.ckpt).exists():
        from solo_queue.model.policy.infer import PolicyPredictor
        predictor = PolicyPredictor(args.ckpt, device="cpu")
    else:
        print("no checkpoint at %s - model tools will be unavailable" % args.ckpt)
    con = schema.connect(args.db) if Path(args.db).exists() else None
    if con is None:
        print("no database at %s - run build_db.py first" % args.db)

    names = model_tools.ChampIndex(schema.names())
    live = draft_state.DraftState(patch=BOARD["patch"], red_first=False)
    for move in MOVES:
        live = live.apply(move)
    keyed = {k: ([draft_state.key(x) for x in v] if isinstance(v, list) else v)
             for k, v in BOARD.items()}
    assert live.board() == keyed, "MOVES no longer reproduces BOARD"

    s = Session(con=con, predictor=predictor,
                champs=names if predictor else None, ablate=ablate,
                store=draft_state.Store(names, real=live))
    print("tools offered:", ", ".join(s.names()) or "none")

    out = s.call("predict_next_move", k=5, **BOARD)
    if out.get("error"):
        print("predict_next_move ->", out["error"])
    else:
        print("predict_next_move -> %s to %s, move %s; %s"
              % (out["side"], out["kind"], out["move_idx"],
                 ", ".join("%s %.1f%%" % (c["name"], 100 * c["p_head"])
                           for c in out["top"][:3])))

    out = s.call("rank_candidates", candidates=["Xayah", "Jinx", "Ashe"], **BOARD)
    if out.get("error"):
        print("rank_candidates ->", out["error"])
    else:
        print("rank_candidates -> %s"
              % ", ".join("%s %.1f%%" % (c["name"], 100 * c["share_of_candidates"])
                          for c in out["candidates"]))

    out = s.call("sql", query=(
        "SELECT champ_key, COUNT(*) n FROM move WHERE kind='ban' "
        "AND split='train' AND patch_rank >= (SELECT MAX(patch_rank)-2 FROM patch) "
        "GROUP BY 1 ORDER BY n DESC LIMIT 3"))
    if out.get("error"):
        print("sql ->", out["error"])
    else:
        print("sql -> bans in the last 3 patches: %s"
              % ", ".join("%s %d" % (r["champ_key"], r["n"]) for r in out["rows"]))

    out = s.call("sql", query="SELECT COUNT(*) n FROM move WHERE kind=1")
    print("sql (the mistake a model makes) ->", out.get("error") or out["rows"],
          "| hint:", (out.get("hint") or "")[:70])

    # --- the state workflow: the board is held, never retyped -----------
    print("\n-- state --")
    out = s.call("get_state")
    print("get_state -> move %s, %s %s, %d champions legal"
          % (out["move_idx"], out["side"], out["kind"], out["n_legal"]))

    sandbox = s.call("clone_state", name="alistar-line")["state"]
    out = s.call("apply_move", state=sandbox, champion="Alistar")
    played = out.get("applied") or {}
    print("apply_move(Alistar) -> played at move %s (%s %s); now move %s, %s %s"
          % (played.get("move_idx"), played.get("side"), played.get("kind"),
             out["move_idx"], out["side"], out["kind"]))

    out = s.call("predict_next_move", state=sandbox, k=5)
    if out.get("error"):
        print("predict on the sandbox ->", out["error"])
    else:
        print("predict on the sandbox -> %s | head agrees with the schedule: %s"
              % (", ".join("%s %.1f%%" % (c["name"], 100 * c["p_head"])
                           for c in out["top"][:3]),
                 out.get("move_matches_state")))

    out = s.call("undo_move", state=sandbox)
    print("undo -> sandbox back at move %s; live board still at move %s"
          % (out["move_idx"], s.store.get(draft_state.REAL_ID).move_idx))

    print("\nsummary:", s.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
