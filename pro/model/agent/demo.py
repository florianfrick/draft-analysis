"""A scripted turn through the tool layer, with the log and the ablation.

Not an agent - it calls the tools in a fixed order so that the wiring, the
instrumentation and the ablation can be checked without an LLM in the loop:

    python pro/model/agent/demo.py                 # all tools
    python pro/model/agent/demo.py --ablate sql    # the same turn, no SQL
    python pro/model/agent/demo.py --ablate predict_next_move

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
from pro.model.agent.engine import Engine                     # noqa: E402
from pro.model.agent.session import Session                   # noqa: E402

BOARD = dict(
    blue_picks=["Aatrox", "Vi", "Ahri"],
    red_picks=["Poppy", "Lee Sin", "Xin Zhao"],
    bans=["Kled", "Olaf", "Jarvan IV", "Nocturne", "Locke", "Rumble",
          "Sylas", "Ryze", "Cassiopeia", "Nautilus"],
    first_pick=0,          # blue first - `red_first=False` in MOVES below
    patch="16.20",
)

# The same board as a chronological move log, which is what a state is. With
# `red_first=False` the schedule is 6 bans, 6 picks, 4 bans, 4 picks - so this
# list reproduces BOARD exactly at move 16, and `main` asserts it does.
#
# Move 16 is deliberate: a *phase-2* pick. The role question can only be asked
# of the head once the ban block is complete (see `model_tools.can_answer_in_lane`),
# and a demo that lands in phase 1 would only show the refusal.
_BANS = BOARD["bans"]
MOVES = (_BANS[:6]
         + ["Aatrox", "Poppy", "Lee Sin", "Vi", "Ahri", "Xin Zhao"]
         + _BANS[6:])


def _short(rows, key, n=3):
    return ", ".join("%s %.1f%%" % (c["name"], 100 * c[key])
                     for c in rows[:n]) or "-"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "pro" / "model" / "checkpoints"
                                          / "nextpick-sharedsplitthrough-las-dk-c-t1-r"
                                          / "best.pt"))
    ap.add_argument("--masker-ckpt", default="",
                    help="a masking checkpoint: needed for a role question "
                         "during the ban phase")
    ap.add_argument("--db", default=str(schema.DB_PATH))
    ap.add_argument("--ablate", default="")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    ablate = {s.strip() for s in args.ablate.split(",") if s.strip()}
    predictor = masker = None
    if Path(args.ckpt).exists():
        from solo_queue.model.policy.infer import PolicyPredictor
        predictor = PolicyPredictor(args.ckpt, device="cpu")
    else:
        print("no checkpoint at %s - the next-move tool will be unavailable"
              % args.ckpt)
    if args.masker_ckpt and Path(args.masker_ckpt).exists():
        from solo_queue.model.policy.infer import PolicyPredictor
        masker = PolicyPredictor(args.masker_ckpt, device="cpu")
    con = schema.connect(args.db) if Path(args.db).exists() else None
    if con is None:
        print("no database at %s - run build_db.py first" % args.db)

    names = model_tools.ChampIndex(schema.names())
    live = draft_state.DraftState(patch=BOARD["patch"], red_first=False,
                                  engine=Engine(names, predictor=predictor,
                                                masker=masker,
                                                patch=BOARD["patch"]))
    for move in MOVES:
        live = live.apply(move)
    keyed = {k: ([draft_state.key(x) for x in v] if isinstance(v, list) else v)
             for k, v in BOARD.items()}
    # `board()` carries the fearless pool even when it is empty, because a
    # reader that cannot tell "no pool" from "pool not reported" will propose a
    # champion from the previous game.
    keyed["unavailable"] = []
    assert live.board() == keyed, "MOVES no longer reproduces BOARD"

    store = draft_state.Store(names, real=live)
    s = Session(con=con, store=store, ablate=ablate)
    print("tools offered:", ", ".join(s.names()) or "none")

    out = s.call("get_state")
    print("get_state -> move %s, %s %s, %d champions legal"
          % (out["move_idx"], out["side"], out["kind"], out["n_legal"]))
    print("             roles: %s"
          % ", ".join("%s %s %.2f" % (r["champion"], r["role"], r["p_lane"])
                      for r in out["roles"]))
    print("             still needs: blue %s | red %s"
          % (out["open_roles"]["blue"], out["open_roles"]["red"]))

    out = s.call("predict_next_move", state="real", k=5)
    if out.get("error"):
        print("predict_next_move ->", out["error"])
    else:
        print("predict_next_move -> %s to %s, move %s; %s"
              % (out["side"], out["kind"], out["move_idx"],
                 _short(out["top"], "p_head")))

    out = s.call("predict_next_move", state="real", role="jungle", side="red", k=3)
    if out.get("error"):
        print("predict_next_move(role) ->", out["error"])
    else:
        print("predict_next_move(red jungle, by %s) -> %s"
              % (out.get("method"), _short(out["top"], "p_in_lane")))

    out = s.call("predict_roles", state="real")
    if out.get("error"):
        print("predict_roles ->", out["error"])
    else:
        print("predict_roles -> blue %s | red %s | open %s"
              % (", ".join("%s %s" % (r["champion"], r["role"])
                           for r in out["blue"]),
                 ", ".join("%s %s" % (r["champion"], r["role"])
                           for r in out["red"]),
                 out["open_roles"]))

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
    sandbox = s.call("clone_state", name="alistar-line")["state"]
    out = s.call("apply_move", state=sandbox, champion="Alistar")
    played = out.get("applied") or {}
    print("apply_move(Alistar) -> played at move %s (%s %s); now move %s, %s %s"
          % (played.get("move_idx"), played.get("side"), played.get("kind"),
             out["move_idx"], out["side"], out["kind"]))
    print("             the role it implies: %s"
          % ", ".join("%s %s %.2f" % (r["champion"], r["role"], r["p_lane"])
                      for r in out["roles"]))

    out = s.call("predict_next_move", state=sandbox, k=5)
    if out.get("error"):
        print("predict on the sandbox ->", out["error"])
    else:
        print("predict on the sandbox -> %s" % _short(out["top"], "p_head"))

    out = s.call("undo_move", state=sandbox)
    print("undo -> sandbox back at move %s; live board still at move %s"
          % (out["move_idx"], s.store.get(draft_state.REAL_ID).move_idx))

    print("\nsummary:", s.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
