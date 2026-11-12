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

from pro.model.agent import schema, model_tools          # noqa: E402
from pro.model.agent.session import Session              # noqa: E402

BOARD = dict(
    blue_picks=["Aatrox", "Vi", "Ahri"],
    red_picks=["Poppy", "Lee Sin"],
    bans=["Kled", "Olaf", "Jarvan IV", "Nocturne", "Locke", "Rumble"],
    first_pick=1,
    patch="16.20",
)


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

    s = Session(con=con, predictor=predictor,
                champs=model_tools.ChampIndex(schema.names()) if predictor else None,
                ablate=ablate)
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

    print("\nsummary:", s.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
