"""Drive the tool layer by hand: real val states, answered blind, scored after.

This is the zero-cost version of the 200-state eval. No API is called - the model
is whoever is reading this - so the sample is small and *stratified* rather than
random, because a random draw would be dominated by boards where the head is
already right and would say nothing about the only slice where an agent can add
value.

The stratification is by **where the head ranks the true next pick**:

    bucket 1      the head is already right        - any agent should match it
    bucket 2-5    the reranker's job               - a good answer moves it up
    bucket 6-20   the agent's slice (~13.65 % of val) - the only place it matters

Two rules make the exercise honest:

**The bucket is hidden while answering.** Printing it would tell the reader which
boards to distrust, which is information a real agent never has. It is assigned
at sampling time, stored in the state file, and only shown by `score`.

**The truth is never printed until `score`.** The boards go out; the answers come
back; the reveal is a separate command.

    python pro/model/agent/handrun.py sample --n 6
    python pro/model/agent/handrun.py call --state A --calls '[{"id":"A",
        "tool":"predict_next_move","args":{"state":"real","k":5}}]'
    python pro/model/agent/handrun.py score --answers '{"A":"Xayah"}'

`call --state A` publishes state A's board as `real` before running the calls,
because no tool takes a board any more: the board is the state, and the sampled
one has to be put somewhere the tools can read it.

`call` goes through `Session`, so the tool mix it reports is the same number the
real loop reports - the instrumentation is not a mock.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pro.model.agent import draft_state, model_tools, schema  # noqa: E402
from pro.model.agent.loop import DEFAULT_CKPT, _load, render_board  # noqa: E402

STATE = ROOT / ".handrun.json"
MIN_MOVE = 11          # the first pick of the second round: a real decision point
BUCKETS = {"1": (1, 1), "2-5": (2, 5), "6-20": (6, 20)}


def _board(con, gameid, move_idx):
    """The board as the model sees it: moves strictly before `move_idx`."""
    rows = con.execute(
        "SELECT kind, side, champ_key FROM move WHERE gameid = ? AND move_idx < ?"
        " ORDER BY move_idx", (gameid, move_idx)).fetchall()
    game = con.execute("SELECT first_pick, patch FROM game WHERE gameid = ?",
                       (gameid,)).fetchone()
    blue = [k for kind, side, k in rows if kind == "pick" and side == "blue"]
    red = [k for kind, side, k in rows if kind == "pick" and side == "red"]
    bans = [k for kind, side, k in rows if kind == "ban"]
    return {"blue_picks": blue, "red_picks": red, "bans": bans,
            "first_pick": int(game[0]), "patch": game[1]}


def _flat_picks(board):
    """`blue_picks` / `red_picks` -> the UI's ten-slot pick list.

    `log_from_board` reads the board the frontend publishes, which is slots
    (blue 0-4, red 5-9), not two per-side lists. Handing it only blue's picks
    would publish a shorter draft than the one that was sampled.
    """
    blue = list(board.get("blue_picks") or [])[:5]
    red = list(board.get("red_picks") or [])[:5]
    return (blue + [None] * (5 - len(blue))
            + red + [None] * (5 - len(red)))


def _candidates(con, limit=None):
    """Every val pick-state at or after `MIN_MOVE`, in a stable order.

    No `LIMIT` before the shuffle: cutting the pool first biases it to
    alphabetically early game ids, which is also early dates. 24k rows is
    nothing to hold in memory, and the seeded shuffle is what makes a run
    reproducible.
    """
    sql = ("SELECT gameid, move_idx, kind, side, champ_key FROM move"
           " WHERE split = 'val' AND kind = 'pick' AND move_idx >= ?"
           " ORDER BY gameid, move_idx")
    if limit:
        sql += " LIMIT %d" % int(limit)
    return con.execute(sql, (MIN_MOVE,)).fetchall()


def _head_rank(predictor, champs, board, truth, kind):
    out = model_tools.predict_next_move(
        predictor, champs, board["blue_picks"], board["red_picks"],
        board["bans"], first_pick=board["first_pick"], patch=board["patch"],
        k=20)
    if out.get("error"):
        return None, out["error"]
    for c in out.get("top") or []:
        if model_tools._norm(c["name"]) == truth:
            return c["rank"], None
    return None, None       # outside the top 20


def sample(con, predictor, champs, n=6, seed=0, per_bucket=2):
    """Fill each bucket, then shuffle so the order leaks nothing."""
    picked, errors = {}, []
    rows = _candidates(con)
    random.Random(seed).shuffle(rows)
    for gameid, move_idx, kind, side, truth in rows:
        # `all()` over an empty dict is True, so the guard has to check that
        # every bucket exists before it can conclude that every bucket is full.
        if len(picked) == len(BUCKETS) and all(len(v) >= per_bucket
                                               for v in picked.values()):
            break
        board = _board(con, gameid, move_idx)
        rank, err = _head_rank(predictor, champs, board, truth, kind)
        if err:
            errors.append(err)
            break
        bucket = next((b for b, (lo, hi) in BUCKETS.items()
                       if rank is not None and lo <= rank <= hi), None)
        if bucket is None or len(picked.get(bucket, [])) >= per_bucket:
            continue
        picked.setdefault(bucket, []).append({
            "gameid": gameid, "move_idx": move_idx, "kind": kind, "side": side,
            "patch": board["patch"], "board": board, "truth": truth,
            "head_rank": rank, "bucket": bucket})
    states = [s for b in sorted(picked) for s in picked[b]]
    random.Random(seed + 1).shuffle(states)
    ids = "ABCDEFGHIJ"
    out = {ids[i]: s for i, s in enumerate(states[:n])}
    STATE.write_text(json.dumps({"states": out, "calls": [], "answers": {}},
                                indent=1), encoding="utf-8")
    return out, errors


def _session():
    return _load(DEFAULT_CKPT, str(schema.DB_PATH), set(), 40)


def cmd_sample(args):
    session = _session()
    states, errors = sample(session.con, session.predictor, session.champs,
                            n=args.n, seed=args.seed)
    for err in errors:
        print("sampling stopped:", err, file=sys.stderr)
    for sid, s in states.items():
        print("=== %s ===  to move: %s %s" % (sid, s["side"], s["kind"]))
        print(render_board(s["board"]))
        print()
    print("%d states. answer with `score --answers '{\"A\": \"<champion>\"}'`"
          % len(states))
    return 0


def cmd_call(args):
    blob = json.loads(STATE.read_text(encoding="utf-8"))
    session = _session()
    if args.state:
        sampled = blob["states"].get(args.state)
        if sampled is None:
            print("no sampled state %r; have: %s"
                  % (args.state, ", ".join(sorted(blob["states"]))),
                  file=sys.stderr)
            return 2
        board = sampled["board"]
        session.store.set_real(draft_state.log_from_board(
            _flat_picks(board), board["bans"],
            red_first=bool(board["first_pick"]), patch=board["patch"],
            unavailable=board.get("unavailable") or (),
            engine=session.store.engine))
        print("published state %s as `real`: move %d, %s %s"
              % (args.state, session.store.get("real").move_idx,
                 sampled["side"], sampled["kind"]))
    raw = args.calls
    # `@file` because a board is ten champion names and six boards is more JSON
    # than a shell argument survives.
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    for spec in json.loads(raw):
        out = session.call(spec["tool"], **(spec.get("args") or {}))
        blob["calls"].append({"id": spec.get("id"), "tool": spec["tool"],
                              "args": spec.get("args") or {},
                              "ok": "error" not in out})
        print("%s %s -> %s" % (spec.get("id"), spec["tool"],
                               _compact(spec["tool"], out)))
    STATE.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    print("\nsummary: %s" % session.summary())
    return 0


def _compact(name, out):
    if "error" in out:
        return "REFUSED: %s" % out["error"]
    if name == "predict_next_move":
        return "%s to %s, move %s | %s" % (
            out["side"], out["kind"], out["move_idx"],
            ", ".join("%s %.1f%% #%s" % (c["name"], 100 * c["p_head"], c["rank"])
                      for c in out["top"]))
    if name == "predict_roles":
        return "blue %s | red %s | open %s" % (
            ", ".join("%s %s %.2f" % (r["champion"], r["role"], r["p_lane"])
                      for r in out["blue"]) or "-",
            ", ".join("%s %s %.2f" % (r["champion"], r["role"], r["p_lane"])
                      for r in out["red"]) or "-",
            out["open_roles"])
    if name == "sql":
        return "%d row(s): %s" % (out["n_rows"], json.dumps(out["rows"][:5]))
    return json.dumps(out)[:300]


def cmd_score(args):
    """Score two different questions, and keep them apart.

    **Accuracy** - did the answer match the truth - is the head's problem. The
    heads are well trained and this project is not trying to beat them, so a miss
    here is usually the head being wrong, not the interface.

    **Fidelity** - did the answer match the head's own top pick - is this
    project's problem. A chat assistant that faithfully carries the model's
    ranking to a user is doing its job even when that ranking is wrong, so this
    is the number to read. It is recomputed here rather than stored, so it works
    on any past run.
    """
    blob = json.loads(STATE.read_text(encoding="utf-8"))
    answers = json.loads(args.answers)
    session = _session()
    hits, faithful, rows = 0, 0, []
    for sid, s in sorted(blob["states"].items()):
        mine = (answers.get(sid) or "").strip()
        board = s["board"]
        out = model_tools.predict_next_move(
            session.predictor, session.champs, board["blue_picks"],
            board["red_picks"], board["bans"], first_pick=board["first_pick"],
            patch=board["patch"], k=1)
        head1 = ((out.get("top") or [{}])[0]).get("name")
        ok = model_tools._norm(mine) == s["truth"]
        agree = head1 is not None and model_tools._norm(mine) == model_tools._norm(head1)
        hits += ok
        faithful += agree
        rows.append((sid, s["bucket"], s["head_rank"], mine or "-", s["truth"],
                     head1 or "-", "HIT" if ok else "miss",
                     "agree" if agree else "differs"))
        blob["answers"][sid] = mine
    print("%-3s %-6s %-5s %-12s %-12s %-12s %-6s %s"
          % ("id", "bucket", "head", "answer", "truth", "head#1", "truth",
             "head"))
    for r in rows:
        print("%-3s %-6s %-5s %-12s %-12s %-12s %-6s %s"
              % (r[0], r[1], r[2] if r[2] is not None else ">20",
                 r[3], r[4], r[5], r[6], r[7]))
    n = len(rows)
    print("\nfidelity (answer == head's own #1): %d/%d = %.1f%%   <- the interface"
          % (faithful, n, 100 * faithful / n if n else 0))
    print("accuracy (answer == truth):          %d/%d = %.1f%%   <- the head's job"
          % (hits, n, 100 * hits / n if n else 0))
    for bucket in sorted(BUCKETS):
        got = [r for r in rows if r[1] == bucket]
        if got:
            print("  bucket %-4s fidelity %d/%d  accuracy %d/%d"
                  % (bucket, sum(1 for r in got if r[7] == "agree"), len(got),
                     sum(1 for r in got if r[6] == "HIT"), len(got)))
    by_tool = {}
    for c in blob["calls"]:
        by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
    print("calls: %d  %s" % (len(blob["calls"]), by_tool))
    STATE.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)
    c = sub.add_parser("call")
    c.add_argument("--calls", required=True)
    c.add_argument("--state", default=None,
                   help="a sampled state id (A, B, ...) to publish as `real` "
                        "before the calls run")
    c.set_defaults(func=cmd_call)
    r = sub.add_parser("score")
    r.add_argument("--answers", required=True)
    r.set_defaults(func=cmd_score)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
