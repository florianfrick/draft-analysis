"""Is the masking model still earning its slot? (role board vs. a table)

The role board (`/api/suggest`) is the masking checkpoint's only remaining job.
This scores it against the two things that could replace it - a per-lane
popularity table, and the next-move head with a lane mask - on the same
held-out games, so the answer is a number instead of an argument.

    python frontend/probe_roleboard.py            # 250 games
    python frontend/probe_roleboard.py 400        # tighter intervals

Target for every cell is the champion that team actually fielded in that lane;
the score is whether it is the model's top-1 / in its top-5. Three board
states, because the board-conditioned part of the model can only show up where
there is a board:

    A  six bans, no picks        - the prior's best case
    B  six bans, six picks       - context exists, the next move is a ban
    C  ten bans, six picks       - context exists, the next move is a pick,
                                   so the head has something valid to answer

`head` is only scored on the side on the clock: `move_plan` returns one
`(kind, team)` per move index, so the other team has no head input at all.
`head_offlane` is the honest counterfactual - the lanes that side is *not*
picking right now, which is the only place the head competes with the board.

Measured 2026-11-10 (400 games): the masked board is ~+3 pp top-5 over the
table once picks are on the board, ~-3 pp at the empty board, and the head
beats it only on the one lane it is actually picking (53.7 % vs ~40 %) while
falling below it on every other lane. See frontend/README.md, "Why the role
board keeps its own model".
"""
import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "propipeline", ROOT / "data" / "pro play" / "pipeline.py")
P = importlib.util.module_from_spec(spec)
sys.modules["propipeline"] = P
spec.loader.exec_module(P)

import server as S                                    # noqa: E402
from solo_queue.model.policy import lanes as L        # noqa: E402
from solo_queue.model.policy.infer import move_plan   # noqa: E402

ARMS = ("masked", "table", "head", "head_offlane", "masked_offlane",
        "random")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    rows, _ = P.build_rows()
    train, held = rows[:-n], rows[-n:]

    # the table's own ordering: how often each champion is picked in each lane.
    # Built on the training rows only - the held-out games must not vote.
    counts = [Counter() for _ in range(5)]
    for r in train:
        for kind, _team, champ, role in r["moves"]:
            if kind == 1 and 0 <= role < 5:
                counts[role][int(champ)] += 1
    table = [[c for c, _ in cnt.most_common(40)] for cnt in counts]

    mapping, _names, _champs = S.load_champ_data()
    inv = {idx: riot for riot, idx in mapping.items()}
    masked = S.load_predictor(S.MASKED_DEFAULT, device="cpu", role="masked")
    head = S.load_predictor(S.POLICY_DEFAULT, device="cpu", role="policy")
    print("masked: %s | head: %s"
          % (Path(S.MASKED_DEFAULT).parent.name,
             Path(S.POLICY_DEFAULT).parent.name))

    for label, n_bans, n_picks in (("A six bans, no picks", 6, 0),
                                   ("B six bans, six picks", 6, 6),
                                   ("C ten bans, six picks", 10, 6)):
        score(label, held, n_bans, n_picks, table, masked, head, mapping, inv)


def score(label, games, n_bans, n_picks, table, masked, head, mapping, inv):
    hit = {a: [0, 0] for a in ARMS}
    cells = {a: 0 for a in ARMS}
    ms_masked = ms_head = 0.0
    n_games = 0

    for r in games:
        bans, picks, per = [], [None] * 10, [0, 0]
        true = [[None] * 5, [None] * 5]
        for kind, team, champ, role in r["moves"]:
            if kind == 1:
                true[team][role] = int(champ)
            elif len(bans) < n_bans:
                bans.append(int(champ))
        placed = [m for m in r["moves"] if m[0] == 1][:n_picks]
        if len(placed) < n_picks or any(c is None for t in true for c in t):
            continue
        open_lane = [[True] * 5, [True] * 5]
        for _kind, team, champ, role in placed:
            picks[team * 5 + per[team]] = int(champ)
            per[team] += 1
            open_lane[team][role] = False

        fp = int(r["first_pick"])
        pid = head.patch_id_for(r.get("patch"))
        taken = set(bans) | {int(m[2]) for m in placed}

        t0 = time.time()
        res = masked.suggest([None if c is None else inv[c] for c in picks],
                             bans, mapping)
        ms_masked += time.time() - t0

        kind, on_clock = move_plan(fp)[n_bans + n_picks]
        now_lane = int(r["moves"][n_bans + n_picks][3]) if kind == 1 else None
        t0 = time.time()
        head_top = {}
        if kind == 1:
            for lane in range(5):
                sug = head.next_move(picks, bans, first_pick=fp, k=5,
                                     patch_id=pid,
                                     allowed=L.legal_champions(lane))
                head_top[lane] = [c["idx"] for c in sug["suggestions"]]
        ms_head += time.time() - t0
        n_games += 1

        for team in (0, 1):
            key = "blue" if team == 0 else "red"
            for lane in range(5):
                if not open_lane[team][lane]:
                    continue
                target = true[team][lane]
                if target in taken:
                    continue
                board = [c["idx"] for c in res[key][S.ROLE_ROWS[lane]]]
                tab = [c for c in table[lane] if c not in taken][:5]
                pool = [c for c in L.legal_champions(lane) if c not in taken]
                for arm, lst in (("masked", board), ("table", tab)):
                    cells[arm] += 1
                    hit[arm][0] += 1 if lst and lst[0] == target else 0
                    hit[arm][1] += 1 if target in lst else 0
                cells["random"] += 1
                hit["random"][1] += 5.0 / max(len(pool), 1)
                if team == on_clock and lane in head_top:
                    lst = head_top[lane]
                    cells["head"] += 1
                    hit["head"][0] += 1 if lst and lst[0] == target else 0
                    hit["head"][1] += 1 if target in lst else 0
                    if lane != now_lane:
                        cells["head_offlane"] += 1
                        hit["head_offlane"][0] += (1 if lst and lst[0] == target
                                                   else 0)
                        hit["head_offlane"][1] += 1 if target in lst else 0
                        cells["masked_offlane"] += 1
                        hit["masked_offlane"][0] += (1 if board and board[0] == target
                                                     else 0)
                        hit["masked_offlane"][1] += 1 if target in board else 0

    print("\n%s (%d games)" % (label, n_games))
    print("  %-14s %8s %8s %7s" % ("arm", "top-1", "top-5", "cells"))
    for arm in ARMS:
        c = max(cells[arm], 1)
        print("  %-14s %7.1f%% %7.1f%% %7d"
              % (arm, 100 * hit[arm][0] / c, 100 * hit[arm][1] / c, cells[arm]))
    print("  per board: masked %.0f ms | head x5 %.0f ms"
          % (1000 * ms_masked / max(n_games, 1),
             1000 * ms_head / max(n_games, 1)))


if __name__ == "__main__":
    main()
