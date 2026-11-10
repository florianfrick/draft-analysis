"""Which side's draft does the model like more?

    python frontend/draftscore.py            # replay 20 held-out boards

There is no win-probability head in the frontend: the value path was measured
at the base rate and turned off (`solo_queue/model/policy/model.py`), and the
win-prediction experiments never left their branch. So the honest question this
answers is the one the loaded model can actually answer: **replay the board
through the policy head and ask how closely each side drafted to the model's
own ranking at the moment it picked.**

For every pick, the board as it stood before that pick is rebuilt from the
draft order (`move_plan`), the head is asked whose turn it is (its real turn,
so nothing is fabricated), and the pick is scored by

  * `rank`  - where the champion sat in the head's legal ranking (1 = top),
  * `prob`  - the head's probability for it,
  * `share` - `prob / prob(best legal)`, which is scale-free, so a pick made
              when 160 champions were legal is comparable with one made when
              150 were.

`share` is the comparison: a side that repeatedly takes the model's first
choice scores near 1.0. It measures *agreement with the model*, not strength -
a draft the model dislikes can win, and the panel says so.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solo_queue.model.policy import infer                    # noqa: E402

SIDE = {0: "blue", 1: "red"}


def chronological_bans(blue_bans, red_bans, first_pick, n=None):
    """Side-grouped ban lists -> the order `build_board` expects.

    The UI keeps `bans.blue` and `bans.red` separately and sends them
    concatenated (`allBansFlat`), so a flat list is blue's bans then red's,
    each in the order they were made. `build_board` stores `ban_ids`
    chronologically and derives each ban's *side* from its slot
    (`BAN_SIDE_BY_SLOT`), so feeding it the grouped list tells the model that
    blue made the first three bans of an alternating block - the sides end up
    shifted. This is a pre-existing mismatch on the next-move path; it is
    fixed here because a replay compares moves, so a shifted ban block would
    change the answer mid-board.

    The mapping is exact: a side's list is its own bans in order, so walking
    the slots and popping from the matching side rebuilds the interleaving.
    `n` truncates to the first n bans, which is what a replay at move m needs.
    """
    from solo_queue.model.policy.model import BAN_SIDE_BY_SLOT

    by_side = {0: list(blue_bans), 1: list(red_bans)}
    out = []
    for slot in range(10):
        if n is not None and len(out) >= n:
            break
        side = BAN_SIDE_BY_SLOT[slot] ^ int(first_pick)
        if not by_side[side]:
            break
        out.append(by_side[side].pop(0))
    return out


def replay(picks, first_pick, plan=None):
    """-> [(move_idx, team, slot)] for the picks on this board, in order.

    The UI board is two blocks of five in *pick order*: blue's slots 0-4 are
    blue's first to fifth pick. The global order is the draft order, so it is
    reconstructed from `move_plan` rather than from the slots - which is also
    why a half-filled board (one side has picked three times, the other twice)
    comes out right.
    """
    plan = plan if plan is not None else infer.move_plan(int(first_pick))
    used = {0: 0, 1: 0}
    out = []
    for move_idx, (kind, team) in enumerate(plan):
        if kind != infer.MOVE_PICK:
            continue
        slot = team * infer.TEAM_SIZE + used[team]
        if slot >= len(picks) or picks[slot] is None:
            continue
        used[team] += 1
        out.append((move_idx, team, slot))
    return out


def _n_bans_before(plan, move_idx):
    return sum(1 for kind, _t in plan[:move_idx] if kind == infer.MOVE_BAN)


def score(pred, picks, blue_bans, red_bans, first_pick, patch_id=None,
          unavailable=None, names=None):
    """Replay the board -> per-side draft scores and the per-pick rows.

    `pred` is a `PolicyPredictor` (the head, not the reranker: this asks the
    model that has a distribution, and it asks it 10 times). `picks` is the
    UI's 10-slot board of champion indices (None for empty), `blue_bans` /
    `red_bans` the two side-grouped ban lists, `names` a name table for the
    display fields.
    """
    plan = infer.move_plan(int(first_pick))
    rows = replay(picks, first_pick, plan=plan)
    if not rows:
        raise ValueError("no picks on the board yet - nothing to compare")
    bans_all = chronological_bans(blue_bans, red_bans, first_pick)

    def nm(i):
        return names[i] if names and 0 <= i < len(names) else str(i)

    pid = pred.patch_id_for(patch_id)
    out_rows = []
    placed = [None] * len(picks)
    for move_idx, team, slot in rows:
        n_bans = _n_bans_before(plan, move_idx)
        bans_now = bans_all[:n_bans]
        res = pred.next_move(placed, bans_now, first_pick=first_pick,
                             k=pred.num_champs, patch_id=pid,
                             unavailable=unavailable)
        if res.get("complete"):
            break
        chosen = int(picks[slot])
        probs = {c["idx"]: c["prob"] for c in res["suggestions"]}
        order = [c["idx"] for c in res["suggestions"]]
        p = probs.get(chosen, 0.0)
        p_best = res["suggestions"][0]["prob"] if res["suggestions"] else 0.0
        out_rows.append({
            "move_idx": int(move_idx),
            "side": SIDE[team],
            "slot": int(slot),
            "champ": chosen,
            "name": nm(chosen),
            "rank": (order.index(chosen) + 1) if chosen in order else None,
            "prob": float(p),
            "best": nm(order[0]) if order else None,
            "best_prob": float(p_best),
            # 1.0 = the model's own first choice; ~0 = it did not rank it
            "share": float(p / p_best) if p_best > 0 else 0.0,
        })
        placed[slot] = chosen

    sides = {}
    for team in (0, 1):
        mine = [r for r in out_rows if r["side"] == SIDE[team]]
        if not mine:
            sides[SIDE[team]] = None
            continue
        sides[SIDE[team]] = {
            "n": len(mine),
            "mean_share": sum(r["share"] for r in mine) / len(mine),
            "mean_rank": (sum(r["rank"] for r in mine if r["rank"])
                          / max(1, len([r for r in mine if r["rank"]]))),
            "top1": sum(1 for r in mine if r["rank"] == 1),
            "worst_rank": max((r["rank"] or 999) for r in mine),
        }
    blue, red = sides.get("blue"), sides.get("red")
    verdict, margin = None, None
    if blue and red:
        margin = blue["mean_share"] - red["mean_share"]
        # A 0.02 gap on a 5-pick sample is not a verdict; the same threshold
        # the explain panel uses for "the list is flat".
        verdict = ("blue" if margin > 0.02 else
                   "red" if margin < -0.02 else "even")
    elif blue or red:
        verdict = "blue" if blue else "red"

    return {"verdict": verdict,
            "margin": None if margin is None else float(margin),
            "blue": blue, "red": red,
            "picks": out_rows,
            "n_picks": len(out_rows),
            "note": ("How closely each side drafted to this model's own "
                     "ranking at the moment it picked - not a win "
                     "probability.")}


# ------------------------------------------------------------------ cli
def main():
    """Replay the last N held-out boards and print the two sides' scores."""
    import importlib.util
    import server as S

    spec = importlib.util.spec_from_file_location(
        "propipeline", ROOT / "data" / "pro play" / "pipeline.py")
    P = importlib.util.module_from_spec(spec)
    sys.modules["propipeline"] = P
    spec.loader.exec_module(P)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rows, _ = P.build_rows()
    mapping, names, _champs = S.load_champ_data()
    pred = S.load_predictor(S.POLICY_DEFAULT, device="cuda", role="policy")
    head = getattr(pred, "head", pred)
    print("head: %s" % Path(S.POLICY_DEFAULT).parent.name)

    agreed = 0
    total = 0
    for r in rows[-n:]:
        bb, rb, picks, per = [], [], [None] * 10, [0, 0]
        for kind, team, champ, role in r["moves"]:
            if kind == infer.MOVE_PICK:
                picks[team * 5 + per[team]] = int(champ)
                per[team] += 1
            elif len(bb) + len(rb) < 10:
                (bb if team == 0 else rb).append(int(champ))
        if any(c is None for c in picks):
            continue
        res = score(head, picks, bb, rb, int(r["first_pick"]),
                    patch_id=head.patch_id_for(r.get("patch")),
                    names=names)
        total += 1
        won = r.get("blue_win")
        won = None if won is None else int(won)
        if res["verdict"] in ("blue", "red") and won is not None:
            agreed += 1 if res["verdict"] == ("blue" if won else "red") else 0
        print("  %-10s blue %.2f (rank %.1f) | red %.2f (rank %.1f) -> %-4s"
              " %s"
              % (str(r.get("gameid", ""))[:10], res["blue"]["mean_share"],
                 res["blue"]["mean_rank"], res["red"]["mean_share"],
                 res["red"]["mean_rank"], res["verdict"],
                 "" if won is None else ("blue won" if won else "red won")))
    if total:
        print("\n%d boards; the higher-scoring side won %d of them"
              % (total, agreed))


if __name__ == "__main__":
    main()
