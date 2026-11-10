"""How much does the head's pick ranking depend on WHOSE turn it is?

    python frontend/probe_roleboard_side.py 40 10 6

The fabricated-turn shortlist (`probe_roleboard_fab.py`) asks the head "suppose
<side> were picking next". This measures whether that question changes the
answer, by comparing the two sides on an otherwise identical board - and
calibrates the result against a control that changes the board content.

Reads:
  tv_side   0.5*sum|p_blue - p_red| over the lane's legal champions
  tv_ctrl   the same, after replacing one enemy pick with another legal champ
  llm_diff  fraction of shared lanes where the reranker's winner differs

Measured 2026-11-10 (52 shared cells): tv_side 0.015 against tv_ctrl 0.147,
90 % identical shortlists, reranker winner differs in 4 %. `to_move` enters the
model only as `side_embed(to_move)`, and the pick counts already determine
whose turn it is, so the flag was redundant in training and carries almost no
weight. See frontend/README.md, "The fabricated turn, measured".
"""
import importlib.util
import sys
import torch

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "propipeline", ROOT / "data" / "pro play" / "pipeline.py")
P = importlib.util.module_from_spec(spec)
sys.modules["propipeline"] = P
spec.loader.exec_module(P)

import rerank as RR                                # noqa: E402
from pro.model.llm import board as BOARD           # noqa: E402
from pro.model.llm import common as C              # noqa: E402
from pro.model.llm import data as D                # noqa: E402
from pro.model.llm.prompt import assemble          # noqa: E402
from solo_queue.model.policy import infer as I     # noqa: E402
from solo_queue.model.policy import lanes as L     # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_BANS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_PICKS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

pred = RR.RerankPredictor(ROOT / "pro/model/checkpoints/llm-lora-rerank5",
                          device="cuda")
model, tok, names = pred.model, pred.tok, pred.names
head = pred.head
from pro.model.llm.common import ANSWER_PREFIX      # noqa: E402


def first_tok(name):
    return tok(ANSWER_PREFIX + name, add_special_tokens=False)["input_ids"][0]


def raw_logits(picks, bans, first_pick, team):
    """Unmasked policy logits for `team` on this board."""
    board = I.build_board(picks, bans, head.num_champs, first_pick, head.device)
    head._fill_roles(board)
    n_picks = int((~board["pad_mask"][0]).sum())
    move_idx = int(board["n_bans"][0]) + n_picks
    batch = I._policy_batch(board, move_idx, I.MOVE_PICK, team,
                            patch_id=head.default_patch_id())
    return head.policy_logits(batch)[0].float()


def lane_probs(logits, lane):
    keep = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
    keep[torch.as_tensor(L.legal_champions(lane), device=logits.device)] = True
    p = torch.softmax(logits.masked_fill(~keep, float("-inf")), -1)
    return p / p.sum()


def tv(p, q):
    return 0.5 * float((p - q).abs().sum())


def top5(p):
    return set(int(i) for i in p.topk(5).indices)


@torch.no_grad()
def rerank(picks, lanes, bans, first_pick, move_idx, patch, team, lane, cands):
    samples = [BOARD.sample_from_board(
        picks, lanes, bans, first_pick, move_idx, patch, I.MOVE_PICK, team,
        target_role=lane, target_champ=c, patch_index=pred.patch_index)
        for c, _p in cands]
    enc, _ = D.collate(samples)
    enc = D.to_device(enc, pred.device)
    prompt = assemble(tok, samples, names, context=pred.context,
                      with_answers=True, prompt_format="v4",
                      candidates=[cands] * len(cands))
    prompt = {k: (v.to(pred.device) if torch.is_tensor(v) else v)
              for k, v in prompt.items()}
    ft = torch.tensor([[first_tok(names[c])] for c, _p in cands],
                      device=pred.device)
    keep = RR.candidate_mask(ft, model.cand_mask.numel(), device=pred.device)
    sc = model.answer_scores(enc, prompt, keep=keep)
    fl, tl = sc["first_logp"].tolist(), sc["total_logp"].tolist()
    win = max(range(len(cands)), key=lambda i: (fl[i], tl[i]))
    return cands[win][0]


rows, _ = P.build_rows()
acc = {"tv_side": [], "tv_ctrl": [], "overlap": [], "shape_side": [],
       "shape_ctrl": [], "tv_det": []}
n_shared = n_llm_diff = n_llm = 0
shown = 0

for r in rows[-N:]:
    bans, picks, per = [], [None] * 10, [0, 0]
    true = [[None] * 5, [None] * 5]
    for kind, team, champ, role in r["moves"]:
        if kind == I.MOVE_PICK:
            true[team][role] = int(champ)
        elif len(bans) < N_BANS:
            bans.append(int(champ))
    placed = [m for m in r["moves"] if m[0] == I.MOVE_PICK][:N_PICKS]
    if len(placed) < N_PICKS or any(c is None for t in true for c in t):
        continue
    for _k, team, champ, role in placed:
        picks[team * 5 + per[team]] = int(champ)
        per[team] += 1
    lanes = L.assign_lanes(picks)
    move_idx = len(bans) + len(placed)
    taken = set(bans) | {int(m[2]) for m in placed}
    fp = int(r["first_pick"])

    # control board: replace one enemy pick with another champ legal in the
    # lane that pick actually occupies. Same shape, different content - the
    # calibration for "a real change moves the distribution this much".
    v_team = int(placed[-1][1])
    v_slot = v_team * 5 + per[v_team] - 1
    v_champ, v_lane = picks[v_slot], lanes[v_slot]
    alt = None
    if v_lane is not None:
        for c in L.legal_champions(v_lane):
            if c not in taken and c != v_champ:
                alt = c
                break
    ctrl = list(picks)
    if alt is not None:
        ctrl[v_slot] = alt

    # The head's logits do not depend on the lane - only the mask does. Two
    # forwards per game instead of two per (lane, team) cell.
    lg = [raw_logits(picks, bans, fp, 0), raw_logits(picks, bans, fp, 1)]
    lg_det = raw_logits(picks, bans, fp, 0)
    lgc = raw_logits(ctrl, bans, fp, 0) if alt is not None else None

    for team in (0, 1):
        for lane in BOARD.remaining_lanes(lanes, team):
            target = true[team][lane]
            if target in taken:
                continue
            p = {t: lane_probs(lg[t], lane) for t in (0, 1)}

            if team == 1 and lane in BOARD.remaining_lanes(lanes, 0):
                n_shared += 1
                acc["tv_side"].append(tv(p[0], p[1]))
                acc["overlap"].append(len(top5(p[0]) & top5(p[1])) / 5.0)
                acc["tv_det"].append(
                    tv(p[0], lane_probs(lg_det, lane)))
                d = lg[0] - lg[1]
                acc["shape_side"].append((float(d.std()), float(d.abs().max())))

                cands = [(c, float(p[0][c])) for c in
                         p[0].topk(5).indices.tolist()]
                w0 = rerank(picks, lanes, bans, fp, move_idx, r.get("patch"),
                            0, lane, cands)
                w1 = rerank(picks, lanes, bans, fp, move_idx, r.get("patch"),
                            1, lane, cands)
                n_llm += 1
                n_llm_diff += 1 if w0 != w1 else 0

                if lgc is not None:
                    pc = lane_probs(lgc, lane)
                    acc["tv_ctrl"].append(tv(p[0], pc))
                    dc = lgc - lg[0]
                    acc["shape_ctrl"].append(
                        (float(dc.std()), float(dc.abs().max())))

                if shown < 6:
                    o = len(top5(p[0]) & top5(p[1]))
                    print("  %-8s tv %.4f  overlap %d/5  |  Blue %s  Red %s"
                          % (C.ROLE_DISPLAY[lane], tv(p[0], p[1]), o,
                             names[w0], names[w1]))
                    shown += 1


def mean(x):
    return sum(x) / max(len(x), 1)


print("\nstate = %d bans + %d picks | %d shared (lane, board) cells"
      % (N_BANS, N_PICKS, n_shared))
print("side effect on the lane distribution: TV %.4f (determinism %.2e)"
      % (mean(acc["tv_side"]), mean(acc["tv_det"])))
print("  control, one enemy pick swapped:    TV %.4f  (%d cells)"
      % (mean(acc["tv_ctrl"]), len(acc["tv_ctrl"])))
print("top-5 overlap between the two sides: %.2f of 5 (%.0f %% identical)"
      % (mean(acc["overlap"]), 100 * mean([1.0 if o == 1.0 else 0.0
                                           for o in acc["overlap"]])))
for name, key in (("side", "shape_side"), ("control", "shape_ctrl")):
    if acc[key]:
        print("logit delta, %-8s sd %.2f | max |d| %.2f"
              % (name, mean([s for s, _m in acc[key]]),
                 mean([m for _s, m in acc[key]])))
print("LLM winner differs between the sides in %d of %d shared lanes (%.0f %%)"
      % (n_llm_diff, n_llm, 100.0 * n_llm_diff / max(n_llm, 1)))
