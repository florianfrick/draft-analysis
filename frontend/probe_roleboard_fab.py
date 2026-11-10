"""Does the fabricated turn replace the role board? (four arms, same cells)

    python frontend/probe_roleboard_fab.py 200 10 6
    python frontend/probe_roleboard_fab.py 100 6 6 full   # ban-phase board

The question: the lane selector already masks the head's pool for the *next
move*; can the same be done for the role board - one lane at a time, with the
turn fabricated - and let the reranker choose? `_policy_batch` takes the team
as an argument, so the off-turn side can be posed as a hypothetical.

Arms, all on identical (game, side, lane) cells so nothing is compared across
two probes (`data.eval_dataset` draws a fixed random subsample, so two probes
with different N do not share a sample):

  masked   the shipped role board (`/api/suggest`, a masking checkpoint)
  table    per-lane popularity, built on the training rows only
  fab_head the head asked "suppose <side> were picking next, for <lane>",
           logits masked to that lane's legal champions
  fab_llm  the same five, reranked by `llm-lora-rerank5` with prompt v4
  random   expectation, 5 draws from the lane's legal pool

`full` poses the hypothetical at the pick move the board will reach
(`move_idx = 10 + n_picks`) instead of the move it is on, which is the only
coherent way to ask a ban-phase board. Measured 2026-11-10: 15.6 % top-1 /
49.5 % top-5 against the masked board's 10.5 / 44.2 with all bans in (774
cells, p < 0.005), and a tie at 6 bans + 6 picks. See frontend/README.md,
"The fabricated turn, measured".
"""
import importlib.util
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "propipeline", ROOT / "data" / "pro play" / "pipeline.py")
P = importlib.util.module_from_spec(spec)
sys.modules["propipeline"] = P
spec.loader.exec_module(P)

import rerank as RR                                # noqa: E402
import server as S                                 # noqa: E402
from pro.model.llm import board as BOARD           # noqa: E402
from pro.model.llm import common as C              # noqa: E402
from pro.model.llm import data as D                # noqa: E402
from pro.model.llm.prompt import assemble          # noqa: E402
from solo_queue.model.policy import infer as I     # noqa: E402
from solo_queue.model.policy import lanes as L     # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
N_BANS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_PICKS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
# At a ban-phase board the next move is a ban, so `move_idx` and `is_ban`
# disagree unless the hypothetical is posed at the pick move the board will
# actually reach: "assume the remaining bans are in". Off by default, so the
# 10-bans numbers stay the ones this script was first measured with.
ASSUME_FULL_BANS = len(sys.argv) > 4 and sys.argv[4] == "full"

pred = RR.RerankPredictor(ROOT / "pro/model/checkpoints/llm-lora-rerank5",
                          device="cuda")
model, tok, names = pred.model, pred.tok, pred.names
head = pred.head
from pro.model.llm.common import ANSWER_PREFIX      # noqa: E402

ARMS = ("masked", "table", "fab_head", "fab_llm", "random")


def first_tok(name):
    return tok(ANSWER_PREFIX + name, add_special_tokens=False)["input_ids"][0]


def head_shortlist(picks, bans, first_pick, team, lane, k=5):
    """The head's top-k for `team` on `lane`, with the turn set to `team`.

    `next_move` takes the side from move_plan(), so the only way to ask about
    the side that is *not* on the clock is to build the batch with that team -
    which is exactly the hypothetical the role board poses.
    """
    board = I.build_board(picks, bans, head.num_champs, first_pick, head.device)
    head._fill_roles(board)
    n_picks = int((~board["pad_mask"][0]).sum())
    n_bans = 10 if ASSUME_FULL_BANS else int(board["n_bans"][0])
    move_idx = n_bans + n_picks
    batch = I._policy_batch(board, move_idx, I.MOVE_PICK, team,
                            patch_id=head.default_patch_id())
    logits = head.policy_logits(batch)[0]
    keep = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
    keep[torch.as_tensor(L.legal_champions(lane), device=logits.device)] = True
    p = torch.softmax(logits.masked_fill(~keep, float("-inf")).float(), -1)
    top = p.topk(k)
    return [(int(i), float(v)) for i, v in zip(top.indices, top.values)]


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
train, held = rows[:-N], rows[-N:]

counts = [Counter() for _ in range(5)]
for r in train:
    for kind, _team, champ, role in r["moves"]:
        if kind == I.MOVE_PICK and 0 <= role < 5:
            counts[role][int(champ)] += 1
table = [[c for c, _ in cnt.most_common(40)] for cnt in counts]

mapping, _n, _c = S.load_champ_data()
inv = {idx: riot for riot, idx in mapping.items()}
masked = S.load_predictor(S.MASKED_DEFAULT, device="cpu", role="masked")
print("masked: %s | head: %s | reranker: llm-lora-rerank5"
      % (Path(S.MASKED_DEFAULT).parent.name,
         Path(S.POLICY_DEFAULT).parent.name))

hit = {a: [0, 0] for a in ARMS}
cells = {a: 0 for a in ARMS}
t_head = t_llm = t_masked = 0.0
same = shared = 0
n_games = 0
shown = 0
# McNemar: cells where one arm is right and the other wrong. The marginals
# above are the same cells, but a 1 pp gap on a 13 % rate is inside the noise,
# and only the paired counts say so.
pairs = {("masked", "fab_llm"): [0, 0], ("masked", "fab_head"): [0, 0]}

for r in held:
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
    n_games += 1

    t0 = time.time()
    res = masked.suggest([None if c is None else inv[c] for c in picks],
                         bans, mapping)
    t_masked += time.time() - t0

    by_lane = {0: {}, 1: {}}
    for team in (0, 1):
        key = "blue" if team == 0 else "red"
        for lane in BOARD.remaining_lanes(lanes, team):
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

            t0 = time.time()
            cands = head_shortlist(picks, bans, fp, team, lane)
            t_head += time.time() - t0
            ids = [c for c, _p in cands]
            t0 = time.time()
            win = rerank(picks, lanes, bans, fp, move_idx, r.get("patch"),
                         team, lane, cands)
            t_llm += time.time() - t0

            by_lane[team][lane] = ids
            if team == 1 and lane in by_lane[0]:
                shared += 1
                same += 1 if ids == by_lane[0][lane] else 0

            cells["fab_head"] += 1
            hit["fab_head"][0] += 1 if ids[0] == target else 0
            hit["fab_head"][1] += 1 if target in ids else 0
            cells["fab_llm"] += 1
            hit["fab_llm"][0] += 1 if win == target else 0
            hit["fab_llm"][1] += 1 if win in ids else 0

            m_ok = bool(board) and board[0] == target
            for other, ok in (("fab_llm", win == target),
                              ("fab_head", ids[0] == target)):
                p = pairs[("masked", other)]
                p[0] += 1 if (m_ok and not ok) else 0
                p[1] += 1 if (ok and not m_ok) else 0

            if shown < 4:
                print("  %s %-8s masked %-11s table %-11s fab %-11s -> %-11s "
                      "(true %s)"
                      % ("Blue" if team == 0 else "Red", C.ROLE_DISPLAY[lane],
                         names[board[0]] if board else "-",
                         names[tab[0]] if tab else "-",
                         names[ids[0]], names[win], names[target]))
                shown += 1

print("\n%d games, %d cells, state = %d bans + %d picks"
      % (n_games, cells["masked"], N_BANS, N_PICKS))
print("  %-10s %7s %7s" % ("arm", "top-1", "top-5"))
for arm in ARMS:
    c = max(cells[arm], 1)
    # fab_llm reorders the head's five, it cannot add one: its recall@5 is
    # fab_head's by construction, so printing it would fake a 100 %.
    five = "-" if arm == "fab_llm" else "%5.1f%%" % (100 * hit[arm][1] / c)
    print("  %-10s %6.1f%% %7s"
          % (arm, 100 * hit[arm][0] / c, five))
print("identical shortlist for both sides in %d of %d shared lanes"
      % (same, shared))
for (a, b), (ab, ba) in pairs.items():
    n = ab + ba
    k = min(ab, ba)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) \
        if n else 1.0
    print("%s right / %s wrong %d, the reverse %d -> exact p %.2f"
          % (a, b, ab, ba, p))
print("per board: masked %.0f ms | head x%d %.0f ms | LLM x%d %.0f ms"
      % (1000 * t_masked / max(n_games, 1), cells["fab_head"],
         1000 * t_head / max(cells["fab_head"], 1), cells["fab_llm"],
         1000 * t_llm / max(cells["fab_llm"], 1)))
