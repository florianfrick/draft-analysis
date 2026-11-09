"""Leakage audit for the warmup result.

69.25 % exact-match against a policy head that gets 18.32 % top-1 is not a
number to report until two things are ruled out:

  1. train/val game overlap - a game in both splits is memorisation, and
     6,000 training games would memorise 200 eval games easily.
  2. the target champion appearing in the rendered board - a one-move
     off-by-one in the prefix tensors would make the task "copy the name".

Both are checked against the real corpus, not a fixture.

    python -m pro.model.llm.audit_leak
"""
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.llm import common as C
from pro.model.llm import data as D
from pro.model.llm.common import N_MOVES, SIDE_NAMES, render_state


def game_key(row):
    """Whatever identifies a game; series+game is what the corpus carries."""
    return (str(row.get("league")), str(row.get("series")),
            str(row.get("game")))


def main():
    rows = C.load_rows("2022")
    split = C.split_rows(rows)
    names = C.champ_names()
    keys, patch_index = D.make_patch_index(rows)
    print("rows %d | train %d val %d test %d"
          % (len(rows), len(split["train"]), len(split["val"]),
             len(split["test"])))

    # ---- 1. split overlap, by index and by game identity
    tr, va = set(split["train"]), set(split["val"])
    print("\n[1] index overlap train/val: %d" % len(tr & va))
    trk = Counter(game_key(rows[i]) for i in split["train"])
    vak = Counter(game_key(rows[i]) for i in split["val"])
    shared = set(trk) & set(vak)
    print("    distinct game keys: train %d val %d shared %d"
          % (len(trk), len(vak), len(shared)))
    if shared:
        ex = list(shared)[:3]
        print("    e.g. %s" % (ex,))

    # ---- 2. does the target champion appear in the board it is asked for?
    va_ds = D.eval_dataset(rows, split["val"], patch_index, seed=0,
                           val_games=200)
    print("\n[2] eval states: %d (%d games x %d moves)"
          % (len(va_ds), len(va_ds) // N_MOVES, N_MOVES))

    hits_board = hits_fearless = 0
    by_ban = Counter()
    examples = []
    for i in range(len(va_ds)):
        s = va_ds[i]
        target = names[int(s["target_champ"])]
        state = render_state(s, names)
        lines = state.split("\n")
        board = "\n".join(ln for ln in lines if not ln.startswith("Fearless"))
        fear = "\n".join(ln for ln in lines if ln.startswith("Fearless"))
        in_board = target in board
        in_fear = target in fear
        hits_board += in_board
        hits_fearless += in_fear
        if in_board:
            by_ban[bool(int(s["is_ban"]))] += 1
            if len(examples) < 3:
                examples.append((i, int(s["move_idx"]), target, state))

    n = len(va_ds)
    print("    target appears in Blue/Red/Bans lines: %d / %d = %.4f"
          % (hits_board, n, hits_board / n))
    print("      of those, ban moves %d / pick moves %d"
          % (by_ban[True], by_ban[False]))
    print("    target appears in the Fearless line:   %d / %d = %.4f"
          % (hits_fearless, n, hits_fearless / n))
    for i, mv, tgt, state in examples:
        print("\n    --- leaking example idx %d move %d target %r ---"
              % (i, mv, tgt))
        for ln in state.split("\n"):
            print("      " + ln)

    # ---- 3. how much of the eval set is even in the training subsample
    print("\n[3] the 6,000-game training subsample:")
    g = torch.Generator().manual_seed(0)
    pick = torch.randperm(len(split["train"]), generator=g)[:6000]
    sub = {split["train"][i] for i in pick.tolist()}
    print("    index overlap with the 4,743-game val split: %d"
          % len(sub & va))
    subk = Counter(game_key(rows[i]) for i in sub)
    print("    game-key overlap with val: %d" % len(set(subk) & set(vak)))

    # ---- 4. sanity: what a constant answer would score
    print("\n[4] reference points")
    print("    policy head top-1 on the 95,160-state val set: %.4f"
          % C.BASELINE_VAL["top1"])
    print("    distinct first tokens for %d names: %d"
          % (len(names), len(C.candidate_tables(
              __import__("transformers").AutoTokenizer.from_pretrained(
                  "Qwen/Qwen3.5-4B"), names)[0])))


if __name__ == "__main__":
    main()
