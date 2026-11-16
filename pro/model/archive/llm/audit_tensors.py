"""Is the answer inside the tensors the encoder receives?

The per-move breakdown is unambiguous: moves 0-6 score 4-16 % exact, moves
7-19 score 99.5-100 %. A jump like that is an index boundary, not learnability.
And it survives `--context soft`, so the board *text* is not the source - the
soft tokens are, which means the encoder's input contains the answer.

This walks the real eval dataset and asks, per row, whether the target champion
is visible in any tensor the encoder reads.
"""
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import common as C
from pro.model.archive.llm import data as D


def main():
    rows = C.load_rows("2022")
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    ds = D.eval_dataset(rows, split["val"], patch_index, seed=0, val_games=200)

    print("keys in a sample:", sorted(ds[0].keys()))
    print()

    stats = Counter()
    by_move = {}
    for i in range(len(ds)):
        s = ds[i]
        t = int(s["target_champ"])
        mv = int(s["move_idx"])
        b = by_move.setdefault(mv, Counter())
        b["n"] += 1

        def note(key, cond):
            if cond:
                stats[key] += 1
                b[key] += 1

        n_pick = int(s["n_picks"]) if "n_picks" in s else None
        n_ban = int(s["n_bans"]) if "n_bans" in s else None

        # visible in the tensors that make up the "board so far"?
        for k in ("pick_ids", "ban_ids", "champ_ids", "blue_ids", "red_ids"):
            if k not in s:
                continue
            v = s[k]
            n = v.numel()
            # try both "all entries" and "first n_pick/n_ban entries"
            if bool((v == t).any()):
                note("%s:any" % k, True)
            cut = n_pick if k.startswith("pick") else n_ban
            if cut is not None and bool((v[:cut] == t).any()):
                note("%s:[:%s]" % (k, "n_pick" if k.startswith("pick")
                                   else "n_ban"), True)

        if "taken_mask" in s:
            note("taken_mask[t]", bool(s["taken_mask"][t]))
        if "unavailable_mask" in s:
            note("unavail_mask[t]", bool(s["unavailable_mask"][t]))
        note("target==pad?", t == 0)

    print("over %d eval states:" % len(ds))
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print("  %-24s %5d  (%.3f)" % (k, v, v / len(ds)))

    print("\nby move index (leak indicator per move):")
    hdr = [k for k, _ in sorted(stats.items(), key=lambda kv: -kv[1])][:5]
    print("  move   n   " + "  ".join("%-18s" % h for h in hdr))
    for mv in sorted(by_move):
        b = by_move[mv]
        print("  %4d %4d   " % (mv, b["n"])
              + "  ".join("%-18s" % ("%.3f" % (b[h] / b["n"])) for h in hdr))


if __name__ == "__main__":
    main()
