"""Print the exact user turn for rows that have a fearless pool.

Answers one question with evidence rather than code-reading: does the prompt
carry the fearless bans? `render_state(include_fearless=True)` is the default,
and this prints it for a real val row so the wording and its position are
visible, plus how often the line is present at all.

Two passes on purpose: the count must cover every row, so it cannot be taken
from a loop that stops as soon as it has an example of each kind.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.llm import common as C          # noqa: E402
from pro.model.llm import data as D            # noqa: E402
from pro.model.llm.prompt import build_text    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-format", default=C.PROMPT_FORMAT,
                    choices=["v1", "v2"])
    args = ap.parse_args(argv)
    rows = C.load_rows("2022")
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    ds = D.eval_dataset(rows, split["val"], patch_index, seed=0, val_games=200)

    n = len(ds)
    pooled = 0
    by_action = {0: [0, 0], 1: [0, 0]}       # [with pool, total]
    examples = {}
    for i in range(n):
        s = ds[i]
        has = bool(s["unavailable_mask"].any())
        pooled += int(has)
        b = int(bool(int(s["is_ban"])))
        by_action[b][1] += 1
        by_action[b][0] += int(has)
        if has and b not in examples:
            examples[b] = s

    for b in (0, 1):
        if b not in examples:
            continue
        s = examples[b]
        prefix, body, ans = build_text(s, names,
                                       answer=names[int(s["target_champ"])],
                                       prompt_format=args.prompt_format)
        print("\n=== %s, move %d/%d, pool of %d ==="
              % ("BAN" if b else "PICK", int(s["move_idx"]) + 1, C.N_MOVES,
                 int(s["unavailable_mask"].sum())))
        print(body)
        print("Answer: " + ans)

    print("\n%d of %d val rows (%.1f%%) have a non-empty fearless pool"
          % (pooled, n, 100.0 * pooled / max(n, 1)))
    for b, label in ((0, "pick"), (1, "ban")):
        with_pool, tot = by_action[b]
        print("  %-4s %d/%d (%.1f%%)" % (label, with_pool, tot,
                                         100.0 * with_pool / max(tot, 1)))
    print("prompt format: %s (render_state default include_fearless: %s)"
          % (args.prompt_format, C.render_state.__defaults__[0]))


if __name__ == "__main__":
    main()
