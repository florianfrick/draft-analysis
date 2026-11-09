"""Is the LLM's val set actually held out *from the policy head*?

    python -m pro.model.llm._split_overlap

The two tracks split the corpus with the same function but different flags:
the head trains on `split_within_patch(val_frac=.1, test_frac=.1,
val_random=False)` (train = the *head* of each patch), the LLM on
`split_within_patch(val_frac=.1, test_frac=0, val_random=True)` (train = a
random 90 % of each patch). Both are legitimate, but they are not the same
partition, so the LLM's val rows are not automatically held out from the head.

That matters for one specific claim: "the LLM's 19.13 % clears the head's
18.32 %". If most of the LLM's val rows were in the head's train, scoring the
head there would measure memorisation, and the two numbers would not be
comparable at all. This prints the overlap, and the intersection that is
honest for both.
"""
import sys

from pro.model.llm import common as C
from pro.model.proonly import common as PC


def game_ids(rows, indices):
    return {rows[i]["gameid"] for i in indices}


def main():
    rows = C.load_rows("2022")
    llm = C.split_rows(rows)

    # The head's split, with the flags its checkpoints were trained with
    # (`pro/model/bans/probe.py` defaults: within-patch, val=tail, test=tail).
    head = PC.PRO_PIPE.split_within_patch(
        rows, val_frac=0.10, test_frac=0.10, min_patch_rows=40,
        pinned_test=PC.PRO_PIPE.TEST_SERIES, strict_pin=True, verbose=False,
        val_random=False)

    g_llm = {"train": game_ids(rows, llm["train"]),
             "val": game_ids(rows, llm["val"]),
             "test": game_ids(rows, llm["test"])}
    g_head = {"train": game_ids(rows, head["train"]),
              "val": game_ids(rows, head["val"]),
              "test": game_ids(rows, head["test"])}

    print("games: llm %d/%d/%d  head %d/%d/%d"
          % (len(g_llm["train"]), len(g_llm["val"]), len(g_llm["test"]),
             len(g_head["train"]), len(g_head["val"]), len(g_head["test"])))
    print()
    print("LLM val games by where the head put them:")
    for part in ("train", "val", "test"):
        n = len(g_llm["val"] & g_head[part])
        print("  head %-5s %6d  %5.1f %%"
              % (part, n, 100.0 * n / max(1, len(g_llm["val"]))))
    print()
    common = g_llm["val"] & (g_head["val"] | g_head["test"])
    print("common held-out (LLM val AND head val/test): %d games (%.1f %% of "
          "the LLM's val)" % (len(common),
                              100.0 * len(common) / max(1, len(g_llm["val"]))))
    print("LLM train games the head also trained on: %d (%.1f %% of LLM train)"
          % (len(g_llm["train"] & g_head["train"]),
             100.0 * len(g_llm["train"] & g_head["train"])
             / max(1, len(g_llm["train"]))))
    if common:
        print()
        print("=> score BOTH models on the %d common games, not on either "
              "track's own val" % len(common))
    return 0


if __name__ == "__main__":
    sys.exit(main())
