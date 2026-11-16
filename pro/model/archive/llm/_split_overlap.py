"""Do the two tracks actually hold the same games out?

    python -m pro.model.archive.llm._split_overlap

The project believed for months that the head and the LLM split the corpus
differently - that the head trained on the *head* of every patch
(`val_random=False`) while the LLM trained on a random 90 %
(`val_random=True`), so that 89.9 % of the LLM's val was inside the head's
train. That number gated every cross-model claim: it shrank the honest
comparison set to a 478-game intersection, it made the reranker train on a
4,303-game pool, and it is quoted in `rerank.py` and in STATUS.

It was wrong. The shipped head trained with `--val-split random --no-test`
(`pro/model/archive/bans/probe.py` defaults; its checkpoint records `test: skipped
(--no-test)` and `init_ckpt = masking-proonly-randomvalsplit-*`), which is the
same `val_frac=0.10, val_random=True` the LLM uses. This script prints both
readings side by side so the correction is auditable rather than asserted.

What to expect (46,915-draft corpus, game granularity):

    head, real flags      train 42,153  val 4,762  test 0
    llm, its SPLIT        train 42,145  val 4,763  test 7
    LLM val inside the real head's train      14 / 4,763 = 0.3 %
    LLM val inside the *assumed* head's train 4,304 / 4,763 = 90.4 %

The 0.3 % is the pinned test series' patch, not leakage: the head run passed
`--no-test`, so its val draw had 7 more games to choose from than the LLM's.
A retrain that keeps `pinned_test=TEST_SERIES` on both sides makes the two val
sets identical, which `test_rerank.py` now pins.
"""
import sys

from pro.model.archive.llm import common as C
from pro.model.proonly import common as PC


def game_ids(rows, indices):
    return {rows[i]["gameid"] for i in indices}


def main():
    rows = C.load_rows("2022")
    llm = C.split_rows(rows)

    # The head's real flags: random val, no test tail.
    real = PC.PRO_PIPE.split_within_patch(
        rows, val_frac=0.10, test_frac=0.0, min_patch_rows=40,
        pinned_test=(), strict_pin=True, verbose=False, val_random=True)
    # The flags this file used to assume, kept so the old number is
    # reproducible and can be retired explicitly rather than forgotten.
    assumed = PC.PRO_PIPE.split_within_patch(
        rows, val_frac=0.10, test_frac=0.10, min_patch_rows=40,
        pinned_test=PC.PRO_PIPE.TEST_SERIES, strict_pin=True, verbose=False,
        val_random=False)

    g = {"llm": {p: game_ids(rows, llm[p]) for p in llm},
         "real": {p: game_ids(rows, real[p]) for p in real},
         "assumed": {p: game_ids(rows, assumed[p]) for p in assumed}}

    for name in ("llm", "real", "assumed"):
        print("%-8s games train %6d  val %5d  test %d"
              % (name, len(g[name]["train"]), len(g[name]["val"]),
                 len(g[name]["test"])))
    print()

    n = len(g["llm"]["val"])
    for name in ("real", "assumed"):
        hit = len(g["llm"]["val"] & g[name]["train"])
        print("LLM val inside the %-8s head's train: %5d / %d = %5.1f %%"
              % (name, hit, n, 100.0 * hit / max(1, n)))
    print()
    print("common held-out set, real flags   : %d games / %d states"
          % (len(g["llm"]["val"] & (g["real"]["val"] | g["real"]["test"])),
             20 * len(g["llm"]["val"] & (g["real"]["val"] | g["real"]["test"]))))
    print("common held-out set, old assumption: %d games / %d states"
          % (len(g["llm"]["val"] & (g["assumed"]["val"] | g["assumed"]["test"])),
             20 * len(g["llm"]["val"] & (g["assumed"]["val"] | g["assumed"]["test"]))))
    print()
    print("reranker pool (`headheld`, the assumed val) that the real head "
          "trained on: %d / %d = %.1f %%"
          % (len((g["assumed"]["val"] | g["assumed"]["test"]) & g["real"]["train"]),
             len(g["assumed"]["val"] | g["assumed"]["test"]),
             100.0 * len((g["assumed"]["val"] | g["assumed"]["test"])
                         & g["real"]["train"])
             / len(g["assumed"]["val"] | g["assumed"]["test"])))
    print()
    print("=> there is one split. Score on `--split heldout` (%d games) or "
          "`--split rr`." % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
