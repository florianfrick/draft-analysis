"""The project's one train/val/test split, in one importable place.

Every number in this repository is only comparable to another number measured on
the *same rows*. That was not true before this file existed: the pre-`nov13`
protocol pinned three Worlds series as a 7-game test set, the `-randomvalsplit-`
pair drew 10% of every patch at random, and the `-nov13-` pair drew a 10% val
and a 5% test tail. Three protocols, three corpora, and a comparison across any
two of them is a comparison of two different questions.

This module is the single definition. It is the split the shipped `-nov13-` pair
was trained on, reproduced exactly:

    corpus    data/pro play/pro_drafts_full.pt   46,939 drafts, 107 patches
    split     split_within_patch, no pin, tail
    train     39,720   (84.62%)
    val        4,771   (10.16%)
    test       2,448   ( 5.22%)

Why this one and not the pinned series:

* **The test set is big enough to rank models.** 2,448 games is 12,240 masked
  slots and 48,960 next-move states. The pinned split was 7 games / 70 slots /
  140 states, where a 95% Wilson interval is ~22 points wide and two models
  whose intervals overlap are tied as far as the split can tell - i.e. it could
  detect a broken model and nothing else.
* **It is reproducible without a pin.** The tail is deterministic: series are
  sorted by (date, gameid) inside each patch and the last ones become test. No
  seed, no hand-maintained list of gameids, nothing to drift.
* **It is what the shipped models actually trained on.** A test split that the
  incumbent was not held out of is not a comparison; it is a home game.
* **No holdout.** A series the corpus has never seen cannot be a test split -
  it can only be a holdout nothing can be trained against, and the moment you
  retrain you have to decide whether to include it. The fractional tail keeps
  every game in the corpus and lets the split do the holding out.

`min_patch_rows=40` keeps 8 patches (including the 25-row newest, 16.22)
entirely in train: their tail would be one or two games, which is noise rather
than a measurement. Consequence, stated because it is easy to misread: **the
newest patch is train-only**, and the test tail belongs to the newest *eligible*
patch. A model is therefore not tested on the very newest patch, only on the
newest patch large enough to have a tail.

What a future model has to do to be comparable
----------------------------------------------
Call `build(rows)` from this module instead of hand-rolling a split call, or
pass the same arguments to `pipeline.split_within_patch`:

    val_frac=0.10, test_frac=0.05, min_patch_rows=40,
    pinned_test=(), val_patches=None, val_random=False

then run `python -m pro.model.test_split.split --check` and confirm the counts
below. `pro/model/test_split/suite.py` refuses to label a model "comparable"
unless its training report records these counts, so a split that drifted is
visible rather than silent.

The manifest
------------
`split_manifest.json` stores the gameids of each split plus a sha256 over the
sorted test gameids. `--check` recomputes and diffs it. The manifest is what
makes the split a contract rather than a convention: a corpus refresh that
moved a single game into test changes the hash, and the check says so.

Usage:
    python -m pro.model.test_split.split            # print the split
    python -m pro.model.test_split.split --check    # verify against the manifest
    python -m pro.model.test_split.split --write    # regenerate the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../pro/model/test_split
MODEL_DIR = HERE.parent                           # .../pro/model
ROOT = MODEL_DIR.parents[1]                       # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"

sys.path.insert(0, str(ROOT))

# The corpus, named by its cache file rather than by an argument string. This is
# `--corpus 2022-full` in `proonly/common.py`: the same CSV globs as `2022`,
# with the corpus freeze REMOVED (`max_date=""`). Its own cache file on purpose
# - reusing `pro_drafts.pt` would either serve frozen rows or overwrite the
# corpus every pre-nov13 number in STATUS.md was measured on.
CORPUS = PRO_DIR / "pro_drafts_full.pt"
MAX_DATE = ""                    # "" = no freeze; None would mean CORPUS_MAX_DATE

# Passed to `pipeline.split_within_patch` verbatim. `seed` is absent on
# purpose: the tail split never draws, so there is no seed to get wrong. (The
# `-randomvalsplit-` protocol did draw, and passing an explicit `seed` there is
# what silently created a second independent 10% - see `llm/_split_overlap.py`.)
SPLIT_ARGS = dict(val_frac=0.10, test_frac=0.05, min_patch_rows=40,
                  pinned_test=(), val_patches=None, val_random=False)

# The contract. `build()` asserts against these; `--check` diffs them.
EXPECTED = {"corpus": 46939, "train": 39720, "val": 4771, "test": 2448}
EXPECTED_SHARES = {"train": 0.8462, "val": 0.1016, "test": 0.0522}

MANIFEST = HERE / "split_manifest.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("pro_pipeline", PRO_DIR / "pipeline.py")


def load_rows(verbose=False):
    """The corpus rows this split is defined on.

    Reads the cache; never rebuilds. `build_rows` would rescan on a changed CSV
    signature, which would change the row order and therefore the split - so a
    caller that wants the contract to hold must not pass `rescan=True`.
    """
    if not CORPUS.exists():
        raise SystemExit(
            "missing corpus cache: %s\n"
            "Rebuild it with:\n"
            "  docker compose run --rm -T dev python pro/model/proonly/train_masked.py "
            "--corpus 2022-full --epochs 0\n"
            "(any proonly run with --corpus 2022-full writes the cache)" % CORPUS)
    rows, _stats = P.build_rows(csv_glob=P.CSV_GLOB, cache_path=CORPUS,
                                max_date=MAX_DATE)
    if verbose:
        print("corpus: %d drafts from %s" % (len(rows), CORPUS.name))
    return rows


def build(rows, strict=True):
    """rows -> {"train": [idx], "val": [idx], "test": [idx]}.

    `strict` asserts the counts in `EXPECTED`. Turn it off only for a smoke
    run on a subsampled corpus, where the counts cannot match.
    """
    splits = P.split_within_patch(rows, strict_pin=False, **SPLIT_ARGS)
    if strict:
        got = {"corpus": len(rows)}
        got.update({k: len(v) for k, v in splits.items()})
        bad = {k: (EXPECTED[k], got[k]) for k in EXPECTED if got[k] != EXPECTED[k]}
        if bad:
            raise AssertionError(
                "the split drifted from the contract:\n"
                + "\n".join("  %-6s expected %d, got %d" % (k, e, g)
                            for k, (e, g) in bad.items())
                + "\nThe corpus or the split arguments changed. Every number "
                  "measured before this is not comparable; either restore the "
                  "corpus or deliberately re-baseline with `--write`.")
    return splits


def shares(splits):
    n = sum(len(v) for v in splits.values()) or 1
    return {k: len(v) / n for k, v in splits.items()}


def gameids(rows, idx):
    return [rows[i]["gameid"] for i in idx]


def fingerprint(rows, test_idx):
    """sha256 over the sorted test gameids.

    Sorted, so a change in row order that leaves the set alone does not look
    like drift - the set is the contract, not the order.
    """
    joined = "\n".join(sorted(gameids(rows, test_idx)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def summarise(rows, splits):
    return {
        "corpus": CORPUS.name,
        "corpus_rows": len(rows),
        "max_date": MAX_DATE or "(none)",
        "args": {k: v for k, v in SPLIT_ARGS.items() if k != "pinned_test"},
        "pinned_test": list(SPLIT_ARGS["pinned_test"]),
        "counts": {k: len(v) for k, v in splits.items()},
        "shares": {k: round(v, 4) for k, v in shares(splits).items()},
        "test_fingerprint": fingerprint(rows, splits["test"]),
        "test_gameids": sorted(gameids(rows, splits["test"])),
    }


def write_manifest(rows, splits, path=MANIFEST):
    payload = summarise(rows, splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False, sort_keys=True)
    return payload


def load_manifest(path=MANIFEST):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify(rows, splits=None, path=MANIFEST):
    """Compare a built split against the manifest. Returns (ok, diffs)."""
    splits = build(rows, strict=False) if splits is None else splits
    man = load_manifest(path)
    if man is None:
        return False, ["no manifest at %s - run --write once" % path]
    got = summarise(rows, splits)
    diffs = []
    for k in ("counts", "corpus_rows", "test_fingerprint"):
        if got[k] != man.get(k):
            diffs.append("%s: manifest %r != current %r"
                         % (k, man.get(k), got[k]))
    if got["args"] != man.get("args"):
        diffs.append("args: manifest %r != current %r"
                     % (man.get("args"), got["args"]))
    return (not diffs), diffs


def sample_games(rows, test_idx, n, seed=0):
    """`n` test games at random, reproducibly.

    For the per-state tables that need "what did each model say *here*". The
    old harness used the 7 pinned series for this and thereby turned an
    illustration into a leaderboard. A random draw from the 2,448 says what it
    is: a sample of the test set, not the test set.
    """
    import random
    pool = list(test_idx)
    if n <= 0 or n >= len(pool):
        return sorted(pool)
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


def _print_summary(rows, splits):
    sh = shares(splits)
    print("corpus  %s  %d drafts" % (CORPUS.name, len(rows)))
    print("args    %s" % {k: v for k, v in SPLIT_ARGS.items()
                          if k != "pinned_test"})
    for k in ("train", "val", "test"):
        print("  %-5s %6d  (%.2f%%)" % (k, len(splits[k]), 100 * sh[k]))
    print("test fingerprint %s" % fingerprint(rows, splits["test"])[:16])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the current split against split_manifest.json")
    ap.add_argument("--write", action="store_true",
                    help="regenerate split_manifest.json from the corpus")
    args = ap.parse_args(argv)

    rows = load_rows(verbose=True)
    splits = build(rows, strict=False)
    _print_summary(rows, splits)

    if args.write:
        payload = write_manifest(rows, splits)
        print("wrote %s (%d test gameids)" % (MANIFEST, len(payload["test_gameids"])))
        return 0
    if args.check:
        ok, diffs = verify(rows, splits)
        if ok:
            print("OK: matches the manifest")
            return 0
        print("MISMATCH:")
        for d in diffs:
            print("  " + d)
        return 1
    ok, diffs = verify(rows, splits)
    print("manifest: %s" % ("matches" if ok else "DIFFERS (%d)" % len(diffs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
