"""One seeded 80/10/10 split, shared by every stage and every experiment.

Why this exists, next to `pipeline.split_series_by_date` and
`pipeline.split_within_patch`. Those two answer temporal questions ("does a
patch's prior transfer forward", "will this work next week") and both pay for
it with a tiny test split: the pinned test is 3 series / 7 games, and the
within-patch tail is 7 games too. Seven games cannot rank two models - the
Wilson interval is ~35 points wide - so every model-vs-model claim in STATUS
had to be made on the 4,768-row val instead, with the test split reduced to a
smoke test.

This module builds the split you need to *compare* models: 80/10/10, seeded,
grouped by series, stratified by patch, ~4,700 test drafts / ~94,000 next-move
states. Absolute numbers here are interpolation-flavoured (test patches are
seen in train, so a patch embedding can memorise) - that is the price of the
power, and it is the right trade for ranking arms. The date split stays the
extrapolation check; this split is the leaderboard.

Reproducibility contract:
  * the only inputs are (seed, corpus, fraction) - no wall clock, no hash order
  * draws are per patch (`Random("<seed>|pro|<patch>")`), so adding or removing
    a patch cannot reshuffle another patch's assignment
  * the manifest stores the series ids themselves plus a sha1 of each split,
    and `load` recomputes the split and refuses to proceed if the hash moved.
    A corpus change is therefore loud, not silent.

Groups are whole series, never rows: fearless draft means game 2 of a series
depends on game 1, so a row-level split leaks.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20261105
FRACS = (0.8, 0.1, 0.1)
# A patch with fewer series than this cannot supply three splits; it goes
# entirely to train. Keeps every split's per-patch slice non-degenerate.
MIN_PATCH_SERIES = 10

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPLIT_DIR = ROOT / "data" / "splits"


def _sha(obj) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _series_of(rows):
    """series_id -> [row indices]. Rows without a series are singletons."""
    out = defaultdict(list)
    for i, row in enumerate(rows):
        sid = row.get("series_id")
        out[sid if sid is not None else "__null_%d" % i].append(i)
    return out


def split_pro(rows, seed=SEED, fracs=FRACS,
              min_patch_series=MIN_PATCH_SERIES):
    """80/10/10 over pro drafts, whole series, stratified by patch.

    Stratifying by patch keeps the patch mix identical across splits, which is
    what makes two arms comparable: without it a split could be 12% newer
    patches and win by accident. `min_patch_series` patches contribute train
    rows only (7 patches / 194 rows on this corpus at the default).
    """
    tr, va, te = fracs
    assert abs(tr + va + te - 1.0) < 1e-9, fracs
    series = _series_of(rows)
    by_patch = defaultdict(list)
    for sid, idxs in series.items():
        by_patch[rows[idxs[0]].get("patch", "?")].append(sid)

    out = {"train": [], "val": [], "test": []}
    for patch in sorted(by_patch):
        sids = sorted(by_patch[patch])          # stable before shuffling
        if len(sids) < min_patch_series:
            out["train"].extend(i for s in sids for i in series[s])
            continue
        rng = random.Random("%s|pro|%s" % (seed, patch))
        rng.shuffle(sids)
        n = len(sids)
        n_test = max(1, int(round(n * te)))
        n_val = max(1, int(round(n * va)))
        # guard: never let val/test eat the training side of a small patch
        if n - n_test - n_val < 1:
            n_test, n_val = 0, 0
        out["test"].extend(i for s in sids[:n_test] for i in series[s])
        out["val"].extend(i for s in sids[n_test:n_test + n_val]
                          for i in series[s])
        out["train"].extend(i for s in sids[n_test + n_val:] for i in series[s])
    for k in out:
        out[k].sort()
    return out


def split_soloq(n_rows, seed=SEED, fracs=FRACS):
    """80/10/10 over soloQ drafts by index.

    No series concept: every soloQ row is an independent game with no fearless
    context, so a row-level split cannot leak. The permutation is drawn from a
    dedicated stream so it is independent of the pro draw.
    """
    tr, va, te = fracs
    perm = list(range(n_rows))
    random.Random("%s|soloq" % seed).shuffle(perm)
    n_test = int(round(n_rows * te))
    n_val = int(round(n_rows * va))
    return {"test": sorted(perm[:n_test]),
            "val": sorted(perm[n_test:n_test + n_val]),
            "train": sorted(perm[n_test + n_val:])}


def describe(rows, splits, label):
    """Counts + composition, for the manifest and for the printed record."""
    n = {k: len(v) for k, v in splits.items()}
    tot = max(1, sum(n.values()))
    patches = {k: len({rows[i].get("patch") for i in v}) for k, v in
               splits.items()}
    series = {k: len({rows[i].get("series_id") for i in v}) for k, v in
              splits.items()}
    return {"label": label, "n": n,
            "frac": {k: round(v / tot, 4) for k, v in n.items()},
            "n_patches": patches, "n_series": series}


def build(rows, soloq_n=0, seed=SEED, fracs=FRACS):
    pro = split_pro(rows, seed=seed, fracs=fracs)
    man = {"seed": seed, "fracs": list(fracs),
           "pro": describe(rows, pro, "pro"),
           "pro_sha": {k: _sha([rows[i].get("series_id") for i in v])
                       for k, v in pro.items()},
           "pro_series": {k: sorted({str(rows[i].get("series_id")) for i in v})
                          for k, v in pro.items()}}
    if soloq_n:
        sq = split_soloq(soloq_n, seed=seed, fracs=fracs)
        man["soloq"] = {"n": {k: len(v) for k, v in sq.items()},
                        "sha": {k: _sha(v) for k, v in sq.items()}}
    return pro, man


def save(man, path=None):
    path = Path(path) if path else SPLIT_DIR / ("pro_seed%d.json" % man["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=1, sort_keys=True), "utf-8")
    return path


def load(rows, path=None, soloq_n=0):
    """Rebuild the split and verify it against the manifest on disk.

    `soloq_n` (the row count of the soloQ payload) additionally verifies the
    soloQ half. Pass it whenever the soloQ split is about to be used - the pro
    half alone does not prove the soloQ half is unchanged.
    """
    path = Path(path) if path else SPLIT_DIR / ("pro_seed%d.json" % SEED)
    man = json.loads(path.read_text("utf-8"))
    pro, fresh = build(rows, soloq_n=soloq_n, seed=man["seed"],
                       fracs=tuple(man["fracs"]))
    if fresh["pro_sha"] != man["pro_sha"]:
        raise RuntimeError(
            "split manifest %s does not match the corpus: the series ids in "
            "one or more splits changed. The corpus is frozen at "
            "CORPUS_MAX_DATE; if you rebuilt it, regenerate the manifest "
            "deliberately instead of training on a silently different split."
            % path)
    if soloq_n and fresh["soloq"]["sha"] != man.get("soloq", {}).get("sha"):
        raise RuntimeError("soloQ split does not match the manifest in %s"
                           % path)
    return pro, man
