"""Data for win prediction: full drafts from the existing pipelines, one split.

What a row is
-------------
A **completed draft**: 10 picks in the canonical slot order
`[B_TOP..B_SUP, R_TOP..R_SUP]`, each with its role, the patch, and the outcome.
Nothing else. Specifically, and on purpose:

* **No bans.** Not read, not embedded, not attended to. The question is "how
  good is this composition", and bans are a *denial* signal whose information
  is already partly in what the opponent could not pick. Adding them is a
  separate experiment with its own ablation.
* **No masking, no prefixes.** One row per game, the finished board. The
  repo's masked/policy datasets exist for other questions; reusing their
  masking here would turn a win-probability number into a partly-information
  number.
* **No recency weights.** The pro split is random *within* each patch, so every
  patch is on both sides of the boundary; down-weighting old games would just
  make old patches unlearnable (the same argument as `proonly`'s
  `--half-life 0`).

Filters, and why they are (mostly) already done upstream
--------------------------------------------------------
* *Remakes / too-short games.* SoloQ: `pipeline.MIN_DURATION_S = 300` is
  applied at scan time. Pro: `pipeline.game_to_draft` drops any game without a
  complete 20-move draft (`incomplete_bans` / `incomplete_picks`), and the
  2022-2026 CSVs contain exactly **one** team-row block under 300 s
  (`ESPORTSTMNT01_3408461`, patch 13.14) which is not in the corpus for an
  unrelated reason (`bad_firstpick`). The loader re-checks nothing about
  duration because there is nothing left to check - the count is in the
  report instead of a filter that would silently match zero rows.
* *`blue_win is None`.* SoloQ: `scan_and_index(with_win=True)` counts those as
  `skipped_no_win` and never caches them. Pro: `bad_result` rows are dropped.
  Both loaders still assert `y in {0,1}` so a future pipeline change cannot
  quietly feed an unlabelled row into a CE loss.

The split
---------
**Random 80/10/10, seeded**, with one structural rule: in pro data whole
*series* stay together (a series is one patch, two teams, and - since 2025 - a
shared fearless pool, so splitting its games across train and test leaks
teams and picks). Series are assigned to the patch of their first game and the
draw happens **within each patch** (`Random("<seed>|<patch>")`), which is what
the brief asks for and what makes the patch embedding a legitimate input: the
same patch is on both sides, so the model interpolates a patch it has seen
rather than memorising an unseen id.

SoloQ is a single patch with no series structure, so it is a plain seeded
permutation.

Everything is reported: `n_train/n_val/n_test`, the per-patch allocation, the
patches too small to hold out (`min_patch_rows`), and the corpus frontier
(newest date in the data) so a number can never be quoted without the data it
was measured on.
"""
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
PRO_DIR = ROOT / "data" / "pro play"
SOLOQ_DIR = ROOT / "data" / "solo queue" / "process matchdata"

N_SLOTS = 10
TEAM_SIZE = 5
NUM_REAL_CHAMPS = 173
IGNORE_INDEX = -100

# One seed for the split and the model init. The brief asks for "the same seed
# as the baseline", and one number is the only way that cannot drift: the
# baseline fits its prior on `train`, so a different split seed would compare
# two models on different data even though both reports say "seed X".
SPLIT_SEED = 20261105
SPLIT_FRACS = (0.80, 0.10, 0.10)

# Patches with fewer games than this stay entirely in train. Their 10% tail is
# one game, which is noise rather than a measurement - the same rule (and the
# same reasoning) as `pipeline.split_within_patch(min_patch_rows=40)`, lowered
# here because this split only needs ~10% of a patch, not a 20% tail.
MIN_PATCH_ROWS = 10


def _load(name, path):
    """Import a module by path: these directories have spaces in their names
    and there are several sibling `pipeline.py` files, so sys.path ordering is
    never safe for them."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load("winpred_pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("winpred_pro_datasets", PRO_DIR / "datasets.py")
SQ_PIPE = _load("winpred_soloq_pipeline", SOLOQ_DIR / "pipeline.py")

CORPORA = {"2022": PRO_DIR / "pro_drafts.pt",
           "all": PRO_DIR / "pro_drafts_all.pt"}


# ------------------------------------------------------------------- table
class DraftTable:
    """A flat, numpy-side view of completed drafts.

    Deliberately not a torch dataset: the split is computed once over arrays,
    and the dataset below only slices. That keeps "which rows are test" a
    property of the data, not of a loader's iteration order.
    """

    __slots__ = ("champ", "role", "y", "patch_id", "game_id", "series_id",
                 "patch", "date", "source")

    def __init__(self, champ, role, y, patch_id, game_id, series_id, patch,
                 date, source):
        self.champ = champ            # [N,10] int16
        self.role = role              # [N,10] int8
        self.y = y                    # [N] int8, 1 = blue won
        self.patch_id = patch_id      # [N] int16 (index into patch_vocab)
        self.game_id = game_id        # list[str]
        self.series_id = series_id    # list[str]
        self.patch = patch            # list[str] normalised
        self.date = date              # list[str]
        self.source = source          # "soloq" | "pro"

    def __len__(self):
        return int(self.y.shape[0])

    def subset(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return DraftTable(self.champ[idx], self.role[idx], self.y[idx],
                          self.patch_id[idx],
                          [self.game_id[i] for i in idx],
                          [self.series_id[i] for i in idx],
                          [self.patch[i] for i in idx],
                          [self.date[i] for i in idx], self.source)

    def info(self):
        """Provenance block, written into every report."""
        return {"source": self.source, "n": len(self),
                "blue_win_rate": float(self.y.mean()),
                "patches": len({p for p in self.patch}),
                "date_min": min(self.date) if self.date else "",
                "date_max": max(self.date) if self.date else ""}


# --------------------------------------------------------------- pro loader
def _pro_rows(corpus="2022", verbose=True):
    cache = CORPORA[corpus]
    csv_glob = PRO_PIPE.CSV_GLOB if corpus == "2022" else PRO_PIPE.CSV_GLOB_ALL
    rows, stats = PRO_PIPE.build_rows(csv_glob=csv_glob, cache_path=cache)
    if verbose:
        print("pro corpus %r: %d drafts (cache %s)"
              % (corpus, len(rows), cache.name))
    return rows, stats


def load_pro(corpus="2022", verbose=True):
    """Completed pro drafts -> (DraftTable, meta).

    Bans are in the rows (`moves`) and are simply never read here; the filter
    below touches only the 10 pick slots and the label.
    """
    rows, stats = _pro_rows(corpus, verbose)
    patch_vocab, patch_index = PRO_DATA.build_patch_vocab(rows)
    champ, role, y, patch_id, gid, sid, pat, date = [], [], [], [], [], [], [], []
    dropped = Counter()
    for row in rows:
        lab = row.get("blue_win")
        if lab is None or int(lab) not in (0, 1):
            dropped["blue_win_none"] += 1
            continue
        c, r = list(row["champ"]), list(row["role"])
        if len(c) != N_SLOTS or len(r) != N_SLOTS:
            dropped["not_10_slots"] += 1
            continue
        if any(v is None or not 0 <= int(v) < NUM_REAL_CHAMPS for v in c):
            dropped["bad_champ"] += 1
            continue
        if any(v is None or not 0 <= int(v) < 5 for v in r):
            dropped["bad_role"] += 1
            continue
        if len(set(c)) != N_SLOTS:
            dropped["duplicate_champ"] += 1
            continue
        champ.append(c)
        role.append(r)
        y.append(int(lab))
        patch_id.append(PRO_DATA.patch_to_id(row.get("patch"), patch_index))
        gid.append(str(row["gameid"]))
        sid.append(str(row.get("series_id") or ("__game_%s" % row["gameid"])))
        pat.append(PRO_PIPE.patch_key(row.get("patch")))
        date.append(str(row.get("date") or ""))

    table = DraftTable(
        np.asarray(champ, dtype=np.int16), np.asarray(role, dtype=np.int8),
        np.asarray(y, dtype=np.int8), np.asarray(patch_id, dtype=np.int16),
        gid, sid, pat, date, "pro")
    meta = {"corpus": corpus, "pipeline_stats": stats,
            "patch_vocab": patch_vocab, "n_patches": len(patch_vocab) + 1,
            "dropped": dict(dropped),
            "corpus_max_date": PRO_PIPE.CORPUS_MAX_DATE,
            "newest_row_date": max(date) if date else ""}
    if verbose:
        print("  kept %d, dropped %s" % (len(table), dict(dropped)))
        print("  patches %d (%s .. %s), newest row %s (corpus pinned at %s)"
              % (len(patch_vocab), patch_vocab[0], patch_vocab[-1],
                 meta["newest_row_date"], PRO_PIPE.CORPUS_MAX_DATE))
    return table, meta


# ------------------------------------------------------------- soloq loader
def _soloq_cache(patch):
    return SOLOQ_DIR / ("drafts_%s.pt" % patch)


def load_soloq(patch="16.20", cache_path=None, verbose=True):
    """Completed soloQ drafts -> (DraftTable, meta).

    Reads the cache `model/pretraining` already builds (`drafts_<patch>.pt`,
    schema 2, with `wins`). A cache without win labels is rejected rather than
    rebuilt silently: this experiment is *only* the label, so a stale cache is
    a hard error, not a warning.

    The rows are `(champ[10], role[10])` tuples - one completed draft each, no
    order and no bans, which is exactly the input this model wants.
    """
    path = Path(cache_path) if cache_path else _soloq_cache(patch)
    if not path.exists():
        raise FileNotFoundError(
            "no soloQ cache at %s.\nBuild it once with the pretraining loader "
            "(python model/pretraining/train.py --limit-drafts 1 --no-wandb) "
            "or point --soloq-cache at an existing drafts_<patch>.pt." % path)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("wins") is None:
        raise RuntimeError(
            "%s has no win labels (schema %s); this experiment needs them. "
            "Rebuild with scan_and_index(..., with_win=True)."
            % (path.name, payload.get("schema", 1)))
    rows, wins = payload["rows"], payload["wins"]
    champ, role, y, gid = [], [], [], []
    dropped = Counter()
    for i, ((c, r), w) in enumerate(zip(rows, wins)):
        if w is None or int(w) not in (0, 1):
            dropped["blue_win_none"] += 1
            continue
        if len(c) != N_SLOTS or len(r) != N_SLOTS:
            dropped["not_10_slots"] += 1
            continue
        if any(not 0 <= int(v) < NUM_REAL_CHAMPS for v in c):
            dropped["bad_champ"] += 1
            continue
        if len(set(int(v) for v in c)) != N_SLOTS:
            dropped["duplicate_champ"] += 1
            continue
        champ.append([int(v) for v in c])
        role.append([int(v) for v in r])
        y.append(int(w))
        gid.append("sq_%d" % i)

    patch_key = PRO_PIPE.patch_key(patch)
    table = DraftTable(
        np.asarray(champ, dtype=np.int16), np.asarray(role, dtype=np.int8),
        np.asarray(y, dtype=np.int8),
        np.full(len(y), 1, dtype=np.int16),      # vocab = [patch] -> id 1
        gid, gid, [patch_key] * len(y), [""] * len(y), "soloq")
    meta = {"corpus": "soloq", "patch": patch_key,
            "patch_vocab": [patch_key], "n_patches": 2,
            "dropped": dict(dropped),
            "cache": path.name, "cache_kept": payload.get("kept"),
            "cache_skipped_no_win": payload.get("skipped_no_win"),
            "min_duration_s": SQ_PIPE.MIN_DURATION_S,
            "newest_row_date": ""}
    if verbose:
        print("soloQ %s: %d drafts (cache %s, no-duration<%.0fs, no remakes)"
              % (patch_key, len(table), path.name,
                 SQ_PIPE.MIN_DURATION_S))
        print("  dropped %s" % dict(dropped))
    return table, meta


# ------------------------------------------------------------------- splits
def _take(seq, target):
    """Pop whole groups off the front of `seq` until `target` rows are taken."""
    out, acc = [], 0
    while seq and acc < target:
        g = seq.pop(0)
        out.append(g)
        acc += len(g)
    return out


def split_soloq(n, seed=SPLIT_SEED, fracs=SPLIT_FRACS):
    """Plain seeded 80/10/10 permutation (one patch, no series structure)."""
    tr, va, te = fracs
    order = list(range(n))
    random.Random("%s|soloq" % seed).shuffle(order)
    n_te = int(round(n * te))
    n_va = int(round(n * va))
    return {"train": np.asarray(order[n_te + n_va:], dtype=np.int64),
            "val": np.asarray(order[n_te:n_te + n_va], dtype=np.int64),
            "test": np.asarray(order[:n_te], dtype=np.int64)}


def split_pro(table, seed=SPLIT_SEED, fracs=SPLIT_FRACS,
              min_patch_rows=MIN_PATCH_ROWS):
    """Random 80/10/10 **within each patch**, whole series kept together.

    Series (rows sharing `series_id`) are drawn, not rows, because two games of
    a series share both teams and - from 2025 - a fearless pool: splitting them
    across the boundary would put the same players and the same removed
    champions on both sides. The draw is seeded per patch
    (`Random("<seed>|<patch>")`), so one patch's allocation does not depend on
    which other patches exist or on iteration order.

    Returns {"train"/"val"/"test": np.int64 arrays} plus a per-patch report in
    `split_report(table, splits, ...)`.
    """
    tr_frac, va_frac, te_frac = fracs
    by_series = defaultdict(list)
    for i, sid in enumerate(table.series_id):
        by_series[sid].append(i)
    # A series belongs to the patch of its first game, so a series played across
    # a patch boundary cannot be the head of one patch and the tail of another.
    by_patch = defaultdict(list)
    for sid, idxs in by_series.items():
        idxs.sort(key=lambda i: table.game_id[i])
        by_patch[table.patch[idxs[0]]].append(idxs)

    out = {"train": [], "val": [], "test": []}
    for patch in sorted(by_patch, key=PRO_PIPE.patch_sort_key):
        groups = [list(g) for g in by_patch[patch]]
        n = sum(len(g) for g in groups)
        if n < min_patch_rows:
            out["train"].extend(i for g in groups for i in g)
            continue
        random.Random("%s|%s" % (seed, patch)).shuffle(groups)
        test = _take(groups, int(round(n * te_frac)))
        val = _take(groups, int(round(n * va_frac)))
        out["test"].extend(i for g in test for i in g)
        out["val"].extend(i for g in val for i in g)
        out["train"].extend(i for g in groups for i in g)

    return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in out.items()}


def split_report(table, splits, seed=SPLIT_SEED, fracs=SPLIT_FRACS,
                 min_patch_rows=MIN_PATCH_ROWS):
    """JSON-able description of the split: sizes, per-patch allocation, checks."""
    n = len(table)
    owner = np.empty(n, dtype=object)
    for name in ("train", "val", "test"):
        owner[splits[name]] = name
    per_patch = {}
    for patch in sorted(set(table.patch), key=PRO_PIPE.patch_sort_key):
        rows = np.flatnonzero(np.asarray(table.patch) == patch)
        if rows.size == 0:
            continue
        c = Counter(owner[rows].tolist())
        per_patch[patch] = {"n": int(rows.size), "train": c.get("train", 0),
                            "val": c.get("val", 0), "test": c.get("test", 0),
                            "held_out": int(rows.size) < min_patch_rows}
    series_sets = {name: {table.series_id[i] for i in splits[name]}
                   for name in ("train", "val", "test")}
    straddle = len(series_sets["train"] & series_sets["test"]) + \
        len(series_sets["train"] & series_sets["val"]) + \
        len(series_sets["val"] & series_sets["test"])
    counts = {name: int(len(splits[name])) for name in ("train", "val", "test")}
    return {
        "seed": seed,
        "fracs": list(fracs),
        "unit": "series" if table.source == "pro" else "game",
        "counts": counts,
        "shares": {k: v / n for k, v in counts.items()},
        "min_patch_rows": min_patch_rows,
        "n_series_straddling": straddle,
        "patches_all_train": [p for p, v in per_patch.items()
                              if v["held_out"]],
        "by_patch": per_patch,
    }


def make_splits(table, seed=SPLIT_SEED, fracs=SPLIT_FRACS,
                min_patch_rows=MIN_PATCH_ROWS, verbose=True):
    """Dispatch to the right splitter and attach the report."""
    if table.source == "pro":
        splits = split_pro(table, seed, fracs, min_patch_rows)
    else:
        splits = split_soloq(len(table), seed, fracs)
    rep = split_report(table, splits, seed, fracs, min_patch_rows)
    if verbose:
        print("split (seed %s, %s-level, %.0f/%.0f/%.0f): train %d / val %d / "
              "test %d" % (seed, rep["unit"], 100 * fracs[0], 100 * fracs[1],
                           100 * fracs[2], rep["counts"]["train"],
                           rep["counts"]["val"], rep["counts"]["test"]))
        print("  series straddling a split: %d (must be 0)" %
              rep["n_series_straddling"])
        if rep["patches_all_train"]:
            print("  %d patches below min_patch_rows stay in train: %s"
                  % (len(rep["patches_all_train"]),
                     ", ".join(rep["patches_all_train"][:8])))
    return splits, rep


# ----------------------------------------------------------------- dataset
class WinDataset(Dataset):
    """One completed draft per example. No masking, no prefixes, no bans."""

    def __init__(self, table, idx=None):
        self.champ = table.champ if idx is None else table.champ[idx]
        self.role = table.role if idx is None else table.role[idx]
        self.y = table.y if idx is None else table.y[idx]
        self.patch_id = (table.patch_id if idx is None
                         else table.patch_id[idx])
        self.game_id = (table.game_id if idx is None
                        else [table.game_id[i] for i in idx])
        self.patch = table.patch if idx is None else \
            [table.patch[i] for i in idx]

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i):
        return {
            "champ_ids": torch.as_tensor(self.champ[i], dtype=torch.long),
            "role_ids": torch.as_tensor(self.role[i], dtype=torch.long),
            "patch_id": torch.tensor(int(self.patch_id[i]), dtype=torch.long),
            "win_label": torch.tensor(float(self.y[i]), dtype=torch.float32),
            "meta": {"game_id": self.game_id[i], "patch": self.patch[i]},
        }


def collate(samples):
    """Stack tensors, keep the meta dicts as a list."""
    out = {}
    for key in samples[0]:
        if key == "meta":
            out[key] = [s[key] for s in samples]
        else:
            out[key] = torch.stack([s[key] for s in samples])
    return out


# -------------------------------------------------- prediction-set I/O
PRED_KEYS = ("y", "p", "game_id", "series_id", "patch", "date")


def save_predictions(path, y, p, table, idx):
    """Write a prediction set: label, probability, and the rows it belongs to.

    The ids are the contract. Two prediction files can only be compared if they
    were scored on the same games, and storing the ids is what lets
    `evaluate.py` prove that instead of assuming it.
    """
    idx = np.asarray(idx, dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        y=np.asarray(y, dtype=np.int8),
        p=np.asarray(p, dtype=np.float32),
        game_id=np.asarray([table.game_id[i] for i in idx], dtype=str),
        series_id=np.asarray([table.series_id[i] for i in idx], dtype=str),
        patch=np.asarray([table.patch[i] for i in idx], dtype=str),
        date=np.asarray([table.date[i] for i in idx], dtype=str))
    return path


def load_predictions(path):
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def align_predictions(pred, game_ids):
    """Reorder a prediction set to `game_ids`, refusing any other row set.

    A subset or superset is an error, not something to intersect: silently
    intersecting would let two models be compared on the rows they happen to
    share, which is exactly the comparison that flatters the worse model.
    """
    want = list(game_ids)
    have = [str(g) for g in pred["game_id"]]
    if len(have) != len(set(have)):
        raise ValueError("prediction file has duplicate game ids")
    if set(have) != set(want):
        missing = len(set(want) - set(have))
        extra = len(set(have) - set(want))
        raise ValueError(
            "prediction file was scored on a different row set "
            "(%d of %d expected ids missing, %d unexpected)"
            % (missing, len(want), extra))
    pos = {g: i for i, g in enumerate(have)}
    order = np.asarray([pos[g] for g in want], dtype=np.int64)
    return {k: v[order] for k, v in pred.items()}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    return path
