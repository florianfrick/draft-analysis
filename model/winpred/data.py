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

import datetime

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

    `domain` is the per-row origin ("soloq" / "pro"). It equals `source` for a
    single-dataset table and is what makes a `--data both` table splittable:
    each domain is split by its own rule, so the soloQ test rows of the joint
    run are the *same rows* as the soloQ-only run's test set. Without that,
    every cross-arm comparison would be a comparison of two different test
    sets.
    """

    __slots__ = ("champ", "role", "y", "patch_id", "game_id", "series_id",
                 "patch", "date", "source", "domain")

    def __init__(self, champ, role, y, patch_id, game_id, series_id, patch,
                 date, source, domain=None):
        self.champ = champ            # [N,10] int16
        self.role = role              # [N,10] int8
        self.y = y                    # [N] int8, 1 = blue won
        self.patch_id = patch_id      # [N] int16 (index into patch_vocab)
        self.game_id = game_id        # list[str]
        self.series_id = series_id    # list[str]
        self.patch = patch            # list[str] normalised
        self.date = date              # list[str]
        self.source = source          # "soloq" | "pro" | "both"
        self.domain = (np.asarray(domain if domain is not None
                                  else [source] * len(y), dtype=object))

    def __len__(self):
        return int(self.y.shape[0])

    def subset(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return DraftTable(self.champ[idx], self.role[idx], self.y[idx],
                          self.patch_id[idx],
                          [self.game_id[i] for i in idx],
                          [self.series_id[i] for i in idx],
                          [self.patch[i] for i in idx],
                          [self.date[i] for i in idx], self.source,
                          self.domain[idx])

    @classmethod
    def concat(cls, tables, source="both"):
        """Concatenate tables in order. They must share a patch vocabulary."""
        return cls(
            np.concatenate([t.champ for t in tables]),
            np.concatenate([t.role for t in tables]),
            np.concatenate([t.y for t in tables]),
            np.concatenate([t.patch_id for t in tables]),
            [g for t in tables for g in t.game_id],
            [s for t in tables for s in t.series_id],
            [p for t in tables for p in t.patch],
            [d for t in tables for d in t.date],
            source,
            np.concatenate([t.domain for t in tables]))

    def domains(self):
        """The distinct domains present, in a stable order."""
        return sorted(set(self.domain.tolist()))

    def info(self):
        """Provenance block, written into every report."""
        out = {"source": self.source, "n": len(self),
               "blue_win_rate": float(self.y.mean()),
               "patches": len({p for p in self.patch}),
               "date_min": min(self.date) if self.date else "",
               "date_max": max(self.date) if self.date else ""}
        if len(self.domains()) > 1:
            out["by_domain"] = {
                d: {"n": int((self.domain == d).sum()),
                    "blue_win_rate": float(self.y[self.domain == d].mean())}
                for d in self.domains()}
        return out


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


def _team_pairs(table, corpus="2022", verbose=False):
    """Per-row (blue team, red team) names, aligned to `table`'s rows."""
    rows, _ = _pro_rows(corpus, verbose=verbose)
    teams = {str(r["gameid"]): (str(r["blue_team"]), str(r["red_team"]))
             for r in rows}
    tb = np.array([teams[g][0] for g in table.game_id], dtype=object)
    tr = np.array([teams[g][1] for g in table.game_id], dtype=object)
    return tb, tr


_PLAYER_INDEX = {}


def _player_index(corpus="2022", verbose=False):
    """`gameid -> ([blue player ids in role order], [red ...])`.

    Read straight from the CSVs rather than through `pipeline.build_rows` on
    purpose: the row schema there is shared with the policy work and the cached
    `pro_drafts.pt` would have to be rebuilt for every consumer just so this
    experiment could see one extra column. The scan is 18 s for 2022-2026 and
    is memoised for the process, which is what a search run needs.

    `playerid` is preferred over `playername` (it survives a rename) and rows
    without either are dropped - a game missing a player simply contributes no
    player term, which is the same treatment an unseen player gets.
    """
    if corpus in _PLAYER_INDEX:
        return _PLAYER_INDEX[corpus]
    import csv
    import glob

    pattern = str(PRO_PIPE.CSV_GLOB if corpus == "2022"
                  else PRO_PIPE.CSV_GLOB_ALL)
    role_of = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                role = role_of.get(row.get("position"))
                side = row.get("side")
                if role is None or side not in ("Blue", "Red"):
                    continue
                pid = ((row.get("playerid") or "").strip()
                       or (row.get("playername") or "").strip())
                gid = row.get("gameid")
                if not gid or not pid:
                    continue
                slot = out.setdefault(gid, ([None] * 5, [None] * 5))
                slot[0 if side == "Blue" else 1][role] = pid
    _PLAYER_INDEX[corpus] = out
    if verbose:
        print("  player index: %d games" % len(out))
    return out


def _roster_columns(table, corpus="2022", verbose=False):
    """Per-row `(blue, red)` tuples of player ids, aligned to `table`."""
    index = _player_index(corpus, verbose=verbose)
    blue, red = [], []
    for g in table.game_id:
        b, r = index.get(str(g), ((), ()))
        blue.append(tuple(p for p in b if p))
        red.append(tuple(p for p in r if p))
    return blue, red


def _age_days(dates):
    """Days *before the corpus frontier*, 0 = newest row possible.

    Relative to `CORPUS_MAX_DATE` and not to the train maximum, because the
    train maximum moves with the split; a decay weight that shifts when the
    split is redrawn would make two runs incomparable for a reason that has
    nothing to do with the model. Unparseable dates count as "newest", which
    fails toward keeping the row rather than silently dropping it.
    """
    try:
        frontier = datetime.date.fromisoformat(str(PRO_PIPE.CORPUS_MAX_DATE)[:10])
    except ValueError:
        frontier = None
    out = np.zeros(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        if frontier is None:
            break
        try:
            out[i] = float((frontier
                            - datetime.date.fromisoformat(str(d)[:10])).days)
        except ValueError:
            out[i] = 0.0
    return out


def _fit_ratings(rows_i, cols_i, vals, n_cols, lam, y, idx, weight):
    """Sparse L2 logistic fit of `logit P(blue) = X r + h`. -> (r, h).

    Shared by the team/player and the champion estimators: the design matrix is
    the only thing that differs between them, and it is passed in as
    coordinate lists so each caller can build it from whatever ids it has.

    `lam` is per-column, which is what lets two blocks in one matrix shrink by
    the same *pseudo-games* rule despite contributing different per-game
    weights (`_bt_ratings` derives it, see the comment there).
    """
    idx = np.asarray(idx, dtype=np.int64)
    X = torch.sparse_coo_tensor(
        torch.as_tensor(np.stack([np.concatenate(rows_i),
                                  np.concatenate(cols_i)])),
        torch.as_tensor(np.concatenate(vals), dtype=torch.float64),
        (idx.size, n_cols), check_invariants=False).coalesce()

    r = torch.zeros(n_cols, dtype=torch.float64, requires_grad=True)
    h = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    yt = torch.as_tensor(y[idx], dtype=torch.float64)
    wt = torch.as_tensor(weight, dtype=torch.float64)
    lam_t = torch.as_tensor(np.asarray(lam, dtype=np.float64))

    opt = torch.optim.LBFGS([r, h], lr=1.0, max_iter=300, history_size=30,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = torch.sparse.mm(X, r.unsqueeze(1)).squeeze(1) + h
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            z, yt, reduction="none")
        loss = ((wt * bce).sum() + 0.5 * (lam_t * r ** 2).sum()) / wt.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return r.detach().numpy(), float(h.detach())


def _bt_ratings(tb, tr, y, idx, k, weight, verbose=False, roster=None, kp=None,
                use_team=True):
    """Bradley-Terry ratings by L2 logistic regression, train rows only.

    The smoothed win rate treats every game as equally informative about a
    team; Bradley-Terry does not, because it conditions on the opponent. A team
    at 60 % in a league of strong teams is not a team at 60 % in a minor
    league, and the marginal rate cannot tell them apart - which matters here
    because the corpus spans several leagues and the whole point of the feature
    is *who is playing*.

    Model: `logit P(blue wins) = r_blue - r_red + h`, with `r` an L2-penalised
    per-team rating and `h` a side intercept. `h` is fitted (ratings should be
    side-agnostic) but **not** returned - the head already owns a `side_bias`,
    and shipping the same constant twice would double-count it.

    `k` keeps its meaning from the rate version by dividing it by the curvature
    of the log-loss at p = 0.5 (0.25 per game): `k` pseudo-games of shrinkage
    is then `lambda = k / 4` in the penalised objective. With `weight` this
    becomes a recency-weighted fit, which is the second reason to prefer it -
    rosters change, and a 2022 win rate is weak evidence about a 2026 team.

    `roster=(blue_players, red_players)` adds a **player block**: a joint fit of
    `r_blue - r_red + mean(r_player_blue) - mean(r_player_red) + h`. A team tag
    is a roster that changes; a player rating is not, so this separates "T1 is
    good" from "T1 still has Faker". `kp` is the player shrinkage (default
    `k`). Returns `[N, 2]` float32 - per-side strength, not ratings - so the
    caller never has to know which blocks were active.
    """
    kp = k if kp is None else kp
    names = sorted({tb[i] for i in idx} | {tr[i] for i in idx}) if use_team \
        else []
    tcol = {t: j for j, t in enumerate(names)}
    n_team = len(names)
    idx = np.asarray(idx, dtype=np.int64)

    rows_i, cols_i, vals = [], [], []
    if n_team:
        ii = np.repeat(np.arange(idx.size), 2)
        cc = np.empty(2 * idx.size, dtype=np.int64)
        cc[0::2] = [tcol[tb[i]] for i in idx]
        cc[1::2] = [tcol[tr[i]] for i in idx]
        rows_i.append(ii)
        cols_i.append(cc)
        vals.append(np.tile([1.0, -1.0], idx.size))

    # Optional player block: each of the five players contributes 1/5, so a
    # side's player term is the *mean* of its five ratings. Mean and not sum
    # because the returned feature must stay on the same scale as the team
    # term - the head consumes the two together and cannot rescale one of them.
    blue_p = red_p = None
    pcol, n_player = {}, 0
    if roster is not None:
        blue_p, red_p = roster
        pnames = sorted({p for i in idx for p in blue_p[i]}
                        | {p for i in idx for p in red_p[i]})
        pcol = {p: n_team + j for j, p in enumerate(pnames)}
        n_player = len(pnames)
        pr, pc, pv = [], [], []
        for n, i in enumerate(idx):
            for p in blue_p[i]:
                pr.append(n)
                pc.append(pcol[p])
                pv.append(0.2)
            for p in red_p[i]:
                pr.append(n)
                pc.append(pcol[p])
                pv.append(-0.2)
        rows_i.append(np.asarray(pr, dtype=np.int64))
        cols_i.append(np.asarray(pc, dtype=np.int64))
        vals.append(np.asarray(pv, dtype=np.float64))

    total = n_team + n_player
    # Per-column penalty so both blocks shrink by *pseudo-games*, not by raw
    # lambda. For a column of per-game weight v the Hessian is 0.25 * v^2, so
    # the shrinkage factor is 0.25 v^2 n / (0.25 v^2 n + lambda); "k games of
    # shrinkage" is therefore lambda = 0.25 * v^2 * k. v = 1 for a team (k/4)
    # and 0.2 for a player (k/100) - getting this backwards shrinks the player
    # block by 6,250 pseudo-games and makes it a silent no-op.
    lam = np.full(total, max(k, 1e-6) / 4.0)
    if n_player:
        lam[n_team:] = 0.25 * 0.2 ** 2 * max(kp, 1e-6)
    rating, side_h = _fit_ratings(rows_i, cols_i, vals, total, lam, y, idx,
                                  weight)

    # Back out a per-side strength for every row: the team term plus the mean
    # of that side's five player ratings. This is the quantity the head sees,
    # and it equals the model's own linear predictor minus the intercept. A
    # team (or player) with no train game is the *average* one, 0 logits - in
    # Bradley-Terry the zero point is the league average, and the head's
    # side_bias carries the intercept that the rate version bakes in.
    sb = np.zeros(len(tb))
    sr = np.zeros(len(tb))
    if n_team:
        sb = sb + np.array([rating[tcol[t]] if t in tcol else 0.0 for t in tb])
        sr = sr + np.array([rating[tcol[t]] if t in tcol else 0.0 for t in tr])
    if n_player:
        def _mean(row):
            cols = [pcol[p] for p in row if p in pcol]
            return float(np.mean(rating[cols])) if cols else 0.0

        sb = sb + np.array([_mean(blue_p[i]) for i in range(len(tb))])
        sr = sr + np.array([_mean(red_p[i]) for i in range(len(tb))])
    if verbose:
        print("  bt: %d teams, %d players, %d train games, k %.1f, h %+.3f"
              % (n_team, n_player, idx.size, k, side_h))
    return np.stack([sb, sr], axis=1).astype(np.float32)


def _champ_cells(table, by_role=False):
    """[N, 10] rating-column index per slot, champion-major (5c + r).

    Champion ids are shared by both domains: `data/pro play/pipeline.py` builds
    its name -> idx map *from* the soloQ `champ_mapping.json`, so a champion has
    the same index in a soloQ row and a pro row. That is what makes a rating
    fitted on one domain applicable to the other.
    """
    champ = np.asarray(table.champ)
    if not by_role:
        return champ
    return 5 * champ + np.asarray(table.role)


def _champ_cells_size(table, by_role=False):
    # +1 because a table may carry the UNK champion index (173), which is a real
    # cell with its own rating rather than an error: it is a handful of rows and
    # dropping them would silently change the split's row count.
    champ = np.asarray(table.champ)
    n_champs = max(NUM_REAL_CHAMPS, int(champ.max()) + 1)
    return n_champs * 5 if by_role else n_champs


def apply_champion_ratings(table, rating, by_role=False):
    """Per-side mean of a champion rating vector -> [N, 2] float32.

    Split out of `champion_strength` so a vector fitted on one domain can be
    applied to another table's rows - the cross-domain test in README 16.
    """
    cells = _champ_cells(table, by_role)
    rating = np.asarray(rating, dtype=np.float64)
    sb = rating[cells[:, :TEAM_SIZE]].mean(axis=1)
    sr = rating[cells[:, TEAM_SIZE:]].mean(axis=1)
    return np.stack([sb, sr], axis=1).astype(np.float32)


def champion_strength(table, train_idx, k=20.0, by_role=False, verbose=False,
                      return_ratings=False):
    """Per-side champion strength logits, fitted on **train rows only**.

    The soloQ analogue of `team_strength`, and the one strength feature that
    exists in both domains: there is no team to rate in solo queue and the
    players are thousands of strangers, but a champion is a champion in both.

    Model: `logit P(blue) = mean(r_5 blue) - mean(r_5 red) + h`, i.e. the same
    Bradley-Terry block as `_bt_ratings`' player block (each of the five picks
    contributes 1/5, so the side term is the mean), fitted on the champion ids
    the table already carries - no scan, and no dependence on the CSV pipelines.

    `by_role=True` indexes `champion x role` (865 cells) *instead of* the
    champion block. Not in addition to it: the two are nested, so a joint fit
    would be identified only by its prior and would land wherever LBFGS stopped
    - the same failure as `--team-mode roster` (README 13). A champion's value
    does depend on its role, so this is the more expressive of the two, at the
    cost of ~350 games per cell instead of ~3,500.

    `k` is pseudo-games of shrinkage: `lambda = 0.25 * (1/5)^2 * k`. With this
    much data it is nearly inert (3,500 games against k = 20), which is the
    point - the estimate, not the prior, is doing the work here.
    """
    cells = _champ_cells(table, by_role)
    n_cols = _champ_cells_size(table, by_role)

    idx = np.asarray(train_idx, dtype=np.int64)
    rows_i = [np.repeat(np.arange(idx.size), N_SLOTS)]
    cols_i = [cells[idx].ravel().astype(np.int64)]
    vals = [np.tile([0.2] * TEAM_SIZE + [-0.2] * TEAM_SIZE, idx.size)]

    lam = np.full(n_cols, 0.25 * 0.2 ** 2 * max(k, 1e-6))
    rating, side_h = _fit_ratings(rows_i, cols_i, vals, n_cols, lam,
                                  table.y.astype(np.float64), idx,
                                  np.ones(idx.size))
    if verbose:
        print("  champ: %d of %d cells seen in train, %d train games, "
              "k %.1f, h %+.3f"
              % (np.unique(cells[idx]).size, n_cols, idx.size, k, side_h))
    feat = apply_champion_ratings(table, rating, by_role=by_role)
    return (feat, rating) if return_ratings else feat


def team_strength(table, train_idx, k=10.0, corpus="2022", verbose=False,
                  mode="rate", half_life_days=0.0, kp=None):
    """Per-row (blue, red) team-strength logits, fitted on **train rows only**.

    Team identity is not a draft feature, so it lives outside the model's normal
    input path and outside the draft-only line entirely: `team_check.py` shows
    it is worth ~0.044 nats on pro against the draft's ~0.005, i.e. an order of
    magnitude more. That is a fact about pro League, not about drafts, and
    folding it into the draft model without saying so would turn "this draft is
    good" into "this team is T1".

    Implementation notes that matter for fairness:

    * **Fitted on train rows only.** A team's rate never sees a val or test row,
      and the split is series-grouped, so a team's test games are not in its own
      history. `prior` is the train base rate, so a team with no train games
      falls back to the null rather than to 0.5.
    * **Shrunk toward that base rate** with weight `k` games (10 ≈ half a
      split): a 2-0 team is not a 100 % winner. Without this the feature
      overfits the long tail of teams with a handful of games.
    * The rate counts a team's wins from *either* side, so the label is
      converted per row (`y` for blue, `1-y` for red).
    * Returns **logits**, not probabilities: the head consumes them linearly and
      a logit keeps a 90 % team from dominating a 55 % one by 8x in raw scale.

    Two knobs beyond `k`:

    * `mode="bt"` swaps the marginal rate for **Bradley-Terry** ratings
      (`_bt_ratings`), which condition on the opponent. The rate is the null
      this has to beat, not the best available estimator. `mode="roster"` adds
      a player block to the same fit, so "T1 is good" is separated from "T1
      still has Faker" - a team tag is a roster that changes.
    * `half_life_days` down-weights old games (0 = off). Rosters change, so a
      team's 2022 record is weak evidence about its 2026 form; the weight is
      relative to `CORPUS_MAX_DATE` so it does not move when the split does.
    """
    tb, tr = _team_pairs(table, corpus, verbose)
    y = table.y.astype(np.float64)
    idx = np.asarray(train_idx, dtype=np.int64)
    base = float(y[idx].mean())

    # Recency weight per *row*; 1.0 everywhere when the knob is off.
    if half_life_days and half_life_days > 0:
        age = _age_days(table.date)[idx]
        weight = np.power(0.5, age / float(half_life_days))
    else:
        weight = np.ones(idx.size, dtype=np.float64)

    if mode == "bt":
        return _bt_ratings(tb, tr, y, idx, k, weight, verbose=verbose)

    if mode == "roster":
        roster = _roster_columns(table, corpus, verbose=verbose)
        return _bt_ratings(tb, tr, y, idx, k, weight, verbose=verbose,
                           roster=roster, kp=kp)

    if mode == "player":
        # Player ratings *instead of* team ratings. The two blocks are close to
        # collinear (a roster is what a team tag is), so a joint fit is only
        # identified by its prior and lands wherever LBFGS happens to stop;
        # this is the version of the question that has one answer.
        roster = _roster_columns(table, corpus, verbose=verbose)
        return _bt_ratings(tb, tr, y, idx, k, weight, verbose=verbose,
                           roster=roster, kp=kp, use_team=False)

    wins, games = Counter(), Counter()
    for n, i in enumerate(idx):
        b, r = tb[i], tr[i]
        w = weight[n]
        games[b] += w
        wins[b] += w * y[i]
        games[r] += w
        wins[r] += w * (1.0 - y[i])
    rate = {t: (wins[t] + k * base) / (games[t] + k) for t in games}

    def _logit(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    sb = np.array([rate.get(t, base) for t in tb])
    sr = np.array([rate.get(t, base) for t in tr])
    return np.stack([_logit(sb), _logit(sr)], axis=1).astype(np.float32)


# ------------------------------------------------------------- soloq loader
def _soloq_cache(patch):
    return SOLOQ_DIR / ("drafts_%s.pt" % patch)


def load_soloq(patch="16.20", cache_path=None, patch_vocab=None, verbose=True):
    """Completed soloQ drafts -> (DraftTable, meta).

    Reads the cache `model/pretraining` already builds (`drafts_<patch>.pt`,
    schema 2, with `wins`). A cache without win labels is rejected rather than
    rebuilt silently: this experiment is *only* the label, so a stale cache is
    a hard error, not a warning.

    The rows are `(champ[10], role[10])` tuples - one completed draft each, no
    order and no bans, which is exactly the input this model wants.

    `patch_vocab` lets a joint run index this table with the *pro* vocabulary
    (16.20 is the newest pro patch too, so it has a row there). SoloQ alone
    gets the one-entry vocabulary `["16.20"]`, which means the patch embedding
    is a constant for every row - honest, and the reason a soloQ-only model
    cannot be said to use the patch as a feature.
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
    vocab = list(patch_vocab) if patch_vocab is not None else [patch_key]
    if patch_key not in vocab:
        raise ValueError("patch %r is not in the supplied vocabulary %s"
                         % (patch_key, vocab[:5]))
    pid = vocab.index(patch_key) + 1
    table = DraftTable(
        np.asarray(champ, dtype=np.int16), np.asarray(role, dtype=np.int8),
        np.asarray(y, dtype=np.int8),
        np.full(len(y), pid, dtype=np.int16),    # every row is the same patch
        gid, gid, [patch_key] * len(y), [""] * len(y), "soloq")
    meta = {"corpus": "soloq", "patch": patch_key,
            "patch_vocab": vocab, "n_patches": len(vocab) + 1,
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


def load_both(corpus="2022", patch="16.20", verbose=True):
    """Pro + soloQ in one table, pro first.

    The patch vocabulary is the pro one, and soloQ's 16.20 rows are indexed
    with it, so the two halves live in one embedding space and a joint model
    sees a real patch id for both. `--data both` therefore differs from the
    single-dataset arms in exactly one way - the rows it trains on - which is
    the point of the comparison.

    The two halves are NOT re-weighted: the joint corpus is 46,879 pro +
    384,356 soloQ drafts, i.e. 89% soloQ. That is deliberate for the first
    sweep (it measures "add pro to soloQ" as a data addition, not as a
    curriculum), and the per-domain test blocks are what show whether the
    minority half was learned at all.
    """
    pro, pro_meta = load_pro(corpus=corpus, verbose=verbose)
    sq, sq_meta = load_soloq(patch=patch,
                             patch_vocab=pro_meta["patch_vocab"],
                             verbose=verbose)
    table = DraftTable.concat([pro, sq], source="both")
    meta = {"corpus": "%s+soloq" % corpus,
            "patch_vocab": pro_meta["patch_vocab"],
            "n_patches": pro_meta["n_patches"],
            "corpus_max_date": pro_meta.get("corpus_max_date", ""),
            "newest_row_date": pro_meta.get("newest_row_date", ""),
            "dropped": {"pro": pro_meta.get("dropped", {}),
                        "soloq": sq_meta.get("dropped", {})},
            "parts": {"pro": len(pro), "soloq": len(sq)},
            "patch": sq_meta["patch"]}
    if verbose:
        print("joint corpus: %d pro + %d soloQ = %d drafts (%.1f%% soloQ)"
              % (len(pro), len(sq), len(table), 100.0 * len(sq) / len(table)))
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
    unit = {"pro": "series", "soloq": "game"}.get(table.source,
                                                  "series (pro) / game (soloq)")
    by_domain = {}
    if len(table.domains()) > 1:
        for dom in table.domains():
            m = table.domain == dom
            by_domain[dom] = {
                "n": int(m.sum()),
                **{name: int(m[splits[name]].sum())
                   for name in ("train", "val", "test")}}
    return {
        "seed": seed,
        "fracs": list(fracs),
        "unit": unit,
        "counts": counts,
        "shares": {k: v / n for k, v in counts.items()},
        "min_patch_rows": min_patch_rows,
        "n_series_straddling": straddle,
        "by_domain": by_domain,
        "patches_all_train": [p for p, v in per_patch.items()
                              if v["held_out"]],
        "by_patch": per_patch,
    }


def make_splits(table, seed=SPLIT_SEED, fracs=SPLIT_FRACS,
                min_patch_rows=MIN_PATCH_ROWS, verbose=True):
    """Dispatch to the right splitter and attach the report.

    A joint table is split **per domain**, each by its own rule, so the soloQ
    test rows of `--data both` are literally the same rows as the soloQ-only
    run's test set (and likewise for pro). Every cross-arm comparison in this
    experiment depends on that: two arms measured on two different test sets
    would not be a comparison.
    """
    if table.source == "both":
        out = {"train": [], "val": [], "test": []}
        for dom in table.domains():
            idxs = np.flatnonzero(table.domain == dom)
            sub = table.subset(idxs)
            s = (split_soloq(len(sub), seed, fracs) if dom == "soloq"
                 else split_pro(sub, seed, fracs, min_patch_rows))
            for k in out:
                out[k].extend(idxs[s[k]].tolist())
        splits = {k: np.asarray(sorted(v), dtype=np.int64)
                  for k, v in out.items()}
    elif table.source == "pro":
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
    """One completed draft per example. No masking, no prefixes, no bans.

    `team_feat` is an optional `[N,2]` array of (blue, red) team-strength
    logits, already computed from **train rows only** by `team_strength`.
    Optional because the draft-only line must stay exactly as it was: an
    experiment that adds team identity answers a different question, and the two
    must not be able to blur into each other.
    """

    def __init__(self, table, idx=None, team_feat=None):
        self.champ = table.champ if idx is None else table.champ[idx]
        self.role = table.role if idx is None else table.role[idx]
        self.y = table.y if idx is None else table.y[idx]
        self.patch_id = (table.patch_id if idx is None
                         else table.patch_id[idx])
        self.game_id = (table.game_id if idx is None
                        else [table.game_id[i] for i in idx])
        self.patch = table.patch if idx is None else \
            [table.patch[i] for i in idx]
        self.team_feat = None if team_feat is None else np.asarray(
            team_feat if idx is None else team_feat[idx], dtype=np.float32)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i):
        out = {
            "champ_ids": torch.as_tensor(self.champ[i], dtype=torch.long),
            "role_ids": torch.as_tensor(self.role[i], dtype=torch.long),
            "patch_id": torch.tensor(int(self.patch_id[i]), dtype=torch.long),
            "win_label": torch.tensor(float(self.y[i]), dtype=torch.float32),
            "meta": {"game_id": self.game_id[i], "patch": self.patch[i]},
        }
        if self.team_feat is not None:
            out["team_feat"] = torch.as_tensor(self.team_feat[i],
                                               dtype=torch.float32)
        return out


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
PRED_KEYS = ("y", "p", "game_id", "series_id", "patch", "date", "domain")


def save_predictions(path, y, p, table, idx):
    """Write a prediction set: label, probability, and the rows it belongs to.

    The ids are the contract. Two prediction files can only be compared if they
    were scored on the same games, and storing the ids is what lets
    `evaluate.py` prove that instead of assuming it. `domain` is stored too so
    a joint run's predictions can be broken down into soloQ and pro after the
    fact, without re-running anything.
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
        date=np.asarray([table.date[i] for i in idx], dtype=str),
        domain=np.asarray(table.domain[idx], dtype=str))
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
