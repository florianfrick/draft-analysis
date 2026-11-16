"""Flat-feature encodings for the non-neural baselines.

The transformer families see a *set* - ten (champion, role) tokens, a ban
block, a fearless pool - through a permutation-invariant encoder. The baselines
in `pro/model/mlp` and `pro/model/forest` answer the deliberately weaker
question: what can you do with the same board flattened into one fixed-length
vector and no attention at all?

That is the only reason this module exists. It is shared rather than duplicated
so the MLP and the forest are given the *same* board and can only differ in how
they model it; two copies of a feature builder is how two baselines quietly
stop being comparable.

Four layouts
------------
Two tasks x two model families. The tasks differ because the questions differ
(next move sees bans and a partial board; masked completion sees a complete
board with one slot hidden and no bans at all - that is what `score_masked`
hands the encoder). The families differ because a 173-way softmax wants a
one-hot and a tree wants a candidate as a column:

    LAYOUTS["next_move"]["mlp"]     3834 dims, one-hot board + masks + scalars
    LAYOUTS["next_move"]["forest"]   549 dims, ids + masks + candidate one-hot
    LAYOUTS["masked"]["mlp"]        1974 dims
    LAYOUTS["masked"]["forest"]      376 dims

The forest layouts put the candidate in the vector, because a tree scores one
candidate at a time (see `pro/model/forest/README.md` for why it cannot be a
173-class classifier at this scale). The MLP layouts do not: it emits all 173
logits in one forward pass, so the candidate axis is its output layer.

Legality
--------
`legal_mask` reproduces `PolicyValueModel._apply_legality` exactly: banned and
picked champions are always illegal, and the fearless pool is illegal *on
picks only*. The baselines rank over the legal candidates so that their top-5
is measured against the same candidate set the transformer's `-inf` masking
produces - `test_suite.py::test_legality_matches_the_policy_model` asserts that
on real boards rather than trusting the reading.

The masked task applies no legality filter at all, because `score_masked` does
not: the encoder ranks over all 173 champions and so do the baselines.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # .../pro/model
ROOT = HERE.parents[1]                            # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"
ROLES_JSON = ROOT / "data" / "champ_roles.json"

NUM_CHAMPS = 173
N_MOVES = 20
N_SLOTS = 10
N_BANS = 10
TEAM_SIZE = 5
N_ROLES = 5
ROLE_FLEX = 5
FEARLESS_SLOTS = 40
MOVE_BAN, MOVE_PICK = 0, 1
# 6 bans, 6 picks, 4 bans, 4 picks (mirrors policy/infer.PHASE_OF)
PHASE_OF = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4
PHASE_NAMES = {0: "ban1", 1: "pick1", 2: "ban2", 3: "pick2"}

# The one-hot index for "nothing here": 173 champions occupy 0..172.
EMPTY = NUM_CHAMPS

# Bumped when a layout changes. `flat.json` records it and the suite refuses to
# score an artifact whose layout it does not know, because a silent feature
# shift would still produce a number - just not this number.
FEATURE_VERSION = 1

# The baselines' whole point is to be small enough that "the transformer wins"
# cannot be explained by it having more capacity. 3 M is the budget both
# trainers enforce before they write an artifact; the shipped policy model is
# 3.35 M, so anything at or under this is genuinely a smaller model.
PARAM_BUDGET = 3_000_000


def check_budget(name, params, unit="parameters"):
    """Refuse to ship a baseline that outgrew its budget."""
    if params > PARAM_BUDGET:
        raise SystemExit(
            "%s has %s %s, over the %s budget - lower the capacity "
            "(fewer trees / smaller --hidden) or the comparison stops being "
            "about architecture"
            % (name, format(int(params), ","), unit,
               format(PARAM_BUDGET, ",")))
    return int(params)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# `patch_key` normalises '16.3' and '16.03' to the same patch. Imported rather
# than reimplemented: two spellings of one patch must not become two features,
# and the split module makes the same argument for the same reason.
_P = _load("pro_pipeline", PRO_DIR / "pipeline.py")
patch_key = _P.patch_key


def _layout(*blocks):
    off, out = 0, {}
    for name, size in blocks:
        out[name] = (off, off + size)
        off += size
    return out, off


NM, NM_DIM = _layout(                      # next move, MLP
    ("slot", N_SLOTS * (NUM_CHAMPS + 1)),
    ("ban", N_BANS * (NUM_CHAMPS + 1)),
    ("taken", NUM_CHAMPS),
    ("unavail", NUM_CHAMPS),
    ("scalar", 8),
)
MK, MK_DIM = _layout(                      # masked, MLP
    ("slot", N_SLOTS * (NUM_CHAMPS + 1)),
    ("role", N_SLOTS * (N_ROLES + 1)),
    ("taken", NUM_CHAMPS),
    ("scalar", 1),
)
FN, FN_DIM = _layout(                      # next move, forest
    ("slot_id", N_SLOTS),
    ("ban_id", N_BANS),
    ("taken", NUM_CHAMPS),
    ("unavail", NUM_CHAMPS),
    ("cand", NUM_CHAMPS),
    ("scalar", 9),
)
FM, FM_DIM = _layout(                      # masked, forest
    ("slot_id", N_SLOTS),
    ("role_id", N_SLOTS),
    ("taken", NUM_CHAMPS),
    ("cand", NUM_CHAMPS),
    ("scalar", 3),
)

LAYOUTS = {
    "next_move": {"mlp": NM_DIM, "forest": FN_DIM},
    "masked": {"mlp": MK_DIM, "forest": FM_DIM},
}


def dim(task, family):
    return LAYOUTS[task][family]


# ------------------------------------------------------------------- the board
def read_state(row, move_idx):
    """Board after `moves[:move_idx]` -> (slot_champ, slot_role, bans, taken, pool).

    Slot champion is `EMPTY` where nothing has been picked. Bans are kept in
    chronological order because their order carries the patch's priority list -
    the same reason `build_prefix` pads `ban_ids` rather than sorting it.
    """
    champ = [EMPTY] * N_SLOTS
    roles = [ROLE_FLEX] * N_SLOTS
    bans = [EMPTY] * N_BANS
    nb = 0
    for kind, team, c, r in row["moves"][:move_idx]:
        if kind == MOVE_BAN:
            bans[nb] = int(c)
            nb += 1
        else:
            slot = int(team) * TEAM_SIZE + int(r)
            champ[slot] = int(c)
            roles[slot] = int(r)
    taken = np.zeros(NUM_CHAMPS, dtype=bool)
    for c in bans[:nb]:
        taken[c] = True
    for c in champ:
        if c != EMPTY:
            taken[c] = True
    pool = np.zeros(NUM_CHAMPS, dtype=bool)
    for c in row.get("unavailable") or ():
        pool[int(c)] = True
    return champ, roles, bans[:nb], taken, pool


def read_masked_state(row, slot):
    """The nine visible picks with `slot` hidden -> (slot_champ, slot_role, taken).

    The hidden slot is `EMPTY` in the champion axis, which is the flat analogue
    of `CHAMP_MASK_ID`. Roles stay complete: the encoder is given the role of
    the slot it must fill, and so is this.
    """
    champ = [int(c) for c in row["champ"]]
    roles = [int(r) for r in row["role"]]
    taken = np.zeros(NUM_CHAMPS, dtype=bool)
    for i, c in enumerate(champ):
        if i != slot:
            taken[c] = True
    champ[slot] = EMPTY
    return champ, roles, taken


def legal_mask(taken, pool, is_ban):
    """True = the champion may be named here.

    Identical to `PolicyValueModel._apply_legality`: the fearless pool is
    illegal on picks only, because a champion picked in an earlier game of the
    series is still legal to ban in this one.
    """
    illegal = taken.copy()
    if not is_ban:
        illegal |= pool
    return ~illegal


def next_move_target(row, move_idx):
    kind, _team, champ, _role = row["moves"][move_idx]
    return int(champ), kind == MOVE_BAN


def illegal_batch(states):
    """bool[B, C] for next-move states: True = not a legal answer here.

    Separate from `build_next_move` because training needs the mask for every
    batch and the feature matrix is the expensive half; rebuilding 3,834 floats
    to recover 173 booleans would double the cost of the loop.
    """
    out = np.zeros((len(states), NUM_CHAMPS), dtype=bool)
    for i, (row, m) in enumerate(states):
        _c, _r, _b, taken, pool = read_state(row, m)
        out[i] = ~legal_mask(taken, pool, row["moves"][m][0] == MOVE_BAN)
    return out


def masked_target(row, slot):
    return int(row["champ"][slot])


def slot_moves(row):
    """slot -> move index, for the masked task's display and pairing."""
    out = {}
    for m, (kind, team, _c, r) in enumerate(row["moves"]):
        if kind == MOVE_PICK:
            out[int(team) * TEAM_SIZE + int(r)] = m
    return out


def patch_id_of(row, patch_index):
    if not patch_index:
        return 0
    return patch_index.get(patch_key(row.get("patch")), 0)


def n_patches_of(patch_index):
    return max(1, max(patch_index.values()) if patch_index else 1)


def role_prior(path=ROLES_JSON):
    """champion -> its modal role, from the static lane counts.

    A constant table rather than something recomputed from rows, so the
    trainer and the scorer cannot disagree about it. `ROLE_FLEX` for a
    champion with no recorded picks.
    """
    out = np.full(NUM_CHAMPS, ROLE_FLEX, dtype=np.int64)
    try:
        counts = json.loads(path.read_text(encoding="utf-8")).get("counts") or {}
    except (OSError, ValueError):
        return out
    for cid, c in counts.items():
        i = int(cid)
        if 0 <= i < NUM_CHAMPS and c:
            out[i] = int(np.argmax(c))
    return out


# --------------------------------------------------------------- MLP builders
def build_next_move(states, patch_index, n_patches=None, out=None):
    """[(row, move_idx)] -> float32[B, NM_DIM].

    Built per batch rather than materialised: 794k states x 3834 float32 is
    12 GB, and the whole training set does not need to exist at once. Legality
    is not in the vector - it is the answer set, not a feature, and
    `illegal_batch` produces it from the same board read.
    """
    np_ = n_patches or n_patches_of(patch_index)
    B = len(states)
    X = np.zeros((B, NM_DIM), dtype=np.float32) if out is None else out
    s0, _ = NM["slot"]
    b0, _ = NM["ban"]
    t0, _ = NM["taken"]
    u0, _ = NM["unavail"]
    c0, _ = NM["scalar"]
    w = NUM_CHAMPS + 1
    for i, (row, m) in enumerate(states):
        champ, _roles, bans, taken, pool = read_state(row, m)
        for k in range(N_SLOTS):
            X[i, s0 + k * w + champ[k]] = 1.0
        for k, c in enumerate(bans):
            X[i, b0 + k * w + c] = 1.0
        X[i, t0:t0 + NUM_CHAMPS] = taken
        X[i, u0:u0 + NUM_CHAMPS] = pool
        kind = row["moves"][m][0]
        X[i, c0 + 0] = m / (N_MOVES - 1)
        X[i, c0 + 1] = PHASE_OF[m] / 3.0
        X[i, c0 + 2] = 1.0 if kind == MOVE_BAN else 0.0
        X[i, c0 + 3] = row["moves"][m][1]
        X[i, c0 + 4] = row["first_pick"]
        X[i, c0 + 5] = len(bans) / N_BANS
        X[i, c0 + 6] = min(len(row.get("unavailable") or ()), FEARLESS_SLOTS) / FEARLESS_SLOTS
        X[i, c0 + 7] = patch_id_of(row, patch_index) / np_
    return X


def build_masked(states, patch_index, n_patches=None, out=None):
    """[(row, slot)] -> float32[B, MK_DIM]."""
    np_ = n_patches or n_patches_of(patch_index)
    B = len(states)
    X = np.zeros((B, MK_DIM), dtype=np.float32) if out is None else out
    s0, _ = MK["slot"]
    r0, _ = MK["role"]
    t0, _ = MK["taken"]
    c0, _ = MK["scalar"]
    w = NUM_CHAMPS + 1
    for i, (row, slot) in enumerate(states):
        champ, roles, taken = read_masked_state(row, slot)
        for k in range(N_SLOTS):
            X[i, s0 + k * w + champ[k]] = 1.0
            X[i, r0 + k * (N_ROLES + 1) + roles[k]] = 1.0
        X[i, t0:t0 + NUM_CHAMPS] = taken
        X[i, c0] = patch_id_of(row, patch_index) / np_
    return X


# ------------------------------------------------------------ forest builders
# One row per (state, candidate), not one row per state: a forest cannot emit
# 173 scores from one row, so the candidate has to be an input. `candidate_lists`
# is parallel to `states` and the output is the concatenation, which lets the
# board be read once per state instead of once per candidate - 173x fewer
# `read_state` calls, and that loop is the whole cost of scoring.
def build_forest_next_move(states, candidate_lists, patch_index,
                           n_patches=None, prior=None):
    """[(row, move_idx)] + per-state candidate lists -> float32[N, FN_DIM].

    Dense and compact on purpose: a tree splits on a threshold, so an integer
    column is a usable feature and a 174-wide one-hot mostly is not. The
    candidate still gets its own one-hot because "is the candidate this
    champion" is the single strongest split a tree can make here.
    """
    np_ = n_patches or n_patches_of(patch_index)
    prior = role_prior() if prior is None else prior
    total = sum(len(c) for c in candidate_lists)
    X = np.zeros((total, FN_DIM), dtype=np.float32)
    a, _ = FN["slot_id"]
    b, _ = FN["ban_id"]
    t0, _ = FN["taken"]
    u0, _ = FN["unavail"]
    k0, _ = FN["cand"]
    c0, _ = FN["scalar"]
    pos = 0
    for (row, m), cands in zip(states, candidate_lists):
        champ, _roles, bans, taken, pool = read_state(row, m)
        kind = row["moves"][m][0]
        n = len(cands)
        rows = np.arange(pos, pos + n)
        X[rows, a:a + N_SLOTS] = champ
        for k, c in enumerate(bans):
            X[rows, b + k] = c
        X[rows, t0:t0 + NUM_CHAMPS] = taken
        X[rows, u0:u0 + NUM_CHAMPS] = pool
        cands = np.asarray(cands, dtype=np.int64)
        X[rows, k0 + cands] = 1.0
        X[rows, c0 + 0] = m
        X[rows, c0 + 1] = PHASE_OF[m]
        X[rows, c0 + 2] = 1.0 if kind == MOVE_BAN else 0.0
        X[rows, c0 + 3] = row["moves"][m][1]
        X[rows, c0 + 4] = row["first_pick"]
        X[rows, c0 + 5] = len(bans)
        X[rows, c0 + 6] = min(len(row.get("unavailable") or ()), FEARLESS_SLOTS)
        X[rows, c0 + 7] = patch_id_of(row, patch_index)
        X[rows, c0 + 8] = prior[cands]
        pos += n
    return X


def build_forest_masked(states, candidate_lists, patch_index, n_patches=None,
                        prior=None):
    """[(row, slot)] + per-state candidate lists -> float32[N, FM_DIM]."""
    prior = role_prior() if prior is None else prior
    total = sum(len(c) for c in candidate_lists)
    X = np.zeros((total, FM_DIM), dtype=np.float32)
    a, _ = FM["slot_id"]
    r0, _ = FM["role_id"]
    t0, _ = FM["taken"]
    k0, _ = FM["cand"]
    c0, _ = FM["scalar"]
    pos = 0
    for (row, slot), cands in zip(states, candidate_lists):
        champ, roles, taken = read_masked_state(row, slot)
        n = len(cands)
        rows = np.arange(pos, pos + n)
        X[rows, a:a + N_SLOTS] = champ
        X[rows, r0:r0 + N_SLOTS] = roles
        X[rows, t0:t0 + NUM_CHAMPS] = taken
        cands = np.asarray(cands, dtype=np.int64)
        X[rows, k0 + cands] = 1.0
        X[rows, c0 + 0] = patch_id_of(row, patch_index)
        X[rows, c0 + 1] = prior[cands]
        X[rows, c0 + 2] = N_SLOTS - 1
        pos += n
    return X


def candidates_for(row, m):
    """The legal candidate set for a next-move state, in ascending id order."""
    _c, _r, _b, taken, pool = read_state(row, m)
    legal = legal_mask(taken, pool, row["moves"][m][0] == MOVE_BAN)
    return np.flatnonzero(legal)


BUILDERS = {
    ("next_move", "mlp"): build_next_move,
    ("masked", "mlp"): build_masked,
    ("next_move", "forest"): build_forest_next_move,
    ("masked", "forest"): build_forest_masked,
}


# ------------------------------------------------------------------ artifacts
# The two artifact files live in one place because they are what makes a
# baseline comparable at all. `registry.py` decides "comparable" by reading
# `protocol.n_train/n_val/n_test` out of `test_report.json`, and it finds the
# model by reading `flat.json`. Two trainers writing that schema independently
# is how one of them ends up silently labelled a training-set number.
def write_flat_artifact(run_dir, *, kind, task, weights, params, meta,
                        val, test, split_counts):
    """Write `flat.json` + `test_report.json` into a checkpoint directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    flat = {"kind": kind, "task": task, "weights": weights,
            "params": int(params), "feature_version": FEATURE_VERSION,
            "features": kind, "in_dim": LAYOUTS[task][kind]}
    flat.update(meta)
    (run_dir / "flat.json").write_text(
        json.dumps(flat, indent=1, ensure_ascii=False), encoding="utf-8")
    protocol = {"corpus": "pro_drafts_full.pt",
                "split": "within-patch (train=head, val=tail); test = a "
                         "fractional tail (no pin, no fixed holdout)",
                "val_scope": "all", "val_split": "tail",
                "kind": kind, "task": task, "params": int(params),
                "n_train": split_counts["train"],
                "n_val": split_counts["val"],
                "n_test": split_counts["test"]}
    report = {"protocol": protocol, "val": val, "test": test}
    (run_dir / "test_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
