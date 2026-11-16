"""Pro draft datasets: next-move (policy) + masked-draft (encoder finetune).

Built on pipeline.py rows (cached pro_drafts.pt). Both emit a per-example
`weight` (recency, from pipeline.assign_weights) so training can skew to
recent meta without touching the val/test distribution.

ProNextMoveDataset: (prefix -> next move) examples from each 20-move
draft. Prefix state uses the SoloQ partial-draft encoding (unpicked
slots = CHAMP_PAD + pad_mask, per solo_queue/model/README.md Sec.3), plus ban list,
turn info (move idx, phase, side to move, first-pick side), the patch
id, fearless unavailability mask, and legality mask (taken |
unavailable) for -inf suppression at the head. Targets: (is_ban,
champ[, role for picks]). Batches add win_label (blue wins) for the
value head - deliberately NOT part of the soloQ pretraining contract.

Three prefix modes, because the 20 prefixes of one game are
near-duplicates of each other (same draft, same label):
  'all'    -> 20 examples/game, every prefix. Val/test only, so metrics
              stay comparable across runs.
  'random' -> 1 example/game/epoch, move index redrawn by set_epoch().
              Training default: the effective sample size of this task
              is len(rows), not 20*len(rows), so unrolling all 20 per
              epoch mostly buys memorisation.
  'k'      -> `prefix_k` DISTINCT examples/game/epoch, redrawn by
              set_epoch(). The middle of that trade: more of the 20
              depths per epoch than 'random', far less repetition than
              'all'. Costs k x the steps of 'random' at the same epoch
              count, so compare at equal epochs only against itself.

`swap_aug` mirrors blue<->red with p=0.5 (see `swap_sides`): doubles the
effective data and enforces the Siamese symmetry the encoder claims.

ProMaskedDraftDataset: full 10-pick pro drafts through the SAME masking
semantics as soloQ pretraining (apply_masking imported from the soloQ
pipeline - single source of truth), so the pretrained encoder can be
finetuned on pro drafts with zero contract drift. Also carries win_label.
"""
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent


def _load_module_by_path(name, path):
    """Import a module from an explicit path (dir names here contain
    spaces, and sibling `pipeline.py` files exist - never rely on
    sys.path ordering for these)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_module_by_path("pro_pipeline", HERE / "pipeline.py")

_SOLOQ_PIPELINE = P.SOLOQ_PROC_DIR / "pipeline.py"
_SQ = _load_module_by_path("soloq_pipeline", _SOLOQ_PIPELINE)

# vocab constants: identical to base_model.py / solo_queue/model/README.md Sec.2
N_SLOTS = 10
TEAM_SIZE = 5
N_MOVES = P.N_MOVES
MOVE_BAN = P.MOVE_BAN
MOVE_PICK = P.MOVE_PICK
NUM_REAL_CHAMPS = 173
CHAMP_MASK_ID = NUM_REAL_CHAMPS
CHAMP_PAD_ID = NUM_REAL_CHAMPS + 1
ROLE_FLEX_ID = 5
IGNORE_INDEX = -100
NUM_ROLES = 5

# move phases: 0=bans1, 1=picks1, 2=bans2, 3=picks2
PHASE_OF = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4

# Fearless pool width. A series is at most 5 games and a game is 10 picks, so
# the pool entering game 5 holds at most 4*10 = 40 champions - a hard bound,
# not a guess (`pipeline.attach_fearless` walks games 1..N-1 in order). The
# corpus maximum is exactly 40, so nothing is ever truncated; the bound exists
# so the token axis is a constant and batches need no per-batch padding.
FEARLESS_SLOTS = 40


def build_fearless_roles(rows):
    """{gameid: {champ: role}} for the champions an earlier series game picked.

    `row["unavailable"]` is a sorted list of champion ids with no roles
    attached, and the role is the interesting half: "their mid pool is gone"
    is a different fact from "40 champions are gone". The roles are
    recoverable by walking each series in game order and remembering what was
    picked, which is what this does.

    Only rows that actually carry a pool get an entry, so the dict is ~10k
    rows rather than 47k. A champion whose role cannot be recovered (the game
    it came from was dropped) falls back to FLEX rather than being dropped -
    the same convention `build_prefix` uses for bans, which have no role.

    The pool membership itself is *not* recomputed here: `row["unavailable"]`
    stays the single source of truth for which champions are gone, because
    `attach_fearless` is what decides whether a series is fearless at all
    (2025+, no internal re-pick). Recomputing it here would let the two
    disagree.
    """
    by_series = {}
    for r in rows:
        sid = r.get("series_id")
        if sid is not None:
            by_series.setdefault(sid, []).append(r)
    out = {}
    for series_rows in by_series.values():
        series_rows.sort(key=lambda x: (x.get("game_number") or 0,
                                        x.get("gameid") or ""))
        seen = {}
        for r in series_rows:
            if r.get("unavailable"):
                out[r["gameid"]] = {c: seen.get(c, ROLE_FLEX_ID)
                                    for c in r["unavailable"]}
            for c, ro in zip(r.get("champ") or (), r.get("role") or ()):
                seen[int(c)] = int(ro)
    return out


# Which team makes ban slot i, given first_pick. The interleave is fixed
# (`pipeline.interleaved_moves`): bans1 alternates first-pick team first,
# bans2 alternates second-pick team first. So the side of a ban is a pure
# function of (slot, first_pick) - it is *derived* rather than carried in the
# batch, which is why `swap_sides` needs no change: it flips `first_pick`, and
# the derived side flips with it. Storing the side instead would have required
# swapping the ban block too, and forgetting that would have silently mirrored
# every ban onto the wrong team.
BAN_SIDE_BY_SLOT = (0, 1, 0, 1, 0, 1, 1, 0, 1, 0)


def ban_sides(first_pick):
    """[10] long: 0 = blue banned that slot, 1 = red, given first_pick.

    `first_pick` may be a tensor [B]; the result is then [B, 10].
    """
    table = torch.tensor(BAN_SIDE_BY_SLOT, dtype=torch.long)
    if torch.is_tensor(first_pick):
        fp = first_pick.long().unsqueeze(1)
        return table.unsqueeze(0) ^ fp
    return table ^ int(first_pick)


def econ_targets(row, num_champs=NUM_REAL_CHAMPS):
    """Raw (gold10_b, gold10_r, gold15_b, gold15_r, xp10_b, xp10_r).

    Missing rows return zeros (use `has_econ` to mask them out);
    missing interpolates nothing - the standardization stats are
    computed on present values only in train.py.
    """
    vals = row.get("econ") or [None] * 6
    return [float(v) if v is not None else 0.0 for v in vals]


def phase_of(move_idx):
    return PHASE_OF[move_idx]


# ---------------------------------------------------------------- patches
#
# Patch conditioning. `patch` is carried in every row as a string
# ('16.03'), but until now it only ever reached the model inside `meta`,
# i.e. it was never a tensor and StateEmbedding had no way to see it.
# That made the first ban of a game unpredictable by construction: with
# no picks on the board, the only thing that determines which champ gets
# banned is the current patch's priority list (measured: model 12.3%
# top-5 on bans1 vs 37.4% for a lookup table fit on the test patches).
PATCH_UNK_ID = 0  # index 0 reserved for unseen patches


# Normalisation and ordering of patch labels live in pipeline.py, because
# split_within_patch has to group rows by the same notion of "one patch" that
# the vocabulary uses. Re-exported here so existing callers keep working.
patch_key = P.patch_key
_patch_sort_key = P.patch_sort_key


def build_patch_vocab(rows):
    """Patch vocabulary over ALL rows -> (sorted keys, {key: id}).

    Built over train+val+test on purpose: a patch label is metadata, not
    a label, so this is not leakage (same reasoning as the champion
    vocab). Index 0 is PATCH_UNK_ID so a patch never seen in training
    (the next patch, at inference time) falls back to a learned row
    instead of crashing. Pass the SAME dict to every split, otherwise
    train/val/test disagree on patch ids.
    """
    keys = sorted({patch_key(r.get("patch")) for r in rows} - {""},
                  key=_patch_sort_key)
    return keys, {k: i + 1 for i, k in enumerate(keys)}


def patch_to_id(patch, patch_index):
    """Map a row's patch string to its vocab id (UNK if unknown)."""
    if not patch_index:
        return PATCH_UNK_ID
    return patch_index.get(patch_key(patch), PATCH_UNK_ID)


def patch_index_from_vocab(vocab):
    """Rebuild {patch_key: id} from a checkpoint's stored `patch_vocab`.

    Diagnostics/inference must use the vocabulary the model was trained
    with, not one rebuilt from whatever rows are on hand: the patch
    embedding rows are indexed by the training-time vocab, so a
    different ordering silently shifts every patch vector.
    """
    return {k: i + 1 for i, k in enumerate(vocab or [])}


def swap_sides(sample):
    """Mirror an example blue<->red. Returns a new dict.

    Valid augmentation because the encoder is Siamese (one set of weights
    for both teams) and carries no team embedding, so the mirrored draft
    is the same draft viewed from the other side, with the win label
    flipped.

    Swapped: champ_ids / role_ids / pad_mask (the two 5-slot team blocks
    trade places - role index is preserved within a team, so the
    role->slot convention survives), to_move, first_pick, win_label, and
    the (blue, red) pairs inside econ_target.

    NOT swapped, deliberately:
      - ban_ids: the ban *array* still needs no change, but no longer because
        bans are team-agnostic. The chronological ban order is a fixed
        function of first_pick (`interleaved_moves`), so a team swap leaves
        slot i holding the same champion and the same "i-th ban of the game".
        Which *side* owns slot i does flip - and that is handled by
        `ban_sides()`, which derives the side from (slot, first_pick) at read
        time. `first_pick` is swapped above, so the derived sides flip for
        free. Storing the side in the batch instead would have required
        swapping the ban block here too, and a missed swap would mirror every
        ban onto the wrong team with nothing to catch it.
      - fearless_ids / fearless_role_ids / n_fearless: the pool comes from
        earlier games of the series and is role-attributed but not
        side-attributed, so a swap inside this game cannot change it.
      - taken_mask / unavailable_mask: sets of champions, not per-team.
      - target_champ / target_role / is_ban / move_idx / phase: the same
        team is still to move and still makes the same move.
      - patch_id: unchanged by a side swap.
    """
    out = dict(sample)
    half = TEAM_SIZE
    for key in ("champ_ids", "role_ids", "pad_mask"):
        t = sample[key]
        out[key] = torch.cat([t[half:], t[:half]], dim=0)
    out["to_move"] = 1 - sample["to_move"]
    out["first_pick"] = 1 - sample["first_pick"]
    out["win_label"] = 1 - sample["win_label"]
    # econ_target is (gold10_b, gold10_r, gold15_b, gold15_r, xp10_b,
    # xp10_r): flip each (blue, red) pair, otherwise the econ head is
    # taught that this side's gold is the other side's.
    e = sample.get("econ_target")
    if e is not None and e.numel() % 2 == 0:
        out["econ_target"] = e.reshape(-1, 2)[:, [1, 0]].reshape(-1)
    return out


def build_prefix(row, move_idx, num_champs=NUM_REAL_CHAMPS, patch_index=None,
                 fearless_roles=None):
    """Encode draft state after moves[:move_idx] -> example dict.

    Slot tensors follow the soloQ partial-draft convention: unpicked
    slots are CHAMP_PAD with role FLEX and pad_mask=True; picked slots
    hold (champ, exact role). Bans so far go in `ban_ids` (PAD-padded),
    fearless champs in `unavailable_mask`, and `taken_mask` holds
    bans+picks only - *not* the fearless pool, because a champion picked
    in an earlier game is still legal to ban. `PolicyValueModel` combines
    the two itself and masks `unavailable_mask` on pick moves only.

    `patch_index` (from build_patch_vocab) turns the row's patch string
    into `patch_id`; None -> PATCH_UNK_ID.
    """
    champ_ids = torch.full((N_SLOTS,), CHAMP_PAD_ID, dtype=torch.long)
    role_ids = torch.full((N_SLOTS,), ROLE_FLEX_ID, dtype=torch.long)
    pad_mask = torch.ones(N_SLOTS, dtype=torch.bool)
    ban_ids = torch.full((10,), CHAMP_PAD_ID, dtype=torch.long)
    n_bans = 0
    for kind, team, champ, role in row["moves"][:move_idx]:
        if kind == MOVE_BAN:
            ban_ids[n_bans] = champ
            n_bans += 1
        else:
            slot = team * TEAM_SIZE + role  # exact role = slot within team
            champ_ids[slot] = champ
            role_ids[slot] = role
            pad_mask[slot] = False
    kind, team, t_champ, t_role = row["moves"][move_idx]
    # The partner of a double-pick block: the champion the SAME team takes on
    # the very next move. Only defined on the first move of a same-team pick
    # pair, because that is the only place the pair is still open - after
    # move_idx the first champion is taken and masked, so a set metric can
    # never see both. `pro/model/archive/proto` uses it as an auxiliary target.
    partner_champ = IGNORE_INDEX
    has_partner = 0
    if kind == MOVE_PICK and move_idx + 1 < N_MOVES:
        nk, nt, nc, _nr = row["moves"][move_idx + 1]
        if nk == MOVE_PICK and nt == team:
            partner_champ = nc
            has_partner = 1
    taken = set(ban_ids[:n_bans].tolist())
    taken.update(champ_ids[~pad_mask].tolist())
    taken_mask = torch.zeros(num_champs, dtype=torch.bool)
    taken_mask[sorted(taken)] = True
    unavailable_mask = torch.zeros(num_champs, dtype=torch.bool)
    if row.get("unavailable"):
        unavailable_mask[sorted(row["unavailable"])] = True
    # Fearless pool as tokens, not just a mask. `unavailable_mask` above is
    # what makes the pool illegal to suggest; these are what let the model
    # *see* it. Without them the pool is information the draft depends on and
    # the model is never told - 21% of the corpus carries a non-empty pool,
    # mean 15.6 champions, so this is not a corner case.
    fearless_ids = torch.full((FEARLESS_SLOTS,), CHAMP_PAD_ID, dtype=torch.long)
    fearless_role_ids = torch.full((FEARLESS_SLOTS,), ROLE_FLEX_ID,
                                   dtype=torch.long)
    pool = sorted(row["unavailable"]) if row.get("unavailable") else []
    role_of = (fearless_roles or {}).get(row["gameid"], {})
    n_fearless = min(len(pool), FEARLESS_SLOTS)
    for k, champ in enumerate(pool[:FEARLESS_SLOTS]):
        fearless_ids[k] = champ
        fearless_role_ids[k] = int(role_of.get(champ, ROLE_FLEX_ID))
    return {
        "champ_ids": champ_ids,          # [10] partial draft, PAD = unpicked
        "role_ids": role_ids,            # [10]
        "pad_mask": pad_mask,            # [10] True = not yet picked
        "ban_ids": ban_ids,              # [10] bans so far, PAD-padded
        "n_bans": torch.tensor(n_bans, dtype=torch.long),
        "taken_mask": taken_mask,        # [C] banned|picked (legality)
        "unavailable_mask": unavailable_mask,  # [C] fearless-picked
        "fearless_ids": fearless_ids,    # [40] pool so far, PAD-padded
        "fearless_role_ids": fearless_role_ids,   # [40]
        "n_fearless": torch.tensor(n_fearless, dtype=torch.long),
        "move_idx": torch.tensor(move_idx, dtype=torch.long),
        "phase": torch.tensor(phase_of(move_idx), dtype=torch.long),
        "to_move": torch.tensor(team, dtype=torch.long),  # 0=blue 1=red
        "first_pick": torch.tensor(row["first_pick"], dtype=torch.long),
        "is_ban": torch.tensor(1 if kind == MOVE_BAN else 0,
                               dtype=torch.long),
        "target_champ": torch.tensor(t_champ, dtype=torch.long),
        "target_role": torch.tensor(
            t_role if kind == MOVE_PICK else IGNORE_INDEX,
            dtype=torch.long),
        "partner_champ": torch.tensor(partner_champ, dtype=torch.long),
        "has_partner": torch.tensor(has_partner, dtype=torch.long),
        "win_label": torch.tensor(row["blue_win"], dtype=torch.long),
        "weight": torch.tensor(float(row.get("weight", 1.0)),
                               dtype=torch.float32),
        # Patch conditioning: the model had no way to see which patch it
        # is predicting for, which is the whole reason bans1 (no picks on
        # the board, patch-only prior) lost to a lookup table.
        "patch_id": torch.tensor(patch_to_id(row.get("patch"), patch_index),
                                 dtype=torch.long),
        # Econ labels: these were missing from build_prefix entirely, so
        # PolicyValueModel's econ head was trained against nothing (only
        # ProMaskedDraftDataset carried them). Wired here so the head
        # gets real supervision; `has_econ` masks the ~9% of rows without.
        "econ_target": torch.tensor(econ_targets(row), dtype=torch.float32),
        "has_econ": torch.tensor(1.0 if row.get("has_econ") else 0.0,
                                 dtype=torch.float32),
        "meta": {"gameid": row["gameid"], "series_id": row["series_id"],
                 "league": row["league"], "patch": row["patch"],
                 "date": row["date"], "blue_team": row["blue_team"],
                 "red_team": row["red_team"]},
    }


class ProNextMoveDataset(Dataset):
    """(prefix -> next move) examples from pro drafts.

    prefix_mode:
      'all'    -> len(rows) * N_MOVES. Index mapping is deterministic
                  (game_idx * 20 + move_idx) so splits/weights at the
                  row level apply cleanly. Val/test.
      'random' -> len(rows). Each game yields ONE example per epoch at a
                  move index redrawn by set_epoch(). Training.
      'k'      -> len(rows) * prefix_k. Each game yields `prefix_k`
                  DISTINCT examples per epoch (a fresh draw without
                  replacement every set_epoch()). Training.

    Why 'random' is the training default: the 20 prefixes of a game are
    the same draft with the same next-move label at different depths, so
    they are near-duplicates. Unrolling all 20 every epoch inflates the
    apparent dataset to 810k while the effective sample size stays at
    len(rows) ~ 46.6k, which is exactly the regime where val loss starts
    climbing after epoch 1. One prefix per game per epoch keeps the same
    information over E epochs, makes each epoch 20x cheaper, and turns
    the redundancy into augmentation instead of repetition.

    'k' is the measured middle: k distinct depths per game per epoch, so
    a run sees k x more of each draft's phase structure per epoch at k x
    the steps. prefix_k=1 is 'random' and prefix_k=N_MOVES is 'all'; the
    two endpoints are rejected as modes rather than silently aliased, so
    a checkpoint's recorded prefix_k always says what it actually did.

    swap_aug: mirror blue<->red with p=0.5 (see `swap_sides`).
    """

    def __init__(self, rows, indices=None, num_champs=NUM_REAL_CHAMPS,
                 patch_index=None, prefix_mode="all", prefix_k=5,
                 swap_aug=False, seed=0, fearless_roles=None):
        if prefix_mode not in ("all", "random", "k"):
            raise ValueError("prefix_mode must be 'all', 'random' or 'k', "
                             "got %r" % (prefix_mode,))
        if prefix_mode == "k" and not 1 < int(prefix_k) < N_MOVES:
            raise ValueError(
                "prefix_mode='k' needs 1 < prefix_k < %d, got %r "
                "(use 'random' for 1, 'all' for %d)"
                % (N_MOVES, prefix_k, N_MOVES))
        self.rows = rows if indices is None else [rows[i] for i in indices]
        self.num_champs = num_champs
        self.patch_index = patch_index
        self.prefix_mode = prefix_mode
        self.prefix_k = int(prefix_k)
        self.swap_aug = swap_aug
        self.seed = seed
        self.epoch = 0
        self._move_of = None
        self._moves = None
        # Built over ALL rows, not `self.rows`: a val row's fearless pool comes
        # from earlier games of its series, which are train rows. Building it
        # over the split would empty every val row's pool and make the fearless
        # arm look inert for the one reason that is not the model's fault.
        self.fearless_roles = (fearless_roles if fearless_roles is not None
                               else build_fearless_roles(rows))
        # Own stream so augmentation draws don't depend on loader order.
        self._aug_rng = torch.Generator().manual_seed(seed + 4241)
        if prefix_mode in ("random", "k"):
            self.set_epoch(0)

    def set_epoch(self, epoch):
        """Redraw the per-game move index (random / k modes).

        Deterministic in (seed, epoch) so a resumed run reproduces the
        same augmentation schedule.
        """
        self.epoch = epoch
        if self.prefix_mode == "all":
            return
        g = torch.Generator().manual_seed(
            (self.seed * 7919 + epoch * 104729) % (2 ** 63 - 1))
        if self.prefix_mode == "random":
            self._move_of = torch.randint(0, N_MOVES, (len(self.rows),),
                                          generator=g)
            return
        # 'k': k distinct depths per game. argsort of one random row per
        # game is the vectorised "sample without replacement" - a per-game
        # randperm loop would be 40k python-level ops every epoch.
        draws = torch.rand((len(self.rows), N_MOVES), generator=g)
        self._moves = torch.argsort(draws, dim=1)[:, :self.prefix_k]

    def __len__(self):
        if self.prefix_mode == "random":
            return len(self.rows)
        if self.prefix_mode == "k":
            return len(self.rows) * self.prefix_k
        return len(self.rows) * N_MOVES

    def __getitem__(self, idx):
        if self.prefix_mode == "random":
            row = self.rows[idx]
            move_idx = int(self._move_of[idx])
        elif self.prefix_mode == "k":
            row_i = idx // self.prefix_k
            row = self.rows[row_i]
            move_idx = int(self._moves[row_i, idx % self.prefix_k])
        else:
            row = self.rows[idx // N_MOVES]
            move_idx = idx % N_MOVES
        sample = build_prefix(row, move_idx, self.num_champs,
                              self.patch_index, self.fearless_roles)
        if self.swap_aug and \
                torch.rand((), generator=self._aug_rng).item() < 0.5:
            sample = swap_sides(sample)
        return sample

    def weights(self):
        """Recency weight per example, aligned with __len__."""
        if self.prefix_mode == "random":
            return [float(r.get("weight", 1.0)) for r in self.rows]
        if self.prefix_mode == "k":
            return [float(r.get("weight", 1.0)) for r in self.rows
                    for _ in range(self.prefix_k)]
        return [float(r.get("weight", 1.0)) for r in self.rows
                for _ in range(N_MOVES)]



class ProMaskedDraftDataset(Dataset):
    """Full 10-pick pro drafts + soloQ-style masking (finetune path).

    Reuses the soloQ pipeline's apply_masking verbatim (BERT-style 0.15
    champ masks, 90/10 role, >=1 mask/draft) so batches satisfy the
    soloQ contract with pad_mask all-False; adds win_label/weight for
    finetuning.

    Econ mode: sample `econ_prob` drafts as random-subset win targets
    (1..9 visible picks, rest PAD, no champ/role masking/labels), used
    for the win probability head on partial drafts (MCTS value on
    incomplete drafts must see this distribution). Deterministic per
    index+epoch via an independent generator stream. `econ_mask_prob`
    switches the FULL-to-partial share of the dataset.

    `patch_index` (from build_patch_vocab) makes each example carry
    `patch_id` [B], which is what the encoder's patch embedding reads. Left
    as None the id is UNK, so a patch-conditioned encoder sees nothing - the
    correct behaviour for a batch whose patches are not in the vocabulary.
    """

    def __init__(self, rows, indices=None, num_champs=NUM_REAL_CHAMPS,
                 mask_prob=0.15, role_keep_prob=0.1, seed=0, epoch=0,
                 econ_mask_prob=0.0, patch_index=None):
        self.rows = rows if indices is None else [rows[i] for i in indices]
        self.num_champs = num_champs
        self.mask_prob = mask_prob
        self.role_keep_prob = role_keep_prob
        self.seed = seed
        self.epoch = epoch
        self.econ_mask_prob = econ_mask_prob
        # None -> patch_id is UNK for every row, i.e. an encoder with a patch
        # embedding behaves as if it had never seen the patch. Pass the vocab
        # from build_patch_vocab to make the patch a real input.
        self.patch_index = patch_index

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _generator(self, idx, salt):
        return torch.Generator().manual_seed(
            (self.seed + idx * 7919 + self.epoch * 104729 + salt * 2654435761)
            % (2 ** 63 - 1))

    def __getitem__(self, idx):
        row = self.rows[idx]
        champ = torch.as_tensor(row["champ"], dtype=torch.long)
        role = torch.as_tensor(row["role"], dtype=torch.long)
        if self.econ_mask_prob > 0 and (
                torch.rand(1, generator=self._generator(idx, 7)).item()
                < self.econ_mask_prob):
            # random-subset win target: keep 1..9 random picks visible
            # (1..9, not 10: keep one team empty states in-distribution -
            # the encoder handles all-PAD teams without NaN since the
            # key-padding clear in base_model._encode)
            keep_n = int(torch.randint(
                1, N_SLOTS, (1,),
                generator=self._generator(idx, 11)).item())
            keep = torch.randperm(N_SLOTS,
                                  generator=self._generator(idx, 13))[:keep_n]
            cids = torch.full((N_SLOTS,), CHAMP_PAD_ID, dtype=torch.long)
            rids = torch.full((N_SLOTS,), ROLE_FLEX_ID, dtype=torch.long)
            pad = torch.ones(N_SLOTS, dtype=torch.bool)
            cids[keep] = champ[keep]
            rids[keep] = role[keep]
            pad[keep] = False
            rm = is_m = torch.zeros(N_SLOTS, dtype=torch.bool)
            clab = rlab = torch.full((N_SLOTS,), IGNORE_INDEX,
                                     dtype=torch.long)
        elif self.mask_prob > 0:
            cids, rids, rm, is_m, clab, rlab = _SQ.apply_masking(
                champ, role, self.num_champs, self.mask_prob,
                self.role_keep_prob, generator=self._generator(idx, 1))
            pad = torch.zeros(N_SLOTS, dtype=torch.bool)
        else:
            cids, rids = champ.clone(), role.clone()
            rm = is_m = torch.zeros(N_SLOTS, dtype=torch.bool)
            clab = rlab = torch.full((N_SLOTS,), IGNORE_INDEX,
                                     dtype=torch.long)
            pad = torch.zeros(N_SLOTS, dtype=torch.bool)
        return {
            "champ_ids": cids, "role_ids": rids, "role_masked": rm,
            "is_masked": is_m, "champ_labels": clab, "role_labels": rlab,
            "pad_mask": pad,
            "patch_id": torch.tensor(
                patch_to_id(row.get("patch"), self.patch_index),
                dtype=torch.long),
            "win_label": torch.tensor(row["blue_win"], dtype=torch.long),
            "econ_target": torch.tensor(
                econ_targets(row), dtype=torch.float32),
            "has_econ": torch.tensor(1.0 if row.get("has_econ") else 0.0,
                                     dtype=torch.float32),
            "weight": torch.tensor(float(row.get("weight", 1.0)),
                                   dtype=torch.float32),
            "meta": {"gameid": row["gameid"], "series_id": row["series_id"],
                     "league": row.get("league", "?"), "patch": row["patch"]},
        }

    def weights(self):
        return [float(r.get("weight", 1.0)) for r in self.rows]


def pro_collate(samples):
    """Stack tensor keys, list the meta dicts."""
    out = {}
    for key in samples[0]:
        if key == "meta":
            out[key] = [s[key] for s in samples]
        else:
            out[key] = torch.stack([s[key] for s in samples])
    return out


def make_weighted_sampler(dataset, seed=0):
    """WeightedRandomSampler over dataset.weights() (recency-skewed
    training distribution). Deterministic given `seed`."""
    from torch.utils.data import WeightedRandomSampler
    w = torch.tensor(dataset.weights(), dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(w, num_samples=len(dataset),
                                 replacement=True, generator=g)

