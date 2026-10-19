"""Pro draft datasets: next-move (policy) + masked-draft (encoder finetune).

Built on pipeline.py rows (cached pro_drafts.pt). Both emit a per-example
`weight` (recency, from pipeline.assign_weights) so training can skew to
recent meta without touching the val/test distribution.

ProNextMoveDataset: unrolls each 20-move draft into 20 examples
(prefix -> next move). Prefix state uses the SoloQ partial-draft encoding
(unpicked slots = CHAMP_PAD + pad_mask, per model/README.md Sec.3), plus
ban list, turn info (move idx, phase, side to move, first-pick side),
fearless unavailability mask, and legality mask (taken | unavailable)
for -inf suppression at the head. Targets: (is_ban, champ[, role for
picks]). Batches add win_label (blue wins) for the future value head -
deliberately NOT part of the soloQ pretraining contract.

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

# vocab constants: identical to soloq_model.py / model/README.md Sec.2
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


def build_prefix(row, move_idx, num_champs=NUM_REAL_CHAMPS):
    """Encode draft state after moves[:move_idx] -> example dict.

    Slot tensors follow the soloQ partial-draft convention: unpicked
    slots are CHAMP_PAD with role FLEX and pad_mask=True; picked slots
    hold (champ, exact role). Bans so far go in `ban_ids` (PAD-padded),
    fearless champs in `unavailable_mask`, and `taken_mask` unions
    bans+picks+unavailable for -inf legality suppression at the head.
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
    taken = set(ban_ids[:n_bans].tolist())
    taken.update(champ_ids[~pad_mask].tolist())
    taken_mask = torch.zeros(num_champs, dtype=torch.bool)
    taken_mask[sorted(taken)] = True
    unavailable_mask = torch.zeros(num_champs, dtype=torch.bool)
    if row.get("unavailable"):
        unavailable_mask[sorted(row["unavailable"])] = True
    return {
        "champ_ids": champ_ids,          # [10] partial draft, PAD = unpicked
        "role_ids": role_ids,            # [10]
        "pad_mask": pad_mask,            # [10] True = not yet picked
        "ban_ids": ban_ids,              # [10] bans so far, PAD-padded
        "n_bans": torch.tensor(n_bans, dtype=torch.long),
        "taken_mask": taken_mask,        # [C] banned|picked (legality)
        "unavailable_mask": unavailable_mask,  # [C] fearless-picked
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
        "win_label": torch.tensor(row["blue_win"], dtype=torch.long),
        "weight": torch.tensor(float(row.get("weight", 1.0)),
                               dtype=torch.float32),
        "meta": {"gameid": row["gameid"], "series_id": row["series_id"],
                 "league": row["league"], "patch": row["patch"],
                 "date": row["date"], "blue_team": row["blue_team"],
                 "red_team": row["red_team"]},
    }


class ProNextMoveDataset(Dataset):
    """20 examples per draft: (prefix state -> next move). Index mapping is
    deterministic (game_idx * 20 + move_idx) so splits/weights at the row
    level apply cleanly. `indices` restricts to a split."""

    def __init__(self, rows, indices=None, num_champs=NUM_REAL_CHAMPS):
        self.rows = rows if indices is None else [rows[i] for i in indices]
        self.num_champs = num_champs

    def __len__(self):
        return len(self.rows) * N_MOVES

    def __getitem__(self, idx):
        row = self.rows[idx // N_MOVES]
        return build_prefix(row, idx % N_MOVES, self.num_champs)

    def weights(self):
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
    """

    def __init__(self, rows, indices=None, num_champs=NUM_REAL_CHAMPS,
                 mask_prob=0.15, role_keep_prob=0.1, seed=0, epoch=0,
                 econ_mask_prob=0.0):
        self.rows = rows if indices is None else [rows[i] for i in indices]
        self.num_champs = num_champs
        self.mask_prob = mask_prob
        self.role_keep_prob = role_keep_prob
        self.seed = seed
        self.epoch = epoch
        self.econ_mask_prob = econ_mask_prob

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
            # key-padding clear in soloq_model._encode)
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
                self.role_keep_prob, self._generator(idx, 1))
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
            "win_label": torch.tensor(row["blue_win"], dtype=torch.long),
            "econ_target": torch.tensor(
                econ_targets(row), dtype=torch.float32),
            "has_econ": torch.tensor(1.0 if row.get("has_econ") else 0.0,
                                     dtype=torch.float32),
            "weight": torch.tensor(float(row.get("weight", 1.0)),
                                   dtype=torch.float32),
            "meta": {"gameid": row["gameid"], "series_id": row["series_id"],
                     "patch": row["patch"]},
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

