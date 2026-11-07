"""Datasets for the LLM experiment.

Deliberately a thin wrapper over `data/pro play/datasets.py`: the prefix
encoding, the PAD convention, the legality masks and the fearlessness pool are
the encoder's contract, and re-deriving them here would let the text prompt and
the soft tokens disagree about what the draft is.

Train uses `prefix_mode='random'` (one prefix per game per epoch, redrawn) and
`swap_aug` - the same recipe the encoder's own policy stage used. Eval uses
`prefix_mode='all'` so every move index is scored, which is the shape the
recorded 95,160-state baseline is measured on.

`--val-games` subsamples eval games. The full val split is 4,758 games /
95,160 states; scoring that costs ~12k forward steps, which is more than the
whole warmup. A 200-game subset is 4,000 states, and the report says so.
"""
import torch

from pro.model.llm.common import DS, N_MOVES, patch_vocab


def make_patch_index(rows):
    keys, index = patch_vocab(rows)
    return keys, index


def train_dataset(rows, indices, patch_index, seed=0):
    return DS.ProNextMoveDataset(
        rows, indices=indices, patch_index=patch_index,
        prefix_mode="random", swap_aug=True, seed=seed)


def eval_dataset(rows, indices, patch_index, seed=0, val_games=0):
    """All 20 prefixes per game, optionally over a seeded subset of games."""
    if val_games and val_games < len(indices):
        g = torch.Generator().manual_seed(seed + 991)
        pick = torch.randperm(len(indices), generator=g)[:val_games].tolist()
        indices = [indices[i] for i in sorted(pick)]
    return DS.ProNextMoveDataset(
        rows, indices=indices, patch_index=patch_index,
        prefix_mode="all", swap_aug=False, seed=seed)


def collate(samples):
    """-> (stacked encoder tensors, the raw sample dicts).

    The sample list is kept because the prompt is built from it: the text and
    the tensors have to come from the same object or a side-swapped example
    would be described backwards.
    """
    batch = {k: torch.stack([s[k] for s in samples])
             for k in samples[0] if k != "meta"}
    return batch, samples


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def answer_names(samples, names):
    return [names[int(s["target_champ"])] for s in samples]


def n_moves():
    return N_MOVES
