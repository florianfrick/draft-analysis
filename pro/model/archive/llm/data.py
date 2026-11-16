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
from torch.utils.data import Dataset

from pro.model.archive.llm.common import DS, N_MOVES, NUM_REAL_CHAMPS, patch_vocab


def make_patch_index(rows, vocab=None):
    """-> (keys, {patch_key: id}) sized to the frozen encoder's own table.

    Defaults to the vocabulary stored in the encoder checkpoint, **not** one
    rebuilt from `rows`. Ids are positions in that table (index 0 is
    `PATCH_UNK_ID`), so a corpus that has moved on to a newer patch maps those
    rows to UNK instead of pushing every id past the table - which is the
    silent-corruption failure `assert_patch_compat` exists to catch. Rebuilding
    from the corpus is only correct when the encoder is about to be trained on
    that corpus; pass `vocab=[]` to force it.
    """
    if vocab is None:
        from . import common as C
        vocab = C.encoder_patch_vocab()
    if vocab:
        return list(vocab), DS.patch_index_from_vocab(vocab)
    return patch_vocab(rows)


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


class PairDataset(Dataset):
    """Explicit (game, move) pairs - the reranker's dataset.

    The reranker can only train on rows whose true answer is in the head's
    candidate list, so neither existing mode fits: `'all'` would spend half its
    steps on rows where every answer is wrong, and `'random'` draws a move
    without knowing whether that move is usable. `pairs` is precomputed once
    from a head pass (`rerank.build_pairs`).

    `per_game=1` is the one-prefix-per-game recipe, with the move redrawn from
    that game's *usable* moves each epoch - so the redundancy the encoder's
    dataset docstring warns about is turned into augmentation, as there, but
    every draw is a row that can be learned from.
    """

    def __init__(self, rows, pairs, patch_index, per_game=1, seed=0,
                 swap_aug=True):
        self.rows = rows
        self.patch_index = patch_index
        self.per_game = max(1, int(per_game))
        self.swap_aug = swap_aug
        self.seed = seed
        # Built over ALL rows: a val game's fearless pool comes from earlier
        # games of its series, which are train rows.
        self.fearless_roles = DS.build_fearless_roles(rows)
        self.by_game = {}
        for r, m in pairs:
            self.by_game.setdefault(int(r), []).append(int(m))
        self.games = sorted(self.by_game)
        self._aug_rng = torch.Generator().manual_seed(seed + 4241)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Redraw which `per_game` usable moves each game contributes."""
        g = torch.Generator().manual_seed(
            (self.seed * 7919 + epoch * 104729) % (2 ** 63 - 1))
        pick = []
        for r in self.games:
            moves = self.by_game[r]
            if len(moves) <= self.per_game:
                pick.extend((r, m) for m in moves)
            else:
                idx = torch.randperm(len(moves), generator=g)[:self.per_game]
                pick.extend((r, moves[int(j)]) for j in idx)
        self._pick = pick

    def __len__(self):
        return len(self._pick)

    def __getitem__(self, idx):
        r, m = self._pick[idx]
        sample = DS.build_prefix(self.rows[r], m, NUM_REAL_CHAMPS,
                                 self.patch_index, self.fearless_roles)
        if self.swap_aug and \
                torch.rand((), generator=self._aug_rng).item() < 0.5:
            sample = DS.swap_sides(sample)
        return sample
