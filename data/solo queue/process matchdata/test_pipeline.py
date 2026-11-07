"""Tests for the SoloQ data pipeline (real shards + synthetic masking).

- test_real_* : run against the actual matchdata corpus (patch 16.20).
  They build the champ mapping (expect 173 champs), scan shards, check
  filters/slot order, and validate a real DataLoader batch with the
  model contract validator (imported from solo_queue/model/tests, not duplicated).
- test_masking_* : fast synthetic checks of apply_masking edge cases.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent / "solo_queue" / "model" / "tests"))

import pipeline as P
from test_soloq_contract import validate_batch_dict

PATCH = "16.20"

_cached = {}


def _shards():
    paths = P.list_shards()
    assert len(paths) > 0, "no matchdata shards found"
    return paths


def _mapping():
    if "mapping" not in _cached:
        _cached["mapping"] = P.build_champ_mapping(_shards(), PATCH,
                                                   save_path=None)
    return _cached["mapping"]


def _indexed(limit=5000):
    key = f"indexed_{limit}"
    if key not in _cached:
        # with_win=True so the win-label tests below have labels to check;
        # rows stay 2-tuples either way, so the other tests are unaffected.
        _cached[key] = P.scan_and_index(_shards(), _mapping(), PATCH,
                                        limit=limit, with_win=True)
    return _cached[key]


def test_champ_mapping_is_173():
    mapping = _mapping()
    assert len(mapping) == 173, f"expected 173 champs on {PATCH}, got {len(mapping)}"
    assert set(mapping.values()) == set(range(173))


def test_slot_order_and_filters():
    mapping = _mapping()
    indexed = _indexed(limit=5000)
    assert indexed["kept"] > 0
    # every kept row: 10 unique champ idx, canonical role order per team
    for champ, role in indexed["rows"]:
        assert role == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        assert len(set(champ)) == 10
        assert all(0 <= c < 173 for c in champ)
    # filters reject: wrong patch, wrong queue, short game, bad position
    good = {"version": "16.20.1", "queue_id": 420, "duration": 1500,
            "players": [{"team": 100 if i < 5 else 200,
                         "champ": list(mapping)[:10][i],
                         "pos": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][i % 5]}
                        for i in range(10)]}
    assert P.record_to_slots(good, mapping, PATCH) is not None
    bad_patch = dict(good, version="16.19.1")
    assert P.record_to_slots(bad_patch, mapping, PATCH) is None
    bad_queue = dict(good, queue_id=400)
    assert P.record_to_slots(bad_queue, mapping, PATCH) is None
    bad_dur = dict(good, duration=60)
    assert P.record_to_slots(bad_dur, mapping, PATCH) is None
    bad_pos = json_like(good)
    bad_pos["players"][0]["pos"] = "Invalid"
    assert P.record_to_slots(bad_pos, mapping, PATCH) is None


def json_like(d):
    import copy
    return copy.deepcopy(d)


def test_real_batch_passes_model_contract():
    mapping = _mapping()
    indexed = _indexed(limit=5000)
    ds = P.SoloQDraftDataset(indexed["rows"], num_champs=173, seed=0)
    assert len(ds) >= 32
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        collate_fn=P.soloq_collate)
    batch = next(iter(loader))
    validate_batch_dict(batch, allow_partial=False)
    # every draft in the batch has >= 1 mask
    assert bool(batch["is_masked"].all(dim=1).any() or
                (batch["is_masked"].sum(dim=1) >= 1).all())


def test_champ_names_cover_mapping():
    mapping = _mapping()
    names = P.fetch_champ_names()  # network, cached to champ_names.json
    assert len(names) == 173
    missing = [r for r in mapping if r not in names]
    assert not missing, f"Data Dragon missing Riot ids: {missing}"
    ordered = P.idx_names(mapping, names)
    assert len(ordered) == 173 and all(isinstance(n, str) for n in ordered)
    assert P.name_of(9999, ordered) == "?"


def test_popularity_baseline_sane():
    indexed = _indexed(limit=5000)
    base = P.compute_popularity_baseline(indexed["rows"], 173)
    assert len(base["global_top"]) == 173
    assert set(base["global_top"]) == set(range(173))
    assert int(base["global_counts"].sum()) == len(indexed["rows"]) * 10
    for r in range(5):
        assert len(base["role_top"][r]) == 173
        # top-1 per role is actually played in that role
        assert int(base["role_counts"][r, base["role_top"][r][0]]) > 0


def test_popularity_baseline_names():
    indexed = _indexed(limit=5000)
    mapping = _mapping()
    names = P.load_champ_names()  # offline: uses cache from previous test
    ordered = P.idx_names(mapping, names)
    base = P.compute_popularity_baseline(indexed["rows"], 173, names=ordered)
    assert len(base["global_top_names"]) == 173
    assert base["global_top_names"][0] == ordered[base["global_top"][0]]
    for r in range(5):
        assert base["role_top_names"][r][0] == ordered[base["role_top"][r][0]]
        # least popular per role resolves to a real name too
        assert isinstance(base["role_top_names"][r][-1], str)


def test_masking_edge_cases():
    champ = list(range(10))
    role = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    # mask_prob=0 still forces >= 1 mask per draft
    out = P.apply_masking(champ, role, 173, mask_prob=0.0,
                          generator=torch.Generator().manual_seed(0))
    assert bool(out[3].sum() == 1)
    # role_keep_prob=1 keeps all roles visible (role_masked empty)
    out = P.apply_masking(champ, role, 173, mask_prob=1.0, role_keep_prob=1.0,
                          generator=torch.Generator().manual_seed(1))
    assert bool(out[3].all()) and not bool(out[2].any())
    assert torch.equal(out[1], torch.tensor(role))
    # role_keep_prob=0 masks every role on champ-masked slots (plus any
    # role-only draws); check the champ-masked subset is fully hidden.
    out = P.apply_masking(champ, role, 173, mask_prob=1.0, role_keep_prob=0.0,
                          role_only_prob=0.0,
                          generator=torch.Generator().manual_seed(1))
    assert torch.equal(out[2], out[3])
    assert bool((out[1][out[3]] == 5).all())


def test_force_one_off_yields_no_masks():
    """The clean value view must have zero masked champions.

    `force_one=True` (the train-time invariant) guarantees >= 1 mask even at
    mask_prob=0, which would corrupt exactly the draft the value head is being
    measured on. `force_one=False` has to leave the draft untouched.
    """
    champ = list(range(10))
    role = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    out = P.apply_masking(champ, role, 173, mask_prob=0.0, role_only_prob=0.0,
                          generator=torch.Generator().manual_seed(0),
                          force_one=False)
    cids, rids, role_masked, is_m, clab, rlab = out
    assert not bool(is_m.any()), "force_one=False must not mask any champ"
    assert not bool(role_masked.any())
    assert torch.equal(cids, torch.tensor(champ))
    assert torch.equal(rids, torch.tensor(role))
    assert bool((clab == P.IGNORE_INDEX).all()) and bool((rlab == P.IGNORE_INDEX).all())


def _record_with_win(blue_win=1, red_win=0):
    mapping = _mapping()
    champs = list(mapping)[:10]
    rec = {"version": "16.20.1", "queue_id": 420, "duration": 1500,
           "teams": {"blue": {"win": blue_win}, "red": {"win": red_win}},
           "players": [{"team": 100 if i < 5 else 200, "champ": champs[i],
                        "pos": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM",
                                "UTILITY"][i % 5]} for i in range(10)]}
    return rec


def test_win_label_parsed_from_record():
    mapping = _mapping()
    blue_won = P.record_to_slots(_record_with_win(1, 0), mapping, PATCH,
                                 return_win=True)
    red_won = P.record_to_slots(_record_with_win(0, 1), mapping, PATCH,
                                return_win=True)
    assert blue_won is not None and blue_won[2] == 1
    assert red_won is not None and red_won[2] == 0
    # the label is what team 100 (blue) got
    assert blue_won[0] == red_won[0], "same draft, same slots"
    # both sides agreeing means no usable result, not a free label
    broken = P.record_to_slots(_record_with_win(0, 0), mapping, PATCH,
                               return_win=True)
    assert broken is not None and broken[2] is None
    # and the default path still returns the 2-tuple
    plain = P.record_to_slots(_record_with_win(1, 0), mapping, PATCH)
    assert len(plain) == 2


def test_dataset_emits_win_label_only_when_asked():
    # 5000, not 2000: the corpus opens with patch 16.19 records, so a small
    # limit stops inside the filtered prefix and yields zero kept rows.
    indexed = _indexed(limit=5000)
    rows, wins = indexed["rows"], indexed["wins"]
    if wins is None:
        pytest.skip("scan_and_index ran without with_win")
    assert len(wins) == len(rows)
    assert set(wins) <= {0, 1}, "labels must be binary, never None"
    with_win = P.SoloQDraftDataset(rows, num_champs=173, wins=wins, seed=0)
    s = with_win[0]
    assert "win_label" in s and int(s["win_label"]) in (0, 1)
    without = P.SoloQDraftDataset(rows, num_champs=173, seed=0)
    assert "win_label" not in without[0], \
        "callers that pass no wins must get the old batch exactly"
    with pytest.raises(ValueError):
        P.SoloQDraftDataset(rows, num_champs=173, wins=wins[:-1])


def test_win_rate_is_plausible():
    """A ~50% base rate is the sanity check that we mapped the label right.

    Reading the red team's flag, or the wrong field, would land far from 0.5.
    """
    indexed = _indexed(limit=5000)
    wins = indexed["wins"]
    if wins is None:
        pytest.skip("scan_and_index ran without with_win")
    rate = sum(wins) / len(wins)
    assert 0.40 < rate < 0.60, f"blue win rate {rate:.3f} is not plausible"
