"""Tests for the pro drafts pipeline (real CSVs + synthetic edge cases).

- test_real_* : run against the actual OraclesElixir corpus (2022-2026
  in this folder). They validate draft conversion, First-Selection move
  interleaving, era-aware fearless/firstPick, series splits, and dataset
  batches (incl. the soloQ contract validator on masked pro drafts).
- test_*_synthetic : fast checks of interleaving/order logic and edge
  cases without touching the ~350MB of CSVs.
"""
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "model" / "tests"))

import pipeline as P
import datasets as D
from test_soloq_contract import validate_batch_dict

NUM_CHAMPS = 173
_cached = {}


def _rows():
    if "rows" not in _cached:
        rows, stats = P.build_rows()
        P.assign_weights(rows)
        _cached["rows"] = rows
        _cached["stats"] = stats
    return _cached["rows"], _cached["stats"]


def _splits():
    if "splits" not in _cached:
        rows, _ = _rows()
        _cached["splits"] = P.split_series_by_date(rows)
    return _cached["splits"]


# ---------------------------------------------------------------- real data

def test_scan_keeps_most_games():
    rows, stats = _rows()
    assert stats["kept"] > 40000
    assert stats["kept"] + stats["skipped"] == stats["total_games"]
    # skip reasons are the documented ones only
    assert set(stats["skip_reasons"]) <= {
        "missing_team_row", "bad_firstpick", "bad_result", "bad_date",
        "missing_players", "bad_position", "incomplete_bans",
        "incomplete_picks", "unknown_champ", "duplicate_champ",
        "pick_mismatch"}
    assert stats["skip_reasons"].get("unknown_champ", 0) == 0, \
        "all pro names must map through the 173-champ ddragon mapping"


def test_draft_rows_valid():
    rows, _ = _rows()
    for row in rows[::97]:  # sample every 97th draft
        assert len(row["champ"]) == 10 and len(row["role"]) == 10
        assert len(set(row["champ"])) == 10
        assert row["role"] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        assert all(0 <= c < NUM_CHAMPS for c in row["champ"])
        assert row["first_pick"] in (0, 1)
        assert row["blue_win"] in (0, 1)
        assert len(row["moves"]) == P.N_MOVES
        bans = [m for m in row["moves"] if m[0] == P.MOVE_BAN]
        picks = [m for m in row["moves"] if m[0] == P.MOVE_PICK]
        assert len(bans) == 10 and len(picks) == 10
        assert len({m[2] for m in row["moves"]}) == 20  # all unique
        # picks carry real roles, bans carry IGNORE
        assert all(m[3] == P.IGNORE_INDEX for m in bans)
        assert all(0 <= m[3] < 5 for m in picks)


def test_first_selection_interleaving_real():
    """Move team pattern matches first pick; picks land in team order."""
    rows, _ = _rows()
    for row in rows[::113]:
        fp, sp = row["first_pick"], 1 - row["first_pick"]
        teams = [m[1] for m in row["moves"]]
        assert teams == [fp, sp, fp, sp, fp, sp] + \
                       [fp, sp, sp, fp, fp, sp] + \
                       [sp, fp, sp, fp] + \
                       [sp, fp, fp, sp]
        # per-team move subsequences match the team's own pick/ban lists
        for team in (0, 1):
            team_picks = [(m[2], m[3]) for m in row["moves"]
                          if m[0] == P.MOVE_PICK and m[1] == team]
            slots = sorted((row["champ"][team * 5 + r], r) for r in range(5))
            assert sorted(team_picks) == slots
            team_bans = [m[2] for m in row["moves"]
                         if m[0] == P.MOVE_BAN and m[1] == team]
            assert len(team_bans) == 5 and len(set(team_bans)) == 5


def test_fearless_era_gating():
    """prior_picks is factual; fearless/unavailable data-driven per series."""
    rows, _ = _rows()
    by_series = defaultdict(list)
    for row in rows:
        by_series[row["series_id"]].append(row)
    # date-ordered: evaluate modern series too (rows are CSV-file ordered:
    # 2022 first, so a flat loop would break out in the old era)
    modern = fearless_n = 0
    for series_rows in sorted(by_series.values(),
                              key=lambda rs: rs[0]["date_ordinal"]):
        if len(series_rows) < 2:
            continue
        series_rows.sort(key=lambda r: r["game_number"])
        seen = set()
        for row in series_rows:
            # prior_picks is always the factual earlier-picks union
            assert row["prior_picks"] == sorted(seen)
            if row["fearless"]:
                assert row["unavailable"] == row["prior_picks"]
                if row["game_number"] > 1:
                    assert not (set(row["champ"]) & seen), row["gameid"]
            else:
                assert row["unavailable"] == []
            if row["date_ordinal"] >= P.FEARLESS_MIN_ORDINAL:
                modern += 1
                fearless_n += row["fearless"]
            seen.update(row["champ"])
        if modern > 2000:
            break
    assert modern > 1000
    assert 0.85 * modern < fearless_n < modern, \
        "most modern series fearless; HW/NEXO/LJL-like exceptions exist"
    # 2026 specifically: fully fearless
    y26 = [r for r in rows if r["date_ordinal"] >=
           datetime(2026, 1, 1).toordinal()]
    later26 = [r for r in y26 if r["game_number"] > 1]
    assert later26 and all(r["fearless"] for r in later26), \
        "non-fearless 2026 series found"


def test_split_no_series_leakage():
    rows, _ = _rows()
    splits = _splits()
    all_idx = sorted(splits["train"] + splits["val"] + splits["test"])
    assert all_idx == list(range(len(rows))), "splits must cover all rows"
    sids = {k: {rows[i]["series_id"] for i in splits[k]}
            for k in ("train", "val", "test")}
    assert not (sids["train"] & sids["val"])
    assert not (sids["train"] & sids["test"])
    assert not (sids["val"] & sids["test"])
    # chronological: newest train date <= oldest val/test date (by series
    # start); allow slack for same-day series boundaries
    tr_max = max(rows[i]["date_ordinal"] for i in splits["train"])
    va_min = min(rows[i]["date_ordinal"] for i in splits["val"])
    te_min = min(rows[i]["date_ordinal"] for i in splits["test"])
    assert va_min >= tr_max - 1 and te_min >= va_min - 1


def test_weights_recency():
    rows, _ = _rows()
    P.assign_weights(rows, half_life_days=75)
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date_ordinal"], []).append(r["weight"])
    ords = sorted(by_date)
    means = [sum(by_date[o]) / len(by_date[o]) for o in ords]
    assert means == sorted(means), "weight must decay w/ age"
    total = sum(r["weight"] for r in rows) / len(rows)
    assert abs(total - 1.0) < 1e-6, "weights mean-normalized"
    P.assign_weights(rows, half_life_days=0)  # disable
    assert all(r["weight"] == 1.0 for r in rows)
    P.assign_weights(rows, half_life_days=75)  # restore


def test_popularity_baseline_train_only():
    rows, _ = _rows()
    # weights mutated by test_weights_recency above - refresh
    P.assign_weights(rows, half_life_days=75)
    splits = _splits()
    base = P.compute_popularity_baseline(rows, splits["train"], NUM_CHAMPS)
    assert len(base["pick_global_top"]) == NUM_CHAMPS
    assert set(base["pick_global_top"]) == set(range(NUM_CHAMPS))
    assert len(base["pick_role_top"]) == 5
    assert len(base["ban_top"]) == NUM_CHAMPS
    assert float(base["pick_global_counts"].sum()) > 0
    # weighted counts: pick totals equal weight-sum x 10 picks; bans x 10
    train_w = sum(rows[i].get("weight", 1.0) for i in splits["train"])
    # float32 accumulation -> loose tolerance, exact logic checked below
    assert abs(float(base["pick_global_counts"].sum()) - train_w * 10) < 1.0
    assert abs(float(base["ban_counts"].sum()) - train_w * 10) < 1.0

def test_nextmove_dataset_shapes_and_consistency():
    rows, _ = _rows()
    splits = _splits()
    ds = D.ProNextMoveDataset(rows, splits["val"])
    assert len(ds) == len(splits["val"]) * 20
    loader = DataLoader(ds, batch_size=16, collate_fn=D.pro_collate)
    batch = next(iter(loader))
    B = 16
    assert batch["champ_ids"].shape == (B, 10)
    assert batch["taken_mask"].shape == (B, NUM_CHAMPS)
    assert batch["unavailable_mask"].shape == (B, NUM_CHAMPS)
    assert batch["target_champ"].shape == (B,)
    # PAD discipline: visible picks + bans made == moves consumed
    for b in range(B):
        mi = int(batch["move_idx"][b])
        n_picks = int((~batch["pad_mask"][b]).sum())
        n_bans = int(batch["n_bans"][b])
        assert n_picks + n_bans == mi
        assert n_bans == min(mi, 6) if mi < 12 else n_bans == mi - 6
        # target legality: never already taken, never fearless-unavailable
        tc = int(batch["target_champ"][b])
        assert not bool(batch["taken_mask"][b, tc])
        assert not bool(batch["unavailable_mask"][b, tc])
        # role targets only on picks
        if int(batch["is_ban"][b]):
            assert int(batch["target_role"][b]) == -100
        else:
            assert 0 <= int(batch["target_role"][b]) < 5


def test_masked_pro_batch_passes_soloq_contract():
    rows, _ = _rows()
    splits = _splits()
    mds = D.ProMaskedDraftDataset(rows, splits["val"][:32], seed=7)
    batch = D.pro_collate([mds[i] for i in range(32)])
    assert batch["pad_mask"].sum() == 0  # full drafts, no PAD
    assert set(batch["win_label"].tolist()) <= {0, 1}
    assert (batch["weight"] > 0).all()
    # validator rejects win_label (deferred in soloQ contract) - strip
    # side-channel keys before validating the shared core.
    core = {k: v for k, v in batch.items()
            if k in ("champ_ids", "role_ids", "role_masked", "is_masked",
                     "champ_labels", "role_labels", "pad_mask")}
    validate_batch_dict(core, allow_partial=False)


def test_weighted_sampler_deterministic():
    rows, _ = _rows()
    P.assign_weights(rows, half_life_days=75)
    splits = _splits()
    ds = D.ProNextMoveDataset(rows, splits["train"])
    s1 = D.make_weighted_sampler(ds, seed=3)
    s2 = D.make_weighted_sampler(ds, seed=3)
    a, b = list(s1)[:4000], list(s2)[:4000]
    assert a == b
    # skew: mean weight of sampled > mean weight of pool (recent oversampled)
    pool = torch.tensor(ds.weights(), dtype=torch.double)
    samp = pool[a]
    assert float(samp.mean()) > float(pool.mean())


def _synth_team_bans(start):
    return {0: [start + i for i in range(5)],
            1: [start + 10 + i for i in range(5)]}


def _synth_team_picks(start):
    # (champ, role) in team pick order pick1..5
    return {0: [(start + i, i % 5) for i in range(5)],
            1: [(start + 50 + i, i % 5) for i in range(5)]}


def test_interleaving_fp_blue_synthetic():
    moves = P.interleaved_moves(0, _synth_team_bans(0), _synth_team_picks(0))
    assert len(moves) == 20
    kinds = [m[0] for m in moves]
    assert kinds == [P.MOVE_BAN] * 6 + [P.MOVE_PICK] * 6 + \
        [P.MOVE_BAN] * 4 + [P.MOVE_PICK] * 4
    # classic order: FP=blue first in phases 1, SP=red first in phases 2
    teams = [m[1] for m in moves]
    assert teams == [0, 1] * 3 + [0, 1, 1, 0, 0, 1] + \
        [1, 0] * 2 + [1, 0, 0, 1]
    # team-side indices consumed in order within each team's moves
    for team in (0, 1):
        bans = [m[2] for m in moves
                if m[0] == P.MOVE_BAN and m[1] == team]
        offset = 0 if team == 0 else 10
        assert bans == [offset + i for i in range(5)]


def test_interleaving_fp_red_mirror():
    moves_b = P.interleaved_moves(0, _synth_team_bans(0),
                                  _synth_team_picks(0))
    moves_r = P.interleaved_moves(1, _synth_team_bans(0),
                                  _synth_team_picks(0))
    assert len(moves_r) == 20
    # full mirror: every move's team flips, kinds/positions identical
    for mb, mr in zip(moves_b, moves_r):
        assert (mr[0], mr[1], mr[3]) == (mb[0], 1 - mb[1], mb[3])
    # phase-1 bans start with SP=blue when red has first pick
    teams = [m[1] for m in moves_r]
    assert teams == [1, 0] * 3 + [1, 0, 0, 1, 1, 0] + \
        [0, 1] * 2 + [0, 1, 1, 0]
    # but a team's own pick order is preserved (pick1..pick5 in sequence)
    for team in (0, 1):
        picks = [m[2] for m in moves_r
                 if m[0] == P.MOVE_PICK and m[1] == team]
        off = 0 if team == 0 else 50
        assert picks == [off + i for i in range(5)]


def test_econ_mode_subset_discipline():
    rows, _ = _rows()
    splits = _splits()
    mds = D.ProMaskedDraftDataset(rows, splits["val"][:200], seed=5,
                                  econ_mask_prob=0.5)
    partial = full = 0
    for i in range(200):
        ex = mds[i]
        assert ex["econ_target"].shape == (6,)
        assert float(ex["has_econ"]) in (0.0, 1.0)
        if bool(ex["pad_mask"].any()):
            partial += 1
            assert not bool(ex["is_masked"].any())
            assert int(ex["champ_labels"].eq(-100).all())
            assert 1 <= int((~ex["pad_mask"]).sum()) <= 9
            assert torch.equal(ex["pad_mask"],
                               ex["champ_ids"] == D.CHAMP_PAD_ID)
        else:
            full += 1
            assert int(ex["champ_labels"].eq(-100).all()) or \
                bool(ex["is_masked"].any())
    assert partial > 50 and full > 50, (partial, full)


def test_prefix_move_targets_real():
    """Spot check: targets match the team's actual pick/ban lists."""
    rows, _ = _rows()
    row = next(r for r in rows if r["first_pick"] == 1 and r["fearless"])
    for mi in (0, 6, 12, 16):
        ex = D.build_prefix(row, mi)
        kind, team, champ, _role = row["moves"][mi]
        assert int(ex["to_move"]) == team
        assert int(ex["is_ban"]) == (kind == P.MOVE_BAN)
        assert int(ex["target_champ"]) == champ
        assert int(ex["phase"]) == D.phase_of(mi)
        assert int(ex["n_bans"]) == sum(
            1 for m in row["moves"][:mi] if m[0] == P.MOVE_BAN)


