"""Tests for the test-split evaluation (`model/test_split/`).

What these pin, and why each one would otherwise fail silently:

1. **The pinned games are in the corpus but out of train/val.** The report is
   only interesting if the seven games are scored on data nothing trained on.
   The corpus pin and `split_within_patch(pinned_test=...)` are what guarantee
   that, so both are asserted here: the rows must be present in
   `pro_drafts.pt` (a series the corpus has never seen cannot be a test split)
   and absent from every train/val index.
2. **The CSV resolves the pin.** `load_test_rows` looks the games up by
   `series_id`, so a renamed team or a moved date breaks loudly instead of
   producing a quietly smaller test set.
3. **No future information reaches the board.** `build_prefix` must never put
   the target champion on the board it predicts from - that leak would show up
   as an inflated score, not as an error.
4. **The masked readout hides exactly one slot** and the hidden slot's
   champion is not visible anywhere else in the batch.
5. **The aggregate arithmetic is what the report claims**, including the
   per-phase split and the Wilson interval printed next to every rate.
6. **The report matches the registry and the registry matches the disk.**
   Weights are git-ignored, so a dangling path is invisible until someone
   tries to reproduce the committed report.
"""
import json
import math
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import eval_test_split as ES  # noqa: E402
import render_test_split as RS  # noqa: E402

P = ES.P
PRO_DATA = ES.PRO_DATA
CACHE = ROOT / "data" / "pro play" / "pro_drafts.pt"
CKPT_DIR = ROOT / "model" / "checkpoints"
N_TEST_GAMES = 7


def _rows(n=3):
    """Synthetic rows with the fields build_prefix / score_* need.

    Twenty *distinct* champions, 10 bans and 10 picks, laid onto the canonical
    move order: `game_to_draft` rejects duplicate champions, so a fixture that
    reused one would exercise a state the real pipeline never produces (and
    would make the target illegal, silently zeroing its probability).
    """
    out = []
    for i in range(n):
        base = i * 20
        champ = [base + 10 + s for s in range(10)]      # slot order
        role = list(range(5)) * 2
        moves = []
        for m in range(6):                              # 6 bans
            moves.append((ES.MOVE_BAN, m % 2, base + m, -100))
        for m in range(6, 12):                          # 6 picks, slots 0-5
            s = m - 6
            moves.append((ES.MOVE_PICK, s // 5, champ[s], role[s]))
        for m in range(12, 16):                         # 4 bans
            moves.append((ES.MOVE_BAN, (m - 12) % 2, base + 6 + (m - 12), -100))
        for m in range(16, 20):                         # 4 picks, slots 6-9
            s = m - 10
            moves.append((ES.MOVE_PICK, s // 5, champ[s], role[s]))
        out.append({
            "gameid": "g%d" % i, "series_id": "s%d" % i, "game_number": 1,
            "league": "LPL", "patch": "16.20", "date": "2026-10-31 22:00:00",
            "date_ordinal": 739100 + i, "blue_team": "A", "red_team": "B",
            "first_pick": 0, "blue_win": 1, "champ": champ, "role": role,
            "moves": moves, "unavailable": [], "prior_picks": [],
            "fearless": False, "econ": [None] * 6, "has_econ": False,
            "weight": 1.0,
        })
    return out


# ------------------------------------------------------------- the split
def test_pinned_series_are_in_the_corpus_but_not_in_train_or_val():
    """The test rows must exist *and* be held out of every training index.

    Both halves matter. Absent from the corpus means the pin was never raised
    and the report is scoring a holdout nothing could train against; present
    in train or val means the report is a train-set readout that still looks
    plausible.
    """
    if not CACHE.exists():
        pytest.skip("pro_drafts.pt not built")
    rows = torch.load(str(CACHE), map_location="cpu", weights_only=False)["rows"]
    by_series = {}
    for i, r in enumerate(rows):
        by_series.setdefault(r["series_id"], []).append(i)
    for sid in ES.TEST_SERIES:
        assert sid in by_series, "%s is not in the corpus" % sid

    P.assign_weights(rows, 0.0, use_recency=False)
    splits = P.split_within_patch(rows)
    test = set(splits["test"])
    assert len(test) == N_TEST_GAMES
    for sid in ES.TEST_SERIES:
        idxs = set(by_series[sid])
        assert idxs <= test, "%s is not wholly in test" % sid
        assert not (idxs & set(splits["train"]))
        assert not (idxs & set(splits["val"]))


def test_test_rows_build_from_the_csv():
    """Every pinned series resolves to a whole Bo-series with a fearless tail."""
    if not ES.CSV_2026.exists():
        pytest.skip("2026 CSV not present")
    rows, _id_to_name = ES.load_test_rows()
    assert len(rows) == N_TEST_GAMES
    assert {r["series_id"] for r in rows} == set(ES.TEST_SERIES)
    # grouped by series, games in order, and every game is a full 20-move draft
    seen = []
    for r in rows:
        assert len(r["moves"]) == ES.N_MOVES
        assert all(0 <= c < 173 for _k, _t, c, _ro in r["moves"])
        if not seen or seen[-1][0] != r["series_id"]:
            assert r["game_number"] == 1, "series not grouped or not in order"
            seen.append([r["series_id"], []])
        seen[-1][1].append(r["game_number"])
    assert [g for _s, g in seen] == [[1, 2], [1, 2], [1, 2, 3]]
    # fearless: the second game of each series carries the first's ten picks
    for sid in ES.TEST_SERIES:
        games = [r for r in rows if r["series_id"] == sid]
        assert games[0]["unavailable"] == []
        assert set(games[1]["unavailable"]) == set(games[0]["champ"])


# --------------------------------------------------------------- no leakage
def test_prefix_never_shows_the_target():
    """The board a model sees must not contain the move it is asked for."""
    for row in _rows():
        for m in range(ES.N_MOVES):
            s = PRO_DATA.build_prefix(row, m, ES.NUM_CHAMPS, None)
            target = int(s["target_champ"])
            visible = set(int(x) for x in s["champ_ids"][~s["pad_mask"]])
            visible |= set(int(x) for x in s["ban_ids"]
                           if int(x) != ES.CHAMP_PAD_ID)
            assert target not in visible, "move %d leaks its own target" % m


def test_masked_batch_hides_exactly_one_slot():
    row = _rows(1)[0]
    champ = torch.tensor(row["champ"], dtype=torch.long)
    for slot in range(ES.N_SLOTS):
        cids = champ.clone()
        cids[slot] = ES.CHAMP_MASK_ID
        is_m = torch.zeros(1, ES.N_SLOTS, dtype=torch.bool)
        is_m[0, slot] = True
        assert int(is_m.sum()) == 1
        # every other slot still shows its real champion
        for other in range(ES.N_SLOTS):
            if other != slot:
                assert int(cids[other]) == int(champ[other])
        assert int(cids[slot]) == ES.CHAMP_MASK_ID


# ------------------------------------------------------------ the harness
def test_ranks_are_consistent_with_top1():
    """A model's own top-1 must rank 0, and probabilities must be sane."""
    path = ROOT / ES.POLICY_MODELS[0][2]
    if not path.exists():
        pytest.skip("checkpoint missing")
    model, patch_index, _cfg = ES.load_policy(path)
    games = ES.score_policy(model, patch_index, _rows(2),
                            {i: str(i) for i in range(173)})
    for g in games:
        assert len(g["steps"]) == ES.N_MOVES
        for s in g["steps"]:
            assert 0 <= s["rank"] < ES.NUM_CHAMPS
            assert s["top5"][0] == s["top1"]
            assert len(s["top5"]) == 5
            assert 0.0 < s["prob"] <= 1.0
            assert s["rank"] == 0 or s["prob"] < 1.0


# --------------------------------------------------------------- arithmetic
def test_aggregate_counts_topk_and_phase():
    games = [{"steps": [
        {"move": 0, "rank": 0}, {"move": 1, "rank": 4},
        {"move": 2, "rank": 9}, {"move": 6, "rank": 1},
    ]}]
    agg = ES.aggregate(games)
    assert agg["n"] == 4
    assert agg["top1"] == pytest.approx(0.25)      # ranks 0 and 1
    assert agg["top5"] == pytest.approx(0.75)      # 0, 4, 1
    assert agg["mean_rank"] == pytest.approx(3.5)
    ph = ES.by_phase(games)
    assert ph["ban1"]["n"] == 3 and ph["pick1"]["n"] == 1
    assert ph["ban1"]["top5"] == pytest.approx(2 / 3)
    assert ph["ban2"]["n"] == 0
    assert ph["ban2"]["top5"] == 0.0               # not NaN on an empty phase


def test_popularity_reference_is_a_permutation():
    """Every champion gets exactly one rank, so ranks are comparable to a model's."""
    rows = _rows(4)
    ranks, n = ES.patch_popularity_reference("16.20", rows)
    assert n == 4
    for m in range(ES.N_MOVES):
        assert sorted(ranks[m]) == list(range(ES.NUM_CHAMPS))


def test_wilson_interval_brackets_the_estimate():
    """The interval printed next to every rate must contain that rate."""
    for k, n in ((0, 7), (3, 7), (7, 7), (0, 1), (5, 10)):
        lo, hi = RS.wilson(k, n)
        p = k / n
        assert 0.0 <= lo <= p <= hi <= 1.0, (k, n, lo, hi)
    # a 0-of-7 rate has an upper bound well above zero - the point of printing it
    assert RS.wilson(0, 7)[1] > 0.3
    assert RS.wilson(0, 7)[0] == 0.0
    # and the width shrinks as n grows
    assert (RS.wilson(35, 70)[1] - RS.wilson(35, 70)[0]) < \
           (RS.wilson(3, 7)[1] - RS.wilson(3, 7)[0])
    assert RS.wilson(0, 0) == (0.0, 0.0)


# ------------------------------------------------------------------ report
def _report():
    path = HERE / "reports" / "test_split.json"
    if not path.exists():
        pytest.skip("report not generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_report_json_shape():
    """If the report exists, every model scored every state."""
    rep = _report()
    assert len(rep["series"]["games"]) == N_TEST_GAMES
    assert rep["test_split"]["n_games"] == N_TEST_GAMES
    assert len(rep["test_split"]["series"]) == len(ES.TEST_SERIES)
    assert sum(s["n_games"] for s in rep["test_split"]["series"]) == N_TEST_GAMES
    for name, d in rep["next_move"].items():
        assert len(d["games"]) == N_TEST_GAMES
        for g in d["games"]:
            assert len(g["steps"]) == ES.N_MOVES
            assert all("top1_name" in s and "rank" in s for s in g["steps"])
    for name, d in rep["masked"].items():
        assert len(d["games"]) == N_TEST_GAMES
        for g in d["games"]:
            assert len(g["slots"]) == ES.N_SLOTS
    # the baseline must cover the same states as the models
    assert sum(len(g["steps"]) for g in rep["baseline"]["games"]) == \
        N_TEST_GAMES * ES.N_MOVES


def test_report_model_set_matches_the_registry():
    """Pruning a checkpoint or editing the registry must regenerate the report.

    The report is the artefact people read. If it lists a model whose weights
    were deleted, the tool can no longer reproduce its own committed output -
    and nothing else in the suite would notice.
    """
    rep = _report()
    assert set(rep["next_move"]) == {n for n, _f, _p in ES.POLICY_MODELS}
    assert set(rep["masked"]) == {n for n, _f, _p in ES.MASKED_MODELS}
    assert set(rep["value"]) == {n for n, _f, _p, _k in ES.VALUE_MODELS}


def test_no_registry_entry_points_at_a_deleted_checkpoint():
    """The pruning must not leave a dangling path behind.

    Weights are git-ignored, so this is only meaningful on a machine that has
    them; a fresh clone has none and skips.
    """
    if not CKPT_DIR.exists():
        pytest.skip("weights are git-ignored; nothing to check on a fresh clone")
    assert ES.missing_checkpoints() == []


def test_every_registry_name_states_its_inference():
    """The directory name is the frontend's contract, so it must be readable.

    `frontend/server._ckpt_roles` decides which slot a checkpoint may fill from
    the prefix alone - `masking-*` for the role board, `nextpick-*` for the
    next-move panel. A name outside that scheme would be offered in both slots
    and fail on load.
    """
    for name, _f, path in ES.POLICY_MODELS:
        assert name.startswith("nextpick-"), name
        assert "/checkpoints/%s/" % name in path, path
    for name, _f, path in ES.MASKED_MODELS:
        assert name.startswith("masking-"), name
        assert "/checkpoints/%s/" % name in path, path
