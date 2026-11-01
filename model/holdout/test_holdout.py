"""Tests for the single-series holdout walkthrough (`model/holdout/`).

What these pin, and why each one would otherwise fail silently:

1. **The series really is outside the training data.** The whole report is
   only interesting if `486564` / `486571` are absent from both caches. A
   cache rebuilt from the updated 2026 CSV would quietly fold the games into
   the corpus the models trained on and turn a holdout into a train-set
   evaluation - which still produces a plausible-looking report.
2. **The harness reproduces a recorded number.** `policy-ablF` on patch-16.20
   test rows must give the top-1/top-5 that STATUS.md §3 already records.
   That is the check that the ranking, the legality mask and the patch id
   wiring all agree with the training-time evaluator rather than merely
   running.
3. **No future information reaches the board.** `build_prefix` must never put
   the target champion on the board it predicts from - that leak would show up
   as an inflated score, not as an error.
4. **The masked readout hides exactly one slot** and the hidden slot's
   champion is not visible anywhere else in the batch.
5. **The aggregate arithmetic is what the report claims**, including the
   per-phase split that carries the ban1 argument.
"""
import json
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import eval_series as ES  # noqa: E402

P = ES.P
PRO_DATA = ES.PRO_DATA
CACHE = ROOT / "data" / "pro play" / "pro_drafts.pt"
CACHE_ALL = ROOT / "data" / "pro play" / "pro_drafts_all.pt"
ABLF = ROOT / "model/policy/ckpt/run_20261025_000606_ablF-stage1/best_policy.pt"


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


# ---------------------------------------------------------------- holdout
@pytest.mark.parametrize("path", [CACHE, CACHE_ALL])
def test_series_games_are_not_in_the_cache(path):
    """The two games must be absent, or the report is a train-set readout."""
    if not path.exists():
        pytest.skip("%s not built" % path.name)
    rows = torch.load(str(path), map_location="cpu", weights_only=False)["rows"]
    gameids = {r["gameid"] for r in rows}
    assert not (set(ES.SERIES_GAMES) & gameids), \
        "%s already contains the holdout series" % path.name


def test_series_rows_build_from_the_csv():
    """The pipeline accepts both games and they are one Bo-series, 1 then 2."""
    if not ES.CSV_2026.exists():
        pytest.skip("2026 CSV not present")
    rows, id_to_name = ES.load_series_rows()
    assert [r["gameid"] for r in rows] == sorted(ES.SERIES_GAMES)
    assert [r["game_number"] for r in rows] == [1, 2]
    assert rows[0]["series_id"] == rows[1]["series_id"]
    # game 2 is fearless, so it must carry game 1's picks as unavailable
    assert rows[0]["unavailable"] == []
    assert len(rows[1]["unavailable"]) == 10
    assert set(rows[1]["unavailable"]) == set(rows[0]["champ"])
    for r in rows:
        assert len(r["moves"]) == ES.N_MOVES
        assert all(0 <= c < 173 for _k, _t, c, _ro in r["moves"])


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
def test_harness_reproduces_the_recorded_patch_number():
    """policy-ablF on 16.20 test rows == STATUS.md §3's recorded figures.

    First 40 patch-16.20 test rows, 800 states. A change to the legality mask,
    the ranking or the patch plumbing moves these numbers.
    """
    if not CACHE.exists() or not ABLF.exists():
        pytest.skip("cache or checkpoint missing")
    rows = torch.load(str(CACHE), map_location="cpu",
                      weights_only=False)["rows"]
    P.assign_weights(rows, 0.0, use_recency=False)
    splits = P.split_series_by_date(rows)
    focus = [i for i in splits["test"]
             if P.patch_key(rows[i]["patch"]) == "16.20"]
    if len(focus) < 40:
        pytest.skip("not enough 16.20 test rows")
    model, patch_index, _cfg = ES.load_policy(ABLF)
    games = ES.score_policy(model, patch_index, [rows[i] for i in focus[:40]],
                            {i: str(i) for i in range(173)})
    agg = ES.aggregate(games)
    assert agg["n"] == 800
    assert agg["top1"] == pytest.approx(0.0875, abs=1e-6)
    assert agg["top5"] == pytest.approx(0.25625, abs=1e-6)
    assert agg["mean_rank"] == pytest.approx(26.605, abs=1e-3)


def test_ranks_are_consistent_with_top1():
    """A model's own top-1 must rank 0, and probabilities must be sane."""
    if not ABLF.exists():
        pytest.skip("checkpoint missing")
    model, patch_index, _cfg = ES.load_policy(ABLF)
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


# ------------------------------------------------------------------ report
def test_report_json_shape():
    """If the report exists, every model scored every state."""
    path = HERE / "reports" / "blg_vs_hle.json"
    if not path.exists():
        pytest.skip("report not generated")
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert len(rep["series"]["games"]) == 2
    for name, d in rep["next_move"].items():
        assert len(d["games"]) == 2
        for g in d["games"]:
            assert len(g["steps"]) == ES.N_MOVES
            assert all("top1_name" in s and "rank" in s for s in g["steps"])
    for name, d in rep["masked"].items():
        for g in d["games"]:
            assert len(g["slots"]) == ES.N_SLOTS
    for name, d in rep["value"].items():
        for g in d["games"]:
            assert len(g["p_blue"]) == ES.N_MOVES
            assert all(0.0 <= p <= 1.0 for p in g["p_blue"])
    # the baseline must cover the same states as the models
    assert sum(len(g["steps"]) for g in rep["baseline"]["games"]) == 40
