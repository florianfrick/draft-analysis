"""Tests for the canonical split, the registry and the scorers.

The split is the load-bearing part: every comparison in this repository is only
meaningful because these rows are the same rows for every model. So the
invariants are asserted rather than assumed - counts, disjointness, and the
manifest fingerprint - and the suite refuses to run when they fail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.test_split import models as M  # noqa: E402
from pro.model.test_split import registry as R  # noqa: E402
from pro.model.test_split import report as REP  # noqa: E402
from pro.model.test_split import split as S  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    rows = S.load_rows()
    splits = S.build(rows)
    return rows, splits


# ------------------------------------------------------------------- the split
def test_the_split_reproduces_the_nov13_counts(corpus):
    """39,720 / 4,771 / 2,448 on 46,939 drafts.

    These are the counts the shipped `-nov13-` pair was trained under, read
    back out of its own `test_report.json`. If this fails, either the corpus
    moved or the split arguments changed, and every number in STATUS.md and in
    this folder's report stops being comparable.
    """
    rows, splits = corpus
    assert len(rows) == S.EXPECTED["corpus"]
    assert {k: len(v) for k, v in splits.items()} == {
        "train": S.EXPECTED["train"], "val": S.EXPECTED["val"],
        "test": S.EXPECTED["test"]}


def test_the_split_args_carry_no_seed():
    """The tail split never draws, so there is no seed to get wrong.

    This is a regression guard with a specific history: the `-randomvalsplit-`
    protocol passed an explicit `seed` that overrode `VAL_SPLIT_SEED` and drew a
    *second, independent* 10% of every patch. 90% of one model's val then sat
    inside another's train, and every cross-model comparison silently shrank to
    a 478-game intersection.
    """
    assert "seed" not in S.SPLIT_ARGS
    assert S.SPLIT_ARGS["val_random"] is False
    assert S.SPLIT_ARGS["pinned_test"] == ()


def test_no_test_game_is_in_train_or_val(corpus):
    """By gameid, not by index - an index check would pass on a duplicate row."""
    rows, splits = corpus
    gid = {k: {rows[i]["gameid"] for i in v} for k, v in splits.items()}
    assert not (gid["test"] & gid["train"])
    assert not (gid["test"] & gid["val"])
    assert not (gid["train"] & gid["val"])


def test_the_manifest_matches_the_corpus(corpus):
    rows, splits = corpus
    ok, diffs = S.verify(rows, splits)
    assert ok, diffs


def test_the_fingerprint_is_order_independent(corpus):
    """The set is the contract, not the order.

    A corpus rebuild that reshuffles rows must not read as a split change.
    """
    rows, splits = corpus
    idx = list(splits["test"])
    assert S.fingerprint(rows, idx) == S.fingerprint(rows, list(reversed(idx)))
    assert S.fingerprint(rows, idx) != S.fingerprint(rows, idx[:-1])


def test_the_test_tail_spans_many_patches(corpus):
    """A tail, not one patch's last week.

    `val_patches=None` is the default, so every patch contributes its newest
    series. If this ever collapses to a handful of patches the test set has
    quietly become a different question.
    """
    rows, splits = corpus
    patches = {rows[i].get("patch") for i in splits["test"]}
    assert len(patches) > 50, "test set spans only %d patches" % len(patches)


def test_sample_games_is_reproducible_and_inside_the_test_set(corpus):
    rows, splits = corpus
    a = S.sample_games(rows, splits["test"], 25, seed=7)
    b = S.sample_games(rows, splits["test"], 25, seed=7)
    c = S.sample_games(rows, splits["test"], 25, seed=8)
    assert a == b
    assert a != c
    assert set(a) <= set(splits["test"])
    assert len(a) == 25
    # asking for more than exists returns the whole set rather than raising
    assert len(S.sample_games(rows, splits["test"], 10 ** 6)) == len(splits["test"])


# ----------------------------------------------------------------- the registry
def test_prefixes_map_to_tasks():
    assert R.task_of("masking-proonly-nov13-through-lcp-df-sw") == "masked"
    assert R.task_of("nextpick-proonly-nov13-k5-through-lcp-df-sw") == "next_move"
    assert R.task_of("role-next-x") == "role_next"
    assert R.task_of("value-something") is None
    assert R.task_of("llm-lora-rerank5") is None


def test_the_transformer_is_scored_on_two_distinct_tasks():
    """One checkpoint, two questions - not one pooled number."""
    name = "nextpick-naivetransformer-role-d512-l8-h8-k5-through-lcp-df-sw"
    assert R.task_of(name) == "next_move"
    assert R.tasks_of(name) == ("next_move", "role_next")
    assert R.tasks_of("role-next-x") == ("role_next",)
    assert R.tasks_of("masking-proonly-nov13-through-lcp-df-sw") == ("masked",)


def test_the_reference_is_first_in_its_task():
    """The incumbent is what everything else is paired against."""
    for task, name in R.REFERENCE.items():
        entries = R.scan(task=task)
        assert entries, "no %s checkpoints found" % task
        assert entries[0].name == name
        assert entries[0].is_reference


def test_checkpoint_file_prefers_best_and_never_latest(tmp_path):
    assert R.checkpoint_file(tmp_path) is None
    (tmp_path / "latest.pt").write_bytes(b"x")
    assert R.checkpoint_file(tmp_path) is None
    (tmp_path / "best.pt").write_bytes(b"x")
    assert R.checkpoint_file(tmp_path).name == "best.pt"


def test_a_foreign_split_is_marked_not_comparable(tmp_path):
    """The label is a fact about the training split, not an opinion.

    A model trained on 42,100/4,768/7 is scored on rows that are in the new
    test set, so its number is a training-set number. The registry has to say
    so, because the alternative is a leaderboard that flatters old models.
    """
    (tmp_path / "best.pt").write_bytes(b"x")
    good = R.Entry(name="masking-x", path=tmp_path / "best.pt", task="masked",
                   track="pro", archived=False)
    R._classify(good, {"n_train": 39720, "n_val": 4771, "n_test": 2448})
    assert good.comparable and good.split_note == ""

    bad = R.Entry(name="masking-y", path=tmp_path / "best.pt", task="masked",
                  track="pro", archived=False)
    R._classify(bad, {"n_train": 42100, "n_val": 4768, "n_test": 7})
    assert not bad.comparable
    assert "different split" in bad.split_note

    unknown = R.Entry(name="masking-z", path=tmp_path / "best.pt",
                      task="masked", track="pro", archived=False)
    R._classify(unknown, {})
    assert not unknown.comparable
    assert "no test_report" in unknown.split_note


def test_split_counts_come_from_the_sibling_report(tmp_path):
    (tmp_path / "test_report.json").write_text(json.dumps(
        {"protocol": {"n_train": 1, "n_val": 2, "n_test": 3, "seed": 0}}),
        encoding="utf-8")
    assert R._split_counts(tmp_path) == {"n_train": 1, "n_val": 2, "n_test": 3}
    (tmp_path / "test_report.json").write_text("{not json", encoding="utf-8")
    assert R._split_counts(tmp_path) == {}


def test_the_archived_store_is_excluded_unless_asked_for():
    active = R.scan(include_archive=False)
    every = R.scan(include_archive=True)
    assert all(not e.archived for e in active)
    assert any(e.archived for e in every)
    assert len(every) >= len(active)


# ------------------------------------------------------------------- scorers
def test_wilson_brackets_the_estimate():
    for hits, n in ((0, 10), (5, 10), (10, 10), (25, 70), (343, 1000)):
        lo, hi = M.wilson(hits, n)
        assert lo <= hits / n <= hi
        assert 0.0 <= lo <= hi <= 1.0
    # the normal approximation gives a zero-width interval at the boundary;
    # Wilson must not, because that is exactly where a broken model lands
    lo, hi = M.wilson(0, 10)
    assert hi > 0.0


def test_aggregate_counts_topk():
    games = [{"gameid": "g1", "steps": [
        {"move": 0, "rank": 0, "prob": 0.5},
        {"move": 1, "rank": 4, "prob": 0.1},
        {"move": 2, "rank": 9, "prob": 0.01},
        {"move": 3, "rank": 50, "prob": 0.001},
    ]}]
    agg = M.aggregate(games, "steps")
    assert agg["n"] == 4
    assert agg["top1"] == 0.25
    assert agg["top5"] == 0.5
    assert agg["top10"] == 0.75
    assert agg["mean_rank"] == pytest.approx(15.75)
    assert agg["mean_prob"] == pytest.approx(0.15275)


def test_paired_hits_aligns_on_the_same_items():
    a = [{"gameid": "g1", "steps": [{"move": 0, "rank": 0},
                                    {"move": 1, "rank": 3}]}]
    b = [{"gameid": "g1", "steps": [{"move": 0, "rank": 5},
                                    {"move": 1, "rank": 0}]}]
    ha, hb = M.paired_hits(a, b, "steps")
    assert ha == [True, False]
    assert hb == [False, True]


def test_mcnemar_counts_discordant_pairs_only():
    """Concordant pairs carry no information about which model is better."""
    a = [True, True, False, False, True]
    b = [True, False, True, False, True]
    a_only, b_only, chi2, p = M.mcnemar(a, b)
    assert (a_only, b_only) == (1, 1)
    assert chi2 == 0.0 and p == pytest.approx(1.0)
    # a clean sweep in one direction is significant
    a = [True] * 20
    b = [False] * 20
    a_only, b_only, chi2, p = M.mcnemar(a, b)
    assert (a_only, b_only) == (20, 0)
    assert p < 1e-4
    # no discordant pairs at all is not evidence of anything
    assert M.mcnemar([True] * 5, [True] * 5)[3] == 1.0


def test_phase_map_matches_the_draft_schedule():
    """6 bans, 6 picks, 4 bans, 4 picks - the order `policy/infer` uses."""
    assert M.PHASE_OF == [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4
    assert len(M.PHASE_OF) == M.N_MOVES == 20


def test_legality_matches_the_policy_model(corpus):
    """The baselines rank over exactly the candidates the policy model allows.

    `_apply_legality` masks taken champions always and the fearless pool on
    picks only - and with -1e4 rather than -inf, so `isfinite` is the wrong
    test. If the baselines' candidate set differed, their top-5 would be
    measured against a different pool than the transformer's, which is the
    quietest way this whole table could become unfair.
    """
    import torch

    from pro.model import baselines as BL

    rows, splits = corpus
    path = (ROOT / "pro/model/checkpoints"
            / "nextpick-proonly-nov13-k5-through-lcp-df-sw/best.pt")
    model, patch_index, _cfg = M.load_policy(path)
    for i in splits["test"][:3]:
        row = rows[i]
        for m in range(BL.N_MOVES):
            sample = M.PRO_DATA.build_prefix(row, m, BL.NUM_CHAMPS, patch_index)
            with torch.no_grad():
                logits = model(M.PRO_DATA.pro_collate([sample]))["policy_logits"][0]
            allowed = (logits > -1e3).numpy()
            mine = BL.candidates_for(row, m)
            assert (mine == np.flatnonzero(allowed)).all(), \
                "move %d of %s: %d candidates vs %d allowed" % (
                    m, row["gameid"], len(mine), int(allowed.sum()))


def test_role_next_states_pin_the_true_role_and_exclude_bans(corpus):
    """The new suite asks 'who fills this role?', not 'what is move m?'.

    build_prefix leaves unpicked slots at FLEX. If the pin were missing, this
    would silently collapse back into the next-move question with a different
    scorer; if ban moves were included, the target would have no true role.
    """
    rows, splits = corpus
    row = rows[splits["test"][0]]
    states = M.role_next_states([row], None, None)
    picks = [m for m, move in enumerate(row["moves"])
             if move[0] == M.MOVE_PICK]
    assert len(states) == len(picks) == 10
    for st, m in zip(states, picks):
        assert st["move"] == m
        assert st["slot"] == row["moves"][m][1] * 5 + row["moves"][m][3]
        assert st["target"] == row["moves"][m][2]
        sample = st["sample"]
        assert int(sample["role_ids"][st["slot"]]) == st["role"]
        assert int(sample["champ_ids"][st["slot"]]) == M.PRO_DATA.CHAMP_PAD_ID
        assert bool(sample["pad_mask"][st["slot"]])


def test_role_next_baseline_uses_only_train_picks(corpus):
    """A train-only (patch, role) table, with no test leakage."""
    rows, splits = corpus
    freq = M.role_next_frequency_tables(rows, splits["train"])
    by_pair, by_role = freq
    assert len(by_pair) > 50
    assert by_role.shape == (5, M.NUM_CHAMPS)
    first_patch = M.P.patch_key(rows[splits["train"][0]].get("patch"))
    expected = sum(1 for i in splits["train"]
                   for move in rows[i]["moves"]
                   if move[0] == M.MOVE_PICK
                   and M.P.patch_key(rows[i].get("patch")) == first_patch)
    assert by_pair[(first_patch, 0)].sum() + by_pair[(first_patch, 1)].sum() \
        + by_pair[(first_patch, 2)].sum() + by_pair[(first_patch, 3)].sum() \
        + by_pair[(first_patch, 4)].sum() == expected


def test_the_flat_scorer_ranks_inside_the_candidate_set(corpus):
    """A target outside the candidate list must raise, not rank 173rd.

    That is the failure mode of a ranking baseline: if the candidate set and
    the corpus disagree, every rank is silently too high and the model looks
    merely mediocre instead of broken.
    """
    from pro.model import baselines as BL

    rows, splits = corpus
    row = rows[splits["test"][0]]

    class OnlyTheFirst:
        def scores(self, states, candidate_lists):
            return [np.zeros(len(c), dtype=np.float32) for c in candidate_lists]

    states = [(row, m) for m in range(BL.N_MOVES)]
    games = M.score_flat(OnlyTheFirst(), "next_move", states, chunk=4)
    assert len(games) == 1 and len(games[0]["steps"]) == BL.N_MOVES
    # all candidates tie, so the strictly-greater rule gives every item rank 0
    assert all(s["rank"] == 0 for s in games[0]["steps"])

    broken = [(row, m) for m in range(BL.N_MOVES)]
    original = BL.candidates_for
    BL.candidates_for = lambda r, m: np.array([], dtype=np.int64)
    try:
        with pytest.raises(RuntimeError, match="not a candidate"):
            M.score_flat(OnlyTheFirst(), "next_move", broken[:1], chunk=4)
    finally:
        BL.candidates_for = original


# -------------------------------------------------------------------- report
def _phase(n=1, top5=0.1):
    return {p: {"n": n, "top1": 0.0, "top5": top5, "mean_rank": 5.0}
            for p in ("ban1", "pick1", "ban2", "pick2")}


def test_the_report_renders_all_sections_and_never_ranks_the_unverified():
    report = {
        "suite": {
            "split": {"corpus": "pro_drafts_full.pt", "corpus_rows": 46939,
                      "counts": {"train": 39720, "val": 4771, "test": 2448},
                      "shares": {"train": 0.8462, "val": 0.1016,
                                 "test": 0.0522},
                      "test_fingerprint": "abc123def4567890", "args": {},
                      "pinned_test": [], "max_date": "(none)"},
            "tasks": ["next_move"], "n_test_games": 2448, "limited": False,
            "baseline": True, "torch": "2", "seconds": 1.0,
        },
        "models": [],
        "next_move": {
            # The frequency table is a normal comparable row now, with its own
            # parameter count and its own paired delta.
            "00-frequency-table": {
                "name": "00-frequency-table", "comparable": True,
                "kind": "frequency", "params": 114180,
                "agg": {"n": 10, "top1": 0.1, "top5": 0.4, "top10": 0.6,
                        "mean_rank": 12.0, "top1_ci": [0.05, 0.2]},
                "by_phase": _phase(),
                "paired": {"vs": R.REFERENCE["next_move"], "this_only": 2,
                           "reference_only": 30, "delta_top1": -0.1,
                           "chi2": 22.0, "p": 1e-6, "n_paired": 10}},
            R.REFERENCE["next_move"]: {
                "name": R.REFERENCE["next_move"], "comparable": True,
                "kind": "policy", "params": 3350879,
                "agg": {"n": 10, "top1": 0.2, "top5": 0.5, "top10": 0.7,
                        "mean_rank": 10.0, "top1_ci": [0.1, 0.3]},
                "by_phase": _phase(top5=0.2)},
            "nextpick-old-arm": {
                "name": "nextpick-old-arm", "comparable": False,
                "kind": "policy", "params": 31897631,
                "split_note": "different split: train=42100",
                "agg": {"n": 10, "top1": 0.9, "top5": 0.95, "top10": 0.99,
                        "mean_rank": 1.0, "top1_ci": [0.8, 0.95]},
                "paired": {"vs": R.REFERENCE["next_move"], "this_only": 7,
                           "reference_only": 0, "delta_top1": 0.7,
                           "chi2": 5.0, "p": 0.01, "n_paired": 10}},
        },
        "role_next": {
            R.REFERENCE["role_next"]: {
                "name": R.REFERENCE["role_next"], "comparable": True,
                "kind": "role_next", "params": 25509037,
                "agg": {"n": 10, "top1": 0.2, "top5": 0.6, "top10": 0.8,
                        "mean_rank": 6.0, "top1_ci": [0.1, 0.3]}},
        },
    }
    md = REP.render(report)
    assert "### Comparable" in md
    assert "### Not comparable" in md
    assert "training-set numbers" in md

    # the frequency table is inside the comparable table, not a prose preamble
    comp = md.split("### Comparable")[1].split("### Not comparable")[0]
    assert "00-frequency-table" in comp
    assert "10.1%" in comp or "10.0%" in comp
    assert "-10.0" in comp                       # its paired delta vs the ref
    assert "114,180" in comp                     # and its size

    # the 90% row must appear, but with no delta: it is not ranked
    line = next(l for l in md.splitlines() if "nextpick-old-arm" in l)
    assert "90.0%" in line
    assert line.rstrip().endswith("| - | - | - |")
    # ...and it still states its size, which is not a claim about accuracy
    assert "31,897,631" in line

    # the split header is the manifest, not a hand-typed number
    assert "39,720 / 4,771 / 2,448".replace(",", "") in md.replace(",", "")

    # the third suite renders, with its own definition rather than a copy of
    # the next-move question
    assert "## Role-conditioned next pick" in md
    assert "true role of that pick" in md
    assert R.REFERENCE["role_next"] in md


def test_every_row_carries_a_param_count_or_an_explicit_dash():
    """A missing size must look missing, not like zero."""
    payload = {"name": "x", "comparable": True,
               "agg": {"n": 1, "top1": 0.5, "top5": 0.5, "top10": 0.5,
                       "mean_rank": 1.0, "top1_ci": [0.1, 0.9]}}
    assert REP.row("x", payload, None, False).split("|")[2].strip() == "-"
    payload["params"] = 42
    assert REP.row("x", payload, None, False).split("|")[2].strip() == "42"


# ------------------------------------------------- the flat baselines' wiring
def test_the_flat_layouts_are_the_widths_the_report_quotes():
    """A layout change silently invalidates every stored baseline."""
    from pro.model import baselines as BL

    assert BL.dim("next_move", "mlp") == BL.NM_DIM
    assert BL.dim("masked", "mlp") == BL.MK_DIM
    assert BL.dim("next_move", "forest") == BL.FN_DIM
    assert BL.dim("masked", "forest") == BL.FM_DIM
    # the one-hot slot block must be exactly wide enough for every champion
    # plus the "nothing here" index, or the highest champion id overruns it
    assert BL.NM["slot"][1] - BL.NM["slot"][0] == BL.N_SLOTS * (BL.NUM_CHAMPS + 1)
    assert BL.EMPTY == BL.NUM_CHAMPS


def test_a_flat_manifest_makes_a_directory_a_baseline(tmp_path):
    """`flat.json` is the discovery contract, not the file extension."""
    d = tmp_path / "nextpick-forest-flat-v9"
    d.mkdir()
    (d / "model.joblib").write_bytes(b"x")
    assert R.checkpoint_file(d) is None          # no torch weights here
    assert R.flat_manifest(d) is None
    (d / "flat.json").write_text(json.dumps(
        {"kind": "forest", "task": "next_move", "weights": "model.joblib",
         "params": 1234, "feature_version": 1}), encoding="utf-8")
    assert R.flat_manifest(d)["kind"] == "forest"
    (d / "test_report.json").write_text(json.dumps(
        {"protocol": {"n_train": 39720, "n_val": 4771, "n_test": 2448}}),
        encoding="utf-8")
    entry = R.Entry(name=d.name, path=d / "model.joblib", task="next_move",
                    track="pro", archived=False, kind="forest", params=1234)
    R._classify(entry, R._split_counts(d))
    assert entry.comparable


def test_the_reported_param_count_is_read_but_not_trusted():
    """`protocol.params` is for display; `suite` counts off the module."""
    assert R._reported_params(Path("/nonexistent")) is None


def test_the_baseline_parameter_budget_is_enforced():
    """Both baselines must stay under 3 M, or "smaller model" stops being true.

    The shipped policy model is 3.35 M, so a baseline at or under this budget
    is genuinely smaller and its shortfall cannot be explained by capacity.
    """
    from pro.model import baselines as BL

    assert BL.PARAM_BUDGET == 3_000_000
    assert BL.check_budget("ok", BL.PARAM_BUDGET) == BL.PARAM_BUDGET
    with pytest.raises(SystemExit, match="budget"):
        BL.check_budget("too-big", BL.PARAM_BUDGET + 1)


def test_params_for_matches_the_module_it_describes():
    """The budget is checked with a formula, so the formula must be the module.

    If `params_for` drifts from `FlatMLP`, `--hidden auto` silently picks a
    width that overruns the budget and the trainer's own check is computed off
    the same wrong number - i.e. it would pass.
    """
    from pro.model.mlp.model import FlatMLP, params_for

    for in_dim, hidden, depth in ((100, 32, 2), (1974, 64, 3), (3834, 40, 1),
                                  (3834, 8, 2)):
        module = FlatMLP(in_dim, hidden=hidden, depth=depth, dropout=0.0)
        assert sum(p.numel() for p in module.parameters()) == params_for(
            in_dim, hidden, depth=depth)


def test_the_mlp_width_fills_the_budget_without_breaking_it():
    """`--hidden auto` must be maximal, not merely legal.

    The two tasks have different input widths, so one shared width would leave
    one arm under-provisioned. Solving per task is the only way both arms get
    the same capacity.
    """
    from pro.model import baselines as BL
    from pro.model.mlp import model as MM

    widths = {}
    for task in ("next_move", "masked"):
        in_dim = BL.dim(task, "mlp")
        hidden = MM.largest_hidden(in_dim)
        assert MM.params_for(in_dim, hidden) <= BL.PARAM_BUDGET
        assert MM.params_for(in_dim, hidden + 1) > BL.PARAM_BUDGET
        assert MM.params_for(in_dim, hidden) > 0.9 * BL.PARAM_BUDGET
        widths[task] = hidden
    # different input widths, therefore different solved widths
    assert widths["next_move"] != widths["masked"]


# ------------------------------------------------- the trainers' mirror of it
def test_the_trainers_canonical_flag_matches_this_split():
    """`--canonical-split` in `proonly/common.py` must mean *this* split.

    Two definitions of one thing is how three incompatible protocols
    accumulated. If the trainer's constant and `split.SPLIT_ARGS` ever drift,
    a model trained with the flag would be labelled non-comparable by this
    suite - and the flag would be lying about what it does.
    """
    from pro.model.proonly import common as PC

    assert PC.CANONICAL_SPLIT["val_frac"] == S.SPLIT_ARGS["val_frac"]
    assert PC.CANONICAL_SPLIT["test_frac"] == S.SPLIT_ARGS["test_frac"]
    assert PC.CANONICAL_SPLIT["min_patch_rows"] == S.SPLIT_ARGS["min_patch_rows"]
    assert PC.CANONICAL_SPLIT["val_scope"] == "all"
    assert PC.CANONICAL_SPLIT["val_split"] == "tail"
    assert S.SPLIT_ARGS["val_random"] is False
    assert S.SPLIT_ARGS["val_patches"] is None
    assert S.SPLIT_ARGS["pinned_test"] == ()


def test_the_canonical_flag_refuses_what_would_change_the_rows():
    """`--no-test` and `--limit-drafts` cannot be reconciled with it."""
    from pro.model.proonly import common as PC

    class Args:
        pass

    a = Args()
    a.val_frac, a.test_frac, a.min_patch_rows = 0.2, 0.1, 10
    a.corpus, a.val_scope, a.val_split = "2022", "latest", "random"
    a.no_test = a.no_pin = a.limit_drafts = 0
    PC.apply_canonical_split(a)
    assert a.corpus == "2022-full" and a.val_frac == 0.10
    assert a.test_frac == 0.05 and a.no_pin is True
    assert a.val_scope == "all" and a.val_split == "tail"

    bad = Args()
    bad.val_frac = bad.test_frac = bad.min_patch_rows = 0
    bad.corpus = bad.val_scope = bad.val_split = ""
    bad.no_pin = bad.limit_drafts = 0
    bad.no_test = True
    with pytest.raises(SystemExit, match="no-test"):
        PC.apply_canonical_split(bad)

    bad.no_test = False
    bad.limit_drafts = 100
    with pytest.raises(SystemExit, match="limit-drafts"):
        PC.apply_canonical_split(bad)
