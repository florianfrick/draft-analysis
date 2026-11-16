"""Score any checkpoint on the project's test split, one task at a time.

Two suites, because the two model families answer two different questions:

* `--task next_move` - "given the first m moves, what is move m?". Prefix only,
  20 states per game. Every `nextpick-*` checkpoint.
* `--task masked` - "given this complete draft with one pick hidden, who was
  it?". The other nine slots, including later picks, are visible. 10 states per
  game. Every `masking-*` checkpoint, which is also the model the role board
  asks.

They are never pooled. Masked completion is an easier question by construction
(it can see the answer's neighbourhood, and the role vector is given), so a
single accuracy number across both families would rank the easier question
higher rather than the better model.

What makes a run a comparison
-----------------------------
1. **Same rows.** `split.py` - 2,448 test games, 99 patches, nothing in train or
   val. Verified against `split_manifest.json` on every run.
2. **Same construction.** Both models are scored by `models.py`, which is the
   code the pre-`nov13` harness used. Legality is applied identically.
3. **Paired against the incumbent.** Every model is scored on the same items and
   compared to the reference with a McNemar on discordant pairs. Unpaired
   differences mostly measure which model drew the easier games.
4. **Split-labelled.** A checkpoint whose `test_report.json` records a different
   split is scored but marked, and never compared against the reference.

Usage:
    python -m pro.model.test_split.suite                    # both suites, full test set
    python -m pro.model.test_split.suite --task next_move
    python -m pro.model.test_split.suite --limit 200         # fast smoke run
    python -m pro.model.test_split.suite --include-archive   # the pre-nov13 arms too
    python -m pro.model.test_split.suite --list              # what would be scored
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.test_split import models as M  # noqa: E402
from pro.model.test_split import registry as R  # noqa: E402
from pro.model.test_split import split as S  # noqa: E402

KEY = {"next_move": "steps", "masked": "slots"}

# The frequency table is not a checkpoint, but it belongs in the same table as
# one: it is the reference point the question is asked against, and pairing it
# against the incumbent turns "the model beats a lookup table" from an assertion
# into a number with a p-value attached. It is named so it sorts above every
# `masking-`/`nextpick-` directory.
BASELINE = "00-frequency-table"


def id_to_name():
    name_to_idx = M.P.load_name_to_idx()[0]
    return {v: k for k, v in name_to_idx.items()}


def flat_states(task, test_rows):
    """Every state of every test game, in the shape `models.score_flat` wants."""
    if task == "next_move":
        return [(row, m) for row in test_rows for m in range(M.N_MOVES)]
    return [(row, s) for row in test_rows for s in range(M.N_SLOTS)]


def score_one(entry, task, test_rows, names):
    """Load, score, aggregate. Returns a dict of everything the row needs."""
    if entry.kind in ("mlp", "forest"):
        # The baseline's weights live beside its `flat.json`, and the loader
        # rebuilds the architecture from that manifest rather than guessing.
        predictor, params, kind = M.load_flat(entry.path.parent)
        games = M.score_flat(predictor, task, flat_states(task, test_rows),
                             names)
        return {"agg": M.aggregate(games, KEY[task]),
                "by_phase": M.by_phase(games) if task == "next_move" else None,
                "games": games, "params": int(params), "kind": kind,
                "n_patches": None}
    if task == "next_move":
        model, patch_index, cfg = M.load_policy(entry.path)
        games = M.score_next_move(model, patch_index, test_rows, names)
        return {"agg": M.aggregate(games, "steps"), "by_phase": M.by_phase(games),
                "games": games,
                "params": sum(p.numel() for p in model.parameters()),
                "kind": "policy", "n_patches": int(cfg.get("n_patches", 0))}
    enc, patch_index, cfg = M.load_masked(entry.path)
    games = M.score_masked(enc, patch_index, test_rows, names)
    return {"agg": M.aggregate(games, "slots"), "by_phase": None,
            "games": games,
            "params": sum(p.numel() for p in enc.parameters()),
            "kind": "masked", "n_patches": int(cfg.get("n_patches", 0))}


def baseline_next_move(rows, splits, test_rows, names):
    """Per-patch move-frequency table built from TRAIN rows only.

    Built per patch because the test set spans 99 patches and a single global
    table would be answering a different question on each. Backed off to the
    patch's own overall counts so it does not tie on a few hundred rows.

    Returns (games, cells): `cells` is the table's size - 20 moves x 173
    champions per patch - which is what it costs to store, and therefore the
    honest entry for its `params` column.
    """
    by_patch = defaultdict(list)
    for i in splits["train"]:
        by_patch[M.P.patch_key(rows[i].get("patch"))].append(rows[i])
    all_train = [rows[i] for i in splits["train"]]

    grouped = defaultdict(list)
    for row in test_rows:
        grouped[M.P.patch_key(row.get("patch"))].append(row)

    games = []
    for patch, patch_rows in sorted(grouped.items()):
        source = by_patch.get(patch) or all_train
        ranks, _n = M.popularity_reference(patch, source)
        games.extend(M.score_popularity(patch_rows, ranks))
    return games, M.N_MOVES * M.NUM_CHAMPS * len(grouped)


def fmt_row(name, agg, extra="", params=None):
    lo, hi = agg["top1_ci"]
    par = "           -" if params is None else "%12s" % format(int(params), ",")
    return ("  %-44s%s  top1 %5.1f%% [%4.1f,%4.1f]  top5 %5.1f%%  "
            "top10 %5.1f%%  rank %5.2f%s"
            % (name, par, 100 * agg["top1"], 100 * lo, 100 * hi,
               100 * agg["top5"], 100 * agg["top10"], agg["mean_rank"], extra))


def attach_paired(report, task, ref_name, ref_games, scored):
    """Second pass: the paired block needs every model's per-item results.

    Kept out of `run_task` so the scoring loop can stream - holding 2,448 games
    of per-state detail for six models is ~100 MB of Python dicts for a number
    that only needs the two booleans per item.

    The frequency table goes through this too. Pairing it against the incumbent
    is the point: "the model beats a lookup table" is an assertion until it has
    a discordant count next to it.
    """
    key = KEY[task]
    for name, payload in report.items():
        if name == ref_name or ref_games is None:
            continue
        games = scored.get(name)
        if games is None:
            continue
        a, b = M.paired_hits(games, ref_games, key)
        a_only, b_only, chi2, p = M.mcnemar(a, b)
        payload["paired"] = {
            "vs": ref_name, "n_paired": len(a),
            "this_only": a_only, "reference_only": b_only,
            "delta_top1": payload["agg"]["top1"] - (sum(b) / max(1, len(b))),
            "chi2": chi2, "p": p,
        }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=("next_move", "masked", "both"),
                    default="both")
    ap.add_argument("--limit", type=int, default=0,
                    help="score a random N of the test games (0 = all 2,448). "
                         "Random, not the first N: the test tail is ordered by "
                         "patch, so a prefix would be one patch's tail")
    ap.add_argument("--sample-games", type=int, default=0,
                    help="also store per-state detail for N randomly drawn "
                         "test games (0 = aggregates only)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --sample-games")
    ap.add_argument("--only", action="append", default=[],
                    help="score only this checkpoint directory (repeatable)")
    ap.add_argument("--include-archive", action="store_true",
                    help="also score the archived pre-nov13 checkpoints")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false",
                    help="skip the frequency-table baseline on next_move")
    ap.add_argument("--list", action="store_true",
                    help="print what would be scored and exit")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    tasks = ["next_move", "masked"] if args.task == "both" else [args.task]
    entries = []
    for task in tasks:
        entries += R.scan(include_archive=args.include_archive, task=task)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e.name in wanted]
        found = {e.name for e in entries}
        missing = wanted - found
        if missing:
            raise SystemExit("not found (or not scorable): %s\nknown:\n%s"
                             % (", ".join(sorted(missing)),
                                R.describe(R.scan(args.include_archive))))
    if not entries:
        raise SystemExit("no checkpoints to score")

    if args.list:
        print("the canonical split:")
        rows = S.load_rows(verbose=False)
        splits = S.build(rows, strict=False)
        print("  %d train / %d val / %d test  (test %d games)"
              % (len(splits["train"]), len(splits["val"]),
                 len(splits["test"]), len(splits["test"])))
        print("\nwould score:")
        print(R.describe(entries))
        return 0

    t_start = time.time()
    rows = S.load_rows(verbose=False)
    splits = S.build(rows)
    ok, diffs = S.verify(rows, splits)
    if not ok:
        raise SystemExit("the split does not match split_manifest.json:\n  "
                         + "\n  ".join(diffs)
                         + "\nEvery number produced now would be incomparable. "
                           "Fix the corpus, or re-baseline with "
                           "`python -m pro.model.test_split.split --write`.")
    print("split: %d train / %d val / %d test, fingerprint %s"
          % (len(splits["train"]), len(splits["val"]), len(splits["test"]),
             S.fingerprint(rows, splits["test"])[:12]))

    test_idx = splits["test"]
    if args.limit:
        test_idx = S.sample_games(rows, test_idx, args.limit, seed=args.seed)
        print("limited run: %d randomly drawn test games (seed %d)"
              % (len(test_idx), args.seed))
    test_rows = [rows[i] for i in test_idx]
    names = id_to_name()

    report = {
        "suite": {
            "split": S.summarise(rows, splits) | {"test_gameids": None},
            "tasks": tasks,
            "n_test_games": len(test_rows),
            "limited": bool(args.limit),
            "baseline": bool(args.baseline),
            "torch": torch.__version__,
            "seconds": None,
        },
        "models": [{"name": e.name, "task": e.task, "track": e.track,
                    "archived": e.archived, "comparable": e.comparable,
                    "kind": e.kind, "params": e.params,
                    "split_note": e.split_note,
                    "path": str(e.path.relative_to(ROOT))} for e in entries],
    }

    for task in tasks:
        task_entries = [e for e in entries if e.task == task]
        if not task_entries:
            continue
        ref = R.reference(task, task_entries)
        ref_games, scored = None, {}
        report[task] = {}
        print("\n== %s ==  (%d games, %d items/model)"
              % (task, len(test_rows),
                 len(test_rows) * (M.N_MOVES if task == "next_move"
                                   else M.N_SLOTS)))
        print("  %-44s%12s  %s" % ("model", "params", "accuracy"))

        if task == "next_move" and args.baseline:
            base_games, cells = baseline_next_move(rows, splits, test_rows,
                                                   names)
            base = {"name": BASELINE, "path": None, "track": "baseline",
                    "archived": False, "comparable": True, "split_note": "",
                    "kind": "frequency", "params": cells, "baseline": True,
                    "n_patches": None, "seconds": 0.0,
                    "split_counts": {k: len(v) for k, v in splits.items()},
                    "agg": M.aggregate(base_games, "steps"),
                    "by_phase": M.by_phase(base_games)}
            report[task][BASELINE] = base
            scored[BASELINE] = base_games
            print(fmt_row(BASELINE, base["agg"], params=cells))

        for entry in task_entries:
            t0 = time.time()
            try:
                out = score_one(entry, task, test_rows, names)
            except Exception as exc:                   # noqa: BLE001
                # One unloadable checkpoint must not take the leaderboard with
                # it: the whole point of scanning a directory is that a future
                # model can be dropped in, and a future model is exactly the
                # thing most likely to have a shape this loader has not seen.
                print("  %-44s FAILED (%s: %s)"
                      % (entry.name, type(exc).__name__, exc))
                report[task][entry.name] = {
                    "name": entry.name, "error": "%s: %s"
                    % (type(exc).__name__, exc),
                    "path": str(entry.path.relative_to(ROOT)),
                    "kind": entry.kind, "params": entry.params,
                    "comparable": entry.comparable,
                    "split_note": entry.split_note}
                continue
            elapsed = time.time() - t0
            games = out.pop("games")
            payload = {
                "name": entry.name,
                "path": str(entry.path.relative_to(ROOT)),
                "track": entry.track, "archived": entry.archived,
                "comparable": entry.comparable, "split_note": entry.split_note,
                "split_counts": entry.split_counts,
                "kind": out.pop("kind"), "params": out.pop("params"),
                "n_patches": out.pop("n_patches"),
                "seconds": round(elapsed, 1), **out,
            }
            if args.sample_games:
                want = set(S.sample_games(rows, test_idx, args.sample_games,
                                          seed=args.seed))
                payload["sample"] = [g for g in games
                                     if g["gameid"] in want]
            report[task][entry.name] = payload
            if entry.is_reference:
                ref_games = games
            scored[entry.name] = games
            print(fmt_row(entry.label(), payload["agg"],
                          "  %5.1fs" % elapsed, params=payload["params"]))

        attach_paired(report[task], task, ref.name if ref else None,
                      ref_games, scored)
        for name, payload in report[task].items():
            p = payload.get("paired")
            if p and p["delta_top1"] is not None:
                print("  %-44s %+5.2f pp vs reference (p %.2g, %d/%d discordant)"
                      % (name, 100 * p["delta_top1"], p["p"],
                         p["this_only"], p["reference_only"]))

    report["suite"]["seconds"] = round(time.time() - t_start, 1)
    out = Path(args.out) if args.out else (
        HERE / "reports" / ("suite%s.json" % ("" if args.task == "both"
                                              else "_" + args.task)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("\nwrote %s  (%.0fs)" % (out, report["suite"]["seconds"]))
    return report


if __name__ == "__main__":
    main()
