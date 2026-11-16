"""Render `reports/suite.json` into a markdown report.

Recomputes nothing. Every number in the output is read from the JSON, so the
report and the run cannot drift - the same rule the old `render_test_split.py`
followed, for the same reason.

Two tables per task, and the split is not cosmetic:

* **comparable** - the checkpoint's recorded training split is the canonical
  one, so these rows are a measurement of the model.
* **not comparable** - everything else, including every archived arm. Those
  models trained on rows that are in this test set, so their numbers are
  training-set numbers. They are printed because "what did the old arm say"
  is occasionally the question, but they are never ranked against the
  comparable rows and the column header says so.

Usage:
    python -m pro.model.test_split.report
    python -m pro.model.test_split.report --in reports/suite.json --out REPORT_suite.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "reports" / "suite.json"
DEFAULT_OUT = HERE / "REPORT_suite.md"

# `suite.BASELINE`, restated rather than imported: `report.py` reads only the
# JSON and must keep working on a run produced before a name change.
BASELINE = "00-frequency-table"

TASK_TITLE = {
    "next_move": "Next move",
    "masked": "Masked completion (the role board's question)",
}
TASK_QUESTION = {
    "next_move": "Given the first *m* moves, what is move *m*? 20 states per "
                 "game, prefix only.",
    "masked": "Given a complete draft with one pick hidden, who was it? 10 "
              "states per game; the other nine slots - including later picks - "
              "are visible.",
}


def pct(x, nd=1):
    return "%.*f%%" % (nd, 100 * x)


def ci(agg, key="top1"):
    lo, hi = agg["%s_ci" % key]
    return "[%.1f, %.1f]" % (100 * lo, 100 * hi)


def fmt_p(p):
    """A p that underflowed to 0 is not zero, it is below what a float holds."""
    if p is None:
        return "-"
    if p == 0.0:
        return "<1e-300"
    return "%.2g" % p


def delta_cell(payload, ref_name):
    p = payload.get("paired")
    if not p or p.get("delta_top1") is None:
        return "-", "-", "-"
    sign = "+" if p["delta_top1"] >= 0 else ""
    return ("%s%.1f" % (sign, 100 * p["delta_top1"]),
            "%d/%d" % (p["this_only"], p["reference_only"]),
            fmt_p(p["p"]))


def row(name, payload, ref_name, show_delta):
    agg = payload.get("agg")
    params = payload.get("params")
    par = "-" if params is None else format(int(params), ",")
    if not agg:
        # A traceback contains newlines, which would end the table row early.
        msg = " ".join(str(payload.get("error", "?")).split())[:140]
        return "| `%s` | %s | - | FAILED: %s | - | - | - | - | - | - | - |" % (
            name, par, msg)
    d, disc, p = delta_cell(payload, ref_name)
    if not show_delta:
        d = disc = p = "-"
    return "| `%s` | %s | %s | %s | %s | %s | %s | %.2f | %s | %s | %s |" % (
        name, par, format(agg["n"], ","), pct(agg["top1"]), ci(agg),
        pct(agg["top5"]), pct(agg["top10"]), agg["mean_rank"], d, disc, p)


HEADER = ("| model | params | n | top-1 | 95% CI | top-5 | top-10 | mean rank "
          "| Δ vs ref (pp) | discordant | p |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")

PARAMS_NOTE = (
    "`params` counts what the model stores. For anything with a weight vector "
    "(the set transformers, the flat MLP) it is `sum(p.numel())` off the loaded "
    "module. For the frequency table it is the table's own size - 20 moves x "
    "173 champions x one per test patch - because that is what it costs to "
    "hold. For the forest it is the total node count across all trees, a tree "
    "ensemble having no weight vector. The three are not the same kind of "
    "quantity and the column is there to size a model, not to rank one.")


def render(report):
    suite = report["suite"]
    sp = suite["split"]
    out = []
    w = out.append

    w("# Test-split evaluation\n")
    w("Every model class on the project's one held-out split. Generated from "
      "`reports/suite.json` by `report.py`; no number here is typed by hand.\n")

    w("## The split\n")
    w("| | |")
    w("|---|---|")
    w("| corpus | `%s` |" % sp["corpus"])
    w("| drafts | %d |" % sp["corpus_rows"])
    w("| split | `split_within_patch`, no pin, tail |")
    w("| train / val / test | %d / %d / %d |"
      % (sp["counts"]["train"], sp["counts"]["val"], sp["counts"]["test"]))
    w("| shares | %.2f%% / %.2f%% / %.2f%% |"
      % tuple(100 * sp["shares"][k] for k in ("train", "val", "test")))
    w("| test games scored | %d%s |"
      % (suite["n_test_games"], " (limited run)" if suite["limited"] else ""))
    w("| test fingerprint | `%s` |" % sp["test_fingerprint"][:16])
    w("")
    w("This is the split the shipped `-nov13-` pair was trained on, and "
      "`split.py --check` verifies it against `split_manifest.json` on every "
      "run. Nothing in this report is measured on a row any comparable model "
      "trained on.\n")

    for task in ("next_move", "masked"):
        block = report.get(task)
        if not block:
            continue
        w("## %s\n" % TASK_TITLE[task])
        w(TASK_QUESTION[task] + "\n")

        models = {k: v for k, v in block.items()}
        from pro.model.test_split.registry import REFERENCE
        ref_name = REFERENCE.get(task)
        if ref_name not in models:
            ref_name = None

        comparable = {k: v for k, v in models.items() if v.get("comparable")}
        others = {k: v for k, v in models.items() if not v.get("comparable")}

        w("### Comparable\n")
        if BASELINE in models:
            w("The first row is the reference point the question is asked "
              "against: a per-patch move-frequency table fitted on train rows "
              "only, ranked under the same legality filter every model gets. "
              "Its `Δ vs ref` is the incumbent's margin over a lookup table.\n")
        w(HEADER)
        for name, payload in comparable.items():
            w(row(name, payload, ref_name, show_delta=(name != ref_name)))

        if others:
            w("\n### Not comparable\n")
            w("Trained on a different split, so some of these test rows are in "
              "their training data. **These are training-set numbers.** Listed "
              "for provenance, not for ranking.\n")
            w(HEADER)
            for name, payload in others.items():
                w(row(name, payload, ref_name, show_delta=False))

        phases = {k: v for k, v in comparable.items() if v.get("by_phase")}
        if phases:
            w("\n### By phase\n")
            w("ban1 is where the board carries no information, so it is where "
              "a patch-conditioned model should separate from a frequency "
              "table - and where the table has historically won.\n")
            cols = ["ban1", "pick1", "ban2", "pick2"]
            w("| model | " + " | ".join(cols) + " |")
            w("|---|" + "---|" * len(cols))
            for name, payload in phases.items():
                cells = [pct(payload["by_phase"][c]["top5"]) for c in cols]
                w("| `%s` | %s |" % (name, " | ".join(cells)))
            w("\nCells are top-5.\n")

    w("\n## Reading it honestly\n")
    w(PARAMS_NOTE + "\n")
    w("The intervals are Wilson 95% on the pooled rate. Two rows whose "
      "intervals overlap are not separated by this split; the `Δ vs ref` "
      "column is the paired McNemar against the incumbent, which is the "
      "sharper test because it only counts items where the two models "
      "disagree. A `p` above 0.05 with a small `Δ` means \"not shown to be "
      "different\", not \"equal\".\n")
    w("The two tasks are never pooled: masked completion is an easier question "
      "by construction, so a single accuracy number across both families would "
      "rank the easier question higher rather than the better model.\n")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        raise SystemExit("no suite run at %s - run `python -m "
                         "pro.model.test_split.suite` first" % src)
    with open(src, encoding="utf-8") as f:
        report = json.load(f)
    md = render(report)
    out = Path(args.out)
    out.write_text(md, encoding="utf-8")
    print("wrote %s (%d lines)" % (out, md.count("\n")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
