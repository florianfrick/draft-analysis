"""Render `reports/test_split.json` into a standalone HTML report + markdown.

The JSON is the evidence; this file only formats it. Nothing is recomputed
here except counts that are trivially derivable (how many steps a model hit),
so the report and the JSON cannot disagree about a number.

Usage:
    python model/test_split/render_test_split.py
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "reports" / "test_split.json"
OUT_HTML = HERE / "REPORT_test_split_evaluation.html"
OUT_MD = HERE / "REPORT_test_split_evaluation.md"

PHASE_OF = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4
PHASE_LABEL = {0: "ban 1", 1: "pick 1", 2: "ban 2", 3: "pick 2"}
PHASE_SHORT = {0: "B1", 1: "P1", 2: "B2", 3: "P2"}


def esc(s):
    return html.escape(str(s))


def fmt_pct(x):
    return "%.0f%%" % (100 * x)


def badge(rank):
    """Rank -> (css class, label)."""
    if rank == 0:
        return "hit1", "1st"
    if rank < 5:
        return "hit5", "%dth" % (rank + 1) if rank + 1 != 2 else "2nd"
    return "miss", "%d" % (rank + 1)


def ordinal(n):
    if 10 <= n % 100 <= 20:
        return "%dth" % n
    return "%d%s" % (n, {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


# --------------------------------------------------------------------- pieces
def draft_board(game):
    """Two columns of bans + picks, in move order, as HTML."""
    bans = {"blue": [], "red": []}
    picks = {"blue": [], "red": []}
    for mv in game["moves"]:
        if mv["kind"] == "ban":
            bans[mv["side"]].append(mv)
        else:
            picks[mv["side"]].append(mv)

    def chips(items, kind):
        out = []
        for mv in items:
            role = ('<span class="role">%s</span>' % esc(mv["role"])
                    if mv["role"] else "")
            out.append('<span class="chip %s">%s%s</span>'
                       % (kind, esc(mv["champ"]), role))
        return "".join(out)

    rows = []
    for side, label, team in (("blue", "BLUE", game["blue_team"]),
                              ("red", "RED", game["red_team"])):
        rows.append(
            '<tr><th class="side %s">%s<span class="team">%s</span></th>'
            '<td>%s</td><td>%s</td></tr>'
            % (side, label, esc(team), chips(picks[side], "pick"),
               chips(bans[side], "ban")))
    return ('<table class="board"><thead><tr><th></th>'
            '<th>picks (in order)</th><th>bans (in order)</th></tr></thead>'
            '<tbody>%s</tbody></table>' % "".join(rows))


def step_table(report, game):
    """One row per move, one column per next-move model."""
    models = list(report["next_move"].items())
    head = ['<th class="mv">#</th>', '<th>phase</th>', '<th>side</th>',
            '<th>actual</th>']
    for name, _d in models:
        head.append('<th class="model">%s</th>' % esc(name))
    head.append('<th class="model base">patch table</th>')

    gid = game["gameid"]
    body = []
    for mv in game["moves"]:
        m = mv["move"]
        cells = ['<td class="mv">%d</td>' % m,
                 '<td class="ph p%d">%s</td>' % (PHASE_OF[m],
                                                 PHASE_LABEL[PHASE_OF[m]]),
                 '<td class="side-cell %s">%s</td>' % (mv["side"],
                                                       mv["side"].upper()),
                 '<td class="actual">%s%s</td>'
                 % (esc(mv["champ"]),
                    ' <span class="role">%s</span>' % esc(mv["role"])
                    if mv["role"] else "")]
        for name, d in models:
            g = next(x for x in d["games"] if x["gameid"] == gid)
            s = g["steps"][m]
            cls, lab = badge(s["rank"])
            tip = "%s | top-5: %s | P(actual)=%.3f" % (
                s["top1_name"], ", ".join(s["top5_names"]), s["prob"])
            cells.append('<td class="pred %s" title="%s">'
                         '<span class="pick">%s</span>'
                         '<span class="rank">%s</span></td>'
                         % (cls, esc(tip), esc(s["top1_name"]), lab))
        b = next(x for x in report["baseline"]["games"]
                 if x["gameid"] == gid)["steps"][m]
        cls, lab = badge(b["rank"])
        cells.append('<td class="pred base %s"><span class="pick">—</span>'
                     '<span class="rank">%s</span></td>' % (cls, lab))
        body.append("<tr>%s</tr>" % "".join(cells))
    return ('<table class="steps"><thead><tr>%s</tr></thead><tbody>%s</tbody>'
            '</table>' % ("".join(head), "".join(body)))


def masked_table(report, game):
    models = list(report["masked"].items())
    head = ['<th class="mv">mv</th>', '<th>role</th>', '<th>actual</th>']
    for name, _d in models:
        head.append('<th class="model">%s</th>' % esc(name))
    gid = game["gameid"]
    body = []
    for i, ref in enumerate(next(iter(report["masked"].values()))["games"]):
        if ref["gameid"] != gid:
            continue
        for j, s in enumerate(ref["slots"]):
            cells = ['<td class="mv">%s</td>' % s["move"],
                     '<td class="ph">%s</td>' % esc(s["role"]),
                     '<td class="actual">%s</td>' % esc(s["actual"])]
            for name, d in models:
                x = d["games"][i]["slots"][j]
                cls, lab = badge(x["rank"])
                tip = "%s | top-5: %s" % (x["top1_name"],
                                          ", ".join(x["top5_names"]))
                cells.append('<td class="pred %s" title="%s">'
                             '<span class="pick">%s</span>'
                             '<span class="rank">%s</span></td>'
                             % (cls, esc(tip), esc(x["top1_name"]), lab))
            body.append("<tr>%s</tr>" % "".join(cells))
    return ('<table class="steps masked"><thead><tr>%s</tr></thead>'
            '<tbody>%s</tbody></table>' % ("".join(head), "".join(body)))


def scoreboard(report):
    rows = []
    for name, d in report["next_move"].items():
        a = d["agg"]
        ph = d["by_phase"]
        rows.append(
            '<tr><td class="name">%s</td><td class="fam">%s</td>'
            '<td class="num">%s</td><td class="num">%s</td>'
            '<td class="num dim">%.1f</td>%s</tr>'
            % (esc(name), esc(d["family"]),
               '<b>%s</b>' % fmt_pct(a["top1"]) if a["top1"] > 0 else
               fmt_pct(a["top1"]),
               wilson_html(a), a["mean_rank"],
               "".join('<td class="num ph">%s</td>'
                       % fmt_pct(ph[k]["top5"])
                       for k in ("ban1", "pick1", "ban2", "pick2"))))
    b = report["baseline"]
    rows.append(
        '<tr class="baseline"><td class="name">patch table</td>'
        '<td class="fam">%s</td><td class="num">%s</td><td class="num">%s</td>'
        '<td class="num dim">%.1f</td>%s</tr>'
        % (esc(b["name"]), fmt_pct(b["agg"]["top1"]),
           wilson_html(b["agg"]),
           b["agg"]["mean_rank"],
           "".join('<td class="num ph">%s</td>' % fmt_pct(b["by_phase"][k]["top5"])
                   for k in ("ban1", "pick1", "ban2", "pick2"))))
    return ('<table class="score"><thead><tr><th>model</th><th>training</th>'
            '<th>top-1</th><th>top-5</th><th>mean rank</th>'
            '<th class="ph">ban1</th><th class="ph">pick1</th>'
            '<th class="ph">ban2</th><th class="ph">pick2</th></tr></thead>'
            '<tbody>%s</tbody></table>' % "".join(rows))


def masked_scoreboard(report):
    rows = []
    for name, d in report["masked"].items():
        a = d["agg"]
        rows.append('<tr><td class="name">%s</td><td class="fam">%s</td>'
                    '<td class="num">%s</td><td class="num">%s</td>'
                    '<td class="num dim">%.1f</td></tr>'
                    % (esc(name), esc(d["family"]), fmt_pct(a["top1"]),
                       wilson_html(a), a["mean_rank"]))
    return ('<table class="score"><thead><tr><th>model</th><th>training</th>'
            '<th>top-1</th><th>top-5 (95%% CI)</th><th>mean rank</th></tr>'
            '</thead><tbody>%s</tbody></table>' % "".join(rows))


CSS = """
:root{
  --bg:#0d1117; --panel:#151b23; --panel2:#1b222c; --line:#283039;
  --ink:#e6edf3; --dim:#8b98a5; --accent:#58a6ff;
  --hit1:#3fb950; --hit5:#d29922; --miss:#6e7b8a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  padding:40px 32px 80px}
.wrap{max-width:1280px;margin:0 auto}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.3px}
h2{font-size:17px;margin:44px 0 12px;padding-bottom:7px;
  border-bottom:1px solid var(--line);font-weight:600}
h3{font-size:14px;margin:26px 0 8px;color:var(--dim);font-weight:600;
  text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--dim);margin:0 0 22px}
p{margin:9px 0}
a{color:var(--accent)}
code{background:var(--panel2);padding:1px 5px;border-radius:4px;
  font-size:12.5px;font-family:ui-monospace,Consolas,monospace}
.prov{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:16px 18px;margin:18px 0}
.prov dl{display:grid;grid-template-columns:150px 1fr;gap:5px 14px;margin:0}
.prov dt{color:var(--dim)}
.prov dd{margin:0}
.board{width:100%;border-collapse:collapse;margin:8px 0 4px}
.board th,.board td{border-bottom:1px solid var(--line);padding:7px 8px;
  text-align:left;vertical-align:top}
.board thead th{font-size:11px;color:var(--dim);text-transform:uppercase;
  letter-spacing:.06em;font-weight:600}
.side{font-size:11px;font-weight:700;letter-spacing:.06em;width:120px}
.side.blue{color:#58a6ff}
.side.red{color:#f85149}
.side .team{display:block;color:var(--dim);font-weight:400;
  text-transform:none;letter-spacing:0;font-size:11.5px}
.chip{display:inline-block;background:var(--panel2);border:1px solid var(--line);
  border-radius:999px;padding:2px 9px;margin:0 4px 4px 0;font-size:12.5px}
.chip.ban{color:var(--dim)}
.chip .role{color:var(--dim);font-size:10.5px;margin-left:5px;
  letter-spacing:.04em}
table.score{width:100%;border-collapse:collapse;font-size:13px}
table.score th{font-size:11px;color:var(--dim);text-transform:uppercase;
  letter-spacing:.06em;font-weight:600;text-align:right;padding:6px 9px;
  border-bottom:1px solid var(--line)}
table.score th:first-child,table.score th:nth-child(2){text-align:left}
table.score td{padding:6px 9px;border-bottom:1px solid var(--line);
  text-align:right;font-variant-numeric:tabular-nums}
table.score td.name{text-align:left;font-family:ui-monospace,Consolas,monospace;
  font-size:12.5px}
table.score td.fam{text-align:left;color:var(--dim);font-size:12px}
table.score tr.baseline td{border-top:1px solid var(--line);color:var(--dim)}
.dim{color:var(--dim)}
table.steps{width:100%;border-collapse:collapse;font-size:12.5px}
table.steps th{font-size:10.5px;color:var(--dim);text-transform:uppercase;
  letter-spacing:.05em;font-weight:600;padding:5px 6px;text-align:left;
  border-bottom:1px solid var(--line);white-space:nowrap}
table.steps th.model{font-family:ui-monospace,Consolas,monospace;
  text-transform:none;letter-spacing:0;font-size:11px}
table.steps td{padding:4px 6px;border-bottom:1px solid rgba(40,48,57,.55);
  vertical-align:middle;white-space:nowrap}
td.mv{color:var(--dim);font-variant-numeric:tabular-nums;width:26px}
td.ph{font-size:10.5px;color:var(--dim);letter-spacing:.04em}
td.ph.p0{color:#a371f7}td.ph.p1{color:#58a6ff}
td.ph.p2{color:#a371f7}td.ph.p3{color:#58a6ff}
td.side-cell{font-size:10.5px;font-weight:700;letter-spacing:.05em}
td.side-cell.blue{color:#58a6ff}td.side-cell.red{color:#f85149}
td.actual{font-weight:600}
td.pred{position:relative;padding-right:34px}
td.pred .pick{display:block;font-size:12px}
td.pred .rank{position:absolute;right:5px;top:50%;transform:translateY(-50%);
  font-size:10px;font-variant-numeric:tabular-nums;border-radius:4px;
  padding:1px 4px;background:var(--panel2);color:var(--miss)}
td.pred.hit1{background:rgba(63,185,80,.13)}
td.pred.hit1 .rank{background:rgba(63,185,80,.25);color:var(--hit1);
  font-weight:700}
td.pred.hit1 .pick{color:#7ee787}
td.pred.hit5{background:rgba(210,153,34,.09)}
td.pred.hit5 .rank{background:rgba(210,153,34,.2);color:var(--hit5)}
td.pred.base .pick{color:var(--dim)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:14px;margin:12px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:12px 14px}
.card-h{display:flex;justify-content:space-between;align-items:baseline;
  margin-bottom:4px}
.card-h b{font-family:ui-monospace,Consolas,monospace;font-size:13px}
.card-h span{color:var(--dim);font-size:11.5px}
.card-f{color:var(--dim);font-size:12px;margin-top:4px}
svg.curve{width:100%;height:auto;display:block}
svg.curve .grid{stroke:#2a323c;stroke-width:1}
svg.curve .grid50{stroke:#4a5563;stroke-width:1;stroke-dasharray:2 3}
svg.curve .axlbl{fill:#6e7b8a;font-size:7.5px}
svg.curve polyline{fill:none;stroke-width:1.8;stroke-linejoin:round}
svg.curve polyline.g1{stroke:#58a6ff}
svg.curve polyline.g2{stroke:#d29922}
svg.curve circle.g1{fill:#58a6ff}svg.curve circle.g2{fill:#d29922}
.legend{color:var(--dim);font-size:11.5px;margin:2px 0 10px}
.legend i{display:inline-block;width:16px;height:2px;vertical-align:middle;
  margin:0 5px 0 0}
.legend i.g1{background:#58a6ff}.legend i.g2{background:#d29922}
.note{background:var(--panel);border-left:2px solid var(--accent);
  border-radius:0 8px 8px 0;padding:12px 16px;margin:14px 0;color:var(--dim)}
.note b{color:var(--ink)}
.note.warn{border-left-color:var(--hit5)}
ul{margin:8px 0;padding-left:20px}li{margin:5px 0}
.kv{font-variant-numeric:tabular-nums}
table.score.appendix td{text-align:left}
table.score.appendix td.path code{font-size:11px;color:var(--dim)}
table.score.appendix tr td:nth-child(5){text-align:right}
"""


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    Reported on every rate in this report because the denominator is 7 games.
    Closed form and deterministic - no bootstrap, so re-rendering the same
    JSON cannot move a number.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    m = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def wilson_cell(agg, key="top5"):
    """'48% (40-56%)' from an aggregate block."""
    n = agg["n"]
    k = int(round(agg[key] * n))
    lo, hi = wilson(k, n)
    return "%s (%s-%s)" % (fmt_pct(agg[key]), fmt_pct(lo), fmt_pct(hi))


def wilson_html(agg, key="top5"):
    """The same, with the bracket dimmed for the HTML table."""
    text = wilson_cell(agg, key)
    rate, _, bracket = text.partition(" ")
    return '%s <span class="dim">%s</span>' % (rate, bracket)


def game_label(game, idx):
    """'BLG vs HLE g2' from a series_id, falling back to the game index."""
    sid = game.get("series_id") or ""
    parts = sid.split("|")
    if len(parts) >= 3:
        return "%s vs %s g%d" % (parts[1], parts[2], game["game_number"])
    return "%s g%d" % (game.get("blue_team", "?"), game["game_number"])


def fearless_leaks(report):
    """Masked slots whose top-5 names a champion already picked in an
    earlier game of the same series.

    Computed rather than hardcoded: the count is a product of how many masked
    models are in the registry and how many games carry a fearless context, so
    it changes whenever either does. `ProMaskedDraftDataset` never receives
    `unavailable`, so this is the scope limit of the masked task as trained.
    """
    games = report["series"]["games"]
    leaks = total = 0
    for gi, game in enumerate(games):
        unavailable = set(game.get("unavailable") or ())
        if not unavailable:
            continue
        for d in report["masked"].values():
            for s in d["games"][gi]["slots"]:
                total += 1
                if unavailable & set(s.get("top5_names", [])):
                    leaks += 1
    return leaks, total


def provenance(report):
    t = report["test_split"]
    rows = [
        ("corpus", "<code>pro_drafts.pt</code>, pinned at "
                   "<code>CORPUS_MAX_DATE = %s</code>" % esc(t["corpus_max_date"])),
        ("split", "<code>pipeline.split_within_patch</code> with "
                  "<code>pinned_test = TEST_SERIES</code>"),
        ("test series", "%d, pinned by <code>series_id</code>" % len(t["series"])),
        ("test games", "%d, patch %s, %s" % (t["n_games"], esc(t["patch"]),
                                             "fearless" if t["fearless"]
                                             else "not fearless")),
        ("train / val", "the head and tail of every patch, minus the pinned "
                        "series - never these games"),
        ("baseline", "patch %s move-frequency table, %d corpus rows"
                     % (esc(t["patch"]), report["baseline"]["n_rows"])),
    ]
    return ('<div class="prov"><dl>%s</dl></div>'
            % "".join("<dt>%s</dt><dd>%s</dd>" % (k, v) for k, v in rows))


def series_table(report):
    rows = []
    for s in report["test_split"]["series"]:
        rows.append('<tr><td class="name">%s</td><td class="fam">%s</td>'
                    '<td class="num">%d</td><td class="num">%d</td></tr>'
                    % (esc(" vs ".join(s["teams"])), esc(s["date"]),
                       s["n_games"], s["blue_wins"]))
    return ('<table class="score"><thead><tr><th>series</th><th>date</th>'
            '<th>games</th><th>blue wins</th></tr></thead>'
            '<tbody>%s</tbody></table>' % "".join(rows))


def class_table(report):
    rows = []
    for group, items in (("next move", report["next_move"]),
                         ("masked", report["masked"])):
        for name, d in items.items():
            rows.append(
                '<tr><td class="name">%s</td><td class="fam">%s</td>'
                '<td class="fam">%s</td><td class="fam"><code>%s</code></td>'
                '</tr>'
                % (esc(name), esc(d.get("family", "")), group, esc(d["path"])))
    return ('<table class="score appendix"><thead><tr><th>checkpoint</th>'
            '<th>training</th><th>answers</th><th>path</th></tr></thead>'
            '<tbody>%s</tbody></table>' % "".join(rows))


def build_html(report):
    t = report["test_split"]
    games = report["series"]["games"]
    n_move_models = len(report["next_move"])
    n_mask_models = len(report["masked"])
    leaks, leak_total = fearless_leaks(report)

    # what a coin-flip-looking interval looks like at this sample size
    lo, hi = wilson(0, len(games))

    H = []
    H.append('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>Test split evaluation</title><style>%s</style></head><body>'
             '<div class="wrap">' % CSS)

    H.append('<h1>Test split evaluation</h1>')
    H.append('<p class="sub">Every model class, scored on the same %d games. '
             'This is the one split nothing trains on.</p>' % t["n_games"])
    H.append(provenance(report))

    H.append('<h2>1. The split</h2>')
    H.append('<p>Three series, %d games, all patch %s, all played on %s. They '
             'are the last three series of Worlds 2026, pinned by '
             '<code>series_id</code> in <code>pipeline.TEST_SERIES</code> so '
             'the test set does not move when the corpus is rebuilt. A '
             'fractional tail inside a patch would be a different set of games '
             'every time; a test split that moves is not a test split.</p>'
             % (t["n_games"], esc(t["patch"]), esc(t["series"][0]["date"])))
    H.append(series_table(report))

    H.append('<h2>2. What is compared</h2>')
    H.append('<p>%d next-move checkpoints and %d masked ones, one per model '
             'class. The directory name states the inference: '
             '<code>masking-*</code> completes a draft and answers the role '
             'board, <code>nextpick-*</code> ranks the next move. '
             '<code>masking-soloqueue</code> is not a class of its own - it is '
             'the encoder <code>masking-finetune</code> starts from, kept as '
             'the reference that says whether fine-tuning on pro drafts bought '
             'anything.</p>' % (n_move_models, n_mask_models))
    H.append(class_table(report))

    H.append('<h2>3. Next move</h2>')
    H.append('<p>20 states per game, %d in total. Top-5 counts how often the '
             'real move was in the model\'s five highest-ranked legal '
             'champions. The bracket is a 95%% Wilson interval on that count, '
             'and it is the most important column here.</p>'
             % (len(games) * 20))
    H.append(scoreboard(report))
    H.append('<p class="legend">top-1 / top-5 / mean rank over all moves. '
             'ban1-pick2 are top-5 within each draft phase.</p>')

    H.append('<h2>4. Masked completion</h2>')
    H.append('<p>Each of a game\'s ten picks is hidden in turn while the other '
             'nine stay visible, so %d slots per model. The models are asked '
             'the question the role board asks.</p>' % (len(games) * 10))
    H.append(masked_scoreboard(report))

    if leaks:
        H.append('<div class="note warn"><b>The masked models cannot express '
                 'fearless.</b> On the games that carry a fearless context, '
                 '%d of %d masked slots put a champion already picked earlier '
                 'in the series into their top-5. The next-move models do get '
                 '<code>unavailable_mask</code> and break fearless zero times. '
                 'That is a scope limit of the masked task as trained, not a '
                 'bug in this harness.</div>' % (leaks, leak_total))

    H.append('<h2>5. How much seven games can say</h2>')
    H.append('<div class="note warn"><b>Almost nothing, on its own.</b> At '
             '%d games one game is %s of the sample, and a 95%% interval on a '
             'rate observed over seven games is roughly %s wide. Any two '
             'models whose intervals overlap are tied as far as this split can '
             'tell, and at this n most of them will. The intervals are printed '
             'next to every rate for exactly that reason.</div>'
             % (t["n_games"], fmt_pct(1.0 / t["n_games"]),
                fmt_pct(hi - lo)))
    H.append('<p>What the split <em>can</em> do is catch a model that is '
             'broken or badly fitted - a next-move model at or below the '
             'frequency table, a masked model that cannot name the champion '
             'at all. It is a smoke test on genuinely unseen games, not a '
             'ranking. For the ranking, the within-patch validation split is '
             'the number to read: it is 100x larger and every model was '
             'selected on it.</p>')
    H.append('<p>The one comparison that survives the sample size is the '
             'per-state one, because every model is scored on the identical '
             'board. That is what the step tables below are for: not "which '
             'model wins" but "here is what each one said at this state, and '
             'here is what happened".</p>')

    H.append('<h2>6. Every state</h2>')
    for i, game in enumerate(games):
        H.append('<h3>%s &middot; %s &middot; %s</h3>'
                 % (esc(game_label(game, i)), esc((game["date"] or "")[:10]),
                    "blue won" if game["blue_win"] else "red won"))
        H.append(draft_board(game))
        H.append(step_table(report, game))
        H.append(masked_table(report, game))

    H.append('<h2>Reproduce</h2>')
    H.append('<pre><code>'
             'docker compose run --rm -T dev python model/test_split/eval_test_split.py\n'
             'docker compose run --rm -T dev python model/test_split/render_test_split.py\n'
             'docker compose run --rm -T dev python -m pytest model/test_split/test_test_split.py -q'
             '</code></pre>')
    H.append('</div></body></html>')
    return "\n".join(H)


def build_md(report):
    t = report["test_split"]
    games = report["series"]["games"]
    leaks, leak_total = fearless_leaks(report)
    lo, hi = wilson(0, len(games))

    L = ["# Test split evaluation", ""]
    L.append("Every model class, scored on the same %d games. This is the one "
             "split nothing trains on." % t["n_games"])
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| corpus | `pro_drafts.pt`, pinned at `CORPUS_MAX_DATE = %s` |"
             % t["corpus_max_date"])
    L.append("| split | `pipeline.split_within_patch` with `pinned_test = "
             "TEST_SERIES` |")
    L.append("| test series | %d, pinned by `series_id` |" % len(t["series"]))
    L.append("| test games | %d, patch %s |" % (t["n_games"], t["patch"]))
    L.append("| baseline | patch %s move-frequency table, %d corpus rows |"
             % (t["patch"], report["baseline"]["n_rows"]))
    L.append("")

    L.append("## 1. The split")
    L.append("")
    L.append("Three series, %d games, all patch %s, all played on %s. The last "
             "three series of Worlds 2026, pinned by `series_id` in "
             "`pipeline.TEST_SERIES` so the test set does not move when the "
             "corpus is rebuilt."
             % (t["n_games"], t["patch"], t["series"][0]["date"]))
    L.append("")
    L.append("| series | date | games | blue wins |")
    L.append("|---|---|---|---|")
    for s in t["series"]:
        L.append("| %s | %s | %d | %d |"
                 % (" vs ".join(s["teams"]), s["date"], s["n_games"],
                    s["blue_wins"]))
    L.append("")

    L.append("## 2. What is compared")
    L.append("")
    L.append("| checkpoint | training | answers | path |")
    L.append("|---|---|---|---|")
    for group, items in (("next move", report["next_move"]),
                         ("masked", report["masked"])):
        for name, d in items.items():
            L.append("| `%s` | %s | %s | `%s` |"
                     % (name, d.get("family", ""), group, d["path"]))
    L.append("")
    L.append("`masking-soloqueue` is not a class of its own - it is the "
             "encoder `masking-finetune` starts from, kept as the reference "
             "that says whether fine-tuning on pro drafts bought anything.")
    L.append("")

    L.append("## 3. Next move")
    L.append("")
    L.append("20 states per game, %d in total." % (len(games) * 20))
    L.append("")
    L.append("| model | training | top-1 | top-5 (95% CI) | mean rank "
             "| ban1 | pick1 | ban2 | pick2 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, d in report["next_move"].items():
        a, ph = d["agg"], d["by_phase"]
        L.append("| `%s` | %s | %s | %s | %.1f | %s | %s | %s | %s |"
                 % (name, d.get("family", ""), fmt_pct(a["top1"]),
                    wilson_cell(a), a["mean_rank"],
                    fmt_pct(ph["ban1"]["top5"]), fmt_pct(ph["pick1"]["top5"]),
                    fmt_pct(ph["ban2"]["top5"]), fmt_pct(ph["pick2"]["top5"])))
    b = report["baseline"]
    L.append("| patch table | %s | %s | %s | %.1f | %s | %s | %s | %s |"
             % (b["name"], fmt_pct(b["agg"]["top1"]),
                wilson_cell(b["agg"]), b["agg"]["mean_rank"],
                fmt_pct(b["by_phase"]["ban1"]["top5"]),
                fmt_pct(b["by_phase"]["pick1"]["top5"]),
                fmt_pct(b["by_phase"]["ban2"]["top5"]),
                fmt_pct(b["by_phase"]["pick2"]["top5"])))
    L.append("")

    L.append("## 4. Masked completion")
    L.append("")
    L.append("Each of a game's ten picks hidden in turn, %d slots per model."
             % (len(games) * 10))
    L.append("")
    L.append("| model | training | top-1 | top-5 (95% CI) | mean rank |")
    L.append("|---|---|---|---|---|")
    for name, d in report["masked"].items():
        a = d["agg"]
        L.append("| `%s` | %s | %s | %s | %.1f |"
                 % (name, d.get("family", ""), fmt_pct(a["top1"]),
                    wilson_cell(a), a["mean_rank"]))
    L.append("")

    if leaks:
        L.append("**The masked models cannot express fearless.** On the games "
                 "that carry a fearless context, %d of %d masked slots put a "
                 "champion already picked earlier in the series into their "
                 "top-5. The next-move models do get `unavailable_mask` and "
                 "break fearless zero times. Scope limit of the masked task as "
                 "trained, not a bug in this harness." % (leaks, leak_total))
        L.append("")

    L.append("## 5. How much seven games can say")
    L.append("")
    L.append("**Almost nothing, on its own.** At %d games one game is %s of "
             "the sample, and a 95%% interval on a rate observed over seven "
             "games is roughly %s wide. Any two models whose intervals overlap "
             "are tied as far as this split can tell, and at this n most of "
             "them will. The intervals are printed next to every rate for "
             "exactly that reason."
             % (t["n_games"], fmt_pct(1.0 / t["n_games"]),
                fmt_pct(hi - lo)))
    L.append("")
    L.append("What the split can do is catch a model that is broken or badly "
             "fitted - a next-move model at or below the frequency table, a "
             "masked model that cannot name the champion at all. It is a smoke "
             "test on genuinely unseen games, not a ranking. For the ranking, "
             "the within-patch validation split is the number to read: it is "
             "100x larger and every model was selected on it.")
    L.append("")

    L.append("## 6. Per-game top-5")
    L.append("")
    head = "| game | " + " | ".join("`%s`" % n
                                    for n in report["next_move"]) + " |"
    L.append(head)
    L.append("|---" * (1 + len(report["next_move"])) + "|")
    for i, game in enumerate(games):
        cells = []
        for _n, d in report["next_move"].items():
            steps = d["games"][i]["steps"]
            cells.append("%d/%d" % (sum(1 for s in steps if s["rank"] < 5),
                                    len(steps)))
        L.append("| %s | %s |" % (game_label(game, i), " | ".join(cells)))
    L.append("")

    L.append("## Reproduce")
    L.append("")
    L.append("```bash")
    L.append("docker compose run --rm -T dev python model/test_split/eval_test_split.py")
    L.append("docker compose run --rm -T dev python model/test_split/render_test_split.py")
    L.append("docker compose run --rm -T dev python -m pytest model/test_split/test_test_split.py -q")
    L.append("```")
    return "\n".join(L) + "\n"


def main():
    report = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    OUT_HTML.write_text(build_html(report), encoding="utf-8")
    OUT_MD.write_text(build_md(report), encoding="utf-8")
    print("wrote %s (%.1f KB)" % (OUT_HTML.name,
                                  OUT_HTML.stat().st_size / 1024))
    print("wrote %s (%.1f KB)" % (OUT_MD.name, OUT_MD.stat().st_size / 1024))


if __name__ == "__main__":
    main()
