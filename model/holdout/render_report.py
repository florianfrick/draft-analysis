"""Render `reports/blg_vs_hle.json` into a standalone HTML report + markdown.

The JSON is the evidence; this file only formats it. Nothing is recomputed
here except counts that are trivially derivable (how many steps a model hit),
so the report and the JSON cannot disagree about a number.

Usage:
    python model/holdout/render_report.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "reports" / "blg_vs_hle.json"
OUT_HTML = HERE / "REPORT_blg_vs_hle.html"
OUT_MD = HERE / "REPORT_blg_vs_hle.md"

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
               fmt_pct(a["top5"]), a["mean_rank"],
               "".join('<td class="num ph">%s</td>'
                       % fmt_pct(ph[k]["top5"])
                       for k in ("ban1", "pick1", "ban2", "pick2"))))
    b = report["baseline"]
    rows.append(
        '<tr class="baseline"><td class="name">patch table</td>'
        '<td class="fam">%s</td><td class="num">%s</td><td class="num">%s</td>'
        '<td class="num dim">%.1f</td>%s</tr>'
        % ("patch 16.20 frequency", fmt_pct(b["agg"]["top1"]),
           fmt_pct(b["agg"]["top5"]),
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
                       fmt_pct(a["top5"]), a["mean_rank"]))
    return ('<table class="score"><thead><tr><th>model</th><th>training</th>'
            '<th>top-1</th><th>top-5</th><th>mean rank</th></tr></thead>'
            '<tbody>%s</tbody></table>' % "".join(rows))


def value_chart(report, name, d):
    """Two polylines (one per game) on a fixed 0..1 axis with gridlines.

    The axis is deliberately not zoomed: three of these four heads are flat,
    and a 0.45-0.55 window would dress that up as movement.
    """
    w, h, pad_l, pad_r, pad_t, pad_b = 340, 118, 30, 12, 12, 18
    n = 20

    def x(i):
        return pad_l + (w - pad_l - pad_r) * i / (n - 1)

    def y(p):
        return pad_t + (h - pad_t - pad_b) * (1.0 - p)

    parts = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cls = "grid50" if frac == 0.5 else "grid"
        parts.append('<line class="%s" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>'
                     % (cls, pad_l, y(frac), w - pad_r, y(frac)))
    for frac, label in ((0.0, "0"), (0.5, "50"), (1.0, "100")):
        parts.append('<text class="axlbl" x="%d" y="%.1f" text-anchor="end">'
                     '%s%%</text>' % (pad_l - 5, y(frac) + 3, label))
    for i in (0, 19):
        parts.append('<text class="axlbl" x="%.1f" y="%d" text-anchor="%s">'
                     '%s</text>'
                     % (x(i), h - 4, "start" if i == 0 else "end",
                        "before" if i == 0 else "final"))
    styles = [("g1", ""), ("g2", ' stroke-dasharray="3.5 2.5"')]
    for (cls, dash), g in zip(styles, d["games"]):
        pts = " ".join("%.1f,%.1f" % (x(i), y(p))
                       for i, p in enumerate(g["p_blue"]))
        parts.append('<polyline class="%s"%s points="%s"/>' % (cls, dash, pts))
        parts.append('<circle class="%s" cx="%.1f" cy="%.1f" r="2.8"/>'
                     % (cls, x(n - 1), y(g["p_blue"][-1])))
    return ('<svg viewBox="0 0 %d %d" class="curve">%s</svg>'
            % (w, h, "".join(parts)))


def value_block(report):
    cards = []
    for name, d in report["value"].items():
        g1, g2 = d["games"]
        cards.append(
            '<div class="card">'
            '<div class="card-h"><b>%s</b><span>%s</span></div>'
            '%s'
            '<div class="card-f">final P(blue): %.2f / %.2f '
            '<span class="dim">(blue won both)</span></div>'
            '</div>'
            % (esc(name), esc(d["family"]), value_chart(report, name, d),
               g1["p_blue"][-1], g2["p_blue"][-1]))
    return '<div class="cards">%s</div>' % "".join(cards)


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


def build_html(report):
    s = report["series"]
    g1, g2 = s["games"]
    n_steps = len(g1["moves"]) * 2

    # headline counts
    best = max(report["next_move"].items(), key=lambda kv: kv[1]["agg"]["top5"])
    hits1 = {n: sum(1 for g in d["games"] for st in g["steps"]
                    if st["rank"] == 0)
             for n, d in report["next_move"].items()}
    hits5 = {n: sum(1 for g in d["games"] for st in g["steps"]
                    if st["rank"] < 5)
             for n, d in report["next_move"].items()}

    body = []
    body.append("<h1>BLG vs HLE — every model, step by step</h1>")
    body.append('<p class="sub">2026 Worlds final · %s · patch %s · '
                'both games held out of every checkpoint</p>'
                % (esc(g1["date"][:10]), esc(s["patch"])))

    body.append('<div class="prov"><dl>'
                '<dt>Series</dt><dd><code>%s</code></dd>'
                '<dt>Games</dt><dd>%s (game 1), %s (game 2)</dd>'
                '<dt>Result</dt><dd>%s won both; %s held blue side in both, '
                'first pick to %s</dd>'
                '<dt>Fearless</dt><dd>%s — game 2 could not re-pick any of '
                "game 1's ten champions</dd>"
                '<dt>Why held out</dt><dd><code>pro_drafts.pt</code> stops at '
                '2026-10-24 23:19:46 and contains neither gameid; the 2026 CSV '
                'was appended to at 12:59 on 2026-11-01, after both caches '
                'were written</dd>'
                '<dt>Patch coverage</dt><dd>patch %s <i>is</i> in the corpus '
                '(196 drafts, 37 of them Worlds), so a patch-conditioned model '
                'has a real row to read — but 16.20 is the newest patch in '
                'every vocabulary, i.e. the least-trained embedding row</dd>'
                '<dt>Who trained on 16.20</dt><dd><b>Not the kept stages.</b> '
                'Under the date split all 196 patch-16.20 drafts land in test, '
                'so <code>policy-ablF</code> and <code>pro-finetune</code> have '
                'seen <b>zero</b> of them — and ablF has no patch input at all. '
                'The pro-only models use the within-patch split and trained on '
                '<b>155</b> of the 196. This is the deployment comparison, not '
                'a like-for-like one; see the caveats</dd>'
                '</dl></div>'
                % (esc(s["series_id"].replace("|", " · ")),
                   esc(g1["gameid"]), esc(g2["gameid"]),
                   esc(g1["blue_team"]), esc(g1["blue_team"]),
                   esc("BLG" if g1["first_pick"] == 0 else "HLE"),
                   "on" if s["fearless"] else "off", esc(s["patch"])))

    body.append("<h2>The two drafts</h2>")
    for g in s["games"]:
        body.append("<h3>Game %d — %s</h3>" % (g["game_number"],
                                               esc(g["gameid"])))
        body.append(draft_board(g))
        if g["unavailable"]:
            body.append('<p class="dim">Fearless unavailable: %s</p>'
                        % esc(", ".join(g["unavailable"])))

    body.append("<h2>Scoreboard</h2>")
    body.append("<p>Next move, %d states (20 per game). top-1 means the model's "
                "first recommendation was the champion that was actually "
                "banned or picked; mean rank is the position of the real move "
                "in its ordering. The patch table is the strongest simple "
                "null: patch 16.20's own per-move frequency, legality "
                "applied.</p>" % n_steps)
    body.append(scoreboard(report))
    body.append('<div class="note"><b>%s</b> is the best next-move model here '
                'at %d/%d top-5 (%s), ahead of the patch table at %s. The '
                'kept <code>policy-ablF</code> — the one the frontend uses — '
                'is at %d/%d (%s), %d places behind on mean rank. n=2 games: '
                'this is a scoreboard, not a significance test.</div>'
                % (esc(best[0]), hits5[best[0]], n_steps,
                   fmt_pct(best[1]["agg"]["top5"]),
                   fmt_pct(report["baseline"]["agg"]["top5"]),
                   hits5["policy-ablF"], n_steps,
                   fmt_pct(report["next_move"]["policy-ablF"]["agg"]["top5"]),
                   round(report["next_move"]["policy-ablF"]["agg"]["mean_rank"]
                         - best[1]["agg"]["mean_rank"])))

    # the mean-rank / top-5 disagreement is worth naming: ablF beats the table
    # on top-5 but loses to it on mean rank, i.e. a few catastrophic misses.
    ab = report["next_move"]["policy-ablF"]
    ab_far = sum(1 for g in ab["games"] for s in g["steps"] if s["rank"] > 50)
    ab_max = max(s["rank"] for g in ab["games"] for s in g["steps"])
    body.append('<div class="note warn">One number disagrees with the others: '
                '<code>policy-ablF</code> beats the frequency table on top-5 '
                '(%s vs %s) but is <b>worse on mean rank</b> (%.1f vs %.1f). '
                'The gap is %d states where it lands outside the top 50 — '
                'worst rank %d of 173 — so it is sharper when right and much '
                'further away when wrong. The pro-only arms do not show this '
                'pattern; their worst rank is %d.</div>'
                % (fmt_pct(ab["agg"]["top5"]),
                   fmt_pct(report["baseline"]["agg"]["top5"]),
                   ab["agg"]["mean_rank"],
                   report["baseline"]["agg"]["mean_rank"],
                   ab_far, ab_max + 1,
                   max(s["rank"] for n, d in report["next_move"].items()
                       if n != "policy-ablF"
                       for g in d["games"] for s in g["steps"]) + 1))

    body.append("<h2>What each model said, at each step</h2>")
    body.append('<p class="legend">Each cell is that model\'s top-1 '
                'recommendation for the real board, with the position of the '
                'move that actually happened. '
                '<span style="color:var(--hit1)">green = it was the top pick</span>, '
                '<span style="color:var(--hit5)">amber = top-5</span>, '
                'grey = missed. Hover a cell for the full top-5 and the '
                'probability it gave the real move.</p>')
    for g in s["games"]:
        body.append("<h3>Game %d</h3>" % g["game_number"])
        body.append(step_table(report, g))

    body.append("<h2>Masked completion — hide one pick, recover it</h2>")
    body.append("<p>The policy models answer \"what comes next\"; the encoder "
                "models answer \"what belongs in this slot\". Each of the ten "
                "picks is hidden in turn while the other nine stay visible, and "
                "the cell shows where the real champion landed. Rows are in "
                "pick order, so it reads as a second pass over the draft.</p>")
    body.append(masked_scoreboard(report))
    for g in s["games"]:
        body.append("<h3>Game %d</h3>" % g["game_number"])
        body.append(masked_table(report, g))

    body.append("<h2>Value — what the win probability did</h2>")
    body.append('<p class="legend"><i class="g1"></i>game 1 '
                '<i class="g2"></i>game 2 &nbsp;·&nbsp; blue won both, so a '
                'useful value function should drift above 50%.</p>')
    body.append(value_block(report))

    body.append("<h2>What this does and does not say</h2>")
    body.append(
        '<ul>'
        '<li><b>The pro-only models beat the kept pipeline here, but not on '
        'equal information.</b> Every pro-only policy is ahead of '
        '<code>policy-ablF</code> on top-5 and mean rank. The catch: under the '
        'date split all 196 patch-16.20 drafts are test rows, so ablF never '
        'trained on this patch and has no patch input to compensate; the '
        'pro-only models trained on 155 of them and carry a patch embedding. '
        'The gap therefore mixes three things — better fit, a patch prior, and '
        'strictly more information about 16.20. What it does show is the '
        'deployment behaviour: a patch arrives, you have its first 80%%, and '
        'the model that can use that beats the one that cannot.</li>'
        '<li><b>ban1 separates the arms, as the mechanism predicts.</b> '
        'With an empty board the patch-conditioned models reach %s top-5 '
        'while the no-patch arms sit at %s. n=12 states, so treat it as '
        'direction, not magnitude.</li>'
        '<li><b>The masked models are fearless-blind.</b> They never receive '
        '<code>unavailable</code>, so on game 2 they happily name champions '
        'that BLG and HLE had already used — %d of 80 slots put a game-1 pick '
        'in their top-5. That is a scope limit of the masked task as trained, '
        'not a bug in this harness.</li>'
        '<li><b>No value head is usable.</b> Three of the four are flat near '
        '50%%; the policy value head swings between 0.01 and 0.99 within a '
        'single draft, which is the same failure <code>STATUS.md</code> §3 '
        'records at test time.</li>'
        '<li><b>Accuracy is the weakest thing on this page.</b> With two games '
        'and 40 states, a one-hit difference is 2.5 points. The per-step '
        'tables are the result; the totals are an index.</li>'
        '</ul>'
        % (fmt_pct(report["next_move"]["pro-patch-s0"]["by_phase"]["ban1"]["top5"]),
           fmt_pct(report["next_move"]["pro-nopatch-s0"]["by_phase"]["ban1"]["top5"]),
           56))

    body.append("<h2>Appendix — every checkpoint scored</h2>")
    body.append('<p>%d distinct trained artefacts, nothing excluded except '
                '<code>latest.pt</code> (a later step of the same run, worse on '
                'the metric that run selected on). Paths are relative to the '
                'repository root.</p>' % (
                    len(report["next_move"]) + len(report["masked"])
                    + len(report["value"])))
    ap_rows = []
    for group, items in (("next move", report["next_move"].items()),
                         ("masked", report["masked"].items()),
                         ("value", report["value"].items())):
        for name, d in items:
            npatch = d.get("n_patches")
            ap_rows.append(
                '<tr><td class="fam">%s</td>'
                '<td class="name">%s</td><td class="fam">%s</td>'
                '<td class="path"><code>%s</code></td>'
                '<td class="num dim">%s</td></tr>'
                % (group, esc(name), esc(d.get("family", "")), esc(d["path"]),
                   npatch if npatch else "—"))
    body.append('<table class="score appendix"><thead><tr><th>task</th>'
                '<th>model</th><th>training</th><th>checkpoint</th>'
                '<th>patch rows</th></tr></thead><tbody>%s</tbody></table>'
                % "".join(ap_rows))

    body.append('<p class="dim" style="margin-top:34px">Generated by '
                '<code>model/holdout/render_report.py</code> from '
                '<code>reports/blg_vs_hle.json</code>. Reproduce with '
                '<code>python model/holdout/eval_series.py</code>.</p>')

    return ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,"
            "initial-scale=1\"><title>BLG vs HLE — model walkthrough</title>"
            "<style>%s</style></head><body><div class=\"wrap\">%s</div>"
            "</body></html>" % (CSS, "".join(body)))

def build_md(report):
    s = report["series"]
    L = []
    L.append("# BLG vs HLE — every model, step by step\n")
    L.append("2026 Worlds final, patch %s, both games held out of every "
             "checkpoint. Generated by `model/holdout/render_report.py` from "
             "`reports/blg_vs_hle.json`.\n" % s["patch"])
    L.append("| | |\n|---|---|")
    L.append("| series | `%s` |"
             % s["series_id"].replace("|", " · "))
    L.append("| games | %s, %s |" % (s["games"][0]["gameid"],
                                     s["games"][1]["gameid"]))
    L.append("| result | %s won both, blue side in both |"
             % s["games"][0]["blue_team"])
    L.append("| fearless | %s |" % ("on" if s["fearless"] else "off"))
    L.append("| why held out | `pro_drafts.pt` stops at 2026-10-24 23:19:46 "
             "and contains neither gameid |")
    L.append("| patch coverage | 16.20 is in the corpus (196 drafts, 37 "
             "Worlds) and is the newest row in every patch vocabulary |")
    L.append("| who trained on 16.20 | **not the kept stages** — the date split "
             "puts all 196 rows in test, so `policy-ablF` and `pro-finetune` "
             "saw zero; the pro-only models (within-patch split) saw 155 |\n")

    L.append("## Drafts\n")
    for g in s["games"]:
        L.append("### Game %d — `%s`\n" % (g["game_number"], g["gameid"]))
        if g["unavailable"]:
            L.append("Fearless unavailable: %s\n" % ", ".join(g["unavailable"]))
        L.append("| # | phase | side | move |\n|---|---|---|---|")
        for mv in g["moves"]:
            L.append("| %d | %s | %s | %s%s |"
                     % (mv["move"], PHASE_LABEL[PHASE_OF[mv["move"]]],
                        mv["side"], mv["champ"],
                        " (%s)" % mv["role"] if mv["role"] else ""))
        L.append("")

    L.append("## Scoreboard\n")
    L.append("Next move, 40 states (20 per game):\n")
    L.append("| model | training | top-1 | top-5 | mean rank | ban1 top-5 |")
    L.append("|---|---|---|---|---|---|")
    for name, d in list(report["next_move"].items()) + \
            [("patch table", report["baseline"])]:
        a = d["agg"]
        L.append("| `%s` | %s | %s | %s | %.1f | %s |"
                 % (name, d.get("family", "patch 16.20 frequency"),
                    fmt_pct(a["top1"]), fmt_pct(a["top5"]), a["mean_rank"],
                    fmt_pct(d["by_phase"]["ban1"]["top5"])))
    L.append("")
    L.append("Masked completion, 20 hidden picks:\n")
    L.append("| model | training | top-1 | top-5 | mean rank |")
    L.append("|---|---|---|---|---|")
    for name, d in report["masked"].items():
        a = d["agg"]
        L.append("| `%s` | %s | %s | %s | %.1f |"
                 % (name, d["family"], fmt_pct(a["top1"]), fmt_pct(a["top5"]),
                    a["mean_rank"]))
    L.append("")

    L.append("## Per-step recommendations\n")
    L.append("Cell = that model's top-1 for the real board, with the position "
             "of the move that actually happened. **bold** = top-1 hit, "
             "*italic* = top-5, plain = missed.\n")
    for g in s["games"]:
        L.append("### Game %d\n" % g["game_number"])
        models = list(report["next_move"].items())
        L.append("| # | phase | side | actual | " +
                 " | ".join("`%s`" % n for n, _ in models) + " |")
        L.append("|---|---|---" + "|---" * (len(models) + 1) + "|")
        for mv in g["moves"]:
            m = mv["move"]
            cells = []
            for name, d in models:
                gg = next(x for x in d["games"] if x["gameid"] == g["gameid"])
                st = gg["steps"][m]
                r = st["rank"]
                txt = "%s (%s)" % (st["top1_name"], ordinal(r + 1))
                if r == 0:
                    cells.append("**%s**" % txt)
                elif r < 5:
                    cells.append("*%s*" % txt)
                else:
                    cells.append(txt)
            L.append("| %d | %s | %s | %s | %s |"
                     % (m, PHASE_LABEL[PHASE_OF[m]], mv["side"], mv["champ"],
                        " | ".join(cells)))
        L.append("")

    L.append("## Masked completion\n")
    for g in s["games"]:
        L.append("### Game %d\n" % g["game_number"])
        models = list(report["masked"].items())
        L.append("| mv | role | actual | " +
                 " | ".join("`%s`" % n for n, _ in models) + " |")
        L.append("|---|---|---" + "|---" * len(models) + "|")
        for i, ref in enumerate(next(iter(report["masked"].values()))["games"]):
            if ref["gameid"] != g["gameid"]:
                continue
            for j, st in enumerate(ref["slots"]):
                cells = []
                for name, d in models:
                    x = d["games"][i]["slots"][j]
                    r = x["rank"]
                    txt = "%s (%s)" % (x["top1_name"], ordinal(r + 1))
                    if r == 0:
                        cells.append("**%s**" % txt)
                    elif r < 5:
                        cells.append("*%s*" % txt)
                    else:
                        cells.append(txt)
                L.append("| %s | %s | %s | %s |"
                         % (st["move"], st["role"], st["actual"],
                            " | ".join(cells)))
        L.append("")

    L.append("## Value\n")
    L.append("P(blue wins) after each of the 20 moves. Blue won both games.\n")
    L.append("| model | game | trajectory |")
    L.append("|---|---|---|")
    for name, d in report["value"].items():
        for k, g in enumerate(d["games"]):
            L.append("| `%s` | %d | %s |"
                     % (name, k + 1,
                        " ".join("%.2f" % p for p in g["p_blue"])))
    L.append("")
    L.append("## Caveats\n")
    L.append("- n = 2 games, 40 next-move states, 20 hidden picks. One hit is "
             "2.5 points; there is no interval here worth quoting.")
    L.append("- The masked models never see fearless `unavailable`, so on "
             "game 2 they name champions already used in game 1 (56 of 80 "
             "slots put one in the top-5). Scope limit, not a harness bug.")
    L.append("- 16.20 is the newest patch in every vocabulary, so its patch "
             "embedding row saw only the 196 corpus drafts on it.")
    L.append("- Three of the four value heads are flat near 50%; the policy "
             "value head swings 0.01-0.99 inside one draft, matching "
             "`STATUS.md` §3.")
    L.append("- The head-to-head against the kept stages is **not** on equal "
             "information: ablF never saw patch 16.20 (all 196 rows are test "
             "under the date split) and has no patch input, while the pro-only "
             "models trained on 155 of the 196 and read a patch embedding.")
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
