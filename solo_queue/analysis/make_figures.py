"""Figures for META_OVERLAP_16.20.md, drawn from meta_overlap_<patch>.json.

Four pictures, one message each. All four are deliberately plotted as
*differences* rather than as two clouds of points: at 262 drafts the pro
numbers are so noisy that a raw x/y scatter reads as "no relationship" when
the honest reading is "no relationship distinguishable from noise".

  fig_presence_rank  who gets drafted: percentile in one world against the other
  fig_wr_diff        who wins: pro WR minus soloQ WR, with each champion's CI
  fig_pair_funnel    who wins together: pair difference against pro sample size
  fig_counter_dev    counter picks: the antisymmetric matchup deviation

Usage: python make_figures.py --patch 16.20
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK = "#1b1b1f"
MUTED = "#6b6b75"
BLUE = "#2f6db3"
ORANGE = "#c4622a"
GRID = "#dcdce2"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="16.20")
    a = ap.parse_args()
    d = json.load(open(os.path.join(HERE, "out", "meta_overlap_%s.json" % a.patch),
                       encoding="utf-8"))
    outdir = os.path.join(HERE, "out", "figs")
    os.makedirs(outdir, exist_ok=True)
    t = d["champ_table"]

    # ---------------------------------------------------------------- presence
    def pct_ranks(rows, key):
        """Average-rank percentile, so ties (37 champions at pro presence 0)
        share one value instead of being spread over 37 arbitrary ranks."""
        order = sorted(rows, key=lambda r: r[key])
        out, i = {}, 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][key] == order[i][key]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]["champ"]] = (i + j) / 2 / (len(order) - 1)
            i = j + 1
        return out

    allp = [r for r in t if r["sq_pres"] > 0]
    live = [r for r in allp if r["pro_pres"] > 0]
    never = [r for r in allp if r["pro_pres"] == 0]
    rp = pct_ranks(allp, "pro_pres")
    rs = pct_ranks(allp, "sq_pres")

    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=170)
    style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", alpha=0.55,
            zorder=1)
    ax.scatter([rs[r["champ"]] for r in live], [rp[r["champ"]] for r in live],
               s=24, color=BLUE, alpha=0.55, edgecolor="none", zorder=3)
    # champions pro never touched: a rug, not a pile-up at percentile 0
    ax.axhline(-0.035, color=GRID, linewidth=0.8)
    ax.scatter([rs[r["champ"]] for r in never], [-0.035] * len(never), s=16,
               color=ORANGE, alpha=0.7, marker="|", zorder=3)
    ax.text(1.0, -0.075, "%d champions never picked or banned in pro" % len(never),
            ha="right", fontsize=7.5, color=ORANGE)

    placed = []
    for r in sorted(live, key=lambda r: -abs(rp[r["champ"]] - rs[r["champ"]])):
        x, y = rs[r["champ"]], rp[r["champ"]]
        if any(abs(x - px) < 0.13 and abs(y - py) < 0.055 for px, py in placed):
            continue
        placed.append((x, y))
        ax.annotate(r["name"], (x, y), textcoords="offset points",
                    xytext=(6 if x < 0.8 else -6, 3),
                    ha="left" if x < 0.8 else "right", fontsize=7.5, color=INK)
        ax.scatter([x], [y], s=34, color=ORANGE, zorder=4, edgecolor="white",
                   linewidth=0.6)
        if len(placed) >= 12:
            break
    ax.set_xlabel("solo queue presence percentile", fontsize=10, color=INK)
    ax.set_ylabel("pro presence percentile", fontsize=10, color=INK)
    ax.set_title("Who gets drafted, patch 16.20\n"
                 "Spearman 0.12 — the two lists share a shape, not an order",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.1, 1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_presence_rank.png"), facecolor="white")
    plt.close(fig)

    # --------------------------------------------------------------- win rate
    # exactly the champions that enter the headline correlation, so the count
    # in the title and the number of points agree
    pts = [r for r in t if r["pro_pick"] >= 5 and r["sq_pick"] >= 2000 and r["sq_wr"]]
    assert len(pts) == d["corr"]["n_champs"], (len(pts), d["corr"]["n_champs"])
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=170)
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--", alpha=0.7, zorder=1)
    off = 0
    for r in pts:
        lo, hi = wilson(r["pro_wr"] * r["pro_pick"], r["pro_pick"])
        delta = 100 * (r["pro_wr"] - r["sq_wr"])
        ax.plot([100 * r["sq_wr"]] * 2, [100 * (lo - r["sq_wr"]), 100 * (hi - r["sq_wr"])],
                color=GRID, linewidth=1.3, zorder=1)
        if not (lo <= r["sq_wr"] <= hi):
            off += 1
            ax.scatter([100 * r["sq_wr"]], [delta], s=44, color=ORANGE, zorder=4,
                       edgecolor="white", linewidth=0.6)
        else:
            ax.scatter([100 * r["sq_wr"]], [delta],
                       s=6 + 0.9 * r["pro_pick"], color=BLUE, alpha=0.5,
                       edgecolor="none", zorder=3)
    cands = [r for r in pts if not (wilson(r["pro_wr"] * r["pro_pick"],
                                          r["pro_pick"])[0] <= r["sq_wr"]
                                    <= wilson(r["pro_wr"] * r["pro_pick"],
                                              r["pro_pick"])[1])]
    cands += sorted([r for r in pts if r["pro_pick"] >= 8],
                    key=lambda r: -abs(r["pro_wr"] - r["sq_wr"]))[:6]
    placed = []
    for r in cands:
        x, y = 100 * r["sq_wr"], 100 * (r["pro_wr"] - r["sq_wr"])
        if any(abs(x - px) < 0.55 and abs(y - py) < 5.5 for px, py in placed):
            continue
        placed.append((x, y))
        ax.annotate(r["name"], (x, y), textcoords="offset points",
                    xytext=(7, 2), fontsize=7.5, color=INK)
    ax.set_xlabel("solo queue win rate (%)", fontsize=10, color=INK)
    ax.set_ylabel("pro win rate minus soloQ win rate (pp)", fontsize=10, color=INK)
    ax.set_title("Who wins\nr = %.2f over %d champions; %d of %d confidence intervals "
                 "exclude\nthe soloQ value, which is what chance alone predicts"
                 % (d["corr"]["wr_pearson"], d["corr"]["n_champs"], off, len(pts)),
                 fontsize=11, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_wr_diff.png"), facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------------ pairs
    pr = [r for r in d["pairs"] if r["sq_wr"] is not None and r["pro_n"] >= 3]
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=170)
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--", alpha=0.7, zorder=1)
    ns = [3, 4, 6, 8, 12, 20, 40, 60]
    for sign in (1, -1):
        ax.plot(ns, [sign * 196 * math.sqrt(0.25 / k) for k in ns], color=ORANGE,
                linewidth=1.4, linestyle=":", alpha=0.9, zorder=2)
    ax.scatter([r["pro_n"] for r in pr],
               [100 * (r["pro_wr"] - r["sq_wr"]) for r in pr],
               s=16, color=BLUE, alpha=0.45, edgecolor="none", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(k) for k in ns])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(2.6, 72)
    ax.text(46, 196 * math.sqrt(0.25 / 46) * 0.62, "±1.96σ from noise alone",
            fontsize=7.5, color=ORANGE, ha="right")
    ax.set_xlabel("pro games the pair appeared in", fontsize=10, color=INK)
    ax.set_ylabel("pro pair WR minus soloQ pair WR (pp)", fontsize=10, color=INK)
    ax.set_title("Champion pairs: r = %.2f over %d pairs\n"
                 "mean |difference| %.1f pp vs %.1f pp expected from noise alone"
                 % (d["pair_corr"]["pearson"], d["pair_corr"]["n"],
                    100 * d["noise"]["pair"]["obs_abs_dev"],
                    100 * d["noise"]["pair"]["exp_abs_dev"]),
                 fontsize=11, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_pair_funnel.png"), facecolor="white")
    plt.close(fig)

    # --------------------------------------------------------- counter picks
    devs = [100 * x for x in d["dev_hist"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=170)
    style(ax)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", alpha=0.8)
    ax.hist(devs, bins=40, color=BLUE, alpha=0.62, edgecolor="white", linewidth=0.5)
    mean = 100 * d["transfer"]["response"]["dev"]
    ax.axvline(mean, color=ORANGE, linewidth=2)
    ax.annotate("mean %+.1f pp" % mean, (mean, ax.get_ylim()[1] * 0.9),
                xytext=(8, 0), textcoords="offset points", fontsize=9, color=ORANGE)
    ax.set_xlabel("soloQ matchup deviation of the responding champion (pp)",
                  fontsize=10, color=INK)
    ax.set_ylabel("pro response picks", fontsize=10, color=INK)
    ax.set_title("Pro's counter picks are counters in solo queue too\n"
                 "65.4% of 933 matchups land on the favourable side of 0",
                 fontsize=11, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_counter_dev.png"), facecolor="white")
    plt.close(fig)
    print("wrote 4 figures to", outdir)


if __name__ == "__main__":
    main()
