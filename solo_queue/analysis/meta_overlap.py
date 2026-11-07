"""How much does the pro draft meta overlap with solo queue, on one patch?

Compares pro (Oracle's Elixir drafts) and ranked solo queue (the DuckDB store)
restricted to a single patch, so no balance change sits between the two
populations. Patch 16.20 is the only patch where both sides have real volume
(soloQ: 279k ranked games; pro: 262 drafts).

Questions, in the order the report asks them:

  1. per champion: presence (picked OR banned) and win rate, both worlds
  2. do the champion *pairs* pro plays get played in soloQ, and do they win
     as much there?
  3. do pro's counter picks (a pick made after the enemy already filled that
     role) beat the same champion in soloQ in the same matchup?
  4. does pro's strength ranking predict soloQ strength at all?

Everything is measured, not asserted: per-champion pro numbers carry a
binomial CI, and the headline correlations come with a null distribution
simulated under "pro win rate == soloQ win rate".

Usage
-----
  python meta_overlap.py --patch 16.20 --queue 420
"""
import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import duckdb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STORE = os.path.join(HERE, "store", "store.duckdb")
PRO_CACHE = os.path.join(ROOT, "data", "pro play", "pro_drafts.pt")
NAMES_PATH = os.path.join(ROOT, "data", "solo queue", "process matchdata",
                          "champ_names.json")
MAPPING_PATH = os.path.join(ROOT, "data", "solo queue", "process matchdata",
                            "champ_mapping.json")
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_SHORT = {"TOP": "TOP", "JUNGLE": "JNG", "MIDDLE": "MID",
              "BOTTOM": "BOT", "UTILITY": "SUP"}
# pro rows carry role as the model's integer id (pipeline.POS_TO_ROLE); the
# store carries it as the Riot position string. Convert at the boundary so
# every join below is on one key space.
ROLE_ID = {0: "TOP", 1: "JUNGLE", 2: "MIDDLE", 3: "BOTTOM", 4: "UTILITY"}

# The pro cache stores the model's DENSE champion index (0..C-1), the soloQ
# store stores the Riot id. Every cross-world join is on the Riot id, so the
# conversion happens once, at load.
_IDX_TO_RIOT = {int(v): int(k) for k, v in
                json.load(open(MAPPING_PATH, encoding="utf-8"))["riot_to_idx"].items()}
MIN_DUR = 300          # below this a soloQ game is a remake / instant surrender
SEED = 20261106


# ---------------------------------------------------------------- statistics
def wilson(w, n, z=1.96):
    """Wilson score interval - the only interval that behaves at small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def boot_stat(items, fn, reps=2000, seed=SEED):
    """Cluster bootstrap of an arbitrary statistic over per-cluster records.

    The 10 picks in one draft are not independent - they share a team, a patch
    read and a game state - so resampling picks would understate the interval.
    Resampling whole drafts does not. `fn` takes the resampled list and returns
    one number (a pooled ratio, or a difference of two pooled ratios).
    """
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        s = [items[rng.randrange(len(items))] for _ in range(len(items))]
        v = fn(s)
        if v is not None:
            out.append(v)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))], fn(items)


# ------------------------------------------------------------------ pro side
def load_pro(patch):
    cache = torch.load(PRO_CACHE, weights_only=False)
    rows = [r for r in cache["rows"] if r["patch"] == patch]
    drafts = []
    for r in rows:
        picks = [(side, _IDX_TO_RIOT[champ], ROLE_ID[role], i)
                 for i, (kind, side, champ, role) in enumerate(r["moves"]) if kind == 1]
        bans = [(side, _IDX_TO_RIOT[champ]) for (kind, side, champ, role) in r["moves"]
                if kind == 0]
        drafts.append({"gameid": r["gameid"], "blue_win": r["blue_win"],
                       "picks": picks, "bans": bans, "league": r["league"],
                       "date": r["date"]})
    return drafts


def pro_tables(drafts):
    """champ -> pick/ban counts and wins; pairs; response picks."""
    pick = Counter()
    pick_w = Counter()
    ban = Counter()
    role_of = defaultdict(Counter)
    pairs = Counter()
    pairs_w = Counter()
    response = Counter()          # (champ, role, opp) -> games where it won
    response_n = Counter()
    blind = Counter()             # (champ, role) -> picks with no enemy in role yet
    blind_w = Counter()
    order_wr = defaultdict(lambda: [0, 0])   # pick slot 1..10 -> [w, n]

    for d in drafts:
        bw = d["blue_win"]
        if bw is None:
            continue
        for side, champ in d["bans"]:
            ban[champ] += 1
        seen_role = {"blue": {}, "red": {}}
        slot = 0
        for side, champ, role, _ in d["picks"]:
            win = 1 if (bw == 1) == (side == 0) else 0
            pick[champ] += 1
            pick_w[champ] += win
            role_of[champ][role] += 1
            slot += 1
            order_wr[slot][0] += win
            order_wr[slot][1] += 1
            opp_side = "red" if side == 0 else "blue"
            if role in seen_role[opp_side]:
                key = (champ, role, seen_role[opp_side][role])
                response[key] += win
                response_n[key] += 1
            else:
                blind[(champ, role)] += 1
                blind_w[(champ, role)] += win
            seen_role["blue" if side == 0 else "red"][role] = champ
        for side in (0, 1):
            ch = sorted(c for s, c, _, _ in d["picks"] if s == side)
            win = 1 if (bw == 1) == (side == 0) else 0
            for i in range(len(ch)):
                for j in range(i + 1, len(ch)):
                    pairs[(ch[i], ch[j])] += 1
                    pairs_w[(ch[i], ch[j])] += win
    return dict(pick=pick, pick_w=pick_w, ban=ban, role_of=role_of,
                pairs=pairs, pairs_w=pairs_w, response=response,
                response_n=response_n, blind=blind, blind_w=blind_w,
                order_wr=dict(order_wr))


# ---------------------------------------------------------------- soloQ side
def soloq_tables(con, patch, queue):
    where = ("patch = ? AND queue_id = ? AND duration >= %d" % MIN_DUR)
    con.execute("CREATE OR REPLACE TEMP TABLE g AS SELECT DISTINCT match_id "
                "FROM players WHERE " + where, [patch, queue])
    con.execute("CREATE OR REPLACE TEMP TABLE p AS SELECT * FROM players "
                "WHERE " + where, [patch, queue])

    champ = con.execute(
        "SELECT champ, count(*) n, sum(win) w FROM p GROUP BY 1").fetchall()
    champ_pos = con.execute(
        "SELECT champ, pos, count(*) n, sum(win) w FROM p "
        "WHERE pos IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY') "
        "GROUP BY 1,2").fetchall()
    pair = con.execute(
        "SELECT a.champ, b.champ, count(*) n, sum(a.win) w "
        "FROM p a JOIN p b ON a.match_id=b.match_id AND a.side=b.side "
        "AND a.champ < b.champ GROUP BY 1,2").fetchall()
    match = con.execute(
        "SELECT a.champ, a.pos, b.champ opp, count(*) n, sum(a.win) w "
        "FROM p a JOIN p b ON a.match_id=b.match_id AND a.pos=b.pos "
        "AND a.side <> b.side "
        "WHERE a.pos IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY') "
        "GROUP BY 1,2,3").fetchall()
    n_games = con.execute("SELECT count(*) FROM g").fetchone()[0]
    return dict(champ={c: (n, w) for c, n, w in champ},
                champ_pos={(c, p): (n, w) for c, p, n, w in champ_pos},
                pair={(a, b): (n, w) for a, b, n, w in pair},
                match={(c, p, o): (n, w) for c, p, o, n, w in match},
                n_games=n_games)


def soloq_bans(con, patch, queue):
    path = os.path.join(HERE, "store", "bans_%s.parquet" % patch)
    if not os.path.exists(path):
        raise SystemExit("no %s - run extract_bans.py first" % path)
    p = path.replace("\\", "/")
    rows = con.execute(f"""
      SELECT ban, count(*) FROM read_parquet('{p}') b
      JOIN g USING (match_id) WHERE ban > 0 GROUP BY 1""").fetchall()
    return {c: n for c, n in rows}


# ------------------------------------------------------------------- report
def pct(x):
    return "%.1f%%" % (100 * x) if x == x else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="16.20")
    ap.add_argument("--queue", type=int, default=420)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = {int(k): v for k, v in
             json.load(open(NAMES_PATH, encoding="utf-8"))["names"].items()}

    drafts = load_pro(a.patch)
    pro = pro_tables(drafts)
    n_drafts = len([d for d in drafts if d["blue_win"] is not None])
    n_teamdrafts = 2 * n_drafts

    con = duckdb.connect(STORE, read_only=True)
    con.execute("SET threads=%d" % (os.cpu_count() or 4))
    sq = soloq_tables(con, a.patch, a.queue)
    sq_ban = soloq_bans(con, a.patch, a.queue)
    con.close()
    n_sq_teamdrafts = 2 * sq["n_games"]

    # ---- 1. per champion presence + win rate -------------------------------
    champs = sorted(set(pro["pick"]) | set(pro["ban"]) | set(sq["champ"]))
    table = []
    for c in champs:
        pn, pw = pro["pick"][c], pro["pick_w"][c]
        pb = pro["ban"][c]
        sn, sw = sq["champ"].get(c, (0, 0))
        sb = sq_ban.get(c, 0)
        p_lo, p_hi = wilson(pw, pn)
        s_lo, s_hi = wilson(sw, sn)
        table.append({
            "champ": c, "name": names.get(c, str(c)),
            "pro_pick": pn, "pro_ban": pb, "pro_pres": (pn + pb) / n_teamdrafts,
            "pro_wr": (pw / pn) if pn else None,
            "pro_wr_lo": p_lo if pn else None, "pro_wr_hi": p_hi if pn else None,
            "sq_pick": sn, "sq_ban": sb,
            "sq_pres": (sn + sb) / n_sq_teamdrafts,
            "sq_wr": (sw / sn) if sn else None,
            "sq_wr_lo": s_lo if sn else None, "sq_wr_hi": s_hi if sn else None,
            "sq_pick_rate": sn / n_sq_teamdrafts,
            "sq_ban_rate": sb / n_sq_teamdrafts,
        })
    table.sort(key=lambda r: (-r["pro_pres"], -r["sq_pres"]))

    # correlation on champions pro actually used, and on soloQ volume
    both = [r for r in table if r["pro_pick"] >= 5 and r["sq_pick"] >= 2000
            and r["pro_wr"] is not None and r["sq_wr"] is not None]
    xs = [r["pro_wr"] for r in both]
    ys = [r["sq_wr"] for r in both]
    r_wr = pearson(xs, ys)
    rho_wr = spearman(xs, ys)

    # null: "pro WR == soloQ WR" -> resample pro wins from Binomial(n, sq_wr)
    rng = random.Random(SEED)
    null_r = []
    for _ in range(2000):
        nx, ny = [], []
        for r in both:
            p = r["sq_wr"]
            w = sum(1 for _ in range(r["pro_pick"]) if rng.random() < p)
            nx.append(w / r["pro_pick"])
            ny.append(p)
        null_r.append(pearson(nx, ny))
    null_r.sort()
    null_hi = null_r[int(0.975 * 2000)]
    # two-sided p: how often does the null produce a correlation at least as
    # large in magnitude as the observed one?
    p_wr = sum(1 for r in null_r if abs(r) >= abs(r_wr)) / len(null_r)

    # how many champions' pro CI actually excludes their soloQ win rate? Under
    # "the two worlds agree" we expect ~5 % by construction.
    excl = sum(1 for r in both if not (r["pro_wr_lo"] <= r["sq_wr"] <= r["pro_wr_hi"]))
    n_excl = {"n": len(both), "excluded": excl, "share": excl / len(both)}

    # presence correlation: only champions that appear somewhere
    pres = [r for r in table if r["pro_pres"] > 0 and r["sq_pres"] > 0]
    r_pres = pearson([r["pro_pres"] for r in pres], [r["sq_pres"] for r in pres])
    rho_pres = spearman([r["pro_pres"] for r in pres], [r["sq_pres"] for r in pres])
    # permutation p: shuffle pro presence across champions, keep the marginals
    px = [r["pro_pres"] for r in pres]
    py = [r["sq_pres"] for r in pres]
    hit = 0
    for _ in range(2000):
        rng.shuffle(px)
        if abs(pearson(px, py)) >= abs(r_pres):
            hit += 1
    p_pres = hit / 2000

    # ---- 2. pairs ----------------------------------------------------------
    pair_rows = []
    for (c1, c2), pn in pro["pairs"].items():
        if pn < 2:
            continue
        pw = pro["pairs_w"][(c1, c2)]
        sn, sw = sq["pair"].get((c1, c2), (0, 0))
        pair_rows.append({
            "a": names.get(c1, str(c1)), "b": names.get(c2, str(c2)),
            "a_id": c1, "b_id": c2, "pro_n": pn, "pro_wr": pw / pn,
            "sq_n": sn, "sq_wr": (sw / sn) if sn else None,
            "sq_plays_per_1k": 1000 * sn / sq["n_games"],
        })
    pair_rows.sort(key=lambda r: -r["pro_n"])

    # synergy: pair WR minus the mean of the two champions' own WR, both worlds
    for r in pair_rows:
        p1 = pro["pick"][r["a_id"]], pro["pick_w"][r["a_id"]]
        p2 = pro["pick"][r["b_id"]], pro["pick_w"][r["b_id"]]
        if p1[0] and p2[0]:
            base = (p1[1] / p1[0] + p2[1] / p2[0]) / 2
            r["pro_syn"] = r["pro_wr"] - base
        else:
            r["pro_syn"] = None
        s1 = sq["champ"].get(r["a_id"], (0, 0))
        s2 = sq["champ"].get(r["b_id"], (0, 0))
        if s1[0] and s2[0] and r["sq_wr"] is not None:
            base = (s1[1] / s1[0] + s2[1] / s2[0]) / 2
            r["sq_syn"] = r["sq_wr"] - base
        else:
            r["sq_syn"] = None

    # ---- 3. response (counter) picks --------------------------------------
    # Every role gets exactly one champ per side, so in every role exactly one
    # pick is "blind" (nobody in that role had picked yet) and exactly one is a
    # response. 262 drafts -> 1310 of each, and the two halves partition all
    # 2620 picks. That is the cleanest available control: same games, same
    # champions, same patch; only the order within the role differs.
    resp = []
    for (c, role, opp), w in pro["response"].items():
        n = pro["response_n"][(c, role, opp)]
        sn, sw = sq["match"].get((c, role, opp), (0, 0))
        resp.append({"champ": names.get(c, str(c)), "champ_id": c, "role": role,
                     "opp": names.get(opp, str(opp)), "opp_id": opp,
                     "pro_n": n, "pro_wr": w / n,
                     "sq_n": sn, "sq_wr": (sw / sn) if sn else None})
    resp.sort(key=lambda r: (-r["pro_n"], -(r["pro_wr"])))

    # per-draft counters, split by side:
    #   [b_resp_w, b_resp_n, b_blind_w, b_blind_n, r_resp_w, r_resp_n, r_blind_w, r_blind_n]
    # Splitting by side matters: in this sample blue wins 56 % and red is the
    # side that answers more often (704 vs 606), so a pooled response-vs-blind
    # contrast would mostly re-measure the side effect.
    by_draft = defaultdict(lambda: [0] * 8)
    cell_rows = []   # (draft, is_response, champ, role, opp, win, sq_n, sq_wr, champ_sq_wr)
    for d in drafts:
        if d["blue_win"] is None:
            continue
        acc = by_draft[d["gameid"]]
        per_role = {"blue": {}, "red": {}}
        win_of = {}
        for side, champ, role, order in d["picks"]:
            sk = "blue" if side == 0 else "red"
            per_role[sk][role] = (champ, order)
            win_of[(sk, role)] = 1 if (d["blue_win"] == 1) == (side == 0) else 0
        for role, b in per_role["blue"].items():
            r = per_role["red"].get(role)
            if r is None:
                continue
            # In each role one side moves first and one second, so the role
            # contributes exactly one blind pick and one response pick.
            first, second = (b, r) if b[1] < r[1] else (r, b)
            pairs = [(first, 0, "blue" if first is b else "red", second[0]),
                     (second, 1, "blue" if second is b else "red", first[0])]
            for item in pairs:
                if item is None:
                    continue
                (champ, _o), is_resp, sk, opp = item
                win = win_of[(sk, role)]
                off = 0 if sk == "blue" else 4
                if is_resp:
                    acc[off] += win
                    acc[off + 1] += 1
                else:
                    acc[off + 2] += win
                    acc[off + 3] += 1
                sn, sw = sq["match"].get((champ, role, opp), (0, 0))
                base = sq["champ"].get(champ, (0, 0))
                cell_rows.append((d["gameid"], is_resp, champ, role, opp, win,
                                  sn, (sw / sn) if sn else None,
                                  (base[1] / base[0]) if base[0] else None))

    def wr(s, a, b):
        return sum(x[a] for x in s) / sum(x[b] for x in s)

    items = [tuple(acc) for acc in by_draft.values()]
    lo_r, hi_r, m_r = boot_stat(items, lambda s: wr(s, 0, 1))
    lo_b, hi_b, m_b = boot_stat(items, lambda s: wr(s, 2, 3))
    # within-side contrast: average of the two sides' (response - blind)
    lo_d, hi_d, m_d = boot_stat(
        items, lambda s: 0.5 * ((wr(s, 0, 1) - wr(s, 2, 3))
                                + (wr(s, 4, 5) - wr(s, 6, 7))))

    # Transfer test, in two versions.
    #
    # (a) cell WR and the edge over the champion's OWN average. Easy to read,
    #     but not clean: if pro's blind picks are champions that are strong in
    #     pro and weak in soloQ (Rumble is exactly that), every counter looks
    #     good in soloQ for free.
    # (b) the antisymmetric deviation. For a matchup (X vs Y) in role R,
    #         dev = (wr(X|Y) - avgX) - (wr(Y|X) - avgY)
    #     which flips sign when the two sides swap, so it is a property of the
    #     matchup and not of the champion pool. dev > 0 means soloQ says the
    #     responding champion wins the exchange. This is the honest test of
    #     "does pro's counter pick work in soloQ too".
    def transfer(rows, floor=200):
        out = []
        for gid, is_resp, champ, role, opp, win, sn, sq_wr, base in rows:
            if sn >= floor and sq_wr is not None and base is not None:
                out.append((sq_wr, sq_wr - base, champ, role, opp))
        return out

    def dev_of(champ, role, opp):
        sn, sw = sq["match"].get((champ, role, opp), (0, 0))
        so, swo = sq["match"].get((opp, role, champ), (0, 0))
        cx = sq["champ"].get(champ, (0, 0))
        cy = sq["champ"].get(opp, (0, 0))
        if min(sn, so, cx[0], cy[0]) < 200:
            return None
        wx, wy = sw / sn, swo / so
        return (wx - cx[1] / cx[0]) - (wy - cy[1] / cy[0])

    tr = {}
    dev_hist = []
    for label, want in (("response", 1), ("blind", 0)):
        rows = [r for r in cell_rows if r[1] == want]
        vals = transfer(rows)
        if not vals:
            continue
        lo_w, hi_w, m_w = boot_stat(
            [(v[0],) for v in vals], lambda s: sum(x[0] for x in s) / len(s))
        lo_e, hi_e, m_e = boot_stat(
            [(v[1],) for v in vals], lambda s: sum(x[0] for x in s) / len(s))
        devs = [d for d in (dev_of(v[2], v[3], v[4]) for v in vals) if d is not None]
        if label == "response":
            dev_hist = devs
        if devs:
            lo_v, hi_v, m_v = boot_stat(
                [(d,) for d in devs], lambda s: sum(x[0] for x in s) / len(s))
            share_pos = sum(1 for d in devs if d > 0) / len(devs)
        else:
            lo_v = hi_v = m_v = share_pos = None
        tr[label] = {"n_cells": len(vals), "n_picks": len(rows),
                     "sq_wr": m_w, "sq_wr_ci": [lo_w, hi_w],
                     "sq_edge": m_e, "sq_edge_ci": [lo_e, hi_e],
                     "share_above_50": sum(1 for v in vals if v[0] > 0.5) / len(vals),
                     "n_dev": len(devs), "dev": m_v, "dev_ci": [lo_v, hi_v],
                     "share_dev_pos": share_pos}
    # coarse split: are the cells pro won in also the cells soloQ likes?
    big = [r for r in cell_rows if r[1] == 1 and r[6] >= 200 and r[7] is not None]
    hi_cells = [r[7] for r in big if r[5] == 1]
    lo_cells = [r[7] for r in big if r[5] == 0]
    tr["split"] = {
        "pro_won_n": len(hi_cells), "pro_lost_n": len(lo_cells),
        "sq_wr_when_pro_won": (sum(hi_cells) / len(hi_cells)) if hi_cells else None,
        "sq_wr_when_pro_lost": (sum(lo_cells) / len(lo_cells)) if lo_cells else None,
    }

    edges = []
    for r in resp:
        sn, sw = r["sq_n"], r["sq_wr"]
        base = sq["champ"].get(r["champ_id"], (0, 0))
        if sn >= 200 and base[0] >= 2000:
            edges.append({
                "champ": r["champ"], "role": r["role"], "opp": r["opp"],
                "pro_n": r["pro_n"], "pro_wr": r["pro_wr"],
                "sq_n": sn, "sq_wr": sw,
                "sq_edge": sw - base[1] / base[0],       # vs own champ average
            })

    # pair transfer, on the pairs thick enough in both worlds
    pq = [r for r in pair_rows if r["pro_n"] >= 3 and r["sq_n"] >= 200
          and r["sq_wr"] is not None]
    pair_corr = {
        "n": len(pq),
        "pearson": pearson([r["pro_wr"] for r in pq], [r["sq_wr"] for r in pq]),
        "spearman": spearman([r["pro_wr"] for r in pq], [r["sq_wr"] for r in pq]),
        "mean_pro": sum(r["pro_wr"] for r in pq) / len(pq) if pq else None,
        "mean_sq": sum(r["sq_wr"] for r in pq) / len(pq) if pq else None,
        "mean_abs_delta": sum(abs(r["pro_wr"] - r["sq_wr"]) for r in pq) / len(pq) if pq else None,
    }
    psyn = [r for r in pair_rows if r["pro_syn"] is not None and r["sq_syn"] is not None
            and r["pro_n"] >= 3]
    pair_corr["syn_n"] = len(psyn)
    pair_corr["syn_pearson"] = (pearson([r["pro_syn"] for r in psyn],
                                        [r["sq_syn"] for r in psyn]) if psyn else None)

    # pick rate and ban rate on their own - presence mixes the two, and they
    # answer different questions ("who gets banned" is the consensus read)
    pr = [r for r in table if r["pro_pick"] > 0 and r["sq_pick"] > 0]
    br = [r for r in table if r["pro_ban"] > 0 and r["sq_ban"] > 0]
    pick_corr = {"n": len(pr),
                 "pearson": pearson([r["pro_pick"] / n_teamdrafts for r in pr],
                                    [r["sq_pick_rate"] for r in pr]),
                 "spearman": spearman([r["pro_pick"] for r in pr],
                                      [r["sq_pick_rate"] for r in pr])}
    ban_corr = {"n": len(br),
                "pearson": pearson([r["pro_ban"] / n_teamdrafts for r in br],
                                   [r["sq_ban_rate"] for r in br]),
                "spearman": spearman([r["pro_ban"] for r in br],
                                     [r["sq_ban_rate"] for r in br])}

    # do pro's *successful* counters hold up in soloQ?
    def cell_wr(sel):
        v = [r["sq_wr"] for r in sel if r["sq_wr"] is not None and r["sq_n"] >= 200]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    good = [r for r in resp if r["pro_n"] >= 3 and r["pro_wr"] >= 0.6]
    bad = [r for r in resp if r["pro_n"] >= 3 and r["pro_wr"] <= 0.4]
    counter_split = {"good_n_cells": len(good), "good_sq_wr": cell_wr(good)[0],
                     "bad_n_cells": len(bad), "bad_sq_wr": cell_wr(bad)[0]}

    # ---- 4. role portability ----------------------------------------------
    port = []
    for c, cnt in pro["role_of"].items():
        tot = sum(cnt.values())
        if tot < 2:
            continue
        pro_role = cnt.most_common(1)[0][0]
        sq_roles = {p: sq["champ_pos"].get((c, p), (0, 0))[0] for p in ROLES}
        sq_tot = sum(sq_roles.values())
        if not sq_tot:
            continue
        sq_role = max(sq_roles, key=sq_roles.get)
        port.append({"champ": names.get(c, str(c)),
                     "pro_role": ROLE_SHORT[pro_role], "pro_n": tot,
                     "sq_role": ROLE_SHORT[sq_role],
                     "sq_share": sq_roles[sq_role] / sq_tot,
                     "same": pro_role == sq_role})

    # ---- 5. how much do the two meta lists agree at the top? ---------------
    def top_set(key, k):
        return [r["champ"] for r in sorted(table, key=lambda r: -r[key])[:k]]

    overlap = {}
    for k in (5, 10, 20, 30):
        pa, pb = set(top_set("pro_pres", k)), set(top_set("sq_pres", k))
        overlap[k] = {"n": len(pa & pb),
                      "champs": sorted(names.get(c, str(c)) for c in pa & pb)}
    top20_sq = set(top_set("sq_pick_rate", 20))
    pick_vol = [r for r in table if r["pro_pick"]]
    overlap["pro_pick_share_in_sq_top20"] = (
        sum(r["pro_pick"] for r in pick_vol if r["champ"] in top20_sq)
        / sum(r["pro_pick"] for r in pick_vol))

    # rank-space divergence: percentile of each champ inside its own world, so
    # a 40 % pro presence and a 40 % soloQ presence are directly comparable
    def pct_rank(rows, key):
        order = sorted(rows, key=lambda r: r[key])
        n = len(order)
        return {r["champ"]: i / max(n - 1, 1) for i, r in enumerate(order)}

    live = [r for r in table if r["pro_pres"] > 0 or r["sq_pres"] > 0]
    rp = pct_rank(live, "pro_pres")
    rs = pct_rank(live, "sq_pres")
    for r in live:
        r["rank_gap"] = rp[r["champ"]] - rs[r["champ"]]
    pro_loved = sorted([r for r in live if r["pro_pick"] >= 5],
                       key=lambda r: -r["rank_gap"])[:10]
    sq_loved = sorted([r for r in live if r["sq_pick"] >= 2000],
                      key=lambda r: r["rank_gap"])[:10]

    # ---- 6. robustness + concentration -------------------------------------
    # The 262 drafts are not evenly spread: LAS alone is more than half. Re-run
    # the headline correlation without it, so the result is not one league's
    # read of the patch.
    top_league = Counter(d["league"] for d in drafts).most_common(1)[0][0]
    sub = [d for d in drafts if d["league"] != top_league]
    sp = pro_tables(sub)
    rows_sub = []
    for r in both:
        c = r["champ"]
        pn, pw = sp["pick"][c], sp["pick_w"][c]
        if pn >= 3:
            rows_sub.append((pw / pn, r["sq_wr"]))
    rob = {"league_dropped": top_league,
           "drafts_left": len(sub),
           "n_champs": len(rows_sub),
           "pearson": (pearson([x for x, _ in rows_sub], [y for _, y in rows_sub])
                       if len(rows_sub) > 3 else None),
           "spearman": (spearman([x for x, _ in rows_sub], [y for _, y in rows_sub])
                        if len(rows_sub) > 3 else None)}

    def concentration(counter, total):
        vals = sorted(counter.values(), reverse=True)
        return {"distinct": len(vals),
                "top10_share": sum(vals[:10]) / total,
                "hhi": sum((v / total) ** 2 for v in vals)}

    conc = {"pro_pick": concentration(pro["pick"], sum(pro["pick"].values())),
            "pro_ban": concentration(pro["ban"], sum(pro["ban"].values())),
            "sq_pick": concentration({c: v[0] for c, v in sq["champ"].items()},
                                     sum(v[0] for v in sq["champ"].values())),
            "sq_ban": concentration(sq_ban, sum(sq_ban.values()))}

    # Noise decomposition. With 5-70 pro picks per champion, the pro win rate
    # has a binomial standard error of 6-22 points, which is the same order as
    # the differences we are trying to read. So compare the observed spread to
    # the spread that pure sampling noise around the soloQ value would produce:
    #   obs = mean |pro_wr - sq_wr|
    #   exp = mean E|k/n - p| under k ~ Binomial(n, p = sq_wr)
    # If obs ~ exp, the champion-level disagreement is noise, not divergence.
    def exp_abs_dev(n, p):
        s = 0.0
        for k in range(n + 1):
            s += abs(k / n - p) * math.comb(n, k) * p ** k * (1 - p) ** (n - k)
        return s

    ch_noise = [(r["pro_pick"], r["pro_wr"], r["sq_wr"]) for r in both]
    noise = {
        "champ": {
            "n": len(ch_noise),
            "obs_abs_dev": sum(abs(a - b) for _, a, b in ch_noise) / len(ch_noise),
            "exp_abs_dev": sum(exp_abs_dev(n, p) for n, _, p in ch_noise) / len(ch_noise),
            "obs_sd": (lambda v: math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)))(
                [a - b for _, a, b in ch_noise]),
            "median_pick": sorted(n for n, _, _ in ch_noise)[len(ch_noise) // 2],
        },
        "pair": {
            "n": len(pq),
            "obs_abs_dev": sum(abs(r["pro_wr"] - r["sq_wr"]) for r in pq) / len(pq),
            "exp_abs_dev": sum(exp_abs_dev(r["pro_n"], r["sq_wr"]) for r in pq) / len(pq),
        },
    }
    # does the gap shrink as the pro sample grows? (it should, if it is noise)
    buckets = []
    for lo, hi in ((3, 5), (6, 10), (11, 20), (21, 100)):
        sel = [r for r in pq if lo <= r["pro_n"] <= hi]
        if sel:
            buckets.append({
                "lo": lo, "hi": hi, "n_pairs": len(sel),
                "obs_abs_dev": sum(abs(r["pro_wr"] - r["sq_wr"]) for r in sel) / len(sel),
                "exp_abs_dev": sum(exp_abs_dev(r["pro_n"], r["sq_wr"]) for r in sel) / len(sel),
            })
    noise["pair_buckets"] = buckets

    out = {
        "patch": a.patch, "queue": a.queue,
        "pro": {"drafts": n_drafts, "team_drafts": n_teamdrafts,
                "picks": sum(pro["pick"].values()),
                "bans": sum(pro["ban"].values()),
                "blue_wr": sum(d["blue_win"] for d in drafts if d["blue_win"] is not None)
                / n_drafts,
                "leagues": Counter(d["league"] for d in drafts).most_common()},
        "soloq": {"games": sq["n_games"], "team_drafts": n_sq_teamdrafts,
                  "picks": sum(v[0] for v in sq["champ"].values()),
                  "bans": sum(sq_ban.values())},
        "corr": {"wr_pearson": r_wr, "wr_spearman": rho_wr, "n_champs": len(both),
                 "null_hi_975": null_hi,
                 "null_med": null_r[1000], "wr_p": p_wr,
                 "pres_pearson": r_pres, "pres_spearman": rho_pres,
                 "pres_n": len(pres), "pres_p": p_pres, "ci_excludes": n_excl},
        "response": {"pro_wr": m_r, "pro_wr_ci": [lo_r, hi_r],
                     "blind_wr": m_b, "blind_wr_ci": [lo_b, hi_b],
                     "diff": m_d, "diff_ci": [lo_d, hi_d],
                     "n_response": sum(v[1] for v in by_draft.values()),
                     "n_blind": sum(v[3] for v in by_draft.values())},
        "transfer": tr, "overlap": overlap, "pair_corr": pair_corr,
        "pick_corr": pick_corr, "ban_corr": ban_corr,
        "counter_split": counter_split,
        "robust": rob, "concentration": conc, "noise": noise,
        "pro_loved": pro_loved, "sq_loved": sq_loved,
        "champ_table": table, "pairs": pair_rows, "response_cells": resp,
        "edges": edges, "portability": port, "dev_hist": dev_hist,
    }
    with open(os.path.join(a.out, "meta_overlap_%s.json" % a.patch), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)

    # ---- markdown artefacts ------------------------------------------------
    with open(os.path.join(a.out, "champ_table_%s.md" % a.patch), "w",
              encoding="utf-8") as fh:
        fh.write("| Champion | Pro pres | Pro pick | Pro ban | Pro WR | 95% CI "
                 "| SoloQ pres | SoloQ pick | SoloQ ban | SoloQ WR | 95% CI |\n")
        fh.write("|---|--:|--:|--:|--:|--|--:|--:|--:|--:|--|\n")
        for r in table:
            fh.write("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s |\n" % (
                r["name"], pct(r["pro_pres"]), r["pro_pick"], r["pro_ban"],
                pct(r["pro_wr"]) if r["pro_wr"] is not None else "-",
                ("%s-%s" % (pct(r["pro_wr_lo"]), pct(r["pro_wr_hi"])))
                if r["pro_wr"] is not None else "-",
                pct(r["sq_pres"]), pct(r["sq_pick_rate"]), pct(r["sq_ban_rate"]),
                pct(r["sq_wr"]) if r["sq_wr"] is not None else "-",
                ("%s-%s" % (pct(r["sq_wr_lo"]), pct(r["sq_wr_hi"])))
                if r["sq_wr"] is not None else "-"))

    with open(os.path.join(a.out, "pairs_%s.md" % a.patch), "w", encoding="utf-8") as fh:
        fh.write("| Pair | Pro n | Pro WR | SoloQ n | SoloQ WR | SoloQ games/1k "
                 "| Pro synergy | SoloQ synergy |\n")
        fh.write("|---|--:|--:|--:|--:|--:|--:|--:|\n")
        for r in pair_rows:
            fh.write("| %s + %s | %d | %s | %d | %s | %.1f | %s | %s |\n" % (
                r["a"], r["b"], r["pro_n"], pct(r["pro_wr"]), r["sq_n"],
                pct(r["sq_wr"]) if r["sq_wr"] is not None else "-",
                r["sq_plays_per_1k"],
                ("%+.1f" % (100 * r["pro_syn"])) if r["pro_syn"] is not None else "-",
                ("%+.1f" % (100 * r["sq_syn"])) if r["sq_syn"] is not None else "-"))

    # ---- console summary ---------------------------------------------------
    print("patch %s | pro %d drafts (%d team-drafts) | soloQ %d games"
          % (a.patch, n_drafts, n_teamdrafts, sq["n_games"]))
    print("  wr  pearson %.3f  spearman %.3f  (n=%d champs, p=%.3f, null 97.5%% = %.3f)"
          % (r_wr, rho_wr, len(both), p_wr, null_hi))
    print("  pres pearson %.3f  spearman %.3f  (n=%d, p=%.3f)"
          % (r_pres, rho_pres, len(pres), p_pres))
    print("  response picks %.4f [%.4f,%.4f] n=%d | blind %.4f [%.4f,%.4f] n=%d"
          % (m_r, lo_r, hi_r, out["response"]["n_response"],
             m_b, lo_b, hi_b, out["response"]["n_blind"]))
    print("  response-blind diff %+.4f [%+.4f,%+.4f]" % (m_d, lo_d, hi_d))
    for k, v in tr.items():
        if k == "split":
            print("  cell split: pro won -> sq wr %.4f (n=%d) | pro lost -> %.4f (n=%d)"
                  % (v["sq_wr_when_pro_won"], v["pro_won_n"],
                     v["sq_wr_when_pro_lost"], v["pro_lost_n"]))
            continue
        print("  %-8s transfer: n_cells %4d  sq wr %.4f [%.4f,%.4f]  edge %+.4f "
              "[%+.4f,%+.4f]  above 50%%: %.1f%%"
              % (k, v["n_cells"], v["sq_wr"], *v["sq_wr_ci"], v["sq_edge"],
                 *v["sq_edge_ci"], 100 * v["share_above_50"]))
        if v.get("dev") is not None:
            print("           antisym dev %+.4f [%+.4f,%+.4f]  positive %.1f%% (n=%d)"
                  % (v["dev"], *v["dev_ci"], 100 * v["share_dev_pos"], v["n_dev"]))
    print("  pairs n=%d  pearson %.3f spearman %.3f  mean pro %.4f sq %.4f  |delta| %.4f"
          % (pair_corr["n"], pair_corr["pearson"], pair_corr["spearman"],
             pair_corr["mean_pro"], pair_corr["mean_sq"], pair_corr["mean_abs_delta"]))
    print("  pick rate r %.3f (n=%d) | ban rate r %.3f (n=%d)"
          % (pick_corr["pearson"], pick_corr["n"], ban_corr["pearson"], ban_corr["n"]))
    print("  pro-good counters (n>=3, wr>=60%%): sq wr %s | pro-bad (<=40%%): %s"
          % (pct(counter_split["good_sq_wr"] or float("nan")),
             pct(counter_split["bad_sq_wr"] or float("nan"))))
    print("  robustness, %s dropped (%d drafts): wr pearson %s"
          % (rob["league_dropped"], rob["drafts_left"],
             ("%.3f (n=%d)" % (rob["pearson"], rob["n_champs"]))
             if rob["pearson"] is not None else "n/a"))
    for k, v in conc.items():
        print("  %-8s distinct %3d  top-10 share %5.1f%%  HHI %.4f"
              % (k, v["distinct"], 100 * v["top10_share"], v["hhi"]))
    print("  noise: champs n=%d obs |dev| %.4f vs null %.4f (sd %.4f, median n %d)"
          % (noise["champ"]["n"], noise["champ"]["obs_abs_dev"],
             noise["champ"]["exp_abs_dev"], noise["champ"]["obs_sd"],
             noise["champ"]["median_pick"]))
    print("         pairs  n=%d obs |dev| %.4f vs null %.4f"
          % (noise["pair"]["n"], noise["pair"]["obs_abs_dev"],
             noise["pair"]["exp_abs_dev"]))
    for b in noise["pair_buckets"]:
        print("         pairs n=%2d-%3d: %3d pairs  obs %.4f  null %.4f"
              % (b["lo"], b["hi"], b["n_pairs"], b["obs_abs_dev"], b["exp_abs_dev"]))
    print("  top-N presence overlap: " + "  ".join(
        "%d:%d" % (k, v["n"]) for k, v in overlap.items() if isinstance(k, int)))
    print("  pro pick volume inside soloQ top-20 pick rate: %.1f%%"
          % (100 * overlap["pro_pick_share_in_sq_top20"]))
    print("\ntop pro presence:")
    for r in table[:15]:
        print("  %-14s pro pres %5.1f%%  wr %s   | sq pick %5.1f%% ban %5.1f%% wr %s"
              % (r["name"], 100 * r["pro_pres"], pct(r["pro_wr"] or float("nan")),
                 100 * r["sq_pick_rate"], 100 * r["sq_ban_rate"],
                 pct(r["sq_wr"] or float("nan"))))
    print("\ntop pro pairs (n>=2):")
    for r in pair_rows[:15]:
        print("  %-12s + %-12s pro n=%2d wr %s | sq n=%5d wr %s"
              % (r["a"], r["b"], r["pro_n"], pct(r["pro_wr"]),
                 r["sq_n"], pct(r["sq_wr"] if r["sq_wr"] is not None else float("nan"))))
    print("\nresponse cells with pro n>=3:")
    for r in [x for x in resp if x["pro_n"] >= 3][:12]:
        print("  %-12s %-3s vs %-12s pro n=%d wr %s | sq n=%5d wr %s"
              % (r["champ"], ROLE_SHORT[r["role"]], r["opp"], r["pro_n"],
                 pct(r["pro_wr"]), r["sq_n"],
                 pct(r["sq_wr"] if r["sq_wr"] is not None else float("nan"))))
    print("\nrole portability: %d/%d champs same role"
          % (sum(1 for p in port if p["same"]), len(port)))
    print("\npro loves (rank gap):")
    for r in pro_loved:
        print("  %-14s pro %5.1f%% | sq %5.1f%% (pick %4.1f%%)"
              % (r["name"], 100 * r["pro_pres"], 100 * r["sq_pres"],
                 100 * r["sq_pick_rate"]))
    print("\nsoloQ loves (rank gap):")
    for r in sq_loved:
        print("  %-14s sq %5.1f%% (pick %4.1f%%) | pro %5.1f%%"
              % (r["name"], 100 * r["sq_pres"], 100 * r["sq_pick_rate"],
                 100 * r["pro_pres"]))


if __name__ == "__main__":
    main()
