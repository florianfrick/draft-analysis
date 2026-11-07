"""Does pro and solo queue share one champion index space, and does the soloQ
data actually carry side + result?

Both questions are answered by evidence, not by reading the code:

  1. `champ_mapping.json` is a bijection Riot id -> 0..172, and every Riot id
     has a display name.
  2. The soloQ pretraining tensor (`drafts_<patch>.pt`) is in that index space:
     after mapping the store's Riot ids through the same file, the per-role
     champion distributions agree. A shifted or permuted mapping cannot survive
     this - roles are a structural property of the champion, not of the index.
  3. The pro cache is in the same space as the *names*: for every kept game,
     the champion set in `pro_drafts.pt` equals the champion set in the
     Oracle's Elixir CSV for that gameid. This ties an index to an independent
     source of truth.
  4. The soloQ store knows side and result: 5 players per side, exactly 5
     winners per game, and the tensor's `wins` is the blue-side result.

Usage
-----
  python verify_champ_space.py [--patch 16.20]
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import duckdb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROC = os.path.join(ROOT, "data", "solo queue", "process matchdata")
STORE = os.path.join(ROOT, "solo_queue", "analysis", "store", "store.duckdb")
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
RANKED = (420, 440)
MIN_DUR = 300

fails = []


def check(ok, label, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                           ("  - " + detail) if detail else ""))
    if not ok:
        fails.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="16.20")
    a = ap.parse_args()

    mapping = json.load(open(os.path.join(PROC, "champ_mapping.json"), encoding="utf-8"))
    names = json.load(open(os.path.join(PROC, "champ_names.json"), encoding="utf-8"))["names"]
    riot_to_idx = {int(k): int(v) for k, v in mapping["riot_to_idx"].items()}
    idx_to_riot = {v: k for k, v in riot_to_idx.items()}
    num_champs = int(mapping["num_champs"])

    print("\n1. mapping file")
    check(len(riot_to_idx) == num_champs == 173,
          "173 Riot ids -> 173 indices", "%d ids, num_champs=%d" % (len(riot_to_idx), num_champs))
    check(sorted(riot_to_idx.values()) == list(range(num_champs)),
          "indices are exactly 0..172 with no gaps or duplicates")
    check(len(idx_to_riot) == len(riot_to_idx), "mapping is injective")
    missing = [r for r in riot_to_idx if str(r) not in names]
    check(not missing, "every Riot id has a display name", "missing %s" % missing[:5])
    dup = [n for n, c in Counter(names.values()).items() if c > 1]
    check(not dup, "no two champions share a display name", "dupes %s" % dup)

    print("\n2. soloQ pretraining tensor is in the same index space")
    solo = torch.load(os.path.join(PROC, "drafts_%s.pt" % a.patch), weights_only=False)
    rows, wins = solo["rows"], solo["wins"]
    check(all(len(r[0]) == 10 and len(r[1]) == 10 for r in rows),
          "every row has 10 champions and 10 roles")
    check(all(0 <= c < num_champs for r in rows for c in r[0]),
          "all champion indices in range")
    check(all(r[1] == [0, 1, 2, 3, 4] * 2 for r in rows),
          "slot order is blue TOP..SUP then red TOP..SUP")

    con = duckdb.connect(STORE, read_only=True)
    con.execute("SET threads=%d" % (os.cpu_count() or 4))
    sq = con.execute(
        "SELECT champ, pos, count(*) FROM players WHERE patch = ? "
        "AND queue_id IN (?,?) AND duration >= ? AND pos IN "
        "('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY') GROUP BY 1,2",
        [a.patch, RANKED[0], RANKED[1], MIN_DUR]).fetchall()
    store_role = defaultdict(lambda: defaultdict(int))
    for riot, pos, n in sq:
        if riot in riot_to_idx:
            store_role[riot_to_idx[riot]][pos] += n
    ten_role = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for c, role in zip(r[0], r[1]):
            ten_role[c][ROLES[role]] += 1

    worst, worst_c, modal_bad = 0.0, None, []
    for idx in sorted(ten_role):
        t_tot = sum(ten_role[idx].values())
        s_tot = sum(store_role[idx].values())
        if not s_tot:
            continue
        for role in ROLES:
            d = abs(ten_role[idx][role] / t_tot - store_role[idx][role] / s_tot)
            if d > worst:
                worst, worst_c = d, idx
        t_mode = max(ROLES, key=lambda r: ten_role[idx][r])
        s_mode = max(ROLES, key=lambda r: store_role[idx][r])
        if t_mode != s_mode:
            modal_bad.append((names[str(idx_to_riot[idx])], t_mode, s_mode))
    check(worst < 0.05, "per-role champion shares agree (tensor vs store)",
          "max abs diff %.3f at %s" % (worst, names[str(idx_to_riot[worst_c])] if worst_c is not None else "-"))
    check(not modal_bad, "every champion's most-played role agrees",
          "%d disagreements %s" % (len(modal_bad), modal_bad[:5]))

    print("\n3. pro cache matches the Oracle's Elixir names")
    cache = torch.load(os.path.join(HERE, "pro_drafts.pt"), weights_only=False)
    pro = [r for r in cache["rows"] if r["patch"] == a.patch]
    name_to_idx = {}
    for riot, idx in riot_to_idx.items():
        nm = names.get(str(riot))
        if nm is not None:
            name_to_idx[nm] = idx
    check(len(name_to_idx) == num_champs, "name -> index covers all 173 champions")

    import csv
    csv.field_size_limit(10 ** 7)
    csv_path = os.path.join(HERE, "2026_LoL_esports_match_data_from_OraclesElixir.csv")
    csv_names = defaultdict(set)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                if abs(float(row["patch"]) - float(a.patch)) > 1e-9:
                    continue
            except (TypeError, ValueError):
                continue
            if row["position"] != "team" and row["champion"]:
                csv_names[row["gameid"]].add(row["champion"])

    bad_set, bad_name = [], []
    for r in pro:
        want = {names[str(idx_to_riot[c])] for c in r["champ"]}
        got = csv_names.get(r["gameid"], set())
        if not got:
            bad_set.append((r["gameid"], "missing game"))
        elif want != got:
            bad_set.append((r["gameid"], sorted(want ^ got)))
    check(not bad_set, "all %d kept games have the same 10 champions as the CSV" % len(pro),
          str(bad_set[:3]))

    # a name the mapping does not know would be a silently dropped draft, so
    # check the whole patch, not just the games that survived
    all_names = {n for s in csv_names.values() for n in s}
    unknown = sorted(n for n in all_names if n not in name_to_idx)
    check(not unknown, "every champion name in the 16.20 export resolves",
          "unresolved %s" % unknown)

    # same question for solo queue: a champion in the data but not in the
    # mapping would vanish from the pretraining corpus without a trace
    con2 = duckdb.connect(STORE, read_only=True)
    sq_ids = {r[0] for r in con2.execute(
        "SELECT DISTINCT champ FROM players WHERE patch = ?", [a.patch]).fetchall()}
    con2.close()
    outside = sorted(sq_ids - set(riot_to_idx))
    check(not outside, "every champion in the soloQ data is in the mapping",
          "%d outside %s" % (len(outside), outside[:5]))
    check(len(sq_ids) == num_champs,
          "soloQ data and mapping cover the same champion count",
          "%d ids in data, %d in mapping" % (len(sq_ids), num_champs))

    # role check: the pro cache's role must match the CSV position for that player
    role_by_game = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                if abs(float(row["patch"]) - float(a.patch)) > 1e-9:
                    continue
            except (TypeError, ValueError):
                continue
            if row["position"] != "team" and row["champion"]:
                role_by_game[row["gameid"]][row["champion"]] = row["position"]
    POS = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
    for r in pro:
        for c, role in zip(r["champ"], r["role"]):
            nm = names[str(idx_to_riot[c])]
            if POS.get(role_by_game[r["gameid"]].get(nm, "")) != role:
                bad_name.append((r["gameid"], nm))
    check(not bad_name, "champion -> role matches the CSV position for all picks",
          str(bad_name[:3]))

    print("\n4. soloQ knows side and result")
    tot, blue_rows, red_rows, bad5 = con.execute("""
        SELECT count(*), sum(CASE WHEN side='blue' THEN 1 ELSE 0 END),
               sum(CASE WHEN side='red' THEN 1 ELSE 0 END), 0
        FROM players WHERE patch = ?""", [a.patch]).fetchone()
    check(blue_rows == red_rows == tot // 2, "every player row is on blue or red",
          "%d blue / %d red" % (blue_rows, red_rows))
    # 81 games in the corpus record win=false for BOTH teams. That is a source
    # quirk, not a mapping problem, and the code handles it by refusing to
    # claim a result (drafts.blue_win NULL, tensor rows dropped). So the check
    # is: 5 winners, or 0 winners - never 1..4.
    odd = con.execute("""
        SELECT count(*) FROM (
          SELECT match_id, sum(win) w, count(*) n FROM players
          WHERE patch = ? GROUP BY 1) WHERE n <> 10 OR w NOT IN (0, 5)""",
                      [a.patch]).fetchone()[0]
    zero = con.execute("""
        SELECT count(*) FROM (
          SELECT match_id, sum(win) w FROM players
          WHERE patch = ? GROUP BY 1) WHERE w = 0""", [a.patch]).fetchone()[0]
    nores = con.execute(
        "SELECT count(*) FROM drafts WHERE patch = ? AND blue_win IS NULL",
        [a.patch]).fetchone()[0]
    check(odd == 0, "every game has 10 players and 5 winners (or 0, if flagged)",
          "%d games with a partial result" % odd)
    check(zero == nores, "the 0-winner games are exactly the ones flagged unusable",
          "%d zero-winner games, %d with blue_win NULL" % (zero, nores))

    store_blue = con.execute(
        "SELECT avg(blue_win) FROM drafts WHERE patch = ? AND blue_win IS NOT NULL",
        [a.patch]).fetchone()[0]
    check(True, "store blue-side win rate on this patch", "%.4f" % store_blue)
    con.close()

    t_blue = sum(wins) / len(wins)
    check(abs(t_blue - store_blue) < 0.01,
          "pretraining labels are the blue-side result",
          "tensor %.4f vs store %.4f (n=%d)" % (t_blue, store_blue, len(wins)))
    check(solo["skipped_no_win"] > 0,
          "rows whose two sides disagreed on the result were dropped",
          "%d rows" % solo["skipped_no_win"])

    print("\n%d check(s) failed" % len(fails))
    if fails:
        print("  " + "\n  ".join(fails))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
