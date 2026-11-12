"""The tools a chat answer may cite, and the cells each one refuses.

Why tools at all: the adapter is good at choosing one name from a shortlist and
bad at recalling facts. `probe_roleboard_gen.py` measured the second half -
7-8 % top-1, exactly one name, and it never learned to emit `<|im_end|>` - so
the chat surface does not ask the model what it knows. It asks it to choose
between things the server computed.

Why *counts* and not *rates*: `_probe_tool_power.py` measured the cells a
champion tool would serve. A champion's marginal is 1,479 games (median over
46,915 drafts), a cross-team pair 24, and the pair a counter-pick question is
actually about - both champions in the same lane - **5**, with 73.6 % of those
cells under 20 games. A rate of 0.60 over 5 games reads exactly like a rate of
0.60 over 500 once it is inside a prompt, so every tool here returns `n` beside
the number and sets `reliable = False` below `MIN_CELL`. The caller prints
"n=5", never "60 %".

The one tool that is never thin is `legal_pool`, because it is not statistics:
it is the board, and it is the thing a 4B model cannot be trusted to
reconstruct from the prompt.

A tool returns

    {"tool": str, "args": {...}, "n": int, "reliable": bool,
     "result": {...}, "note": str | None}

so that a caller can decide what to say without re-deriving the power.
"""
import threading
from collections import Counter

from llm import N_MOVES

# Below this a rate is a rumour. 30 is a judgement, not a derivation: at n=30
# the 95 % interval on a rate near 0.5 is +/-18 pp, which is already wider than
# any effect this project has measured, so it is the *floor* for quoting a rate
# at all - not a claim that 30 is enough.
MIN_CELL = 30

# How many historical boards the conditional tool reads. 120 is ~3 s of the
# retrieval index's coarse pass on the first call and is cached by the caller's
# LRU, not here; larger mostly adds weaker matches, because the exact-overlap
# ranking is what orders them and the tail has overlap 1-2.
SIMILAR_DEFAULT = 120


# ------------------------------------------------------------------- board
class Board:
    """The live board in the shape these tools need.

    `picks` is the UI's 10-slot pick-order board (None for an empty slot),
    `bans` the flat ban list, `unavailable` the fearless pool. Every champion
    argument in this module is an *index*, never a Riot id - the mapping stops
    at the HTTP edge, the same as everywhere else in the frontend.
    """

    __slots__ = ("picks", "bans", "unavailable", "first_pick", "patch",
                 "num_champs")

    def __init__(self, picks, bans, first_pick=0, unavailable=None,
                 patch=None, num_champs=173):
        self.picks = [None if p is None else int(p) for p in picks]
        self.bans = [int(b) for b in bans]
        self.unavailable = [int(u) for u in (unavailable or [])]
        self.first_pick = 1 if int(first_pick) else 0
        self.patch = patch
        self.num_champs = int(num_champs)

    @property
    def placed(self):
        return [c for c in self.picks if c is not None]

    @property
    def move_idx(self):
        """Moves made so far. A ban is a move, so the two counts just add."""
        return len(self.bans) + len(self.placed)

    @property
    def taken(self):
        return set(self.bans) | set(self.placed)

    @property
    def complete(self):
        return self.move_idx >= N_MOVES

    def legal(self):
        """Champions that can still be named, fearless pool included."""
        gone = self.taken | set(self.unavailable)
        return [i for i in range(self.num_champs) if i not in gone]


def _result(tool, args, n, reliable, result, note=None):
    return {"tool": tool, "args": args, "n": int(n), "reliable": bool(reliable),
            "result": result, "note": note}


# ------------------------------------------------------------------ corpus
class Corpus:
    """Corpus-wide counters, built once and lazily.

    The pair counters are the expensive part - 46,915 drafts x 30 cells is
    ~1.4M increments, about two seconds - so they are built behind a lock on
    first use rather than at import. Importing this module stays free, and a
    request that only needs `legal_pool` never pays for them.
    """

    def __init__(self, index):
        self.index = index
        self.rows = index.rows
        self._lock = threading.Lock()
        self._picks = None
        self._wins = None
        self._same_team = None
        self._opposing = None
        self._same_role = None

    # -- build
    def _build(self):
        picks, wins = Counter(), Counter()
        same_team, opposing, same_role = Counter(), Counter(), Counter()
        for row in self.rows:
            champ = row.get("champ") or []
            blue = [int(c) for c in champ[:5] if c is not None]
            red = [int(c) for c in champ[5:] if c is not None]
            blue_win = row.get("blue_win")
            for c in blue:
                picks[c] += 1
                if blue_win is not None and int(blue_win) == 1:
                    wins[c] += 1
            for c in red:
                picks[c] += 1
                if blue_win is not None and int(blue_win) == 0:
                    wins[c] += 1
            for team in (blue, red):
                for i, a in enumerate(team):
                    for b in team[i + 1:]:
                        same_team[(min(a, b), max(a, b))] += 1
            for a in blue:
                for b in red:
                    opposing[(min(a, b), max(a, b))] += 1
            # Role order: champ[i] is blue's i-th role, champ[5 + i] is red's.
            for i in range(min(5, len(blue), len(red))):
                a, b = blue[i], red[i]
                same_role[(min(a, b), max(a, b))] += 1
        with self._lock:
            if self._picks is None:
                (self._picks, self._wins, self._same_team,
                 self._opposing, self._same_role) = (
                    picks, wins, same_team, opposing, same_role)
        return self._picks, self._wins, self._same_team, self._opposing, \
            self._same_role

    def _ensure(self):
        return (self._picks, self._wins, self._same_team, self._opposing,
                self._same_role) if self._picks is not None else self._build()

    # -- cells
    def marginal(self, idx):
        picks, wins, _st, _op, _sr = self._ensure()
        return picks.get(int(idx), 0), wins.get(int(idx), 0)

    def pair(self, a, b):
        """-> (same_team, opposing, same_role) counts for an unordered pair."""
        _p, _w, st, op, sr = self._ensure()
        key = (min(int(a), int(b)), max(int(a), int(b)))
        return st.get(key, 0), op.get(key, 0), sr.get(key, 0)

    # -- retrieval
    def similar(self, board, k=SIMILAR_DEFAULT):
        """Historical boards that match this one, with the move played from it."""
        if not board.taken or board.complete:
            return [], False
        return self.index.similar(board.taken, board.move_idx, board.first_pick,
                                  k=k, patch=board.patch)


# ------------------------------------------------------------------- tools
def legal_pool(corpus, board, probs, names, k=12, lane=None):
    """What can still be named, best-ranked by the head.

    `probs` is the head's distribution over *all* champions for this move, as
    idx -> float. The caller has it already - the chat loop needs the full
    vector to blend a user's insistence against - and recomputing it here would
    double the forward pass for no gain.

    `lane` restricts to the champions that play it. It filters the legal set
    rather than the head's top-k, for the reason recorded in
    `infer.next_move`: filtering a top-k empties the list whenever the head's
    favourites do not play the lane, which is 13 % of pick moves.
    """
    legal = board.legal()
    if lane is not None:
        from roles import legal_champions
        allowed = set(legal_champions(int(lane)))
        legal = [c for c in legal if c in allowed]
    ranked = sorted(legal, key=lambda c: (-float(probs.get(c, 0.0)), c))[:k]
    return _result(
        "legal_pool", {"lane": lane}, len(legal), True,
        {"n_legal": len(legal),
         "n_taken": len(board.taken),
         "n_unavailable": len(board.unavailable),
         "top": [{"idx": c, "name": names[c],
                  "prob": round(float(probs.get(c, 0.0)), 5)}
                 for c in ranked]},
        note=None if legal else "no legal champion is left for this move")


def champ_evidence(corpus, board, idx, names, k=SIMILAR_DEFAULT):
    """What the corpus says about one champion, on this board and overall.

    Two numbers, kept apart on purpose:

      marginal     picks and wins over the whole corpus. Well powered (median
                   1,479 games) and nearly worthless, because it cannot see the
                   board: a champion Bradley-Terry prior is almost all of the
                   0.0031 nats of draft signal the outcome data holds
                   (STATUS.md §17). It is the part the head already knows.
      conditional  over the historical boards that match this one, how often the
                   next move *was* this champion, and how often the side that
                   took it won. This is the number a coach asks for, and the one
                   that is usually too thin to quote - so the tool says which.
    """
    idx = int(idx)
    games, relaxed = corpus.similar(board, k=k)
    n_pick_moves = 0
    n_next = 0
    n_decided = 0
    n_won = 0
    for g in games:
        if g["next_kind"] != "pick":
            continue
        n_pick_moves += 1
        if g["next_champ_idx"] != idx:
            continue
        n_next += 1
        blue_win = g.get("blue_win")
        if blue_win is None:
            continue
        n_decided += 1
        took_blue = g["next_side"] == "blue"
        if int(blue_win) == (1 if took_blue else 0):
            n_won += 1

    picks, wins = corpus.marginal(idx)
    share = (n_next / n_pick_moves) if n_pick_moves else None
    # Same rule as the marginal: a rate is only reported when the cell can
    # carry one. `share` is the exception and is always reported, because it is
    # a count over the boards that matched (`n_next` of `n_pick_moves`) and the
    # caller sees both halves of the fraction.
    rate = (n_won / n_decided) if n_decided >= MIN_CELL else None
    if n_pick_moves == 0:
        # Two different nothings, and they read differently to a coach: an empty
        # board has no prefix to match, a full one can have no overlap. Say
        # which, because "0 of 0" invites the reader to think the tool failed.
        note = ("no board to condition on yet - the marginal is all there is"
                if not board.taken else
                "no historical board shares a champion with this one")
    else:
        thin = []
        if n_next < MIN_CELL:
            thin.append("only %d of %d similar boards played it next"
                        % (n_next, n_pick_moves))
        if n_decided < MIN_CELL:
            thin.append("only %d of those games have an outcome" % n_decided)
        note = "; ".join(thin) if thin else None
    return _result(
        "champ_evidence",
        {"name": names[idx], "move_idx": board.move_idx, "k": k},
        n_pick_moves, n_next >= MIN_CELL,
        {"marginal": {"picks": picks, "wins": wins,
                      "rate": (round(wins / picks, 4)
                               if picks >= MIN_CELL else None),
                      "reliable": picks >= MIN_CELL},
         "conditional": {"n_boards": len(games),
                         "n_pick_moves": n_pick_moves,
                         "n_next": n_next,
                         "n_decided": n_decided,
                         "n_won": n_won,
                         "share": None if share is None else round(share, 4),
                         "rate": None if rate is None else round(rate, 4),
                         "reliable": n_next >= MIN_CELL},
         "relaxed_side": bool(relaxed)},
        note=note)


def next_move_counts(corpus, board, names, k=SIMILAR_DEFAULT, top=8):
    """What the similar boards actually played next - the whole choice, not one cell.

    Added because the per-champion cell is usually empty. At move 9, a *named*
    champion is the next move in 0-2 of the 120 nearest boards: asking
    `champ_evidence` for one champion asks one cell of a ten-way choice, and the
    probe already said those cells are thin. This aggregates the same 120 boards
    the way the decision is actually made, so the leader is 8-15 games instead of
    0-2. Still not a rate worth quoting, and the tool says so, but it is a count
    a coach can act on.
    """
    games, relaxed = corpus.similar(board, k=k)
    counts = Counter()
    for g in games:
        counts[g["next_champ_idx"]] += 1
    n = sum(counts.values())
    ranked = counts.most_common(int(top))
    return _result(
        "next_move_counts", {"move_idx": board.move_idx, "k": k},
        n, bool(ranked) and ranked[0][1] >= MIN_CELL,
        {"n_boards": len(games),
         "n_moves": n,
         "top": [{"idx": c, "name": names[c], "n": v,
                  "share": round(v / n, 4) if n else None}
                 for c, v in ranked],
         "relaxed_side": bool(relaxed)},
        note=(None if ranked else "no similar board to count"))


def pair_evidence(corpus, a, b, names):
    """How often two champions have met, in the three senses that differ.

    The distinction is the whole point of the tool. `same_team` is a synergy
    question, `opposing` is a matchup question at team granularity, and
    `same_role` is the one a counter-pick claim is actually about - and it is
    the thinnest: median 5 games across the corpus, 73.6 % of cells under 20.
    """
    a, b = int(a), int(b)
    same_team, opposing, same_role = corpus.pair(a, b)
    cells = {"same_team": same_team, "opposing": opposing,
             "same_role": same_role}
    thin = [k for k, v in cells.items() if v < MIN_CELL]
    return _result(
        "pair_evidence", {"a": names[a], "b": names[b]},
        opposing, same_role >= MIN_CELL,
        {"counts": cells,
         "reliable": {k: v >= MIN_CELL for k, v in cells.items()},
         "reversed": a > b},
        note=("thin cells (n<%d): %s" % (MIN_CELL, ", ".join(thin))
              if thin else None))


def similar_drafts(corpus, board, names, k=5):
    """The historical boards the conditional numbers were computed from.

    Returned so the reply can name a game instead of asserting a tendency.
    """
    games, relaxed = corpus.similar(board, k=k)
    return _result(
        "similar_drafts", {"move_idx": board.move_idx, "k": k},
        len(games), bool(games),
        {"games": [{"gameid": g["gameid"], "patch": g["patch"],
                    "league": g["league"], "date": g["date"],
                    "blue_team": g["blue_team"], "red_team": g["red_team"],
                    "next_kind": g["next_kind"], "next_side": g["next_side"],
                    "next_champ": names[g["next_champ_idx"]],
                    "blue_win": g["blue_win"], "overlap": g["overlap"]}
                   for g in games],
         "relaxed_side": bool(relaxed)},
        note=("no historical board shares a champion with this one"
              if not games else None))


TOOLS = {"legal_pool": legal_pool, "champ_evidence": champ_evidence,
         "next_move_counts": next_move_counts, "pair_evidence": pair_evidence,
         "similar_drafts": similar_drafts}


def run(corpus, board, tool, names, probs=None, **args):
    """Dispatch by name, so an HTTP route can expose them without a table.

    Champion arguments arrive as names or indices; both are accepted because
    the chat parser works in names and the rest of the frontend in indices.
    """
    if tool not in TOOLS:
        raise ValueError("unknown tool %r (have: %s)"
                         % (tool, ", ".join(sorted(TOOLS))))
    for key in ("idx", "a", "b"):
        if isinstance(args.get(key), str):
            args[key] = _idx_of(names, args[key])
    if tool == "legal_pool":
        return legal_pool(corpus, board, probs or {}, names, **args)
    if tool == "champ_evidence":
        return champ_evidence(corpus, board, args.pop("idx"), names, **args)
    if tool == "pair_evidence":
        return pair_evidence(corpus, args.pop("a"), args.pop("b"), names)
    return TOOLS[tool](corpus, board, names, **args)


def _idx_of(names, name):
    key = str(name).strip().lower()
    for i, n in enumerate(names):
        if n.lower() == key:
            return i
    raise ValueError("unknown champion %r" % (name,))


# ----------------------------------------------------------------- selftest
def _selftest():
    """Run against the real corpus; prints the cells the README quotes."""
    import llm
    index = llm.ProIndex.load()
    corpus = Corpus(index)
    names = _load_names(173)
    empty = Board([None] * 10, [], first_pick=0, patch="16.20")
    print("corpus %d drafts" % len(corpus.rows))
    lp = legal_pool(corpus, empty, {i: 1.0 / 173 for i in range(173)}, names)
    print("legal_pool n=%d top=%s" % (lp["n"], [c["name"] for c in
                                                lp["result"]["top"][:3]]))
    ev = champ_evidence(corpus, empty, _idx_of(names, "Vi"), names)
    print("empty board, Vi -> n=%d reliable=%s note=%r"
          % (ev["n"], ev["reliable"], ev["note"]))
    print("  marginal", ev["result"]["marginal"])

    # A real ban-phase board: six bans, three picks, blue first pick.
    bans = [_idx_of(names, n) for n in
            ("Yuumi", "Zeri", "Azir", "Renata Glasc", "Sejuani", "Maokai")]
    picks = [_idx_of(names, n) for n in ("Vi", "Orianna", "Rakan")] + [None] * 7
    board = Board(picks, bans, first_pick=0, patch="16.20")
    print("\nboard: %d bans, %d picks, move_idx %d"
          % (len(bans), len(picks) - picks.count(None), board.move_idx))
    sd = similar_drafts(corpus, board, names, k=3)
    print("similar_drafts n=%d %s"
          % (sd["n"], [(g["gameid"], g["next_champ"]) for g in
                       sd["result"]["games"]]))
    for who in ("Kled", "Vi"):
        ev = champ_evidence(corpus, board, _idx_of(names, who), names)
        c = ev["result"]["conditional"]
        print("champ_evidence %-5s n=%d reliable=%-5s next %d/%d (share %s) "
              "won %d/%d (rate %s) note=%r"
              % (who, ev["n"], ev["reliable"], c["n_next"], c["n_pick_moves"],
                 c["share"], c["n_won"], c["n_decided"], c["rate"], ev["note"]))
    nm = next_move_counts(corpus, board, names, top=5)
    print("next_move_counts n=%d reliable=%s %s"
          % (nm["n"], nm["reliable"],
             [(c["name"], c["n"]) for c in nm["result"]["top"]]))
    pe = pair_evidence(corpus, _idx_of(names, "Kled"), _idx_of(names, "Olaf"),
                       names)
    print("pair_evidence Kled/Olaf -> %s  note=%r"
          % (pe["result"]["counts"], pe["note"]))
    return 0


def _load_names(n):
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    proc = root / "data" / "solo queue" / "process matchdata"
    with open(proc / "champ_mapping.json", encoding="utf-8") as f:
        mapping = {int(k): int(v) for k, v in json.load(f)["riot_to_idx"].items()}
    with open(proc / "champ_names.json", encoding="utf-8") as f:
        riot_to_name = {int(k): v for k, v in json.load(f)["names"].items()}
    names = ["Riot%d" % i for i in range(n)]
    for riot, i in mapping.items():
        if riot in riot_to_name and i < n:
            names[i] = riot_to_name[riot]
    return names


if __name__ == "__main__":
    raise SystemExit(_selftest())
