"""How well-powered is each statistic a champion tool could serve?

    python -m pro.model.llm._probe_tool_power

Asked because "give the LLM a champion-stats tool" is the obvious way to
ground a chat answer in numbers, and the question is not whether the numbers
can be computed - they can - but whether a specific query returns a cell with
enough games behind it to be worth saying out loud. A 4B model will cite
whatever it is handed, so a table that answers "Vi vs Lee Sin, jungle" with
n=5 is worse than no table: it reads like evidence.

Three granularities, and they are not equally powered:

  marginal      one champion, any board. Median 1,479 games per champion
                (min 16, p10 120) over 46,915 drafts. Usable.
  cross-team    any two champions on opposite sides. 13,244 of 14,878
                unordered pairs occur at all; median 24 games, and 46.3 % of
                pairs have fewer than 20.
  same-role     the pair a counter-pick question is actually about (both in
                the same lane). Median **5** games, 73.6 % of pairs under 20.

So the marginal is a fact, the cross-team pair is a weak fact, and the lane
pair is a coin flip for most champions. That is the argument for a
*conditional count* tool - "in N games with this board prefix, X won W" - over
a *rate* tool: the count can be reported honestly and refused when it is thin,
whereas a rate of 0.60 out of 5 games looks exactly like a rate of 0.60 out of
500 once it is in a prompt.

Note also that the marginal is the one quantity the head already knows: a
champion Bradley-Terry prior is nearly all of the 0.0031 nats of draft signal
in the outcome data (STATUS.md §17). A tool that serves champion strength is
serving the least interesting part of what the model already has, and it is
the part that is *unconditional* - it cannot see the board, so where it
disagrees with the head it is the one that is wrong.
"""
from collections import Counter

from pro.model.llm import common as C

N_CHAMPS = 173


def main():
    rows = C.load_rows("2022")
    print("drafts: %d" % len(rows))

    seen, pairs, lane_pairs = Counter(), Counter(), Counter()
    for r in rows:
        champ, blue_win = r["champ"], int(r["blue_win"])
        for c in champ:
            if c is not None:
                seen[int(c)] += 1
        blue = [int(c) for c in champ[:5] if c is not None]
        red = [int(c) for c in champ[5:] if c is not None]
        for b in blue:
            for e in red:
                pairs[(min(b, e), max(b, e))] += 1
        for i in range(5):
            b, e = champ[i], champ[5 + i]
            if b is not None and e is not None:
                lane_pairs[(min(int(b), int(e)), max(int(b), int(e)))] += 1

    def report(name, c, n_possible):
        v = sorted(c.values())
        thin = sum(1 for x in v if x < 20)
        print("\n%s" % name)
        print("  distinct cells   %d of %d" % (len(v), n_possible))
        print("  games per cell   median %d   p10 %d   max %d"
              % (v[len(v) // 2], v[len(v) // 10], v[-1]))
        print("  cells under 20   %d (%.1f %%)" % (thin, 100.0 * thin / len(v)))
        print("  observations in those cells  %.1f %%"
              % (100.0 * sum(x for x in v if x < 20) / sum(v)))

    report("marginal (one champion, any board)", seen, N_CHAMPS)
    report("cross-team pair (any two champions)", pairs,
           N_CHAMPS * (N_CHAMPS - 1) // 2)
    report("same-role pair (a counter question)", lane_pairs,
           N_CHAMPS * (N_CHAMPS - 1) // 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
