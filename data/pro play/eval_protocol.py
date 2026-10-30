"""Shared evaluation protocol for the pro draft models.

Why this module exists. The old protocol reported one pooled number per split
(`val top-5 0.3796` vs `test top-5 0.2558`) and treated the drop as model
degradation. It is not, or not only: measured on this corpus, a model-free
popularity baseline fitted on train scores **0.0879 on val and 0.0707 on
test** (ratio 0.81), because the two windows differ in patch novelty and in
league mix. The model's ratio is 0.67, so roughly half the apparent
"generalisation loss" is the window getting harder, and the rest is real.
A pooled number cannot tell those apart, and it also hides heads that are
broken outright (pro test `win_auc` was 0.4969, chance, and `econ_skill`
-0.153, worse than predicting the mean).

What this module provides:

* `patch_honest_val` - a validation window restricted to patches that never
  appear in train, so val sits in the same condition as test (test is 100%
  unseen patches; the raw val window is only 81.5% unseen).
* `RankReport` - accumulates the rank of the true move/champion and slices it
  by patch, league tier, move index and pick/ban, so a mix shift shows up as a
  per-slice change instead of as one mysterious number.
* `wilson` - a CI for every rate. Patch 16.20 has 196 drafts and 84 series;
  a bare "top-5 = 0.28" invites over-reading, "0.28 +/- 0.02 (n=3920)" does not.
* `baseline_relative` - model top-k over the popularity baseline top-k on the
  same rows, the only score that is comparable across windows of different
  difficulty. This is what checkpoint selection uses.

Reporting conveniences, not truth claims: `LEAGUE_TIERS` is an explicit,
editable list. Anything not listed is reported as "other" rather than being
silently folded into a tier.
"""
import math
from collections import defaultdict

# ------------------------------------------------------------------ leagues
# Riot's tier-1 domestic leagues + the international events. Everything else
# (academies, national leagues, second divisions) is "other". Edit freely -
# the point is that the report shows the mix instead of hiding it.
TIER1 = {
    "LPL", "LCK", "LEC", "LCS", "LTA", "LTA N", "LTA S", "LCP",
    "PCS", "VCS", "LJL", "LLA", "CBLOL", "TCL",
}
INTERNATIONAL = {
    "WLDs", "Worlds", "MSI", "EWC", "First Stand", "KeSPA Cup",
    "Rift Rivals", "All-Star",
}
# known academy / second-division labels, kept explicit so the tier is
# auditable rather than inferred
ACADEMY = {"LCKC", "NACL", "EM", "LAS", "PRM", "AL", "NLC", "LFL", "LFL2",
           "LIT", "HC", "HM", "CD", "ROL", "LES", "EBL", "RL", "LPLOL",
           "LRS", "LRN", "Ultraliga", "PG Nationals", "Hitpoint", "LCL"}


def league_tier(league):
    """major / international / academy / other, from the explicit sets above."""
    lg = (league or "").strip()
    if lg in INTERNATIONAL:
        return "international"
    if lg in TIER1:
        return "major"
    if lg in ACADEMY:
        return "academy"
    return "other"


# --------------------------------------------------------------------- split
def train_patches(rows, train_idx):
    """Set of patch labels that appear anywhere in the training split."""
    return {rows[i].get("patch") for i in train_idx}


def patch_honest_val(rows, splits, min_rows=400, verbose=True):
    """Val restricted to patches that never occur in train.

    Rationale: test is 100% unseen patches, so a val window that contains
    familiar patches measures a condition the model will not be judged in.
    On this corpus the filter drops 18.5% of val rows (all of patch 16.07-16.08,
    the tail of train's own patch range) and keeps 2,458.

    Falls back to the full window - with a printed warning - if filtering would
    leave fewer than `min_rows` rows, because an empty val is worse than a
    slightly optimistic one.
    """
    seen = train_patches(rows, splits["train"])
    kept = [i for i in splits["val"] if rows[i].get("patch") not in seen]
    if len(kept) < min_rows:
        if verbose:
            print("patch-honest val: only %d rows survive (min %d) -> keeping "
                  "the full val window and reporting the mix instead"
                  % (len(kept), min_rows))
        return list(splits["val"]), {"honest": False, "kept": len(kept),
                                     "dropped": 0}
    dropped = len(splits["val"]) - len(kept)
    if verbose:
        print("patch-honest val: %d rows (%d dropped, %.1f%%, all familiar "
              "patches)" % (len(kept), dropped,
                            100.0 * dropped / max(1, len(splits["val"]))))
    return kept, {"honest": True, "kept": len(kept), "dropped": dropped}


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial rate. Returns (lo, hi).

    Wilson rather than normal-approx because the slices get small (patch 16.20
    is 84 series) and the normal interval misbehaves near 0/1 and under-covers.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def auc(scores, labels):
    """Tie-averaged rank AUC (Mann-Whitney). 0.5 on a degenerate split."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(pairs)
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # average ranks over ties, 1-based
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[t] = avg
        i = j + 1
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def baseline_relative(model_topk, base_topk):
    """Model top-k divided by the popularity top-k on the same rows.

    >1 means the model beats a patch-frequency lookup table; the *ratio* is
    what is comparable across windows, because the denominator carries the
    window's intrinsic difficulty.
    """
    if not base_topk:
        return float("nan")
    return model_topk / base_topk


# ------------------------------------------------------------------ baseline
def move_popularity_ranks(rows, indices, num_champs, n_moves=20):
    """Popularity baseline for NEXT-MOVE prediction.

    For each move index, the weighted-frequency rank of every champion: at
    move 3 the baseline "knows" that move 3 is usually a ban of whatever is
    banned most at move 3, which is a much stronger baseline than one global
    pick list. Returns ranks[move_idx][champ_id] -> 0 is most likely.

    Weights are the recency weights, so the baseline reflects the same
    recency emphasis the model is trained under.
    """
    counts = [defaultdict(float) for _ in range(n_moves)]
    for i in indices:
        row = rows[i]
        w = float(row.get("weight", 1.0))
        for m, move in enumerate(row["moves"]):
            if m < n_moves:
                counts[m][int(move[2])] += w
    ranks = []
    for m in range(n_moves):
        c = counts[m]
        order = sorted(range(num_champs), key=lambda x: (-c.get(x, 0.0), x))
        rank = [0] * num_champs
        for pos, champ in enumerate(order):
            rank[champ] = pos
        ranks.append(rank)
    return ranks


def pick_popularity_ranks(rows, indices, num_champs):
    """Popularity baseline for COMPOSITION metrics.

    A masked model predicts a hidden champion from the rest of the draft; the
    question is whether that beats simply naming the most-picked champions of
    the training era. Returns rank[champ_id] -> 0 is most picked. Weighted by
    the recency weights, like everything else here.

    This is the *weak* null (it ignores which slot is being predicted). Prefer
    `slot_popularity_ranks` when the slot index is known, because in pro drafts
    slot -> role is fixed and the model sees the role.
    """
    counts = defaultdict(float)
    for i in indices:
        row = rows[i]
        w = float(row.get("weight", 1.0))
        for c in row["champ"]:
            counts[int(c)] += w
    order = sorted(range(num_champs), key=lambda x: (-counts.get(x, 0.0), x))
    rank = [0] * num_champs
    for pos, champ in enumerate(order):
        rank[champ] = pos
    return rank


def slot_popularity_ranks(rows, indices, num_champs, n_slots=10):
    """Per-slot popularity baseline for composition metrics.

    Returns ranks[slot][champ_id] -> 0 is the most common champion in that
    slot. Slot 0 is always the blue top laner in a pro draft, so this null
    model already knows "slot 0 is a top laner" and is a much stronger - and
    therefore fairer - baseline than a single global pick list. A global list
    scores 0% at some slots simply because the most-picked champions never
    appear there, which makes the relative score meaningless.
    """
    counts = [defaultdict(float) for _ in range(n_slots)]
    for i in indices:
        row = rows[i]
        w = float(row.get("weight", 1.0))
        for slot, c in enumerate(row["champ"]):
            if slot < n_slots:
                counts[slot][int(c)] += w
    ranks = []
    for slot in range(n_slots):
        c = counts[slot]
        order = sorted(range(num_champs), key=lambda x: (-c.get(x, 0.0), x))
        rank = [0] * num_champs
        for pos, champ in enumerate(order):
            rank[champ] = pos
        ranks.append(rank)
    return ranks


def self_popularity_topk(rows, indices, num_champs, k=5, get=None):
    """Reference: top-k of a frequency table built on THESE rows themselves.

    This is the "just use this patch's pick rates" bar, and it is the one that
    matters on a new patch: the train-era table is stale there, so a model can
    look good (relative > 1) purely because the baseline decayed. If the model
    does not beat the same-patch table, then for that patch a frequency table
    is as good as the model.

    Each row is excluded from the table it is scored against (leave-one-out),
    so this is not leakage - it is "the table a human would write after
    watching every game except this one". Without that, the table contains the
    answer and the reference is inflated.

    `get(row)` yields (index, champ_id) pairs; the default is the 10 draft
    slots, pass a move lambda for next-move metrics.

    `k` may be an int (returns {"n", "top<k>"}) or a sequence (returns every
    requested k), so one pass can serve both the top-1 and top-5 reference.
    """
    ks = (k,) if isinstance(k, int) else tuple(k)
    if get is None:
        def get(row):
            return list(enumerate(row["champ"]))

    counts = defaultdict(lambda: defaultdict(int))
    pairs = []
    for i in indices:
        row = rows[i]
        for idx, champ in get(row):
            counts[idx][int(champ)] += 1
            pairs.append((idx, int(champ)))
    if not pairs:
        return {("top%d" % kk): float("nan") for kk in ks} | {"n": 0}

    # suffix[idx][c] = number of champions with count >= c in that slot
    suffix = {}
    for idx, c in counts.items():
        hist = [0] * (max(c.values()) + 1)
        for v in c.values():
            hist[v] += 1
        suf = [0] * (len(hist) + 1)
        for v in range(len(hist) - 1, -1, -1):
            suf[v] = suf[v + 1] + hist[v]
        suffix[idx] = suf

    hits = {kk: 0 for kk in ks}
    for idx, champ in pairs:
        c = counts[idx]
        n_c = c[champ]
        suf = suffix[idx]
        # After removing this row champ has n_c-1; everyone else is unchanged.
        # Rank = #{d != champ : count_d > n_c-1} = #{d : count_d >= n_c} - 1,
        # i.e. only *strictly more frequent* champions outrank it, so tied
        # champions all count as hits. Breaking ties by champion id instead
        # would make a one-row table score 0.4 at k=5 purely because of id
        # order, which is an artifact, not a baseline.
        rank = (suf[n_c] if n_c < len(suf) else 0) - 1
        for kk in ks:
            hits[kk] += int(rank < kk)
    out = {"n": len(pairs)}
    for kk in ks:
        out["top%d" % kk] = hits[kk] / len(pairs)
    return out


def self_move_topk(rows, indices, num_champs, k=(1, 5)):
    """Same-patch leave-one-out reference for NEXT-MOVE metrics.

    Grouped by move index, because move 3 is a ban and move 14 is a pick; a
    single global list would be a strawman nobody would actually use.
    """
    def get(row):
        return [(m, int(mv[2])) for m, mv in enumerate(row["moves"])]
    return self_popularity_topk(rows, indices, num_champs, k=k, get=get)


def self_slot_topk(rows, indices, num_champs, k=(1, 5)):
    """Same-patch leave-one-out reference for COMPOSITION metrics (per slot)."""
    return self_popularity_topk(rows, indices, num_champs, k=k)


# ------------------------------------------------------------------- reports
class RankReport:
    """Per-example "rank of the true answer" accumulator with slicing.

    One accumulator serves both model kinds, because both reduce to the same
    question - where did the truth rank in the model's ordering?
      * next-move (policy):   rank of the actual move among all champions
      * composition (masked): rank of the hidden champion among all champions
    The only difference is the label in `kind` and what `idx` means (move index
    vs draft slot).
    """

    def __init__(self, ks=(1, 5)):
        self.ks = tuple(sorted(ks))
        self._patch = []
        self._league = []
        self._idx = []
        self._kind = []
        self._rank = []

    def __len__(self):
        return len(self._rank)

    def add(self, ranks, patch, league, idx, kind):
        """Append one batch. All arguments are equal-length sequences (or
        scalars, which are broadcast)."""
        ranks = _as_list(ranks)
        n = len(ranks)
        self._patch.extend(_broadcast(patch, n))
        self._league.extend(_broadcast(league, n))
        self._idx.extend(_broadcast(idx, n))
        self._kind.extend(_broadcast(kind, n))
        self._rank.extend(int(r) for r in ranks)

    # -- slicing ---------------------------------------------------------
    def _key(self, name, i):
        if name == "patch":
            return self._patch[i]
        if name == "league":
            return self._league[i]
        if name == "tier":
            return league_tier(self._league[i])
        if name == "idx":
            return self._idx[i]
        if name == "kind":
            return self._kind[i]
        raise KeyError(name)

    def metrics(self, where=None):
        """{n, top1, top5, ci1, ci5, mean_rank} over the selected examples."""
        ranks = [r for i, r in enumerate(self._rank)
                 if where is None or where(i)]
        n = len(ranks)
        out = {"n": n}
        if not n:
            for k in self.ks:
                out["top%d" % k] = float("nan")
            return out
        for k in self.ks:
            hits = sum(1 for r in ranks if r < k)
            lo, hi = wilson(hits, n)
            out["top%d" % k] = hits / n
            out["ci%d" % k] = (lo, hi)
        out["mean_rank"] = sum(ranks) / n
        return out

    def by(self, name, min_n=1, order=None):
        """Group metrics by a slice name: patch | league | tier | idx | kind."""
        groups = defaultdict(list)
        for i in range(len(self._rank)):
            groups[self._key(name, i)].append(i)
        out = {}
        for key, idxs in groups.items():
            if len(idxs) < min_n:
                continue
            sel = set(idxs)
            out[key] = self.metrics(where=lambda i, s=sel: i in s)
        if order == "natural":
            out = dict(sorted(out.items(), key=lambda kv: str(kv[0])))
        elif order == "n":
            out = dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))
        return out

    def where(self, name, value):
        """Metrics restricted to one slice value, e.g. ("patch", "16.20")."""
        return self.metrics(where=lambda i: self._key(name, i) == value)

    def summary(self, key="top5", min_n=1):
        """Compact {slice: (metric, n, ci)} for printing."""
        return {k: (v[key], v["n"], v.get("ci5" if key == "top5" else "ci1"))
                for k, v in self.by("patch", min_n=min_n).items()}


def format_table(report, key="top5", title="", indent=0):
    """Human-readable per-slice table. `report` is RankReport.by(...) output."""
    pad = " " * indent
    lines = []
    if title:
        lines.append(pad + title)
    ci_key = "ci%d" % int(key.replace("top", ""))
    lines.append(pad + "  %-14s %7s %8s %s" % ("slice", "n", key, "95% CI"))
    for name, m in report.items():
        lo, hi = m.get(ci_key, (float("nan"), float("nan")))
        lines.append(pad + "  %-14s %7d %8.4f  [%.4f, %.4f]"
                     % (name, m["n"], m.get(key, float("nan")), lo, hi))
    return "\n".join(lines)


def compare(rep, base_rep, name, key="top5", min_n=1, order="natural"):
    """Model metrics next to the popularity baseline for the same slices.

    Returns a list of row dicts so callers can print it or dump it as JSON.
    `rel` is the baseline-relative score (model / baseline); it is None when
    no baseline was supplied.
    """
    a = rep.by(name, min_n=min_n, order=order)
    b = base_rep.by(name, min_n=min_n, order=order) if base_rep else {}
    ci_key = "ci%d" % int(key[3:])
    rows = []
    for slice_name, m in a.items():
        bm = b.get(slice_name) or {}
        base_val = bm.get(key)
        rows.append({
            "slice": slice_name,
            "n": m["n"],
            key: m[key],
            "ci": m.get(ci_key),
            "base": base_val,
            "rel": (baseline_relative(m[key], base_val)
                    if base_val else None),
        })
    return rows


def format_compare(rows, key="top5", title="", indent=0):
    """Table: slice | n | model@k | baseline@k | relative."""
    pad = " " * indent
    out = []
    if title:
        out.append(pad + title)
    out.append(pad + "  %-14s %7s %8s %9s %7s" %
               ("slice", "n", "model", "baseline", "rel"))
    for r in rows:
        base = "-" if r["base"] is None else "%.4f" % r["base"]
        rel = "-" if r["rel"] is None else "%.2fx" % r["rel"]
        out.append(pad + "  %-14s %7d %8.4f %9s %7s"
                   % (str(r["slice"]), r["n"], r[key], base, rel))
    return "\n".join(out)


# ------------------------------------------------------- stratified report
# The patch the model is actually asked about. 16.20 is the newest patch in
# the corpus (196 drafts / 84 series, 2026-10-09 -> 2026-10-24); it is the only
# window where "will this work next week" is answerable at all.
FOCUS_PATCH = "16.20"

# Emitted in this order; anything not listed still appears, at the end.
_TIER_ORDER = ("major", "international", "academy", "other")


def _patch_key(p):
    """Sort patches numerically ("16.9" < "16.10"), unparseable ones last."""
    try:
        return tuple(int(x) for x in str(p).split("."))
    except (TypeError, ValueError):
        return (1 << 30,)


def _idx_key(i):
    """Sort move indices / slots numerically, non-numeric labels last.

    Needed because insertion order is not reliable: the composition evaluator
    visits masked slots in (draft, slot) order, so with ~1.5 masks per draft the
    first-appearance order is e.g. 2,1,3,6,9,0,... rather than 0..9. Sorting by
    string would put move 10 between 1 and 2.
    """
    try:
        return (0, int(i), "")
    except (TypeError, ValueError):
        return (1, 0, str(i))


def _ordered(d, prefer=None, keyfn=None):
    """Reorder a dict for stable, human-sane report output."""
    if keyfn is not None:
        return dict(sorted(d.items(), key=lambda kv: keyfn(kv[0])))
    if prefer is not None:
        rank = {k: i for i, k in enumerate(prefer)}
        return dict(sorted(d.items(),
                           key=lambda kv: (rank.get(kv[0], len(prefer)),
                                           str(kv[0]))))
    return d


def focus_block(rep, patch, base_rep=None, self_ref=None, key="top5"):
    """One patch, with CI and BOTH reference points.

    `base_rep` is the train-fitted popularity table: the weak null, and the
    one whose decay makes a stale model look good on a new patch.
    `self_ref` is the leave-one-out same-patch table: the strong null, i.e.
    "the table a human writes after watching every game on this patch except
    this one". If `rel_self` is ~1 the model has added nothing on that patch.
    """
    m = rep.where("patch", patch)
    if not m["n"]:
        return {"patch": patch, "n": 0}
    ci_key = "ci%d" % int(key[3:])
    out = {"patch": patch, "n": m["n"], key: m[key],
           "ci": m.get(ci_key), "top1": m.get("top1"),
           "ci1": m.get("ci1"), "mean_rank": m.get("mean_rank")}
    base = base_rep.where("patch", patch).get(key) if base_rep else None
    out["base"] = base
    out["rel"] = baseline_relative(m[key], base) if base else None
    if self_ref:
        ref = self_ref.get(key)
        out["self_ref"] = ref
        out["rel_self"] = baseline_relative(m[key], ref) if ref else None
    return out


def stratified(rep, base_rep=None, focus=None, self_ref=None, key="top5",
               min_n=1, slices=("patch", "tier", "league", "idx", "kind"),
               label=None):
    """JSON-able report: overall, every slice, and a focus-patch block.

    This is the thing that gets written to `test_report.json`. A pooled number
    cannot distinguish "the model got worse" from "the window got harder", and
    it hides heads that are broken outright; a per-slice table with n and a CI
    makes both visible in one screen.
    """
    out = {}
    if label:
        out["label"] = label
    out["overall"] = rep.metrics()
    for name in slices:
        # league is alphabetical; patch/tier/kind are re-ordered below and
        # idx must keep insertion order (0..19), which a string sort breaks
        rows = compare(rep, base_rep, name, key=key, min_n=min_n,
                       order="natural" if name == "league" else None)
        d = {str(r["slice"]): {k: v for k, v in r.items()
                               if k != "slice"} for r in rows}
        if name == "patch":
            d = _ordered(d, keyfn=_patch_key)
        elif name == "idx":
            d = _ordered(d, keyfn=_idx_key)
        elif name == "tier":
            d = _ordered(d, prefer=_TIER_ORDER)
        elif name == "kind":
            d = _ordered(d, prefer=("pick", "ban", "role_visible",
                                    "role_hidden"))
        out["by_" + name] = d
    if focus is not None:
        out["focus"] = focus_block(rep, focus, base_rep, self_ref, key=key)
    return out


def format_stratified(report, key="top5", indent=0):
    """Human-readable version of `stratified(...)`."""
    pad = " " * indent
    out = []
    ov = report["overall"]
    lo, hi = ov.get("ci%d" % int(key[3:]), (float("nan"), float("nan")))
    out.append(pad + "overall n=%d %s=%.4f [%.4f, %.4f] top1=%.4f"
               % (ov["n"], key, ov.get(key, float("nan")), lo, hi,
                  ov.get("top1", float("nan"))))
    for name, d in report.items():
        if not name.startswith("by_"):
            continue
        out.append(pad + "  " + name[3:] + ":")
        for slice_name, m in d.items():
            base = "-" if m.get("base") is None else "%.4f" % m["base"]
            rel = "-" if m.get("rel") is None else "%.2fx" % m["rel"]
            out.append(pad + "    %-14s %7d %8.4f %9s %7s"
                       % (slice_name, m["n"], m[key], base, rel))
    f = report.get("focus")
    if f and f.get("n"):
        out.append(pad + "  focus patch %s: n=%d %s=%.4f [%.4f, %.4f] "
                   "top1=%.4f | train-table rel %s | same-patch rel %s"
                   % (f["patch"], f["n"], key, f[key],
                      f["ci"][0], f["ci"][1], f["top1"],
                      "-" if f.get("rel") is None else "%.2fx" % f["rel"],
                      "-" if f.get("rel_self") is None
                      else "%.2fx" % f["rel_self"]))
    return "\n".join(out)


def _as_list(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _broadcast(x, n):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError("length %d != %d ranks" % (len(x), n))
        return list(x)
    return [x] * n
