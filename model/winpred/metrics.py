"""Win-probability metrics: the five the experiment is judged on.

Everything here takes `y` (1 = blue won) and `p` (P(blue wins)) as flat
arrays, so a model, a prediction file and the prior baseline are all scored by
exactly the same code. That is the point: a comparison is only as good as the
claim that both sides were measured identically.

Why these five
--------------
* **Log-loss** - the proper scoring rule the model is trained on (CE). This is
  the number the run selects its epoch on, so it is the primary metric.
* **Brier** - proper, but bounded and therefore less punishing to a single
  confident mistake; it is the metric STATUS.md §6/§9 already gates value
  heads on ("do not surface a win probability before a value head beats the
  constant's Brier"), so it keeps this work comparable to that record.
* **ECE** - a proper scoring rule can be good while the probabilities are
  unusable as probabilities (over-confident but well ranked). ECE is the
  calibration check, and calibration is what makes "p > 0.65" mean anything.
* **ROC-AUC** - rank-only, so it is the metric that is *invariant* to the side
  bias and to the base rate. It answers "does the draft order the two sides
  correctly", separately from "is the number calibrated".
* **Precision/recall at high confidence** - the operational question. A tool
  that says "blue 71%" is only useful if the 71% bucket actually wins ~71% of
  the time (precision) and if it is confident often enough to matter (recall /
  coverage). Everything in between 0.35 and 0.65 is deliberately counted as
  "no opinion".

Skill scores (BSS / log-loss skill) are reported against the *base rate of the
scored set*, i.e. the best constant predictor on those rows. That is the honest
denominator: 0.6931 log-loss sounds bad until you notice the base rate's own
log-loss is 0.6929, and a Brier of 0.2495 is exactly the base rate.

Statistics. Every rate carries a Wilson interval, and paired comparisons carry
a **grouped** bootstrap: whole series are resampled, not rows, because two games
of a series share teams, patch and fearless pool. `bootstrap_delta` therefore
supports only the metrics that decompose into a per-row sum (log-loss, Brier,
accuracy) - an AUC difference needs the ranks, not the sums, and is reported
without an interval rather than with a wrong one.
"""
import math

import numpy as np

EPS = 1e-6

# Defaults for the "high confidence" band. 0.35/0.65 is the experiment's own
# definition (see the branch brief): below 0.35 predict red, above 0.65 predict
# blue, and treat the middle as an abstention.
CONF_LO = 0.35
CONF_HI = 0.65


# --------------------------------------------------------------- utilities
def _clean(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("y and p must be the same length (%d vs %d)"
                         % (y.size, p.size))
    if y.size == 0:
        raise ValueError("empty prediction set")
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite probability in p")
    bad = np.unique(y[~np.isin(y, (0.0, 1.0))])
    if bad.size:
        raise ValueError("y must be 0/1, found %s" % bad[:5])
    return y, np.clip(p, EPS, 1.0 - EPS)


def base_rate(y):
    return float(np.asarray(y, dtype=np.float64).mean())


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial rate (same as eval_protocol)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------- core metrics
def log_loss(y, p):
    y, p = _clean(y, p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y, p):
    y, p = _clean(y, p)
    return float(np.mean((p - y) ** 2))


def roc_auc(y, p):
    """Tie-averaged rank AUC (Mann-Whitney). 0.5 on a degenerate split.

    A constant predictor has every score tied, so it scores exactly 0.5 - which
    is what makes this the metric that ignores the side bias.
    """
    y, p = _clean(y, p)
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks_sorted = np.empty(p.size, dtype=np.float64)
    i = 0
    while i < sp.size:
        j = i
        while j + 1 < sp.size and sp[j + 1] == sp[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(p.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def accuracy(y, p, thr=0.5):
    y, p = _clean(y, p)
    return float(np.mean((p > thr) == (y == 1)))


def calibration_bins(y, p, n_bins=15):
    """Equal-width reliability bins: [{lo, hi, n, conf, acc}, ...].

    Empty bins are omitted rather than reported as zero-count rows, so the ECE
    sum never divides by zero. Equal-width (not equal-mass) on purpose: the
    question is "does the 0.6-0.66 bucket win 63% of the time", which only
    equal-width bins can answer.
    """
    y, p = _clean(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        out.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                    "n": n, "conf": float(p[m].mean()),
                    "acc": float(y[m].mean())})
    return out


def ece(y, p, n_bins=15):
    """Expected calibration error, plus MCE and the bins it came from."""
    y, p = _clean(y, p)
    bins = calibration_bins(y, p, n_bins)
    n = float(y.size)
    gaps = [(b["n"] / n, abs(b["acc"] - b["conf"])) for b in bins]
    return {"ece": float(sum(w * g for w, g in gaps)),
            "mce": float(max((g for _, g in gaps), default=0.0)),
            "bins": bins}


def confident_metrics(y, p, lo=CONF_LO, hi=CONF_HI):
    """Precision/recall in the two decisive buckets, and the abstention rate.

    A "blue call" is p > hi, a "red call" is p < lo; anything between is an
    abstention. Precision is measured on the *predicted side* (so the red
    bucket's precision is the rate at which p < lo actually means red won), and
    recall is the share of that side's real wins the model was confident about.

    The pooled row is selective prediction: `precision` is the hit rate on the
    confident subset and `coverage` is how much of the test set it covers.
    """
    y, p = _clean(y, p)
    n = y.size
    n_blue = int((y == 1).sum())
    n_red = int((y == 0).sum())
    call_blue = p > hi
    call_red = p < lo
    nb = int(call_blue.sum())
    nr = int(call_red.sum())
    hit_blue = int((call_blue & (y == 1)).sum())
    hit_red = int((call_red & (y == 0)).sum())
    n_conf = nb + nr
    prec_blue = hit_blue / nb if nb else float("nan")
    prec_red = hit_red / nr if nr else float("nan")
    return {
        "lo": lo, "hi": hi,
        "n_blue_calls": nb, "n_red_calls": nr, "n_confident": n_conf,
        "coverage": n_conf / n,
        # blue bucket: does "blue > 0.65" mean blue wins, and how much of blue's
        # real wins does it cover?
        "precision_blue": prec_blue,
        "precision_blue_ci": wilson(hit_blue, nb) if nb else (0.0, 1.0),
        "recall_blue": hit_blue / n_blue if n_blue else float("nan"),
        # red bucket, mirrored
        "precision_red": prec_red,
        "precision_red_ci": wilson(hit_red, nr) if nr else (0.0, 1.0),
        "recall_red": hit_red / n_red if n_red else float("nan"),
        # pooled: hit rate on everything the model had an opinion about
        "precision_confident": (hit_blue + hit_red) / n_conf if n_conf
        else float("nan"),
    }


def skill_scores(y, p):
    """Brier skill and log-loss skill against the best constant on these rows.

    Both are `1 - model / constant`, so 0 = no better than always predicting
    the base rate and negative = worse than it. This is the only comparison
    that does not move when the window's base rate moves.
    """
    y, p = _clean(y, p)
    r = float(y.mean())
    r = min(max(r, EPS), 1.0 - EPS)
    const_brier = float(np.mean((r - y) ** 2))
    const_ll = float(-(r * math.log(r) + (1 - r) * math.log(1 - r)))
    b = brier(y, p)
    ll = log_loss(y, p)
    return {"brier_skill": 1.0 - b / const_brier if const_brier else float("nan"),
            "log_loss_skill": 1.0 - ll / const_ll if const_ll else float("nan"),
            "const_brier": const_brier, "const_log_loss": const_ll}


# ------------------------------------------------------------------ reports
def full_report(y, p, lo=CONF_LO, hi=CONF_HI, n_bins=15, with_bins=False):
    """Every metric for one prediction vector, as a JSON-able dict."""
    y, p = _clean(y, p)
    n = int(y.size)
    n_blue = int((y == 1).sum())
    acc = accuracy(y, p)
    rep = {
        "n": n,
        "n_blue_wins": n_blue,
        "base_rate": n_blue / n,
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "roc_auc": roc_auc(y, p),
        "accuracy": acc,
        "accuracy_ci": wilson(int(round(acc * n)), n),
        "mean_p": float(p.mean()),
    }
    cal = ece(y, p, n_bins)
    rep["ece"] = cal["ece"]
    rep["mce"] = cal["mce"]
    if with_bins:
        rep["bins"] = cal["bins"]
    rep.update(skill_scores(y, p))
    rep["confident"] = confident_metrics(y, p, lo, hi)
    return rep


def format_report(rep, name="model", indent=0):
    """One block of human-readable text for a `full_report` dict."""
    pad = " " * indent
    c = rep["confident"]
    lo, hi = rep["accuracy_ci"]
    lines = [
        "%s%s  n=%d  base=%.4f  mean_p=%.4f" % (pad, name, rep["n"],
                                                 rep["base_rate"],
                                                 rep["mean_p"]),
        "%s  log-loss %.4f (skill %+.4f)  Brier %.4f (skill %+.4f)  "
        "ECE %.4f  MCE %.4f"
        % (pad, rep["log_loss"], rep["log_loss_skill"], rep["brier"],
           rep["brier_skill"], rep["ece"], rep["mce"]),
        "%s  ROC-AUC %.4f   accuracy %.4f [%.4f, %.4f]"
        % (pad, rep["roc_auc"], rep["accuracy"], lo, hi),
        "%s  confident (p<%.2f or p>%.2f): %d of %d games (%.1f%% coverage)"
        % (pad, c["lo"], c["hi"], c["n_confident"], rep["n"],
           100.0 * c["coverage"]),
        "%s    blue calls n=%d  precision %.4f  recall %.4f"
        % (pad, c["n_blue_calls"], c["precision_blue"], c["recall_blue"]),
        "%s    red  calls n=%d  precision %.4f  recall %.4f"
        % (pad, c["n_red_calls"], c["precision_red"], c["recall_red"]),
    ]
    return "\n".join(lines)


# ------------------------------------------------------- paired comparisons
_DECOMPOSABLE = ("log_loss", "brier", "accuracy")


def _per_group_sums(y, p, groups):
    """Per-group (n, -loglik, sq_err, correct) so a metric is a sum over groups.

    Returns (keys, sums[n,4]). Only metrics that are a plain mean over rows can
    be bootstrapped this way; that is why `bootstrap_delta` rejects AUC.
    """
    keys = {}
    idx = np.empty(len(groups), dtype=np.int64)
    for i, g in enumerate(groups):
        idx[i] = keys.setdefault(g, len(keys))
    sums = np.zeros((len(keys), 4), dtype=np.float64)
    np.add.at(sums, idx, np.stack([
        np.ones_like(p),
        -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        (p - y) ** 2,
        ((p > 0.5) == (y == 1)).astype(np.float64),
    ], axis=1))
    return sums


def bootstrap_delta(y, p_ref, p_new, groups, metric="log_loss", n_boot=2000,
                    seed=0):
    """Paired group-bootstrap of `metric(new) - metric(ref)`.

    Whole groups (series) are resampled with replacement, so the interval
    carries the within-series correlation instead of pretending 10 games of one
    series are 10 independent draws. Negative delta = the new model is better
    for log-loss/Brier (lower is better); for accuracy the sign is the natural
    one, so read `metric` before reading the sign.
    """
    if metric not in _DECOMPOSABLE:
        raise ValueError("bootstrap_delta supports %s, not %r (AUC is not a "
                         "per-row sum)" % (", ".join(_DECOMPOSABLE), metric))
    y, p_ref = _clean(y, p_ref)
    _, p_new = _clean(y, p_new)
    if len(groups) != y.size:
        raise ValueError("groups must align with y")
    col = {"log_loss": 1, "brier": 2, "accuracy": 3}[metric]
    a = _per_group_sums(y, p_ref, groups)[:, [0, col]]
    b = _per_group_sums(y, p_new, groups)[:, [0, col]]
    n_groups = a.shape[0]
    if n_groups < 2:
        raise ValueError("need at least 2 groups to bootstrap")

    def value(s):
        tot = s[:, 0].sum()
        return s[:, 1].sum() / tot if tot else float("nan")

    obs = value(b) - value(a)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        pick = rng.integers(0, n_groups, n_groups)
        draws[k] = value(b[pick]) - value(a[pick])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"metric": metric, "delta": float(obs), "lo": float(lo),
            "hi": float(hi), "n_boot": n_boot, "n_groups": n_groups,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "frac_draws_lt_zero": float(np.mean(draws < 0))}
