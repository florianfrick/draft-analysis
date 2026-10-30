"""Ban decisions are a search problem, so score them as decisions, not predictions.

Why this file exists
--------------------
The policy head is scored on "was this ban the one that happened?" That is the
wrong objective for a ban. A ban is an adversarial choice: it is correct when it
removes something the opponent wanted *and* you did not, which can be both
correct and surprising. Measured on this corpus the policy is 0.98x a per-move
frequency table on ban phase 1 and 2.38x on ban phase 2 - the model has no
opinion beyond "the popular ban" when the board is empty, because with no picks
on the board the opponent's preferences are not in the input at all.

The decision, written out
-------------------------
For a banning team t at a state s, and each legal champion c:

    denial(c) = what the opponent loses if c is gone
    self(c)   = what I lose if c is gone
    ban c  iff  denial(c) > self(c)

which is depth-1 of  c* = argmin_c max_c' V(opp | s-c, opp picks c'), i.e. a
negamax over a value function V. No accuracy target appears anywhere.

A ban only has a value through lookahead
----------------------------------------
V(s) depends on the champions *already picked*, so removing an unpicked champion
from the pool does not change V(s) at all. A ban is only worth something because
of what would have been picked later. That is the formal reason bans are a
search problem, and it has a sharp consequence: with a purely additive V over a
global champion strength, every ban has *exactly* the same value
(denial(c) = self(c) = w_c for all c), so the search has no opinion and any ban
is optimal. `--model global` demonstrates this - the score spread is identically
zero in both ban phases, even though that model predicts wins *better* than the
role-conditioned one.

A ban decision therefore exists only to the extent that V is state-dependent,
and the cheapest state dependence is *role need*: which of my five roles are
still unfilled, and which of theirs are. That gives one further structural
result, measured here rather than assumed: in ban phase 1 both teams still need
all five roles from the same pool, so every ban removes the same value from each
side and the score is flat for the *role* model too. Ban phase 1 is
unidentifiable under any symmetric additive V - not under-served by the model,
unidentifiable - and only an asymmetry (interaction terms, or team-specific
champion preference) can create a decision there. Ban phase 2 has one, because
six picks have made the two teams' remaining roles differ.

Value function
--------------
    V(s) = sigmoid(b + sum_blue W[role, champ] - sum_red W[role, champ])

W is fitted on pro TRAIN only, patch-honest, and validated before any of the
numbers below are believed (see the AUC gate in the report).

What is deliberately NOT in V
-----------------------------
Pairwise synergy/counter terms. `soloq data analysis/experiments/
pair_reliability.py` already measured this: after removing the additive part,
the split-half correlation of pair residuals is 0.09 and the residual SD
(0.0228) is essentially the binomial noise SD (0.0220), leaving a real pair
effect of ~0.006 winrate points against 0.022 of sampling noise per cell. Adding
those features *lowers* AUC (pro 0.5507 -> 0.5346). A pairwise V is not
estimable at this data scale, so the interaction structure has to come from
somewhere else (role need here, class complementarity or a learned encoder
later).

Usage
-----
    python model/policy/ban_search.py
    python model/policy/ban_search.py --model global        # the control
    python model/policy/ban_search.py --limit-decisions 500 --no-bootstrap
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # .../model/policy
MODEL_DIR = HERE.parent                         # .../model
ROOT = MODEL_DIR.parent                         # .../repo
PRO_DIR = ROOT / "data" / "pro play"

sys.path.insert(0, str(HERE))


def _load_module_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load_module_by_path("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load_module_by_path("pro_datasets", PRO_DIR / "datasets.py")
EP = _load_module_by_path("eval_protocol", PRO_DIR / "eval_protocol.py")

N_SLOTS = 10
NUM_ROLES = 5
MOVE_BAN = 0
MOVE_PICK = 1
# Ban phases of the 20-move pro draft: 6 bans, 6 picks, 4 bans, 4 picks.
# PHASE_OF[move] is 0/1/2/3 = ban1/pick1/ban2/pick2, so the ban phases are
# 0 and 2 - not 0 and 1, which is the off-by-one that used to KeyError here.
PHASE_OF = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4
BAN_PHASES = {0: "ban1", 2: "ban2"}


# --------------------------------------------------------------- value model
def value_features(rows, indices, num_champs):
    """Sparse design matrix for the role-conditioned additive model.

    One column per (role, champion); +1 if that champion fills that role on
    blue, -1 if on red. The intercept absorbs the blue-side base rate.
    """
    import scipy.sparse as sp

    n = len(indices)
    ri, ci, vv = [], [], []
    y = np.zeros(n, dtype=np.float64)
    for k, i in enumerate(indices):
        row = rows[i]
        y[k] = float(row["blue_win"])
        for slot in range(N_SLOTS):
            team = slot // 5
            role = slot % 5
            champ = int(row["champ"][slot])
            ri.append(k)
            ci.append(role * num_champs + champ)
            vv.append(1.0 if team == 0 else -1.0)
    X = sp.csr_matrix((vv, (ri, ci)), shape=(n, NUM_ROLES * num_champs))
    return X, y


def global_features(rows, indices, num_champs):
    """Control: champion strength with no role. Collapses the ban score to 0."""
    import scipy.sparse as sp

    n = len(indices)
    ri, ci, vv = [], [], []
    y = np.zeros(n, dtype=np.float64)
    for k, i in enumerate(indices):
        row = rows[i]
        y[k] = float(row["blue_win"])
        for slot in range(N_SLOTS):
            team = slot // 5
            champ = int(row["champ"][slot])
            ri.append(k)
            ci.append(champ)
            vv.append(1.0 if team == 0 else -1.0)
    X = sp.csr_matrix((vv, (ri, ci)), shape=(n, num_champs))
    return X, y


def fit_logistic(X, y, C):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, fit_intercept=True, max_iter=3000,
                           solver="lbfgs")
    m.fit(X, y)
    return m


def as_role_weights(model, num_champs, kind):
    """Reshape a fitted model into W[role, champ] (+ intercept)."""
    coef = model.coef_.ravel()
    if kind == "global":
        # broadcast the global strength across roles: this is the point of the
        # control - every role sees the same ordering, so a ban removes the same
        # amount from both teams and the score difference is identically zero
        W = np.tile(coef[None, :], (NUM_ROLES, 1))
    else:
        W = coef.reshape(NUM_ROLES, num_champs)
    return W, float(model.intercept_[0])


# ---------------------------------------------------------------- draft state
@dataclass
class DraftState:
    """What is already decided, at the moment a ban has to be chosen."""
    taken: np.ndarray            # bool [C] banned / picked / fearless-unavailable
    filled: tuple                # (frozenset blue roles, frozenset red roles)
    blue_sum: float              # sum of W over blue's picks so far
    red_sum: float               # sum of W over red's picks so far

    def copy(self):
        return DraftState(self.taken.copy(), self.filled, self.blue_sum,
                          self.red_sum)


def state_at(row, move_idx, W):
    """Replay row["moves"][:move_idx] into a DraftState."""
    n = W.shape[1]
    taken = np.zeros(n, dtype=bool)
    filled = [set(), set()]
    blue = red = 0.0
    for kind, team, champ, role in row["moves"][:move_idx]:
        taken[int(champ)] = True
        if kind == MOVE_PICK:
            filled[team].add(int(role))
            if team == 0:
                blue += W[int(role), int(champ)]
            else:
                red += W[int(role), int(champ)]
    for c in (row.get("unavailable") or []):
        taken[int(c)] = True
    return DraftState(taken, (frozenset(filled[0]), frozenset(filled[1])),
                      blue, red)


def top_two_per_role(W, taken, filled):
    """(best value, best champ, second value) for each role still to be filled.

    Roles already filled get (0, -1, 0) so they contribute nothing.
    """
    n = W.shape[1]
    m0 = np.zeros(NUM_ROLES)
    a0 = np.full(NUM_ROLES, -1, dtype=int)
    m1 = np.zeros(NUM_ROLES)
    for r in range(NUM_ROLES):
        if r in filled:
            continue
        row = np.where(taken, -np.inf, W[r])
        order = np.argpartition(-row, 1)[:2]
        # argpartition does not order the two, sort them
        order = order[np.argsort(-row[order])]
        a0[r] = int(order[0])
        m0[r] = float(row[order[0]])
        m1[r] = float(row[order[1]]) if len(order) > 1 else float(row[order[0]])
    return m0, a0, m1


def completion_values(W, state):
    """Value of the rest of the draft for each side, per candidate ban.

    Returns (base_blue, base_red, loss_blue, loss_red) where base_* is the
    completion value with nothing banned and loss_*[c] is how much that drops if
    champion c is banned.

    Exact under a per-role greedy completion: banning c can only change a role's
    best available champion if c *was* that champion, in which case the role
    falls back to its second best. Flex conflicts (one champion being the best
    in two roles at once) are the one approximation; `double_claim_rate` reports
    how often that happens.
    """
    m0_b, a0_b, m1_b = top_two_per_role(W, state.taken, state.filled[0])
    m0_r, a0_r, m1_r = top_two_per_role(W, state.taken, state.filled[1])
    n = W.shape[1]
    loss_b = np.zeros(n)
    loss_r = np.zeros(n)
    for r in range(NUM_ROLES):
        if a0_b[r] >= 0:
            loss_b[a0_b[r]] += m0_b[r] - m1_b[r]
        if a0_r[r] >= 0:
            loss_r[a0_r[r]] += m0_r[r] - m1_r[r]
    return m0_b.sum(), m0_r.sum(), loss_b, loss_r


def double_claim_rate(W, state):
    """Share of remaining roles whose best champion is shared with another role.

    The per-role greedy counts such a champion twice, which slightly overstates
    both teams' completion value. Reported so the approximation is visible.
    """
    _, a0_b, _ = top_two_per_role(W, state.taken, state.filled[0])
    _, a0_r, _ = top_two_per_role(W, state.taken, state.filled[1])
    hits = 0
    total = 0
    for a0 in (a0_b, a0_r):
        vals = [int(x) for x in a0 if x >= 0]
        total += len(vals)
        hits += len(vals) - len(set(vals))
    return (hits / total) if total else 0.0


def ban_probabilities(W, b, state, legal, side):
    """P(`side` wins) for every candidate ban in `legal`.

    V is evaluated on the completed draft, so this is the depth-1 view of the
    negamax: ban c, let both teams fill their remaining roles greedily, read the
    value. `side` is the banning team (0 blue, 1 red); a ban is good for that
    side when it lowers the *opponent's* completion more than its own.
    """
    base_b, base_r, loss_b, loss_r = completion_values(W, state)
    lp = (b + state.blue_sum + base_b - loss_b[legal]
          - (state.red_sum + base_r - loss_r[legal]))
    p_blue = 1.0 / (1.0 + np.exp(-np.clip(lp, -30.0, 30.0)))
    return p_blue if side == 0 else 1.0 - p_blue


# ------------------------------------------------------------------ analysis
def auc_of(W, b, rows, indices, num_champs):
    """Win AUC of the fitted V on a set of drafts (sanity gate)."""
    scores, labels = [], []
    for i in indices:
        row = rows[i]
        blue = red = 0.0
        for slot in range(N_SLOTS):
            team = slot // 5
            role = slot % 5
            v = W[role, int(row["champ"][slot])]
            if team == 0:
                blue += v
            else:
                red += v
        scores.append(b + blue - red)
        labels.append(int(row["blue_win"]))
    return EP.auc(scores, labels)


def analyse(rows, splits, W, b, move_rank, num_champs, limit=0, seed=0,
            keep_details=False):
    """Regret of the observed ban under the search score, by phase."""
    rng = np.random.default_rng(seed)
    decisions = []
    for i in splits["test"]:
        row = rows[i]
        for m, (kind, team, champ, _role) in enumerate(row["moves"]):
            if kind != MOVE_BAN:
                continue
            decisions.append((i, m, int(team), int(champ)))
    if limit and len(decisions) > limit:
        sel = rng.choice(len(decisions), size=limit, replace=False)
        decisions = [decisions[k] for k in sorted(sel)]

    out = {}
    detail = []
    for i, m, team, obs in decisions:
        row = rows[i]
        state = state_at(row, m, W)
        legal = np.where(~state.taken)[0]
        if len(legal) < 2 or obs not in legal:
            continue
        p = ban_probabilities(W, b, state, legal, team)
        # frequency-table ban: most common champion at this move in TRAIN
        ranks = np.array([move_rank[m][int(c)] for c in legal])
        freq = int(legal[int(np.argmin(ranks))])
        best_idx = int(np.argmax(p))
        obs_idx = int(np.searchsorted(legal, obs))
        freq_idx = int(np.searchsorted(legal, freq))
        phase = BAN_PHASES[PHASE_OF[m]]
        rec = {
            "patch": row["patch"], "phase": phase, "move": m,
            "n_legal": int(len(legal)),
            "p_best": float(p[best_idx]),
            "p_obs": float(p[obs_idx]),
            "p_freq": float(p[freq_idx]),
            "p_random": float(p.mean()),
            "regret_obs": float(p[best_idx] - p[obs_idx]),
            "regret_freq": float(p[best_idx] - p[freq_idx]),
            "regret_random": float(p[best_idx] - p.mean()),
            "obs_is_best": bool(obs_idx == best_idx),
            "freq_is_best": bool(freq_idx == best_idx),
            "obs_eq_freq": bool(obs == freq),
            "rank_obs": int((p > p[obs_idx]).sum()),
            "rank_freq": int((p > p[freq_idx]).sum()),
            "spread": float(p.max() - p.min()),
        }
        out.setdefault(phase, []).append(rec)
        if keep_details:
            detail.append({"gameid": row["gameid"], "move": m, "phase": phase,
                           "obs": obs, "best": int(legal[best_idx]),
                           "freq": freq, **rec})
    return out, detail


def summarise(recs):
    if not recs:
        return {"n": 0}
    keys = ("regret_obs", "regret_freq", "regret_random", "p_best", "p_obs",
            "p_freq", "p_random", "spread", "rank_obs", "rank_freq")
    out = {"n": len(recs)}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in recs]))
    for k in ("obs_is_best", "freq_is_best", "obs_eq_freq"):
        out[k] = float(np.mean([float(r[k]) for r in recs]))
    out["regret_obs_p90"] = float(np.percentile(
        [r["regret_obs"] for r in recs], 90))
    # Paired edge: how much better than an average legal ban the observed /
    # frequency ban is, *under this V*, with a z-score. Comparing the two mean
    # regrets directly is wrong - they are measured on the same decisions, so
    # the difference is per-decision and its own SE applies. Under "humans ban
    # at random" the expected edge is exactly 0.
    for tag, key in (("obs", "p_obs"), ("freq", "p_freq")):
        d = np.array([r[key] - r["p_random"] for r in recs])
        out["edge_" + tag] = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        out["edge_%s_se" % tag] = se
        # epsilon rather than a 0 guard: a constant difference has zero SE and
        # therefore infinite z, which must not silently read as "no effect"
        out["edge_%s_z" % tag] = float(d.mean() / max(se, 1e-12))
    # A phase is "flat" when the score is identical for every legal ban, so
    # there is no decision to make and every regret is 0 by construction. That
    # is a structural statement about the value function, not a weak result, and
    # it must not be averaged into a headline number.
    out["flat"] = bool(out["spread"] < 1e-9)
    return out


def bootstrap_stability(rows, splits, num_champs, W, b, move_rank, kind,
                        n_boot, n_decisions, seed=0):
    """Does the recommended ban survive a refit on resampled training data?

    A ranking that flips every time the training set is perturbed cannot be
    acted on, however clean its regret numbers look. This is the empirical
    version of "is the decision signal bigger than the model's error".
    """
    rng = np.random.default_rng(seed)
    train = np.asarray(splits["train"])
    # a fixed, representative sample of decisions to re-score each replicate
    pool = []
    for i in splits["test"]:
        for m, (k, _t, _c, _r) in enumerate(rows[i]["moves"]):
            if k == MOVE_BAN:
                pool.append((i, m))
    if len(pool) > n_decisions:
        sel = rng.choice(len(pool), size=n_decisions, replace=False)
        pool = [pool[k] for k in sorted(sel)]

    def top_bans(W_, b_):
        picks = []
        for i, m in pool:
            row = rows[i]
            team = int(row["moves"][m][1])
            state = state_at(row, m, W_)
            legal = np.where(~state.taken)[0]
            p = ban_probabilities(W_, b_, state, legal, team)
            picks.append(int(legal[int(np.argmax(p))]))
        return picks

    base = top_bans(W, b)
    agree = np.zeros(len(pool))
    for _ in range(n_boot):
        idx = rng.choice(train, size=len(train), replace=True)
        X, y = (global_features(rows, idx, num_champs) if kind == "global"
                else value_features(rows, idx, num_champs))
        m_ = fit_logistic(X, y, C=0.03)
        W_, b_ = as_role_weights(m_, num_champs, kind)
        alt = top_bans(W_, b_)
        agree += np.array([float(a == c) for a, c in zip(base, alt)])
    return {"n_boot": n_boot, "n_decisions": len(pool),
            "top1_agreement": float(agree.mean() / max(1, n_boot))}


# ----------------------------------------------------------------------- cli
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Ban decision search + regret")
    ap.add_argument("--model", choices=("role", "global"), default="role",
                    help="role = role-conditioned additive (default); "
                         "global = the control that proves the score is zero "
                         "without state dependence")
    ap.add_argument("--limit-decisions", type=int, default=0)
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--n-boot", type=int, default=6)
    ap.add_argument("--boot-decisions", type=int, default=300)
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    rows, _ = PRO_PIPE.build_rows(limit=0)
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_series_by_date(rows)
    splits["val"], info = EP.patch_honest_val(rows, splits)
    num_champs = PRO_DATA.NUM_REAL_CHAMPS
    print("rows %d (train %d / val %d / test %d), patch-honest val %s"
          % (len(rows), len(splits["train"]), len(splits["val"]),
             len(splits["test"]), info))

    # --- fit on train only, choose C on patch-honest val --------------------
    feat = global_features if args.model == "global" else value_features
    Xtr, ytr = feat(rows, splits["train"], num_champs)
    Xva, yva = feat(rows, splits["val"], num_champs)
    best = None
    for C in (0.01, 0.03, 0.1):
        m = fit_logistic(Xtr, ytr, C)
        p = m.predict_proba(Xva)[:, 1]
        a = EP.auc(list(p), list(yva))
        print("  C=%-5s val AUC %.4f" % (C, a))
        if best is None or a > best[0]:
            best = (a, C, m)
    val_auc, C, model = best
    W, b = as_role_weights(model, num_champs, args.model)
    print("chosen C=%s (val AUC %.4f)" % (C, val_auc))

    # --- the gate: V has to beat chance before any ban number is believable -
    test_auc = auc_of(W, b, rows, splits["test"], num_champs)
    print("value function: val AUC %.4f | test AUC %.4f  (gate: >0.5)"
          % (val_auc, test_auc))

    # --- ban regret --------------------------------------------------------
    move_rank = EP.move_popularity_ranks(rows, splits["train"], num_champs)
    res, detail = analyse(rows, splits, W, b, move_rank, num_champs,
                          limit=args.limit_decisions, seed=args.seed)
    print("\nban decisions on test (%s model)" % args.model)
    print("  %-6s %7s %9s %9s %9s %8s %8s %8s %8s"
          % ("phase", "n", "regret_h", "regret_f", "regret_rnd", "h=best",
             "f=best", "h=freq", "spread"))
    summary = {}
    for phase in ("ban1", "ban2"):
        s = summarise(res.get(phase, []))
        summary[phase] = s
        if not s["n"]:
            continue
        if s["flat"]:
            print("  %-6s %7d %9s %9s %9s %8s %8s %8s %8.5f"
                  % (phase, s["n"], "-", "-", "-", "-", "-", "-",
                     s["spread"]))
            continue
        print("  %-6s %7d %9.5f %9.5f %9.5f %7.1f%% %7.1f%% %7.1f%% %8.5f"
              % (phase, s["n"], s["regret_obs"], s["regret_freq"],
                 s["regret_random"], 100 * s["obs_is_best"],
                 100 * s["freq_is_best"], 100 * s["obs_eq_freq"],
                 s["spread"]))
        print("         edge over an average legal ban: human %+.5f "
              "(z %+.1f)   freq %+.5f (z %+.1f)"
              % (s["edge_obs"], s["edge_obs_z"], s["edge_freq"],
                 s["edge_freq_z"]))

    # Headline: pooled over the phases where a decision actually exists. A flat
    # phase would add zeros and make the search look harmless, which is exactly
    # the wrong conclusion.
    live = [r for ph, v in res.items() if not summarise(v)["flat"] for r in v]
    pooled = summarise(live)
    summary["all"] = pooled
    summary["flat_phases"] = [ph for ph in ("ban1", "ban2")
                              if res.get(ph) and summarise(res[ph])["flat"]]
    if pooled.get("n"):
        print("  %-6s %7d %9.5f %9.5f %9.5f %7.1f%% %7.1f%% %7.1f%% %8.5f"
              % ("all*", pooled["n"], pooled["regret_obs"],
                 pooled["regret_freq"], pooled["regret_random"],
                 100 * pooled["obs_is_best"], 100 * pooled["freq_is_best"],
                 100 * pooled["obs_eq_freq"], pooled["spread"]))
        print("  * only phases with a non-degenerate score (%s); flat: %s"
              % ("+".join(sorted({ph for ph in ("ban1", "ban2")
                                  if res.get(ph)
                                  and not summarise(res[ph])["flat"]})),
                 ",".join(summary["flat_phases"]) or "none"))

    # --- patch 16.20 focus -------------------------------------------------
    focus = [r for r in live if r["patch"] == EP.FOCUS_PATCH]
    summary["focus_patch"] = summarise(focus)
    if focus:
        s = summary["focus_patch"]
        if s["flat"]:
            print("\n  patch %s: n=%d  score flat (no decision to make)"
                  % (EP.FOCUS_PATCH, s["n"]))
        else:
            print("\n  patch %s: n=%d  regret_human %.5f  regret_freq %.5f  "
                  "human-is-best %.1f%%  spread %.5f"
                  % (EP.FOCUS_PATCH, s["n"], s["regret_obs"],
                     s["regret_freq"], 100 * s["obs_is_best"], s["spread"]))

    # --- is the ranking real? ---------------------------------------------
    stability = None
    if not args.no_bootstrap:
        print("\nbootstrap rank stability (%d refits on resampled train) ..."
              % args.n_boot)
        stability = bootstrap_stability(
            rows, splits, num_champs, W, b, move_rank, args.model,
            args.n_boot, args.boot_decisions, seed=args.seed)
        print("  same recommended ban after refit: %.1f%% of %d decisions"
              % (100 * stability["top1_agreement"],
                 stability["n_decisions"]))
    summary["stability"] = stability

    verdict = _verdict(pooled, test_auc, stability,
                       summary.get("flat_phases"), args.model)
    print("\n" + verdict)
    summary["verdict"] = verdict
    summary["value"] = {"val_auc": val_auc, "test_auc": test_auc, "C": C,
                        "model": args.model}

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary,
             "details": detail[:2000] if detail else []}, indent=1),
            encoding="utf-8")
        print("\nwrote %s (%.0fs)" % (args.json, time.time() - t0))
    return summary


def _verdict(pooled, test_auc, stability, flat_phases=None, model="role"):
    """Say plainly whether these numbers can be acted on."""
    flat_phases = flat_phases or []
    lines = []
    if flat_phases:
        if model == "global":
            why = ("a global additive V orders champions identically for every "
                   "role, so banning c removes exactly the same value from "
                   "both teams in every unfilled role and the score is "
                   "identically zero")
        else:
            why = ("both teams still needed the same roles from the same pool, "
                   "so every legal ban removed exactly the same value from "
                   "each side")
        lines.append(
            "Structurally flat: %s. Because %s. No accuracy figure, and no "
            "search, can rank those decisions under a symmetric additive V - "
            "it needs an asymmetry (interaction terms or team-specific "
            "champion preference), not a better model."
            % (", ".join(flat_phases), why))
    if not pooled.get("n"):
        lines.append("No phase with a non-degenerate score was analysed.")
        return "\n".join(lines)
    if test_auc < 0.52:
        lines.append(
            "V is too weak (test AUC %.3f) for its ban ranking to mean "
            "anything: the score spread across candidate bans (%.5f winrate "
            "points) is far smaller than the model's own error. Fix the value "
            "function before trusting any regret number."
            % (test_auc, pooled["spread"]))
    elif abs(pooled["edge_obs_z"]) < 2.0:
        lines.append(
            "Observed bans are indistinguishable from an average legal ban "
            "under this V (edge %+.5f, z %+.1f, n=%d): humans score at the "
            "middle of the legal set, so V is not representing what they "
            "optimise. The search is measuring V, not the game."
            % (pooled["edge_obs"], pooled["edge_obs_z"], pooled["n"]))
    else:
        lines.append(
            "Observed bans beat an average legal ban by %+.5f winrate points "
            "(z %+.1f), so V carries some of what players optimise - but the "
            "gap between the best ban and the observed one is only %.5f."
            % (pooled["edge_obs"], pooled["edge_obs_z"],
               pooled["regret_obs"]))
    if stability is not None:
        a = stability["top1_agreement"]
        if a >= 0.9:
            lines.append(
                "The recommended ban survives refitting (%.0f%% of decisions "
                "keep it), so the ranking is reproducible." % (100 * a))
        elif a >= 0.75:
            lines.append(
                "The recommended ban survives refitting only %.0f%% of the "
                "time: reproducible in aggregate, not per decision."
                % (100 * a))
        else:
            lines.append(
                "And %.0f%% of recommendations change when the training set is "
                "resampled, so this ranking is not reproducible enough to act "
                "on - the score differences are smaller than the estimation "
                "noise in W." % (100 * (1 - a)))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
