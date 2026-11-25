"""A parameterised pairwise-matchup tool over the soloQ analytical store.

The agent already has `sql` for the *pro* corpus (SQLite), but the soloQ
matchup data lives in a different store - a DuckDB cube (`cube2`) built by
`solo_queue/analysis/store.py`. The agent cannot reach it through `sql`, and it
should not have to write DuckDB SQL (its escape hatch is a different schema). So
this tool wraps the precomputed cube with parameters instead of free text, and
attaches the honesty the pairwise numbers need:

  * a **baseline** - the champion's own win rate with no partner filter, over the
    same patch scope - so the *marginal* effect of the partner is visible rather
    than buried in the champion's own strength;
  * a **reliable** flag on `n >= min_n`, because `pair_reliability` showed the
    pairwise deviations are mostly champion strength plus noise;
  * the exact query is returned, like the `sql` tool, so a wrong answer is
    debuggable;
  * the numbers are **soloQ**, a different population from the pro drafts the
    head trained on, so they are a complementary signal, not a substitute for
    `predict_next_move`.

The connection is opened lazily and cached: importing this module is cheap
(duckdb is only imported inside `_connect`), and a missing/locked store is a
returned error, never a raised one - a loop must not die on a tool call.
"""
from __future__ import annotations

import os
import json
import re
import threading
from pathlib import Path

# The store is large and DuckDB holds a file lock on Windows even read-only, so
# the connection is a process-wide singleton opened on first use.
_CON = None
_LOCK = threading.Lock()

_STATS = ["win", "gold", "dmg", "cs", "kills", "deaths", "assists", "vision"]
# The cube already carries per-cell averages; across cells, weight by n.
_WEIGHTED = ", ".join("SUM(n*%s)/SUM(n) AS %s" % (s, s) for s in _STATS) \
    if False else ", ".join("SUM(n*%s)/NULLIF(SUM(n),0) AS %s" % (s, s)
                            for s in _STATS)

_ROLE_ALIASES = {
    "top": "TOP", "jng": "TOP", "jg": "TOP", "jungle": "TOP", "jungler": "TOP",
    "mid": "MIDDLE", "middle": "MIDDLE",
    "bot": "BOTTOM", "bottom": "BOTTOM", "adc": "BOTTOM",
    "support": "SUPPORT", "sup": "SUPPORT", "supp": "SUPPORT",
    "utility": "SUPPORT", "util": "SUPPORT",
}
_VALID_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT")

# `solo_queue/analysis/store.py` resolves its data path to `solo_queue/data/...`,
# but the matchdata actually lives at the repo-root `data/` directory, so its
# `champ_names()` / `champ_idx_map()` return empty at runtime. The store still
# *queries* correctly (it holds prebuilt data), so we load the champion map
# directly from the correct path instead of through that module.
_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parents[2]                       # .../pro/model/agent -> repo root
_NAMES_PATH = _PROJ / "data" / "solo queue" / "process matchdata" \
    / "champ_names.json"

_NAMES_CACHE = None


def _names():
    """riot id (int) -> display name, from the patch-pinned names file."""
    global _NAMES_CACHE
    if _NAMES_CACHE is None:
        if _NAMES_PATH.exists():
            data = json.loads(_NAMES_PATH.read_text(encoding="utf-8"))
            _NAMES_CACHE = {int(k): v for k, v in data.get("names", data).items()}
        else:
            _NAMES_CACHE = {}
    return _NAMES_CACHE


def _norm(text):
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower()).strip()


def available():
    """Whether the soloQ store exists and can be offered as a tool.

    Does not open the file - `connect` would lock it, and the explorer may be
    holding it. Presence is enough to advertise the spec. A deployment that
    ships neither the analysis package nor duckdb simply has no store: that is
    an absent tool, not a broken session.
    """
    try:
        from solo_queue.analysis.store import DEFAULT_OUT
    except ImportError:
        return False
    return os.path.exists(os.path.join(DEFAULT_OUT, "store.duckdb"))


def _connect():
    global _CON
    if _CON is None:
        with _LOCK:
            if _CON is None:
                from solo_queue.analysis.store import DEFAULT_OUT, connect
                _CON = connect(DEFAULT_OUT, read_only=True)
    return _CON


def _resolve(token, champs=None):
    """Name or Riot id -> Riot id (the space `cube2` uses).

    Riot ids pass straight through. Names are resolved against the soloQ name map
    loaded from the repo-root data file (see `_names`): the pro corpus and the
    soloQ store share *names* but not dense index ordering, so the name is the
    only safe bridge. Fuzzy matching ('mundo' -> 'Dr. Mundo', 'kaisa' ->
    'Kai'Sa') uses the same ends-with rule as the agent's `ChampIndex`.
    """
    # Riot id passed straight through.
    try:
        riot = int(token)
    except (TypeError, ValueError):
        riot = None
    if riot is not None:
        if riot in _names():
            return riot
        raise ValueError("riot id %r is not in the soloQ store" % (token,))
    # Name -> riot id via the soloQ name map.
    norm_map = {}
    for riot, name in _names().items():
        nk = _norm(name)
        norm_map.setdefault(nk, riot)
        norm_map.setdefault(nk.replace(" ", ""), riot)
    nk = _norm(token)
    if not nk:
        raise ValueError("unknown champion %r" % (token,))
    if nk in norm_map:
        return norm_map[nk]
    # Fuzzy: a single champion whose normalized name ends with the query, the
    # same rule ChampIndex uses so 'mundo' finds 'Dr. Mundo'.
    hits = [r for n, r in norm_map.items()
            if n.endswith(" " + nk) or n.replace(" ", "").endswith(nk)]
    if len(set(hits)) == 1:
        return hits[0]
    raise ValueError("unknown champion %r - use a name or a Riot id"
                     % (token,))


def _dense_names():
    """riot id -> soloQ display name, for echoing the resolved pair."""
    return _names()


def run(champion, partner, kind="vs", patch=None, role=None, min_n=30,
        champs=None):
    """-> {champion, partner, kind, patch, observed, baseline, delta_win,
    reliable, role, query, note} or {"error": ...}.

    `kind` is 'vs' (against) or 'with' (alongside). `patch`, when given, is a
    patch label like '16.20'; omitted pools every patch and weights by n. `role`,
    when given, restricts the matchup to that lane (a wide-table scan, which is
    slower) and is echoed in the result. Never raises - every failure is a dict
    the loop can read and recover from.
    """
    try:
        k = str(kind or "vs").lower()
        if k not in ("vs", "with"):
            return {"error": "kind must be 'vs' or 'with', got %r" % (kind,)}
        if role is not None:
            rk = _ROLE_ALIASES.get(str(role).lower())
            if rk not in _VALID_ROLES:
                return {"error": "role must be one of %s, got %r"
                        % (", ".join(_VALID_ROLES), role)}
            role = rk

        try:
            a = _resolve(champion, champs)
            b = _resolve(partner, champs)
        except ValueError as exc:
            return {"error": str(exc)}
        names = _dense_names()
        a_name = names.get(a, str(a))
        b_name = names.get(b, str(b))
        min_n = int(min_n or 30)

        con = _connect()
        try:
            base_filter = "champ=? AND partner=? AND kind=?"
            params = [a, b, k]
            if patch:
                base_filter += " AND patch=?"
                params.append(str(patch))

            cur = con.execute(
                "SELECT SUM(n) AS n, %s FROM cube2 WHERE %s"
                % (_WEIGHTED, base_filter), params)
            cols = [d[0] for d in cur.description]
            row = cur.fetchone()
            total = (row[0] if row else None) or 0
        except Exception as exc:  # noqa: BLE001 - lock / SQL errors read as refusals
            return {"error": "soloQ query failed: %s" % exc}

        observed = None
        if total >= 1:
            d = dict(zip(cols, row))
            observed = {"n": int(d["n"])}
            for s in _STATS:
                v = d.get(s)
                observed[s] = round(float(v), 4) if v is not None else None

        # Baseline: the champion's own win rate (and gold/dmg) over the same
        # patch scope, with no partner filter. The delta is the marginal effect.
        bfilter = "champ=?"
        bparams = [a]
        if patch:
            bfilter += " AND patch=?"
            bparams.append(str(patch))
        br = con.execute(
            "SELECT SUM(n) AS n, SUM(n*win)/NULLIF(SUM(n),0) AS win, "
            "SUM(n*gold)/NULLIF(SUM(n),0) AS gold, "
            "SUM(n*dmg)/NULLIF(SUM(n),0) AS dmg FROM champ_base WHERE %s"
            % bfilter, bparams).fetchone()
        baseline = None
        if br and br[0]:
            baseline = {"n": int(br[0]),
                        "win": round(float(br[1]), 4) if br[1] is not None else None,
                        "gold": round(float(br[2]), 4) if br[2] is not None else None,
                        "dmg": round(float(br[3]), 4) if br[3] is not None else None}

        delta_win = None
        if observed and observed.get("win") is not None and baseline \
                and baseline.get("win") is not None:
            delta_win = round(observed["win"] - baseline["win"], 4)

        # Optional lane-scoped scan over the wide table.
        role_scan = None
        if role is not None:
            try:
                rr = con.execute("""
                    SELECT count(*) n, avg(win) win, avg(gold) gold, avg(dmg) dmg
                    FROM wide
                    WHERE champ=? AND pos=?
                      AND (CASE WHEN ? THEN list_contains(enemy_champs, ?)
                               ELSE list_contains(ally_champs, ?) END)
                """, [a, role, k == "vs", b, b]).fetchone()
                if rr and rr[0]:
                    role_scan = {"role": role, "n": int(rr[0]),
                                 "win": round(float(rr[1]), 4) if rr[1] is not None else None,
                                 "gold": round(float(rr[2]), 4) if rr[2] is not None else None,
                                 "dmg": round(float(rr[3]), 4) if rr[3] is not None else None}
            except Exception as exc:  # noqa: BLE001
                role_scan = {"role": role, "error": "lane scan failed: %s" % exc}

        # The query string, for debuggability (a wrong answer must be auditable).
        query = "cube2: champ=%s partner=%s kind=%s patch=%s" % (a, b, k, patch)
        if role is not None:
            query += " (lane scan pos=%s)" % role

        note = (
            "These are SOLO QUEUE rates, a different population from the pro "
            "drafts the next-move head learned on - a complementary signal, "
            "not a substitute for predict_next_move. The cube is a precomputed "
            "2-way aggregate, so it says nothing about a full 5v5 lineup. "
        )
        if observed is None:
            note += ("No %s cell exists for this pair%s - the champion and "
                     "partner never met in the data, so there is no matchup to "
                     "report."
                     % (k, " on patch %s" % patch if patch else ""))
        else:
            note += ("reliable = (n >= %d); below that the average is mostly "
                     "noise. delta_win is this matchup's win rate minus the "
                     "champion's own win rate over the same patch scope, so a "
                     "small delta means the partner barely moves the needle - "
                     "most pairwise deviations do."
                     % min_n)
        return {"champion": a_name, "partner": b_name, "kind": k,
                "patch": patch, "role": role, "observed": observed,
                "baseline": baseline, "delta_win": delta_win,
                "reliable": bool(observed and observed.get("n", 0) >= min_n),
                "lane_scan": role_scan, "query": query, "note": note}
    except Exception as exc:  # noqa: BLE001 - a tool must never raise into the loop
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


TOOL_SPEC = {
    "name": "soloq_matchup",
    "description": (
        "Pairwise win-rate and stat matchup for two champions, from the solo "
        "queue analytical store (a different population from the pro drafts the "
        "head trained on). Use it when a recommendation turns on a specific "
        "matchup - 'does Ahri beat Zed in mid?', 'does Alistar pair well with "
        "this ADC?' - that the next-move model cannot answer because it scores "
        "champions independently. It does NOT tell you what to pick by itself.\n\n"
        "Parameters:\n"
        "  champion  (required) a champion name ('Ahri') or Riot id (103)\n"
        "  partner   (required) the other champion, name or Riot id\n"
        "  kind      'vs' (against; default) or 'with' (alongside / in lane)\n"
        "  patch     optional patch label like '16.20'; omit to pool all patches\n"
        "  role      optional lane (TOP/JUNGLE/MIDDLE/BOTTOM/SUPPORT) to scope "
        "the matchup to that lane\n"
        "  min_n     reliability threshold (default 30); the result flags "
        "reliable=false below it\n\n"
        "The result carries the observed cell, the champion's own baseline win "
        "rate over the same patch scope, delta_win (the marginal effect), and a "
        "reliable flag. Treat n under min_n as an anecdote: most pairwise "
        "deviations are the champion's own strength plus noise, not a real "
        "matchup edge. A missing cell (n=0) means the pair never met in the "
        "data."),
    "parameters": {
        "type": "object",
        "properties": {
            "champion": {"type": "string",
                         "description": "a champion name or Riot id"},
            "partner": {"type": "string",
                        "description": "the other champion, name or Riot id"},
            "kind": {"type": "string", "enum": ["vs", "with"], "default": "vs",
                     "description": "against or alongside"},
            "patch": {"type": "string", "description": "patch label e.g. '16.20'; "
                      "omit to pool all patches (weighted by sample size)"},
            "role": {"type": "string", "enum": ["TOP", "JUNGLE", "MIDDLE",
                                                "BOTTOM", "SUPPORT"],
                     "description": "restrict the matchup to one lane (slower "
                                    "wide-table scan)"},
            "min_n": {"type": "integer", "default": 30, "minimum": 1,
                      "description": "reliability threshold; reliable=false below"},
        },
        "required": ["champion", "partner"],
    },
}
