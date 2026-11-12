"""The escape hatch: one read-only SQL tool with mechanical guards.

The failure mode of this tier is different from the model tier's. A wrong
model answer is legible - it arrives with a rank attached. A wrong SQL answer
is silent: a valid query, a small n, a confident sentence. So the guards here
are mechanical rather than advisory:

    read-only connection        writes are impossible, not discouraged
    single statement            no stacked queries, no PRAGMA
    SELECT / WITH only          checked before execution, not after
    row cap 200 + `truncated`   a bare SELECT * is a context-window event
    2 s deadline                via a progress handler, aborts mid-query
    the query is returned       every result carries the SQL that produced it

The last one is not a nicety: without it a wrong answer cannot be debugged, and
the unsupported-claim metric - "does the reply assert a number no tool
returned" - cannot be computed.

`hint` is the part that pays for itself in an agent loop. A flash model writes
`WHERE kind = 1` or `WHERE champ = 'Kai'Sa'` on the first try often enough that
the error message has to teach, not just fail.
"""
from __future__ import annotations

import re
import sqlite3
import time

MAX_ROWS = 200
TIMEOUT_S = 2.0

_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_LEAD = re.compile(r"^\s*(select|with)\b", re.I)


class SqlRefused(Exception):
    """A guard fired. The message is written for the model, not for a log."""


def _clean(query):
    q = _COMMENT.sub(" ", query or "").strip()
    if not q:
        raise SqlRefused("empty query")
    # One statement only. A trailing semicolon is fine; anything after it is not.
    body = q[:-1] if q.endswith(";") else q
    if ";" in body:
        raise SqlRefused("one statement per call - stacked queries are refused")
    if not _LEAD.match(body):
        raise SqlRefused("only SELECT and WITH are allowed; this tool cannot "
                         "write, and PRAGMA/ATTACH are refused")
    return body


def hint(query, n_rows):
    """A targeted nudge for the mistakes this schema actually invites."""
    q = (query or "").lower()
    out = []
    if re.search(r"\bkind\s*=\s*['\"]?[01]['\"]?", q):
        out.append("kind is text: 'ban' or 'pick' (0/1 will match nothing)")
    if re.search(r"\bside\s*=\s*['\"]?[01]['\"]?", q):
        out.append("side is text: 'blue' or 'red'")
    if re.search(r"\bchamp\s*=\s*'", q) and "champ_key" not in q:
        out.append("join champ on champ.key (normalised, no apostrophes) "
                   "instead of matching champ_idx to a name")
    if re.search(r"\bpatch\s*(>=|<=|>|<)\s*'", q):
        out.append("patch ids are not ordered by string or by number; use "
                   "patch_rank, or date_ordinal for recency")
    if "split" not in q:
        out.append("no split filter: this mixes train, val and test rows. Add "
                   "split = 'train' unless the question is about evaluation")
    if n_rows == 0 and not out:
        out.append("0 rows - check the filters before concluding the data is "
                   "empty")
    return " | ".join(out) or None


def run(con, query, max_rows=MAX_ROWS, timeout=TIMEOUT_S):
    """Execute one query. Returns {query, columns, rows, n_rows, truncated, ...}.

    Never raises for a bad query: an agent loop needs a refusal it can read and
    recover from, so every failure comes back as `{"error": ..., "hint": ...}`.
    """
    started = time.monotonic()
    try:
        body = _clean(query)
    except SqlRefused as exc:
        return {"error": str(exc), "query": query}

    deadline = started + timeout

    def _watchdog():
        return 1 if time.monotonic() > deadline else 0

    try:
        con.set_progress_handler(_watchdog, 2000)
        cur = con.execute("SELECT * FROM (%s) LIMIT %d" % (body, max_rows + 1))
        columns = [d[0] for d in cur.description] if cur.description else []
        raw = cur.fetchmany(max_rows + 1)
    except sqlite3.Error as exc:
        # `interrupted` is this tool's own watchdog, not a broken query, and the
        # difference matters: "no rows" and "aborted" lead to opposite
        # conclusions, and the second one is actionable. A model that reads
        # `rows: 0` reports that the corpus has nothing; a model that reads
        # "aborted after 2.0 s" narrows the query or says it could not check.
        if "interrupt" in str(exc).lower():
            return {"error": "the query was aborted after %.1f s" % timeout,
                    "timed_out": True, "query": body,
                    "hint": "narrow it: filter on kind / move_idx / role / "
                            "split, or aggregate fewer rows. A GROUP BY over "
                            "the whole move table (938k rows) is a full scan."}
        return {"error": "%s: %s" % (type(exc).__name__, exc), "query": body,
                "hint": hint(body, 0)}
    finally:
        con.set_progress_handler(None, 0)

    truncated = len(raw) > max_rows
    rows = [dict(zip(columns, r)) for r in raw[:max_rows]]
    return {"query": body, "columns": columns, "rows": rows,
            "n_rows": len(rows), "truncated": truncated,
            "elapsed_ms": round(1000 * (time.monotonic() - started), 1),
            "hint": hint(body, len(rows))}


TOOL_SPEC = {
    "name": "sql",
    "description": (
        "Run one read-only SQL query over the pro draft corpus (46,915 games, "
        "107 patches, one row per ban/pick). Use it for hyperspecific "
        "questions the other tools do not cover - a champion's ban rate in the "
        "last 3 patches, how often a pairing met in lane, what a team does "
        "after losing the first two bans. Always select the count next to any "
        "rate, and treat a count under 30 as an anecdote rather than a rate."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "a single SELECT"},
        },
        "required": ["query"],
    },
}
