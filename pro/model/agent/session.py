"""One tool registry, with a log and an off switch.

Two things the experiment needs that the tools themselves do not provide:

**A log.** Every call is recorded with its arguments, its latency, whether it
succeeded and how many rows it returned. Without it the only measurement is the
final pick, and the interesting question - did the agent use the models, or did
it fall back to SQL and argue with itself - is invisible. `mix()` is the specific
number the design turns on: if more than half the calls are `sql`, the
high-level tier is under-specified, not the SQL tier over-powered.

**An ablation.** `ablate={"sql"}` removes a tool from the advertised specs *and*
refuses it at call time. Both halves matter: hiding the spec tests the agent as
it would be shipped, and refusing the call catches the model that invents a tool
name it was never offered. Refusals are logged as `ablated` rather than as
errors, because "it tried and could not" is a result, not a failure.

Every tool call returns a dict and never raises. An agent loop that dies on a
bad argument measures the harness, not the model.
"""
from __future__ import annotations

import time

from . import model_tools, sql_tool


class Session:
    def __init__(self, con=None, predictor=None, champs=None, ablate=(),
                 max_calls=24, rerank=None, max_rows=sql_tool.MAX_ROWS):
        self.con = con
        self.predictor = predictor
        self.champs = champs
        self.ablate = set(ablate or ())
        self.max_calls = int(max_calls)
        self.rerank = rerank
        self.max_rows = max_rows
        self.log = []
        self._started = time.monotonic()

    # ------------------------------------------------------------- specs
    def available(self):
        out = []
        if self.predictor is not None and self.champs is not None:
            out += [model_tools.PREDICT_SPEC, model_tools.RANK_SPEC]
        if self.con is not None:
            out.append(sql_tool.TOOL_SPEC)
        return [s for s in out if s["name"] not in self.ablate]

    def names(self):
        return [s["name"] for s in self.available()]

    # -------------------------------------------------------------- call
    def call(self, name, **args):
        entry = {"i": len(self.log), "name": name,
                 "args": _trim(args), "t": round(time.monotonic()
                                                 - self._started, 2)}
        started = time.monotonic()
        try:
            if name in self.ablate:
                out = {"error": "tool %r is not available in this run" % name,
                       "ablated": True}
                entry["ablated"] = True
            elif len(self.log) >= self.max_calls:
                out = {"error": "tool budget of %d calls is exhausted"
                                 % self.max_calls, "budget": True}
                entry["budget"] = True
            else:
                out = self._dispatch(name, args)
        except Exception as exc:  # noqa: BLE001 - a loop must not die here
            out = {"error": "%s: %s" % (type(exc).__name__, exc)}
        entry["ms"] = round(1000 * (time.monotonic() - started), 1)
        entry["ok"] = "error" not in out
        if isinstance(out, dict) and out.get("n_rows") is not None:
            entry["n_rows"] = out["n_rows"]
        if not entry["ok"]:
            entry["error"] = out.get("error")
        self.log.append(entry)
        return out

    def _dispatch(self, name, args):
        # A tool the session is not advertising is not a tool. This is what
        # stops a model from calling `predict_next_move` in a run where no
        # checkpoint was loaded, and getting an AttributeError instead of the
        # list of what it could have asked for.
        if name not in self.names():
            raise ValueError("unknown tool %r; available: %s"
                             % (name, ", ".join(self.names()) or "none"))
        if name == "sql":
            if self.con is None:
                raise ValueError("no corpus database is open")
            return sql_tool.run(self.con, args.get("query"),
                                max_rows=self.max_rows)
        if name == "predict_next_move":
            return model_tools.predict_next_move(
                self.predictor, self.champs,
                args.get("blue_picks") or [], args.get("red_picks") or [],
                args.get("bans") or [],
                first_pick=args.get("first_pick") or 0,
                patch=args.get("patch"), k=args.get("k") or 8,
                unavailable=args.get("unavailable") or (), rerank=self.rerank)
        if name == "rank_candidates":
            return model_tools.rank_candidates(
                self.predictor, self.champs,
                args.get("blue_picks") or [], args.get("red_picks") or [],
                args.get("bans") or [], args.get("candidates") or [],
                first_pick=args.get("first_pick") or 0, patch=args.get("patch"),
                rerank=self.rerank)
        raise ValueError("unknown tool %r; available: %s"
                         % (name, ", ".join(self.names()) or "none"))

    # ---------------------------------------------------------- reporting
    def mix(self):
        """-> {tool: n} plus `sql_share`, the number the design is judged on."""
        counts = {}
        for e in self.log:
            counts[e["name"]] = counts.get(e["name"], 0) + 1
        n = len(self.log)
        return {"calls": n, "by_tool": counts,
                "sql_share": round(counts.get("sql", 0) / n, 3) if n else 0.0,
                "errors": sum(1 for e in self.log if not e["ok"]),
                "ablated": sum(1 for e in self.log if e.get("ablated")),
                "budget_refusals": sum(1 for e in self.log
                                       if e.get("budget"))}

    def summary(self):
        out = self.mix()
        out["ablate"] = sorted(self.ablate)
        out["seconds"] = round(time.monotonic() - self._started, 1)
        return out


def _trim(args, limit=240):
    """Keep the log readable: a full 46k-row argument set is not a log entry."""
    out = {}
    for k, v in (args or {}).items():
        s = str(v)
        out[k] = s if len(s) <= limit else s[:limit] + "..."
    return out
