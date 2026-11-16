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

**The store is the only door to a model.** There is no board-as-argument path
left: `predict_next_move` and `predict_roles` are methods on a state, and a
session without a store has no state, so it offers neither. That is deliberate -
a second way to name a board is a second thing that can disagree with the first,
and the disagreement is invisible from the answer.
"""
from __future__ import annotations

import time

from . import draft_state, model_tools, soloq_tool, sql_tool, state_tools
from .engine import Engine

_STATE_NAMES = {"get_state", "clone_state", "apply_move", "undo_move",
                "list_states"}
_MODEL_NAMES = {"predict_next_move", "predict_roles"}
_DATA_NAMES = {"soloq_matchup"}


class Session:
    def __init__(self, con=None, store=None, ablate=(), max_calls=24,
                 max_rows=sql_tool.MAX_ROWS, predictor=None, champs=None,
                 masker=None):
        self.con = con
        self.ablate = set(ablate or ())
        self.max_calls = int(max_calls)
        self.max_rows = max_rows
        # A store is opt-in, but `champs` is enough to build one, because the
        # store needs names rather than weights. Passing a predictor without a
        # store is the common case for a script, so it builds one too - and a
        # session given neither has no state and therefore no model tools.
        if store is None and champs is not None:
            store = draft_state.Store(champs, engine=Engine(
                champs=champs, predictor=predictor, masker=masker))
        elif store is not None and (predictor is not None or masker is not None):
            # A checkpoint loaded into a session whose board is already open.
            engine = store.engine
            store.attach(Engine(champs=store.champs, predictor=predictor,
                                masker=masker or getattr(engine, "masker", None),
                                patch=getattr(engine, "patch", None)))
        self.store = store
        self.log = []
        self._started = time.monotonic()

    # The models are the store's, not the session's. Kept as properties because
    # every caller and test asks the session what it has.
    @property
    def predictor(self):
        return getattr(self.store.engine, "predictor", None) if self.store else None

    @property
    def champs(self):
        return getattr(self.store.engine, "champs", None) if self.store else None

    @property
    def masker(self):
        return getattr(self.store.engine, "masker", None) if self.store else None

    # ------------------------------------------------------------- specs
    def available(self):
        out = []
        if self.store is not None:
            out += state_tools.SPECS
            engine = self.store.engine
            # Offered when either model is loaded: the role question can be
            # answered by a masking checkpoint alone, and hiding the tool would
            # make the ablation look like a capability gap.
            if engine.predictor is not None or engine.masker is not None:
                out.append(model_tools.PREDICT_SPEC)
            # Roles need the champion index and the corpus table, not weights.
            if engine.champs is not None:
                out.append(model_tools.ROLES_SPEC)
        if self.con is not None:
            out.append(sql_tool.TOOL_SPEC)
        # Offered whenever the soloQ analytical store is present on disk. The
        # connection is opened lazily by the tool, so advertising it costs
        # nothing here, and an agent with no store simply does not see it.
        if soloq_tool.available():
            out.append(soloq_tool.TOOL_SPEC)
        return [s for s in out if s["name"] not in self.ablate]

    def names(self):
        return [s["name"] for s in self.available()]

    # -------------------------------------------------------------- call
    def call(self, tool, **args):
        # The first parameter is `tool`, not `name`, because a tool argument can
        # be called `name` (clone_state takes one) and `call(name, **{"name": x})`
        # is a TypeError. The loop dispatches model-chosen arguments, so this is
        # not a theoretical collision.
        name = tool
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
        if name in _STATE_NAMES:
            return self._state_call(name, args)
        if name == "sql":
            if self.con is None:
                raise ValueError("no corpus database is open")
            return sql_tool.run(self.con, args.get("query"),
                                max_rows=self.max_rows)
        if name in _MODEL_NAMES:
            return self._model_call(name, args)
        if name in _DATA_NAMES:
            return soloq_tool.run(
                args.get("champion"), args.get("partner"),
                kind=args.get("kind") or "vs", patch=args.get("patch"),
                role=args.get("role"), min_n=args.get("min_n"),
                champs=self.champs)
        raise ValueError("unknown tool %r; available: %s"
                         % (name, ", ".join(self.names()) or "none"))

    # -------------------------------------------------------------- state
    def _state_call(self, name, args):
        store = self.store
        if name == "get_state":
            return state_tools.get(store, args.get("state") or draft_state.REAL_ID)
        if name == "clone_state":
            return state_tools.clone(store,
                                     args.get("from_state")
                                     or draft_state.REAL_ID,
                                     name=args.get("name"))
        if name == "apply_move":
            if not args.get("state"):
                raise ValueError("apply_move needs a state id; call get_state "
                                 "or list_states first")
            return state_tools.apply(store, args["state"], args.get("champion"),
                                     kind=args.get("kind"), side=args.get("side"))
        if name == "undo_move":
            if not args.get("state"):
                raise ValueError("undo_move needs a state id")
            return state_tools.undo(store, args["state"])
        return state_tools.listing(store)

    def _model_call(self, name, args):
        """Both model tools are methods on a state; the id is the only board.

        A state id that does not resolve is an `UnknownState` from the store,
        which reads as "unknown state 'x'; live states: real, s1" - the same
        refusal a state tool gives, so a model has one thing to learn.
        """
        sid = args.get("state") or draft_state.REAL_ID
        st = self.store.get(sid)
        if name == "predict_roles":
            out = st.predict_roles()
        else:
            out = st.predict_next_move(k=args.get("k") or 8,
                                       role=args.get("role"),
                                       side=args.get("side"))
        out["state"] = st.summary(sid)
        return out

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
                "budget_refusals": sum(1 for e in self.log if e.get("budget"))}

    def summary(self):
        out = self.mix()
        out["ablate"] = sorted(self.ablate)
        out["seconds"] = round(time.monotonic() - self._started, 1)
        if self.store is not None:
            out["models"] = self.store.engine.describe()
        return out


def _trim(args, limit=240):
    """Keep the log readable: a full 46k-row argument set is not a log entry."""
    out = {}
    for k, v in (args or {}).items():
        s = str(v)
        out[k] = s if len(s) <= limit else s[:limit] + "..."
    return out
