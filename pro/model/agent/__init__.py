"""The agent's tool layer: a queryable corpus, the models as tools, and a log.

Not an agent. This package is the part that has to be *correct* before any
LangGraph loop is worth writing: what the tools return, what they refuse, and
what they record. The loop that decides which tool to call is the easy part and
the part most likely to change; these three modules are what the experiment
actually measures.

    schema    build the SQLite corpus, and the schema text the model is shown
    sql_tool  one read-only query tool with mechanical guards
    model_tools  predict_next_move / rank_candidates over the trained heads
    session   the registry, the call log, and the ablation switch

See `pro/model/agent/README.md` for why the schema looks the way it does.
"""
from .model_tools import (ChampIndex, predict_next_move, rank_candidates,
                          PREDICT_SPEC, RANK_SPEC)
from .session import Session
from .sql_tool import MAX_ROWS, TOOL_SPEC as SQL_SPEC, SqlRefused

__all__ = ["ChampIndex", "MAX_ROWS", "PREDICT_SPEC", "RANK_SPEC", "SQL_SPEC",
           "Session", "SqlRefused", "predict_next_move", "rank_candidates"]
