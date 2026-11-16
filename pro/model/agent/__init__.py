"""The agent's tool layer: a queryable corpus, the models as a draft object.

This package is the part that has to be *correct* before any loop is worth
running: what the tools return, what they refuse, and what they record.

    schema       build the SQLite corpus, and the schema text the model is shown
    sql_tool     one read-only query tool with mechanical guards
    model_tools  the model calls: next move, roles, and one role's next pick
    engine       the models bound to a state, and the routing between them
    draft_state  the draft as an object: moves, roles, and its own questions
    session      the registry, the call log, and the ablation switch

Two modules sit on top and are **not** imported here, so that the tool layer
stays importable without `langgraph` or a network stack:

    llm_client   an OpenAI-compatible chat client with tool calling
    loop         the bounded tool-calling graph, over `Session`
    mcp_server   the same tools over the MCP stdio wire protocol
    suggest      the one-click recommendation: prompt, budgets, answer parser

See `pro/model/agent/README.md` for why the schema looks the way it does.
"""
from .draft_state import DraftState, Role, Store
from .engine import Engine
from .model_tools import (ChampIndex, PREDICT_SPEC, ROLES_SPEC, assignment,
                          predict_in_lane, predict_next_move, predict_roles)
from .session import Session
from .sql_tool import MAX_ROWS, TOOL_SPEC as SQL_SPEC, SqlRefused

__all__ = ["ChampIndex", "DraftState", "Engine", "MAX_ROWS", "PREDICT_SPEC",
           "ROLES_SPEC", "Role", "SQL_SPEC", "Session", "SqlRefused", "Store",
           "assignment", "predict_in_lane", "predict_next_move", "predict_roles"]
