"""The state tools: see a draft, clone it, play a move, take it back.

These are the verbs the model uses instead of retyping a board. The important
property is not convenience - it is that a champion name can no longer arrive
wrong. `apply_move` resolves the name, checks it against the board and against
the schedule, and refuses with a sentence the model can act on.

The workflow the specs describe, in the order the prompt gives it:

    get_state                       read the live board, its roles included
    clone_state                     make a sandbox to explore in
    apply_move                      "if blue picks Alistar here"
    predict_next_move(state=...)    "then red is likely to ..."
    undo_move                       take it back and try the other branch

`predict_next_move` and `predict_roles` take the same `state` id, so the board
never appears in a tool call. That is the whole point: the model can still *read*
the board - it has to, to reason about comps - but it never has to reproduce it.

`get_state` carries the role assignment because the state does. Which role each
pick fills, how confident that is, and what each side still needs arrive with
the board, so a question about "red's jungle" needs no second call to find out
that red has no jungle yet.
"""
from __future__ import annotations

from . import draft_state

_GET = {
    "name": "get_state",
    "description": (
        "Read a draft state: the board, whose move it is, how many champions "
        "are still legal, and which role every pick fills with the corpus's "
        "confidence in it (`roles`) plus what each side still has to fill "
        "(`open_roles`). Call this before answering anything about the board, "
        "and again after changing a sandbox. `real` is the live board. The "
        "`fearless` list is champions picked in earlier games of this series - "
        "they are not on the board and they are not available."),
    "parameters": {
        "type": "object",
        "properties": {"state": {"type": "string", "default": "real"}},
    },
}

_CLONE = {
    "name": "clone_state",
    "description": (
        "Copy a state into a new sandbox and return its id. Use this before "
        "exploring - the live board (`real`) is read-only, and every hypothetical "
        "must happen on a clone. Cloning the live board and playing one move is "
        "how you answer 'what if blue picks X here'."),
    "parameters": {
        "type": "object",
        "properties": {
            "from_state": {"type": "string", "default": "real"},
            "name": {"type": "string",
                     "description": "a short label, e.g. 'alistar-line'"},
        },
    },
}

_APPLY = {
    "name": "apply_move",
    "description": (
        "Play a champion into the next slot of a sandbox and return the new "
        "board, with the role assignment it now implies. The kind and side come "
        "from the draft schedule, so you only name the champion; pass kind/side "
        "only if you mean to check them, and a disagreement is refused. A "
        "champion already on the board is refused, and so is one from the "
        "fearless pool."),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {"type": "string"},
            "champion": {"type": "string"},
            "kind": {"type": "string", "enum": ["ban", "pick"]},
            "side": {"type": "string", "enum": ["blue", "red"]},
        },
        "required": ["state", "champion"],
    },
}

_UNDO = {
    "name": "undo_move",
    "description": (
        "Take back the last move in a sandbox, so you can try the other branch "
        "from the same position. Returns the restored board and its roles."),
    "parameters": {
        "type": "object",
        "properties": {"state": {"type": "string"}},
        "required": ["state"],
    },
}

_LIST = {
    "name": "list_states",
    "description": ("Every live state: id, label, how many moves deep it is, and "
                    "whether it is read-only."),
    "parameters": {"type": "object", "properties": {}},
}

SPECS = [_GET, _CLONE, _APPLY, _UNDO, _LIST]


def get(store, state=draft_state.REAL_ID):
    st = store.get(state)
    return dict(st.summary(state), label=store.label(state),
                rendered=st.render(state, store.display),
                readonly=(state == draft_state.REAL_ID))


def clone(store, from_state=draft_state.REAL_ID, name=None):
    state_id = store.clone(from_state, name=name)
    st = store.get(state_id)
    return dict(st.summary(state_id), label=store.label(state_id),
                rendered=st.render(state_id, store.display), cloned_from=from_state)


def apply(store, state, champion, kind=None, side=None):
    """Play one champion and report both the move made and the position after it.

    `applied` exists because the summary describes the *next* move: without it
    "move 12, red ban" reads as if the champion just played were that ban.
    """
    due = store.get(state).to_move() or {}
    st = store.apply(state, champion, kind=kind, side=side)
    return dict(st.summary(state), label=store.label(state),
                rendered=st.render(state, store.display),
                applied=dict(due, champion=store.display(store.resolve(champion))))


def undo(store, state):
    st = store.undo(state)
    return dict(st.summary(state), label=store.label(state),
                rendered=st.render(state, store.display))


def listing(store):
    states = store.list()
    return {"n_states": len(states), "states": states}
