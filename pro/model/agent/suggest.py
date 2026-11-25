"""One-click agent suggestion: the reranking arm, served instead of measured.

`pro/model/archive/llm/eval_api_agent.py` measures a hosted chat model that reads the
held draft state, calls `predict_next_move` with a k it chooses and the lane
table, then commits to one champion. That is the middle-man job the retired Qwen
reranker used to do, with the tools and state management in place of an adapter.
This module is the *same configuration* as a product surface, so the button in
the frontend's agent panel and the 4,000-state measurement cannot drift:

  * the same question,
  * the same explained system prompt (`prompt_explained`, which tells the model
    where the head's numbers come from and how often they are wrong),
  * `sql` and `soloq_matchup` ablated (the free-form data tools are not offered
    here - this turn is a rerank of the head, not a data dive), and the same
    step/call budgets,
  * one turn on `real`, whose board is read-only.

The measurement is **accuracy** (answer == the pro's actual move); the panel's
job is **fidelity** (answer == the head's own #1 on this board, or a reasoned
overrule of it). So the answer is parsed into one champion - the final
`PICK: <name>` / `BAN: <name>` line the prompt asks for - and shipped next to
the head's #1. Nothing here claims the agent is right; it says what it chose
and whether it deferred to the model it was given.

A second arm in this module ranks, rather than commits. `rank_question` /
`parse_ranking` ask the same ablated agent to *order* the head's top-k rather
than to pick one, and read back a list of `NAME (NN%)` lines. The list is honest
by construction: `parse_ranking` only accepts champions the head listed - it
matches names against the store, and fills any untouched slot from the head's
own shortlist - so the agent relabels and reorders the head, it does not invent.
The panel renders the ranking beside the head's own #1, and re-runs it on every
board change so it is never stale.

The parser is deliberately a second implementation of the harness's
`answer_of`, not an import of it: `eval_api_agent` imports the training-side
`pro.model.archive.llm` stack (transformers, the corpus dataset), which the frontend's
base image does not have. The rules are the same - marker first, then the first
sentence, then the body; longest name wins - and are pinned on both sides.
"""
from __future__ import annotations

import re

from . import draft_state, prompt_explained

QUESTION = ("Recommend the single best move for the side on the clock. Name the "
            "champion first, then justify it in at most five sentences.")

# The measured arm's budget: 5 graph steps, 8 tool calls, no data tools. A
# suggestion is one question with one shortlist to read, not an open-ended
# conversation; the ordinary `/api/agent` turn keeps its larger budget and its
# corpus tools (including `soloq_matchup`). Ablating `sql` and `soloq_matchup`
# here keeps this turn a rerank of the head, not a data dive - the design that
# `session.py`'s docstring calls out.
MAX_STEPS = 5
MAX_CALLS = 8
ABLATE = frozenset({"sql", "soloq_matchup"})

# The ranking arm (`/api/agent-rank` + the panel's "agent suggestions" list):
# the same ablated explained-prompt agent, the same budget, but asked to *order*
# the head's top-k instead of commit to one. `RANK_K` is the panel's list length
# (kept equal to the head's own #next-move count so the two columns line up);
# the head's shortlist it reranks is fetched at k=20 and trimmed to the legal
# pool, so the agent has room to promote a deep pick without the parser falling
# back to the head's order for every slot.
RANK_K = 8
RANK_STEPS = 5
RANK_CALLS = 8
RANK_ABLATE = frozenset({"sql", "soloq_matchup"})


def system():
    """The prompt for this turn; the ordinary agent turn keeps `loop.SYSTEM`."""
    return prompt_explained.system_prompt()


def config():
    """What answered, for the panel. A prompt edit must be visible here."""
    return {"question": QUESTION, "system": "explained",
            "ablate": sorted(ABLATE), "max_steps": MAX_STEPS,
            "max_calls": MAX_CALLS}


# `PICK: **Leona**` / `BAN: Leona` on the last line. The marker is a request,
# not a guarantee, so the prose fallbacks below stay; `parse` records which path
# was taken so a run full of `body` rows reads as partly a parser's result.
_MARKER = re.compile(
    r"\b(PICK|BAN)\s*:\s*\**\s*([A-Za-z'.\- ]{2,24}?)\s*\**\s*$",
    re.IGNORECASE)


def _norm(text):
    """Alphanumerics only: `Kai'Sa` -> `kaisa`, `Dr. Mundo` -> `drmundo`."""
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _resolve(name, champs):
    try:
        return champs.resolve(name)
    except (ValueError, TypeError, AttributeError):
        return None


def _first_named(text, champs):
    """-> the first champion id named in `text`, or None.

    Left-to-right, longest match wins, overlaps dropped: without that `Vi`
    matches inside `Viktor` and a one-name answer reads as two. Same rule as
    `eval_api_agent._extract`, which the measured reports are parsed with.
    """
    norm = _norm(text)
    if not norm:
        return None
    hits = []
    for idx, name in enumerate(champs.names):
        n = _norm(name)
        if not n:
            continue
        start = norm.find(n)
        if start >= 0:
            hits.append((start, -len(n), idx))
    hits.sort()
    taken = []
    for start, neg_len, idx in hits:
        end = start - neg_len
        if any(not (end <= a or start >= b) for a, b in taken):
            continue
        return idx
    return None


def _marker_line(text):
    tail = str(text).strip().split("\n")[-1]
    m = _MARKER.search(tail)
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2).strip(" *_.")


def parse_answer(text, champs, state=None):
    """-> {champion, idx, parse, kind, kind_matches_clock, legal, note}.

    `legal` is about *this board*, not about the champion existing: a ban, a
    pick or a fearless-unavailable champion named by the model is a refusal of
    the constraint and must not be offered as a playable move. `kind` comes from
    the marker when there is one, otherwise from the clock; a disagreement is
    reported rather than silently resolved.
    """
    out = {"champion": None, "idx": None, "parse": "empty", "kind": None,
           "kind_matches_clock": None, "legal": False, "note": ""}
    if not text or not str(text).strip():
        return out
    due = state.to_move() if state is not None else None
    clock = (due or {}).get("kind")

    mark_kind, mark_name = _marker_line(text)
    idx = mode = kind = None
    if mark_name:
        idx = _resolve(mark_name, champs)
        if idx is not None:
            mode, kind = "marker", mark_kind
    if idx is None:
        first = re.split(r"(?<=[.!?])\s+", str(text).strip(), maxsplit=1)[0]
        idx = _first_named(first, champs)
        if idx is not None:
            mode, kind = "first-sentence", clock
    if idx is None:
        idx = _first_named(str(text), champs)
        if idx is not None:
            mode, kind = "body", clock
    if idx is None:
        out.update(parse="none", note="no champion name in the answer")
        return out

    name = champs.name(idx)
    taken = state.taken() if state is not None else set()
    blocked = state.unavailable if state is not None else ()
    illegal = draft_state.key(name) in taken or draft_state.key(name) in blocked
    out.update(champion=name, idx=int(idx), parse=mode, kind=kind or clock,
               kind_matches_clock=(clock is None or (kind or clock) == clock),
               legal=not illegal,
               note=("%s is on the board or in the fearless pool" % name)
               if illegal else "")
    return out


def head_fidelity(state, champs, suggestion):
    """The head's own #1 on this board, and whether the answer deferred to it.

    Fidelity is recomputed here rather than read from the agent's tool log: the
    model may have asked for k=20, cloned a state, or named no champion at all,
    and the panel's claim must be about the head's argmax on `real`. With no
    next-move model loaded the honest answer is None, not an empty string.
    """
    try:
        res = state.predict_next_move(k=1)
    except (ValueError, AttributeError):
        return None
    top = (res.get("top") or [None])[0]
    if not top:
        return None
    name = top["name"]
    idx = _resolve(name, champs)
    chosen = (suggestion or {}).get("champion")
    return {"champion": name, "idx": idx,
            "p_head": top.get("p_head"),
            "patch": res.get("patch_resolved"),
            "patch_known": res.get("patch_known"),
            "matches": (bool(chosen) and idx is not None
                        and draft_state.key(chosen) == draft_state.key(name))}


# ------------------------------------------------------------------- ranking
def rank_question(k=RANK_K, kind="pick"):
    """Ask the agent to rank the top-k moves for the side on the clock.

    Same honesty rule as the single-move arm: it must read the head's own
    ranking (via `predict_next_move`) and reorder *that*, not invent a list.
    The `NAME (NN%)` format is what `parse_ranking` reads back, so the
    confidence the panel shows is the agent's own number where it gave one, and
    the head's `p_head` where it did not. `kind` is "pick" or "ban" from the
    clock, so the prompt names the right move type.
    """
    return (
        "Rank the top %d %ss the side on the clock should make next, from best "
        "to worst. First call `predict_next_move` (and `predict_roles` if you "
        "are choosing among roles) to read the head's own legal shortlist, then "
        "reorder and relabel those candidates by how strong each is here. List "
        "exactly %d champions, one per line, each as `NAME (NN%%)` where NN%% "
        "is your confidence that this is the best available move. Do not name a "
        "champion the head did not list; if you trust the head, echo its order."
        % (k, kind, k))


def rank_config():
    """What answered, for the panel. Mirrors `config()` for the single arm."""
    return {"question": "rank top %d" % RANK_K, "system": "explained",
            "ablate": sorted(RANK_ABLATE), "max_steps": RANK_STEPS,
            "max_calls": RANK_CALLS, "k": RANK_K}


def _is_illegal(state, key):
    """A taken or fearless-unavailable champion named by the model is a refusal
    of the board's constraint, not a playable move."""
    if state is None:
        return False
    return key in state.taken() or key in state.unavailable


def _rank_from_head(head_top, k, champs):
    """Fallback when the turn returned no text: the head's own top-k."""
    out = []
    for h in (head_top or [])[:k]:
        p = h.get("p_head")
        if p is None or p <= 0:
            continue
        name = h.get("name")
        out.append({"name": name, "idx": _resolve(name, champs),
                    "conf": p, "source": "head", "legal": True, "note": ""})
    return {"suggestions": out, "parse": "head" if out else "none"}


def parse_ranking(text, champs, state=None, k=RANK_K, head_top=None):
    """-> {suggestions, parse}.

    `suggestions` is a list of `{"name", "idx", "conf", "source", "legal",
    "note"}`, best-first, at most `k`. `conf` is a share in [0, 1]; `source` is
    "model" for a champion the agent named with its own `(NN%)`, or "head" for a
    slot filled from the head's shortlist (no agent number, so `p_head`). A
    `conf <= 0` item is dropped, and any slot the agent left unnamed is filled
    from `head_top`, so the panel always shows `k` champions and never invents
    one the head did not list.

    `head_top` is the head's own legal shortlist, as a list of
    `{"name", "p_head"}` (fetched at k=20 and already trimmed to legality).

    The percentage may carry one decimal place: the hosted arms write `34%`,
    the local 4B writes `34.2%`, and both are the same claim.
    """
    if not text or not str(text).strip():
        return _rank_from_head(head_top, k, champs)
    # One regex over the whole answer: every champion name, longest first, as an
    # alternation, each required to be followed by `(NN%)`. Longest-first means
    # `Viktor` is tried before `Vi`, so a name is never matched inside a longer
    # one; case-insensitive so the agent's casing cannot hide a champion; the
    # trailing `(NN%)` is what `rank_question` asked for and what we read back.
    #
    # The percentage accepts a decimal (`15.6%`). The hosted arms write
    # integers and always did, so this changes none of their numbers - but a
    # local 4B asked for `NN%` answers with one decimal place, and an
    # integer-only pattern discarded its whole list and fell back to the head
    # without saying so. A parser that silently drops a valid answer is worse
    # than one that is slightly too permissive.
    names_sorted = sorted(champs.names, key=len, reverse=True)
    pat = re.compile(
        r"(?P<name>" + "|".join(re.escape(n) for n in names_sorted) + r")"
        r"\s*\**\s*\(\s*(?P<pct>\d+(?:\.\d+)?)\s*%\s*\)", re.IGNORECASE)
    seen = set()
    raw = []
    for m in pat.finditer(str(text)):
        name = m.group("name")
        key = draft_state.key(name)
        if key in seen:           # the agent repeating itself keeps the first
            continue
        seen.add(key)
        raw.append((m.start(), name, float(m.group("pct"))))
    # The agent lists best-to-worst top-to-bottom, so the panel order is the
    # order the names appear in the text, not the order the regex found them.
    raw.sort()

    suggestions = []
    for _start, name, conf in raw:
        if conf <= 0:
            continue
        key = draft_state.key(name)
        illegal = _is_illegal(state, key)
        idx = _resolve(name, champs)
        # The model's casing is not authoritative - render the store's canonical
        # (proper-case) name so the column never shows "AHRI" just because the
        # LLM wrote it that way.
        canon = champs.name(idx) if idx is not None else name
        suggestions.append({
            "name": canon, "idx": idx,
            "conf": conf / 100.0, "source": "model", "legal": not illegal,
            "note": ("%s is on the board or in the fearless pool" % canon)
                    if illegal else ""})
    # Fill untouched slots from the head, head-sourced, until we reach k.
    for h in (head_top or []):
        if len(suggestions) >= k:
            break
        hname = h.get("name")
        if not hname:
            continue
        hkey = draft_state.key(hname)
        if hkey in seen:
            continue
        p = h.get("p_head")
        if p is None or p <= 0:
            continue
        seen.add(hkey)
        suggestions.append({
            "name": hname, "idx": _resolve(hname, champs),
            "conf": p, "source": "head", "legal": True, "note": ""})

    if not suggestions:
        return {"suggestions": [], "parse": "none"}
    return {"suggestions": suggestions[:k],
            "parse": "model" if raw else "head"}
