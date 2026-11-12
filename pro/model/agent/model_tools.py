"""The models, as tools. This is the tier that carries the judgment.

Two tools, not four, and the split is deliberate:

`predict_next_move` answers "what should we pick" - the head's distribution,
optionally re-ordered by the reranker. It exists because the head is good at
this (50.5 % top-5 on the shared val) and an LLM is not, and because the
numbers are useless to a reader who cannot ask for them.

`rank_candidates` answers "Kled or Olaf", which is what a coach actually asks
and what the reranker was trained for - a k-way comparison of a *named* set
rather than a top-k. It is the one place where the LLM's framing and the
model's strength line up, so it is the tool to reach for first.

Three things this module refuses to do, each for a measured reason:

- **It never returns a bare probability.** `p_head` is a share of the legal
  pool (164 champions), `p_rerank` is a share of the candidate set (5). Those
  are different claims wearing the same number, so every entry carries `rank`
  as well and the result carries `note`.
- **It always echoes the patch the model actually used.** Pass 16.22 and the
  head maps it to UNK and answers patch-agnostically without saying so;
  `patch_known: false` is the only signal that this happened.
- **It validates the board.** A champion already banned, or a full board,
  comes back as a refusal with the specific error. An agent that learns to
  trust a distribution computed for an illegal board is worse than one with
  no tools at all.
"""
from __future__ import annotations

import re

ROLES = ("top", "jungle", "mid", "bot", "support")


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


class ChampIndex:
    """Name -> dense index, tolerant of the ways people type champion names.

    Same normalisation as `schema.key` and `frontend/chat._norm`; longest key
    first so 'Dr. Mundo' is not read as 'Dr'. Ints pass through, because a
    caller holding an index should not have to look the name back up.
    """

    def __init__(self, names):
        self.names = list(names)
        self.by_key = {}
        for i, name in enumerate(self.names):
            k = _norm(name)
            self.by_key.setdefault(k, i)
            self.by_key.setdefault(k.replace(" ", ""), i)

    def resolve(self, value):
        if value is None:
            return None
        if isinstance(value, int):
            if 0 <= value < len(self.names):
                return value
            raise ValueError("champion index %r is out of range" % (value,))
        k = _norm(value)
        for cand in (k, k.replace(" ", "")):
            if cand in self.by_key:
                return self.by_key[cand]
        # last resort: the query is the last word of a key, so 'mundo' finds
        # 'Dr. Mundo' and 'sin' finds 'Lee Sin'
        if k:
            hits = [i for kk, i in self.by_key.items()
                    if kk.endswith(" " + k) or kk.replace(" ", "").endswith(k)]
            if len(set(hits)) == 1:
                return hits[0]
        raise ValueError("unknown champion %r - use a dDragon name or an index"
                         % (value,))

    def resolve_all(self, values):
        return [self.resolve(v) for v in (values or [])]

    def name(self, idx):
        return self.names[idx]


def _head(predictor):
    return getattr(predictor, "head", predictor)


def _num_champs(head):
    return int(getattr(head, "num_champs", 173))


def _patch_state(predictor, patch):
    """-> (patch_id, resolved_key|None, known).

    Patch ids are 1-based row numbers into the checkpoint's `patch_vocab`, not
    the label - `patch_key_for` does `patch_vocab[pid - 1]`, and id 0 is UNK.
    So a label has to be looked up in the vocabulary rather than parsed, and
    anything not found is UNK: the head will then answer patch-agnostically
    without saying so, which is why `known` is returned rather than inferred
    by the caller.
    """
    if patch is None:
        return None, None, True
    holder = predictor if hasattr(predictor, "patch_vocab") else _head(predictor)
    vocab = [str(k) for k in (getattr(holder, "patch_vocab", None) or [])]
    pid = None
    if isinstance(patch, int):
        fn = getattr(holder, "patch_id_for", None)
        pid = fn(patch) if fn is not None else patch
    else:
        want = _norm(str(patch))
        for i, k in enumerate(vocab):
            if _norm(k) == want:
                pid = i + 1
                break
    resolved = None
    fn = getattr(holder, "patch_key_for", None)
    if fn is not None and pid is not None:
        try:
            resolved = fn(pid)
        except (ValueError, KeyError, TypeError):
            resolved = None
    elif pid is not None and 1 <= pid <= len(vocab):
        resolved = vocab[pid - 1]
    known = resolved is not None and _norm(str(resolved)) == _norm(str(patch))
    return pid, resolved, known


def _state(res):
    """The move the answer is about. `next_move` returns these at the top
    level; older callers nested them under `state`.

    `side` comes back as 0/1. It is reported as 'blue'/'red' because a bare 0
    in a transcript reads as "no side", and the model would have to know the
    convention to get it right.
    """
    nested = res.get("state") or {}
    out = {}
    for k in ("kind", "side", "move_idx", "phase", "complete", "n_legal"):
        out[k] = res[k] if k in res else nested.get(k)
    if isinstance(out.get("side"), int):
        out["side"] = "blue" if out["side"] == 0 else "red"
    return out


def _check(blue, red, bans):
    """Validate a board before a model sees it.

    A distribution computed for an illegal board is worse than no answer: it is
    indistinguishable from a good one.
    """
    if len(blue) > 5 or len(red) > 5:
        raise ValueError("a side has five slots; got %d blue and %d red"
                         % (len(blue), len(red)))
    if len(bans) > 10:
        raise ValueError("a draft has 10 bans; got %d" % len(bans))
    seen = {}
    for c in list(blue) + list(red) + list(bans):
        seen[c] = seen.get(c, 0) + 1
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        raise ValueError("champion index %d is on the board twice - a champion "
                         "cannot be picked and banned, or picked twice"
                         % dupes[0])


def _slots(blue, red):
    """blue/red slot lists -> the head's 10-slot layout (blue 0-4, red 5-9).

    Two side lists rather than one 10-slot array because that is how the
    question is actually asked - "blue has Aatrox Vi Ahri, red has Poppy Lee
    Sin" - and a single sparse list cannot say which side the third name
    belongs to. Placement is by slot order within each side, which is what the
    head was trained on (canonical blue top..support, then red).
    """
    return list(blue) + [None] * (5 - len(blue)) + list(red) + [None] * (5 - len(red))


def predict_next_move(predictor, champs, blue_picks=(), red_picks=(), bans=(),
                      first_pick=0, patch=None, k=8, unavailable=(),
                      rerank=None):
    """The head's answer for this board, with its units attached.

    Returns {model, kind, side, move_idx, n_legal, patch_requested,
    patch_resolved, patch_known, top: [{name, rank, p_head, p_rerank}], note}.
    """
    blue_idx = champs.resolve_all(blue_picks)
    red_idx = champs.resolve_all(red_picks)
    bans_idx = champs.resolve_all(bans)
    unavail_idx = champs.resolve_all(unavailable)
    _check(blue_idx, red_idx, bans_idx)
    head = _head(predictor)
    pid, resolved, known = _patch_state(predictor, patch)

    res = head.next_move(_slots(blue_idx, red_idx), bans_idx,
                         first_pick=1 if first_pick else 0, k=int(k),
                         unavailable=list(unavail_idx), patch_id=pid)
    sugg = res.get("suggestions") or []
    state = _state(res)
    legal = _num_champs(head) - len(blue_idx) - len(red_idx) - len(bans_idx) \
        - len(unavail_idx)
    if state.get("n_legal"):
        legal = int(state["n_legal"])

    top = [{"name": champs.name(int(c["idx"])), "rank": i + 1,
            "p_head": round(float(c["prob"]), 5)}
           for i, c in enumerate(sugg)]
    note = ("p_head is a share of the %d legal champions. " % legal
            if legal > 0 else "p_head is a share of the legal pool. ")
    if rerank is not None:
        try:
            scores = rerank(_slots(blue_idx, red_idx), bans_idx,
                            1 if first_pick else 0,
                            [int(c["idx"]) for c in sugg], pid) or {}
        except Exception as exc:  # noqa: BLE001 - the head's list still stands
            scores, note = {}, note + ("rerank failed (%s), showing the head; "
                                       % exc)
        if scores:
            z = sum(float(v) for v in scores.values()) or 1.0
            for c in top:
                v = scores.get(int(champs.resolve(c["name"])), None)
                if v is not None:
                    c["p_rerank"] = round(float(v) / z, 5)
            note += ("p_rerank is a share of these %d candidates, not of the "
                     "pool - do not compare the two." % len(top))
    if patch is not None and not known:
        note += (" patch %s is not in the model's patch table (UNK), so this "
                 "answer is patch-agnostic." % patch)
    return {"model": str(getattr(predictor, "model", "policy")),
            "kind": state.get("kind"), "side": state.get("side"),
            "move_idx": state.get("move_idx"), "n_legal": legal,
            "patch_requested": patch, "patch_resolved": resolved,
            "patch_known": known, "top": top, "note": note}


def rank_candidates(predictor, champs, blue_picks=(), red_picks=(), bans=(),
                    candidates=(), first_pick=0, patch=None, rerank=None):
    """Score a *named* set of candidates against each other.

    The head is asked for its whole distribution, not a top-k, so a champion it
    ranks 60th still gets a number and the comparison is between the candidates
    rather than between two different pools.
    """
    cand_idx = champs.resolve_all(candidates)
    if len(cand_idx) < 2:
        raise ValueError("rank_candidates needs at least two candidates; use "
                         "predict_next_move for a single suggestion")
    if len(set(cand_idx)) != len(cand_idx):
        raise ValueError("the same champion was passed twice")
    blue_idx = champs.resolve_all(blue_picks)
    red_idx = champs.resolve_all(red_picks)
    bans_idx = champs.resolve_all(bans)
    _check(blue_idx, red_idx, bans_idx)
    on_board = [c for c in cand_idx
                if c in blue_idx or c in red_idx or c in bans_idx]
    if on_board:
        raise ValueError("%s is already on the board, so it cannot be ranked "
                         "for this move" % champs.name(on_board[0]))
    head = _head(predictor)
    pid, resolved, known = _patch_state(predictor, patch)

    res = head.next_move(_slots(blue_idx, red_idx), bans_idx,
                         first_pick=1 if first_pick else 0,
                         k=_num_champs(head), patch_id=pid)
    prob = {int(c["idx"]): float(c["prob"]) for c in res.get("suggestions") or []}
    total = sum(prob.get(c, 0.0) for c in cand_idx)
    rows = []
    for c in cand_idx:
        p = prob.get(c, 0.0)
        rows.append({"name": champs.name(c), "p_head": round(p, 5),
                     "share_of_candidates": round(p / total, 4) if total else None,
                     "rank_in_pool": 1 + sum(1 for v in prob.values() if v > p)})
    rows.sort(key=lambda r: -r["p_head"])
    for i, r in enumerate(rows):
        r["rank_in_candidates"] = i + 1
    note = ("rank_in_pool is against all %d legal champions; "
            "share_of_candidates is the renormalised comparison you asked for."
            % len(prob))
    if patch is not None and not known:
        note += (" patch %s is UNK for this model, so the answer is "
                 "patch-agnostic." % patch)
    return {"model": str(getattr(predictor, "model", "policy")),
            "patch_requested": patch, "patch_resolved": resolved,
            "patch_known": known, "candidates": rows, "note": note}


_BOARD_PROPS = {
    "blue_picks": {"type": "array", "items": {"type": "string"},
                   "description": "blue's picks in slot order (top, jungle, "
                                  "mid, bot, support); omit unfilled slots"},
    "red_picks": {"type": "array", "items": {"type": "string"},
                  "description": "red's picks in slot order"},
    "bans": {"type": "array", "items": {"type": "string"},
             "description": "all bans, in the order they were made"},
    "first_pick": {"type": "integer", "enum": [0, 1],
                   "description": "1 if blue has the first pick of the game"},
    "patch": {"type": "string", "description": "e.g. '16.20'"},
}

PREDICT_SPEC = {
    "name": "predict_next_move",
    "description": (
        "The trained next-move model's answer for this board: the top champions "
        "for the move on the clock, each with its share of the legal pool and "
        "its rank. Use this instead of guessing what pros pick. It does not "
        "explain itself, and it cannot tell blue from red."),
    "parameters": {
        "type": "object",
        "properties": dict(_BOARD_PROPS, k={"type": "integer", "default": 8}),
        "required": ["blue_picks", "red_picks", "bans"],
    },
}

RANK_SPEC = {
    "name": "rank_candidates",
    "description": (
        "Compare a named set of two to five champions for the move on the "
        "clock - 'Kled or Olaf here?'. Returns each one's rank among all legal "
        "champions and its share within the set. Prefer this over "
        "predict_next_move when the question is a choice between specific "
        "names."),
    "parameters": {
        "type": "object",
        "properties": dict(_BOARD_PROPS, candidates={
            "type": "array", "items": {"type": "string"},
            "minItems": 2, "maxItems": 5}),
        "required": ["blue_picks", "red_picks", "bans", "candidates"],
    },
}
