"""The models, as functions a draft state can call.

Four questions, and each one is a different object rather than a different
parameter:

`predict_next_move` answers "what should be played" - the head's distribution
for the move on the clock, with `rank` and its share of the pool. It exists
because the head is good at this (50.5 % top-5 on the shared val) and an LLM is
not, and because the numbers are useless to a reader who cannot ask for them.
`k` is the caller's: the head's top-20 covers 85 % of the answers and its top-5
51 %, so how wide to read is a decision the caller should make.

`predict_roles` answers "what does this team still need" - the one-to-one role
assignment of each side's picks, with the champion's own history as the
confidence, and which roles are still open. It reads the corpus table, not a
head, so it costs a dict lookup and cannot disagree with itself between calls.

`predict_in_lane` answers "what does the model think red's jungle is" - the same
head, asked a narrower question: its logits masked to the champions that play
that lane, for a side that may not be on the clock. It is the query the role
board is built from, and it is *not* a filter over `predict_next_move`: filtering
an unconstrained top-k leaves an empty list whenever the head's five favourites
happen not to play the lane, which is 13 % of pick moves.

`assignment` is the shared primitive under `predict_roles` and `predict_in_lane`:
the 10-slot board -> one role per filled slot, injective per team.

**There is no `rank_candidates`.** Comparing a named set is what an LLM is
actually good at, and it is now the harnessed model's job: it reads the head's
list and says why. A tool that renormalised the head's numbers over a shortlist
the model wrote itself produced confident answers to the wrong question - the
shortlist could omit the head's top pick, and then the comparison was internally
consistent and irrelevant. Measured in `pro/model/archive/llm/README.md`.

Three things this module refuses to do, each for a measured reason:

- **It never returns a bare probability.** `p_head` is a share of the legal pool
  (about 164 champions), `p_lane` is a share of one champion's own picks, and
  `p_in_lane` is a share of the champions that play one lane. Those are
  different claims wearing the same number, so every entry carries `rank` as well
  and the result carries `note`.
- **It always echoes the patch the model actually used.** Pass 16.22 and the
  head maps it to UNK and answers patch-agnostically without saying so;
  `patch_known: false` is the only signal that this happened.
- **It validates the board.** A champion already banned, or a full board, comes
  back as a refusal with the specific error. An agent that learns to trust a
  distribution computed for an illegal board is worse than one with no tools.
"""
from __future__ import annotations

import math
import re

from solo_queue.model.policy import lanes as _lanes

ROLES = ("top", "jungle", "mid", "bot", "support")
# `base_model.ROLE_FLEX_ID`, duplicated because importing it pulls torch into a
# module the state service imports before any weights exist. It is the INPUT id
# meaning "this pick's role is unknown", exactly like MASK for a champion.
ROLE_FLEX_ID = 5

_ROLE_ALIASES = {
    "top": 0, "jng": 1, "jg": 1, "jungle": 1, "jungler": 1, "mid": 2,
    "middle": 2, "bot": 3, "bottom": 3, "adc": 3, "support": 4, "sup": 4,
    "supp": 4, "utility": 4, "util": 4,
}


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def role_index(role):
    """'jungle' / 'JNG' / 1 -> 1. Refuses anything the table does not have.

    The role has to arrive as one of five, because it is used to mask logits:
    an unrecognised role silently read as 0 would answer a question about the
    top lane that nobody asked.
    """
    if isinstance(role, int):
        if 0 <= role < len(ROLES):
            return role
        raise ValueError("role %r is not one of %s" % (role, ", ".join(ROLES)))
    key = _norm(role)
    if key in _ROLE_ALIASES:
        return _ROLE_ALIASES[key]
    raise ValueError("unknown role %r - use one of %s"
                     % (role, ", ".join(ROLES)))


class ChampIndex:
    """Name -> dense index, tolerant of the ways people type champion names.

    Same normalisation as `schema.key` and `webapp/chat._norm`; longest key
    first so 'Dr. Mundo' is not read as 'Dr'. Ints pass through, because a
    caller holding an index should not have to look the name back up.

    **The order has to be the corpus order** (`schema.names()`), not any
    alphabetical or hand-written list: every checkpoint's output layer and
    `data/champ_roles.json` are keyed by that dense index, so an index built
    from a different list answers confidently about the wrong champions. The
    role table is the quiet one - it never raises, it just places Aatrox in the
    top lane.
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
    """The *wrapped* head, for introspection only (`num_champs`, `patch_vocab`).

    Not the model to ask. A `RerankPredictor` holds a `PolicyPredictor` in
    `.head` and re-orders its shortlist, so `_head(pred).next_move(...)` answers
    with the head and throws the reranker away - the exact bug that made the
    agent's next-pick tool ignore the 4B model. Ask `predictor.next_move`.
    """
    return getattr(predictor, "head", predictor)


def _serving(predictor):
    """The predictor whose `next_move` is the answer: reranker if there is one.

    `hasattr` rather than a type check because callers pass duck-typed
    predictors (the test suite's stubs hold a `.head` and nothing else), and
    because a wrapper that only delegates attributes is not the model.
    """
    if predictor is not None and hasattr(predictor, "next_move"):
        return predictor
    return _head(predictor)


def label(predictor):
    """A short name for the model, never the module.

    `str(getattr(predictor, "model", "policy"))` looks safe and is not: the
    shipped `PolicyPredictor.model` is a property returning the torch module, so
    the tool result carried the entire `PolicyValueModel(...)` repr - two
    thousand characters of architecture dump in a JSON payload the model is
    supposed to read, repeated on every call. Prefer a string label, then the
    predictor's kind.
    """
    for attr in ("label", "kind", "ckpt_path"):
        value = getattr(predictor, attr, None)
        if isinstance(value, str) and value:
            return value
    return "policy"


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
    head was trained on; the encoder is permutation-invariant within a team, so
    pick order is fine and a role ordering would be a second guess.
    """
    return list(blue) + [None] * (5 - len(blue)) + list(red) + [None] * (5 - len(red))


def _top(p, champs, k, illegal=()):
    """A probability vector -> the ranked rows every tool returns."""
    keep = [(int(i), float(v)) for i, v in enumerate(p)
            if float(v) > 0 and int(i) not in illegal]
    keep.sort(key=lambda kv: -kv[1])
    return [{"name": champs.name(i), "rank": n + 1, "p": v}
            for n, (i, v) in enumerate(keep[:k])]


# ------------------------------------------------------------------ roles
def assignment(champs, picks):
    """The 10-slot board -> one role per filled slot, with its confidence.

    `picks` is blue's five in pick order then red's five, `None` for an empty
    slot (the head's own layout, so this and `predict_next_move` see one board).

    **One-to-one, per side.** A team's five roles are distinct, and two flex
    champions that both list jungle would both claim it under a per-slot
    argmax. `lanes.assign_lanes` maximises the summed log-likelihood over the
    <=120 injective assignments per team instead, which is what splits them: a
    single-role champion (113 of 173) claims its lane and pushes the flex one
    elsewhere.

    `p_lane` is *that champion's* share of its own picks in that lane - 0.98 is
    a dedicated laner, 0.45 a genuine flex. It is not a probability that the
    team wants the lane, and it is not a prediction about this game. It is
    returned rather than a bare role name because "jungle, 0.42" and "jungle,
    0.99" are different claims about the same placement.
    """
    assigned = _lanes.assign_lanes(list(picks))
    out = []
    for slot, champ in enumerate(picks):
        if champ is None or slot >= len(assigned):
            continue
        lane = assigned[slot]
        if lane is None:                       # cannot happen; be loud
            raise ValueError("the table could not place %s in a role"
                             % champs.name(champ))
        out.append({"slot": slot,
                    "side": "blue" if slot < 5 else "red",
                    "pick": (slot % 5) + 1,
                    "champion": champs.name(int(champ)),
                    "role": ROLES[int(lane)],
                    "role_id": int(lane),
                    "p_lane": round(math.exp(_lanes.lane_logp(int(champ), lane)), 4)})
    return out


def open_roles(rows):
    """The roles a side still has to fill, given its assignment rows."""
    out = {}
    for side in ("blue", "red"):
        taken = {r["role"] for r in rows if r["side"] == side}
        out[side] = [r for r in ROLES if r not in taken]
    return out


def legal_in_lane(champs, taken=()):
    """-> {role: how many champions that play it are still draftable here}.

    The number a caller needs to decide whether a lane is worth filling from a
    shortlist at all: a role with four legal champions is a different decision
    from one with thirty.
    """
    taken = set(taken or ())
    out = {}
    for lane in range(len(_lanes.table()["roles"])):
        pool = _lanes.legal_champions(lane)
        out[ROLES[int(lane)]] = sum(1 for c in pool if c not in taken)
    return out


def predict_roles(champs, blue_picks=(), red_picks=(), bans=(), unavailable=()):
    """Which role each pick fills, how sure that is, and what is still open.

    Reads `data/champ_roles.json` (46,915 drafts of pick-role pairs) rather than
    a head, so it costs a dict lookup and an exhaustive search over at most 120
    injective assignments per team - it cannot be wrong about a patch and it
    cannot disagree with itself between calls.

    The 96 %-agreement number belongs to this table, not to the role head: the
    head's own argmax agrees with the corpus 67 % of the time, and serving
    recall@5 follows (30.2 % -> 37.7 % over 1,500 pick moves, ceiling 38.8 %).
    """
    blue_idx = champs.resolve_all(blue_picks)
    red_idx = champs.resolve_all(red_picks)
    bans_idx = champs.resolve_all(bans)
    unavail_idx = champs.resolve_all(unavailable)
    _check(blue_idx, red_idx, bans_idx)

    rows = assignment(champs, _slots(blue_idx, red_idx))
    taken = set(blue_idx) | set(red_idx) | set(bans_idx) | set(unavail_idx)
    return {"model": "champ-role table",
            "blue": [{k: v for k, v in r.items() if k != "slot"}
                     for r in rows if r["side"] == "blue"],
            "red": [{k: v for k, v in r.items() if k != "slot"}
                    for r in rows if r["side"] == "red"],
            "open_roles": open_roles(rows),
            "legal_in_lane": legal_in_lane(champs, taken),
            "note": ("p_lane is that champion's own share of its picks in that "
                     "role across the pro corpus, not a prediction for this "
                     "game. A low p_lane means the champion's history is split "
                     "across roles and this placement is the best joint fit for "
                     "the team, not a confident call. The assignment is "
                     "one-to-one per side. open_roles are the roles this side "
                     "still has to fill; legal_in_lane counts draftable "
                     "champions per role.")}


# -------------------------------------------------------------- next move
def predict_next_move(predictor, champs, blue_picks=(), red_picks=(), bans=(),
                      first_pick=0, patch=None, k=8, unavailable=()):
    """The serving model's answer for this board, with its units attached.

    "Serving model" is the point: when the slot holds the 4B reranker, this
    returns the reranker's order over the head's shortlist, not the head's own
    order. `model` in the result names which one answered.

    Returns {model, kind, side, move_idx, n_legal, patch_requested,
    patch_resolved, patch_known, top: [{name, rank, p_head}], note}.
    """
    blue_idx = champs.resolve_all(blue_picks)
    red_idx = champs.resolve_all(red_picks)
    bans_idx = champs.resolve_all(bans)
    unavail_idx = champs.resolve_all(unavailable)
    _check(blue_idx, red_idx, bans_idx)
    head = _head(predictor)
    pid, resolved, known = _patch_state(predictor, patch)
    # The schema declares 1..20 but a model can send anything; a k of 0 would
    # return an empty list and read as "the model has no answer".
    k = max(1, min(20, int(k)))

    res = _serving(predictor).next_move(_slots(blue_idx, red_idx), bans_idx,
                                        first_pick=1 if first_pick else 0,
                                        k=int(k),
                                        unavailable=list(unavail_idx),
                                        patch_id=pid)
    sugg = res.get("suggestions") or []
    state = _state(res)
    legal = _num_champs(head) - len(blue_idx) - len(red_idx) - len(bans_idx) \
        - len(unavail_idx)
    if state.get("n_legal"):
        legal = int(state["n_legal"])

    top = [{"name": champs.name(int(c["idx"])), "rank": i + 1,
            "p_head": round(float(c["prob"]), 5)}
           for i, c in enumerate(sugg)]
    note = ""
    if patch is not None and not known:
        note += ("patch %s is not in the model's patch table (UNK), so this "
                 "answer is patch-agnostic." % patch)
    return {"model": label(predictor),
            "kind": state.get("kind"), "side": state.get("side"),
            "move_idx": state.get("move_idx"), "n_legal": legal,
            "patch_requested": patch, "patch_resolved": resolved,
            "patch_known": known, "top": top, "note": note}


def can_answer_in_lane(due, n_bans):
    """Whether the head can be asked a lane question about this board.

    Two conditions, both measured (`webapp/probe_roleboard_fab.py`):

    * the ban block must be complete - on a ban-phase board a pick-phase
      hypothetical has no honest form (6 bans with a 10-ban move index), and
      the two models tie there (10.8 % vs 10.1 % top-1, p = 0.81);
    * the next move must be a pick - the head has no pick to rank at a ban
      move, so `team=` would answer "which ban for that side", a different
      question wearing the same answer shape.

    The first version of this gate routed every pick move to the head, which
    was wrong for picks1: at 6 bans + 3 picks the head's top-5 is 39.9 %
    against the masked board's 45.7 %. Three picks is not enough board.
    """
    return bool(due) and due.get("kind") == "pick" and int(n_bans) >= 10


def predict_in_lane(predictor, champs, role, side=None, blue_picks=(),
                    red_picks=(), bans=(), first_pick=0, patch=None, k=5,
                    unavailable=(), due=None, masker=None):
    """The serving model's ranking *inside one role*, for either side.

    This is the query behind "what does the base model think red's jungle is",
    and behind the frontend's role board. Three answers, and the result says
    which one it gave (`method`):

    * **head** - `next_move(allowed=<champions that play the role>, team=...)`.
      The mask is applied to the LOGITS, before the top-k, so the result is a
      real ranking inside the role. Filtering an unconstrained top-k instead
      empties the list 13 % of the time, because the head's five favourites
      often do not play the lane at all.
    * **rerank** - the same call, on a slot that holds the 4B reranker: the
      lane mask still decides *which* champions are eligible (it is applied to
      the head's logits), and the adapter then orders the top twenty of them.
      So this is the one place the reranker serves a role question, and it does
      it without a role head - the role is a candidate-set constraint, exactly
      as it is for the head. What it cannot do is the ban-phase cell below,
      which needs a MASK query.
    * **head, hypothetical** - the same call with `team=` naming the side that
      is not on the clock. `move_idx` and `kind` still come from the board, so
      this is "suppose red were picking now", which is what the role board asks
      for five of its ten cells. Measured caveat: the two sides come out nearly
      identical, because the pick counts already determine whose turn it is.
    * **masker** - a masking checkpoint's role-conditioned completion. Needed
      only when the next-move model cannot be asked (`can_answer_in_lane`):
      during the ban phase. It is a different model and a different training
      condition, so the result says so rather than passing it off as the head's.

    `p_in_lane` is a share of the champions that play that role, not of the
    pool - 5 % here is not comparable with 5 % from `predict_next_move`.
    """
    lane = role_index(role)
    role_name = ROLES[int(lane)]
    blue_idx = champs.resolve_all(blue_picks)
    red_idx = champs.resolve_all(red_picks)
    bans_idx = champs.resolve_all(bans)
    unavail_idx = champs.resolve_all(unavailable)
    _check(blue_idx, red_idx, bans_idx)
    k = max(1, min(20, int(k)))
    pid, resolved, known = _patch_state(predictor, patch)

    on_clock = (due or {}).get("side")
    team = 0 if (side or on_clock or "blue") == "blue" else 1
    if side is not None and side not in ("blue", "red"):
        raise ValueError("side must be 'blue' or 'red', got %r" % (side,))
    hypothetical = bool(on_clock) and (side or on_clock) != on_clock

    base = {"role": role_name,
            "side": "blue" if team == 0 else "red",
            "hypothetical": hypothetical,
            "patch_requested": patch, "patch_resolved": resolved,
            "patch_known": known}

    if predictor is not None and can_answer_in_lane(due, len(bans_idx)):
        serving = _serving(predictor)
        pool = _lanes.legal_champions(lane)
        res = serving.next_move(_slots(blue_idx, red_idx), bans_idx,
                                first_pick=1 if first_pick else 0, k=int(k),
                                unavailable=list(unavail_idx), patch_id=pid,
                                allowed=pool, team=team)
        state = _state(res)
        # `method` names the model that answered, and the panel routes on
        # "masker". A reranker re-orders the lane-constrained shortlist the
        # head produced, so the numbers are the reranker's while the candidate
        # set is still the head's - saying "head" here would hide that.
        method = "rerank" if getattr(serving, "kind", "") == "rerank" else "head"
        top = [{"name": champs.name(int(c["idx"])), "rank": i + 1,
                "p_in_lane": round(float(c["prob"]), 5)}
               for i, c in enumerate(res.get("suggestions") or [])]
        note = ("p_in_lane is a share of the %d champions the table says play "
                "%s and that are still draftable - not of the whole pool, so "
                "do not compare it with an unconstrained call."
                % (len([c for c in pool
                        if c not in set(blue_idx) | set(red_idx)
                        | set(bans_idx) | set(unavail_idx)]), role_name))
        if hypothetical:
            note += (" This is the side that is NOT on the clock: the question "
                     "is posed as if %s were picking now, and the two sides "
                     "come out nearly identical because the pick counts "
                     "already determine whose turn it is."
                     % base["side"])
        if patch is not None and not known:
            note += (" patch %s is UNK for this model, so the answer is "
                     "patch-agnostic." % patch)
        return dict(base, model=label(predictor), method=method,
                    move_idx=state.get("move_idx"), kind=state.get("kind"),
                    top=top, note=note)

    if masker is None:
        raise ValueError(
            "this board is in its ban phase, where the next-move head cannot "
            "answer a role question - a masking checkpoint is needed for that, "
            "and none is loaded. Ask predict_next_move for the move on the "
            "clock instead, or ask about a role once the bans are done.")

    # The masking path. A slot's position within its team carries no meaning
    # (the encoder is permutation-invariant), so any empty slot of that side
    # can hold the question; its role id is what the model reads.
    picks = _slots(blue_idx, red_idx)
    free = [team * 5 + i for i in range(5) if picks[team * 5 + i] is None]
    if not free:
        raise ValueError("%s already has all five picks - there is no slot "
                         "left to ask a role question about"
                         % ("blue" if team == 0 else "red"))
    slot = free[0]
    role_ids = [ROLE_FLEX_ID] * 10
    for row in assignment(champs, picks):
        role_ids[row["slot"]] = row["role_id"]
    role_ids[slot] = lane
    logits, probs = masker.complete_slot(picks, role_ids, slot)
    illegal = set(blue_idx) | set(red_idx) | set(bans_idx) | set(unavail_idx)
    p = [0.0 if i in illegal else float(v) for i, v in enumerate(probs)]
    total = sum(p)
    if total <= 0:
        raise ValueError("the masking model put no mass on a legal champion "
                         "for %s %s" % (base["side"], role_name))
    top = []
    for row in _top(p, champs, k):
        row["p_in_lane"] = round(row.pop("p") / total, 5)
        top.append(row)
    return dict(base, model=label(masker), method="masker",
                move_idx=(due or {}).get("move_idx"), kind=(due or {}).get("kind"),
                top=top,
                note=("p_in_lane is a share of the legal champions that play "
                      "%s, renormalised after the fearless pool and the bans "
                      "were masked. This answer comes from a masking "
                      "checkpoint, not from the next-move head: the head cannot "
                      "be asked a role question during the ban phase."
                      % role_name))


# ------------------------------------------------------------------ specs
# No tool takes a board any more. The board lives in a state (`get_state`), and
# a tool that also accepted nine champion names would be a second way to say the
# same thing - which is the failure the state service exists to remove. The
# `first_pick=1 means red` inversion is now stated once, in `draft_state`.
PREDICT_SPEC = {
    "name": "predict_next_move",
    "description": (
        "The trained next-move model's answer for the board in `state`: the top "
        "k champions for the move on the clock, each with its share of the "
        "legal pool and its rank. Use this instead of guessing what pros pick. "
        "It does not explain itself, and it cannot tell blue from red.\n\n"
        "Choose k yourself. The model's top-1 is right about one time in five, "
        "its top-5 about half, and its top-20 about 85 % of the time, so k=1 "
        "throws away half the answers and k=20 is the widest read that still "
        "pays for itself. Read a narrow list when the board is ordinary and a "
        "wide one when you expect a flex or a counter.\n\n"
        "Pass `role` to narrow the question to one role - 'what does the model "
        "think red's jungle is' - and `side` when you mean a side that is not "
        "on the clock. The numbers then count only champions that play that "
        "role, so they are NOT comparable with an unconstrained call."),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {"type": "string", "default": "real",
                      "description": "the draft state to answer for; `real` is "
                                     "the live board"},
            "k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20,
                  "description": "how many candidates to return (1-20)"},
            "role": {
                "type": "string",
                "enum": list(ROLES),
                "description": "restrict the answer to champions that play this "
                               "role. The result says whether the next-move "
                               "head or a masking checkpoint answered."},
            "side": {
                "type": "string", "enum": ["blue", "red"],
                "description": "which side to answer for. Default: the side on "
                               "the clock. Naming the other side is a "
                               "hypothetical and only works on a pick move."},
        },
        "required": ["state"],
    },
}

ROLES_SPEC = {
    "name": "predict_roles",
    "description": (
        "Which role each pick fills, how sure that is, and which roles a side "
        "still has to fill. The role comes from each champion's own pick "
        "history in the pro corpus (96 % agreement with the real roles), not "
        "from a prediction about this game; `p_lane` is that champion's share "
        "of its own picks in that role, so 0.98 is a dedicated laner and 0.45 "
        "is a flex. The assignment is one-to-one per side, so two flex "
        "champions are split rather than both claiming the same role. Call it "
        "before recommending a champion: it tells you what the side still "
        "needs, and `legal_in_lane` says how many champions can actually fill "
        "each open role."),
    "parameters": {
        "type": "object",
        "properties": {"state": {"type": "string", "default": "real"}},
        "required": ["state"],
    },
}
