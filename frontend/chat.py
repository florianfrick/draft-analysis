"""The chat surface: a message in, a grounded recommendation out.

The design is the inverted loop, and it exists because of what the model
measured badly at. Free generation from the adapter is 7-8 % top-1 with exactly
one name and no stop token (`probe_roleboard_gen.py`); selection from a
shortlist is 33.85 % and the format does most of that work. So the chat does not
ask the model to *recall* anything. The server computes the legal pool, the
head's distribution over it, and the corpus evidence; the model - or, by
default, a template - only has to say which of those numbers matters.

**Insistence enters as a prior, not as a rewrite.** A user who says "Kled is
incredibly good here" is not asking for the ranking to be replaced, they are
adding evidence the head does not have. So it is added in log space:

    log p'(c) = log p_head(c) + lambda * 1[c == named]

which is a likelihood ratio: `lambda` is how many nats of prior the user is
contributing. Renormalising afterwards means the boost has to come out of the
other candidates, which is the behaviour you want - a pin on Kled should cost
Xayah its probability, not create mass.

`lambda` is therefore *comparable across messages*, and the whole point of
scoring it is that "incredibly good" and "alright" must not produce the same
answer. They will not: the ladder below spans 0.5 to 10 nats, and a champion
sitting at rank 47 needs about 7.6 nats to reach the top of a 173-champion
distribution.

The ladder is uncalibrated - nobody has asked two coaches to label a corpus of
utterances - so every reply states the lambda it used and the rank the named
champion moved from and to. That makes the guess auditable instead of hidden:
if the reply reads wrong, the number that produced it is on screen.
"""
import math
import re

import tools

# How many nats of prior each kind of utterance is worth. Ordered strongest
# first; the first rung whose cue appears in the clause wins.
#
# These are a ladder, not measurements. The only thing asserted is the order and
# the rough spacing: a "must" has to beat a "maybe" by enough to change the
# answer, and 10 nats moves a champion from anywhere in the distribution to the
# top, which is what "I insist" means.
LADDER = (
    ("pin", 15.0, ("must", "insist", "have to", "has to", "no matter what",
                   "100%", "definitely", "for sure", "lock in", "locked")),
    ("strong", 6.0, ("incredibly", "extremely", "very good", "really good",
                     "great", "amazing", "best", "clearly", "obvious")),
    ("normal", 3.0, ("i think", "i'd", "i would", "prefer", "like", "good",
                     "should", "want", "rate")),
    ("weak", 1.5, ("alright", "okay", "ok", "fine", "maybe", "consider",
                   "could", "might", "decent")),
)
MENTION_LAM = 0.5

# Negation flips the sign: "Kled is bad here" is evidence *against*, which is the
# same mechanism with lambda < 0 and is strictly more useful than refusing it.
# It gets its own magnitude ladder for the same reason the positive one exists -
# "he is bad" and "don't pick him" are not equally strong claims.
NEGATIVE_STRONG = 6.0
NEGATIVE_WEAK = 3.0
NEGATIVE = ("bad", "weak", "poor", "don't", "dont", "do not", "avoid", "hate",
            "not ", "never", "instead", "against", "worse", "terrible", "awful")
NEGATIVE_CUES = ("bad", "poor", "hate", "avoid", "never", "worse", "terrible",
                 "awful")

# Names that a coach types and dDragon does not use. Kept deliberately short:
# every entry is a guess about how people talk, so it earns its place only when
# the real name would not match.
ALIASES = {
    "mundo": "Dr. Mundo",
    "nunu": "Nunu & Willump",
    "willump": "Nunu & Willump",
    "tf": "Twisted Fate",
    "ww": "Warwick",
    "j4": "Jarvan IV",
    "jarvan": "Jarvan IV",
    "asol": "Aurelion Sol",
    "kog": "Kog'Maw",
    "renata": "Renata Glasc",
    "tahm": "Tahm Kench",
}

_CLAUSE = re.compile(r"[.,;!?\n]+")
# The only champion name containing a full stop, and the only reason clause
# splitting needs a special case at all: "Dr. Mundo" would otherwise split into
# "Dr" and "Mundo" and lose whatever qualifies it.
_DR_MUNDO = re.compile(r"\bdr\.?\s*mundo\b", re.I)


def _norm(text):
    """Lowercase, and every run of non-alphanumerics becomes one space.

    So `Kai'Sa`, `Kaisa` and `kai sa` all become a key this module can match.
    """
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def name_index(names):
    """-> {normalised key: champion idx}, longest key first on lookup.

    Both the spaced and the run-together form are registered, because
    `Kai'Sa` is typed as `kaisa` about as often as `kai sa`, and `Dr. Mundo`
    as `dr mundo` or `mundo`.
    """
    out = {}
    for i, name in enumerate(names):
        key = _norm(name)
        if not key:
            continue
        out.setdefault(key, i)
        out.setdefault(key.replace(" ", ""), i)
    for alias, name in ALIASES.items():
        key = _norm(name)
        if key in out:
            out.setdefault(alias, out[key])
    return out


def parse(message, names):
    """-> {"mentions": [...], "side": None|0|1, "complete": bool}

    A keyword scan, not NLU: the clause that contains a champion name also
    contains the words that qualify it. That is wrong for "I don't think X is
    bad" and right for everything a coach actually types, and it is cheap enough
    to run on every keystroke-length message.
    """
    index = name_index(names)
    keys = sorted(index, key=len, reverse=True)
    side = None
    low = _norm(message)
    if re.search(r"\bfor (the )?blue\b|\bblue side\b|\bon blue\b", low):
        side = 0
    elif re.search(r"\bfor (the )?red\b|\bred side\b|\bon red\b", low):
        side = 1

    mentions = []
    whole = _norm(_DR_MUNDO.sub("Dr Mundo", message or ""))
    for clause in _CLAUSE.split(_DR_MUNDO.sub("Dr Mundo", message or "")):
        norm = _norm(clause)
        if not norm:
            continue
        found, consumed = [], set()
        for key in keys:
            if not key or key in consumed:
                continue
            pattern = r"\b" + re.escape(key) + r"\b"
            hit = re.search(pattern, norm)
            if hit:
                idx = index[key]
                if idx not in [f[1] for f in found]:
                    found.append((hit.start(), idx, names[idx]))
                consumed.add(key)
        if not found:
            continue
        # Message order, not key length: the first name the user said is the one
        # the recommendation should be steered by.
        found.sort()
        label, lam = _stance(norm, whole)
        neg = _negation(norm) or _negation(whole)
        if neg is not None:
            lam = -max(neg, abs(lam) if label != "mention" else 0.0)
            label = "against-" + label
        for _pos, idx, name in found:
            mentions.append({"idx": idx, "name": name, "stance": label,
                             "lam": lam})
    return {"mentions": mentions, "side": side, "raw": message,
            "empty": not (message or "").strip()}


def _stance(norm, whole=""):
    """The strongest cue in the clause; the message's cue if the clause has none.

    The fallback matters because coaches write "don't pick Kled, he is bad" -
    the qualifier lands in the second clause, and without it the first clause's
    mention would be scored as a bare mention.
    """
    for label, lam, cues in LADDER:
        if any(cue in norm for cue in cues):
            return label, lam
    if whole and whole != norm:
        for label, lam, cues in LADDER:
            if any(cue in whole for cue in cues):
                return label, lam
    return "mention", MENTION_LAM


def _negation(norm):
    """-> how many nats of evidence *against*, or None."""
    if not any(neg in norm for neg in NEGATIVE):
        return None
    return (NEGATIVE_STRONG if any(c in norm for c in NEGATIVE_CUES)
            else NEGATIVE_WEAK)


# ------------------------------------------------------------- the loop
def invert(head_probs, board, idx, lam):
    """-> the head's ranking, the ranking after `lam` nats on `idx`, and both
    positions. Pure: no model, no corpus, so it is the part worth testing.

    `head_probs` is the head's softmax over all champions for this move. It is
    renormalised over the legal set first, because the head does not hard-mask
    legality - it learns it - so an illegal champion can carry real mass and
    would otherwise win a ranking it cannot be picked in.
    """
    legal = board.legal()
    if not legal:
        raise ValueError("no legal champion: the board is full")
    total = sum(float(head_probs.get(c, 0.0)) for c in legal)
    if total <= 0:
        raise ValueError("the head put no mass on any legal champion")
    p = {c: float(head_probs.get(c, 0.0)) / total for c in legal}

    before_rank = _rank(p, idx)
    before_p = p.get(idx) if idx is not None else None
    # How many nats it would take to make `idx` the pick. Reported so the reply
    # can say what the wording was worth instead of silently landing somewhere:
    # the ladder is a guess, but the gap it is being compared against is exact.
    lam_needed = None
    if idx is not None and idx in p and len(p) > 1:
        top_other = max((v for c, v in p.items() if c != idx), default=0.0)
        if p[idx] > 0 and top_other > 0:
            lam_needed = math.log(top_other / p[idx])
    if idx is not None and lam and idx in p:
        p[idx] = p[idx] * math.exp(lam)
        z = sum(p.values())
        p = {c: v / z for c, v in p.items()}
    after_rank = _rank(p, idx)

    ranking = sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"p": p, "ranking": ranking,
            "before_rank": before_rank, "after_rank": after_rank,
            "before_p": before_p, "after_p": p.get(idx) if idx is not None
            else None,
            "n_legal": len(legal), "lam": float(lam or 0.0),
            "lam_needed": lam_needed,
            "moved": (before_rank != after_rank)}


def _rank(p, idx):
    if idx is None or idx not in p:
        return None
    return 1 + sum(1 for v in p.values() if v > p[idx])


# ---------------------------------------------------------------- replies
def _pct(x):
    return "%.2f %%" % (100.0 * x) if x is not None else "n/a"


def _times(n):
    return "once" if n == 1 else "%d times" % n


def compose_local(board, names, parsed, inv, evidence):
    """The default reply: the numbers, in the order a coach would ask for them.

    Deliberately not prose for its own sake. Every clause is one tool result,
    and the thin ones carry their `n` so the reader can discount them.
    """
    lines = []
    if board.complete:
        return ("The draft is complete - all twenty moves are on the board, so "
                "there is no next move to rank.")
    pool = evidence.get("legal_pool")
    counts = evidence.get("next_move_counts")
    top = inv["ranking"][:3]

    if pool and pool["result"]["top"]:
        best = pool["result"]["top"][0]
        lines.append("The head's pick is %s (%.1f %% of the legal mass), then %s."
                     % (best["name"], 100.0 * best["prob"],
                        ", ".join(c["name"] for c in pool["result"]["top"][1:3])
                        or "nothing close"))

    if counts and counts["result"]["top"]:
        c = counts["result"]["top"]
        head = ", ".join("%s %d" % (x["name"], x["n"]) for x in c[:3])
        lines.append("The %d nearest boards played %s."
                     % (counts["result"]["n_boards"], head))
        if not counts["reliable"]:
            lines.append("That leader is %d games - a count, not a rate."
                         % c[0]["n"])

    m = parsed["mentions"][0] if parsed["mentions"] else None
    if m:
        idx, name, lam = m["idx"], m["name"], m["lam"]
        if idx not in inv["p"]:
            lines.append("%s is already off the board, so there is nothing to "
                         "%s." % (name, "boost" if lam > 0 else "suppress"))
        else:
            lines.append("%s sits at rank %d of %d (%s) for the head; reading "
                         "that as %+.1f nats moves it to rank %d (%s)."
                         % (name, inv["before_rank"], inv["n_legal"],
                            _pct(inv["before_p"]), lam, inv["after_rank"],
                            _pct(inv["after_p"])))
            if lam > 0 and inv["after_rank"] == 1:
                lines.append("That is enough to make it the pick.")
            elif lam > 0 and inv["lam_needed"] is not None:
                lines.append("It would take %.1f nats to make it the pick."
                             % inv["lam_needed"])
            elif lam < 0 and inv["after_rank"] not in (None, inv["n_legal"]):
                lines.append("To push it off the list entirely would take more "
                             "than this.")
        if parsed["mentions"][1:]:
            other = parsed["mentions"][1]
            pair = evidence.get("pair_evidence")
            if pair:
                cc = pair["result"]["counts"]
                lines.append("%s and %s have been on the same team %s, "
                             "opposing %s, and facing each other in lane %s."
                             % (name, other["name"], _times(cc["same_team"]),
                                _times(cc["opposing"]),
                                _times(cc["same_role"])))

    if parsed["side"] is not None:
        lines.append("Note: the model answers for the side on the clock, and "
                     "does not differentiate the two sides "
                     "(TV 0.015 vs 0.147 for a real swap), so a request framed "
                     "for the other side changes nothing.")
    return " ".join(lines)


def respond(corpus, board, head_probs, names, message, client=None,
            k_similar=tools.SIMILAR_DEFAULT):
    """The whole chat turn: parse, gather evidence, invert, reply.

    Returns a dict the UI can render without re-deriving anything:
    `reply`, `ranking`, `pick`, `mentions`, `evidence`, `tools_used`.
    """
    parsed = parse(message, names)
    evidence, used = {}, []

    def take(name, **kw):
        try:
            res = tools.run(corpus, board, name, names, probs=head_probs, **kw)
        except ValueError:
            return None
        evidence[name] = res
        used.append(name)
        return res

    if not board.complete:
        take("legal_pool", k=5)
        take("next_move_counts", k=k_similar, top=5)

    primary = parsed["mentions"][0] if parsed["mentions"] else None
    if primary and not board.complete:
        take("champ_evidence", idx=primary["idx"], k=k_similar)
    if len(parsed["mentions"]) > 1:
        a, b = parsed["mentions"][0], parsed["mentions"][1]
        take("pair_evidence", a=a["idx"], b=b["idx"])

    if board.complete:
        return {"reply": compose_local(board, names, parsed, _empty_inv(), {}),
                "ranking": [], "pick": None, "mentions": parsed["mentions"],
                "evidence": {}, "tools_used": [], "parsed": parsed}

    inv = invert(head_probs, board,
                 primary["idx"] if primary else None,
                 primary["lam"] if primary else 0.0)
    reply = compose_local(board, names, parsed, inv, evidence)
    source = "local"
    if client is not None:
        text = _compose_llm(client, board, names, parsed, inv, evidence)
        if text:
            reply, source = text, getattr(client, "model", "api")

    ranking = [{"idx": c, "name": names[c], "prob": round(p, 5)}
               for c, p in inv["ranking"][:8]]
    return {"reply": reply, "source": source, "ranking": ranking,
            "pick": ranking[0] if ranking else None,
            "mentions": parsed["mentions"], "evidence": evidence,
            "tools_used": used, "parsed": parsed,
            "lambda": inv["lam"],
            "moved": {"name": primary["name"] if primary else None,
                      "before_rank": inv["before_rank"],
                      "after_rank": inv["after_rank"],
                      "before_p": inv["before_p"],
                      "after_p": inv["after_p"],
                      "lam_needed": inv["lam_needed"]} if primary else None}


def _empty_inv():
    return {"ranking": [], "p": {}, "before_rank": None, "after_rank": None,
            "before_p": None, "after_p": None, "n_legal": 0, "lam": 0.0,
            "lam_needed": None, "moved": False}


LLM_SYSTEM = """You are a League of Legends draft coach answering a question \
about one board. You are given the model's numbers and the corpus evidence as \
JSON.

Rules, in order:
1. Cite only the numbers in the input. Never invent a game, a team, a patch \
change or a statistic.
2. Where a count is below 30, say "n=" and treat it as an anecdote, not a rate.
3. You are explaining a ranking, not replacing it. If you disagree, say so in \
one clause and leave the ranking alone.
4. At most 90 words.
5. Reply with JSON only: {"reply": "..."}"""


def _compose_llm(client, board, names, parsed, inv, evidence):
    """Ask a remote model to phrase the same numbers. Any failure -> None.

    The local composer is the shipped path and this is the optional one, so it
    is not allowed to break a turn: a timeout, a refusal, a JSON error and a
    hallucinated number all fall back to the template.
    """
    import json
    ctx = {
        "move_idx": board.move_idx,
        "first_pick": board.first_pick,
        "patch": board.patch,
        "message": parsed.get("raw"),
        "mentions": [{"name": m["name"], "lambda": m["lam"],
                      "stance": m["stance"]} for m in parsed["mentions"]],
        "ranking": [{"name": names[c], "prob": round(p, 5)}
                    for c, p in inv["ranking"][:5]],
        "evidence": {k: {"n": v["n"], "reliable": v["reliable"],
                         "note": v["note"], "result": v["result"]}
                     for k, v in evidence.items()},
    }
    try:
        raw = client([{"role": "system", "content": LLM_SYSTEM},
                      {"role": "user", "content": json.dumps(ctx, default=str)}])
        text = json.loads(raw).get("reply")
        return text.strip() if isinstance(text, str) and text.strip() else None
    except Exception:  # noqa: BLE001 - the template is always a valid answer
        return None
