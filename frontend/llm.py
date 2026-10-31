"""Grounded explanation for a draft recommendation (plan-model-v2.md P3.3).

Three parts, each usable without a network:

  ProIndex   what pros actually did from a board like this one
  bullets()  what the patch notes say about the candidates (optional: the
             corpus is local-only, and an absent corpus simply means no patch
             evidence - see `.gitignore` and STATUS.md P3.1 for why)
  explain()  prompt -> LLM -> validated answer, cached on disk

The rule that keeps this honest: **every claim must cite evidence that
resolves.** A game citation must be a gameid in the index; a patch citation
must be a line that appears verbatim in the notes we handed the model. Bad
citations are dropped and the call is retried once with the problems fed
back; if it still fails, the answer comes back with `unverified` populated
rather than quietly shipping invented evidence.

Why retrieval rather than the model's own judgement: the policy network
knows a distribution over moves, not why. The corpus knows what strong teams
did when they saw this board, and the patch notes know what changed. Those
are the two things a coach cannot get from the probability list, and neither
requires training - which is why this is the one P3 item that cannot be
invalidated by a metric the way the attribute table was.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

N_MOVES = 20
ROLE_NAMES = ["TOP", "JGL", "MID", "BOT", "SUP"]
PHASE_NAMES = ["bans1", "picks1", "bans2", "picks2"]
PRO_DRAFTS = ROOT / "data" / "pro play" / "pro_drafts.pt"
PATCH_DIR = ROOT / "data" / "patch notes" / "clean"
CACHE_DIR = HERE / ".cache" / "explain"

# Bump when the prompt or the answer schema changes: the cache key includes
# it, so old answers cannot be served against a new contract.
PROMPT_VERSION = "explain-1"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_BULLET_CHARS = 700


def phase_of(move_idx):
    return 0 if move_idx < 6 else 1 if move_idx < 12 else 2 if move_idx < 16 else 3


def patch_key(patch):
    """"16.20" -> (16, 20) for ordering; None if it is not a patch number."""
    try:
        major, minor = str(patch).strip().split(".")[:2]
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def _norm(text):
    """Whitespace-insensitive comparison key for quoted patch lines."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


# --------------------------------------------------------------- retrieval
class ProIndex:
    """Nearest historical boards, with the move that was played from them.

    Two stages, for a reason. Scanning every row's prefix at every query is
    46k x n set builds; instead each row keeps its *final* draft as one
    173-bit int, so the coarse pass is a single `int.bit_count()` sweep
    (~20 ms for 46k rows, no numpy). Only the top `coarse` rows are then
    re-ranked on the champions that were already on the board at the query
    move - which is the number that matters: a row that shares champions
    with us *later* is not evidence about this decision.

    Rows are filtered to the same move index and the same first-pick side.
    If that leaves too few, the side filter is dropped and the result says
    so, because a weak match from the other side is still worth showing -
    labelled - rather than returning nothing.
    """

    def __init__(self, rows):
        self.rows = rows
        self.masks = []
        for row in rows:
            mask = 0
            for _kind, _team, champ, _role in row["moves"]:
                mask |= 1 << int(champ)
            self.masks.append(mask)
        self.n = len(rows)
        # Newest patch in the corpus: the default for "which notes apply"
        # when the caller does not name one.
        self.latest_patch = None
        best = (-1, -1)
        for row in rows:
            key = patch_key(row.get("patch"))
            if key and key > best:
                best, self.latest_patch = key, row.get("patch")

    @classmethod
    def load(cls, path=None):
        import torch
        payload = torch.load(str(path or PRO_DRAFTS), map_location="cpu",
                             weights_only=False)
        rows = payload["rows"] if isinstance(payload, dict) else payload
        return cls(rows)

    @staticmethod
    def _prefix_champs(row, n_moves):
        return {int(c) for _k, _t, c, _r in row["moves"][:n_moves]}

    def similar(self, champs_so_far, move_idx, first_pick, k=5, coarse=600,
                patch=None):
        """-> (games best-first, relaxed_side). Empty list if nothing matches.

        Ranking is overlap first, then patch proximity, then recency. Overlap
        has to dominate - a same-patch board that shares nothing is not
        evidence - but on equal overlap an eight-patch-old game is worse
        evidence than a recent one, because the meta moved, so patch distance
        breaks the tie before date does.
        """
        query = 0
        for c in champs_so_far:
            query |= 1 << int(c)
        # Coarse: overlap with the row's whole draft. Cheap, and only used to
        # decide which rows deserve the exact comparison below.
        scored = []
        for i, mask in enumerate(self.masks):
            ov = (mask & query).bit_count()
            if ov:
                scored.append((ov, i))
        if not scored:
            return [], False
        scored.sort(reverse=True)
        shortlist = [i for _ov, i in scored[:coarse]]
        wanted = set(int(c) for c in champs_so_far)
        q_patch = patch_key(patch) if patch else None

        def rank(indices):
            out = []
            for i in indices:
                row = self.rows[i]
                if len(row["moves"]) != N_MOVES:
                    continue
                prefix = self._prefix_champs(row, move_idx)
                inter = len(prefix & wanted)
                union = len(prefix | wanted) or 1
                r_patch = patch_key(row.get("patch"))
                gap = (0 if q_patch is None or r_patch is None
                       else abs(r_patch[0] - q_patch[0]) * 100
                       + abs(r_patch[1] - q_patch[1]))
                out.append((inter / union, -gap,
                            int(row.get("date_ordinal") or 0), i))
            out.sort(reverse=True)
            return out

        same_side = [i for i in shortlist
                     if int(self.rows[i]["first_pick"]) == int(first_pick)]
        ranked, relaxed = rank(same_side), False
        if len(ranked) < k:
            ranked, relaxed = rank(shortlist), True
        return ([self._describe(i, move_idx, score)
                 for score, _gap, _d, i in ranked[:k]], relaxed)

    def _describe(self, i, move_idx, score):
        row = self.rows[i]
        kind, team, champ, role = row["moves"][move_idx]
        date = row.get("date")
        return {
            "gameid": row.get("gameid"),
            "patch": row.get("patch"),
            "league": row.get("league"),
            "date": str(date)[:10] if date is not None else None,
            "blue_team": row.get("blue_team"),
            "red_team": row.get("red_team"),
            "blue_win": (None if row.get("blue_win") is None
                         else int(row["blue_win"])),
            "overlap": round(float(score), 3),
            "next_kind": "ban" if int(kind) == 0 else "pick",
            "next_side": "blue" if int(team) == 0 else "red",
            "next_champ_idx": int(champ),
            "next_role": (None if int(kind) == 0 else int(role)),
        }


# ------------------------------------------------------------ patch notes
def _notes_path(patch):
    """'16.20' -> clean/patch_16_20.md. None if the corpus lacks it."""
    if not patch:
        return None
    name = "patch_%s.md" % str(patch).strip().replace(".", "_")
    path = PATCH_DIR / name
    return path if path.exists() else None


def _sections(text):
    """Split a clean patch-notes file into {normalised champion: body}."""
    out = {}
    parts = re.split(r"^####\s+", text, flags=re.M)
    for part in parts[1:]:
        lines = part.splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if body:
            out[_norm(name)] = (name, body)
    return out


def bullets(patch, champ_names, max_chars=MAX_BULLET_CHARS):
    """-> ({name: text}, missing) for the champions named in this patch."""
    path = _notes_path(patch)
    if path is None:
        return {}, list(champ_names)
    sections = _sections(path.read_text(encoding="utf-8"))
    found, missing = {}, []
    for name in champ_names:
        hit = sections.get(_norm(name))
        if hit is None:
            missing.append(name)
            continue
        body = hit[1]
        if len(body) > max_chars:
            body = body[:max_chars].rsplit("\n", 1)[0] + " ..."
        found[name] = body
    return found, missing


# ---------------------------------------------------------------- the prompt
SYSTEM = """You are a League of Legends draft analyst explaining a \
recommendation to a coach.

Rules, in order of importance:
1. Cite only the evidence you are given. Never invent a game, a team, a \
patch change or a statistic. If the evidence does not support a claim, say so.
2. Every item in "evidence" must be copied from the inputs: a game id from \
RETRIEVED GAMES, or a line copied verbatim from PATCH NOTES.
3. You are explaining the model's recommendation, not replacing it. If you \
disagree, put that in "risk" - do not change "recommend".
4. Be concrete about mechanism (what the pick does to the draft), not about \
tier lists.
5. Reply with JSON only, no prose around it.

Schema:
{"recommend": "<one of the CANDIDATES>",
 "why": "<at most 3 sentences>",
 "their_answer": "<the opponent's most likely response and why>",
 "risk": "<the main way this goes wrong>",
 "evidence": [{"type": "game", "id": "<gameid>"},
              {"type": "patch", "line": "<verbatim line>"}]}"""


def build_prompt(board, suggestions, games, notes, notes_available=True):
    """board: dict from the server; suggestions: [(name, prob)].

    `notes_available` separates "this patch has no notes in the corpus" from
    "the notes were read and these champions were not changed". They are
    different claims, and a model told the first one will hedge where the
    second one should make it stop talking about patch changes.
    """
    lines = ["PATCH: %s" % (board.get("patch") or "unknown"),
             "MOVE: %d of %d (%s), %s to move, first pick: %s"
             % (board["move_idx"], N_MOVES,
                PHASE_NAMES[phase_of(board["move_idx"])],
                board["side"], board.get("first_pick_side", "?")),
             "BANS: %s" % (", ".join(board.get("bans") or []) or "none"),
             "PICKS: blue %s | red %s"
             % (", ".join(board.get("blue") or []) or "none",
                ", ".join(board.get("red") or []) or "none"),
             "",
             "CANDIDATES (model probability):",
             ]
    for name, prob in suggestions:
        lines.append("  - %s  %.1f%%" % (name, 100.0 * prob))
    lines += ["", "RETRIEVED GAMES (same board size, same first-pick side, "
                  "most similar first):"]
    if not games:
        lines.append("  (none - the corpus has no comparable board)")
    for g in games:
        who = "blue" if g["next_side"] == "blue" else "red"
        role = (" as %s" % ROLE_NAMES[g["next_role"]]
                if g.get("next_role") is not None else "")
        result = ("blue won" if g["blue_win"] else "red won"
                  if g["blue_win"] is not None else "result unknown")
        lines.append("  - %s | patch %s | %s %s | %s vs %s | next: %s %s %s%s "
                     "| %s | overlap %.2f"
                     % (g["gameid"], g["patch"], g["league"] or "?",
                        g["date"] or "?", g["blue_team"], g["red_team"],
                        who, g["next_kind"], g["next_champ"], role,
                        result, g["overlap"]))
    lines += ["", "PATCH NOTES (champion changes this patch):"]
    if not notes:
        lines.append("  (no champion-level changes for the candidates on this "
                     "patch)" if notes_available else
                     "  (the corpus has no notes for this patch - do not "
                     "claim anything about patch changes)")
    for name, text in notes.items():
        lines.append("  [%s]\n%s" % (name, text))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]


# ------------------------------------------------------------- validation
def _strip_fences(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def validate(answer, games, notes, candidate_names):
    """-> (clean_answer, problems). Bad evidence is dropped, not shipped."""
    problems = []
    if not isinstance(answer, dict):
        return None, ["answer is not a JSON object"]
    by_name = {_norm(n): n for n in candidate_names}
    rec = answer.get("recommend")
    if _norm(rec) not in by_name:
        problems.append("recommend %r is not one of the candidates" % rec)
        rec = None
    else:
        rec = by_name[_norm(rec)]
    game_ids = {str(g["gameid"]) for g in games}
    note_text = _norm(" ".join(notes.values()))
    kept, unverified = [], []
    for item in answer.get("evidence") or []:
        if not isinstance(item, dict):
            unverified.append(item)
            continue
        kind = str(item.get("type", "")).lower()
        if kind == "game":
            gid = str(item.get("id", ""))
            if gid in game_ids:
                kept.append({"type": "game", "id": gid})
            else:
                unverified.append(item)
                problems.append("game %r was not in the retrieved set" % gid)
        elif kind == "patch":
            line = item.get("line", "")
            if len(_norm(line)) >= 12 and _norm(line) in note_text:
                kept.append({"type": "patch", "line": line})
            else:
                unverified.append(item)
                problems.append("patch line not found in the notes: %.60r"
                                % line)
        else:
            unverified.append(item)
            problems.append("unknown evidence type %r" % kind)
    clean = {"recommend": rec,
             "why": (answer.get("why") or "").strip(),
             "their_answer": (answer.get("their_answer") or "").strip(),
             "risk": (answer.get("risk") or "").strip(),
             "evidence": kept,
             "unverified": unverified}
    if not clean["why"]:
        problems.append("no explanation given")
    return clean, problems


# ------------------------------------------------------------------ client
def client_from_env(env=None):
    """-> callable(messages) -> str, or None if no key is configured.

    An OpenAI-compatible chat-completions endpoint, called through urllib so
    the frontend gains no dependency. Blocking: the server is threaded, so a
    slow call costs one request, not the process.
    """
    env = os.environ if env is None else env
    key = (env.get("LLM_API_KEY") or "").strip()
    if not key:
        return None
    base = (env.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = env.get("LLM_MODEL") or DEFAULT_MODEL
    timeout = float(env.get("LLM_TIMEOUT") or 45)

    def call(messages):
        body = json.dumps({"model": model, "messages": messages,
                           "temperature": 0.2,
                           "response_format": {"type": "json_object"}}
                          ).encode("utf-8")
        req = urllib.request.Request(
            base + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]

    call.model = model
    return call


# ------------------------------------------------------------------- cache
def _cache_key(payload):
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


def _cache_read(key):
    path = CACHE_DIR / (key + ".json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_write(key, value):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / (key + ".json")).write_text(
            json.dumps(value, indent=2), encoding="utf-8")
    except OSError:
        pass          # a cache miss is not worth failing the request over


# ------------------------------------------------------------------ public
def explain(board, suggestions, index, client=None, k_games=5, use_cache=True,
            force=False, names=None):
    """Retrieve, prompt, validate, cache.

    `board` carries move_idx, side, first_pick_side, patch, bans, blue, red
    and `champ_ids` (the champion ids on the board, for retrieval).
    `names` maps champion idx -> display name; without it the retrieved games
    report ids, which is fine for tests and useless for a prompt.
    Returns dict with `retrieval`, `patch`, `explanation` (or None), `llm`.
    """
    games, relaxed = index.similar(
        board.get("champ_ids") or [], board["move_idx"],
        board.get("first_pick", 0), k=k_games, patch=board.get("patch"))
    for g in games:
        idx = g.pop("next_champ_idx")
        g["next_champ"] = names[idx] if names and idx < len(names) else idx
    top = [n for n, _p in suggestions[:3]]
    notes, missing = bullets(board.get("patch"), top)
    available = _notes_path(board.get("patch")) is not None

    out = {"retrieval": {"games": games, "relaxed_side": relaxed,
                         "n_indexed": index.n},
           "patch": {"patch": board.get("patch"), "notes": notes,
                     "missing": missing, "available": available},
           "explanation": None,
           "llm": {"configured": client is not None,
                   "model": getattr(client, "model", None)}}
    if client is None:
        out["llm"]["reason"] = ("set LLM_API_KEY (and optionally "
                                "LLM_BASE_URL / LLM_MODEL) to get prose; the "
                                "retrieved evidence is returned regardless")
        return out

    key = _cache_key({"v": PROMPT_VERSION,
                      "model": getattr(client, "model", None),
                      "board": {k: board.get(k) for k in
                                ("move_idx", "side", "first_pick", "patch",
                                 "bans", "blue", "red")},
                      "cands": [(n, round(float(p), 4)) for n, p in suggestions],
                      "games": [g["gameid"] for g in games]})
    if use_cache and not force:
        hit = _cache_read(key)
        if hit is not None:
            hit["cached"] = True
            out["explanation"] = hit
            return out

    messages = build_prompt(board, suggestions, games, notes,
                            notes_available=available)
    names = [n for n, _p in suggestions]
    problems, clean = ["no attempt"], None
    for attempt in range(2):
        if attempt:                      # feed the failures back, once
            messages = messages + [
                {"role": "user",
                 "content": "That answer was rejected: %s\nFix it and reply "
                            "with JSON only." % "; ".join(problems[:4])}]
        try:
            raw = client(messages)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            out["llm"]["error"] = "call failed: %s" % exc
            return out
        clean, problems = validate(_strip_fences(raw), games, notes, names)
        if clean is not None and not problems:
            break
    if clean is None:
        out["llm"]["error"] = "unparseable answer: %s" % "; ".join(problems[:3])
        return out
    clean["model"] = getattr(client, "model", None)
    clean["cached"] = False
    clean["ok"] = not problems
    out["explanation"] = clean
    if use_cache:
        _cache_write(key, clean)
    return out
