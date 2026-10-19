"""Local-only prototype frontend for the SoloQ draft model.

Run:    python frontend/server.py --port 8777
Open:   http://127.0.0.1:8777   (binds 127.0.0.1 only - no external access)

UI: 5 ban + 5 pick slots per team (horizontal, role-free; the next pro
pick/ban to move is HIGHLIGHTED, never enforced - sequence follows
model/README.md draft format, mirrored when red has first pick), the
model's likely role assignments for the picked champs (role head on a
FLEX-role query), role-conditioned top-5 suggestions for all 5 roles per
team (flex picks flagged), dDragon champion art
(https://ddragon.leagueoflegends.com/cdn/16.20.1/img/champion/... via the
manifest, cached, deterministic name fallback offline), and a walkthrough
of real pro games (GET /api/drafts + GET /api/draft) showing what the
model suggested vs what the pros actually moved.

Checkpoints - BOTH families load:
  * pretraining  model/pretraining/ckpt/run_*/*.pt (default: the best.pt
    with highest val champ_top5)
  * pro finetune model/finetune/ckpt/run_*/*.pt (best_mask/best_win/
    latest). Those payloads wrap the shared encoder under `encoder.` plus
    win/econ outcome heads under `heads.`; we strip the prefix, drop the
    heads (no MCTS here) and serve suggestions from the finetuned encoder.
Select a model via --ckpt <path> (CLI), the header dropdown
(GET /api/checkpoints + POST /api/load), or the "From file..." button
(POST /api/upload -> saved under model/uploaded/).

Inference contract (model/README.md): every empty slot is a MASK query
with its ROLE GIVEN (role-conditional, matches the 10% role-visible
training condition); no PADs ever, so a team is never fully padded.
Bans are NOT model inputs - applied as a post-filter (the model already
-inf's already-picked visible champs). suggest() runs 1 FLEX-role pass
(role head over the picked champs) plus <=5 role-conditioned MASK passes
(every role queried for both teams); pick-order slots are fine - the
encoder is permutation-invariant within a team.
"""
import argparse
import glob
import json
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import torch

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT = FRONTEND_DIR.parent                     # .../lol draft analysis
PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"
MODEL_DIR = ROOT / "model"
STATIC_DIR = FRONTEND_DIR / "static"
CKPT_GLOB = str(ROOT / "model" / "pretraining" / "ckpt" / "run_*" / "best.pt")
CKPT_PATTERNS = (                 # what the dropdown offers (no torch.load)
    ("pretrain", str(ROOT / "model" / "pretraining" / "ckpt"
                     / "run_*" / "*.pt")),
    ("finetune", str(ROOT / "model" / "finetune" / "ckpt" / "run_*" / "*.pt")),
    ("uploaded", str(ROOT / "model" / "uploaded" / "*.pt")),
)
UPLOAD_DIR = ROOT / "model" / "uploaded"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024      # generous; ckpts are 20-150 MB
LOAD_LOCK = threading.Lock()              # serialize runtime model swaps
# dDragon static champion art (model patch 16.20)
DDRAGON_VERSION = "16.20.1"
DDRAGON_CDN = "https://ddragon.leagueoflegends.com/cdn/"
DDRAGON_CACHE = FRONTEND_DIR / ".cache" / ("champion_%s.json" % DDRAGON_VERSION)
# real pro games for the walkthrough (cached pro_drafts.pt)
PRO_DRAFTS_PT = ROOT / "data" / "pro play" / "pro_drafts.pt"
sys.path.insert(0, str(MODEL_DIR))

from soloq_model import (SoloQPretrainedModel, ROLE_FLEX_ID,  # noqa: E402
                         NUM_ROLES)

ROLE_ROWS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
N_SLOTS = 10


# Names dDragon stores under a different file id (heuristic fallback only;
# the manifest lookup is exact and needs none of these).
DDRAGON_NAME_FIX = {"Wukong": "MonkeyKing", "Nunu & Willump": "Nunu",
                    "LeBlanc": "Leblanc", "Renata Glasc": "Renata"}

_ICON_LOCK = threading.Lock()
_ICON_CACHE = None


def icon_url(name):
    """Deterministic dDragon icon URL from a display name (offline path)."""
    ident = DDRAGON_NAME_FIX.get(name) or re.sub(r"[^A-Za-z]", "", name)
    return "%s%s/img/champion/%s.png" % (DDRAGON_CDN, DDRAGON_VERSION, ident)


def ddragon_icons():
    """riot-id (int) -> icon URL from dDragon champion.json.
    Memory + file cache; network optional (5s timeout, failure -> {}
    so icon_url() fills every entry)."""
    global _ICON_CACHE
    with _ICON_LOCK:
        if _ICON_CACHE is not None:
            return _ICON_CACHE
        data = None
        try:
            if DDRAGON_CACHE.is_file():
                data = json.loads(DDRAGON_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt cache: refetch
            data = None
        if data is None:
            try:
                url = "%s%s/data/en_US/champion.json" % (DDRAGON_CDN,
                                                         DDRAGON_VERSION)
                with urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                DDRAGON_CACHE.parent.mkdir(parents=True, exist_ok=True)
                DDRAGON_CACHE.write_text(json.dumps(data), encoding="utf-8")
            except Exception:  # noqa: BLE001 - offline: heuristic URLs
                data = None
        icons = {}
        if isinstance(data, dict) and data.get("data"):
            for c in data["data"].values():
                try:
                    icons[int(c["key"])] = "%s%s/img/champion/%s" % (
                        DDRAGON_CDN, DDRAGON_VERSION, c["image"]["full"])
                except (KeyError, ValueError, TypeError):
                    continue
        _ICON_CACHE = icons
        return icons


def load_champ_data():
    """-> (riot_to_idx, names_by_idx, champions list for the UI).
    Every champion carries a dDragon `img` URL."""
    with open(PROC_DIR / "champ_mapping.json", "r", encoding="utf-8") as f:
        mapping = {int(k): int(v)
                   for k, v in json.load(f)["riot_to_idx"].items()}
    with open(PROC_DIR / "champ_names.json", "r", encoding="utf-8") as f:
        riot_to_name = {int(k): v
                        for k, v in json.load(f)["names"].items()}
    num = len(mapping)
    names_by_idx = [f"Riot{i}" for i in range(num)]
    for riot, i in mapping.items():
        if riot in riot_to_name:
            names_by_idx[i] = riot_to_name[riot]
    icons = ddragon_icons()
    champions = sorted(({"idx": i, "riot": riot, "name": names_by_idx[i],
                         "img": icons.get(riot)
                         or icon_url(names_by_idx[i])}
                        for riot, i in mapping.items()),
                       key=lambda c: c["name"].lower())
    return mapping, names_by_idx, champions


def find_best_ckpt():
    """Best = highest val champ_top5, tie -> newest; retry partial writes."""
    paths = sorted(glob.glob(CKPT_GLOB))
    if not paths:
        raise FileNotFoundError("no checkpoint under %s" % CKPT_GLOB)
    best_path, best_score = None, (-1.0, -1.0)
    for p in paths:
        payload = None
        for _ in range(3):
            try:
                payload = torch.load(p, map_location="cpu")
                break
            except Exception:  # noqa: BLE001 - file being written, retry
                time.sleep(1.0)
        if payload is None:
            continue
        score = (float(payload.get("metrics", {}).get("champ_top5", -1.0)),
                 Path(p).stat().st_mtime)
        if score >= best_score:
            best_path, best_score = p, score
    if best_path is None:
        raise RuntimeError("all candidate checkpoints unreadable")
    return Path(best_path), float(best_score[0])


class Predictor:
    """Wraps the loaded checkpoint; one forward -> top-5 per empty slot."""

    def __init__(self, ckpt_path, device="cpu"):
        payload = torch.load(str(ckpt_path), map_location="cpu")
        cfg = payload["config"]
        self.num_champs = int(cfg["num_champs"])
        state = payload["model"]
        # Finetune checkpoints store ProDraftModel.state_dict(): the shared
        # encoder under `encoder.` plus win/econ outcome heads under
        # `heads.`. Suggestions only need the encoder (its own champ/role
        # heads are part of it); the outcome heads are dropped here.
        if any(k.startswith("encoder.") for k in state):
            state = {k[len("encoder."):]: v for k, v in state.items()
                     if k.startswith("encoder.")}
            self.kind = "finetune"
        else:
            self.kind = "pretrain"
        self.model = SoloQPretrainedModel(
            num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_intra_layers=cfg["num_intra"],
            num_cross_layers=cfg["num_cross"],
            dropout=cfg.get("dropout", 0.0))
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.ckpt_path = str(ckpt_path)
        self.val_top5 = float(payload.get("metrics", {})
                              .get("champ_top5", -1))
        self.epoch = payload.get("metrics", {}).get("epoch")
        self.stage = payload.get("stage")   # finetune only (0/1/2)

    def _forward(self, champ_ids, role_ids):
        batch = {"champ_ids": champ_ids, "role_ids": role_ids,
                 "is_masked": champ_ids == self.num_champs,
                 "pad_mask": torch.zeros(1, N_SLOTS, dtype=torch.bool)}
        return self.model({k: v.to(self.device) for k, v in batch.items()})

    def _top5(self, logits, illegal):
        # Output classes are REAL champions only: champ_head is
        # Linear(D, num_champs), so [MASK]/[PAD] exist solely in the
        # 175-wide INPUT embedding and can never be suggested. The bounds
        # filter locks that contract in (model/README.md Sec.2).
        logits = logits.clone()
        logits[illegal] = float("-inf")
        probs = logits.softmax(-1)
        n_legal = int((~illegal).sum())
        top = probs.topk(max(1, min(5, n_legal)))
        return [{"idx": int(i), "prob": float(p)}
                for i, p in zip(top.indices, top.values)
                if float(p) > 0 and int(i) < self.num_champs]

    @torch.no_grad()
    def suggest(self, picks, bans_idx, mapping):
        """picks: len-10 riot ids or None - blue 0-4 / red 5-9 in PICK
        ORDER (slots carry no role meaning); bans_idx: banned champ idx.
        -> {"roles": [10 x [{role, prob}...]] (filled slots, desc),
            "blue"/"red": {role_name: [{idx, prob, flex?}] <=5}}

        Pass 1 queries picked champs with a FLEX role INPUT (= "role
        unknown", the same way [MASK] means "champion unknown") -> the
        5-way role head gives likely role assignments. Later passes hold
        each pick at its predicted role and cycle ALL 5 roles through the
        MASKed slots, so both teams get a role-conditioned top-5 per role
        (the 10% role-visible training condition). Bans/picks are
        post-filtered.

        Special tokens are input-only (model/README.md Sec.2): every
        returned idx is a real champion (< num_champs, never [MASK]/
        [PAD]) and every returned role is in 0..4 (never FLEX - the role
        head is Linear(D, 5)).
        """
        if len(picks) != N_SLOTS:
            raise ValueError("picks must have 10 entries")
        champ0 = [None] * N_SLOTS
        picked_idx = []
        for s, riot in enumerate(picks):
            if riot is None:
                continue
            if riot not in mapping:
                raise ValueError("unknown champion id %s" % riot)
            champ0[s] = mapping[riot]
            picked_idx.append(champ0[s])
        if len(set(picked_idx)) != len(picked_idx):
            raise ValueError("duplicate champion picked")
        illegal = torch.zeros(self.num_champs, dtype=torch.bool)
        for i in list(bans_idx) + picked_idx:
            if 0 <= i < self.num_champs:
                illegal[i] = True
        filled = [c is not None for c in champ0]
        champ_ids = torch.tensor(
            [[self.num_champs if c is None else c for c in champ0]],
            dtype=torch.long)

        # pass 1: (real champ, FLEX role input = "role unknown") ->
        # likely role assignments. FLEX is INPUT-only, exactly like the
        # [MASK] champ input: role_head is Linear(D, NUM_ROLES), so FLEX
        # cannot come back out - the filter defends it anyway.
        out = self._forward(champ_ids, torch.full(
            (1, N_SLOTS), ROLE_FLEX_ID, dtype=torch.long))
        roles, pred_role = [], [None] * N_SLOTS
        for s in range(N_SLOTS):
            if not filled[s]:
                roles.append([])
                continue
            probs = out["role_logits"][0, s].softmax(-1)
            order = torch.argsort(probs, descending=True)
            real = [int(r) for r in order if 0 <= int(r) < NUM_ROLES]
            roles.append([{"role": r, "prob": float(probs[r])}
                          for r in real])
            pred_role[s] = real[0]

        # passes: role-conditioned MASK queries, every role x both teams.
        # One pass covers (5 - picks) roles per team; <=5 passes total.
        empties = {0: [s for s in range(0, 5) if not filled[s]],
                   1: [s for s in range(5, 10) if not filled[s]]}
        queue = {0: list(range(5)), 1: list(range(5))}   # roles left
        passes = 0
        for t in (0, 1):
            if empties[t]:
                passes = max(passes, -(-5 // len(empties[t])))
        names = {0: "blue", 1: "red"}
        sugs = {team: {ROLE_ROWS[r]: [] for r in range(5)}
                for team in ("blue", "red")}
        for _ in range(passes):
            role_ids = [(pred_role[s] if filled[s] else ROLE_FLEX_ID)
                        for s in range(N_SLOTS)]
            assigned = []
            for t in (0, 1):
                for s in empties[t]:
                    if queue[t]:
                        r = queue[t].pop(0)
                        role_ids[s] = r
                        assigned.append((s, t, r))
            out = self._forward(
                champ_ids, torch.tensor([role_ids], dtype=torch.long))
            for s, t, r in assigned:
                # r always comes from the role queue (0..4): results are
                # only read from REAL-role-conditioned MASK queries -
                # FLEX-conditioned slots never contribute a shown result.
                sugs[names[t]][ROLE_ROWS[r]] = self._top5(
                    out["champ_logits"][0, s], illegal)

        # flex flag: same champ suggested under 2+ roles of one team
        for t in (0, 1):
            team = names[t]
            seen = {}
            for rname in ROLE_ROWS:
                for c in sugs[team][rname]:
                    seen[c["idx"]] = seen.get(c["idx"], 0) + 1
            for rname in ROLE_ROWS:
                for c in sugs[team][rname]:
                    if seen[c["idx"]] > 1:
                        c["flex"] = True
        return {"roles": roles, "blue": sugs["blue"], "red": sugs["red"]}


def _rel(p):
    """Project-relative POSIX-ish path; absolute passthrough if outside."""
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def model_info(pred):
    """JSON-safe description of the loaded checkpoint (header + loaders)."""
    p = Path(pred.ckpt_path)
    info = {"ckpt": "%s/%s" % (p.parent.name, p.name),
            "path": _rel(p),
            "val_top5": pred.val_top5,
            "epoch": pred.epoch,
            "kind": pred.kind}
    if pred.stage is not None:
        info["stage"] = pred.stage
    return info


def resolve_ckpt_path(spec, confine=True):
    """Checkpoint spec -> absolute .pt Path.
    confine=True (HTTP): must resolve inside the project - no `..` escapes.
    """
    p = Path(str(spec)).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if confine:
        try:
            p.relative_to(ROOT.resolve())
        except ValueError:
            raise ValueError("path outside project: %s" % spec)
    if p.suffix.lower() != ".pt":
        raise ValueError("expected a .pt checkpoint, got %s" % p.name)
    if not p.is_file():
        raise FileNotFoundError("no checkpoint at %s" % p)
    return p


def list_checkpoints():
    """Every .pt under the known ckpt dirs; cheap (payload not read)."""
    out = []
    for kind, pattern in CKPT_PATTERNS:
        for hit in sorted(glob.glob(pattern)):
            pp = Path(hit)
            out.append({"path": _rel(hit), "kind": kind,
                        "label": "[%s] %s/%s" % (kind, pp.parent.name,
                                                pp.name)})
    return out


def load_predictor(path, device=None):
    """Build a new Predictor first, then swap it in (old one stays on
    failure) - concurrent /api/suggest requests keep a valid model."""
    with LOAD_LOCK:
        pred = Predictor(path, device=device or APP.get("device", "cpu"))
        APP["predictor"] = pred
    return pred


_DRAFT_ROWS = None


def load_drafts():
    """Lazy-load the cached pro drafts (46.6k rows) on first use."""
    global _DRAFT_ROWS
    if _DRAFT_ROWS is None:
        payload = torch.load(str(PRO_DRAFTS_PT), map_location="cpu")
        _DRAFT_ROWS = payload["rows"]
    return _DRAFT_ROWS


def draft_summary(r, i):
    return {"i": i, "date": r["date"], "league": r["league"],
            "blue": r["blue_team"], "red": r["red_team"],
            "game_number": r["game_number"], "patch": r["patch"],
            "first_pick": r["first_pick"],
            "winner": "blue" if r["blue_win"] else "red"}


def list_drafts(q="", limit=100, offset=0, random_sample=False):
    """Filtered (date-descending) page of real games for the UI dropdown.
    -> (total_matches, [draft_summary x<=limit])."""
    rows = load_drafts()
    needle = q.strip().lower()
    hits = []
    for i, r in enumerate(rows):
        if needle and needle not in r["blue_team"].lower() \
                and needle not in r["red_team"].lower() \
                and needle not in str(r["league"]).lower() \
                and needle not in r["gameid"].lower() \
                and needle not in r["date"]:
            continue
        hits.append(i)
    hits.sort(key=lambda i: rows[i]["date"], reverse=True)
    if random_sample:
        random.shuffle(hits)
    page = hits[offset:offset + limit]
    return len(hits), [draft_summary(rows[i], i) for i in page]


def get_draft(i):
    """Full game -> summary + the 20 real moves (Riot ids/names/roles)."""
    rows = load_drafts()
    if not 0 <= i < len(rows):
        raise FileNotFoundError("no draft #%s" % i)
    r = rows[i]
    inv = {idx: riot for riot, idx in APP["mapping"].items()}
    names = APP["names_by_idx"]
    moves = []
    for kind, side, champ, role in r["moves"]:
        idx = int(champ)
        moves.append({"kind": "ban" if int(kind) == 0 else "pick",
                      "side": int(side), "idx": idx,
                      "riot": inv.get(idx), "name": names[idx],
                      "role": int(role) if int(role) >= 0 else None})
    out = draft_summary(r, i)
    out["gameid"] = r["gameid"]
    out["moves"] = moves
    return out


APP = {}  # main(): mapping, names_by_idx, champions, predictor, device


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/champions":
            pred = APP["predictor"]
            self._send(200, {
                "champions": APP["champions"],
                "roles": ROLE_ROWS,
                "model": model_info(pred),
            })
            return
        if path == "/api/drafts":
            qs = parse_qs(urlparse(self.path).query)
            try:
                limit = min(500, max(1, int(qs.get("limit", ["100"])[0])))
                offset = max(0, int(qs.get("offset", ["0"])[0]))
            except ValueError:
                limit, offset = 100, 0
            try:
                total, drafts = list_drafts(
                    q=qs.get("q", [""])[0], limit=limit, offset=offset,
                    random_sample=qs.get("random", ["0"])[0] in ("1", "true"))
            except Exception as exc:  # noqa: BLE001 - missing/bad data file
                self._send(500, {"error": "cannot load pro drafts: %s" % exc})
                return
            self._send(200, {"total": total, "drafts": drafts})
            return
        if path == "/api/draft":
            qs = parse_qs(urlparse(self.path).query)
            try:
                i = int(qs.get("i", ["-1"])[0])
            except ValueError:
                i = -1
            try:
                self._send(200, get_draft(i))
            except FileNotFoundError as exc:
                self._send(404, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - corrupt row
                self._send(500, {"error": "cannot load draft: %s" % exc})
            return
        if path == "/api/checkpoints":
            pred = APP.get("predictor")
            self._send(200, {"checkpoints": list_checkpoints(),
                             "current": (model_info(pred)["path"]
                                         if pred else None)})
            return
        if path.startswith("/api/"):
            self._send(404, {"error": "unknown endpoint"})
            return
        rel = "index.html" if path == "/" else path.lstrip("/")
        fpath = (STATIC_DIR / rel).resolve()
        if not str(fpath).startswith(str(STATIC_DIR.resolve())) \
                or not fpath.is_file():
            self._send(404, {"error": "not found"}, "text/plain")
            return
        ctype = {".html": "text/html", ".js": "text/javascript",
                 ".css": "text/css"}.get(fpath.suffix, "application/octet-stream")
        self._send(200, fpath.read_bytes(), ctype)

    def _read_json(self, max_bytes=1_000_000):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= max_bytes:
            raise ValueError("request body missing or too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_POST(self):  # noqa: N802
        route = self.path.split("?")[0]
        if route == "/api/suggest":
            self._handle_suggest()
        elif route == "/api/load":
            self._handle_load()
        elif route == "/api/upload":
            self._handle_upload()
        else:
            self._send(404, {"error": "unknown endpoint"})

    def _handle_suggest(self):
        try:
            payload = self._read_json()
            picks = payload["picks"]              # len 10, riot id or None
            bans = payload.get("bans", [])        # riot ids
            mapping, names = APP["mapping"], APP["names_by_idx"]
            bans_idx = [mapping[b] for b in bans if b in mapping]
            pred = APP["predictor"]               # one stable reference
            res = pred.suggest(picks, bans_idx, mapping)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        out = {"blue": {}, "red": {}, "roles": res["roles"],
               "model": model_info(pred)}
        for team in ("blue", "red"):
            for role, cands in res[team].items():
                lst = []
                for c in cands:
                    cand = {"name": names[c["idx"]], "prob": c["prob"],
                            "idx": c["idx"]}
                    if c.get("flex"):
                        cand["flex"] = True
                    lst.append(cand)
                out[team][role] = lst
        self._send(200, out)

    def _handle_load(self):
        """POST {"path": "<repo-relative .pt>"} -> swap the live model."""
        try:
            path = resolve_ckpt_path(self._read_json()["path"])
            pred = load_predictor(path)
        except FileNotFoundError as exc:
            self._send(404, {"error": str(exc)})
            return
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - foreign/broken ckpt
            self._send(500, {"error": "cannot load checkpoint: %s" % exc})
            return
        print("checkpoint <- %s (%s)" % (pred.ckpt_path, pred.kind))
        self._send(200, {"model": model_info(pred)})

    def _handle_upload(self):
        """Raw .pt bytes with ?name=x.pt -> model/uploaded/x.pt, then load."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                raise ValueError("empty body")
            if length > MAX_UPLOAD_BYTES:
                self._send(413, {"error": "checkpoint larger than %d MB"
                                 % (MAX_UPLOAD_BYTES // (1024 * 1024))})
                return
            data = self.rfile.read(length)     # always consume the body
            qs = parse_qs(urlparse(self.path).query)
            name = Path(qs.get("name", [""])[0]).name
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.pt", name):
                raise ValueError("file name must look like model.pt")
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = UPLOAD_DIR / name
            dest.write_bytes(data)
        except (ValueError, KeyError) as exc:
            self._send(400, {"error": str(exc)})
            return
        try:
            pred = load_predictor(dest)
        except Exception as exc:  # noqa: BLE001 - drop the useless file
            dest.unlink(missing_ok=True)
            self._send(400, {"error": "not a usable checkpoint: %s" % exc})
            return
        print("checkpoint <- %s (uploaded)" % pred.ckpt_path)
        self._send(200, {"model": model_info(pred)})


def main():
    global APP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps GPU free; cuda also fine")
    ap.add_argument("--ckpt", default="",
                    help="checkpoint .pt path - pretraining best.pt or a "
                         "finetune best_mask/best_win/latest.pt "
                         "(default: best pretraining by val champ_top5)")
    args = ap.parse_args()

    mapping, names_by_idx, champions = load_champ_data()
    if args.ckpt:
        ckpt = resolve_ckpt_path(args.ckpt, confine=False)
    else:
        ckpt, val = find_best_ckpt()
        print("best pretraining ckpt: %s (val top-5 %s)" % (ckpt, val))
    predictor = Predictor(ckpt, device=args.device)
    print("checkpoint: %s (%s)" % (ckpt, predictor.kind))
    APP.update(mapping=mapping, names_by_idx=names_by_idx,
               champions=champions, predictor=predictor,
               device=args.device)
    print("champions: %d, device: %s" % (len(mapping), args.device))

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("frontend: http://127.0.0.1:%d (local only)" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()

