"""Local-only prototype frontend for the SoloQ draft model.

Run:    python frontend/server.py --port 8777
Open:   http://127.0.0.1:8777   (binds 127.0.0.1 only - no external access)

UI: 5 ban + 5 pick slots per team (horizontal, role-free; the next pro
pick/ban to move is HIGHLIGHTED, never enforced - sequence follows
model/README.md draft format, mirrored when red has first pick), a role
board per team (TOP..UTILITY, picked champs placed in their assigned role,
empty roles showing the champion that role is looking for), dDragon
champion art (https://ddragon.leagueoflegends.com/cdn/16.20.1/img/champion/
... via the manifest, cached, deterministic name fallback offline), and a
walkthrough of real pro games (GET /api/drafts + GET /api/draft) showing
what the model suggested vs what the pros actually moved.

Two model slots, loaded at the same time and chosen independently:
  * role model (masking: `masking-*` checkpoints) - the role head and the
    per-role MASK queries, i.e. /api/suggest.
  * next-move model (`nextpick-*` checkpoints) - /api/next-move and
    /api/explain. Optional (--no-policy runs the role board alone).
Every kept artefact lives in model/checkpoints/<name>/ and the directory
name *is* the contract: the prefix states which inference the checkpoint
can do, the suffix which stage produced it.

  masking-soloqueue   soloQ-pretrained encoder (masking-finetune's init)
  masking-finetune    soloQ encoder fine-tuned on pro drafts
  masking-proonly     pro drafts only, from scratch (through 2026-10-31)
  nextpick-policy     soloQ encoder + policy head
  nextpick-proonly    pro drafts only, encoder + policy head from scratch
                      (through 2026-10-31)

A `-through-<league>-<a>-<b>` suffix states which series the training data ends
on, so two runs of the same stage are distinguishable and a stale artefact
cannot pass for a fresh one:

  masking-proonly-through-wlds-ig-tes    all 46,879 drafts, no test split
  nextpick-proonly-through-wlds-ig-tes   the same data, next-move head
  masking-proonly-randomvalsplit-through-wlds-ig-tes    same data, val drawn
  nextpick-proonly-randomvalsplit-through-wlds-ig-tes   at random per patch

The `-randomvalsplit-` pair is the shipped default: same corpus and frontier as
the other `-through-` pair, but val is a seeded random 10% of *every* patch
instead of each patch's tail, so 90% of every patch - including its end - is
trained on. Neither pair holds a held-out number (both train with `--no-test`),
so `STATUS.md` §7 - the pinned-test scores - still belongs to the originals; the
only honest comparison for these is `compare.py --split shared --other old`
against an original, which is in §7 too.

`_ckpt_roles` reads that prefix, so the dropdown never offers a policy
checkpoint to the role board. Select via --ckpt / --policy-ckpt (CLI), the
two header dropdowns (GET /api/checkpoints + POST /api/load with a `role`),
or "From file..." (POST /api/upload -> saved under model/uploaded/, offered
in both slots and validated on load).

Inference contract (model/README.md): every empty slot is a MASK query
with its ROLE GIVEN (role-conditional, matches the 10% role-visible
training condition); no PADs ever, so a team is never fully padded.
Bans are NOT model inputs - applied as a post-filter (the model already
-inf's already-picked visible champs). suggest() runs 1 FLEX-role pass
(role head over the picked champs -> the one-to-one role assignment, with
the slot just filled pinned first) plus <=5 role-conditioned MASK passes
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
CKPT_DIR = ROOT / "model" / "checkpoints"
# Only the masking family: `nextpick-*/best.pt` also matches a bare
# `*/best.pt` glob, and picking a policy checkpoint as the fallback role model
# would fail at load. The name says which inference, so the glob uses it.
CKPT_GLOB = str(CKPT_DIR / "masking-*" / "best.pt")
CKPT_PATTERNS = (                 # what the dropdown offers (no torch.load)
    ("checkpoints", str(CKPT_DIR / "*" / "*.pt")),
    ("uploaded", str(ROOT / "model" / "uploaded" / "*.pt")),
)
# Shipped defaults, one per job. The role board needs a *masking* checkpoint
# (role head + MASK queries); the next-move panel needs a *nextpick* one. They
# are separate slots so both can be loaded at once - the model that answers
# "which role / what is this team still missing" is not the model that ranks
# the next pick.
#
# Both point at the `-randomvalsplit-` refresh: same architecture and corpus as
# the earlier `-through-` pair (42,121 train), but val is a seeded random 10% of
# every patch rather than each patch's tail, so the end of every patch is
# trained on and the selection signal is 4,758 games instead of 52. Trained with
# `--no-test`, so they have no pinned-test score - §7 of STATUS.md is the
# originals' number, not theirs. Override with --ckpt / --policy-ckpt.
MASKED_DEFAULT = CKPT_DIR / "masking-proonly-randomvalsplit-through-wlds-ig-tes" / "best.pt"
POLICY_DEFAULT = CKPT_DIR / "nextpick-proonly-randomvalsplit-through-wlds-ig-tes" / "best.pt"
UPLOAD_DIR = ROOT / "model" / "uploaded"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024      # generous; ckpts are 20-150 MB
LOAD_LOCK = threading.Lock()              # serialize runtime model swaps
# dDragon static champion art (model patch 16.20)
DDRAGON_VERSION = "16.20.1"
DDRAGON_CDN = "https://ddragon.leagueoflegends.com/cdn/"
DDRAGON_CACHE = FRONTEND_DIR / ".cache" / ("champion_%s.json" % DDRAGON_VERSION)
# real pro games for the walkthrough (cached pro_drafts.pt)
PRO_DRAFTS_PT = ROOT / "data" / "pro play" / "pro_drafts.pt"
# The repo ROOT (parent of `model/`) goes on the path so `import model`
# resolves to the package; `soloq_model.py` lives in `model/pretraining/`
# now, so the package path is the only stable way to import it.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRONTEND_DIR))             # so `import llm` works

from model.pretraining.soloq_model import (SoloQPretrainedModel,  # noqa: E402
                                           ROLE_FLEX_ID, NUM_ROLES,
                                           load_encoder_state,
                                           attr_matrix_for_config)
from model.policy import infer  # noqa: E402  (next-move policy)
import llm  # noqa: E402  (frontend/llm.py: retrieval + grounded explanation)

ROLE_ROWS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
N_SLOTS = 10

# --- breakdown scoring -------------------------------------------------------
# Mirrors model/test_split/eval_test_split.py so the breakdown page reproduces
# the test-split report's per-move methodology (real patch id, exact role-keyed
# slots, target rank/prob) instead of the UI's inferred-role / UNK-patch
# next-move path. Kept self-contained: importing the pro pipeline here would
# pull heavy data-processing modules into the (already import-light) server.
_PHASE_OF = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4
PHASE_NAMES = {0: "Ban Phase 1", 1: "Pick Phase 1",
               2: "Ban Phase 2", 3: "Pick Phase 2"}
_MOVE_BAN, _MOVE_PICK = 0, 1
_ROLE_FLEX_ID = 5
_IGNORE_INDEX = -100


def _patch_key(patch):
    """Mirror data/pro play/pipeline.py:patch_key ('16.3' -> '16.03')."""
    s = str(patch or "").strip()
    if not s:
        return ""
    maj, _, mnr = s.partition(".")
    if not mnr:
        mnr = "0"
    try:
        return "%d.%02d" % (int(maj), int(mnr))
    except ValueError:
        return s


def _patch_index_of(pred):
    """This model's {patch_key: id} vocab, or None (-> UNK id 0)."""
    vocab = getattr(pred, "patch_vocab", None)
    if not vocab:
        cfg = getattr(pred, "config", None) or getattr(pred, "cfg", None) or {}
        vocab = cfg.get("patch_vocab")
    if not vocab:
        return None
    return {k: i + 1 for i, k in enumerate(vocab)}


def _patch_id_for(pred, row):
    pidx = _patch_index_of(pred)
    if not pidx:
        return 0
    return pidx.get(_patch_key(row.get("patch")), 0)


def _build_prefix(row, m, num_champs, patch_id):
    """One policy prefix example in the pro-data convention: slots are
    team*5 + role (role-keyed), unpicked = PAD, the real patch is fed."""
    pad = num_champs + 1
    champ_ids = torch.full((10,), pad, dtype=torch.long)
    role_ids = torch.full((10,), _ROLE_FLEX_ID, dtype=torch.long)
    pad_mask = torch.ones(10, dtype=torch.bool)
    ban_ids = torch.full((10,), pad, dtype=torch.long)
    n_bans = 0
    for kind, team, champ, role in row["moves"][:m]:
        if kind == _MOVE_BAN:
            ban_ids[n_bans] = champ
            n_bans += 1
        else:
            slot = team * 5 + int(role)
            champ_ids[slot] = champ
            role_ids[slot] = int(role)
            pad_mask[slot] = False
    kind, team, t_champ, t_role = row["moves"][m]
    taken = set(ban_ids[:n_bans].tolist())
    taken.update(champ_ids[~pad_mask].tolist())
    taken_mask = torch.zeros(num_champs, dtype=torch.bool)
    taken_mask[sorted(taken)] = True
    unavailable_mask = torch.zeros(num_champs, dtype=torch.bool)
    if row.get("unavailable"):
        unavailable_mask[sorted(row["unavailable"])] = True
    return {
        "champ_ids": champ_ids, "role_ids": role_ids, "pad_mask": pad_mask,
        "ban_ids": ban_ids, "n_bans": torch.tensor(n_bans, dtype=torch.long),
        "taken_mask": taken_mask, "unavailable_mask": unavailable_mask,
        "move_idx": torch.tensor(m, dtype=torch.long),
        "phase": torch.tensor(_PHASE_OF[m], dtype=torch.long),
        "to_move": torch.tensor(team, dtype=torch.long),
        "first_pick": torch.tensor(int(row["first_pick"]), dtype=torch.long),
        "is_ban": torch.tensor(1 if kind == _MOVE_BAN else 0, dtype=torch.long),
        "target_champ": torch.tensor(int(t_champ), dtype=torch.long),
        "target_role": torch.tensor(
            int(t_role) if kind == _MOVE_PICK else _IGNORE_INDEX,
            dtype=torch.long),
        "win_label": torch.tensor(int(row["blue_win"]), dtype=torch.long),
        "weight": torch.tensor(1.0, dtype=torch.float32),
        "patch_id": torch.tensor(int(patch_id), dtype=torch.long),
        "econ_target": torch.zeros(6, dtype=torch.float32),
        "has_econ": torch.tensor(0.0, dtype=torch.float32),
    }


def _collate(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


def _row_ban_indices(row):
    return [int(c) for kind, _t, c, _r in row["moves"] if kind == _MOVE_BAN]


def score_policy_model(pred, row, idx_to_img, names):
    """20 states/game -> top-1/top-5/rank/prob for the real next move."""
    nc = pred.num_champs
    pid = _patch_id_for(pred, row)
    samples = [_build_prefix(row, m, nc, pid) for m in range(len(row["moves"]))]
    batch = {k: v.to(pred.device) for k, v in _collate(samples).items()}
    with torch.no_grad():
        logits = pred.model(batch)["policy_logits"].float()
    probs = torch.softmax(logits, dim=-1)
    rows_out, ranks = {}, []
    top1 = top5 = 0
    for m in range(len(row["moves"])):
        lg = logits[m]
        target = int(batch["target_champ"][m])
        top5_idx = torch.topk(lg, 5).indices.tolist()
        top1_idx = int(lg.argmax())
        rank = int((lg > lg[target]).sum())          # 0-based
        prob = float(probs[m, target])
        hit1 = (top1_idx == target)
        hit5 = (target in top5_idx)
        top1 += int(hit1)
        top5 += int(hit5)
        ranks.append(rank + 1)
        rows_out[str(m)] = {
            "kind": "ban" if int(batch["is_ban"][m]) else "pick",
            "side": int(batch["to_move"][m]),
            "real_idx": target, "real_name": names[target],
            "real_img": idx_to_img.get(target),
            "top1_idx": top1_idx, "top1_name": names[top1_idx],
            "top1_img": idx_to_img.get(top1_idx),
            "top5": [{"idx": int(i), "name": names[int(i)],
                      "img": idx_to_img.get(int(i)),
                      "prob": float(probs[m, int(i)])} for i in top5_idx],
            "target_rank": rank + 1, "target_prob": prob,
            "hit_top1": bool(hit1), "hit_top5": bool(hit5),
        }
    n = len(row["moves"])
    return {"kind": "policy",
            "score": {"top1": top1 / n, "top5": top5 / n,
                      "mean_rank": sum(ranks) / n, "n": n},
            "rows": rows_out}


def score_masked_model(pred, row, idx_to_img, names):
    """Hide each picked slot in turn; did the model return the real champ?

    Mirrors eval_test_split.score_masked: the board carries the exact roles and
    the real patch id, and the rank is taken over the FULL distribution so
    the scoreboard matches the test-split report. The displayed top-5 is
    filtered to legal champions (not already taken / banned) for UX.
    """
    nc = pred.num_champs
    pad = nc + 1
    champ = torch.full((10,), pad, dtype=torch.long)
    role = torch.full((10,), _ROLE_FLEX_ID, dtype=torch.long)
    slot_move = {}
    for m, (kind, team, c, r) in enumerate(row["moves"]):
        if kind == _MOVE_PICK:
            slot = team * 5 + int(r)
            champ[slot] = c
            role[slot] = int(r)
            slot_move[slot] = m
    pid = _patch_id_for(pred, row)
    bans = _row_ban_indices(row)
    rows_out, ranks = {}, []
    top1 = top5 = 0
    for slot in range(10):
        cids = champ.clone()
        cids[slot] = nc                                  # MASK id
        is_m = torch.zeros(1, 10, dtype=torch.bool)
        is_m[0, slot] = True
        batch = {
            "champ_ids": cids.unsqueeze(0).to(pred.device),
            "role_ids": role.unsqueeze(0).to(pred.device),
            "pad_mask": torch.zeros(1, 10, dtype=torch.bool,
                                   device=pred.device),
            "is_masked": is_m.to(pred.device),
            "patch_id": torch.tensor([pid], dtype=torch.long,
                                    device=pred.device),
        }
        with torch.no_grad():
            lg = pred.model(batch)["champ_logits"][0, slot].float()
        target = int(champ[slot])
        rank = int((lg > lg[target]).sum())              # 0-based
        full_probs = torch.softmax(lg, dim=-1)
        prob = float(full_probs[target])
        illegal = torch.zeros(nc, dtype=torch.bool)
        for j in range(10):
            if j != slot and int(champ[j]) < nc:
                illegal[int(champ[j])] = True
        for c in bans:
            illegal[c] = True
        lg_d = lg.clone()
        lg_d[illegal] = float("-inf")
        top5_idx = torch.topk(lg_d, 5).indices.tolist()
        top1_idx = int(lg_d.argmax())
        hit1 = (top1_idx == target and not bool(illegal[target]))
        hit5 = (target in top5_idx)
        top1 += int(hit1)
        top5 += int(hit5)
        ranks.append(rank + 1)
        m = slot_move.get(slot)
        rows_out[str(m)] = {
            "kind": "pick", "side": slot // 5,
            "real_idx": target, "real_name": names[target],
            "real_img": idx_to_img.get(target),
            "top1_idx": top1_idx, "top1_name": names[top1_idx],
            "top1_img": idx_to_img.get(top1_idx),
            "top5": [{"idx": int(i), "name": names[int(i)],
                      "img": idx_to_img.get(int(i)),
                      "prob": float(full_probs[int(i)])}
                     for i in top5_idx if int(i) < nc],
            "target_rank": rank + 1, "target_prob": prob,
            "hit_top1": bool(hit1), "hit_top5": bool(hit5),
        }
    n = 10
    return {"kind": "masked",
            "score": {"top1": top1 / n, "top5": top5 / n,
                      "mean_rank": sum(ranks) / n, "n": n},
            "rows": rows_out}


def build_breakdown(game_i, model_paths, max_models=8):
    """Score one pro game under each selected model (policy + masked)."""
    rows = load_drafts()
    if not 0 <= game_i < len(rows):
        raise FileNotFoundError("no draft #%s" % game_i)
    row = rows[game_i]
    names = APP["names_by_idx"]
    idx_to_img = {c["idx"]: c["img"] for c in APP["champions"]}
    moves = []
    for m, (kind, team, champ, role) in enumerate(row["moves"]):
        idx = int(champ)
        moves.append({"i": m, "kind": "ban" if kind == _MOVE_BAN else "pick",
                      "side": int(team), "idx": idx, "name": names[idx],
                      "img": idx_to_img.get(idx),
                      "role": int(role) if int(role) >= 0 else None,
                      "phase": PHASE_NAMES[_PHASE_OF[m]]})
    models_out = []
    for spec in model_paths[:max_models]:
        path = resolve_ckpt_path(spec)                  # confines to project
        try:
            pred = make_predictor(path)
        except Exception as exc:                        # broken/foreign ckpt
            models_out.append({"path": _rel(path), "error": str(exc)})
            continue
        if pred.kind == "policy":
            res = score_policy_model(pred, row, idx_to_img, names)
        else:
            res = score_masked_model(pred, row, idx_to_img, names)
        res["path"] = _rel(path)
        res["label"] = "[%s] %s/%s" % (
            res["kind"], Path(path).parent.name, Path(path).name)
        res["val_top5"] = getattr(pred, "val_top5", None)
        models_out.append(res)
    game = {
        "i": game_i, "date": row["date"], "league": row["league"],
        "blue": row["blue_team"], "red": row["red_team"],
        "patch": row["patch"], "first_pick": int(row["first_pick"]),
        "winner": "blue" if row["blue_win"] else "red",
        "gameid": row["gameid"], "moves": moves,
    }
    return {"game": game, "models": models_out}


# Names dDragon stores under a different file id (heuristic fallback only;
# the manifest lookup is exact and needs none of these).
DDRAGON_NAME_FIX = {"Wukong": "MonkeyKing", "Nunu & Willump": "Nunu",
                    "LeBlanc": "Leblanc", "Renata Glasc": "Renata"}

_ICON_LOCK = threading.Lock()
_ICON_CACHE = None

_INDEX_LOCK = threading.Lock()
_PRO_INDEX = None


def pro_index():
    """The 46,875-draft retrieval index, built once per process (~1 s)."""
    global _PRO_INDEX
    if _PRO_INDEX is None:
        with _INDEX_LOCK:
            if _PRO_INDEX is None:
                _PRO_INDEX = llm.ProIndex.load(PRO_DRAFTS_PT)
    return _PRO_INDEX


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
    """Best = highest val champ_top5, tie -> newest; retry partial writes.

    A fallback only (MASKED_DEFAULT is tried first). The scores it compares were
    measured on different val sets - the originals on the every-patch split, the
    `-randomvalsplit-` pair on a random 10% of every patch, the `-through-` pair
    on 16.20's tail - and a larger, more predictable val has a higher ceiling, so
    this is a preference order, not a ranking.
    """
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


def peek_kind(payload):
    """Which family a payload belongs to, from its state-dict key names."""
    state = payload.get("model") or {}
    for key in state:
        if key.startswith("policy_head."):
            return "policy"
        if key.startswith("heads."):
            return "finetune"
    return "pretrain"


def _assign_roles(role_dists, filled, newest=None):
    """One-to-one slot -> role assignment per team.

    The role head is per-slot, so two picks on one team can argmax the same
    role. A team's five roles are distinct, so collisions are resolved with
    an exhaustive search over injective assignments (<=5 picks -> <=120
    leaves).

    `newest` (the slot just filled) is pinned to its own argmax first and
    the rest are then optimised jointly. That ordering matters: a pure
    sum-of-probabilities optimum happily leaves an older pick in a
    contested role - pick Vi (JUNGLE .97) while Pantheon (JUNGLE .60, TOP
    .30) is already there and the optimum keeps Pantheon in JUNGLE and
    pushes Vi to TOP, so the board looks like it ignored the click. Pinning
    the newest pick is lexicographic: the pick the user just made keeps the
    role the model likes best for it, and the older ones move aside.

    Returns a length-10 list of role ids (None for empty slots).
    """
    out = [None] * len(role_dists)
    for team in (0, 1):
        slots = [s for s in range(team * 5, team * 5 + 5)
                 if filled[s] and role_dists[s]]
        if not slots:
            continue
        if newest in slots:
            first, rest = newest, [s for s in slots if s != newest]
        else:
            first, rest = None, slots
        used = [False] * 5
        if first is not None:
            for cand in role_dists[first]:        # already prob-descending
                r = cand["role"]
                if 0 <= r < 5 and not used[r]:
                    out[first] = r
                    used[r] = True
                    break
        cur = {}
        best = {"score": -1.0, "assign": {}}

        def rec(i, score):
            if i == len(rest):
                if score > best["score"]:
                    best["score"] = score
                    best["assign"] = dict(cur)
                return
            for cand in role_dists[rest[i]]:
                r = cand["role"]
                if not (0 <= r < 5) or used[r]:
                    continue
                used[r] = True
                cur[rest[i]] = r
                rec(i + 1, score + cand["prob"])
                used[r] = False
                del cur[rest[i]]

        rec(0, 0.0)
        for s, r in best["assign"].items():
            out[s] = r
    return out


class Predictor:
    """Wraps the loaded checkpoint; one forward -> top-5 per empty slot."""

    def __init__(self, ckpt_path, device="cpu", payload=None):
        payload = payload if payload is not None else torch.load(
            str(ckpt_path), map_location="cpu")
        cfg = payload["config"]
        self.config = cfg            # patch_vocab lives here for patch models
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
            dropout=cfg.get("dropout", 0.0),
            # patch-conditioned checkpoints carry `embed.patch_lookup.*`;
            # without the matching `n_patches` load_encoder_state raises.
            n_patches=cfg.get("n_patches", 0),
            attr_matrix=attr_matrix_for_config(cfg))
        # load_encoder_state tolerates the opt-in value path (`pma.*`,
        # `win_head.*`) that pretraining checkpoints carry, and still raises
        # on a real mismatch. The frontend never needs the value head.
        load_encoder_state(self.model, state)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.ckpt_path = str(ckpt_path)
        self.metric = "champ top5"
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
    def complete_slot(self, champ_idx, role_idx, slot):
        """Masked completion of one pick slot (the breakdown page).

        Hide `slot` (set to the [MASK] id) and return its full champion
        logits + softmax. Like `_forward`, the patch input is omitted so the
        model runs with its UNK patch - the same convention the rest of the
        frontend uses for masked models.
        """
        cids = torch.tensor(
            [[self.num_champs if c is None else int(c) for c in champ_idx]],
            dtype=torch.long)
        rids = torch.tensor([[int(r) for r in role_idx]], dtype=torch.long)
        cids[0, slot] = self.num_champs
        out = self._forward(cids, rids)
        lg = out["champ_logits"][0, slot].float()
        p = torch.softmax(lg, dim=-1)
        return lg, p

    @torch.no_grad()
    def suggest(self, picks, bans_idx, mapping, last_slot=None,
                unavailable_idx=None):
        """picks: len-10 riot ids or None - blue 0-4 / red 5-9 in PICK
        ORDER (slots carry no role meaning); bans_idx: banned champ idx;
        last_slot: the slot just filled (wins a contested role);
        unavailable_idx: fearless pool (champions picked in earlier games
        of the series).
        -> {"roles": [10 x [{role, prob}...]] (filled slots, desc),
            "assignment": [10 x role id | None] (one per team, distinct),
            "blue"/"red": {role_name: [{idx, prob, flex?}] <=5}}

        Pass 1 queries picked champs with a FLEX role INPUT (= "role
        unknown", the same way [MASK] means "champion unknown") -> the
        5-way role head gives likely role assignments. Later passes hold
        each pick at its predicted role and cycle ALL 5 roles through the
        MASKed slots, so both teams get a role-conditioned top-5 per role
        (the 10% role-visible training condition). Bans/picks are
        post-filtered.

        Fearless is a *post-filter* here, not an input: the masked task was
        never trained with an unavailable mask, so the model cannot know a
        champion is gone. Filtering keeps an illegal champion out of the
        answer but does not make the remaining ranking fearless-aware -
        the probability mass was computed as if the champion were
        available. The UI says so when a pool is active.

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
        for i in list(bans_idx) + picked_idx + list(unavailable_idx or ()):
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
        roles = []
        for s in range(N_SLOTS):
            if not filled[s]:
                roles.append([])
                continue
            probs = out["role_logits"][0, s].softmax(-1)
            order = torch.argsort(probs, descending=True)
            real = [int(r) for r in order if 0 <= int(r) < NUM_ROLES]
            roles.append([{"role": r, "prob": float(probs[r])}
                          for r in real])
        # One consistent, collision-free role per team. The role head is
        # per-slot, so two picks can argmax the same role; the assignment
        # below drives both the MASK queries and the UI's role board. The
        # slot the user just filled is pinned to its own argmax, so a new
        # pick is never displaced by an older one.
        pred_role = _assign_roles(roles, filled, newest=last_slot)

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
        return {"roles": roles, "assignment": pred_role,
                "blue": sugs["blue"], "red": sugs["red"]}


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
            "kind": pred.kind,
            "metric": getattr(pred, "metric", "champ top5")}
    if getattr(pred, "stage", None) is not None:
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


def _ckpt_roles(path, kind):
    """Which model slots a checkpoint can fill, from its path alone.

    The UI has two slots (role model / next-move model) and offering the
    whole list in both would make most picks fail on load. torch.load'ing
    every checkpoint at startup just to classify them is not worth it, and
    it is not necessary: every kept artefact lives in
    model/checkpoints/<name>/, and the name states the inference it serves.

      masking-*    masked champion + role completion -> the role board
      nextpick-*   next-move ranking                -> the next-move panel
      value-*      outcome head only                -> neither (not servable)

    Paths outside that layout (uploads, checkpoints from before the rename)
    fall back to the old family rules, and anything still unrecognised is
    offered in both slots and validated on load.
    """
    if kind == "value":
        return []                       # outcome head only, not servable
    text = Path(path).as_posix()
    m = re.search(r"/checkpoints/([^/]+)/", text)
    if m:
        name = m.group(1)
        if name.startswith("masking-"):
            return ["masked"]
        if name.startswith("nextpick-"):
            return ["policy"]
        if name.startswith("value-"):
            return []
    # legacy families: still reachable via an upload of an old checkpoint
    if kind == "policy":
        return ["policy"]
    if kind in ("pretrain", "finetune"):
        return ["masked"]
    if "/proonly/" in text:
        m = re.search(r"run_\d+_\d+_([mp])-", text)
        if m:
            return ["masked"] if m.group(1) == "m" else ["policy"]
    return ["masked", "policy"]


def list_checkpoints():
    """Every .pt under the known ckpt dirs; cheap (payload not read).

    `latest.pt` is skipped: it is whatever the last epoch wrote, which is
    only useful for resuming, not for serving. The dropdown should offer
    the checkpoints that were selected on validation.
    """
    out = []
    for kind, pattern in CKPT_PATTERNS:
        for hit in sorted(glob.glob(pattern)):
            pp = Path(hit)
            if pp.name == "latest.pt":
                continue
            rel = _rel(hit)
            label = ("%s/%s" % (pp.parent.name, pp.name)
                     if kind == "checkpoints"
                     else "[%s] %s/%s" % (kind, pp.parent.name, pp.name))
            out.append({"path": rel, "kind": kind,
                        "roles": _ckpt_roles(hit, kind),
                        "label": label})
    return out


_MODEL_CACHE = {}                      # (path, device) -> predictor


def make_predictor(path, device=None):
    """Build (or fetch a cached) predictor for a checkpoint path.

    One file can hold three different model families, so the payload is
    read once here and dispatched: a policy checkpoint becomes a
    `PolicyPredictor`, everything else stays a masked `Predictor`. The
    result is cached per (path, device) so the breakdown endpoint can
    score several models without reloading the same multi-hundred-MB
    weights repeatedly.
    """
    device = device or APP.get("device", "cpu")
    key = (str(path), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    payload = torch.load(str(path), map_location="cpu")
    kind = peek_kind(payload)
    if kind == "policy":
        pred = infer.PolicyPredictor(path, device=device, payload=payload)
    else:
        pred = Predictor(path, device=device, payload=payload)
    # The patch vocabulary is stored at the checkpoint top level (both
    # families); keep it on the predictor so breakdown scoring can map a
    # game's patch string to the embedding row the model was trained with.
    pred.patch_vocab = (payload.get("patch_vocab")
                        or (payload.get("config") or {}).get("patch_vocab"))
    _MODEL_CACHE[key] = pred
    return pred


def load_predictor(path, device=None, role="auto"):
    """Build a model first, then swap it into its slot (the old one stays on
    failure) - concurrent requests keep a valid model.

    `role` files the model: a policy checkpoint answers the next move, every
    other kind answers the role board. The two slots are independent, so a
    masked role model and a policy next-move model are loaded at the same
    time; "auto" picks the slot from the checkpoint kind.
    """
    with LOAD_LOCK:
        pred = make_predictor(path, device=device)
        slot = "policy" if pred.kind == "policy" else "masked"
        if role in ("masked", "policy") and role != slot:
            raise ValueError("%s is a %s checkpoint, not a %s one"
                             % (Path(path).name, pred.kind, role))
        APP[slot] = pred
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
            # fearless is data-driven per series (2025+ mostly, 2026 always)
            "fearless": bool(r.get("fearless")),
            "n_unavailable": len(r.get("unavailable") or ()),
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
    # The real fearless pool for this game - champions picked in earlier
    # games of the same series (pipeline.attach_fearless). Empty for a
    # non-fearless series and for game 1. The UI seeds its pool from this
    # when a real game is loaded, so a walkthrough of G3 starts with the
    # 20 champions that are actually gone.
    out["unavailable"] = [inv[int(c)] for c in (r.get("unavailable") or ())
                          if int(c) in inv]
    return out


APP = {}  # main(): mapping, names_by_idx, champions, device,
          #        masked (role model), policy (next-move model)


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
            masked, policy = APP.get("masked"), APP.get("policy")
            # The green "next move" highlight is derived from this plan on the
            # client, synchronously. It must not wait for /api/next-move: that
            # answer describes the board it was requested for, so an answer in
            # flight is always one board behind. infer.move_plan is the single
            # source of truth (checked against pipeline.interleaved_moves in
            # model/policy/test_infer.py), so publish it rather than copying
            # the order into app.js.
            move_plan = {
                str(fp): [["ban" if kind == infer.MOVE_BAN else "pick", int(t)]
                          for kind, t in infer.move_plan(fp)]
                for fp in (0, 1)
            }
            # Patch conditioning belongs to the loaded policy checkpoint, so
            # its vocabulary ships with the same payload that says which
            # models are live. A checkpoint with no patch table publishes an
            # empty list and the UI disables the selector instead of
            # pretending the choice does something.
            patches = policy.patch_options() if policy else []
            self._send(200, {
                "champions": APP["champions"],
                "roles": ROLE_ROWS,
                "move_plan": move_plan,
                "patches": patches,
                "default_patch": infer.DEFAULT_PATCH,
                "default_patch_id": (policy.default_patch_id() if policy
                                     else infer.PATCH_UNK_ID),
                "model": model_info(masked) if masked else None,
                "policy_model": model_info(policy) if policy else None,
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
            self._send(200, {
                "checkpoints": list_checkpoints(),
                "current": {slot: (model_info(APP[slot])["path"]
                                   if APP.get(slot) else None)
                            for slot in ("masked", "policy")},
            })
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
        elif route == "/api/next-move":
            self._handle_next_move()
        elif route == "/api/explain":
            self._handle_explain()
        elif route == "/api/load":
            self._handle_load()
        elif route == "/api/upload":
            self._handle_upload()
        elif route == "/api/breakdown":
            self._handle_breakdown()
        else:
            self._send(404, {"error": "unknown endpoint"})

    def _handle_suggest(self):
        try:
            payload = self._read_json()
            pred = APP.get("masked")              # one stable reference
            if pred is None:
                raise ValueError(
                    "no role model loaded - pick a masked checkpoint "
                    "(pretraining / finetune / proonly best.pt) as the "
                    "role model")
            picks = payload["picks"]              # len 10, riot id or None
            bans = payload.get("bans", [])        # riot ids
            fearless = payload.get("fearless") or []   # riot ids
            last_slot = payload.get("last_slot")
            if last_slot is not None:
                last_slot = int(last_slot)
                if not 0 <= last_slot < N_SLOTS:
                    raise ValueError("last_slot must be a slot index 0-9")
            mapping, names = APP["mapping"], APP["names_by_idx"]
            bans_idx = [mapping[b] for b in bans if b in mapping]
            unavail_idx = [mapping[f] for f in fearless if f in mapping]
            res = pred.suggest(picks, bans_idx, mapping, last_slot=last_slot,
                               unavailable_idx=unavail_idx)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        out = {"blue": {}, "red": {}, "roles": res["roles"],
               "assignment": res["assignment"],
               # the masked model cannot consume the pool, so this is a
               # filter, not a condition - the UI warns when it is non-zero
               "fearless_filtered": len(unavail_idx),
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

    def _handle_next_move(self):
        """POST {picks, bans, first_pick, fearless, patch_id} -> next move.

        `picks` is the UI's 10-slot pick-order board (riot ids or null),
        `bans` the flat ban list, `fearless` the champions picked in
        earlier games of the same series, and `patch_id` a row of the
        loaded checkpoint's patch vocabulary (omitted -> the default
        patch, usually 16.20). The board is mapped to indices here and
        handed to infer.PolicyPredictor; every champion index it returns is
        translated back to a name so the client never needs the mapping.

        One answer per board: the ranked list for the side on the clock.
        There is no separate ban ranking - a pro-like ban is the denial
        decision, and the rollout that used to approximate it was removed
        (STATUS.md §5/§8).
        """
        try:
            payload = self._read_json()
            pred = APP.get("policy")
            if pred is None:
                raise ValueError(
                    "no next-move policy loaded - pick a best_policy.pt as "
                    "the next-move model to use this panel")
            mapping, names = APP["mapping"], APP["names_by_idx"]
            picks = payload.get("picks") or [None] * 10
            bans = payload.get("bans") or []
            fearless = payload.get("fearless") or []
            first_pick = 1 if int(payload.get("first_pick", 0)) else 0
            picks_idx = [None if p is None else mapping[p] for p in picks]
            bans_idx = [mapping[b] for b in bans]
            unavail_idx = [mapping[f] for f in fearless]
            pid = payload.get("patch_id")
            pid = pred.default_patch_id() if pid is None else pid
            res = pred.next_move(picks_idx, bans_idx, first_pick=first_pick,
                                 k=int(payload.get("k", 8)),
                                 unavailable=unavail_idx, patch_id=pid)
        except (ValueError, KeyError, TypeError, IndexError,
                json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return

        self._send(200, {
            "model": model_info(pred),
            "state": {k: res[k] for k in ("move_idx", "kind", "side", "phase",
                                          "first_pick", "warning", "complete",
                                          "n_unavailable")},
            # what the model actually conditioned on (an unknown or
            # out-of-range request is reported as the UNK row it fell back to)
            "patch_id": res["patch_id"],
            "patch": pred.patch_key_for(res["patch_id"]),
            "suggestions": [{"idx": c["idx"], "prob": c["prob"],
                             "name": names[c["idx"]]}
                            for c in res["suggestions"]],
            "roles": res["roles"],
        })

    def _handle_explain(self):
        """POST {picks, bans, first_pick, patch_id, k, k_games, patch, force}
        -> top-k, retrieved evidence, and prose when an LLM key is configured.

        `patch_id` conditions the model exactly as /api/next-move does;
        `patch` overrides only which patch NOTES are quoted (default: the
        selected patch, else the newest in the corpus). Without a key this
        still returns the evidence. That ordering is deliberate: the
        retrieval is the part that has to be right either way, and it is the
        part a coach can use on its own.
        """
        try:
            payload = self._read_json()
            pred = APP.get("policy")
            if pred is None:
                raise ValueError(
                    "/api/explain needs a next-move policy checkpoint; none "
                    "is loaded")
            mapping, names = APP["mapping"], APP["names_by_idx"]
            picks = payload.get("picks") or [None] * 10
            bans = payload.get("bans") or []
            first_pick = 1 if int(payload.get("first_pick", 0)) else 0
            picks_idx = [None if p is None else mapping[p] for p in picks]
            bans_idx = [mapping[b] for b in bans]
            pid = payload.get("patch_id")
            pid = pred.default_patch_id() if pid is None else pid
            res = pred.next_move(picks_idx, bans_idx, first_pick=first_pick,
                                 k=int(payload.get("k", 5)), patch_id=pid)
            index = pro_index()
        except (ValueError, KeyError, TypeError, IndexError,
                json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - corpus missing/unreadable
            self._send(500, {"error": "cannot build the draft index: %s" % exc})
            return

        board = {
            "move_idx": int(res["move_idx"]),
            "side": "blue" if int(res["side"]) == 0 else "red",
            "first_pick": first_pick,
            "first_pick_side": "red" if first_pick else "blue",
            # The patch selector is the default for the notes too, so the
            # prose talks about the patch the model was actually
            # conditioned on. An explicit `patch` in the body still wins.
            "patch": (payload.get("patch")
                      or pred.patch_key_for(res["patch_id"])
                      or index.latest_patch),
            "bans": [names[i] for i in bans_idx],
            "blue": [names[i] for i in picks_idx[:5] if i is not None],
            "red": [names[i] for i in picks_idx[5:] if i is not None],
            "champ_ids": [i for i in picks_idx if i is not None] + bans_idx,
        }
        suggestions = [(names[c["idx"]], float(c["prob"]))
                       for c in res["suggestions"]]
        try:
            explained = llm.explain(
                board, suggestions, index, client=llm.client_from_env(),
                k_games=int(payload.get("k_games", 5)),
                force=bool(payload.get("force")), names=names)
        except Exception as exc:  # noqa: BLE001 - one panel, not the server
            self._send(500, {"error": "explain failed: %s" % exc})
            return
        out = {
            "model": model_info(pred),
            "state": {k: res[k] for k in ("move_idx", "kind", "side", "phase",
                                          "first_pick", "warning", "complete")},
            "suggestions": [{"idx": c["idx"], "prob": c["prob"],
                             "name": names[c["idx"]]}
                            for c in res["suggestions"]],
            "board": board,
        }
        out.update(explained)
        self._send(200, out)

    def _handle_load(self):
        """POST {"path": ".pt", "role": "masked"|"policy"|"auto"} -> swap
        that slot's model. `role` defaults to auto (decided by the kind);
        a null path unloads the slot, which is how the UI turns the
        next-move panel off without touching the role board."""
        try:
            payload = self._read_json()
            role = payload.get("role") or "auto"
            if role not in ("auto", "masked", "policy"):
                raise ValueError("role must be masked, policy or auto")
            spec = payload.get("path")
            if not spec:
                if role == "auto":
                    raise ValueError("unloading a model needs an explicit role")
                with LOAD_LOCK:
                    APP.pop(role, None)
                print("%s model unloaded" % role)
                self._send(200, {"model": None, "role": role})
                return
            path = resolve_ckpt_path(spec)
            pred = load_predictor(path, role=role)
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
        self._send(200, {"model": model_info(pred),
                         "role": "policy" if pred.kind == "policy"
                         else "masked"})

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
        self._send(200, {"model": model_info(pred),
                         "role": "policy" if pred.kind == "policy"
                         else "masked"})

    def _handle_breakdown(self):
        """POST {game_i, models:[repo-relative .pt]} -> per-move breakdown.

        Scores one pro game under every selected checkpoint and returns the
        real move next to each model's prediction. Does NOT touch the live
        model: `make_predictor` caches by path, so the header model and the
        breakdown selections stay independent.
        """
        try:
            payload = self._read_json()
            game_i = int(payload.get("game_i", -1))
            model_paths = payload.get("models") or []
            if not isinstance(model_paths, list):
                raise ValueError("models must be a list of .pt paths")
            if not model_paths:
                raise ValueError("select at least one model")
            result = build_breakdown(game_i, model_paths)
        except (ValueError, KeyError, TypeError, IndexError,
                FileNotFoundError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - one page, not the server
            self._send(500, {"error": "breakdown failed: %s" % exc})
            return
        self._send(200, result)


def main():
    global APP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8777)
    # Default stays loopback-only: this server has no auth and exposes the
    # draft model, so binding it to a routable interface must be opt-in.
    # Needed only when running inside a container, where the published port
    # arrives on eth0 rather than on the container's own loopback.
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; 127.0.0.1 (default) is local-only, "
                         "use 0.0.0.0 only inside a container")
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps GPU free; cuda also fine")
    ap.add_argument("--ckpt", default="",
                    help="role-model checkpoint (a masking-* checkpoint) - "
                         "drives the role board and the per-role targets "
                         "(default: model/checkpoints/"
                         "masking-proonly-randomvalsplit-through-wlds-ig-tes"
                         "/best.pt, "
                         "else the best masking checkpoint by val champ_top5)")
    ap.add_argument("--policy-ckpt", default="",
                    help="next-move checkpoint (a nextpick-* checkpoint) - "
                         "loaded alongside the role model, not instead of it "
                         "(default: model/checkpoints/"
                         "nextpick-proonly-randomvalsplit-through-wlds-ig-tes"
                         "/best.pt)")
    ap.add_argument("--no-policy", action="store_true",
                    help="run the role board only, without a next-move model")
    args = ap.parse_args()

    mapping, names_by_idx, champions = load_champ_data()
    APP.update(device=args.device)

    # Slot 1 - the role model (masked): role board + per-role targets.
    if args.ckpt:
        masked_ckpt = resolve_ckpt_path(args.ckpt, confine=False)
    elif MASKED_DEFAULT.is_file():
        masked_ckpt = MASKED_DEFAULT
    else:
        masked_ckpt, val = find_best_ckpt()
        print("best pretraining ckpt: %s (val top-5 %s)" % (masked_ckpt, val))
    # Slot 2 - the next-move model (policy). Optional: the role board works
    # without it, and --no-policy runs the frontend masked-only.
    policy_ckpt = None
    if not args.no_policy:
        if args.policy_ckpt:
            policy_ckpt = resolve_ckpt_path(args.policy_ckpt, confine=False)
        elif POLICY_DEFAULT.is_file():
            policy_ckpt = POLICY_DEFAULT

    masked = load_predictor(masked_ckpt, device=args.device, role="masked")
    print("role model:   %s (%s)" % (masked_ckpt, masked.kind))
    if policy_ckpt is not None:
        policy = load_predictor(policy_ckpt, device=args.device, role="policy")
        print("next-move:    %s (%s)" % (policy_ckpt, policy.kind))
    else:
        print("next-move:    (none - role board only)")
    APP.update(mapping=mapping, names_by_idx=names_by_idx, champions=champions)
    print("champions: %d, device: %s" % (len(mapping), args.device))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("frontend: http://%s:%d%s"
          % (args.host, args.port,
             " (local only)" if args.host == "127.0.0.1" else ""))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()

