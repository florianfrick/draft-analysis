"""Local-only prototype frontend for the SoloQ draft model.

Run:    python frontend/server.py --port 8777
Open:   http://127.0.0.1:8777   (binds 127.0.0.1 only - no external access)

Five things, and the board is the only one that is not a question:

  **the board** - 5 ban + 5 pick slots per team (horizontal, role-free; the next
  pro pick/ban to move is HIGHLIGHTED, never enforced - sequence follows
  solo_queue/model/README.md draft format, mirrored when red has first pick),
  dDragon champion art (https://ddragon.leagueoflegends.com/cdn/16.20.1/img/
  champion/ ... via the manifest, cached, deterministic name fallback offline),
  a fearless series bar, and a walkthrough of real pro games (GET /api/drafts +
  GET /api/draft) showing what the pros actually moved.

  **the next move** (`POST /api/next-move`) - the head's ranked list for the
  side on the clock, each champion with its share of the legal pool.

  **the role board** (`POST /api/suggest`) - each team's five roles, the picked
  champions placed in the role the corpus says they fill, and for every role
  still open the champion the next-move head would take for it. One cell is one
  `predict_in_lane` question; the answer says which model answered it
  (`method`), because the head can only be asked on a pick move.

  **the agent** (`POST /api/agent`) - one turn of an LLM that holds the board
  as an object (`pro/model/agent/`), calls the models and the corpus through
  tools, and answers in prose. This replaced the breakdown page, the draft-score
  comparison and the rule-based chat: one surface that can be asked anything,
  instead of four that each answered a fixed question.

  **the agent's suggestion** (`POST /api/agent-suggest`) - the same LLM, one
  click, asked the fixed reranking question (`pro/model/agent/suggest.py`) with
  SQL ablated and the explained prompt: it reads the live state, calls the head
  and the lane table through tools, and commits to one champion. The panel shows
  the parsed move, the head's own #1 for comparison, and the tool log.

Two model slots, loaded at the same time and chosen independently:
  * role model (masking: `masking-*` checkpoints) - answers a role question
    during the ban phase, when the head cannot. Drives the role board's empty
    cells in phase 1.
  * next-move model - the 4B reranker (`llm-*`, the default when it can be
    built) or the head (`nextpick-*`, the fallback). Answers the board question,
    the role question on a pick move, and the agent's `predict_next_move`.
    Optional (`--no-policy` runs the role board alone, and the agent still
    works - it just cannot rank a move).

Both slots are bound into one `engine.Engine`, which is what a draft state asks
its questions through. The server never hands a board to a model function: it
publishes the board to the state store, and the state carries the answers.
That publish is also where the rest of the game arrives - patch, first pick,
the fearless pool and which game of the series this is - so an agent turn reads
one object rather than a board retyped into a prompt.

Every kept artefact lives in <track>/model/checkpoints/<name>/ - the track is
`solo_queue/` or `pro/` - and the directory name *is* the contract: the prefix
states which inference the checkpoint can do, the suffix which stage produced it.
Models are chosen with `--ckpt` / `--policy-ckpt` when the server starts. There
is no browser picker: the header only reports which two checkpoints are live.
Both defaults are pro-track, because the pro-only recipe is the better model.

  masking-soloqueue   soloQ-pretrained encoder (masking-finetune's init)
  masking-finetune    soloQ encoder fine-tuned on pro drafts
  masking-proonly     pro drafts only, from scratch (through 2026-10-31)
  nextpick-policy     soloQ encoder + policy head
  nextpick-proonly    pro drafts only, encoder + policy head from scratch
                      (through 2026-10-31)
  nextpick-bans-*     the pro-only policy with the ban-awareness flags on
                      (`ban_side`, `ban_fearless`, `counter_rank`); selectable
                      via --policy-ckpt, no longer the default
  nextpick-proto-ensemble   `ensemble.json`: the mean of three nextpick
                      members' policy logits (selectable, no longer default)
  llm-lora-rerank5    the 4.2 B Qwen adapter over the head's top 20 (a
                      directory: adapter/ + projector.pt; the trunk comes from
                      the `llm` image). `--rerank` serves it; it is not the
                      default (README, "What the reranker is worth")

A `-through-<league>-<a>-<b>` suffix states which series the training data ends
on, so two runs of the same stage are distinguishable and a stale artefact
cannot pass for a fresh one:

  masking-proonly-through-wlds-ig-tes    all 46,879 drafts, no test split
  nextpick-proonly-through-wlds-ig-tes   the same data, next-move head
  masking-proonly-randomvalsplit-through-wlds-ig-tes    same data, val drawn
  nextpick-proonly-randomvalsplit-through-wlds-ig-tes   at random per patch
  masking-proonly-nov13-through-lcp-df-sw   46,939 drafts (2026-11-13 export)
  nextpick-proonly-nov13-k5-through-lcp-df-sw   the same, 5 prefixes/game

BOTH slots ship on the `-nov13-` pair - `MASKED_DEFAULT` and `POLICY_DEFAULT`.
That pair is a different corpus, not a newer run of the old one: the corpus
freeze is lifted (`--corpus 2022-full`, 46,939 drafts through 2026-11-13), the
pinned test series are ordinary rows (`--no-pin`), and the split is the
fractional 85/10/5 within-patch tail. The policy also trains on `--prefix-k 5`
(5 distinct prefixes per game per epoch) instead of one.

So **none of its numbers are comparable with the `-wlds-ig-tes` rows** in
`STATUS.md` §7: different corpus, different split, different test set. On its
own 2,448-game fractional test split it scores top-5 0.5083 (rel 3.36x), and a
paired bootstrap puts it +2.55 pp over the same recipe with one prefix
(CI [+2.22, +2.87], p 2.7e-51). Use that split, not §7, for any comparison.

The `-randomvalsplit-` pair and the two former next-move defaults
(`nextpick-bans-aware-through-wlds-ig-tes`, the pro-only policy plus all three
ban-awareness flags, and the 3-member `nextpick-proto-ensemble/ensemble.json`)
remain **CLI-selectable**. Their numbers must not be added to the nov13 pair's
or to each other: the ban-aware arm's +0.60 pp (0.5056 vs 0.4995 top-5,
paired bootstrap p = 4.3e-8) is a **val** score on the split its control
trained and selected under and is a single seed (`pro/model/archive/bans/README.md`);
the ensemble's +1.33 pp was measured on 439 draft rows neither it nor its
predecessor had trained on, a different and harder set.

**Where the weights live.** `pro/model/checkpoints/` holds exactly the shipped
pair and nothing else; every other checkpoint named above is under
`pro/model/archive/checkpoints/`. That is why `PRO_CKPT_DIR` (defaults) and
`ARCHIVE_CKPT_DIR` (everything a flag can reach) are separate constants: a
future edit cannot promote an archived arm to a default by dropping a directory
in the wrong store. `pro/model/test_split/suite.py` is the harness that scores
any of them on the canonical split, and it is the number to use for a
comparison - not §7.

Select with `--ckpt` / `--policy-ckpt` / `--rerank` / `--no-policy` at startup.
The
`GET /api/checkpoints`, `POST /api/load` and `POST /api/upload` routes still
exist as a local scripting surface (and are tested), but no page offers them:
swapping weights mid-draft would leave every answer in flight describing the
previous model. `_ckpt_roles` reads the directory prefix so a policy
checkpoint can never be filed into the role slot.

The next-move slot ships the head. The 4B reranker is one flag away
(`--rerank`), and it is not the default because the number that would have made
it the default does not survive a served comparison:

    reranker, task line names the lane ("for Top lane")   34.27 %
    reranker, no lane (what this panel actually asks)     20.85 %
    head that proposed those candidates (bans-aware)      18.25 %
    shipped head, same rows, served                       24.90 %

Four arms, one slice of 4,000 held-out states, and only the first is above the
head - because `target_role` in the training rows is the *true* lane of the
pro's pick, i.e. that arm is handed part of the answer. The 20.85 % row is the
panel's question, and it is on candidates that were not legality-filtered, so it
is not even a clean +2.6 against the 18.25 it is usually quoted against; the
24.90 is the shipped head on the same rows but the `-nov13-` head trained on a
different split, so it is not a held-out number either. `--rerank` therefore
stays opt-in and a build failure is fatal. The role board is the surface where
the lane *is* the question, and it is the reason to keep the wiring:
`pro/model/archive/llm/eval_role_cells.py` measures it (result in STATUS.md §9).

The mechanism behind the 13.4 pp gap: the training rows carry the pro's *true*
role, so the lane clause is a hint about the answer, and the adapter learned to
use it. A role-board cell names a lane as the *question* ("what is red's
jungle"), which is a different and legitimate use of the same clause - that is
why the flag is worth keeping even though the panel's arm is thin.

Inference contract (solo_queue/model/README.md): every empty slot is a MASK query
with its ROLE GIVEN (role-conditional, matches the 10% role-visible
training condition); no PADs ever, so a team is never fully padded.
Bans are NOT model inputs - applied as a post-filter (the model already
-inf's already-picked visible champs). The masking path runs one
role-conditioned MASK pass per cell asked about; pick-order slots are fine -
the encoder is permutation-invariant within a team.

The role *assignment* is no longer the role head's. It comes from
`data/champ_roles.json` through `model_tools.assignment`, which maximises the
summed log-likelihood over <=120 injective assignments per team: the head's own
argmax agrees with the corpus 67 % of the time and collides two flex champions
on one role, the table agrees 96 % and cannot collide.
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
# Two artefact stores, one per track. Nothing is shared: a checkpoint trained
# on solo queue lives under `solo_queue/`, everything trained on pro drafts
# lives under `pro/`. The UI offers both, because the role board has a
# solo-queue option and the next-move panel does not.
SOLOQ_CKPT_DIR = ROOT / "solo_queue" / "model" / "checkpoints"
PRO_CKPT_DIR = ROOT / "pro" / "model" / "checkpoints"
# Everything that is not the shipped pair lives here. Reached into by
# `--rerank` / `--policy-ckpt` / `--ckpt`, never by a default.
ARCHIVE_CKPT_DIR = ROOT / "pro" / "model" / "archive" / "checkpoints"
CKPT_DIRS = (SOLOQ_CKPT_DIR, PRO_CKPT_DIR)
STATIC_DIR = FRONTEND_DIR / "static"
# Only the masking family: `nextpick-*/best.pt` also matches a bare
# `*/best.pt` glob, and picking a policy checkpoint as the fallback role model
# would fail at load. The name says which inference, so the glob uses it.
CKPT_GLOB = [str(d / "masking-*" / "best.pt") for d in CKPT_DIRS]
CKPT_PATTERNS = tuple(            # what /api/checkpoints lists (no torch.load)
    [("checkpoints", str(d / "*" / "*.pt")) for d in CKPT_DIRS]
    # an ensemble manifest is a selectable model, not a checkpoint payload
    + [("checkpoints", str(d / "*" / "ensemble.json")) for d in CKPT_DIRS]
    # a reranker is a *directory* (adapter/ + projector.pt), and the directory
    # is the thing you name; listing `projector.pt` instead would offer a file
    # that is one piece of it
    + [("checkpoints", str(d / "llm-*")) for d in CKPT_DIRS]
    + [("uploaded", str(FRONTEND_DIR / "uploaded" / "*.pt"))]
)
# Shipped defaults, one per job. The role board needs a *masking* checkpoint
# (role head + MASK queries); the next-move panel needs a *nextpick* one. They
# are separate slots so both can be loaded at once - the model that answers
# "which role / what is this team still missing" is not the model that ranks
# the next pick.

MASKED_DEFAULT = PRO_CKPT_DIR / "masking-proonly-nov13-through-lcp-df-sw" / "best.pt"

POLICY_DEFAULT = PRO_CKPT_DIR / "nextpick-proonly-nov13-k5-through-lcp-df-sw" / "best.pt"

# The reranker: a *directory* (adapter/ + projector.pt), not a best.pt, because
# it is a 4.2 B trunk plus a LoRA plus the encoder's projector. It answers the
# same `next_move` interface as POLICY_DEFAULT, so it fills the same slot.
#
# The number depends on which question is asked, and the two differ by 13 pp:
#
#   lane named in the task line ("for Top lane")   34.27 % exact  (4,000 states)
#   no lane (this panel's unconstrained next move) 20.85 % exact  (same states)
#   the head that proposed those candidates       18.25 % raw     (same states)
#   the shipped head, served, same rows           24.90 %         (same states)
#
# `eval_rerank5_mask4k.json` and `eval_rerank5_nolane4k.json` are the first two;
# `eval_served_banaware4k.json` and `eval_served_head4k.json` are the last two.
# The 13.4 pp is the lane clause, not drafting skill, and `target_role` in the
# training rows is the *true* lane of the pro's pick. The 18.32 % this used to
# quote is `C.BASELINE_VAL` - the *bans-aware* head over 95,160 states, not the
# shipped `-nov13-` head over these 4,000 - so it was never the row to compare
# against either.
#
# And the lane-named row is not a capability the repo lacked: the masker already
# answers a lane-conditioned query at 2.5 M parameters (35.7 % top-1 on 70
# slot-mode queries, STATUS §7), on *more* information - `masked_ranks_slot`
# keeps the nine other picks including later ones and the true role vector,
# where this model sees the prefix only. Its own larger reports are 20.2 % /
# 21.3 % on 4,193 / 8,069 slots, so that is a band rather than a rank. Either
# way the 34.27 % is not evidence that a 4.2 B model was needed.
#
# It is **not** the default. `--rerank` opts in, and a failure to build it is
# fatal: a server that refuses to start beats one that quietly answers with a
# different model than the flag asked for. The numbers above are why - the
# panel's question is the 20.85 % row, and it costs ~2 s per board.
RERANK_DEFAULT = ARCHIVE_CKPT_DIR / "llm-lora-rerank5"
# Still selectable in the dropdown and still pinned by the ensemble tests: a
# `.json` manifest is a model, and `read_ensemble` + logit-averaging is the
# contract the +1.33 pp was measured under. Do not "simplify" it to one .pt.
ENSEMBLE_CKPT = ARCHIVE_CKPT_DIR / "nextpick-proto-ensemble" / "ensemble.json"
UPLOAD_DIR = FRONTEND_DIR / "uploaded"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024      # generous; ckpts are 20-150 MB
LOAD_LOCK = threading.Lock()              # serialize runtime model swaps
# dDragon static champion art (model patch 16.20)
DDRAGON_VERSION = "16.20.1"
DDRAGON_CDN = "https://ddragon.leagueoflegends.com/cdn/"
DDRAGON_CACHE = FRONTEND_DIR / ".cache" / ("champion_%s.json" % DDRAGON_VERSION)
# Real pro games for the walkthrough. Two caches exist and they answer different
# questions:
#   pro_drafts.pt       the FROZEN corpus (CORPUS_MAX_DATE=2026-11-08). Every
#                       recorded metric is measured on it, so it must not move.
#   pro_drafts_full.pt  the same scan with no freeze (the test split's corpus),
#                       a strict superset: +21 games through 2026-11-13.
# Browsing real games is not a measurement, so the walkthrough reads the
# superset when it is on disk and falls back to the frozen cache otherwise -
# that keeps the dropdown showing the newest series (the Worlds final, the LCP
# games on 16.22) without touching the freeze or any recorded number.
PRO_DRAFTS_PT = ROOT / "data" / "pro play" / "pro_drafts.pt"
PRO_DRAFTS_FULL_PT = ROOT / "data" / "pro play" / "pro_drafts_full.pt"
# The repo ROOT goes on the path so `import solo_queue` / `import pro` resolve;
# the base encoder lives in `solo_queue/model/pretraining/base_model.py`
# now, so the package path is the only stable way to import it.
sys.path.insert(0, str(ROOT))

from solo_queue.model.pretraining.base_model import (SiameseBase,  # noqa: E402
                                           load_encoder_state,
                                           attr_matrix_for_config)
from solo_queue.model.policy import infer  # noqa: E402  (next-move policy)
from pro.model.agent import draft_state, model_tools  # noqa: E402
from pro.model.agent import schema as agent_schema  # noqa: E402
from pro.model.agent import llm_client  # noqa: E402  (the agent's model)
from pro.model.agent.engine import Engine  # noqa: E402

ROLE_ROWS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
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


_STATE_LOCK = threading.Lock()
_STATE_STORE = None

# A sandbox is a list of states (one per move, for undo), so an unbounded store
# is an unbounded leak in a process that never restarts. 32 is far more than an
# exploration needs and small enough that a runaway loop cannot exhaust memory.
MAX_SANDBOXES = 32


def state_store():
    """The live draft state, and the sandboxes an assistant may explore in.

    Needs the name table and nothing else, so a board can be published before a
    model is loaded and `/api/state` keeps working in a `--no-policy` install.
    The models arrive later, through `_bind_models`, which swaps the engine under
    every state the store already holds.

    This is the one object the frontend and the agent share. The frontend writes
    it (`_publish_state`, once per request that carries a board); the agent reads
    it as `real` and clones it to explore. Nothing on the agent side can write
    `real`, which is why the panel can be left open while a hypothetical is
    being tested.
    """
    global _STATE_STORE
    if _STATE_STORE is None:
        with _STATE_LOCK:
            if _STATE_STORE is None:
                names = APP.get("names_by_idx")
                if not names:
                    raise ValueError("champion names are not loaded yet")
                champs = model_tools.ChampIndex(names)
                _STATE_STORE = draft_state.Store(champs, engine=_engine(champs))
    return _STATE_STORE


def _engine(champs=None):
    """The models the state service can reach, built from the two live slots.

    One engine per store, not one per request: the role assignment is a table
    lookup that every state carries, so rebuilding the binding on every call
    would recompute it for every sandbox as well.

    The masker is the role slot and the predictor the next-move slot, the two
    slots the CLI selects at startup. Either may be None - an
    engine with no predictor still assigns roles, and one with no masker still
    ranks the move on the clock.
    """
    champs = champs or model_tools.ChampIndex(APP["names_by_idx"])
    return Engine(champs, predictor=APP.get("policy"),
                  masker=APP.get("masked"), patch=APP.get("patch"))


def _bind_models():
    """Re-attach the engine after a checkpoint swap.

    The board outlives the weights: a model loaded while a draft is open has to
    answer about *that* draft, so the store is rebound rather than rebuilt. The
    sandboxes are rebound too - an exploration made under the old model would
    otherwise keep answering with it.
    """
    store = state_store()
    return store.attach(_engine(store.champs))


def _agent_session(ablate=(), max_calls=None):
    """A `Session` over the live store, its engine, and the corpus.

    The corpus database is optional: without it the SQL tool is simply not
    offered, and the model and state tools still answer. The store carries the
    engine, so this does not pass a predictor - doing that would build a second
    engine and silently drop the masker.

    `ablate` and `max_calls` are per turn: the free-form agent turn asks
    anything, while the one-click suggestion runs the measured reranking arm
    (no SQL, a smaller budget) through the same session and the same store.
    """
    from pro.model.agent.session import Session

    con = None
    if Path(agent_schema.DB_PATH).exists():
        con = agent_schema.connect(str(agent_schema.DB_PATH))
    return Session(con=con, store=state_store(), ablate=ablate,
                   max_calls=int(max_calls or APP.get("agent_calls", 24)))


def _head_top_legal(st, k=20):
    """The head's own top-k legal champions, as {"name", "p_head"}.

    Used by `/api/agent-rank` as the parser's fallback fill - any slot the agent
    left unnamed is filled from here, so the panel never shows fewer than k names
    and never shows one the head did not produce. `predict_next_move` already
    masks bans and fearless, so `top` is the legal pool.
    """
    try:
        res = st.predict_next_move(k=k)
    except (ValueError, AttributeError):
        return []
    return [{"name": c["name"], "p_head": c["p_head"]}
            for c in (res.get("top") or [])]


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
    paths = sorted(p for g in CKPT_GLOB for p in glob.glob(g))
    if not paths:
        raise FileNotFoundError("no checkpoint under %s" % (CKPT_GLOB,))
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


class Predictor:
    """Wraps the loaded masking checkpoint; one forward -> one slot completed.

    The role assignment is not here any more. It comes from
    `data/champ_roles.json` (`model_tools.assignment`), which agrees with the
    corpus 96 % of the time against this model's role head at 67 %, and which
    cannot give two champions on one team the same role. What is left is the one
    question only this checkpoint can answer: a role-conditioned completion
    during the ban phase, when the next-move head cannot be asked.
    """

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
        self.model = SiameseBase(
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

    @torch.no_grad()
    def complete_slot(self, champ_idx, role_idx, slot):
        """Masked completion of one pick slot, with its role given.

        Hide `slot` (set to the [MASK] id) and return its full champion logits
        + softmax. `model_tools.predict_in_lane` calls this when the board is in
        its ban phase - the one question the next-move head cannot be asked.

        The patch input is omitted so the model runs with its UNK patch, the
        same convention the rest of the frontend uses for masked models: this
        checkpoint was trained without a patch table.
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
    # An ensemble has no single val run: it is scored on the rows neither it
    # nor the checkpoint it replaced had trained on. Publish that number as
    # `eval` (with its set and n) instead of passing it off as a val score.
    ens = getattr(pred, "ensemble", None)
    if ens:
        info["ensemble"] = {"members": ens.get("members", []),
                            "eval": ens.get("eval") or {}}
    return info


def resolve_ckpt_path(spec, confine=True):
    """Checkpoint spec -> absolute .pt / ensemble .json / reranker directory.

    A file is a single model; a directory is one only when its name says
    `llm-`, because that is the one artefact that is a *folder*: the 4B
    trunk lives in the image and what is on disk is `adapter/` plus
    `projector.pt`. Anything else that is a directory is still a mistake, and
    the refusal names the file inside it rather than "no such file".

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
    if p.is_dir():
        if p.name.startswith("llm-"):
            return p
        raise ValueError("%s is a directory - name the checkpoint inside it "
                         "(best.pt or ensemble.json)" % p.name)
    if p.suffix.lower() not in (".pt", ".json"):
        raise ValueError("expected a .pt checkpoint or an ensemble .json, "
                         "got %s" % p.name)
    if not p.is_file():
        raise FileNotFoundError("no checkpoint at %s" % p)
    return p


def read_ensemble(path):
    """Ensemble manifest -> (member paths, eval block, note).

    A manifest is the artefact that names a set of checkpoints to average. It
    lives next to its members so `<track>/model/checkpoints/<name>/` stays the
    single contract, and members are resolved relative to it (never absolute,
    never outside the project).
    """
    man = json.loads(Path(path).read_text("utf-8"))
    members = man.get("members") or []
    if not members:
        raise ValueError("ensemble manifest %s lists no members" % path)
    base = Path(path).resolve().parent
    out = []
    for m in members:
        mp = (base / m).resolve()
        try:
            mp.relative_to(ROOT.resolve())
        except ValueError:
            raise ValueError("ensemble member outside project: %s" % m)
        if mp.suffix.lower() != ".pt" or not mp.is_file():
            raise FileNotFoundError("ensemble member missing: %s" % m)
        out.append(mp)
    return out, man.get("eval") or {}, man.get("note") or ""


def _ckpt_roles(path, kind):
    """Which model slots a checkpoint can fill, from its path alone.

    There are two slots (role model / next-move model) and offering the
    whole list in both would make most picks fail on load. torch.load'ing
    every checkpoint at startup just to classify them is not worth it, and
    it is not necessary: every kept artefact lives in
    <track>/model/checkpoints/<name>/, and the name states the inference it
    serves. The track does not affect the classification - `masking-*` is the
    role board whether it was trained on solo queue or on pro drafts.

      masking-*    masked champion + role completion -> the masker
      nextpick-*   next-move ranking                -> the head
      value-*      outcome head only                -> neither (not servable)
      llm-*        the 4B reranker over the head's  -> the next-move slot
                   shortlist (a directory)

    An `llm-*` folder is the reranker: it answers `next_move` by re-ordering the
    head's shortlist, and it needs `transformers`, `peft`, `bitsandbytes` and a
    GPU, so it is only reachable from the `llm` image (`serve.sh --rerank`).
    Offering it from the base image would list a model the server cannot build,
    so `make_predictor` reports the missing package instead of raising on import.

    Paths outside that layout (uploads, checkpoints from before the rename)
    fall back to the old family rules, and anything still unrecognised is
    offered in both slots and validated on load.
    """
    if kind == "value":
        return []                       # outcome head only, not servable
    if Path(path).name == "projector.pt":
        # One piece of the reranker, not a model: the directory is the model.
        return []
    text = Path(path).as_posix()
    # `(?:/|$)`: a reranker is the directory itself, so its path ends at the
    # name and the old `/checkpoints/<name>/` pattern matched nothing.
    m = re.search(r"/checkpoints/([^/]+)(?:/|$)", text)
    if m:
        name = m.group(1)
        if name.startswith("masking-"):
            return ["masked"]
        if name.startswith("nextpick-"):
            return ["policy"]
        if name.startswith("llm-"):
            return ["policy"]          # the reranker fills the next-move slot
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


def slot_of(pred):
    """Which of the two slots a built predictor fills.

    The role board is the masker and nothing else; every other kind - the head,
    an ensemble, the reranker - answers `next_move`, so it belongs in the
    next-move slot.

    Written once because three places ask the question (`load_predictor` and
    both `/api/load` and `/api/upload` replies), and both one-liners that were
    tried here were wrong for one of the three families:

      `"policy" if kind == "policy" else "masked"`  - right for the masker and
        the head, wrong for the reranker (kind "rerank" -> filed as a masker)
      `"masked" if kind == "masked" else "policy"`  - right for the reranker,
        wrong for the masker, because `Predictor.kind` is "pretrain"/"finetune"
        (it names the *stage* the checkpoint came from, not the inference it
        serves) and no checkpoint ever reports "masked". That version made
        `load_predictor(role="masked")` raise on every real startup.

    So the masker is "everything that is a `Predictor`", and the test is on the
    kinds that class produces.
    """
    kind = getattr(pred, "kind", None)
    return "masked" if kind in ("masked", "pretrain", "finetune") else "policy"


def list_checkpoints():
    """Every .pt under the known ckpt dirs; cheap (payload not read).

    `latest.pt` is skipped: it is whatever the last epoch wrote, which is
    only useful for resuming, not for serving. The list should contain the
    checkpoints that were selected on validation.
    """
    out = []
    for kind, pattern in CKPT_PATTERNS:
        for hit in sorted(glob.glob(pattern)):
            pp = Path(hit)
            if pp.name == "latest.pt":
                continue
            rel = _rel(hit)
            if pp.name == "projector.pt":
                # The retired reranker's entry point: a 4B adapter this server
                # cannot answer from. Skipped rather than offered and refused.
                continue
            label = ("%s/%s" % (pp.parent.name, pp.name)
                     if kind == "checkpoints"
                     else "[%s] %s/%s" % (kind, pp.parent.name, pp.name))
            out.append({"path": rel, "kind": kind,
                        "roles": _ckpt_roles(hit, kind),
                        "label": label})
    return out


_MODEL_CACHE = {}                      # (path, device) -> predictor


def make_reranker(path, device=None, head_path=None):
    """Build the 4B reranker, or say why this image cannot.

    The import is inside the function on purpose: `transformers`/`peft` live in
    the `llm` image only, and `frontend/server.py` has to import cleanly in the
    base one. A missing package is a one-line explanation, not a traceback at
    startup.

    `head_path` is the proposer the reranker re-orders. Callers that already
    resolved the head pass it; otherwise it is whatever the next-move slot
    currently holds (so a runtime `/api/load` of a reranker keeps the head the
    panel is already using) and `POLICY_DEFAULT` is the last resort.
    """
    try:
        from pro.model.archive.llm.rerank_predictor import RerankPredictor
    except Exception as exc:                       # ImportError, mostly
        raise ValueError(
            "%s is the 4B reranker and this image has no LLM stack (%s). "
            "Serve it with `bash frontend/serve.sh --rerank`, which uses the "
            "llm image and --device cuda." % (Path(path).name, exc))
    if head_path is None:
        pol = APP.get("policy")
        if getattr(pol, "kind", None) == "rerank":
            head_path = pol.head_path
        elif getattr(pol, "kind", None) == "policy":
            # `ckpt_path`, not `label`: an ensemble's label is the manifest
            # `.json`, which is not something `PolicyPredictor` can load twice.
            cp = Path(str(getattr(pol, "ckpt_path", "") or ""))
            if cp.suffix.lower() == ".pt":
                head_path = cp
    head_path = Path(head_path) if head_path else POLICY_DEFAULT
    return RerankPredictor(path, head_path, device=device or "cuda")


def make_predictor(path, device=None, head_path=None):
    """Build (or fetch a cached) predictor for a checkpoint path.

    One file can hold three different model families, so the payload is
    read once here and dispatched: a policy checkpoint becomes a
    `PolicyPredictor`, everything else stays a masked `Predictor`. An
    `ensemble.json` manifest becomes a `PolicyPredictor` over its members
    (mean of the policy logits - the measured quantity). A *directory* is the
    reranker (`adapter/` + `projector.pt`).

    The result is cached per (path, device) so a swap back to a previously
    loaded checkpoint does not re-read its weights.
    """
    device = device or APP.get("device", "cpu")
    key = (str(path), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if Path(path).is_dir():
        # A reranker: adapter/ + projector.pt + a 4B trunk.
        pred = make_reranker(path, device=device, head_path=head_path)
        _MODEL_CACHE[key] = pred
        return pred
    if Path(path).name == "projector.pt":
        raise ValueError("%s is the retired local reranker's projector - the "
                         "agent asks the head and reads the corpus itself, so "
                         "this checkpoint has nothing to answer"
                         % Path(path).parent.name)
    if Path(path).suffix.lower() == ".json":
        members, ev, note = read_ensemble(path)
        pred = infer.PolicyPredictor(members, device=device, label=path)
        pred.ensemble = {"members": [m.name for m in members],
                         "eval": ev, "note": note}
        pred.patch_vocab = pred.patch_vocab or []
        _MODEL_CACHE[key] = pred
        return pred
    payload = torch.load(str(path), map_location="cpu")
    kind = peek_kind(payload)
    if kind == "policy":
        pred = infer.PolicyPredictor(path, device=device, payload=payload)
    else:
        pred = Predictor(path, device=device, payload=payload)
    # The patch vocabulary is stored at the checkpoint top level (both
    # families); keep it on the predictor so the state's patch label can be
    # mapped to the embedding row the model was trained with.
    pred.patch_vocab = (payload.get("patch_vocab")
                        or (payload.get("config") or {}).get("patch_vocab"))
    _MODEL_CACHE[key] = pred
    return pred


def load_predictor(path, device=None, role="auto"):
    """Build a model first, then swap it into its slot (the old one stays on
    failure) - concurrent requests keep a valid model.

    `role` files the model: a policy checkpoint is the head, every other kind is
    the masker. The two slots are independent, so a masking checkpoint and a
    next-move head are loaded at the same time; "auto" picks the slot from the
    checkpoint kind.

    The store is rebound after the swap, not before: a board published under the
    old model has to be answerable by the new one, and an engine that still held
    the old weights would answer `/api/suggest` with a model the header does
    not name. Only the local `/api/load` route can do this now - the browser
    has no model picker, so a normal session never swaps weights mid-draft.
    """
    with LOAD_LOCK:
        pred = make_predictor(path, device=device)
        slot = slot_of(pred)
        if role in ("masked", "policy") and role != slot:
            raise ValueError("%s is a %s checkpoint, not a %s one"
                             % (Path(path).name, pred.kind, role))
        APP[slot] = pred
        if _STATE_STORE is not None:
            _bind_models()
    return pred


_DRAFT_ROWS = None


def load_drafts():
    """Lazy-load the cached pro drafts (~46.9k rows) on first use.

    Prefers the unfrozen superset (`pro_drafts_full.pt`) so the walkthrough can
    show games played after `CORPUS_MAX_DATE`; the frozen cache is the fallback.
    Both are `{"rows": [...]}` with the same row schema, so `i` stays an index
    into whichever file was loaded - callers must not mix ids across a reload.
    """
    global _DRAFT_ROWS
    if _DRAFT_ROWS is None:
        path = (PRO_DRAFTS_FULL_PT if PRO_DRAFTS_FULL_PT.exists()
                else PRO_DRAFTS_PT)
        payload = torch.load(str(path), map_location="cpu")
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


def _row_key(role):
    """A model role name -> the UI's `TOP..SUPPORT` key.

    The two vocabularies are not the same list in a different case: the model's
    is `model_tools.ROLES` (top, jungle, mid, bot, support) and the UI's is
    `ROLE_ROWS` (TOP, JUNGLE, MIDDLE, BOTTOM, SUPPORT). This is the one edge
    that converts, so a role key can never be compared across the two.
    """
    up = str(role).upper()
    return up if up in ROLE_ROWS else ROLE_ROWS[model_tools.ROLES.index(role)]


def _suggest_model(engine):
    """What answered the role board: the head, else the masker, else nothing."""
    for pred in (engine.predictor, engine.masker):
        if pred is not None:
            return model_info(pred)
    return None


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
            # solo_queue/model/policy/test_infer.py), so publish it rather than copying
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
        if path == "/api/state":
            self._handle_state()
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
        elif route == "/api/load":
            self._handle_load()
        elif route == "/api/upload":
            self._handle_upload()
        elif route == "/api/state":
            self._handle_state()
        elif route == "/api/agent":
            self._handle_agent()
        elif route == "/api/agent-suggest":
            self._handle_agent_suggest()
        elif route == "/api/agent-rank":
            self._handle_agent_rank()
        else:
            self._send(404, {"error": "unknown endpoint"})

    # ------------------------------------------------------------- state
    def _publish_state(self, payload):
        """Publish the board the user is looking at as the live state.

        Called by every handler that receives a full board, so `real` is never
        more than one request stale. The UI speaks riot ids and the store speaks
        champion indices; the mapping stops at this edge, as it does everywhere
        else in the frontend.

        A board the store refuses is *recorded*, not raised: there is no
        board-as-argument fallback left to fall back to, so the honest answer is
        to keep the last good state and let the panel show `state_error` instead
        of turning one unplayable board into a 400 on every route.
        """
        mapping = APP.get("mapping") or {}
        try:
            picks = payload.get("picks") or [None] * N_SLOTS
            picks_idx = [None if p is None else mapping[p] for p in picks]
            flat = [mapping[b] for b in (payload.get("bans") or [])
                    if b in mapping]
            blue = [mapping[b] for b in (payload.get("bans_blue") or [])
                    if b in mapping]
            red = [mapping[b] for b in (payload.get("bans_red") or [])
                   if b in mapping]
            # The split only when it carries bans: a payload sending both is
            # normal, but an empty pair next to a populated flat list means the
            # split is the stale half and the flat list is the board.
            split = bool(blue or red)
            # The fearless pool: champions picked in earlier games of this
            # series. It is not derivable from the board, so if it is dropped
            # here the head is free to suggest one of them - which it did.
            #
            # Absent is not empty. The panel sends `fearless` on every board it
            # publishes, but a caller that forwards a board without the pool is
            # saying nothing about it, and reading that as "no pool" would clear
            # a real one. `[]` is the explicit "this series has no pool".
            raw = payload.get("fearless")
            pool = (list(state_store().get(draft_state.REAL_ID).unavailable)
                    if raw is None
                    else [mapping[f] for f in raw if f in mapping])
            state, _old = state_store().set_board(
                picks_idx, flat,
                red_first=bool(int(payload.get("first_pick") or 0)),
                patch=self._state_patch(payload),
                bans_blue=blue if split else None,
                bans_red=red if split else None,
                unavailable=pool,
                series=self._series_lines(payload))
        except (ValueError, KeyError, TypeError) as exc:
            APP["state_error"] = str(exc)
            return None
        APP["state_error"] = None
        return state

    def _series_lines(self, payload):
        """The series bar as lines the state renders and `get_state` reports.

        The board itself cannot say which game of a series it is: ten picks and
        ten bans look the same in G1 and G5, and the fearless pool is only the
        *effect* of the earlier games on legality. The panel sends the game
        number, the number of games and a compact per-game count; those are
        rendered here from numbers only, so a client cannot put prose into the
        model's prompt.

        Absent is not empty. Every board route publishes the series, but a
        partial caller that forwards a board without it is saying nothing about
        the series, and clearing the lines would make the next agent turn
        describe G1 of a G3 draft.
        """
        raw = payload.get("series")
        if raw is None:
            return list(state_store().get(draft_state.REAL_ID).series)
        if not isinstance(raw, dict):
            return []

        def num(value, hi):
            try:
                return max(0, min(hi, int(value)))
            except (TypeError, ValueError):
                return 0

        game, of = num(raw.get("game"), 99), num(raw.get("of"), 99)
        if not game or not of or game > of:
            return []
        lines = ["series: game %d of %d" % (game, of)]
        games = raw.get("games")
        if isinstance(games, list):
            rows = []
            for i, g in enumerate(games[:12]):
                if not isinstance(g, dict):
                    continue
                rows.append("G%d %d/10%s" % (i + 1, num(g.get("picks"), 10),
                                             " (closed)" if g.get("closed")
                                             else ""))
            if rows:
                lines.append("series games: " + ", ".join(rows))
        if game > 1:
            lines.append("series: champions picked in games 1-%d are "
                         "unavailable here (the fearless pool above lists them)"
                         % (game - 1))
        else:
            lines.append("series: game 1 - nothing is fearless-unavailable yet")
        return lines

    def _state_patch(self, payload):
        """The patch *label* the head will actually condition on.

        The UI sends a `patch_id` (a row of the checkpoint's vocabulary), not a
        label, and the head silently falls back to its default when none arrives.
        A state that said "patch: not given" while the head answered 16.20 would
        make the agent attribute a patch-specific ranking to no patch at all, so
        the label is resolved here through the same `patch_key_for` the rest of
        the frontend uses.
        """
        if payload.get("patch"):
            return payload.get("patch")
        pred = APP.get("policy")
        pid = payload.get("patch_id")
        if pid is None and hasattr(pred, "default_patch_id"):
            pid = pred.default_patch_id()
        key_for = getattr(pred, "patch_key_for", None)
        return key_for(pid) if key_for is not None else None

    def _state_payload(self):
        """One shape for `real` and for the sandboxes, so the panel cannot drift.

        Built from `Store.list()` rather than from `summary()` so `label`,
        `depth` and `readonly` are on `real` too - the panel decides whether to
        offer an edit button from that flag, and a missing key would read as
        false.
        """
        store = state_store()
        rows = store.list()
        for row in rows:
            row["rendered"] = store.get(row["state"]).render(row["state"],
                                                             store.display)
        return {"real": next(r for r in rows
                             if r["state"] == draft_state.REAL_ID),
                "states": rows, "error": APP.get("state_error")}

    def _handle_state(self):
        """GET -> the live state and its sandboxes. POST -> one operation.

        POST {op, state, champion, name}:

            clone   copy `state` (default real) into a new sandbox
            apply   play `champion` (a dDragon name) into `state`
            undo    take back the last move in `state`
            drop    delete a sandbox
            reset   drop every sandbox, keeping `real`

        `real` is refused for apply/undo/drop. It is written by `_publish_state`
        and by nothing else, which is what lets a panel explore a hypothetical
        while the board on screen keeps moving.
        """
        if self.command == "GET":
            try:
                self._send(200, self._state_payload())
            except ValueError as exc:          # names not loaded yet
                self._send(400, {"error": str(exc)})
            return
        try:
            payload = self._read_json()
            op = payload.get("op")
            store = state_store()
            sid = payload.get("state") or draft_state.REAL_ID
            if op == "clone":
                if len(store.ids()) > MAX_SANDBOXES:
                    raise ValueError("too many sandboxes (%d) - drop one first"
                                     % len(store.ids()))
                out = {"state": store.clone(sid, payload.get("name"))}
            elif op == "apply":
                st = store.apply(sid, payload.get("champion"),
                                 kind=payload.get("kind"),
                                 side=payload.get("side"))
                out = {"state": sid, "move_idx": st.move_idx}
            elif op == "undo":
                out = {"state": sid, "move_idx": store.undo(sid).move_idx}
            elif op == "drop":
                store.drop(sid)
                out = {"dropped": sid}
            elif op == "reset":
                for other in store.ids():
                    if other != draft_state.REAL_ID:
                        store.drop(other)
                out = {"dropped": "all sandboxes"}
            else:
                raise ValueError("unknown op %r (clone, apply, undo, drop, "
                                 "reset)" % (op,))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        out.update(self._state_payload())
        self._send(200, out)

    def _handle_agent(self):
        """POST {message, picks, bans, first_pick, fearless} -> one agent turn.

        The only route that calls a remote model, and the only one whose answer
        is prose. Three things have to be true, and each missing one is a
        sentence rather than a traceback: `langgraph` importable, a key in the
        environment, a model loaded. The board comes from the store, not from
        the payload - that is what `/api/state` is for, and it is why the model
        is never asked to retype nine champion names.

        The board in the payload is published first, so a turn always answers
        about the board on screen. Everything the agent does to it after that
        happens in a sandbox: `real` is written here and nowhere else.
        """
        try:
            from pro.model.agent import loop
        except ImportError as exc:
            self._send(503, {"error": "the agent loop needs langgraph: %s" % exc})
            return
        try:
            payload = self._read_json()
            message = (payload.get("message") or "").strip()
            if not message:
                raise ValueError("empty message")
            self._publish_state(payload)
            sess = _agent_session()
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        try:
            # `agent_client` is a seam: a test (or a local model) can answer
            # without a network, and with nothing set the environment decides.
            out = loop.run(sess, message, client=APP.get("agent_client"),
                           state_id=draft_state.REAL_ID,
                           max_steps=int(payload.get("max_steps") or 8))
        except Exception as exc:  # noqa: BLE001 - one panel, not the server
            self._send(500, {"error": "agent failed: %s" % exc})
            return
        # The board the turn was about, and every sandbox it left behind: an
        # exploration is only useful if the user can see what was tried.
        state = self._state_payload()
        out["state"] = state["real"]
        out["states"] = state["states"]
        self._send(200, out)

    def _handle_agent_suggest(self):
        """POST {board...} -> one agent turn that commits to a single move.

        The button in the agent panel. It is *not* `/api/next-move` with prose
        on top: that route asks the head for a ranked list, while this one asks
        the agent - the same configuration `eval_api_agent.py` measured - to
        read the held state, call `get_state`, `predict_roles` and
        `predict_next_move` with a k it chooses, and commit to one champion.
        That is the reranking job the retired local adapter used to do, done
        with tools and state management instead of a 4B checkpoint.

        The whole answer is a claim about *this* board, so a board the store
        refuses is a 400 rather than a stale answer: exactly the rule
        `/api/next-move` follows, and the opposite of `/api/suggest`'s lenient
        role board. Everything else degrades to a sentence - no langgraph
        (503), no key (`stop="no client"`), no head (`head=None`).

        The recommendation is parsed from the answer's final `PICK:`/`BAN:`
        line so the panel can show it as a move; the head's own #1 is computed
        here, on `real`, for the fidelity chip. Fidelity is this component's
        metric, accuracy belongs to the head - nothing here claims either.
        """
        try:
            from pro.model.agent import loop, suggest
        except ImportError as exc:
            self._send(503, {"error": "the agent loop needs langgraph: %s" % exc})
            return
        try:
            payload = self._read_json()
            if self._publish_state(payload) is None:
                raise ValueError(APP.get("state_error")
                                 or "the store refused this board")
            store = state_store()
            st = store.get(draft_state.REAL_ID)
            sess = _agent_session(ablate=suggest.ABLATE,
                                  max_calls=suggest.MAX_CALLS)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return

        state = self._state_payload()
        if st.to_move() is None:
            self._send(200, {
                "question": suggest.QUESTION, "answer": None,
                "stop": "draft complete", "model": None,
                "state": state["real"], "states": state["states"],
                "complete": True, "calls": [], "summary": sess.summary(),
                "suggestion": suggest.parse_answer(None, store.champs, st),
                "head": None, "config": suggest.config(),
                "error": state["error"]})
            return
        try:
            out = loop.run(sess, suggest.QUESTION,
                           client=APP.get("agent_client"),
                           system=suggest.system(),
                           state_id=draft_state.REAL_ID,
                           max_steps=suggest.MAX_STEPS)
        except Exception as exc:  # noqa: BLE001 - one panel, not the server
            self._send(500, {"error": "agent failed: %s" % exc})
            return
        # Re-read the state after the turn: the agent may have left sandboxes
        # behind, and an exploration is only useful if the panel can see it.
        state = self._state_payload()
        out["state"] = state["real"]
        out["states"] = state["states"]
        out["complete"] = False
        out["error"] = out.get("error") or state["error"]
        out["config"] = suggest.config()
        # `real` is read-only to the tools, but a checkpoint swap mid-turn
        # rebinds the engine under every state; re-read it so the fidelity chip
        # and the answer are about the weights that are live now.
        live = store.get(draft_state.REAL_ID)
        out["suggestion"] = suggest.parse_answer(out.get("answer"),
                                                 store.champs, live)
        out["head"] = suggest.head_fidelity(live, store.champs,
                                            out["suggestion"])
        self._send(200, out)

    def _handle_agent_rank(self):
        """POST {board...} -> the agent's top-k rerank of the head's shortlist.

        The ranking arm of `suggest.py`, exposed for the panel's "agent
        suggestions" column. It is the same ablated, explained-prompt agent and
        the same budget as the single-move `/api/agent-suggest`, but asked to
        *order* the head's top-k rather than commit to one. The list comes back
        honest by construction: `parse_ranking` only accepts champions the head
        listed, and fills any slot the agent left unnamed from the head's own
        legal shortlist, so the panel always shows `RANK_K` names and never
        invents one the head did not produce.

        `head_top` is fetched at k=20 and trimmed to the legal pool (the head
        already masks bans and fearless), so the agent has room to promote a deep
        pick without the parser falling back to the head's order everywhere.

        Like the other agent routes it publishes the board first and degrades to a
        sentence - no langgraph (503), no key (`stop="no client"`, and the head's
        own list with `source="head"`), no head (`suggestions` from the parser
        alone). A complete draft returns an empty list with `complete: true`.
        """
        try:
            from pro.model.agent import loop, suggest
        except ImportError as exc:
            self._send(503, {"error": "the agent loop needs langgraph: %s" % exc})
            return
        try:
            payload = self._read_json()
            if self._publish_state(payload) is None:
                raise ValueError(APP.get("state_error")
                                 or "the store refused this board")
            store = state_store()
            st = store.get(draft_state.REAL_ID)
            sess = _agent_session(ablate=suggest.RANK_ABLATE,
                                  max_calls=suggest.RANK_CALLS)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return

        state = self._state_payload()
        if st.to_move() is None:
            self._send(200, {
                "question": suggest.rank_question(suggest.RANK_K),
                "suggestions": [], "complete": True,
                "answer": None,
                "state": state["real"], "states": state["states"],
                "stop": "draft complete", "model": None,
                "summary": sess.summary(), "config": suggest.rank_config(),
                "head": None, "error": state["error"]})
            return

        # The head's own legal shortlist, for the parser's fallback fill.
        k = suggest.RANK_K
        head_top = _head_top_legal(st, k=20)
        # The chat the user has been having, carried as rerank context so a
        # conversation that changed the agent's mind changes this list too.
        seed = payload.get("messages") or None
        try:
            out = loop.run(sess,
                           suggest.rank_question(k, st.to_move()["kind"]),
                           client=APP.get("agent_client"),
                           system=suggest.system(),
                           state_id=draft_state.REAL_ID,
                           max_steps=suggest.RANK_STEPS, seed=seed)
        except Exception as exc:  # noqa: BLE001 - one panel, not the server
            self._send(500, {"error": "agent failed: %s" % exc})
            return

        # Re-read the state: the agent may have cloned it for a hypothetical.
        state = self._state_payload()
        out["state"] = state["real"]
        out["states"] = state["states"]
        out["complete"] = False
        out["error"] = out.get("error") or state["error"]
        out["config"] = suggest.rank_config()
        live = store.get(draft_state.REAL_ID)
        ranked = suggest.parse_ranking(out.get("answer"), store.champs, live,
                                       k=k, head_top=head_top)
        out["suggestions"] = ranked["suggestions"]
        out["parse"] = ranked["parse"]
        # Fidelity to the head's #1, recomputed on the live board, for the chip.
        out["head"] = suggest.head_fidelity(live, store.champs, None)
        self._send(200, out)

    def _handle_suggest(self):
        """POST {picks, bans, first_pick, fearless, k} -> the role board.

        One `predict_in_lane` question per open cell: for each side, for each
        role the assignment says is still unfilled, "which champion would the
        model take for that role". The mask is applied to the logits *before*
        the top-k, so the answer is a real ranking inside the role rather than
        an unconstrained list filtered afterwards (which empties the cell 13 %
        of the time, because the head's five favourites often do not play it).

        The result says which model answered, per cell:

          head     the next-move head, on a pick move with the bans complete
          masker   a masking checkpoint, during the ban phase, where the head
                   has no in-distribution form for the question

        The *assignment* is not a model answer. It comes from
        `data/champ_roles.json`, is one-to-one per side, and travels on the
        state - so this route reads it instead of computing a second, possibly
        different one.

        A cell that cannot be answered is left out with its reason in `notes`
        rather than failing the board: one refused lane should not blank the
        other nine.

        Role keys are the UI's `TOP..SUPPORT`; the model's own vocabulary is
        lowercase (`model_tools.ROLES`). This handler is the one place that
        converts between them, and the client normalises them through
        `/api/champions`' `roles` list - never by assuming a numeric index.
        """
        try:
            payload = self._read_json()
            self._publish_state(payload)
            store = state_store()
            st = store.get(draft_state.REAL_ID)
            engine = store.engine
            if engine.predictor is None and engine.masker is None:
                raise ValueError(
                    "no model loaded - the role board needs either the "
                    "next-move head (best_policy.pt) or a masking checkpoint "
                    "(best.pt)")
            k = max(1, min(5, int(payload.get("k") or 3)))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return

        assignment = [None] * N_SLOTS
        conf = [None] * N_SLOTS
        for row in st.roles:
            assignment[row.slot] = _row_key(row.role)
            conf[row.slot] = row.p_lane
        out = {"blue": {}, "red": {},
               "roles": [r.as_dict() for r in st.roles],
               "assignment": assignment, "conf": conf,
               "open_roles": {side: [_row_key(r) for r in roles]
                              for side, roles in st.open_roles().items()},
               # The head masks the pool before its top-k, so this is a real
               # condition on the pick path and a post-filter on the masking
               # path. The count is published either way; `methods` says which.
               "fearless_filtered": len(st.unavailable),
               "complete": st.to_move() is None,
               "methods": {}, "notes": {}, "method": "none"}
        if st.to_move() is None:
            self._send(200, dict(out, model=_suggest_model(engine)))
            return

        methods = set()
        for side in ("blue", "red"):
            for role in st.open_roles()[side]:
                key = _row_key(role)
                try:
                    res = st.predict_next_move(k=k, role=role, side=side)
                except ValueError as exc:
                    out["notes"]["%s/%s" % (side, key)] = str(exc)
                    continue
                methods.add(res["method"])
                out["methods"]["%s/%s" % (side, key)] = res["method"]
                out[side][key] = [{"name": c["name"],
                                   "idx": store.champs.resolve(c["name"]),
                                   "prob": c["p_in_lane"]}
                                  for c in res["top"]]
        # Same champion under two roles of one team: it plays both, so the cell
        # is a real choice rather than a duplicate of the other one.
        for side in ("blue", "red"):
            seen = {}
            for cands in out[side].values():
                for c in cands:
                    seen[c["name"]] = seen.get(c["name"], 0) + 1
            for cands in out[side].values():
                for c in cands:
                    if seen[c["name"]] > 1:
                        c["flex"] = True
        out["method"] = ("mixed" if len(methods) > 1
                         else (methods.pop() if methods else "none"))
        self._send(200, dict(out, model=_suggest_model(engine)))

    def _handle_next_move(self):
        """POST {picks, bans, first_pick, fearless, patch_id, k} -> the head.

        The same call the agent makes - `predict_next_move` on the live state -
        rendered instead of narrated. It publishes the board first, so this
        answer and `/api/suggest` describe one board rather than two.

        One answer per board: the ranked list for the side on the clock. There
        is no separate ban ranking - a pro-like ban is the denial decision, and
        the rollout that used to approximate it was removed (STATUS.md §5/§8).
        """
        try:
            payload = self._read_json()
            # `/api/suggest` may keep the last good board and let the panel show
            # `state_error`; this route may not. Its whole answer is a
            # distribution over *this* board, and one computed for a board the
            # store refused is indistinguishable from a good one.
            if self._publish_state(payload) is None:
                raise ValueError(APP.get("state_error")
                                 or "the store refused this board")
            store = state_store()
            st = store.get(draft_state.REAL_ID)
            engine = store.engine
            if engine.predictor is None:
                raise ValueError(
                    "no next-move model loaded - load a best_policy.pt to rank "
                    "the move on the clock")
            k = max(1, min(20, int(payload.get("k") or 8)))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        if st.to_move() is None:
            self._send(200, {"model": model_info(engine.predictor),
                             "state": {"move_idx": st.move_idx, "kind": None,
                                       "side": None, "complete": True,
                                       "n_legal": st.legal_count(),
                                       "n_unavailable": len(st.unavailable),
                                       "first_pick": 1 if st.red_first else 0},
                             "patch": st.patch, "suggestions": [], "note": ""})
            return
        try:
            res = st.predict_next_move(k=k)
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(200, {
            "model": model_info(engine.predictor),
            "state": {"move_idx": res["move_idx"], "kind": res["kind"],
                      "side": 0 if res["side"] == "blue" else 1,
                      "n_legal": res["n_legal"], "complete": False,
                      "n_unavailable": len(st.unavailable),
                      "first_pick": 1 if st.red_first else 0},
            # The patch the model actually conditioned on: an unknown or
            # out-of-range request is reported as the row it fell back to.
            "patch": res["patch_resolved"],
            "suggestions": [{"idx": store.champs.resolve(c["name"]),
                             "name": c["name"], "prob": c["p_head"]}
                            for c in res["top"]],
            # The tool's own note: which pool the shares are of, and whether
            # the patch it was asked for is one it knows.
            "note": res["note"],
        })

    def _handle_load(self):
        """POST {"path": ".pt", "role": "masked"|"policy"|"auto"} -> swap
        that slot's model. `role` defaults to auto (decided by the kind);
        a null path unloads the slot, which is how the UI turns the role board
        or the head off without touching the other one."""
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
                    if _STATE_STORE is not None:
                        _bind_models()
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
        self._send(200, {"model": model_info(pred), "role": slot_of(pred)})

    def _handle_upload(self):
        """Raw .pt bytes with ?name=x.pt -> frontend/uploaded/x.pt, then load."""
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
        self._send(200, {"model": model_info(pred), "role": slot_of(pred)})

def main():
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
                    help="cpu keeps GPU free and serves the head fine; the 4B "
                         "reranker needs cuda (it is skipped, not fatal, when "
                         "this is cpu)")
    ap.add_argument("--ckpt", default="",
                    help="role-model checkpoint (a masking-* checkpoint) - "
                         "drives the role board and the per-role targets "
                         "(default: pro/model/checkpoints/"
                         "masking-proonly-nov13-through-lcp-df-sw"
                         "/best.pt, "
                         "else the best masking checkpoint by val champ_top5)")
    ap.add_argument("--policy-ckpt", default="",
                    help="next-move checkpoint, an ensemble.json manifest, or "
                         "an llm-* reranker directory - loaded alongside the "
                         "role model, not instead of it (default: "
                         "pro/model/checkpoints/"
                         "nextpick-proonly-nov13-k5-through-lcp-df-sw/best.pt; "
                         "the -randomvalsplit- pair, "
                         "nextpick-bans-aware-through-wlds-ig-tes/best.pt and "
                         "nextpick-proto-ensemble/ensemble.json are still "
                         "selectable here)")
    ap.add_argument("--rerank", nargs="?", const=str(RERANK_DEFAULT),
                    default=None, metavar="DIR",
                    help="serve the 4B reranker as the next-move model "
                         "(default DIR: %s) instead of the head. Same slot as "
                         "--policy-ckpt; it re-orders that head's shortlist, so "
                         "the head still proposes. Needs a GPU and the `llm` "
                         "image, and a failure to build it is fatal. Read "
                         "frontend/README.md 'What the reranker is worth' "
                         "first: it is worth ~2 s/board and, on the panel's "
                         "unconstrained question, a few pp"
                         % RERANK_DEFAULT.name)
    ap.add_argument("--no-policy", action="store_true",
                    help="run without a next-move model: the role board falls "
                         "back to the masking checkpoint, and the agent can "
                         "read the board but not rank a move")
    ap.add_argument("--agent-calls", type=int, default=24,
                    help="tool-call budget per agent turn (default 24)")
    args = ap.parse_args()

    mapping, names_by_idx, champions = load_champ_data()
    APP.update(device=args.device, agent_calls=args.agent_calls)

    # Slot 1 - the masker: answers a role question during the ban phase, and is
    # the whole role board in a --no-policy install.
    if args.ckpt:
        masked_ckpt = resolve_ckpt_path(args.ckpt, confine=False)
    elif MASKED_DEFAULT.is_file():
        masked_ckpt = MASKED_DEFAULT
    else:
        masked_ckpt, val = find_best_ckpt()
        print("best pretraining ckpt: %s (val top-5 %s)" % (masked_ckpt, val))
    # Slot 2 - the next-move model: the head by default, the 4B reranker on
    # request.
    #
    # The reranker was going to be the default. It is not, because the number
    # that justified it does not survive a served comparison (frontend/README.md
    # "What the reranker is worth", STATUS.md §9):
    #
    #   * its 34.27 % is measured with the *true lane* in the task line, and the
    #     lane is the answer to a question the panel does not ask;
    #   * without that clause - the panel's actual question - it is 20.85 %, on
    #     candidates that were not legality-filtered;
    #   * the head it was measured against is `nextpick-bans-aware-*`, not the
    #     shipped `-nov13-k5-` head.
    #
    # So `--rerank` is an explicit opt-in, and an explicit request that cannot
    # be served is fatal: a server that refuses to start beats one that quietly
    # answers with a different model than the flag asked for. The role board is
    # the surface where the lane *is* the question (`predict_in_lane`), and it is
    # the reason to keep the flag wired at all.
    policy_ckpt, want_rerank = None, False
    if not args.no_policy:
        if args.rerank is not None:
            policy_ckpt = resolve_ckpt_path(args.rerank, confine=False)
            want_rerank = True
        elif args.policy_ckpt:
            policy_ckpt = resolve_ckpt_path(args.policy_ckpt, confine=False)
            want_rerank = Path(policy_ckpt).is_dir()
        elif POLICY_DEFAULT.is_file():
            policy_ckpt = POLICY_DEFAULT

    masked = load_predictor(masked_ckpt, device=args.device, role="masked")
    print("role model:   %s (%s)" % (masked_ckpt, masked.kind))
    policy = None
    if policy_ckpt is not None:
        if not want_rerank:
            policy = load_predictor(policy_ckpt, device=args.device,
                                    role="policy")
        else:
            reason = None
            if not (str(args.device).startswith("cuda")
                    and torch.cuda.is_available()):
                reason = ("--device is %s and the 4.2 B trunk is 4-bit "
                          "quantized, which needs a GPU" % args.device)
            if reason is None:
                try:
                    policy = load_predictor(policy_ckpt, device=args.device,
                                            role="policy")
                except Exception as exc:           # noqa: BLE001 - see above
                    reason = "%s: %s" % (type(exc).__name__, exc)
            if policy is None:
                raise SystemExit("cannot serve the reranker %s: %s"
                                 % (Path(policy_ckpt).name, reason))
    if policy is not None:
        print("next-move:    %s (%s)" % (policy.ckpt_path, policy.kind))
        if getattr(policy, "kind", None) == "rerank":
            print("              head %s proposes, the adapter re-orders the "
                  "top %d" % (policy.head_path.name, policy.n_candidates))
    else:
        print("next-move:    (none - the agent can read the board only)")
    APP.update(mapping=mapping, names_by_idx=names_by_idx, champions=champions)
    # The agent's model is whatever LLM_API_KEY names; `client=None` means the
    # environment decides, so this line reports the decision rather than making
    # it. A key is a capability, not a preference: without one the panel still
    # opens and says so.
    cfg = llm_client.describe()
    print("agent:        %s" % ("%s at %s" % (cfg["model"], cfg["base_url"])
                                if llm_client.client_from_env() is not None
                                else "no LLM_API_KEY - the panel will say so"))
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

