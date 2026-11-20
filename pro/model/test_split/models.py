"""Loaders and scorers shared by the eval suite.

Nothing here decides *what* to score or *which* split - that is `split.py` and
`registry.py`. This module only knows how to turn a checkpoint into answers on a
board, and it is deliberately the same code the pre-`nov13` harness used, so the
numbers this suite produces are on the same construction as the ones in
STATUS.md rather than a quiet re-derivation.

Two scorers, one per question, because the two model families answer two
different things and must not be scored by the same function:

* `score_next_move` - given the first m moves, what is move m? 20 states per
  game, prefix only, nothing after the target is visible.
* `score_masked` - given the complete draft with one pick hidden, who was it?
  10 states per game, and the other nine slots *including later picks* are
  visible.
* `score_role_next` - given the same prefix but TOLD the true role of the
  upcoming pick, who is it? 10 pick states per game, no later picks. This is
  the role board's cell-level question, scored separately from both above.

The asymmetry is the whole point and it is not a bug: these are different
questions, which is why they are reported separately and never averaged.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

sys.path.insert(0, str(ROOT))
from solo_queue.model.pretraining.base_model import (  # noqa: E402
    CHAMP_MASK_ID, SiameseBase, attr_matrix_for_config, load_encoder_state)
from pro.model import baselines as BL  # noqa: E402

PRO_DIR = ROOT / "data" / "pro play"

NUM_CHAMPS = 173
N_MOVES = 20
N_SLOTS = 10
MOVE_BAN, MOVE_PICK = 0, 1
PHASE_NAMES = {0: "ban1", 1: "pick1", 2: "ban2", 3: "pick2"}
# 6 bans, 6 picks, 4 bans, 4 picks (mirrors policy/infer.PHASE_OF)
PHASE_OF = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4
ROLE_NAMES = {0: "TOP", 1: "JNG", 2: "MID", 3: "BOT", 4: "SUP"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")
P = _load("pro_pipeline", PRO_DIR / "pipeline.py")
SOLOQ_MODEL = ROOT / "solo_queue" / "model"
FIN_MODEL = _load("finetune_model", SOLOQ_MODEL / "finetune" / "model.py")
POLICY_MODEL = _load("policy_model", SOLOQ_MODEL / "policy" / "model.py")
PolicyValueModel = POLICY_MODEL.PolicyValueModel


# --------------------------------------------------------------------- models
def _enc_state(blob):
    """Encoder weights out of a checkpoint, with or without the `encoder.` prefix."""
    sd = blob["model"]
    if any(k.startswith("encoder.") for k in sd):
        return {k[len("encoder."):]: v for k, v in sd.items()
                if k.startswith("encoder.")}
    return sd


def build_encoder(blob, n_patches=None):
    cfg = blob["config"]
    n_p = int(cfg.get("n_patches", 0) if n_patches is None else n_patches)
    enc = SiameseBase(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        with_value=bool(cfg.get("with_value")),
        value_seeds=cfg.get("value_seeds", 4),
        attr_matrix=attr_matrix_for_config(cfg), n_patches=n_p)
    load_encoder_state(enc, _enc_state(blob))
    return enc


def patch_vocab_of(blob):
    """The checkpoint's own patch table, never None.

    None would make every patch score UNK and hide the patch conditioning the
    suite is partly there to measure. Both conventions are tolerated: the
    finetune stage stores `patch_vocab` at the top level of the payload, the
    policy stage nests it inside `config`.
    """
    pv = blob.get("patch_vocab")
    if pv is None:
        pv = (blob.get("config") or {}).get("patch_vocab")
    return PRO_DATA.patch_index_from_vocab(pv)


def load_policy(path):
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = blob["config"]
    if str(cfg.get("arch", "")).startswith("naive_transformer"):
        # Single-stack model: it builds its own 61-token sequence from the
        # ordinary prefix batch, so the scorer below needs no changes. The
        # checkpoint declares its architecture; guessing from weight names
        # would let an old checkpoint load into the wrong model.
        from pro.model.naivetransformer.model import load_from_checkpoint
        model, patch_index, cfg = load_from_checkpoint(path)
        return model, patch_index, cfg
    n_p = int(cfg.get("n_patches", 0))
    enc = build_encoder(blob, n_p)
    with_value = any(k.startswith("value_head.") for k in blob["model"])
    # The four ban-awareness switches are module-set flags, not hyperparameters:
    # leaving them at their defaults makes a ban-aware checkpoint fail to load
    # with "unexpected keys" (`counter.*`, `ban_context.*_embed.*`,
    # `state_embed.pool_proj.*`). They live in the checkpoint config for exactly
    # this reason, and every pre-nov13 arm needs at least one of them.
    model = PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg.get("state_dim", 64),
        n_econ=cfg.get("n_econ", 0), dropout=cfg.get("dropout", 0.0),
        n_patches=n_p, with_value=with_value,
        ban_side=bool(cfg.get("ban_side", False)),
        ban_fearless=bool(cfg.get("ban_fearless", False)),
        counter_rank=int(cfg.get("counter_rank", 0) or 0),
        pool_order=str(cfg.get("pool_order", "absolute")))
    model.load_state_dict(blob["model"])
    model.eval()
    return model, patch_vocab_of(blob), cfg


def load_masked(path):
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    enc = build_encoder(blob)
    enc.eval()
    return enc, patch_vocab_of(blob), blob["config"]


# -------------------------------------------------------------------- scoring
def _rank_of(logits, target):
    return int((logits > logits[target]).sum().item())


def _topk(logits, k):
    return torch.topk(logits, min(k, logits.numel())).indices.tolist()


def score_next_move(model, patch_index, rows, id_to_name):
    """20 states per game -> rank/top-k/prob for the move that actually happened."""
    games = []
    for row in rows:
        samples = [PRO_DATA.build_prefix(row, m, NUM_CHAMPS, patch_index)
                   for m in range(N_MOVES)]
        batch = PRO_DATA.pro_collate(samples)
        with torch.no_grad():
            logits = model(batch)["policy_logits"]
        probs = torch.softmax(logits, dim=-1)
        steps = []
        for m in range(N_MOVES):
            lg = logits[m]
            target = int(batch["target_champ"][m])
            if not torch.isfinite(lg[target]):
                # A target the model masks out means the legality construction
                # disagrees with the corpus, which would silently score the
                # model on an unanswerable question.
                raise RuntimeError("target %d illegal at move %d (%s)"
                                   % (target, m, row["gameid"]))
            top5 = _topk(lg, 5)
            steps.append({
                "move": m,
                "phase": PHASE_NAMES[PHASE_OF[m]],
                "rank": _rank_of(lg, target),
                "top1_name": id_to_name[int(lg.argmax())],
                "top5_names": [id_to_name[c] for c in top5],
                "prob": float(probs[m, target]),
            })
        games.append({"gameid": row["gameid"], "steps": steps})
    return games


def score_masked(enc, patch_index, rows, id_to_name):
    """Hide each picked slot in turn; does the model return the real champion?

    One slot at a time, batch of one, exactly as `proonly/compare.masked_ranks_slot`
    and the training-time eval do it. Batching the ten slots together would let
    a model see nine masked slots at once - a different question, and not the
    one the role board asks.
    """
    games = []
    for row in rows:
        champ = torch.tensor(row["champ"], dtype=torch.long)
        role = torch.tensor(row["role"], dtype=torch.long)
        slot_move = {}
        for m, (kind, team, _c, r) in enumerate(row["moves"]):
            if kind == MOVE_PICK:
                slot_move[team * 5 + r] = m
        pid = PRO_DATA.patch_to_id(row.get("patch"), patch_index)
        slots = []
        for slot in range(N_SLOTS):
            cids = champ.clone()
            cids[slot] = CHAMP_MASK_ID
            is_m = torch.zeros(1, N_SLOTS, dtype=torch.bool)
            is_m[0, slot] = True
            batch = {
                "champ_ids": cids.unsqueeze(0),
                "role_ids": role.unsqueeze(0),
                "pad_mask": torch.zeros(1, N_SLOTS, dtype=torch.bool),
                "is_masked": is_m,
                "patch_id": torch.tensor([pid], dtype=torch.long),
            }
            with torch.no_grad():
                out = enc(batch)
            lg = out["champ_logits"][0, slot]
            target = int(champ[slot])
            top5 = _topk(lg, 5)
            slots.append({
                "slot": slot,
                "move": slot_move.get(slot),
                "role": ROLE_NAMES.get(int(role[slot]), "?"),
                "actual": id_to_name[target],
                "rank": _rank_of(lg, target),
                "top1_name": id_to_name[int(lg.argmax())],
                "top5_names": [id_to_name[c] for c in top5],
                "prob": float(torch.softmax(lg, -1)[target]),
            })
        slots.sort(key=lambda s: (s["move"] is None, s["move"]))
        games.append({"gameid": row["gameid"], "slots": slots})
    return games


# ------------------------------------------------ role-conditioned next pick
# A third protocol, deliberately separate from both suites above: prefix only
# (no later picks), but the TRUE role of the upcoming pick is pinned into the
# target slot's role input. This is the cell-level shape of the role board's
# question - "who belongs in red MID given what is known now?" - without the
# full-draft future leakage of `score_masked`. Ban moves are excluded: their
# target has no true role (`build_prefix` records IGNORE_INDEX), so the question
# "who fills this role?" is undefined there.
def role_next_states(rows, patch_index, fearless_roles):
    """One state per PICK move: prefix + the target slot's true role pinned."""
    states = []
    for row in rows:
        patch = P.patch_key(row.get("patch"))
        for m, move in enumerate(row["moves"]):
            kind, team, champ, role = move
            if kind != MOVE_PICK:
                continue
            team, role, champ = int(team), int(role), int(champ)
            slot = team * 5 + role
            sample = PRO_DATA.build_prefix(
                row, m, NUM_CHAMPS, patch_index, fearless_roles)
            # build_prefix leaves unpicked slots at ROLE_FLEX_ID. Pinning the
            # target role is the entire experiment: the model is TOLD which
            # role it is filling instead of having to guess it from the pick
            # order. The champion stays PAD and pad_mask stays True.
            sample["role_ids"][slot] = role
            states.append({
                "sample": sample, "row": row, "move": m, "slot": slot,
                "role": role, "target": champ, "patch": patch,
            })
    return states


def role_next_frequency_tables(rows, train_idx):
    """Per (patch, role) champion counts from TRAIN picks only, plus backoff."""
    by_pair = defaultdict(lambda: np.zeros(NUM_CHAMPS, dtype=np.float64))
    by_role = np.zeros((5, NUM_CHAMPS), dtype=np.float64)
    for i in train_idx:
        row = rows[i]
        patch = P.patch_key(row.get("patch"))
        for kind, _team, champ, role in row["moves"]:
            if kind != MOVE_PICK:
                continue
            by_pair[(patch, int(role))][int(champ)] += 1.0
            by_role[int(role)][int(champ)] += 1.0
    return dict(by_pair), by_role


def role_next_table_size(freq):
    """What the lookup table costs to hold: one row per (patch, role, champ)."""
    by_pair, by_role = freq
    return len(by_pair) * NUM_CHAMPS + by_role.shape[0] * NUM_CHAMPS


def _role_next_item(state, rank, id_to_name=None):
    item = {
        "slot": state["slot"], "move": state["move"],
        "role": ROLE_NAMES.get(state["role"], "?"), "rank": int(rank),
    }
    if id_to_name:
        item["actual"] = id_to_name[state["target"]]
    return item


def score_role_next(model, patch_index, rows, names, fearless_roles, freq,
                    chunk=160):
    """Rank the upcoming pick's champion, given its prefix and true role."""
    states = role_next_states(rows, patch_index, fearless_roles)
    by_game = {}
    for start in range(0, len(states), chunk):
        part = states[start:start + chunk]
        batch = PRO_DATA.pro_collate([s["sample"] for s in part])
        with torch.no_grad():
            logits = model(batch)["slot_logits"]
        for i, state in enumerate(part):
            slot = state["slot"]
            lg = logits[i, slot]
            target = state["target"]
            if not bool(torch.isfinite(lg[target])):
                raise RuntimeError(
                    "target %d is masked at role_next state %s move %d"
                    % (target, state["row"]["gameid"], state["move"]))
            by_game.setdefault(state["row"]["gameid"], []).append(
                _role_next_item(state, _rank_of(lg, target), names))
    games = []
    for gid, items in by_game.items():
        items.sort(key=lambda s: (s["move"], s["slot"]))
        games.append({"gameid": gid, "slots": items})
    return games


def score_role_next_frequency(states, freq):
    """The same items and legality filter, ranked by a train-only lookup table."""
    by_pair, by_role = freq
    by_game = {}
    for state in states:
        table = by_pair.get((state["patch"], state["role"]))
        if table is None:
            table = by_role[state["role"]]
        sample = state["sample"]
        legal = np.flatnonzero(~sample["taken_mask"].numpy()
                               & ~sample["unavailable_mask"].numpy())
        target = state["target"]
        if target not in legal:
            raise RuntimeError(
                "target %d is illegal at role_next state %s move %d"
                % (target, state["row"]["gameid"], state["move"]))
        scores = table[legal]
        # Same tie rule as eval_role.py: a deterministic id tie-break keeps a
        # sparse per-patch table from being penalised for counts that tie.
        rank = int(np.count_nonzero(
            (scores > table[target])
            | ((scores == table[target]) & (legal < target))))
        by_game.setdefault(state["row"]["gameid"], []).append(
            _role_next_item(state, rank))
    games = []
    for gid, items in by_game.items():
        items.sort(key=lambda s: (s["move"], s["slot"]))
        games.append({"gameid": gid, "slots": items})
    return games


# ------------------------------------------------- flat baselines (mlp/forest)
# One scorer for both non-neural families. They differ in *where the candidate
# lives* - the MLP's output layer, the forest's input row - but both expose
# `scores(states, candidate_lists)`, so the ranking below is written once. Two
# rankers would be two chances to measure the same thing differently.
def load_flat(run_dir):
    """Load a `pro/model/{mlp,forest}` checkpoint -> (predictor, params, kind)."""
    run_dir = Path(run_dir)
    flat = json.loads((run_dir / "flat.json").read_text(encoding="utf-8"))
    if flat.get("feature_version") != BL.FEATURE_VERSION:
        raise RuntimeError(
            "%s was trained on flat-feature version %r, this suite builds %r"
            % (run_dir.name, flat.get("feature_version"), BL.FEATURE_VERSION))
    kind = flat.get("kind")
    if kind == "mlp":
        from pro.model.mlp.model import load as load_mlp
    elif kind == "forest":
        from pro.model.forest.model import load as load_forest
    else:
        raise RuntimeError("unknown flat kind %r in %s"
                           % (kind, run_dir / "flat.json"))
    predictor, params = (load_mlp if kind == "mlp" else load_forest)(run_dir, flat)
    return predictor, params, kind


def candidates_of(task, row, k):
    """The candidates a flat model ranks. Sorted ascending, always."""
    if task == "next_move":
        return BL.candidates_for(row, k)
    return np.arange(NUM_CHAMPS, dtype=np.int64)


def score_flat(predictor, task, states, id_to_name=None, chunk=512):
    """(row, k) states -> the same `games` shape the torch scorers return.

    Chunked because the MLP's feature matrix is 3,834 floats per state: the
    whole 48,960-state next-move suite at once is 750 MB for a single model.
    """
    key = "steps" if task == "next_move" else "slots"
    by_game = {}
    for b in range(0, len(states), chunk):
        st = states[b:b + chunk]
        cl = [candidates_of(task, row, k) for row, k in st]
        sc = predictor.scores(st, cl)
        for (row, k), cands, s in zip(st, cl, sc):
            target = int(row["moves"][k][2] if task == "next_move"
                         else row["champ"][k])
            pos = int(np.searchsorted(cands, target))
            if pos >= len(cands) or int(cands[pos]) != target:
                raise RuntimeError(
                    "target %d is not a candidate at %s %d (%s): the baseline "
                    "is being asked something it has excluded"
                    % (target, task, k, row["gameid"]))
            # Same tie rule as the torch path's `_rank_of`: strictly greater
            # counts as ahead, so a tie is shared rather than lost.
            item = {"rank": int((s > s[pos]).sum())}
            if task == "next_move":
                item["move"] = k
                item["phase"] = PHASE_NAMES[PHASE_OF[k]]
            else:
                item["slot"] = k
                item["move"] = BL.slot_moves(row).get(k)
                item["role"] = ROLE_NAMES.get(int(row["role"][k]), "?")
            if id_to_name:
                order = np.argsort(-s, kind="stable")[:5]
                item["top1_name"] = id_to_name[int(cands[order[0]])]
                item["top5_names"] = [id_to_name[int(cands[j])] for j in order]
                if task == "masked":
                    item["actual"] = id_to_name[target]
            by_game.setdefault(row["gameid"], []).append(item)
    games = []
    for gid, items in by_game.items():
        items.sort(key=lambda s: (s.get("move") is None, s.get("move") or 0,
                                  s.get("slot", 0)))
        games.append({"gameid": gid, key: items})
    return games


# ------------------------------------------------------------------ aggregate
def wilson(hits, n, z=1.96):
    """Wilson score interval - the only honest error bar at these rates.

    The normal approximation collapses at 0 and 1, which is exactly where a
    broken model lands, and it is the case the suite most needs to report.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(games, key):
    """Pooled top-1/5/10, mean rank, mean probability, and their intervals."""
    items = [s for g in games for s in g[key]]
    n = len(items)
    out = {"n": n}
    for k in (1, 5, 10):
        hits = sum(1 for s in items if s["rank"] < k)
        lo, hi = wilson(hits, n)
        out["top%d" % k] = hits / max(1, n)
        out["top%d_ci" % k] = [lo, hi]
    out["mean_rank"] = sum(s["rank"] for s in items) / max(1, n)
    with_prob = [s["prob"] for s in items if "prob" in s]
    if with_prob:
        out["mean_prob"] = sum(with_prob) / len(with_prob)
    return out


def by_phase(games):
    """Top-1/5 per draft phase, on next-move states only.

    ban1 is the phase where the board carries no information, so it is where a
    patch-conditioned model should separate from a frequency table. It is also
    where the frequency table has historically won, which makes the cut worth
    reporting rather than averaging away.
    """
    out = {}
    for phase in range(4):
        ranks = [s["rank"] for g in games for s in g["steps"]
                 if PHASE_OF[s["move"]] == phase]
        n = len(ranks)
        out[PHASE_NAMES[phase]] = {
            "n": n,
            "top1": sum(1 for r in ranks if r < 1) / max(1, n),
            "top5": sum(1 for r in ranks if r < 5) / max(1, n),
            "mean_rank": sum(ranks) / max(1, n),
        }
    return out


def paired_hits(games_a, games_b, key):
    """Per-item correctness for two models on the SAME items.

    Paired is the only way to compare two models here: the items differ hugely
    in difficulty (ban1 is nearly unpredictable, pick2 nearly determined), so an
    unpaired difference mostly measures which model got the easier sample.
    """
    a = {(g["gameid"], s.get("slot", s["move"])): s["rank"] < 1
         for g in games_a for s in g[key]}
    b = {(g["gameid"], s.get("slot", s["move"])): s["rank"] < 1
         for g in games_b for s in g[key]}
    keys = sorted(set(a) & set(b))
    return [a[k] for k in keys], [b[k] for k in keys]


def mcnemar(a_hits, b_hits):
    """Exact-ish McNemar on two paired hit vectors.

    Returns (a_only, b_only, chi2, p). The chi-square is the continuity-corrected
    form and the p is from a normal approximation, which is adequate once
    `a_only + b_only` is above ~10 and is the standard used elsewhere in this
    repo. Below that, read the discordant counts directly - the suite prints
    them for exactly this reason.
    """
    a_only = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    b_only = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    n = a_only + b_only
    if n == 0 or a_only == b_only:
        # Equal discordant counts is the *maximum* agreement with the null, so
        # the answer is p = 1. The continuity-corrected chi-square would report
        # 1/n here (~0.48 at n=2), which is conservative in the wrong
        # direction: it makes "no evidence at all" look like weak evidence.
        return a_only, b_only, 0.0, 1.0
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return a_only, b_only, chi2, p


def popularity_reference(patch, corpus_rows, n_moves=N_MOVES):
    """Per-move frequency table on one patch, backed off to patch frequency.

    A bare per-move count ties constantly (a patch has a few hundred rows), so
    0.05 x the patch-wide count breaks ties toward the patch's overall priority
    order rather than toward low champion ids.
    """
    per_move = [dict() for _ in range(n_moves)]
    overall = dict()
    n = 0
    for row in corpus_rows:
        if P.patch_key(row.get("patch")) != patch:
            continue
        n += 1
        for m, move in enumerate(row["moves"]):
            c = int(move[2])
            per_move[m][c] = per_move[m].get(c, 0.0) + 1.0
            overall[c] = overall.get(c, 0.0) + 1.0
    ranks = []
    for m in range(n_moves):
        score = {c: per_move[m].get(c, 0.0) + 0.05 * overall.get(c, 0.0)
                 for c in range(NUM_CHAMPS)}
        order = sorted(range(NUM_CHAMPS), key=lambda c: (-score[c], c))
        rank = [0] * NUM_CHAMPS
        for pos, c in enumerate(order):
            rank[c] = pos
        ranks.append(rank)
    return ranks, n


def score_popularity(rows, base_ranks):
    """The frequency table, scored on the identical boards and legality filter.

    Legality matters: without it the table ranks champions already banned or
    picked while every model has them at -inf, a ~10-place handicap on 173
    champions that would flatter the models by several points.
    """
    games = []
    for row in rows:
        steps = []
        taken = set()
        for m, (_k, _t, c, _ro) in enumerate(row["moves"]):
            legal = [x for x in range(NUM_CHAMPS) if x not in taken]
            order = sorted(legal, key=lambda x: base_ranks[m][x])
            rank = {champ: pos for pos, champ in enumerate(order)}
            steps.append({"move": m, "phase": PHASE_NAMES[PHASE_OF[m]],
                          "rank": rank[int(c)]})
            taken.add(int(c))
        games.append({"gameid": row["gameid"], "steps": steps})
    return games
