"""Score every kept checkpoint on the test split, state by state.

Why this file exists
--------------------
Every pooled number answers "how good is it on average" and never "what would
it have said here". This script scores the same seven held-out games with every
model class, so each model's actual recommendation at each real state is
visible next to what actually happened - and so the classes can be compared on
data none of them trained on.

The test split
--------------
Three series, seven games, patch 16.20, all played on the last day of Worlds
2026: Bilibili Gaming vs Hanwha Life Esports (2 games), Karmine Corp vs
Movistar KOI (2), LYON vs LØS (3). They are `pipeline.TEST_SERIES`, pinned by
`series_id`, and `split_within_patch` holds them out of train and val by id.
The gameids are resolved from the CSV by series id, so this file has no list of
its own to drift.

They *are* in `pro_drafts.pt` now - that is the point. `CORPUS_MAX_DATE` was
raised to 2026-11-01 so the corpus contains them, and the split, not the corpus
pin, is what keeps them out of training. A series the corpus has never seen
cannot be a test split; it can only be a holdout nothing can be trained
against.

Nothing here rebuilds a cache: `build_rows` would see a changed CSV signature
and rescan, so the rows are scanned straight from the CSV and kept in memory.

Two readouts, because the models answer two different questions:

* **next move** (policy models) - 20 states per game. What did it recommend,
  where did the actual move rank, what probability did it put on it.
* **masked completion** (encoder models) - 10 picks per game, each hidden in
  turn. Does it recover the champion that was actually picked into that slot.
  Ordered by the move index of the pick, so it reads as a per-step table too.

No value readout: no value checkpoint is kept, and the frontend has no slot for
one. That is a gap in the comparison, not an untested result.

Statistics discipline: this is 7 games, 140 next-move states and 70 masked
slots. No p-values. Every rate is printed with a Wilson interval, because at
this n the intervals overlap for almost any pair of models - that is the
honest reading, and the per-state tables are the part that carries information,
since every model is scored on the identical board.

Usage:
    python pro/model/test_split/eval_test_split.py
    python pro/model/test_split/eval_test_split.py --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent            # .../pro/model/test_split
MODEL_DIR = HERE.parent                           # .../model
ROOT = MODEL_DIR.parents[1]                           # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"

sys.path.insert(0, str(ROOT))
from solo_queue.model.pretraining.base_model import (CHAMP_MASK_ID, CHAMP_PAD_ID, ROLE_FLEX_ID,  # noqa: E402
                         SiameseBase, attr_matrix_for_config,
                         load_encoder_state)

NUM_CHAMPS = 173
N_MOVES = 20
N_SLOTS = 10
MOVE_BAN, MOVE_PICK = 0, 1
PHASE_NAMES = {0: "ban1", 1: "pick1", 2: "ban2", 3: "pick2"}
# 6 bans, 6 picks, 4 bans, 4 picks (mirrors policy/infer.PHASE_OF)
PHASE_OF = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4
ROLE_NAMES = {0: "TOP", 1: "JNG", 2: "MID", 3: "BOT", 4: "SUP"}

# The test split is `pipeline.TEST_SERIES`, not a list kept here: three
# series, seven games, all patch 16.20, all played on the last day of Worlds
# 2026. The gameids are resolved from the CSV by series_id, so the report and
# the split cannot drift apart - if the pin changes, this follows.
CSV_2026 = PRO_DIR / "2026_LoL_esports_match_data_from_OraclesElixir.csv"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")
EP = _load("eval_protocol", PRO_DIR / "eval_protocol.py")
SOLOQ_MODEL = ROOT / "solo_queue" / "model"
FIN_MODEL = _load("finetune_model", SOLOQ_MODEL / "finetune" / "model.py")
POLICY_MODEL = _load("policy_model", SOLOQ_MODEL / "policy" / "model.py")
PolicyValueModel = POLICY_MODEL.PolicyValueModel
PrefixValueModel = FIN_MODEL.PrefixValueModel
ProDraftModel = FIN_MODEL.ProDraftModel

TEST_SERIES = list(P.TEST_SERIES)


# --------------------------------------------------------------------- models
# One checkpoint per model class, each in its own track's store
# (`solo_queue/model/checkpoints/` or `pro/model/checkpoints/`). The name is the
# contract (see frontend/server._ckpt_roles): `masking-*` answers masked
# completion and the role board, `nextpick-*` answers next move. This file is the
# one place that deliberately spans both tracks - it exists to measure whether
# solo-queue initialisation buys anything over pro-only.
#
# `masking-soloqueue` is not a class of its own - it is the soloQ-pretrained
# encoder that `masking-finetune` starts from, kept as the reference point
# that says whether fine-tuning on pro drafts bought anything. `latest.pt`
# files are skipped: same run, later step, worse on the metric the run
# selected on.
POLICY_MODELS = [
    ("nextpick-policy", "soloQ encoder + policy head",
     "solo_queue/model/checkpoints/nextpick-policy/best_policy.pt"),
    ("nextpick-proonly", "pro-only, from scratch",
     "pro/model/checkpoints/nextpick-proonly/best.pt"),
]

MASKED_MODELS = [
    ("masking-soloqueue", "soloQ pretrain only",
     "solo_queue/model/checkpoints/masking-soloqueue/best.pt"),
    ("masking-finetune", "soloQ -> pro finetune",
     "solo_queue/model/checkpoints/masking-finetune/best_mask.pt"),
    ("masking-proonly", "pro-only, from scratch",
     "pro/model/checkpoints/masking-proonly/best.pt"),
]

# No value checkpoint is kept. The frontend has no slot for one
# (`_ckpt_roles` returns [] for `value-*`), and the value path measured at
# chance in every earlier arm, so it is outside this comparison rather than
# an untested gap in it.
VALUE_MODELS = []


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
    # Tolerates both conventions: finetune stores ``patch_vocab`` at the
    # top level of the payload, while the policy stage nests it inside
    # ``config``. Either way we want the dict, never None (None would make
    # every patch score UNK and hide the patch conditioning we measure).
    pv = blob.get("patch_vocab")
    if pv is None:
        pv = (blob.get("config") or {}).get("patch_vocab")
    return PRO_DATA.patch_index_from_vocab(pv)


def load_policy(path):
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = blob["config"]
    n_p = int(cfg.get("n_patches", 0))
    enc = build_encoder(blob, n_p)
    with_value = any(k.startswith("value_head.") for k in blob["model"])
    model = PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg.get("state_dim", 64),
        n_econ=cfg.get("n_econ", 0), dropout=cfg.get("dropout", 0.0),
        n_patches=n_p, with_value=with_value)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, patch_vocab_of(blob), cfg


def load_masked(path):
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    enc = build_encoder(blob)
    enc.eval()
    return enc, patch_vocab_of(blob), blob["config"]


def load_value(path, kind):
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = blob["config"]
    if kind == "prefix":
        enc = build_encoder(blob)
        model = PrefixValueModel(
            enc, ban_heads=cfg.get("ban_heads", 4),
            ban_seeds=cfg.get("ban_seeds", 4),
            ban_dropout=cfg.get("ban_dropout", 0.0),
            use_bans=cfg.get("use_bans", True))
        model.load_state_dict(blob["model"])
    elif kind == "soloq":
        model = build_encoder(blob)
    elif kind == "draft":
        enc = build_encoder(blob)
        model = ProDraftModel(enc, n_econ=cfg.get("n_econ", 6))
        model.load_state_dict(blob["model"])
    elif kind == "policy":
        enc = build_encoder(blob, 0)
        model = PolicyValueModel(
            enc, embed_dim=cfg["embed_dim"], state_dim=cfg.get("state_dim", 64),
            n_econ=cfg.get("n_econ", 0), dropout=cfg.get("dropout", 0.0),
            n_patches=0, with_value=True)
        model.load_state_dict(blob["model"])
    else:
        raise ValueError(kind)
    model.eval()
    platt = tuple(blob.get("platt") or (1.0, 0.0))
    return model, kind, platt


# ----------------------------------------------------------------------- data
def load_test_rows():
    """Scan the 2026 CSV for the pinned test series. Never touches a cache.

    Resolved by `series_id` rather than by gameid so there is one source of
    truth: `pipeline.TEST_SERIES` is what the split holds out, so it is also
    what the report scores. Raises if a pinned series is not in the CSV, and
    if the row count is not the 7 the split promises.
    """
    name_to_idx = P.load_name_to_idx()[0]
    games = P.assign_series(P.group_games([str(CSV_2026)]))
    pinned = set(TEST_SERIES)
    rows = []
    for g in games:
        if g.get("series_id") not in pinned:
            continue
        out, reason = P.game_to_draft(g, name_to_idx)
        if out is None:
            raise RuntimeError("%s rejected by the pipeline: %s"
                               % (g["gameid"], reason))
        out.setdefault("fearless", False)
        rows.append(out)
    got = {r["series_id"] for r in rows}
    if got != pinned:
        raise RuntimeError("CSV has %d of %d pinned series; missing: %s"
                           % (len(got), len(pinned),
                              ", ".join(sorted(pinned - got))))
    P.attach_fearless(rows)
    rows.sort(key=lambda r: (r["series_id"], r["game_number"]))
    id_to_name = {v: k for k, v in name_to_idx.items()}
    return rows, id_to_name


def patch_popularity_reference(patch, rows_cache):
    """Per-move frequency table on one patch, backed off to patch frequency.

    196 corpus rows on 16.20, so a bare per-move count ties constantly; adding
    0.05 * (patch-wide count) breaks ties toward the patch's overall priority
    order instead of toward low champion ids.
    """
    per_move = [dict() for _ in range(N_MOVES)]
    overall = dict()
    n = 0
    for row in rows_cache:
        if P.patch_key(row.get("patch")) != patch:
            continue
        n += 1
        for m, move in enumerate(row["moves"]):
            c = int(move[2])
            per_move[m][c] = per_move[m].get(c, 0.0) + 1.0
            overall[c] = overall.get(c, 0.0) + 1.0
    ranks = []
    for m in range(N_MOVES):
        score = {c: per_move[m].get(c, 0.0) + 0.05 * overall.get(c, 0.0)
                 for c in range(NUM_CHAMPS)}
        order = sorted(range(NUM_CHAMPS), key=lambda c: (-score[c], c))
        rank = [0] * NUM_CHAMPS
        for pos, c in enumerate(order):
            rank[c] = pos
        ranks.append(rank)
    return ranks, n


# -------------------------------------------------------------------- scoring
def _rank_of(logits, target):
    return int((logits > logits[target]).sum().item())


def _topk(logits, k):
    return torch.topk(logits, k).indices.tolist()


def score_policy(model, patch_index, rows, id_to_name):
    """20 states per game -> top-1/top-5/rank/prob for the real next move."""
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
                raise RuntimeError("target %d illegal at move %d" % (target, m))
            top5 = _topk(lg, 5)
            steps.append({
                "move": m,
                "rank": _rank_of(lg, target),
                "top1": int(lg.argmax()),
                "top1_name": id_to_name[int(lg.argmax())],
                "top5": top5,
                "top5_names": [id_to_name[c] for c in top5],
                "prob": float(probs[m, target]),
            })
        games.append({"gameid": row["gameid"], "steps": steps})
    return games


def score_masked(enc, patch_index, rows, id_to_name):
    """Hide each picked slot in turn; does the model return the real champ?"""
    games = []
    for row in rows:
        champ = torch.tensor(row["champ"], dtype=torch.long)
        role = torch.tensor(row["role"], dtype=torch.long)
        # slot -> the move index at which that pick happened
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
                "top1": int(lg.argmax()),
                "top1_name": id_to_name[int(lg.argmax())],
                "top5": top5,
                "top5_names": [id_to_name[c] for c in top5],
                "prob": float(torch.softmax(lg, -1)[target]),
            })
        slots.sort(key=lambda s: (s["move"] is None, s["move"]))
        games.append({"gameid": row["gameid"], "slots": slots})
    return games


def score_value(model, kind, rows, patch_index=None):
    games = []
    for row in rows:
        samples = [PRO_DATA.build_prefix(row, m, NUM_CHAMPS, patch_index)
                   for m in range(N_MOVES)]
        batch = PRO_DATA.pro_collate(samples)
        with torch.no_grad():
            if kind == "prefix":
                logit = model(batch)["win_logit"]
            elif kind == "soloq":
                logit = model.forward_value(batch)
            elif kind == "draft":
                # PoolOutcomeHeads emits [logit blue, logit red]; the blue-win
                # logit is the difference (softmax over two classes).
                two = model(batch)["win_logits"]
                logit = two[:, 0] - two[:, 1]
            elif kind == "policy":
                logit = model(batch)["value_logit"]
            else:
                raise ValueError(kind)
        logit = logit.reshape(-1)
        if logit.numel() != len(samples):
            raise RuntimeError("%s produced %d logits for %d states"
                               % (kind, logit.numel(), len(samples)))
        games.append({"gameid": row["gameid"],
                      "logits": [float(x) for x in logit]})
    return games


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# --------------------------------------------------------------------- report
def aggregate(games, key="steps", hit_keys=("rank",)):
    n = sum(len(g[key]) for g in games)
    out = {"n": n}
    for k in (1, 5):
        out["top%d" % k] = sum(
            1 for g in games for s in g[key] if s["rank"] < k) / max(1, n)
    out["mean_rank"] = sum(s["rank"] for g in games for s in g[key]) / max(1, n)
    with_prob = [s["prob"] for g in games for s in g[key] if "prob" in s]
    if with_prob:
        out["mean_prob"] = sum(with_prob) / len(with_prob)
    return out


def by_phase(games):
    """Top-1/top-5 per draft phase, on next-move states only.

    The branch's mechanism claim is that the patch embedding pays off exactly
    where the board carries no information (ban1) - so the phase split is the
    interesting cut even at n=40.
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


def missing_checkpoints():
    """Checkpoint paths this machine does not have.

    Weights are git-ignored, so a fresh clone has none of them. Name the list
    up front instead of raising `FileNotFoundError` from inside a loader.
    """
    wanted = [p for _n, _f, p in POLICY_MODELS + MASKED_MODELS]
    wanted += [p for _n, _f, p, _k in VALUE_MODELS]
    return [p for p in wanted if not (ROOT / p).exists()]


def series_summary(rows):
    """One entry per pinned series, in the order the split lists them."""
    out = []
    for sid in TEST_SERIES:
        games = [r for r in rows if r["series_id"] == sid]
        if not games:
            continue
        out.append({
            "series_id": sid,
            "teams": [sid.split("|")[1], sid.split("|")[2]],
            "date": (games[0]["date"] or "")[:10],
            "n_games": len(games),
            "blue_wins": sum(1 for r in games if r["blue_win"]),
            "gameids": [r["gameid"] for r in games],
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "reports" / "test_split.json"))
    args = ap.parse_args(argv)

    missing = missing_checkpoints()
    if missing:
        raise SystemExit(
            "missing checkpoints - weights are git-ignored, so train them "
            "first:\n  " + "\n  ".join(missing))

    rows, id_to_name = load_test_rows()
    print("test split: %d games, %d series, patch %s"
          % (len(rows), len(TEST_SERIES), rows[0]["patch"]))
    for r in rows:
        print("  %s game %d: %s (blue) vs %s, blue_win=%d, first_pick=%d"
              % (r["gameid"], r["game_number"], r["blue_team"],
                 r["red_team"], r["blue_win"], r["first_pick"]))

    # corpus rows, for the patch popularity baseline only - read straight from
    # the cache, never rebuilt
    cache = torch.load(str(PRO_DIR / "pro_drafts.pt"), map_location="cpu",
                       weights_only=False)
    base_ranks, base_n = patch_popularity_reference(rows[0]["patch"],
                                                    cache["rows"])
    print("popularity baseline: %d corpus rows on patch %s"
          % (base_n, rows[0]["patch"]))

    report = {
        "test_split": {
            "pinned": list(TEST_SERIES),
            "corpus_max_date": P.CORPUS_MAX_DATE,
            "patch": rows[0]["patch"],
            "league": rows[0]["league"],
            "n_games": len(rows),
            "fearless": bool(rows[0]["fearless"]),
            "series": series_summary(rows),
        },
        "series": {
            "patch": rows[0]["patch"],
            "games": [{
                "gameid": r["gameid"], "game_number": r["game_number"],
                "series_id": r["series_id"],
                "date": r["date"], "blue_team": r["blue_team"],
                "red_team": r["red_team"], "blue_win": r["blue_win"],
                "first_pick": r["first_pick"],
                "unavailable": [id_to_name[c] for c in r["unavailable"]],
                "moves": [{
                    "move": m, "kind": "ban" if k == MOVE_BAN else "pick",
                    "side": "blue" if t == 0 else "red",
                    "champ": id_to_name[c],
                    "role": None if k == MOVE_BAN else ROLE_NAMES.get(ro, "?"),
                } for m, (k, t, c, ro) in enumerate(r["moves"])],
            } for r in rows],
        },
        "baseline": {"name": "patch %s move-frequency table" % rows[0]["patch"],
                     "n_rows": base_n},
    }

    # ---- next move
    report["next_move"] = {}
    print("\n== next move ==")
    for name, family, path in POLICY_MODELS:
        model, patch_index, cfg = load_policy(ROOT / path)
        games = score_policy(model, patch_index, rows, id_to_name)
        agg = aggregate(games)
        report["next_move"][name] = {"family": family, "path": path,
                                     "n_patches": int(cfg.get("n_patches", 0)),
                                     "games": games, "agg": agg,
                                     "by_phase": by_phase(games)}
        print("  %-16s top1 %.3f top5 %.3f mean rank %.2f"
              % (name, agg["top1"], agg["top5"], agg["mean_rank"]))

    # baseline, scored on the same states. Legality is applied here too: the
    # table would otherwise rank champions that are already banned or picked,
    # while every model has them at -inf. That is a ~10-place handicap on a
    # 173-champion list, so leaving it out would flatter the models.
    base_steps = []
    for r in rows:
        steps = []
        taken = set()
        for m, (_k, _t, c, _ro) in enumerate(r["moves"]):
            legal = [x for x in range(NUM_CHAMPS) if x not in taken]
            order = sorted(legal, key=lambda x: base_ranks[m][x])
            rank = {champ: pos for pos, champ in enumerate(order)}
            steps.append({"move": m, "rank": rank[int(c)]})
            taken.add(int(c))
        base_steps.append({"gameid": r["gameid"], "steps": steps})
    report["baseline"]["agg"] = aggregate(base_steps)
    report["baseline"]["by_phase"] = by_phase(base_steps)
    report["baseline"]["games"] = base_steps
    print("  %-16s top1 %.3f top5 %.3f mean rank %.2f"
          % ("popularity", report["baseline"]["agg"]["top1"],
             report["baseline"]["agg"]["top5"],
             report["baseline"]["agg"]["mean_rank"]))

    # ---- masked completion
    report["masked"] = {}
    print("\n== masked completion (one pick hidden at a time) ==")
    for name, family, path in MASKED_MODELS:
        enc, patch_index, cfg = load_masked(ROOT / path)
        games = score_masked(enc, patch_index, rows, id_to_name)
        agg = aggregate(games, key="slots")
        report["masked"][name] = {"family": family, "path": path,
                                  "n_patches": int(cfg.get("n_patches", 0)),
                                  "games": games, "agg": agg}
        print("  %-18s top1 %.3f top5 %.3f mean rank %.2f"
              % (name, agg["top1"], agg["top5"], agg["mean_rank"]))

    # ---- value
    report["value"] = {}
    print("\n== value (P blue wins after each move) ==")
    for name, family, path, kind in VALUE_MODELS:
        model, kind, platt = load_value(ROOT / path, kind)
        games = score_value(model, kind, rows)
        a, b = platt
        for g in games:
            g["p_blue"] = [_sigmoid(a * x + b) for x in g["logits"]]
        report["value"][name] = {"family": family, "path": path,
                                 "platt": list(platt), "games": games}
        last = [g["p_blue"][-1] for g in games]
        print("  %-16s P(blue) final: %s"
              % (name, ", ".join("%.3f" % p for p in last)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("\nwrote %s" % out)
    return report


if __name__ == "__main__":
    main()
