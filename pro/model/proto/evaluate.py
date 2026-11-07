"""One evaluator for the seeded split: exact move, block partner, paired tests.

The metric set, and why each piece is here:

* **exact top-1/5/10** - the product question ("is the champion the team took
  in the panel"). Kept as the headline because it is the only thing comparable
  to the popularity baseline and to every number in STATUS.
* **block partner top-1/5** - at the FIRST move of a same-team double-pick
  block, where does the champion the team takes *next* rank? This is the
  exchangeability diagnostic: it asks the model to know the pair before the
  pair is half-consumed. Measured on the pre-legality distribution, which is
  the only place the question is answerable (once move m+1 has happened, both
  members are taken and masked). Block Jaccard was rejected for exactly this
  reason - see the record.
* **per-move and ban/pick slices** - a pooled number hides which phase moved.
* **paired tests** - two arms on identical rows. McNemar on the discordant
  pairs plus a game-level bootstrap of the delta, because marginal Wilson
  intervals overlapping says nothing about the difference.

Usage:
    python pro/model/proto/evaluate.py --ckpt A.pt --ckpt B.pt --split test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRO_DIR = ROOT / "data" / "pro play"
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load("pro_datasets", PRO_DIR / "datasets.py")
EP = _load("eval_protocol", PRO_DIR / "eval_protocol.py")

from solo_queue.model import PolicyValueModel                                   # noqa: E402
from solo_queue.model.pretraining.base_model import SiameseBase       # noqa: E402
from solo_queue.model.pretraining.base_model import attr_matrix_for_config     # noqa: E402
from pro.model.proto import split as SPLIT                               # noqa: E402
from pro.model.proto.policy import ProtoPolicy, build_prior              # noqa: E402

MOVE_PICK = PRO_DATA.MOVE_PICK
N_MOVES = PRO_DATA.N_MOVES


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--ensemble", action="store_true",
                    help="average the logits of all --ckpt instead of scoring "
                         "them separately")
    ap.add_argument("--split", default="test",
                    choices=("train", "val", "test", "both"))
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="")
    return ap.parse_args(argv)


def build_model(blob, device, rows=None, patch_index=None, train_idx=None):
    cfg = blob["config"]
    enc = SiameseBase(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        attr_matrix=attr_matrix_for_config(cfg),
        n_patches=cfg.get("n_patches", 0))
    # A prototype arm can have a wider head (slot_mask) or extra parameters
    # (block/prior); PolicyValueModel would silently load it with strict=False
    # and produce a plausible wrong number, so the class is chosen from cfg.
    flags = (cfg.get("block_head"), cfg.get("slot_mask"),
             cfg.get("prior_bias"))
    kw = dict(embed_dim=cfg["embed_dim"], state_dim=cfg["state_dim"],
              n_econ=cfg.get("n_econ", 0),
              dropout=cfg.get("head_dropout", 0.0),
              n_patches=cfg.get("n_patches", 0),
              with_value=cfg.get("with_value", True))
    if any(flags):
        model = ProtoPolicy(enc, block_head=bool(flags[0]),
                            slot_mask=bool(flags[1]),
                            prior_bias=bool(flags[2]), **kw).to(device)
    else:
        model = PolicyValueModel(enc, **kw).to(device)
    if cfg.get("prior_bias"):
        model.set_prior(build_prior(rows, train_idx, patch_index,
                                    cfg["n_patches"], PRO_DATA.N_MOVES,
                                    PRO_DATA.patch_to_id))
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError("checkpoint mismatch: missing=%s unexpected=%s"
                           % (list(missing)[:5], list(unexpected)[:5]))
    model.eval()
    return model, cfg


@torch.no_grad()
def run(models, rows, indices, patch_index, cfg, device, batch_size):
    """Per-example arrays: rank of truth, rank of the block partner.

    `models` is a list: more than one means the logits are averaged (an
    ensemble). Arms this close are separated by less than the seed spread, so
    the ensemble is the honest test of whether the differences carry signal.
    """
    ds = PRO_DATA.ProNextMoveDataset(
        rows, indices, num_champs=cfg["num_champs"],
        patch_index=patch_index, prefix_mode="all", swap_aug=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=PRO_DATA.pro_collate)
    by_game = {rows[i]["gameid"]: rows[i] for i in indices}
    rank, prank, move, ban, game, partner_n = [], [], [], [], [], []
    games = sorted(by_game)
    gid = {g: i for i, g in enumerate(games)}
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        cl = models[0](batch)["policy_logits"]
        for extra in models[1:]:
            cl = cl + extra(batch)["policy_logits"]
        if len(models) > 1:
            cl = cl / len(models)
        tc = batch["target_champ"]
        truth = cl.gather(1, tc[:, None]).squeeze(1)
        r = (cl > truth[:, None]).sum(1)
        for j in range(tc.numel()):
            m = int(batch["move_idx"][j])
            g = batch["meta"][j]["gameid"]
            mv = by_game[g]["moves"]
            rank.append(int(r[j]))
            move.append(m)
            ban.append(int(batch["is_ban"][j]))
            game.append(gid[g])
            # partner = the next move, only when it is the same team's pick
            if m + 1 < N_MOVES and mv[m][0] == MOVE_PICK \
                    and mv[m + 1][0] == MOVE_PICK \
                    and mv[m + 1][1] == mv[m][1]:
                p = mv[m + 1][2]
                prank.append(int((cl[j] > cl[j, p]).sum()))
                partner_n.append(1)
            else:
                prank.append(-1)
                partner_n.append(0)
    return {"rank": np.array(rank, np.int32), "prank": np.array(prank, np.int32),
            "move": np.array(move, np.int8), "ban": np.array(ban, np.int8),
            "game": np.array(game, np.int32),
            "partner": np.array(partner_n, np.int8),
            "games": games}


def block_base_ranks(rows, indices, move_rank, num_champs):
    """Popularity-table partner rank, on the same examples."""
    out = []
    for i in indices:
        mv = rows[i]["moves"]
        for m in range(N_MOVES):
            if m + 1 < N_MOVES and mv[m][0] == MOVE_PICK \
                    and mv[m + 1][0] == MOVE_PICK \
                    and mv[m + 1][1] == mv[m][1]:
                out.append(move_rank[m][mv[m + 1][2]])
            else:
                out.append(-1)
    return np.array(out, np.int32)


def metrics(a, base_prank=None):
    r, p, ban = a["rank"], a["prank"], a["ban"]
    pm = a["partner"] == 1
    out = {"n": int(r.size)}
    for k in (1, 5, 10):
        out["top%d" % k] = float((r < k).mean())
    out["mean_rank"] = float(r.mean())
    out["ban_top1"] = float((r[ban == 1] < 1).mean())
    out["ban_top5"] = float((r[ban == 1] < 5).mean())
    out["pick_top1"] = float((r[ban == 0] < 1).mean())
    out["pick_top5"] = float((r[ban == 0] < 5).mean())
    out["by_move"] = {str(m): {"n": int((a["move"] == m).sum()),
                              "top1": float((r[a["move"] == m] < 1).mean()),
                              "top5": float((r[a["move"] == m] < 5).mean())}
                      for m in range(N_MOVES)}
    if pm.any():
        out["partner_n"] = int(pm.sum())
        out["partner_top1"] = float((p[pm] < 1).mean())
        out["partner_top5"] = float((p[pm] < 5).mean())
        out["partner_mean_rank"] = float(p[pm].mean())
        if base_prank is not None:
            bp = base_prank[pm]
            out["partner_base_top1"] = float((bp < 1).mean())
            out["partner_base_top5"] = float((bp < 5).mean())
    return out


def mcnemar(hit_a, hit_b):
    """Exact-ish McNemar on two aligned hit vectors -> (b, c, chi2, p)."""
    b = int(np.sum(hit_a & ~hit_b))
    c = int(np.sum(~hit_a & hit_b))
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # survival function of chi2 with 1 df = erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return b, c, chi2, p


def game_bootstrap(a, b, k=5, n_boot=2000, seed=0):
    """Bootstrap the top-k difference by resampling whole games."""
    rng = np.random.default_rng(seed)
    ha = (a["rank"] < k).astype(np.float64)
    hb = (b["rank"] < k).astype(np.float64)
    g = a["game"]
    n_games = int(g.max()) + 1
    order = np.argsort(g, kind="stable")
    gs = g[order]
    bounds = np.searchsorted(gs, np.arange(n_games + 1))
    d = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_games, n_games)
        idx = np.concatenate([np.arange(bounds[i], bounds[i + 1])
                              for i in pick])
        d.append(ha[idx].mean() - hb[idx].mean())
    d = np.sort(np.array(d))
    return float(ha.mean() - hb.mean()), float(d[int(0.025 * n_boot)]), \
        float(d[int(0.975 * n_boot)])


def fmt(m, label):
    line = ("%-22s n=%-6d top1 %.4f top5 %.4f top10 %.4f | ban %.4f pick %.4f"
            % (label, m["n"], m["top1"], m["top5"], m["top10"],
               m["ban_top5"], m["pick_top5"]))
    if "partner_n" in m:
        line += ("\n%-22s partner n=%-6d top1 %.4f top5 %.4f rank %.1f"
                 % ("", m["partner_n"], m["partner_top1"],
                    m["partner_top5"], m["partner_mean_rank"]))
        if "partner_base_top5" in m:
            line += "   (base top1 %.4f top5 %.4f)" % (
                m["partner_base_top1"], m["partner_base_top5"])
    return line


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    rows, _ = PRO_PIPE.build_rows()
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits, man = SPLIT.load(rows)
    _, patch_index = PRO_DATA.build_patch_vocab(rows)
    names = args.name or [Path(c).parent.name for c in args.ckpt]

    targets = (["val", "test"] if args.split == "both" else [args.split])
    payload = {}
    for split_name in targets:
        idxs = splits[split_name]
        move_rank = EP.move_popularity_ranks(rows, splits["train"],
                                             PRO_DATA.NUM_REAL_CHAMPS)
        base_pr = block_base_ranks(rows, idxs, move_rank,
                                   PRO_DATA.NUM_REAL_CHAMPS)
        print("\n=== %s: %d drafts, %d states (seed %d)"
              % (split_name, len(idxs), len(idxs) * N_MOVES, man["seed"]))
        res = {}
        arrays = {}
        if args.ensemble:
            models, cfg = [], None
            for ckpt in args.ckpt:
                blob = torch.load(str(ckpt), map_location="cpu",
                                  weights_only=False)
                mdl, cfg = build_model(blob, device, rows=rows,
                                       patch_index=patch_index,
                                       train_idx=splits["train"])
                models.append(mdl)
            label = "ensemble(%d)" % len(models)
            a = run(models, rows, idxs, patch_index, cfg, device,
                    args.batch_size)
            m = metrics(a, base_pr)
            print(fmt(m, label))
            res[label] = m
            arrays[label] = a
            del models
            torch.cuda.empty_cache()
        for ckpt, name in zip(args.ckpt, names):
            blob = torch.load(str(ckpt), map_location="cpu",
                              weights_only=False)
            model, cfg = build_model(blob, device, rows=rows,
                                     patch_index=patch_index,
                                     train_idx=splits["train"])
            a = run([model], rows, idxs, patch_index, cfg, device,
                    args.batch_size)
            m = metrics(a, base_pr)
            print(fmt(m, name))
            res[name] = m
            arrays[name] = a
            del model
            torch.cuda.empty_cache()
        # paired comparisons against the first scored entry
        if len(arrays) > 1:
            ref = list(arrays)[0]
            res["paired_vs_" + ref] = {}
            for name in list(arrays)[1:]:
                for k in (1, 5):
                    b, c, chi2, p = mcnemar(arrays[ref]["rank"] < k,
                                            arrays[name]["rank"] < k)
                    d, lo, hi = game_bootstrap(arrays[ref], arrays[name], k=k)
                    res["paired_vs_" + ref][name + "_top%d" % k] = {
                        "delta": d, "ci": [lo, hi], "a_only": b, "b_only": c,
                        "chi2": chi2, "p": p}
                    print("  paired top%d %s vs %s: %+.4f  boot [%+.4f, "
                          "%+.4f]  mcnemar a-only %d b-only %d p=%.3g"
                          % (k, name, ref, d, lo, hi, b, c, p))
                pm = arrays[ref]["partner"] == 1
                if pm.any():
                    b, c, chi2, p = mcnemar(arrays[ref]["prank"][pm] < 5,
                                            arrays[name]["prank"][pm] < 5)
                    print("  paired partner@5 %s vs %s: a-only %d b-only %d "
                          "p=%.3g" % (name, ref, b, c, p))
                    res["paired_vs_" + ref][name + "_partner5"] = {
                        "a_only": b, "b_only": c, "chi2": chi2, "p": p}
        payload[split_name] = res

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1), "utf-8")
        print("\nwrote", args.json)
    return payload


if __name__ == "__main__":
    main()
