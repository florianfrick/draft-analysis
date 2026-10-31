"""Does the observed ban target a champion the opponent actually wanted?

The question this answers
------------------------
"Suppose my opponent drafts like my model does. Then the ban should be
whatever my model thinks they will pick next." That rule is testable
without any value function: take a real ban, find the opponent's first
pick after it, and ask where the banned champion ranks in the policy's
pick distribution at that moment.

    rank 1-3   the ban targeted something they demonstrably wanted
               -> the premise holds, and the search should find it
    rank ~87   the ban is unrelated to their pick distribution
               (uniform over ~173 legal champs)
               -> humans ban for other reasons, and no amount of
                  predicting their pick will reproduce their bans

Both numbers are reported against a popularity baseline (the frequency
table of the same move on the same split), because the policy's pick
distribution is itself close to frequency on ban1 - if the model's rank
matches the frequency rank, the model added nothing.

Caveat, stated up front: the state used is the opponent's *actual* first
pick, so the bans that happened between my ban and that pick are already
on the board. This measures intent given how the draft developed; it is
not a rollout of "what if I had banned c instead". That is the right
question for validating the premise, and the wrong question for scoring
a decision rule.

Usage:
    python model/policy/ban_intent.py \
        --ckpt model/policy/ckpt/run_20261025_000606_ablF-stage1/best_policy.pt
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import train as T  # noqa: E402
from model import PolicyValueModel  # noqa: E402
from soloq_model import SoloQPretrainedModel  # noqa: E402
from soloq_model import attr_matrix_for_config  # noqa: E402

PRO_DATA = T.PRO_DATA
EP = T.EP
PHASE_NAMES = {0: "bans1", 1: "picks1", 2: "bans2", 3: "picks2"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Ban intent analysis")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--phase", default="all",
                    choices=("all", "bans1", "bans2"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--half-life", type=float, default=75.0,
                    help="recency weighting; only affects row['weight'], "
                         "which this analysis does not use, but load_pro_rows "
                         "needs it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default="")
    return ap.parse_args(argv)


def build_model(blob, device):
    cfg = blob["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        attr_matrix=attr_matrix_for_config(cfg))
    model = PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg["state_dim"],
        n_econ=cfg["n_econ"], dropout=cfg.get("head_dropout", 0.0),
        n_patches=cfg.get("n_patches", 0),
        with_value=cfg.get("with_value", True)).to(device)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError("checkpoint mismatch: missing=%s unexpected=%s"
                           % (list(missing)[:6], list(unexpected)[:6]))
    model.eval()
    return model, cfg


def collect_targets(rows, idxs, phase="all"):
    """(row, ban move, banned champ, opponent's next pick move) tuples.

    Only bans whose opponent has a later pick *in the same game* are
    usable: a ban2 by the team that picks last has no "what would they
    have picked" to check against, because the game ends.
    """
    want = {"all": (0, 2), "bans1": (0,), "bans2": (2,)}[phase]
    out = []
    for i in idxs:
        moves = rows[i]["moves"]
        for m, (kind, team, champ, _role) in enumerate(moves):
            if kind != PRO_DATA.MOVE_BAN:
                continue
            if PRO_DATA.PHASE_OF[m] not in want:
                continue
            opp = 1 - team
            p = next((j for j in range(m + 1, len(moves))
                      if moves[j][0] == PRO_DATA.MOVE_PICK
                      and moves[j][1] == opp), None)
            if p is None:
                continue
            out.append({"row": i, "ban_move": m, "ban_team": team,
                        "champ": champ, "pick_move": p})
    return out


@torch.no_grad()
def rank_under_policy(model, rows, targets, num_champs, patch_index,
                      device, batch_size):
    """Rank of each banned champ in the opponent's pick distribution.

    Grouped by pick_move so each batch is a single draft state - 20
    forwards per split instead of one per ban.

    The banned champ is illegal at that state, so its logit is -inf and
    every rank would tie at the bottom. Clearing it from `taken_mask`
    restores its raw logit; the other champions' logits are untouched
    because legality is applied after the head, so the ranking of
    everyone else is unaffected.
    """
    by_move = defaultdict(list)
    for t in targets:
        by_move[t["pick_move"]].append(t)
    for t in targets:
        t["rank"] = None
        t["n_legal"] = None
    for p, group in sorted(by_move.items()):
        samples = [PRO_DATA.build_prefix(rows[g["row"]], p, num_champs,
                                         patch_index) for g in group]
        batch = PRO_DATA.pro_collate(samples)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        c = torch.tensor([g["champ"] for g in group], device=device)
        taken = batch["taken_mask"].clone()
        taken[torch.arange(len(c), device=device), c] = False
        batch["taken_mask"] = taken
        logits = model(batch)["policy_logits"]
        tgt = logits.gather(1, c[:, None]).squeeze(1)
        ranks = (logits > tgt[:, None]).sum(1)
        n_legal = torch.isfinite(logits).sum(1)
        for g, r, n in zip(group, ranks.tolist(), n_legal.tolist()):
            g["rank"] = int(r)
            g["n_legal"] = int(n)
    return targets


def popularity_ranks(rows, idxs, targets, num_champs):
    """Same rank, but scored by the split's own frequency table.

    Leave-one-out on the *picking* team's move: counts are taken over the
    other rows of the split at the same move index, so the row being
    scored does not vote for itself. This is the bar the policy has to
    beat - on ban1 it was 0.98x the frequency table, i.e. level.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for i in idxs:
        for m, (kind, _team, champ, _role) in enumerate(rows[i]["moves"]):
            if kind == PRO_DATA.MOVE_PICK:
                counts[m][champ] += 1
    out = []
    for t in targets:
        c = counts[t["pick_move"]]
        mine = c.get(t["champ"], 0)
        rank = sum(1 for v in c.values() if v > mine)
        out.append(rank)
    return out


def summarize(ranks, n_legal):
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    mean_legal = sum(n_legal) / n
    return {
        "n": n,
        "mean_rank": sum(ranks) / n,
        "median_rank": sorted(ranks)[n // 2],
        "mean_uniform_rank": (mean_legal + 1) / 2.0,
        "top1": sum(r < 1 for r in ranks) / n,
        "top3": sum(r < 3 for r in ranks) / n,
        "top5": sum(r < 5 for r in ranks) / n,
        "top10": sum(r < 10 for r in ranks) / n,
        "n_legal": mean_legal,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, cfg = build_model(blob, device)
    print("ckpt: %s (step %s)" % (args.ckpt, blob.get("step")))

    rows, splits = T.load_pro_rows(args)
    idxs = splits[args.split]
    num_champs = cfg["num_champs"]
    patch_index = PRO_DATA.patch_index_from_vocab(cfg.get("patch_vocab"))
    print("%s: %d drafts" % (args.split, len(idxs)))

    targets = collect_targets(rows, idxs, args.phase)
    print("ban decisions with an opponent pick after them: %d" % len(targets))
    targets = rank_under_policy(model, rows, targets, num_champs,
                                patch_index, device, args.batch_size)
    targets = [t for t in targets if t["rank"] is not None]
    pop = popularity_ranks(rows, idxs, targets, num_champs)

    report = {"ckpt": args.ckpt, "split": args.split, "phase": args.phase,
              "by_phase": {}, "baseline": {}}
    for ph in sorted({PRO_DATA.PHASE_OF[t["ban_move"]] for t in targets}):
        sel = [t for t in targets if PRO_DATA.PHASE_OF[t["ban_move"]] == ph]
        idx_sel = [k for k, t in enumerate(targets)
                   if PRO_DATA.PHASE_OF[t["ban_move"]] == ph]
        name = PHASE_NAMES[ph]
        report["by_phase"][name] = summarize(
            [t["rank"] for t in sel], [t["n_legal"] for t in sel])
        report["baseline"][name] = summarize(
            [pop[k] for k in idx_sel],
            [t["n_legal"] for t in sel])
    report["all"] = summarize([t["rank"] for t in targets],
                              [t["n_legal"] for t in targets])
    report["baseline"]["all"] = summarize(pop,
                                          [t["n_legal"] for t in targets])

    print("\n%-8s %7s %8s %9s %8s %7s %7s %7s"
          % ("phase", "n", "mean_rk", "uniform", "model@3", "base@3",
             "model@5", "base@5"))
    for name in list(report["by_phase"]) + ["all"]:
        m = report["all"] if name == "all" else report["by_phase"][name]
        b = report["baseline"][name]
        print("%-8s %7d %8.2f %9.2f %8.3f %7.3f %7.3f %7.3f"
              % (name, m["n"], m["mean_rank"], m["mean_uniform_rank"],
                 m["top3"], b["top3"], m["top5"], b["top5"]))
    print("\nuniform-rank column = what a random ban would score on average.")

    out = Path(args.out) if args.out else HERE / "ban_intent.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
