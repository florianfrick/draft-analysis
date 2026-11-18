"""Role-conditioned next-pick evaluation on the canonical test split.

For every prefix of every test game and every still-empty pick slot, rank the
champion that actually ended up in that slot. This is NOT the next_move suite:
the query is "who belongs in red MID given what is known now", and it is asked
even when the next real move is a ban. Never pool the two numbers.

Run:
    docker compose run --rm -T dev python pro/model/naivetransformer/eval_role.py \
        --ckpt pro/model/checkpoints/<role-checkpoint>/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.proonly import common as C  # noqa: E402
from pro.model.naivetransformer import model as NT  # noqa: E402
from pro.model.test_split import split as S  # noqa: E402

P = C.PRO_DATA
ROLE_NAMES = ["TOP", "JNG", "MID", "BOT", "SUP"]
NUM_CHAMPS = 173


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--batch-size", type=int, default=160)
    ap.add_argument("--limit-games", type=int, default=0)
    return ap.parse_args(argv)


def frequency_tables(rows, train_idx):
    """Per (patch, role) champion counts from TRAIN rows only."""
    by_pair = defaultdict(lambda: np.zeros(NUM_CHAMPS, dtype=np.float64))
    by_role = np.zeros((5, NUM_CHAMPS), dtype=np.float64)
    for i in train_idx:
        row = rows[i]
        patch = P.patch_key(row.get("patch"))
        for kind, _team, champ, role in row["moves"]:
            if kind != P.MOVE_PICK:
                continue
            by_pair[(patch, int(role))][int(champ)] += 1.0
            by_role[int(role)][int(champ)] += 1.0
    return by_pair, by_role


def score_table(by_pair, by_role, patch, role):
    t = by_pair.get((patch, role))
    return by_pair[(patch, role)] if t is not None else by_role[role]


def new_slice():
    return {"n": 0, "hits": {1: 0, 5: 0, 10: 0},
            "base_hits": {1: 0, 5: 0, 10: 0}, "rank_sum": 0}


def bump(table, rank, base_rank):
    table["n"] += 1
    table["rank_sum"] += rank
    for k in table["hits"]:
        table["hits"][k] += int(rank < k)
        table["base_hits"][k] += int(base_rank < k)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    rows = S.load_rows(verbose=False)
    splits = S.build(rows)
    ok, diffs = S.verify(rows, splits)
    if not ok:
        raise SystemExit("split does not match manifest:\n" + "\n".join(diffs))
    test_idx = splits["test"]
    if args.limit_games:
        test_idx = test_idx[:args.limit_games]
    test_rows = [rows[i] for i in test_idx]
    by_pair, by_role = frequency_tables(rows, splits["train"])
    model, patch_index, cfg = NT.load_from_checkpoint(args.ckpt, device)
    fearless_roles = P.build_fearless_roles(rows)
    print("role eval: %d test games, %d states/game, checkpoint %s"
          % (len(test_rows), 20, args.ckpt), flush=True)

    slices = defaultdict(new_slice)
    states = []
    t0 = time.time()
    for row in test_rows:
        for move in range(20):
            sample = P.build_prefix(row, move, NUM_CHAMPS, patch_index,
                                    fearless_roles)
            states.append({
                "sample": sample,
                "patch": P.patch_key(row.get("patch")),
                "is_ban": int(sample["is_ban"]),
                "side": int(sample["to_move"]),
                "n_fearless": int(sample["n_fearless"]),
            })

    for start in range(0, len(states), args.batch_size):
        chunk = states[start:start + args.batch_size]
        batch = P.pro_collate([s["sample"] for s in chunk])
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            slot_logits = model(batch)["slot_logits"]
        logits_np = slot_logits.detach().cpu().numpy()
        taken_np = batch["taken_mask"].cpu().numpy()
        unavail_np = batch["unavailable_mask"].cpu().numpy()
        empty_np = batch["pad_mask"].cpu().numpy()
        targets_np = batch["slot_target"].cpu().numpy()
        for i, meta in enumerate(chunk):
            empty = np.flatnonzero(empty_np[i])
            for slot in empty.tolist():
                target = int(targets_np[i, slot])
                lg = logits_np[i, slot]
                rank = int(np.count_nonzero(lg > lg[target]))
                role = slot % 5
                legal = np.flatnonzero(~taken_np[i] & ~unavail_np[i])
                table = score_table(by_pair, by_role, meta["patch"], role)
                scores = table[legal]
                base_rank = int(np.count_nonzero(
                    (scores > table[target])
                    | ((scores == table[target]) & (legal < target))))
                keys = ["overall",
                        "next_ban" if meta["is_ban"] else "next_pick",
                        "side_blue" if meta["side"] == 0 else "side_red",
                        "role_" + ROLE_NAMES[role],
                        "fearless_yes" if meta["n_fearless"] else "fearless_no"]
                if meta["is_ban"] and meta["n_fearless"]:
                    keys.append("ban_with_fearless")
                for key in keys:
                    bump(slices[key], rank, base_rank)
        if (start // args.batch_size) % 20 == 0:
            print("  %d/%d states (%.0fs)" % (start + len(chunk), len(states),
                                             time.time() - t0), flush=True)

    def summarise(t):
        n = max(1, t["n"])
        out = {"n": t["n"], "mean_rank": t["rank_sum"] / n}
        for k in (1, 5, 10):
            out["top%d" % k] = t["hits"][k] / n
            out["base_top%d" % k] = t["base_hits"][k] / n
        return out

    report = {
        "protocol": {
            "ckpt": args.ckpt, "arch": cfg.get("arch"),
            "slot_head": cfg.get("slot_head", "shared"),
            "n_test_games": len(test_rows), "n_states": len(states),
            "split": "%d/%d/%d" % (len(splits["train"]), len(splits["val"]),
                                   len(splits["test"])),
            "baseline": "train-only per-patch per-role frequency, legal-filtered",
            "note": "rank of the eventual occupant of each empty slot; "
                    "slices are nested and must not be summed",
        },
        "slices": {k: summarise(v) for k, v in sorted(slices.items())},
    }
    out = Path(args.out) if args.out else \
        HERE / "logs" / ("role_eval_" + Path(args.ckpt).parent.name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for key in ("overall", "next_ban", "next_pick", "side_blue", "side_red",
                "fearless_yes", "fearless_no", "ban_with_fearless"):
        v = report["slices"].get(key)
        if v:
            print("%-20s n=%-7d top1 %.4f (base %.4f)  top5 %.4f (base %.4f)"
                  % (key, v["n"], v["top1"], v["base_top1"], v["top5"],
                     v["base_top5"]), flush=True)
    print("wrote", out, flush=True)
    return report


if __name__ == "__main__":
    main()
