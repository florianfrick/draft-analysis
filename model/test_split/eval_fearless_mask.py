"""What does an output-only fearless mask actually buy the masked models?

The three `masking-*` checkpoints have no `unavailable` input: the task they
were trained on never told them a champion was gone. So the only zero-cost
way to make them respect fearless is on the output side - set the pool to
-inf and renormalise, which is exactly what `frontend/server._top5` does.

This measures whether that is enough, on the four fearless games of the test
split (40 slots per model). Three readings per slot:

  raw        no mask at all - the model's literal top-5
  +taken     bans and other picks masked (what the frontend always did)
  +pool      taken *and* the fearless pool masked (what it does now)

The comparison that matters is `+taken` vs `+pool`, because the real pick is
never in the pool. That makes the mask monotone: -inf cannot reorder the
champions that remain, and the target stays legal, so its rank can only
improve. "Masking does not hurt" is therefore arithmetic, not a finding.
The finding is the absolute quality after masking, and the size of the
pool's share of the model's probability mass - that is what says whether
the remaining ranking is still informed or just the tail of a distribution
whose head was removed.

Usage:
    python model/test_split/eval_fearless_mask.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_test_split as E  # noqa: E402

N_SLOTS = 10
NUM_CHAMPS = E.NUM_CHAMPS
MOVE_PICK = E.MOVE_PICK
PHASE_OF = E.PHASE_OF


def wilson(hits, n, z=1.96):
    """95% interval. At n=40 a bare rate is worth little on its own."""
    if not n:
        return (0.0, 0.0)
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _masked_logits(lg, illegal):
    out = lg.clone()
    if illegal:
        out[torch.tensor(sorted(illegal), dtype=torch.long)] = float("-inf")
    return out


def rank_of(lg, illegal, target):
    return int((_masked_logits(lg, illegal) > lg[target]).sum().item())


def role_prior(rows):
    """P(champ | role) over the whole pro corpus - the no-model baseline."""
    counts = torch.zeros(5, NUM_CHAMPS)
    for r in rows:
        for k, _t, c, ro in r["moves"]:
            if int(k) == MOVE_PICK:
                counts[int(ro), int(c)] += 1.0
    return counts


def policy_reference(rows, report_path):
    """Top-1/top-5 of the fearless-aware policy on the same pick states."""
    if not report_path.exists():
        return {}
    rep = json.load(open(report_path))
    out = {}
    for name, blob in (rep.get("next_move") or {}).items():
        ranks = []
        for g in blob["games"]:
            row = next((r for r in rows if r["gameid"] == g["gameid"]), None)
            if row is None or not row["unavailable"]:
                continue
            for s in g["steps"]:
                if PHASE_OF[s["move"]] in (1, 3):     # pick phases
                    ranks.append(s["rank"])
        n = len(ranks)
        out[name] = {
            "n": n,
            "top1": sum(1 for r in ranks if r < 1) / n if n else 0.0,
            "top5": sum(1 for r in ranks if r < 5) / n if n else 0.0,
        }
    return out


def score_model(enc, patch_index, rows, prior):
    """Every slot of every game, under all three masks."""
    buckets = {"fearless": [], "game1": []}
    for row in rows:
        champ = torch.tensor(row["champ"], dtype=torch.long)
        role = torch.tensor(row["role"], dtype=torch.long)
        pool = set(int(c) for c in row["unavailable"])
        pid = E.PRO_DATA.patch_to_id(row.get("patch"), patch_index)
        board = set(int(c) for _k, _t, c, _r in row["moves"])
        key = "fearless" if pool else "game1"
        for slot in range(N_SLOTS):
            cids = champ.clone()
            cids[slot] = E.CHAMP_MASK_ID
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
                lg = enc(batch)["champ_logits"][0, slot].float()
            target = int(champ[slot])
            taken = set(board) - {target}     # the hidden slot is unseen
            raw5 = torch.topk(lg, 5).indices.tolist()
            pool5 = torch.topk(_masked_logits(lg, taken | pool), 5).indices.tolist()
            # role-frequency prior over the same legal set
            pr = prior[int(role[slot])].clone()
            pr[torch.tensor(sorted(taken | pool), dtype=torch.long)] = -1.0
            mass = (torch.softmax(lg, -1)[torch.tensor(sorted(pool))].sum()
                    if pool else torch.tensor(0.0))
            buckets[key].append({
                "game": int(row["game_number"]),
                "n_pool": len(pool),
                "raw_rank": E._rank_of(lg, target),
                "raw5_has_taken": any(c in taken for c in raw5),
                "raw5_has_pool": any(c in pool for c in raw5),
                "raw1_in_pool": int(lg.argmax()) in pool,
                "taken_rank": rank_of(lg, taken, target),
                "pool_rank": rank_of(lg, taken | pool, target),
                "prior_rank": int((pr > pr[target]).sum().item()),
                "pool_mass": float(mass),
            })
    return buckets


def summarise(ss):
    n = len(ss)
    out = {"n": n}
    for key, rank in (("raw", "raw_rank"), ("taken", "taken_rank"),
                      ("pool", "pool_rank"), ("prior", "prior_rank")):
        h1 = sum(s[rank] < 1 for s in ss)
        h5 = sum(s[rank] < 5 for s in ss)
        out[key] = {"top1": h1 / n, "top5": h5 / n,
                    "top1_ci": wilson(h1, n), "top5_ci": wilson(h5, n),
                    "mean_rank": sum(s[rank] for s in ss) / n}
    hard = [s for s in ss if s["raw1_in_pool"]]
    out["hard"] = {
        "n": len(hard),
        "top1": sum(s["pool_rank"] < 1 for s in hard) / len(hard) if hard else 0.0,
        "top5": sum(s["pool_rank"] < 5 for s in hard) / len(hard) if hard else 0.0,
    }
    out["raw5_pool"] = sum(s["raw5_has_pool"] for s in ss)
    out["raw5_taken"] = sum(s["raw5_has_taken"] for s in ss)
    out["pool_mass"] = sum(s["pool_mass"] for s in ss) / n
    return out


def main():
    missing = E.missing_checkpoints()
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))

    rows, _id_to_name = E.load_test_rows()
    cache = torch.load(str(E.PRO_DIR / "pro_drafts.pt"), map_location="cpu",
                       weights_only=False)
    prior = role_prior(cache["rows"])
    print("test split: %d games, fearless pool sizes %s"
          % (len(rows), [len(r["unavailable"]) for r in rows]))

    report = {"test_split": {"n_games": len(rows),
                             "pool_sizes": [len(r["unavailable"]) for r in rows]},
              "models": {}}
    for name, family, path in E.MASKED_MODELS:
        enc, patch_index, _cfg = E.load_masked(E.ROOT / path)
        buckets = score_model(enc, patch_index, rows, prior)
        g1 = summarise(buckets["game1"])
        fl = summarise(buckets["fearless"])
        report["models"][name] = {"family": family, "path": path,
                                  "game1": g1, "fearless": fl}
        print("\n== %s (%s) ==" % (name, family))
        print("  G1, pool empty (n=%d)   taken top1 %.3f top5 %.3f | "
              "pool top1 %.3f top5 %.3f"
              % (g1["n"], g1["taken"]["top1"], g1["taken"]["top5"],
                 g1["pool"]["top1"], g1["pool"]["top5"]))
        print("  fearless (n=%d)" % fl["n"])
        for key, label in (("raw", "raw, no mask "), ("taken", "+taken      "),
                           ("pool", "+taken+pool "), ("prior", "role prior  ")):
            print("    %s top1 %.3f top5 %.3f [%.2f, %.2f] mean rank %.2f"
                  % (label, fl[key]["top1"], fl[key]["top5"],
                     fl[key]["top5_ci"][0], fl[key]["top5_ci"][1],
                     fl[key]["mean_rank"]))
        print("    raw top-5 with a pool champion: %d/%d"
              % (fl["raw5_pool"], fl["n"]))
        print("    pool share of probability mass: %.3f" % fl["pool_mass"])
        print("    when raw top-1 was in the pool (n=%d): top1 %.3f top5 %.3f"
              % (fl["hard"]["n"], fl["hard"]["top1"], fl["hard"]["top5"]))

    pol = policy_reference(rows, HERE / "reports" / "test_split.json")
    report["next_move_reference"] = pol
    if pol:
        print("\n== next-move policy, same fearless pick states (live "
              "conditioning, no future picks) ==")
        for name, v in pol.items():
            print("  %-18s top1 %.3f top5 %.3f (n=%d)"
                  % (name, v["top1"], v["top5"], v["n"]))

    out = HERE / "reports" / "fearless_mask.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\nwrote %s" % out)
    return report


if __name__ == "__main__":
    main()
