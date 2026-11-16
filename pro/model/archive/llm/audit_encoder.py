"""Raw tensors at the boundary, and what the encoder's own head scores here.

Moves 0-6 score ~10 % exact, moves 7-19 score ~100 %, and it holds with the
board text removed. The tensor-level leak checks came back clean, so either the
prefix tensors encode the answer in a way the checks above do not name, or the
recorded 18.32 % policy baseline was measured on a different path.

The second question is settled by running the encoder's *own* policy head on
these exact batches: if it scores ~18 % the data is fine and the leak is in the
LLM path; if it scores ~100 % the dataset path is the leak.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import common as C
from pro.model.archive.llm import data as D


def main():
    rows = C.load_rows("2022")
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    ds = D.eval_dataset(rows, split["val"], patch_index, seed=0, val_games=200)

    # ---- raw tensors for the first game, moves 5..9
    print("first game's 20 states, moves 5-9")
    for i in range(5, 10):
        s = ds[i]
        t = int(s["target_champ"])
        print("\n  move_idx %d | is_ban %d | n_bans %d | target %r (id %d)"
              % (int(s["move_idx"]), int(s["is_ban"]), int(s["n_bans"]),
                 names[t], t))
        for k in ("champ_ids", "role_ids", "pad_mask", "taken_mask"):
            v = s[k]
            if k == "taken_mask":
                print("    %-12s sum=%d  target_set=%s"
                      % (k, int(v.sum()), bool(v[t])))
            else:
                print("    %-12s %s" % (k, v.tolist()))
        print("    target_role  %d" % int(s["target_role"]))

    # ---- the encoder's own policy head on this exact path
    print("\n" + "=" * 72)
    print("encoder policy head on this eval path")
    print("=" * 72, flush=True)
    enc, cfg, blob = C.build_frozen_encoder(C.ENCODER_CKPT, device="cuda")
    head = blob.get("head") if isinstance(blob, dict) else None
    print("checkpoint keys:", sorted(blob.keys()) if isinstance(blob, dict)
          else type(blob))
    enc.eval()

    ld = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False, collate_fn=D.collate, num_workers=0)

    correct = total = 0
    per_move = {}
    with torch.no_grad():
        for batch, samples in ld:
            batch = D.to_device(batch, "cuda")
            h = enc._encode(batch)
            if head is None:
                print("no head in the checkpoint; cannot score")
                return
            logits = head(h)
            pred = logits.argmax(dim=-1)
            tgt = batch["target_champ"]
            ok = (pred == tgt)
            correct += int(ok.sum())
            total += ok.numel()
            for j, s in enumerate(samples):
                mv = int(s["move_idx"])
                d = per_move.setdefault(mv, [0, 0])
                d[0] += int(ok[j])
                d[1] += 1

    print("\noverall top-1 %.4f (%d states)" % (correct / total, total))
    print("recorded baseline    %.4f" % C.BASELINE_VAL["top1"])
    print("\nby move:")
    for mv in sorted(per_move):
        c, n = per_move[mv]
        print("  move %2d  %.4f  (n=%d)" % (mv, c / n, n))


if __name__ == "__main__":
    main()
