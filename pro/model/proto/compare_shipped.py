"""Is the prototype ensemble actually better than the checkpoint the frontend ships?

The two were trained on *different* splits, so neither side's own val/test
number is comparable:

* shipped `nextpick-proonly-randomvalsplit-*` trained on 42,121 drafts and
  selected on a seeded random 10% of every patch (`val_random=True`,
  `test_frac=0`). It has no held-out test at all.
* the prototype arms trained on 37,501 drafts (80% of the seeded series split)
  and are scored on its 4,670-draft test split.

The only honest common set is rows that NEITHER saw:

    shipped_val  AND  proto_test      (strict: ~470 drafts)
    shipped_val  AND  (proto_val+test) (wide: ~950 drafts, proto arms selected
                                        on proto_val, so slightly kind to them)

`shipped_val` is reconstructed exactly: the split function is deterministic in
(rows, seed, fractions), so if the reconstructed val has the checkpoint's own
recorded `n` states the reconstruction is the same draw.

Usage:
    python pro/model/proto/compare_shipped.py [--wide]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from pro.model.proto import evaluate as E                             # noqa: E402
from pro.model.proto import split as SPLIT                            # noqa: E402

SHIPPED = (HERE.parent / "checkpoints" /
           "nextpick-proonly-randomvalsplit-through-wlds-ig-tes" / "best.pt")
ENSEMBLE = [HERE / "runs" / n / "best.pt"
            for n in ("base64", "combo64", "unfreeze")]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", action="store_true",
                    help="use shipped_val & (proto val+test) instead of the "
                         "strict shipped_val & proto test")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    rows, _ = E.PRO_PIPE.build_rows()
    splits, man = SPLIT.load(rows)
    _, patch_index = E.PRO_DATA.build_patch_vocab(rows)

    # -- reconstruct the shipped checkpoint's val draw -----------------
    # `pinned_test=None`: with test_frac=0 the pin would only *remove* the
    # pinned series from the pool, which shifts the draw. The checkpoint's
    # recorded n=95,160 is exactly the unpinned draw (42,121/4,758/0).
    shipped = E.PRO_PIPE.split_within_patch(
        rows, val_frac=0.10, test_frac=0.0, val_random=True,
        pinned_test=None, seed=E.PRO_PIPE.VAL_SPLIT_SEED)
    blob = torch.load(str(SHIPPED), map_location="cpu", weights_only=False)
    want = blob["metrics"]["n"]
    got = len(shipped["val"]) * E.PRO_DATA.N_MOVES
    print("shipped val reconstructed: %d drafts / %d states (ckpt records %d)"
          % (len(shipped["val"]), got, want))
    if got != want:
        raise SystemExit("reconstruction does not match the checkpoint - the "
                         "split parameters are not the ones that produced it; "
                         "refusing to compare on a guessed set")

    held = set(splits["val"]) | set(splits["test"]) if args.wide \
        else set(splits["test"])
    idxs = sorted(set(shipped["val"]) & held)
    print("honest common set: %d drafts, %d states (proto %s)"
          % (len(idxs), len(idxs) * E.PRO_DATA.N_MOVES,
             "val+test" if args.wide else "test only"))
    if len(idxs) < 200:
        raise SystemExit("common set too small to rank models")

    move_rank = E.EP.move_popularity_ranks(rows, splits["train"],
                                           E.PRO_DATA.NUM_REAL_CHAMPS)
    base_pr = E.block_base_ranks(rows, idxs, move_rank,
                                 E.PRO_DATA.NUM_REAL_CHAMPS)

    def load(path):
        b = torch.load(str(path), map_location="cpu", weights_only=False)
        m, cfg = E.build_model(b, device, rows=rows, patch_index=patch_index,
                               train_idx=splits["train"])
        return m, cfg

    arms = [("shipped", [SHIPPED]),
            ("base64", [HERE / "runs/base64/best.pt"]),
            ("combo64", [HERE / "runs/combo64/best.pt"]),
            ("unfreeze36", [HERE / "runs/unfreeze/best.pt"]),
            ("ensemble3", ENSEMBLE)]
    res, arrays = {}, {}
    for name, paths in arms:
        models, cfg = [], None
        for p in paths:
            m, cfg = load(p)
            models.append(m)
        a = E.run(models, rows, idxs, patch_index, cfg, device,
                  args.batch_size)
        m = E.metrics(a, base_pr)
        print(E.fmt(m, name))
        res[name], arrays[name] = m, a
        del models
        torch.cuda.empty_cache()

    print("\npaired vs shipped (positive = better than shipped):")
    res["paired_vs_shipped"] = {}
    for name in list(arrays)[1:]:
        for k in (1, 5):
            b, c, chi2, p = E.mcnemar(arrays["shipped"]["rank"] < k,
                                      arrays[name]["rank"] < k)
            d, lo, hi = E.game_bootstrap(arrays["shipped"], arrays[name], k=k)
            res["paired_vs_shipped"]["%s_top%d" % (name, k)] = {
                "delta": d, "ci": [lo, hi], "p": p}
            print("  %-11s top%d %+.4f  boot [%+.4f, %+.4f]  p=%.3g"
                  % (name, k, -d, -hi, -lo, p))
        pm = arrays["shipped"]["partner"] == 1
        if pm.any():
            b, c, chi2, p = E.mcnemar(arrays["shipped"]["prank"][pm] < 5,
                                      arrays[name]["prank"][pm] < 5)
            print("  %-11s partner@5  shipped-only %d %s-only %d p=%.3g"
                  % (name, b, name, c, p))
            res["paired_vs_shipped"]["%s_partner5" % name] = {
                "shipped_only": b, "other_only": c, "p": p}

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1), "utf-8")
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
