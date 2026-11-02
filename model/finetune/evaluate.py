"""Composition evaluation for a masked/finetuned checkpoint.

This is the metric set for the *pretrained* objective. A masked model was never
trained to predict the next move, so scoring it on next-move accuracy measures
a task it does not do. What it does do is model which 10 champions form a
plausible draft, and that is what this script measures:

  * leave-one-out masking - hide exactly one slot, predict it. Every draft
    contributes 10 predictions instead of the ~1.5 a 15%-mask sampler yields,
    and the comparison across slots is balanced by construction.
  * top-1 / top-5 / top-10 accuracy on the hidden champion.
  * bits per slot - mean negative log-likelihood of the true champion under
    the model, in bits. This is the composition density: unlike accuracy it
    uses the whole distribution, so a model that ranks the truth 2nd is not
    scored the same as one that ranks it 200th.
  * popularity baseline - the most-picked champion *in that slot* over the
    training era. Per slot, not global: slot 0 is always the blue top laner,
    so the null already knows the role, which is the fair comparison for a
    model that is shown the role too. A composition model has to beat that or
    it is just a patch-frequency table.

Slices: patch, league tier, slot index. Val is patch-honest (patches absent
from train) because test is 100% unseen patches.

Usage:
    python evaluate.py --ckpt ckpt/run_x/best_mask.pt
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent            # .../model/finetune
MODEL_DIR = HERE.parent                           # .../model
ROOT = MODEL_DIR.parent                           # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"

sys.path.insert(0, str(ROOT))
from model.pretraining.soloq_model import SoloQPretrainedModel      # noqa: E402
from model.pretraining.soloq_model import attr_matrix_for_config    # noqa: E402


def _load_module_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load_module_by_path("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load_module_by_path("pro_datasets", PRO_DIR / "datasets.py")
EP = _load_module_by_path("eval_protocol", PRO_DIR / "eval_protocol.py")

N_SLOTS = 10
IGNORE_INDEX = -100


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Composition (masked) evaluation")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--focus-patch", default="16.20")
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="")
    return ap.parse_args(argv)


class LeaveOneOut(Dataset):
    """One example per (draft, slot): that slot's champion is hidden.

    Role stays visible on purpose - the question is "which champion completes
    this comp", not "which role is missing", and the pretrained model was
    trained with role-visible masks (the 90/10 split) so this is in
    distribution for it.
    """

    def __init__(self, rows, indices):
        self.rows = [rows[i] for i in indices]
        self.indices = list(indices)

    def __len__(self):
        return len(self.rows) * N_SLOTS

    def __getitem__(self, k):
        row = self.rows[k // N_SLOTS]
        slot = k % N_SLOTS
        champ = torch.as_tensor(row["champ"], dtype=torch.long)
        role = torch.as_tensor(row["role"], dtype=torch.long)

        cids = champ.clone()
        cids[slot] = PRO_DATA.CHAMP_MASK_ID
        is_m = torch.zeros(N_SLOTS, dtype=torch.bool)
        is_m[slot] = True
        clab = torch.full((N_SLOTS,), IGNORE_INDEX, dtype=torch.long)
        clab[slot] = champ[slot]
        return {
            "champ_ids": cids,
            "role_ids": role.clone(),          # role visible
            "role_masked": torch.zeros(N_SLOTS, dtype=torch.bool),
            "is_masked": is_m,
            "champ_labels": clab,
            "role_labels": torch.full((N_SLOTS,), IGNORE_INDEX,
                                      dtype=torch.long),
            "pad_mask": torch.zeros(N_SLOTS, dtype=torch.bool),
            "win_label": torch.tensor(row["blue_win"], dtype=torch.long),
            "econ_target": torch.tensor(
                PRO_DATA.econ_targets(row), dtype=torch.float32),
            "has_econ": torch.tensor(1.0 if row.get("has_econ") else 0.0,
                                     dtype=torch.float32),
            "weight": torch.tensor(float(row.get("weight", 1.0)),
                                   dtype=torch.float32),
            "meta": {"gameid": row["gameid"], "series_id": row["series_id"],
                     "patch": row["patch"], "league": row.get("league")},
        }


def build_model(blob, device):
    cfg = blob["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        attr_matrix=attr_matrix_for_config(cfg))
    from model import ProDraftModel
    model = ProDraftModel(enc, n_econ=cfg.get("n_econ", 0),
                          head_dropout=0.0).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def score(model, loader, slot_rank, device):
    rep = EP.RankReport(ks=(1, 5, 10))
    base = EP.RankReport(ks=(1, 5, 10))
    nll_sum, nll_n = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        cl = out["champ_logits"]                     # [B, C]
        m = batch["is_masked"]
        logits = cl[m]
        truth = batch["champ_labels"][m]
        ranks = (logits > logits.gather(1, truth[:, None])).sum(1)

        # one masked slot per example, so the slot index is recoverable
        slots = m.float().argmax(dim=1)              # [B] -> slot per example
        has = m.any(dim=1)
        if not bool(has.all()):
            # defensive: the leave-one-out set always has one mask, but a
            # future caller might not, and a silent length mismatch would
            # misalign every slice below
            logits, truth, ranks = logits[has], truth[has], ranks[has]
        slot_list = slots[has].tolist()
        patches = [mm["patch"] for mm, k in zip(batch["meta"], has.tolist())
                   if k]
        leagues = [mm["league"] for mm, k in zip(batch["meta"], has.tolist())
                   if k]
        kinds = ["role_visible"] * len(slot_list)
        rep.add(ranks.tolist(), patches, leagues, slot_list, kinds)
        # per-slot baseline: the null knows which role that slot is
        base.add([slot_rank[s][c] for s, c in zip(slot_list, truth.tolist())],
                 patches, leagues, slot_list, kinds)

        nll_sum += float(F.cross_entropy(logits, truth, reduction="sum"))
        nll_n += truth.numel()
    nll = {"n": nll_n}
    if nll_n:
        nats = nll_sum / nll_n
        nll["nats"] = nats
        nll["bits"] = nats / math.log(2.0)
    return rep, base, nll


def emit(name, rep, base, nll, rows, idxs, focus, min_n, num_champs, out):
    print("\n" + "=" * 68)
    print("%s: %d predictions over %d drafts" % (name, len(rep), len(idxs)))
    m = rep.metrics()
    bm = base.metrics()
    print("  pooled  top1 %.4f  top5 %.4f [%.4f, %.4f]  top10 %.4f"
          "  base@5 %.4f  rel %.2fx"
          % (m["top1"], m["top5"], m["ci5"][0], m["ci5"][1], m["top10"],
             bm["top5"], EP.baseline_relative(m["top5"], bm["top5"])))
    if nll:
        print("  density %.4f bits/slot (%.4f nats, n=%d)"
              % (nll["bits"], nll["nats"], nll["n"]))
    # The bar that actually matters on a new patch: a frequency table built on
    # that patch's own games, leave-one-out. The train-era table is stale
    # there, so `rel` against it flatters the model.
    self_ref = None
    if focus:
        focus_idx = [i for i in idxs if rows[i].get("patch") == focus]
        if focus_idx:
            self_ref = EP.self_slot_topk(rows, focus_idx, num_champs, k=(1, 5))
    report = EP.stratified(rep, base_rep=base, focus=focus, self_ref=self_ref,
                           min_n=min_n, label=name)
    print(EP.format_stratified(report, indent=2))

    out.setdefault(name, {})["pooled"] = m
    out[name]["baseline"] = bm
    out[name]["density"] = nll
    out[name]["stratified"] = report
    return out


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    blob = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    model, cfg = build_model(blob, device)
    print("checkpoint: %s (step %s)" % (args.ckpt, blob.get("step")))
    print("config: embed %d, econ %d" % (cfg["embed_dim"],
                                         cfg.get("n_econ", 0)))

    rows, _ = PRO_PIPE.build_rows(limit=args.limit_drafts)
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_series_by_date(rows)
    honest, hinfo = EP.patch_honest_val(rows, splits)
    print("splits: train %d / val %d / test %d (patch-honest val %d)"
          % (len(splits["train"]), len(splits["val"]), len(splits["test"]),
             len(honest)))

    slot_rank = EP.slot_popularity_ranks(rows, splits["train"],
                                         cfg["num_champs"])
    out = {}
    for name, idxs in (("val_raw", splits["val"]), ("val_honest", honest),
                       ("test", splits["test"])):
        ds = LeaveOneOut(rows, idxs)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=PRO_DATA.pro_collate)
        rep, base, nll = score(model, loader, slot_rank, device)
        emit(name, rep, base, nll, rows, idxs, args.focus_patch,
             args.min_n, cfg["num_champs"], out)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"ckpt": str(args.ckpt), "step": blob.get("step"),
                       "config": cfg, "patch_honest": hinfo,
                       "reports": out}, f, indent=1)
        print("\nwrote", args.json)
    return out


if __name__ == "__main__":
    main()
