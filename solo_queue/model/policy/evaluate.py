"""Honest evaluation for a policy checkpoint (next-move metrics).

What this replaces. `train.py` reported one pooled val number and one pooled
test number and the drop between them was read as model degradation. Measured
here, a model-free popularity baseline drops from 0.0879 (val) to 0.0707
(test) on the same rows, so most of that "drop" is the window getting harder.
This script reports, for every slice:

    model@k, baseline@k, relative = model / baseline, n, 95% CI

and it scores two val variants side by side: the raw val window and the
patch-honest one (patches absent from train), because test is 100% unseen
patches while raw val is only 81.5%.

Next-move metrics only. The masked/pretrained model is scored by
`solo_queue/model/finetune/evaluate.py` instead, because predicting the *next* move is
not what it was trained to do.

Usage:
    python evaluate.py --ckpt ckpt/run_x/best_policy.pt
    python evaluate.py --ckpt ... --focus-patch 16.20 --json out.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent            # .../solo_queue/model/policy
MODEL_DIR = HERE.parent                           # .../model
ROOT = MODEL_DIR.parents[1]                           # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"

sys.path.insert(0, str(ROOT))
from solo_queue.model.pretraining.base_model import SiameseBase      # noqa: E402
from solo_queue.model.pretraining.base_model import attr_matrix_for_config    # noqa: E402
from solo_queue.model.pretraining.base_model import load_encoder_state        # noqa: E402
from solo_queue.model import PolicyValueModel                # noqa: E402


def _load_module_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRO_PIPE = _load_module_by_path("pro_pipeline", PRO_DIR / "pipeline.py")
PRO_DATA = _load_module_by_path("pro_datasets", PRO_DIR / "datasets.py")
EP = _load_module_by_path("eval_protocol", PRO_DIR / "eval_protocol.py")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Honest policy evaluation")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--focus-patch", default="16.20",
                    help="patch to call out explicitly (the current one)")
    ap.add_argument("--half-life", type=float, default=75.0,
                    help="recency half-life; must match training so the "
                         "baseline uses the same weights")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=200,
                    help="hide slices with fewer examples than this")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="")
    return ap.parse_args(argv)


def build_model(blob, device):
    cfg = blob["config"]
    enc = SiameseBase(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        attr_matrix=attr_matrix_for_config(cfg))
    model = PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg["state_dim"],
        n_econ=cfg["n_econ"], dropout=cfg.get("head_dropout", 0.0),
        n_patches=cfg.get("n_patches", 0),
        # Older checkpoints always have a value head; new runs default it
        # off (--value-weight 0), so the flag has to come from the config
        # or load_state_dict raises on missing/unexpected value_head.*.
        with_value=cfg.get("with_value", True)).to(device)
    # The policy checkpoint holds the whole PolicyValueModel (ban_context,
    # state_embed, heads), so load it directly - load_encoder_state would
    # reject those keys, correctly, because they are not encoder keys.
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint does not match the model: missing=%s unexpected=%s"
            % (list(missing)[:6], list(unexpected)[:6]))
    model.eval()
    return model, cfg


@torch.no_grad()
def score(model, loader, move_rank, cfg, device, econ_mean=None,
          econ_std=None):
    """Rank of the true move, plus the baseline rank on the same examples."""
    rep = EP.RankReport(ks=(1, 5, 10))
    base = EP.RankReport(ks=(1, 5, 10))
    win_p, win_y = [], []
    # econ in the trainer's standardized units, so the RMSE and skill are
    # comparable to the numbers printed during training (raw gold/xp values
    # are ~1e4 and depend on patch drift)
    econ_sq = econ_tgt_sum = econ_tgt_sq = 0.0
    econ_n = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        cl, tc = out["policy_logits"], batch["target_champ"]
        truth = cl.gather(1, tc[:, None]).squeeze(1)
        ranks = (cl > truth[:, None]).sum(1)
        n = tc.numel()

        patches = [m["patch"] for m in batch["meta"]]
        leagues = [m["league"] for m in batch["meta"]]
        idxs = batch["move_idx"].tolist()
        kinds = ["ban" if b else "pick" for b in batch["is_ban"].tolist()]
        rep.add(ranks.tolist(), patches, leagues, idxs, kinds)

        tc_l = tc.tolist()
        base_ranks = [move_rank[m][c] for m, c in zip(idxs, tc_l)]
        base.add(base_ranks, patches, leagues, idxs, kinds)

        if "value_logit" in out:
            win_p.extend(torch.sigmoid(out["value_logit"].squeeze(-1))
                         .float().cpu().tolist())
            win_y.extend(batch["win_label"].float().cpu().tolist())

        if cfg["n_econ"] > 0 and "econ_pred" in out:
            keep = batch["has_econ"].bool()
            if bool(keep.any()):
                tgt = batch["econ_target"][:, :cfg["n_econ"]]
                if econ_mean is not None and econ_std is not None:
                    tgt = (tgt - econ_mean) / econ_std
                pred = out["econ_pred"][keep]
                econ_sq += float(((pred - tgt[keep]) ** 2).sum())
                econ_tgt_sum += float(tgt[keep].double().sum())
                econ_tgt_sq += float((tgt[keep].double() ** 2).sum())
                econ_n += int(keep.sum()) * cfg["n_econ"]
    win = {}
    if win_y:
        win["n"] = len(win_y)
        win["auc"] = EP.auc(win_p, win_y)
        win["base_rate"] = sum(win_y) / len(win_y)
        win["brier"] = sum((p - y) ** 2 for p, y in zip(win_p, win_y)) / len(win_y)
        b = win["base_rate"]
        win["brier_const"] = b * (1 - b)
        win["skill"] = (1 - win["brier"] / win["brier_const"]
                        if win["brier_const"] else 0.0)
    if econ_n:
        mse = econ_sq / econ_n
        mean = econ_tgt_sum / econ_n
        var = max(1e-12, econ_tgt_sq / econ_n - mean * mean)
        win["econ_rmse"] = mse ** 0.5
        win["econ_rmse_const"] = var ** 0.5
        win["econ_skill"] = 1 - (mse ** 0.5) / (var ** 0.5)
    return rep, base, win


def league_mix(rows, idxs):
    from collections import Counter
    c = Counter(rows[i].get("league") or "?" for i in idxs)
    n = max(1, sum(c.values()))
    return {k: (v, v / n) for k, v in c.most_common()}


def emit(name, rep, base, win, rows, idxs, focus, min_n, num_champs, out):
    print("\n" + "=" * 68)
    print("%s: %d examples, %d drafts" % (name, len(rep), len(idxs)))
    m = rep.metrics()
    bm = base.metrics() if base else {}
    print("  pooled  model@1 %.4f  model@5 %.4f [%.4f, %.4f]  base@5 %.4f"
          "  rel %.2fx"
          % (m["top1"], m["top5"], m["ci5"][0], m["ci5"][1],
             bm.get("top5", float("nan")),
             EP.baseline_relative(m["top5"], bm.get("top5", 0.0))))
    if win:
        print("  value   auc %.4f  brier %.5f (const %.5f) skill %+.4f  n %d"
              % (win.get("auc", float("nan")), win.get("brier", float("nan")),
                 win.get("brier_const", float("nan")),
                 win.get("skill", float("nan")), win["n"]))
        if "econ_rmse" in win:
            print("  econ    rmse %.4f (const %.4f) skill %+.4f"
                  % (win["econ_rmse"], win["econ_rmse_const"],
                     win["econ_skill"]))

    rows_tbl = EP.compare(rep, base, "patch", min_n=min_n, order="natural")
    print(EP.format_compare(rows_tbl, title="  by patch", indent=2))
    print(EP.format_compare(
        EP.compare(rep, base, "tier", min_n=min_n, order="n"),
        title="  by league tier", indent=2))
    print(EP.format_compare(
        EP.compare(rep, base, "idx", min_n=min_n),
        title="  by move index", indent=2))
    print(EP.format_compare(
        EP.compare(rep, base, "kind", min_n=min_n, order="n"),
        title="  by ban/pick", indent=2))

    # The bar that matters on a new patch: a per-move frequency table built on
    # that patch's own games, leave-one-out. The train-era table is stale
    # there, so `rel` against it flatters the model.
    self_ref = None
    if focus:
        focus_idx = [i for i in idxs if rows[i].get("patch") == focus]
        if focus_idx:
            self_ref = EP.self_move_topk(rows, focus_idx, num_champs, k=(1, 5))
    report = EP.stratified(rep, base_rep=base, focus=focus, self_ref=self_ref,
                           min_n=min_n, label=name)
    print(EP.format_stratified(report, indent=2))
    if focus and report.get("focus", {}).get("n"):
        mix = league_mix(rows, idxs)
        print("      league mix of the whole split (top 5): %s"
              % ", ".join("%s %.0f%%" % (k, 100 * v[1])
                          for k, v in list(mix.items())[:5]))

    out.setdefault(name, {})["pooled"] = m
    out[name]["baseline"] = bm
    out[name]["value"] = win
    out[name]["stratified"] = report
    out[name]["league_mix"] = {k: {"n": v[0], "share": v[1]}
                               for k, v in league_mix(rows, idxs).items()}
    return out


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    blob = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    model, cfg = build_model(blob, device)
    print("checkpoint: %s (step %s)" % (args.ckpt, blob.get("step")))
    print("config: embed %d, state %d, econ %d, patches %d"
          % (cfg["embed_dim"], cfg["state_dim"], cfg["n_econ"],
             cfg.get("n_patches", 0)))

    rows, _ = PRO_PIPE.build_rows(limit=args.limit_drafts)
    PRO_PIPE.assign_weights(rows, args.half_life)
    splits = PRO_PIPE.split_series_by_date(rows)
    honest, hinfo = EP.patch_honest_val(rows, splits)
    print("splits: train %d / val %d / test %d (patch-honest val %d)"
          % (len(splits["train"]), len(splits["val"]), len(splits["test"]),
             len(honest)))

    _, patch_index = PRO_DATA.build_patch_vocab(rows)
    num_champs = cfg["num_champs"]
    move_rank = EP.move_popularity_ranks(rows, splits["train"], num_champs)

    def loader(idxs):
        ds = PRO_DATA.ProNextMoveDataset(
            rows, idxs, num_champs=num_champs, patch_index=patch_index,
            prefix_mode="all", swap_aug=False)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=PRO_DATA.pro_collate)

    out = {}
    econ_mean = blob.get("econ_mean")
    econ_std = blob.get("econ_std")
    if econ_mean is not None:
        econ_mean = torch.as_tensor(econ_mean, device=device,
                                    dtype=torch.float32)[:cfg["n_econ"]]
        econ_std = torch.as_tensor(econ_std, device=device,
                                   dtype=torch.float32)[:cfg["n_econ"]]
    for name, idxs in (("val_raw", splits["val"]), ("val_honest", honest),
                       ("test", splits["test"])):
        rep, base, win = score(model, loader(idxs), move_rank, cfg, device,
                               econ_mean, econ_std)
        emit(name, rep, base, win, rows, idxs, args.focus_patch,
             args.min_n, num_champs, out)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"ckpt": str(args.ckpt), "step": blob.get("step"),
                       "config": cfg, "patch_honest": hinfo,
                       "reports": out}, f, indent=1)
        print("\nwrote", args.json)
    return out


if __name__ == "__main__":
    main()
