"""Does the model actually use the bans, or just know they exist?

    python pro/model/bans/probe.py --ckpt pro/model/checkpoints/<run>/best.pt

A better top-5 does not prove ban-awareness, and neither does a lower loss: the
legality mask alone guarantees that no banned champion is ever suggested, which
makes a ban-blind model look correct on every ban-related metric the repo has
had so far. This probe separates the two channels.

The ablation keeps legality fixed and removes only the *information*: the ban
content (`ban_ids`) is replaced with PAD while `taken_mask` still contains the
same champions, so the model still cannot pick them - it simply no longer knows
which ones they are. Same for the fearless pool. Anything that changes is the
model using bans as knowledge rather than as a constraint.

Three numbers, each answering a different question:

`blindfold` - top-1 / top-5 on pick moves with the ban content hidden. A model
    that leans on bans loses accuracy; one that ignores them does not. Reported
    against the same model's full-input score, so it is paired.

`top1_changed` - how often hiding ONE ban flips the argmax. This is the
    counterfactual the branch is about: "if that champion were not gone, would
    you pick differently?" A model whose ban influence is a constant offset
    across candidates almost never flips, because the offset cancels in the
    argmax; a model with a per-candidate term flips often.

`spread` - std/mean of the logit change across the still-legal candidates, when
    one ban is removed. Near 0 means the ban moved every candidate by the same
    amount, i.e. it carried no *candidate-specific* information at all - which
    is the precise failure mode the output-side counter term exists to fix.
    Compare `spread` between the ban-blind and the ban-aware checkpoint; the
    absolute value is not meaningful on its own.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR / "proonly"))
sys.path.insert(0, str(ROOT))

import common as C  # noqa: E402

PRO_DATA, EP = C.PRO_DATA, C.EP
POLICY_MODEL = C.POLICY_MODEL


def load_model(ckpt, device):
    """Rebuild exactly the module set the checkpoint was trained with."""
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    enc = C.build_encoder(cfg["num_champs"], cfg["embed_dim"],
                          cfg["num_intra"], cfg["num_cross"],
                          cfg["num_heads"], cfg.get("dropout", 0.0),
                          cfg.get("n_patches", 0))
    model = C.PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg.get("state_dim", 64),
        n_econ=0, dropout=cfg.get("dropout", 0.0),
        n_patches=cfg.get("n_patches", 0), with_value=False,
        ban_side=bool(cfg.get("ban_side", False)),
        ban_fearless=bool(cfg.get("ban_fearless", False)),
        counter_rank=int(cfg.get("counter_rank", 0) or 0))
    own = set(model.state_dict())
    keep = {k: v for k, v in payload["model"].items() if k in own}
    missing = [k for k in own - set(keep)
               if k.startswith(("encoder.", "policy_head.", "state_embed.",
                                "ban_context.", "counter."))]
    if missing:
        raise SystemExit("%s lacks policy weights: %s" % (ckpt, missing[:6]))
    model.load_state_dict(keep, strict=False)
    return model.to(device).eval(), cfg, payload


def _to(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _ranks(logits, target):
    truth = logits.gather(1, target[:, None]).squeeze(1)
    return (logits > truth[:, None]).sum(1)


def _expand(batch, rep):
    """Repeat every row `rep` times along the batch axis -> [B*rep, ...]."""
    n = batch["ban_ids"].shape[0]
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
            out[k] = v.repeat_interleave(rep, dim=0)
        else:
            out[k] = v
    return out


@torch.no_grad()
def probe(model, loader, device, max_states=0, want_ablation=True):
    """Blindfold + single-ban counterfactuals on pick moves."""
    n_full = n_pick = 0
    hits_full = {1: 0, 5: 0}
    hits_blind = {1: 0, 5: 0}
    hits_poolblind = {1: 0, 5: 0}
    by_bans = {}
    by_pool = {}
    flips = 0
    flip_n = 0
    spreads = []
    seen = 0
    # DUPLICATE_LOGIT_NEG is -1e4: a candidate that is illegal in either
    # forward pass must not enter the counterfactual statistics, or its
    # "unchanged" -1e4 would look like a perfectly stable logit.
    ILLEGAL = -1e3

    for batch in loader:
        batch = _to(batch, device)
        B = batch["ban_ids"].shape[0]
        logits = model(batch)["policy_logits"]
        target = batch["target_champ"]
        is_pick = batch["is_ban"] == 0

        ranks = _ranks(logits, target)
        for k in (1, 5):
            hits_full[k] += int((ranks[is_pick] < k).sum())
        n_pick += int(is_pick.sum())
        n_full += B

        # --- blindfold: hide the ban content, keep the legality mask
        blind = dict(batch)
        blind["ban_ids"] = torch.full_like(batch["ban_ids"],
                                           POLICY_MODEL.CHAMP_PAD_ID)
        lb = model(blind)["policy_logits"]
        rb = _ranks(lb, target)
        for k in (1, 5):
            hits_blind[k] += int((rb[is_pick] < k).sum())

        nb = (batch["ban_ids"] != POLICY_MODEL.CHAMP_PAD_ID).sum(1)
        for b_i in range(B):
            if not bool(is_pick[b_i]):
                continue
            e = by_bans.setdefault(int(nb[b_i]), {"n": 0, "h5": 0, "b5": 0})
            e["n"] += 1
            e["h5"] += int(ranks[b_i] < 5)
            e["b5"] += int(rb[b_i] < 5)

        # --- split by whether a fearless pool exists at all. This is the
        # mechanism check: a gain that lives only on pooled rows is the
        # fearless channel working; a gain spread evenly is something else
        # (a wider ban path) and should be reported as such.
        if "fearless_ids" in batch:
            has_pool = (batch["fearless_ids"]
                        != POLICY_MODEL.CHAMP_PAD_ID).any(1)
            for key, sel in (("pool", has_pool & is_pick),
                             ("no_pool", ~has_pool & is_pick)):
                if bool(sel.any()):
                    e = by_pool.setdefault(key, {"n": 0, "h1": 0, "h5": 0,
                                                 "b5": 0})
                    e["n"] += int(sel.sum())
                    e["h1"] += int((ranks[sel] < 1).sum())
                    e["h5"] += int((ranks[sel] < 5).sum())
                    e["b5"] += int((rb[sel] < 5).sum())

        # --- hide the fearless pool, keep legality
        if "fearless_ids" in batch:
            pb = dict(batch)
            pb["fearless_ids"] = torch.full_like(batch["fearless_ids"],
                                                 POLICY_MODEL.CHAMP_PAD_ID)
            rp = _ranks(model(pb)["policy_logits"], target)
            for k in (1, 5):
                hits_poolblind[k] += int((rp[is_pick] < k).sum())

        # --- one-ban counterfactuals on pick moves only
        if want_ablation:
            idx = is_pick.nonzero(as_tuple=True)[0]
            if idx.numel():
                sub = {k: (v[idx] if torch.is_tensor(v) and v.dim() >= 1
                           and v.shape[0] == B else v) for k, v in batch.items()}
                reps = int(nb[idx].max().item())
                if reps > 0:
                    exp = _expand(sub, reps)
                    # copy r removes ban slot r; slots past a state's own
                    # ban count are PAD and drop out below
                    slot = torch.arange(reps, device=device).repeat(
                        idx.numel())
                    rows = torch.arange(exp["ban_ids"].shape[0], device=device)
                    row_ok = exp["ban_ids"][rows, slot] != \
                        POLICY_MODEL.CHAMP_PAD_ID
                    b_ids = exp["ban_ids"].clone()
                    b_ids[rows, slot] = POLICY_MODEL.CHAMP_PAD_ID
                    exp["ban_ids"] = b_ids
                    lf = model(sub)["policy_logits"].repeat_interleave(
                        reps, dim=0)
                    la = model(exp)["policy_logits"]
                    legal = (lf > ILLEGAL) & (la > ILLEGAL)   # [N,C]
                    sel = row_ok & legal.any(1)
                    if sel.any():
                        d = torch.where(legal, lf - la,
                                        torch.zeros_like(lf))
                        flips += int((lf[sel].argmax(1)
                                      != la[sel].argmax(1)).sum())
                        flip_n += int(sel.sum())
                        m = legal.float()
                        n_leg = m.sum(1).clamp_min(1.0)
                        mu = (d * m).sum(1) / n_leg
                        sd = ((((d - mu[:, None]) ** 2) * m).sum(1)
                              / n_leg).clamp_min(0).sqrt()
                        mean_abs = (d * m).abs().sum(1) / n_leg
                        row_spread = sd / (mean_abs + 1e-9)
                        spreads.extend(row_spread[sel].tolist())
        seen += B
        if max_states and seen >= max_states:
            break

    out = {
        "n_states": n_full,
        "n_pick_states": n_pick,
        "full": {("top%d" % k): hits_full[k] / max(1, n_pick)
                 for k in (1, 5)},
        "blindfold": {("top%d" % k): hits_blind[k] / max(1, n_pick)
                      for k in (1, 5)},
        "pool_blindfold": {("top%d" % k): hits_poolblind[k] / max(1, n_pick)
                           for k in (1, 5)},
        "top1_changed": flips / max(1, flip_n),
        "n_counterfactuals": flip_n,
        "spread": (sum(spreads) / len(spreads)) if spreads else None,
        "by_n_bans": {},
        "by_pool": {},
    }
    for k in sorted(by_bans):
        e = by_bans[k]
        out["by_n_bans"][str(k)] = {
            "n": e["n"], "top5": e["h5"] / e["n"],
            "blind_top5": e["b5"] / e["n"],
            "delta": (e["h5"] - e["b5"]) / e["n"]}
    for k in sorted(by_pool):
        e = by_pool[k]
        out["by_pool"][k] = {"n": e["n"], "top1": e["h1"] / e["n"],
                             "top5": e["h5"] / e["n"],
                             "blind_top5": e["b5"] / e["n"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", action="append", required=True,
                    help="repeatable; each is probed on the same rows")
    ap.add_argument("--corpus", choices=sorted(C.CORPORA), default="2022")
    ap.add_argument("--split", default="test",
                    choices=("val", "test", "latest-val"))
    ap.add_argument("--val-split", choices=("tail", "random"), default="random",
                    help="must match the split the checkpoints trained with")
    ap.add_argument("--val-scope", choices=("all", "latest"), default="all")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--min-patch-rows", type=int, default=40)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--half-life", type=float, default=0.0)
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--no-ablation", dest="ablation", action="store_false",
                    default=True,
                    help="skip the one-ban counterfactuals (they are ~10x the "
                         "cost); keeps the blindfold and the pool slice")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    device = torch.device(args.device)

    rows, splits, _vocab, patch_index = C.load_split(args, verbose=False)
    fearless_roles = PRO_DATA.build_fearless_roles(rows)
    ds = PRO_DATA.ProNextMoveDataset(
        rows, splits[args.split], num_champs=173, patch_index=patch_index,
        prefix_mode="all", swap_aug=False, fearless_roles=fearless_roles)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=PRO_DATA.pro_collate)
    print("%s split: %d states (%d games)"
          % (args.split, len(ds), len(splits[args.split])))

    results = {}
    for ckpt in args.ckpt:
        model, cfg, _ = load_model(ckpt, device)
        r = probe(model, loader, device, max_states=args.max_states,
                  want_ablation=args.ablation)
        r["config"] = {k: cfg.get(k) for k in
                       ("ban_side", "ban_fearless", "counter_rank")}
        results[ckpt] = r
        print("\n%s" % ckpt)
        print("  ban-awareness: side %s, fearless %s, counter rank %s"
              % (r["config"]["ban_side"], r["config"]["ban_fearless"],
                 r["config"]["counter_rank"]))
        print("  pick top1 %.4f  top5 %.4f   (n=%d)"
              % (r["full"]["top1"], r["full"]["top5"], r["n_pick_states"]))
        print("  blindfold     top1 %.4f  top5 %.4f   d_top5 %+.4f"
              % (r["blindfold"]["top1"], r["blindfold"]["top5"],
                 r["blindfold"]["top5"] - r["full"]["top5"]))
        print("  pool blindfold      top5 %.4f   d_top5 %+.4f"
              % (r["pool_blindfold"]["top5"],
                 r["pool_blindfold"]["top5"] - r["full"]["top5"]))
        print("  one-ban counterfactuals: n=%d  top1 changed %.4f  spread %.4f"
              % (r["n_counterfactuals"], r["top1_changed"],
                 r["spread"] if r["spread"] is not None else float("nan")))
        if r["by_pool"]:
            print("  top5 by fearless pool present:")
            for k, e in r["by_pool"].items():
                print("      %-8s n=%-6d top1 %.4f  top5 %.4f  blind %.4f"
                      % (k, e["n"], e["top1"], e["top5"], e["blind_top5"]))
        print("  top5 by bans visible:  n_bans   full   blind   delta")
        for k, e in r["by_n_bans"].items():
            print("                         %-6s %.4f  %.4f  %+.4f  (n=%d)"
                  % (k, e["top5"], e["blind_top5"], e["delta"], e["n"]))
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print("\nwrote", p)


if __name__ == "__main__":
    main()
