"""Pro-only masked modelling: no soloQ pretraining, no soloQ rehearsal.

    python model/proonly/train_masked.py --run-name p1-patch --patch-emb
    python model/proonly/train_masked.py --run-name p1-nopatch --no-patch-emb

What this measures, and what it deliberately does not
----------------------------------------------------
Every other stage in this repo starts from a soloQ-pretrained encoder
(`model/pretraining`) and treats pro data as a finetune. This script drops
that: the encoder is randomly initialised and sees pro drafts only, so the
number it reports is the answer to "how far does 37k pro drafts get you on
their own". Nothing here reads a soloQ shard - there is no rehearsal path to
disable, which is the point (a rehearsal term would make the result a
pretraining ablation rather than a pro-only one).

Split: `pipeline.split_within_patch`. Train is the head of every patch, val
and test are its tail, series never straddle. Under the repo's global date
split the test patches are unseen, so a patch id is UNK on every test row and
a patch embedding can only memorise; this split keeps each patch on both sides
so the question becomes whether the head of a patch teaches its tail.

Patch embedding (`--patch-emb`, default on; `--no-patch-emb` is the ablation):
one zero-initialised vector per patch, added to every slot before the
LayerNorm (a segment embedding). At step 0 the two arms are bit-identical, so
the comparison is one flag and one seed, not two networks.

Objective: BERT-style masked champion modelling, verbatim from soloQ
(`apply_masking`, single source of truth) - 15% of slots corrupted, predict
champion (CE over 173) + role (CE, weight 0.3). No win head: this stage is
about the completion objective, and the value path has its own script.

Selection: `rel_champ_top5` on val = top-5 over the per-slot popularity
baseline fit on train. Baseline-relative rather than raw, because the raw
number tracks how easy the window is, and this window is much easier than the
patch-honest one (the same patch is in train).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

PRO_PIPE, PRO_DATA, EP = C.PRO_PIPE, C.PRO_DATA, C.EP
IGNORE_INDEX = -100


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pro-only masked modelling")
    C.add_corpus_args(ap)
    # model
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-intra", type=int, default=2)
    ap.add_argument("--num-cross", type=int, default=1)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="0.1, not the soloQ 0.0: 37k pro drafts against a "
                         "~2.5M-parameter encoder is the regime where the "
                         "soloQ runs were underfitting and this one overfits")
    ap.add_argument("--patch-emb", dest="patch_emb", action="store_true",
                    default=True,
                    help="add the patch embedding (default; --no-patch-emb is "
                         "the ablation)")
    ap.add_argument("--no-patch-emb", dest="patch_emb", action="store_false")
    # objective
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--role-keep-prob", type=float, default=0.1)
    ap.add_argument("--lambda-role", type=float, default=0.3)
    # optimisation
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lol-draft-proonly")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out-dir", default="",
                    help="write the run here instead of ckpt/run_<ts>_<name>. "
                         "Used to give the kept artefacts a stable, "
                         "self-describing name (model/checkpoints/masking-proonly).")
    return ap.parse_args(argv)


def masked_loss(out, batch, lambda_role):
    m = batch["is_masked"]
    lc = F.cross_entropy(out["champ_logits"][m], batch["champ_labels"][m])
    rm = batch["role_labels"] != IGNORE_INDEX
    lr = F.cross_entropy(out["role_logits"][rm], batch["role_labels"][rm])
    return lc + lambda_role * lr, lc, lr


@torch.no_grad()
def evaluate(model, loader, slot_rank, args, device, slices=None):
    """Masked top-k on the given split, with the slot baseline on the same rows.

    `slot_rank` (EP.slot_popularity_ranks fit on TRAIN) is the strong null: in
    a pro draft slot -> role is fixed, so the table already knows "slot 0 is a
    top laner". A global pick list would be a straw man here.
    """
    model.eval()
    n = 0
    hits = {1: 0.0, 5: 0.0, 10: 0.0}
    base_hits = {1: 0.0, 5: 0.0}
    cond5 = cond_n = 0
    role_hid1 = role_hid_n = 0
    role_champ1 = role_champ_n = 0
    loss_sum = 0.0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        loss, _, _ = masked_loss(out, batch, args.lambda_role)
        m = batch["is_masked"]
        cnt = int(m.sum())
        if cnt == 0:
            continue
        loss_sum += float(loss) * cnt
        cl, ct = out["champ_logits"][m], batch["champ_labels"][m]
        truth = cl.gather(1, ct[:, None]).squeeze(1)
        ranks = (cl > truth[:, None]).sum(1)
        for k in hits:
            hits[k] += float((ranks < k).sum())
        di, si = m.nonzero(as_tuple=True)
        base_ranks = [slot_rank[s][c] for s, c in zip(si.tolist(),
                                                      ct.tolist())]
        for k in base_hits:
            base_hits[k] += sum(1 for r in base_ranks if r < k)
        vis = ~batch["role_masked"][m]
        cond_n += int(vis.sum())
        if bool(vis.any()):
            cond5 += int((ranks[vis] < 5).sum())
        hid = batch["role_masked"][m]
        if bool(hid.any()):
            role_hid1 += int((out["role_logits"][m].argmax(1)[hid]
                              == batch["role_labels"][m][hid]).sum())
            role_hid_n += int(hid.sum())
        ro = batch["role_masked"] & (~m) & (batch["role_labels"] != IGNORE_INDEX)
        if bool(ro.any()):
            role_champ1 += int((out["role_logits"][ro].argmax(1)
                                == batch["role_labels"][ro]).sum())
            role_champ_n += int(ro.sum())
        if slices is not None:
            metas = batch.get("meta") or [{}] * int(m.shape[0])
            patches = [metas[int(d)].get("patch") for d in di.tolist()]
            slices.add(ranks.tolist(), patches, base_ranks)
        n += cnt
    out_m = {"mask_loss": loss_sum / max(1, n), "n_masked": n}
    for k in hits:
        out_m["champ_top%d" % k] = hits[k] / max(1, n)
    for k in base_hits:
        out_m["base_champ_top%d" % k] = base_hits[k] / max(1, n)
    out_m["rel_champ_top5"] = EP.baseline_relative(
        out_m["champ_top5"], out_m["base_champ_top5"])
    out_m["role_cond_top5"] = cond5 / max(1, cond_n)
    out_m["role_hidden_top1"] = role_hid1 / max(1, role_hid_n)
    out_m["role_given_champ_top1"] = role_champ1 / max(1, role_champ_n)
    return out_m


def run_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb
        key = C.MODEL_DIR / "pretraining" / "wandb_key.txt"
        if "WANDB_API_KEY" not in os.environ and key.exists():
            os.environ["WANDB_API_KEY"] = key.read_text().strip()
        return wandb.init(project=args.wandb_project,
                          name=args.run_name or None, config=vars(args),
                          dir=str(HERE), settings=wandb.Settings(init_timeout=60))
    except Exception as exc:  # noqa: BLE001 - logging never kills training
        print("wandb disabled: %s" % exc)
        return None


def save_ckpt(path, model, args, cfg, metrics, step, epoch, patch_vocab,
              resume=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(),
               "config": cfg, "metrics": metrics, "step": step, "epoch": epoch,
               "patch_vocab": patch_vocab, "corpus": args.corpus,
               "patch_emb": bool(args.patch_emb)}
    if resume is not None:
        payload["resume"] = resume
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path))


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    rows, splits, patch_vocab, patch_index = C.load_split(args)
    num_champs = 173
    n_patches = (len(patch_vocab) + 1) if args.patch_emb else 0

    def loader(idxs, mask_prob, shuffle, seed, patch_index=patch_index):
        ds = PRO_DATA.ProMaskedDraftDataset(
            rows, idxs, num_champs=num_champs, mask_prob=mask_prob,
            role_keep_prob=args.role_keep_prob, seed=seed, epoch=0,
            econ_mask_prob=0.0, patch_index=patch_index)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=PRO_DATA.pro_collate,
                          num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available())

    tr_loader = loader(splits["train"], args.mask_prob, True, args.seed)
    # Eval views are fixed (EVAL_SEED), not seed-dependent: the patch arm and
    # the seed arms must all be scored on the same masked slots.
    va_loader = loader(splits["val"], args.mask_prob, False, C.EVAL_SEED)
    te_loader = loader(splits["test"], args.mask_prob, False, C.EVAL_SEED)

    model = C.build_encoder(num_champs, args.embed_dim, args.num_intra,
                            args.num_cross, args.num_heads, args.dropout,
                            n_patches).to(device)
    cfg = {"num_champs": num_champs, "embed_dim": args.embed_dim,
           "num_intra": args.num_intra, "num_cross": args.num_cross,
           "num_heads": args.num_heads, "dropout": args.dropout,
           "with_value": False, "attr_dim": 0, "n_patches": n_patches}
    print("patch embedding: %s (%d rows incl. UNK)"
          % ("ON" if n_patches else "OFF", n_patches))
    print("model: %.2fM params, device %s, amp %s"
          % (C.param_count(model) / 1e6, device, amp))

    slot_rank = EP.slot_popularity_ranks(rows, splits["train"], num_champs)
    steps_per_epoch = math.ceil(len(tr_loader))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    warm = max(1, int(total_steps * args.warmup_frac))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run = run_wandb(args)
    run_dir = Path(args.out_dir) if args.out_dir else \
        HERE / "ckpt" / ("run_" + time.strftime("%Y%m%d_%H%M%S") +
                         (("_" + args.run_name) if args.run_name else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    print("checkpoints ->", run_dir)

    best = -1.0
    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = 0.0
        nb = 0
        for batch in tr_loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v)
                     else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(batch)
                loss, lc, lr = masked_loss(out, batch, args.lambda_role)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss at step %d" % global_step)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            agg += float(loss)
            nb += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            va = evaluate(model, va_loader, slot_rank, args, device)
            msg = ("ep %d step %d train %.4f | val top1 %.4f top5 %.4f "
                   "base5 %.4f rel %.3fx rolecond5 %.4f (%.0fs)"
                   % (epoch, global_step, agg / max(1, nb), va["champ_top1"],
                      va["champ_top5"], va["base_champ_top5"],
                      va["rel_champ_top5"], va["role_cond_top5"],
                      time.time() - t0))
            print(msg)
            if run is not None:
                import wandb
                wandb.log({"train/loss": agg / max(1, nb),
                           "train/lr": sched.get_last_lr()[0],
                           **{"val/" + k: v for k, v in va.items()}},
                          step=global_step)
            if va["rel_champ_top5"] > best:
                best = va["rel_champ_top5"]
                save_ckpt(run_dir / "best.pt", model, args, cfg,
                          dict(va, epoch=epoch), global_step, epoch,
                          patch_vocab)
        save_ckpt(run_dir / "latest.pt", model, args, cfg,
                  {"epoch": epoch}, global_step, epoch, patch_vocab,
                  resume={"opt": opt.state_dict(), "sched": sched.state_dict(),
                          "step": global_step, "epoch": epoch,
                          "best": best, "run_dir": str(run_dir)})
        if args.max_steps and global_step >= args.max_steps:
            break

    # ---- test report on the selected checkpoint
    report = {"protocol": {
        "corpus": args.corpus, "patch_emb": bool(args.patch_emb),
        "n_patches": n_patches,
        "split": "within-patch (train=head, val=tail); "
                 "test = the pinned TEST_SERIES",
        "val_frac": args.val_frac,
        "test_frac": args.test_frac, "seed": args.seed,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "selection": "rel_champ_top5 (val)",
        "model": {k: cfg[k] for k in ("embed_dim", "num_intra", "num_cross",
                                      "num_heads", "dropout")},
        "params": C.param_count(model)}}
    blob = torch.load(str(run_dir / "best.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(blob["model"])
    report["val"] = blob["metrics"]
    slices = C.SliceAccumulator()
    te = evaluate(model, te_loader, slot_rank, args, device, slices=slices)
    report["test"] = te
    report["test"]["slices"] = slices.report()
    print("TEST top1 %.4f top5 %.4f (base %.4f, rel %.3fx) rolecond5 %.4f"
          % (te["champ_top1"], te["champ_top5"], te["base_champ_top5"],
             te["rel_champ_top5"], te["role_cond_top5"]))
    years = report["test"]["slices"]["by_year"]
    for y in sorted(years):
        r = years[y]
        print("  %s: n=%d top5 %.4f base %.4f rel %.2fx"
              % (y, r["n"], r["top5"], r["base_top5"], r.get("rel_top5", 0)))
    with open(run_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("done (%s)" % run_dir)
    if run is not None:
        import wandb
        wandb.finish()
    return report


def main():
    report = train(parse_args())
    out = HERE / "last_masked.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"protocol": report["protocol"], "val": report["val"],
                   "test": {k: v for k, v in report["test"].items()
                            if k != "slices"}}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
