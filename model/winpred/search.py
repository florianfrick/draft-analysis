"""Val-only hyperparameter search on pro drafts.

Why this exists rather than a shell loop around `train.py`: the search has to be
*cheap* and it has to be *honest*. Cheap, because every pro arm so far peaks in
the first 5-12 epochs of 40 and then overfits (README §5, §7) - so the budget is
20 epochs with `--patience 4`. Honest, because `--no-test` drops the held-out
rows before the loaders are built: a configuration chosen on test is not a
held-out number, and it is very easy to run forty configs and then quote the
best test score.

Nothing in this file prints a test metric. The winning config gets one run
through `train.py` afterwards, on the same split, and that number is the result.

    python model/winpred/search.py --grid size
    python model/winpred/search.py --grid regularise --out reports/search.json

Each grid is a list of (name, extra CLI tokens) over one base recipe, so a
result can be reproduced by pasting the tokens into `train.py`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.winpred.train import parse_args, train    # noqa: E402

BASE = ["--data", "pro", "--loss", "ce", "--seed", "20261105",
        "--epochs", "20", "--patience", "4", "--no-test", "--no-wandb"]

# Grids that are not pro. soloQ is 8x the rows (307 k train), so a config costs
# ~2 min rather than 30 s; the same split seed is used so every arm in a grid is
# comparable to the next, and --no-test keeps the 38,436 held-out rows out of
# the loop exactly as it does for pro.
SOLOQ = ["--data", "soloq", "--loss", "ce", "--seed", "20261105",
         "--epochs", "20", "--patience", "4", "--no-test", "--no-wandb"]
BASE_OVERRIDES = {"soloq_base": SOLOQ, "soloq_champ": SOLOQ}

GRIDS = {
    # Does the model want less capacity? Every arm overfits by epoch ~8, and
    # 37 k rows against 2.6 M parameters is the obvious suspect.
    "size": [
        ("d256", []),
        ("d192", ["--embed-dim", "192"]),
        ("d128", ["--embed-dim", "128"]),
        ("d128-i1", ["--embed-dim", "128", "--num-intra", "1"]),
        ("d96-i1", ["--embed-dim", "96", "--num-intra", "1"]),
        ("d128-h4", ["--embed-dim", "128", "--num-heads", "4"]),
    ],
    # If capacity is the problem, regularisation is the other half of the fix.
    "regularise": [
        ("d128-drop3", ["--embed-dim", "128", "--dropout", "0.3"]),
        ("d128-drop4", ["--embed-dim", "128", "--dropout", "0.4"]),
        ("d128-wd1", ["--embed-dim", "128", "--weight-decay", "0.1"]),
        ("d128-wd1-drop3", ["--embed-dim", "128", "--weight-decay", "0.1",
                            "--dropout", "0.3"]),
        ("d192-drop3-wd1", ["--embed-dim", "192", "--dropout", "0.3",
                            "--weight-decay", "0.1"]),
        ("d256-drop4-wd1", ["--dropout", "0.4", "--weight-decay", "0.1"]),
    ],
    # Optimisation: with early stopping at epoch ~6 the cosine schedule never
    # finishes, so the effective learning rate is whatever the curve had left.
    "optim": [
        ("d128-lr5e4", ["--embed-dim", "128", "--lr", "5e-4"]),
        ("d128-lr2e3", ["--embed-dim", "128", "--lr", "2e-3"]),
        ("d128-bs256", ["--embed-dim", "128", "--batch-size", "256"]),
        ("d128-bs1024", ["--embed-dim", "128", "--batch-size", "1024"]),
        ("d128-nopatch", ["--embed-dim", "128", "--no-patch-emb"]),
        ("d128-pma", ["--embed-dim", "128", "--pool", "pma"]),
    ],
    # The 'size' grid came back monotone in capacity (d256 0.6867 < d192 0.6882
    # < d128 0.6888), so the overfitting is not a parameter-count problem. Push
    # the other way and see where it turns.
    "scale": [
        ("d384", ["--embed-dim", "384"]),
        ("d512", ["--embed-dim", "512"]),
        ("d256-i3", ["--num-intra", "3"]),
        ("d384-i3", ["--embed-dim", "384", "--num-intra", "3"]),
        ("d512-drop3", ["--embed-dim", "512", "--dropout", "0.3"]),
        ("d384-wd1", ["--embed-dim", "384", "--weight-decay", "0.1"]),
    ],
    # The corpus question: 'all' adds 2017-2021 (46,879 -> 81,227 drafts).
    # NOT comparable to the '2022' numbers - different test rows.
    "corpus": [
        ("all-d256", ["--corpus", "all"]),
        ("all-d384", ["--corpus", "all", "--embed-dim", "384"]),
        ("all-d512", ["--corpus", "all", "--embed-dim", "512"]),
        ("all-d256-drop3", ["--corpus", "all", "--dropout", "0.3"]),
    ],
    # Team strength changes the regime: it is worth an order of magnitude more
    # than the draft (README §10), so the encoder no longer has to carry the
    # signal and the optimum capacity may move.
    "team": [
        ("t-d128", ["--embed-dim", "128", "--team-feat"]),
        ("t-d256", ["--team-feat"]),
        ("t-d384", ["--embed-dim", "384", "--team-feat"]),
        ("t-d256-k5", ["--team-feat", "--team-k", "5"]),
        ("t-d256-k20", ["--team-feat", "--team-k", "20"]),
        ("t-d256-drop3", ["--team-feat", "--dropout", "0.3"]),
    ],
    # Team strength is where the signal is, so the *estimator* is now the
    # bottleneck. Bradley-Terry conditions on the opponent (a 60 % team in a
    # strong league is not a 60 % team in a minor league) and the half-life
    # lets a roster's recent games count more than its 2022 record. Base recipe
    # is the best rate arm from the 'team' grid.
    "team_bt": [
        ("bt-k10", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt"]),
        ("bt-k4", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                   "--team-k", "4"]),
        ("bt-k20", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                    "--team-k", "20"]),
        ("bt-hl365", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                      "--team-half-life", "365"]),
        ("bt-hl730", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                      "--team-half-life", "730"]),
        ("rate-hl365", ["--team-feat", "--dropout", "0.3",
                        "--team-half-life", "365"]),
    ],
    # bt-k10 selects at **epoch 1** and then degrades, which is not "capacity"
    # in the usual sense: the encoder is barely trained and the head has
    # already saturated on the team feature. So shrink the head (it is the part
    # that can memorise a 2-D input) and slow the schedule down.
    "team_head": [
        ("h32", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                 "--head-dim", "32"]),
        ("h64", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                 "--head-dim", "64"]),
        ("h128", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                  "--head-dim", "128"]),
        ("h64-lr5e4", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                       "--head-dim", "64", "--lr", "5e-4"]),
        ("lr5e4", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                   "--lr", "5e-4"]),
        ("h64-wd1", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                     "--head-dim", "64", "--weight-decay", "0.1"]),
    ],
    # h32 (0.6407) beat the full-width head (0.6475) by more than Bradley-Terry
    # itself gained, and beat h64 (0.6541) as well - so the head is not "an MLP
    # that wants a certain size", it wants to be *small*, i.e. close to the
    # linear stack `team_check.py` fits. Find where that stops being true.
    "head_small": [
        ("h8", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                "--head-dim", "8"]),
        ("h16", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                 "--head-dim", "16"]),
        ("h24", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                 "--head-dim", "24"]),
        ("h32-drop2", ["--team-feat", "--dropout", "0.2", "--team-mode", "bt",
                       "--head-dim", "32"]),
        ("h32-drop4", ["--team-feat", "--dropout", "0.4", "--team-mode", "bt",
                       "--head-dim", "32"]),
        ("h32-lr5e4", ["--team-feat", "--dropout", "0.3", "--team-mode", "bt",
                       "--head-dim", "32", "--lr", "5e-4"]),
    ],
    # Does separating "T1 is good" from "T1 still has Faker" help? The player
    # block is nearly collinear with the team block (a roster is what a team
    # tag *is*), so the joint fit has to be shrunk hard to say anything: kp is
    # the player pseudo-count and the grid is about finding out whether any
    # value beats plain bt (which is h24's 0.6347).
    "roster": [
        ("r-kp10", ["--team-feat", "--dropout", "0.3", "--team-mode", "roster",
                    "--head-dim", "24"]),
        ("r-kp50", ["--team-feat", "--dropout", "0.3", "--team-mode", "roster",
                    "--head-dim", "24", "--team-player-k", "50"]),
        ("r-kp200", ["--team-feat", "--dropout", "0.3", "--team-mode", "roster",
                     "--head-dim", "24", "--team-player-k", "200"]),
        ("r-kp1000", ["--team-feat", "--dropout", "0.3",
                      "--team-mode", "roster", "--head-dim", "24",
                      "--team-player-k", "1000"]),
    ],
    # The joint fit was ill-posed (the two blocks are collinear), so drop the
    # team block and ask the question directly: are player ratings a better
    # strength estimate than team tags? Raw ratings say yes by ~0.033 nats
    # (val 0.6062 vs 0.6395), and the median team fields 3 different rosters
    # over this corpus, which is why. Now see what the encoder adds.
    "player": [
        ("p-kp10", ["--team-feat", "--dropout", "0.3", "--team-mode", "player",
                    "--head-dim", "24"]),
        ("p-kp20", ["--team-feat", "--dropout", "0.3", "--team-mode", "player",
                    "--head-dim", "24", "--team-player-k", "20"]),
        ("p-kp10-drop2", ["--team-feat", "--dropout", "0.2",
                          "--team-mode", "player", "--head-dim", "24"]),
        ("p-kp10-h32", ["--team-feat", "--dropout", "0.3",
                        "--team-mode", "player", "--head-dim", "32"]),
    ],
    # soloQ, first question: does the pro finding about the head transfer? The
    # committed soloQ arm (0.68838 val) is d256 / dropout 0.2 / full-width head
    # / 40 epochs selecting at epoch 31.
    "soloq_base": [
        ("h256", []),
        ("h64", ["--head-dim", "64"]),
        ("h32", ["--head-dim", "32"]),
        ("h24", ["--head-dim", "24"]),
        ("h24-drop1", ["--head-dim", "24", "--dropout", "0.1"]),
        ("h24-i1", ["--head-dim", "24", "--num-intra", "1"]),
    ],
    # soloQ, second question: a champion Bradley-Terry prior is worth 0.68987 on
    # its own (865 champion x role cells) - the same as the whole 2.6 M-param
    # encoder. Handing it to the head leaves the encoder the interactions.
    # The h24-i1 recipe from 'soloq_base' carries over (num_intra 1 was the one
    # change that both improved val and trained steadily to epoch 19 instead of
    # peaking at epoch 2).
    "soloq_champ": [
        ("c-h24-role-i1", ["--head-dim", "24", "--num-intra", "1",
                           "--champ-feat", "--champ-by-role"]),
        ("c-h24-i1", ["--head-dim", "24", "--num-intra", "1",
                      "--champ-feat"]),
        ("c-h24-role", ["--head-dim", "24", "--champ-feat",
                        "--champ-by-role"]),
        ("c-h256-role-i1", ["--num-intra", "1", "--champ-feat",
                            "--champ-by-role"]),
        ("c-d128-role-i1", ["--embed-dim", "128", "--head-dim", "24",
                            "--num-intra", "1", "--champ-feat",
                            "--champ-by-role"]),
        ("c-h24-role-i1-k5", ["--head-dim", "24", "--num-intra", "1",
                              "--champ-feat", "--champ-by-role",
                              "--champ-k", "5"]),
    ],
}


def run_one(name, tokens, scratch, base=None):
    out = Path(scratch) / name
    t0 = time.time()
    args = parse_args((base or BASE) + list(tokens) + ["--out-dir", str(out)])
    report = train(args)
    v = report["val"]
    return {"name": name, "tokens": list(tokens),
            "val_log_loss": v["log_loss"], "val_auc": v["roc_auc"],
            "val_brier": v["brier"],
            "epoch": report["protocol"]["selected_epoch"],
            "params": report["protocol"]["params"],
            "corpus": report["protocol"]["corpus"],
            "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", required=True, choices=sorted(GRIDS))
    ap.add_argument("--scratch", default=str(ROOT / "model" / "checkpoints"
                                             / "_search"))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    base = BASE_OVERRIDES.get(args.grid, BASE)
    rows = []
    for name, tokens in GRIDS[args.grid]:
        print("\n" + "#" * 72)
        print("# %s  (%s)" % (name, " ".join(tokens) or "base recipe"))
        print("#" * 72)
        try:
            rows.append(run_one(name, tokens, args.scratch, base))
        except Exception as exc:                      # noqa: BLE001
            print("FAILED %s: %s" % (name, exc))
            rows.append({"name": name, "tokens": list(tokens), "error": str(exc)})

    ok = [r for r in rows if "val_log_loss" in r]
    ok.sort(key=lambda r: r["val_log_loss"])
    print("\n" + "=" * 72)
    print("%-20s %10s %9s %8s %6s %10s %7s" % ("config", "val ll", "val auc",
                                              "val brier", "ep", "params",
                                              "sec"))
    for r in ok:
        print("%-20s %10.5f %9.4f %8.5f %6d %10d %7.0f"
              % (r["name"], r["val_log_loss"], r["val_auc"], r["val_brier"],
                 r["epoch"], r["params"], r["seconds"]))
    for r in rows:
        if "error" in r:
            print("%-20s FAILED: %s" % (r["name"], r["error"][:80]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"grid": args.grid, "base": base, "rows": rows}, f,
                      indent=1)
            f.write("\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
