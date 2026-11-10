"""The reranker's two numbers, and the split that makes them honest.

    python -m pro.model.llm.rerank --split heldout --k 1,3,5,10,20
    python -m pro.model.llm.rerank --split rr

A reranker over the next-pick head's candidates cannot score better than
**recall@k** - the share of rows whose true answer is in the list it is given.
That is the ceiling, and it is also the share of rows that can be trained on at
all: a row whose answer is not in the list has nothing to learn. So recall@k is
measured before anything is trained, not after.

**There is one split.** `C.split_rows` is the only partition in the project:
the head's trainer, both LLM stages and this file all read the same
`split_within_patch` call, so the head's val *is* the LLM's val and one number
can be quoted against the other. This file used to reconstruct the head's split
from `val_frac=.1, test_frac=.1, val_random=False` on the theory that the head
trained on the *head* of every patch; the shipped head actually trained with
`--val-split random --no-test` (its checkpoint records `test: skipped
(--no-test)` and was initialised from `masking-proonly-randomvalsplit-*`).
Under the wrong reconstruction 90.4 % of the LLM's val looked like head-train;
under the real one it is **0.3 %** (14 of 4,763 games). See
`_split_overlap.py`, which now prints both.

Consequences, and the reason `common` is gone:

  heldout   eval on the whole shared val, no training rows - 4,763 games /
            95,260 states. The ceiling measurement, and the honest
            cross-model set. SE ~0.15 pp at a 35 % top-1.
  rr        the shared val cut in half by game: 2,381 games train the
            reranker, 2,382 evaluate it. Both halves are held out from the
            head, so the candidate lists look like serving on *both* sides -
            which the old pool did not (90.5 % of `headheld` was head-train).

Caveat either way: the head's `best.pt` was *selected* on its val, so val is
mildly optimistic as a held-out set. Unavoidable without retraining the head
with a separate selection split; noted rather than corrected.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pro.model.llm import common as C
from pro.model.llm import data as D
from pro.model.proonly import common as PC

HEAD_CKPT = C.CKPT_DIR / "nextpick-bans-aware-through-wlds-ig-tes" / "best.pt"
SPLIT_JSON = C.CKPT_DIR / "llm-rerank-splits.json"
# The reranker's own train/eval cut of the shared val. A module constant for
# the same reason `VAL_SPLIT_SEED` is one: two invocations must not disagree
# about which half a game is on.
RR_SPLIT_SEED = 20261105


def load_head(ckpt, device):
    """Rebuild exactly the module set the checkpoint was trained with.

    Same construction as `pro/model/bans/probe.py:load_model` - duplicated
    rather than imported because that module lives outside the package and
    runs `sys.path` surgery at import time. The config carries the ban flags,
    and they are read from it, never defaulted: a checkpoint trained without
    `ban_fearless` rebuilt with it on would score different rows differently.
    """
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    enc = PC.build_encoder(cfg["num_champs"], cfg["embed_dim"],
                           cfg["num_intra"], cfg["num_cross"],
                           cfg["num_heads"], cfg.get("dropout", 0.0),
                           cfg.get("n_patches", 0))
    model = PC.PolicyValueModel(
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
    return model.to(device).eval(), cfg


def head_split(rows):
    """The shared split - the one partition every stage trains and scores on.

    Deliberately *not* a reconstruction from flags: a second call to
    `split_within_patch` with guessed flags is how this file spent months
    scoring a 478-game sliver. Read it from `C.SPLIT` and the tracks cannot
    drift.

    The shipped head's own run differs in one flag: it passed `--no-test`, so
    it had no pinned test tail and drew its val from a 7-game-larger pool.
    That moves 14 of 4,763 games (0.3 %) into its train - negligible here, and
    exactly zero after a retrain that keeps `pinned_test=TEST_SERIES`, which
    `test_rerank.py` pins as identical across the three parts.
    """
    return C.split_rows(rows)


def _rr_halves(games, seed=RR_SPLIT_SEED):
    """Deterministic 50/50 cut of the shared val, at game granularity."""
    games = sorted(games)
    rng = random.Random(seed)
    rng.shuffle(games)
    half = len(games) // 2
    return set(games[:half]), set(games[half:])


def split_games(rows, which="rr"):
    """-> (train indices, eval indices) as indices into `rows`.

    Returned at *game* granularity: the dataset expands each game to its 20
    prefixes, so keeping games whole is what stops a game's prefixes straddling
    the two sides.

      rr        half the shared val trains, half evaluates (see module doc)
      heldout   no train rows; the whole shared val is the eval. Ceiling only.
    """
    split = head_split(rows)
    val_games = {rows[i]["gameid"] for i in split["val"]}
    if which == "rr":
        train, evalg = _rr_halves(val_games)
    elif which == "heldout":
        train, evalg = set(), val_games
    elif which in ("common", "headheld", "llmval"):
        raise SystemExit(
            "--split %s is retired: it came from reconstructing the head's "
            "split with val_random=False, which the shipped head never used. "
            "Use --split rr (train + eval) or --split heldout (ceiling only)."
            % which)
    else:
        raise SystemExit("unknown split %r" % which)
    keep = train | evalg
    idx = {}
    for i, row in enumerate(rows):
        if row["gameid"] in keep:
            idx[row["gameid"]] = i
    tr = sorted(idx[g] for g in train if g in idx)
    ev = sorted(idx[g] for g in evalg if g in idx)
    return tr, ev


def candidates(logits, k):
    """-> (ids [B,k], prob [B,k]) over the *legal* champions.

    `policy_logits` is already legality-masked with -1e4 on taken/unavailable
    (`PolicyValueModel._apply_legality`), so a plain softmax puts ~0 on the
    illegal ones and the top-k can never contain a champion that is off the
    board. That is the property the reranker relies on: its answer is legal by
    construction, so the illegal-rate metric disappears rather than improving.
    """
    k = min(k, logits.shape[1])
    top = logits.topk(k, dim=1)
    return top.indices, torch.softmax(logits, dim=1).gather(1, top.indices)


def ranks(logits, target):
    truth = logits.gather(1, target[:, None]).squeeze(1)
    return (logits > truth[:, None]).sum(1)


@torch.no_grad()
def measure(model, loader, device, ks=(1, 3, 5, 10, 20), limit=0):
    """recall@k, the rank distribution, and the split by move kind."""
    n = 0
    hits = {k: 0 for k in ks}
    by_kind = {}
    rank_hist = {}
    seen_batches = 0
    for batch, _samples in loader:
        batch = D.to_device(batch, device)
        logits = model(batch)["policy_logits"]
        r = ranks(logits, batch["target_champ"])
        is_ban = batch["is_ban"].bool()
        n += r.numel()
        for k in ks:
            hits[k] += int((r < k).sum())
        for kind, sel in (("pick", ~is_ban), ("ban", is_ban)):
            if not bool(sel.any()):
                continue
            e = by_kind.setdefault(kind, {"n": 0, **{k: 0 for k in ks}})
            e["n"] += int(sel.sum())
            for k in ks:
                e[k] += int((r[sel] < k).sum())
        for v in r.tolist():
            b = 0 if v < 1 else (1 if v < 5 else (5 if v < 10 else
                                                 (10 if v < 20 else 20)))
            rank_hist[b] = rank_hist.get(b, 0) + 1
        seen_batches += 1
        if limit and seen_batches >= limit:
            break
    out = {"n_states": n,
           "recall": {"top%d" % k: hits[k] / max(1, n) for k in ks},
           "by_kind": {kk: {"n": e["n"],
                            **{"top%d" % k: e[k] / max(1, e["n"]) for k in ks}}
                       for kk, e in by_kind.items()},
           "rank_hist": {str(k): v / max(1, n)
                         for k, v in sorted(rank_hist.items())}}
    return out


def keep_mask(cand_first, target_first, n_vocab):
    """[B, V] bool: the candidate list's first tokens, plus the target's.

    This is the reranker's decode constraint *and* its loss mask. The target is
    unioned in because a row whose answer is missing from the list would
    otherwise get a +inf loss; counting it as wrong is the honest reading, and
    the eval path needs the row in the denominator to say so.
    """
    B, k = cand_first.shape
    keep = torch.zeros(B, n_vocab, dtype=torch.bool, device=cand_first.device)
    rows = torch.arange(B, device=cand_first.device)[:, None].expand(B, k)
    live = cand_first >= 0
    keep[rows[live], cand_first[live]] = True
    keep[torch.arange(B, device=cand_first.device), target_first] = True
    return keep


@torch.no_grad()
def build_pairs(rows, game_idx, patch_index, head, device, k,
                batch_size=256):
    """-> ([(row_index, move_idx)], n_rows) for rows with the answer in top-k.

    A single pass over every (game, move) of `game_idx`, keeping the ones a
    k-way reranker could actually get right. That list *is* the training set:
    a row whose answer is not in the list has no learnable answer, and
    including it would teach the model that the list is usually wrong when it
    usually is not.
    """
    ds = D.DS.ProNextMoveDataset(rows, indices=game_idx,
                                 patch_index=patch_index, prefix_mode="all",
                                 swap_aug=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=D.collate)
    pairs, n, base = [], 0, 0
    for batch, _samples in loader:
        batch = D.to_device(batch, device)
        r = ranks(head(batch)["policy_logits"], batch["target_champ"])
        # `pos` is an index *within the batch*; the dataset position is
        # `base + pos`, and the game is that position's quotient by N_MOVES.
        # Using `pos` directly silently maps every batch onto the first
        # `batch_size // N_MOVES` games - 45,407 pairs collapsed onto 13 games,
        # which trains a reranker that has seen 13 drafts. The assertion below
        # is what makes that impossible to miss.
        for pos in (r < k).nonzero().flatten().tolist():
            g = base + pos
            pairs.append((game_idx[g // C.N_MOVES], g % C.N_MOVES))
        base += r.numel()
        n += r.numel()
    if base != len(ds):
        raise SystemExit("build_pairs: loader covered %d of %d rows"
                         % (base, len(ds)))
    n_games = len({g for g, _m in pairs})
    if n_games < 0.5 * len(game_idx):
        raise SystemExit("build_pairs: %d distinct games for %d inputs - the "
                         "position mapping is wrong" % (n_games, len(game_idx)))
    return pairs, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(HEAD_CKPT))
    ap.add_argument("--split", default="heldout", choices=("rr", "heldout"))
    ap.add_argument("--k", default="1,3,5,10,20")
    ap.add_argument("--corpus", default="2022", choices=("2022", "all"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="")
    ap.add_argument("--build-pairs", default="",
                    help="write the training pool's usable (game, move) pairs "
                         "to this path, for `finetune_lora --pairs`")
    ap.add_argument("--pair-k", type=int, default=5)
    ap.add_argument("--dump-splits", action="store_true",
                    help="write the game-index lists the trainer reuses")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    ks = tuple(int(x) for x in args.k.split(","))

    rows = C.load_rows(args.corpus)
    keys, patch_index = D.make_patch_index(rows)
    train_idx, eval_idx = split_games(rows, args.split)
    print("%s: %d eval games, %d training games"
          % (args.split, len(eval_idx), len(train_idx)), flush=True)

    ds = D.eval_dataset(rows, eval_idx, patch_index, seed=0, val_games=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=D.collate)
    print("%d states" % len(ds), flush=True)

    model, cfg = load_head(args.ckpt, device)
    print("head: ban_side %s ban_fearless %s counter_rank %s"
          % (cfg.get("ban_side"), cfg.get("ban_fearless"),
             cfg.get("counter_rank")), flush=True)

    r = measure(model, loader, device, ks, limit=args.limit_batches)
    r["split"] = args.split
    r["n_games"] = len(eval_idx)
    r["ckpt"] = str(args.ckpt)

    print("\nrecall@k (ceiling for a k-way reranker)")
    for k in ks:
        print("  top-%-2d %.4f" % (k, r["recall"]["top%d" % k]))
    for kind, e in r["by_kind"].items():
        print("  %-4s n=%-6d %s" % (kind, e["n"], "  ".join(
            "top%d %.4f" % (k, e["top%d" % k]) for k in ks)))
    print("\nrank distribution")
    for k, v in r["rank_hist"].items():
        lo = int(k)
        label = ("#1" if lo == 0 else
                 ("#2-5" if lo == 1 else
                  ("#6-10" if lo == 5 else ("#11-20" if lo == 10 else ">20"))))
        print("  %-7s %.4f" % (label, v))

    if args.build_pairs:
        if not train_idx:
            raise SystemExit("--build-pairs needs a training pool; --split rr "
                             "gives one (half the shared val, held out from "
                             "the head). --split heldout has none.")
        pairs, n = build_pairs(rows, train_idx, patch_index, model, device,
                               args.pair_k)
        out = {"k": args.pair_k, "split": args.split, "n_rows": n,
               "n_games": len(train_idx), "pairs": pairs}
        p = Path(args.build_pairs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
        print("\ntrainable pairs at k=%d: %d of %d rows (%.1f %%) over %d "
              "games" % (args.pair_k, len(pairs), n,
                         100.0 * len(pairs) / max(1, n), len(train_idx)))
        print("wrote %s" % p)

    if args.dump_splits:
        SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_JSON.write_text(json.dumps(
            {"rr": split_games(rows, "rr"),
             "heldout": split_games(rows, "heldout")}, indent=0))
        print("\nwrote %s" % SPLIT_JSON)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1))
        print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
