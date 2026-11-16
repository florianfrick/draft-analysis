"""Score a warmup or fine-tuned checkpoint, sliced by what the move *is*.

The aggregate number is not the interesting one. Two questions decide whether
this architecture is worth a CoT dataset:

1. **Can it tell a pick from a ban?** The encoder's state embedding knows the
   phase, and the prompt says "Blue to ban" in words, so the question is
   whether the output side reflects it. A model that scores well overall but
   predicts picks on ban moves has learned the marginal champion frequency and
   nothing about the move. The per-phase and ban/pick rows are what separate
   those two stories.

2. **Does it ever recommend a champion that is already gone?** Free generation
   has no -inf mask, so a fearless-picked champion can be suggested at all. The
   policy head breaks this zero times (STATUS.md §7); this counts how often the
   LLM does, which is the cost of moving the decision into text.

Slices come from per-row values the model already returns, so this is one pass
over the eval set, not one pass per slice.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import build as B          # noqa: E402
from pro.model.archive.llm import common as C        # noqa: E402
from pro.model.archive.llm import data as D          # noqa: E402
from pro.model.archive.llm import rerank as R        # noqa: E402
from pro.model.archive.llm.metrics import chance_levels  # noqa: E402
from pro.model.archive.llm.prompt import assemble, candidate_first_ids  # noqa: E402

SLICE_NAMES = {0: "bans1", 1: "picks1", 2: "bans2", 3: "picks2"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--projector", required=True)
    p.add_argument("--adapter", default="",
                   help="LoRA adapter dir (stage 2); empty = frozen LLM")
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--split", default="llmval",
                   choices=["llmval", "heldout", "rr"],
                   help="llmval = the shared val, subsampled by --val-games; "
                        "heldout = the same games, all states; rr = the "
                        "reranker's eval half. All three are held out from "
                        "the head, so all three carry cross-model numbers")
    p.add_argument("--val-games", type=int, default=200)
    p.add_argument("--rerank", type=int, default=0,
                   help="K > 0 restricts the answer to the next-pick head's "
                        "top-K candidates (prompt v4 + a first-token mask)")
    p.add_argument("--head", default=str(R.HEAD_CKPT))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--n-soft", type=int, default=10)
    p.add_argument("--context", default="soft+text",
                   choices=["soft+text", "soft", "cand"])
    p.add_argument("--prompt-format", default=None,
                   choices=["v1", "v2", "v3", "v4"],
                   help="v2 labels the turn and the pick's lane; v1 is the "
                        "format every number before 2026-11-08 was measured "
                        "on; v4 adds the candidate list and is implied by "
                        "--rerank. Default: v4 with --rerank, else %s"
                        % C.PROMPT_FORMAT)
    p.add_argument("--loss-on-eos", action="store_true")
    p.add_argument("--no-lane", action="store_true",
                   help="drop 'for <lane>' from the task line, which is what "
                        "the served reranker's prompt looks like when the "
                        "caller did not constrain a role")
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


class Bucket:
    """Per-row accumulators for one slice."""

    def __init__(self):
        self.n = self.exact = self.rank = self.top1 = 0.0
        self.ill1 = self.ill5 = self.n_tok = self.off = 0.0
        # decode-time legality: the same readout with gone champions at -inf
        self.mrank = self.mtop1 = self.mtop5 = 0.0

    def add(self, row, mask):
        k = float(mask.sum())
        if not k:
            return
        self.n += k
        self.exact += float((row["exact"] * mask).sum())
        self.rank += float((row["rank"] * mask).sum())
        self.top1 += float(((row["rank"] == 0).float() * mask).sum())
        if "masked_rank" in row:
            self.mrank += float((row["masked_rank"] * mask).sum())
            self.mtop1 += float(((row["masked_rank"] == 0).float()
                                 * mask).sum())
            self.mtop5 += float(((row["masked_rank"] < 5).float()
                                 * mask).sum())
        self.n_tok += float((row["n_tok"] * mask).sum())
        if "off" in row:
            self.off += float((row["off"] * mask).sum())
        if "ill1" in row:
            self.ill1 += float((row["ill1"] * mask).sum())
            self.ill5 += float((row["ill5"] * mask).sum())

    def out(self):
        if not self.n:
            return {"n": 0}
        out = {"n": int(self.n),
               "exact": self.exact / self.n,
               "ft_top1": self.top1 / self.n,
               "mean_rank": self.rank / self.n,
               "tokens_per_row": self.n_tok / self.n,
               "top1_offlist": self.off / self.n,
               "top1_illegal": self.ill1 / self.n,
               "top5_illegal": self.ill5 / self.n}
        if self.mtop1 or self.mrank:
            out["ft_top1_masked"] = self.mtop1 / self.n
            out["ft_top5_masked"] = self.mtop5 / self.n
            out["mean_rank_masked"] = self.mrank / self.n
        return out


def main(argv=None):
    args = parse_args(argv)
    if args.prompt_format is None:
        args.prompt_format = "v4" if args.rerank else C.PROMPT_FORMAT
    if args.rerank and args.prompt_format != "v4":
        raise SystemExit("--rerank needs --prompt-format v4 (the candidate "
                         "list is what the answer is constrained to)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = C.load_rows(args.corpus)
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    n_unk, worst = C.patch_coverage(rows, patch_index)
    if n_unk:
        print("patch: %d drafts (%.2f%%) are newer than the encoder's table "
              "-> UNK %s" % (n_unk, 100.0 * n_unk / len(rows),
                             [w[0] for w in worst][:4]), flush=True)
    names = C.champ_names()
    if args.split == "llmval":
        va_ds = D.eval_dataset(rows, split["val"], patch_index, seed=0,
                               val_games=args.val_games)
    else:
        _, eval_idx = R.split_games(rows, args.split)
        va_ds = D.eval_dataset(rows, eval_idx, patch_index, seed=0,
                               val_games=0)
        print("%s split: %d games, %d states"
              % (args.split, len(eval_idx), len(va_ds)), flush=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=D.collate, num_workers=0)

    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)
    C.assert_patch_compat(cfg, patch_index)
    llm = B.load_llm(args.model, quantize=True, device=device.index or 0)
    B.freeze(llm)
    if args.adapter:
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, args.adapter)
        llm.eval()
    d_llm = llm.get_input_embeddings().weight.shape[1]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = B.build_draft_llm(enc, llm, tok, names, cfg["embed_dim"], d_llm,
                              args.hidden, n_soft=args.n_soft).to(device)
    blob = B.load_projector(model, args.projector)
    ck_fmt = (blob.get("args") or {}).get("prompt_format")
    if ck_fmt and ck_fmt != args.prompt_format:
        print("  !! %s was trained with prompt_format=%s; scoring it with "
              "%s - the text no longer matches the soft tokens"
              % (args.projector, ck_fmt, args.prompt_format))
    model.eval()

    head = None
    n_vocab = int(llm.get_input_embeddings().weight.shape[0])
    if args.rerank:
        head, _head_cfg = R.load_head(args.head, device)
        print("rerank: k=%d from %s" % (args.rerank, Path(args.head).parent.name),
              flush=True)
    ceiling_hits = ceiling_n = 0

    overall = Bucket()
    by_ban = {0: Bucket(), 1: Bucket()}
    by_phase = {k: Bucket() for k in SLICE_NAMES}
    by_move = defaultdict(Bucket)
    by_role = {k: Bucket() for k in range(len(C.ROLE_DISPLAY))}
    # Illegal suggestions are only *avoidable* on rows that carry a fearless
    # pool: without one, "illegal" means "already banned or picked", which the
    # board states in plain text. Splitting by pool is what separates "the model
    # cannot read the board" from "the model ignores the fearless list".
    by_pool = {(a, p): Bucket() for a in (0, 1) for p in (0, 1)}

    with torch.no_grad():
        for i, (batch, samples) in enumerate(va_ld):
            if args.limit_batches and i >= args.limit_batches:
                break
            batch = D.to_device(batch, device)
            cands = keep = None
            if args.no_lane:
                # The served prompt (RerankPredictor) has no lane unless the
                # caller constrained one, so `task_line` drops "for <lane>".
                # Measured here rather than assumed: the difference is one
                # phrase, but it is one phrase in the prompt that was trained
                # with it. -100 is the out-of-range value `task_line` reads as
                # "no lane", the same one bans already carry.
                for s in samples:
                    s["target_role"] = -100
            if head is not None:
                # The head sees the *same* batch the LLM does, so a side-swapped
                # or otherwise augmented row gets candidates for the state that
                # was actually rendered - there is no cache to fall out of sync.
                hlog = head(batch)["policy_logits"]
                cid, cp = R.candidates(hlog, args.rerank)
                cands = [[(int(c), float(p)) for c, p in zip(a, b)]
                         for a, b in zip(cid.tolist(), cp.tolist())]
                rk = R.ranks(hlog, batch["target_champ"])
                ceiling_hits += int((rk < args.rerank).sum())
                ceiling_n += rk.numel()
            prompt = assemble(tok, samples, names, context=args.context,
                              loss_on_eos=args.loss_on_eos, device=device,
                              prompt_format=args.prompt_format,
                              candidates=cands)
            if head is not None:
                keep = R.keep_mask(
                    candidate_first_ids(tok, names, cands).to(device),
                    prompt["ans_ids"][:, 0], n_vocab)
            out = model(batch, prompt, keep=keep)
            row = {k: v.float() for k, v in out["row"].items()}
            ones = torch.ones_like(row["exact"])
            overall.add(row, ones)
            is_ban = batch["is_ban"].float()
            by_ban[1].add(row, is_ban)
            by_ban[0].add(row, 1.0 - is_ban)
            for ph in SLICE_NAMES:
                by_phase[ph].add(row, (batch["phase"] == ph).float())
            for mv in range(C.N_MOVES):
                by_move[mv].add(row, (batch["move_idx"] == mv).float())
            # `target_role` is -100 on bans, so this slice is picks only
            for r in by_role:
                by_role[r].add(row, (batch["target_role"] == r).float())
            has_pool = batch["unavailable_mask"].any(dim=1).float()
            for a in (0, 1):
                for p in (0, 1):
                    by_pool[(a, p)].add(row, ((is_ban == a) & (has_pool == p))
                                        .float())

    first, _ = C.candidate_tables(tok, names)
    report = {
        "checkpoint": {"projector": args.projector,
                       "adapter": args.adapter or None,
                       "context": args.context,
                       "prompt_format": args.prompt_format},
        "val_games": len(va_ds) // C.N_MOVES,
        "split": args.split,
        "rerank": ({"k": args.rerank,
                    "ceiling_recall": ceiling_hits / max(1, ceiling_n),
                    "n_states": ceiling_n} if args.rerank else None),
        "chance_first_token": chance_levels(len(first)),
        "baseline_policy_val": C.BASELINE_VAL,
        "overall": overall.out(),
        "by_action": {"pick": by_ban[0].out(), "ban": by_ban[1].out()},
        "by_phase": {SLICE_NAMES[k]: v.out() for k, v in by_phase.items()},
        "by_move": {str(k): v.out() for k, v in sorted(by_move.items())},
        "by_role": {C.ROLE_DISPLAY[k]: v.out() for k, v in by_role.items()},
        "by_pool": {"%s_pool=%d" % ("ban" if a else "pick", p): v.out()
                    for (a, p), v in sorted(by_pool.items())},
    }

    def fmt(d):
        if not d.get("n"):
            return "n=0"
        line = ("n=%-5d exact %.3f ft@1 %.3f rank %.1f off@1 %.3f ill@1 %.3f"
                % (d["n"], d["exact"], d["ft_top1"], d["mean_rank"],
                   d["top1_offlist"], d["top1_illegal"]))
        if "ft_top1_masked" in d:
            line += (" | masked ft@1 %.3f ft@5 %.3f rank %.1f"
                     % (d["ft_top1_masked"], d["ft_top5_masked"],
                        d["mean_rank_masked"]))
        return line

    print("\n== overall ==\n  " + fmt(report["overall"]))
    if args.rerank:
        ceil = ceiling_hits / max(1, ceiling_n)
        acc = report["overall"].get("ft_top1", 0.0)
        print("\n== rerank k=%d ==" % args.rerank)
        print("  head recall@%d (the ceiling) %.4f over %d states"
              % (args.rerank, ceil, ceiling_n))
        print("  model k-way accuracy %.4f  = %.0f %% of the ceiling"
              % (acc, 100.0 * acc / max(1e-9, ceil)))
        print("  (ft@5 saturates at 1.000 when k <= 5 - that is the mask "
              "working, not a metric)")
    print("\n== by action ==")
    for k, v in report["by_action"].items():
        print("  %-5s %s" % (k, fmt(v)))
    print("\n== by phase ==")
    for k, v in report["by_phase"].items():
        print("  %-7s %s" % (k, fmt(v)))
    print("\n== by role (picks only) ==")
    for k, v in report["by_role"].items():
        print("  %-7s %s" % (k, fmt(v)))
    print("\n== by action x fearless pool ==")
    for k, v in report["by_pool"].items():
        print("  %-12s %s" % (k, fmt(v)))
    print("\n== by move ==")
    for k, v in report["by_move"].items():
        print("  move %-3s %s" % (k, fmt(v)))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1),
                                  encoding="utf-8")
        print("\nwrote %s" % args.out)
    return report


if __name__ == "__main__":
    main()
