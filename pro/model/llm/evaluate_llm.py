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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.llm import build as B          # noqa: E402
from pro.model.llm import common as C        # noqa: E402
from pro.model.llm import data as D          # noqa: E402
from pro.model.llm.metrics import chance_levels  # noqa: E402
from pro.model.llm.prompt import assemble    # noqa: E402

SLICE_NAMES = {0: "bans1", 1: "picks1", 2: "bans2", 3: "picks2"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--projector", required=True)
    p.add_argument("--adapter", default="",
                   help="LoRA adapter dir (stage 2); empty = frozen LLM")
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--val-games", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--n-soft", type=int, default=10)
    p.add_argument("--context", default="soft+text",
                   choices=["soft+text", "soft"])
    p.add_argument("--loss-on-eos", action="store_true")
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


class Bucket:
    """Per-row accumulators for one slice."""

    def __init__(self):
        self.n = self.exact = self.rank = self.top1 = 0.0
        self.ill1 = self.ill5 = self.n_tok = 0.0

    def add(self, row, mask):
        k = float(mask.sum())
        if not k:
            return
        self.n += k
        self.exact += float((row["exact"] * mask).sum())
        self.rank += float((row["rank"] * mask).sum())
        self.top1 += float(((row["rank"] == 0).float() * mask).sum())
        self.n_tok += float((row["n_tok"] * mask).sum())
        if "ill1" in row:
            self.ill1 += float((row["ill1"] * mask).sum())
            self.ill5 += float((row["ill5"] * mask).sum())

    def out(self):
        if not self.n:
            return {"n": 0}
        return {"n": int(self.n),
                "exact": self.exact / self.n,
                "ft_top1": self.top1 / self.n,
                "mean_rank": self.rank / self.n,
                "tokens_per_row": self.n_tok / self.n,
                "top1_illegal": self.ill1 / self.n,
                "top5_illegal": self.ill5 / self.n}


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = C.load_rows(args.corpus)
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    va_ds = D.eval_dataset(rows, split["val"], patch_index, seed=0,
                           val_games=args.val_games)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=D.collate, num_workers=0)

    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)
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
    B.load_projector(model, args.projector)
    model.eval()

    overall = Bucket()
    by_ban = {0: Bucket(), 1: Bucket()}
    by_phase = {k: Bucket() for k in SLICE_NAMES}
    by_move = defaultdict(Bucket)

    with torch.no_grad():
        for i, (batch, samples) in enumerate(va_ld):
            if args.limit_batches and i >= args.limit_batches:
                break
            batch = D.to_device(batch, device)
            prompt = assemble(tok, samples, names, context=args.context,
                              loss_on_eos=args.loss_on_eos, device=device)
            out = model(batch, prompt)
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

    first, _ = C.candidate_tables(tok, names)
    report = {
        "checkpoint": {"projector": args.projector,
                       "adapter": args.adapter or None,
                       "context": args.context},
        "val_games": len(va_ds) // C.N_MOVES,
        "chance_first_token": chance_levels(len(first)),
        "baseline_policy_val": C.BASELINE_VAL,
        "overall": overall.out(),
        "by_action": {"pick": by_ban[0].out(), "ban": by_ban[1].out()},
        "by_phase": {SLICE_NAMES[k]: v.out() for k, v in by_phase.items()},
        "by_move": {str(k): v.out() for k, v in sorted(by_move.items())},
    }

    def fmt(d):
        if not d.get("n"):
            return "n=0"
        return ("n=%-5d exact %.3f ft@1 %.3f rank %.1f ill@1 %.3f"
                % (d["n"], d["exact"], d["ft_top1"], d["mean_rank"],
                   d["top1_illegal"]))

    print("\n== overall ==\n  " + fmt(report["overall"]))
    print("\n== by action ==")
    for k, v in report["by_action"].items():
        print("  %-5s %s" % (k, fmt(v)))
    print("\n== by phase ==")
    for k, v in report["by_phase"].items():
        print("  %-7s %s" % (k, fmt(v)))
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
