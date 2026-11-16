"""Are the soft tokens load-bearing at all, or just order-blind?

    bash pro/model/archive/llm/run.sh python -m pro.model.archive.llm._probe_soft \
        --projector pro/model/archive/checkpoints/llm-lora-rerank5/projector.pt \
        --adapter   pro/model/archive/checkpoints/llm-lora-rerank5/adapter \
        --limit-batches 20

The earlier probe permuted and reversed the ten soft vectors and moved the
reranker's top-1 by 0.7 pp with 94 % identical picks. That answers "is the
*order* meaningful" and nothing else: permuting a set preserves the set, so a
model that reads the block as a bag is invariant by construction. It is equally
consistent with "the model uses an order-free summary" and with "the model
ignores the block entirely", and those two have very different consequences -
the second would mean the encoder -> projector -> soft-token path is dead weight
at inference, and the reranker is a pure text model.

So this probe destroys the *content* instead of the order, one arm per way it
could matter:

  normal    the real tokens
  zero      all ten zeroed - the block carries nothing
  mean      all ten replaced by their own mean - keeps the summary, kills the
            per-slot structure (this is the arm that separates "bag" from
            "ignored": a bag survives `mean`)
  shuffle   the previous board's tokens (batch rolled by one) - right
            distribution, wrong board, so it isolates board-specific content
            from "some vectors are present"
  pad       the projector's own PAD embedding in all ten slots - in
            distribution, and independent of the board
  perm      a fixed permutation - the original probe, kept for continuity

`zero` is **not** the clean "no information" arm and should not be read as one:
a row of zeros is out of distribution for the trunk, so its cost mixes "the
block said nothing" with "the block said something the model has never seen".
`pad` is the arm that means "no board information, in the model's own
language", and `shuffle` is the arm that means "the right kind of information
about the wrong board". Those two decide the question; `zero` only bounds how
badly the prefix can be corrupted.

Every arm scores the same rows in the same order (`shuffle=False`), with the
same head, the same shortlist and the same first-token mask, so the only
difference is the block. `ft_top1` is the reranker's own metric: the share of
rows whose true answer is the candidate the LLM ranked first. Rows whose truth
is not in the head's top-5 are wrong in every arm and dilute all of them
equally, so the *between-arm* differences are the measurement.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import build as B          # noqa: E402
from pro.model.archive.llm import common as C         # noqa: E402
from pro.model.archive.llm import data as D           # noqa: E402
from pro.model.archive.llm import rerank as R         # noqa: E402
from pro.model.archive.llm.prompt import (            # noqa: E402
    assemble, candidate_first_ids)

ARMS = ("normal", "pad", "shuffle", "mean", "zero", "perm")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--projector", required=True)
    p.add_argument("--adapter", default="")
    p.add_argument("--head", default=str(R.HEAD_CKPT))
    p.add_argument("--split", default="heldout", choices=["heldout", "rr"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--n-soft", type=int, default=10)
    p.add_argument("--limit-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--context", default="soft+text",
                   choices=["soft+text", "cand"],
                   help="cand = board hidden, candidate list kept; the block "
                        "is then the only source of board information")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = C.load_rows("2022")
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    _, eval_idx = R.split_games(rows, args.split)
    ds = D.eval_dataset(rows, eval_idx, patch_index, seed=0, val_games=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=D.collate, num_workers=0)
    print("%s: %d games, %d states (first %d batches)"
          % (args.split, len(eval_idx), len(ds), args.limit_batches),
          flush=True)

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
    B.load_projector(model, args.projector)
    model.eval()
    head, _ = R.load_head(args.head, device)
    n_vocab = int(llm.get_input_embeddings().weight.shape[0])
    perm = torch.randperm(args.n_soft,
                          generator=torch.Generator().manual_seed(args.seed))

    original = model.soft_tokens
    pad_vec = model.projector.pad_token.detach().to(device)

    def arm_fn(arm):
        def f(enc_batch):
            s = original(enc_batch)
            if arm == "normal":
                return s
            if arm == "zero":
                return torch.zeros_like(s)
            if arm == "mean":
                return s.mean(dim=1, keepdim=True).expand_as(s).contiguous()
            if arm == "shuffle":
                return s.roll(1, dims=0)
            if arm == "pad":
                return pad_vec.to(s.dtype).view(1, 1, -1).expand_as(s).contiguous()
            if arm == "perm":
                return s[:, perm.to(s.device), :].contiguous()
            raise SystemExit("unknown arm %r" % arm)
        return f

    hits = {a: 0 for a in ARMS}
    # Per-row outcomes, so every arm can be compared to `normal` as a paired
    # test. The unpaired difference is the wrong statistic here: the arms see
    # identical rows and agree on ~90 % of them, so the paired discordance is
    # what carries the evidence.
    per_row = {a: [] for a in ARMS}
    n = 0
    with torch.no_grad():
        for i, (batch, samples) in enumerate(loader):
            if i >= args.limit_batches:
                break
            batch = D.to_device(batch, device)
            hlog = head(batch)["policy_logits"]
            cid, _cp = R.candidates(hlog, args.k)
            cands = [[(int(c), 0.0) for c in row] for row in cid.tolist()]
            prompt = assemble(tok, samples, names, context=args.context,
                              loss_on_eos=False, device=device,
                              prompt_format="v4", candidates=cands)
            keep = R.keep_mask(
                candidate_first_ids(tok, names, cands).to(device),
                prompt["ans_ids"][:, 0], n_vocab)
            for arm in ARMS:
                model.soft_tokens = arm_fn(arm)
                out = model(batch, prompt, keep=keep)
                ok = (out["row"]["rank"] == 0)
                hits[arm] += int(ok.sum())
                per_row[arm].append(ok.cpu())
            n += batch["target_champ"].shape[0]
            print("  %d rows" % n, end="\r", flush=True)

    model.soft_tokens = original
    print("\n%d rows, k=%d, %s, context=%s"
          % (n, args.k, Path(args.projector).parent.name, args.context))
    base = hits["normal"] / max(1, n)
    base_row = torch.cat(per_row["normal"])
    for arm in ARMS:
        acc = hits[arm] / max(1, n)
        line = "  %-8s ft_top1 %.4f   %+6.2f pp vs normal" % (
            arm, acc, 100.0 * (acc - base))
        if arm != "normal":
            other = torch.cat(per_row[arm])
            b_only = int((base_row & ~other).sum())   # normal right, arm wrong
            a_only = int((~base_row & other).sum())   # normal wrong, arm right
            p = _mcnemar_p(b_only, a_only)
            line += "   discordant %d/%d  p=%.3f" % (b_only, a_only, p)
        print(line)
    print("\n`pad` ≈ `normal` -> the block is a format slot, not information:")
    print("   nothing about the board reaches the model through it.")
    print("`shuffle` ≈ `normal` -> the content is not board-specific.")
    print("`mean` ≈ `normal` -> the per-slot structure is unnecessary (a bag).")
    print("`zero` << `normal` -> out-of-distribution, not evidence about content.")
    return 0


def _mcnemar_p(b_only, a_only):
    """Exact two-sided binomial test on the discordant pairs."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    k = min(b_only, a_only)
    tail = sum(_binom(n, i) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def _binom(n, k):
    from math import comb
    return comb(n, k)


if __name__ == "__main__":
    sys.exit(main())
