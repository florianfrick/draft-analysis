"""Run the local 4B on the agent's own prompts, base vs adapter.

The hosted-API arms all landed within 4 pp of the next-pick head, and the
adapter is +13 pp over it. Two explanations are still standing: the adapter has
draft-specific weights, or the adapter has a *restricted answer space* (the
`keep` mask over five first-tokens). The second one predicts something the API
cannot do and this script can: ask the local model the agent's question in
prose, with no mask, and see what it produces.

Two arms, because they answer different questions:

  chat   six neutral questions on a real board. Does the adapter still write
         prose, or did the reranker's answer-token loss eat it? That is the
         catastrophic-forgetting check, and it is qualitative on purpose - the
         question is whether the register survives, not whether a benchmark
         moved.
  rank   the panel's job: the agent's board, the head's shortlist, and a rank
         instruction, generated freely, read back with `suggest.parse_ranking`
         and scored against the move the pro actually made.

One deliberate deviation from `/api/agent-rank`: the API arm *fetches* the
shortlist with a `predict_next_move` tool call, because it has tools. The local
model has none, so the shortlist is written into the prompt instead. Same
information, different plumbing - and the plumbing is the API arm's one genuine
advantage, so handing it over is the generous reading.

`--which base` loads the same trunk with no adapter, which is the control: any
difference is the 21.2 M adapter weights, not the checkpoint, the split, or the
board.

  python -m pro.model.archive.llm.eval_agent_local --which adapter --mode chat --limit 6
  python -m pro.model.archive.llm.eval_agent_local --which base --mode rank --limit 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.agent import model_tools, suggest                    # noqa: E402
from pro.model.agent.eval_rank_api import (                         # noqa: E402
    DEFAULT_HEAD, LockedHead, _state_of, head_top)
from pro.model.archive.llm import build as B                                # noqa: E402
from pro.model.archive.llm import common as C                               # noqa: E402
from pro.model.archive.llm import data as D                                 # noqa: E402
from pro.model.archive.llm.prompt import PREFIX, tokenize_prompt            # noqa: E402
from solo_queue.model.policy import infer                           # noqa: E402

# Neutral, and deliberately not answerable with a champion name: if the adapter
# has collapsed into "emit one name and stop", these are the prompts that show
# it. Two of the six still need the board, so "it stopped reading the board" and
# "it stopped writing prose" stay separable.
CHAT_QUESTIONS = [
    "In one sentence, what is a lane swap and why do pro teams use it?",
    "Explain what the fearless draft rule means for the ban phase.",
    "Which side has the better composition on this board, and why?",
    "What is the biggest risk for the side on the clock here?",
    "Name one champion that would be a bad pick here and say why, in one line.",
    "Summarise this draft in two sentences for someone who has never watched "
    "pro play.",
]

# The rank question, rewritten for a model with no tool call. `suggest.
# rank_question` says "first call `predict_next_move`"; that instruction is a
# dead end here, and leaving it in would test whether the model hallucinates a
# tool call rather than whether it can rank.
RANK_QUESTION = (
    "Rank the top %d %ss the side on the clock should make next, from best to "
    "worst. The head's own legal shortlist is listed above, best first, with "
    "the head's confidence in each. Reorder and relabel those candidates by how "
    "strong each is on this board. List exactly %d champions, one per line, "
    "each as `NAME (NN%%)` where NN%% is your confidence that this is the best "
    "available move. Do not name a champion the head did not list; if you trust "
    "the head, echo its order."
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--which", default="adapter", choices=["base", "adapter"])
    p.add_argument("--mode", default="chat", help="chat, rank, or chat,rank")
    p.add_argument("--checkpoint", default="llm-lora-full2",
                   help="checkpoint dir under pro/model/checkpoints (adapter "
                        "+ projector); ignored with --which base")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--head", default=DEFAULT_HEAD)
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--limit", type=int, default=6,
                   help="rows (one seeded prefix per game)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=suggest.RANK_K)
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help="0 = per-mode default (chat 100, rank 220)")
    p.add_argument("--chat-format", default="chat", choices=["chat", "raw"],
                   help="'chat' appends <|im_start|>assistant before generating "
                        "(the format the base trunk expects); 'raw' is the "
                        "completion format the adapter was trained on, where "
                        "the answer follows the text with no turn marker. The "
                        "base model emits EOS immediately in 'raw', so a "
                        "single-format comparison measures the template, not "
                        "the weights")
    p.add_argument("--out", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="print the first prompt of each mode and exit")
    return p.parse_args(argv)


def build_model(args, device):
    """The `evaluate_llm` recipe, minus the dataset: encoder, projector, trunk."""
    ckpt = HERE.parent / "checkpoints" / args.checkpoint
    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)
    llm = B.load_llm("Qwen/Qwen3.5-4B", quantize=True, device=device.index or 0)
    B.freeze(llm)
    if args.which == "adapter":
        from peft import PeftModel
        adapter = ckpt / "adapter"
        if not adapter.exists():
            raise SystemExit("%s has no adapter/ - use --which base" % ckpt)
        llm = PeftModel.from_pretrained(llm, str(adapter))
        llm.eval()
    d_llm = llm.get_input_embeddings().weight.shape[1]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = B.build_draft_llm(enc, llm, tok, C.champ_names(), cfg["embed_dim"],
                              d_llm, 1024, n_soft=10).to(device)
    if args.which == "adapter":
        B.load_projector(model, str(ckpt / "projector.pt"))
    model.eval()
    return model, tok


def prompt_ids(tok, body):
    """`tokenize_prompt` plus the two tensors `inputs_embeds`/`spliced_mask`
    read. The placeholder is inserted by index, never searched for in text, so
    the soft block cannot land inside a word."""
    ids, ph = tokenize_prompt(tok, PREFIX, body)
    t = torch.tensor([ids], dtype=torch.long)
    return {"ids": t, "attn": torch.ones_like(t), "ph_idx": ph}


@torch.no_grad()
def generate(model, tok, enc, body, max_new, eos_id):
    """Greedy, one row, no padding - `pad_token_id == <|im_end|>` here, so a
    batch would stop on another row's answer."""
    device = model.trunk.embed_tokens.weight.device
    prompt = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in prompt_ids(tok, body).items()}
    soft = model.soft_tokens(enc)
    emb = model.inputs_embeds(prompt, soft)
    attn = model.spliced_mask(prompt)
    t0 = time.time()
    out = model.llm.generate(inputs_embeds=emb, attention_mask=attn,
                             max_new_tokens=max_new, do_sample=False,
                             num_beams=1, eos_token_id=eos_id,
                             pad_token_id=eos_id)
    # with inputs_embeds the output holds *only* the new tokens
    return tok.decode(out[0], skip_special_tokens=True).strip(), time.time() - t0


def rank_body(state, head, k, kind):
    listing = "\n".join("  %d. %s - %.1f%%"
                        % (i + 1, h["name"], 100 * h["p_head"])
                        for i, h in enumerate(head[:k]))
    return "\n".join([
        state.render("real", None),
        "",
        "The head's shortlist (%d legal %ss, best first):" % (len(head[:k]), kind),
        listing,
        "",
        RANK_QUESTION % (k, kind, k),
        "",
    ])


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = C.load_rows(args.corpus)
    split = C.split_rows(rows)
    patch_index = D.make_patch_index(rows)[1]
    names = C.champ_names()
    champs = model_tools.ChampIndex(names)

    # `eval_dataset` is the shared val: held out from the adapter's train split,
    # so a good score here is not memorisation.
    ds = D.eval_dataset(rows, split["val"], patch_index, seed=args.seed,
                        val_games=args.limit)
    n_moves = D.n_moves()          # prefix_mode="all": 20 rows per game
    rng = random.Random(args.seed)
    picks = [rng.randrange(n_moves) for _ in range(args.limit)]

    model, tok = build_model(args, device)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    # CPU on purpose: the card is holding the 4B, and the head is 2.5 M params.
    pred = LockedHead(infer.PolicyPredictor(args.head, device="cpu"))
    print("%s | %s | %d rows | device %s"
          % (args.which, args.checkpoint if args.which == "adapter" else "-",
             args.limit, device), flush=True)

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    out = {"which": args.which, "checkpoint": args.checkpoint,
           "chat_format": args.chat_format, "modes": modes, "records": []}

    for g in range(args.limit):
        sample = ds[g * n_moves + picks[g]]
        enc, _ = D.collate([sample])
        enc = D.to_device(enc, device)
        state = _state_of(sample, names)
        truth = names[int(sample["target_champ"])]
        kind = "ban" if int(sample["phase"]) in (0, 2) else "pick"
        head = head_top(pred, champs, state, k=max(args.k, 20))
        base = {"gameid": sample["meta"]["gameid"],
                "move_idx": int(sample["move_idx"]), "kind": kind,
                "truth": truth,
                "head_top1": head[0]["name"] if head else None,
                # kept so the invention rate is computable offline: the panel's
                # contract is "reorder the head, never name something it did
                # not list", and a local model is free to break it.
                "head_names": [h["name"] for h in head]}

        if "chat" in modes:
            board = state.render("real", None)
            tail = ("<|im_start|>assistant\n" if args.chat_format == "chat"
                    else "\n")
            for q in CHAT_QUESTIONS:
                text, secs = generate(model, tok, enc,
                                      board + "\n\n" + q + "\n" + tail,
                                      args.max_new_tokens or 100, eos_id)
                if args.dry_run:
                    print("\n--- chat prompt ---\n%s\n\n%s\n" % (board, q))
                    print("--- answer ---\n%s" % text)
                    return 0
                out["records"].append(dict(base, mode="chat", question=q,
                                           answer=text, seconds=round(secs, 2)))

        if "rank" in modes:
            body = rank_body(state, head, args.k, kind)
            if args.chat_format == "chat":
                body += "<|im_start|>assistant\n"
            text, secs = generate(model, tok, enc, body,
                                  args.max_new_tokens or 220, eos_id)
            parsed = suggest.parse_ranking(text, champs, state, args.k, head)
            got = [s["name"] for s in parsed["suggestions"]]
            rec = dict(base, mode="rank", answer=text, parsed=parsed["parse"],
                       got=got, own_numbers=sum(1 for s in parsed["suggestions"]
                                                if s["source"] == "model"),
                       in1=bool(got) and got[0] == truth,
                       in5=truth in got[:5],
                       head_in1=head[0]["name"] == truth,
                       seconds=round(secs, 2))
            out["records"].append(rec)
            if args.dry_run:
                print("\n--- rank prompt ---\n%s\n--- answer ---\n%s\n"
                      "\nparsed=%s own_numbers=%d got=%s truth=%s"
                      % (body, text, rec["parsed"], rec["own_numbers"], got,
                         truth))
                return 0
            print("  %-3s %-8s top1=%-14s truth=%-14s own=%d"
                  % (rec["move_idx"], kind, got[0] if got else "-", truth,
                     rec["own_numbers"]), flush=True)

    recs = out["records"]
    summary = {}
    for mode in modes:
        rows_ = [r for r in recs if r["mode"] == mode]
        if not rows_:
            continue
        s = {"n": len(rows_)}
        if mode == "rank":
            s["top1"] = sum(r["in1"] for r in rows_) / len(rows_)
            s["top5"] = sum(r["in5"] for r in rows_) / len(rows_)
            s["head_top1"] = sum(r["head_in1"] for r in rows_) / len(rows_)
            s["own_numbers"] = sum(r["own_numbers"] for r in rows_) / len(rows_)
            s["parse"] = {p: sum(1 for r in rows_ if r["parsed"] == p)
                          for p in {r["parsed"] for r in rows_}}
        s["seconds"] = round(sum(r["seconds"] for r in rows_) / len(rows_), 2)
        summary[mode] = s
    out["summary"] = summary
    print(json.dumps(summary, indent=1))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
