"""Is the 98.6 % first-token accuracy real?

Two things can make a frozen 4B look clairvoyant when it is not:

  1. an off-by-one in the scored position. A causal LM turns `hidden[p]` into
     a distribution over token p+1, so the position that *predicts* the first
     answer token is the last prompt token, not the first answer token. If
     `splice_positions` were off by one, the model would be reading the answer
     token it has already consumed.
  2. the board text doing all the work. The encoder's own policy head scores
     18.32 % top-1 from the same state, so the soft tokens cannot support 98 %;
     if the text alone supports it, the LLM is reasoning from the board and the
     projector is not the story.

This prints, for real eval rows, the top-5 at `pos` and at `pos-1`, plus the
metrics for `--context soft+text` and `--context soft`.

    python -m pro.model.llm.audit_positions --projector pro/model/checkpoints/llm-warmup-segfix/projector.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.llm import build as B          # noqa: E402
from pro.model.llm import common as C        # noqa: E402
from pro.model.llm import data as D          # noqa: E402
from pro.model.llm.prompt import assemble, splice_positions  # noqa: E402


def top5(tok, logrow):
    v, i = logrow.topk(5)
    return [(tok.decode([int(t)]), round(float(x), 2)) for x, t in zip(v, i)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--projector", required=True)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--n-soft", type=int, default=10)
    p.add_argument("--hidden", type=int, default=1024)
    args = p.parse_args(argv)

    device = torch.device("cuda")
    rows = C.load_rows("2022")
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()

    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)
    llm = B.load_llm(args.model, quantize=True, device=0)
    B.freeze(llm)
    B.enable_checkpointing(llm)
    d_llm = llm.get_input_embeddings().weight.shape[1]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = B.build_draft_llm(enc, llm, tok, names, cfg["embed_dim"], d_llm,
                              args.hidden, n_soft=args.n_soft).to(device)
    blob = B.load_projector(model, args.projector)
    print("loaded projector, val ce in the report: %s"
          % (blob.get("val", {}).get("ce")), flush=True)
    model.eval()

    va_ds = D.eval_dataset(rows, split["val"], patch_index, seed=0, val_games=4)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=D.collate, num_workers=0)

    for ctx in ("soft+text", "soft"):
        print("\n" + "=" * 72)
        print("context = %s" % ctx)
        print("=" * 72, flush=True)
        n_ok_p = n_ok_m = n = 0
        shown = 0
        with torch.no_grad():
            for batch, samples in va_ld:
                batch = D.to_device(batch, device)
                pr = assemble(tok, samples, names, context=ctx, device=device)

                soft = model.soft_tokens(batch)
                E = model.inputs_embeds(pr, soft)
                hidden = model.trunk(
                    inputs_embeds=E,
                    attention_mask=model.spliced_mask(pr),
                    use_cache=False).last_hidden_state

                pos = splice_positions(pr, model.n_soft)      # [B,K]
                head = model.llm.get_output_embeddings()
                tgt = pr["ans_ids"]
                valid = tgt >= 0

                def gather(h, shift):
                    p = (pos + shift).clamp(min=0)
                    sel = h.gather(1, p.unsqueeze(-1).expand(
                        -1, -1, h.shape[-1]))
                    return head(sel).float().log_softmax(-1)

                lg_p = gather(hidden, 0)      # hidden AT the answer token
                lg_m = gather(hidden, -1)     # hidden BEFORE the answer token
                pred_p = lg_p.argmax(-1)
                pred_m = lg_m.argmax(-1)
                n_ok_p += int(((pred_p == tgt) & valid).sum())
                n_ok_m += int(((pred_m == tgt) & valid).sum())
                n += int(valid.sum())

                if shown < args.rows:
                    for r in range(min(2, tgt.shape[0])):
                        if shown >= args.rows:
                            break
                        k = int(pr["n_ans"][r])
                        body = tok.decode(pr["ids"][r].tolist()[-40:])
                        print("\n--- row %d | %d answer tokens | ctx %s"
                              % (shown, k, ctx))
                        print("  tail of prompt: ...%r" % body)
                        print("  target       : %s"
                              % tok.decode(tgt[r, :k].tolist()))
                        print("  at pos   top5: %s" % (top5(tok, lg_p[r, 0]),))
                        print("  at pos-1 top5: %s" % (top5(tok, lg_m[r, 0]),))
                        shown += 1
        print("\n  token accuracy at pos  : %.4f  (%d tokens)"
              % (n_ok_p / n, n))
        print("  token accuracy at pos-1: %.4f" % (n_ok_m / n))
        print("  -> the convention that predicts is the one that scores.", flush=True)


if __name__ == "__main__":
    main()
