"""Read one game's 20 predictions next to their targets, no metric code.

Everything so far agrees with itself but the number is not credible, so this
prints raw strings for a single game: the task line, the target, and the
model's top-1 and top-3 at the scored position, for both contexts.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import build as B
from pro.model.archive.llm import common as C
from pro.model.archive.llm import data as D
from pro.model.archive.llm.prompt import assemble, splice_positions


def main():
    rows = C.load_rows("2022")
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    ds = D.eval_dataset(rows, split["val"], patch_index, seed=0, val_games=1)
    print("states in this game: %d" % len(ds))

    device = torch.device("cuda")
    enc, cfg, _ = C.build_frozen_encoder(C.ENCODER_CKPT, device=device)
    llm = B.load_llm("Qwen/Qwen3.5-4B", quantize=True, device=0)
    B.freeze(llm)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = B.build_draft_llm(enc, llm, tok, names, cfg["embed_dim"],
                              llm.get_input_embeddings().weight.shape[1], 1024,
                              n_soft=10).to(device)
    B.load_projector(model, "pro/model/archive/checkpoints/llm-warmup-segfix/projector.pt")
    model.eval()

    for ctx in ("soft+text", "soft"):
        print("\n" + "=" * 78)
        print("context %s - one game, all 20 moves" % ctx)
        print("=" * 78, flush=True)
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                pr = assemble(tok, [s], names, context=ctx, device=device)
                batch, _ = D.collate([s])
                batch = D.to_device(batch, device)
                soft = model.soft_tokens(batch)
                hidden = model.trunk(
                    inputs_embeds=model.inputs_embeds(pr, soft),
                    attention_mask=model.spliced_mask(pr),
                    use_cache=False).last_hidden_state
                pos = splice_positions(pr, model.n_soft)[:, 0]
                sel = hidden[0, int(pos)].unsqueeze(0).unsqueeze(0)
                lg = model.llm.get_output_embeddings()(sel)[0, 0].float()
                top = lg.topk(3).indices.tolist()
                tgt = tok.decode(pr["ans_ids"][0][pr["ans_ids"][0] >= 0].tolist())
                body = tok.decode(pr["ids"][0].tolist())
                task = body.split("only.")[0].split("Task:")[-1]
                print("  move %2d | target %-14r | top1 %-16r | top3 %s"
                      % (int(s["move_idx"]), tgt,
                         tok.decode([top[0]]),
                         [tok.decode([t]) for t in top]))


if __name__ == "__main__":
    main()
