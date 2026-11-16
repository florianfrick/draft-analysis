"""Are the move>=7 targets degenerate, or is the model really predicting them?

Moves 7-19 score ~100 % exact while moves 0-6 score ~10 %. If the 200 val games
share almost the same champion at a given slot, then "always say Jinx" scores
100 % and nothing is being predicted. This prints the target distribution and
the model's predictions side by side, so the two explanations separate.
"""
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import build as B
from pro.model.archive.llm import common as C
from pro.model.archive.llm import data as D
from pro.model.archive.llm.prompt import assemble


def main():
    rows = C.load_rows("2022")
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    ds = D.eval_dataset(rows, split["val"], patch_index, seed=0, val_games=200)

    # ---- what are the targets, without any model involved?
    print("target distribution over the 200 val games, per move")
    per_move = {}
    for i in range(len(ds)):
        s = ds[i]
        per_move.setdefault(int(s["move_idx"]), []).append(
            names[int(s["target_champ"])])
    for mv in sorted(per_move):
        c = Counter(per_move[mv])
        top, n = c.most_common(1)[0]
        print("  move %2d  n=%3d  distinct %3d  mode %-14s %3d (%.2f)"
              % (mv, len(per_move[mv]), len(c), top, n, n / len(per_move[mv])))

    # ---- and what does the model say on those same rows?
    print("\nnow the model, same rows (context soft+text)")
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

    ld = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False,
                                     collate_fn=D.collate, num_workers=0)
    preds = {}
    with torch.no_grad():
        for batch, samples in ld:
            batch = D.to_device(batch, device)
            pr = assemble(tok, samples, names, context="soft+text",
                          device=device)
            soft = model.soft_tokens(batch)
            hidden = model.trunk(
                inputs_embeds=model.inputs_embeds(pr, soft),
                attention_mask=model.spliced_mask(pr),
                use_cache=False).last_hidden_state
            from pro.model.archive.llm.prompt import splice_positions
            pos = splice_positions(pr, model.n_soft)
            sel = hidden.gather(
                1, pos[:, :1].unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
            lg = model.llm.get_output_embeddings()(sel)[:, 0].float()
            # rank the target among the 136 candidate first tokens
            first, _ = C.candidate_tables(tok, names)
            first = first.to(device)
            sub = lg[:, first]
            pred = sub.argmax(dim=-1)
            for j, s in enumerate(samples):
                mv = int(s["move_idx"])
                preds.setdefault(mv, []).append(
                    (tok.decode([int(first[pred[j]])]).strip(),
                     names[int(s["target_champ"])]))

    for mv in sorted(preds):
        c = Counter(p for p, _ in preds[mv])
        top, n = c.most_common(1)[0]
        ok = sum(1 for p, t in preds[mv] if p == t) / len(preds[mv])
        print("  move %2d  predicted-mode %-14s %3d (%.2f)  first-token acc %.3f"
              % (mv, top, n, n / len(preds[mv]), ok))


if __name__ == "__main__":
    main()
