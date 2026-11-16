"""Does the model follow the label, or the board?

The tell: at move 0 the board is empty (`Bans: -`) and the model still names the
right champion's first token. A model cannot know that. So the question is not
"how good is it" but "is it reading the answer".

The discriminating experiment: keep the board identical, replace the target with
a random *other* champion, and score again.
  - accuracy tracks the fake label  -> the answer is in the input (copy task)
  - accuracy stays on the true label -> it is reading the board
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

    g = torch.Generator().manual_seed(7)

    def predict(sample, target_id):
        s = dict(sample)
        s["target_champ"] = torch.tensor(target_id)
        pr = assemble(tok, [s], names, context="soft+text", device=device)
        batch, _ = D.collate([s])
        batch = D.to_device(batch, device)
        with torch.no_grad():
            soft = model.soft_tokens(batch)
            hidden = model.trunk(
                inputs_embeds=model.inputs_embeds(pr, soft),
                attention_mask=model.spliced_mask(pr),
                use_cache=False).last_hidden_state
            pos = int(splice_positions(pr, model.n_soft)[0, 0])
            lg = model.llm.get_output_embeddings()(
                hidden[:, pos].unsqueeze(1))[0, 0].float()
        return tok.decode([int(lg.argmax())])

    print("%-4s %-16s %-12s %-12s %-12s" %
          ("mv", "true target", "pred@true", "fake target", "pred@fake"))
    n_true = n_fake = n = 0
    for i in range(len(ds)):
        s = ds[i]
        true_id = int(s["target_champ"])
        fake_id = int(torch.randint(0, len(names), (1,), generator=g))
        while fake_id == true_id:
            fake_id = int(torch.randint(0, len(names), (1,), generator=g))
        p_true = predict(s, true_id)
        p_fake = predict(s, fake_id)
        n += 1
        n_true += p_true.strip().startswith(names[true_id].split()[0][:2])
        n_fake += p_fake.strip().startswith(names[fake_id].split()[0][:2])
        print("%-4d %-16r %-12r %-12r %-12r"
              % (int(s["move_idx"]), names[true_id], p_true,
                 names[fake_id], p_fake))

    print("\nprediction follows the TRUE label: %d/%d" % (n_true, n))
    print("prediction follows the FAKE label: %d/%d" % (n_fake, n))
    print("\nIf the fake column tracks too, the answer is in the input.")


if __name__ == "__main__":
    main()
