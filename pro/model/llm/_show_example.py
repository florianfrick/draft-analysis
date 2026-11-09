"""One training example, end to end: tensors -> text -> ids -> soft tokens ->
what Qwen emits.

Everything printed here is read off the objects the trainers actually use
(`D.train_dataset`, `prompt.assemble`, `DraftLLM.forward`), not re-derived, so
the dump cannot drift from the run. Three sections:

  1. the row as the encoder sees it (tensors, shapes, names),
  2. the row as the LLM sees it (the literal prompt, the placeholder index, the
     token ids, and which position predicts which answer token),
  3. the answer side: the soft tokens the projector emits, then the model's
     top-k at each scored position, the greedy name under teacher forcing, and
     a free-running 8-token greedy generation.

Section 3 runs twice - once with the freshly calibrated (random) projector,
which is step 0 of the warmup, and once with the trained checkpoint - because
the interesting quantity is the difference, and a dump that only shows the
trained model invites the reader to think the LLM did that on its own.
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pro.model.llm import build as B          # noqa: E402
from pro.model.llm import common as C        # noqa: E402
from pro.model.llm import data as D          # noqa: E402
from pro.model.llm.prompt import (assemble, build_text, splice_positions,  # noqa: E402
                                  tokenize_answer, tokenize_prompt)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--encoder", default=C.ENCODER_CKPT)
    p.add_argument("--projector", default=str(C.CKPT_DIR / "llm-warmup-fixed"
                                             / "projector.pt"))
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--n-soft", type=int, default=10)
    p.add_argument("--context", default="soft+text",
                   choices=["soft+text", "soft"])
    p.add_argument("--prompt-format", default=C.PROMPT_FORMAT,
                   choices=["v1", "v2", "v3"],
                   help="v2 labels the turn and the pick's lane; "
                        "v1 is the format every number before "
                        "2026-11-08 was measured on")
    p.add_argument("--move", type=int, default=9,
                   help="move index (0-19) to look for; 9 = picks1 last move")
    p.add_argument("--any-move", action="store_true",
                   help="take the first sample, whatever move it is")
    p.add_argument("--scan", type=int, default=4000,
                   help="how many train samples to scan for --move")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--n-gen", type=int, default=8)
    return p.parse_args(argv)


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def find_sample(ds, cand, move, ntok, want_pool=True, limit=4000):
    """First sample at `move` (and, by default, with a fearless pool).

    `cand` is the list of dataset indices whose row carries a pool, tried
    first. Scanning the whole split instead would be misleading rather than
    just slow: rows are chronological and fearless drafting only starts
    2025-01-13, so the first few thousand train rows have no pool at all and a
    prefix scan finds nothing.

    Among the matches, the first one whose champion name is more than one token
    is preferred, so the position table shows the shift for j>0 as well. A
    single-token name is a perfectly ordinary row (22/173), it just makes a
    less informative dump.
    """
    cand_set = set(cand)
    order = list(cand) + [i for i in range(min(len(ds), limit))
                          if i not in cand_set]
    fallback = None
    for i in order:
        s = ds[i]
        if move >= 0 and int(s["move_idx"]) != move:
            continue
        if want_pool and not bool(s["unavailable_mask"].any()):
            continue
        if ntok[int(s["target_champ"])] > 1:
            return i, s
        if fallback is None:
            fallback = (i, s)
    if fallback is not None:
        return fallback
    raise SystemExit("no train sample matched move=%d pool=%s" % (move,
                                                                 want_pool))


def show_row(s, names):
    print("game %s  %s vs %s  (%s, %s, %s)"
          % (s["meta"]["gameid"], s["meta"]["blue_team"],
             s["meta"]["red_team"], s["meta"]["league"], s["meta"]["patch"],
             s["meta"]["date"]))
    print("move_idx %d  phase %s  to_move %s  is_ban %d  first_pick %s"
          % (int(s["move_idx"]), C.PHASE_NAMES[int(s["phase"])],
             C.SIDE_NAMES[int(s["to_move"])], int(s["is_ban"]),
             C.SIDE_NAMES[int(s["first_pick"])]))
    print()
    print("  %-18s %-10s %s" % ("tensor", "shape", "value"))
    for k in ("champ_ids", "role_ids", "pad_mask", "ban_ids", "n_bans",
              "taken_mask", "unavailable_mask", "fearless_ids", "n_fearless",
              "patch_id", "target_champ", "weight"):
        v = s[k]
        if v.dim() == 0:
            shown = str(v.item())
        elif v.numel() <= 12:
            shown = str(v.tolist())
        else:
            shown = ("%d hot / %d entries" % (int(v.sum()), v.numel())
                     if v.dtype == torch.bool else "%d entries" % v.numel())
        print("  %-18s %-10s %s" % (k, tuple(v.shape), shown))
    print()
    print("  champ_ids as names: %s"
          % ", ".join(names[int(c)] if not bool(p) else "-"
                      for c, p in zip(s["champ_ids"], s["pad_mask"])))
    print("  ban_ids  as names: %s"
          % ", ".join(names[int(c)] for c in s["ban_ids"][:int(s["n_bans"])]))
    print("  answer (target_champ %d): %s"
          % (int(s["target_champ"]), names[int(s["target_champ"])]))
    print("  taken %d, fearless-unavailable %d champions"
          % (int(s["taken_mask"].sum()), int(s["unavailable_mask"].sum())))


def show_prompt(s, names, tok, context, prompt_format=None):
    prefix, body, ans = build_text(s, names, context=context,
                                   answer=names[int(s["target_champ"])],
                                   prompt_format=prompt_format)
    print("--- prefix (constant for every example) ---")
    print(repr(prefix))
    print("--- body (board + task, the part that changes) ---")
    print(body)
    print("--- answer text ---")
    print(repr(ans))

    ids, ph = tokenize_prompt(tok, prefix, body)
    ans_ids = tokenize_answer(tok, ans)
    print("\nplaceholder index (== len(tokenized prefix)) : %d" % ph)
    print("prompt ids before splice                     : %d" % len(ids))
    print("answer ids                                   : %d %s"
          % (len(ans_ids), ans_ids))
    print("answer tokens decoded                        : %s"
          % " + ".join(repr(tok.decode([t])) for t in ans_ids))
    print("spliced input length                         : %d"
          % (len(ids) - 1 + 10))
    print("\nfirst prompt ids : %s ..." % ids[:8])
    print("last prompt ids  : ... %s   (ph at %d -> %s)"
          % (ids[-6:], ph, tok.convert_ids_to_tokens([ids[ph]])))
    return prefix, body, ans, ids, ph, ans_ids


def show_alignment(tok, s, names, ids, ph, ans_ids, model, prompt):
    """Position table: what is fed where, and what that position must predict."""
    L0 = len(ids)
    K = len(ans_ids)
    print("\nThe input stops one token before each target, so position "
          "len(prompt)+j-1 predicts answer[j]:")
    print("  %-8s %-8s %-26s %-16s" % ("pos_pre", "pos_spliced", "token at that "
                                       "position", "target it must predict"))
    # `prompt["ans_pos_pre"]` is the authority; the arithmetic below only
    # explains it. Compare, and say so if they ever disagree.
    for j in range(K):
        pos_pre = L0 + j - 1
        if int(prompt["ans_pos_pre"][0, j]) != pos_pre:
            print("  !! prompt['ans_pos_pre'][0,%d] = %d, arithmetic says %d"
                  % (j, int(prompt["ans_pos_pre"][0, j]), pos_pre))
        pos_spl = int(splice_positions(prompt, model.n_soft)[0, j])
        # Positions past the prompt are fed the *previous* answer token: the
        # input is `prompt + answer[:-1]`, so the model never sees answer[j].
        fed = (ids[pos_pre] if pos_pre < L0
               else ans_ids[pos_pre - L0])
        fed_s = "%r" % tok.decode([fed])
        tgt_s = "%r (id %d)" % (tok.decode([ans_ids[j]]), ans_ids[j])
        print("  %-8d %-8d %-26s %-16s" % (pos_pre, pos_spl, fed_s, tgt_s))
    print("  (last prompt token is %r; everything after the placeholder "
          "shifts right by %d)"
          % (tok.decode([ids[L0 - 1]]), model.n_soft - 1))


def show_soft(model, enc_batch, tok, emb_rms):
    with torch.no_grad():
        h = model.encoder._encode(enc_batch)
        soft = model.projector(h, enc_batch["pad_mask"])
    pad = enc_batch["pad_mask"][0]
    print("\nencoder output h      %s  dtype %s"
          % (tuple(h.shape), h.dtype))
    print("projector output      %s  dtype %s   (d_llm = %d)"
          % (tuple(soft.shape), soft.dtype, soft.shape[-1]))
    print("log_scale             %.4f  (calibrated to the embedding RMS)"
          % float(model.projector.log_scale.detach().exp()))
    print("embedding-table RMS   %.4f ; soft-token RMS %.4f"
          % (emb_rms, float(soft[0][~pad].float().pow(2).mean().sqrt())))
    print("slot  live/pad   h RMS   soft RMS")
    for k in range(model.n_soft):
        print("  %2d    %-5s    %6.3f   %6.3f"
              % (k, "PAD" if bool(pad[k]) else "live",
                 float(h[0, k].float().pow(2).mean().sqrt()),
                 float(soft[0, k].float().pow(2).mean().sqrt())))


def show_topk(logp_row, model, tok, names, ill_champ=None, k=8):
    """Top-k over the full 248,320 vocabulary at one scored position.

    `ill_champ` [C] is taken|fearless-unavailable for this row, so a candidate
    token can be labelled with the champions it stands for *that are still
    available*. That is the fearless-ban question in one line: free generation
    has no -inf mask, so the model can name a champion that is already gone.
    """
    probs = logp_row.float().exp()
    top = probs.topk(k)
    for rank, (pid, p) in enumerate(zip(top.indices.tolist(),
                                        top.values.tolist())):
        s = tok.decode([pid])
        on = bool(model.cand_mask[pid])
        owners = ""
        if on:
            pos = int(torch.searchsorted(
                model.cand_first,
                torch.tensor([pid], device=model.cand_first.device)).item())
            champs = model.cand_owner[pos].nonzero().flatten().tolist()
            if ill_champ is None:
                owners = "  -> " + "/".join(names[c] for c in champs)
            else:
                live = [names[c] for c in champs if not bool(ill_champ[c])]
                dead = [names[c] for c in champs if bool(ill_champ[c])]
                owners = ("  -> " + ("/".join(live) if live
                                     else "ALL GONE (" + "/".join(dead) + ")"))
        print("   %2d  id %-7d %-10r p %.4f %s%s"
              % (rank + 1, pid, s, p, "name-prefix" if on else "OFF-LIST ",
                 owners))


@torch.no_grad()
def readout(model, enc_batch, prompt, tok, names, k=8):
    """Teacher-forced top-k per scored position + the greedy name."""
    out = model(enc_batch, prompt)
    soft = model.soft_tokens(enc_batch)
    E = model.inputs_embeds(prompt, soft)
    hidden = model.trunk(inputs_embeds=E,
                         attention_mask=model.spliced_mask(prompt),
                         use_cache=False).last_hidden_state
    pos = splice_positions(prompt, model.n_soft)
    sel = hidden.gather(1, pos.clamp(min=0).unsqueeze(-1)
                        .expand(-1, -1, hidden.shape[-1]))
    logits = model.llm.get_output_embeddings()(sel)
    logp = logits.float().log_softmax(dim=-1)
    return out, logp


@torch.no_grad()
def generate(model, enc_batch, prompt, tok, n_new=8):
    """Greedy decode from the prompt *without* the answer in it."""
    soft = model.soft_tokens(enc_batch)
    ids, p = prompt["ids"], prompt["ph_idx"]
    emb = model.trunk.embed_tokens
    E = torch.cat([emb(ids[:, :p]), soft, emb(ids[:, p + 1:])], dim=1)
    attn = model.spliced_mask(prompt)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    got = []
    for _ in range(n_new):
        h = model.trunk(inputs_embeds=E, attention_mask=attn,
                        use_cache=False).last_hidden_state
        nid = int(model.llm.get_output_embeddings()(h[:, -1:])[0, -1].argmax())
        if nid == eos:
            got.append(("<|im_end|>", nid))
            break
        got.append((tok.decode([nid]), nid))
        E = torch.cat([E, emb(torch.tensor([[nid]], device=E.device))], dim=1)
        attn = torch.cat([attn, attn.new_ones((1, 1))], dim=1)
    return got


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    rows = C.load_rows(args.corpus)
    split = C.split_rows(rows)
    keys, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    tr = D.train_dataset(rows, split["train"], patch_index, seed=0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ntok = [len(tokenize_answer(tok, C.ANSWER_PREFIX + n)) for n in names]

    pool_idx = [i for i, r in enumerate(split["train"])
                if rows[r].get("unavailable")]
    idx, s = find_sample(tr, pool_idx, -1 if args.any_move else args.move,
                         ntok=ntok, want_pool=True, limit=args.scan)
    s2 = tr[(idx + 1) % len(tr)]
    batch, samples = D.collate([s, s2])

    rule("1. THE ROW AS THE ENCODER SEES IT  (train split, index %d)" % idx)
    show_row(s, names)

    rule("2. THE ROW AS THE LLM SEES IT")
    prefix, body, ans, ids, ph, ans_ids = show_prompt(s, names, tok,
                                                      args.context,
                                                      args.prompt_format)
    print("\n--- the same text with the soft block drawn in ---")
    print((prefix + "\u27e6SOFT x %d\u27e7" % args.n_soft + body
           ).replace("<|im_start|>", "\n<|im_start|>")
          .replace("<|im_end|>", "<|im_end|>\n"))

    rule("3. THE BATCH (batch_size 2, as the warmup ran)")
    for k in ("champ_ids", "role_ids", "pad_mask", "ban_ids", "patch_id",
              "taken_mask", "unavailable_mask", "target_champ"):
        print("  %-18s %-16s %s" % (k, tuple(batch[k].shape), batch[k].dtype))
    prompt2 = assemble(tok, samples, names, context=args.context,
                       device=device, prompt_format=args.prompt_format)
    for k in ("ids", "attn", "ans_pos_pre", "ans_ids", "n_ans"):
        print("  prompt[%-12s] %-16s %s"
              % (k, tuple(prompt2[k].shape), prompt2[k].dtype))
    print("  ph_idx = %d (one int, constant across the batch - that is what "
          "makes the splice a single cat)" % prompt2["ph_idx"])
    print("  row 0 length %d, row 1 length %d -> the batch is padded to %d"
          % (int(prompt2["attn"][0].sum()), int(prompt2["attn"][1].sum()),
             prompt2["ids"].shape[1]))
    print("  sections 5 and 6 below score row 0 on its own, unpadded")

    rule("4. MODEL")
    enc, cfg, _ = C.build_frozen_encoder(args.encoder, device=device)
    C.assert_patch_compat(cfg, patch_index)
    llm = B.load_llm(args.model, quantize=True, device=device.index or 0)
    B.freeze(llm)
    d_llm = llm.get_input_embeddings().weight.shape[1]
    model = B.build_draft_llm(enc, llm, tok, names, cfg["embed_dim"], d_llm,
                              args.hidden, n_soft=args.n_soft).to(device)
    model.eval()
    print("  d_in %d -> hidden %d -> d_llm %d, %d soft tokens"
          % (cfg["embed_dim"], args.hidden, d_llm, model.n_soft))
    print("  projector params %d, encoder params %d (frozen), LLM %s"
          % (C.param_count(model.projector), C.param_count(model.encoder),
             "4-bit frozen"))
    emb_rms = float(model.calibration["embed_rms"])

    enc_batch = D.to_device(D.collate([s])[0], device)
    prompt = assemble(tok, [s], names, context=args.context, device=device,
                      prompt_format=args.prompt_format)
    prompt_gen = assemble(tok, [s], names, context=args.context,
                          with_answers=False, device=device,
                          prompt_format=args.prompt_format)

    for stage, load in (("A. UNTRAINED projector (step 0)", False),
                        ("B. TRAINED projector (%s)"
                         % Path(args.projector).parent.name, True)):
        if load:
            blob = B.load_projector(model, args.projector)
            model.eval()
            rule("6. THE SAME ROW AFTER WARMUP  (%s)" % stage)
            print("  loaded %s  (n_soft %d, calibration %s)"
                  % (Path(args.projector).name, blob["n_soft"],
                     {k: round(v, 4) for k, v in blob["calibration"].items()}))
        else:
            rule("5. THE SAME ROW BEFORE TRAINING  (%s)" % stage)

        show_soft(model, enc_batch, tok, emb_rms)
        if not load:
            show_alignment(tok, s, names, ids, ph, ans_ids, model, prompt)

        out, logp = readout(model, enc_batch, prompt, tok, names, args.topk)
        print("\n  loss %.4f   ce %.4f   tokens %d"
              % (float(out["loss"]), out["ce"], out["n_tokens"]))
        print("  exact %d   ft_top1 %d   ft_rank %.0f   off-list %d"
              % (int(out["exact"]), int(out["ft_top1"]), out["ft_rank"],
                 int(out["top1_offlist"])))

        print("\n  top-%d over the full vocabulary at position 0 "
              "(the token right after 'Answer:'):" % args.topk)
        ill_champ = (enc_batch["taken_mask"][0]
                     | enc_batch["unavailable_mask"][0])
        show_topk(logp[0, 0], model, tok, names, ill_champ, args.topk)
        print("  (%d of the 136 name-prefix candidates still name an available "
              "champion for this row)" % int((~model._cand_illegal(enc_batch)
                                              [0]).sum()))

        print("\n  argmax at each scored position (teacher-forced readout):")
        greedy = []
        for j in range(int(prompt["n_ans"][0])):
            lp = logp[0, j]
            pid = int(lp.argmax())
            greedy.append(pid)
            print("   j=%d  %-12r p %.4f   %s   target %r"
                  % (j, tok.decode([pid]), float(lp[pid].exp()),
                     "HIT " if pid == int(prompt["ans_ids"][0, j]) else "miss",
                     tok.decode([int(prompt["ans_ids"][0, j])])))
        print("  readout name: %r   (truth %r)"
              % (tok.decode(greedy), names[int(s["target_champ"])]))

        print("\n  free-running greedy generation (prompt only, no answer "
              "tokens in the input):")
        got = generate(model, enc_batch, prompt_gen, tok, args.n_gen)
        print("    " + "".join(t for t, _ in got).strip())
        print("    tokens: %s" % " ".join("%r/%d" % (t, i) for t, i in got))


if __name__ == "__main__":
    main()
