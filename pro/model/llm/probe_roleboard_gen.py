"""Zero-shot role board from the LLM: does it need the masked checkpoint?

The role board is 10 cells - five lanes for each team - and the masked model
answers all of them in ~11 ms. This asks whether the LLM can answer them with no
shortlist at all, so the frontend could ship one model instead of two.

Two readouts, because they fail differently:

  gen     the v2 prompt (the one the base LLM was trained to answer), greedy
          decode, first name parsed off the front. One answer per cell.
  ro      one forward pass, then the model's own distribution over the 173
          names at the "Answer:" position. Top-5 first *tokens*, expanded to
          names and ranked by whole-name log-prob - the `ft_top*` readout, with
          no shortlist and no decoding.

Both are scored against the champion that team actually fielded in that lane,
on held-out games, at a fixed board (10 bans + 6 picks).

    python pro/model/llm/probe_roleboard_gen.py 80 ro  llm-lora-rerank5
    python pro/model/llm/probe_roleboard_gen.py 40 gen llm-lora-full2

Result (2026-11-10, 80 games): masked board 40.6 % top-5 / 9.4 % top-1 in 11 ms;
this 29.4-34.3 % / 8.7-9.4 % in 0.9-1.3 s, and generation 7.2-7.9 % top-1 with a
single answer. Not a replacement - see frontend/README.md.
"""
import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]   # pro/model/llm -> repo
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "propipeline", ROOT / "data" / "pro play" / "pipeline.py")
P = importlib.util.module_from_spec(spec)
sys.modules["propipeline"] = P
spec.loader.exec_module(P)

import rerank as RR                                # noqa: E402
from pro.model.llm import board as BOARD           # noqa: E402
from pro.model.llm import common as C              # noqa: E402
from pro.model.llm import data as D                # noqa: E402
from pro.model.llm.prompt import assemble          # noqa: E402
from solo_queue.model.policy import lanes as L     # noqa: E402
from solo_queue.model.policy.infer import MOVE_PICK   # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_BANS, N_PICKS = 10, 6
MODE = sys.argv[2] if len(sys.argv) > 2 else "both"

NAME = sys.argv[3] if len(sys.argv) > 3 else "llm-lora-rerank5"
if NAME != "llm-lora-rerank5":
    # the v2-trained adapter was never taught the candidate line; letting the
    # loader through is the whole point of the comparison
    RR.PROMPT_FORMAT = "v2"
pred = RR.RerankPredictor(ROOT / "pro/model/checkpoints" / NAME, device="cuda")
model, tok, names = pred.model, pred.tok, pred.names
eos_id = tok.convert_tokens_to_ids("<|im_end|>")

BY_LEN = sorted(((n.lower(), i) for i, n in enumerate(names)),
                key=lambda p: -len(p[0]))


def parse_name(text):
    """Longest-prefix match: 'Ambessa Velma (Top' -> Ambessa, 'Ashe.' -> Ashe."""
    t = text.lower().lstrip()
    for name, idx in BY_LEN:
        if t.startswith(name):
            rest = t[len(name):len(name) + 1]
            if not rest or not rest.isalpha():
                return idx
    return None


def batch_prompts(samples, fmt=None):
    enc, _ = D.collate(samples)
    enc = D.to_device(enc, pred.device)
    prompt = assemble(tok, samples, names, context=pred.context,
                      with_answers=False,
                      prompt_format=fmt or "v2")
    prompt = {k: (v.to(pred.device) if torch.is_tensor(v) else v)
              for k, v in prompt.items()}
    return enc, prompt


@torch.no_grad()
def readout(samples):
    """-> [B, V] log-probs at the position that predicts the first answer token."""
    enc, prompt = batch_prompts(samples)
    soft = model.soft_tokens(enc)
    emb = model.inputs_embeds(prompt, soft)
    attn = model.spliced_mask(prompt)
    logits = model.llm(inputs_embeds=emb, attention_mask=attn).logits
    last = attn.sum(1) - 1
    rows = logits[torch.arange(len(samples), device=logits.device), last]
    return rows.float().log_softmax(-1)


@torch.no_grad()
def generate_one(sample):
    """One row, no padding: pad_token_id == <|im_end|> here, so a batch stops
    on its own first pad. 0.3 s each, which is the honest cost."""
    enc, prompt = batch_prompts([sample], fmt="v2")   # the generation prompt
    soft = model.soft_tokens(enc)
    emb = model.inputs_embeds(prompt, soft)
    attn = model.spliced_mask(prompt)
    out = model.llm.generate(inputs_embeds=emb, attention_mask=attn,
                             max_new_tokens=6, do_sample=False, num_beams=1,
                             eos_token_id=eos_id)
    # with inputs_embeds the output holds *only* the new tokens
    return tok.decode(out[0], skip_special_tokens=True).strip()


first_ids = [int(x) for x in model.cand_first.tolist()]
owner = model.cand_owner                      # [136, 173] bool
rows, _ = P.build_rows()

stat = {"gen_top1": [0, 0], "gen_off": 0, "ro_top1": [0, 0],
        "ro_g5": [0, 0], "ro_g1": [0, 0], "ro_n5": [0, 0],
        "ro_n5b": [0, 0], "ro_t1b": [0, 0]}
t_gen = t_ro = 0.0
shown = 0

for r in rows[-N:]:
    bans, picks, per = [], [None] * 10, [0, 0]
    true = [[None] * 5, [None] * 5]
    for kind, team, champ, role in r["moves"]:
        if kind == MOVE_PICK:
            true[team][role] = int(champ)
        elif len(bans) < N_BANS:
            bans.append(int(champ))
    placed = [m for m in r["moves"] if m[0] == MOVE_PICK][:N_PICKS]
    if len(placed) < N_PICKS or any(c is None for t in true for c in t):
        continue
    for _k, team, champ, role in placed:
        picks[team * 5 + per[team]] = int(champ)
        per[team] += 1
    lanes = L.assign_lanes(picks)
    move_idx = len(bans) + len(placed)
    taken = set(bans) | {int(m[2]) for m in placed}

    names_by_cell = []
    cells = [(team, lane) for team in (0, 1)
             for lane in BOARD.remaining_lanes(lanes, team)
             if true[team][lane] not in taken]
    if not cells:
        continue
    samples = [BOARD.sample_from_board(
        picks, lanes, bans, int(r["first_pick"]), move_idx, r.get("patch"),
        MOVE_PICK, team, target_role=lane, target_champ=0,
        patch_index=pred.patch_index) for team, lane in cells]

    if MODE in ("both", "gen"):
        t0 = time.time()
        texts = [generate_one(x) for x in samples]
        t_gen += time.time() - t0
        for (team, lane), text in zip(cells, texts):
            got = parse_name(text)
            stat["gen_top1"][1] += 1
            stat["gen_top1"][0] += 1 if got == true[team][lane] else 0
            stat["gen_off"] += 1 if got is None else 0
            if shown < 6:
                print("  gen %-12s -> %-28r %s"
                      % (C.ROLE_DISPLAY[lane], text[:28],
                         names[got] if got is not None else "NO MATCH"))
                shown += 1

    if MODE in ("both", "ro"):
        t0 = time.time()
        lp = readout(samples)
        t_ro += time.time() - t0
        for i, (team, lane) in enumerate(cells):
            target = true[team][lane]
            order = lp[i, first_ids].argsort(descending=True)
            top5 = order[:5].tolist()
            tgt_first = next((g for g, fid in enumerate(first_ids)
                              if owner[g, target]), None)
            stat["ro_g5"][1] += 1
            stat["ro_g5"][0] += 1 if tgt_first in top5 else 0
            stat["ro_g1"][1] += 1
            stat["ro_g1"][0] += 1 if tgt_first == top5[0] else 0
            names_by_cell.append((target, [c for g in top5
                                           for c in range(len(names))
                                           if owner[g, c]],
                                  (picks, lanes, bans, int(r["first_pick"]),
                                   move_idx, r.get("patch"), team, lane)))

    # stage 2: rank the names inside those groups by the whole-name log-prob,
    # exactly as rerank.pick_winner does (first token, then the full name)
    flat, owner_of = [], []
    for ci, (target, cands, ctx) in enumerate(names_by_cell):
        for c in cands:
            flat.append(BOARD.sample_from_board(
                ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5], MOVE_PICK,
                ctx[6], target_role=ctx[7], target_champ=c,
                patch_index=pred.patch_index))
            owner_of.append(ci)
    if flat:
        t0 = time.time()
        enc2, prompt2 = batch_prompts(flat)
        sc = model.answer_scores(enc2, prompt2)
        first = sc["first_logp"].tolist()
        total = sc["total_logp"].tolist()
        ntok = sc["n_tok"].tolist()
        for ci in range(len(names_by_cell)):
            idx = [j for j, o in enumerate(owner_of) if o == ci]
            order = sorted(idx, key=lambda j: (-first[j], -total[j]))
            target = names_by_cell[ci][0]
            stat["ro_top1"][1] += 1
            stat["ro_top1"][0] += 1 if flat[order[0]]["target_champ"] == \
                target else 0
            top5names = [flat[j]["target_champ"] for j in order[:5]]
            stat["ro_n5"][1] += 1
            stat["ro_n5"][0] += 1 if target in top5names else 0
            # length-normalised: total_logp / tokens. A 1-token name ("Vi")
            # otherwise always beats a 2-token one ("Viego") inside a group.
            order2 = sorted(idx, key=lambda j: (-first[j], -total[j] / max(ntok[j], 1)))
            top5b = [flat[j]["target_champ"] for j in order2[:5]]
            stat["ro_n5b"][1] += 1
            stat["ro_n5b"][0] += 1 if target in top5b else 0
            stat["ro_t1b"][1] += 1
            stat["ro_t1b"][0] += 1 if flat[order2[0]]["target_champ"] == \
                target else 0
        t_ro += time.time() - t0

print("\n%s | %d games, state = %d bans + %d picks" % (NAME, N, N_BANS, N_PICKS))
if stat["gen_top1"][1]:
    n = stat["gen_top1"][1]
    print("generation  top-1 %5.1f %%  | not a name %4.1f %% | %.2f s/batch"
          % (100 * stat["gen_top1"][0] / n, 100 * stat["gen_off"] / n,
             t_gen / max(n, 1)))
if stat["ro_top1"][1]:
    n = stat["ro_top1"][1]
    print("readout     top-1 %5.1f %%  (two-stage: group, then whole-name)"
          % (100 * stat["ro_top1"][0] / n))
    print("            name top-5 %5.1f %% (len-normalised %5.1f %%) | "
          "top-1 %5.1f %% (len-norm %5.1f %%)"
          % (100 * stat["ro_n5"][0] / max(stat["ro_n5"][1], 1),
             100 * stat["ro_n5b"][0] / max(stat["ro_n5b"][1], 1),
             100 * stat["ro_top1"][0] / n,
             100 * stat["ro_t1b"][0] / max(stat["ro_t1b"][1], 1)))
    print("            group recall@5 %5.1f %% | argmax-group hit %5.1f %% | "
          "%.2f s/board"
          % (100 * stat["ro_g5"][0] / n,
             100 * stat["ro_g1"][0] / max(stat["ro_g1"][1], 1),
             t_ro / max(n, 1)))
