"""Prompt construction and soft-token splicing.

The LLM sees three things per example:

  [constant chat prefix]  [10 soft tokens]  [board text + task + answer]

The **constant** part is what makes the splice cheap: because the soft block
sits at a fixed position (the first thing in the user turn), every row of a
batch has its placeholder at the same token index, so

    emb(ids[:, :p])  ++  soft[B, 10, d]  ++  emb(ids[:, p+1:])

is a single `torch.cat` with no per-row bookkeeping and no position ids to
fix up. That is the only reason the placeholder is at the top of the turn
rather than inline before the task.

Two decisions worth stating, because they are measurement decisions:

**The board is rendered as text as well as soft tokens.** The user's sketch
put `[DRAFT_TOKENS]` in a prompt with no board. That asks a frozen 4B model to
recover entity identity ("which champion is in which slot", "is Azir already
banned") from a 10-vector continuous channel, while a projector that has seen
a few thousand examples tries to teach it - and nothing stops the model from
suggesting a champion that is already taken, because there is no legality mask
in free generation. Rendering the board as text gives the model the *facts*
natively and leaves the soft tokens to carry the encoder's *judgement*, which
is the only part text cannot express. `--context soft` turns the text off so
the pure-channel variant stays measurable.

**The target is the champion name, exactly as asked.** Worth knowing before
the run: only 22 of the 173 names are a single token under this tokenizer, and
there are 136 distinct first tokens (24 collision groups covering 61 names, the
largest standing for 6 champions), so "a 1:1 mapping to discrete entity tokens"
is not what the tokenizer offers - the target is a 1-5 token sequence. the
metrics therefore report both per-token accuracy and whole-name exact match, and
`first_token_rank` is reported as a 136-way number and never compared to the
policy head's 173-way one.
"""
import torch

from pro.model.llm.common import (ANSWER_PREFIX, IGNORE_INDEX, PHASE_NAMES,
                              render_state, task_line)

SYSTEM = "You are an expert League of Legends draft engine."

PREFIX = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
          "<|im_start|>user\n")
ANSWER_CUE = "Answer:"
EOS = "<|im_end|>"


def build_text(sample, names, context="soft+text", answer=None,
               extra_instruction=None, prompt_format=None, candidates=None):
    """-> (pre_placeholder_text, post_placeholder_text, answer_text).

    `answer_text` is empty when `answer` is None (the generation prompt).
    Splitting the three pieces lets the caller tokenize them separately, which
    is what guarantees every target token is a whole token rather than one
    straddling the prompt/answer boundary.

    `prompt_format` selects the board rendering (`common.render_state`): `v2`
    is the default and labels the turn and the lane; `v1` reproduces the text
    every number recorded before 2026-11-08 was measured on; `v4` adds the
    candidate list and needs `candidates` = [(champ_id, prob), ...] for *this*
    row, which the caller gets from the next-pick head.
    """
    if context == "soft":
        state = ""
    else:
        state = render_state(sample, names, prompt_format=prompt_format,
                             candidates=candidates)
    parts = []
    if state:
        parts.append(state)
    if extra_instruction:
        parts.append(extra_instruction)
    parts.append(task_line(sample, prompt_format=prompt_format))
    body = "\n".join(parts) + "\n" + ANSWER_CUE
    ans_text = "" if answer is None else ANSWER_PREFIX + answer
    return PREFIX, body, ans_text


def tokenize_prompt(tok, prefix, body):
    """Prefix + one placeholder id + body -> (ids [L], placeholder index).

    The placeholder is *inserted by index*, not written into the text and
    searched for afterwards. A text marker ("<|draft_state|>") is not in the
    Qwen vocabulary and the BPE splits it across neighbours, which would put
    the soft block inside a word; building the id list directly makes that
    impossible and costs the same. The prefix is tokenized separately from the
    body, so the only boundary the tokenizer cannot merge across is the one
    where the soft tokens go anyway.
    """
    ph = tok.convert_tokens_to_ids("<|im_end|>")   # any fixed, single id
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    bod = tok(body, add_special_tokens=False)["input_ids"]
    return pre + [ph] + bod, len(pre)


def tokenize_answer(tok, answer_text):
    """' Vi' -> its ids. Separate call, so the answer is never merged with ':'."""
    return tok(answer_text, add_special_tokens=False)["input_ids"]


def assemble(tok, samples, names, context="soft+text", with_answers=True,
             loss_on_eos=False, extra_instruction=None, device=None,
             prompt_format=None, candidates=None):
    """Batch -> token ids, the placeholder index and the scored positions.

    `names` is the champion name table; the per-row answer is looked up here
    from `target_champ`, so a caller cannot pair a row with another row's
    champion.

    "Scored" is defined by `ans_ids`, not by a -100 mask over the whole
    sequence: the model computes logits only at these positions (the vocabulary
    is 248k wide, so a full-sequence logit tensor is the largest allocation in
    the step), and the loss is exactly the cross-entropy over them. Everything
    else - system text, board, task line, EOS - is context and never enters the
    loss, which is what the brief asked for.

    `loss_on_eos` appends the `<|im_end|>` to the scored positions so the model
    also learns to stop. Off by default: an untrained projector will happily
    spend its gradient on the stop token instead of the name.

    Returns dict:
      ids         [B, L]  prompt + answer[:-1], i.e. the model's input; the
                          answer itself is never in it
      attn        [B, L]  1 for real tokens, 0 for right padding
      ph_idx      int     index of the placeholder (constant across the batch)
      ans_pos_pre [B, K]  *predicting* positions in this (pre-splice) sequence,
                          one before each answer token; -1 pad
      ans_ids     [B, K]  the scored token ids, -1 pad
      n_ans       [B]     how many scored tokens each row really has
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.convert_tokens_to_ids(EOS)
    eos_ids = tok(EOS, add_special_tokens=False)["input_ids"]

    seqs, scored, n_ans, starts, ph_idx = [], [], [], [], None
    for i, s in enumerate(samples):
        ans_name = names[int(s["target_champ"])] if with_answers else None
        prefix, body, ans = build_text(s, names, context=context,
                                       answer=ans_name,
                                       extra_instruction=extra_instruction,
                                       prompt_format=prompt_format,
                                       candidates=(candidates[i]
                                                   if candidates else None))
        ids, p = tokenize_prompt(tok, prefix, body)
        ph_idx = p
        tail = []
        if with_answers:
            tail = tokenize_answer(tok, ans)
            if not tail:
                raise RuntimeError("empty answer for %r" % (ans_name,))
        starts.append(len(ids))                 # first scored position
        # The input must stop *before* each target: position `len(ids)+j-1`
        # predicts answer token j, so the last prompt token predicts answer[0].
        # Appending the answer to the input (`ids + tail`) instead makes the
        # task a copy of the label - measured by corrupting the target: the
        # prediction follows the corrupted name, so every number it produces
        # is a copying score, not a prediction score.
        target = tail + (eos_ids if (with_answers and loss_on_eos) else [])
        scored.append(target)
        n_ans.append(len(target))
        seqs.append((ids + tail + eos_ids)[:len(ids) + len(target)])

    L = max(len(x) for x in seqs)
    B = len(seqs)
    K = max(1, max(n_ans))
    ids_t = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    apos = torch.full((B, K), -1, dtype=torch.long)
    aid = torch.full((B, K), -1, dtype=torch.long)
    for i, ids in enumerate(seqs):
        ids_t[i, :len(ids)] = torch.tensor(ids)
        attn[i, :len(ids)] = 1
        for j, tok_id in enumerate(scored[i]):
            apos[i, j] = starts[i] + j - 1
            aid[i, j] = tok_id
    out = {"ids": ids_t, "attn": attn, "ph_idx": ph_idx,
           "ans_pos_pre": apos, "ans_ids": aid,
           "n_ans": torch.tensor(n_ans, dtype=torch.long)}
    if device is not None:
        out = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in out.items()}
    return out


def candidate_first_ids(tok, names, candidates):
    """Per row: the first-token ids of its candidate names -> [B, k], -1 pad.

    The reranker's answer is a champion *name*, so constraining it to the list
    means constraining its **first token** to the candidates' first tokens.
    Collisions make this a constraint over groups rather than over names: 24
    first-token groups cover 61 of the 173 champions, so two candidates can
    share a first token. That is harmless - both are legal answers and the
    following tokens separate them - but it means the mask's width is at most
    k and sometimes less, so callers must not assume k distinct columns.

    `-1` marks padding, not "no candidate": every row in a reranker batch has
    the same k candidates, so the padding only appears when k exceeds the
    longest list.
    """
    rows = []
    for cand in candidates:
        ids = [tok(ANSWER_PREFIX + names[int(c)], add_special_tokens=False)
               ["input_ids"][0] for c, _p in cand]
        rows.append(ids)
    k = max(1, max(len(r) for r in rows)) if rows else 1
    out = torch.full((len(rows), k), -1, dtype=torch.long)
    for i, ids in enumerate(rows):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def splice_positions(batch, n_soft):
    """Pre-splice answer positions -> positions in the spliced sequence.

    Everything at or after the placeholder shifts right by `n_soft - 1`.
    """
    apos = batch["ans_pos_pre"]
    ph = batch["ph_idx"]
    shifted = apos.clone()
    live = apos >= 0
    shifted[live] = apos[live] + (n_soft - 1)
    return shifted
