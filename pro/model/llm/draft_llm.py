"""encoder -> projector -> Qwen: the model this experiment trains.

    10 slot vectors h [B,10,256]        (frozen set-transformer encoder)
        -> LayerNorm -> MLP -> [B,10,2560]
        -> spliced into Qwen's input embeddings at the placeholder position
        -> frozen (or LoRA-adapted) Qwen3.5-4B
        -> logits at the answer positions only

Why the logits are gathered instead of returned for the whole sequence: the
vocabulary is 248,320, so a [B, L, V] tensor is ~100 MB per row and would be
the single largest allocation in the step. Only the 1-5 answer positions are
ever read, so the LM head is applied to a gathered [B, K, d] slice. That is
the difference between fitting in 8 GB and not.

The loss is computed here rather than by passing `labels=` to the model for
the same reason, and because the metrics (per-token accuracy, whole-name exact
match, first-token rank) all need the logits anyway.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pro.model.llm.prompt import splice_positions

N_SLOTS = 10


class SoftPromptProjector(nn.Module):
    """Slot vectors -> LLM-width soft tokens, with a learned slot for PAD.

    A PAD slot is not "a slot whose vector is zero": the encoder's PAD slots
    still carry attention context, so projecting them would inject eight
    meaningless-but-confident tokens into an early-draft prompt where only two
    slots are real. They are replaced by one shared learned vector instead, so
    the model can learn to ignore them.

    `log_scale` starts at whatever makes the projected tokens' RMS match the
    token embedding table's. An untrained MLP whose output norm is 20x the
    text embeddings is the usual reason a soft prompt trains for a few hundred
    steps and then plateaus: every layer's LayerNorm sees an out-of-range
    input. One scalar fixes the scale and stays learnable, so the model can
    still decide otherwise.
    """

    def __init__(self, d_in=256, d_llm=2560, hidden=1024, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_llm)
        self.drop = nn.Dropout(dropout)
        self.pad_token = nn.Parameter(torch.zeros(d_llm))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h, pad_mask):
        """h [B,10,D], pad_mask [B,10] True = PAD -> [B,10,d_llm]."""
        x = self.fc2(self.drop(F.gelu(self.fc1(self.norm(h)))))
        x = x * self.log_scale.exp()
        pad = self.pad_token.to(x.dtype).view(1, 1, -1)
        return torch.where(pad_mask.unsqueeze(-1), pad.expand_as(x), x)

    @torch.no_grad()
    def calibrate(self, embed_weight, n_sample=4096):
        """Set log_scale so the projected RMS matches the text embeddings'.

        The projector may still be on the CPU when this runs (it is built
        before `.to(device)`), so the probe batch goes on the *projector's*
        device and only the scalar target is read off the embedding table.
        """
        idx = torch.randint(0, embed_weight.shape[0], (n_sample,),
                            device=embed_weight.device)
        target = float(embed_weight[idx].float().pow(2).mean().sqrt())
        dev = next(self.parameters()).device
        h = torch.randn(64, N_SLOTS, self.norm.normalized_shape[0], device=dev)
        out = self.fc2(F.gelu(self.fc1(self.norm(h)))).float()
        cur = float(out.pow(2).mean().sqrt().clamp_min(1e-6))
        self.log_scale.fill_(math.log(target / cur))
        return target, cur, float(self.log_scale.exp())


def _trunk(llm):
    """The inner transformer (the object holding `embed_tokens`).

    A breadth-first walk rather than a fixed attribute path, because the depth
    differs between the stages: a plain `Qwen3_5ForCausalLM` is `llm.model`,
    while a peft model is `llm.base_model.model.model`. Hard-coding either
    path works in one stage and raises in the other.
    """
    queue, seen = [llm], set()
    while queue:
        m = queue.pop(0)
        if id(m) in seen:
            continue
        seen.add(id(m))
        if hasattr(m, "embed_tokens"):
            return m
        for attr in ("model", "base_model"):
            nxt = getattr(m, attr, None)
            if nxt is not None and id(nxt) not in seen:
                queue.append(nxt)
    raise RuntimeError("cannot find the transformer trunk under %s"
                       % type(llm).__name__)


class DraftLLM(nn.Module):
    """Frozen encoder + trainable projector + (frozen or LoRA) LLM.

    `cand_first` [M] are the distinct first tokens of the 173 champion names
    and `cand_owner` [M, C] says which champions each one stands for. Both are
    derived from the tokenizer and the name table, never hand-written, so the
    metric cannot drift from what the model is actually asked to emit.

    `cand_mask` [V] is the same set as a boolean over the vocabulary, because
    `ft_top*` renormalises over the M candidates and therefore cannot see the
    thing a free decoder would emit first: a token that is not a name prefix at
    all. `top1_offlist` is that fraction, and it is the honest gap between the
    teacher-forced readout and "what would this model actually say".
    """

    def __init__(self, encoder, projector, llm, cand_first, cand_owner,
                 n_soft=N_SLOTS):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.n_soft = n_soft
        self.register_buffer("cand_first", cand_first, persistent=False)
        self.register_buffer("cand_owner", cand_owner, persistent=False)
        mask = torch.zeros(int(llm.get_input_embeddings().weight.shape[0]),
                           dtype=torch.bool)
        mask[cand_first] = True
        self.register_buffer("cand_mask", mask, persistent=False)
        self.trunk = _trunk(llm)

    # ---------------------------------------------------------------- pieces
    @property
    def dtype(self):
        return self.trunk.embed_tokens.weight.dtype

    def soft_tokens(self, enc_batch):
        """-> [B, n_soft, d_llm] in the LLM's dtype."""
        with torch.no_grad():
            h = self.encoder._encode(enc_batch)
        return self.projector(h, enc_batch["pad_mask"]).to(self.dtype)

    def inputs_embeds(self, prompt, soft):
        emb = self.trunk.embed_tokens
        ids, p = prompt["ids"], prompt["ph_idx"]
        return torch.cat([emb(ids[:, :p]), soft, emb(ids[:, p + 1:])], dim=1)

    def spliced_mask(self, prompt):
        attn, p = prompt["attn"], prompt["ph_idx"]
        B = attn.shape[0]
        return torch.cat([attn[:, :p], attn.new_ones((B, self.n_soft)),
                          attn[:, p + 1:]], dim=1)

    def _answer_logp(self, enc_batch, prompt, keep=None):
        """-> [B, K, V] log-probabilities at the scored answer positions.

        The one place the trunk is run, so training, eval and the reranker's
        candidate scoring cannot drift apart. `keep` is applied to the first
        answer position only (see `forward`).
        """
        soft = self.soft_tokens(enc_batch)
        E = self.inputs_embeds(prompt, soft)
        hidden = self.trunk(inputs_embeds=E,
                            attention_mask=self.spliced_mask(prompt),
                            use_cache=False).last_hidden_state

        pos = splice_positions(prompt, self.n_soft)   # [B,K] spliced positions
        sel = hidden.gather(
            1, pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        logits = self.llm.get_output_embeddings()(sel)   # [B,K,V]
        if keep is not None:
            logits = logits.clone()
            logits[:, 0] = logits[:, 0].masked_fill(~keep, float("-inf"))
        return logits.float().log_softmax(dim=-1)

    @torch.no_grad()
    def answer_scores(self, enc_batch, prompt, keep=None):
        """Per-row score of the answer the prompt carries.

        Serving needs the numbers `forward` throws away: the reranker builds one
        row per candidate, so `first_logp` [B] is that candidate's log-prob at
        the decision point (after `keep`) and `total_logp` [B] its whole name.
        Picking the argmax of `first_logp` over the k rows is exactly what
        `ft_top1` measured, which is why the served number and the reported one
        are the same quantity.
        """
        logp = self._answer_logp(enc_batch, prompt, keep)
        target = prompt["ans_ids"]
        valid = target >= 0
        tok_ll = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return {"first_logp": tok_ll[:, 0],
                "total_logp": (tok_ll * valid).sum(dim=1),
                "n_tok": valid.sum(dim=1)}

    def forward(self, enc_batch, prompt, compute_metrics=True, keep=None):
        """`keep` [B, V] bool is the reranker's decode constraint.

        It masks the **first** answer position to the candidate list's first
        tokens *before* the log-softmax, so the loss is a proper k-way
        cross-entropy over the candidates rather than a full-vocabulary one
        with a smaller argmax set. The remaining answer tokens are untouched:
        they spell out which of the k was chosen, and a first-token collision
        between two candidates is resolved there.

        `keep` must contain the target's first token or the loss is +inf; the
        eval path therefore unions the target in, which keeps the denominator
        intact and counts those rows as wrong - the honest reading, since a
        decoder constrained to a list that lacks the answer cannot be right.
        """
        logp = self._answer_logp(enc_batch, prompt, keep)
        target = prompt["ans_ids"]
        valid = target >= 0
        tok_ll = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        loss = -(tok_ll * valid).sum() / valid.sum().clamp_min(1)

        out = {"loss": loss, "n_tokens": int(valid.sum())}
        if compute_metrics:
            with torch.no_grad():
                out.update(self._metrics(logp, tok_ll, target, valid,
                                         enc_batch))
        return out

    # --------------------------------------------------------------- metrics
    def _metrics(self, logp, tok_ll, target, valid, enc_batch):
        """Teacher-forced numbers, all restricted to the answer positions.

        `ft_top*` is a **136-way** number (the distinct first tokens of the 173
        names), not a 173-way one - see the module docstring of prompt.py.
        `*_legal` drops candidates whose every champion is already taken or
        fearless-unavailable, which is the only way to ask "would it have said
        something legal" without running a decoding loop.
        """
        pred = logp.argmax(dim=-1)                    # [B,K]
        hit = (pred == target) & valid
        # The first token a free decoder would emit, over the full vocabulary.
        # `ft_top*` below renormalises over the 136 name prefixes and so cannot
        # see an off-list argmax; this is the gap between the teacher-forced
        # readout and what the model would actually start saying.
        full0 = logp[:, 0].argmax(dim=-1)             # [B]
        off = ~self.cand_mask[full0]                  # [B]
        out = {
            "ce": float(-(tok_ll * valid).sum() / valid.sum().clamp_min(1)),
            "tok_acc": float(hit.sum().float() / valid.sum().clamp_min(1)),
            "exact": float((hit | ~valid).all(dim=1).float().mean()),
            "top1_offlist": float(off.float().mean()),
        }

        first = logp[:, 0][:, self.cand_first]        # [B,M] over candidates
        # `target[:,0]` is a *vocabulary* id; the candidate tables are indexed
        # by position in `cand_first`. Comparing the two directly silently
        # scores every row as rank 0 (nothing is ever equal), so the mapping
        # is explicit and asserted.
        tgt = self._cand_pos(target[:, 0])            # [B] candidate index
        rank = self._rank(first, tgt)
        out["ft_top1"] = float((rank == 0).float().mean())
        out["ft_top5"] = float((rank < 5).float().mean())
        out["ft_rank"] = float(rank.float().mean())
        # Per-row values so a caller can slice (ban vs pick, per phase) without
        # re-running the model. Small - one scalar per row per key.
        out["row"] = {"exact": (hit | ~valid).all(dim=1).float(),
                      "rank": rank.float(), "n_tok": valid.sum(dim=1),
                      "off": off.float()}

        cand_ill = self._cand_illegal(enc_batch)      # [B,M] bool or None
        if cand_ill is not None:
            # Does it *recommend* something illegal? This is the fearless-ban
            # question in its measurable form: the policy head breaks it zero
            # times because it applies a -inf mask, and free generation has no
            # such mask, so the only defence is whether the model learned the
            # board from the prompt. Counted over all rows (not just the ones
            # where the answer is legal), because a suggestion of an
            # already-taken champion is wrong regardless of the label.
            order = first.argsort(dim=-1, descending=True)
            ill_sorted = cand_ill.gather(1, order)
            out["top1_illegal"] = float(ill_sorted[:, 0].float().mean())
            out["top5_illegal"] = float(
                ill_sorted[:, :5].any(dim=1).float().mean())
            out["row"]["ill1"] = ill_sorted[:, 0].float()
            out["row"]["ill5"] = ill_sorted[:, :5].any(dim=1).float()

            # ---------------------------------------------------------------
            # Decode-time legality. This is what a *decoder* scores, not what
            # the teacher-forced readout scores: a generator that knows the
            # rules masks every candidate whose champions are all taken or
            # fearless-picked, and renormalises over the rest. Scored on ALL
            # rows, unlike `ft_*_legal` below, because a masked decoder cannot
            # emit an illegal champion at all - restricting the denominator to
            # the rows where the label happens to be legal would flatter it.
            #
            # It costs nothing to train for: `ft_top*` are the unmasked
            # numbers, so the delta between them is exactly the free gain from
            # masking, and no weight changes.
            legal_rank = self._rank(first.masked_fill(cand_ill, -1e4), tgt)
            out["ft_top1_masked"] = float((legal_rank == 0).float().mean())
            out["ft_top5_masked"] = float((legal_rank < 5).float().mean())
            out["ft_rank_masked"] = float(legal_rank.float().mean())
            out["row"]["masked_rank"] = legal_rank.float()

            # Only rows whose true answer survives the legality filter can be
            # scored: a champion picked earlier in the series is still legal
            # to *ban* (the corpus has 8 such bans), so masking it would
            # punish the model for the filter, not for the prediction.
            ok = ~cand_ill.gather(1, tgt.unsqueeze(1)).squeeze(1)
            if int(ok.sum()) > 0:
                out["ft_top1_legal"] = float(
                    (legal_rank == 0).float()[ok].mean())
                out["ft_top5_legal"] = float((legal_rank < 5).float()[ok].mean())
                out["legal_n"] = int(ok.sum())
        return out

    def _cand_pos(self, tok_ids):
        """Vocabulary id -> position in `cand_first` (raises if absent)."""
        pos = torch.searchsorted(self.cand_first, tok_ids.contiguous())
        if not bool((self.cand_first[pos.clamp(max=self.cand_first.numel() - 1)]
                     == tok_ids).all()):
            raise RuntimeError("an answer's first token is not a candidate - "
                               "candidate_tables and the answers disagree")
        return pos

    def _rank(self, scores, tgt):
        order = scores.argsort(dim=-1, descending=True)
        return (order == tgt.unsqueeze(1)).float().argmax(dim=1)

    def _cand_illegal(self, enc_batch):
        """[B,M] True where *every* champion behind that first token is
        already gone.

        The rule is the policy head's training rule (`_apply_legality`):
        `taken_mask` (banned or picked this game) is always illegal, and a
        fearless-picked champion is illegal on **pick** moves only - a champion
        picked in an earlier series game can still be banned in this one, and
        the corpus has 8 ban moves whose target is exactly such a champion.
        Masking those would charge the model for the filter, not the
        prediction.

        `infer._policy_batch` is deliberately stricter and passes
        `taken | unavailable` for bans too; that is a serving choice, not the
        rule, so it is not what the decode-time mask here implements.
        """
        taken = enc_batch.get("taken_mask")
        if taken is None:
            return None
        ill = taken
        unavail = enc_batch.get("unavailable_mask")
        if unavail is not None:
            is_pick = ~enc_batch["is_ban"].bool()        # [B] True = pick
            ill = ill | (unavail & is_pick.unsqueeze(1))
        owned = self.cand_owner.float()                  # [M,C]
        n_ill = ill.float() @ owned.T                    # [B,M]
        return n_ill >= owned.sum(dim=1).unsqueeze(0)

    def legal_first_mask(self, enc_batch):
        """[B,M] bool, True where that first token may be emitted.

        This is the mask a decoder applies before argmax/sampling. It is the
        public entry point for it: `_cand_illegal` is kept as the negation
        because the metrics ask the opposite question ("did it recommend
        something gone"), and inverting at one site is cheaper to read than a
        boolean flag threaded through both.
        """
        ill = self._cand_illegal(enc_batch)
        return None if ill is None else ~ill

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
