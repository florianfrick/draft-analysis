"""The finetuned reranker, served as the next-move predictor.

`PolicyPredictor.next_move` returns a distribution over 173 champions. This
class returns the same dict with the same keys, but the `suggestions` list is
the *reranker's* order over the head's shortlist, so every caller - the
next-move panel, `/api/suggest`, the role board, the agent's
`predict_next_move` tool - keeps working unchanged.

Why it exists: on 4,000 held-out states the head's top-1 is 18.32 % and this
pipeline's is 34.27 % (+16.0), because the reranker converts 67.0 % of a
shortlist the head only converts 35.8 % of. The cost is a 4.2 B model in the
loop, so it needs `--device cuda` and the `llm` image (transformers + peft +
bitsandbytes); `webapp/server.py` falls back to the head when either is
missing rather than failing to start.

What it does *not* change: the candidate set. The head proposes, the reranker
orders. So top-5 recall is the head's 51.18 % either way - only the ordering
inside those five improves. That is why this cannot rescue the role board
(see `webapp/README.md`), and why the honest description of the gain is
"picks the right one of five twice as often", not "finds more moves".

The board reaches the prompt as text and as the ten soft tokens, exactly as in
training (`prompt.assemble` + `DraftLLM.soft_tokens`), because a text-only
prompt loses 3.5 pp (`_probe_soft.py`'s `pad` arm) and an adapter fed a
different board's tokens is not the model that was measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pro.model.archive.llm import board as BOARD                     # noqa: E402
from pro.model.archive.llm import build as B                         # noqa: E402
from pro.model.archive.llm import common as C                        # noqa: E402
from pro.model.archive.llm import data as D                          # noqa: E402
from pro.model.archive.llm import rerank as R                        # noqa: E402
from pro.model.archive.llm.prompt import assemble, candidate_first_ids  # noqa: E402
from solo_queue.model.policy import lanes as L               # noqa: E402
from solo_queue.model.policy import infer                    # noqa: E402

MOVE_BAN, MOVE_PICK = 0, 1


class RerankPredictor:
    """`next_move`, answered by the Qwen reranker over the head's shortlist."""

    kind = "rerank"
    # `model_info` in webapp/server.py reads these off every predictor. The
    # reranker has no val run of its own: its number is a held-out *rerank*
    # score (34.27 % exact, STATUS.md §9), not a `champ top5` on a training
    # split, and putting it in the val column would invite exactly the
    # cross-corpus comparison the STATUS note forbids. So they stay None and
    # `metric` names what the number actually is.
    metric = "rerank top1"
    val_top5 = None
    epoch = None

    def __init__(self, ckpt_dir, head_path, device="cuda", n_candidates=20,
                 encoder=C.ENCODER_CKPT, model_id="Qwen/Qwen3.5-4B",
                 hidden=1024, n_soft=10):
        ckpt_dir = Path(ckpt_dir)
        if not (ckpt_dir / "adapter").is_dir():
            raise ValueError("%s has no adapter/ - not a reranker checkpoint"
                             % ckpt_dir)
        self.device = torch.device(device)
        self.n_candidates = int(n_candidates)
        # `label` is what `model_tools.label` and the tool payloads show, so it
        # is the checkpoint's name, not its path.
        self.label = ckpt_dir.name
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = str(ckpt_dir)
        self.head_path = Path(head_path)

        # The head proposes. It stays on CPU: the card is holding the 4B, and
        # 2.5 M parameters are not worth the 300 MB.
        self.head = infer.PolicyPredictor(head_path, device="cpu")

        rows = C.load_rows("2022")
        self.names = C.champ_names()
        self._pi = D.make_patch_index(rows)[1]
        self._patch_vocab = []
        enc, cfg, vocab = C.build_frozen_encoder(encoder, device=self.device)
        self._patch_vocab = list(vocab)
        C.assert_patch_compat(cfg, self._pi)

        llm = B.load_llm(model_id, quantize=True,
                         device=self.device.index or 0)
        B.freeze(llm)
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, str(ckpt_dir / "adapter"))
        llm.eval()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        d_llm = llm.get_input_embeddings().weight.shape[1]
        self.model = B.build_draft_llm(enc, llm, tok, self.names,
                                       cfg["embed_dim"], d_llm, hidden,
                                       n_soft=n_soft).to(self.device)
        B.load_projector(self.model, str(ckpt_dir / "projector.pt"))
        self.model.eval()
        self.tok = tok
        self.n_vocab = int(llm.get_input_embeddings().weight.shape[0])

    # ---------------------------------------------------------------- helpers
    def patch_str(self, patch_id):
        """Encoder patch id -> the corpus patch name, for the prompt's line.

        `next_move` receives an id (that is the head's interface); the text
        wants the name. An out-of-range or UNK id becomes "" so the prompt
        says nothing rather than the wrong patch. A string is passed through:
        `patch_id_for` accepts either, and the agent's tool sends the name.

        Patch ids are **1-based** rows into the vocab - index 0 is reserved for
        UNK (`infer.PATCH_UNK_ID`, `patch_key_for` does `vocab[pid - 1]`). The
        first version of this used `vocab[pid]`, which named the *next* patch
        in every prompt: no error, no warning, just a board from the wrong
        patch.
        """
        if isinstance(patch_id, str):
            return patch_id
        try:
            pid = int(patch_id)
        except (TypeError, ValueError):
            return ""
        if 1 <= pid <= len(self._patch_vocab):
            return str(self._patch_vocab[pid - 1])
        return ""

    def lane_of(self, allowed):
        """`allowed` -> the lane it names, or -100 when it names none.

        The task line says "Choose the best Blue Side pick for Top lane" only
        when the lane is known (`common.task_line`), and every pick row the
        reranker trained on had a lane. The next-move panel asks without one,
        so the served prompt is the no-lane branch - in-distribution for bans,
        a phrase shorter for picks. When the caller *does* constrain to a lane
        (the role board, `next_move(allowed=...)`) the lane is recoverable
        exactly, by matching the allowed set against the lane table, so the
        prompt gets its trained branch back.

        The comparison must use `lanes.legal_champions`, which is what the
        callers pass: `lanes.table()` is a dict of dicts (`roles`, `counts`,
        `share`, `floor`), not `{lane: champions}`, and iterating it compares
        champion sets against the role-name list. That version returned -100 on
        every call, so the role board silently got the no-lane prompt - the
        one thing this method exists to avoid.
        """
        if allowed is None:
            return -100
        try:
            want = {int(x) for x in allowed}
        except (TypeError, ValueError):
            return -100
        for lane in range(5):
            if {int(c) for c in L.legal_champions(lane)} == want:
                return int(lane)
        return -100

    @torch.no_grad()
    def _scores(self, picks, bans, first_pick, move_idx, kind, team, cands,
                patch_id, unavailable, target_role=-100):
        """-> [M] log-prob of each candidate at the first answer position.

        One sample per candidate, batched: `sample_from_board` takes the answer
        as `target_champ`, and `assemble` builds the same board text for every
        row, so the batch is M copies of one prompt with M different answers.
        """
        lanes = L.assign_lanes(picks)
        patch = self.patch_str(patch_id)
        samples = []
        for cand in cands:
            samples.append(BOARD.sample_from_board(
                picks, lanes, bans, int(first_pick), int(move_idx), patch,
                kind, int(team), target_role=int(target_role),
                target_champ=int(cand["idx"]), fearless=tuple(unavailable or ()),
                patch_index=self._pi))
        enc, _ = D.collate(samples)
        enc = D.to_device(enc, self.device)
        cand_pairs = [(int(c["idx"]), float(c["prob"])) for c in cands]
        # `with_answers` must stay True even though we are not scoring a label:
        # the answer *is* the candidate. `assemble` puts the answer text in the
        # input and scores the position before each answer token, so this is a
        # prediction, not a copy - and `ans_ids[:, 0]` is what `keep_mask`
        # unions in, so `with_answers=False` would leave it at -1, mask every
        # candidate out and return -inf for all of them (NaN after softmax).
        prompt = assemble(self.tok, samples, self.names, context="soft+text",
                          with_answers=True, device=self.device,
                          prompt_format="v4", candidates=[cand_pairs] * len(samples))
        prompt = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in prompt.items()}
        keep = R.keep_mask(
            candidate_first_ids(self.tok, self.names, [cand_pairs] *
                                len(samples)).to(self.device),
            prompt["ans_ids"][:, 0], self.n_vocab)
        out = self.model.answer_scores(enc, prompt, keep)
        return out["first_logp"]

    # ------------------------------------------------------------- interface
    @torch.no_grad()
    def next_move(self, picks, bans, first_pick=0, k=8, unavailable=None,
                  patch_id=infer.PATCH_UNK_ID, allowed=None, team=None):
        """The head's answer, re-ordered by the reranker.

        Every key of the returned dict comes from the head unchanged except
        `suggestions`; `roles` in particular is the head's, because the
        reranker has no role head and the role board is not its job.
        """
        n = max(int(k), self.n_candidates)
        base = self.head.next_move(picks, bans, first_pick=first_pick, k=n,
                                   unavailable=unavailable, patch_id=patch_id,
                                   allowed=allowed, team=team)
        if base.get("complete") or not base["suggestions"]:
            return base
        cands = base["suggestions"][:self.n_candidates]
        logp = self._scores(picks, bans, first_pick, base["move_idx"],
                            MOVE_BAN if base["kind"] == "ban" else MOVE_PICK,
                            base["side"], cands, patch_id, unavailable,
                            target_role=self.lane_of(allowed))
        # The reranker's distribution is defined on the shortlist, so it is a
        # *share of the twenty* - not of the legal pool, which is the unit every
        # caller of `next_move` is promised (`model_tools` says so twice, and
        # the panel prints `prob` as a percentage of the pool). So the share is
        # scaled by the head's own mass on these twenty: the ordering is the
        # reranker's, the total is the head's, and a 12 % top-1 means the same
        # thing it meant before this class existed. `p_rerank` keeps the raw
        # within-shortlist share for anyone who wants to see the reweighting.
        share = torch.softmax(logp.float(), dim=0)
        mass = sum(float(c["prob"]) for c in cands)
        p = share * mass
        order = torch.argsort(share, descending=True).tolist()
        base["suggestions"] = [{"idx": int(cands[i]["idx"]),
                                "prob": float(p[i]),
                                "p_rerank": float(share[i]),
                                "p_head": float(cands[i]["prob"])}
                               for i in order][:int(k)]
        base["rerank"] = {"checkpoint": self.label,
                          "n_candidates": len(cands),
                          "head_mass": mass}
        return base
