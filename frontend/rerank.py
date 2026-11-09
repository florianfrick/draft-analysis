"""The LoRA reranker as a next-move model.

`nextpick-*` checkpoints answer the next move with a 2.49 M-parameter head. The
`llm-*` checkpoints are a different thing wearing the same interface: the head
still produces the shortlist, and a fine-tuned Qwen3.5-4B chooses from it. On the
478 games held out from both models that is 35.31 % against the head's 19.14 %
top-1, with the head's recall@5 (51.85 %) as the ceiling.

Three things about serving it that are not obvious from the training side:

* **It needs the head anyway.** The shortlist is the head's, so this object owns
  a `PolicyPredictor` and delegates `next_move` to it. That also means the patch
  table, the legality mask and the fearless handling are the head's, not a
  second implementation.
* **The board is indexed by lane, not by pick order** (`datasets.build_prefix`:
  `slot = team * 5 + role`). The frontend's slots are pick order, so every pick
  has to be given its lane before the prompt can be built. `roles.assign_roles`
  supplies them from the head's own role distribution, which is the only role
  information available before the pick exists.
* **The lane of the pick being asked for is a decision, not a fact.** In the
  corpus it is the role the pro took. Here it is either forced (four lanes
  filled) or guessed, and the response says which (`lane_assumed`).

The score is the one that was measured: each candidate is scored as a complete
answer, and the winner is the argmax at the **first** answer position under the
same k-way mask training used - so `rerank.candidates[*].prob` and the reported
`ft_top1` are the same quantity. `total_logp` (the whole name) only breaks
first-token ties.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solo_queue.model.policy import infer  # noqa: E402
from roles import assign_roles             # noqa: E402

IMPORT_ERROR = None
try:
    from pro.model.llm import board as BOARD
    from pro.model.llm import build as B
    from pro.model.llm import common as C
    from pro.model.llm import data as D
    from pro.model.llm.prompt import assemble
except Exception as exc:               # noqa: BLE001 - reported, not raised
    IMPORT_ERROR = "%s: %s" % (type(exc).__name__, exc)

HEAD_CKPT = (ROOT / "pro" / "model" / "checkpoints"
             / "nextpick-bans-aware-through-wlds-ig-tes" / "best.pt")

# The base model, when a checkpoint does not record one. Every `llm-*` run so
# far was trained on this; `projector.pt:args.model` is authoritative when set.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

# The prompt format the reranker was trained on. Not a default to be overridden:
# a v2 projector cannot read a v4 prompt, and `test_v4_is_v2_plus_one_candidate_line`
# pins that the two differ by exactly the candidate line.
PROMPT_FORMAT = "v4"


def unavailable_reason():
    """Why this machine cannot serve the reranker, or None if it can.

    The frontend is normally run in the base image on CPU, where none of this is
    available - so the check is explicit rather than letting the first request
    fail with an ImportError three layers down.
    """
    if IMPORT_ERROR is not None:
        return "the LLM modules did not import (%s)" % IMPORT_ERROR
    try:
        import bitsandbytes  # noqa: F401
        import peft          # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:           # noqa: BLE001
        return ("transformers/peft/bitsandbytes are not installed here (%s). "
                "Run the frontend in the LLM image: "
                "bash pro/model/llm/run.sh python frontend/server.py ..."
                % exc)
    if not torch.cuda.is_available():
        return ("no CUDA device: the reranker is a 4.2B model in 4-bit "
                "(~4.5 GB VRAM). On CPU one request takes minutes, so it is "
                "refused rather than silently shipped as unusable")
    return None


def _local(path):
    """`/workspace/x` (what a training run recorded) -> this checkout's x."""
    if not path:
        return None
    p = Path(str(path))
    if p.is_absolute() and str(p).startswith("/workspace/"):
        return ROOT / str(p)[len("/workspace/"):]
    if not p.is_absolute():
        return ROOT / p
    return p


def pick_winner(first_logp, total_logp, first_tok):
    """Index of the reranker's choice. The served rule, in one place.

    `first_logp` is each candidate's log-prob at the first answer position
    under the k-way mask, so the argmax over candidates is the masked
    first-token argmax - the quantity `ft_top1` measured. Two names can share a
    first token (24 collision groups cover 61 of the 173 champions), and the
    first position cannot separate them, so the winner among the tied rows is
    the one with the larger whole-name log-prob.
    """
    win = int(max(range(len(first_logp)), key=lambda i: float(first_logp[i])))
    tied = [i for i, t in enumerate(first_tok) if int(t) == int(first_tok[win])]
    if len(tied) > 1:
        win = max(tied, key=lambda i: float(total_logp[i]))
    return win


def candidate_mask(first_tok, vocab, device=None):
    """The k-way decode constraint: the candidates' first tokens, for *every* row.

    The one thing this must not do is give each row its own token. That mask is
    satisfiable by construction - a row allowed only its own answer scores
    log P = 0 - and it looks like a very confident model instead of a broken
    constraint. Every row sees the same set, which is what makes the softmax
    comparable across candidates.

    Width is at most k and can be less: 24 first-token groups cover 61 of the
    173 champions, so two candidates can share a first token.
    """
    allowed = torch.unique(first_tok.reshape(-1))
    keep = torch.zeros(first_tok.reshape(-1).numel(), vocab, dtype=torch.bool,
                       device=device)
    keep[:, allowed] = True
    return keep


class RerankPredictor:
    """Head shortlist -> LLM choice. `kind = "policy"` so it fills that slot."""

    kind = "policy"
    # `/api/next-move` forwards a `lane` only to predictors that say they want
    # one; the head answers a lane-free question and would reject the argument.
    accepts_lane = True

    def __init__(self, ckpt_dir, device="cpu", head_ckpt=None, k=None,
                 model_id=None, label=None, warmup=True):
        if IMPORT_ERROR is not None:
            raise RuntimeError(unavailable_reason())
        self.ckpt_dir = Path(ckpt_dir)
        proj_path = self.ckpt_dir / "projector.pt"
        if not proj_path.exists():
            raise FileNotFoundError("%s has no projector.pt - an llm-* "
                                    "checkpoint is a folder, not a .pt file"
                                    % self.ckpt_dir)
        blob = torch.load(str(proj_path), map_location="cpu",
                          weights_only=False)
        args = blob.get("args") or {}
        fmt = args.get("prompt_format")
        if fmt and fmt != PROMPT_FORMAT:
            raise ValueError(
                "%s was trained with prompt_format=%s, but serving it needs %s "
                "(the candidate line). Retrain or pass the matching checkpoint."
                % (self.ckpt_dir.name, fmt, PROMPT_FORMAT))
        self.args = args
        self.k = int(k or args.get("rerank") or 5)
        self.hidden = int(args.get("hidden", 1024))
        self.n_soft = int(blob.get("n_soft") or args.get("n_soft") or 10)
        self.context = args.get("context") or "soft+text"
        self.names = C.champ_names()

        # The shortlist comes from the head, so it is loaded first and every
        # patch/legality question is answered by it.
        head = _local(head_ckpt or args.get("head")) or HEAD_CKPT
        self.head = infer.PolicyPredictor(head, device=device)
        self.patch_vocab = self.head.patch_vocab
        self.patch_index = C.DS.patch_index_from_vocab(self.patch_vocab)
        self.head_ckpt = str(head)

        self.device = torch.device(device)
        enc_name = args.get("encoder") or C.ENCODER_CKPT
        self.encoder, cfg, _ = C.build_frozen_encoder(enc_name,
                                                      device=device)
        self.num_champs = int(cfg["num_champs"])

        model_id = model_id or args.get("model") or DEFAULT_MODEL
        llm = B.load_llm(model_id, quantize=True,
                         device=self.device.index or 0)
        B.freeze(llm)
        adapter = self.ckpt_dir / "adapter"
        if adapter.exists():
            llm = B.attach_lora(llm, quantized=True,
                                adapter_path=str(adapter), trainable=False)
        self.adapter = str(adapter) if adapter.exists() else None
        d_llm = llm.get_input_embeddings().weight.shape[1]

        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = B.build_draft_llm(
            self.encoder, llm, self.tok, self.names, cfg["embed_dim"], d_llm,
            self.hidden, n_soft=self.n_soft).to(self.device)
        B.load_projector(self.model, str(proj_path))
        self.model.eval()

        # What `model_info` publishes. `val_top5` is deliberately None: the
        # stored val numbers are a k-way accuracy over a 5-candidate list, and
        # printing it in a "val top-5" slot would read as 93 %.
        self.ckpt_path = str(proj_path)
        self.label = label or self.ckpt_dir.name
        self.val_top5 = None
        self.epoch = args.get("epochs")
        self.stage = None
        self.metric = "LLM k-way"
        val = blob.get("val") or {}
        self.rerank_eval = {
            "k": self.k,
            "kway": val.get("ft_top1"),
            "exact": val.get("exact"),
            "n": val.get("n_rows"),
        }
        if warmup:
            self._warmup()

    def _warmup(self):
        """One throwaway request, because the first one is 70x slower.

        Measured: the first `next_move` after loading takes ~53 s and the ones
        after it ~0.7 s. The difference is Triton compiling the hybrid
        attention kernels and `causal_conv1d` falling back to its reference
        implementation. Paying that at startup - where it is printed as part of
        the load - is much better than paying it on the user's first click with
        nothing on screen to explain the wait.
        """
        t0 = time.time()
        try:
            self.next_move([None] * C.N_SLOTS, [], first_pick=0, k=self.k)
            print("rerank warmup: %.1fs (first request is 70x the steady "
                  "state)" % (time.time() - t0), flush=True)
        except Exception as exc:        # noqa: BLE001 - warmup is best-effort
            print("rerank warmup failed (first request will be slow): %s"
                  % exc, flush=True)

    # ------------------------------------------------------------- delegate
    def default_patch_id(self):
        return self.head.default_patch_id()

    def patch_options(self):
        return self.head.patch_options()

    def patch_key_for(self, pid):
        return self.head.patch_key_for(pid)

    def next_move(self, picks, bans, first_pick=0, k=8, unavailable=None,
                  patch_id=None, lane=None):
        """Head shortlist, then the LLM's choice. Same contract as the head.

        `k` is how many chips the UI shows; the rerank itself always uses the
        checkpoint's own k (5), because that is the list the model was trained
        to choose from.

        `patch_id` defaults to the head's own default rather than to UNK: the
        patch is a *string* in the prompt ("Patch: 16.20"), and "Patch: None"
        is text no training example ever contained.
        """
        if patch_id is None:
            patch_id = self.default_patch_id()
        res = self.head.next_move(picks, bans, first_pick=first_pick,
                                  k=max(int(k), self.k), unavailable=unavailable,
                                  patch_id=patch_id)
        res["rerank"] = None
        if res.get("complete"):
            return res
        cands = res["suggestions"][:self.k]
        if not cands:
            return res
        try:
            self._rerank(res, cands, picks, bans, unavailable, lane)
        except Exception as exc:        # noqa: BLE001 - one panel, not the server
            # The head's answer is still a valid answer. Losing the rerank must
            # not lose the recommendation, so the failure is reported beside it.
            res["rerank"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        return res

    # --------------------------------------------------------------- internals
    def _rerank(self, res, cands, picks, bans, unavailable, lane):
        filled = [p is not None for p in picks]
        lanes = assign_roles(res.get("roles") or [], filled)
        side = int(res["side"])
        is_ban = res["kind"] == "ban"
        open_lanes = [] if is_ban else BOARD.remaining_lanes(lanes, side)

        role, source = None, None
        if not is_ban:
            if lane is not None:
                role, source = int(lane), "user"
            elif len(open_lanes) == 1:
                role, source = open_lanes[0], "board"   # forced, nothing guessed
            elif open_lanes:
                role, source = open_lanes[0], "default"  # a guess, and labelled
            else:
                raise ValueError("the board has all five lanes filled but the "
                                 "move is still a pick")

        patch_str = self.patch_key_for(res.get("patch_id")) or "unknown"
        kind = C.DS.MOVE_BAN if is_ban else C.DS.MOVE_PICK
        samples = [BOARD.sample_from_board(
            picks, lanes, bans, res["first_pick"], res["move_idx"], patch_str,
            kind, side, target_role=role, target_champ=c["idx"],
            fearless=unavailable or (), num_champs=self.num_champs,
            patch_index=self.patch_index) for c in cands]

        enc_batch, _ = D.collate(samples)
        enc_batch = D.to_device(enc_batch, self.device)
        cand_list = [(int(c["idx"]), float(c["prob"])) for c in cands]
        prompt = assemble(self.tok, samples, self.names, context=self.context,
                          with_answers=True, device=self.device,
                          prompt_format=PROMPT_FORMAT,
                          candidates=[cand_list] * len(samples))

        # The k-way mask: the candidates' first tokens, the same set for every
        # row. `ans_ids[:, 0]` is the candidate's own first token, so the mask
        # and the answers cannot disagree about which tokens are allowed.
        first_tok = prompt["ans_ids"][:, 0]
        keep = candidate_mask(first_tok, int(self.model.cand_mask.numel()),
                              device=self.device)

        with torch.no_grad():
            sc = self.model.answer_scores(enc_batch, prompt, keep=keep)
        first_lp = sc["first_logp"].float().cpu()
        total_lp = sc["total_logp"].float().cpu()
        probs = first_lp.softmax(dim=0)

        win = pick_winner(first_lp, total_lp, first_tok.cpu())

        rows = []
        for i, c in enumerate(cands):
            rows.append({"idx": int(c["idx"]), "name": self.names[int(c["idx"])],
                         "head_prob": float(c["prob"]),
                         "prob": float(probs[i]),
                         "logp": float(total_lp[i]),
                         "chosen": i == win})
        for s in res["suggestions"]:
            hit = next((r for r in rows if r["idx"] == s["idx"]), None)
            if hit is not None:
                s["llm_prob"] = hit["prob"]
                s["llm_pick"] = hit["chosen"]

        # The chips are read left to right as a ranking, so the answering
        # model's order has to be the one on screen. The LLM scored the
        # shortlist, so its probabilities order it; candidates the LLM did not
        # see keep the head's order behind them (-1 sinks them, and the sort is
        # stable, so they stay in the head's sequence among themselves).
        res["suggestions"].sort(key=lambda s: s.get("llm_prob", -1.0),
                                reverse=True)

        res["rerank"] = {
            "k": len(cands),
            "chosen_idx": int(cands[win]["idx"]),
            "chosen": self.names[int(cands[win]["idx"])],
            "lane": None if is_ban else int(role),
            "lane_name": None if is_ban else C.ROLE_DISPLAY[int(role)],
            # "board" = four lanes filled, so this one is forced; "user" = the
            # drafter chose it; "default" = the board does not decide it yet and
            # the server picked the first open lane, which is a guess.
            "lane_source": source,
            "lane_assumed": source == "default",
            "lanes_open": open_lanes,
            "candidates": rows,
            "head_top1_agrees": bool(win == 0),
        }
