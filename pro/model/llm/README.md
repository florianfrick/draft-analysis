# Draft encoder -> soft prompt -> Qwen3.5

Branch `exp/finetune-llm`. Two stages, both here, both runnable:

| stage | script | encoder | projector | LLM |
|---|---|---|---|---|
| 1. warmup | `warmup_projector.py` | frozen | **trained** | frozen |
| 2. fine-tune | `finetune_lora.py` | frozen | **trained** | **QLoRA** |

```
python -m model.llm.warmup_projector --max-train 3000 --epochs 2
python -m model.llm.finetune_lora --init-projector <warmup>/projector.pt
python -m model.llm.evaluate_llm --projector <ckpt>/projector.pt [--adapter <ckpt>/adapter]
```

Run them through `bash pro/model/llm/run.sh ...` (see *Environment*).

## What it is

```
frozen set-transformer encoder           10 slot vectors h [B,10,256]
  -> LayerNorm -> MLP(256->1024->2560)  -> [B,10,2560] soft tokens
  -> spliced into Qwen's input embeddings at a fixed position
  -> frozen Qwen3.5-4B (4-bit)          -> logits at the answer tokens only
```

The encoder is `masking-proonly-randomvalsplit-through-wlds-ig-tes`, the
shipped pro-only masked model. It is not modified: `_encode` is called under
`no_grad`, so the experiment cannot quietly retrain the thing it is measuring.

## Three decisions that differ from the brief, and why

**1. The board is rendered as text as well as soft tokens.** The sketch put
`[DRAFT_TOKENS]` in a prompt with no board. That asks a frozen 4B model to
recover entity identity - which champion is in which slot, which are banned -
from ten continuous vectors, and free generation has no `-inf` legality mask,
so nothing stops it naming a champion that is already gone. The text carries
the *facts*; the soft tokens carry the encoder's *judgement*, which is the part
text cannot express. `--context soft` restores the pure-channel variant, so the
question "does the channel work on its own" stays answerable.

**2. Fearless bans are structurally invisible to the soft prompt.** The encoder
never receives `unavailable_mask` - it is a policy-stage input, applied as a
logit mask (`solo_queue/model/policy/model.py:_apply_legality`). So the ten slot vectors
*cannot* encode the fearless pool; only the prompt text can. 18.3% of eval rows
have a non-empty fearless pool, so this is not a corner case. The measurable form of the
question is `top1_illegal` in `evaluate_llm.py`: how often the model's first
choice is a champion that is already taken or fearless-unavailable. The policy
head scores 0 by construction; this measures what moving the decision into text
costs.

**3. The target is a 2-5 token sequence, not a token.** Under this tokenizer
only 22 of the 173 names are one token (`" Vi"` is, `" Kai'Sa"` is four), and
there are 136 distinct first tokens for 173 names. So "a clean 1:1 mapping to
discrete entity tokens" is not something the tokenizer offers, and a
first-token readout is a **136-way** problem with ties - `candidate_tables`
builds the owner matrix so the report and the legality filter can say which
champions a candidate stands for. `ft_top*` is never compared to the policy
head's 173-way numbers.

Consequence for the loss: cross-entropy is over the answer tokens only (system
text, board, task line and EOS are context). `--loss-on-eos` adds the stop
token to the scored set; it is off by default because an untrained projector
will spend its gradient learning to stop instead of learning to name.

## Why this is not a policy head

The existing `nextpick-*` head is a 173-way classifier that scores
**0.183 top-1 / 0.499 top-5** on 95,160 val states. A 4B model asked to
*generate* a 2-5 token name from a soft prompt is solving a strictly harder
problem with a much weaker output interface, so it will not beat that, and the
report says so rather than pretending otherwise. What the LLM adds is the thing
the classifier cannot do: a reason. Stage 2 is therefore a step towards the CoT
dataset, not a replacement for the policy.

## Environment

`lol-draft:dev` has no C compiler, and Qwen3.5's hybrid linear-attention layers
fall back to a reference PyTorch implementation without `flash-linear-attention`
- which is Triton, which JIT-compiles. `pro/model/llm/Dockerfile` adds `gcc` and
`libc6-dev` and points `PATH` at the venv; `run.sh` supplies the mounts so the
HF and Triton caches survive between runs.

The Python packages live in `pro/model/llm/.venv` (inside the bind mount, so they
outlive `--rm` and do not invalidate the 11.7 GB base image):
`transformers 5.22.2`, `peft 0.21.2`, `accelerate 1.15.0`,
`bitsandbytes 0.50.2`, `flash-linear-attention 0.6.0`.

> **The venv moved with the folder on 2026-11-07** (`model/llm/.venv` →
> `pro/model/llm/.venv`). Its `bin/python` resolves through the symlink and
> still finds the right prefix, but the `pip`/`accelerate` console scripts have
> the old absolute path in their shebang. Use `python -m pip` inside it, or
> rebuild it (`python -m venv .venv && .venv/bin/python -m pip install ...`).

## Measured cost on an 8 GB RTX 4060

These are the numbers that decided the defaults, not guesses. Measured with
`bench_step.py`, which prints the allocator state every step.

| configuration | step time | peak reserved |
|---|---|---|
| bf16 weights, no training | - | 8.4 GB (does not fit) |
| 4-bit, batch 2, grad-accum 8, seq ~140 | 3.1-5.1 s | 6.99 GB |
| 4-bit, batch 4, seq ~140 | dies after ~8 steps | 7.75 GB and climbing |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not a micro-optimisation
here, it is the difference between a usable run and a 5.6 s/step crawl. Without
it the default size-bucketed allocator fragments against this workload's
variable sequence lengths, reserves 7.46 -> 9.09 GB over 12 steps, and on WDDM
every step after that spills to host RAM. With it, reserved is flat.

**Batch 4 does not fit**, even though 6.90 GB of reserved looks like it should:
it survives a short probe and then dies, and the failure is disguised as
`RuntimeError: CUDA driver error: device not ready` inside whatever allocation
comes next (`new_empty`, `torch.empty`, `silu`) - never as an OOM. Batch 2 with
`--grad-accum 8` is the operating point: same effective batch, ~4.4
examples/s, so the full 42,121-game split is ~2.5 h/epoch.

## Measured results

Stage 1, projector only (encoder and LLM frozen), 6,000 games, 2 epochs,
batch 2 x grad-accum 8, lr 3e-3 cosine. Val = 200 games = 4,000 states.

| | epoch 0 | epoch 1 | chance | encoder policy head |
|---|---|---|---|---|
| val ce | 2.5065 | 2.3623 | - | - |
| exact (whole name) | 3.33 % | 4.28 % | ~0 | 18.32 % top-1 |
| ft_top1 (136-way) | 4.25 % | 6.00 % | 0.74 % | - |
| ft_top5 | 15.88 % | 20.15 % | 3.68 % | - |
| mean first-token rank | 39.0 | 34.6 | 68 | - |
| top5_illegal | 4.10 % | 4.30 % | - | 0 % (masked) |

**The channel carries real signal (5.8-8x chance on top-1) but is ~3x worse
than the encoder's own policy head reading the same state.** On this budget, 10
soft tokens through a 2.89M-param MLP are not enough for a frozen 4B to name
the next champion. The curve is still improving at epoch 1, so this is not a
ceiling - but "more of the same" is not obviously the answer either.

### The result this replaced, and why it was wrong

The first run scored **69.25 % exact / 98.6 % ft_top1**, 3.8x the policy head.
It was a copy task. `assemble` returned `ids = prompt + answer + eos`, and
`inputs_embeds` embeds everything after the placeholder, so the answer was in
the model's input - and the scored position was `len(prompt) + j`, i.e. *at* the
answer token, one too late. A causal model can read the token it is scored on.

Found by corrupting the target and watching the prediction follow the
corruption (`audit_label.py`): change `Ivern` to `Dr. Mundo` and the model says
` Dr`. The per-move breakdown was the first hint - moves 0-6 at ~10 % exact,
moves 7-19 at 99.5-100 % - and at move 0 the board is empty, so nothing could
have been predicted. Fix: score at `len(prompt) + j - 1` and truncate the input
to `prompt + answer[:-1]`. `test_scored_positions_predict_rather_than_read_the_answer`
guards it; the old test only checked the rendered *text*, not the tensor that is
actually embedded.

## Files

| file | what |
|---|---|
| `common.py` | corpus, split, names, board rendering, frozen encoder |
| `prompt.py` | chat template, soft-token splice, scored positions |
| `draft_llm.py` | projector + assembled model + metrics |
| `build.py` | 4-bit loader, LoRA attach, save/load |
| `data.py` | datasets (thin wrapper over `data/pro play/datasets.py`) |
| `metrics.py` | count-based accumulators |
| `train_util.py` | eval loop, LR schedule, param groups |
| `evaluate_llm.py` | sliced scoring: ban vs pick, per phase, legality |
| `bench_step.py` | allocator/throughput probe; prints reserved per step |
| `audit_leak.py` | split overlap + target-in-board check on the real corpus |
| `audit_label.py` | corrupt the target; does the prediction follow? |
| `audit_positions.py` | which position actually predicts, and text vs soft split |
| `audit_tensors.py` | is the answer in any tensor the encoder reads |
| `audit_encoder.py` | raw tensors at the move-6/7 boundary |
| `audit_targets.py` | target distribution per move (is the val sample degenerate) |
| `audit_eyeball.py` | one game's 20 predictions next to their targets |
| `test_llm_prompt.py` | contract tests for the splice (9 tests, CPU only) |

## Next step

A bootstrapped CoT SFT dataset: sample reasoning traces from the stage-2 model,
keep the ones that terminate in the correct champion, and train on those. That
is written up in `STATUS.md` §9 / the handoff note in this folder.
