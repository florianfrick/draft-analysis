"""Model construction: the 4-bit LLM, the projector, and the optional LoRA.

Kept out of the two trainer scripts because the warmup and the fine-tune must
build *exactly* the same object - a projector trained in one file and reloaded
by another that disagreed about `hidden`, `n_soft` or the calibration would
fail silently in the worst way (it would still run).

Memory notes that decided the defaults, measured on an 8 GB RTX 4060:

* bf16 weights are 8.4 GB, so even the *frozen* warmup has to be 4-bit. nf4
  with double quantisation lands at 3.1 GB including the 248k x 2560 embedding
  table, which stays bf16 because bitsandbytes does not quantise embeddings.
* Without gradient checkpointing a batch of 4 x 200 tokens peaks at 15.5 GB
  (WDDM silently spills to system RAM, so it "works" and is 4x slower).
  Checkpointing brings it under the card.
"""
import torch

from pro.model.llm.common import candidate_tables
from pro.model.llm.draft_llm import DraftLLM, SoftPromptProjector

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def load_llm(model_id, quantize=True, device=0, dtype=torch.bfloat16):
    """Text-only Qwen3.5 in 4-bit (default) or bf16.

    `Qwen3_5ForCausalLM` is used rather than the checkpoint's own
    `Qwen3_5ForConditionalGeneration`: the checkpoint is multimodal, and the
    vision tower (0.33B params, 0.67 GB bf16 = 7.2 % of the file) plus the MTP
    head (0.12B, 2.6 %) are weights a text-only soft-prompt task never touches.
    Measured from the safetensors headers, and confirmed by the resident size:
    the text trunk is 4.21B params, which is 3.06 GB in nf4 + bf16 embeddings,
    against the 3.14 GB the build actually allocates.
    """
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM
    kwargs = dict(device_map={"": device}, dtype=dtype)
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    llm = Qwen3_5ForCausalLM.from_pretrained(model_id, **kwargs)
    llm.config.use_cache = False
    return llm


def enable_checkpointing(llm):
    """Gradient checkpointing, tolerating the API differences across versions."""
    try:
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        return True
    except TypeError:
        llm.gradient_checkpointing_enable()
        return True


def freeze(llm):
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()
    return llm


def attach_lora(llm, r=16, alpha=32, dropout=0.05, quantized=True,
                adapter_path=""):
    """QLoRA adapters on the attention and MLP projections.

    `prepare_model_for_kbit_training` casts the norms to fp32, turns the cache
    off and freezes the base - it is what makes a 4-bit base trainable at all.
    It also requires a *quantized* model, so it is skipped when the caller
    loaded bf16 (`--no-4bit`), where a plain `freeze` already did the same job.

    `adapter_path` loads an existing adapter *instead of* a fresh one, which is
    how a finished run is continued for another cycle without re-deriving it.
    The r/alpha/dropout arguments are then ignored: the shapes come from the
    saved adapter, and a mismatch would be a different model, not a resume.
    """
    from peft import (LoraConfig, PeftModel, get_peft_model,
                      prepare_model_for_kbit_training)
    if quantized:
        llm = prepare_model_for_kbit_training(
            llm, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter_path:
        model = PeftModel.from_pretrained(llm, adapter_path,
                                          is_trainable=True)
        model.enable_input_require_grads()
        return model
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    model = get_peft_model(llm, cfg)
    model.enable_input_require_grads()
    return model


def build_draft_llm(encoder, llm, tok, names, d_in, d_llm, hidden,
                    n_soft=10, dropout=0.0):
    """Projector + candidate tables + the assembled model, calibrated."""
    projector = SoftPromptProjector(d_in=d_in, d_llm=d_llm, hidden=hidden,
                                    dropout=dropout)
    first, owner = candidate_tables(tok, names)
    model = DraftLLM(encoder, projector, llm, first, owner, n_soft=n_soft)
    target_rms, raw_rms, scale = projector.calibrate(
        model.trunk.embed_tokens.weight)
    model.calibration = {"embed_rms": target_rms, "raw_out_rms": raw_rms,
                         "scale": scale}
    return model


def save_projector(model, path, extra=None):
    torch.save({
        "projector": model.projector.state_dict(),
        "calibration": getattr(model, "calibration", {}),
        "n_soft": model.n_soft,
        **(extra or {}),
    }, str(path))


def load_projector(model, path):
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    model.projector.load_state_dict(blob["projector"])
    return blob
