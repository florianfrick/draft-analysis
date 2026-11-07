"""LLM experiment: draft encoder -> soft prompt -> Qwen3.5.

Stage 1  `warmup_projector.py`   encoder frozen, LLM frozen, MLP trained
Stage 2  `finetune_lora.py`      encoder frozen, projector + QLoRA trained
"""
