import importlib
import torch

print("torch %s cuda %s" % (torch.__version__, torch.version.cuda))
for m in ["fla", "causal_conv1d", "flash_attn", "triton", "bitsandbytes"]:
    try:
        mod = importlib.import_module(m)
        print("%-14s OK      %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("%-14s MISSING %s" % (m, type(e).__name__))
