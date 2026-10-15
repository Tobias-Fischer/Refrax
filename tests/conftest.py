import os

# Make unsupported MPS ops fail loudly instead of silently running on the CPU.
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
