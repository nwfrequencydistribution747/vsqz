"""
vsqz — Memory-Efficient Training & Inference for 24GB GPUs
====================================================================
Dual-mode toolkit: training (optimizer compression) and inference (KV-cache).

Training mode:
  - GaLore: Gradient low-rank projection (ICML 2024)
  - LISA: Layer-wise importance sampling (2024)
  - Q-GaLore: QLoRA + GaLore + LISA combined
  - DeepSpeed ZeRO-1 CPU offload
  - FP16/BF16 optimizer states
  - INT8 quantized optimizer states
  - Sparse gradient encoding (COO format)
  - Gradient delta tracking (rsync-inspired)

Inference mode:
  - KV-Cache H.264: I/P/B-frame token management
  - Adaptive per-head quantization (AV1-style bit allocation)
  - ADPCM attention scaling per head
  - Attention sink preservation (StreamingLLM)
  - 2x context length at same VRAM

File format:
  - .vsqz: Universal format (weights + optimizer state + SHA-256 + Recovery Record)
  - .vsq: Training checkpoint (optimizer state only)

Usage:
    from vsqz import VRAMSqueeze
    squeezer = VRAMSqueeze(model, mode="training", preset="13B_24GB")
    squeezer.save_vsqz("model.vsqz")     # One file: weights + training state
    squeezer.load_vsqz("model.vsqz")      # Full resume
    # Or: VRAMSqueeze(model, mode="inference", i_window=256, p_window=512)
"""

from .galore import GaLoreWrapper, estimate_galore_vram_savings
from .lisa import LISASampler
from .q_galore import QGaLoreTrainer, can_run_20b_on_24gb
from .deepspeed_offload import DeepSpeedCPUOffload
from .fp16_states import FP16OptimizerStates
from .int8_states import Int8OptimizerStates
from .sparse_grad import SparseGradientEncoder
from .gradient_delta import GradientDeltaTracker
from .vram_estimator import VRAMEstimator, estimate_technique_savings
from .wrapper import VRAMSqueeze

from .inference import KVCacheCompressor
from .vsqz_format import save_vsqz, load_vsqz_weights, load_vsqz_full, peek_vsqz

# Auto-activate HuggingFace plugin (Model.from_pretrained(".vsqz") just works)
try:
    from . import hf_plugin as _hf  # noqa: F401 — side-effect import
except Exception:
    pass  # HF not installed — .vsqz loading works via vsqz_format directly

__version__ = "0.4.0"
__all__ = [
    "VRAMSqueeze",
    "KVCacheCompressor",
    "GaLoreWrapper",
    "LISASampler",
    "QGaLoreTrainer",
    "DeepSpeedCPUOffload",
    "FP16OptimizerStates",
    "Int8OptimizerStates",
    "SparseGradientEncoder",
    "GradientDeltaTracker",
    "VRAMEstimator",
    "estimate_galore_vram_savings",
    "can_run_20b_on_24gb",
    "estimate_technique_savings",
    "save_vsqz",
    "load_vsqz_weights",
    "load_vsqz_full",
    "peek_vsqz",
]
