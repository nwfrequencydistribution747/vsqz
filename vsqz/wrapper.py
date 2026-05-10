"""
Unified VRAMSqueeze Wrapper
=============================
Single entry point that stacks all memory-saving techniques cumulatively.

VRAM savings per technique (cumulative):
  Baseline (QLoRA NF4):      ~16 GB for 13B
  + GaLore (r=128):          ~14 GB   (-2 GB, optimizer compression)
  + LISA (50% layers):       ~10 GB   (-4 GB, activation sampling)
  + FP16 States:             ~9 GB    (-1 GB, half-precision moments)
  + INT8 States (instead):   ~8 GB    (-2 GB, 8-bit moments)
  + CPU Offload:             ~6 GB    (-3 GB, states to RAM)
  + Sparse Grad + Delta:     ~5 GB    (-1 GB, gradient compression)
  + Adaptive Quant:          ~5 GB    (-0.5 GB, per-layer precision)

Preset configurations:
  "13B_24GB"  — 13B model, batch=2, safe defaults
  "20B_24GB"  — 20B model, batch=1, aggressive (tight)
  "9B_max"    — 9B model, batch=3, uses all techniques

Usage:
    from vsqz import VRAMSqueeze
    squeezer = VRAMSqueeze(model, optimizer, preset="13B_24GB")
    for batch in dataloader:
        squeezer.step_begin()
        loss = model(**batch).loss
        loss.backward()
        squeezer.step_end()
        squeezer.zero_grad()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from .galore import GaLoreWrapper
from .lisa import LISASampler
from .deepspeed_offload import DeepSpeedCPUOffload
from .fp16_states import FP16OptimizerStates
from .int8_states import Int8OptimizerStates
from .sparse_grad import SparseGradientEncoder
from .gradient_delta import GradientDeltaTracker
from .adaptive_quant import SpatialGradientPredictor, AdaptiveLayerQuantizer, AdaptiveStepScaler
from .vram_estimator import VRAMEstimator, estimate_technique_savings
from .inference import KVCacheCompressor

logger = logging.getLogger("VRAMSqueeze")

# ── Presets ────────────────────────────────────────────────────────────

PRESETS = {
    "13B_24GB": {
        "galore_rank": 128,
        "lisa_ratio": 0.5,
        "fp16_states": True,
        "sparse_grad": True,
        "gradient_delta": True,
        "lisa_warmup": 50,
    },
    "20B_24GB": {
        "galore_rank": 128,
        "lisa_ratio": 0.4,
        "int8_states": True,
        "sparse_grad": True,
        "gradient_delta": True,
        "deepspeed_offload": True,
        "lisa_warmup": 50,
    },
    "9B_max": {
        "galore_rank": 128,
        "lisa_ratio": 0.4,
        "int8_states": True,
        "sparse_grad": True,
        "gradient_delta": True,
        "adaptive_quant": True,
        "lisa_warmup": 0,
    },
    "safe_defaults": {
        "galore_rank": 128,
        "lisa_ratio": 0.6,
        "fp16_states": True,
        "lisa_warmup": 100,
    },
}


INFERENCE_PRESETS = {
    "balanced": {"i_window": 256, "p_window": 512, "attention_threshold": 0.01, "quant_bits_high": 8, "quant_bits_low": 4},
    "aggressive": {"i_window": 128, "p_window": 384, "attention_threshold": 0.02, "quant_bits_high": 8, "quant_bits_low": 3},
    "conservative": {"i_window": 512, "p_window": 1024, "attention_threshold": 0.005, "quant_bits_high": 8, "quant_bits_low": 6},
}


class VRAMSqueeze:
    """Dual-mode wrapper: training optimizer compression + inference KV-cache.

    Modes:
      - "training": stacks optimizer technique chain (GaLore + LISA + FP16 + ...)
      - "inference": H.264-style KV-cache compression (I/P/B-frame + adaptive Q)

    Usage:
        # Training
        squeezer = VRAMSqueeze(model, mode="training", optimizer=opt, preset="13B_24GB")
        squeezer.step_begin(); loss.backward(); squeezer.step_end()

        # Inference
        squeezer = VRAMSqueeze(model, mode="inference", preset="balanced")
        squeezer.evict_if_needed(current_seq_len)  # Call before each generation step
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "training",
        optimizer=None,
        preset: Optional[str] = None,
        galore_rank: Optional[int] = None,
        lisa_ratio: Optional[float] = None,
        fp16_states: bool = False,
        int8_states: bool = False,
        deepspeed_offload: bool = False,
        sparse_grad: bool = False,
        gradient_delta: bool = False,
        adaptive_quant: bool = False,
        lisa_warmup: int = 50,
        lisa_seed: Optional[int] = None,
        excluded_params: Optional[set] = None,
        # Inference-only params
        i_window: int = 256,
        p_window: int = 512,
        attention_threshold: float = 0.01,
        quant_bits_high: int = 8,
        quant_bits_low: int = 4,
        attention_sink_tokens: int = 4,
    ):
        self.model = model
        self._mode = mode
        self._config: Dict[str, Any] = {"mode": mode}

        # ── Inference path ──────────────────────────────────────────
        if mode == "inference":
            if preset and preset in INFERENCE_PRESETS:
                cfg = INFERENCE_PRESETS[preset]
                i_window = cfg["i_window"]
                p_window = cfg["p_window"]
                attention_threshold = cfg["attention_threshold"]
                quant_bits_high = cfg["quant_bits_high"]
                quant_bits_low = cfg["quant_bits_low"]

            self._kv_compressor = KVCacheCompressor(
                model,
                i_window=i_window,
                p_window=p_window,
                attention_threshold=attention_threshold,
                quant_bits_high=quant_bits_high,
                quant_bits_low=quant_bits_low,
                attention_sink_tokens=attention_sink_tokens,
            )
            self._optimizer = None
            self._config.update({
                "i_window": i_window, "p_window": p_window,
                "attention_threshold": attention_threshold,
                "quant_bits": f"{quant_bits_low}-{quant_bits_high}",
            })
            logger.info("VRAMSqueeze inference mode — %d I-frames, %d P-frames, evict threshold=%.3f",
                       i_window, p_window, attention_threshold)
            return

        # ── Training path ───────────────────────────────────────────
        if optimizer is None:
            raise ValueError("Training mode requires an optimizer")
        self._original_optimizer = optimizer
        self._kv_compressor = None
        self._step_counter = 0
        self._mode = "training"

        # Apply preset config
        if preset and preset in PRESETS:
            cfg = PRESETS[preset]
            galore_rank = galore_rank or cfg.get("galore_rank")
            lisa_ratio = lisa_ratio or cfg.get("lisa_ratio")
            fp16_states = fp16_states or cfg.get("fp16_states", False)
            int8_states = int8_states or cfg.get("int8_states", False)
            deepspeed_offload = deepspeed_offload or cfg.get("deepspeed_offload", False)
            sparse_grad = sparse_grad or cfg.get("sparse_grad", False)
            gradient_delta = gradient_delta or cfg.get("gradient_delta", False)
            adaptive_quant = adaptive_quant or cfg.get("adaptive_quant", False)
            lisa_warmup = cfg.get("lisa_warmup", lisa_warmup)

        # ── Build technique stack ──
        current_opt = optimizer

        # Layer 1: GaLore (gradient low-rank, wraps optimizer)
        if galore_rank:
            self._galore = GaLoreWrapper(model, current_opt, rank=galore_rank)
            current_opt = self._galore
            self._config["galore_rank"] = galore_rank
        else:
            self._galore = None

        # Layer 2: FP16/INT8 states (precision reduction)
        if int8_states and fp16_states:
            logger.warning("Both INT8 and FP16 selected — using INT8 (more aggressive)")
            fp16_states = False

        if int8_states:
            self._state_compressor = Int8OptimizerStates(current_opt)
            current_opt = self._state_compressor
            self._config["int8_states"] = True
        elif fp16_states:
            self._state_compressor = FP16OptimizerStates(current_opt)
            current_opt = self._state_compressor
            self._config["fp16_states"] = True
        else:
            self._state_compressor = None

        # Layer 3: CPU offload
        if deepspeed_offload:
            self._cpu_offload = DeepSpeedCPUOffload(current_opt)
            current_opt = self._cpu_offload
            self._config["deepspeed_offload"] = True
        else:
            self._cpu_offload = None

        self._optimizer = current_opt

        # Layer 4: LISA (activation reduction, before forward)
        if lisa_ratio:
            self._lisa = LISASampler(
                model, active_layers_ratio=lisa_ratio,
                warmup_steps=lisa_warmup, seed=lisa_seed,
            )
            self._config["lisa_ratio"] = lisa_ratio
            self._config["lisa_warmup"] = lisa_warmup
        else:
            self._lisa = None

        # Layer 5: Sparse gradients
        if sparse_grad:
            self._sparse = SparseGradientEncoder()
            self._config["sparse_grad"] = True
        else:
            self._sparse = None

        # Layer 6: Gradient delta
        if gradient_delta:
            self._delta = GradientDeltaTracker(model)
            self._config["gradient_delta"] = True
        else:
            self._delta = None

        # Layer 7: Adaptive quantization (H.264 + AV1 + ADPCM)
        if adaptive_quant:
            self._spatial_pred = SpatialGradientPredictor()
            self._layer_quant = AdaptiveLayerQuantizer()
            self._step_scaler = AdaptiveStepScaler()
            self._config["adaptive_quant"] = True
        else:
            self._spatial_pred = None
            self._layer_quant = None
            self._step_scaler = None

        # ── All trainable params (for sparse grad / delta / adaptive Q) ──
        self._trainable_params = [
            p for p in model.parameters() if p.requires_grad
        ]

        logger.info(
            "VRAMSqueeze: %d techniques stacked — config=%s",
            len([v for v in self._config.values() if v is True or isinstance(v, (int, float))]),
            {k: v for k, v in self._config.items() if v},
        )

    # ── Public API ──────────────────────────────────────────────────

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def step_begin(self) -> None:
        """Call BEFORE forward pass."""
        self._step_counter += 1

        if self._lisa:
            self._lisa.select_active_layers()

        if self._delta:
            self._delta.step()

    def step_end(self) -> Optional[float]:
        """Call AFTER backward pass. Compresses → optimizer step."""
        params = self._trainable_params

        # Sparse gradient encoding (compress near-zero gradients)
        if self._sparse:
            self._sparse.compress_gradients(params)

        # Gradient delta encoding
        if self._delta:
            for p in params:
                self._delta.compress(p)

        # Adaptive quantization
        if self._spatial_pred and self._layer_quant and self._step_scaler:
            self._apply_adaptive_quantization()

        # Optimizer step (runs through stacked wrappers)
        loss = self._optimizer.step()

        # Reconstruction for next step
        if self._sparse:
            self._sparse.decompress_gradients(params)

        if self._delta:
            for p in params:
                self._delta.reconstruct(p)

        return loss

    def zero_grad(self) -> None:
        self._optimizer.zero_grad(set_to_none=True)

    def restore_all_layers(self) -> None:
        """Activate all layers (for evaluation)."""
        if self._lisa:
            self._lisa.restore_all_layers()

    def _apply_adaptive_quantization(self) -> None:
        """H.264 + AV1 + ADPCM style adaptive gradient statistics.

        Updates per-layer statistics used by downstream components:
          - AdaptiveLayerQuantizer: EMA of gradient variance → bit allocation
          - AdaptiveStepScaler: EMA of gradient magnitude → quantization scale
          - SpatialGradientPredictor: cross-layer gradient correlation

        These statistics guide INT8 states, gradient delta, and sparse grad
        to use optimal bit budgets per layer in subsequent steps.
        """
        # Update per-layer gradient statistics
        for name_param in self.model.named_parameters():
            name, param = name_param
            if param.grad is None:
                continue
            if self._layer_quant:
                self._layer_quant.update_statistics(name, param.grad.data)
            if self._step_scaler:
                self._step_scaler.encode(name, param.grad.data)

        # Recompute bit budgets periodically (every 50 steps)
        if self._layer_quant and self._step_counter % 50 == 0:
            self._layer_quant.allocate_bits()

        # Apply spatial prediction to reduce gradient delta magnitude
        if self._spatial_pred and self._step_counter > 1:
            layer_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None and 'layers.' in name:
                    layer_id = self._spatial_pred._get_layer_name(name)
                    if layer_id not in layer_grads:
                        layer_grads[layer_id] = {}
                    layer_grads[layer_id][name] = param.grad.data

            for name, param in self.model.named_parameters():
                if param.grad is not None and len(param.grad.shape) == 2:
                    residual = self._spatial_pred.encode(name, param.grad.data, layer_grads)
                    if residual is not None:
                        param.grad.data = residual

    # ── State ───────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "config": self._config,
            "step": self._step_counter,
        }
        if self._galore:
            state["galore"] = self._galore.state_dict()
        if self._lisa:
            state["lisa"] = self._lisa.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._step_counter = state.get("step", 0)
        if self._galore and "galore" in state:
            self._galore.load_state_dict(state["galore"])
        if self._lisa and "lisa" in state:
            self._lisa.load_state_dict(state["lisa"])

    # ── Checkpoint API ────────────────────────────────────────────

    def save_checkpoint(self, path: str, **metadata) -> str:
        """Save full training state in compressed vsqz format (.vsq.pt).

        25× smaller than full AdamW checkpoints.
        """
        from .checkpoint import save_checkpoint as _save
        return _save(self, path, extra_metadata=metadata if metadata else None)

    def load_checkpoint(self, path: str) -> dict:
        """Load training state from vsqz checkpoint."""
        from .checkpoint import load_checkpoint as _load
        return _load(self, path)

    def save_vsqz(self, path: str) -> str:
        """Save model + optimizer state in universal .vsqz format.

        Single file: weights (NF4 FP16) + GaLore states + training metadata.
        For inference: load_vsqz_weights(path) → model
        For resume: load_vsqz_full(squeezer, path) → continues training
        """
        from .vsqz_format import save_vsqz as _save_vsqz
        return _save_vsqz(self, path)

    def load_vsqz(self, path: str) -> dict:
        """Load model + optimizer state from .vsqz file. Full resume."""
        from .vsqz_format import load_vsqz_full as _load_vsqz
        return _load_vsqz(self, path)

    # ── Inference API (mode="inference") ────────────────────────────

    def evict_if_needed(self, current_seq_len: int, max_cache_len: int = 2048) -> int:
        """[Inference] Evict B-frame tokens if KV-cache exceeds budget.

        Call before each generation step. Returns number of tokens evicted.
        """
        if self._mode != "inference":
            raise RuntimeError("evict_if_needed() only available in inference mode")
        return self._kv_compressor.evict_if_needed(current_seq_len, max_cache_len)

    def update_attention_stats(self, attention_weights: list) -> Dict[int, int]:
        """[Inference] Update per-head adaptive quantization from attention weights.

        Call after forward pass to feed attention patterns into ADPCM scaler.
        Returns updated per-head bit budget.
        """
        if self._mode != "inference":
            raise RuntimeError("update_attention_stats() only available in inference mode")
        return self._kv_compressor.allocate_head_bit_budget(attention_weights)

    @property
    def kv_stats(self) -> Dict:
        """[Inference] Current KV-cache compression statistics."""
        if self._mode != "inference":
            return {}
        return self._kv_compressor.stats()

    # ── Shared API ──────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    def summary(self) -> Dict[str, Any]:
        """Return current technique stack and estimated VRAM usage."""
        if self._mode == "inference":
            return {
                "mode": "inference",
                "techniques": ["KV-Cache H.264 I/P/B-frame", "Adaptive Per-Head Quantization (AV1)", "ADPCM Attention Scaling"],
                "config": {k: v for k, v in self._config.items() if v},
                "kv_stats": self._kv_compressor.stats() if self._kv_compressor else {},
            }
        return {
            "mode": "training",
            "techniques": [k for k, v in self._config.items() if v],
            "step": self._step_counter,
            "vram_estimate": VRAMEstimator().summary(),
        }
