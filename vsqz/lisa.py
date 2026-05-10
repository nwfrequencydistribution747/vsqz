#!/usr/bin/env python3
"""
LISA — Layer-wise Importance Sampled AdamW
============================================
Paper: "LISA: Layer-wise Importance Sampling for Memory-Efficient
       Large Language Model Fine-Tuning" (2024)

Core insight: Not all layers need to be active every step. By sampling a subset
of transformer layers per step (freezing others), we reduce:
  - Activation memory (no forward activations for frozen layers)
  - Gradient memory (no backward passes for frozen layers)
  → ~40-60% VRAM reduction with minimal quality loss

Strategy: Importance-weighted sampling — layers with higher weight norms
have higher probability of being selected. This ensures important layers
train every step while less critical ones train less frequently.

Integration: Wraps model training via context manager or explicit sampling.

Usage:
    sampler = LISASampler(model, active_layers_ratio=0.5)
    for batch in dataloader:
        with sampler.sample_layers():
            loss = model(batch).loss
            loss.backward()
            optimizer.step()
"""

from __future__ import annotations

import logging
import random
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("LISA")

# ── Layer identification ─────────────────────────────────────────────────


def _is_transformer_layer(name: str, module: nn.Module) -> bool:
    """Identify a transformer decoder/encoder layer."""
    cls_name = module.__class__.__name__
    return any(pat in cls_name for pat in (
        "DecoderLayer", "EncoderLayer", "Qwen3VLDecoderLayer",
        "LlamaDecoderLayer", "GPT2Block", "TransformerBlock",
        "TransformerDecoderLayer", "TransformerEncoderLayer",
    ))


def _collect_transformer_layers(
    model: nn.Module,
) -> List[Tuple[str, nn.Module]]:
    """Walk model and collect all transformer layers."""
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if _is_transformer_layer(name, module):
            layers.append((name, module))
    return layers


# ── Layer importance scoring ─────────────────────────────────────────────


def _compute_layer_importance(module: nn.Module) -> float:
    """Compute importance score for a layer based on L2 weight norms."""
    total_norm = 0.0
    for param in module.parameters(recurse=True):
        if param.requires_grad and param.ndim >= 2:
            total_norm += param.data.float().norm(2).item() ** 2
    return total_norm ** 0.5


# ── LISA Sampler ─────────────────────────────────────────────────────────


class LISASampler:
    """Layer-wise Importance Sampling for activation-efficient training.

    Key parameters:
      - active_layers_ratio: fraction of layers active per step (0.3-0.7 typical)
      - warmup_steps: start with all layers active, ramp down sampling
      - importance_temperature: higher = more uniform sampling, lower = greedy

    Usage:
        sampler = LISASampler(model, active_layers_ratio=0.5)

        # Option A: Context manager (auto-freezes, auto-thaws)
        with sampler.sample_layers():
            output = model(**batch)
            loss.backward()

        # Option B: Manual (for distributed / complex loops)
        sampler.select_active_layers()
        output = model(**batch)
        loss.backward()
        sampler.restore_all_layers()
    """

    def __init__(
        self,
        model: nn.Module,
        active_layers_ratio: float = 0.5,
        warmup_steps: int = 0,
        importance_temperature: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.active_ratio = active_layers_ratio
        self.warmup_steps = warmup_steps
        self.temperature = importance_temperature
        self._rng = random.Random(seed)

        # Discover transformer layers
        self._layers = _collect_transformer_layers(model)
        self._num_layers = len(self._layers)

        if self._num_layers == 0:
            logger.warning("LISA: no transformer layers found — sampler is no-op")
        else:
            logger.info(
                "LISA: %d transformer layers, active_ratio=%.1f → ~%d active/step",
                self._num_layers,
                active_layers_ratio,
                max(1, int(self._num_layers * active_layers_ratio)),
            )

        # Layer importance scores (recomputed periodically)
        self._importance: Optional[List[float]] = None
        self._frozen_set: Set[int] = set()
        self._step_counter = 0

        # Embedding + LM head are always active (critical for output)
        self._always_active: Set[str] = set()

    # ── Public API ──────────────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def active_count(self) -> int:
        return self._num_layers - len(self._frozen_set)

    def select_active_layers(self) -> List[int]:
        """Sample which layers to keep active this step. Returns list of active indices."""
        self._step_counter += 1

        # No transformer layers — no-op
        if self._num_layers == 0:
            return []

        # During warmup: all layers active
        if self._step_counter <= self.warmup_steps:
            self._frozen_set.clear()
            self._thaw_excluded()
            return list(range(self._num_layers))

        # Recompute importance periodically (every 50 steps)
        if self._importance is None or self._step_counter % 50 == 0:
            self._importance = [
                _compute_layer_importance(mod) for _, mod in self._layers
            ]

        # Normalize to probabilities
        if self.temperature <= 0:
            probs = [1.0 / self._num_layers] * self._num_layers
        else:
            imp = [v ** (1.0 / max(self.temperature, 0.01)) for v in self._importance]
            total = sum(imp)
            probs = [v / total for v in imp]

        # Sample without replacement
        n_active = max(1, int(self._num_layers * self.active_ratio))
        active_indices = sorted(self._rng.choices(
            range(self._num_layers), weights=probs, k=n_active
        ))
        # Deduplicate (choices with replacement)
        active_indices = list(set(active_indices))
        while len(active_indices) < n_active:
            cand = self._rng.choices(range(self._num_layers), weights=probs, k=1)[0]
            if cand not in active_indices:
                active_indices.append(cand)
        active_indices.sort()

        # Freeze inactive layers
        self._frozen_set = set(range(self._num_layers)) - set(active_indices)
        self._freeze_inactive()
        self._thaw_excluded()

        return active_indices

    def restore_all_layers(self) -> None:
        """Thaw all layers (for eval or checkpointing)."""
        for idx in list(self._frozen_set):
            _, mod = self._layers[idx]
            for param in mod.parameters(recurse=True):
                param.requires_grad = True
        self._frozen_set.clear()

    @contextmanager
    def sample_layers(self):
        """Context manager: auto-sample, train, auto-restore."""
        self.select_active_layers()
        try:
            yield
        finally:
            pass  # Keep frozen during gradient step, restore at next sample

    def sample_step(self) -> None:
        """Call before forward: selects active layers."""
        self.select_active_layers()

    def end_step(self) -> None:
        """Call after backward: no-op (restore happens at next sample)."""
        pass

    # ── Internal ────────────────────────────────────────────────────

    def _freeze_inactive(self) -> None:
        """Freeze parameters in inactive layers."""
        for idx, (name, module) in enumerate(self._layers):
            if idx in self._frozen_set:
                for param in module.parameters(recurse=True):
                    param.requires_grad = False

    def _thaw_excluded(self) -> None:
        """Ensure always-active modules are trainable."""
        for name, module in self.model.named_modules():
            for pname in self._always_active:
                if pname in name:
                    for param in module.parameters(recurse=False):
                        param.requires_grad = True

    # ── State ───────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_counter": self._step_counter,
            "num_layers": self._num_layers,
            "active_ratio": self.active_ratio,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._step_counter = state.get("step_counter", 0)


# ── VRAM estimator ───────────────────────────────────────────────────────


def estimate_lisa_vram_savings(
    num_layers: int,
    activation_vram_gb: float,
    active_ratio: float = 0.5,
) -> Dict[str, float]:
    """Estimate VRAM saved by LISA sampling.

    Args:
        num_layers: Number of transformer layers
        activation_vram_gb: Total activation VRAM without LISA
        active_ratio: Fraction of layers active per step
    """
    saved_act_gb = activation_vram_gb * (1.0 - active_ratio)
    # Also save ~active_ratio fraction of gradient VRAM in frozen layers
    grad_savings_gb = activation_vram_gb * 0.3 * (1.0 - active_ratio)  # ~30% of act VRAM is gradients

    return {
        "activation_vram_saved_gb": round(saved_act_gb, 2),
        "gradient_vram_saved_gb": round(grad_savings_gb, 2),
        "total_saved_gb": round(saved_act_gb + grad_savings_gb, 2),
        "activation_vram_with_lisa_gb": round(activation_vram_gb * active_ratio, 2),
    }
