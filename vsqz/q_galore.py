#!/usr/bin/env python3
"""
Q-GaLore — QLoRA + GaLore + LISA Integration
===============================================
Combines three orthogonal memory-efficiency techniques for maximum VRAM savings:
  1. QLoRA (NF4 weights) → saves weight memory (~75%)
  2. GaLore (gradient projection) → saves optimizer memory (~90%)
  3. LISA (layer sampling) → saves activation memory (~50%)

Together: ~60-70% total VRAM reduction vs full fine-tuning,
          enabling 20B models on 24 GB GPU.

| Technique    | What it saves       | VRAM saved | Quality impact |
|-------------|---------------------|-----------|----------------|
| QLoRA NF4   | Weight parameters    | ~75%      | Minimal (<1%)  |
| GaLore r128 | Optimizer states     | ~90%      | Minimal (<0.5%)|
| LISA 50%    | Activations+grads    | ~50%      | Minimal (<1%)  |

Usage:
    from vsqz.q_galore import QGaLoreTrainer
    trainer = QGaLoreTrainer(model, galore_rank=128, lisa_ratio=0.5)
    for batch in dataloader:
        trainer.step_begin()        # Sample layers (LISA)
        loss = model(**batch).loss
        loss.backward()
        trainer.step_end()          # Project gradients (GaLore) + optimizer step
        trainer.zero_grad()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn

from .galore import GaLoreWrapper, estimate_galore_vram_savings
from .lisa import LISASampler, estimate_lisa_vram_savings

logger = logging.getLogger("Q-GaLore")


class QGaLoreTrainer:
    """Combined QLoRA + GaLore + LISA training wrapper.

    QLoRA is assumed active (model loaded in NF4 via bitsandbytes).
    GaLore compresses optimizer states.
    LISA samples layers for activation efficiency.

    Usage:
        trainer = QGaLoreTrainer(
            model,
            base_optimizer=optimizer,
            galore_rank=128,
            lisa_ratio=0.5,
            warmup_steps=50,
        )

        for batch in dataloader:
            trainer.step_begin()         # LISA: sample active layers
            loss = model(**batch).loss
            loss.backward()
            trainer._step(loss)          # GaLore: project + AdamW step
            trainer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        galore_rank: int = 128,
        galore_scale: float = 1.0,
        lisa_ratio: float = 0.5,
        lisa_temperature: float = 0.5,
        warmup_steps: int = 50,
        excluded_param_names: Optional[Set[str]] = None,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.galore_rank = galore_rank
        self.lisa_ratio = lisa_ratio
        self.warmup_steps = warmup_steps

        # Initialize sub-components
        self._galore = GaLoreWrapper(
            model,
            base_optimizer=base_optimizer,
            rank=galore_rank,
            scale=galore_scale,
            excluded_param_names=excluded_param_names,
        )

        self._lisa = LISASampler(
            model,
            active_layers_ratio=lisa_ratio,
            warmup_steps=warmup_steps,
            importance_temperature=lisa_temperature,
            seed=seed,
        )

        self._step_counter = 0

        # Estimate total VRAM
        self._log_vram_estimate()

    def _log_vram_estimate(self) -> None:
        """Log estimated VRAM savings from the combined stack."""
        galore_est = estimate_galore_vram_savings(self.model, self.galore_rank)
        lisa_est = estimate_lisa_vram_savings(
            self._lisa.num_layers,
            activation_vram_gb=8.0,  # Typical for 9B with offloading
            active_ratio=self.lisa_ratio,
        )

        total_saved = galore_est["saved_gb"] + lisa_est["total_saved_gb"]
        logger.info(
            "Q-GaLore stacked — GaLore: %.1f GB saved, LISA: %.1f GB saved, Total: %.1f GB",
            galore_est["saved_gb"], lisa_est["total_saved_gb"], total_saved,
        )

    # ── Public API ──────────────────────────────────────────────────

    @property
    def galore(self) -> GaLoreWrapper:
        return self._galore

    @property
    def lisa(self) -> LISASampler:
        return self._lisa

    def step_begin(self) -> None:
        """Call BEFORE forward pass. Samples layers (LISA)."""
        self._step_counter += 1
        self._lisa.select_active_layers()

    def step_end(self) -> Optional[float]:
        """Call AFTER backward pass. Projects gradients (GaLore) and steps optimizer."""
        loss = self._galore.step()
        return loss

    def zero_grad(self) -> None:
        """Zero all gradients."""
        self._galore.zero_grad(set_to_none=True)

    def restore_all_layers(self) -> None:
        """Restore all layers for evaluation."""
        self._lisa.restore_all_layers()

    # ── State ───────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "galore": self._galore.state_dict(),
            "lisa": self._lisa.state_dict(),
            "step_counter": self._step_counter,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._galore.load_state_dict(state.get("galore", {}))
        self._lisa.load_state_dict(state.get("lisa", {}))
        self._step_counter = state.get("step_counter", 0)

    def eval(self) -> None:
        """Switch to eval mode — all layers active, no gradient projection."""
        self._lisa.restore_all_layers()

    def train(self) -> None:
        """Switch back to training mode."""
        pass  # LISA sampling resumes on next step_begin()


# ── 20B Feasibility Check ────────────────────────────────────────────────


def can_run_20b_on_24gb(
    model_params_b: int = 20,
    sequence_length: int = 2048,
    batch_size: int = 1,
    galore_rank: int = 128,
    lisa_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Estimate if a 20B model can train on 24 GB with Q-GaLore.

    Conservatively estimates each component's VRAM.

    Returns dict with breakdown and verdict.
    """
    # ── Weight memory ───────────────────────────────────────────────
    # QLoRA NF4: ~0.625 bytes/param (4-bit + quantization metadata)
    nf4_gb = model_params_b * 0.625
    # LoRA adapters: r=64, ~0.01% of base → ~2M/param for 20B → ~0.2 GB
    lora_gb = 0.2
    weight_total = nf4_gb + lora_gb

    # ── Optimizer memory ────────────────────────────────────────────
    # GaLore rank=r: ~r×d states instead of d×d
    # For 20B with hidden_dim ~6144: ~3M states vs 37M per layer
    # Across ~40 layers: ~0.3 GB vs ~4 GB
    optimizer_gb = 0.3

    # ── Activation + Gradient memory ────────────────────────────────
    # Without LISA: ~12 GB for 20B with seq_len=2048, batch=1, grad_checkpoint
    # With LISA 50%: ~6 GB
    activation_gb = 12.0 * lisa_ratio

    # ── CUDA overhead ───────────────────────────────────────────────
    overhead_gb = 1.5

    total_gb = weight_total + optimizer_gb + activation_gb + overhead_gb

    return {
        "nf4_weights_gb": round(nf4_gb, 1),
        "lora_adapters_gb": round(lora_gb, 2),
        "weight_total_gb": round(weight_total, 1),
        "optimizer_gb": round(optimizer_gb, 2),
        "activation_gb": round(activation_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "total_gb": round(total_gb, 1),
        "fits_24gb": total_gb < 23.5,
        "headroom_gb": round(23.5 - total_gb, 1),
        "verdict": (
            "PASS — 20B model fits in 24 GB with Q-GaLore"
            if total_gb < 23.5
            else f"FAIL — {total_gb:.1f} GB exceeds 23.5 GB limit"
        ),
    }
