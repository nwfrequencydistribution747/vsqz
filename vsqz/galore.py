#!/usr/bin/env python3
"""
GaLore — Gradient Low-Rank Projection Optimizer
==================================================
Paper: "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
       Jiawei Zhao et al., ICML 2024

Core insight: Gradients become low-rank during training. Instead of storing
full AdamW optimizer states (2×d×d entries), project gradients to rank r via
SVD and run AdamW on the compressed representation.

VRAM savings: 2×m×n → 2×r×(m+n)  (typically 20-30× reduction for AdamW states)

Integration: Wraps a standard optimizer (AdamW). Transparent to training loop.

Usage:
    from training.ga_lore import GaLoreWrapper
    optimizer = GaLoreWrapper(model, base_optimizer=AdamW, rank=128, scale=1.0)
    # Training loop unchanged: loss.backward(), optimizer.step(), optimizer.zero_grad()
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger("GaLore")

_SUPPORTED_LAYERS = {nn.Linear, nn.Embedding}
_NORM_LAYERS = {
    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
    nn.GroupNorm, nn.RMSNorm,
}


def _extract_weight(module: nn.Module, name: str) -> Optional[torch.nn.Parameter]:
    """Safely extract a weight parameter from a module by name."""
    param = getattr(module, name, None)
    if isinstance(param, torch.nn.Parameter):
        return param
    return None


def _layer_rank(shape: Tuple[int, ...], max_rank: int) -> int:
    """Determine rank for a given weight shape — proportional to min(m,n)."""
    if len(shape) != 2:
        return min(max_rank, shape[0])  # e.g. embedding
    m, n = shape
    return min(max_rank, m // 8, n // 8)


class GaLoreModule:
    """Per-layer GaLore state: stores compressed gradient projections + AdamW states."""

    def __init__(self, weight: torch.nn.Parameter, rank: int, scale: float = 1.0):
        self.weight = weight
        self.shape = weight.shape  # (m, n) or (vocab, dim)
        self.rank = rank
        self.scale = scale
        self.dtype = weight.dtype
        self.device = weight.device

        # Only 2D weights (Linear layers) benefit from SVD projection
        if len(self.shape) == 2:
            m, n = self.shape
            r = rank
            # Projected gradient: G ≈ P @ Q  where P ∈ R^{m×r}, Q ∈ R^{r×n}
            # AdamW state is stored on P and Q separately
            self.P: Optional[torch.Tensor] = None
            self.Q: Optional[torch.Tensor] = None
            # AdamW states for P
            self.m_P: Optional[torch.Tensor] = None
            self.v_P: Optional[torch.Tensor] = None
            # AdamW states for Q
            self.m_Q: Optional[torch.Tensor] = None
            self.v_Q: Optional[torch.Tensor] = None
        else:
            # 1D weights (bias, norm) — fallback to full AdamW
            self.m_full: Optional[torch.Tensor] = None
            self.v_full: Optional[torch.Tensor] = None

    def project_gradient(self, G: torch.Tensor) -> None:
        """Decompose gradient G ≈ P @ Q via truncated SVD."""
        if len(G.shape) != 2:
            return  # Skip non-2D

        with torch.no_grad():
            if G.shape[0] >= G.shape[1]:
                # Tall matrix: SVD on G
                U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
                r = min(self.rank, len(S))
                U_r = U[:, :r]
                S_r = S[:r]
                Vh_r = Vh[:r, :]
            else:
                # Wide matrix: SVD on G^T for efficiency
                U, S, Vh = torch.linalg.svd(G.T.float(), full_matrices=False)
                r = min(self.rank, len(S))
                Vh_r = U[:, :r].T
                S_r = S[:r]
                U_r = Vh[:r, :].T

            # Normalize by sqrt(scale) to control magnitude
            norm = torch.sqrt(S_r.sum()) / r if S_r.sum() > 0 else 1.0
            scale_sqrt = math.sqrt(self.scale) / (norm + 1e-8)

            self.P = U_r * S_r.sqrt().unsqueeze(0) * scale_sqrt  # (m, r)
            self.Q = S_r.sqrt().unsqueeze(1) * Vh_r * scale_sqrt  # (r, n)

    def reconstruct_gradient(self) -> Optional[torch.Tensor]:
        """Reconstruct full gradient: G̃ = P @ Q."""
        if self.P is None or self.Q is None:
            return None
        return (self.P @ self.Q).to(dtype=self.dtype)


class GaLoreWrapper(torch.optim.Optimizer):
    """Memory-efficient AdamW via gradient low-rank projection.

    Replaces full-rank optimizer states (2 × m × n per weight matrix) with
    low-rank projections (2 × r × (m + n)), typically 20-30× smaller.

    Usage:
        base_opt = torch.optim.AdamW(..., foreach=False)
        wrapper = GaLoreWrapper(model, base_optimizer=base_opt, rank=128)
        # Training loop:
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            wrapper.step()          # Replaces optimizer.step()
            wrapper.zero_grad()     # Replaces optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        rank: int = 128,
        scale: float = 1.0,
        update_gap: int = 1,
        excluded_param_names: Optional[Set[str]] = None,
    ):
        self.model = model
        self.rank = rank
        self.scale = scale
        self.update_gap = update_gap
        self.step_counter = 0
        self._excluded = excluded_param_names or set()

        # Identify which parameters to apply GaLore to
        self._galore_modules: Dict[int, GaLoreModule] = {}
        self._full_params: List[torch.nn.Parameter] = []

        # Separate optimizer for full-rank params (bias, norm, excluded)
        self._full_optimizer = base_optimizer

        # Inherit param_groups from base optimizer (includes 'lr' for scheduler)
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state

        self._init_modules()

        logger.info(
            "GaLore: rank=%d scale=%.2f gap=%d — %d layers compressed, %d full-rank",
            rank, scale, update_gap, len(self._galore_modules), len(self._full_params),
        )

    def _init_modules(self) -> None:
        """Walk model parameters and assign GaLore modules to 2D weights."""
        for name, module in self.model.named_modules():
            if any(excl in name for excl in self._excluded):
                continue

            for pname, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{pname}" if name else pname

                if not param.requires_grad:
                    continue
                if pname in self._excluded or full_name in self._excluded:
                    continue
                if pname == "weight" and isinstance(module, tuple(_NORM_LAYERS)):
                    self._full_params.append(param)
                    param._galore_full = True
                    continue
                if pname == "bias" or pname == "weight" and isinstance(module, tuple(_NORM_LAYERS)):
                    self._full_params.append(param)
                    param._galore_full = True
                    continue

                # LoRA adapters (lora_A, lora_B) are small — apply GaLore
                # to save optimizer VRAM on these trainable weights
                # Base model weights are frozen (QLoRA) — no optimizer state needed
                if isinstance(module, tuple(_SUPPORTED_LAYERS)):
                    layer_rank = _layer_rank(param.shape, self.rank)
                    if layer_rank > 0 and len(param.shape) == 2:
                        gm = GaLoreModule(param, layer_rank, self.scale)
                        self._galore_modules[id(param)] = gm
                        param._galore_module = gm
                        continue

                # Default: full-rank
                self._full_params.append(param)

        # Build parameter group for full optimizer
        if self._full_params:
            self._full_optimizer.param_groups[0]["params"] = self._full_params

    def step(self, closure=None) -> Optional[float]:
        """Perform one optimization step: project gradients → compressed AdamW → reconstruct."""
        self.step_counter += 1

        # ── 1. Project gradients to low rank + run compressed AdamW ──
        for param_id, gm in self._galore_modules.items():
            param = gm.weight
            if param.grad is None:
                continue

            G = param.grad.data
            if len(G.shape) != 2:
                continue

            # SVD projection to low rank
            gm.project_gradient(G)

            # Run AdamW on compressed components P and Q
            if gm.P is None or gm.Q is None:
                continue

            lr = self._full_optimizer.param_groups[0]["lr"]
            wd = self._full_optimizer.param_groups[0].get("weight_decay", 0.0)
            betas = self._full_optimizer.param_groups[0].get("betas", (0.9, 0.999))
            eps = self._full_optimizer.param_groups[0].get("eps", 1e-8)

            for comp, m_comp, v_comp, comp_name in [
                (gm.P, gm.m_P, gm.v_P, "P"),
                (gm.Q, gm.m_Q, gm.v_Q, "Q"),
            ]:
                if comp is None:
                    continue
                comp_grad = comp if comp.grad is None else comp.grad
                if comp_grad is None:
                    comp_grad = comp  # initial step: use component directly

                # Init AdamW state
                if m_comp is None:
                    gm.__dict__[f"m_{comp_name}"] = torch.zeros_like(comp)
                    gm.__dict__[f"v_{comp_name}"] = torch.zeros_like(comp)
                    m_comp = gm.__dict__[f"m_{comp_name}"]
                    v_comp = gm.__dict__[f"v_{comp_name}"]

                # AdamW update
                m_comp.mul_(betas[0]).add_(comp_grad, alpha=1 - betas[0])
                v_comp.mul_(betas[1]).addcmul_(comp_grad, comp_grad, value=1 - betas[1])

                # Bias correction
                m_hat = m_comp / (1 - betas[0] ** self.step_counter)
                v_hat = v_comp / (1 - betas[1] ** self.step_counter)

                # Weight decay
                if wd > 0:
                    comp.mul_(1 - lr * wd)

                # Update compressed component
                comp.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

            # ── Reconstruct full gradient from compressed components ──
            G_reconstructed = gm.reconstruct_gradient()
            if G_reconstructed is not None:
                param.grad.data.copy_(G_reconstructed)

            # Recompute P and Q after AdamW update (for next step)
            gm.project_gradient(G_reconstructed if G_reconstructed is not None else G)

        # ── 2. Standard optimizer step on full-rank params ──
        loss = self._full_optimizer.step(closure)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero all gradients."""
        self._full_optimizer.zero_grad(set_to_none=set_to_none)
        for gm in self._galore_modules.values():
            if gm.weight.grad is not None:
                gm.weight.grad = None

    def state_dict(self) -> Dict[str, Any]:
        """Serialize GaLore state for checkpointing."""
        return {
            "rank": self.rank,
            "scale": self.scale,
            "update_gap": self.update_gap,
            "step_counter": self.step_counter,
            "num_galore_layers": len(self._galore_modules),
            "num_full_params": len(self._full_params),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore GaLore state from checkpoint."""
        self.step_counter = state.get("step_counter", 0)


# ── Estimator for VRAM planning ──────────────────────────────────────────


def estimate_galore_vram_savings(
    model: nn.Module,
    galore_rank: int = 128,
    bytes_per_param: int = 4,
) -> Dict[str, float]:
    """Estimate VRAM savings from GaLore on a given model.

    Returns dict with: optimizer_vram_before_gb, optimizer_vram_after_gb, saved_gb.
    """
    total_full = 0
    total_compressed = 0

    for name, module in model.named_modules():
        if isinstance(module, tuple(_SUPPORTED_LAYERS)):
            for pname, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if pname == "weight" and len(param.shape) == 2:
                    m, n = param.shape
                    r = _layer_rank((m, n), galore_rank)
                    # Full AdamW: 2 states × m × n
                    total_full += 2 * m * n
                    # GaLore: 2 states × (m×r + r×n) for P and Q
                    total_compressed += 2 * (m * r + r * n)
                else:
                    total_full += 2 * param.numel()
                    total_compressed += 2 * param.numel()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.RMSNorm)):
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    total_full += 2 * param.numel()
                    total_compressed += 2 * param.numel()

    gb_full = total_full * bytes_per_param / (1024 ** 3)
    gb_compressed = total_compressed * bytes_per_param / (1024 ** 3)
    saved = gb_full - gb_compressed

    return {
        "optimizer_vram_before_gb": round(gb_full, 2),
        "optimizer_vram_after_gb": round(gb_compressed, 2),
        "saved_gb": round(saved, 2),
        "compression_ratio": round(gb_full / max(gb_compressed, 1e-8), 1),
    }
