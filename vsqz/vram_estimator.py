"""
Unified VRAM Estimator
=======================
Calculates VRAM consumption for any combination of the supported techniques.
Use to plan model size, batch size, and technique selection before training.

Usage:
    from vsqz import VRAMEstimator
    est = VRAMEstimator(model_params_b=13, batch_size=2, seq_len=2048)
    est.with_technique("galore", rank=128)
    est.with_technique("lisa", ratio=0.5)
    est.with_technique("fp16_states")
    print(est.summary())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class VRAMEstimator:
    """Estimate VRAM usage with any combination of VRAM-squeeze techniques.

    Configuration presets:
      - "13B_24GB": 13B model, batch=2, all techniques active
      - "9B_24GB_max": 9B model, batch=2, galore+lisa (safe defaults)
      - "20B_borderline": 20B model, batch=1, all techniques active (tight)
    """

    PRESETS = {
        "13B_24GB": {
            "model_params_b": 13, "batch_size": 2, "seq_len": 2048,
            "techniques": {"galore": 128, "lisa": 0.5, "fp16_states": True},
        },
        "9B_24GB_max": {
            "model_params_b": 9, "batch_size": 2, "seq_len": 2048,
            "techniques": {"galore": 128, "lisa": 0.5, "int8_states": True},
        },
        "20B_borderline": {
            "model_params_b": 20, "batch_size": 1, "seq_len": 2048,
            "techniques": {"galore": 128, "lisa": 0.5, "fp16_states": True},
        },
    }

    def __init__(
        self,
        model_params_b: float = 9,
        batch_size: int = 1,
        seq_len: int = 2048,
        preset: Optional[str] = None,
    ):
        if preset and preset in self.PRESETS:
            p = self.PRESETS[preset]
            model_params_b = p["model_params_b"]
            batch_size = p["batch_size"]
            seq_len = p["seq_len"]
            self._techniques: Dict[str, Any] = dict(p["techniques"])
        else:
            self._techniques: Dict[str, Any] = {}

        self._model_params_b = model_params_b
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._total_gb = 0.0
        self._breakdown: Dict[str, float] = {}
        self._calculate()

    def with_technique(self, name: str, **kwargs):
        """Add a technique to the estimator before calculation."""
        self._techniques[name] = kwargs.get(name, True)
        self._calculate()
        return self

    def _calculate(self) -> None:
        """Compute VRAM budget with active techniques."""
        params_b = self._model_params_b
        batch = self._batch_size
        seq = self._seq_len

        # Base weights (QLoRA NF4)
        nf4_gb = params_b * 0.625
        lora_gb = 0.2  # LoRA adapters r=64

        # Optimizer states (AdamW m+v)
        # LoRA trainable params ≈ 0.01% of base model
        trainable_m = params_b * 1e9 * 0.0001  # ~1M for 13B model
        opt_state_bytes = trainable_m * 4 * 2  # FP32 × 2 states

        # Apply techniques
        if "fp16_states" in self._techniques:
            opt_state_bytes *= 0.5  # BF16 halves
        if "int8_states" in self._techniques:
            opt_state_bytes *= 0.5  # 8-bit further halves
        if "deepspeed_offload" in self._techniques:
            opt_state_bytes *= 0.0  # All to CPU
        if "galore" in self._techniques:
            rank = self._techniques.get("galore", 128)
            # GaLore compression ratio for a dense layer
            hidden = int(params_b * 5000 ** 0.5)  # rough hidden_dim
            full_states = 2 * hidden * hidden * 4
            compressed = 2 * rank * (hidden + hidden) * 4
            galore_ratio = compressed / max(full_states, 1)
            opt_state_bytes *= galore_ratio

        opt_gb = opt_state_bytes / (1024 ** 3)

        # Activations
        # Rough: 2× hidden × batch × seq / offloading_factor
        hidden_dim = 4096  # conservative
        act_factor = 8.0  # bytes per activation element (incl. gradient checkpoints)
        act_gb = (hidden_dim * batch * seq * act_factor * 40) / (1024 ** 3)  # 40 layers

        if "lisa" in self._techniques:
            ratio = self._techniques.get("lisa", 0.5)
            act_gb *= ratio

        # Activation offloading (QLoRA default)
        act_gb *= 0.7

        overhead_gb = 1.5  # CUDA context + cuBLAS workspaces

        total = nf4_gb + lora_gb + opt_gb + act_gb + overhead_gb

        self._breakdown = {
            "nf4_weights": round(nf4_gb, 2),
            "lora_adapters": round(lora_gb, 2),
            "optimizer_states": round(opt_gb, 2),
            "activations": round(act_gb, 2),
            "overhead": round(overhead_gb, 2),
        }
        self._total_gb = round(total, 2)

    def summary(self) -> Dict[str, Any]:
        return {
            "model_params_b": self._model_params_b,
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "techniques": list(self._techniques.keys()),
            "breakdown": dict(self._breakdown),
            "total_gb": self._total_gb,
            "fits_24gb": self._total_gb < 23.5,
            "headroom_gb": round(23.5 - self._total_gb, 2),
            "verdict": (
                f"PASS — {self._total_gb} GB fits in 24 GB"
                if self._total_gb < 23.5
                else f"FAIL — {self._total_gb} GB exceeds 23.5 GB limit"
            ),
        }

    def __repr__(self) -> str:
        s = self.summary()
        lines = [
            f"VRAM Estimator — {s['model_params_b']}B, batch={s['batch_size']}",
            f"  Techniques: {', '.join(s['techniques']) or 'none'}",
        ]
        for k, v in s["breakdown"].items():
            lines.append(f"  {k}: {v} GB")
        lines.append(f"  ─────────────────")
        lines.append(f"  Total: {s['total_gb']} GB | Headroom: {s['headroom_gb']} GB")
        lines.append(f"  {s['verdict']}")
        return "\n".join(lines)


def estimate_technique_savings(
    model_params_b: float = 13,
    batch_size: int = 2,
    seq_len: int = 2048,
) -> Dict[str, Dict[str, float]]:
    """Compare VRAM savings of each technique individually."""
    results = {}
    for name in ("galore", "lisa", "fp16_states", "int8_states", "deepspeed_offload"):
        est_base = VRAMEstimator(model_params_b, batch_size, seq_len)
        est_with = VRAMEstimator(model_params_b, batch_size, seq_len)
        est_with.with_technique(name)
        saved = est_base._total_gb - est_with._total_gb
        results[name] = {
            "base_gb": est_base._total_gb,
            "with_gb": est_with._total_gb,
            "saved_gb": round(saved, 2),
        }
    return results
