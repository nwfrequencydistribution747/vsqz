"""Multi-model inference with shared base + delta reference-swap (GPU-native)."""
from __future__ import annotations

import json
import os
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .converter_io import _load_source, _fmt_bytes
from .vsqz_format import _read_vsqz


def _torch_available():
    try:
        import torch
        return torch
    except ImportError:
        return None


class ModelSwarm:
    """Load base once on GPU, apply deltas for multi-model inference.

    Usage:
        swarm = ModelSwarm("qwen-base.vsqz", ["current.delta.vsqz", "best.delta.vsqz"])
        swarm.load()
        print(swarm.status())
    """

    def __init__(self, base_path: str, delta_paths: List[str], device: str = "cuda"):
        self.base_path = Path(base_path)
        self.delta_paths = [Path(d) for d in delta_paths]
        self.device = device
        self._base_tensors: Dict[str, "torch.Tensor"] = {}
        self._models: Dict[str, Dict[str, "torch.Tensor"]] = {}
        self._loaded = False
        self._base_size = 0
        self._base_sha = ""  # File-level SHA from base .vsqz (for delta verification)

    @property
    def _torch(self):
        t = _torch_available()
        if t is None:
            raise ImportError("PyTorch required for GPU model loading. pip install torch")
        return t

    @property
    def _on_gpu(self) -> bool:
        return self.device == "cuda" and self._torch.cuda.is_available()

    def load(self, quiet: bool = False) -> "ModelSwarm":
        """Load base tensors on GPU, apply all deltas. Base shared across models."""
        t0 = _time.time()
        device = "cuda" if self._on_gpu else "cpu"

        # ── Load base as numpy first (mmap/stream from file) ──────────
        base_src = str(self.base_path)
        base_np: Dict[str, np.ndarray] = {}
        self._base_sha = ""
        if base_src.endswith('.vsqz'):
            h, td = _read_vsqz(base_src)
            self._base_sha = h.get("sha256", "")
            for name in sorted(h["tensors"]):
                e = h["tensors"][name]
                d = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(
                    e.get("dtype", "float16"), np.float16)
                base_np[name] = np.frombuffer(td[name], dtype=d).reshape(e["shape"])
        elif base_src.endswith('.gguf'):
            from .stream_diff import _stream_gguf
            for name, tensor in _stream_gguf(base_src):
                base_np[name] = tensor
        elif self.base_path.is_dir() or self.base_path.suffix in (".safetensors", ".bin", ".pt", ".pth"):
            base_np, _ = _load_source(self.base_path)
        else:
            raise ValueError(f"Unsupported base format: {base_src}")

        # ── Move base to GPU (shared single copy) ─────────────────────
        torch = self._torch
        for name in sorted(base_np):
            t = torch.from_numpy(base_np[name].copy())  # copy to own memory
            self._base_tensors[name] = t.to(device=device)

        self._base_size = sum(t.numel() * t.element_size() for t in self._base_tensors.values())
        del base_np  # free CPU copy

        model_name = self.base_path.stem.replace('.vsqz', '').replace('.gguf', '')
        self._models[model_name] = self._base_tensors  # same dict = same GPU memory
        if not quiet:
            extra = f" on {device.upper()}" if self._on_gpu else " on CPU"
            print(f"  Base: {model_name} ({_fmt_bytes(self._base_size)}{extra} in {_time.time()-t0:.1f}s)")

        # ── Apply deltas ──────────────────────────────────────────────
        for dp in self.delta_paths:
            if dp.suffix == '.vsqz':
                h, td = _read_vsqz(str(dp))
                if not h.get("delta"):
                    print(f"  ⚠️  {dp.name} is not a delta file — skipping")
                    continue

                # Verify base SHA — use authoritative file-level SHA from base
                if h.get("base_sha256"):
                    if h["base_sha256"] != self._base_sha:
                        print(f"  ⚠️  {dp.name}: BASE SHA MISMATCH — wrong base? Skipping.")
                        continue

                # Build merged model: base tensors by reference, delta tensors as new GPU tensors
                merged: Dict[str, "torch.Tensor"] = {}
                for name in sorted(self._base_tensors):
                    merged[name] = self._base_tensors[name]  # SHARED GPU pointer

                delta_new_bytes = 0
                for name in sorted(td):
                    e = h["tensors"][name]
                    d = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(
                        e.get("dtype", "float16"), np.float16)
                    delta_t = torch.from_numpy(
                        np.frombuffer(td[name], dtype=d).reshape(e["shape"]).copy()
                    ).to(device=device)
                    if name in merged:
                        delta_new_bytes += delta_t.numel() * delta_t.element_size()
                    merged[name] = delta_t  # overwrites shared base with delta-specific tensor

                dname = dp.stem.replace('.delta', '').replace('.vsqz', '')
                self._models[dname] = merged
                if not quiet:
                    total_mb = sum(t.numel() * t.element_size() for t in merged.values())
                    print(f"  + {dname}: +{_fmt_bytes(delta_new_bytes)} → {_fmt_bytes(total_mb)} total")

        self._loaded = True
        if not quiet:
            vram_total = self._gpu_vram_used()
            base_shared = self._base_size
            delta_only = vram_total - base_shared
            saved = (sum(t.numel() * t.element_size() for t in next(iter(self._models.values())).values())
                     * (len(self._models) - 1))
            print(f"  Loaded {len(self._models)} models in {_time.time()-t0:.1f}s")
            print(f"  GPU VRAM: base={_fmt_bytes(base_shared)} + deltas={_fmt_bytes(delta_only)} "
                  f"= {_fmt_bytes(vram_total)} (saved {_fmt_bytes(saved)} via sharing)")
        return self

    def _gpu_vram_used(self) -> int:
        """Total unique GPU VRAM used (base + unique delta tensors)."""
        if not self._loaded:
            return 0
        # Unique tensors by id()
        seen = set()
        total = 0
        for name in sorted(self._base_tensors):
            tid = id(self._base_tensors[name])
            if tid not in seen:
                seen.add(tid)
                total += self._base_tensors[name].numel() * self._base_tensors[name].element_size()
        for mname, model in self._models.items():
            for tname, tensor in model.items():
                tid = id(tensor)
                if tid not in seen:
                    seen.add(tid)
                    total += tensor.numel() * tensor.element_size()
        return total

    @property
    def models(self) -> List[str]:
        return list(self._models.keys())

    def status(self) -> str:
        """CLI status output."""
        if not self._loaded:
            return "Not loaded. Call .load() first."

        model_count = len(self._models)
        total_vram = self._gpu_vram_used()
        without_vsqz = model_count * self._base_size if self._base_size > 0 else 0
        saved = without_vsqz - total_vram if without_vsqz > 0 else 0
        pct = (saved / without_vsqz * 100) if without_vsqz > 0 else 0
        loc = "GPU" if self._on_gpu else "CPU"

        lines = [
            f"",
            f"  Models loaded: {model_count}",
            f"  Location:      {loc}",
            f"  VRAM used:     {_fmt_bytes(total_vram)}",
        ]
        if without_vsqz > 0:
            lines.append(f"  Without vsqz:  {_fmt_bytes(without_vsqz)}")
            lines.append(f"  Saved:         {_fmt_bytes(saved)} ({pct:.0f}%)")
        lines.append(f"  Base:          {self.models[0]} ({_fmt_bytes(self._base_size)} shared)")
        for name in self.models[1:]:
            size = sum(t.numel() * t.element_size() for t in self._models[name].values())
            lines.append(f"  Delta:         {name} ({_fmt_bytes(size)} total)")
        return "\n".join(lines)

    def status_json(self) -> dict:
        if not self._loaded:
            return {"loaded": False}
        model_count = len(self._models)
        total_vram = self._gpu_vram_used()
        without_vsqz = model_count * self._base_size if self._base_size > 0 else 0
        return {
            "models_loaded": model_count,
            "location": "GPU" if self._on_gpu else "CPU",
            "vram_used": _fmt_bytes(total_vram),
            "vram_without_vsqz": _fmt_bytes(without_vsqz),
            "vram_saved": _fmt_bytes(without_vsqz - total_vram),
            "base": f"{self.models[0]} ({_fmt_bytes(self._base_size)} shared)",
            "models": {n: _fmt_bytes(sum(t.numel() * t.element_size() for t in self._models[n].values()))
                      for n in self.models},
        }

    def get_state_dict(self, model_name: str) -> Optional[Dict[str, "torch.Tensor"]]:
        return self._models.get(model_name)

    def get_tensors(self) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, Dict[str, "torch.Tensor"]]]:
        """Return (base_tensors, {name: merged_tensors})."""
        return self._base_tensors, self._models
