"""Multi-model inference with shared base + delta reference-swap."""
from __future__ import annotations

import json
import os
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .converter_io import _load_source, _fmt_bytes
from .vsqz_format import _read_vsqz


class ModelSwarm:
    """Load base once, apply deltas for multi-model inference.

    Usage:
        swarm = ModelSwarm("qwen-base.vsqz", ["current.delta.vsqz", "best.delta.vsqz"])
        swarm.load()
        print(swarm.status())
        output = swarm.generate("current", "What is the capital of France?")
    """

    def __init__(self, base_path: str, delta_paths: List[str], device: str = "cuda"):
        self.base_path = Path(base_path)
        self.delta_paths = [Path(d) for d in delta_paths]
        self.device = device
        self._base_tensors: Dict[str, np.ndarray] = {}
        self._models: Dict[str, Dict[str, np.ndarray]] = {}
        self._loaded = False
        self._base_size = 0

    def load(self, quiet: bool = False) -> "ModelSwarm":
        """Load base tensors and apply all deltas."""
        t0 = _time.time()

        # Load base
        base_src = str(self.base_path)
        if base_src.endswith('.vsqz'):
            h, td = _read_vsqz(base_src)
            for name in sorted(h["tensors"]):
                e = h["tensors"][name]
                d = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(
                    e.get("dtype", "float16"), np.float16)
                self._base_tensors[name] = np.frombuffer(td[name], dtype=d).reshape(e["shape"])
        elif base_src.endswith('.gguf'):
            # Use streaming GGUF reader — max 1 tensor in RAM
            from .stream_diff import _stream_gguf
            for name, tensor in _stream_gguf(base_src):
                self._base_tensors[name] = tensor
        elif self.base_path.is_dir() or self.base_path.suffix in (".safetensors", ".bin", ".pt", ".pth"):
            self._base_tensors, _ = _load_source(self.base_path)
        else:
            raise ValueError(f"Unsupported base format: {base_src}")

        self._base_size = sum(t.nbytes for t in self._base_tensors.values())

        # Load base model
        model_name = self.base_path.stem.replace('.vsqz', '').replace('.gguf', '')
        self._models[model_name] = self._base_tensors
        if not quiet:
            print(f"  Base: {model_name} ({_fmt_bytes(self._base_size)} in {_time.time()-t0:.1f}s)")

        # Apply deltas
        for dp in self.delta_paths:
            if dp.suffix == '.vsqz':
                h, td = _read_vsqz(str(dp))
                if not h.get("delta"):
                    print(f"  ⚠️  {dp.name} is not a delta file — skipping")
                    continue
                # Verify SHA if available
                if h.get("base_sha256"):
                    import hashlib
                    expected = h["base_sha256"]
                    actual = hashlib.sha256(
                        b"".join(n.encode() + self._base_tensors[n].astype(np.float16).tobytes()
                                 for n in sorted(self._base_tensors))
                    ).hexdigest()
                    if expected != actual:
                        print(f"  ⚠️  {dp.name}: BASE SHA MISMATCH — wrong base? Skipping.")
                        continue

                merged = {k: v.copy() for k, v in self._base_tensors.items()}
                for name in sorted(td):
                    e = h["tensors"][name]
                    d = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(
                        e.get("dtype", "float16"), np.float16)
                    merged[name] = np.frombuffer(td[name], dtype=d).reshape(e["shape"])

                dname = dp.stem.replace('.delta', '').replace('.vsqz', '')
                self._models[dname] = merged
                if not quiet:
                    delta_mb = sum(t.nbytes for t in merged.values()) - self._base_size
                    print(f"  + {dname}: delta {_fmt_bytes(Path(dp).stat().st_size)} → model {_fmt_bytes(sum(t.nbytes for t in merged.values()))}")

        self._loaded = True
        if not quiet:
            print(f"  Loaded {len(self._models)} models in {_time.time()-t0:.1f}s")
        return self

    @property
    def models(self) -> List[str]:
        return list(self._models.keys())

    def status(self) -> str:
        """CLI status output."""
        if not self._loaded:
            return "Not loaded. Call .load() first."

        total_vram = sum(t.nbytes for t in next(iter(self._models.values())).values())
        model_count = len(self._models)
        without_vsqz = model_count * self._base_size if self._base_size > 0 else 0
        saved = without_vsqz - total_vram if without_vsqz > 0 else 0
        pct = (saved / without_vsqz * 100) if without_vsqz > 0 else 0

        lines = [
            f"",
            f"  Models loaded: {model_count}",
            f"  VRAM used:     {_fmt_bytes(total_vram)}",
        ]
        if without_vsqz > 0:
            lines.append(f"  Without vsqz:  {_fmt_bytes(without_vsqz)}")
            lines.append(f"  Saved:         {_fmt_bytes(saved)} ({pct:.0f}%)")
        lines.append(f"  Base:          {self.models[0]} ({_fmt_bytes(self._base_size)} shared)")
        for name in self.models[1:]:
            size = sum(t.nbytes for t in self._models[name].values())
            lines.append(f"  Delta:         {name} ({_fmt_bytes(size)} total)")
        return "\n".join(lines)

    def status_json(self) -> dict:
        if not self._loaded:
            return {"loaded": False}
        total_vram = sum(t.nbytes for t in next(iter(self._models.values())).values())
        without_vsqz = len(self._models) * self._base_size if self._base_size > 0 else 0
        return {
            "models_loaded": len(self._models),
            "vram_used": _fmt_bytes(total_vram),
            "vram_without_vsqz": _fmt_bytes(without_vsqz),
            "vram_saved": _fmt_bytes(without_vsqz - total_vram),
            "base": f"{self.models[0]} ({_fmt_bytes(self._base_size)} shared)",
            "models": {n: _fmt_bytes(sum(t.nbytes for t in self._models[n].values())) for n in self.models},
        }

    def get_state_dict(self, model_name: str) -> Optional[Dict[str, np.ndarray]]:
        return self._models.get(model_name)

    def get_tensors(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """Return (base_tensors, {name: merged_tensors}). For HF model loading."""
        return self._base_tensors, self._models
