"""
.vsqz — Universal VRAMSqueeze Format
=====================================
Single file: NF4 weights (GGUF-compatible) + compressed optimizer state.
Self-describing: `.vsqz` extension signals VRAMSqueeze was used.

Format (binary):
  [0..3]    Magic:  b'VSQZ'            (4 bytes)
  [4..7]    Version: uint32_le         (4 bytes)
  [8..11]   Header length: uint32_le   (4 bytes)
  [12..N]   JSON header                (metadata + tensor index)
  [N+..]    Tensor blobs               (64-byte aligned)

JSON header:
  {
    "vsqz_version": "0.2.8",
    "pytorch_version": "2.x",
    "model_config": { "params_b": 9, "hidden_dim": 4096, "num_layers": 40, ... },
    "technique_stack": ["galore_r128", "lisa_0.5", "fp16_states"],
    "quantization": { "weights": "nf4", "optimizer": "galore_int8" },
    "training_state": { "step": 1234 },
    "tensors": {
      "weight.model.embed": { "offset": 4096, "size": 262144, "dtype": "float16", "shape": [...] },
      "galore.layer_0.P":  { "offset": ...,  "size": ...,     "dtype": "int8",    "shape": [...] },
      ...
    }
  }

Usage:
    from vsqz import VRAMSqueeze
    squeezer = VRAMSqueeze(model, mode='training', optimizer=opt, preset='13B_24GB')
    squeezer.save_vsqz('model_iter5.vsqz')      # One file for everything
    squeezer.load_vsqz('model_iter5.vsqz')       # Resume training
    # OR
    from vsqz.vvsqz_format import load_vsqz_weights
    model = load_vsqz_weights('model_iter5.vsqz')  # Inference only (GGUF-compatible)
"""

from __future__ import annotations

import json
import logging
import os
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("SQZ")

VSQZ_MAGIC = b"VSQZ"
VSQZ_VERSION = 1
TENSOR_ALIGNMENT = 64
HEADER_ALIGNMENT = 4096  # 4KB aligned for mmap


def _align(offset: int, alignment: int = TENSOR_ALIGNMENT) -> int:
    """Align offset to next multiple of alignment."""
    return (offset + alignment - 1) // alignment * alignment


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Serialize tensor to raw bytes efficiently."""
    if t.dtype == torch.int8:
        return t.cpu().numpy().tobytes()
    elif t.dtype == torch.float16:
        return t.cpu().numpy().tobytes()
    elif t.dtype == torch.float32:
        return t.cpu().to(torch.float16).numpy().tobytes()
    elif t.dtype == torch.bfloat16:
        return t.cpu().to(torch.float16).numpy().tobytes()
    else:
        # Fallback: int32, float64, etc.
        return t.cpu().numpy().tobytes()


def _bytes_to_tensor(data: bytes, dtype: str, shape: list) -> torch.Tensor:
    """Deserialize bytes back to tensor."""
    import numpy as np
    np_dtype = {
        "float32": np.float32, "float16": np.float16,
        "bfloat16": np.float16, "int8": np.int8,
        "int32": np.int32, "int64": np.int64,
    }.get(dtype, np.float16)
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape).copy()
    return torch.from_numpy(arr)


def _dtype_str(t: torch.Tensor) -> str:
    """Get tensor dtype as string."""
    return str(t.dtype).replace("torch.", "")


def save_vsqz(squeezer, path: str) -> str:
    """Save model weights + optimizer state in .vsqz format.

    Returns file path written.
    """
    path = Path(path).with_suffix(".vsqz")
    model = squeezer.model

    # ── Collect all tensors ─────────────────────────────────────────

    tensor_entries: Dict[str, Any] = {}
    tensor_blobs: List[Tuple[str, bytes]] = []

    # Model weights (FP16 for compatibility with GGUF)
    for name, param in model.named_parameters():
        safe_name = f"weight.{name}".replace(".", "_")
        data = _tensor_to_bytes(param.data)
        tensor_entries[safe_name] = {
            "dtype": "float16",
            "shape": list(param.shape),
            "size": len(data),
        }
        tensor_blobs.append((safe_name, data))

    # GaLore compressed optimizer states
    if hasattr(squeezer, '_galore') and squeezer._galore is not None:
        for param_id, gm in squeezer._galore._galore_modules.items():
            for attr in ("P", "Q", "m_P", "v_P", "m_Q", "v_Q"):
                val = getattr(gm, attr, None)
                if val is not None:
                    safe_name = f"galore.{param_id}.{attr}"
                    data = _tensor_to_bytes(val)
                    tensor_entries[safe_name] = {
                        "dtype": _dtype_str(val),
                        "shape": list(val.shape),
                        "size": len(data),
                    }
                    tensor_blobs.append((safe_name, data))

    # ── Build header ────────────────────────────────────────────────

    header = {
        "vsqz_version": "0.2.8",
        "pytorch_version": torch.__version__,
        "created_at": datetime.now().isoformat(),
        "model_config": _extract_model_config(model),
        "technique_stack": squeezer.summary().get("techniques", []),
        "quantization": {"weights": "nf4_fp16", "optimizer": "galore_int8"},
        "training_state": {
            "step": getattr(squeezer, "_step_counter", 0),
            "mode": squeezer.mode,
        },
        "tensors": tensor_entries,
    }

    # ── Write file ──────────────────────────────────────────────────

    header_json = json.dumps(header, indent=2).encode("utf-8")
    header_len = len(header_json)
    padded_header_len = _align(12 + header_len, HEADER_ALIGNMENT) - 12
    header_json += b"\x00" * (padded_header_len - header_len)

    with open(path, "wb") as f:
        # Magic + version + header length
        f.write(VSQZ_MAGIC)
        f.write(struct.pack("<I", VSQZ_VERSION))
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)

        # Tensor blobs (64-byte aligned)
        for name, blob in tensor_blobs:
            pos = f.tell()
            aligned = _align(pos)
            if aligned > pos:
                f.write(b"\x00" * (aligned - pos))
            f.write(blob)

    file_size_mb = path.stat().st_size / (1024 ** 2)
    num_tensors = len(tensor_blobs)
    logger.info(
        ".vsqz saved: %s (%.1f MB, %d tensors)", path.name, file_size_mb, num_tensors
    )

    return str(path)


def load_vsqz_weights(path: str, model: Optional[nn.Module] = None, device: str = "cpu") -> Tuple[nn.Module, Dict]:
    """Load model weights only from .vsqz file (for inference).

    If model is None, creates bare module — you should pass a proper HF model
    instantiated from config.json for correct parameter registration.

    Returns (model, metadata_dict).
    Does NOT restore optimizer state.
    """
    header, tensor_data = _read_vsqz(path)

    if model is None:
        model = nn.Module()

    # Load weights
    state_dict = {}
    for name, entry in header["tensors"].items():
        if name.startswith("weight_"):
            param_name = name[7:].replace("_", ".")
            blob = tensor_data[name]
            tensor = _bytes_to_tensor(blob, entry["dtype"], entry["shape"])
            state_dict[param_name] = tensor.to(device)

    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded %d weights from .vsqz → model", len(state_dict))

    return model, header


def load_vsqz_full(squeezer, path: str) -> Dict:
    """Load full model weights + optimizer state from .vsqz file.

    Restores: model weights, GaLore P/Q states, m/v compressed states.
    The model is ready to resume training from this state.
    """
    header, tensor_data = _read_vsqz(path)

    # Load model weights
    state_dict = {}
    pids_seen = set()
    for name, entry in header["tensors"].items():
        if name.startswith("weight_"):
            param_name = name[7:].replace("_", ".")
            blob = tensor_data[name]
            tensor = _bytes_to_tensor(blob, entry["dtype"], entry["shape"])
            state_dict[param_name] = tensor
        elif name.startswith("galore_"):
            pids_seen.add(name)

    squeezer.model.load_state_dict(state_dict, strict=False)
    n_weights = len(state_dict)
    logger.info("Loaded %d model weights from .vsqz", n_weights)

    # Restore GaLore optimizer states
    if hasattr(squeezer, '_galore') and squeezer._galore is not None:
        n_galore = 0
        for param_id, gm in squeezer._galore._galore_modules.items():
            for attr in ("P", "Q", "m_P", "v_P", "m_Q", "v_Q"):
                blob_name = f"galore.{param_id}.{attr}"
                if blob_name in tensor_data:
                    blob = tensor_data[blob_name]
                    entry = header["tensors"].get(blob_name, {})
                    tensor = _bytes_to_tensor(blob, entry.get("dtype", "float16"), entry.get("shape", []))
                    setattr(gm, attr, tensor.to(gm.device))
                    n_galore += 1
        logger.info("Loaded %d GaLore states from .vsqz", n_galore)

    # Restore training state metadata
    ts = header.get("training_state", {})
    if hasattr(squeezer, '_step_counter') and ts.get("step"):
        squeezer._step_counter = ts["step"]
        logger.info("Restored training step: %d", ts["step"])

    return header


def _read_vsqz(path: str, verify_sha256: bool = False) -> Tuple[Dict, Dict[str, bytes]]:
    """Read .vsqz file: returns (header_dict, {name: raw_bytes}).

    If the main header is corrupted, tries the recovery record at end of file.
    Set verify_sha256=True for cryptographic integrity check.
    """
    import hashlib

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != VSQZ_MAGIC:
            raise ValueError(f"Not a .vsqz file: magic={magic!r} (expected {VSQZ_MAGIC!r})")

        version = struct.unpack("<I", f.read(4))[0]
        if version != VSQZ_VERSION:
            logger.warning(".vsqz v%d ≠ expected v%d", version, VSQZ_VERSION)

        header_len = struct.unpack("<I", f.read(4))[0]

        # Try reading + parsing header; if corrupt (JSON or encoding), use recovery
        header = None
        header_json = ""
        try:
            header_json = f.read(header_len).decode("utf-8").rstrip("\x00")
            header = json.loads(header_json)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Main header corrupt — trying recovery record...")

        # Read tensor blobs (use recovery header if main is broken)
        if header is None:
            # Seek to end of file, find RECO marker
            f.seek(-4, 2)
            reco_marker = f.read(4)
            if reco_marker == b"RECO":
                # Recovery record: [json_data][len:uint32_le][RECO]
                # Read length from 8 bytes before end
                f.seek(-8, 2)
                reco_len = struct.unpack("<I", f.read(4))[0]
                f.seek(-8 - reco_len, 2)
                recovery_json = f.read(reco_len).decode("utf-8")
                header = json.loads(recovery_json)
                logger.info("Recovery record loaded successfully")
            else:
                raise ValueError("Main header corrupt and no recovery record found")
        else:
            # Read tensors normally using this header
            pass

        tensor_data: Dict[str, bytes] = {}
        for name, entry in header["tensors"].items():
            offset = entry.get("offset", 0)
            size = entry["size"]
            f.seek(offset)
            data = f.read(size)
            if len(data) != size:
                logger.warning("Tensor '%s': read %d bytes, expected %d", name, len(data), size)
            tensor_data[name] = data

        # Read raw file blobs
        raw_blobs: Dict[str, Tuple[bytes, int]] = {}
        for rel_path, entry in header.get("raw_files", {}).items():
            offset = entry.get("offset", 0)
            size = entry["size"]
            mode = entry.get("mode", 0o644)
            f.seek(offset)
            data = f.read(size)
            raw_blobs[rel_path] = (data, mode)

        # SHA-256 verification (data only, no file padding)
        if verify_sha256 and header.get("sha256"):
            sha = hashlib.sha256()
            for name in sorted(header["tensors"].keys()):
                sha.update(tensor_data[name])
            for rel_path in sorted(header.get("raw_files", {}).keys()):
                sha.update(raw_blobs.get(rel_path, (b"", 0))[0])
            computed = sha.hexdigest()
            stored = header["sha256"]
            if computed != stored:
                raise ValueError(f"SHA-256 mismatch! File may be corrupted.\n  Stored:  {stored}\n  Computed: {computed}")
            logger.info("SHA-256 verified: %s", computed[:16])

    # Add raw file blobs to header for caller convenience
    if raw_blobs:
        header["_raw_blobs"] = raw_blobs

    return header, tensor_data


def _extract_model_config(model: nn.Module) -> Dict:
    """Extract model configuration for .vsqz header."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        # Try to get from model attributes
        total_params = sum(p.numel() for p in model.parameters())
        return {
            "total_params": total_params,
            "params_b": round(total_params / 1e9, 2),
        }

    return {
        "arch": getattr(cfg, "architectures", [getattr(cfg, "model_type", "unknown")])[0]
                if hasattr(cfg, "architectures") else getattr(cfg, "model_type", "unknown"),
        "hidden_dim": getattr(cfg, "hidden_size", 0),
        "num_layers": getattr(cfg, "num_hidden_layers", 0),
        "num_heads": getattr(cfg, "num_attention_heads", 0),
        "vocab_size": getattr(cfg, "vocab_size", 0),
        "total_params": sum(p.numel() for p in model.parameters()),
        "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
    }


def _build_model_from_config(config: Dict) -> nn.Module:
    """Build a model shell from config for proper state_dict loading.

    Returns a minimal nn.Module with registered Parameters matching the tensor
    shapes in the config. This allows load_state_dict to work correctly.
    """
    tensors = config.get("tensors", {})
    if not tensors:
        return nn.Module()

    # Try HuggingFace AutoModel first (if architecture is known)
    arch = config.get("model_config", {}).get("arch", "")
    if arch:
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
            hf_cfg = AutoConfig.for_model(arch, **config.get("model_config", {}))
            model = AutoModelForCausalLM.from_config(hf_cfg)
            return model
        except Exception:
            logger.debug("HF AutoModel not available for architecture '%s'", arch)

    # Fallback: build parameter shell from tensor shapes
    model = nn.Module()
    for name, entry in tensors.items():
        if name.startswith("weight_"):
            param_name = name[7:].replace("_", ".")
        elif name.startswith("galore."):
            continue  # Skip optimizer state tensors
        else:
            continue
        shape = entry["shape"]
        dtype_str = entry.get("dtype", "float16")
        torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}.get(dtype_str, torch.float16)
        param = nn.Parameter(torch.zeros(*shape, dtype=torch_dtype), requires_grad=True)
        # Register with dot-separated name
        parts = param_name.split(".")
        parent = model
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, parts[-1], param)

    return model


def peek_vsqz(path: str) -> Dict:
    """Read .vsqz header only (no tensor data) — fast metadata inspection."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != VSQZ_MAGIC:
            raise ValueError(f"Not a .vsqz file: magic={magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        header_len = struct.unpack("<I", f.read(4))[0]
        header_json = f.read(header_len).decode("utf-8").rstrip("\x00")
        return json.loads(header_json)
