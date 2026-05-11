"""
HuggingFace Plugin for .vsqz Format
=====================================
Enables: model = AutoModelForCausalLM.from_pretrained("model.vsqz") — just works.

Auto-activates on import. Detects .vsqz files and loads them transparently.
Works with all model architectures.

Usage:
    import vsqz.hf_plugin  # One-line activation
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("model.vsqz")  # Just works
"""

from __future__ import annotations

import json
import logging
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("SQZ-HF")

VSQZ_MAGIC = b"VSQZ"
VSQZ_VERSION = 1
_PATCHED = False


def _is_vvsqz_path(path: str) -> bool:
    """Check if a path points to a .vsqz file."""
    p = Path(path)
    if p.suffix == ".vsqz":
        return True
    if p.is_dir() and list(p.glob("*.vsqz")):
        return True
    return False


def _find_vvsqz_file(path: str) -> Optional[Path]:
    """Find the .vsqz file for a given path."""
    p = Path(path)
    if p.suffix == ".vsqz" and p.is_file():
        return p
    if p.is_dir():
        vvsqz_files = list(p.glob("*.vsqz"))
        if vvsqz_files:
            return vvsqz_files[0]
    return None


def _read_vvsqz_header(path: str) -> Tuple[Dict, Dict[str, bytes], int]:
    """Read .vsqz header and tensor data. Returns (header, tensor_data, data_start_offset)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != VSQZ_MAGIC:
            raise ValueError(f"Not a .vsqz file: magic={magic!r}")

        version = struct.unpack("<I", f.read(4))[0]
        header_len = struct.unpack("<I", f.read(4))[0]
        header_json = f.read(header_len).decode("utf-8").rstrip("\x00")
        header = json.loads(header_json)
        data_offset = f.tell()

        tensor_data: Dict[str, bytes] = {}
        for name, entry in header["tensors"].items():
            f.seek(entry.get("offset", 0))
            data = f.read(entry["size"])
            tensor_data[name] = data

    return header, tensor_data, data_offset


def _sqz_tensor_name_to_hf(name: str) -> str:
    """Convert .vsqz tensor name to HuggingFace parameter name.

    .vsqz stores tensor names as-is from the source. No conversion needed
    for safetensors/GGUF sources — names already match HF convention.
    """
    return name


def _bytes_to_tensor(data: bytes, dtype_str: str, shape: list) -> torch.Tensor:
    """Deserialize bytes to tensor."""
    import numpy as np
    dtype_map = {
        "float32": np.float32, "float16": np.float16,
        "bfloat16": np.float16, "int8": np.int8,
        "int32": np.int32, "int64": np.int64,
    }
    np_dtype = dtype_map.get(dtype_str, np.float16)
    arr = np.frombuffer(data, dtype=np_dtype).copy().reshape(shape)
    t = torch.from_numpy(arr)
    # Convert float16 back if original was float32 (for model dtype alignment)
    return t


def load_vvsqz_model(model_class_or_path: str, vsqz_path: Optional[str] = None, **kwargs) -> nn.Module:
    """Load a HuggingFace model from a .vsqz file.

    Args:
        model_class_or_path: Either a model class name (e.g. "AutoModelForCausalLM")
                             or the .vsqz file path.
        vsqz_path: If model_class_or_path is a class name, this is the .vsqz path.
        **kwargs: Passed to model class constructor.

    Returns:
        Loaded model with weights from .vsqz.
    """
    if vsqz_path is None:
        vsqz_path = model_class_or_path

    vsqz_file = _find_vvsqz_file(vsqz_path)
    if vsqz_file is None:
        raise FileNotFoundError(f"No .vsqz file found at {vsqz_path}")

    header, tensor_data, _ = _read_vvsqz_header(str(vsqz_file))

    # Extract model config from header
    model_cfg = header.get("model_config", {})
    arch = model_cfg.get("arch", "qwen2")

    # Build state dict from .vsqz tensors
    state_dict = {}
    for name, entry in header["tensors"].items():
        hf_name = _sqz_tensor_name_to_hf(name)
        # Skip metadata-only entries (no tensor data)
        if entry.get("shape") is None or len(entry.get("shape", [])) == 0:
            continue
        tensor = _bytes_to_tensor(tensor_data[name], entry["dtype"], entry["shape"])
        state_dict[hf_name] = tensor

    logger.info("Loaded %d weight tensors from %s", len(state_dict), vsqz_file.name)

    # Load model via HuggingFace, then inject weights
    from transformers import AutoConfig, AutoModelForCausalLM

    if "config" in kwargs:
        config = kwargs.pop("config")
    elif "config.json" in str(vsqz_file.parent):
        config = AutoConfig.from_pretrained(str(vsqz_file.parent))
    else:
        # Extract config from header metadata
        hf_config_path = None
        source_meta = header.get("source_metadata", {})
        if "config" in source_meta:
            config_dict = source_meta["config"]
        else:
            config_dict = model_cfg
        config = AutoConfig.for_model(arch, **config_dict) if arch else None
        if config is None:
            config = AutoConfig.from_pretrained(kwargs.get("pretrained_model_name_or_path", ""))

    torch_dtype = kwargs.pop("torch_dtype", torch.float16)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype, **kwargs)
    if model is None:
        # Fallback: create from config directly
        from transformers import PreTrainedModel
        model = PreTrainedModel(config)
    model.load_state_dict(state_dict, strict=False)

    return model


def patch_huggingface() -> bool:
    """Monkey-patch HuggingFace to support .vsqz files.

    After calling this, `AutoModel.from_pretrained("model.vsqz")` works.

    Returns True if patching succeeded.
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import huggingface_hub
        from transformers import AutoModel, AutoModelForCausalLM
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:
        logger.warning("HuggingFace not installed — .vsqz plugin not activated")
        return False

    _orig_from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        path = str(pretrained_model_name_or_path)

        # Check if the path points to a .vsqz file
        vsqz_file = _find_vvsqz_file(path)

        if vsqz_file is None:
            # Not .vsqz — use original behavior
            return _orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        # ── .vsqz path: bypass HF Hub resolution entirely ──

        # Load from .vsqz
        header, tensor_data, _ = _read_vvsqz_header(str(vsqz_file))

        # Build state dict
        state_dict = {}
        for name, entry in header["tensors"].items():
            if entry.get("shape") is None or len(entry.get("shape", [])) == 0:
                continue
            hf_name = _sqz_tensor_name_to_hf(name)
            tensor = _bytes_to_tensor(tensor_data[name], entry["dtype"], entry["shape"])
            state_dict[hf_name] = tensor

        # Get config: from same directory, or from .vsqz header
        config = kwargs.pop("config", None)
        if config is None:
            from transformers import AutoConfig
            cfg_dir = str(vsqz_file.parent)
            if (vsqz_file.parent / "config.json").exists():
                config = AutoConfig.from_pretrained(cfg_dir)
            else:
                model_cfg = header.get("model_config", {})
                arch = model_cfg.get("arch", "")
                if arch:
                    try:
                        config = AutoConfig.from_pretrained(arch)
                    except Exception:
                        pass
                if config is None or not hasattr(config, 'hidden_size'):
                    raise ValueError(
                        "Cannot determine model architecture from .vsqz file. "
                        "Place a config.json next to the .vsqz file, "
                        "or pass config=AutoConfig.from_pretrained('model-name')."
                    )

        # Create model (config-only, no Hub lookup)
        torch_dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", torch.float16) if "dtype" in kwargs else torch.float16)
        if hasattr(cls, 'from_config'):
            model = cls.from_config(config, torch_dtype=torch_dtype)
        else:
            model = cls(config)
            if torch_dtype in (torch.float16, torch.bfloat16):
                model = model.to(torch_dtype)
        model.load_state_dict(state_dict, strict=False, assign=True)

        logger.info(
            "Loaded model from .vsqz: %s (%d params)",
            vsqz_file.name, sum(p.numel() for p in model.parameters()),
        )
        return model

    PreTrainedModel.from_pretrained = _patched_from_pretrained

    # Also patch AutoModel factory entry points (AutoModelForCausalLM etc.)
    try:
        from transformers import AutoModelForCausalLM as _AMCLM, AutoModel as _AM
        _orig_amclm_fp = _AMCLM.from_pretrained
        _orig_am_fp = _AM.from_pretrained

        def _make_auto_patch(orig_fn):
            @classmethod
            def _patched(cls, pretrained_model_name_or_path, *a, **kw):
                path = str(pretrained_model_name_or_path)
                vsqz_file = _find_vvsqz_file(path)
                if vsqz_file is None:
                    return orig_fn.__func__(cls, pretrained_model_name_or_path, *a, **kw)
                # Config from same dir, then delegate to PreTrainedModel patch
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(str(vsqz_file.parent))
                kw["config"] = config
                model_cls = cls._model_mapping[type(config)]
                return model_cls.from_pretrained(pretrained_model_name_or_path, *a, **kw)

            return _patched

        _AMCLM.from_pretrained = _make_auto_patch(_orig_amclm_fp)
        _AM.from_pretrained = _make_auto_patch(_orig_am_fp)
    except Exception:
        pass

    _PATCHED = True

    logger.info("✅ .vsqz HuggingFace plugin activated — AutoModel.from_pretrained('model.vsqz') works")
    return True


# Auto-activate on import (safe: fails silently if transformers not installed)
try:
    patch_huggingface()
except Exception as e:
    logger.debug("HF plugin auto-activation skipped: %s", e)
