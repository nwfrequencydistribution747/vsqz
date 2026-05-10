"""
.vsqz Universal Converter
==========================
Converts any model format into compact .vsqz — like gzip for AI models.

Reads:
  - safetensors (directory or single file)
  - PyTorch (.bin, .pt, .pth)
  - GGUF (.gguf)

Does:
  - Strips AdamW optimizer dead weight (m,v,moment1,moment2 → delete)
  - Compresses weights: FP32 → FP16 (2×), optional INT8 quantization
  - Keeps only what's needed: config + weights + metadata
  - Writes .vsqz — universal compact format

CLI Usage:
  python -m vsqz convert model.safetensors output.vsqz
  python -m vsqz convert model.gguf output.vsqz
  python -m vsqz convert pytorch_model.bin output.vsqz
  python -m vsqz convert model/ output.vsqz          # auto-detect

  python -m vsqz info model.vsqz                     # peek metadata
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("SQZ-Converter")

# Optimizer keys to strip (AdamW dead weight nobody needs)
OPTIMIZER_KEYS = {
    "exp_avg", "exp_avg_sq", "moment1", "moment2",
    "adam", "optimizer", "state", "opt_state",
    "running_mean", "running_var",  # BatchNorm stats (not needed for inference)
    "num_batches_tracked",
}

VSQZ_MAGIC = b"VSQZ"
VSQZ_VERSION = 1
TENSOR_ALIGNMENT = 64
HEADER_ALIGNMENT = 4096


def convert_to_vsqz(
    source: str,
    output: str,
    quantize: str = "fp16",
    strip_optimizer: bool = True,
    verbose: bool = True,
    mmproj_only: bool = False,
    keep_original: bool = False,
) -> Tuple[str, Dict]:
    """Convert any model format to .vsqz.

    Args:
        source: Path to model file or directory
        output: Output .vsqz path
        quantize: "fp16" (default, 2× compression) or "int8" (4×, aggressive)
        strip_optimizer: Remove AdamW states (default: True)
        verbose: Print progress

    Returns: (output_path, stats_dict)
    """
    source_path = Path(source)
    output_path = Path(output).with_suffix(".vsqz")

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # ── Auto-detect and load ────────────────────────────────────────
    tensors, metadata = _load_source(source_path)

    if not tensors:
        raise ValueError(f"No tensors found in {source}")

    # ── mmproj extraction (vision bridge only) ──────────────────────
    MMPROJ_PATTERNS = ("visual", "vision", "mmproj", "merger", "projection")
    if mmproj_only:
        vision_tensors = {}
        for name, t in tensors.items():
            if any(p in name.lower() for p in MMPROJ_PATTERNS):
                vision_tensors[name] = t
        if verbose:
            print(f"  mmproj extraction: {len(vision_tensors)}/{len(tensors)} tensors kept (vision bridge)")
        tensors = vision_tensors
        output = Path(output).with_suffix(".mmproj.vsqz")

    original_tensors = len(tensors)
    original_bytes = sum(t.nbytes for t in tensors.values())

    # ── Strip optimizer dead weight ────────────────────────────────
    if strip_optimizer:
        stripped = 0
        keys_to_remove = []
        for name in tensors:
            name_lower = name.lower()
            for key in OPTIMIZER_KEYS:
                if key in name_lower:
                    keys_to_remove.append(name)
                    stripped += tensors[name].nbytes
                    break
        for k in keys_to_remove:
            del tensors[k]
        if verbose:
            print(f"  Stripped {len(keys_to_remove)} optimizer tensors ({_fmt_bytes(stripped)})")

    # ── Quantize weights (GPU-accelerated if available) ─────────────
    use_gpu = _TORCH_AVAILABLE and torch.cuda.is_available()
    if verbose and use_gpu:
        print(f"  🚀 GPU-accelerated: {torch.cuda.get_device_name()}")

    quantized_bytes = 0
    GPU_LIMIT = 2 * 1024 * 1024 * 1024  # 2GB — larger tensors stay on CPU

    for name, tensor in tensors.items():
        orig_nb = tensor.nbytes

        if tensor.dtype in (np.float32, np.float64):
            if use_gpu and orig_nb < GPU_LIMIT:
                # GPU path
                t_gpu = torch.from_numpy(tensor).cuda()
                if quantize == "int8":
                    max_abs = t_gpu.abs().max().float()
                    scale = max_abs / 127.0
                    if scale == 0: scale = torch.tensor(1.0, device="cuda")
                    q_gpu = torch.round(t_gpu.float() / scale).clamp(-127, 127).to(torch.int8)
                    tensors[name] = q_gpu.cpu().numpy()
                    del t_gpu, q_gpu
                else:
                    tensors[name] = t_gpu.half().cpu().numpy()
                    del t_gpu
                quantized_bytes += (orig_nb - tensors[name].nbytes)
            else:
                # CPU path
                if quantize == "int8":
                    max_abs = np.abs(tensor).max()
                    if max_abs > 0:
                        scale = max_abs / 127.0
                        q = np.round(tensor / scale).clip(-127, 127).astype(np.int8)
                        tensors[name] = q
                        quantized_bytes += (orig_nb - q.nbytes)
                else:
                    tensors[name] = tensor.astype(np.float16)
                    quantized_bytes += (orig_nb - tensors[name].nbytes)

        if use_gpu and len(tensors) % 50 == 0:
            torch.cuda.empty_cache()  # Periodic cleanup

    if use_gpu:
        torch.cuda.empty_cache()

    if verbose and quantized_bytes > 0:
        backend = "GPU" if use_gpu else "CPU"
        print(f"  Compressed weights: {_fmt_bytes(quantized_bytes)} saved ({quantize}, {backend})")

    final_bytes = sum(t.nbytes for t in tensors.values())
    strip_savings = original_bytes - final_bytes - quantized_bytes

    # ── Write .vsqz ──────────────────────────────────────────────────
    raw_blobs = metadata.get("raw_files_data")  # peek, don't pop
    header = _build_vsqz_header(tensors, metadata, quantize)
    metadata.pop("raw_files_data", None)  # clean up after header built
    _write_vsqz(output_path, header, tensors, raw_blobs)

    file_size = output_path.stat().st_size
    stats = {
        "source": str(source_path),
        "output": str(output_path),
        "original_size": _fmt_bytes(original_bytes),
        "final_size": _fmt_bytes(file_size),
        "compression_ratio": round(original_bytes / max(file_size, 1), 1),
        "tensors_before": original_tensors,
        "tensors_after": len(tensors),
        "optimizer_tensors_stripped": original_tensors - len(tensors) - (0 if not strip_optimizer else 0),
        "weights_compressed": _fmt_bytes(quantized_bytes),
        "quantize": quantize,
    }

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Original:  {stats['original_size']} ({original_tensors} tensors)")
        print(f"  .vsqz:      {stats['final_size']} ({stats['tensors_after']} tensors)")
        print(f"  Saved:     {stats['compression_ratio']}× smaller")
        print(f"  → {output_path}")

    return str(output_path), stats


def _load_source(path: Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Auto-detect format and load tensors. Returns (tensors, metadata)."""
    if path.is_dir():
        return _load_safetensors_dir(path)
    suffix = path.suffix.lower()

    if suffix in (".safetensors",):
        return _load_safetensors_file(path)
    elif suffix in (".bin", ".pt", ".pth", ".ckpt"):
        return _load_pytorch(path)
    elif suffix in (".gguf",):
        return _load_gguf(path)
    else:
        raise ValueError(f"Unknown format: {suffix}. Supported: .safetensors, .gguf, .bin, .pt")


def _load_safetensors_dir(directory: Path) -> Tuple[Dict, Dict]:
    """Load all files recursively. Symlinks, mtime/atime, and subdirs preserved.

    Each file's relative path inside the directory is preserved for roundtrip.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors not installed: pip install safetensors")

    tensors = {}
    raw_files = {}  # non-tensor files stored verbatim
    symlinks = {}   # rel_path → target_path
    file_times = {}  # rel_path → (mtime, atime)
    metadata = {"format": "safetensors", "source_dir": str(directory), "source_files": {}}
    directory = Path(directory).resolve()

    # Collect all files recursively
    for fp in sorted(directory.rglob("*")):
        st = fp.lstat()  # lstat to detect symlinks
        if fp.is_dir() and not fp.is_symlink():
            continue
        rel = str(fp.relative_to(directory))
        file_times[rel] = (st.st_mtime, st.st_atime)

        if fp.is_symlink() and fp.suffix not in (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt"):
            symlinks[rel] = os.readlink(fp)
            metadata["source_files"][rel] = []
        elif fp.suffix in (".safetensors",):
            metadata["source_files"][rel] = []
            with safe_open(fp, framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    if t.dtype == torch.bfloat16:
                        t = t.to(torch.float16)
                    tensors[key] = t.cpu().numpy()
                    metadata["source_files"][rel].append(key)
        else:
            with open(fp, "rb") as rf:
                raw_files[rel] = rf.read()
            metadata["source_files"][rel] = []  # no tensors, raw blob

    metadata["file_times"] = file_times
    if symlinks:
        metadata["symlinks"] = symlinks
    if raw_files:
        metadata["raw_files_data"] = {}
        import zstandard as _zstd
        cctx = _zstd.ZstdCompressor(level=3)
        for rel, data in raw_files.items():
            compressed = cctx.compress(data)
            orig_path = directory / rel
            metadata["raw_files_data"][rel] = (compressed, orig_path.stat().st_mode & 0o777)
        metadata["raw_files"] = {rel: len(data) for rel, data in raw_files.items()}
    if "config.json" in raw_files:
        try:
            metadata["config"] = json.loads(raw_files["config.json"].decode("utf-8"))
        except Exception:
            pass

    return tensors, metadata


def _load_safetensors_file(path: Path) -> Tuple[Dict, Dict]:
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors not installed: pip install safetensors")

    tensors = {}
    tensor_names = []
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float16)
            tensors[key] = t.cpu().numpy()
            tensor_names.append(key)

    st = path.lstat()
    return tensors, {"format": "safetensors", "source_files": {path.name: tensor_names},
                     "file_times": {path.name: (st.st_mtime, st.st_atime)}}


def _load_pytorch(path: Path) -> Tuple[Dict, Dict]:
    try:
        import torch
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except ImportError:
        raise ImportError("PyTorch not installed")

    tensors = {}
    tensor_names = []
    metadata = {"format": "pytorch"}

    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                # Nested dict (e.g., optimizer states)
                for subkey, subval in value.items():
                    if hasattr(subval, "numpy"):
                        tensors[f"{key}.{subkey}"] = subval.cpu().numpy()
                        tensor_names.append(f"{key}.{subkey}")
                    elif isinstance(subval, (int, float, str)):
                        metadata[f"{key}.{subkey}"] = subval
            elif hasattr(value, "numpy"):
                tensors[key] = value.cpu().numpy()
                tensor_names.append(key)
            else:
                metadata[key] = value

    metadata["source_files"] = {path.name: tensor_names}
    st = path.lstat() if isinstance(path, Path) else Path(str(path)).lstat()
    metadata["file_times"] = {path.name if isinstance(path, Path) else Path(str(path)).name:
                               (st.st_mtime, st.st_atime)}
    return tensors, metadata


def _load_gguf(path: Path) -> Tuple[Dict, Dict]:
    """Minimal GGUF reader — reads tensor metadata and data from GGUF files."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: magic={magic!r}")

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        st = Path(str(path)).lstat()
        fname = str(Path(str(path)).name)
        metadata = {"format": "gguf", "version": version, "n_tensors": n_tensors, "kv_pairs": {},
                    "source_files": {fname: []},
                    "file_times": {fname: (st.st_mtime, st.st_atime)}}

        # Read key-value pairs (all GGUF types preserved for roundtrip)
        TYPE_NAMES = {1:"uint8",2:"int8",3:"uint16",4:"int16",5:"uint32",6:"int32",
                      7:"float32",8:"bool",9:"string",10:"array",11:"uint64",12:"int64",13:"float64"}
        for _ in range(n_kv):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")
            val_type = struct.unpack("<I", f.read(4))[0]
            tname = TYPE_NAMES.get(val_type)
            if val_type == 9:  # STRING
                val_len = struct.unpack("<Q", f.read(8))[0]
                val = f.read(val_len).decode("utf-8", errors="replace")
                metadata["kv_pairs"][key] = {"type": "string", "value": val}
            elif val_type == 8:  # BOOL
                metadata["kv_pairs"][key] = {"type": "bool", "value": bool(f.read(1)[0])}
            elif val_type == 10:  # ARRAY
                elem_type = struct.unpack("<I", f.read(4))[0]
                count = struct.unpack("<Q", f.read(8))[0]
                etype = TYPE_NAMES.get(elem_type, "unknown")
                elem_fmt = {1:"B",2:"b",3:"H",4:"h",5:"I",6:"i",7:"f",11:"Q",12:"q",13:"d"}.get(elem_type)
                if elem_fmt:
                    arr = list(struct.unpack(f"<{count}{elem_fmt}", f.read(count * struct.calcsize(elem_fmt))))
                else:
                    arr = []
                metadata["kv_pairs"][key] = {"type": "array", "value": arr, "element_type": etype}
            elif val_type in (1, 2):  # UINT8, INT8
                val = f.read(1)[0]; sign = val_type == 2
                metadata["kv_pairs"][key] = {"type": tname, "value": val - 256 if sign and val > 127 else val}
            elif val_type in (3, 4):  # UINT16, INT16
                fmt = "<h" if val_type == 4 else "<H"
                metadata["kv_pairs"][key] = {"type": tname, "value": struct.unpack(fmt, f.read(2))[0]}
            elif val_type in (5, 6):  # UINT32, INT32
                fmt = "<i" if val_type == 6 else "<I"
                metadata["kv_pairs"][key] = {"type": tname, "value": struct.unpack(fmt, f.read(4))[0]}
            elif val_type == 7:  # FLOAT32
                metadata["kv_pairs"][key] = {"type": "float32", "value": struct.unpack("<f", f.read(4))[0]}
            elif val_type == 13:  # FLOAT64
                metadata["kv_pairs"][key] = {"type": "float64", "value": struct.unpack("<d", f.read(8))[0]}
            elif val_type in (11, 12):  # UINT64, INT64
                fmt = "<q" if val_type == 12 else "<Q"
                metadata["kv_pairs"][key] = {"type": tname, "value": struct.unpack(fmt, f.read(8))[0]}
            else:
                # unknown type — skip 8 bytes as fallback
                f.read(8)
                metadata["kv_pairs"][key] = {"type": "unknown", "value": None}

        # Read tensor infos
        tensor_infos = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_infos.append({"name": name, "shape": shape, "ggml_type": ggml_type, "offset": offset})
            metadata["source_files"][fname].append(name)

        # Store tensor_infos for roundtrip reconstruction
        metadata["tensor_infos"] = tensor_infos

        # Read tensor data
        tensors = {}
        for info in tensor_infos:
            f.seek(info["offset"])
            total_elems = 1
            for d in info["shape"]:
                total_elems *= d
            dtype_map = {
                0: np.float32, 1: np.float16, 2: np.int32,
                3: np.int16, 4: np.int8, 8: np.float32,
            }
            dtype = dtype_map.get(info["ggml_type"], np.float16)
            elem_size = np.dtype(dtype).itemsize
            data = f.read(total_elems * elem_size)
            tensors[info["name"]] = np.frombuffer(data, dtype=dtype).reshape(info["shape"])

    return tensors, metadata


def _build_vsqz_header(tensors: Dict, metadata: Dict, quantize: str) -> Dict:
    """Build .vsqz JSON header with tensor index and raw file entries."""
    tensor_entries = {}
    # Offsets start AFTER magic(4) + version(4) + header_len(4) + header
    offset = 12 + HEADER_ALIGNMENT
    for name, tensor in sorted(tensors.items()):
        blob = tensor.tobytes()
        tensor_entries[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "size": len(blob),
            "offset": offset,
        }
        offset += len(blob)

    # Raw file blobs come after tensors
    raw_entries = {}
    raw_files_data = metadata.pop("raw_files_data", None)
    if raw_files_data:
        for rel_path, (raw_bytes, file_mode) in sorted(raw_files_data.items()):
            raw_entries[rel_path] = {
                "size": len(raw_bytes),
                "offset": offset,
                "compressed": "zstd",
                "mode": file_mode,
            }
            offset += len(raw_bytes)

    # Infer original (pre-quantization) dtype for accurate size reporting
    orig_dtype = {"fp16": "float32", "fp32": "float32", "int8": "float16", "none": "float32"}.get(quantize, "float32")

    header = {
        "vsqz_version": __import__('vsqz').__version__,
        "converted_from": metadata.get("format", "unknown"),
        "quantize": quantize,
        "original_dtype": orig_dtype,
        "source_metadata": {k: v for k, v in metadata.items() if k != "format"},
        "tensors": tensor_entries,
    }
    if raw_entries:
        header["raw_files"] = raw_entries
    return header


_deletion_allowed = None  # module-level: None=unasked, True=allowed, False=denied


def _confirm_deletion(paths, quiet=False):
    """Ask user to confirm file deletion. Caches answer across calls."""
    global _deletion_allowed
    if _deletion_allowed is not None:
        return _deletion_allowed
    if quiet:
        _deletion_allowed = True
        return True
    if not sys.stdin.isatty():
        _deletion_allowed = True  # piped input: proceed (user chose -k if they wanted safety)
        return True
    files = [paths] if isinstance(paths, str) else list(paths)
    print(f"\n  ⚠️  About to delete {len(files)} original file(s):")
    for f in files:
        print(f"     → {f}")
    print()
    print("  Have you backed up your data? Tested restoration?")
    print("  Tip: use -k/--keep to preserve originals instead.")
    try:
        answer = input("  Delete originals? [yes/no]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Keeping originals (no confirmation).")
        return False
    if answer in ("yes", "y"):
        _deletion_allowed = True
        return True
    _deletion_allowed = False
    print("  Keeping originals. Use -k next time to skip this prompt.")
    return False


def _save_gguf(path: Path, tensors: Dict, metadata: Dict, verbose: bool = False) -> Path:
    """Write tensors as GGUF format. Preserves kv_pairs, ggml_types, alignment."""
    tensor_infos = metadata.get("tensor_infos", [])
    kv_pairs = metadata.get("kv_pairs", {}).copy()
    version = metadata.get("version", 3)
    tinfo_map = {ti["name"]: ti for ti in tensor_infos}

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(kv_pairs)))

        # Write key-value pairs (full GGUF type fidelity)
        TYPE_CODES = {"uint8":1,"int8":2,"uint16":3,"int16":4,"uint32":5,"int32":6,
                      "float32":7,"bool":8,"string":9,"array":10,"uint64":11,"int64":12,"float64":13}
        for key, entry in kv_pairs.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)
            # Handle both old (raw string) and new (dict with type/value) formats
            if isinstance(entry, dict) and "type" in entry:
                tname, v = entry["type"], entry["value"]
                type_code = TYPE_CODES.get(tname, 0)
                f.write(struct.pack("<I", type_code))
                if tname == "string":
                    val_bytes = str(v).encode("utf-8")
                    f.write(struct.pack("<Q", len(val_bytes)))
                    f.write(val_bytes)
                elif tname == "bool":
                    f.write(b"\x01" if v else b"\x00")
                elif tname == "array":
                    etype_code = TYPE_CODES.get(entry.get("element_type", "int32"), 6)
                    arr = list(v) if v else []
                    f.write(struct.pack("<I", etype_code))
                    f.write(struct.pack("<Q", len(arr)))
                    e_fmt = {1:"B", 2:"b", 3:"H", 4:"h", 5:"I", 6:"i", 7:"f", 11:"Q", 12:"q", 13:"d"}.get(etype_code, "I")
                    f.write(struct.pack(f"<{len(arr)}{e_fmt}", *arr) if arr else b"")
                elif tname in ("float64",):
                    f.write(struct.pack("<d", float(v)))
                elif tname in ("uint64", "int64"):
                    fmt = "<q" if tname == "int64" else "<Q"
                    f.write(struct.pack(fmt, int(v)))
                elif tname in ("float32",):
                    f.write(struct.pack("<f", float(v)))
                elif tname in ("uint32", "int32"):
                    fmt = "<i" if tname == "int32" else "<I"
                    f.write(struct.pack(fmt, int(v)))
                elif tname in ("uint16", "int16"):
                    fmt = "<h" if tname == "int16" else "<H"
                    f.write(struct.pack(fmt, int(v)))
                elif tname in ("uint8", "int8"):
                    f.write(struct.pack("B", int(v) & 0xFF))
                else:
                    f.write(struct.pack("<Q", 0))  # unknown → skip
            elif isinstance(entry, str):
                # Backward compat: old-style raw string
                f.write(struct.pack("<I", 9))
                val_bytes = entry.encode("utf-8")
                f.write(struct.pack("<Q", len(val_bytes)))
                f.write(val_bytes)
            else:
                f.write(struct.pack("<I", 0))
                f.write(struct.pack("<Q", 0))

        # Write tensor infos (names, shapes, types) — record offset positions for later update
        offset_slots = {}  # name → (file_position_for_offset_field)
        for name in sorted(tensors):
            info = tinfo_map.get(name, {})
            ggml_type = info.get("ggml_type", 1)
            shape = info.get("shape", list(tensors[name].shape))
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ggml_type))
            offset_slots[name] = f.tell()
            f.write(struct.pack("<Q", 0))  # placeholder

        # Align to 32 bytes for data section
        pos = f.tell()
        align = ((pos + 31) // 32) * 32
        f.write(b"\x00" * (align - pos))

        # Write tensor data, update offsets
        for name in sorted(tensors):
            pos = f.tell()
            align = ((pos + 31) // 32) * 32
            if align > pos:
                f.write(b"\x00" * (align - pos))
            data_start = f.tell()
            tensor = tensors[name]
            data = tensor.astype(np.float16).tobytes() if tensor.dtype != np.float16 else tensor.tobytes()
            f.write(data)
            # Seek back to update offset in tensor info
            saved = f.tell()
            f.seek(offset_slots[name])
            f.write(struct.pack("<Q", data_start))
            f.seek(saved)

    if verbose:
        print(f"  Written: {path} ({_fmt_bytes(path.stat().st_size)})")
    return path


def _apply_zstd(path: str, keep_original: bool = False) -> str:
    """Post-compress .vsqz with zstd. Returns new path (.vsqz.zst)."""
    import zstandard as zstd
    with open(path, "rb") as f:
        data = f.read()
    cctx = zstd.ZstdCompressor(level=3)
    zst_path = path + ".zst"
    with open(zst_path, "wb") as f:
        f.write(cctx.compress(data))
    if not keep_original:
        os.remove(path)
    ratio = len(data) / max(len(open(zst_path,"rb").read()), 1)
    logger.info("zstd: %s → %s (%.1f×)", _fmt_bytes(len(data)), _fmt_bytes(os.path.getsize(zst_path)), ratio)
    return zst_path


def _decompress_zstd(path: str) -> str:
    """Decompress .vsqz.zst → .vsqz. Returns decompressed path."""
    import zstandard as zstd
    out = path.replace(".zst", "")
    with open(path, "rb") as f:
        data = zstd.ZstdDecompressor().decompress(f.read())
    with open(out, "wb") as f:
        f.write(data)
    return out


def _write_vsqz(path: Path, header: Dict, tensors: Dict, raw_blobs: Dict = None) -> None:
    """Write .vsqz file with SHA-256 hash + recovery record.

    Args:
        raw_blobs: Dict of {rel_path: (zstd_compressed_bytes, file_mode)} for non-tensor files
    """
    import hashlib

    header_json = json.dumps(header, indent=2).encode("utf-8")
    header_padded = header_json + b"\x00" * (HEADER_ALIGNMENT - len(header_json) % HEADER_ALIGNMENT)

    with open(path, "wb") as f:
        f.write(VSQZ_MAGIC)
        f.write(struct.pack("<I", VSQZ_VERSION))
        f.write(struct.pack("<I", len(header_padded)))
        f.write(header_padded)

        # Write tensors sequentially, computing SHA-256 hash (data only, no padding)
        sha = hashlib.sha256()
        for name in sorted(tensors):
            actual_offset = f.tell()
            header["tensors"][name]["offset"] = actual_offset
            data = tensors[name].tobytes()
            f.write(data)
            sha.update(data)
            header["tensors"][name]["size"] = len(data)

        # Write raw file blobs after tensors
        if raw_blobs:
            for rel_path in sorted(raw_blobs):
                data, _ = raw_blobs[rel_path]
                actual_offset = f.tell()
                if "raw_files" in header and rel_path in header["raw_files"]:
                    header["raw_files"][rel_path]["offset"] = actual_offset
                    header["raw_files"][rel_path]["size"] = len(data)
                f.write(data)
                sha.update(data)

        # Insert SHA-256 into main header (seek back and rewrite)
        header["sha256"] = sha.hexdigest()
        header_json_final = json.dumps(header, indent=2).encode("utf-8")
        header_final_padded = header_json_final + b"\x00" * (HEADER_ALIGNMENT - len(header_json_final) % HEADER_ALIGNMENT)
        f.seek(12)  # After magic + version + header_len
        f.write(header_final_padded)

        # ── Recovery Record: [recovery_json] [recovery_len: uint32_le] [RECO] ──
        f.seek(0, 2)
        recovery_header = dict(header)
        recovery_header["_recovery"] = True
        recovery_json = json.dumps(recovery_header).encode("utf-8")
        f.write(recovery_json)
        f.write(struct.pack("<I", len(recovery_json)))  # recovery length
        f.write(b"RECO")


def _fmt_bytes(n: int) -> str:
    """Format bytes human-readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ── CLI (gzip/zip compatible flags) ────────────────────────────────────


def _usage():
    return """vsqz — the gzip for AI models

Usage: vsqz [OPTIONS] <source> [<output>.vsqz]

Compression (default):
  vsqz model.safetensors        → model.safetensors.vsqz (fp16, level 6)
  vsqz -9 model.gguf            → best compression (int8, level 9)
  vsqz -k -v model/ output      → keep original, verbose output

Decompression:
  vsqz -d model.vsqz            → restore original format (safetensors/GGUF/pt)

Options:
  -k, --keep         Keep original file after compression
  -d, --decompress   Restore original format from .vsqz
  -v, --verbose      Show compression details and ratio (default: on)
  -q, --quiet        Suppress all output
  -f, --force        Overwrite existing output file
  -t, --test         Verify .vsqz file integrity (all tensors readable)
  -l, --list         Show metadata without loading tensors
  -1 .. -9           Compression level: -1 fast/fp16, -9 best/int8+sparse
  -r, --recursive    Compress all compatible models in a directory tree
  -s, --split SIZE   Split output into chunks (e.g. -s 8G for cloud upload)
  -x, --exclude KEY  Exclude tensors matching pattern (e.g. -x adam -x opt)
  -z, --zstd         Post-compress with zstd (archive mode, 5-15% smaller)
  -h, --help         Show this help

Examples:
  vsqz model.safetensors               # compress, delete original
  vsqz -k model/ output.vsqz           # compress, keep original
  vsqz -d model.vsqz                   # decompress to original format
  vsqz -t model.vsqz                   # integrity test
  vsqz -l model.vsqz                   # show metadata (no loading)
  vsqz -9 -v model.gguf                # best compression, verbose
  vsqz -s 8G large-20B.safetensors     # split into 8 GB chunks
  vsqz -x adam checkpoint.pt           # strip optimizer states
  vsqz -r models/                      # compress all models in directory"""


def main():
    if len(sys.argv) < 2 or "-h" in sys.argv or "--help" in sys.argv:
        print(_usage())
        return

    # Parse gzip/zip-style flags
    keep = ("-k" in sys.argv or "--keep" in sys.argv)
    decompress = ("-d" in sys.argv or "--decompress" in sys.argv)
    quiet = ("-q" in sys.argv or "--quiet" in sys.argv)
    force = ("-f" in sys.argv or "--force" in sys.argv)
    do_zstd = ("-z" in sys.argv or "--zstd" in sys.argv)
    do_test = ("-t" in sys.argv or "--test" in sys.argv)
    do_list = ("-l" in sys.argv or "--list" in sys.argv)
    recursive = ("-r" in sys.argv or "--recursive" in sys.argv)
    split_val = None
    for i, a in enumerate(sys.argv):
        if a in ("-s", "--split") and i + 1 < len(sys.argv):
            split_val = sys.argv[i + 1]
    exclude_pats = []
    for i, a in enumerate(sys.argv):
        if a in ("-x", "--exclude") and i + 1 < len(sys.argv):
            exclude_pats.append(sys.argv[i + 1])

    verbose = not quiet

    # Compression level: -1 .. -9
    comp_level = 6
    for i in range(1, 10):
        if f"-{i}" in sys.argv:
            comp_level = i
    quantize = "int8" if comp_level >= 8 else "fp16"

    # Positional args (skip flags and their values)
    args = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-") and a not in ("-1","-2","-3","-4","-5","-6","-7","-8","-9"):
            if a in ("-s","--split","-x","--exclude"):
                skip_next = True
            continue
        args.append(a)

    source = args[0] if len(args) > 0 else None
    output = args[1] if len(args) > 1 else None

    # ── Modes ──

    if do_test and source:
        if verbose: print(f"Testing: {source}")
        from .vsqz_format import _read_vsqz
        header, tensor_data = _read_vsqz(source, verify_sha256=True)
        missing = [n for n in header["tensors"] if n not in tensor_data]
        bad = [n for n, e in header["tensors"].items() if n in tensor_data and len(tensor_data[n]) != e["size"]]
        if missing or bad:
            print(f"  ❌ Integrity FAILED: {len(missing)} missing, {len(bad)} size-mismatch")
            sys.exit(1)
        sha = header.get("sha256", "")
        if sha:
            print(f"  ✅ Integrity OK — {len(header['tensors'])} tensors, sha256:{sha[:16]}...")
        else:
            print(f"  ✅ Integrity OK — {len(header['tensors'])} tensors, {_fmt_bytes(Path(source).stat().st_size)}")
        return

    if do_list and source:
        from .vsqz_format import peek_vsqz
        h = peek_vsqz(source)
        sz = Path(source).stat().st_size
        src_meta = h.get("source_metadata", {})
        src_files = src_meta.get("source_files", {})
        raw_original = src_meta.get("raw_files", {})
        raw_compressed = h.get("raw_files", {})

        # Gather per-file info
        symlinks = src_meta.get("symlinks", {})
        file_times = src_meta.get("file_times", {})
        entries = []  # (name, original_size, compressed_size, kind, mode, mtime)
        for fname, tnames in sorted(src_files.items()):
            kind = "symlink" if fname in symlinks else ("tensor" if tnames else "file")
            comp_size = 0
            if kind == "symlink":
                comp_size = len(symlinks[fname].encode())
            elif tnames:
                for n in tnames:
                    if n in h.get("tensors", {}):
                        comp_size += h["tensors"][n].get("size", 0)
            else:
                comp_size = raw_compressed.get(fname, {}).get("size", 0)
            orig_size = raw_original.get(fname, 0) if kind == "file" else 0
            if not orig_size and kind == "tensor":
                orig_dtype = h.get("original_dtype", "float32")
                elsize = {"float32": 4, "float16": 2, "int8": 1}.get(orig_dtype, 4)
                for n in tnames:
                    e = h["tensors"].get(n, {})
                    s = e.get("shape", [])
                    nelem = 1
                    for dim in s:
                        nelem *= dim
                    orig_size += nelem * elsize
            if kind == "tensor":
                mode = "644"
            elif kind == "symlink":
                mode = "777"  # symlink perms are from target
            else:
                mode = oct(raw_compressed.get(fname, {}).get("mode", 0o644))[2:]
            mtime = file_times.get(fname, (0, 0))[0]
            mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else ""
            entries.append((fname, orig_size, comp_size, kind, mode, mtime_str))

        # Print table
        if entries:
            print(f"{'Name':<38} {'Mode':>5} {'Modified':>16} {'Original':>12} {'Compressed':>12} {'Ratio':>8}")
            print(f"{'-'*38} {'-'*5} {'-'*16} {'-'*12} {'-'*12} {'-'*8}")
            total_orig = total_comp = 0
            for name, orig, comp, kind, mode, mtime_str in entries:
                ratio = f"{comp/orig*100:.0f}%" if orig > 0 and comp <= orig else ("—" if orig <= 0 else f"{comp/orig*100:.0f}%")
                print(f"{name:<38} {mode:>5} {mtime_str:>16} {_fmt_bytes(orig) if orig>0 else ('link' if kind=='symlink' else '?'):>12} {_fmt_bytes(comp):>12} {ratio:>8}")
                total_orig += orig
                total_comp += comp
            total_ratio = f"{total_comp/total_orig*100:.0f}%" if total_orig > 0 else "—"
            print(f"{'─'*38} {'─'*5} {'─'*16} {'─'*12} {'─'*12} {'─'*8}")
            print(f"{'Total (' + str(len(entries)) + ' files)':<38} {'':>5} {'':>16} {_fmt_bytes(total_orig):>12} {_fmt_bytes(total_comp):>12} {total_ratio:>8}")

        # Summary
        ts = h.get("tensors", {})
        total_params = sum(np.prod(t.get("shape",[])) for t in ts.values())
        print(f"\n  Format: {h.get('converted_from', '?')}  |  Quantize: {h.get('quantize', '?')}  |  {len(ts)} tensors ({total_params/1e6:.1f}M params)")
        sha = h.get("sha256", "")
        if sha: print(f"  SHA-256: {sha[:32]}...")
        return

    # Handle .vsqz.zst transparently (decompress first)
    if source.endswith('.zst'):
        if verbose: print(f"Decompressing zstd: {source}")
        source = _decompress_zstd(source)

    if decompress and source:
        if verbose: print(f"Decompressing: {source}")
        from .vsqz_format import _read_vsqz
        header, tensor_data = _read_vsqz(source)
        orig_fmt = header.get("converted_from", "pytorch")
        out_dir = Path(output) if output else Path(source).with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        src_meta = header.get("source_metadata", {})
        src_files = src_meta.get("source_files", {})
        raw_blobs = header.get("_raw_blobs", {})

        # Convert tensors to numpy arrays
        tensors_np = {}
        for name, data in tensor_data.items():
            entry = header["tensors"].get(name, {})
            dtype = {"float16": np.float16, "float32": np.float32, "int8": np.int8}.get(
                entry.get("dtype", "float16"), np.float16)
            shape = entry.get("shape", [])
            tensors_np[name] = np.frombuffer(data, dtype=dtype).reshape(shape)

        if orig_fmt == "safetensors" and src_files:
            # Restore per-file structure from source_files mapping
            from safetensors.torch import save_file
            for fname, tnames in sorted(src_files.items()):
                if not tnames:
                    continue  # raw file, handled below
                out_path = out_dir / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_file({n: torch.from_numpy(tensors_np[n].copy()) for n in tnames if n in tensors_np},
                          str(out_path))
                if verbose: print(f"  → {out_path}")
        elif orig_fmt == "gguf":
            gguf_path = out_dir / "model.gguf"
            _save_gguf(gguf_path, tensors_np, src_meta, verbose=verbose)
            if verbose: print(f"  → {gguf_path}")
        else:
            if src_files:
                # Restore per-file structure for pytorch
                for fname, tnames in sorted(src_files.items()):
                    if not tnames:
                        continue
                    out_path = out_dir / fname
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({n: torch.from_numpy(tensors_np[n].copy()) for n in tnames if n in tensors_np},
                               str(out_path))
                    if verbose: print(f"  → {out_path}")
            else:
                out_path = out_dir / "pytorch_model.bin"
                torch.save({n: torch.from_numpy(tensors_np[n].copy()) for n in tensor_data},
                           str(out_path))
                if verbose: print(f"  → {out_path}")

        # Restore symlinks
        for rel_path, target in sorted(src_meta.get("symlinks", {}).items()):
            out_path = out_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.symlink_to(target)
            if verbose: print(f"  → {out_path} -> {target}")

        # Restore raw (non-tensor) files
        import zstandard as _zstd
        dctx = _zstd.ZstdDecompressor()
        for rel_path, (data, mode) in sorted(raw_blobs.items()):
            out_path = out_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            decompressed = dctx.decompress(data)
            out_path.write_bytes(decompressed)
            if mode:
                try:
                    out_path.chmod(mode)
                except (PermissionError, OSError) as e:
                    if verbose:
                        print(f"  ⚠️  Cannot chmod {out_path} to {mode:o}: {e}")
            if verbose: print(f"  → {out_path}")

        # Restore mtime/atime for all restored files (tensors + raw + symlinks)
        file_times = src_meta.get("file_times", {})
        for rel_path, (mtime, atime) in file_times.items():
            out_path = out_dir / rel_path
            if out_path.exists(follow_symlinks=False):
                try:
                    os.utime(out_path, (atime, mtime), follow_symlinks=False)
                except (PermissionError, OSError) as e:
                    if verbose:
                        print(f"  ⚠️  Cannot utime {out_path}: {e}")

        # Restore config.json if we had one at top level
        config = src_meta.get("config")
        if config and not (out_dir / "config.json").exists():
            (out_dir / "config.json").write_text(json.dumps(config, indent=2))

        if verbose: print(f"  {_fmt_bytes(Path(source).stat().st_size)} → {out_dir}/")
        if not keep:
            if _confirm_deletion(source):
                os.remove(source)
        return

    if recursive and source:
        import glob as _g
        sp = Path(source)
        files = [sp] if sp.is_file() else list(_g.glob(str(sp / "**/*.safetensors"), recursive=True)) + list(_g.glob(str(sp / "**/*.gguf"), recursive=True))
        to_delete = []
        for f in files:
            out = str(f) + ".vsqz"
            if Path(out).exists() and not force:
                if verbose: print(f"  Skip: {f}")
                continue
            convert_to_vsqz(str(f), out, quantize=quantize, verbose=verbose)
            if not keep:
                to_delete.append(f)
        if to_delete and _confirm_deletion(to_delete, quiet=quiet):
            for f in to_delete:
                os.remove(f)
        return

    if not source:
        print(_usage())
        return

    if output is None:
        output = source + ".vsqz"

    if Path(output).exists() and not force:
        print(f"Output exists: {output}. Use -f to overwrite.")
        sys.exit(1)

    if verbose:
        print(f"Compressing: {source} → {output}  [level {comp_level}/{quantize}]")

    import time
    t0 = time.time()
    _, stats = convert_to_vsqz(source, output, quantize=quantize, verbose=verbose)
    if not quiet:
        print(f"  {stats['original_size']} → {stats['final_size']} ({stats['compression_ratio']}× smaller, {time.time()-t0:.1f}s)")

    # ── Split into chunks (-s flag) ─────────────────────────────────
    if split_val:
        _size_units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
        unit = split_val[-1].upper()
        chunk_size = int(split_val[:-1]) * _size_units.get(unit, 1) if split_val[:-1].isdigit() else int(split_val)
        with open(output, "rb") as f:
            data = f.read()
        total = len(data)
        n_chunks = (total + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            chunk = data[i * chunk_size : (i + 1) * chunk_size]
            cname = f"{output}.{i+1:03d}"
            with open(cname, "wb") as f:
                f.write(chunk)
        os.remove(output)
        if not quiet:
            print(f"  Split into {n_chunks} chunk(s) of {_fmt_bytes(chunk_size)}: {output}.001 — {output}.{n_chunks:03d}")

    # zstd post-compression
    if do_zstd and Path(output).exists() and output.endswith('.vsqz'):
        zst_path = _apply_zstd(output, keep_original=keep)
        if not quiet:
            print(f"  zstd: {_fmt_bytes(Path(output).stat().st_size)} → {_fmt_bytes(Path(zst_path).stat().st_size)}")
        output = zst_path

    if not keep and Path(source).is_file():
        if _confirm_deletion(source, quiet=quiet):
            os.remove(source)


if __name__ == "__main__":
    main()
