"""
converter_io.py — Pure I/O and utility functions for vsqz converter.
No business logic — just loading, writing, and formatting.
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

# ── Constants ─────────────────────────────────────────────────────────

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

# ── Deletion confirmation (module-level cache) ────────────────────────

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


# ── Utility ───────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    """Format bytes human-readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# GPU auto-detection for tensor offloading
_GPU_AVAILABLE = False
_GPU_OOM = False
try:
    import torch as _tcheck
    if _tcheck.cuda.is_available():
        _GPU_AVAILABLE = True
except Exception:
    pass  # noqa

_VL_PREFIXES = ("v.blk", "v.patch_embd", "v.post_ln", "v.position_embd", "mm.",
                "visual.", "vision_tower", "vision_encoder", "vision_model",
                "multi_modal_projector", "mlp1.", "siglip",
                "perceiver", "resampler",
                # Audio
                "audio_encoder", "speech_encoder", "whisper", "wav2vec2",
                "hubert", "audio_tower")


def _filter_vision_tensors(tensors):
    """Extract vision-encoder tensors from a full model dict. Works across all VL architectures."""
    vision = {}
    for name, tensor in tensors.items():
        lower = name.lower()
        if any(lower.startswith(p) for p in _VL_PREFIXES):
            vision[name] = tensor
    return vision


def _to_gpu(tensors: dict) -> dict:
    """Move tensors to GPU if available and beneficial. Falls back to CPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return tensors
        free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        total_bytes = sum(t.nbytes for t in tensors.values())
        if free_vram > total_bytes * 1.3:  # 30% headroom
            return {n: torch.from_numpy(t).cuda() for n, t in tensors.items()}
    except Exception:
        pass  # noqa
    return tensors


def _compute_delta(base_tensors, variant_tensors, tolerance=0):
    """Return tensors that differ between base and variant.

    Returns:
        (shared_count, delta_tensors) — delta_tensors contains only differing entries.
    """
    shared = 0
    deltas = {}
    for name in sorted(variant_tensors):
        v = variant_tensors[name]
        if name in base_tensors:
            b = base_tensors[name]
            if b.shape == v.shape and b.dtype == v.dtype:
                # Handle both numpy and torch tensors
                if hasattr(b, 'cpu'):  # torch tensor
                    if _tcheck.equal(b.cpu(), v.cpu() if hasattr(v, 'cpu') else _tcheck.from_numpy(v)):
                        shared += 1
                        continue
                elif np.array_equal(b, v):
                    shared += 1
                    continue
        # Tensor differs or doesn't exist in base — include in delta
        deltas[name] = v.copy()
    return shared, deltas


# ── Source loaders ────────────────────────────────────────────────────

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
            # config.json may be invalid or binary — non-critical
            pass  # noqa

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
                    elif isinstance(subval, np.ndarray):
                        tensors[f"{key}.{subkey}"] = subval
                        tensor_names.append(f"{key}.{subkey}")
                    elif isinstance(subval, (int, float, str)):
                        metadata[f"{key}.{subkey}"] = subval
            elif hasattr(value, "numpy"):
                tensors[key] = value.cpu().numpy()
                tensor_names.append(key)
            elif isinstance(value, np.ndarray):
                tensors[key] = value
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
            arr = np.frombuffer(data, dtype=dtype).reshape(info["shape"]).copy()
            # GPU auto-offload: if available and VRAM sufficient
            if _GPU_AVAILABLE and not _GPU_OOM:
                try:
                    import torch as _t
                    sz = arr.nbytes
                    free = _t.cuda.get_device_properties(0).total_memory - _t.cuda.memory_allocated()
                    if free > sz * 1.5 + 512 * 1024 * 1024:
                        tensors[info["name"]] = _t.from_numpy(arr).cuda()
                        continue
                    else:
                        _GPU_OOM = True  # stop trying
                except Exception:
                    _GPU_OOM = True
            tensors[info["name"]] = arr

    return tensors, metadata


# ── Header builder ────────────────────────────────────────────────────

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
            "sha256": "0" * 64,  # placeholder — filled by _write_vsqz (same length, no overflow)
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


# ── Writer ─────────────────────────────────────────────────────────────

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
            # Per-tensor SHA for blazing-fast diff comparisons (90%+ skip)
            header["tensors"][name]["sha256"] = hashlib.sha256(data).hexdigest()

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


# ── GGUF Serializer ───────────────────────────────────────────────────

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


# ── Zstd helpers ──────────────────────────────────────────────────────

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
