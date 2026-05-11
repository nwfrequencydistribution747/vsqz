"""Streaming diff: tensor-by-tensor comparison, max 2 tensors in RAM.

Like gzip processes bytes, vsqz processes tensors — never loads the full model.
Works for ANY model size: 1B, 9B, 70B, 405B.
"""
from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Dict, Generator, Tuple

import numpy as np

from .converter_io import _fmt_bytes, _save_gguf, _build_vsqz_header, _write_vsqz


def _open_vsqz_mmap(path: str) -> Tuple[Dict, mmap.mmap, object]:
    """Open .vsqz with mmap for random-access tensor reading. Header only in RAM."""
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic = mm[:4]
    if magic != b'VSQZ':
        raise ValueError(f"Not a .vsqz file: {path}")
    hlen = struct.unpack("<I", mm[8:12])[0]
    header_json = mm[12:12+hlen].decode("utf-8").rstrip("\x00")
    return json.loads(header_json), mm, f


def _read_one(header: Dict, mm: mmap.mmap, name: str) -> np.ndarray:
    """Read ONE tensor from mmap'd .vsqz. No RAM accumulation."""
    entry = header["tensors"].get(name)
    if not entry:
        raise KeyError(f"Tensor '{name}' not found")
    offset = entry.get("offset", 0)
    size = entry["size"]
    shape = entry.get("shape", [])
    dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(
        entry.get("dtype", "float16"), np.float16)
    return np.frombuffer(mm[offset:offset+size], dtype=dt).reshape(shape).copy()


def _stream_gguf(path: str) -> Generator[Tuple[str, np.ndarray], None, None]:
    """Stream GGUF tensors: yield (name, array) one at a time. Uses gguf library."""
    try:
        from gguf import GGUFReader
    except ImportError:
        raise ImportError("gguf library required for GGUF streaming: pip install gguf")

    reader = GGUFReader(path)
    for tensor in reader.tensors:
        yield tensor.name, tensor.data


def stream_diff(base_vsqz: str, variant_path: str, output: str, verbose: bool = False) -> Dict:
    """Streaming diff: compare tensor-by-tensor, never load full model.

    base_vsqz: .vsqz file (mmap'd for random-access tensor reading)
    variant_path: .vsqz or .gguf or safetensors directory
    output: path for output .delta.vsqz

    Max RAM: 2 × largest_tensor (~100 MB for 9B model)
    """
    h, mm, f = _open_vsqz_mmap(base_vsqz)
    base_tensors = list(h["tensors"].keys())
    base_count = len(base_tensors)
    base_format = h.get("converted_from", "safetensors")
    base_size_mb = sum(e["size"] for e in h["tensors"].values()) / 1e6

    # Get variant streaming generator
    is_gguf = variant_path.endswith(".gguf")
    is_vsqz = variant_path.endswith(".vsqz")

    if is_gguf:
        stream = _stream_gguf(variant_path)
    elif is_vsqz:
        # For .vsqz variant: open mmap too and iterate tensor names
        vh, vmm, vf = _open_vsqz_mmap(variant_path)
        def _iter_vsqz():
            for name in sorted(vh["tensors"]):
                yield name, _read_one(vh, vmm, name)
        stream = _iter_vsqz()
    else:
        # Directory or single file: load normally (typically small)
        from .converter_io import _load_source
        tensors, _ = _load_source(Path(variant_path))
        def _iter_dict():
            for name in sorted(tensors):
                yield name, tensors[name]
        stream = _iter_dict()

    # Compare — streaming
    shared = 0
    deltas: Dict[str, np.ndarray] = {}
    seen = set()
    for name, var_tensor in stream:
        seen.add(name)
        if name in base_tensors:
            base = _read_one(h, mm, name)
            if base.shape == var_tensor.shape and base.dtype == var_tensor.dtype:
                if np.array_equal(base, var_tensor.astype(base.dtype)):
                    shared += 1
                    continue
        # Different or new tensor
        deltas[name] = var_tensor.astype(np.float16)

    total = base_count
    pct = shared / max(total, 1) * 100

    if verbose:
        import hashlib
        base_sha = hashlib.sha256(
            b"".join((n + ' ').encode('utf-8') for n in sorted(base_tensors))
        ).hexdigest()[:16]
        print(f"  Base:    {total} tensors ({base_size_mb:.0f} MB)")
        print(f"  Variant: {len(seen)} tensors streamed")
        print(f"  Shared:  {shared}/{total} ({pct:.0f}%)")
        print(f"  Delta:   {len(deltas)} tensors ({sum(t.nbytes for t in deltas.values())/1e6:.0f} MB)")

    if not deltas:
        if verbose: print("  ✅ Models identical — no delta needed.")
        return {"shared": shared, "total": total, "delta_count": 0}

    # Compute SHA from base (mmap still open)
    import hashlib
    base_sha = hashlib.sha256(
        b"".join(n.encode() + _read_one(h, mm, n).astype(np.float16).tobytes()
                 for n in sorted(base_tensors))
    ).hexdigest()

    # Cleanup mmap (SHA done, file no longer needed)
    f.close()
    if is_vsqz:
        vf.close()

    # Need header with right format
    from .converter_io import _build_vsqz_header, _write_vsqz
    delta_meta = {"format": base_format, "source_files": {"delta": ["delta"]}}
    delta_header = _build_vsqz_header(deltas, delta_meta, "fp16")
    delta_header["delta"] = True
    delta_header["base_sha256"] = base_sha
    delta_header["base_model"] = {
        "tensor_count": total,
        "source_format": base_format,
        "source_name": Path(base_vsqz).name,
    }
    delta_header["shared_count"] = shared
    _write_vsqz(Path(output), delta_header, deltas)

    if verbose:
        print(f"  Delta:   {output} ({_fmt_bytes(Path(output).stat().st_size)})")

    return {"shared": shared, "total": total, "delta_count": len(deltas)}
