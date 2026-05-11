"""
converter_restore.py — Shared restore functions for vsqz converter.
Used by both decompress/recompress and --rediff paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .converter_io import _save_gguf


def _restore_tensors(tensors_np, header, out_dir, verbose=False):
    """Write tensors to output directory in original format. Shared: decompress + rediff."""
    orig_fmt = header.get("converted_from", "pytorch")
    src_files = header.get("source_metadata", {}).get("source_files", {})
    if orig_fmt == "safetensors" and src_files:
        from safetensors.torch import save_file
        for fname, tnames in sorted(src_files.items()):
            if not tnames: continue
            out_path = out_dir / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({n: torch.from_numpy(tensors_np[n].copy()) for n in tnames if n in tensors_np}, str(out_path))
            if verbose: print(f"  → {out_path}")
    elif orig_fmt == "gguf":
        _save_gguf(out_dir / "model.gguf", tensors_np,
                   header.get("source_metadata", {}), verbose=verbose)
    else:
        if src_files:
            for fname, tnames in sorted(src_files.items()):
                if not tnames: continue
                out_path = out_dir / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({n: torch.from_numpy(tensors_np[n].copy()) for n in tnames if n in tensors_np}, str(out_path))
                if verbose: print(f"  → {out_path}")
        else:
            out_path = out_dir / "pytorch_model.bin"
            torch.save({n: torch.from_numpy(tensors_np[n].copy()) for n in tensors_np}, str(out_path))
            if verbose: print(f"  → {out_path}")


def _restore_raw_files(out_dir, base_header, raw_deltas=None, verbose=False):
    """Restore non-tensor files from base + deltas. Shared: decompress + rediff."""
    import zstandard as _zstd, base64 as _b64
    dctx = _zstd.ZstdDecompressor()
    raw_deltas = raw_deltas or {}
    raw_written = 0

    if base_header.get("_raw_blobs"):
        for rel, (data, mode) in base_header["_raw_blobs"].items():
            if rel in raw_deltas and raw_deltas[rel][0] == "removed":
                continue
            rp = out_dir / rel
            rp.parent.mkdir(parents=True, exist_ok=True)
            if rel in raw_deltas and raw_deltas[rel][0] in ("changed", "new"):
                raw_data = raw_deltas[rel][1]
                if isinstance(raw_data, str):
                    raw_data = _b64.b64decode(raw_data)
                rp.write_bytes(dctx.decompress(raw_data))
                if len(raw_deltas[rel]) > 2:
                    try: rp.chmod(raw_deltas[rel][2])
                    except OSError: pass
                if len(raw_deltas[rel]) > 3 and raw_deltas[rel][3] is not None:
                    mtime = raw_deltas[rel][3]
                    atime = raw_deltas[rel][4] if len(raw_deltas[rel]) > 4 and raw_deltas[rel][4] is not None else mtime
                    try: os.utime(rp, (atime, mtime), follow_symlinks=False)
                    except OSError: pass
            else:
                rp.write_bytes(dctx.decompress(data))
                if mode:
                    try: rp.chmod(mode)
                    except OSError: pass
            raw_written += 1

    # Symlinks from delta
    for rel, entry in raw_deltas.items():
        if entry[0] == "symlink":
            rp = out_dir / rel
            rp.parent.mkdir(parents=True, exist_ok=True)
            if rp.exists(follow_symlinks=False): rp.unlink()
            rp.symlink_to(entry[1])
            if len(entry) > 2:
                try: os.utime(rp, (entry[2], entry[2]), follow_symlinks=False)
                except OSError: pass
            raw_written += 1

    # New files from delta (not in base _raw_blobs)
    for rel, entry in raw_deltas.items():
        if entry[0] != "new" or rel in base_header.get("_raw_blobs", {}):
            continue
        rp = out_dir / rel
        if rp.exists(): continue
        rp.parent.mkdir(parents=True, exist_ok=True)
        if len(entry) > 1 and entry[1] is not None:
            data = entry[1]
            if isinstance(data, str): data = _b64.b64decode(data)
            rp.write_bytes(dctx.decompress(data))
        if len(entry) > 2:
            try: rp.chmod(entry[2])
            except OSError: pass
        if len(entry) > 3 and entry[3] is not None:
            mtime = entry[3]
            atime = entry[4] if len(entry) > 4 and entry[4] is not None else mtime
            try: os.utime(rp, (atime, mtime), follow_symlinks=False)
            except OSError: pass
        raw_written += 1

    if verbose and raw_written:
        print(f"  + {raw_written} raw files restored")
    return raw_written


def _auto_rejoin_split(path: Path) -> Path:
    """If 'path' is a split archive, auto-rejoin and return combined path.

    Tries two naming conventions:
    - path.001, path.002, ... (e.g., base.vsqz.001)
    - prefix.001, prefix.002, ... (e.g., base.001 from split prefix 'base')
    Returns original path if no split chunks found or if path already exists.
    """
    if path.exists():
        return path

    # Try convention 1: path.001 (e.g., model.vsqz.001)
    first = Path(str(path) + ".001")
    if first.exists():
        chunks = sorted(path.parent.glob(path.name + ".*"))
        if chunks:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".vsqz")
            with open(tmp.name, "wb") as out:
                for c in sorted(chunks):
                    out.write(c.read_bytes())
            return Path(tmp.name)

    # Try convention 2: prefix.001 (e.g., base.001 from split prefix 'base')
    stem = str(path).rsplit(".", 1)[0] if "." in str(path) else str(path)
    first2 = Path(stem + ".001")
    if first2.exists():
        chunks = sorted(Path(stem).parent.glob(Path(stem).name + ".*"))
        if chunks:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".vsqz")
            with open(tmp.name, "wb") as out:
                for c in sorted(chunks):
                    out.write(c.read_bytes())
            return Path(tmp.name)

    return path
