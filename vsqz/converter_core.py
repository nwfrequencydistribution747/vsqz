"""
converter_core.py — Archive operations: convert, decompress, list, test, recursive, compress.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time as _time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .converter_io import (
    VSQZ_MAGIC, VSQZ_VERSION, TENSOR_ALIGNMENT, HEADER_ALIGNMENT, OPTIMIZER_KEYS,
    _TORCH_AVAILABLE as _TIA, logger,
    _fmt_bytes, _load_source, _build_vsqz_header, _write_vsqz,
    _decompress_zstd, _confirm_deletion,
    _filter_vision_tensors, _save_gguf,
)
from .converter_restore import _restore_tensors, _restore_raw_files


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
        quantize: "fp16" (default, 2x compression) or "int8" (4x, aggressive)
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
        print(f"  Saved:     {stats['compression_ratio']}x smaller")
        print(f"  -> {output_path}")

    return str(output_path), stats


def _do_test(source, verbose):
    """--test: Verify .vsqz file integrity."""
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


def _do_list(source):
    """--list: Show metadata without loading tensors."""
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

    # Delta self-description
    if h.get("delta"):
        bi = h.get("base_model", {})
        print(f"\n  🧬 Delta file — requires base model:")
        print(f"    Architecture:  {bi.get('architecture', '?')}")
        print(f"    Parameters:    {bi.get('total_params', 0)/1e9:.1f}B")
        print(f"    Layers:        {bi.get('layer_count', '?'):>4}  |  Tensors: {bi.get('tensor_count', '?'):>4}  |  Hidden: {bi.get('hidden_dim', '?') or '?'}")
        print(f"    Vocab:         {bi.get('vocab_size', '?'):>6}  |  Source: {bi.get('source_name', '?')}")
        print(f"    Base built:    {bi.get('source_mtime', '?')}  |  Delta created: {bi.get('delta_created', '?')}")
        print(f"    Base SHA-256:  {h.get('base_sha256', '?')[:32]}...")
        print(f"    Shared tensors: {h.get('shared_count', '?')}/{bi.get('tensor_count', '?')}")


def _do_decompress(source, output, keep, verbose):
    """--decompress: Restore original format from .vsqz."""
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

    # Restore tensors using shared function
    _restore_tensors(tensors_np, header, out_dir, verbose=verbose)

    # Restore raw files + symlinks
    _restore_raw_files(out_dir, header, verbose=verbose)

    # Restore symlinks (separate from raw blobs)
    for rel_path, target in sorted(src_meta.get("symlinks", {}).items()):
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists(follow_symlinks=False):
            out_path.unlink()
        out_path.symlink_to(target)
        if verbose: print(f"  -> {out_path} -> {target}")

    # Restore mtime/atime for all restored files
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

    if verbose: print(f"  {_fmt_bytes(Path(source).stat().st_size)} -> {out_dir}/")
    if not keep:
        if _confirm_deletion(source):
            os.remove(source)


def _do_mmproj(source, output, verbose=False):
    """Extract vision/audio encoder subset as GGUF (mmproj replacement).

    Works for any VL/audio model: Qwen-VL, LLaVA, Phi-4-Vision-Audio, etc.
    No HF conversion needed — pure tensor name filtering.
    """
    if not output:
        output = source + "-mmproj.gguz" if source.endswith('.vsqz') else source.rstrip('/') + '-mmproj.gguf'

    if verbose: print(f"Extracting mmproj: {source} → {output}")

    # Load source (HF dir, .vsqz, .gguf, .safetensors)
    tensors, meta = _load_source(Path(source))
    vision = _filter_vision_tensors(tensors)

    if not vision:
        print(f"  ⚠️  No vision/audio tensors found. Model may not be VL/audio.")
        print(f"  Detected prefixes: {_VL_PREFIXES}")
        sys.exit(1)

    if verbose:
        total = sum(t.nbytes for t in vision.values())
        print(f"  Vision tensors: {len(vision)} ({_fmt_bytes(total)})")

    # Write as GGUF
    gguf_meta = {"format": "gguf", "tensor_infos": [], "kv_pairs": {},
                 "version": 3, "source_files": {"mmproj.gguf": list(vision.keys())}}
    _save_gguf(Path(output), vision, gguf_meta, verbose=verbose)

    if verbose:
        print(f"  ✅ mmproj: {output} ({_fmt_bytes(Path(output).stat().st_size)})")


def _do_recursive(source, quantize, keep, force, verbose, quiet):
    """--recursive: Compress all compatible models in a directory tree."""
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


def _do_compress(source, output, keep, force, verbose, quiet, quantize, comp_level, do_zstd, split_val):
    """Normal compression: source -> output.vsqz, with optional split/zstd."""
    if Path(output).exists() and not force:
        print(f"Output exists: {output}. Use -f to overwrite.")
        sys.exit(1)

    if verbose:
        print(f"Compressing: {source} -> {output}  [level {comp_level}/{quantize}]")

    t0 = _time.time()
    _, stats = convert_to_vsqz(source, output, quantize=quantize, verbose=verbose)
    if not quiet:
        print(f"  {stats['original_size']} -> {stats['final_size']} ({stats['compression_ratio']}x smaller, {_time.time()-t0:.1f}s)")

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
        from .converter_io import _apply_zstd
        zst_path = _apply_zstd(output, keep_original=keep)
        if not quiet:
            print(f"  zstd: {_fmt_bytes(Path(output).stat().st_size)} -> {_fmt_bytes(Path(zst_path).stat().st_size)}")
        output = zst_path

    if not keep and Path(source).is_file():
        if _confirm_deletion(source, quiet=quiet):
            os.remove(source)
