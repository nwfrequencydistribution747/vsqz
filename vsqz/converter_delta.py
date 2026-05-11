"""
converter_delta.py — Delta operations: --diff, --serve, --rediff.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import numpy as np

from .converter_io import (
    VSQZ_MAGIC, VSQZ_VERSION, TENSOR_ALIGNMENT, HEADER_ALIGNMENT, OPTIMIZER_KEYS,
    _TORCH_AVAILABLE, logger,
    _fmt_bytes, _compute_delta, _load_source, _build_vsqz_header, _write_vsqz,
)
from .converter_restore import _restore_tensors, _restore_raw_files, _auto_rejoin_split


def _do_diff(source, variant, extra_output, verbose):
    """--diff: Compute delta between base and variant."""
    delta_out = extra_output if extra_output else (variant + ".delta.vsqz")
    if verbose: print(f"Computing delta: {source} vs {variant}")

    # Require .vsqz base for reproducible SHA-256
    if not source.endswith('.vsqz'):
        print("  Error: base must be a .vsqz file. Compress first: vsqz base.gguf base.vsqz")
        sys.exit(1)

    # Auto-rejoin split archives
    source_path = _auto_rejoin_split(Path(source))
    variant_path = Path(variant) if variant.endswith('.vsqz') else Path(variant)
    if variant.endswith('.vsqz'):
        variant_path = _auto_rejoin_split(variant_path)

    from .vsqz_format import _read_vsqz
    bh, bt = _read_vsqz(str(source_path), verify_sha256=True)
    base_tensors = {}
    for n in sorted(bh["tensors"]):
        e = bh["tensors"][n]
        d = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(e.get("dtype","float16"), np.float16)
        base_tensors[n] = np.frombuffer(bt[n], dtype=d).reshape(e.get("shape",[]))
    import hashlib as _hl2
    base_sha = _hl2.sha256(
        b"".join(
            n.encode() + base_tensors[n].astype(np.float16).tobytes()
            for n in sorted(base_tensors)
        )
    ).hexdigest()
    base_meta = {"format": bh.get("converted_from","safetensors")}

    # Load variant — may be .vsqz or raw format
    if variant.endswith('.vsqz'):
        from .vsqz_format import _read_vsqz as _rv3
        vh, vt = _rv3(str(variant_path), verify_sha256=True)
        var_tensors = {}
        for n in sorted(vh["tensors"]):
            e = vh["tensors"][n]
            d = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(e.get("dtype","float16"), np.float16)
            var_tensors[n] = np.frombuffer(vt[n], dtype=d).reshape(e.get("shape",[]))
        var_meta = {"format": vh.get("converted_from","safetensors"),
                   "raw_files": vh.get("source_metadata",{}).get("raw_files",{})}
    else:
        var_tensors, var_meta = _load_source(Path(variant))

    # Cast variant to base dtype for fair comparison
    var_normalized = {}
    for n, v in var_tensors.items():
        if n in base_tensors and v.shape == base_tensors[n].shape:
            var_normalized[n] = v.astype(base_tensors[n].dtype)
        else:
            var_normalized[n] = v

    shared, deltas = _compute_delta(base_tensors, var_normalized)
    total = len(base_tensors)
    pct = shared / max(total, 1) * 100

    # Compare raw files from base and variant (.vsqz or directory)
    base_raws = bh.get("source_metadata", {}).get("raw_files", {})
    var_raws = var_meta.get("raw_files", {}) if "raw_files" in var_meta else {}
    raw_deltas = {}
    all_raw = set(base_raws) | set(var_raws)
    base_raw_blobs = bh.get("_raw_blobs", {})
    var_is_dir = Path(variant).is_dir() if not variant.endswith('.vsqz') else False

    # Read variant raw blobs (from .vsqz or filesystem)
    import zstandard as _zstd5
    var_raw_blobs = {}
    if variant.endswith('.vsqz'):
        # Use full _read_vsqz to get _raw_blobs
        from .vsqz_format import _read_vsqz as _rv5
        vh_raw, _ = _rv5(variant)
        var_raw_blobs = vh_raw.get("_raw_blobs", {})
        # Decompress for comparison
        dctx_r = _zstd5.ZstdDecompressor()
        for rp, entry in var_raw_blobs.items():
            # entry is (compressed_data, mode)
            var_raw_blobs[rp] = (dctx_r.decompress(entry[0]), entry[1])

    for rp in sorted(all_raw):
        if rp in var_raws and rp not in base_raws:
            # New file in variant
            if var_is_dir:
                fp = Path(variant) / rp
                st = fp.lstat()
                raw_deltas[rp] = ("new", fp.read_bytes(), st.st_mode & 0o777,
                                  st.st_mtime, st.st_atime)
            elif rp in var_raw_blobs:
                raw_deltas[rp] = ("new", var_raw_blobs[rp][0], var_raw_blobs[rp][1])
            else:
                raw_deltas[rp] = ("new", None)
        elif rp in base_raws and rp not in var_raws:
            raw_deltas[rp] = ("removed", None)
        elif rp in base_raws and rp in var_raws:
            # Compare: base bytes vs variant bytes
            base_bytes = None
            if base_raw_blobs and rp in base_raw_blobs:
                base_bytes = _zstd5.ZstdDecompressor().decompress(base_raw_blobs[rp][0])
            var_bytes = None
            if var_is_dir and (Path(variant) / rp).exists():
                fp = Path(variant) / rp
                st = fp.lstat()
                var_bytes = fp.read_bytes()
                var_mode = st.st_mode & 0o777
                if base_bytes is None or base_bytes != var_bytes:
                    raw_deltas[rp] = ("changed", var_bytes, var_mode,
                                      st.st_mtime, st.st_atime)
            elif rp in var_raw_blobs:
                var_bytes = var_raw_blobs[rp][0]
                var_mode = var_raw_blobs[rp][1]
                if base_bytes is None or base_bytes != var_bytes:
                    raw_deltas[rp] = ("changed", var_bytes, var_mode)
                    # No mtime available from .vsqz source

    if raw_deltas:
        import zstandard as _zstd2, base64 as _b64
        cctx = _zstd2.ZstdCompressor(level=3)
        raw_deltas_compressed = {}
        for rp, entry in raw_deltas.items():
            if entry[0] in ("changed", "new") and len(entry) > 1 and entry[1] is not None:
                compressed = cctx.compress(entry[1])
                raw_deltas_compressed[rp] = (entry[0], _b64.b64encode(compressed).decode(),
                                             entry[2] if len(entry) > 2 else 0o644,
                                             entry[3] if len(entry) > 3 else None,
                                             entry[4] if len(entry) > 4 else None)
            else:
                raw_deltas_compressed[rp] = entry
        raw_deltas = raw_deltas_compressed

    if verbose:
        print(f"  Base: {len(base_tensors)} tensors ({_fmt_bytes(sum(t.nbytes for t in base_tensors.values()))})")
        print(f"  Variant: {len(var_tensors)} tensors ({_fmt_bytes(sum(t.nbytes for t in var_tensors.values()))})")
        print(f"  Shared: {shared}/{total} tensors ({pct:.1f}%)")
        print(f"  Delta: {len(deltas)} tensors ({_fmt_bytes(sum(t.nbytes for t in deltas.values()))})")
        if raw_deltas:
            print(f"  Raw delta: {len(raw_deltas)} files")

    if not deltas:
        print("  ✅ Models are identical — no delta needed.")
        return

    # Build comprehensive base metadata for self-documenting deltas
    total_params = sum(int(np.prod(t.shape)) for t in base_tensors.values())
    embed_key = next((n for n in base_tensors if 'embed' in n.lower()), None)
    vocab_size = int(base_tensors[embed_key].shape[0]) if embed_key else None
    hidden_dim = int(base_tensors[embed_key].shape[1]) if embed_key and len(base_tensors[embed_key].shape) > 1 else None
    layer_count = len([n for n in base_tensors if 'layers.' in n])  # approximate
    # Collect GGUF-style metadata if available
    kv_pairs = bh.get("source_metadata", {}).get("kv_pairs", {})
    arch = kv_pairs.get("general.architecture", {})
    arch_name = arch.get("value") if isinstance(arch, dict) else bh.get("model_config", {}).get("arch", "unknown")

    base_info = {
        "sha256": base_sha,
        "tensor_count": len(base_tensors),
        "total_params": total_params,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "layer_count": layer_count,
        "architecture": str(arch_name) if arch_name else "unknown",
        "source_format": bh.get("converted_from", "safetensors"),
        "source_name": Path(source).name,
        "source_mtime": datetime.datetime.fromtimestamp(
            Path(source).stat().st_mtime
        ).isoformat(),
        "delta_created": datetime.datetime.now().isoformat(),
    }
    if kv_pairs:
        base_info["gguf_metadata"] = kv_pairs

    # Write delta .vsqz — uses the verified base SHA from the .vsqz header
    delta_meta = {
        "format": base_meta.get("format", "safetensors"),
        "source_files": {"delta": ["delta"]},
    }
    delta_header = _build_vsqz_header(deltas, delta_meta, "fp16")
    delta_header["delta"] = True
    delta_header["base_sha256"] = base_sha
    delta_header["base_model"] = base_info
    delta_header["shared_count"] = shared
    if raw_deltas:
        delta_header["raw_file_deltas"] = raw_deltas
    _write_vsqz(Path(delta_out), delta_header, deltas)
    if verbose:
        print(f"  Delta written: {delta_out} ({_fmt_bytes(Path(delta_out).stat().st_size)})")


def _do_serve(source, args, verbose, show_status=False):
    """--serve: Multi-model shared base + deltas via ModelSwarm."""
    from .serve import ModelSwarm

    deltas = [a for a in args[1:] if a.endswith('.vsqz') or a.endswith('.gguf')]
    if not deltas:
        print("Usage: vsqz --serve base_model delta1.vsqz [delta2.vsqz ...]")
        sys.exit(1)

    if verbose: print(f"Serving multi-model: {source}")
    swarm = ModelSwarm(source, deltas)
    swarm.load(quiet=not verbose)

    if show_status:
        print(swarm.status())
    elif verbose:
        print(swarm.status())


def _do_rediff(source, delta_in, delta_out, verbose):
    """--rediff: Reconstruct from base + delta."""
    # Syntax: vsqz --rediff base.vsqz delta.vsqz -o reconstructed.gguf
    if not delta_out:
        print("Usage: vsqz --rediff base.vsqz delta.vsqz -o reconstructed.safetensors")
        sys.exit(1)

    if verbose:
        print(f"Reconstructing: {source} + {delta_in} → {delta_out}")

    # Load base (.vsqz or raw format) — auto-rejoin split archives
    if source.endswith('.vsqz'):
        source_path = _auto_rejoin_split(Path(source))
        from .vsqz_format import _read_vsqz
        bh, bt = _read_vsqz(str(source_path), verify_sha256=True)
        base_tensors = {}
        for n in sorted(bh["tensors"]):
            e = bh["tensors"][n]
            d = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(e.get("dtype","float16"), np.float16)
            base_tensors[n] = np.frombuffer(bt[n], dtype=d).reshape(e.get("shape",[]))
        base_meta = {"format": bh.get("converted_from","safetensors"),
                    "raw_files": bh.get("source_metadata",{}).get("raw_files",{})}
    else:
        base_tensors, base_meta = _load_source(Path(source))
    base_size = sum(t.nbytes for t in base_tensors.values())

    # Load delta
    from .vsqz_format import _read_vsqz
    dh, dd = _read_vsqz(delta_in)
    if not dh.get("delta"):
        print("  Error: delta must be a .vsqz delta file (use --diff to create)")
        sys.exit(1)

    # SHA verification
    import hashlib as _hl
    base_sha = _hl.sha256(
        b"".join(n.encode() + base_tensors[n].astype(np.float16).tobytes()
                 for n in sorted(base_tensors))
    ).hexdigest()
    expected_sha = dh.get("base_sha256", "")
    if expected_sha != base_sha:
        bi = dh.get("base_model", {})
        print(f"  ⚠️  BASE MISMATCH: delta expects {bi.get('architecture','?')}")
        print(f"      Expected SHA: {expected_sha[:16]}...")
        print(f"      Loaded SHA:   {base_sha[:16]}...")
        sys.exit(1)

    # Apply delta
    for name in sorted(dd):
        entry = dh["tensors"][name]
        d_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(
            entry.get("dtype", "float16"), np.float16)
        arr = np.frombuffer(dd[name], dtype=d_dt).reshape(entry["shape"])
        base_tensors[name] = arr
        # Cast to base dtype if needed
        if name in base_tensors and base_tensors[name].dtype != arr.dtype:
            base_tensors[name] = base_tensors[name].astype(arr.dtype)

    # Write reconstructed tensors + raw files using shared functions
    out_path = Path(delta_out)
    _MODEL_EXT = {".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt"}
    is_dir = (delta_out.endswith('/') or delta_out.endswith('\\')
             or out_path.suffix not in _MODEL_EXT)
    out_dir = out_path if is_dir else out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build minimal header for _restore_tensors
    from .vsqz_format import _read_vsqz as _rv2
    base_header, _ = _rv2(str(Path(source).resolve()))
    restore_header = {
        "converted_from": base_meta.get("format", "safetensors"),
        "source_metadata": base_header.get("source_metadata", {}),
    }
    _restore_tensors(base_tensors, restore_header, out_dir, verbose=verbose)

    # Restore raw files with delta changes applied
    raw_deltas = dh.get("raw_file_deltas", {})
    _restore_raw_files(out_dir, base_header, raw_deltas, verbose=verbose)

    # Restore symlinks from base (not affected by delta)
    base_meta_src = base_header.get("source_metadata", {})
    for rel_path, target in sorted(base_meta_src.get("symlinks", {}).items()):
        rp = out_dir / rel_path
        rp.parent.mkdir(parents=True, exist_ok=True)
        if rp.exists(follow_symlinks=False):
            rp.unlink()
        rp.symlink_to(target)
        if verbose: print(f"  → {rp} -> {target}")

    out_size = sum(t.nbytes for t in base_tensors.values())
    if verbose:
        print(f"  Reconstructed: {len(base_tensors)} tensors ({_fmt_bytes(out_size)})")
        print(f"  Source: base {_fmt_bytes(base_size)} + delta {_fmt_bytes(Path(delta_in).stat().st_size)}")
