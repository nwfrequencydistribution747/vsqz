"""
converter_cli.py — CLI entry point for vsqz converter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .argparse_vsqz import parse_args
from .converter_core import (
    _do_test, _do_list, _do_decompress, _do_recursive, _do_compress, _do_mmproj, _do_update,
)
from .converter_delta import _do_diff, _do_serve, _do_rediff
from .converter_io import _decompress_zstd, _fmt_bytes


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
  --diff             Compute delta between base and variant (only differing weights)
  --rediff           Reconstruct full model from base + delta
  --serve            Multi-model: load base once, apply deltas on top
  -o, --output OUT   Output file (for --diff/--rediff)
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
  vsqz -r models/                      # compress all models in directory
  vsqz --diff base.vsqz fine.gguf -o delta.vsqz    # compute delta (shared weights)
  vsqz --rediff base.pt delta.vsqz -o fine.gguf    # reconstruct from base+delta
  vsqz --serve base.vsqz delta1.vsqz delta2.vsqz   # multi-model: shared base + deltas
  vsqz -u old_model.vsqz new_model.vsqz              # upgrade to latest format (adds per-tensor SHA)
  vsqz --mmproj HF_model_dir/ -o mmproj.gguf        # extract vision encoder (all VL archs)"""


def main():
    parsed = parse_args()
    if parsed.help:
        print(_usage())
        return

    keep = parsed.keep
    decompress = parsed.decompress
    quiet = parsed.quiet
    force = parsed.force
    do_zstd = parsed.zstd
    do_test = parsed.test
    do_list = parsed.list
    do_diff = parsed.diff
    do_serve = parsed.serve
    do_rediff = parsed.rediff
    do_mmproj = parsed.mmproj
    do_update = parsed.update
    server_port = parsed.port
    recursive = parsed.recursive
    split_val = parsed.split
    exclude_pats = parsed.exclude
    verbose = parsed.verbose
    comp_level = parsed.comp_level
    quantize = parsed.quantize
    source = parsed.source
    output = parsed.args[1] if len(parsed.args) >= 2 else None  # second positional
    extra_output = parsed.output  # -o/--output flag value

    # ── Modes ──

    if do_test and source:
        _do_test(source, verbose)
        return

    if do_list and source:
        _do_list(source)
        return

    # ── Diff ────────────────────────────────────────────────────────
    if do_diff and source and output:
        _do_diff(source, output, extra_output, verbose)
        return

    # ── Serve ───────────────────────────────────────────────────────
    if do_serve and source:
        deltas = [a for a in parsed.args[1:] if a.endswith('.vsqz') or a.endswith('.gguf')]
        if not deltas:
            print("Usage: vsqz --serve base_model delta1.vsqz [delta2.vsqz ...]")
            print("       vsqz --serve base_model delta1.vsqz --status")
            print("       vsqz --serve base_model delta1.vsqz --port 8081")
            sys.exit(1)

        if server_port > 0:
            from .server import serve_models
            serve_models(source, deltas, base_port=server_port)
            import signal; signal.pause()
        else:
            _do_serve(source, parsed.args, verbose)
        return

    # ── Rediff ──────────────────────────────────────────────────────
    if do_rediff and source and output:
        _do_rediff(source, output, extra_output, verbose)
        return

    # ── Update ──────────────────────────────────────────────────────
    if do_update and source:
        _do_update(source, extra_output or output, verbose)
        return

    # ── mmproj ──────────────────────────────────────────────────────
    # Optional: vsqz --mmproj base.vsqz delta.vsqz -o mmproj.gguf
    if do_mmproj and source:
        delta_file = output if output and (output.endswith('.delta.vsqz') or '.delta.vsqz' in output) else None
        _do_mmproj(source, extra_output, delta=delta_file, verbose=verbose)
        return

    # Handle .vsqz.zst transparently (decompress first)
    if source and source.endswith('.zst'):
        if verbose: print(f"Decompressing zstd: {source}")
        source = _decompress_zstd(source)

    if decompress and source:
        _do_decompress(source, output, keep, verbose)
        return

    if recursive and source:
        _do_recursive(source, quantize, keep, force, verbose, quiet)
        return

    if not source:
        print(_usage())
        return

    if output is None:
        output = source + ".vsqz"

    _do_compress(source, output, keep, force, verbose, quiet, quantize, comp_level, do_zstd, split_val)


if __name__ == "__main__":
    main()
