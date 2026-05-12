"""Argparse-based CLI for vsqz. Shared by all entry points."""

import argparse as _ap
import sys


def parse_args():
    """Parse all flags. Returns Namespace with typed values."""
    p = _ap.ArgumentParser(description="vsqz — gzip for AI models", add_help=False)

    # Modes
    p.add_argument("-d", "--decompress", action="store_true")
    p.add_argument("-l", "--list", action="store_true")
    p.add_argument("-t", "--test", action="store_true")
    p.add_argument("--diff", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--rediff", action="store_true")
    p.add_argument("--mmproj", action="store_true")
    p.add_argument("-u", "--update", action="store_true", help="Upgrade old .vsqz to latest format")
    p.add_argument("--status", action="store_true", help="Show VRAM comparison stats with --serve")
    p.add_argument("--port", type=int, default=0, metavar="PORT", help="Base port for multi-model server")

    # Behavior
    p.add_argument("-k", "--keep", action="store_true")
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", default=True)
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("-z", "--zstd", action="store_true")
    p.add_argument("-h", "--help", action="store_true")

    # Compression level: -1..-9
    for n in range(1, 10):
        p.add_argument(f"-{n}", dest="level", action="store_const", const=n)

    # With-value flags
    p.add_argument("-s", "--split", type=str, default=None, metavar="SIZE")
    p.add_argument("-x", "--exclude", action="append", default=[], metavar="KEY")
    p.add_argument("-o", "--output", type=str, default=None, metavar="OUT")

    # Positional: source [output]
    p.add_argument("args", nargs="*", help="<source> [<output>]")

    parsed, _ = p.parse_known_args()

    # Derived values
    parsed.verbose = not parsed.quiet
    parsed.comp_level = parsed.level or 6
    parsed.quantize = "int8" if parsed.comp_level >= 8 else "fp16"
    parsed.source = parsed.args[0] if len(parsed.args) >= 1 else None
    # -o/--output takes priority over second positional arg
    pos_output = parsed.args[1] if len(parsed.args) >= 2 else None
    if parsed.output is None:
        parsed.output = pos_output

    return parsed
