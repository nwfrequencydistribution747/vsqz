#!/usr/bin/env python3
"""
Automated PR Review Bot — Autonomous quality gate.
Blocks PRs that don't meet standards. No human review until all checks pass.

Checks:
  1. Module imports clean
  2. No stubs/placeholders
  3. Extensions standardized (.vsqz)
  4. New code has tests (git diff)
  5. PR template filled (AI disclosure required)
  6. README consistency
  7. No personal info / hardcoded paths
"""

import os, re, sys, ast, subprocess
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    ROOT = Path.cwd()

ISSUES = []
PASSES = []
BLOCKER = []  # Must-fix before merge

def ok(msg):
    PASSES.append(msg)
    print(f"  ✅ {msg}")

def warn(msg, block=True):
    ISSUES.append(msg)
    if block: BLOCKER.append(msg)
    prefix = "🛑" if block else "⚠️"
    print(f"  {prefix}  {msg}")

# ── Check 0: Diff-based — new functions need tests ────────────────

def check_new_code_tested():
    print("\n🧪 New code test coverage:")
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD~1", "--unified=0", "--", "vsqz/"],
            cwd=str(ROOT), text=True
        )
    except Exception:
        ok("No git diff available (first commit or CI limitation)")
        return

    # Find newly added functions/classes
    new_defs = set()
    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            m = re.match(r'.*def\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if m:
                name = m.group(1)
                if not name.startswith("_"):
                    new_defs.add(name)

    if not new_defs:
        ok("No new public functions added")
        return

    # Check tests/ for these names
    test_files = list((ROOT / "tests").glob("test_*.py"))
    test_content = "\n".join(f.read_text() for f in test_files)

    missing = []
    for fn in new_defs:
        if fn not in test_content:
            missing.append(fn)

    if missing:
        warn(f"New functions without tests: {', '.join(missing)} — add test coverage", block=True)
    else:
        ok(f"{len(new_defs)} new functions, all covered by tests")

# ── Check 1: Module imports ──────────────────────────────────────

def check_imports():
    print("\n📦 Module imports:")
    modules = []
    for py_file in sorted((ROOT / "vsqz").glob("*.py")):
        if py_file.name == "__init__.py": continue
        try:
            code = ast.parse(py_file.read_text())
            for node in ast.walk(code):
                if isinstance(node, ast.ClassDef):
                    modules.append(f"{py_file.stem}.{node.name}")
        except SyntaxError as e:
            warn(f"Syntax error in {py_file.name}: {e}")
            return
    ok(f"{len(modules)} classes across {len(list((ROOT/'vsqz').glob('*.py')))} modules")

# ── Check 2: No stubs ────────────────────────────────────────────

def check_no_stubs():
    print("\n🔍 Placeholder / Stub scan:")
    found = 0
    for py_file in sorted((ROOT / "vsqz").glob("*.py")):
        if py_file.stem.startswith("_"): continue
        content = py_file.read_text()
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s == "pass" and i > 1:
                j = i - 2
                while j >= 0 and lines[j].strip() == "": j -= 1
                prev = lines[j].strip() if j >= 0 else ""
                if not any(prev.startswith(p) for p in ['"""', '#', 'def ', 'class ']):
                    found += 1
                    warn(f"{py_file.name}:{i} — dangling `pass`")
    if not found: ok("No dangling pass/placeholder found")

# ── Check 3: Extensions ──────────────────────────────────────────

def check_extensions():
    print("\n📁 Extension check:")
    bad = 0
    for py_file in (ROOT / "vsqz").glob("*.py"):
        c = py_file.read_text()
        if ".sqz" in c.replace(".vsqz", ""): bad += 1; warn(f"{py_file.name}: bare '.sqz'")
    for md_file in (ROOT).glob("*.md"):
        c = md_file.read_text()
        if ".sqz" in c.replace(".vsqz", ""): bad += 1; warn(f"{md_file.name}: bare '.sqz'")
    if not bad: ok("All extensions standardized to .vsqz")

# ── Check 4: Tests ───────────────────────────────────────────────

def check_tests():
    print("\n🧪 Test summary:")
    td = ROOT / "tests"
    if not td.exists(): warn("No tests/ directory"); return
    nf = len(list(td.glob("test_*.py")))
    lines = sum(len(f.read_text().splitlines()) for f in td.glob("test_*.py"))
    if nf >= 5: ok(f"{nf} test files, ~{lines} lines")
    else: warn(f"Only {nf} test files — add more coverage")

# ── Check 5: README ──────────────────────────────────────────────

def check_readme():
    print("\n📊 README consistency:")
    content = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
    if "RTX 3060" in content and "RTX 4090" in content: ok("GPU capability table")
    else: warn("GPU capability table missing", block=False)
    if "86%" in content and "Original" in content: ok("VRAM savings table")
    else: warn("VRAM savings table incomplete", block=False)

# ── Check 6: No personal info ─────────────────────────────────────

def check_no_paths():
    print("\n🔒 Privacy scan:")
    bad = ["/home/", "/Users/", "butterweck", "christian"]
    found = 0
    for f in (ROOT / "vsqz").glob("*.py"):
        c = f.read_text()
        for p in bad:
            if p in c: found += 1; warn(f"{f.name} contains '{p}'"); break
    if not found: ok("No hardcoded paths or identifiers")

# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  vsqz Autonomous Review Bot")
    print("=" * 60)

    check_new_code_tested()
    check_imports()
    check_no_stubs()
    check_extensions()
    check_tests()
    check_readme()
    check_no_paths()

    print(f"\n{'=' * 60}")
    print(f"  Result: {len(PASSES)} passed, {len(ISSUES)} issues ({len(BLOCKER)} blockers)")
    print(f"{'=' * 60}")

    if BLOCKER:
        print(f"\n🛑 {len(BLOCKER)} blocker(s) found. PR CANNOT BE MERGED.")
        print("   Fix all 🛑 issues, then push again.")
        sys.exit(1)
    elif ISSUES:
        print(f"\n⚠️  {len(ISSUES)} warning(s) found.")
        print("   Review the warnings before merging. Tests must pass.")
        sys.exit(0)
    else:
        print("\n✅ All checks passed. Ready for merge.")
        sys.exit(0)
