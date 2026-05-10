"""
v0.2.0-dev — VRAM Savings Benchmark
====================================
Measures actual VRAM savings per technique and combination.
Run on idle GPU. No trading deps.

Usage: python tests/benchmark_vram.py [--quick] [--all]

Output: Table showing VRAM before/after with actual GiB measurements.
"""

import gc, sys, time, torch, torch.nn as nn
from torch.optim import AdamW
from vsqz import VRAMSqueeze, VRAMEstimator


def _vram_gb() -> float:
    """Current PyTorch GPU memory allocated in GiB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def _reset_vram():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


class BenchmarkModel(nn.Module):
    """Medium model for VRAM measurement — simulates a LoRA trainable subset."""
    def __init__(self, hidden=2048, layers=12):
        super().__init__()
        self.embed = nn.Embedding(1000, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        ])
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.embed(x).mean(dim=1)
        for b in self.blocks: x = b(x)
        return self.head(x)


def measure_technique(name, model, opt, **kwargs):
    """Measure VRAM with a given technique active. Returns delta in GiB."""
    _reset_vram()
    m = model if model else BenchmarkModel()
    o = opt(m.parameters(), lr=1e-4) if opt else AdamW(m.parameters(), lr=1e-4)

    if name == "baseline":
        sq = None
    else:
        sq = VRAMSqueeze(m, mode="training", optimizer=o, **kwargs)

    # Forward + backward + step
    try:
        x = torch.randint(0, 1000, (2, 64), device="cuda" if torch.cuda.is_available() else "cpu")
        if sq: sq.step_begin()
        loss = m(x.to(next(m.parameters()).device)).sum()
        loss.backward()
        if sq:
            sq.step_end()
        else:
            o.step()
        if sq: sq.zero_grad()
        else: o.zero_grad()
    except torch.OutOfMemoryError:
        return float("inf"), float("inf")

    vram = _vram_gb()
    del m, o, sq, x, loss
    _reset_vram()
    return round(vram, 2)


def run_benchmark(quick=False):
    results = []

    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU available — running CPU-only estimates (not real measurements)")
        print("   Install on a GPU machine for real VRAM numbers.\n")

    vram_budget = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8
    if vram_budget < 12:
        hidden, layers = 512, 3
    elif vram_budget < 20:
        hidden, layers = 1024, 4
    else:
        hidden, layers = 1024, 6  # Smaller baseline that actually fits
    if quick:
        hidden, layers = 512, 3

    configs = [
        ("Baseline (no vsqz)", {}),
        ("GaLore r=128", {"galore_rank": 128, "lisa_ratio": 1.0, "fp16_states": False}),
        ("LISA 50%", {"galore_rank": None, "lisa_ratio": 0.5, "fp16_states": False}),
        ("FP16 States", {"galore_rank": None, "lisa_ratio": 1.0, "fp16_states": True}),
        ("GaLore + FP16", {"galore_rank": 128, "lisa_ratio": 1.0, "fp16_states": True}),
        ("GaLore + LISA", {"galore_rank": 128, "lisa_ratio": 0.5, "fp16_states": False}),
        ("GaLore + LISA + FP16", {"galore_rank": 128, "lisa_ratio": 0.5, "fp16_states": True}),
    ]

    baseline_vram = None
    for name, cfg in configs:
        print(f"  Measuring: {name}...", end=" ", flush=True)
        m = BenchmarkModel(hidden=hidden, layers=layers)
        m = m.cuda() if torch.cuda.is_available() else m
        o = AdamW(m.parameters(), lr=1e-4)

        _reset_vram()
        sq = None
        if cfg:
            sq = VRAMSqueeze(m, mode="training", optimizer=o, **cfg)

        vram = 0
        oom = False
        try:
            x = torch.randint(0, 1000, (2, 64)).cuda() if torch.cuda.is_available() else torch.randint(0, 1000, (2, 64))
            if sq: sq.step_begin()
            out = m(x); loss = out.sum(); loss.backward()
            if sq: sq.step_end()
            else: o.step()
            if sq: sq.zero_grad()
            else: o.zero_grad()
            torch.cuda.synchronize()
            vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except torch.OutOfMemoryError:
            oom = True
            vram = 25.0  # >24 = OOM

        if baseline_vram is None:
            baseline_vram = vram

        saved = round(baseline_vram - vram, 2) if not oom else float("-inf")
        results.append({
            "technique": name,
            "vram_gb": round(vram, 2),
            "saved_gb": saved,
            "reduction_pct": round(saved / max(baseline_vram, 0.01) * 100, 1),
            "oom": oom,
        })
        print(f"{vram:.1f} GB (saved {saved:+.1f} GB)")
        del m, o, sq, x, out, loss
        _reset_vram()

    # Print table
    print(f"\n{'=' * 65}")
    print(f"  VRAM Benchmark — {'Quick' if quick else 'Full'} Mode (hidden={hidden}, layers={layers})")
    print(f"{'=' * 65}")
    print(f"  {'Technique':<30} {'VRAM':>6} {'Saved':>7} {'Reduction':>9}")
    print(f"  {'-' * 30} {'-' * 6} {'-' * 7} {'-' * 9}")
    for r in results:
        status = "OOM" if r["oom"] else f"{r['saved_gb']:+.1f} GB"
        print(f"  {r['technique']:<30} {r['vram_gb']:>5.1f}G {status:>7} {r['reduction_pct']:>8.1f}%")
    print(f"{'=' * 65}")
    print(f"  Baseline: {baseline_vram:.1f} GB → Best: {min(r['vram_gb'] for r in results):.1f} GB")
    print(f"  Total savings: {max(r['saved_gb'] for r in results):.1f} GB")

    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_benchmark(quick=quick)
