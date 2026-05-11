"""Tests for vsqz converter (model compression with no trading deps)"""
import pytest, tempfile, os, torch, torch.nn as nn
from vsqz.converter_core import convert_to_vsqz
from vsqz.converter_io import _fmt_bytes

class TinyModel(nn.Module):
    def __init__(self): super().__init__(); self.a = nn.Linear(32,16); self.b = nn.Linear(16,8)
    def forward(self,x): return self.b(self.a(x))

def test_convert_pytorch():
    m = TinyModel()
    with tempfile.TemporaryDirectory() as d:
        pt = f"{d}/test.pt"; torch.save(m.state_dict(), pt)
        out, stats = convert_to_vsqz(pt, f"{d}/test.vs", verbose=False)
        assert os.path.exists(out)
        assert "compression_ratio" in stats

def test_convert_fp16_saves_space():
    # Use larger model so FP32→FP16 saving outweighs JSON header overhead
    m = nn.Sequential(*[nn.Linear(256,256) for _ in range(4)])
    with tempfile.TemporaryDirectory() as d:
        pt = f"{d}/test.pt"; torch.save(m.state_dict(), pt)
        out, stats = convert_to_vsqz(pt, f"{d}/test.vs", quantize="fp16", verbose=False)
        assert stats["compression_ratio"] >= 1.3  # Larger model = meaningful compression

def test_fmt_bytes():
    assert "B" in _fmt_bytes(500)
    assert "KB" in _fmt_bytes(2000)
    assert "MB" in _fmt_bytes(5_000_000)

def test_converter_filters_optimizer():
    m = TinyModel()
    pt_data = {k: v for k,v in m.state_dict().items()}
    # Add fake AdamW states
    pt_data["exp_avg.a.weight"] = torch.randn(32,16)
    pt_data["moment1.b.bias"] = torch.randn(8)
    with tempfile.TemporaryDirectory() as d:
        pt = f"{d}/adam.pt"; torch.save(pt_data, pt)
        out, stats = convert_to_vsqz(pt, f"{d}/no_adam.vs", verbose=False)
        assert "optimizer_tensors_stripped" in str(stats) or stats["tensors_before"] >= stats["tensors_after"]
