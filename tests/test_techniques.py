"""Tests for vsqz novel techniques (sparse grad, delta, adaptive quant)"""
import pytest, torch, torch.nn as nn
from vsqz.sparse_grad import SparseGradientEncoder
from vsqz.gradient_delta import GradientDeltaTracker
from vsqz.adaptive_quant import SpatialGradientPredictor, AdaptiveLayerQuantizer, AdaptiveStepScaler

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64,64) for _ in range(3)])
    def forward(self, x):
        for l in self.layers: x = l(x); return x

def test_sparse_encoder():
    m = TinyModel(); loss=m(torch.randn(2,64)).sum(); loss.backward()
    enc = SparseGradientEncoder(sparsity_threshold=1e-3)
    params = [p for p in m.parameters() if p.grad is not None]
    ratio = enc.compress_gradients(params)
    enc.decompress_gradients(params)
    assert 0.0 <= ratio <= 1.0

def test_sparse_stats():
    enc = SparseGradientEncoder()
    s = enc.stats; assert "steps" in s

def test_delta_tracker_init():
    m = TinyModel()
    dt = GradientDeltaTracker(m)
    assert dt._step_counter == 0

def test_delta_compress():
    m = TinyModel(); loss=m(torch.randn(2,64)).sum(); loss.backward()
    dt = GradientDeltaTracker(m)
    for p in m.parameters():
        if p.grad is not None:
            result = dt.compress(p)
            if result is not None:
                saved, orig = result
                assert saved >= 0

def test_spatial_predictor():
    sp = SpatialGradientPredictor()
    name = "layers.1.linear.weight"
    assert "layers." in sp._get_layer_name(name)

def test_adaptive_layer_quantizer():
    aq = AdaptiveLayerQuantizer()
    g = torch.randn(64,64)
    aq.update_statistics("layers.0.weight", g)
    aq.update_statistics("layers.1.weight", g * 2)
    bits = aq.allocate_bits()
    assert len(bits) >= 1

def test_adaptive_step_scaler():
    sc = AdaptiveStepScaler()
    v = torch.randn(32,32) * 0.1
    q, scale = sc.encode("test", v)
    assert q.dtype == torch.int8
    decoded, _ = sc.decode("test", q)
    assert decoded.shape == (32,32)
