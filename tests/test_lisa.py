"""Tests for vsqz.lisa"""
import pytest, torch, torch.nn as nn
from vsqz.lisa import LISASampler, _collect_transformer_layers

class MiniTransformer(nn.Module):
    def __init__(self, dim=128, layers=6):
        super().__init__()
        self.embed = nn.Embedding(500, dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=dim, nhead=4, batch_first=True)
            for _ in range(layers)])
        self.head = nn.Linear(dim, 2)
    def forward(self, x): return self.head(self.embed(x))

def test_collect_layers():
    m = MiniTransformer()
    layers = _collect_transformer_layers(m)
    assert len(layers) == 6

def test_lisa_init():
    m = MiniTransformer()
    s = LISASampler(m, active_layers_ratio=0.5)
    assert s.num_layers == 6

def test_lisa_select():
    m = MiniTransformer()
    s = LISASampler(m, active_layers_ratio=0.5, seed=42)
    active = s.select_active_layers()
    assert 2 <= len(active) <= 4

def test_lisa_restore():
    m = MiniTransformer()
    s = LISASampler(m, active_layers_ratio=0.3, seed=42)
    s.select_active_layers()
    pre = s.active_count
    s.restore_all_layers()
    assert s.active_count == 6

def test_lisa_warmup():
    m = MiniTransformer()
    s = LISASampler(m, active_layers_ratio=0.5, warmup_steps=3, seed=42)
    for i in range(5):
        s.select_active_layers()
        if i < 3: assert s.active_count == 6
        else: assert s.active_count < 6

def test_lisa_context():
    m = MiniTransformer()
    s = LISASampler(m, active_layers_ratio=0.5, seed=42)
    with s.sample_layers():
        assert s.active_count < 6
