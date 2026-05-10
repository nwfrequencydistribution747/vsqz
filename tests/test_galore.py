"""Tests for vsqz.galore"""
import pytest, torch, torch.nn as nn
from torch.optim import AdamW
from vsqz.galore import GaLoreWrapper

class MiniModel(nn.Module):
    def __init__(self, dim=128, layers=4):
        super().__init__()
        self.embed = nn.Embedding(500, dim)
        self.layers = nn.ModuleList([nn.Linear(dim,dim) for _ in range(layers)])
        self.head = nn.Linear(dim, 2)
    def forward(self, x):
        x = self.embed(x)
        for l in self.layers: x = l(x)
        return self.head(x)

def test_galore_init():
    m = MiniModel(); o = AdamW(m.parameters(), lr=1e-4)
    w = GaLoreWrapper(m, o, rank=32)
    assert isinstance(w, torch.optim.Optimizer)

def test_galore_step():
    m = MiniModel(); o = AdamW(m.parameters(), lr=1e-4)
    w = GaLoreWrapper(m, o, rank=32)
    x = torch.randint(0,500,(2,8)); loss=m(x).sum(); loss.backward(); w.step(); w.zero_grad()

def test_galore_multiple_steps():
    m = MiniModel(); o = AdamW(m.parameters(), lr=1e-4)
    w = GaLoreWrapper(m, o, rank=32)
    for _ in range(5):
        x = torch.randint(0,500,(2,8)); loss=m(x).mean(); loss.backward(); w.step(); w.zero_grad()

def test_galore_state_dict():
    m = MiniModel(); o = AdamW(m.parameters(), lr=1e-4)
    w = GaLoreWrapper(m, o, rank=32)
    x = torch.randint(0,500,(2,8)); loss=m(x).mean(); loss.backward(); w.step()
    s = w.state_dict(); assert "rank" in s; w.load_state_dict(s)
