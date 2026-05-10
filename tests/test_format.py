"""Tests for vsqz format I/O"""
import pytest, tempfile, os, torch, torch.nn as nn
from torch.optim import AdamW
from vsqz import VRAMSqueeze
from vsqz.vsqz_format import peek_vsqz, load_vsqz_weights, _read_vsqz

class TinyModel(nn.Module):
    def __init__(self): super().__init__(); self.linear = nn.Linear(16,8)
    def forward(self,x): return self.linear(x)

def test_save_load_vsqz():
    m = TinyModel(); o = AdamW(m.parameters(),lr=1e-4)
    sq = VRAMSqueeze(m, mode='training', optimizer=o, galore_rank=8, fp16_states=True)
    for _ in range(3):
        loss=m(torch.randn(2,16)).sum(); loss.backward(); sq.step_end(); sq.zero_grad()
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/test.vsqz"
        sp = sq.save_vsqz(p)
        assert os.path.exists(sp)
        h = peek_vsqz(sp)
        assert 'tensors' in h

def test_checkpoint_save_load():
    m = TinyModel(); o = AdamW(m.parameters(),lr=1e-4)
    sq = VRAMSqueeze(m, mode='training', optimizer=o, galore_rank=8, fp16_states=True)
    for _ in range(3):
        loss=m(torch.randn(2,16)).sum(); loss.backward(); sq.step_end(); sq.zero_grad()
    with tempfile.TemporaryDirectory() as d:
        cp = sq.save_checkpoint(f"{d}/test.vsq.pt")
        assert os.path.exists(cp)
        m2 = TinyModel(); o2 = AdamW(m2.parameters(),lr=1e-4)
        sq2 = VRAMSqueeze(m2, mode='training', optimizer=o2, galore_rank=8, fp16_states=True)
        meta = sq2.load_checkpoint(cp)
        assert 'step' in meta or 'config' in meta

def test_peek_metadata():
    m = TinyModel(); o = AdamW(m.parameters(),lr=1e-4)
    sq = VRAMSqueeze(m, mode='training', optimizer=o, galore_rank=8)
    for _ in range(3):
        loss=m(torch.randn(2,16)).sum(); loss.backward(); sq.step_end(); sq.zero_grad()
    with tempfile.TemporaryDirectory() as d:
        sp = sq.save_vsqz(f"{d}/test.vsqz")
        h = peek_vsqz(sp)
        assert 'vsqz_version' in h or 'vram_squeeze_version' in h
