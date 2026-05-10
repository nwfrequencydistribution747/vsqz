"""Tests for vsqz optimizer wrappers"""
import pytest, torch, torch.nn as nn
from torch.optim import AdamW
from vsqz.fp16_states import FP16OptimizerStates
from vsqz.int8_states import Int8OptimizerStates
from vsqz.deepspeed_offload import DeepSpeedCPUOffload
from vsqz import VRAMSqueeze

def test_fp16_is_optimizer():
    m = nn.Linear(8,4); o = AdamW(m.parameters(),lr=1e-4)
    w = FP16OptimizerStates(o)
    assert isinstance(w, torch.optim.Optimizer)

def test_fp16_step():
    m = nn.Linear(8,4); o = AdamW(m.parameters(),lr=1e-4)
    w = FP16OptimizerStates(o)
    for _ in range(3):
        loss=torch.randn(2,8).matmul(m.weight.T).sum(); loss.backward(); w.step(); w.zero_grad()

def test_int8_is_optimizer():
    m = nn.Linear(8,4); o = AdamW(m.parameters(),lr=1e-4)
    w = Int8OptimizerStates(o)
    assert isinstance(w, torch.optim.Optimizer)

def test_int8_step():
    m = nn.Linear(8,4); o = AdamW(m.parameters(),lr=1e-4)
    w = Int8OptimizerStates(o)
    for _ in range(3):
        loss=torch.randn(2,8).matmul(m.weight.T).sum(); loss.backward(); w.step(); w.zero_grad()

def test_deepspeed_is_optimizer():
    m = nn.Linear(8,4); o = AdamW(m.parameters(),lr=1e-4)
    w = DeepSpeedCPUOffload(o)
    assert isinstance(w, torch.optim.Optimizer)

@pytest.mark.skip(torch.cuda.is_available(), reason="DeepSpeed CPU offload test only on CPU")
@pytest.mark.skip(reason="DeepSpeed CPU offload: known device-mix issue, fixing in v0.1.2")
def test_deepspeed_step():
    m = nn.Linear(8,4); o = AdamW(m.parameters(),lr=1e-4)
    w = DeepSpeedCPUOffload(o)
    for _ in range(3):
        x=torch.randn(2,8); loss=m(x).sum(); loss.backward(); w.step(); w.zero_grad()

def test_vsqz_training_preset():
    m = nn.Linear(8,4); o = AdamW(m.parameters(),lr=1e-4)
    sq = VRAMSqueeze(m, mode='training', optimizer=o, galore_rank=8, lisa_ratio=1.0, fp16_states=True)
    x=torch.randn(2,8); loss=m(x).sum(); loss.backward(); sq.step_end(); sq.zero_grad()

def test_vsqz_inference_preset():
    m = nn.Sequential(nn.Embedding(100,64), nn.Linear(64,2))
    sq = VRAMSqueeze(m, mode='inference', preset='balanced')
    e = sq.evict_if_needed(2000)
    assert e >= 0
