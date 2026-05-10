"""INT8 Optimizer States — wraps any optimizer."""
from __future__ import annotations
import logging, torch
from torch.optim import Optimizer

logger = logging.getLogger("Int8State")

class Int8OptimizerStates(Optimizer):
    def __init__(self, base_optimizer, num_bits=8):
        self._base = base_optimizer; self._num_bits = num_bits; self._step_counter = 0; self._scales = {}
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state
        logger.info("Int8OptimizerStates: %d-bit", num_bits)

    def step(self, closure=None):
        self._step_counter += 1; self._dequantize_all(); loss = self._base.step(closure); self._quantize_all(); torch.cuda.empty_cache(); return loss

    def _dequantize_all(self):
        for group in self._base.param_groups:
            for p in group["params"]:
                state = self._base.state.get(p)
                if state is None: continue
                for k in ("exp_avg","exp_avg_sq","moment1","moment2"):
                    t = state.get(k)
                    if t is not None and t.dtype==torch.int8:
                        scale = self._scales.get(id(p),{}).get(k)
                        if scale is not None: state[k] = t.float()*scale.to(t.device)

    def _quantize_all(self):
        for group in self._base.param_groups:
            for p in group["params"]:
                state = self._base.state.get(p)
                if state is None: continue
                for k in ("exp_avg","exp_avg_sq","moment1","moment2"):
                    t = state.get(k)
                    if t is not None and t.dtype.is_floating_point:
                        scale = t.abs().max().float()/(2**(self._num_bits-1)-1)
                        if scale==0: scale = torch.tensor(1.0, device=t.device)
                        state[k] = torch.round(t.float()/scale).clamp(-128,127).to(torch.int8)
                        self._scales.setdefault(id(p),{})[k] = scale.to("cpu")

    def zero_grad(self, set_to_none=True): self._base.zero_grad(set_to_none=set_to_none)
    def state_dict(self): return {"base": self._base.state_dict(), "step": self._step_counter}
    def load_state_dict(self, d): self._base.load_state_dict(d["base"]); self._step_counter = d.get("step",0)
