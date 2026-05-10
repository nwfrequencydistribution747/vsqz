"""FP16/BF16 Optimizer States — wraps any optimizer."""
from __future__ import annotations
import logging, torch
from torch.optim import Optimizer

logger = logging.getLogger("FP16State")

class FP16OptimizerStates(Optimizer):
    def __init__(self, base_optimizer, dtype=torch.bfloat16):
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"dtype must be float16 or bfloat16, got {dtype}")
        # Bypass Optimizer.__init__ to avoid property conflicts — inherit purely for isinstance checks
        self._base = base_optimizer
        self._dtype = dtype
        self._step_counter = 0
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state
        logger.info("FP16OptimizerStates: dtype=%s", dtype)

    def step(self, closure=None):
        self._step_counter += 1
        self._decompress_states()
        loss = self._base.step(closure)
        self._compress_states()
        torch.cuda.empty_cache()
        return loss

    def _decompress_states(self):
        for group in self._base.param_groups:
            for p in group["params"]:
                state = self._base.state.get(p)
                if state is None: continue
                for k in ("exp_avg","exp_avg_sq","moment1","moment2"):
                    t = state.get(k)
                    if t is not None and t.dtype == self._dtype:
                        state[k] = t.to(dtype=torch.float32)

    def _compress_states(self):
        for group in self._base.param_groups:
            for p in group["params"]:
                state = self._base.state.get(p)
                if state is None: continue
                for k in ("exp_avg","exp_avg_sq","moment1","moment2"):
                    t = state.get(k)
                    if t is not None and t.dtype != self._dtype:
                        state[k] = t.to(dtype=self._dtype)

    def zero_grad(self, set_to_none=True):
        self._base.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"base": self._base.state_dict(), "step": self._step_counter, "dtype": str(self._dtype)}

    def load_state_dict(self, d):
        self._base.load_state_dict(d["base"])
        self._step_counter = d.get("step",0)
