"""DeepSpeed CPU Offload — wraps any optimizer."""
from __future__ import annotations
import logging, torch
from torch.optim import Optimizer

logger = logging.getLogger("DeepSpeedOffload")

class DeepSpeedCPUOffload(Optimizer):
    def __init__(self, base_optimizer, pin_memory=True):
        # Bypass Optimizer.__init__ to avoid property conflicts
        self._base = base_optimizer; self._pin = pin_memory; self._step_counter = 0
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state
        self._offload_all_states()

    def _offload_all_states(self):
        for group in self._base.param_groups:
            for p in group["params"]:
                state = self._base.state.get(p)
                if state is None: continue
                for k in ("exp_avg","exp_avg_sq","moment1","moment2"):
                    t = state.get(k)
                    if t is not None and t.device.type!="cpu":
                        state[k] = t.to("cpu").pin_memory() if self._pin else t.to("cpu")

    def step(self, closure=None):
        self._step_counter += 1
        for group in self._base.param_groups:
            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                state = self._base.state.get(p)
                if state is None: continue
                for k in ("exp_avg","exp_avg_sq","moment1","moment2"):
                    t = state.get(k)
                    if t is not None and t.device.type=="cpu":
                        state[k] = t.to("cuda", non_blocking=True)
        loss = self._base.step(closure)
        self._offload_all_states()
        return loss

    def zero_grad(self, set_to_none=True): self._base.zero_grad(set_to_none=set_to_none)
    def state_dict(self): return {"base": self._base.state_dict(), "step": self._step_counter}
    def load_state_dict(self, d): self._base.load_state_dict(d["base"]); self._step_counter = d.get("step",0)

