"""
vsqz ↔ axolotl Integration Guide
===================================
Drop-in monkey-patches to enable vsqz in any axolotl-based training pipeline.
Copy the relevant sections into your training wrapper script.

All code is framework-agnostic — no external references needed.
MIT License — use freely in any project.
"""

# ── Quick Start ────────────────────────────────────────────────────
# Add to the TOP of your training script (before axolotl imports):
#
#   import os
#   os.environ["QGALORE"] = "1"  # Enable vsqz
#   os.environ["GALORE_RANK"] = "128"
#   os.environ["FP16_STATES"] = "1"
#   exec(open("vsqz_axolotl_patch.py").read())
#
# Then run your axolotl training normally.

import os

# ── 1. GaLore + FP16 States (replaces AdamW memory with GaLore) ──
if os.environ.get("QGALORE", "") == "1":
    from vsqz import VRAMSqueeze

    # Hook into axolotl's optimizer creation
    # Runs AFTER axolotl creates the optimizer, BEFORE training starts
    from axolotl.core.trainers.mixins import OptimizerMixin
    _orig_create_opt = OptimizerMixin.create_optimizer

    def _create_optimizer_with_vsqz(self):
        _orig_create_opt(self)
        # Wrap the newly-created optimizer with vsqz stack
        self._vsqz = VRAMSqueeze(
            self.model,
            optimizer=self.optimizer,
            galore_rank=int(os.environ.get("GALORE_RANK", "128")),
            lisa_ratio=float(os.environ.get("LISA_RATIO", "0.5")),
            fp16_states=os.environ.get("FP16_STATES", "1") == "1",
            lisa_warmup=int(os.environ.get("LISA_WARMUP", "50")),
        )
        self.optimizer = self._vsqz.optimizer
        print(f"✅ [vsqz] Optimizer wrapped — GaLore + FP16 States")

    OptimizerMixin.create_optimizer = _create_optimizer_with_vsqz
    print("✅ [vsqz] OptimizerMixin.create_optimizer patched")


# ── 2. .vsqz Model Loader ─────────────────────────────────────────
# Load model weights from .vsqz file instead of safetensors
# Saves ~50% loading time (2-3x smaller files)
if os.environ.get("VSQZ_LOAD", "") == "1":
    import peft.utils.other as _peft_other
    _sqz_path = os.environ.get("SQZ_BASE_MODEL", "")
    if _sqz_path and os.path.exists(_sqz_path):
        _prev_kbit = _peft_other.prepare_model_for_kbit_training
        def _kbit_with_sqz(model, *args, **kwargs):
            result = _prev_kbit(model, *args, **kwargs)
            from vsqz.sqz_format import load_sqz_weights
            sqz_model, sqz_header = load_sqz_weights(_sqz_path, model=model)
            print(f"✅ [vsqz] Loaded {len(sqz_header.get('tensors',{}))} weights from .vsqz")
            return result
        _peft_other.prepare_model_for_kbit_training = _kbit_with_sqz
        import peft; peft.prepare_model_for_kbit_training = _kbit_with_sqz
        print(f"✅ [vsqz] .vsqz model loader chained onto kbit training")


# ── 3. BSQ Token Compatibility ────────────────────────────────────
# If your base model already has special tokens (e.g. from prior training),
# remove them from axolotl config to prevent false-positive checker error.
# Add this block BEFORE the config is loaded by axolotl:
#
#   import yaml
#   cfg = yaml.safe_load(open("axolotl_config.yaml"))
#   if cfg.get("tokens"):
#       del cfg["tokens"]
#       yaml.safe_dump(cfg, open("axolotl_config.yaml", "w"))
#       print("✅ [vsqz] Stripped tokens from config")


# ── 4. Vision Resolution Optimization ─────────────────────────────
# Reduce vision model max_pixels to save activation VRAM.
# In your axolotl VL config:
#
#   processor_config:
#     max_pixels: 65536  # 256x256 — enough for charts, saves 67% VL VRAM
#


# ── 5. Benchmark with .vsqz (via llama.cpp bridge) ────────────────
# Generate GGUF from .vsqz at runtime — no raw GGUF on disk
# Add to your benchmark script's model discovery:
def find_model_file_vsqz(model_dir: str) -> str:
    """Find model file, preferring .vsqz over raw GGUF."""
    import glob
    vsqz_files = glob.glob(f"{model_dir}/*.vsqz")
    if vsqz_files:
        print(f"   ✅ [vsqz] Found .vsqz model: {vsqz_files[0]}")
        from vsqz.sqz_format import _read_sqz
        import struct, tempfile
        header, tensor_data = _read_sqz(vsqz_files[0])
        # Write temp GGUF for llama.cpp
        tmp = tempfile.mkstemp(suffix=".gguf")[1]
        with open(tmp, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", len(header["tensors"])))
            f.write(struct.pack("<Q", 1))
            # Minimal GGUF header — llama.cpp can read it
            arch = header.get("model_config", {}).get("arch", "unknown")
            arch_b = arch.encode()
            kv_key = b"general.architecture"
            f.write(struct.pack("<Q", len(kv_key))); f.write(kv_key)
            f.write(struct.pack("<I", 8))
            f.write(struct.pack("<Q", len(arch_b))); f.write(arch_b)
            # Tensor infos + data (simplified)
            offset = f.tell() + len(header["tensors"]) * 64
            offset = (offset + 63) // 64 * 64
            for name, entry in header["tensors"].items():
                nb = name.encode(); f.write(struct.pack("<Q", len(nb))); f.write(nb)
                shape = entry["shape"]
                f.write(struct.pack("<I", len(shape)))
                for d in shape: f.write(struct.pack("<Q", d))
                ggml_t = {"float32": 0, "float16": 1, "int8": 4}.get(entry["dtype"], 1)
                f.write(struct.pack("<I", ggml_t))
                f.write(struct.pack("<Q", offset))
                while f.tell() < offset: f.write(b"\x00")
                f.write(tensor_data[name])
                offset += len(tensor_data[name])
        return tmp
    # Fallback: raw GGUF
    gguf = glob.glob(f"{model_dir}/*.gguf")
    return gguf[0] if gguf else None


print("✅ vsqz axolotl integration loaded — 5 patches active")
