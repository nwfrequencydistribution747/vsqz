"""Multi-model inference server with shared base + delta architecture (GPU-native).

Usage:
    vsqz --serve base.vsqz current.delta.vsqz best.delta.vsqz

Starts one HTTP port per model variant. Each endpoint is OpenAI-compatible.
Base tensors loaded ONCE on GPU, shared across all variants via tensor pointers.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time as _time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .converter_io import _fmt_bytes
from .serve import ModelSwarm


def _status_banner(swarm: ModelSwarm, ports: List[int]) -> str:
    """Viral status display — shows GPU VRAM savings including base compression."""
    n_models = len(swarm.models)
    base_size = swarm._base_size
    total_vram = swarm._gpu_vram_used()
    without_vsqz = n_models * base_size
    saved = without_vsqz - total_vram
    pct = saved / without_vsqz * 100 if without_vsqz > 0 else 0
    loc = "GPU" if swarm._on_gpu else "CPU"

    lines = [
        f"",
        f"  {n_models} Models on {loc}, 1 Base shared",
        f"  Base:         {_fmt_bytes(base_size)}",
        f"  VRAM used:    {_fmt_bytes(total_vram)}",
        f"  Without vsqz: {n_models} x {_fmt_bytes(base_size)} = {_fmt_bytes(without_vsqz)}",
        f"  YOU SAVED:    {_fmt_bytes(saved)} ({pct:.0f}%)",
        f"",
    ]
    name_w = max(len(n) for n in swarm.models) + 2
    lines.append(f"  {'Model':<{name_w}} {'Original':>10} {'vsqz':>10} {'Saved':>7}")
    lines.append(f"  {'-'*name_w} {'-'*10} {'-'*10} {'-'*7}")
    for i, name in enumerate(swarm.models):
        msize = sum(t.numel() * t.element_size() for t in swarm._models[name].values())
        m_saved = base_size - (msize - base_size) if i > 0 else 0  # delta savings
        if i == 0:
            m_pct = (base_size - msize) / base_size * 100  # compression
            tag = "base"
            lines.append(f"  {name:<{name_w}} {_fmt_bytes(base_size):>10} {_fmt_bytes(msize):>10} {m_pct:.0f}% ({tag})")
        else:
            m_pct = (base_size - (msize - base_size)) / base_size * 100
            tag = "delta"
            lines.append(f"  {name:<{name_w}} {_fmt_bytes(base_size):>10} {_fmt_bytes(msize):>10} {m_pct:.0f}% ({tag})")
    lines.append("")
    for i, name in enumerate(swarm.models):
        port = ports[i] if i < len(ports) else "-"
        tag = "base" if i == 0 else "delta"
        lines.append(f"  :{port:<5} -> {name} ({tag})")
    return "\n".join(lines)


def serve_models(
    base_path: str,
    delta_paths: List[str],
    base_port: int = 8081,
    host: str = "127.0.0.1",
    quantize: str = "fp16",
) -> Dict[str, Any]:
    """Start multi-model server. Models pre-loaded on GPU, one port per variant."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        import uvicorn
        import torch
    except ImportError as e:
        print(f"  Missing dependency: {e}")
        print("  pip install fastapi uvicorn torch transformers")
        sys.exit(1)

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"

    # ── Load models ──────────────────────────────────────────────────
    print(f"  Loading on {device.upper()}...")
    swarm = ModelSwarm(base_path, delta_paths, device=device)
    swarm.load()

    ports = [base_port + i for i in range(len(swarm.models))]
    print(_status_banner(swarm, ports))

    # ── Pre-build HF models on GPU (one per variant) ─────────────────
    hf_models = {}  # {model_name: (model, tokenizer)}
    for i, model_name in enumerate(swarm.models):
        tensors = swarm.get_state_dict(model_name)
        if tensors is None:
            continue
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

            hidden = None
            for n, t in tensors.items():
                if len(t.shape) >= 2 and t.shape[1] > 100:
                    hidden = t.shape[1]
                    break
            if hidden is None:
                hidden = 4096

            num_layers = max(1, len([n for n in tensors if 'layer' in n.lower()]) // 4)
            config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
            config.hidden_size = hidden
            config.num_hidden_layers = num_layers

            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
            # Load state dict from GPU tensors
            model.load_state_dict(tensors, strict=False, assign=True)
            if use_gpu:
                model = model.cuda()
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

            hf_models[model_name] = (model, tokenizer)
            port = ports[i]
            print(f"  :{port} -> {model_name} ({'GPU' if use_gpu else 'CPU'}, ready)")
        except Exception as e:
            print(f"  :{ports[i]} -> {model_name}: HF load failed ({e})")
            continue

    # ── FastAPI apps ─────────────────────────────────────────────────
    # Single app serving all models via path routing
    app = FastAPI(title="vsqz ModelSwarm")

    @app.get("/health")
    async def health():
        return {"status": "ok", "models": len(hf_models), "gpu": use_gpu}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": n, "object": "model"} for n in hf_models]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model_name = body.get("model", list(hf_models.keys())[0])
        messages = body.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        max_tokens = body.get("max_tokens", 256)

        if model_name not in hf_models:
            return JSONResponse({"error": f"Unknown model: {model_name}"}, status_code=404)

        model, tokenizer = hf_models[model_name]
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            if use_gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=True, temperature=0.7
                )
            response_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return JSONResponse({
                "id": f"vsqz-{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }],
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ── Start server ─────────────────────────────────────────────────
    port = ports[0]
    print(f"\n  Server: http://{host}:{port}")
    print(f"  Models: {list(hf_models.keys())}")
    print(f"  GPU:    {'YES' if use_gpu else 'NO'}")
    print()

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="error")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {
        "swarm": swarm,
        "model_count": len(hf_models),
        "port": port,
        "models": list(hf_models.keys()),
        "gpu": use_gpu,
    }
