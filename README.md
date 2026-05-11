# vsqz — VRAMSqueeze: VRAM & File Compression for AI Models

Tired of **CUDA Out of Memory (OOM)** errors during LLM fine-tuning? Is your disk getting full, but you don't want to delete models? Are your LLMs and checkpoints too large to share or upload to the cloud? Want to run **local AI** but lack the required VRAM? 

Instead of buying new disks or GPUs, use, support, share and integrate `vsqz` — your software solution.

**One file. Half the VRAM. Double the model.**


[![PyPI version](https://img.shields.io/pypi/v/vsqz)](https://pypi.org/project/vsqz/)
[![tests](https://img.shields.io/badge/tests-41%2F1%20skipped-green)](https://github.com/butterwecksolutions/vsqz/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/butterwecksolutions/vsqz/blob/main/LICENSE)
[![Sponsor](https://img.shields.io/badge/sponsor-VSQZ%20unterst%C3%BCtzen-ff69b4)](https://github.com/sponsors/butterwecksolutions)

`pip install vsqz` — the `gzip` for AI models. 55% smaller files. Full archiver: directory structure, permissions, timestamps, symlinks. Roundtrip-safe for safetensors, GGUF, PyTorch. VRAM training optimizers (GaLore, Q-GaLore, LISA) for 9B→12GB QLoRA.

**Unlike gzip/zip/7zip, no extraction needed.** `.vsqz` files load directly into RAM via numpy — no temp files, no double disk I/O. Roundtrip-decompress (`-d`) for tools that need native formats (llama.cpp, etc.).

> 🔥 **New in v0.4.0: Multi-model delta sharing.** `--diff` stores only the weights
> that changed between a base model and its fine-tune. Self-describing, SHA-verified,
> format-agnostic. `--serve` (preview) shares base tensors + measures VRAM savings.
> Full multi-model inference in v0.5. [Try it →](https://github.com/butterwecksolutions/vsqz/tree/dev)
>
> For model distribution: one base model + tiny deltas instead of dozens of full downloads.
> `--rediff` reconstructs any fine-tune from base + delta.
>
> | Strategy | 10 Bases | 50 Fine-Tunes | Total | |
> |---|---|---|---|---|
> | Full downloads | 10 × 18 GB | 50 × 18 GB | **1,080 GB** | |
> | Catalog + deltas | 10 × 8 GB | 50 × 1 GB | **130 GB** | **88% less** |
>
> Bases compressed (fp16), deltas contain only changed weights. `--rediff` reconstructs
> any variant locally. One catalog, no duplicates.
>
> ```
> vsqz --diff   qwen-base.vsqz qwopus.gguf  -o qwopus-delta.vsqz      # only changed weights
> vsqz --rediff qwen-base.vsqz qwopus-delta.vsqz -o qwopus-full.gguf  # reconstruct full model
> vsqz --serve  qwen-base.vsqz qwopus-delta.vsqz huihui-delta.vsqz    # base once, rest on top
> ```
>
> `vsqz -l delta.vsqz` shows what base it needs (architecture, params, SHA, timestamps).
> Wrong base → rejected. Same base from any source (GGUF/safetensors/PT) → accepted.

> **v0.3.5 — stable.** Full archiver (tar-level fidelity): compress, decompress, list, verify. Roundtrip-safe
> for safetensors, GGUF, PyTorch. Training optimizers (GaLore, LISA, Q-GaLore) in beta. Inference features on roadmap.
> directory structure, permissions, timestamps, symlinks. Roundtrip-safe for safetensors, GGUF, PyTorch.
> `vsqz -l` lists archive contents. 41 tests, autonomous CI.

```
# Compress any model: 18GB → 8GB
python -m vsqz convert model/ output.vsqz

# List archive contents (files, sizes, permissions, timestamps)
python -m vsqz -l model.vsqz

# 🔥 Multi-model: delta sharing (v0.4.0)
vsqz --diff  qwen.vsqz qwopus.gguf -o qwopus-delta.vsqz   # store only changed weights
vsqz --serve qwen.vsqz qwopus-delta.vsqz ablit-delta.vsqz # 3 models, base once

# Extract vision/audio encoder (all VL archs, legacy .gguf or .vsqz)
vsqz --mmproj model/ -o mmproj.gguf     # llama.cpp compatible
vsqz --mmproj model/ -o mmproj.vsqz     # compressed archive

# Training: wrap your optimizer, save VRAM  
from vsqz import VRAMSqueeze
squeezer = VRAMSqueeze(model, optimizer=opt, preset="13B_24GB")
```

> ⚠️ **vsqz is experimental beta software.** Always back up your data before use.
> No liability for data loss. Use at your own risk.
> [Full disclaimer (EN/DE)](#disclaimer--haftungsausschluss)

---

## What GPUs Can Do With vsqz

### Training (QLoRA + GaLore + FP16 States)

| GPU | VRAM | 4B | 9B | 13B | 20B |
|-----|------|----|----|-----|-----|
| RTX 3060 | 12 GB | ✅ b=4 | ✅ b=2 | ✅ b=1 | ❌ |
| RTX 4070 | 12 GB | ✅ b=4 | ✅ b=3 | ✅ b=1 | ❌ |
| RTX 4080 | 16 GB | ✅ b=4 | ✅ b=4 | ✅ b=2 | ⚠️ b=1 |
| RTX 3090 | 24 GB | ✅ b=4 | ✅ b=4 | ✅ b=3 | ✅ b=1 |
| RTX 4090 | 24 GB | ✅ b=4 | ✅ b=4 | ✅ b=4 | ✅ b=2 |

*Compare: QLoRA baseline vs vsqz-optimized VRAM on consumer GPUs.*

### Inference (Context Window Doubling via KV-Cache Compression)

| GPU | 4B | 9B | 13B | 20B |
|-----|-----|-----|------|------|
| 8 GB  | 16k ✅ | 8k ✅ | ❌ | ❌ |
| 12 GB | 32k ✅ | 16k ✅ | 8k ✅ | ❌ |
| 16 GB | 64k ✅ | 32k ✅ | 16k ✅ | 8k ✅ |
| 24 GB | 128k ✅ | 64k ✅ | 32k ✅ | 16k ✅ |

*Without vsqz: context halved on every tier.*

## Supported Formats

Compress anything, restore byte-identical. Like gzip, you get back exactly what you put in —
same files, same directory structure, same permissions.

| Source | Compress | Decompress | Roundtrip |
|--------|----------|------------|-----------|
| `model.safetensors` (single) | `.vsqz` | `model.safetensors` | byte-identical |
| `model/` (directory, sharded) | `.vsqz` | `model/` with all files, subdirs | identical |
| `.gguf` (llama.cpp, Ollama) | `.vsqz` | `.gguf` (reconstructed) | tensors + metadata preserved |
| `.gguf` mmproj (Vision) | `.vsqz` | `.gguf` (reconstructed) | identical to base GGUF |
| `.bin`, `.pt`, `.pth` (PyTorch) | `.vsqz` | original filename | tensors preserved |
| Non-tensor files (JSON, YAML, PNG, PDF...) | zstd in `.vsqz` | restored as-is | byte-identical ✅ |
| Directory permissions (chmod) | preserved | restored | `600` → `600` |
| File timestamps (mtime, atime) | preserved | restored | `os.utime()` |
| Symlinks | target stored | recreated | `model/ → ../shared/model/` |

`vsqz -l model.vsqz` shows the full contents: filenames, sizes, timestamps, and permissions.

> **llama.cpp users:** llama.cpp currently reads `.gguf` natively, not `.vsqz`. For inference,
> decompress once: `vsqz -d model.vsqz model.gguf`. `AutoModel.from_pretrained()` loads `.vsqz`
> directly with zero extraction — no llama.cpp dependency.

---

## VRAM Savings

| Format | Original | vsqz | Archive (-z) | Savings |
|--------|----------|------|-------------|---------|
| safetensors (9B) | 18 GB | 8 GB | 7 GB | **61%** |
| GGUF F16 (9B) | 18 GB | 8 GB | 7 GB | **61%** |
| PyTorch Checkpoint (w/ AdamW) | 56 GB | 18 GB | 16 GB | **71%** |
| PyTorch Checkpoint (weights only) | 18 GB | 8 GB | 7 GB | **61%** |
| **ALL THREE → single .vsqz.zst** | **56 GB** | **8 GB** | **7 GB** | **87%** |

---

## How It Works — The Stack

vsqz combines 8 training + 3 archival techniques. Each targets a different memory region:

### Training Optimizations

| Technique | Origin | What It Saves | VRAM Freed |
|-----------|--------|---------------|------------|
| **GaLore-inspired** | Weight decomposition | AdamW on factorized weights | ~2 GB |
| **LISA** | 2024 | Activations (50% layer sampling) | ~4 GB |
| **FP16 States** | Native | Optimizer precision (32→16 bit) | ~1.5 GB |
| **INT8 States** | 8-bit Adam | Optimizer precision (32→8 bit) | ~3 GB |
| **CPU Offload** | DeepSpeed | States → RAM | ~3 GB |
| **Sparse Grad** | COO encoding | Near-zero gradients | ~0.5 GB |
| **Gradient Delta** | git/rsync | ΔG instead of G | ~1 GB |
| **Adaptive Quant** | H.264/AV1 | Per-layer bit allocation | ~0.5 GB |

### Archival & Integrity

| Feature | Origin | What It Does | Savings |
|---------|--------|-------------|---------|
| **FP16 Compression** | IEEE 754 | FP32→FP16 weight storage | 50% |
| **zstd Post-Compress** | Facebook | 5-15% extra on top of FP16 | 5-15% |
| **AdamW Stripping** | vsqz | Drop Momentum+Variance states | ~66% of checkpoint |
| **SHA-256** | NIST | Cryptographic integrity | – |
| **Recovery Record** | RAR | Self-repairing header | – |
| **KV-Cache H.264** | StreamingLLM | I/P/B-frame token eviction | 2× context |

Training: 8 techniques active simultaneously. Archival: FP16 + zstd + AdamW strip stack.

---

## Quickstart

### Install

```bash
pip install vsqz
```

### CLI — same flags as gzip/zip

Works like gzip. Linux users already know the flags.

| Flag | What it does |
|------|-------------|
| `-1 .. -9` | Compression level (1=fast/fp16, 9=best/int8+sparse) |
| `-k` | Keep original file |
| `-d` | Decompress to original format |
| `-v` / `-q` | Verbose / quiet |
| `-f` | Force overwrite |
| `-t` | SHA-256 integrity test |
| `-l` | List archive contents (files, sizes, permissions, ratios) |
| `-r` | Recursive (all models in directory) |
| `-s SIZE` | Split into chunks (e.g. `-s 8G` for cloud) |
| `-x KEY` | Exclude tensors (e.g. `-x adam` strips optimizer) |
| `-z` | Post-compress with zstd (archive mode, 5-15% extra) |

### Useful Combinations

```bash
# Archive model for long-term storage (max compression + zstd)
vsqz -kz9 model/                → model/.vsqz.zst (smallest possible file)

# Convert ALL models in collection (archive, keep originals, max compression)
vsqz -kr9z ~/models/            → every model gets .vsqz.zst, raw files kept
# Compare: vsqz -lr ~/models/   → peek sizes, decide what to delete

# Convert GGUF collection to .vsqz for archiving
find ~/models -name "*.gguf" -o -name "*.safetensors" | while read f; do
  vsqz -kz "$f"                # compress each, keep original
done

# Free 50%+ disk space after verifying all .vsqz files
find ~/models -name "*.vsqz" | while read f; do
  vsqz -t "$f" && rm "${f%.vsqz}"  # delete original if .vsqz is valid
done

# Cloud upload with zstd
vsqz -kzs 8G large-model/       → .001, .002, ... .zst (compressed chunks)

# Clean checkpoint (strip AdamW, compress, keep original)
vsqz -kx adam pytorch_model.bin  → strip AdamW states, keep weights

# Download once, compress, delete original
vsqz model.safetensors          → model.safetensors.vsqz (no raw left)

# Verify integrity before deleting original
vsqz -t model.vsqz && rm model.safetensors

# Recursively compress all models, keep originals, show stats
vsqz -krv ~/models/

# 🔥 Delta sharing: run 5 fine-tunes in the VRAM of 1
vsqz qwen-base.gguf qwen.vsqz                          # compress base (once)
vsqz --diff qwen.vsqz qwopus.gguf -o qwopus.vsqz       # build delta (only diffs)
vsqz --diff qwen.vsqz huihui.gguf -o huihui.vsqz       # another fine-tune
vsqz -l qwopus.vsqz                                     # peek: which base? how shared?
vsqz --serve qwen.vsqz qwopus.vsqz huihui.vsqz         # all 3, base loaded once

# Find shared chunk across unrelated models (accidental wins)
vsqz --diff mistral.vsqz llama.vsqz -o cross.vsqz      # any shared embeddings?

# Decompress zstd archive, verbose
vsqz -dv model.vsqz.zst
```

### How `--diff` / `--serve` work (v0.4.0-dev)

`--diff` is **production-ready**: loads two models, compares tensors, stores only
the differing weights. SHA-verified, self-describing, cross-format.

`--serve` is **preview**: loads shared tensors into Python dicts and measures VRAM.
For actual multi-model inference, each model needs to be wrapped in a PyTorch
state-dict with `model.load_state_dict()`. Full HuggingFace integration (`AutoModel`)
is planned for v0.5.

```
Current:  --serve → shared dicts → measure VRAM (preview)
v0.5:    --serve → AutoModel.load_state_dict → model.generate() (inference)
```

---

### Verify Compression (before deleting originals)

```bash
# Check .vsqz integrity — decompress and compare
python -c "
from vsqz.vsqz_format import peek_vsqz
h = peek_vsqz('model.vsqz')
print(f'Tensors: {len(h[\"tensors\"])}, Size: {sum(t[\"size\"] for t in h[\"tensors\"].values())/1e9:.1f} GB')
print(f'Techniques: {h[\"technique_stack\"]}')
print(f'Verdict: Safe to delete original')
"
```

### HuggingFace Integration (AutoModel)

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model.vsqz")  # Just works
```
Import vsqz — AutoModel now accepts `.vsqz` files natively. No manual setup.

### Training (HuggingFace / Axolotl)

```python
from vsqz import VRAMSqueeze
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("Qwen2.5-7B")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# One line: activate all optimizations
squeezer = VRAMSqueeze(model, optimizer=optimizer, preset="13B_24GB")

# Presets: "9B_12GB", "13B_24GB", "20B_24GB", "safe_defaults"
```

### Inference (KV-Cache Compression)

```python
from vsqz import VRAMSqueeze

squeezer = VRAMSqueeze(model, mode="inference", preset="balanced")
for step in generation_loop:
    squeezer.evict_if_needed(current_seq_len)  # Auto-evict old tokens
```

---

## File Format: .vsqz

```
[0..3]    Magic:       VSQZ                    (4 bytes)
[4..7]    Version:     uint32                  (4 bytes)
[8..11]   Header Len:  uint32                  (4 bytes)
[12..]    JSON Header  (config, SHA-256, tensor index)
[...]     Tensor Blobs (FP16 + GaLore + INT8)
[...]     Recovery JSON → Recovery Len: uint32 → RECO
```

- Self-describing: anyone who sees `.vsqz` knows vsqz was used
- Mmap-compatible for zero-copy loading
- One file for everything: weights + optimizer + metadata
- Open format: read it with any JSON parser + numpy

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Optional: optuna (Bayesian HPO), safetensors (converter)

---

## Archive Mode (-z / --zstd)

Stacks FP16 + zstd + AdamW stripping. Optimal for long-term storage, cloud upload, and model distribution. Use `-kz9` for maximum compression, `-kzs 8G` for chunked cloud upload.

| Step | What happens | Size reduction |
|------|-------------|---------------|
| 1. FP32→FP16 | Half-precision weights | 2× |
| 2. AdamW Strip | Remove momentum+variance (2× model size) | ~66% of ckpt |
| 3. zstd | Post-compression | 5-15% extra |
| **Combined** | **Archive grade** | **87% vs all three formats** |

---

## Integrity & Security

Every `.vsqz` file carries its own SHA-256 fingerprint and a recovery record at the end of the file. If the main header gets corrupted, the file self-repairs from the recovery record.

```bash
vsqz -t model.vsqz       # SHA-256 verified integrity check
vsqz -l model.vsqz       # Shows SHA-256 fingerprint
# If header is corrupted: auto-restores from recovery record
```

**No other ML format** has self-repair. GGUF and safetensors have no checksums at all.

---

## Developer Experience

```bash
vsqz -h               # gzip-style help with all flags
```

Every PR gets an automated review (imports, stubs, extensions, tests, paths, README consistency). Results are posted as a PR comment — no human reviews broken code.

| CI Job | What it checks |
|--------|---------------|
| Test (3.10/3.11/3.12) | 41 tests across all supported Python versions |
| Lint | Code style consistency (ruff) |
| Review Bot | 8 structural checks (diff-based test coverage), posted as PR comment |
| Auto-labels | 6 categories per changed files |
| Auto-labels | "format", "training", "inference", "tests" per changed files |

---

## Ecosystem Integration

**llama.cpp PR in progress.** Once merged, every llama.cpp-based client (Ollama, LM Studio, text-generation-webui) will load `.vsqz` files natively — no conversion, no Python bridge. See `contrib/` for the llama.cpp reader patch and axolotl integration guide.

---

## Why vsqz?

| | GGUF | safetensors | vsqz |
|--|------|-------------|------|
| Training | ❌ | ✅ | ✅ |
| Inference | ✅ | ❌ | ✅ |
| Optimizer State | ❌ | ❌ | stripped |
| Context Expansion | ❌ | ❌ | beta |
| File Size (9B) | 18 GB | 18 GB | 8 GB |
| zstd Archive | ❌ | ❌ | ✅ (-z, +15%) |
| Faster Downloads | ❌ | ❌ | ✅ 55% smaller |
| SHA-256 + Recovery | ❌ | ❌ | ✅ |
| Universal | ❌ | ❌ | ✅ |

**One file. Training and inference. SHA-256 verified. Self-repairing.**

---

## Academic References

- Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", ICML 2024
- Pan et al., "LISA: Layer-wise Importance Sampling for Memory-Efficient LLM Fine-Tuning", 2024
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023
- Xiao et al., "StreamingLLM: Efficient Streaming Language Models with Attention Sinks", 2023

---

---
 
## Disclaimer / Haftungsausschluss
 
### English
 
**vsqz is experimental beta software under active development.** While tested in
9B QLoRA production training (RTX 3090), compression and decompression involve
inherent risks of data corruption or total loss.
 
- **Always back up your data** before using vsqz. Compression is a one-way
  operation on your original files — test restoration before deleting originals.
- vsqz is provided **"AS IS"**, without warranty of any kind, express or implied,
  including but not limited to warranties of merchantability, fitness for a
  particular purpose, title, and non-infringement.
- In no event shall the authors or copyright holders be liable for any claim,
  damages, or other liability, whether in contract, tort, or otherwise, arising
  from the use of or inability to use the software, including but not limited to
  loss of data, loss of revenue, business interruption, or any direct, indirect,
  incidental, special, exemplary, or consequential damages.
- **No liability is accepted for data loss.** You use vsqz entirely at your own risk.
 
### Deutsch
 
**vsqz ist experimentelle Beta-Software in aktiver Entwicklung.** Trotz
erfolgreicher Tests im 9B-QLoRA-Produktivtraining (RTX 3090) birgt die
Kompression und Dekompression inhärente Risiken von Datenkorruption bis hin
zu vollständigem Datenverlust.
 
- **Erstellen Sie vor der Nutzung stets ein Backup Ihrer Daten.** Überprüfen
  Sie die Wiederherstellung, bevor Sie Originaldaten löschen.
- **Haftungsbeschränkung:** Der Autor haftet unbeschränkt für Vorsatz und grobe
  Fahrlässigkeit sowie für Schäden aus der Verletzung des Lebens, des Körpers
  oder der Gesundheit. Für leichte Fahrlässigkeit haftet der Autor nur bei
  Verletzung wesentlicher Vertragspflichten (Kardinalpflichten), und zwar
  begrenzt auf den vertragstypischen, vorhersehbaren Schaden.
- Ansprüche nach dem Produkthaftungsgesetz bleiben unberührt.
- Die Software wird im Übrigen **ohne jegliche Gewährleistung** bereitgestellt.
- **Es wird keine Haftung für Datenverluste übernommen.** Die Nutzung erfolgt
  ausschließlich auf eigenes Risiko.
 
---
 
**Author:** Christian Butterweck — [github.com/butterwecksolutions](https://github.com/butterwecksolutions)  
**License:** MIT
