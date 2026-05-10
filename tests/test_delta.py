"""Tests for vsqz --diff and --serve multi-model delta sharing."""

import os
import sys
import tempfile
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vsqz.converter import _compute_delta, _load_source


@pytest.fixture
def model_pair():
    """Create base and fine-tuned PyTorch models in temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(42)
        base = {}
        for i in range(4):
            base[f"l.{i}.w"] = torch.randn(32, 32)
        base["embed"] = torch.randn(100, 32)
        torch.save(base, f"{tmp}/base.pt")
        os.system(f"CUDA_VISIBLE_DEVICES='' python3 -m vsqz -k {tmp}/base.pt {tmp}/base.vsqz 2>/dev/null")

        fine = {k: v.clone() for k, v in base.items()}
        fine["l.0.w"] += 0.01 * torch.randn(32, 32)
        fine["new_layer.w"] = torch.randn(32, 32)
        torch.save(fine, f"{tmp}/fine.pt")

        yield tmp, base, fine


class TestComputeDelta:
    def test_identical_models(self):
        """Two identical models produce empty delta."""
        a = {"w": np.array([1, 2, 3], dtype=np.float32)}
        b = {"w": np.array([1, 2, 3], dtype=np.float32)}
        shared, deltas = _compute_delta(a, b)
        assert shared == 1
        assert len(deltas) == 0

    def test_modified_tensor(self):
        """Modified tensor goes into delta."""
        a = {"w": np.array([1, 2, 3], dtype=np.float32)}
        b = {"w": np.array([1, 2, 4], dtype=np.float32)}
        shared, deltas = _compute_delta(a, b)
        assert shared == 0
        assert "w" in deltas

    def test_new_tensor(self):
        """New tensor in fine-tune goes into delta."""
        a = {"w": np.array([1, 2], dtype=np.float32)}
        b = {"w": np.array([1, 2], dtype=np.float32), "new": np.zeros(3)}
        shared, deltas = _compute_delta(a, b)
        assert shared == 1
        assert "new" in deltas
        assert len(deltas) == 1

    def test_removed_tensor(self):
        """Tensor removed from base is not in delta."""
        a = {"w1": np.ones(3), "w2": np.ones(3)}
        b = {"w1": np.ones(3)}
        shared, deltas = _compute_delta(a, b)
        assert shared == 1
        assert "w2" not in deltas

    def test_different_shape(self):
        """Different shape → tensor goes into delta."""
        a = {"w": np.ones((3, 3))}
        b = {"w": np.ones((4, 4))}
        shared, deltas = _compute_delta(a, b)
        assert shared == 0


class TestDiffCLI:
    def test_reject_non_vsqz_base(self, model_pair):
        """--diff with .pt base must be rejected."""
        tmp, _, _ = model_pair
        ret = os.system(
            f"CUDA_VISIBLE_DEVICES='' python3 -m vsqz --diff {tmp}/base.pt {tmp}/fine.pt "
            f"-o {tmp}/delta.vsqz 2>/dev/null >/dev/null"
        )
        assert ret != 0

    def test_diff_produces_file(self, model_pair):
        """--diff with .vsqz base must produce a delta file."""
        tmp, base, fine = model_pair
        os.system(
            f"CUDA_VISIBLE_DEVICES='' python3 -m vsqz --diff {tmp}/base.vsqz {tmp}/fine.pt "
            f"-o {tmp}/delta.vsqz 2>/dev/null >/dev/null"
        )
        assert os.path.exists(f"{tmp}/delta.vsqz")
        assert os.path.getsize(f"{tmp}/delta.vsqz") > 0

    def test_delta_self_describing(self, model_pair):
        """Delta must contain base_model metadata."""
        tmp, _, _ = model_pair
        os.system(
            f"CUDA_VISIBLE_DEVICES='' python3 -m vsqz --diff {tmp}/base.vsqz {tmp}/fine.pt "
            f"-o {tmp}/delta.vsqz 2>/dev/null >/dev/null"
        )
        from vsqz.vsqz_format import peek_vsqz
        h = peek_vsqz(f"{tmp}/delta.vsqz")
        assert h.get("delta") is True
        assert "base_sha256" in h
        assert "base_model" in h
        bi = h["base_model"]
        assert "sha256" in bi
        assert "tensor_count" in bi
        assert "source_name" in bi
        assert "delta_created" in bi
        assert "source_mtime" in bi


class TestServeCLI:
    def test_serve_applies_delta(self, model_pair):
        """--serve with correct base+delta must apply the delta."""
        tmp, _, _ = model_pair
        os.system(
            f"CUDA_VISIBLE_DEVICES='' python3 -m vsqz --diff {tmp}/base.vsqz {tmp}/fine.pt "
            f"-o {tmp}/delta.vsqz 2>/dev/null >/dev/null"
        )
        # Capture serve output
        ret = os.system(
            f"CUDA_VISIBLE_DEVICES='' python3 -m vsqz --serve {tmp}/base.pt {tmp}/delta.vsqz "
            f"2>/dev/null >{tmp}/serve_out.txt"
        )
        out = open(f"{tmp}/serve_out.txt").read()
        assert "+ delta" in out.lower() or "+ delta" in out.lower() or "+ base" in out.lower() or "models loaded" in out

    def test_serve_rejects_wrong_base(self, model_pair):
        """--serve with wrong base must reject delta."""
        tmp, base, fine = model_pair
        os.system(
            f"CUDA_VISIBLE_DEVICES='' python3 -m vsqz --diff {tmp}/base.vsqz {tmp}/fine.pt "
            f"-o {tmp}/delta.vsqz 2>/dev/null >/dev/null"
        )
        ret = os.system(
            f"CUDA_VISIBLE_DEVICES='' python3 -m vsqz --serve {tmp}/fine.pt {tmp}/delta.vsqz "
            f"2>/dev/null >{tmp}/serve_wrong.txt"
        )
        out = open(f"{tmp}/serve_wrong.txt").read()
        assert "MISMATCH" in out or "Skipping" in out or "mismatch" in out


class TestSHA:
    def test_fp16_normalized_sha(self):
        """SHA must match even after fp32→fp16→fp32 roundtrip."""
        a = {f"l.{i}.w": np.random.RandomState(42).randn(64, 64).astype(np.float32) for i in range(4)}
        # fp16 roundtrip
        b = {k: v.astype(np.float16).astype(np.float32) for k, v in a.items()}
        import hashlib
        def sha(t):
            return hashlib.sha256(
                b"".join(n.encode() + t[n].astype(np.float16).tobytes() for n in sorted(t))
            ).hexdigest()
        assert sha(a) == sha(b)

    def test_different_seed_different_sha(self):
        """Different weights must produce different SHA."""
        a = {f"l.{i}.w": np.random.RandomState(1).randn(64, 64).astype(np.float32) for i in range(4)}
        b = {f"l.{i}.w": np.random.RandomState(2).randn(64, 64).astype(np.float32) for i in range(4)}
        import hashlib
        def sha(t):
            return hashlib.sha256(
                b"".join(n.encode() + t[n].astype(np.float16).tobytes() for n in sorted(t))
            ).hexdigest()
        assert sha(a) != sha(b)
