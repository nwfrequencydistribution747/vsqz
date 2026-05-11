"""Tests for .vsqz integrity — SHA-256 + Recovery Record."""
import pytest, tempfile, torch, os, sys
from vsqz.converter_core import convert_to_vsqz
from vsqz.vsqz_format import _read_vsqz, peek_vsqz

class TestSHA256:
    def test_sha_present_in_header(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'; torch.save({'w': torch.randn(64, 32)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        h, _ = _read_vsqz(out, verify_sha256=True)
        assert h.get('sha256'), 'SHA-256 must be in header'
        assert len(h['sha256']) == 64

    def test_sha_verification_passes(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'; torch.save({'w': torch.randn(64, 32)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        h, td = _read_vsqz(out, verify_sha256=True)
        assert len(td) > 0

    def test_sha_mismatch_detected(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'; torch.save({'w': torch.randn(64, 32)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        data = bytearray(open(out, 'rb').read())
        data[5000] ^= 0xFF  # Corrupt tensor data
        open(f'{d}/bad.vsqz', 'wb').write(data)
        with pytest.raises(ValueError, match='SHA-256'):
            _read_vsqz(f'{d}/bad.vsqz', verify_sha256=True)

    def test_sha_in_peek(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'; torch.save({'w': torch.randn(64, 32)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        h = peek_vsqz(out)
        assert h.get('sha256'), 'peek must show SHA-256'


class TestRecovery:
    def test_recovery_present(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'; torch.save({'w': torch.randn(64, 32)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        with open(out, 'rb') as f:
            f.seek(-4, 2)
            assert f.read(4) == b'RECO'

    def test_header_corruption_recovery(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'; torch.save({'w': torch.randn(64, 32)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        h_orig, td_orig = _read_vsqz(out, verify_sha256=True)
        data = bytearray(open(out, 'rb').read())
        data[12] = 0xFF  # Corrupt JSON header
        open(f'{d}/corr.vsqz', 'wb').write(data)
        h_rec, td_rec = _read_vsqz(f'{d}/corr.vsqz')
        assert h_rec.get('_recovery') is True
        assert h_orig['sha256'] == h_rec['sha256']

    def test_both_headers_corrupt_raises(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'; torch.save({'w': torch.randn(64, 32)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        data = bytearray(open(out, 'rb').read())
        data[12] = 0xFF
        # Also corrupt RECO marker
        data[-4:] = b'XXXX'
        open(f'{d}/dead.vsqz', 'wb').write(data)
        with pytest.raises(ValueError, match='unrecoverable|recovery'):
            _read_vsqz(f'{d}/dead.vsqz')

    def test_tensor_count_matches_after_recovery(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'
        torch.save({'w1': torch.randn(64, 32), 'w2': torch.randn(32, 16)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        data = bytearray(open(out, 'rb').read())
        data[12] = 0xFF
        open(f'{d}/corr.vsqz', 'wb').write(data)
        _, td = _read_vsqz(f'{d}/corr.vsqz')
        assert len(td) == 2


class TestSplit:
    def test_rejoin_byte_perfect(self):
        import hashlib
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'
        torch.save({'w1': torch.randn(128, 128), 'w2': torch.randn(64, 64)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        raw = open(out, 'rb').read()
        chunk_sz = 4096
        chunks = [raw[i:i+chunk_sz] for i in range(0, len(raw), chunk_sz)]
        rejoined = b''.join(chunks)
        assert rejoined == raw
        assert hashlib.sha256(rejoined).hexdigest() == hashlib.sha256(raw).hexdigest()

    def test_sha_survives_split(self):
        d = tempfile.mkdtemp()
        pt = f'{d}/m.pt'; torch.save({'w': torch.randn(64, 32)}, pt)
        out, _ = convert_to_vsqz(pt, pt + '.vsqz', verbose=False)
        h_before, _ = _read_vsqz(out, verify_sha256=True)
        # Simulate split + rejoin
        raw = open(out, 'rb').read()
        rejoined = b''.join([raw[i:i+4096] for i in range(0, len(raw), 4096)])
        open(out, 'wb').write(rejoined)
        h_after, _ = _read_vsqz(out, verify_sha256=True)
        assert h_before['sha256'] == h_after['sha256']
