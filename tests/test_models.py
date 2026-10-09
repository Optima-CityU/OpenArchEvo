"""Block interface, legacy imports and fingerprint extraction (skipped without PyTorch)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from openarchevo.config import FingerprintConfig, ProbeConfig  # noqa: E402
from openarchevo.fingerprint.schema import NUM_FEATURES  # noqa: E402
from openarchevo.models import load_block_class  # noqa: E402

BLOCKS = sorted((Path(__file__).resolve().parents[1] / "examples" / "blocks").glob("*.py"))

LEGACY_PROGRAM = '''
import torch, torch.nn as nn
from src.models.spike.neuron import registry as neuron_registry
from src.models.spike.surrogate import piecewise_quadratic_surrogate

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, neuron_type="bptt", **kwargs):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x, **kwargs):
        s = self.neuron(x.transpose(1, 2)).transpose(1, 2)
        return x + self.linear(s), None, s.flatten()
'''


@pytest.mark.parametrize("path", BLOCKS, ids=lambda p: p.stem)
def test_block_interface(path):
    block = load_block_class(path)(d_model=32, num_heads=4, neuron_type="bptt")
    x = torch.randn(2, 70, 32, requires_grad=True)
    y, reg, spikes = block(x)
    assert y.shape == x.shape and spikes.dim() == 1
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}
    (y.sum() + (reg if reg is not None else 0)).backward()
    assert x.grad is not None


def test_legacy_program_imports():
    block = load_block_class(LEGACY_PROGRAM)(d_model=16, num_heads=2)
    assert block(torch.randn(1, 10, 16))[0].shape == (1, 10, 16)


def test_fingerprint_is_deterministic():
    from openarchevo.fingerprint.extract import extract_fingerprint

    cfg = FingerprintConfig(neuron_type="bptt", probe=ProbeConfig(batch_size=4, seq_len=32), structure_seq_len=48)
    code = BLOCKS[0].read_text()
    fp1, fp2 = extract_fingerprint(code, cfg, "cpu"), extract_fingerprint(code, cfg, "cpu")
    assert len(fp1) == NUM_FEATURES and fp1 == pytest.approx(fp2)
