"""Spiking building blocks and the fixed wrapper (requires PyTorch)."""

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "openarchevo.models needs PyTorch. Install it (CPU or the CUDA wheel for your driver, "
        "https://pytorch.org/get-started/locally/), or `pip install 'openarchevo[zoo]'` for the "
        "archived zoo programs (torch, flash-linear-attention, einops)."
    ) from exc


from openarchevo.models.neurons import build_neuron, piecewise_quadratic_surrogate, registry
from openarchevo.models.program import BLOCK_CLASS, install_legacy_spike_alias, load_block_class, missing_imports
from openarchevo.models.spiking_linear import SpikingLinear
from openarchevo.models.wrapper import SequenceModel

__all__ = [
    "BLOCK_CLASS", "SequenceModel", "SpikingLinear", "build_neuron", "install_legacy_spike_alias",
    "load_block_class", "missing_imports", "piecewise_quadratic_surrogate", "registry",
]
