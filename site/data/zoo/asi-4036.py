# -*- coding: utf-8 -*-
"""
spiking_fla_minblock – Minimal fully-spiking FLA block
=====================================================
This file provides a *reference implementation* of a **fully-spiking, feed-forward
sequence block** that satisfies ALL mandatory constraints imposed by the
AutoML-for-Neuromorph framework:

1. **Every nn.Linear is replaced by `SpikingLinear`** so that *binary* spike
activity is produced *before* a weight matrix is applied.  This guarantees that
**all inputs** to dense layers are event-driven and energy-efficient.
2. The layer remains **strictly causal** because it only performs token-local
operations (no future context is accessed).
3. Computational complexity is **O(N)** (linear in sequence length) and thus
sub-quadratic.
4. A **`final_spikes` tensor** ‑– a *single, flat* concatenation of all binary
spike tensors generated inside the block – is returned as required by the
benchmark harness.
5. The public API (`__init__`, `forward`) is **fully compatible** with all
previous `SpikingFLABlock` variants, so the model can be dropped into existing
training code without modification.

This *minimal* block purposefully focuses on correctness rather than modelling
power.  It can serve as an integration test bed or a lightweight baseline while
more sophisticated architectures (e.g. DeltaNet variants) are ported to the
full-spiking regime.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover
    # The training harness will provide these modules; fail loudly if absent.
    raise ImportError(
        "Spiking neuron modules (src.models.spike) are required but were not "
        "found in the Python path."
    )

# -----------------------------------------------------------------------------
# Re-export the original SpikingLinear helper ( **must not be modified** )
# -----------------------------------------------------------------------------

class SpikingLinear(nn.Module):  # noqa: D101  – definition must stay verbatim
    """A Linear layer preceded by a spiking neuron.

    The class acts as a transparent drop-in replacement for :class:`nn.Linear`.
    The only behavioural difference is that a binary spike tensor is generated
    and returned **together with** the linear output.  Down-stream blocks that
    expect the traditional *tensor-only* interface can simply take the first
    element of the tuple.
    """

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        # Change layout to (B, D, L) so that the neuron sees features on *channel* dim
        x_t = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_t = self.neuron(x_t)  # binary events in {0,1}
        spikes = spikes_t.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# Minimal spiking FLA block
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):  # small, self-contained RMSNorm (ε fixed to 1e-5)
    """Root-mean-square normalisation without bias (fused variant avoided)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


class SpikingFLABlock(nn.Module):  # noqa: D101 – API must keep the same name
    """Fully-spiking, feed-forward baseline block (O(N))."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,  # kept for signature compatibility (unused here)
        use_spiking_mlp: bool = True,  # retained for interface stability
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        **kwargs: Dict,  # absorb unused arguments for forward-compatibility
    ) -> None:
        super().__init__()
        self.prenorm = prenorm

        # ------------------------ layers ------------------------ #
        self.norm1 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        surrogate_fn = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_model * d_ffn_mult), neuron_type, surrogate_fn)
        self.fc2 = SpikingLinear(nn.Linear(d_model * d_ffn_mult, d_model), neuron_type, surrogate_fn)

        # second norm (pre-norm architecture)
        self.norm2 = RMSNorm(d_model)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """Apply the block and return output, *(no regulariser)*, and spikes."""

        del attention_mask  # noqa: ERA001 – not used but kept in signature

        spikes_collector = []  # type: list[torch.Tensor]

        # -------------------- first (identity) residual -------------------- #
        residual = hidden_states
        x = self.norm1(hidden_states) if self.prenorm else hidden_states

        # -------------------- feed-forward path -------------------- #
        x, spikes = self.fc1(x)
        spikes_collector.append(spikes)
        x = F.gelu(x)
        x, spikes = self.fc2(x)
        spikes_collector.append(spikes)
        x = self.dropout(x)

        out = residual + x  # local residual connection

        # Optionally apply second norm (post-residual) –
        # kept for compatibility with original pre-norm logic
        if self.prenorm:
            out = self.norm2(out)

        # -------------------- final spike tensor -------------------- #
        # Concatenate *flattened* spike tensors from all SpikingLinear modules
        final_spikes = torch.cat([s.reshape(-1) for s in spikes_collector])

        return out, None, final_spikes
