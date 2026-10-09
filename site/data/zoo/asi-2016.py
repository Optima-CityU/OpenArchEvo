# -*- coding: utf-8 -*-
"""
SpikingFLABlock – Parallel Spike-Gated Filter (PSF)
==================================================
Identifier: spiking_fla_psf

Paradigm shift
--------------
Traditional DeltaNet-derived blocks rely on *temporal associative rules* (Δ-rule)
plus multi-scale depth-wise convolutions.  Those designs repeatedly fuse several
**analog** branches (convolution, value, delta) through soft or hard gates.

The *Parallel Spike-Gated Filter (PSF)* explores an *orthogonal principle* based
on **event-driven multiplicative modulation**:

1.  A *spiking pathway* generates **binary gating masks** that decide – *per
    token, per feature* – whether the content update of a *continuous pathway*
    should be applied.
2.  The continuous pathway is a lightweight two-layer feed-forward filter that
    also operates on spikes (every Linear ⇒ `SpikingLinear`) but produces
    dense values.
3.  The final token update is the element-wise product *content × gate*,
    preserving strict causality and O(N) complexity without any Δ-rule or
    attention-like interactions.

The PSF therefore replaces the heavily reused *additive* fusion of analogue
branches with a *multiplicative* spike-controlled filter, opening a new design
space inspired by thalamo-cortical modulation where sparse gating neurons
control information flow of dense cortical columns.

Implementation guarantees
------------------------
• **All** linear projections are wrapped by `SpikingLinear`.
• A flat tensor `final_spikes` containing **every** spike fired inside the
  block is returned.
• Uses only `einops.rearrange` for reshaping; batch/sequence dimensions are
  inferred at runtime.
• Causal, purely token-wise computation ⇒ O(N) time / memory.
• `SpikingFLABlock` API (constructor & forward) remains unchanged.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------------------------------------------------------
# Optional import of spike modules (do *not* fail if unavailable at import time)
# -----------------------------------------------------------------------------
try:
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except ImportError:  # pragma: no cover
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    print("Spiking behaviour will be unavailable and code will raise at runtime.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None

# =============================================================================
# SpikingLinear (MUST remain bit-identical – do not modify)
# =============================================================================
class SpikingLinear(nn.Module):
    """A Linear module preceded by a spiking neuron layer.

    Parameters
    ----------
    linear_layer : nn.Linear
        The standard PyTorch Linear layer whose *inputs* will be the spikes.
    neuron_type : str
        Key in ``src.models.spike.neuron.registry`` identifying which neuron to
        instantiate.
    surrogate_function : callable
        Surrogate gradient function used by the spiking neurons.
    """

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ``SpikingLinear`` expects spikes in (B, D, L) format.
        spikes_t = self.neuron(x.transpose(-1, -2))  # (B, D, L)
        spikes = spikes_t.transpose(-1, -2)          # back to (B, L, D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _get_surrogate():
    if piecewise_quadratic_surrogate is None:
        raise ImportError("Spiking surrogate gradient implementation missing.")
    return piecewise_quadratic_surrogate()

# -----------------------------------------------------------------------------
# Main Block – MUST keep class name & interface *exactly*
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Parallel Spike-Gated Filter (PSF) block.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    num_heads : int
        Only kept for interface compatibility (unused by PSF, but may be logged
        upstream).  Must divide ``d_model`` if >1 because gating is performed
        per-feature.
    use_spiking_mlp : bool, optional
        Kept for API compatibility – has no effect (the PSF **always** uses
        spiking layers).
    d_ffn_mult : int, optional
        Hidden expansion multiplier of the internal feed-forward filter.
    dropout : float, optional
        Drop-out rate applied on the spike-gated update.
    prenorm : bool, optional
        Whether to apply RMSNorm before the block (same default as older
        architectures).
    neuron_type : str, optional
        Identifier of the spiking neuron to retrieve from ``registry``.
    **kwargs : Dict
        Silently ignored – preserves external configurability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_spiking_mlp: bool = True,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry not found – cannot build SpikingFLABlock.")

        self.d_model = d_model
        self.prenorm = prenorm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(d_model)

        # ------------------------------------------------------------------
        # 1.  Spiking *gate* pathway  – produces binary mask (B, L, D)
        # ------------------------------------------------------------------
        gate_linear = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = SpikingLinear(gate_linear, neuron_type, _get_surrogate())

        # ------------------------------------------------------------------
        # 2.  Continuous *content* pathway  – lightweight 2-layer FFN
        #     (Both layers are SpikingLinear ⇒ still operate on spike inputs.)
        # ------------------------------------------------------------------
        hidden_dim = d_model * d_ffn_mult

        ff_linear_1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.ff_proj1 = SpikingLinear(ff_linear_1, neuron_type, _get_surrogate())

        ff_linear_2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.ff_proj2 = SpikingLinear(ff_linear_2, neuron_type, _get_surrogate())

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, D)
        *args,  # ignore extra positional for compatibility
        **kwargs,  # accept arbitrary keyword args for compatibility
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """Forward pass returning (output, reg_loss=None, final_spikes)."""

        spikes_collector: List[torch.Tensor] = []

        # --------------------------------------------------------------
        # Optional RMS pre-normalisation
        # --------------------------------------------------------------
        x = self.norm(hidden_states) if self.prenorm else hidden_states

        # --------------------------------------------------------------
        # 1. Gate pathway (binary mask)
        # --------------------------------------------------------------
        _gate_out, gate_spikes = self.gate_proj(x)  # shape: (B, L, D)
        spikes_collector.append(gate_spikes)
        gate_mask = gate_spikes.float()             # 0/1 mask – broadcast-safe

        # --------------------------------------------------------------
        # 2. Content pathway – 2-layer spiking FFN
        # --------------------------------------------------------------
        ff_hidden, ff_spikes1 = self.ff_proj1(x)
        spikes_collector.append(ff_spikes1)
        ff_hidden = F.silu(ff_hidden)

        ff_out, ff_spikes2 = self.ff_proj2(ff_hidden)
        spikes_collector.append(ff_spikes2)

        # --------------------------------------------------------------
        # 3. Spike-gated update (multiplicative modulation)
        # --------------------------------------------------------------
        gated_update = gate_mask * ff_out  # element-wise (B, L, D)
        gated_update = self.dropout(gated_update)

        # Residual connection – identical to previous FLABlock API
        out = hidden_states + gated_update

        # --------------------------------------------------------------
        # 4. Aggregate *all* spikes into a flat tensor as required
        # --------------------------------------------------------------
        final_spikes = torch.cat([rearrange(s, "b l d -> (b l d)") for s in spikes_collector])

        return out, None, final_spikes
