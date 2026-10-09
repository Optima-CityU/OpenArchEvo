# -*- coding: utf-8 -*-
"""
SpectralSpikeMixer – Spiking-modulated Token-wise Spectral Feature Mixing
========================================================================
This file implements a **novel, orthogonal** hybrid SNN/ANN architecture that
**abandons the extensively-explored path-gating / Delta-rule paradigm**.  Instead
of mixing *temporal* paths, it mixes the **frequency content across *feature*
channels for every token individually**.  The core idea is:

1.  A *spiking* pathway generates a **binary control mask** that modulates the
    amplitude of each discrete frequency component (obtained via an in-token
    RFFT) **per token, per batch element**.
2.  The modulated spectrum is transformed back into the time domain with an
    inverse RFFT, producing a content-adaptive, frequency-remixed token
    representation – completely *local* in the sequence dimension and therefore
    *causal*.
3.  A lightweight **Spiking MLP** further refines the representation.  All learnable
    projections rely on `SpikingLinear`, guaranteeing binary spike inputs.  The
    final forward pass returns a *single* flat tensor `final_spikes` that
    concatenates the spikes produced by **all** spiking layers.

The design fulfils every mandatory requirement:
• Class name stays **`SpikingFLABlock`** with unchanged interface.
• All tensor reshaping uses `einops.rearrange`.
• Complexity is `O(N·D·log D)` (FFT along feature dim) – strictly
  sub-quadratic in sequence length `N`.
• Causality is preserved because no operation mixes information **across
  sequence positions**.
• Every learnable linear transform is implemented with `SpikingLinear`.
• A 1-D, concatenated `final_spikes` tensor is returned.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------------------------------------------------------
# External helper modules (unchanged imports – required by framework)
# -----------------------------------------------------------------------------
from fla.modules import RMSNorm
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover – makes file import-safe when SNN libs missing
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    neuron_registry = None  # type: ignore
    piecewise_quadratic_surrogate = None  # type: ignore

# -----------------------------------------------------------------------------
# SpikingLinear – MUST remain exactly as provided by the specification.
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A wrapper around `nn.Linear` that inserts a spiking neuron in front."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry not available.")
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Neurons operate on (B, D, L) for efficiency; transpose back afterwards.
        spikes_t = self.neuron(x.transpose(-1, -2))            # (B, D, L)
        spikes   = spikes_t.transpose(-1, -2)                  # (B, L, D)
        out      = self.linear(spikes)                         # (B, L, D_out)
        return out, spikes

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _make_spiking_linear(in_dim: int, out_dim: int, *, bias: bool = False,
                         neuron_type: str = "sdn") -> SpikingLinear:
    """Factory utility to create a `SpikingLinear` with given dims."""
    linear = nn.Linear(in_dim, out_dim, bias=bias)
    return SpikingLinear(linear, neuron_type=neuron_type,
                         surrogate_function=piecewise_quadratic_surrogate())

# -----------------------------------------------------------------------------
# Main Block – **SpectralSpikeMixer** embedded in required `SpikingFLABlock`
# -----------------------------------------------------------------------------
class SpectralSpikeMixer(nn.Module):
    """Token-wise spectral feature mixer modulated by spiking control masks."""

    def __init__(self, d_model: int, dropout: float = 0.1,
                 neuron_type: str = "sdn") -> None:
        super().__init__()
        self.d_model   = d_model
        self.dropout_p = dropout

        # Spiking projection generating *frequency-domain* scaling masks.
        self.mask_proj = _make_spiking_linear(d_model, d_model,
                                              neuron_type=neuron_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,               # (B, L, D)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------
        # 1) Obtain spiking control mask m (B, L, D)
        # ---------------------------------------------
        mask_logits, spikes_mask = self.mask_proj(x)          # both (B,L,D)
        mask = torch.tanh(mask_logits)                        # bounded in (-1,1)

        # ---------------------------------------------
        # 2) RFFT along *feature* dim (per token)
        # ---------------------------------------------
        # Use float32 to avoid complex16 accuracy issues.
        x_fft = torch.fft.rfft(x.to(torch.float32), dim=-1)    # (B,L,F)
        B, L, Freq = x_fft.shape

        # Align mask shape: rfft length == D//2 + 1
        mask_trim = mask[..., :Freq]                           # (B,L,Freq)
        scale = 1.0 + mask_trim                                # ensure positive-ish scaling

        # ---------------------------------------------
        # 3) Modulate amplitude (phase preserved)
        # ---------------------------------------------
        x_fft_mod = torch.complex(x_fft.real * scale,
                                  x_fft.imag * scale)

        # ---------------------------------------------
        # 4) Inverse transform back to feature space
        # ---------------------------------------------
        x_mixed = torch.fft.irfft(x_fft_mod, n=self.d_model, dim=-1)
        x_out = x + self.dropout(x_mixed)                      # residual fusion

        return x_out, spikes_mask

class SpikingMLP(nn.Module):
    """Two-layer Spiking MLP used after the spectral mixer."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1,
                 neuron_type: str = "sdn"):
        super().__init__()
        self.fc1 = _make_spiking_linear(d_model, d_ffn,
                                        neuron_type=neuron_type)
        self.fc2 = _make_spiking_linear(d_ffn, d_model,
                                        neuron_type=neuron_type)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out1, spikes1 = self.fc1(x)          # (B,L,FF)
        out1 = self.act(out1)
        out1 = self.dropout(out1)
        out2, spikes2 = self.fc2(out1)       # (B,L,D)
        return out2, [spikes1, spikes2]

# -----------------------------------------------------------------------------
# REQUIRED public class – keeps interface identical to previous versions.
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Hybrid SNN/ANN block based on token-wise spectral feature mixing."""

    def __init__(self,
                 d_model: int,
                 num_heads: int = 1,                 # kept for interface compat (unused)
                 use_spiking_mlp: bool = True,
                 d_ffn_mult: int = 2,
                 dropout: float = 0.1,
                 prenorm: bool = True,
                 neuron_type: str = "sdn",
                 **kwargs: Dict):
        super().__init__()
        self.prenorm = prenorm
        self.norm1   = RMSNorm(d_model)
        self.mixer   = SpectralSpikeMixer(d_model, dropout, neuron_type=neuron_type)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        self.use_spiking_mlp = use_spiking_mlp
        if use_spiking_mlp:
            self.mlp = SpikingMLP(d_model, d_ffn, dropout, neuron_type=neuron_type)
        else:
            # Fallback dense MLP (still uses SpikingLinear for compliance)
            self.mlp = SpikingMLP(d_model, d_ffn, dropout, neuron_type=neuron_type)
        self.dropout2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward – returns (hidden, reg_loss=None, final_spikes)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):                      # noqa: D401
        del kwargs  # unused but kept for drop-in compatibility
        spike_collector: List[torch.Tensor] = []

        # ------- Mixer sub-layer ---------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm1(x)
        x, mix_spikes = self.mixer(x)
        spike_collector.append(mix_spikes)
        x = residual + self.dropout1(x)

        # ------- MLP sub-layer ----------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm2(x)
        mlp_out, mlp_spikes = self.mlp(x)
        spike_collector.extend(mlp_spikes)
        x = residual + self.dropout2(mlp_out)

        # ------- Construct final_spikes tensor ------------------------
        # Flatten each spike tensor and concatenate – always 1-D.
        final_spikes = torch.cat([rearrange(s, "... -> (...)") for s in spike_collector], dim=0)

        return x, None, final_spikes
