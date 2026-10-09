# -*- coding: utf-8 -*-
"""
spiking_fla_minimal_causal_conv – Lightweight Causal Convolutional SNN Block
==============================================================================
This implementation introduces a *radically simplified* yet biologically-motivated
sequence block that **converts every linear projection into a spiking operation**
via `SpikingLinear` and assembles them into a causal depth-wise convolutional
state-space pathway followed by a spiking feed-forward network.  The design
prioritises:

1.  UNIVERSAL SPIKING COMPLIANCE – every nn.Linear is now preceded by a spike
    neuron (see `SpikingLinear`).  All inputs to linear layers are therefore
    *binary spike events* rather than analogue activations.
2.  SUB-QUADRATIC COMPLEXITY – the only sequence-length dependent operation is
    a depth-wise 1-D convolution whose complexity is **O(N·k)** (linear in
    sequence length).  No soft-max attention is used.
3.  CHUNK-FREE CAUSALITY – causal padding ensures future tokens never
    contribute to current activations.
4.  FINAL SPIKE TRACE – the forward pass returns a `final_spikes` tensor that
    concatenates the binary outputs of *all* spiking neurons triggered inside
    the block (projection + FFN layers) complying with the specification.
5.  INTERFACE PARITY – the class name (`SpikingFLABlock`), constructor
    signature, and `forward()` signature remain unchanged.  Any extraneous
    `**kwargs` are silently accepted for drop-in compatibility with existing
    training code.

This streamlined architecture purposefully forgoes the heavy DeltaNet routing
machinery to **focus on perfect spiking correctness** under the new
constraints while retaining the residual / prenorm scaffold familiar to FLA
models.  It may serve as a clean reference for future SNN-centric experimentation.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm  # Using lightweight RMSNorm only

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError as e:  # pragma: no cover – runtime environment without spiking libs
    raise ImportError(
        "Spiking neuron modules (src.models.spike) are required for this SNN block."
    ) from e

# -----------------------------------------------------------------------------
# Re-use canonical SpikingLinear – MUST NOT BE MODIFIED per specification
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear layer preceded by a spiking neuron producing binary events."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_t = self.neuron(x_t)  # binary spikes
        spikes = spikes_t.transpose(-1, -2)  # back to (B, L, D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# Utility: Causal depth-wise 1-D convolution (per-feature independent kernel)
# -----------------------------------------------------------------------------
class DepthwiseCausalConv1d(nn.Module):
    """Depth-wise causal convolution with learnable FIR filter (linear time)."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        # shape: (channels, 1, kernel_size) so groups==channels = depth-wise
        weight = torch.zeros(channels, 1, kernel_size)
        with torch.no_grad():
            weight[..., -1] = 1.0  # start as identity / residual copy
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, D, L)
        x_ch = rearrange(x, "b l d -> b d l")
        # causal padding on the left (past context)
        x_pad = F.pad(x_ch, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, self.weight, groups=x_ch.shape[1])  # still (B, D, L)
        y = rearrange(y, "b d l -> b l d")
        return y

# -----------------------------------------------------------------------------
# Spiking Feed-Forward Network (2× SpikingLinear + GELU)
# -----------------------------------------------------------------------------
class SpikingFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
    ) -> None:
        super().__init__()
        surrogate = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_hidden, bias=False), neuron_type, surrogate)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_hidden, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_collector: List[torch.Tensor] = []
        x, s1 = self.fc1(x)
        spikes_collector.append(s1)
        x = self.act(x)
        x = self.dropout(x)
        x, s2 = self.fc2(x)
        spikes_collector.append(s2)
        return x, spikes_collector

# -----------------------------------------------------------------------------
# Main Block – maintains original class name & signatures
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Lightweight causal SNN sequence block with universal spiking projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,  # kept for API compatibility – not used here
        use_spiking_mlp: bool = True,  # ignored – always spiking now
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        conv_kernel: int = 7,
        **kwargs,  # absorb extraneous arguments for compatibility
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Sequence mixing pathway: Spiking projection -> causal depth-wise conv -> Spiking projection
        surrogate = piecewise_quadratic_surrogate()
        self.in_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        self.conv = DepthwiseCausalConv1d(d_model, kernel_size=conv_kernel)
        self.out_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.norm2 = RMSNorm(d_model)
        d_hidden = d_model * d_ffn_mult
        self.ffn = SpikingFFN(d_model, d_hidden, dropout=dropout, neuron_type=neuron_type)
        self.dropout2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass: returns (hidden, reg_loss=None, final_spikes)."""

        spike_accumulator: List[torch.Tensor] = []  # collect all spike tensors

        # ----- Sequence mixing path -----
        residual = x
        if self.prenorm:
            x = self.norm1(x)
        x, s_in = self.in_proj(x)
        spike_accumulator.append(s_in)
        x = self.conv(x)
        x, s_out = self.out_proj(x)
        spike_accumulator.append(s_out)
        x = residual + self.dropout1(x)

        # ----- Feed-forward path -----
        residual = x
        if self.prenorm:
            x = self.norm2(x)
        ffn_out, ffn_spikes = self.ffn(x)
        spike_accumulator.extend(ffn_spikes)
        x = residual + self.dropout2(ffn_out)

        # ----- Concatenate all spike events -----
        final_spikes = torch.cat([s.flatten() for s in spike_accumulator], dim=0)

        # reg_loss is None – placeholder for compatibility
        return x, None, final_spikes
