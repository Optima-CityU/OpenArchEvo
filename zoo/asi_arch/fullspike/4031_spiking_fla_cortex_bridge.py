# -*- coding: utf-8 -*-
"""
CortexBridge – Dual-Stream Spiking-Controlled Convolutional Memory (CB-SCM)
===========================================================================
Identifier: spiking_fla_cortex_bridge

Paradigm Shift
--------------
This block breaks with the previous *multi-branch fusion* and *delta-rule*
family by introducing **two parallel yet functionally specialised streams**:

1.  A **Spiking-Controller stream** that emits *binary* gating signals via
    `SpikingLinear` projections.  These sparse, event-driven signals modulate
    the continuous pathway and therefore *control* information flow rather
    than carrying the content themselves.

2.  A **Continuous-Content stream** that builds a causal, depth-wise
    convolutional memory over the hidden representation.  The content stream
    never performs an `nn.Linear` – all point-wise projections are executed by
    `SpikingLinear`, guaranteeing that every learnable affine transform receives
    binary spikes as input.

The spiking gates (dorsal stream) multiplicatively blend the raw content with
its local memory trace (ventral stream), realising a biologically inspired
"cortico-cortical feedback loop".  No softmax/attention, MoE fusion or
Δ-kernel is used – establishing an **orthogonal design space** with strictly
O(N·k) complexity.

Output
------
`SpikingFLABlock.forward` returns `(hidden_out, None, final_spikes)` where
`final_spikes` is the flat, concatenated tensor of **all** spike tensors
produced inside the block, satisfying the required interface.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "Spiking neuron modules (src.models.spike) are required but not found."
    ) from err

# ---------------------------------------------------------------------------
# SpikingLinear wrapper (MUST remain EXACTLY as provided by the framework)
# ---------------------------------------------------------------------------


class SpikingLinear(nn.Module):
    """Linear transformation *preceded* by a spiking neuron."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Neuron expects channel dimension => transpose L and D
        spikes_t = self.neuron(x.transpose(-1, -2))  # (B, D, L)
        spikes = spikes_t.transpose(-1, -2)  # (B, L, D)
        out = self.linear(spikes)
        return out, spikes  # output as float, spikes as binary


# ---------------------------------------------------------------------------
# Utility: causal depth-wise 1-D conv (per-head)
# ---------------------------------------------------------------------------


class _DepthwiseCausalConv1d(nn.Module):
    """Depth-wise 1-D convolution with causal padding per (head, channel)."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.randn(num_heads * head_dim, 1, self.kernel_size) / math.sqrt(
            head_dim * self.kernel_size
        )
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        x_perm = rearrange(x, "b l h d -> b (h d) l")  # group per channel
        x_pad = F.pad(x_perm, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, self.weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


# ---------------------------------------------------------------------------
# RMSNorm – (unchanged minimal variant)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square normalisation (element-wise affine)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (…, D)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.scale


# ---------------------------------------------------------------------------
# Main Block – SpikingFLABlock (MANDATORY name)
# ---------------------------------------------------------------------------


class SpikingFLABlock(nn.Module):
    """Dual-stream spiking-controlled convolutional memory block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        conv_kernel_size: int = 5,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        surrogate_function=piecewise_quadratic_surrogate(),
        **kwargs,  # retains interface compatibility
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.prenorm = prenorm

        # Normalisations --------------------------------------------------
        self.norm_in = RMSNorm(d_model)
        self.norm_out = RMSNorm(d_model)

        # Spiking controller projections ----------------------------------
        self.gate_proj = SpikingLinear(
            nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate_function
        )  # produces per-head gate logits

        # Content projections ---------------------------------------------
        self.value_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function
        )

        # Depth-wise causal conv memory (content stream) -------------------
        self.dw_conv = _DepthwiseCausalConv1d(num_heads, self.head_dim, kernel_size=conv_kernel_size)

        # Output projection -----------------------------------------------
        self.out_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function
        )

        # Dropout ----------------------------------------------------------
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):  # x: (B, L, D)
        """Return: hidden_out, None, final_spikes"""

        residual = x
        if self.prenorm:
            x = self.norm_in(x)

        # ---------------- Spiking controller -------------------------
        gate_logits, spikes_gate = self.gate_proj(x)  # (B, L, H)
        gate = torch.sigmoid(gate_logits)  # (0,1) per-head gate

        # ---------------- Content path -------------------------------
        val_proj, spikes_val = self.value_proj(x)  # (B, L, D)
        val_heads = rearrange(val_proj, "b l (h d) -> b l h d", h=self.num_heads)
        mem = self.dw_conv(val_heads)  # local causal memory

        # ---------------- Fusion (spike-controlled) ------------------
        fused = gate.unsqueeze(-1) * mem + (1.0 - gate.unsqueeze(-1)) * val_heads
        fused_flat = rearrange(fused, "b l h d -> b l (h d)")

        # ---------------- Output projection & residual --------------
        out_proj, spikes_out = self.out_proj(self.norm_out(fused_flat))
        hidden_out = residual + self.dropout(out_proj)

        # ---------------- Collect spikes ---------------------------
        final_spikes = torch.cat(
            [spikes_gate.flatten(), spikes_val.flatten(), spikes_out.flatten()], dim=0
        )

        return hidden_out, None, final_spikes
