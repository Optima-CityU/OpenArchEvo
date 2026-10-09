# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""
SpikingFLABlock – Dual-Stream Causal Cortex (DSC-Block)
=====================================================
This implementation introduces a *dual–stream hybrid* design that
runs a **spiking control stream** in parallel with a **continuous
content stream**.  The control stream generates per-token, per-feature
gates (via spikes) that modulate the continuous content pathway.  A
causal depth-wise convolution augments the content stream with local
context.  All linear transformations are implemented with `SpikingLinear`,
thereby ensuring spike generation before every weight matrix.  A flat
concatenation of every spike tensor is returned as `final_spikes`.
The overall complexity is **O(N)** – convolution and element-wise gates –
so sub-quadratic requirements are satisfied.  No attention or quadratic
matrix products are used.

Key architectural departures from prior patterns
------------------------------------------------
1.  *Parallel control & content pathways* rather than a single fused
    attention / delta-rule core.
2.  *Spiking-driven gating* – the control pathway consumes spikes and
    emits gates that multiplicatively modulate a dense pathway.
3.  *Causal depth-wise convolution* introduces local context without
    attention, keeping operations linear in sequence length.
4.  *Spike aggregation* – every spike tensor is collected and returned
    to the caller, encouraging external regularisation or hardware
    routing.

All mandatory constraints from the developer instructions are respected:
•   `SpikingLinear` class is kept *exactly* as provided.
•   Every `nn.Linear` is wrapped by a `SpikingLinear` module.
•   `einops.rearrange()` is used for every reshape.
•   Forward signature and class name `SpikingFLABlock` are preserved.
•   A flat `final_spikes` tensor is returned.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover – make file import-safe for non-spiking envs
    print(
        "Warning: Spiking neuron modules (src.models.spike) not found.\n"
        "The resulting model will not run – import the required modules first."
    )
    neuron_registry = None  # type: ignore
    piecewise_quadratic_surrogate = None  # type: ignore

# ---------------------------------------------------------------------------
#  SpikingLinear (MUST remain exactly as specified – DO NOT EDIT)
# ---------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear Module with Spiking Neurons before."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        output = self.linear(spikes)
        return output, spikes

# ---------------------------------------------------------------------------
#  Helper Modules
# ---------------------------------------------------------------------------
class DepthwiseCausalConv1d(nn.Module):
    """Causal depth-wise 1-D convolution (kernel over *sequence* dimension)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # Weight shape: (C, 1, K)
        weight = torch.zeros(channels, 1, self.kernel_size)
        with torch.no_grad():
            weight[..., -1] = 1.0  # identity initialisation
            weight += 0.01 * torch.randn_like(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, C)
        x_t = rearrange(x, "b l c -> b c l")  # (B, C, L)
        x_pad = F.pad(x_t, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, self.weight, groups=x_t.shape[1])  # (B, C, L)
        y = rearrange(y, "b c l -> b l c")
        return y

class SpikingMLP(nn.Module):
    """Two-layer feed-forward MLP entirely built from SpikingLinear."""

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
        surrogate_function=piecewise_quadratic_surrogate(),
    ) -> None:
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")
        hidden_dim = d_model * hidden_mult
        self.lin1 = SpikingLinear(
            nn.Linear(d_model, hidden_dim, bias=False), neuron_type, surrogate_function
        )
        self.lin2 = SpikingLinear(
            nn.Linear(hidden_dim, d_model, bias=False), neuron_type, surrogate_function
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, spike_list: List[torch.Tensor]) -> torch.Tensor:
        x1, s1 = self.lin1(x)
        spike_list.append(s1)
        x1 = F.silu(x1)
        x1 = self.dropout(x1)
        x2, s2 = self.lin2(x1)
        spike_list.append(s2)
        return x2

# ---------------------------------------------------------------------------
#  Main Block
# ---------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Dual-Stream Causal Cortex block mixing spiking control and content paths."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,  # kept for interface compatibility (unused internally)
        use_spiking_mlp: bool = True,
        d_ffn_mult: int = 4,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        conv_kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        # --- Book-keeping & modules -------------------------------------------------
        self.prenorm = prenorm
        self.neuron_type = neuron_type
        self.dropout_rate = dropout
        surrogate_fn = piecewise_quadratic_surrogate()

        # Normalisations -------------------------------------------------------------
        self.norm_content = RMSNorm(d_model)
        self.norm_mlp = RMSNorm(d_model)

        # Content pathway linear (GLU-style) ----------------------------------------
        self.content_linear = SpikingLinear(
            nn.Linear(d_model, 2 * d_model, bias=False), neuron_type, surrogate_fn
        )
        # Control pathway linear -----------------------------------------------------
        self.control_linear = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_fn
        )

        # Local context via causal depth-wise convolution ----------------------------
        self.local_conv = DepthwiseCausalConv1d(d_model, kernel_size=conv_kernel_size)

        # Optional second MLP --------------------------------------------------------
        self.use_spiking_mlp = bool(use_spiking_mlp)
        if self.use_spiking_mlp:
            self.mlp = SpikingMLP(
                d_model=d_model,
                hidden_mult=d_ffn_mult,
                dropout=dropout,
                neuron_type=neuron_type,
                surrogate_function=surrogate_fn,
            )
        self.dropout = nn.Dropout(dropout)

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, D).

        Returns
        -------
        Tuple[torch.Tensor, None, torch.Tensor]
            (output, None, final_spikes)
        """
        spike_collector: List[torch.Tensor] = []

        residual = x
        if self.prenorm:
            x = self.norm_content(x)

        # ---- Content pathway -----------------------------------------------------
        content_out, spikes_c = self.content_linear(x)
        spike_collector.append(spikes_c)
        # Split for gated linear unit (GLU)
        # content_out: (B, L, 2D) -> ((B, L, D), (B, L, D))
        content_split = rearrange(content_out, "b l (two d) -> b l two d", two=2)
        content_part_a = content_split[:, :, 0]
        content_part_b = content_split[:, :, 1]

        # ---- Control pathway (spiking) -------------------------------------------
        control_out, spikes_ctrl = self.control_linear(x)
        spike_collector.append(spikes_ctrl)
        gate = torch.sigmoid(control_out)  # (B, L, D) in (0,1)

        # ---- Fuse gated content --------------------------------------------------
        gated_content = gate * content_part_a + (1.0 - gate) * content_part_b

        # ---- Local convolutional context ----------------------------------------
        local_context = self.local_conv(gated_content)

        fused = gated_content + local_context
        fused = self.dropout(fused)
        x = residual + fused  # residual connection

        # ---- Optional MLP --------------------------------------------------------
        if self.use_spiking_mlp:
            res2 = x
            if self.prenorm:
                x = self.norm_mlp(x)
            mlp_out = self.mlp(x, spike_collector)
            mlp_out = self.dropout(mlp_out)
            x = res2 + mlp_out

        # ---- Final spike aggregation -------------------------------------------
        final_spikes: torch.Tensor
        if spike_collector:
            final_spikes = torch.cat([s.flatten() for s in spike_collector])
        else:
            final_spikes = torch.empty(0, device=x.device)

        return x, None, final_spikes
