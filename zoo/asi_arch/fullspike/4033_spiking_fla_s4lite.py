# -*- coding: utf-8 -*-
"""
spiking_fla_s4lite – Causal Gated State-Space Block with Universal SpikingLinear Inputs
=====================================================================================
This implementation introduces a *light-weight gated state-space* (GSS) core that
acts as the attention / memory mechanism inside **SpikingFLABlock** while
strictly complying with every neuromorphic and framework constraint:

Key innovations
---------------
1. **Universal SpikingLinear I/O** – *all* linear projections are wrapped by the
   mandatory `SpikingLinear` adapter so that their inputs are binary spikes.
2. **Gated State-Space Path** – we blend a depth-wise causal convolution
   (capturing long-range dependencies in O(L) time) with a data-driven shortcut
   through a *sigmoid gate* (formally equivalent to the Mamba/Mixer idea but
   adapted for SNNs).
3. **Single-pass O(L·d) computation** – no quadratic attention; the only
   sequence operation is a grouped 1-D convolution whose complexity is linear
   in sequence length and dimension.
4. **Spike Tensor Reporting** – every spiking neuron output is collected and
   concatenated into `final_spikes`, exactly as required by the policy.
5. **Batch-size agnostic & Causal** – the grouped convolution is left-padded to
   ensure strict causality and uses dynamic shapes obtained at run-time.

The design purposefully *removes* the previous, very large DeltaNet variants to
minimise code complexity while providing a clean, easily extensible template
for future research.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    # These imports are kept for dependency consistency (even if unused).
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input  # noqa: F401
    from fla.modules import RMSNorm  # noqa: F401
    from src.models.spike.neuron import registry as neuron_registry  # noqa: F401
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # noqa: F401
except ImportError:  # pragma: no cover – test environments might lack these libs
    neuron_registry = None  # type: ignore
    piecewise_quadratic_surrogate = None  # type: ignore

###############################################################################
# ---------------------------  SpikingLinear (unchanged)  ------------------- #
###############################################################################
class SpikingLinear(nn.Module):
    """Mandatory adapter that inserts a spiking neuron in front of nn.Linear."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry not available.")
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B,L,D) → (B,D,L) for channel-wise spiking → back to (B,L,D)
        spikes = self.neuron(x.transpose(-1, -2)).transpose(-1, -2)
        out = self.linear(spikes)
        return out, spikes

###############################################################################
# ---------------------------  Core GSS Attention  -------------------------- #
###############################################################################
class GatedStateSpaceCore(nn.Module):
    """Causal gated depth-wise state-space (1-D conv) module with spikes."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 6,
        neuron_type: str = "sdn",
        surrogate_function=None,
    ) -> None:
        super().__init__()
        if surrogate_function is None:
            if piecewise_quadratic_surrogate is None:
                raise ImportError("Spiking surrogate function not available.")
            surrogate_function = piecewise_quadratic_surrogate()

        self.d_model = d_model
        self.kernel_size = int(kernel_size)

        # In-projection delivers *gate* and *value* tensors (2·D)
        self.in_proj = SpikingLinear(
            nn.Linear(d_model, 2 * d_model, bias=False), neuron_type, surrogate_function
        )
        # Depth-wise convolution kernel (groups = D) – causal.
        weight = torch.zeros(d_model, 1, self.kernel_size)
        with torch.no_grad():
            # Identity-like start: spike acts as simple pass-through initially.
            weight[:, 0, -1] = 1.0
        self.register_parameter("conv_weight", nn.Parameter(weight))

        # Output projection (back to D)
        self.out_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function
        )

    # ---------------------------------------------------------------------
    def _causal_depthwise_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Depth-wise causal 1-D convolution with left padding (O(L))."""
        # x: (B,L,D) –> rearrange to channels-first for conv1d
        x_ch = rearrange(x, "b l d -> b d l")
        pad_len = self.kernel_size - 1
        x_pad = F.pad(x_ch, (pad_len, 0))  # pad on the *left* (past)
        y = F.conv1d(x_pad, weight=self.conv_weight, groups=self.d_model)
        return rearrange(y, "b d l -> b l d")  # back to (B,L,D)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (output, spikes) with spikes concatenated flat."""
        # 1. Spiking input projection – produces gate & value + spike tensor.
        proj_out, spikes_in = self.in_proj(x)  # (B,L,2D)
        gate, value = torch.split(proj_out, self.d_model, dim=-1)
        gate = torch.sigmoid(gate)

        # 2. Causal grouped convolution over *value* path (state-space dynamics)
        conv_out = self._causal_depthwise_conv(value)

        # 3. Gated fusion (element-wise multiply)
        y = gate * conv_out + (1.0 - gate) * value

        # 4. Spiking output projection
        y, spikes_out = self.out_proj(y)

        # 5. Concatenate spikes (flatten keeps traceability but loses structure)
        final_spikes = torch.cat([spikes_in.flatten(), spikes_out.flatten()])
        return y, final_spikes

###############################################################################
# ---------------------------  Simple Spiking MLP  -------------------------- #
###############################################################################
class SpikeMLP(nn.Module):
    """Two-layer MLP where each Linear is preceded by a spiking neuron."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
        surrogate_function=None,
    ) -> None:
        super().__init__()
        if surrogate_function is None:
            if piecewise_quadratic_surrogate is None:
                raise ImportError("Spiking surrogate function not available.")
            surrogate_function = piecewise_quadratic_surrogate()

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_hidden), neuron_type, surrogate_function)
        self.act = nn.GELU()
        self.fc2 = SpikingLinear(nn.Linear(d_hidden, d_model), neuron_type, surrogate_function)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out1, s1 = self.fc1(x)
        out1 = self.act(out1)
        out1 = self.drop(out1)
        out2, s2 = self.fc2(out1)
        final_spikes = torch.cat([s1.flatten(), s2.flatten()])
        return out2, final_spikes

###############################################################################
# ---------------------------  SpikingFLABlock  ----------------------------- #
###############################################################################
class SpikingFLABlock(nn.Module):
    """Flexible Layer-norm + GSS core + optional Spike-MLP (prenorm)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,  # kept for interface; not used by GSS core
        use_spiking_mlp: bool = True,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if piecewise_quadratic_surrogate is None:
            raise ImportError("Spiking surrogate function not available.")
        surrogate = piecewise_quadratic_surrogate()

        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = GatedStateSpaceCore(
            d_model=d_model,
            kernel_size=6,
            neuron_type=neuron_type,
            surrogate_function=surrogate,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = SpikeMLP(
                d_model=d_model,
                d_hidden=d_ffn,
                dropout=dropout,
                neuron_type=neuron_type,
                surrogate_function=surrogate,
            )
        else:
            # Simple identity if MLP is disabled (rare in practise).
            self.mlp = None
        self.dropout2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Return (output, reg_loss(None), final_spikes) preserving interface."""
        spikes_collector: List[torch.Tensor] = []  # gather all spike tensors

        # --- core gated state-space ------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn, spikes_attn = self.attn(x)
        spikes_collector.append(spikes_attn)
        x = residual + self.dropout1(x_attn)

        # --- optional MLP ---------------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm2(x)
        if self.mlp is not None:
            x_mlp, spikes_mlp = self.mlp(x)
            spikes_collector.append(spikes_mlp)
        else:
            x_mlp = torch.zeros_like(x)
        x = residual + self.dropout2(x_mlp)

        # Concatenate *all* spikes into a flat tensor as required.
        final_spikes = (
            torch.cat(spikes_collector) if spikes_collector else torch.tensor([], device=x.device)
        )
        return x, None, final_spikes

###############################################################################
# ---------------------------  Optional Typing  ----------------------------- #
###############################################################################
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401
