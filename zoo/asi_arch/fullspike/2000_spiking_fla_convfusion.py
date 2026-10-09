# -*- coding: utf-8 -*-
"""
spiking_fla_convfusion
======================
A lightweight, fully-spiking FLA block that replaces every linear layer with
`SpikingLinear` and therefore guarantees that **all** learnable affine
transformations operate exclusively on binary spike tensors.  The design keeps
sub-quadratic (actually linear) complexity by employing a *causal depth-wise
convolution* instead of soft-max attention.  This yields an architecture that
is:

* strictly causal – left-padded convolution prevents future leakage,
* O(N·d) – depth-wise convolution and element-wise operations dominate cost,
* batch-agnostic – all shapes are inferred at run-time, never hard-coded,
* spike-native – every Linear layer is wrapped by `SpikingLinear`.

The block exposes the same public API (`SpikingFLABlock`) used throughout the
code-base so it can be swapped in seamlessly.  It returns a flat
`final_spikes` tensor that concatenates every spike generated inside the
module – a requirement for downstream neuromorphic monitoring.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm  # keep dependency list unchanged

try:
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except ImportError:  # pragma: no cover – allow import even when SNN libs absent
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    print("SpikingFLABlock will not be operational without them.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None

# -----------------------------------------------------------------------------
# SpikingLinear – must be kept *exactly* as provided in the specification -----
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """Linear transformation preceded by a spiking neuron layer.

    The wrapped ``nn.Linear`` never sees analogue activations – only binary
    spikes emitted by ``self.neuron``.  The forward call returns both the
    analogue *output* and the *binary spikes*.
    """

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):  # noqa: D401,E501
        super().__init__()
        if neuron_registry is None:
            raise ImportError(
                "Spiking neuron registry is unavailable – ensure neuromorphic "
                "modules are importable before constructing SpikingLinear.")
        if neuron_type not in neuron_registry:
            raise ValueError(f"Unknown neuron type '{neuron_type}'. Available: {list(neuron_registry.keys())}")
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        x_t = x.transpose(-1, -2)  # (B, D, L) – neuron expects channel dim first
        spikes_t = self.neuron(x_t)  # binary spikes (B, D, L)
        spikes = spikes_t.transpose(-1, -2)  # back to (B, L, D)
        out = self.linear(spikes)  # analogue output produced from binary input
        return out, spikes

# -----------------------------------------------------------------------------
# Simple spike-native MLP -------------------------------------------------------
# -----------------------------------------------------------------------------
class _SpikeMLP(nn.Module):
    """Two-layer MLP where **both** affine layers are `SpikingLinear`."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
    ) -> None:
        super().__init__()
        if piecewise_quadratic_surrogate is None:
            raise ImportError("Surrogate gradient function is unavailable.")
        surrogate = piecewise_quadratic_surrogate()
        # first affine
        linear1 = nn.Linear(d_model, d_ffn, bias=False)
        self.fc1 = SpikingLinear(linear1, neuron_type=neuron_type, surrogate_function=surrogate)
        # second affine
        linear2 = nn.Linear(d_ffn, d_model, bias=False)
        self.fc2 = SpikingLinear(linear2, neuron_type=neuron_type, surrogate_function=surrogate)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # first spike + GELU activation
        x1, spikes1 = self.fc1(x)
        x1 = F.gelu(x1)
        # second spike + dropout
        x2, spikes2 = self.fc2(self.dropout(x1))
        return x2, [spikes1, spikes2]

# -----------------------------------------------------------------------------
# Depth-wise causal convolution helper -----------------------------------------
# -----------------------------------------------------------------------------
class _DepthwiseCausalConv1d(nn.Module):
    """Length-preserving, causal, depth-wise 1-D convolution.

    Each channel is convolved independently (groups == channels) with an
    *identity-initialised* kernel to keep the signal path intact at start-up.
    Complexity: O(N · kernel_size) – linear in sequence length.
    """

    def __init__(self, channels: int, kernel_size: int = 7, init_std: float = 0.02) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        # weight shape: (channels, 1, K)
        weight = torch.zeros(channels, 1, self.kernel_size)
        with torch.no_grad():
            weight[..., -1] = 1.0  # identity (Dirac) at the last, causal tap
            weight.add_(torch.randn_like(weight) * init_std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, C)
        # rearrange to (B, C, L)
        x_ch = x.transpose(-1, -2)
        x_pad = F.pad(x_ch, (self.kernel_size - 1, 0))  # causal left pad
        y = F.conv1d(x_pad, weight=self.weight, groups=x_ch.shape[1])
        # back to (B, L, C)
        return y.transpose(-1, -2)

# -----------------------------------------------------------------------------
# Main FLABlock – spike-native convolutional fusion ----------------------------
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Fully spiking FLA block with depth-wise causal convolution core.

    The block consists of:
    1. *Input projection* – spike-linear transformation of the model state.
    2. *Causal convolution* – depth-wise, identity-initialised O(N) memory.
    3. *Output projection* – spike-linear back to model dimension.
    4. *Spike-MLP* – optional two-layer MLP, again fully spike-linear.

    All residual paths preserve causality, and the block returns a
    concatenated ``final_spikes`` tensor containing **all** binary spikes
    emitted inside the module (input proj, output proj, MLP layers).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,  # kept for API compatibility; unused in this variant
        use_spiking_mlp: bool,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        conv_kernel: int = 7,
        **kwargs: Dict,  # ignore extra args for forward-compatibility
    ) -> None:
        super().__init__()
        if piecewise_quadratic_surrogate is None:
            raise ImportError("Surrogate gradient function is unavailable.")
        surrogate = piecewise_quadratic_surrogate()
        self.prenorm = prenorm
        self.dropout_rate = dropout
        # normalisations
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # spike-linear projections (input/output)
        lin_in = nn.Linear(d_model, d_model, bias=False)
        self.in_proj = SpikingLinear(lin_in, neuron_type=neuron_type, surrogate_function=surrogate)
        lin_out = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = SpikingLinear(lin_out, neuron_type=neuron_type, surrogate_function=surrogate)

        # causal depth-wise convolution acting on the *spike* activations
        self.conv = _DepthwiseCausalConv1d(channels=d_model, kernel_size=conv_kernel)
        self.dropout = nn.Dropout(dropout)

        # optional spike-native MLP
        self.mlp: Optional[_SpikeMLP] = None
        if use_spiking_mlp:
            print(f"INFO: Layer uses fully-spiking MLP with neuron '{neuron_type}'.")
            d_ffn = d_model * d_ffn_mult
            self.mlp = _SpikeMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        # parameter placeholder for compatibility (unused)
        self.num_heads = num_heads

    # ------------------------------------------------------------------
    # Forward -----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None, torch.Tensor]:  # noqa: D401,E501
        """Forward pass retaining legacy return signature.

        Returns
        -------
        x_out : torch.Tensor
            The transformed hidden state with residual connections applied.
        None
            Placeholder for ``reg_loss`` (not used in this lightweight block).
        final_spikes : torch.Tensor
            **Flat 1-D** tensor containing the concatenated binary spikes from
            *all* spiking neurons inside the block.
        """
        del kwargs  # no op – keep API stable without extra side effects
        spike_collector: List[torch.Tensor] = []

        # 1) (optional) pre-norm ------------------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm1(x)

        # 2) spike-linear projection -------------------------------------------
        x_proj, spikes_in = self.in_proj(x)
        spike_collector.append(spikes_in)

        # 3) causal depth-wise convolution (O(N)) -------------------------------
        x_conv = self.conv(x_proj)  # (B, L, D)

        # 4) spike-linear output projection ------------------------------------
        x_conv_proj, spikes_out = self.out_proj(x_conv)
        spike_collector.append(spikes_out)

        # residual connection ---------------------------------------------------
        x = residual + self.dropout(x_conv_proj)

        # 5) MLP block -----------------------------------------------------------
        if self.mlp is not None:
            residual_mlp = x
            if self.prenorm:
                x = self.norm2(x)
            x_mlp, mlp_spikes = self.mlp(x)
            spike_collector.extend(mlp_spikes)
            x = residual_mlp + self.dropout(x_mlp)
        else:
            # still apply second norm for consistency if prenorm=False
            if not self.prenorm:
                x = self.norm2(x)

        # ----------------------------------------------------------------------
        # Concatenate *all* spikes emitted in the block into a single tensor
        # ----------------------------------------------------------------------
        final_spikes = torch.cat([s.flatten() for s in spike_collector]) if spike_collector else torch.empty(0)

        # As per legacy API: return (output, reg_loss=None, final_spikes)
        return x, None, final_spikes
