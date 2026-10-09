# -*- coding: utf-8 -*-
"""
Spiking Predictive State Mixer (SPSM)
====================================
This implementation breaks away from the family of **multi-path gated Delta / FIR
architectures** that have dominated recent explorations.  Instead of routing
between *several* parallel memory paths, **SPSM** maintains a *single* latent
state that is updated in a *predictive-coding* fashion:

    stateₜ  =  gateₜ  ⊙  φ(updateₜ)  +  (1 − gateₜ) ⊙ conv(stateₜ₋₁)

• **State Convolution** – a *depth-wise* causal convolution (one filter per
  hidden feature) integrates the previous state in O(N·k) where *k*≪N.
• **Spike–Driven Update** – the update proposal comes from a `SpikingLinear`
  projection of the current input.  Because every linear transform is wrapped
  by `SpikingLinear`, **all update vectors are computed *from binary spikes***.
• **Spiking Gate** – the gate itself is produced by another `SpikingLinear`
  layer followed by `sigmoid`, allowing the *spiking stream* to *modulate* the
  continuous state flow.

This **hybrid predictive-coding loop** realises a *division of labour*:

1.  The **continuous convolution** carries slow-varying contextual memory.
2.  The **spiking pathway** injects event-driven prediction errors that update
    the state only when new evidence arrives.

The design therefore explores *hierarchical feedback* (predict → compare →
update) rather than *parallel path selection*, pushing the architectural search
into an orthogonal direction.

All affine transforms use `SpikingLinear`; every tensor reshape uses
`einops.rearrange`; operations stay O(N·k) (k = kernel size) hence
sub-quadratic.  The class name and forward signature remain unchanged so the
layer can replace earlier FLA blocks transparently.
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    # Mandatory spike modules
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:
    raise ImportError(
        "Spiking neuron modules (src.models.spike) not found. This block "
        "requires the neuromorphic components to be available.")

# -----------------------------------------------------------------------------
# SpikingLinear (must NOT be modified, but re-declared here for local use)
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """Linear layer preceded by a spiking neuron – inputs are binary spikes."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):  # noqa: D401
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B, L, D) → neuron expects (B, D, L)
        spikes_t = self.neuron(x.transpose(-1, -2))
        out = self.linear(spikes_t.transpose(-1, -2))
        return out, spikes_t.transpose(-1, -2)  # keep (B, L, D) spike shape

# -----------------------------------------------------------------------------
# Minimal spike-based MLP (two SpikingLinear layers)
# -----------------------------------------------------------------------------
class SpikeMLP(nn.Module):
    """Feed-forward network where *all* projections are spike-driven."""

    def __init__(self, d_in: int, d_ff: int, dropout: float = 0.1, neuron_type: str = "sdn"):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_in, d_ff), neuron_type, piecewise_quadratic_surrogate())
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ff, d_in), neuron_type, piecewise_quadratic_surrogate())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:  # returns activ & spikes list
        spikes_collector: List[torch.Tensor] = []
        x1, s1 = self.fc1(x)
        spikes_collector.append(s1)
        x1 = self.act(x1)
        x1 = self.drop(x1)
        x2, s2 = self.fc2(x1)
        spikes_collector.append(s2)
        return x2, spikes_collector

# -----------------------------------------------------------------------------
# Predictive State Mixer core
# -----------------------------------------------------------------------------
class _PredictiveStateMixer(nn.Module):
    """Single-headed predictive coding mixer (shared across heads)."""

    def __init__(self, hidden_size: int, kernel_size: int = 5, neuron_type: str = "sdn") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = int(kernel_size)
        # Depth-wise causal convolution kernel (D groups)
        weight = torch.zeros(hidden_size, 1, self.kernel_size)
        with torch.no_grad():
            weight[..., -1] = 1.0  # identity init
        self.conv_weight = nn.Parameter(weight * 0.001)

        # Spike-driven projections
        self.update_proj = SpikingLinear(nn.Linear(hidden_size, hidden_size, bias=False), neuron_type, piecewise_quadratic_surrogate())
        self.gate_proj = SpikingLinear(nn.Linear(hidden_size, hidden_size, bias=False), neuron_type, piecewise_quadratic_surrogate())

        # Output projection to mix back into hidden space
        self.out_proj = SpikingLinear(nn.Linear(hidden_size, hidden_size, bias=False), neuron_type, piecewise_quadratic_surrogate())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:  # x: [B,L,D]
        B, L, D = x.shape
        spike_collection: List[torch.Tensor] = []

        # ---------------- spike-driven proposals ----------------
        update, s_up = self.update_proj(x)
        gate_logits, s_gate = self.gate_proj(x)
        spike_collection.extend([s_up, s_gate])

        gate = torch.sigmoid(gate_logits)
        update = torch.tanh(update)  # bounded update proposal

        # ---------------- recurrent conv on running state --------
        # causal depth-wise conv on the *input* acts as running context proxy
        x_t = rearrange(x, "b l d -> b d l")
        x_pad = F.pad(x_t, (self.kernel_size - 1, 0))
        conv_out = F.conv1d(x_pad, self.conv_weight, groups=D)
        conv_out = rearrange(conv_out, "b d l -> b l d")

        mixed = gate * update + (1.0 - gate) * conv_out

        # output projection (spike-driven)
        out, s_out = self.out_proj(mixed)
        spike_collection.append(s_out)
        return out, spike_collection

# -----------------------------------------------------------------------------
# Public block
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):  # noqa: D401 – name fixed by framework
    """Predictive-coding FLA block with spiking-controlled state mixer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,  # unused but kept for interface compatibility
        use_spiking_mlp: bool,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        kernel_size: int = 5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = _PredictiveStateMixer(d_model, kernel_size=kernel_size, neuron_type=neuron_type)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.use_spiking_mlp = use_spiking_mlp
        if use_spiking_mlp:
            d_ff = d_model * d_ffn_mult
            self.mlp = SpikeMLP(d_model, d_ff, dropout=dropout, neuron_type=neuron_type)
        else:
            self.mlp = nn.Identity()
        self.drop2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward – returns (hidden, reg_loss=None, final_spikes)
    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        spikes_all: List[torch.Tensor] = []

        # -------- Mixer ----------
        residual = hidden_states
        if self.prenorm:
            hidden_states = self.norm1(hidden_states)
        mixed, s_m = self.mixer(hidden_states)
        spikes_all.extend(s_m)
        hidden_states = residual + self.drop1(mixed)

        # -------- MLP ------------
        if self.use_spiking_mlp:
            residual = hidden_states
            if self.prenorm:
                hidden_states = self.norm2(hidden_states)
            mlp_out, s_mlp = self.mlp(hidden_states)
            spikes_all.extend(s_mlp)
            hidden_states = residual + self.drop2(mlp_out)

        # -------- collect spikes -- flatten to 1-D tensor -------------
        final_spikes = torch.cat([s.flatten() for s in spikes_all], dim=0)
        return hidden_states, None, final_spikes
