# -*- coding: utf-8 -*-
"""
spiking_fla_simplenet (2024-06-10)
----------------------------------
A minimal but fully-compliant spiking SSM/Delta block that **ensures every
Linear layer receives discrete binary spikes**.  All ``nn.Linear`` instances
are wrapped by the provided ``SpikingLinear`` adapter, making the incoming
activations event-driven.  The core sequence model is a light variant of the
Delta kernel (chunk-wise associative scan) which preserves O(N) complexity and
strict causality.  A straightforward two-layer spiking MLP supplies additional
non-linearity.

The class ``SpikingFLABlock`` fully replaces the previous implementation while
retaining the exact public interface.  It collects spikes from **all** internal
``SpikingLinear`` modules and returns them concatenated as the required
``final_spikes`` tensor.

Design goals
============
1. **Total Spiking Compliance** – *every* linear projection (Q, K, V, β and
   output) is driven by binary spikes.
2. **Simplicity & Robustness** – remove auxiliary paths (rotary, mixers, etc.)
   to minimise surface for accidental ANN shortcuts.  The remaining operations
   are depth-wise convolutions and chunked recurrent scans – both naturally
   neuromorphic-friendly.
3. **Sub-quadratic Complexity** – the Delta scan is O(N) and uses a fixed
   ``chunk_size`` to bound inversion cost.
4. **Batch-size Agnosticism** – shapes are derived at runtime via
   ``einops.rearrange``; no hard-coded batch assumptions exist.

Note
====
The original ``SpikingLinear`` adapter is copied verbatim to satisfy the
*no-modification* constraint.  Re-declaration is harmless – Python keeps the
first identical definition.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm
from fla.modules.l2norm import l2norm

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None

# -----------------------------------------------------------------------------
# Mandatory adapter – copied verbatim (do not modify) --------------------------
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear Module with Spiking Neurons in front."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transpose to (B, D, L) so the neuron operates over feature channels
        spikes_transposed = self.neuron(x.transpose(-1, -2))  # binary spikes
        spikes = spikes_transposed.transpose(-1, -2)  # back to (B, L, D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# Helper – chunk-wise Delta rule (no adaptive decay for minimal footprint) -----
# -----------------------------------------------------------------------------

def _delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,Dk)
    k: torch.Tensor,  # (B,H,L,Dk)
    v: torch.Tensor,  # (B,H,L,Dv)
    beta: torch.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Linear-time associative scan as used in previous DeltaNet variants."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg_4d = (0, 0, 0, pad_len)
        q = F.pad(q, pad_cfg_4d)
        k = F.pad(k, pad_cfg_4d)
        v = F.pad(v, pad_cfg_4d)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Unit-norm map & gating ---------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape ------------------------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )
    tri_full = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)

    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (
            attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()
        ).sum(-2)
    eye = torch.eye(chunk_size, dtype=attn_inv.dtype, device=attn_inv.device)
    attn_inv = attn_inv + eye

    u = attn_inv @ v
    w = attn_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o  # (B,H,L,Dv)

# -----------------------------------------------------------------------------
# Lightweight spiking feed-forward network ------------------------------------
# -----------------------------------------------------------------------------
class _SpikingFFN(nn.Module):
    """Two-layer spiking MLP with SiLU activation and dropout."""

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 2,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
    ) -> None:
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available.")
        d_hidden = int(d_model * hidden_mult)
        surrogate = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_hidden, bias=False), neuron_type, surrogate)
        self.fc2 = SpikingLinear(nn.Linear(d_hidden, d_model, bias=False), neuron_type, surrogate)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, spike_collector: List[torch.Tensor]) -> torch.Tensor:
        x, s1 = self.fc1(x)
        spike_collector.append(s1)
        x = self.act(x)
        x = self.dropout(x)
        x, s2 = self.fc2(x)
        spike_collector.append(s2)
        return x

# -----------------------------------------------------------------------------
# Core spiking Delta (attention) module ---------------------------------------
# -----------------------------------------------------------------------------
class _SpikingDeltaNet(nn.Module):
    """Minimal spiking variant of DeltaNet (Q/K/V + O projections)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        dropout: float = 0.0,
        neuron_type: str = "sdn",
    ) -> None:
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        if self.key_dim % num_heads or self.value_dim % num_heads:
            raise ValueError("projected dims must divide num_heads")
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        surrogate = piecewise_quadratic_surrogate()

        # Projections ---------------------------------------------------------
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(self.head_v_dim)

    def forward(
        self, x: torch.Tensor, spike_collector: List[torch.Tensor]
    ) -> torch.Tensor:
        # Q/K/V/Beta projections --------------------------------------------
        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        beta_raw, s_b = self.b_proj(x)
        spike_collector.extend([s_q, s_k, s_v, s_b])

        # Shape to heads -----------------------------------------------------
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        beta = rearrange(beta_raw.sigmoid(), "b l h -> b h l")

        # Delta kernel -------------------------------------------------------
        out = _delta_rule_chunkwise(q, k, v, beta)  # (B,H,L,Dv)
        out = rearrange(out, "b h l d -> b l h d")
        out = self.norm(out)
        out = rearrange(out, "b l h d -> b l (h d)")

        # Output projection --------------------------------------------------
        out, s_o = self.o_proj(out)
        spike_collector.append(s_o)
        out = self.dropout(out)
        return out

# -----------------------------------------------------------------------------
# Public block (keeps original class name) ------------------------------------
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Fully-spiking FLA block – attention (Delta) + feed-forward."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_spiking_mlp: bool = True,  # kept for API-compat (always True here)
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = _SpikingDeltaNet(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            neuron_type=neuron_type,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = _SpikingFFN(
            d_model=d_model,
            hidden_mult=d_ffn_mult,
            dropout=dropout,
            neuron_type=neuron_type,
        )
        self.dropout2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # forward – signature unchanged (kwargs ignored for compatibility) --
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        spike_collector: List[torch.Tensor] = []

        residual = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn = self.attn(x, spike_collector)
        x = residual + self.dropout1(x_attn)

        residual = x
        if self.prenorm:
            x = self.norm2(x)
        x_mlp = self.mlp(x, spike_collector)
        x = residual + self.dropout2(x_mlp)

        # Concatenate all spike tensors into a flat vector -------------------
        final_spikes = torch.cat([s.flatten() for s in spike_collector]) if spike_collector else torch.empty(0)

        return x, None, final_spikes
