# -*- coding: utf-8 -*-
"""
spiking_fla_delta_htfs – Spiking DeltaNet with Hard-Threshold Fully-Spiking (HTFS) projections
=============================================================================================
This evolution hardens the "all–spikes-before-linear" requirement by replacing **every** Linear
projection with the provided ``SpikingLinear`` wrapper.  The design keeps the proven
chunk-wise Δ-rule global path and dual-scale (short + long) causal FIR convolutions but
simplifies the hierarchical router to a *parameter-efficient*, **per-head static softmax** that
mixes three streams:
    1.  local_fir  – (short + long) fused local context
    2.  delta_out  – O(N) recurrent global memory
    3.  v_direct   – identity / value bypass
Reducing the router to a temperature-controlled, learnable per-head vector eliminates dozens of
Linear layers, ensuring that *all* remaining dense projections are preceded by spiking neurons as
mandated.  All surfaces preserve the original interfaces so the layer is a drop-in replacement.

Key properties
--------------
* **All-Spiking Projections** – q/k/v, β, output projections and the 2-layer FFN all use
  ``SpikingLinear`` therefore receive binary spikes.
* **Sub-quadratic** – identical Δ-rule implementation (O(N · d)) with fixed 32-token chunks.
* **Batch-Agnostic & Causal** – uses einops for all reshapes; never assumes batch size.
* **final_spikes Tensor** – every spike tensor is concatenated and returned as ``final_spikes``
  to comply with the specification.
* **Minimal Parameter Overhead** – the router is a single learnable logit vector per head; the
  rest reuses existing efficient primitives.

The public API of ``SpikingFLABlock`` is unchanged.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional, List, Dict, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution, FusedRMSNormGated  # noqa: F401
from fla.modules.l2norm import l2norm

try:
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except ImportError:  # pragma: no cover – unit-test environments without SNN deps
    neuron_registry = None  # type: ignore
    piecewise_quadratic_surrogate = None  # type: ignore

################################################################################
# SpikingLinear – kept *exactly* as provided (do NOT modify)                   #
################################################################################

class SpikingLinear(nn.Module):
    """Linear preceded by a spiking neuron – definition mandated by framework."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(-1, -2)  # (B,L,D) -> (B,D,L)
        spikes_t = self.neuron(x_t)
        spikes = spikes_t.transpose(-1, -2)  # back to (B,L,D)
        out = self.linear(spikes)
        return out, spikes

################################################################################
# Helper utilities                                                             #
################################################################################

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU that stays strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)

################################################################################
# Core chunk-wise Δ-rule implementation (unchanged, O(N))                     #
################################################################################

# pylint: disable=too-many-locals, too-many-statements

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,D_k)
    k: torch.Tensor,  # (B,H,L,D_k)
    v: torch.Tensor,  # (B,H,L,D_v)
    beta: torch.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    excl_mask = torch.triu(tri_mask, 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(excl_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

################################################################################
# Depth-wise causal FIR convolution                                            #
################################################################################

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal 1-D FIR convolution with identity init."""

    def __init__(self, *, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity tap
        if noise_std > 0:
            filt.add_(torch.randn_like(filt) * noise_std)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

################################################################################
# Spiking-only Feed-Forward (2-layer)                                          #
################################################################################

class FullySpikingFFN(nn.Module):
    """Two-layer feed-forward network where *both* layers are SpikingLinear."""

    def __init__(self, d_model: int, expansion: int, dropout: float = 0.1, neuron_type: str = "sdn"):
        super().__init__()
        if neuron_registry is None or piecewise_quadratic_surrogate is None:
            raise ImportError("Spiking neuron modules are not available.")
        surrogate_fn = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, expansion), neuron_type, surrogate_fn)
        self.fc2 = SpikingLinear(nn.Linear(expansion, d_model), neuron_type, surrogate_fn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes: List[torch.Tensor] = []
        x, s1 = self.fc1(x)
        spikes.append(s1)
        x = F.gelu(x)
        x = self.dropout(x)
        x, s2 = self.fc2(x)
        spikes.append(s2)
        return x, spikes

################################################################################
# DeltaNet – ALL-SPIKING projections + lightweight static router               #
################################################################################

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401 – for type hints only


class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Spiking DeltaNet with simplified static router (per-head learnable softmax)."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int = 4,
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 64,
        use_beta: bool = True,
        allow_neg_eigval: bool = False,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        **kwargs,
    ) -> None:
        super().__init__()
        if neuron_registry is None or piecewise_quadratic_surrogate is None:
            raise ImportError("Spiking neuron modules are not available.")
        surrogate_fn = piecewise_quadratic_surrogate()

        self.hidden_size = d_model
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.allow_neg_eigval = allow_neg_eigval
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # --- dimensions
        self.key_dim = d_model
        self.value_dim = d_model
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # --- all-spiking projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), "sdn", surrogate_fn)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), "sdn", surrogate_fn)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), "sdn", surrogate_fn)
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), "sdn", surrogate_fn)

        # --- short convolution enhancer for q,k,v (mandatory for local context)
        self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=4, activation="silu")
        self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=4, activation="silu")
        self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=4, activation="silu")

        # --- FIR branches
        self.fir_short = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_short_kernel)
        self.fir_long = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_long_kernel)

        # --- Per-head static router logits (3-way)
        self.router_logits = nn.Parameter(torch.zeros(num_heads, 3))
        # Temperature parameter (>0) per head for sharper routing if needed
        self.log_temp = nn.Parameter(torch.zeros(num_heads) + math.log(1.0))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    # pylint: disable=too-many-statements

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes_collector: List[torch.Tensor] = []

        # ---- Spiking projections --------------------------------------
        q, s = self.q_proj(hidden_states)
        spikes_collector.append(s)
        k, s = self.k_proj(hidden_states)
        spikes_collector.append(s)
        v, s = self.v_proj(hidden_states)
        spikes_collector.append(s)

        q = self.q_conv1d(q)
        k = self.k_conv1d(k)
        v = self.v_conv1d(v)

        # ---- head split ----------------------------------------------
        # FIX: Ensure q, k, v are tensors before passing to rearrange
        if isinstance(q, tuple):
            q = q[0]
        if isinstance(k, tuple):
            k = k[0]
        if isinstance(v, tuple):
            v = v[0]
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        value_direct = v  # identity path

        # ---- β for delta memory path ----------------------------------
        if self.use_beta:
            beta_logits, s = self.b_proj(hidden_states)
            spikes_collector.append(s)
            beta = torch.sigmoid(beta_logits)
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Δ-rule global path ---------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, _ = delta_rule_chunkwise(q_d, k_d, v_d, beta_d, chunk_size=32)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---- FIR local paths -----------------------------------------
        local_short = self.fir_short(value_direct)
        local_long = self.fir_long(value_direct)
        local_comb = local_short + local_long  # simple add keeps params minimal

        # ---- Static router (per-head) ---------------------------------
        temp = torch.exp(self.log_temp).view(1, 1, self.num_heads, 1)  # (1,1,H,1)
        # FIX: Elementwise multiply per-head router logits by per-head temp (broadcast dims!)
        w = torch.softmax(self.router_logits * temp.view(self.num_heads, 1), dim=-1)  # (H,3)
        w = w.view(1, 1, self.num_heads, 3)

        out = (
            w[..., 0:1] * local_comb +
            w[..., 1:2] * delta_out +
            w[..., 2:3] * value_direct
        )

        out = rearrange(out, "b l h d -> b l (h d)")

        # ---- Collect spikes into single tensor ------------------------
        def unpack_spikes(collected):
            # flatten tuple-of-tensors or just tensor
            if isinstance(collected, tuple) or isinstance(collected, list):
                return [item.flatten() for item in collected]
            return [collected.flatten()]

        all_spikes = []
        for spk in spikes_collector:
            all_spikes.extend(unpack_spikes(spk))
        final_spikes = torch.cat(all_spikes, dim=0)

        return out, final_spikes

################################################################################
# Spiking-First Layer-Attention Block                                         #
################################################################################

class SpikingFLABlock(nn.Module):
    """Transformer-style block with fully-spiking projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_spiking_mlp: bool = True,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        **kwargs,
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = FullySpikingFFN(d_model=d_model, expansion=d_ffn, dropout=dropout, neuron_type=neuron_type)
        else:
            raise ValueError("use_spiking_mlp must be True to satisfy all-spiking constraint.")
        self.dropout2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):
        final_spikes: List[torch.Tensor] = []

        residual = x
        if self.prenorm:
            x = self.norm1(x)
        attn_out, spikes_attn = self.attn(x, **kwargs)
        x = residual + self.dropout1(attn_out)
        final_spikes.append(spikes_attn)

        residual = x
        if self.prenorm:
            x = self.norm2(x)
        mlp_out, mlp_spikes = self.mlp(x)
        x = residual + self.dropout2(mlp_out)
        final_spikes.extend(mlp_spikes)

        # Flatten & concat all spike tensors
        final_spikes_tensor = torch.cat([sp.flatten() for sp in final_spikes], dim=0)
        return x, None, final_spikes_tensor
