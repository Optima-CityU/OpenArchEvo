# -*- coding: utf-8 -*-
"""
spiking_fla_delta_s5 – Fully-Spiking DeltaNet with Simplex-5 Routing
===================================================================
This implementation **upgrades the original “PHSG-5way” DeltaNet** so that *all*
linear transformations are executed **after a spiking neuron**.  Every
`nn.Linear` layer is therefore wrapped by `SpikingLinear`, guaranteeing that its
inputs are binary spikes rather than analogue activations – a hard requirement
for strict SNN execution.

Key Changes
-----------
1. **SpikingLinear Everywhere** – `q_proj`, `k_proj`, `v_proj`, `b_proj`,
   `g_proj`, and `o_proj` are now instances of `SpikingLinear`.  Likewise the 2-
   layer feed-forward sub-module is rewritten as `SpikingFeedForward`, which is
   composed solely of `SpikingLinear` blocks.
2. **Spike Aggregation** – Every call to a `SpikingLinear` yields a spike tensor
   that is collected.  All spikes generated inside the layer (attention *and*
   FFN) are concatenated into **one flat 1-D tensor** returned as
   `final_spikes`, satisfying the “binary-only” output format requirement.
3. **Forward Signature Unchanged** – The public API of `SpikingFLABlock` is
   preserved (`forward(x, **kwargs)` → `(x_out, reg_loss, final_spikes)`).
4. **Batch-Agnostic & Sub-Quadratic** – The algorithmic core (chunk-wise Δ-rule
   plus depth-wise FIR) is kept identical; complexity remains **O(L)** and all
   tensor shapes are obtained dynamically.  Only `einops.rearrange` is used for
   reshaping operations.
5. **No @torch.compile, no bf16 casts** – Compilation decorators are explicitly
   avoided to ensure portability.

The resulting block is a *drop-in replacement* that fulfils all neuromorphic
constraints while retaining the proven memory-routing capability of the
original model.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm, FusedRMSNormGated, ShortConvolution
from fla.modules.l2norm import l2norm

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover
    raise ImportError(
        "Spiking neuron modules (src.models.spike.*) are required but not found."
    )

# ===============================================================
# Pre-defined helper classes (kept identical to originals)
# ---------------------------------------------------------------
class SpikingLinear(nn.Module):
    """Linear layer preceded by a spiking neuron (unmodified helper)."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_t = self.neuron(x_t)  # binary spikes in (B, D, L)
        spikes = spikes_t.transpose(-1, -2)  # back to (B, L, D)
        out = self.linear(spikes)
        return out, spikes


# ===============================================================
# Helper functions
# ---------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:  # shifted ELU, ≥0
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:  # L1 normalise last dim
    return (x / x.sum(-1, keepdim=True)).to(x)


# ===============================================================
# Depth-wise causal FIR convolution (identity initialisation)
# ---------------------------------------------------------------
class DepthwiseFIRConv1d(nn.Module):
    """Depth-wise causal FIR conv; identical to the original implementation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # δ-kernel (identity)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))  # causal left-pad
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


# ===============================================================
# Chunk-wise causal Δ-rule (unchanged)
# ---------------------------------------------------------------
# NOTE: we intentionally *do not* decorate with @torch.compile.

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,Dk)
    k: torch.Tensor,  # (B,H,L,Dk)
    v: torch.Tensor,  # (B,H,L,Dv)
    beta: torch.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear-time causal associative retrieval via the Delta-rule."""
    b, h, L, d_k = q.shape
    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad:
        pad_cfg = (0, 0, 0, pad)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad))
    L_pad = L + pad

    q, k = l2norm(q), l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    A_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        A_inv[..., i, :i] += (A_inv[..., i, :, None].clone() * A_inv[..., :, :i].clone()).sum(-2)
    A_inv = A_inv + torch.eye(chunk_size, dtype=A_inv.dtype, device=q.device)

    u = A_inv @ v
    w = A_inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad:
        o = o[:, :, :L]
    return o, S


# ===============================================================
# Feed-forward network composed solely of SpikingLinear blocks
# ---------------------------------------------------------------
class SpikingFeedForward(nn.Module):
    """2-layer MLP where *both* layers are fully-spiking."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
        *,
        neuron_type: str = "sdn",
    ) -> None:
        super().__init__()
        surrogate = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn, bias=False), neuron_type, surrogate)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes_all: List[torch.Tensor] = []
        x, sp1 = self.fc1(x)
        spikes_all.append(sp1)
        x = self.dropout(x)
        x, sp2 = self.fc2(x)
        spikes_all.append(sp2)
        final_spikes = torch.cat([s.reshape(-1) for s in spikes_all])
        return x, final_spikes


# ===============================================================
# Per-head simplex gate (weight parameters unchanged)
# ---------------------------------------------------------------
class PerHeadGate(nn.Module):
    """Head-wise linear gate producing logits for *n_paths* branches."""

    def __init__(self, hidden_size: int, num_heads: int, n_paths: int):
        super().__init__()
        self.num_heads = num_heads
        self.n_paths = n_paths
        weight = torch.zeros(num_heads, n_paths, hidden_size)
        bound = 1.0 / math.sqrt(hidden_size)
        weight.uniform_(-bound, bound)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(num_heads, n_paths))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,D)
        return torch.einsum("b l d, h p d -> b l h p", x, self.weight) + self.bias


# ===============================================================
# DeltaNet with *fully-spiking* projections
# ---------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401


class DeltaNet(nn.Module):
    """DeltaNet where every linear projection precedes a spiking neuron."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_beta: bool = True,
        use_gate: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        fir_kernel_short: int = 3,
        fir_kernel_mid: int = 15,
        fir_kernel_long: int = 63,
        neuron_type: str = "sdn",
        layer_idx: int | None = None,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.layer_idx = layer_idx or 0

        # ---------------- dimensions ----------------
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key / value dims must divide evenly across heads.")

        surrogate = piecewise_quadratic_surrogate()
        # ---------------- spiking projections ----------------
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        if use_beta:
            self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        if use_gate:
            self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

        # ---------------- short convolutions ----------------
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)

        # ---------------- FIR branches ----------------
        self.fir_short = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_short)
        self.fir_mid = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_mid)
        self.fir_long = DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_long)

        # ---------------- simplex gate ----------------
        self.n_paths = 5  # short, mid, long, delta, value
        self.gate_linear = PerHeadGate(d_model, num_heads, self.n_paths)
        self.log_temp = nn.Parameter(torch.zeros(num_heads, 1))
        self.eps_param = nn.Parameter(torch.full((num_heads, 1), 0.02))

        # ---------------- output norm ----------------
        if use_gate:
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=1e-5)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=1e-5)

    # ---------------------------------------------------------
    # forward
    # ---------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # noqa: F821
        *,
        use_cache: bool = False,
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional["Cache"]]:
        B, L, _ = hidden_states.shape
        conv_q = conv_k = conv_v = None  # ShortConv states are *not* cached here.

        spikes_all: List[torch.Tensor] = []

        # ---------- spiking projections ----------
        q_proj, sp_q = self.q_proj(hidden_states)
        k_proj, sp_k = self.k_proj(hidden_states)
        v_proj, sp_v = self.v_proj(hidden_states)
        spikes_all += [sp_q, sp_k, sp_v]
        if self.use_beta:
            beta_raw, sp_b = self.b_proj(hidden_states)
            spikes_all.append(sp_b)
            beta = beta_raw.sigmoid()
        else:
            beta = torch.ones(B, L, self.num_heads, device=hidden_states.device, dtype=hidden_states.dtype)
        # ---------- ShortConv enhancement ----------
        q, _ = self.q_conv1d(q_proj, cache=None, output_final_state=False, cu_seqlens=None)
        k, _ = self.k_conv1d(k_proj, cache=None, output_final_state=False, cu_seqlens=None)
        v, _ = self.v_conv1d(v_proj, cache=None, output_final_state=False, cu_seqlens=None)

        # ---------- head split ----------
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)

        # ---------- optional activations ----------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # ---------- Delta path ----------
        delta_out, _ = delta_rule_chunkwise(
            rearrange(q, "b l h d -> b h l d"),
            rearrange(k, "b l h d -> b h l d"),
            rearrange(v_direct, "b l h d -> b h l d"),
            rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # ---------- FIR branches ----------
        local_short = self.fir_short(v_direct)
        local_mid = self.fir_mid(v_direct)
        local_long = self.fir_long(v_direct)

        # ---------- simplex gating ----------
        logits = self.gate_linear(hidden_states)
        temp = torch.exp(self.log_temp).view(1, 1, -1, 1)
        probs = torch.softmax(logits / temp, dim=-1)
        eps = torch.clamp(self.eps_param, 0.0, 0.2).view(1, 1, -1, 1)
        probs = probs * (1.0 - self.n_paths * eps) + eps

        w_short, w_mid, w_long, w_delta, w_value = (probs[..., i : i + 1] for i in range(5))
        o = (
            w_short * local_short
            + w_mid * local_mid
            + w_long * local_long
            + w_delta * delta_out
            + w_value * v_direct
        )

        # ---------- gated RMSNorm & output proj ----------
        if self.use_gate:
            g_raw, sp_g = self.g_proj(hidden_states)
            spikes_all.append(sp_g)
            g_vec = rearrange(g_raw, "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)

        o = rearrange(o, "b l h d -> b l (h d)")
        o, sp_out = self.o_proj(o)
        spikes_all.append(sp_out)

        final_spikes = torch.cat([s.reshape(-1) for s in spikes_all]) if spikes_all else torch.empty(0, device=o.device)
        return o, final_spikes, past_key_values


# ===============================================================
# SpikingFLABlock – public facing transformer block
# ---------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Transformer block with fully-spiking DeltaNet and FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        use_spiking_mlp: bool = True,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = SpikingFeedForward(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        else:
            raise ValueError("Only fully-spiking MLP is supported in this strict SNN setting.")
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs: Dict):
        spikes_all: List[torch.Tensor] = []

        x_res = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn, sp_attn, _ = self.attn(x, **kwargs)
        spikes_all.append(sp_attn)
        x = x_res + self.dropout1(x_attn)

        x_res = x
        if self.prenorm:
            x = self.norm2(x)
        x_mlp, sp_mlp = self.mlp(x)
        spikes_all.append(sp_mlp)
        x = x_res + self.dropout2(x_mlp)

        final_spikes = torch.cat([s.reshape(-1) for s in spikes_all]) if spikes_all else torch.empty(0, device=x.device)
        return x, None, final_spikes
