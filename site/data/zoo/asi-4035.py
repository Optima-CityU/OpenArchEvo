# -*- coding: utf-8 -*-
"""
spiking_fla_simplenet – Minimal-complexity SNN Delta-rule block with universal
SpikingLinear injection
================================================================================
This implementation **replaces every Linear layer by a SpikingLinear wrapper**
that inserts a trainable spiking-neuron in front of the weight matrix.  It keeps
O(N) chunk-wise Δ-rule global context propagation while dramatically
simplifying the original hierarchical gating logic – thereby avoiding large-
scale refactoring while still satisfying the *"all inputs to Linear come from
binary spikes"* requirement.

Key design points
-----------------
1.  **Universal SpikingLinear** – All projections (`q/k/v/o`) *and* MLP layers
    (`fc1/fc2`) are instances of `SpikingLinear`.  The class is re-used without
    modification so its contractual interface `(output, spikes)` remains
    intact.
2.  **Spike aggregation** – Every call to `SpikingLinear` returns a binary spike
    tensor.  All such tensors are collected during the forward pass and
    concatenated into a single 1-D `final_spikes` tensor that is returned by
    the outer `SpikingFLABlock`.
3.  **Sub-quadratic complexity** – Context mixing still relies on the proven
    `delta_rule_chunkwise` kernel (O(N·d)) and causal, depth-wise FIR
    convolutions; no softmax attention is introduced.
4.  **Batch-agnostic & Chunked** – The code uses `einops.rearrange` for all
    reshaping and never assumes a fixed batch/sequence length.
5.  **Interface compatibility** – `SpikingFLABlock` keeps its original
    signature `(x, **kwargs) -> Tuple[x_out, reg_loss, final_spikes]` so it
    remains a drop-in replacement.
6.  **Reasonable defaults** – New components are enabled by default, require no
    extra config parameters, and preserve all mandatory kwargs pass-through
    behaviour.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

try:
    # Local spike library (provided by benchmark harness)
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except ImportError:  # pragma: no cover
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None

# -----------------------------------------------------------------------------
# SpikingLinear – **must remain exactly identical** to original implementation
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear module preceded by a spiking neuron layer.

    NOTE:  Implementation is *unchanged* to guarantee interface and behavioural
    compatibility across the code-base.  The layer returns a tuple
    ``(output, spikes)`` so that downstream blocks can both use the linear
    projection *and* collect the emitted binary spikes.
    """

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry not available – cannot instantiate SpikingLinear.")
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_t = self.neuron(x_t)
        spikes = spikes_t.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:  # shifted ELU keeps >0
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # row-sum normalisation
    return (x / (x.sum(-1, keepdim=True) + 1e-6)).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule implementation (identical to prior proven kernel)
# -----------------------------------------------------------------------------

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B, H, L, D_k)
    k: torch.Tensor,  # (B, H, L, D_k)
    v: torch.Tensor,  # (B, H, L, D_v)
    beta: torch.Tensor,  # (B, H, L)
    *,
    chunk_size: int = 32,
):
    """Causal, O(N) associative scan kernel (see DeltaNet paper for derivation)."""
    bsz, n_heads, seq_len, d_k = q.shape
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    seq_len_pad = seq_len + pad_len

    # Normalisation & scaling -------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk re-shaping --------------------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (c n) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(bsz, n_heads, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    excl_mask = torch.triu(tri_mask, 1)

    for idx in range(seq_len_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(excl_mask, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :seq_len]
    return out, S

# -----------------------------------------------------------------------------
# Depth-wise causal FIR convolution (identity initialised)
# -----------------------------------------------------------------------------
class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal 1-D FIR convolution with identity init."""

    def __init__(self, *, num_heads: int, head_dim: int, kernel_size: int = 5, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity kernel
        if noise_std > 0:
            filt.add_(torch.randn_like(filt) * noise_std)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, H, D)
        bsz, seq_len, n_heads, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight, groups=n_heads * d)
        return rearrange(y, "b (h d) l -> b l h d", h=n_heads)

# -----------------------------------------------------------------------------
# Minimal DeltaNet variant with universal SpikingLinear wrappers
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):
    """Simplified DeltaNet preserving Δ-rule path + dual FIR branches.

    All Linear projections are instances of `SpikingLinear`; emitted spike
    tensors are collected and returned to the caller.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        fir_short_kernel: int = 5,
        fir_long_kernel: int = 63,
        conv_size: int = 4,
        prenorm_qk_act: str = "silu",
        norm_eps: float = 1e-5,
        neuron_type: str = "sdn",
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry not available – cannot instantiate DeltaNet.")

        self.hidden_size = d_model
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.head_k_dim = int(d_model * expand_k) // num_heads
        self.head_v_dim = int(d_model * expand_v) // num_heads
        self.key_dim = self.head_k_dim * num_heads
        self.value_dim = self.head_v_dim * num_heads

        surrogate_fn = piecewise_quadratic_surrogate() if callable(piecewise_quadratic_surrogate) else None

        # Projections (all SpikingLinear)
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate_fn)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate_fn)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate_fn)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate_fn)

        # Short causal convolution to enrich local context (optional but cheap)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu")
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu")
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")

        # FIR branches for local context
        self.fir_short = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_short_kernel)
        self.fir_long = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_v_dim, kernel_size=fir_long_kernel)

        # Output projection & norm
        self.out_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate_fn)

        # Storage for spikes
        self._spike_bucket: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Return (output, reg_loss=None, spikes_list)."""
        # Reset spike bucket for new forward pass
        self._spike_bucket.clear()

        # === Projections =====================================================
        q_lin, s_q = self.q_proj(hidden_states)
        k_lin, s_k = self.k_proj(hidden_states)
        v_lin, s_v = self.v_proj(hidden_states)
        beta_lin, s_b = self.b_proj(hidden_states)  # (B, L, H)
        self._spike_bucket.extend([s_q, s_k, s_v, s_b])

        # Optional short convolution enrichment --------------------------------
        q_lin = self.q_conv1d(q_lin)[0]  # ShortConvolution returns (out, None)
        k_lin = self.k_conv1d(k_lin)[0]
        v_lin = self.v_conv1d(v_lin)[0]

        # Reshape into heads ----------------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        # Activation & normalisation -------------------------------------------
        if kwargs.get("qk_act", "silu") == "relu":
            q, k = F.relu(q), F.relu(k)
        elif kwargs.get("qk_act", "silu") == "elu":
            q, k = _elu_plus_one(q), _elu_plus_one(k)
        if kwargs.get("qk_norm", "l2") == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # === Δ-rule global path ===============================================
        beta = torch.sigmoid(beta_lin)
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, _ = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # === Local FIR paths ===================================================
        local_short = self.fir_short(v)
        local_long = self.fir_long(v)

        # === Simple gated fusion ==============================================
        # We employ a lightweight content-free gate: 0.5 * global + 0.5 * (short+long)/2
        local_comb = 0.5 * (local_short + local_long)
        fused = 0.5 * local_comb + 0.5 * delta_out

        # === Norm + output projection =========================================
        fused = self.out_norm(fused)
        fused = rearrange(fused, "b l h d -> b l (h d)")
        out, s_out = self.o_proj(fused)
        self._spike_bucket.append(s_out)

        # Concatenate spikes ----------------------------------------------------
        final_spikes = torch.cat([s.flatten() for s in self._spike_bucket]) if self._spike_bucket else fused.new_empty(0)

        return out, None, final_spikes

# -----------------------------------------------------------------------------
# Spiking-only MLP – simple 2-layer feed-forward with SpikingLinear
# -----------------------------------------------------------------------------
class SpikingMLP(nn.Module):
    """Two-layer MLP where **both** Linear layers are SpikingLinear."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1, neuron_type: str = "sdn"):
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry not available – cannot instantiate SpikingMLP.")
        surrogate_fn = piecewise_quadratic_surrogate() if callable(piecewise_quadratic_surrogate) else None
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_fn)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_fn)

    def forward(self, x: torch.Tensor):
        spikes_bucket: List[torch.Tensor] = []
        h, s1 = self.fc1(x)
        spikes_bucket.append(s1)
        h = self.act(h)
        h = self.dropout(h)
        out, s2 = self.fc2(h)
        spikes_bucket.append(s2)
        return out, spikes_bucket

# -----------------------------------------------------------------------------
# SpikingFLABlock – top-level residual block exposing `final_spikes`
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Residual block: (Norm → DeltaNet) + (Norm → MLP) with universal spikes."""

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
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)

        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        else:
            # Even if a non-spiking MLP is requested, we still wrap its Linear layers
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        self.dropout2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):
        spike_collector: List[torch.Tensor] = []

        # === Attention / DeltaNet ============================================
        residual = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn, reg_loss, attn_spikes = self.attn(x, **kwargs)
        spike_collector.append(attn_spikes)
        x = residual + self.dropout1(x_attn)

        # === MLP =============================================================
        residual = x
        if self.prenorm:
            x = self.norm2(x)
        x_mlp, mlp_spikes = self.mlp(x)
        spike_collector.extend(mlp_spikes)
        x = residual + self.dropout2(x_mlp)

        # === Final spike tensor ==============================================
        final_spikes = torch.cat([s.flatten() for s in spike_collector if s.numel() > 0]) if spike_collector else x.new_empty(0)

        return x, reg_loss, final_spikes
