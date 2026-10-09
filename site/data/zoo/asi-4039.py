# -*- coding: utf-8 -*-
"""
spiking_fla_hblend – Hybrid FIR-Delta Spiking Block
===================================================
This implementation introduces a *fully-spiking* sequence modelling block that
combines

1.  A **causal Delta-rule global memory** path (O(N) chunk-wise implementation)
    for unlimited context propagation.
2.  A **depth-wise causal FIR convolution** (local branch) that efficiently
    captures short-range patterns.
3.  A **lightweight two-way gated fusion** whose *every* linear transformation
    is preceded by a **spike neuron**, enforced via the `SpikingLinear` wrapper.
4.  A **spiking feed-forward network** (SpikeFFN) used as the MLP sub-block.

All linear layers inside the block are created as `SpikingLinear`, guaranteeing
that their *inputs* are binary spike signals.  The block therefore fully
conforms to the "all-spike-in" requirement.

Complexity is strictly **O(N)** (chunk-wise Delta path + depth-wise 1-D conv)
with constant memory.  The class name (`SpikingFLABlock`) and public interface
are preserved for seamless drop-in replacement.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# External helper modules ----------------------------------------------------
from fla.modules import RMSNorm  # RMSNorm does not use nn.Linear internally
from fla.modules.l2norm import l2norm

try:
    # We rely on the existing spiking registry provided by the framework
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError as _err:
    raise ImportError(
        "Spiking neuron modules not found – ensure src.models.spike is on the PYTHONPATH.") from _err

###############################################################################
# --- Spiking wrappers ------------------------------------------------------- #
###############################################################################

class SpikingLinear(nn.Module):
    """A *linear* transformation whose **input** is first converted to binary
    spikes.  The wrapped ``nn.Linear`` instance is created *outside* and passed
    in so we keep initialisation identical to a vanilla dense layer.
    """

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B,L,D) -> transpose -> spike -> back-transpose so shapes stay intact
        spikes_t = self.neuron(x.transpose(-1, -2))          # (B,D,L)
        spikes = spikes_t.transpose(-1, -2)                  # (B,L,D)
        out = self.linear(spikes)
        return out, spikes

###############################################################################
# --- Lightweight depth-wise FIR convolution (local branch) ------------------#
###############################################################################

class DepthwiseFIRConv1d(nn.Module):
    """Per-head depth-wise causal FIR convolution with identity initialisation."""

    def __init__(self, *, num_heads: int, head_dim: int, kernel_size: int, noise_std: float = 1e-2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        filt[..., -1] = 1.0  # identity (δ) tap at current step
        if noise_std > 0:
            filt.add_(torch.randn_like(filt) * noise_std)
        self.filters = nn.Parameter(filt)  # (H,D,K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,L,H,D)
        b, l, h, d = x.shape
        weight = rearrange(self.filters, 'h d k -> (h d) 1 k')
        x_fd = rearrange(x, 'b l h d -> b (h d) l')
        x_pad = F.pad(x_fd, (self.kernel_size - 1, 0))   # causal left pad
        y = F.conv1d(x_pad, weight, groups=h * d)
        return rearrange(y, 'b (h d) l -> b l h d', h=h)

###############################################################################
# --- Chunk-wise Delta rule (O(N)) -------------------------------------------#
###############################################################################

# NOTE: identical mathematics to prior variants, only einops based reshaping

def _delta_rule_chunkwise(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, *, chunk_size: int = 32):
    """Causal associative Δ-rule retrieval.

    Shapes
    ------
    q, k, v : (B,H,L,D)
    beta    : (B,H,L)
    Returns : (B,H,L,D)
    """
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        # --- FIX: ensure the generator expression materialises into a list ---
        q, k, v = [F.pad(t, pad_cfg) for t in (q, k, v)]
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # normalise & beta
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # chunk reshape (einops ensures batch-agnostic)
    q, k, v, k_beta = map(lambda t: rearrange(t, 'b h (n c) d -> b h n c d', c=chunk_size), (q, k, v, k_beta))
    n_chunks = q.shape[2]

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(tri, 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    for blk in range(n_chunks):
        q_i, k_i = q[:, :, blk], k[:, :, blk]
        att_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, blk] - w[:, :, blk] @ S
        out[:, :, blk] = q_i @ S + att_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, 'b h n c d -> b h (n c) d')
    if pad_len:
        out = out[:, :, :L]
    return out

###############################################################################
# --- SpikeFFN (MLP) ---------------------------------------------------------#
###############################################################################

class SpikeFFN(nn.Module):
    """Spiking feed-forward network: GELU(SpikeLinear) → Drop → SpikeLinear."""

    def __init__(self, d_model: int, hidden_mult: int = 2, dropout: float = 0.1, *, neuron_type: str = 'sdn'):
        super().__init__()
        surrogate = piecewise_quadratic_surrogate()
        hidden_dim = d_model * hidden_mult
        self.fc1 = SpikingLinear(nn.Linear(d_model, hidden_dim), neuron_type, surrogate)
        self.fc2 = SpikingLinear(nn.Linear(hidden_dim, d_model), neuron_type, surrogate)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_all: List[torch.Tensor] = []
        x, s1 = self.fc1(x)
        spikes_all.append(s1)
        x = self.act(x)
        x = self.dropout(x)
        x, s2 = self.fc2(x)
        spikes_all.append(s2)
        return x, spikes_all

###############################################################################
# --- SpikingFLABlock --------------------------------------------------------#
###############################################################################

class SpikingFLABlock(nn.Module):
    """Fully-spiking FIR-Delta gated block (drop-in replacement)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_spiking_mlp: bool = True,  # kept for API compatibility, always spiking
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = 'sdn',
        fir_kernel: int = 5,
        delta_chunk: int = 32,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads.'
        self.prenorm = prenorm
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.delta_chunk = int(delta_chunk)

        surrogate = piecewise_quadratic_surrogate()

        # Normalisations -------------------------------------------------
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Projections ----------------------------------------------------
        self.q_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)

        # Local branch ---------------------------------------------------
        self.local_fir = DepthwiseFIRConv1d(num_heads=num_heads, head_dim=self.head_dim, kernel_size=fir_kernel)

        # Gating ---------------------------------------------------------
        self.gate_proj = SpikingLinear(nn.Linear(d_model, num_heads * 2, bias=True), neuron_type, surrogate)

        # Output projection ---------------------------------------------
        self.out_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward ---------------------------------------------------
        self.mlp = SpikeFFN(d_model, hidden_mult=d_ffn_mult, dropout=dropout, neuron_type=neuron_type)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):  # noqa: D401
        """Return ``output, None, final_spikes`` to match legacy signature."""
        spikes_accum: List[torch.Tensor] = []

        # --- Attention-like FIR/Delta fusion ---------------------------
        residual = x
        if self.prenorm:
            x = self.norm1(x)

        # ---- Spiking projections ----
        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        spikes_accum.extend([s_q, s_k, s_v])

        # reshape into heads -------------------------------------------
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)  # (B,H,L,D_h)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v_h = rearrange(v, 'b l (h d) -> b l h d', h=self.num_heads)  # keep (B,L,H,D_h) for local branch & delta value

        # beta – fixed 1 (no projection) -------------------------------
        beta = torch.ones_like(q[..., 0])  # (B,H,L)

        # ---- Delta path (global) ----
        delta_out = _delta_rule_chunkwise(q, k, rearrange(v_h, 'b l h d -> b h l d'), beta, chunk_size=self.delta_chunk)
        delta_out = rearrange(delta_out, 'b h l d -> b l h d')

        # ---- Local FIR path ---------
        local_out = self.local_fir(v_h)  # (B,L,H,D_h)

        # ---- Gating ---------------
        gate_logits, s_gate = self.gate_proj(x)          # (B,L,H*2)
        spikes_accum.append(s_gate)
        gate_logits = rearrange(gate_logits, 'b l (h s) -> b l h s', h=self.num_heads, s=2)
        weights = F.softmax(gate_logits, dim=-1)         # (...,2)

        # Fuse ----------------------------------------------------------
        o = weights[..., 0:1] * local_out + weights[..., 1:2] * delta_out
        o = rearrange(o, 'b l h d -> b l (h d)')         # back to (B,L,D)

        # Output proj + dropout ----------------------------------------
        o, s_out = self.out_proj(o)
        spikes_accum.append(s_out)
        o = self.dropout(o)
        x = residual + o                                  # residual add

        # --- Feed-forward --------------------------------------------
        residual_ff = x
        if self.prenorm:
            x = self.norm2(x)
        x_ff, spikes_ff = self.mlp(x)
        spikes_accum.extend(spikes_ff)
        x = residual_ff + self.dropout(x_ff)

        # --- Final spike tensor ---------------------------------------
        final_spikes = torch.cat([s.flatten() for s in spikes_accum]) if spikes_accum else x.new_zeros(0)
        return x, None, final_spikes
