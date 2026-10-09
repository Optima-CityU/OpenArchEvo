# -*- coding: utf-8 -*-
"""
spiking_fla_delta_snn
=====================
A minimal yet powerful fully–spiking Delta-rule building block that complies with
all strict requirements of the evolution framework:

1. Every *learnable* linear projection is replaced by **SpikingLinear** so its
   input is first converted to *binary* spikes emitted by a registered spiking
   neuron.
2. The associative Δ-rule kernel is retained for *O(L)* sub-quadratic sequence
   processing while keeping strict causality via chunked scanning.
3. All tensor reshaping is done with **einops.rearrange** – no view/reshape –
   and the implementation is completely *batch-size agnostic*.
4. Each forward pass returns ``final_spikes`` – a **flat, concatenated** tensor
   that contains the binary spikes produced by *all* spiking neurons inside the
   block (attention + MLP).
5. Class name **SpikingFLABlock** and public signatures are **unchanged** so the
   new implementation is a drop-in replacement for existing training / eval
   pipelines.

The design purposefully avoids external helper modules that might include
non-spiking ``nn.Linear`` layers (e.g. ShortConvolution) and instead focuses on
an extremely clean, easy-to-verify spiking pipeline.  This ensures full
compliance with the requirement that *every* linear projection is preceded by a
spiking neuron while still delivering competitive sequence modelling capacity
through the Δ-rule path.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm  # keeps parameter-free normalisation
from fla.modules.l2norm import l2norm  # unit-norm projection for delta rule

# -----------------------------------------------------------------------------
# Optional import of spiking neuron registry (falls back gracefully during docs)
# -----------------------------------------------------------------------------
try:
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except Exception:  # pragma: no cover – allow doc build without spikes
    neuron_registry = None
    piecewise_quadratic_surrogate = None

# -----------------------------------------------------------------------------
# Core spiking building block – MUST remain *exactly* as in legacy code
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A wrapper that feeds inputs through a spiking neuron *before* a Linear."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking modules not available – ensure src.models.spike is on PYTHONPATH.")
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transpose to (B, D, L) → neuron expects channels-first for event simulation
        spikes_t = self.neuron(x.transpose(-1, -2))  # binary spikes
        spikes = spikes_t.transpose(-1, -2)          # back to (B, L, D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# O(L) chunk-wise associative Δ-rule kernel (identical maths, pure PyTorch)
# -----------------------------------------------------------------------------
# pylint: disable=too-many-locals

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B, H, L, D_k)
    k: torch.Tensor,  # (B, H, L, D_k)
    v: torch.Tensor,  # (B, H, L, D_v)
    beta: torch.Tensor,  # (B, H, L)
    *,
    chunk_size: int = 32,
):
    """Associative scan with causal masking – **O(L)** time / O(1) memory per head."""
    b, h, L, d_k = q.shape

    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # ------------------------------------------------------------------
    # Normalise queries / keys to unit length for kernel stability
    # ------------------------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Chunk reshape – einops guarantees safety across arbitrary batch sizes
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_full = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    # Running state S is kept causal per head
    S = q.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S  # (B, H, L, D_v)

# -----------------------------------------------------------------------------
# Fully-spiking DeltaNet variant (attention / memory path only)
# -----------------------------------------------------------------------------
class SpikingDeltaNet(nn.Module):
    """A lean Δ-rule attention block where *all* linear projections are spiking."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        neuron_type: str = "sdn",
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry unavailable – cannot build network.")

        self.d_model = d_model
        self.num_heads = num_heads

        # Key / value dimensions (can be scaled independently if desired)
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "dims must divide heads"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate() if piecewise_quadratic_surrogate is not None else None

        # Spiking projections ------------------------------------------------
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)

        # Output projection back to model dimension
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

        # Lightweight RMS normalisation per head after Δ-rule mixing
        self.o_norm = RMSNorm(self.head_v_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, D)
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:

        spikes_accumulator = []  # collect binary spikes for *all* spiking neurons

        # --------------- Spiking projections ---------------------------
        q_lin, q_spk = self.q_proj(hidden_states)
        k_lin, k_spk = self.k_proj(hidden_states)
        v_lin, v_spk = self.v_proj(hidden_states)
        beta_lin, beta_spk = self.b_proj(hidden_states)

        spikes_accumulator += [q_spk, k_spk, v_spk, beta_spk]

        # Reshape for multi-head processing (einops ensures safety)
        q = rearrange(q_lin, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b h l d", h=self.num_heads)
        beta = rearrange(beta_lin.sigmoid(), "b l h -> b h l")  # gating scalar per head / token

        # --------------- Δ-rule kernel ---------------------------------
        out_d, _ = delta_rule_chunkwise(q, k, v, beta)  # (B, H, L, D_v)
        out = rearrange(out_d, "b h l d -> b l h d")  # back to (B, L, H, D_v)

        # Normalise & merge heads
        out = self.o_norm(out)
        out_flat = rearrange(out, "b l h d -> b l (h d)")

        # Final spiking projection to model dimension
        out_proj, o_spk = self.o_proj(out_flat)
        spikes_accumulator.append(o_spk)

        # --------------- Concatenate *all* spikes into single flat tensor ----
        final_spikes = torch.cat([s.flatten() for s in spikes_accumulator])

        return out_proj, None, final_spikes

# -----------------------------------------------------------------------------
# Fully-spiking MLP (two-layer) – again, only spiking linears allowed
# -----------------------------------------------------------------------------
class SpikingMLP(nn.Module):
    """A 2-layer feed-forward block with spiking linear projections."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
    ) -> None:
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking modules required for SpikingMLP")

        surrogate = piecewise_quadratic_surrogate() if piecewise_quadratic_surrogate is not None else None

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes_acc: list[torch.Tensor] = []

        h, sp1 = self.fc1(x)
        spikes_acc.append(sp1)
        h = self.act(h)
        h = self.drop(h)
        out, sp2 = self.fc2(h)
        spikes_acc.append(sp2)

        final_spikes = torch.cat([s.flatten() for s in spikes_acc])
        return out, final_spikes

# -----------------------------------------------------------------------------
# Public Spiking FLA Block – drop-in replacement with identical interface
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """FLA block with fully-spiking attention (Δ-rule) and spiking MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
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
        self.attn = SpikingDeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        else:
            raise ValueError("This evolved block *requires* the spiking MLP to satisfy the all-spiking constraint.")
        self.drop2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward – returns (hidden, reg_loss, final_spikes)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        res = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn, reg_loss, attn_spikes = self.attn(x, **kwargs)
        x = res + self.drop1(x_attn)

        res2 = x
        if self.prenorm:
            x = self.norm2(x)
        x_mlp, mlp_spikes = self.mlp(x)
        x = res2 + self.drop2(x_mlp)

        final_spikes = torch.cat([attn_spikes.flatten(), mlp_spikes.flatten()])
        return x, reg_loss, final_spikes
