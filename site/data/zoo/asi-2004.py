# -*- coding: utf-8 -*-
"""
spiking_fla_cond_fuse – Conditioned Spiking Fusion DeltaNet
===========================================================
This file introduces a *fully spiking* variant of the original DeltaNet block.
All `nn.Linear` layers have been replaced with the mandatory `SpikingLinear`
wrapper so that *binary spike tensors* are presented to *every* linear weight
matrix.  A compact **Conditioned-Fusion Gate** (CFG) blends the original
Delta-rule global path with a value path and a depth-wise FIR local path while
retaining sub-quadratic (linear-time) complexity.

Key design highlights
---------------------
1. **Full Spiking Compliance** – Every projection (`q/k/v`, beta, output, gate)
   and every MLP affine transformation (feed-forward) uses the `SpikingLinear`
   wrapper.  A flat tensor `final_spikes` that concatenates *all* spike events
   generated inside the block is returned.
2. **Conditioned-Fusion Gate (CFG)** – A lightweight, per-head two-branch gate
   conditions on hidden state statistics (mean + ℓ2-norm) to allocate mixture
   weights between *global* (Delta) and *local* (Value+FIR) paths.  A fixed
   minimum floor ε prevents path starvation and ensures continuous gradient
   flow without expensive multi-branch softmaxes.
3. **Efficiency & Causality** – The core Delta rule kernel is kept identical
   and chunk-wise O(N). The gate adds only O(Nd) element-wise cost. Depth-wise
   FIR convolution is causal (left-padded) and linear-time.
4. **Batch Size Agnosticism** – All tensor reshaping relies exclusively on
   `einops.rearrange`, never on hard-coded batch dimensions.

The public API (class names, signatures, **kwargs) remains unchanged so that
this module can drop-in replace previous implementations without touching
up-stream code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover
    print("Warning: Spiking neuron modules (src.models.spike) not found.  Spiking components will be unavailable.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None

# ================================================================
# Utility – mandatory SpikingLinear wrapper (MUST stay unchanged)
# ================================================================
class SpikingLinear(nn.Module):
    """Linear transformation preceded by a spiking neuron."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry not found – cannot build SpikingLinear")
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,L,D] – convert to [B,D,L] for neuron which expects channels first
        spikes_t = self.neuron(x.transpose(-1, -2))  # binary spikes [B,D,L]
        spikes = spikes_t.transpose(-1, -2)          # back to [B,L,D]
        out = self.linear(spikes)                    # dense projection on spikes
        return out, spikes

# ---------------------------------------------------------------
# Depth-wise causal FIR conv (identity init)
# ---------------------------------------------------------------
class DepthwiseFIRConv1d(nn.Module):
    """Per-head causal FIR conv with identity (delta) kernel initialisation."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = int(kernel_size)
        filt = torch.zeros(num_heads, head_dim, self.kernel_size)
        with torch.no_grad():
            filt[..., -1] = 1.0  # identity kernel (current timestep)
        self.filters = nn.Parameter(filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,L,H,D]
        b, l, h, d = x.shape
        weight = rearrange(self.filters, "h d k -> (h d) 1 k")  # groups=h*d
        x_f = rearrange(x, "b l h d -> b (h d) l")
        # causal left-pad so index alignment is preserved
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

# ---------------------------------------------------------------
# Chunk-wise Delta kernel (unchanged numerics, compile-friendly)
# ---------------------------------------------------------------
# @torch.compile   # disabled per guidelines
def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Associative O(N) Delta rule with chunked causal computation."""
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

    tri_full = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(tri_full, 1)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=inv.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
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
    return o, S.detach()

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # typing alias

# ---------------------------------------------------------------
# Spiking DeltaNet with Conditioned-Fusion Gate
# ---------------------------------------------------------------
class DeltaNet(nn.Module):
    """Fully-spiking DeltaNet layer with lightweight conditioned fusion gate."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        expand_v: float = 2.0,
        neuron_type: str = "sdn",
        min_floor: float = 0.05,
        fir_kernel_size: int = 5,
        conv_size: int = 4,
        chunk_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        if neuron_registry is None or piecewise_quadratic_surrogate is None:
            raise ImportError("Spiking neuron registry/surrogate not available.")
        surrogate_fn = piecewise_quadratic_surrogate()

        self.d_model = int(d_model)
        self.head_dim = head_dim or (self.d_model // num_heads)
        self.num_heads = num_heads
        self.key_dim = self.num_heads * self.head_dim
        self.head_v_dim = int(self.head_dim * expand_v)
        self.value_dim = self.num_heads * self.head_v_dim
        self.expand_v = expand_v
        self.chunk_size = chunk_size
        self.min_floor = float(min_floor)

        # ------------- projections (all spiking) -------------
        self.q_proj = SpikingLinear(nn.Linear(self.d_model, self.key_dim, bias=False), neuron_type, surrogate_fn)
        self.k_proj = SpikingLinear(nn.Linear(self.d_model, self.key_dim, bias=False), neuron_type, surrogate_fn)
        self.v_proj = SpikingLinear(nn.Linear(self.d_model, self.value_dim, bias=False), neuron_type, surrogate_fn)
        self.b_proj = SpikingLinear(nn.Linear(self.d_model, self.num_heads, bias=False), neuron_type, surrogate_fn)

        # ------------- optional short conv preconditioning -------------
        self.q_conv = ShortConvolution(self.key_dim, conv_size, activation="silu")
        self.k_conv = ShortConvolution(self.key_dim, conv_size, activation="silu")
        self.v_conv = ShortConvolution(self.value_dim, conv_size, activation="silu")

        # ------------- local FIR branch -------------
        self.fir_local = DepthwiseFIRConv1d(self.num_heads, self.head_v_dim, kernel_size=fir_kernel_size)

        # ------------- fusion gate (per-head, two-path) -------------
        # Features per head: hidden vector (d_model) + mean & l2 (2 scalars)
        gate_in_dim = self.d_model + 2  # correct per-head feature dimension
        self.gate_fc = SpikingLinear(
            nn.Linear(gate_in_dim, 1, bias=True),  # output single scalar per (token, head)
            neuron_type,
            surrogate_fn,
        )

        # ------------- output normalisation & projection -------------
        self.out_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, self.d_model, bias=False), neuron_type, surrogate_fn)

        # container for spikes aggregated inside forward
        self._spikes: List[torch.Tensor] = []

    # -----------------------------------------------------------
    # forward
    # -----------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,L,D]
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None]:
        # Preserve original shapes for later padding restoration
        orig_bsz, orig_seq_len, _ = hidden_states.shape

        # (optional) unpad variable-length sequences for efficiency
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -orig_seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)
        else:
            cu_seqlens = None

        # After possible unpadding
        b_after, seq_after, _ = hidden_states.shape  # b_after may be 1 when unpadded

        # ------------- Q/K/V/Beta projections (collect spikes) -------------
        q_lin, spikes_q = self.q_proj(hidden_states)
        k_lin, spikes_k = self.k_proj(hidden_states)
        v_lin, spikes_v = self.v_proj(hidden_states)
        beta_raw, spikes_b = self.b_proj(hidden_states)

        self._spikes = [spikes_q, spikes_k, spikes_v, spikes_b]

        # ------------- Short conv refinement -------------
        # (ShortConvolution preserves batch/seq dims.)
        q_lin, _ = self.q_conv(q_lin, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        k_lin, _ = self.k_conv(k_lin, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        v_lin, _ = self.v_conv(v_lin, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)

        # ------------- Reshape into heads -------------
        q = rearrange(q_lin, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_v_dim)
        beta = torch.sigmoid(beta_raw)
        beta = rearrange(beta, "b l h -> b h l")

        # ------------- Global Delta path -------------
        o_delta, _ = delta_rule_chunkwise(q, k, v, beta, chunk_size=self.chunk_size)
        o_delta = rearrange(o_delta, "b h l d -> b l h d")

        # ------------- Local path (value + FIR) -------------
        v_direct = rearrange(v, "b h l d -> b l h d")  # same layout as delta output
        v_local = self.fir_local(v_direct)
        local_out = 0.5 * (v_direct + v_local)  # simple average inside local branch

        # ------------- Conditioned Fusion Gate -------------
        # statistics per token – mean + l2 of hidden state (computed on possibly unpadded sequence)
        hs_mean = hidden_states.mean(dim=-1, keepdim=True)              # [B_after,L_after,1]
        hs_l2 = hidden_states.norm(dim=-1, keepdim=True)                # [B_after,L_after,1]
        hs_stats = rearrange(torch.cat([hs_mean, hs_l2], dim=-1), "b l s -> b l 1 s").expand(-1, -1, self.num_heads, -1)
        gate_in = torch.cat([
            hidden_states.unsqueeze(2).expand(-1, -1, self.num_heads, -1),  # [B_after,L_after,H,D]
            hs_stats,  # [B_after,L_after,H,2]
        ], dim=-1)  # -> [B_after,L_after,H,D+2]

        # reshape to 3-D so that SpikingLinear receives [B_g,L,C]
        gate_in_3d = rearrange(gate_in, "b l h c -> (b h) l c")  # [B_after*H, L_after, C]
        gate_logits_3d, spikes_gate = self.gate_fc(gate_in_3d)  # [B_after*H, L_after, 1]
        self._spikes.append(spikes_gate)

        # collapse back to [B_after, L_after, H]
        gate_logits = gate_logits_3d.squeeze(-1).view(b_after, self.num_heads, seq_after).permute(0, 2, 1)

        # mixture weight with min-floor ε
        w_delta = torch.sigmoid(gate_logits)  # (0,1)
        w_delta = w_delta * (1.0 - 2 * self.min_floor) + self.min_floor
        w_local = 1.0 - w_delta
        w_delta = w_delta.unsqueeze(-1)  # [B_after,L_after,H,1]
        w_local = w_local.unsqueeze(-1)

        o = w_delta * o_delta + w_local * local_out  # [B_after,L_after,H,Dv]

        # ------------- Output norm + projection (spiking) -------------
        o = self.out_norm(o)
        o_lin_in = rearrange(o, "b l h d -> b l (h d)")
        o_proj, spikes_o = self.o_proj(o_lin_in)
        self._spikes.append(spikes_o)

        if attention_mask is not None:
            # o_proj currently [1, L_unpad, D]; restore to [orig_bsz, orig_seq_len, D]
            o_proj = pad_input(o_proj.squeeze(0), indices, orig_bsz, orig_seq_len)

        # ------------- concatenate all spikes into single tensor -------------
        self.final_spikes = torch.cat([s.flatten() for s in self._spikes])  # type: ignore[attr-defined]

        return o_proj, None, None  # reg_loss=None, cache=None

# ---------------------------------------------------------------
# Spiking-only MLP (2-layer) – every Linear wrapped
# ---------------------------------------------------------------
class SpikingMLP(nn.Module):
    """Two-layer MLP where both affine layers are preceded by spiking neurons."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1, neuron_type: str = "sdn"):
        super().__init__()
        if neuron_registry is None or piecewise_quadratic_surrogate is None:
            raise ImportError("Spiking modules unavailable – cannot build SpikingMLP")
        surrogate_fn = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_fn)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_fn)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out1, spikes1 = self.fc1(x)
        out1 = self.dropout(out1)
        out2, spikes2 = self.fc2(out1)
        final_spikes = torch.cat([spikes1.flatten(), spikes2.flatten()])
        return out2, final_spikes

# ---------------------------------------------------------------
# Public block wrapper – SpikingFLABlock (interface preserved)
# ---------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Full spiking FLA block: (DeltaNet + MLP) with prenorm & residuals."""

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
        self.mlp = (
            SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
            if use_spiking_mlp
            else nn.Identity()
        )
        self.dropout2 = nn.Dropout(dropout)

    # -----------------------------------------------------------
    # forward
    # -----------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):  # noqa: D401
        residual = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn, _reg, _ = self.attn(x, **kwargs)  # reg_loss not used currently
        x = residual + self.dropout1(x_attn)

        residual = x
        if self.prenorm:
            x = self.norm2(x)

        if isinstance(self.mlp, SpikingMLP):
            x_mlp, mlp_spikes = self.mlp(x)
        else:
            x_mlp, mlp_spikes = x, torch.tensor([], device=x.device)
        x = residual + self.dropout2(x_mlp)

        # collect spikes from attention & MLP
        attn_spikes = getattr(self.attn, "final_spikes", torch.tensor([], device=x.device))
        final_spikes = (
            torch.cat([attn_spikes, mlp_spikes])
            if (mlp_spikes.numel() or attn_spikes.numel())
            else torch.tensor([], device=x.device)
        )
        return x, None, final_spikes
