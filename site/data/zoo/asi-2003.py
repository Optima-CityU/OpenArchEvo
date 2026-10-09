# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""
spiking_fla_msquota_router
==========================
A minimalist yet powerful evolution that unifies the strongest empirical
insights gathered so far:
  • Multi–scale fusion (local depth–wise causal convolution  +  global
    delta-rule memory)  — proven to boost both local precision and long
    range reasoning while keeping O(N) complexity.
  • Statistic–aware, **quota–regularised router** guaranteeing ≥δ mass on
    the global (delta) branch to avoid path starvation.  The quota is a
    *learnable* parameter per-head initialised to 0.1.
  • End-to-end **spiking linear projections** – every nn.Linear has been
    replaced by SpikingLinear so that inputs of *all* dense layers are
    binary spike trains produced by surrogate-gradient neurons.
  • Complete spike tracing – the layer returns a flat `final_spikes`
    tensor concatenating **all** binary spikes generated inside the
    attention block *and* the optional MLP, fulfilling the interface
    mandate.

The implementation maintains strict causality, sub-quadratic O(N) run-time
(via the proven chunk-wise delta kernel) and is batch-size agnostic thanks
to pervasive use of einops.rearrange for shape manipulation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm
from fla.modules.l2norm import l2norm
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover – fail-graceful for docs
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None

###############################################################################
# Helper modules                                                              #
###############################################################################

class SpikingLinear(nn.Module):
    """A Linear layer preceded by a spiking neuron producing binary spikes."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # neuron operates on (B, D, L) for efficiency, return to (B, L, D)
        spikes_t = self.neuron(x.transpose(-1, -2))  # binary spikes
        spikes = spikes_t.transpose(-1, -2)
        out = self.linear(spikes)
        return out, spikes  # downstream tensor & the spike events


class _DepthwiseCausalConv1d(nn.Module):
    """Depth-wise causal 1-D convolution (O(N·k)), one filter per head*dim."""

    def __init__(self, num_heads: int, head_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        weight = torch.zeros(num_heads * head_dim, 1, self.kernel_size)
        with torch.no_grad():
            weight[..., -1] = 1.0  # delta initialisation
            weight += 0.02 * torch.randn_like(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, H, D)
        b, l, h, d = x.shape
        x_ch = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_ch, (self.kernel_size - 1, 0))  # left pad for causality
        y = F.conv1d(x_pad, self.weight, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)


###############################################################################
# Delta-rule kernel (unchanged, proven O(N))                                   #
###############################################################################

# @torch.compile – avoided for stability

def _delta_rule_chunkwise(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, *, chunk_size: int = 32):
    """Associative delta memory in causal chunks.

    q,k,v: (B, H, L, Dk/Dv)   beta: (B, H, L)
    Returns: (B, H, L, Dv)
    """
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

    q, k, v, k_beta = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))

    mask_fut = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    mask_diag = torch.triu(mask_fut | torch.eye(chunk_size, dtype=torch.bool, device=q.device), 0)

    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask_diag, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask_fut, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S


###############################################################################
# DeltaNet with multi-scale quota router                                       #
###############################################################################

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache


class DeltaNet(nn.Module):
    """Multi-scale DeltaNet with statistic-aware, quota-regularised router.

    Two branches:
      • *delta*  – global associative memory (chunk-wise O(N)).
      • *local*  – depth-wise causal convolution (kernel = 7 by default).

    A learnable quota δ (per head) guarantees P_delta ≥ δ preventing global
    path starvation.  All dense projections are spiking.
    """

    def __init__(
        self,
        d_model: int | None = None,
        hidden_size: int = 512,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        expand_v: float = 2.0,
        # router / quota params
        local_kernel_size: int = 7,
        delta_quota_init: float = 0.1,
        neuron_type: str = "sdn",
        use_output_gate: bool = False,
        norm_eps: float = 1e-5,
        chunk_size: int = 32,
        layer_idx: int | None = None,
        **kwargs,
    ):
        super().__init__()
        if neuron_registry is None or piecewise_quadratic_surrogate is None:
            raise ImportError("Spiking neuron modules are not available.")

        surrogate_fn = piecewise_quadratic_surrogate()
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        self.head_k_dim = head_dim
        self.key_dim = num_heads * head_dim

        self.expand_v = expand_v
        self.head_v_dim = int(self.head_k_dim * expand_v)
        self.value_dim = self.head_v_dim * num_heads

        self.chunk_size = chunk_size
        self.layer_idx = layer_idx or 0

        # ------------- Spiking projections ----------------------------
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_fn)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_fn)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_fn)
        self.b_proj = SpikingLinear(nn.Linear(hidden_size, num_heads, bias=False), neuron_type, surrogate_fn)
        # router logits (two paths) local & delta
        self.router_proj = SpikingLinear(nn.Linear(hidden_size, num_heads * 2, bias=True), neuron_type, surrogate_fn)

        # output gate (optional)
        self.use_output_gate = bool(use_output_gate)
        if self.use_output_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_fn)

        # local branch
        self.local_conv = _DepthwiseCausalConv1d(num_heads, self.head_v_dim, kernel_size=local_kernel_size)

        # output projection back to model dim
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_fn)

        # quota parameter per-head (sigmoid ensures in (0,1))
        self.delta_quota_logit = nn.Parameter(torch.full((num_heads,), math.log(delta_quota_init / (1 - delta_quota_init))))

        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"], torch.Tensor]:
        B, L, _ = hidden_states.shape

        # --------------------------------------------------------------
        # Projections (spiking) & collect spikes
        # --------------------------------------------------------------
        spike_buffer: List[torch.Tensor] = []
        def _proj(layer: SpikingLinear, inp: torch.Tensor):
            out, spk = layer(inp)
            spike_buffer.append(spk)
            return out

        q = _proj(self.q_proj, hidden_states)
        k = _proj(self.k_proj, hidden_states)
        v = _proj(self.v_proj, hidden_states)
        beta = torch.sigmoid(_proj(self.b_proj, hidden_states))  # (B, L, H)
        router_logits = _proj(self.router_proj, hidden_states)   # (B, L, H*2)

        # reshape heads ------------------------------------------------
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        beta = rearrange(beta, "b l h -> b h l")

        # delta branch -------------------------------------------------
        delta_out, recurrent_state = _delta_rule_chunkwise(q, k, v, beta, chunk_size=self.chunk_size)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # local branch -------------------------------------------------
        v_direct = rearrange(v, "b h l d -> b l h d")  # (B,L,H,D)
        local_out = self.local_conv(v_direct)

        # router weights ----------------------------------------------
        router_logits = rearrange(router_logits, "b l (h p) -> b l h p", h=self.num_heads, p=2)
        weights = torch.softmax(router_logits, dim=-1)  # (B,L,H,2)

        # apply per-head quota on delta path
        quota = torch.sigmoid(self.delta_quota_logit)  # (H,)
        quota = quota.clamp(min=0.0, max=0.5)          # safety (≤0.5)
        quota = quota.view(1, 1, self.num_heads, 1)    # broadcast
        weights = weights * (1.0 - quota)              # scale all
        weights[..., 1:2] += quota                     # delta index 1

        fused = weights[..., 0:1] * local_out + weights[..., 1:2] * delta_out  # (B,L,H,Dv)

        # optional output gate ----------------------------------------
        if self.use_output_gate:
            gate_vec = _proj(self.g_proj, hidden_states)  # (B,L,V)
            gate = rearrange(gate_vec, "b l (h d) -> b l h d", h=self.num_heads)
            fused = fused * torch.sigmoid(gate)  # simple element-wise gate

        out = self.o_norm(fused)

        # project back -------------------------------------------------
        out_flat = rearrange(out, "b l h d -> b l (h d)")
        out_proj = _proj(self.o_proj, out_flat)  # final spikes collected inside

        # cache update stub (no state beyond recurrent_state needed for now)
        if past_key_values is not None and use_cache:
            past_key_values.update(dict(recurrent_state=recurrent_state), self.layer_idx, L)

        final_spikes = torch.cat([s.flatten() for s in spike_buffer])
        return out_proj, None, past_key_values, final_spikes


###############################################################################
# Spiking MLP                                                                 #
###############################################################################

class SpikingMLP(nn.Module):
    """Two-layer MLP using spiking linear projections throughout."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1, neuron_type: str = "sdn"):
        super().__init__()
        if neuron_registry is None or piecewise_quadratic_surrogate is None:
            raise ImportError("Spiking neuron modules are not available.")
        surrogate_fn = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_fn)
        self.drop = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_fn)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spike_buf: List[torch.Tensor] = []
        y1, s1 = self.fc1(x)
        spike_buf.append(s1)
        y1 = F.gelu(y1)
        y1 = self.drop(y1)
        y2, s2 = self.fc2(y1)
        spike_buf.append(s2)
        final_spikes = torch.cat([s.flatten() for s in spike_buf])
        return y2, final_spikes


###############################################################################
# Main Block                                                                  #
###############################################################################

class SpikingFLABlock(nn.Module):
    """Full layer comprising DeltaNet + optional SpikingMLP (prenorm)."""

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
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)

        d_ffn = int(d_model * d_ffn_mult)
        self.use_spiking_mlp = bool(use_spiking_mlp)
        if self.use_spiking_mlp:
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        else:
            raise ValueError("SpikingFLABlock mandates use_spiking_mlp=True for full spiking compliance.")
        self.drop2 = nn.Dropout(dropout)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):
        spike_collector: List[torch.Tensor] = []

        # ----- attention part -------------------------------------
        x_res = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn, _, _, attn_spk = self.attn(x, **kwargs)
        spike_collector.append(attn_spk)
        x = x_res + self.drop1(x_attn)

        # ----- MLP part -------------------------------------------
        x_res = x
        if self.prenorm:
            x = self.norm2(x)
        x_mlp, mlp_spk = self.mlp(x)
        spike_collector.append(mlp_spk)
        x = x_res + self.drop2(x_mlp)

        final_spikes = torch.cat(spike_collector)
        # reg_loss = None placeholder for compatibility
        return x, None, final_spikes
