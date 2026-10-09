# -*- coding: utf-8 -*-
"""
SpikingDeltaConvMixer – Fully-Spiking DeltaNet with Multi-Scale Depthwise Convolution
===================================================================================
This evolutionary step **spikes** every linear transformation in the original
`DepthwiseConvPointMixer` while **keeping the O(N) computational profile**.  All
`nn.Linear` layers are wrapped inside `SpikingLinear`, which places a spiking
neuron in front of the weight matrix.  During the forward pass **all generated
binary spike tensors are collected and concatenated** so the caller can inspect
real spike events (not averaged firing-rates).

Key Highlights
--------------
1. **End-to-End Spiking Path** – query / key / value projections, gate MLP,
   kernel mixing and output projection are all converted to `SpikingLinear`.
   Consequently **every tensor entering a weight matrix is a spike-train**.

2. **Spike Collection Utility** – every `SpikingLinear` returns `(output,
   spikes)`; the layer concatenates all spikes into `self._spikes`, which the
   outer `SpikingFLABlock` retrieves and finally returns as `final_spikes`.

3. **Interface Preservation** – the class names (`DeltaNet`,
   `MultiScaleDepthwiseConv1d`, `SpikingFLABlock`) and all public signatures are
   **unchanged** so the new code can be dropped into existing pipelines.

4. **Batch-Agnostic & Sub-Quadratic** – all computations rely on
   `einops.rearrange` for safe reshaping, and the associative Δ-rule kernel
   keeps the overall complexity **O(N)**.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, FusedRMSNormGated, ShortConvolution
from fla.modules.l2norm import l2norm

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None  # type: ignore[assignment]

# ----------------------------------------------------------------------------
# SpikingLinear (MUST remain unmodified)
# ----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear layer preceded by a spiking neuron (binary activation)."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor):  # -> (output, spikes)
        # spike neuron expects (B, D, L); we use last dim = length / tokens
        spikes = self.neuron(x.transpose(-1, -2))  # (B,D,L)
        spikes = spikes.transpose(-1, -2)          # back to (B,L,D)
        out = self.linear(spikes)
        return out, spikes

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Shifted ELU (+1) – strictly positive."""
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:
    """L1 normalisation along last dimension."""
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# ----------------------------------------------------------------------------
# Core O(N) Δ-rule kernel (unchanged numerics, chunk-wise algorithm)
# ----------------------------------------------------------------------------

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B H L Dk)
    k: torch.Tensor,  # (B H L Dk)
    v: torch.Tensor,  # (B H L Dv)
    beta: torch.Tensor,  # (B H L)
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

    tri_inc = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_inc, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=q.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    tri_exc = torch.triu(tri_inc, 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_exc, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

# ----------------------------------------------------------------------------
# Multi-Scale Depth-wise Conv1d **with spiking kernel & channel mixing**
# ----------------------------------------------------------------------------

class MultiScaleDepthwiseConv1d(nn.Module):
    """Depth-wise causal convolutions at multiple scales + spiking mixers."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 15, 31),
        neuron_type: str = "sdn",
        surrogate_function=piecewise_quadratic_surrogate(),
    ) -> None:
        super().__init__()
        assert neuron_registry is not None, "Spiking neuron registry unavailable."
        self.kernel_sizes = list(kernel_sizes)
        self.num_heads = num_heads
        self.head_dim = head_dim
        in_channels = num_heads * head_dim

        # depth-wise convs ---------------------------------------------------
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=k,
                    groups=in_channels,
                    bias=False,
                )
                for k in self.kernel_sizes
            ]
        )
        for conv in self.convs:
            nn.init.normal_(conv.weight, std=0.02)

        # spiking kernel mixer ----------------------------------------------
        mix_in = len(self.kernel_sizes) * head_dim
        self.kernel_mix = SpikingLinear(
            nn.Linear(mix_in, head_dim, bias=False), neuron_type, surrogate_function
        )

        # spiking channel mixer (1×1) ---------------------------------------
        total_hidden = num_heads * head_dim
        self.channel_mixer = SpikingLinear(
            nn.Linear(total_hidden, total_hidden, bias=False), neuron_type, surrogate_function
        )

    def forward(self, x: torch.Tensor):  # x: (B L H D)
        b, L, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")  # (B,C,L)

        conv_outs: List[torch.Tensor] = []
        for k, conv in zip(self.kernel_sizes, self.convs):
            x_pad = F.pad(x_f, (k - 1, 0))
            conv_outs.append(conv(x_pad))  # (B,C,L)
        y = torch.cat(conv_outs, dim=1)  # (B, C*|K|, L)

        y = rearrange(
            y,
            "b (h kd) l -> b l h kd",
            h=self.num_heads,
            kd=len(self.kernel_sizes) * d,
        )

        # --- spiking kernel mix -------------------------------------------
        y, spikes_mix = self.kernel_mix(y)  # (B,L,H,D)

        # --- spiking channel (head) mix -----------------------------------
        y_flat = rearrange(y, "b l h d -> b l (h d)")
        y_flat, spikes_ch = self.channel_mixer(y_flat)  # (B,L,H*D)
        y = rearrange(y_flat, "b l (h d) -> b l h d", h=h)

        # gather spikes -----------------------------------------------------
        layer_spikes = torch.cat([spikes_mix.flatten(), spikes_ch.flatten()])
        return y, layer_spikes

# ----------------------------------------------------------------------------
# DeltaNet (fully-spiking)
# ----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache

class DeltaNet(nn.Module):
    """Fully-Spiking DeltaNet layer (global Δ-rule + local conv + gated fusion)."""

    def __init__(
        self,
        *,
        mode: str = "mscmix_spike",
        d_model: Optional[int] = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: Optional[int] = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        ms_kernel_sizes: Tuple[int, int, int] = (3, 15, 31),
        fusion_hidden_mult: int = 2,
        gate_bias_init: float = 1.0,
        neuron_type: str = "sdn",
        surrogate_function=piecewise_quadratic_surrogate(),
        **kwargs,
    ):  # noqa: D401
        super().__init__()
        assert neuron_registry is not None, "Spiking neuron registry unavailable."

        # ---------------- basic params ------------------------------------
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # ---------------- dimensions -------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("Key / Value dims must divide num_heads")

        # ---------------- spiking projections -----------------------------
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, num_heads, bias=False), neuron_type, surrogate_function)

        # ---------------- optional short conv branch ----------------------
        if not self.use_short_conv:
            raise UserWarning("ShortConvolution is mandatory in this variant.")
        act = "silu" if qk_activation == "silu" else None
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)

        # ---------------- local multi-scale conv branch -------------------
        self.local_conv = MultiScaleDepthwiseConv1d(
            num_heads=num_heads,
            head_dim=self.head_v_dim,
            kernel_sizes=ms_kernel_sizes,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
        )

        # ---------------- spiking fusion gate MLP -------------------------
        gate_in_dim = hidden_size + 3 * num_heads  # hidden + norms per head
        fusion_hidden = hidden_size * fusion_hidden_mult
        self.fusion_fc1 = SpikingLinear(nn.Linear(gate_in_dim, fusion_hidden, bias=True), neuron_type, surrogate_function)
        self.fusion_fc2 = SpikingLinear(nn.Linear(fusion_hidden, num_heads * 3, bias=True), neuron_type, surrogate_function)
        # identity bias so value path preferred early
        with torch.no_grad():
            self.fusion_fc2.linear.bias[num_heads * 2 :] = gate_bias_init  # type: ignore[index]

        # ---------------- output normalisation / projection ---------------
        if self.use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=1e-5)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=1e-5)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

        # container for latest spike tensor --------------------------------
        self._spikes: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B,L,D)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        # --------------------------------------------------------------
        # optional unpadding for variable-length batches
        # --------------------------------------------------------------
        orig_batch_size, orig_seq_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -orig_seq_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # --------------------------------------------------------------
        # Spiking projections + short conv
        # --------------------------------------------------------------
        spikes_collector: List[torch.Tensor] = []
        q_lin, s_q = self.q_proj(hidden_states)
        k_lin, s_k = self.k_proj(hidden_states)
        v_lin, s_v = self.v_proj(hidden_states)
        spikes_collector.extend([s_q, s_k, s_v])

        q_lin, _ = self.q_conv1d(q_lin, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        k_lin, _ = self.k_conv1d(k_lin, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        v_lin, _ = self.v_conv1d(v_lin, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)

        # --------------------------------------------------------------
        # reshape heads
        # --------------------------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_lin, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v_lin, "b l (h d) -> b l h d", d=self.head_v_dim)

        # --------------------------------------------------------------
        # activations & norms
        # --------------------------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # --------------------------------------------------------------
        # beta gate (spiking)
        # --------------------------------------------------------------
        if self.use_beta:
            beta_raw, s_beta = self.b_proj(hidden_states)
            spikes_collector.append(s_beta)
            beta = beta_raw.sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --------------------------------------------------------------
        # global Δ-rule memory (O(N))
        # --------------------------------------------------------------
        q_g = rearrange(q, "b l h d -> b h l d")
        k_g = rearrange(k, "b l h d -> b h l d")
        v_g = rearrange(v_direct, "b l h d -> b h l d")
        beta_g = rearrange(beta, "b l h -> b h l")
        delta_out, _ = delta_rule_chunkwise(q_g, k_g, v_g, beta_g)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # --------------------------------------------------------------
        # local multi-scale conv branch (spiking mixers inside)
        # --------------------------------------------------------------
        ms_out, s_local = self.local_conv(v_direct)
        spikes_collector.append(s_local)

        # --------------------------------------------------------------
        # branch norms (per-token, per-head)
        # --------------------------------------------------------------
        def _bn(x: torch.Tensor):
            return x.abs().mean(dim=-1)

        norms = [_bn(t) for t in (ms_out, delta_out, v_direct)]  # each: (B, L, H)

        # Use current (possibly unpadded) sequence length for all ops
        gate_feats = torch.cat(
            [hidden_states, norms[0], norms[1], norms[2]],
            dim=-1,
        )  # (B, L_eff, hidden + 3*H)

        # --------------------------------------------------------------
        # spiking gate MLP
        # --------------------------------------------------------------
        h_gate, s_fc1 = self.fusion_fc1(gate_feats)
        spikes_collector.append(s_fc1)
        h_gate = F.gelu(h_gate)
        logits, s_fc2 = self.fusion_fc2(h_gate)
        spikes_collector.append(s_fc2)
        logits = rearrange(logits, "b l (h c) -> b l h c", h=self.num_heads, c=3)
        w = torch.softmax(logits, dim=-1)

        o = w[..., 0:1] * ms_out + w[..., 1:2] * delta_out + w[..., 2:3] * v_direct

        # --------------------------------------------------------------
        # optional gated RMSNorm
        # --------------------------------------------------------------
        if self.use_gate:
            g_vec, s_g = self.g_proj(hidden_states)
            spikes_collector.append(s_g)
            g_vec = rearrange(g_vec, "b l (h d) -> b l h d", d=self.head_v_dim)
            o = self.o_norm(o, g_vec)
        else:
            o = self.o_norm(o)

        # --------------------------------------------------------------
        # output projection (spiking)
        # --------------------------------------------------------------
        o = rearrange(o, "b l h d -> b l (h d)")
        o, s_out = self.o_proj(o)
        spikes_collector.append(s_out)

        # --------------------------------------------------------------
        # re-pad if necessary
        # --------------------------------------------------------------
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, orig_batch_size, orig_seq_len)

        # --------------------------------------------------------------
        # store spike tensor for outer module
        # --------------------------------------------------------------
        if spikes_collector:
            self._spikes = torch.cat([s.flatten() for s in spikes_collector])
        else:
            self._spikes = None

        return o, None, past_key_values

# ----------------------------------------------------------------------------
# Simple fully-spiking MLP
# ----------------------------------------------------------------------------
class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1, neuron_type: str = "sdn"):
        super().__init__()
        assert neuron_registry is not None, "Spiking neuron registry unavailable."
        surrogate = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate)

    def forward(self, x: torch.Tensor):
        spikes_list: List[torch.Tensor] = []
        x, s1 = self.fc1(x)
        spikes_list.append(s1)
        x = self.dropout(F.gelu(x))
        x, s2 = self.fc2(x)
        spikes_list.append(s2)
        spikes = torch.cat([s.flatten() for s in spikes_list])
        return x, spikes

# ----------------------------------------------------------------------------
# SpikingFLABlock – final container that returns **final_spikes**
# ----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_spiking_mlp: bool,
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
        if use_spiking_mlp:
            d_ffn = d_model * d_ffn_mult
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        else:
            raise ValueError("use_spiking_mlp must be True – fully spiking architecture required.")
        self.dropout2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):  # -> (output, reg_loss, final_spikes)
        spike_bucket: List[torch.Tensor] = []
        residual = x
        if self.prenorm:
            x = self.norm1(x)

        x_attn, reg_loss, _ = self.attn(x, **kwargs)  # attn stores spikes internally
        x = residual + self.dropout1(x_attn)

        # collect attention spikes
        if getattr(self.attn, "_spikes", None) is not None:
            spike_bucket.append(self.attn._spikes)  # type: ignore[arg-type]

        residual = x
        if self.prenorm:
            x = self.norm2(x)

        x_mlp, mlp_spikes = self.mlp(x)
        x = residual + self.dropout2(x_mlp)
        spike_bucket.append(mlp_spikes)

        final_spikes = torch.cat([s.flatten() for s in spike_bucket]) if spike_bucket else None
        return x, reg_loss, final_spikes
