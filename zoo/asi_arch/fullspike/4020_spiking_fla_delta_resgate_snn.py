# -*- coding: utf-8 -*-
"""
spiking_fla_delta_resgate_snn
=============================
This file provides an **SNN-first** re-implementation of the original
`delta_net_ms_resgate` layer where **every Linear layer input is now a
binary spike tensor**.  Each dense projection (Q, K, V, β, gate MLP,
output projection, etc.) has been replaced by a `SpikingLinear` wrapper
that inserts a spiking neuron *in front of* the weight matrix as
required by the mission brief.

Key implementation notes
------------------------
1. **SpikingLinear Usage** – All `nn.Linear` modules are wrapped by our
   mandatory `SpikingLinear` class.  This guarantees the layer consumes
   binary spikes and returns both the analogue output and the spike
   tensor itself.
2. **Spike Collection** – A `spike_collector: list` can be passed via
   `**kwargs` to accumulate *all* spike tensors generated during the
   forward pass.  This is exploited by `SpikingFLABlock` to build the
   final, flattened `final_spikes` tensor required by the evaluation
   harness.
3. **Interface Stability** – Public class names (`DeltaNet`,
   `SpikingFLABlock`) and forward signatures are **unchanged**.  The new
   optional `spike_collector` kwarg is fully backward compatible thanks
   to `**kwargs`.
4. **Complexity & Causality** – All algorithmic components (depth-wise
   FIR, chunkwise Δ-rule) remain O(N).  No quadratic attention
   operations were introduced.
5. **einops everywhere** – Every reshape uses `einops.rearrange` to
   guarantee shape-safety and batch agnosticism.
6. **No @torch.compile decorators** – Kept disabled for maximum
   robustness.

The surrounding MLP helper (`SpikingMLP`) has been updated to follow the
same conventions, and `SpikingFLABlock` now concatenates *all* collected
spikes before returning.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import ShortConvolution, RMSNorm, FusedRMSNormGated
from fla.modules.l2norm import l2norm

try:
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except ImportError:  # pragma: no cover
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    print("SpikingMLP/SpikingLinear will not be available.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None

# -----------------------------------------------------------------------------
# Mandatory SpikingLinear wrapper (MUST stay unmodified)
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear layer preceded by a spiking neuron."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)  # type: ignore[index]
        self.linear = linear_layer

    def forward(self, x: torch.Tensor):  # -> (output, spikes)
        spikes_t = self.neuron(x.transpose(-1, -2))  # (B, D, L)
        spikes = spikes_t.transpose(-1, -2)  # (B, L, D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# Helper activations / norms
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:  # positive ELU
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # L1 normalise last dim
    s = x.sum(-1, keepdim=True).clamp(min=1e-6)
    return (x / s).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (unchanged, still O(N))
# -----------------------------------------------------------------------------
# pylint: disable=too-many-locals,too-many-statements

def delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 32,
):  # noqa: C901 – complex but battle-tested
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # Normalise & gate ---------------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape into blocks ------------------------------------------------------
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    mask_future = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for blk in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, blk], k[:, :, blk]
        local = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask_future, 0)
        u_i = u[:, :, blk] - w[:, :, blk] @ S
        o[:, :, blk] = q_i @ S + local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Depth-wise causal FIR – same as original (identity initialisation)
# -----------------------------------------------------------------------------
class DepthwiseMultiScaleFIR(nn.Module):
    """Parallel depth-wise causal convolutions with identity initialisation."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31),
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        total_ch = num_heads * head_dim
        self.filters = nn.ParameterList()
        for k in kernel_sizes:
            w = torch.zeros(total_ch, 1, k)
            with torch.no_grad():
                w[:, 0, -1] = 1.0  # identity at last tap
                w.add_(init_std * torch.randn_like(w))
            self.filters.append(nn.Parameter(w))
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_ch = rearrange(x, "b l h d -> b (h d) l")
        outs: List[torch.Tensor] = []
        for k, w in zip(self.kernel_sizes, self.filters):
            y = F.conv1d(F.pad(x_ch, (k - 1, 0)), w, groups=h * d)
            outs.append(rearrange(y, "b (h d) l -> b l h d", h=h))
        return outs

# -----------------------------------------------------------------------------
# Optional cache typing helper
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401

# -----------------------------------------------------------------------------
# Main SNN DeltaNet layer
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DeltaNet with Reserve Gating – **every Linear is now spiking**."""

    def __init__(
        self,
        *,
        mode: str = "ms_resgate_snn",
        d_model: int | None = None,
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
        layer_idx: int | None = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        # new hyper-params ---------------------------------------------------
        ms_kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31),
        gate_hidden_mult: int = 2,
        delta_floor: float = 0.05,
        temp_init: float = 1.0,
        neuron_type: str = "sdn",
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron registry not found – cannot build SpikingLinear layers.")
        if d_model is not None:
            hidden_size = d_model

        # Basic fields --------------------------------------------------------
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.delta_floor = float(delta_floor)
        self.neuron_type = neuron_type
        self.surr_fn = piecewise_quadratic_surrogate()

        # Dimensions ----------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "Dims must divide heads"
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Spiking projections --------------------------------------------------
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, self.surr_fn)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, self.surr_fn)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, self.surr_fn)
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, num_heads, bias=False), neuron_type, self.surr_fn)

        # Short convolution wrappers -----------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")

        # Multi-scale FIR ------------------------------------------------------
        self.local_fir = DepthwiseMultiScaleFIR(num_heads=num_heads, head_dim=self.head_v_dim, kernel_sizes=ms_kernel_sizes)
        self.num_scales = len(ms_kernel_sizes)

        # Reserve gate parameters --------------------------------------------
        self.n_local_paths = self.num_scales + 1  # conv branches + direct value
        self.n_paths = self.n_local_paths + 1     # + delta path

        gate_in_dim = hidden_size + num_heads * self.n_paths
        self.gate_fc1 = SpikingLinear(
            nn.Linear(gate_in_dim, hidden_size * gate_hidden_mult, bias=True), neuron_type, self.surr_fn
        )
        self.gate_fc2 = SpikingLinear(
            nn.Linear(hidden_size * gate_hidden_mult, num_heads * self.n_paths, bias=True), neuron_type, self.surr_fn
        )

        # Temperature & statistic scale --------------------------------------
        self.gate_log_temp = nn.Parameter(torch.ones(num_heads) * temp_init)
        self.alpha_stat = nn.Parameter(torch.ones(self.n_paths) * 0.1)
        with torch.no_grad():  # encourage Δ-path early
            bias_view = self.gate_fc2.linear.bias.view(num_heads, self.n_paths)
            bias_view[:, self.n_local_paths] += 2.0

        # Output projection & norm -------------------------------------------
        if self.use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, self.surr_fn)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, self.surr_fn)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        *,
        use_cache: bool = False,
        output_attentions: bool = False,
        spike_collector: Optional[List[torch.Tensor]] = None,
        **kwargs: Dict,
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:  # type: ignore[name-defined]
        if spike_collector is None:
            spike_collector = []

        # --- optional unpadding --------------------------------------------
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (B, L)"
        B_orig, L_orig, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -L_orig:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b l d -> (b l) d"), indices).unsqueeze(0)

        # --- spike projections --------------------------------------------
        q_raw, s_q = self.q_proj(hidden_states)
        k_raw, s_k = self.k_proj(hidden_states)
        v_raw, s_v = self.v_proj(hidden_states)
        spike_collector.extend([s_q, s_k, s_v])
        if self.use_beta:
            beta_raw, s_b = self.b_proj(hidden_states)
            spike_collector.append(s_b)
        else:
            beta_raw = torch.ones(*hidden_states.shape[:2], self.num_heads, device=hidden_states.device)

        # --- short convs ----------------------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q_lin, conv_q = self.q_conv1d(q_raw, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(k_raw, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(v_raw, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # --- head split & activation ---------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # --- β for Δ-rule ----------------------------------------------------
        beta = beta_raw.sigmoid() if self.use_beta else beta_raw  # (B,L,H)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --- Δ-rule global memory -------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, rec_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # --- local FIR branches + direct value ------------------------------
        conv_branches = self.local_fir(v)  # list len = num_scales
        value_out = v

        branches: List[torch.Tensor] = conv_branches + [value_out, delta_out]  # locals..., value, delta

        # --- branch statistics ---------------------------------------------
        branch_stats = torch.stack([b_.abs().mean(dim=-1) for b_ in branches], dim=-1)  # (B,L,H,P)

        # --- fusion gate ----------------------------------------------------
        gate_input = torch.cat(
            [hidden_states, rearrange(branch_stats, "b l h p -> b l (h p)")], dim=-1
        )  # (B,L,D + H*P)

        x_g1, s_g1 = self.gate_fc1(gate_input)
        spike_collector.append(s_g1)
        x_g1_act = F.gelu(x_g1)
        gate_logits_raw, s_g2 = self.gate_fc2(x_g1_act)
        spike_collector.append(s_g2)
        gate_logits = rearrange(gate_logits_raw, "b l (h p) -> b l h p", h=self.num_heads, p=self.n_paths)
        gate_logits = gate_logits + self.alpha_stat.view(1, 1, 1, self.n_paths) * branch_stats
        temp = F.softplus(self.gate_log_temp) + 1e-4
        gate_logits = gate_logits * temp.view(1, 1, self.num_heads, 1)

        # --- reserve gating --------------------------------------------------
        logits_local = gate_logits[..., : self.n_local_paths]
        logit_delta = gate_logits[..., self.n_local_paths]
        w_delta = torch.sigmoid(logit_delta) * (1.0 - self.delta_floor) + self.delta_floor  # (B,L,H)
        w_delta_exp = w_delta.unsqueeze(-1)
        w_local = F.softmax(logits_local, dim=-1) * (1.0 - w_delta_exp)
        weights = torch.cat([w_local, w_delta_exp], dim=-1)  # (B,L,H,P)

        # --- fuse outputs ----------------------------------------------------
        stacked = torch.stack(branches, dim=-2)  # (B,L,H,P,D)
        out = (weights.unsqueeze(-1) * stacked).sum(dim=-2)  # (B,L,H,D)

        # --- cache update ----------------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=rec_state_new,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=L_orig,
            )

        # --- output projection ----------------------------------------------
        if self.use_gate:
            g_vec_raw, s_gproj = self.g_proj(hidden_states)
            spike_collector.append(s_gproj)
            g_vec = rearrange(g_vec_raw, "b l (h d) -> b l h d", h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)

        out = rearrange(out, "b l h d -> b l (h d)")
        out_raw, s_oproj = self.o_proj(out)
        spike_collector.append(s_oproj)

        # --- re-pad if needed -----------------------------------------------
        if attention_mask is not None:
            out_raw = pad_input(out_raw.squeeze(0), indices, B_orig, L_orig)

        return out_raw, None, past_key_values

# -----------------------------------------------------------------------------
# SNN-aware MLP helper (two SpikingLinear layers)
# -----------------------------------------------------------------------------
class SpikingMLP(nn.Module):
    """Position-wise feedforward MLP using spike-driven Linear layers."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
    ) -> None:
        super().__init__()
        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available.")
        surr_fn = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surr_fn)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surr_fn)

    def forward(self, x: torch.Tensor, *, spike_collector: List[torch.Tensor]):  # noqa: D401
        x1, s1 = self.fc1(x)
        spike_collector.append(s1)
        x1_act = F.gelu(x1)
        x1_act = self.dropout(x1_act)
        x2, s2 = self.fc2(x1_act)
        spike_collector.append(s2)
        return x2

# -----------------------------------------------------------------------------
# Full SNN FLA Block (attn + MLP)
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """FLA block composed of (SNN) DeltaNet attention and Spiking MLP."""

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
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            print(f"INFO: SpikingFLABlock uses SpikingMLP with '{neuron_type}' neurons.")
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        else:
            raise ValueError("use_spiking_mlp must be True for full SNN compliance.")
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs):  # noqa: D401
        spike_collector: List[torch.Tensor] = []
        x_res = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn, _, _ = self.attn(x, spike_collector=spike_collector, **kwargs)
        x = x_res + self.dropout1(x_attn)

        # MLP -----------------------------------------------------------------
        x_res = x
        if self.prenorm:
            x = self.norm2(x)
        x_mlp = self.mlp(x, spike_collector=spike_collector)
        x = x_res + self.dropout2(x_mlp)

        # Concatenate all spike tensors into a single 1-D vector --------------
        if spike_collector:
            final_spikes = torch.cat([s.flatten() for s in spike_collector])
        else:
            final_spikes = torch.empty(0, device=x.device)

        return x, None, final_spikes
