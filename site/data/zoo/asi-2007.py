# -*- coding: utf-8 -*-
"""
spiking_fla_resgate_slinear – Reserve–Gated DeltaNet with *full* SpikingLinear usage
===================================================================================
This file rewrites the previous *delta_net_ms_resgate* implementation so that **all
linear layers are preceded by a spiking neuron** via the *SpikingLinear* wrapper
(given by the assignment).  Every Linear projection, gate MLP layer, output
projection and MLP feed-forward now employs spike activations.  All resulting
binary spike tensors are aggregated and returned as the **final_spikes** output
of `SpikingFLABlock.forward` so that downstream training loops can apply custom
regularisation (e.g. firing-rate penalties) or sparsity measurements.

Core guarantees
---------------
• Public APIs remain unchanged (`DeltaNet`, `SpikingFLABlock`).
• All shapes & causal/O(N) complexity are preserved – only Linear→SpikingLinear
  substitutions plus minimal glue code were introduced.
• `einops.rearrange` is used for every reshape.
• Batch-size and sequence-length agnosticism are preserved (no hard-coded dims).
• *No* existing import is removed; additional lightweight utilities are defined
  locally.

Implementation notes
--------------------
1. **SLWrapper** – thin adaptor that embeds the mandated `SpikingLinear` but
   returns only the *dense* output so that the surrounding code stays intact.
   The binary spikes are cached into `last_spikes` for later aggregation.
2. **DeltaNet** – every former `nn.Linear` is now an `SLWrapper`.  The original
   `fusion_gate_mlp` (`nn.Sequential`) is replaced by two explicit linear layers
   (`fusion_gate_fc1`, `fusion_gate_fc2`) so that both are spiking.
3. **SpikingMLP** – both internal Linear layers are changed to `SLWrapper`.
4. **SpikingFLABlock** – collects references to every `SLWrapper` and
   concatenates their `last_spikes` into the required `final_spikes` tensor
   (flat 1-D).  The forward method now returns `(x, reg_loss, final_spikes)`.

The rest of the numerical pipeline (chunk-wise Δ-rule, depth-wise FIR, reserve
routing) is *identical* to the validated implementation, guaranteeing that the
behavioural upgrades come solely from the spike-driven linear processing.
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
except ImportError:
    print("Warning: Spiking neuron modules (src.models.spike) not found.\n" "SpikingMLP will not be available.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None

# -----------------------------------------------------------------------------
# Existing mandatory SpikingLinear (unchanged) ---------------------------------
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear layer preceded by a spiking neuron (provided externally)."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)  # type: ignore[index]
        self.linear = linear_layer

    def forward(self, x: torch.Tensor):  # -> Tuple[dense_out, spikes]
        x_t = x.transpose(-1, -2)  # (B,L,D) -> (B,D,L)
        spikes_t = self.neuron(x_t)
        spikes = spikes_t.transpose(-1, -2)  # (B,D,L) -> (B,L,D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# Thin wrapper that *looks* like nn.Linear to the rest of the code -------------
# -----------------------------------------------------------------------------
class SLWrapper(nn.Module):
    """Adapter around *SpikingLinear* that caches spikes but returns *only* the dense output.

    The cached spikes are exposed through the attribute ``last_spikes`` so that
    higher-level blocks can aggregate them into a single tensor at the end of
    the forward pass.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *,
                 neuron_type: str = "sdn", surrogate_function=None):
        super().__init__()
        if surrogate_function is None:
            # fallback to global variable, or default
            surrogate_function = piecewise_quadratic_surrogate() if "piecewise_quadratic_surrogate" in globals() and callable(piecewise_quadratic_surrogate) else None
        base_linear = nn.Linear(in_features, out_features, bias=bias)
        self.slinear = SpikingLinear(base_linear, neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.last_spikes: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, spikes = self.slinear(x)
        self.last_spikes = spikes
        return out

# -----------------------------------------------------------------------------
# Helper activations & utilities ----------------------------------------------
# -----------------------------------------------------------------------------

def elu_p1(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise Δ-rule kernel (identical numerics – O(N) ) ------------------------
# -----------------------------------------------------------------------------

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,Dk)
    k: torch.Tensor,  # (B,H,L,Dk)
    v: torch.Tensor,  # (B,H,L,Dv)
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

    q, k, v, k_beta = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size), (q, k, v, k_beta))

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

    for idx in range(L_pad // chunk_size):
        q_i = q[:, :, idx]
        k_i = k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask_future, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Multi-scale depth-wise FIR convolution --------------------------------------
# -----------------------------------------------------------------------------
class DepthwiseMultiScaleFIR(nn.Module):
    """Parallel depth-wise causal convolutions with identity initialisation."""

    def __init__(self, *, num_heads: int, head_dim: int, kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31), init_std: float = 0.02):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        channels = num_heads * head_dim
        self.filters = nn.ParameterList()
        for k in kernel_sizes:
            w = torch.zeros(channels, 1, k)
            with torch.no_grad():
                w[:, 0, -1] = 1.0
                w.add_(init_std * torch.randn_like(w))
            self.filters.append(nn.Parameter(w))
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # x: (B,L,H,D)
        b, L, h, d = x.shape
        x_f = rearrange(x, "b l h d -> b (h d) l")
        outs: List[torch.Tensor] = []
        for k_size, filt in zip(self.kernel_sizes, self.filters):
            x_pad = F.pad(x_f, (k_size - 1, 0))
            y = F.conv1d(x_pad, weight=filt, groups=h * d)
            outs.append(rearrange(y, "b (h d) l -> b l h d", h=h))
        return outs

# -----------------------------------------------------------------------------
# Optional typing helper -------------------------------------------------------
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache  # noqa: F401 – only for type hints

# -----------------------------------------------------------------------------
# DeltaNet – unchanged numerics, *all* linear layers now Spiking ---------------
# -----------------------------------------------------------------------------
class DeltaNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Reserve-Gated DeltaNet with comprehensive SpikingLinear adoption."""

    def __init__(
        self,
        *,
        mode: str = "ms_resgate_spike",
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
        ms_kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31),
        gate_hidden_mult: int = 2,
        delta_floor: float = 0.05,
        temp_init: float = 1.0,
        neuron_type: str = "sdn",
        **kwargs: "Dict",
    ) -> None:
        super().__init__()
        # The following global check avoids NameError due to missing surrogate_function
        if neuron_registry is None or ("piecewise_quadratic_surrogate" not in globals() and 'piecewise_quadratic_surrogate' not in locals()):  # type: ignore[name-defined]
            raise ImportError("Spiking neuron modules are not available.")

        surrogate_fn = piecewise_quadratic_surrogate() if "piecewise_quadratic_surrogate" in globals() and callable(piecewise_quadratic_surrogate) else None
        if d_model is not None:
            hidden_size = d_model

        # store basic config --------------------------------------------------
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx or 0
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.delta_floor = float(delta_floor)

        # dimensions ----------------------------------------------------------
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        assert self.key_dim % num_heads == 0 and self.value_dim % num_heads == 0, "Dims must divide num_heads"

        # projections (now spiking) ------------------------------------------
        self.q_proj = SLWrapper(hidden_size, self.key_dim, bias=False, neuron_type=neuron_type, surrogate_function=surrogate_fn)
        self.k_proj = SLWrapper(hidden_size, self.key_dim, bias=False, neuron_type=neuron_type, surrogate_function=surrogate_fn)
        self.v_proj = SLWrapper(hidden_size, self.value_dim, bias=False, neuron_type=neuron_type, surrogate_function=surrogate_fn)
        if use_beta:
            self.b_proj = SLWrapper(hidden_size, num_heads, bias=False, neuron_type=neuron_type, surrogate_function=surrogate_fn)

        # short convolutions --------------------------------------------------
        if use_short_conv:
            act = "silu" if qk_activation == "silu" else None
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation=act, bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation="silu", bias=conv_bias)
        else:
            raise UserWarning("ShortConvolution is mandatory for DeltaNet variants.")

        # multi-scale FIR ------------------------------------------------------
        self.local_fir = DepthwiseMultiScaleFIR(num_heads=num_heads, head_dim=self.head_v_dim, kernel_sizes=ms_kernel_sizes)
        self.num_scales = len(ms_kernel_sizes)

        # reserve gate – individual spiking linears ---------------------------
        self.n_local_paths = self.num_scales + 1  # conv branches + direct value
        self.n_paths = self.n_local_paths + 1    # + delta
        gate_in_dim = hidden_size + num_heads * self.n_paths
        hidden_gate_dim = hidden_size * gate_hidden_mult

        # two-layer MLP, both layers spiking
        self.fusion_fc1 = SLWrapper(gate_in_dim, hidden_gate_dim, bias=True, neuron_type=neuron_type, surrogate_function=surrogate_fn)
        self.fusion_fc2 = SLWrapper(hidden_gate_dim, num_heads * self.n_paths, bias=True, neuron_type=neuron_type, surrogate_function=surrogate_fn)

        # temperature parameters ---------------------------------------------
        self.gate_log_temp = nn.Parameter(torch.ones(num_heads) * temp_init)

        # statistic scaling parameter per path
        self.alpha_stat = nn.Parameter(torch.ones(self.n_paths) * 0.1)

        # output norm / projection -------------------------------------------
        if use_gate:
            self.g_proj = SLWrapper(hidden_size, self.value_dim, bias=False, neuron_type=neuron_type, surrogate_function=surrogate_fn)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = SLWrapper(self.value_dim, hidden_size, bias=False, neuron_type=neuron_type, surrogate_function=surrogate_fn)

    # ----------------------------------------------------------------------
    # Forward --------------------------------------------------------------
    # ----------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,  # type: ignore[name-defined]
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,  # kept for API
        **kwargs: "Dict",
    ) -> Tuple[torch.Tensor, None, Optional["Cache"]]:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "attention_mask must be (batch, seq_len)"
        B_orig, T_orig, _ = hidden_states.shape
        indices = None
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -T_orig:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # cached conv state --------------------------------------------------
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        # projections + short conv ------------------------------------------
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_lin, conv_q = self.q_conv1d(q_lin, cache=conv_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k_lin, conv_k = self.k_conv1d(k_lin, cache=conv_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v_lin, conv_v = self.v_conv1d(v_lin, cache=conv_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        # head split ---------------------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b l h d", h=self.num_heads)

        # activations / norms ----------------------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = F.relu(q), F.relu(k)
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError
        if self.qk_norm == "sum":
            q, k = sum_norm(q), sum_norm(k)

        # beta gate ---------------------------------------------------------
        if self.use_beta:
            beta = self.b_proj(hidden_states)
            beta = beta.sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Δ-rule global memory ---------------------------------------------
        q_d = rearrange(q, "b l h d -> b h l d")
        k_d = rearrange(k, "b l h d -> b h l d")
        v_d = rearrange(v, "b l h d -> b h l d")
        beta_d = rearrange(beta, "b l h -> b h l")
        delta_out, recurrent_state_new = delta_rule_chunkwise(q_d, k_d, v_d, beta_d)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # local FIR branches ----------------------------------------------
        conv_branches = self.local_fir(v)  # list len=num_scales
        value_out = v
        branches: List[torch.Tensor] = conv_branches + [value_out, delta_out]

        branch_stats = torch.stack([b_.abs().mean(dim=-1) for b_ in branches], dim=-1)  # (B,L,H,P)

        # gate MLP ----------------------------------------------------------
        gate_input = torch.cat([hidden_states, rearrange(branch_stats, "b l h p -> b l (h p)")], dim=-1)
        hidden_gate = self.fusion_fc1(gate_input)
        hidden_gate = F.gelu(hidden_gate)
        gate_logits = self.fusion_fc2(hidden_gate)
        gate_logits = rearrange(gate_logits, "b l (h p) -> b l h p", h=self.num_heads, p=self.n_paths)
        gate_logits = gate_logits + self.alpha_stat.view(1, 1, 1, self.n_paths) * branch_stats
        temp = F.softplus(self.gate_log_temp) + 1e-4
        gate_logits = gate_logits * temp.view(1, 1, self.num_heads, 1)

        logits_local = gate_logits[..., : self.n_local_paths]
        logit_delta = gate_logits[..., self.n_local_paths]

        w_delta = torch.sigmoid(logit_delta) * (1.0 - self.delta_floor) + self.delta_floor
        w_delta_exp = w_delta.unsqueeze(-1)
        w_local = F.softmax(logits_local, dim=-1) * (1.0 - w_delta_exp)
        weights = torch.cat([w_local, w_delta_exp], dim=-1)  # (B,L,H,P)

        stacked = torch.stack(branches, dim=-2)  # (B,L,H,P,D)
        out = (weights.unsqueeze(-1) * stacked).sum(dim=-2)

        # cache update ------------------------------------------------------
        if past_key_values is not None and use_cache:
            past_key_values.update(
                recurrent_state=recurrent_state_new,
                conv_state=(conv_q, conv_k, conv_v),
                layer_idx=self.layer_idx,
                offset=T_orig,
            )

        # output projection -------------------------------------------------
        if self.use_gate:
            g_vec = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            out = self.o_norm(out, g_vec)
        else:
            out = self.o_norm(out)
        out = rearrange(out, "b l h d -> b l (h d)")
        out = self.o_proj(out)

        # re-pad ------------------------------------------------------------
        if attention_mask is not None:
            out = pad_input(out.squeeze(0), indices, B_orig, T_orig)

        return out, None, past_key_values

# -----------------------------------------------------------------------------
# Spiking MLP ------------------------------------------------------------------
# -----------------------------------------------------------------------------
class SpikingMLP(nn.Module):
    """Two-layer MLP with spike activations before *both* linear transformations."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1, neuron_type: str = "sdn"):
        super().__init__()
        if neuron_registry is None or ("piecewise_quadratic_surrogate" not in globals() and 'piecewise_quadratic_surrogate' not in locals()):
            raise ImportError("Spiking neuron modules are not available.")
        surrogate_fn = piecewise_quadratic_surrogate() if "piecewise_quadratic_surrogate" in globals() and callable(piecewise_quadratic_surrogate) else None
        self.fc1 = SLWrapper(d_model, d_ffn, bias=True, neuron_type=neuron_type, surrogate_function=surrogate_fn)
        self.neuron_type = neuron_type
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SLWrapper(d_ffn, d_model, bias=True, neuron_type=neuron_type, surrogate_function=surrogate_fn)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Concatenate spikes from both internal layers
        spikes = torch.cat([
            self.fc1.last_spikes.flatten(),
            self.fc2.last_spikes.flatten(),
        ])
        return x, spikes

# -----------------------------------------------------------------------------
# High-level FLA block with spike aggregation ---------------------------------
# -----------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    """FLA block (Attn + MLP) with global spike aggregation."""

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
            print(f"INFO: Layer uses SpikingMLP with {neuron_type} neuron.")
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        else:
            raise ValueError("SpikingFLABlock requires use_spiking_mlp=True for full spiking compliance.")
        self.dropout2 = nn.Dropout(dropout)

        # Collect all SLWrapper references for spike aggregation -----------
        self._spike_modules: List[SLWrapper] = []
        for m in self.modules():
            if isinstance(m, SLWrapper):
                self._spike_modules.append(m)

    def forward(self, x: torch.Tensor, **kwargs):
        # Reset spike caches ------------------------------------------------
        for m in self._spike_modules:
            m.last_spikes = None

        # Attention --------------------------------------------------------
        x_res = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn, reg_loss, _ = self.attn(x, **kwargs)
        x = x_res + self.dropout1(x_attn)

        # MLP -------------------------------------------------------------
        x_res = x
        if self.prenorm:
            x = self.norm2(x)
        x_mlp, mlp_spikes = self.mlp(x)
        x = x_res + self.dropout2(x_mlp)

        # Aggregate all spikes --------------------------------------------
        spike_tensors: List[torch.Tensor] = []
        for m in self._spike_modules:
            if m.last_spikes is not None:
                spike_tensors.append(m.last_spikes.flatten())
        # Ensure at least mlp spikes are included (fallback)
        spike_tensors.append(mlp_spikes.flatten())
        final_spikes = torch.cat(spike_tensors) if spike_tensors else torch.empty(0, device=x.device)

        return x, reg_loss, final_spikes
