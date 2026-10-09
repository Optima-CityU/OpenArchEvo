# ==========================================
# 💡 NAS Idea / Motivation:
# Idea:
# The primary innovation in this evolved `SpikingFLABlock`, specifically within the `DeltaNet` attention mechanism, is the introduction of a more direct and dedicated spike-based modulation for the `beta` (decay factor) and `gate` parameters. 
# 
# **Motivation for Change:**
# 
# 1.  **Limitations of Previous Spike Modulation (Version 2):** In the preceding version, aggregated mean spike activities (from the `q`, `k`, and `v` projections) were concatenated directly with the `hidden_states` tensor before being fed into the `b_proj` (beta projection) and `g_proj` (gate projection) `SpikingLinear` layers. While this allowed the network to *consider* spike activity, it forced the `b_proj` and `g_proj` layers to learn combined representations of both the original analog features and the spike statistics. This approach might not provide the most fine-grained or biologically plausible way for spikes to directly influence these critical attention parameters.
# 
# 2.  **Enhanced Neuromorphic Nature:** The goal is to move towards a more explicit spike-driven control. By dedicating separate `SpikingLinear` pathways for spike-based modulation, we allow the network to explicitly learn *how* spike activity should scale or modify the `beta` and `gate` values, rather than implicitly learning this within a mixed feature space. This aligns better with neuromorphic principles where local activity can dynamically regulate synaptic efficacy or gating.
# 
# 3.  **Increased Modularity and Interpretability:** Decoupling the primary projection of `hidden_states` from the spike-based modulation makes the model more modular. The `b_proj` and `g_proj` can focus on extracting relevant information from the `hidden_states`, while the new `spike_beta_mod_proj` and `spike_gate_mod_proj` layers learn a specific, direct mapping from spike patterns to a modulation signal.
# 
# **Implemented Changes:**
# 
# 1.  **Dedicated Spike Modulation Projections within `DeltaNet`:**
#     *   Two new `SpikingLinear` layers, `self.spike_beta_mod_proj` and `self.spike_gate_mod_proj`, have been introduced in the `__init__` method of `DeltaNet`. These layers are specifically designed to process the `aggregated_spikes` (mean spike activities from `q_proj`, `k_proj`, `v_proj`).
#     *   `self.spike_beta_mod_proj` projects the 3-dimensional `aggregated_spikes` to `self.num_heads` dimensions, matching the output dimension of `beta`.
#     *   `self.spike_gate_mod_proj` projects the 3-dimensional `aggregated_spikes` to `self.head_v_dim` dimensions, matching the output dimension of the gate for a single head.
# 
# 2.  **Multiplicative Spike-Based Modulation in `DeltaNet.forward`:**
#     *   The `b_proj` and `g_proj` now take only the original `hidden_states` as input, simplifying their primary function.
#     *   If `self.use_spike_beta_modulation` is enabled, the `aggregated_spikes` are passed through `self.spike_beta_mod_proj` to generate `spike_beta_mod_raw`. This raw output is then passed through a `sigmoid` activation to produce `beta_modulation`, which acts as a scaling factor between 0 and 1. The original `beta` value is then multiplicatively scaled by `(1 + beta_modulation)` to allow for dynamic amplification based on spike activity.
#     *   Similarly, if `self.use_spike_gate_modulation` is enabled, `aggregated_spikes` are processed by `self.spike_gate_mod_proj` to produce `spike_gate_mod_raw`. After a `sigmoid` activation to create `gate_modulation`, the original `g` (gate) value is scaled by `(1 + gate_modulation)`.
#     *   All new spike tensors from `spike_beta_mod_proj` and `spike_gate_mod_proj` are collected into `all_spikes`.
# 
# **Expected Performance Improvements:**
# 
# *   **Enhanced Adaptivity:** By allowing spikes to directly and multiplicatively modulate crucial parameters (`beta` and `gate`), the model gains a more adaptive and dynamic control over the attention mechanism's behavior. High or low spike activity in certain parts of the network can directly influence the decay rates and gating strengths, leading to more context-sensitive information processing.
# *   **Improved Long-Range Dependency:** Finer control over `beta` can significantly impact the memory decay, potentially improving the model's ability to capture and retain long-range dependencies effectively.
# *   **Spike Efficiency:** This direct modulation encourages the network to learn meaningful correlations between spike patterns and functional parameter adjustments, potentially leading to more efficient utilization of spike information.
# *   **Cognitive Potential:** A more responsive and spike-aware attention mechanism can contribute to superior cognitive capabilities by allowing the network to dynamically adjust its focus and information integration based on the internal, event-driven (spiking) activity.
# 
# These changes further integrate neuromorphic principles into the architecture, maintaining sub-quadratic complexity and adhering to all SNN and implementation requirements, including the consistent use of `SpikingLinear` and proper spike collection for `final_spikes`.
# Spike-driven Explicit Neuromodulation
# ==========================================

"""Implements a SpikingFLABlock with refined spike activity regularization and direct spike-driven modulation of attention parameters.
"""
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ==================== SNN Related Imports and Helper Classes ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder SNN modules.")
    # neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None
# --- End of Placeholders ---

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

# ================================================================

def softmax(x):
    return F.softmax(x, dim=-1)

def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    
    padded_len = l + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        # Use non-inplace operations to ensure autograd safety
        attn_intra = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask, 0)
        o_inter = q_i @ S
        u_i = u[:, :, i] - w[:, :, i] @ S # This needs to be calculated before S is updated
        o[:, :, i] = o_inter + attn_intra @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')

    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List
if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

class DeltaNet(nn.Module):
    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', use_spike_beta_modulation: bool=True, use_spike_gate_modulation: bool=True, **kwargs) -> Dict:
        super().__init__()

        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")

        surrogate_function = piecewise_quadratic_surrogate()

        self.mode = mode
        self.qk_norm = qk_norm
        assert self.qk_norm in ['l2', 'sum']
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)

        self.use_beta = use_beta
        self.use_spike_beta_modulation = use_spike_beta_modulation
        if self.use_beta:
            # b_proj now only takes hidden_states as input
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), neuron_type, surrogate_function)
            if self.use_spike_beta_modulation:
                # New: Dedicated SpikingLinear for spike-based beta modulation
                # Input dim 3 for mean_spikes_q, mean_spikes_k, mean_spikes_v
                self.spike_beta_mod_proj = SpikingLinear(nn.Linear(3, self.num_heads, bias=True), neuron_type, surrogate_function)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation=None 
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation=None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation=None
            )

        self.use_spike_gate_modulation = use_spike_gate_modulation
        if use_gate:
            # g_proj now only takes hidden_states as input
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            if self.use_spike_gate_modulation:
                # New: Dedicated SpikingLinear for spike-based gate modulation
                # Input dim 3 for mean_spikes_q, mean_spikes_k, mean_spikes_v
                self.spike_gate_mod_proj = SpikingLinear(nn.Linear(3, self.head_v_dim * self.num_heads, bias=True), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: Unpack[Dict]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        all_spikes = {}

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "Expected attention_mask as a 0-1 matrix"

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        q, spikes_q_proj = self.q_proj(hidden_states)
        k, spikes_k_proj = self.k_proj(hidden_states)
        v, spikes_v_proj = self.v_proj(hidden_states)

        all_spikes['q_proj'] = spikes_q_proj
        all_spikes['k_proj'] = spikes_k_proj
        all_spikes['v_proj'] = spikes_v_proj

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            q, conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.qk_norm == 'l2':
            q = l2norm(q)
            k = l2norm(k)
        elif self.qk_norm == 'sum':
            q = sum_norm(q)
            k = sum_norm(k)

        # Aggregate spike activity for modulation, needed for both beta and potential gate modulation
        if (self.use_beta and self.use_spike_beta_modulation) or (self.use_gate and self.use_spike_gate_modulation):
            mean_spikes_q = spikes_q_proj.float().mean(dim=-1, keepdim=True)
            mean_spikes_k = spikes_k_proj.float().mean(dim=-1, keepdim=True)
            mean_spikes_v = spikes_v_proj.float().mean(dim=-1, keepdim=True)
            aggregated_spikes = torch.cat([mean_spikes_q, mean_spikes_k, mean_spikes_v], dim=-1)
        else:
            aggregated_spikes = None # Not used

        if self.use_beta:
            # beta_proj takes only hidden_states as input
            beta, spikes_beta = self.b_proj(hidden_states)
            beta = beta.sigmoid()
            all_spikes['b_proj'] = spikes_beta

            if self.use_spike_beta_modulation and aggregated_spikes is not None:
                spike_beta_mod_raw, spikes_beta_mod = self.spike_beta_mod_proj(aggregated_spikes)
                beta_modulation = torch.sigmoid(spike_beta_mod_raw) # Scale between 0 and 1
                beta = beta * (1 + beta_modulation) # Multiplicative modulation, factor > 1
                all_spikes['spike_beta_mod'] = spikes_beta_mod
        else:
            beta = torch.ones_like(q[..., 0])

        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state.get('recurrent_state', None) if last_state is not None else None
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')

        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')

        if past_key_values is not None:
            cache_data = {
                'recurrent_state': recurrent_state,
                'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None
            }
            if self.layer_idx is not None:
                past_key_values[self.layer_idx] = cache_data

        if self.use_gate:
            # g_proj takes only hidden_states as input
            g, spikes_g = self.g_proj(hidden_states)
            all_spikes['g_proj'] = spikes_g

            if self.use_spike_gate_modulation and aggregated_spikes is not None:
                spike_gate_mod_raw, spikes_gate_mod = self.spike_gate_mod_proj(aggregated_spikes)
                gate_modulation = torch.sigmoid(spike_gate_mod_raw)
                g = g * (1 + gate_modulation) # Multiplicative modulation, factor > 1
                all_spikes['spike_gate_mod'] = spikes_gate_mod

            g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')

        o, spikes_o = self.o_proj(o)
        all_spikes['o_proj'] = spikes_o

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, all_spikes # Still returning None for reg_loss as it's handled at block level.

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x)
        x = self.dropout(x)
        x, spike2 = self.fc2(x)
        return x, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', spike_regularization_weight: float=0.001, use_spike_beta_modulation: bool=True, use_spike_gate_modulation: bool=True, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Pass the new 'use_spike_beta_modulation' and 'use_spike_gate_modulation' parameters to DeltaNet
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, use_spike_beta_modulation=use_spike_beta_modulation, use_spike_gate_modulation=use_spike_gate_modulation, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        print(f"INFO: Layer uses SpikingMLP with {neuron_type} neuron, without additional ANN activation.")
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

        self.spike_regularization_weight = spike_regularization_weight

    def forward(self, x, **kwargs):
        all_spikes_list: List[torch.Tensor] = []

        x_res = x
        if self.prenorm: 
            x = self.norm1(x)

        x_attn, reg_loss_from_attn, attn_spikes_dict = self.attn(x, **kwargs) # renamed reg_loss to reg_loss_from_attn for clarity. DeltaNet currently returns None for it.

        if attn_spikes_dict and isinstance(attn_spikes_dict, dict):
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        x = x_res + self.dropout1(x_attn)

        x_res = x
        if self.prenorm: 
            x = self.norm2(x)

        x_mlp, mlp_spikes1, mlp_spikes2 = self.mlp(x)

        if mlp_spikes1 is not None:
            all_spikes_list.append(mlp_spikes1)
        if mlp_spikes2 is not None:
            all_spikes_list.append(mlp_spikes2)

        x = x_res + self.dropout2(x_mlp)

        final_spikes = None
        current_reg_loss = None
        if all_spikes_list:
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

            if self.spike_regularization_weight > 0:
                spike_activity = final_spikes.float().mean()
                current_reg_loss = self.spike_regularization_weight * spike_activity

        # Note: reg_loss_from_attn from DeltaNet is currently None, so the logic for combining losses is effectively a placeholder.
        # If DeltaNet were to return a regularization loss, this would correctly sum them.
        total_reg_loss = current_reg_loss

        return x, total_reg_loss, final_spikes