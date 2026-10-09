"""Implements a SpikingFLABlock with enhanced spike-driven adaptive gating and recurrent state stability for improved neuromorphic integration and sequence modeling performance.
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
from fla.modules import FusedRMSNormGated, RMSNorm
from fla.modules.l2norm import l2norm

# ==================== SNN Related Imports and Helper Classes ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder SNN modules.")
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return input
        @staticmethod
        def backward(ctx, grad_output): return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None
# --- End of Placeholders ---

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)
        # Generate spikes (binary signal)
        spikes_transposed = self.neuron(x_transposed)
        # (B, D, L) -> (B, L, D) to match original input shape for linear layer
        spikes = spikes_transposed.transpose(-1, -2)
        # Linear layer operates on binary spikes
        output = self.linear(spikes)
        # output is continuous, spikes are binary
        return output, spikes

# ================================================================

def delta_rule_chunkwise(q, k, v, beta, chunk_size=32, state_clamp_val: float=10.0):
    """Delta-rule Chunkwise parallel operator with state clamping for stability."""
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len), value=0.0) # Ensure beta is padded with 0.0, as exp(0)=1 for initial state

    padded_len = l + pad_len
    # q, k are L2 normalized *before* chunking and attention calculation
    # This normalization operates on the continuous outputs from SpikingLinear
    q = l2norm(q)
    k = l2norm(k)

    # v is scaled by beta
    v = v * beta[..., None] # beta is (B, H, L), v is (B, H, L, D_v)
    k_beta = k * beta[..., None]

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])

    # Intra-chunk attention calculation, modified for delta-rule
    # The attention calculation itself needs to be robust. Assuming 'attn' here is a component to derive 'u' and 'w'
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0) # This original part seems specific, keeping it as is.
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
        u_i = u[:, :, i] - w[:, :, i] @ S
        o[:, :, i] = o_inter + attn_intra @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        S = torch.clamp(S, -state_clamp_val, state_clamp_val) # State clamping added here
    o = rearrange(o, 'b h n c d -> b h (n c) d')

    if pad_len > 0:
        o = o[:, :, :l, :]
    return o, S

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

class DeltaNet(nn.Module):
    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=False, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', state_clamp_val: float=10.0, **kwargs) -> 'DeltaNet':
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
        if self.use_short_conv:
            raise ValueError("ShortConvolution is not supported in this SNN-pure DeltaNet implementation.")

        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx
        self.state_clamp_val = state_clamp_val # Store for forward pass

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)

        self.use_beta = use_beta
        if self.use_beta:
            self.beta_spike_proj = SpikingLinear(nn.Linear(1, self.num_heads, bias=False), neuron_type, surrogate_function)


        if use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional['Cache']=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: 'Unpack[Dict]') -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        all_spikes: Dict[str, torch.Tensor] = {}

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "Expected attention_mask as a 0-1 matrix"

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None and cu_seqlens is None: # Added condition to only unpad if not already padded via cu_seqlens
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states_for_proj = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0) # Unsqueeze to match expected batch dim
        else:
            hidden_states_for_proj = hidden_states

        q_analog, spikes_q_proj = self.q_proj(hidden_states_for_proj)
        k_analog, spikes_k_proj = self.k_proj(hidden_states_for_proj)
        v_analog, spikes_v_proj = self.v_proj(hidden_states_for_proj)

        all_spikes['q_proj'] = spikes_q_proj
        all_spikes['k_proj'] = spikes_k_proj
        all_spikes['v_proj'] = spikes_v_proj

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q_analog, k_analog))
        v = rearrange(v_analog, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.qk_norm == 'l2':
            q = l2norm(q)
            k = l2norm(k)
        # elif self.qk_norm == 'sum':
            # q = sum_norm(q)
            # k = sum_norm(k)

        if self.use_beta:
            k_spike_mean_activity_for_beta = spikes_k_proj.float().mean(dim=-1, keepdim=True) # (B, L, 1)
            beta_proj_analog_out, spikes_beta_proj = self.beta_spike_proj(k_spike_mean_activity_for_beta)
            beta = beta_proj_analog_out.sigmoid()
            all_spikes['beta_spike_proj'] = spikes_beta_proj
        else:
            beta = torch.ones_like(q[..., 0])

        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state.get('recurrent_state', None) if last_state is not None else None
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')

        # Pass state_clamp_val to delta_rule_chunkwise
        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v, beta=beta, state_clamp_val=self.state_clamp_val)
        o = rearrange(o, 'b h l d -> b l h d')

        if past_key_values is not None:
            cache_data = {
                'recurrent_state': recurrent_state,
            }
            if self.layer_idx is not None:
                past_key_values[self.layer_idx] = cache_data

        if self.use_gate:
            g_out_analog, spikes_g = self.g_proj(hidden_states_for_proj)
            g = rearrange(g_out_analog, '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
            all_spikes['g_proj'] = spikes_g
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')

        o_final_analog, spikes_o = self.o_proj(o)
        all_spikes['o_proj'] = spikes_o

        if attention_mask is not None and cu_seqlens is None: # Only pad back if it was unpadded here
            o_final_analog = pad_input(o_final_analog.squeeze(0), indices, batch_size, q_len) # Squeeze batch dim added for `index_first_axis`

        return o_final_analog, None, all_spikes # Original returns None for reg_loss

# Helper function placeholder (if sum_norm was ever intended to be used in DeltaNet)
# def sum_norm(x):
#     return x / (x.sum(dim=-1, keepdim=True) + 1e-8)

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_spiking_mlp: bool=True, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', state_clamp_val: float=10.0, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        # Pass state_clamp_val to DeltaNet
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, state_clamp_val=state_clamp_val, **kwargs)

        surrogate_function = piecewise_quadratic_surrogate()

        self.num_heads = num_heads

        # Head-wise Spike-Adaptive Gating for Attention:
        self.attn_head_gate_proj = SpikingLinear(nn.Linear(num_heads, d_model, bias=False), neuron_type, surrogate_function)
        # Added learnable gain and bias for attention gate
        self.attn_gate_gain = nn.Parameter(torch.ones(1)) # Initialize to 1, softplus will ensure positive
        self.attn_gate_bias = nn.Parameter(torch.zeros(1)) # Initialize to 0

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        if not use_spiking_mlp:
             raise ValueError("use_spiking_mlp must be True for SpikingFLABlock to ensure SNN nature.")

        # Inlined SpikingMLP components
        self.mlp_fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.mlp_dropout_inner = nn.Dropout(dropout)
        self.mlp_fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

        # Spiking-Adaptive Gating for MLP: Input is only mean_mlp_spike_activity (1) -> output d_model
        self.mlp_gate_input_dim = 1
        self.mlp_gate_proj = SpikingLinear(nn.Linear(self.mlp_gate_input_dim, d_model, bias=False), neuron_type, surrogate_function)
        # Added learnable gain and bias for MLP gate
        self.mlp_gate_gain = nn.Parameter(torch.ones(1)) # Initialize to 1
        self.mlp_gate_bias = nn.Parameter(torch.zeros(1)) # Initialize to 0

    def forward(self, x: torch.Tensor, past_key_values: Optional['Cache']=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        reg_loss = None # DeltaNet might return a reg_loss, but current implementation does not.

        # --- Attention Block ---
        x_res1 = x
        x_norm1 = self.norm1(x) if self.prenorm else x

        x_attn_output_continuous, delta_reg_loss, attn_spikes_dict = self.attn(x_norm1, past_key_values=past_key_values, use_cache=use_cache, layer_idx=layer_idx, **kwargs)

        # If DeltaNet ever returns a meaningful reg_loss, it will be assigned here.
        if delta_reg_loss is not None:
             reg_loss = delta_reg_loss

        attention_branch_spikes_for_gating: List[torch.Tensor] = []
        if attn_spikes_dict:
            for spike_name, spike_tensor in attn_spikes_dict.items():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

                    # For attention gating, prioritize key spikes for head-wise aggregation
                    # Check if the tensor is empty before proceeding
                    if spike_name == 'k_proj' and spike_tensor.numel() > 0:
                        attention_branch_spikes_for_gating.append(spike_tensor)

        mean_attn_spike_activity_per_head: torch.Tensor
        if attention_branch_spikes_for_gating:
            # We expect 'k_proj' spikes here. This is (B, L, D_key)
            k_proj_spikes = attention_branch_spikes_for_gating[0] # Assuming only k_proj spikes are added here

            # Rearrange k_proj_spikes to (B, L, num_heads, head_k_dim)
            rearranged_k_spikes = rearrange(k_proj_spikes.float(), 'b l (h d) -> b l h d', h=self.num_heads)
            # Calculate mean activity per head, resulting in (B, L, num_heads)
            mean_attn_spike_activity_per_head = rearranged_k_spikes.mean(dim=-1) # (B, L, num_heads)
        else:
            # Shape should be (B, L, num_heads)
            mean_attn_spike_activity_per_head = torch.zeros(x_res1.shape[0], x_res1.shape[1], self.num_heads, device=x_res1.device)

        branch_output_attn = x_attn_output_continuous

        # Head-wise Spike-Adaptive Gating for Attention
        gating_factor_cont_attn, s_gate_attn = self.attn_head_gate_proj(mean_attn_spike_activity_per_head)
        all_spikes_list.append(s_gate_attn)
        # Apply adaptive gain and bias before sigmoid to make gating more flexible
        gating_factor_attn = 1.0 + F.softplus(self.attn_gate_gain) * torch.sigmoid(gating_factor_cont_attn + self.attn_gate_bias)

        x_after_attn_gated = x_res1 + (branch_output_attn * gating_factor_attn)

        # --- MLP Block ---
        x_res2 = x_after_attn_gated
        x_norm2 = self.norm2(x_after_attn_gated) if self.prenorm else x_after_attn_gated

        # Inlined MLP operations
        x_fc1_analog_out, s_mlp1 = self.mlp_fc1(x_norm2)
        # Re-introducing non-linearity after the first linear layer in MLP for better capacity
        x_activated_mlp1 = F.gelu(x_fc1_analog_out)

        x_after_mlp_inner_dropout = self.mlp_dropout_inner(x_activated_mlp1)
        x_mlp_output_continuous, s_mlp2 = self.mlp_fc2(x_after_mlp_inner_dropout)

        all_spikes_list.extend([s_mlp1, s_mlp2])

        mlp_branch_spikes_for_gating: List[torch.Tensor] = [s_mlp1, s_mlp2]

        mean_mlp_spike_activity: torch.Tensor
        if mlp_branch_spikes_for_gating:
            # Filter None tensors and ensure numel() > 0 before flattening and concatenating
            valid_spikes = [s.float().flatten(2) for s in mlp_branch_spikes_for_gating if s is not None and s.numel() > 0]
            if valid_spikes: # Check if there are any valid spikes left after filtering
                flat_mlp_spikes_for_mean = torch.cat(valid_spikes, dim=-1)
                mean_mlp_spike_activity = flat_mlp_spikes_for_mean.mean(dim=-1, keepdim=True) # (B, L, 1)
            else:
                mean_mlp_spike_activity = torch.zeros_like(x_res2[..., :1]) # Fallback to zeros
        else:
            mean_mlp_spike_activity = torch.zeros_like(x_res2[..., :1]) # Fallback to zeros

        branch_output_mlp = x_mlp_output_continuous

        # Spike-Adaptive Gating for MLP
        gating_factor_cont_mlp, s_gate_mlp = self.mlp_gate_proj(mean_mlp_spike_activity)
        all_spikes_list.append(s_gate_mlp)
        # Apply adaptive gain and bias before sigmoid to make gating more flexible
        gating_factor_mlp = 1.0 + F.softplus(self.mlp_gate_gain) * torch.sigmoid(gating_factor_cont_mlp + self.mlp_gate_bias)

        x_final_output_gated = x_res2 + (branch_output_mlp * gating_factor_mlp)

        # Filter out None tensors and flatten valid tensors for final output
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None and s.numel() > 0]) if all_spikes_list else torch.empty(0, device=x.device, dtype=torch.bool)

        return x_final_output_gated, reg_loss, final_spikes