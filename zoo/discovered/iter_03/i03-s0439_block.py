"""Implements a SpikingFLABlock with spike-gated residual connections and simplified MLP for enhanced neuromorphic integration.
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
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D) # Binary spikes
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
        attn_intra = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn_intra @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')

    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

class DeltaNet(nn.Module):
    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> 'DeltaNet':
        super().__init__()

        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")

        surrogate_function = piecewise_quadratic_surrogate()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
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

        # All nn.Linear layers are replaced by SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), neuron_type, surrogate_function)
            # New: Spike-based modulator for beta
            # It takes the mean spike activity (a scalar per token) and projects it to num_heads
            self.spike_beta_mod_proj = SpikingLinear(nn.Linear(1, self.num_heads, bias=True), neuron_type, surrogate_function)


        # ShortConvolution layers remain non-spiking as per instructions and SNN-ANN hybrid design
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
        else:
            # If short conv is not used, provide an activation for v, similar to v_conv1d's internal activation
            self.v_activation = nn.SiLU()


        if use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional['Cache']=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: 'Unpack[Dict]') -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional['Cache']]:
        all_spikes = {} # Used to collect all spikes

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

        # SpikingLinear layers take analog input and produce analog output and binary spikes
        q, spikes_q_proj = self.q_proj(hidden_states)
        k, spikes_k_proj = self.k_proj(hidden_states)
        v, spikes_v_proj = self.v_proj(hidden_states)

        all_spikes['q_proj'] = spikes_q_proj
        all_spikes['k_proj'] = spikes_k_proj
        all_spikes['v_proj'] = spikes_v_proj

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            q, conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            # Apply activation for v if short conv is not used, to match v_conv1d behavior
            v = self.v_activation(v)


        # Apply qk_activation only if not 'identity', centralized here for consistency
        if self.qk_activation == 'silu':
            q, k = F.silu(q), F.silu(k)
        elif self.qk_activation == 'relu':
            q, k = q.relu(), k.relu()
        elif self.qk_activation == 'elu':
            q, k = elu_p1(q), elu_p1(x)
        # If self.qk_activation == 'identity', no additional activation applied to q, k here.


        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.qk_norm == 'l2':
            q = l2norm(q)
            k = l2norm(k)
        elif self.qk_norm == 'sum':
            q = sum_norm(q)
            k = sum_norm(k)


        if self.use_beta:
            beta, spikes_beta = self.b_proj(hidden_states)
            beta = beta.sigmoid()
            all_spikes['b_proj'] = spikes_beta

            # Integrate spike activity for beta modulation
            if spikes_k_proj is not None:
                # Calculate mean spike activity of keys across the feature dimension
                # Resulting shape: (B, L, 1)
                k_spike_mean_activity = spikes_k_proj.float().mean(dim=-1, keepdim=True)
                # Project the mean spike activity to the number of heads using a SpikingLinear layer
                # Resulting shape: (B, L, num_heads)
                beta_spike_mod, spikes_beta_mod = self.spike_beta_mod_proj(k_spike_mean_activity)
                all_spikes['spike_beta_mod'] = spikes_beta_mod
                # Apply a soft modulation (e.g., tanh) and then multiply
                beta = beta * (1 + beta_spike_mod.tanh()) # Using tanh to keep modulation around 1 (can be slightly >1 or <1)

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
            g, spikes_g = self.g_proj(hidden_states)
            g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
            all_spikes['g_proj'] = spikes_g
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')

        o, spikes_o = self.o_proj(o)
        all_spikes['o_proj'] = spikes_o

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        # Removed self.activation and self.dropout to streamline the MLP towards a more spiking-native design.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_out, spike1 = self.fc1(x)
        # No non-spiking activation or dropout applied here, relying on SpikingLinear's internal neuron for non-linearity.
        x_fc2_out, spike2 = self.fc2(x_fc1_out)
        return x_fc2_out, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # DeltaNet also uses SpikingLinear layers internally.
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        # Always use SpikingMLP to ensure all linear layers are spiking-enabled.
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

        # New: Spike-gating for attention residual connection
        # It takes a scalar summary of spike activity and projects it to d_model for gating.
        self.attn_spike_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, piecewise_quadratic_surrogate())
        # New: Learnable scaling factor for the attention spike gate.
        self.attn_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

        # New: Spike-gating for MLP residual connection
        # Similar to attention, takes (B, L, 1) mean spike activity and projects to d_model.
        self.mlp_spike_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, piecewise_quadratic_surrogate())
        # New: Learnable scaling factor for the MLP spike gate.
        self.mlp_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        x_res = x
        if self.prenorm:
            x = self.norm1(x)

        # DeltaNet returns analog output, optional regularization loss, and a dictionary of collected spikes.
        x_attn, reg_loss, attn_spikes = self.attn(x, **kwargs)

        # Collect spikes from the attention mechanism and prepare for spike-gating
        attn_block_spike_activity = None
        if attn_spikes and isinstance(attn_spikes, dict):
            # Prioritize 'k_proj' spikes for attention gating as they are already used for beta modulation
            k_spikes_for_gating = attn_spikes.get('k_proj')
            if k_spikes_for_gating is not None:
                attn_block_spike_activity = k_spikes_for_gating.float().mean(dim=-1, keepdim=True) # (B, L, 1)
            else:
                # Fallback: if 'k_proj' spikes are not available, take the mean activity of the first available spike tensor
                for spike_name, spike_tensor in attn_spikes.items():
                    if spike_tensor is not None:
                        attn_block_spike_activity = spike_tensor.float().mean(dim=-1, keepdim=True)
                        break

            # Collect all spikes for final output
            for spike_tensor in attn_spikes.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        # Apply spike-dependent gating to the attention output
        if attn_block_spike_activity is not None:
            # The analog output of the SpikingLinear layer will be (B, L, d_model)
            attn_gate_analog_output, attn_gate_binary_spikes = self.attn_spike_gate_proj(attn_block_spike_activity)
            all_spikes_list.append(attn_gate_binary_spikes) # Collect spikes from the gate
            # Use sigmoid and learnable scaling to modulate the attention output
            attn_gate = self.attn_gate_scale * torch.sigmoid(attn_gate_analog_output)
            x_attn = x_attn * attn_gate # Element-wise gating


        x = x_res + self.dropout1(x_attn)

        x_res = x
        if self.prenorm:
            x = self.norm2(x)

        # SpikingMLP returns its final analog output and individual spike tensors for its two layers.
        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x)

        # Collect spikes from the MLP and prepare for spike-gating
        mlp_spikes_for_gating_means = []
        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
            mlp_spikes_for_gating_means.append(mlp_spike1.float().mean(dim=-1, keepdim=True)) # (B, L, 1)
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)
            mlp_spikes_for_gating_means.append(mlp_spike2.float().mean(dim=-1, keepdim=True)) # (B, L, 1)

        # Apply spike-dependent gating to the MLP output
        if mlp_spikes_for_gating_means:
            # Average the mean activities from fc1 and fc2 for the MLP gate's input
            mlp_block_spike_activity = torch.mean(torch.stack(mlp_spikes_for_gating_means), dim=0) # (B, L, 1)
            # The analog output of the SpikingLinear layer will be (B, L, d_model)
            mlp_gate_analog_output, mlp_gate_binary_spikes = self.mlp_spike_gate_proj(mlp_block_spike_activity)
            all_spikes_list.append(mlp_gate_binary_spikes) # Collect spikes from the gate
            # Use sigmoid and learnable scaling to modulate the MLP output
            mlp_gate = self.mlp_gate_scale * torch.sigmoid(mlp_gate_analog_output)
            x_mlp = x_mlp * mlp_gate # Element-wise gating

        x = x_res + self.dropout2(x_mlp)

        final_spikes = None
        if all_spikes_list:
            # Concatenate all collected binary spike tensors into a single flattened tensor.
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

        return x, reg_loss, final_spikes