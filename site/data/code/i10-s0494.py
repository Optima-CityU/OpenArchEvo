# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution
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
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

# ================================================================

def comba_chunkwise_parallel(q, k, v, g, feedback_net, chunk_size=64, initial_state=None):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len), value=0.0)
    
    q = l2norm(q)
    k = l2norm(k)
    
    q, k, v, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, g]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)

    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        
        # --- Closed-loop feedback stability enhancement ---
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        s_feat = (S / s_norm).mean(dim=(-1, -2), keepdim=True).flatten(2) # [B, H, 1]
        
        # Simplified feedback net output and modulation (f_i from [B,H,1] to [B,H])
        f_i_raw = feedback_net(s_feat) # [B, H, 1]
        f_i = (torch.sigmoid(f_i_raw.squeeze(-1)) * 4.0) - 2.0 # Scale to [-2, 2] using sigmoid for smoother control
        
        # --- Gate numerical computation alignment ---
        log_decay = F.logsigmoid(g_i + f_i.unsqueeze(-1)) # [B, H, C]
        d_cumsum = torch.cumsum(log_decay, dim=-1)
        
        decay_rel = d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0)
        
        # 3. Compute output
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = attn_intra @ v_i
        
        o[:, :, i] = o_inter + o_intra
        
        # --- State update stability enhancement ---
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update
        
        # Strict renormalization after each chunk update
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) # Scale down if norm is high

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Adaptive Spike Gain Scales applied to the input of SpikingLinear layers
        self.q_gain_scale = nn.Parameter(torch.ones(1, 1, d_model)) # Scale d_model features before q_proj
        self.k_gain_scale = nn.Parameter(torch.ones(1, 1, d_model)) # Scale d_model features before k_proj
        self.v_gain_scale = nn.Parameter(torch.ones(1, 1, d_model)) # Scale d_model features before v_proj
        self.g_gain_scale = nn.Parameter(torch.ones(1, 1, d_model)) # Scale d_model features before g_proj

        # Projections using SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        # Simplified Feedback Network: From nn.Sequential(Linear, Tanh, Linear) to just a Linear layer
        self.feedback_net = nn.Linear(1, 1) # Outputs a scalar per (B, H) after squeeze

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, use_cache: Optional[bool]=False, **kwargs):
        all_spikes = {}

        # Apply adaptive spike gain scales before projections
        q_input = x * self.q_gain_scale
        k_input = x * self.k_gain_scale
        v_input = x * self.v_gain_scale
        g_input = x * self.g_gain_scale

        q, s_q = self.q_proj(q_input)
        k, s_k = self.k_proj(k_input)
        v, s_v = self.v_proj(v_input)
        g, s_g = self.g_proj(g_input)

        all_spikes.update({'q_proj':s_q, 'k_proj':s_k, 'v_proj':s_v, 'g_proj':s_g})

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
             last_state = past_key_values[self.layer_idx]
             if last_state is not None:
                 if 'conv_state' in last_state and self.use_short_conv:
                     conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

        if self.use_short_conv:
            q, conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=None)
            k, conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=None)
            v, conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=None)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l h -> b h l')

        recurrent_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state is not None and 'recurrent_state' in last_state:
                recurrent_state = last_state['recurrent_state']

        # Execute core parallel operator
        o, final_recurrent_state = comba_chunkwise_parallel(q, k, v, g, self.feedback_net, initial_state=recurrent_state)

        if use_cache and past_key_values is not None and self.layer_idx is not None:
            cache_data = {
                'recurrent_state': final_recurrent_state,
                'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None
            }
            past_key_values[self.layer_idx] = cache_data

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o_proj'] = s_o

        return o, None, all_spikes # None for reg_loss, handled at block level

# The SpikingMLP class is removed as its functionality is integrated into SpikingFLABlock

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', spike_regularization_weight: float=0.001, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Comba attention mechanism
        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        surrogate_function = piecewise_quadratic_surrogate() # Get surrogate function once

        # --- Gated Spiking Feed-Forward Network (GSFFN) components ---
        # Parameters for adaptive spike gain scales
        self.ffn_main_gain_scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.ffn_gate_gain_scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.ffn_proj_out_gain_scale = nn.Parameter(torch.ones(1, 1, d_ffn))

        # Projections for the main and gate paths, using SpikingLinear
        self.ffn_proj_main = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.ffn_proj_gate = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)

        self.ffn_dropout = nn.Dropout(dropout)

        # Output projection for the FFN, using SpikingLinear
        self.ffn_proj_out = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)
        # --- End GSFFN components ---

        self.dropout2 = nn.Dropout(dropout)
        self.spike_regularization_weight = spike_regularization_weight

    def forward(self, x, attention_mask: Optional[torch.Tensor]=None, past_key_values=None, use_cache: Optional[bool]=False, **kwargs):
        all_spikes_list: List[torch.Tensor] = []

        batch_size_orig, q_len_orig = x.shape[0], x.shape[1]
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len_orig:])
            x_processed = index_first_axis(rearrange(x, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        else:
            x_processed = x

        x_res_attn = x
        if self.prenorm: 
            x_normalized_attn = self.norm1(x_processed)
        else:
            x_normalized_attn = x_processed

        x_attn, _, attn_spikes_dict = self.attn(x_normalized_attn, past_key_values=past_key_values, use_cache=use_cache, **kwargs)

        if attn_spikes_dict and isinstance(attn_spikes_dict, dict):
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        if attention_mask is not None:
            x_attn = pad_input(x_attn.squeeze(0), indices, batch_size_orig, q_len_orig)

        x = x_res_attn + self.dropout1(x_attn)

        # --- Gated Spiking Feed-Forward Network (GSFFN) ---
        x_res_ffn = x
        if self.prenorm: 
            if attention_mask is not None:
                x_ffn_input = self.norm2(index_first_axis(rearrange(x, "b s ... -> (b s) ..."), indices).unsqueeze(0))
            else:
                x_ffn_input = self.norm2(x)
        else: 
            if attention_mask is not None:
                x_ffn_input = index_first_axis(rearrange(x, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            else:
                x_ffn_input = x

        # Apply adaptive spike gain scales before FFN projections
        x_main_scaled = x_ffn_input * self.ffn_main_gain_scale
        x_gate_scaled = x_ffn_input * self.ffn_gate_gain_scale

        # Main projection
        x_main_analog, main_spikes = self.ffn_proj_main(x_main_scaled)
        all_spikes_list.append(main_spikes)

        # Gate projection
        x_gate_analog, gate_spikes = self.ffn_proj_gate(x_gate_scaled)
        all_spikes_list.append(gate_spikes)

        # Gated activation
        # Apply GELU to the main path, Sigmoid to the gate path, then element-wise product
        gated_activation = F.gelu(x_main_analog) * F.sigmoid(x_gate_analog)
        gated_activation = self.ffn_dropout(gated_activation) # Apply dropout after gating

        # Final FFN output projection
        x_proj_out_scaled = gated_activation * self.ffn_proj_out_gain_scale
        x_mlp_out, proj_out_spikes = self.ffn_proj_out(x_proj_out_scaled)
        all_spikes_list.append(proj_out_spikes)
        # --- End GSFFN ---

        if attention_mask is not None:
            x_mlp_out = pad_input(x_mlp_out.squeeze(0), indices, batch_size_orig, q_len_orig)

        x = x_res_ffn + self.dropout2(x_mlp_out)

        final_spikes = None
        current_reg_loss = None
        if all_spikes_list:
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

            if self.spike_regularization_weight > 0:
                spike_activity = final_spikes.float().mean()
                current_reg_loss = self.spike_regularization_weight * spike_activity

        return x, current_reg_loss, final_spikes