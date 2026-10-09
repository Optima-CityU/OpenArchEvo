# 22_SpikingRetNet.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.modules import RMSNorm

# ==================== SNN Imports ====================
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

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2) 
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)
        output = self.linear(spikes)
        return output, spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x)
        x = F.gelu(x) 
        x, spike2 = self.fc2(self.dropout(x))
        return x, spike1, spike2

# ==================== White-box Operator (RetNet) ====================

def get_gamma_decay(n_heads, device):
    s = torch.log(torch.tensor(1 / 32, device=device))
    gamma = 1 - torch.exp(torch.linspace(s, torch.log(torch.tensor(1 / 2, device=device)), n_heads))
    return gamma

def retention_chunkwise(q, k, v, gamma, chunk_size=64, initial_state=None):
    """RetNet Chunkwise Logic."""
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v]]
        
    gamma_chunk = (gamma.view(1, h, 1, 1) ** chunk_size)
    
    indices = torch.arange(chunk_size, device=q.device)
    diff = indices.unsqueeze(0) - indices.unsqueeze(1)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device))
    decay_mask = (gamma.view(h, 1, 1) ** diff.view(1, chunk_size, chunk_size)) * mask.view(1, chunk_size, chunk_size)

    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    # [MODIFIED 2] Use passed initial_state to initialize S
    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = torch.zeros(b, h, d_k, d_v, device=q.device)
        
    o = torch.zeros_like(v)
    
    gamma_vec = gamma.view(h, 1, 1) ** (torch.arange(chunk_size, device=q.device).view(1, chunk_size, 1) + 1)
    gamma_rev = gamma.view(h, 1, 1) ** (chunk_size - torch.arange(chunk_size, device=q.device).view(1, chunk_size, 1))

    for i in range(q.shape[2]): 
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        
        # 1. Intra-chunk
        attn = (q_i @ k_i.transpose(-1, -2)) * decay_mask.unsqueeze(0) 
        o_intra = attn @ v_i
        
        # 2. Inter-chunk
        o_inter = (q_i * gamma_vec.unsqueeze(0)) @ S
        
        o[:, :, i] = o_intra + o_inter
        
        # 3. Update State
        k_decayed = k_i * gamma_rev.unsqueeze(0)
        kv_chunk = k_decayed.transpose(-1, -2) @ v_i
        
        S = S * gamma_chunk + kv_chunk
        
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    return o, S

# ==================== Main Layer ====================

class RetNet(nn.Module):
    def __init__(self, d_model, num_heads, expand_k=1.0, expand_v=1.0, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)

        surrogate = piecewise_quadratic_surrogate()

        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate) 
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

        self.o_norm = RMSNorm(self.value_dim // num_heads)

        # Initialize gamma
        # Note: We use register_buffer but need to ensure it moves to device properly in forward if not automatic
        self.register_buffer('gamma', get_gamma_decay(num_heads, torch.device('cpu')))

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'g': s_g})

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        if self.gamma.device != q.device:
            self.gamma = self.gamma.to(q.device)

        # [MODIFIED 3] Read Past Key Values
        last_state = None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        # [MODIFIED 4] Pass last_state
        o, final_state = retention_chunkwise(q, k, v, self.gamma, initial_state=last_state)

        # [MODIFIED 5] Write Past Key Values
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                past_key_values.append({'recurrent_state': final_state})
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_state

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)

        g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
        o = o * F.silu(g)

        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

# ==================== Block Wrapper ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RetNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.norm2 = RMSNorm(d_model)

        # RetNet typically uses larger FFN, but we keep it configurable
        d_ffn = d_model * 2

        if use_spiking_mlp:
            self.mlp = SpikingMLP(d_model, d_ffn, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))

    def forward(self, x, **kwargs):
        all_spikes_list = []

        x_norm = self.norm1(x)
        # Pass kwargs to support layer_idx etc.
        x_attn, _, attn_spikes = self.attn(x_norm, **kwargs)
        x = x + x_attn
        if attn_spikes: all_spikes_list.extend(attn_spikes.values())

        x_norm = self.norm2(x)
        if isinstance(self.mlp, SpikingMLP):
            x_mlp, s1, s2 = self.mlp(x_norm)
            all_spikes_list.extend([s1, s2])
        else:
            x_mlp = self.mlp(x_norm)
        x = x + x_mlp

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None
        return x, None, final_spikes