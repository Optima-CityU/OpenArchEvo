# 29_SpikingABC.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return input
        @staticmethod
        def backward(ctx, grad_output): return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x):
        x_transposed = x.transpose(-1, -2)
        spikes = self.neuron(x_transposed).transpose(-1, -2)
        return self.linear(spikes), spikes

# ==================== White-box Operator (ABC: Two-Pass Linear Attention) ====================

def linear_attn_chunkwise(q, k, v, chunk_size=64, initial_state=None, scale=1.0):
    """Standard Linear Attention Chunkwise Parallel.
    O = Q @ (Sum K^T V)
    No decay involved (RetNet with gamma=1).
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # Padding
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v]]
        
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = torch.zeros(b, h, d_k, d_v, device=q.device)
        
    o_list = []
    
    # Pre-compute causal mask for intra-chunk
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    
    for i in range(q.shape[2]):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        
        # 1. Intra-Chunk
        # A = Q @ K^T (Causal)
        attn = q_i @ k_i.transpose(-1, -2)
        attn = attn.masked_fill(mask, 0)
        
        o_intra = attn @ v_i
        
        # 2. Inter-Chunk
        # O_inter = Q @ S
        o_inter = q_i @ S
        
        o_list.append(o_intra + o_inter)
        
        # 3. Update State
        # S += K^T @ V
        S = S + (k_i.transpose(-1, -2) @ v_i)
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    
    return o, S

# ==================== ABC Operator (Two-Pass Wrapper) ====================

def abc_chunkwise(q, k, v, s, chunk_size=64, initial_state=None):
    """Parallel ABC: Calls Linear Attention twice."""
    if initial_state is not None:
        S1_init, S2_init = initial_state
    else:
        S1_init, S2_init = None, None
        
    # Pass 1: Q=q, K=k, V=s
    # Scale q by d_k ** -0.5
    scale = q.shape[-1] ** -0.5
    
    # logits = LinearAttn(q, k, s)
    # Output of Pass 1 is [B, H, L, M] (scores over slots)
    logits, S1 = linear_attn_chunkwise(q, k, s, chunk_size, initial_state=S1_init, scale=scale)
    
    # Softmax
    probs = F.softmax(logits, dim=-1)
    
    # Pass 2: Q=probs, K=s, V=v
    # Scale is 1.0 (probs already normalized)
    # o = LinearAttn(probs, s, v)
    o, S2 = linear_attn_chunkwise(probs, s, v, chunk_size, initial_state=S2_init, scale=1.0)
    
    return o, (S1, S2)

# ==================== Main Layer ====================

class ABC(nn.Module):
    def __init__(self, d_model, num_heads, num_slots=64, expand_v=1.0, use_short_conv=True, conv_size=4, neuron_type='sdn', clamp_min=-32.0, clamp_max=32.0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.value_dim = int(d_model * expand_v)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        surrogate = piecewise_quadratic_surrogate()

        # Projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        # S projection: projects to [num_heads * num_slots]
        self.s_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)

        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.value_dim // num_heads)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(d_model, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(d_model, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            self.s_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu') 
            self.g_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        s, s_s = self.s_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 's': s_s, 'g': s_g})

        if self.use_short_conv:
            # Cache logic omitted for brevity, assuming standard pattern
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)
            s, _ = self.s_conv1d(s)
            g, _ = self.g_conv1d(g)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l (h d) -> b h l d', h=self.num_heads)
        s = rearrange(s, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)

        s = s.clamp(min=self.clamp_min, max=self.clamp_max)

        last_state = None
        if past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = abc_chunkwise(q, k, v, s, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            state_dict = {'recurrent_state': final_state}
            if len(past_key_values) <= layer_idx:
                past_key_values.append(state_dict)
            else:
                past_key_values[layer_idx] = state_dict

        o = rearrange(o, 'b h l d -> b l h d')
        g = rearrange(g, 'b h l d -> b l h d')
        o = self.g_norm(o, g)

        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

# [SpikingMLP and SpikingFLABlock definitions remain standard...]

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

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = ABC(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        all_spikes_list = []
        x_res = x
        if self.prenorm: x = self.norm1(x)
        x_attn, _, attn_spikes = self.attn(x, **kwargs)
        if attn_spikes: all_spikes_list.extend(attn_spikes.values())
        x = x_res + self.dropout1(x_attn)
        x_res = x
        if self.prenorm: x = self.norm2(x)
        if isinstance(self.mlp, SpikingMLP):
            x_mlp, s1, s2 = self.mlp(x)
            all_spikes_list.extend([s1, s2])
        else:
            x_mlp = self.mlp(x)
        x = x + x_mlp
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None
        return x, None, final_spikes