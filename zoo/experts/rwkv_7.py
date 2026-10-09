# 28_SpikingRWKV7.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm

# [SNN Imports]
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

# ==================== White-box Operator (RWKV-7 Goose Logic) ====================

def rwkv7_chunkwise(r, w, k, v, a, chunk_size=64, initial_state=None):
    """RWKV7 Parallel Logic.
    State Update: S_t = S_{t-1} * decay + K_t^T * (V_t - K_t * a_t)
    """
    b, h, l, d = k.shape
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        r, w, k, v, a = [F.pad(x, (0, 0, 0, pad_len)) for x in [r, w, k, v, a]]
    
    # Decay
    w = torch.exp(-F.softplus(w))
    
    r = rearrange(r, 'b h (n c) d -> b h n c d', c=chunk_size)
    w = rearrange(w, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    a = rearrange(a, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, d, d, device=k.device)
        
    o_list = []
    block_size = 32
    
    # [Pre-compute V_net]
    # v_net = v - k * a
    # This logic is element-wise and local, safe to do before slicing.
    # Note: K*a is element-wise product.
    v_net = v - (k * a)
    
    for i in range(r.shape[2]):
        r_chunk, w_chunk, k_chunk, v_net_chunk = r[:, :, i], w[:, :, i], k[:, :, i], v_net[:, :, i]
        
        o_chunk_slices = []
        S_new_slices = []
        
        for start_idx in range(0, d, block_size):
            end_idx = min(start_idx + block_size, d)
            
            # Slice K dimension
            k_i = k_chunk[..., start_idx:end_idx]
            r_i = r_chunk[..., start_idx:end_idx]
            w_i = w_chunk[..., start_idx:end_idx]
            S_i = S[..., start_idx:end_idx, :]
            v_net_full = v_net_chunk # Use full V dimension
            
            # --- Intra-Chunk ---
            w_log = torch.log(w_i + 1e-8)
            w_cumsum = torch.cumsum(w_log, dim=-2)
            
            decay_matrix = w_cumsum.unsqueeze(-2) - w_cumsum.unsqueeze(-3)
            mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device), diagonal=1)
            decay_matrix = decay_matrix.masked_fill(mask.unsqueeze(-1), -float('inf'))
            decay_mask = torch.exp(decay_matrix)
            
            # Attn = (R_i * K_j * Mask)
            attn_score = torch.einsum('b h i d, b h j d, b h i j d -> b h i j', r_i, k_i, decay_mask)
            
            o_intra = attn_score @ v_net_full
            
            # --- Inter-Chunk ---
            decay_from_start = torch.exp(w_cumsum)
            r_decayed = r_i * decay_from_start
            o_inter = r_decayed @ S_i
            
            o_chunk_slices.append(o_intra + o_inter)
            
            # --- Update State ---
            total_decay = torch.exp(w_log.sum(dim=-2)).unsqueeze(-1)
            decay_to_end = torch.exp(w_cumsum[:, :, -1, :].unsqueeze(-2) - w_cumsum)
            k_weighted = k_i * decay_to_end
            
            current_contrib = k_weighted.transpose(-1, -2) @ v_net_full
            
            S_new_slices.append(S_i * total_decay + current_contrib)
            
        S = torch.cat(S_new_slices, dim=-2)
        o_chunk = sum(o_chunk_slices)
        o_list.append(o_chunk)
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    return o, S

# ==================== Main Layer ====================

class RWKV7(nn.Module):
    def __init__(self, d_model, num_heads, expand_k=1.0, expand_v=1.0, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)

        # Ensure K=V dim for element-wise 'a'
        assert self.key_dim == self.value_dim

        surrogate = piecewise_quadratic_surrogate()

        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_w = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_a = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_g = nn.Parameter(torch.ones(1, 1, d_model))

        self.r_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.w_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.a_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.value_dim // num_heads)

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}
        x_prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)

        def mix(p, x_c, x_l): return x_c * p + x_l * (1 - p)
        xr, xw = mix(self.time_mix_r, x, x_prev), mix(self.time_mix_w, x, x_prev)
        xk, xv = mix(self.time_mix_k, x, x_prev), mix(self.time_mix_v, x, x_prev)
        xa, xg = mix(self.time_mix_a, x, x_prev), mix(self.time_mix_g, x, x_prev)

        r, s_r = self.r_proj(xr)
        w, s_w = self.w_proj(xw)
        k, s_k = self.k_proj(xk)
        v, s_v = self.v_proj(xv)
        a, s_a = self.a_proj(xa)
        g, s_g = self.g_proj(xg)
        all_spikes.update({'r': s_r, 'w': s_w, 'k': s_k, 'v': s_v, 'a': s_a, 'g': s_g})

        r = rearrange(r, 'b l (h d) -> b h l d', h=self.num_heads)
        w = rearrange(w, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        a = rearrange(a, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l (h d) -> b h l d', h=self.num_heads)

        last_state = None
        if past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = rwkv7_chunkwise(r, w, k, v, a, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                past_key_values.append({'recurrent_state': final_state})
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_state

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.g_norm(o)
        g = rearrange(g, 'b h l d -> b l h d')
        o = o * F.silu(g)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o
        return o, None, all_spikes

# [Block Wrapper]

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
        self.attn = RWKV7(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
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