# 27_SpikingRWKV6.py
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
        # Linear layer expects [B, L, D]
        return self.linear(spikes), spikes

# ==================== White-box Operator (RWKV-6 Eagle Logic) ====================

def rwkv6_chunkwise(r, k, v, w, u, chunk_size=64, initial_state=None):
    """RWKV6 Parallel Logic with Feature Slicing.
    Fixes: Dimension mismatch when d_k != d_v.
    """
    b, h, l, d_k = k.shape
    d_v = v.shape[-1]  # [FIX] Get explicit value dimension
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        r, k, v, w = [F.pad(x, (0, 0, 0, pad_len)) for x in [r, k, v, w]]
    
    # Decay: exp(-softplus(w)) -> (0, 1)
    w = torch.exp(-F.softplus(w))
    
    r = rearrange(r, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    w = rearrange(w, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    if initial_state is not None:
        S = initial_state 
    else:
        # [FIX] State is [D_k, D_v]
        S = torch.zeros(b, h, d_k, d_v, device=k.device) 
        
    o_list = []
    
    # Feature Slicing on Key Dimension (d_k)
    # We slice K, R, W, and the K-dimension of S.
    # V is kept full.
    block_size = 32
    
    for i in range(r.shape[2]): # Chunk Loop
        r_chunk = r[:, :, i]
        k_chunk = k[:, :, i]
        v_chunk = v[:, :, i] # Full V
        w_chunk = w[:, :, i]
        
        o_chunk_slices = []
        S_new_slices = []
        
        # Loop over Key dimension
        for start_idx in range(0, d_k, block_size):
            end_idx = min(start_idx + block_size, d_k)
            
            # Slice K, R, W
            k_i = k_chunk[..., start_idx:end_idx] 
            r_i = r_chunk[..., start_idx:end_idx] 
            w_i = w_chunk[..., start_idx:end_idx]
            
            # Slice State along K dimension [..., Block, D_v]
            S_i = S[..., start_idx:end_idx, :]
            
            # Use full V and sliced U
            v_i = v_chunk 
            u_i = u[..., start_idx:end_idx] 
            
            # --- 1. Intra-Chunk ---
            w_log = torch.log(w_i + 1e-8)
            w_cumsum = torch.cumsum(w_log, dim=-2) 
            
            decay_matrix = w_cumsum.unsqueeze(-2) - w_cumsum.unsqueeze(-3)
            mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device), diagonal=1)
            decay_matrix = decay_matrix.masked_fill(mask.unsqueeze(-1), -float('inf'))
            decay_mask = torch.exp(decay_matrix)
            
            # A = (R_i * K_j * Mask) -> Sum over d (Block) -> [B, H, i, j]
            attn_score = torch.einsum('b h i d, b h j d, b h i j d -> b h i j', r_i, k_i, decay_mask)
            
            # O_intra = A @ V_full -> [B, H, C, D_v]
            o_intra_block = attn_score @ v_i 
            
            # Bonus Term (u)
            # u: [H, Block] -> broadcast
            u_broadcast = u_i.unsqueeze(0).unsqueeze(2)
            bonus_score = (r_i * u_broadcast * k_i).sum(dim=-1, keepdim=True) # [B, H, C, 1]
            o_bonus = bonus_score * v_i
            
            o_intra_block = o_intra_block + o_bonus
            
            # --- 2. Inter-Chunk ---
            decay_from_start = torch.exp(w_cumsum) 
            
            # R_decayed: [..., Block]
            r_decayed = r_i * decay_from_start
            
            # O_inter = R_decayed @ S_i (Block, D_v) -> [..., D_v]
            o_inter_block = r_decayed @ S_i
            
            # [FIXED] Dimensions now match (both are D_v)
            o_chunk_slices.append(o_intra_block + o_inter_block)
            
            # --- 3. Update State ---
            total_decay = torch.exp(w_log.sum(dim=-2)).unsqueeze(-1)
            decay_to_end = torch.exp(w_cumsum[:, :, -1, :].unsqueeze(-2) - w_cumsum)
            k_weighted = k_i * decay_to_end
            
            # Update: K^T @ V -> [Block, C] @ [C, D_v] -> [Block, D_v]
            current_chunk_contrib = k_weighted.transpose(-1, -2) @ v_i
            
            S_new_slice = S_i * total_decay + current_chunk_contrib
            S_new_slices.append(S_new_slice)
            
        # Concat State K-dim
        S = torch.cat(S_new_slices, dim=-2)
        
        # Sum outputs from all K-blocks
        o_chunk = sum(o_chunk_slices)
        o_list.append(o_chunk)
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    return o, S

# ==================== Main Layer ====================

class RWKV6(nn.Module):
    def __init__(self, d_model, num_heads, expand_k=1.0, expand_v=2.0, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)

        surrogate = piecewise_quadratic_surrogate()

        # Time Shift Params
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_w = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_g = nn.Parameter(torch.ones(1, 1, d_model))

        self.r_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.w_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate) 
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate) 

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.value_dim // num_heads)

        # Bonus u
        self.u = nn.Parameter(torch.randn(num_heads, self.key_dim // num_heads))

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}

        # Time Shift
        x_prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)

        def mix(mu, x, x_p): return x * mu + x_p * (1 - mu)

        xr = mix(self.time_mix_r, x, x_prev)
        xk = mix(self.time_mix_k, x, x_prev)
        xv = mix(self.time_mix_v, x, x_prev)
        xw = mix(self.time_mix_w, x, x_prev)
        xg = mix(self.time_mix_g, x, x_prev)

        r, s_r = self.r_proj(xr)
        k, s_k = self.k_proj(xk)
        v, s_v = self.v_proj(xv)
        w, s_w = self.w_proj(xw)
        g, s_g = self.g_proj(xg)
        all_spikes.update({'r': s_r, 'k': s_k, 'v': s_v, 'w': s_w, 'g': s_g})

        r = rearrange(r, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        w = rearrange(w, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l (h d) -> b h l d', h=self.num_heads)

        # State
        last_state = None
        if past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = rwkv6_chunkwise(r, k, v, w, self.u, initial_state=last_state)

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

# [Block Wrapper SpikingFLABlock]

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
        self.attn = RWKV6(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
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