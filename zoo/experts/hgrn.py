# 25_SpikingHGRN.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm

# [Import Common SNN Components: SpikingLinear, etc.]
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return input
        @staticmethod
        def backward(ctx, grad_output): return grad_output

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x):
        x_transposed = x.transpose(-1, -2)
        spikes = self.neuron(x_transposed).transpose(-1, -2)
        return self.linear(spikes), spikes

# ==================== White-box Operator (HGRN) ====================

def hgrn_chunkwise(g, x, chunk_size=64, initial_state=None):
    """HGRN Chunkwise Parallel Logic (Element-wise Recurrence).
    Formula: h_t = h_{t-1} * exp(g_t) + x_t
    
    Parallelization:
        Uses Feature Slicing to compute the [C, C, D] decay mask without OOM.
    """
    b, h, l, d = x.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        g, x = [F.pad(t, (0, 0, 0, pad_len)) for t in [g, x]]
        
    g = rearrange(g, 'b h (n c) d -> b h n c d', c=chunk_size)
    x = rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, d, device=x.device)
        
    o_list = []
    
    # Feature Block Slicing to prevent OOM
    block_size = 32
    
    for i in range(g.shape[2]): # Chunk Loop
        g_chunk = g[:, :, i] # [B, H, C, D]
        x_chunk = x[:, :, i]
        
        o_chunk_slices = []
        S_new_slices = []
        
        for start_idx in range(0, d, block_size):
            end_idx = min(start_idx + block_size, d)
            
            # Slice
            g_i = g_chunk[..., start_idx:end_idx]
            x_i = x_chunk[..., start_idx:end_idx]
            S_i = S[..., start_idx:end_idx]
            
            # 1. Intra-Chunk (Parallel)
            # g is log-decay. Cumsum gives potentials.
            g_cumsum = torch.cumsum(g_i, dim=-2)
            
            # Mask: exp(G_i - G_j)
            decay_matrix = g_cumsum.unsqueeze(-2) - g_cumsum.unsqueeze(-3)
            mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device), diagonal=1)
            decay_matrix = decay_matrix.masked_fill(mask.unsqueeze(-1), -float('inf'))
            decay_mask = torch.exp(decay_matrix)
            
            # O_intra = Mask @ X
            # [B, H, C, C, D_block] @ [B, H, C, D_block] -> [B, H, C, D_block]
            # Summing over the second C dimension (j)
            o_intra = torch.einsum('b h i j d, b h j d -> b h i d', decay_mask, x_i)
            
            # 2. Inter-Chunk
            decay_from_start = torch.exp(g_cumsum)
            o_inter = S_i.unsqueeze(2) * decay_from_start
            
            o_chunk_slices.append(o_intra + o_inter)
            
            # 3. Update State
            # Total decay for the chunk
            total_decay = torch.exp(g_i.sum(dim=-2))
            # Decay from j to end
            decay_to_end = torch.exp(g_cumsum[:, :, -1].unsqueeze(-2) - g_cumsum)
            current_chunk_contrib = (x_i * decay_to_end).sum(dim=-2)
            
            S_new_slice = S_i * total_decay + current_chunk_contrib
            S_new_slices.append(S_new_slice)
            
        # Concatenate slices
        S = torch.cat(S_new_slices, dim=-1)
        o_chunk = torch.cat(o_chunk_slices, dim=-1)
        o_list.append(o_chunk)
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    return o, S

# ==================== Main Layer ====================

class HGRN(nn.Module):
    def __init__(self, d_model, num_heads=None, expand_ratio=1.0, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        if num_heads is None: num_heads = d_model // 64
        self.num_heads = num_heads
        self.hidden_dim = int(d_model * expand_ratio)

        surrogate = piecewise_quadratic_surrogate()

        # HGRN Projections
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.i_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)

        self.o_proj = SpikingLinear(nn.Linear(self.hidden_dim, d_model, bias=False), neuron_type, surrogate)
        self.norm = RMSNorm(self.hidden_dim)

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}

        g, s_g = self.g_proj(x)
        i, s_i = self.i_proj(x)
        all_spikes.update({'g': s_g, 'i': s_i})

        g = rearrange(g, 'b l (h d) -> b h l d', h=self.num_heads)
        i = rearrange(i, 'b l (h d) -> b h l d', h=self.num_heads)

        # [MODIFICATION for Parallel Stability]
        # Use logsigmoid instead of sigmoid for 'g'. 
        # This keeps 'g' in log-space (negative values), ensuring exp(g) is in (0, 1).
        # Standard HGRN uses sigmoid(g), but for parallel cumsum, log-space is required.
        g = F.logsigmoid(g)

        i = F.silu(i)

        last_state = None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = hgrn_chunkwise(g, i, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                past_key_values.append({'recurrent_state': final_state})
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_state

        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.norm(o)
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

# [SpikingFLABlock Wrapper would go here]
# ... (Same as above) ...

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
        self.attn = HGRN(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
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