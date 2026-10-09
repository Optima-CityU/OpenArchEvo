# 23_SpikingLightNet.py
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fla.modules import RMSNorm

# ==================== SNN Imports and Helpers ====================
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

# ==================== White-box Operator (LightNet) ====================

def lightnet_chunk_step(q_i, k_i, v_i, g_i, S_i, mask):
    """Helper function for a single chunk calculation to be checkpointed."""
    kv_i = k_i * v_i
    
    g_cumsum = torch.cumsum(g_i, dim=-2)
    
    # Efficient Decay Mask Calculation
    # shape: (B, H, C, C, D_block)
    decay_matrix = g_cumsum.unsqueeze(-2) - g_cumsum.unsqueeze(-3)
    
    # Apply Mask (Passed in to avoid recreating)
    decay_matrix = decay_matrix.masked_fill(mask.unsqueeze(-1), -float('inf'))
    decay_mask = torch.exp(decay_matrix)
    
    # Output Calculation
    o_intra = torch.einsum('b h i j d, b h j d -> b h i d', decay_mask, kv_i)
    o_intra = o_intra * q_i
    
    # 2. Inter-Chunk
    decay_from_start = torch.exp(g_cumsum)
    o_inter = S_i.unsqueeze(2) * decay_from_start * q_i
    
    o_chunk = o_intra + o_inter
    
    # 3. Compute New State
    # Only need the last row relative to current positions for state update
    total_decay = torch.exp(g_i.sum(dim=-2))
    
    # decay_to_end: decay from position j to the end of the chunk
    decay_to_end = torch.exp(g_cumsum[:, :, -1].unsqueeze(-2) - g_cumsum)
    current_chunk_contrib = (kv_i * decay_to_end).sum(dim=-2)
    
    S_new = S_i * total_decay + current_chunk_contrib
    
    return o_chunk, S_new

def lightnet_chunkwise(q, k, v, g, chunk_size=64, initial_state=None):
    """Optimized LightNet Chunkwise Logic.
    Optimizations:
    1. Gradient Checkpointing: Drastically reduces memory from O(L) to O(chunk_size).
    2. Lazy KV Computation: Avoids storing full KV tensor.
    3. Pre-allocated Mask: Reduces overhead.
    """
    b, h, l, d = q.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v, g = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v, g]]
        
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    g = rearrange(g, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, d, device=q.device)
        
    o_list = []
    
    # [Optimization] Pre-compute mask once
    # Mask is upper triangular with diagonal=1 (causal masking usually excludes future)
    # Your logic used: i - j. If i < j, it's future. 
    # Your original code: triu(ones, diagonal=1).masked_fill(-inf). 
    # This means i >= j is kept. Correct.
    mask_template = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    
    # Feature Block Size
    block_size = 32 
    
    for i in range(q.shape[2]): # Chunk Loop
        q_chunk = q[:, :, i]
        k_chunk = k[:, :, i] # [Optimization] Don't pre-calc kv
        v_chunk = v[:, :, i]
        g_chunk = g[:, :, i]
        
        o_chunk_slices = []
        S_new_slices = []
        
        for start_idx in range(0, d, block_size):
            end_idx = min(start_idx + block_size, d)
            
            # Slices (Views)
            q_i = q_chunk[..., start_idx:end_idx]
            k_i = k_chunk[..., start_idx:end_idx]
            v_i = v_chunk[..., start_idx:end_idx]
            g_i = g_chunk[..., start_idx:end_idx]
            S_i = S[..., start_idx:end_idx]
            
            # [Optimization] Use Checkpoint
            # Checkpointing trades compute (re-running this function in backward) for memory.
            # We pass S_i as input so gradients flow correctly to previous states.
            if torch.is_grad_enabled():
                o_slice, S_new_slice = checkpoint(
                    lightnet_chunk_step, 
                    q_i, k_i, v_i, g_i, S_i, mask_template,
                    use_reentrant=False # Recommended setting for modern PyTorch
                )
            else:
                o_slice, S_new_slice = lightnet_chunk_step(q_i, k_i, v_i, g_i, S_i, mask_template)
            
            o_chunk_slices.append(o_slice)
            S_new_slices.append(S_new_slice)
            
        # Concatenate slices
        S = torch.cat(S_new_slices, dim=-1)
        o_chunk = torch.cat(o_chunk_slices, dim=-1)
        o_list.append(o_chunk)
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    
    if pad_len > 0: o = o[..., :l, :]
    
    return o, S

class LightNet(nn.Module):
    def __init__(self, d_model, num_heads, expand_ratio=2, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = int(d_model * expand_ratio)
        self.head_dim = self.hidden_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        self.q_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)

        self.o_proj = SpikingLinear(nn.Linear(self.hidden_dim, d_model, bias=False), neuron_type, surrogate)
        self.norm = RMSNorm(self.hidden_dim)

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
        g = rearrange(g, 'b l (h d) -> b h l d', h=self.num_heads)

        g = F.logsigmoid(g) 

        last_state = None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = lightnet_chunkwise(q, k, v, g, initial_state=last_state)

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

# ==================== Block Wrapper ====================

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
        self.attn = LightNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
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

        # [MODIFIED] **kwargs ensures past_key_values and layer_idx are passed to self.attn
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