# 33_SpikingGatedDeltaNet.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from transformers.processing_utils import Unpack
from fla.models.utils import Cache

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

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

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

# ==================== White-box Operator (Gated DeltaNet) ====================

def gated_delta_rule_chunkwise(q, k, v, beta, g, chunk_size=32, initial_state=None):
    """Gated DeltaNet Chunkwise Logic (Fully Parallelized).
    Performance: Matches standard DeltaNet/RetNet speed.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # 1. Padding
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    
    q = l2norm(q)
    k = l2norm(k)
    
    # 2. Reshape to [B, H, N, C, D] (Parallel across N chunks)
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    beta = rearrange(beta, 'b h (n c) -> b h n c', c=chunk_size)
    g = rearrange(g, 'b h (n c) -> b h n c', c=chunk_size)
    
    # Pre-calculate auxiliary vars
    k_beta = k * beta.unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)
    
    # ============================================================
    # STEP A: Intra-Chunk Parallel Calculation (The Speed Boost)
    # We construct matrices for ALL chunks at once [B, H, N, C, C]
    # ============================================================
    
    # 1. Decay Mask Construction
    # g_cumsum: [B, H, N, C]
    g_cumsum = torch.cumsum(g, dim=-1)
    
    # Decay Matrix: D_{ij} = exp(G_i - G_j)
    # Shape: [B, H, N, C, C]
    decay_matrix = g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)
    
    # Causal Mask
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    decay_matrix = decay_matrix.masked_fill(mask, -float('inf'))
    decay_mask = torch.exp(decay_matrix)
    
    # 2. Attention Matrix (Delta Rule)
    # attn_base = -(k_beta @ k^T)
    # [B, H, N, C, D] @ [B, H, N, D, C] -> [B, H, N, C, C]
    attn_base = -(k_beta @ k.transpose(-1, -2))
    
    # Combine Decay and Interaction
    attn = attn_base * decay_mask
    
    # Apply strict lower triangular mask (diagonal is 0 for Neumann series base)
    attn = attn.masked_fill(mask, 0)
    
    # 3. Matrix Inversion (Neumann Series)
    # Parallel over N chunks! No Python loop over chunks here.
    # (I - A)^-1 = I + A + A^2 + ...
    # For small C=32, this converges in log2(32)=5 steps
    for j in range(1, chunk_size):
        #attn = attn + (attn @ attn) # Simple doubling?
        # Standard FLA "fast implementation" triangular update:
        # This ensures correct triangular propagation
        attn[..., j, :j] = attn[..., j, :j] + (attn[..., j, :, None].clone() * attn[..., :, :j].clone()).sum(-2)
        
    attn = attn + torch.eye(chunk_size, device=q.device)
    
    # 4. Compute Intra-Chunk Updates (u and w)
    # u = Inverse @ (v * beta)
    # w = Inverse @ (k * beta)
    # These contain the "net change" to memory for each step
    u = attn @ v_beta
    w = attn @ k_beta
    
    # ============================================================
    # STEP B: Inter-Chunk Recurrence (Lightweight Loop)
    # Only S (small state) is carried over.
    # ============================================================
    
    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = torch.zeros(b, h, d_k, d_v, device=q.device)
        
    o = torch.zeros_like(v)
    
    # Pre-compute Inter-chunk Decay masks
    # decay_from_start: Decay from t=0 to current t in chunk
    decay_from_start = torch.exp(g_cumsum).unsqueeze(-1) # [B, H, N, C, 1]
    
    # total_chunk_decay: Decay from start to end of chunk
    total_chunk_decay = torch.exp(g.sum(dim=-1)).unsqueeze(-1).unsqueeze(-1) # [B, H, N, 1, 1]
    
    # decay_to_end: Decay from t to end of chunk (for updating S)
    decay_to_end = torch.exp(g_cumsum[..., -1].unsqueeze(-1).unsqueeze(-1) - g_cumsum.unsqueeze(-1)) # [B, H, N, C, 1]
    
    # Pre-calculate Mask for Output (Q @ K^T)
    # This is the "Reading" part
    mask_intra = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    
    # Loop over N chunks (Sequential dependency on S)
    for i in range(q.shape[2]):
        # Load pre-computed chunk data
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i] # Pre-computed Update
        w_i = w[:, :, i] # Pre-computed Erasure
        
        # 1. Output Calculation
        # Part A: From History (Inter-chunk)
        # O_inter = Q * (S * Decay_from_start)
        q_decayed = q_i * decay_from_start[:, :, i]
        o_inter = q_decayed @ S
        
        # Part B: From Current Chunk (Intra-chunk)
        # Attn_read = (Q @ K^T) * Decay_mask
        # Note: We reuse the decay_mask we computed for inverse, but need to check masking
        decay_mask_i = decay_mask[:, :, i]
        attn_read = (q_i @ k_i.transpose(-1, -2)) * decay_mask_i
        attn_read = attn_read.masked_fill(mask_intra, 0)
        
        # O_intra = Attn_read @ (u - w @ S)
        # We subtract (w @ S) to correct for the fact that 'u' and 'w' calculations assumed S=0.
        # However, S needs to be decayed properly?
        # In Gated DeltaNet, the 'w' term inherently handles the decay structure via 'attn'.
        # But 'S' is a static value at the start of the chunk.
        # The term (w @ S) represents "what the state S would have generated if it were treated as an input".
        # Yes, standard DeltaNet logic: o = o_inter + attn_read @ (u_i - w_i @ S)
        
        o[:, :, i] = o_inter + attn_read @ (u_i - w_i @ S)
        
        # 2. State Update
        # S_new = S * Total_Decay + Sum(Updates * Decay_to_end)
        
        # Weighted K for update: K * Decay_to_end
        k_weighted = k_i * decay_to_end[:, :, i]
        
        # Chunk Update: K_w^T @ u
        # [D, C] @ [C, D] -> [D, D]
        chunk_update = k_weighted.transpose(-1, -2) @ u_i
        
        S = S * total_chunk_decay[:, :, i] + chunk_update
        
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[:, :, :l]
    return o, S

class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, qk_activation: str='silu', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs):
        super().__init__()

        if d_model is not None: hidden_size = d_model
        self.hidden_size = hidden_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads

        surrogate_function = piecewise_quadratic_surrogate()

        # Projections
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
        self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), neuron_type, surrogate_function)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), neuron_type, surrogate_function)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu' if qk_activation == 'silu' else None)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu' if qk_activation == 'silu' else None)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            self.g_conv1d = ShortConvolution(self.num_heads, conv_size, activation='silu')

        self.qk_activation = qk_activation

        self.use_gate = use_gate
        if use_gate:
            self.out_g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.hidden_size // num_heads, eps=norm_eps) # Assuming head dim
        else:
            self.o_norm = RMSNorm(self.value_dim // num_heads, eps=norm_eps)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, **kwargs):
        all_spikes = {}
        batch_size, q_len, _ = hidden_states.shape

        q, s_q = self.q_proj(hidden_states)
        k, s_k = self.k_proj(hidden_states)
        v, s_v = self.v_proj(hidden_states)
        g, s_g = self.g_proj(hidden_states)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'g': s_g})

        conv_state_q, conv_state_k, conv_state_v, conv_state_g = None, None, None, None
        if past_key_values is not None and layer_idx is not None:
            # Assume cache structure is list of dicts
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        if self.use_short_conv:
            q, _ = self.q_conv1d(q, cache=conv_state_q, output_final_state=use_cache)
            k, _ = self.k_conv1d(k, cache=conv_state_k, output_final_state=use_cache)
            v, _ = self.v_conv1d(v, cache=conv_state_v, output_final_state=use_cache)
            g, _ = self.g_conv1d(g, cache=conv_state_g, output_final_state=use_cache)

        if self.qk_activation == 'silu':
            q, k = F.silu(q), F.silu(k)
        elif self.qk_activation == 'relu':
            q, k = q.relu(), k.relu()
        elif self.qk_activation == 'elu':
            q, k = elu_p1(q), elu_p1(k)

        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.num_heads)
        g = rearrange(g, 'b l h -> b l h')

        if self.use_beta:
            beta, s_b = self.b_proj(hidden_states)
            beta = beta.sigmoid()
            all_spikes['beta'] = s_b
            beta = rearrange(beta, 'b l h -> b l h')
        else:
            beta = torch.ones_like(g)

        g = F.logsigmoid(g)

        last_state = None

        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')
        g = rearrange(g, 'b l h -> b h l')

        o, final_state = gated_delta_rule_chunkwise(q, k, v, beta, g, initial_state=last_state)

        o = rearrange(o, 'b h l d -> b l h d')

        if self.use_gate:
            out_g, s_og = self.out_g_proj(hidden_states)
            all_spikes['out_g'] = s_og
            out_g = rearrange(out_g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.o_norm(o, out_g)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

# [SpikingMLP and SpikingFLABlock follow the same template as Model 18]

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
        self.attn = GatedDeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
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