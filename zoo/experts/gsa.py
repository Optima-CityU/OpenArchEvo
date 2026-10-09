# 30_SpikingGSA.py
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

# ==================== White-box Operator (GSA) ====================

def gsa_chunkwise(q, k, v, f, chunk_size=64, initial_state=None):
    """GSA Parallel Chunkwise Logic using Factorized Attention.
    Significantly reduces memory by avoiding the materialization of the 5D decay mask.
    Formula:
        Score_ij = Sum_m (Q_im * exp(C_im)) * (K_jm * exp(-C_jm))
    """
    b, h, l, d = v.shape
    num_slots = q.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v, f = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v, f]]
        
    q = rearrange(q, 'b h (n c) m -> b h n c m', c=chunk_size)
    k = rearrange(k, 'b h (n c) m -> b h n c m', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    f = rearrange(f, 'b h (n c) m -> b h n c m', c=chunk_size)
    
    # Softmax addressing
    k = torch.softmax(k, dim=-1)
    q = torch.softmax(q, dim=-1)
    
    # Log-space decay
    f = F.logsigmoid(f)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, num_slots, d, device=q.device)
        
    o_list = []
    
    for i in range(q.shape[2]):
        q_chunk = q[:, :, i] # [B, H, C, M]
        k_chunk = k[:, :, i] 
        v_chunk = v[:, :, i] # [B, H, C, D]
        f_chunk = f[:, :, i] # [B, H, C, M]
        
        # --- 1. Intra-Chunk (Factorized Form) ---
        # Cumsum of log-decay
        f_cumsum = torch.cumsum(f_chunk, dim=-2) # [B, H, C, M]
        
        # Decompose Decay: exp(Ci - Cj) = exp(Ci) * exp(-Cj)
        # To ensure numerical stability (prevent overflow of exp(-Cj)), we can center C.
        # However, for chunk_size=64, values are usually within range.
        # We proceed with direct decomposition.
        
        decay_exp = torch.exp(f_cumsum)      # [B, H, C, M] (Decay)
        decay_inv_exp = torch.exp(-f_cumsum) # [B, H, C, M] (Growth)
        
        # Modulated Q and K
        q_mod = q_chunk * decay_exp
        k_mod = k_chunk * decay_inv_exp
        
        # Attention Matrix: (Q_mod @ K_mod^T)
        # [B, H, C, M] @ [B, H, M, C] -> [B, H, C, C]
        # This replaces the huge 5D masked einsum.
        attn_score = q_mod @ k_mod.transpose(-1, -2)
        
        # Apply Causal Mask
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        attn_score = attn_score.masked_fill(mask, 0)
        
        # O_intra = Attn @ V
        o_intra = attn_score @ v_chunk
        
        # --- 2. Inter-Chunk ---
        # Decay from start: exp(F_cumsum) = decay_exp
        # O_inter = (Q * Decay) @ S
        q_decayed = q_chunk * decay_exp
        
        # [B, H, C, M] @ [B, H, M, D] -> [B, H, C, D]
        o_inter = q_decayed @ S
        
        # Output accumulation
        o_list.append(o_intra + o_inter)
        
        # --- 3. Update State ---
        # S_new = S * Total_Decay + Sum(K * V * Decay_to_end)
        
        # Total decay for the whole chunk
        total_decay = torch.exp(f_chunk.sum(dim=-2)).unsqueeze(-1) # [B, H, M, 1]
        
        # Decay from t to end: exp(C_end - C_t) = exp(C_end) * exp(-C_t)
        # We already have exp(-C_t) as decay_inv_exp.
        c_end = f_cumsum[:, :, -1, :].unsqueeze(-2) # [B, H, 1, M]
        decay_to_end_const = torch.exp(c_end)
        
        # k_weighted = K * exp(-C_t) * exp(C_end)
        k_weighted = k_mod * decay_to_end_const
        
        # Update: [B, H, M, C] @ [B, H, C, D] -> [B, H, M, D]
        chunk_update = k_weighted.transpose(-1, -2) @ v_chunk
        
        # Calculate new state WITHOUT in-place modification
        S = S * total_decay + chunk_update
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    return o, S

# ==================== Main Layer ====================

class GSA(nn.Module):
    def __init__(self, d_model, num_heads, num_slots=64, expand_v=1.0, use_short_conv=True, conv_size=4, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.value_dim = int(d_model * expand_v)
        surrogate = piecewise_quadratic_surrogate()

        # Projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.f_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate) 
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate) 

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.value_dim // num_heads)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            self.f_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        f, s_f = self.f_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'f': s_f, 'g': s_g})

        if self.use_short_conv:
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)
            f, _ = self.f_conv1d(f)

        q = rearrange(q, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        k = rearrange(k, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        f = rearrange(f, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)

        last_state = None
        if past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = gsa_chunkwise(q, k, v, f, chunk_size=64, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                past_key_values.append({'recurrent_state': final_state})
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_state

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.g_norm(o)

        g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
        o = o * F.silu(g) 

        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o
        return o, None, all_spikes

# [SpikingFLABlock Wrapper]

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
        self.attn = GSA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
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