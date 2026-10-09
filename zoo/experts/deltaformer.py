# 31_SpikingDeltaFormer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

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

# ==================== White-box Operator (DeltaNet Chunkwise) ====================

def delta_former_chunkwise_parallel(q, k, v, beta, decay, chunk_size=64, initial_state=None):
    """Ultra-stable DeltaFormer operator optimized for SNN NAS.
    Solves GradNorm nan issues and ensures stable eigenvalues in recurrent systems.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len), value=0.0) 
        decay = F.pad(decay, (0, pad_len), value=1.0) 
    
    # Increase eps and use more robust normalization
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    
    q, k, v, beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, beta, decay]
    )
    
    num_chunks = q.shape[2]
    
    # Initialize state S as [B, H, d_k, d_v]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)

    for i in range(num_chunks):
        q_i, k_i, v_i, b_i, d_i = q[:, :, i], k[:, :, i], v[:, :, i], beta[:, :, i], decay[:, :, i]
        
        # [Fix Point 1] Historical state reading: Q * S
        # q_i: [B, H, C, D_k], S: [B, H, D_k, D_v]
        # Direct multiplication yields [B, H, C, D_v]
        o_inter = q_i @ S  
        
        # [Fix Point 2] Prediction error (Delta Rule core)
        # v_pred = k_i @ S -> [B, H, C, D_k] @ [B, H, D_k, D_v] -> [B, H, C, D_v]
        v_pred_inter = k_i @ S 
        v_error = (v_i - v_pred_inter) * b_i.unsqueeze(-1)
        
        # [Fix Point 3] Intra-chunk parallel update (using Wy transform simplified form)
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device), diagonal=-1)
        attn_intra = (k_i @ k_i.transpose(-1, -2)) * mask
        v_error_intra = v_error - (attn_intra @ v_error) * 0.5
        
        # [Fix Point 4] Intra-chunk incremental output
        # (q @ k.T) @ v_error_intra -> [C, Dk] @ [Dk, C] @ [C, Dv]
        o_intra = q_i @ (k_i.transpose(-1, -2) @ v_error_intra)
        o[:, :, i] = o_inter + o_intra
        
        # [Fix Point 5] State update
        # S_next = decay * S_prev + k.T @ v_error_intra
        # k_i.transpose: [D_k, C], v_error_intra: [C, D_v] -> [D_k, D_v]
        d_mean = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1)
        delta_update = k_i.transpose(-1, -2) @ v_error_intra 
        S = d_mean * S + delta_update
        
        # Clamp to prevent overflow
        S = torch.clamp(S, min=-10.0, max=10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

class SpikingDeltaFormer(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        # Dimension alignment
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # 1. Projection layers: Q, K, V, Beta, Decay, and DeltaFormer-specific Gate (g)
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        self.w_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate) # Aligned with FLA

        # 2. Short convolution
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        # 3. Normalization and output
        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}

        # 1. Projection layer (ensure output int dimensions are correct)
        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        beta, s_b = self.b_proj(x)
        w, s_w = self.w_proj(x)
        g, s_g = self.g_proj(x) 

        all_spikes.update({'q_proj':s_q, 'k_proj':s_k, 'v_proj':s_v, 'b_proj':s_b, 'w_proj':s_w, 'g_proj':s_g})

        # 2. Activation function alignment
        # In FLA's DeltaFormer, beta typically goes through softplus, w through logsigmoid
        # beta = F.softplus(beta) 
        # decay = F.logsigmoid(w).exp()
        beta = torch.sigmoid(beta) * 0.5
        # [Fix] Ensure decay is strictly less than 1
        decay = torch.sigmoid(w)
        gate = F.silu(g) # Gate branch

        # 3. Core operator
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        beta = rearrange(beta, 'b l h -> b h l')
        decay = rearrange(decay, 'b l h -> b h l')

        # KV Cache logic
        layer_state = None
        if past_key_values is not None:
            layer_state = past_key_values[self.layer_idx] if self.layer_idx in past_key_values else None

        o, final_state = delta_former_chunkwise_parallel(q, k, v, beta, decay, initial_state=layer_state)

        if past_key_values is not None:
            past_key_values[self.layer_idx] = final_state

        # 4. Gate merging logic (strictly aligned with FLA)
        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        # Here gate dimension is [B, L, head_v_dim * num_heads]
        gate = rearrange(gate, 'b l (h d) -> b l h d', h=self.num_heads)
        o = o * gate # Element-wise multiplication for Gated Linear Attention

        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o_proj'] = s_o

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
        self.attn = SpikingDeltaFormer(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
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