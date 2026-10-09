# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026, Ruoyu Zhao
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, List
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ==================== SNN Base Components (Consistent with DeltaNet) ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return (input > 0).float()
        @staticmethod
        def backward(ctx, grad_output): return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: (lambda x: DummySurrogate.apply(x))}
    piecewise_quadratic_surrogate = lambda: None

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2) # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2) # (B, D, L) -> (B, L, D)
        output = self.linear(spikes)
        return output, spikes

# ==================== White-box MetaLA Operator (Parallel Chunk) ====================

def metala_chunk_parallel(q, k, v, gk, chunk_size=64, initial_state=None):
    """Parallel implementation of MetaLA core operator.
    Aligned with MetaLA logic: S_t = exp(gk_t) * S_{t-1} + k_t^T * v_t
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # 1. Padding handling
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        gk = F.pad(gk, (0, 0, 0, pad_len), value=-10.0) # Strong decay padding

    # 2. Reshape to Chunks
    q, k, v, gk = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, gk]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, gk_i = q[:, :, i], k[:, :, i], v[:, :, i], gk[:, :, i]
        
        # [Step 1] Compute intra-chunk decay matrix
        gk_cumsum = torch.cumsum(gk_i, dim=-2) # [B, H, C, Dk]
        
        # Fix 1: Apply mask before exp to avoid Inf
        decay_diff = gk_cumsum.unsqueeze(-2) - gk_cumsum.unsqueeze(-3)
        # Set unnecessary upper triangle to negative infinity, exp(-inf) = 0.0
        decay_diff = decay_diff.masked_fill(~mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).bool(), float('-inf'))
        decay_rel = torch.exp(decay_diff)

        # [Step 2] Intra-chunk parallel output
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_rel).sum(dim=-1) # [B, H, C, C]
        o_intra = attn_intra @ v_i
        
        # [Step 3] Inter-chunk history reading
        decay_from_start = torch.exp(gk_cumsum) # [B, H, C, Dk]
        o_inter = (q_i * decay_from_start) @ S
        
        o[:, :, i] = o_inter + o_intra
        
        # [Step 4] State update
        decay_total = torch.exp(gk_i.sum(dim=-2)).unsqueeze(-1) # [B, H, Dk, 1]
        
        decay_scalars = torch.exp(gk_cumsum[..., -1, :].unsqueeze(-2) - gk_cumsum) # [B, H, C, Dk]
        k_decayed = (k_i * decay_scalars).transpose(-1, -2) # [B, H, Dk, C]
        
        S = S * decay_total + (k_decayed @ v_i)
        
        # Fix 2: Remove S = torch.clamp(S, -10.0, 10.0)
        # Release gradient channel for ListOps long sequence backpropagation

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingMetaLA Token Mixer ====================

class SpikingMetaLA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, layer_idx: int=None, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layer_idx = layer_idx

        self.dk = d_model // 2
        self.head_dim = d_model // num_heads
        self.key_dim = self.dk // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # 1. Spiking Projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.dk, bias=False), neuron_type, surrogate)
        self.k_gate = SpikingLinear(nn.Linear(d_model, self.dk, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=True), neuron_type, surrogate)
        self.out_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)

        # 2. MetaLA-specific Self-Augmentation parameters
        self.aug_balance = nn.Parameter(torch.zeros(self.dk))

        # 3. Short Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            bias=False,
            kernel_size=4,
            groups=d_model,
            padding=3,
        )
        self.group_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}

        # [Step 1] Short Conv
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[..., :x.shape[-1]]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)

        # [Step 2] Spiking Projections
        q_raw, s_q = self.q_proj(x)
        kg_raw, s_kg = self.k_gate(x)
        v_raw, s_v = self.v_proj(x)
        g_raw, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k_gate': s_kg, 'v': s_v, 'g': s_g})

        # [Step 3] MetaLA Logic (Discretization)
        gk = F.logsigmoid(kg_raw) / 16.0
        k = 1.0 - torch.exp(gk)

        # Reshape for Multi-head
        q = rearrange(q_raw, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v_raw, 'b l (h d) -> b h l d', h=self.num_heads)
        gk = rearrange(gk, 'b l (h d) -> b h l d', h=self.num_heads)

        # [Step 4] Core Operator (Parallel Chunk)
        initial_state = None
        if past_key_values is not None and self.layer_idx in past_key_values:
            initial_state = past_key_values[self.layer_idx]

        o, final_state = metala_chunk_parallel(q, k, v, gk, initial_state=initial_state)

        if past_key_values is not None:
            past_key_values[self.layer_idx] = final_state

        # [Step 5] Self-Augmentation
        aug_balance = rearrange(self.aug_balance, '(h d) -> h d', h=self.num_heads)

        # [FIX]: Fix self-augmentation logic, use 'k' instead of 'q' to build augk for correct Q-K similarity bypass
        augk = k * aug_balance.unsqueeze(1) # [B, H, L, Dk] 
        aug_w = (q * augk).sum(dim=-1, keepdim=True) # [B, H, L, 1]

        o = o + torch.sigmoid(aug_w) * v

        # [Step 6] Output
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        output, s_o = self.out_proj(o * F.silu(g_raw))
        all_spikes['out'] = s_o

        return output, None, all_spikes

# ==================== SpikingMLP (Consistent with DeltaNet) ====================

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

# ==================== SpikingFLABlock (Unified Interface) ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingMetaLA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        all_spikes_list: List[torch.Tensor] = []

        # 1. Attention Path
        x_res = x
        if self.prenorm: x = self.norm1(x)
        x_attn, reg_loss, attn_spikes = self.attn(x, **kwargs)
        if attn_spikes: all_spikes_list.extend(attn_spikes.values())
        x = x_res + self.dropout1(x_attn)

        # 2. MLP Path
        x_res = x
        if self.prenorm: x = self.norm2(x)
        if isinstance(self.mlp, SpikingMLP):
            x_mlp, s1, s2 = self.mlp(x)
            all_spikes_list.extend([s1, s2])
        else:
            x_mlp = self.mlp(x)
        x = x_res + self.dropout2(x_mlp)

        # 3. Spikes Collection
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None]) if all_spikes_list else None

        return x, reg_loss, final_spikes