# 37_SpikingMamba.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
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

# ==================== White-box Operator (Mamba S6 Scan) ====================

def mamba_chunk_scan_parallel(x, dt, A, B, C, D, chunk_size=64, initial_state=None):
    """Mamba parallel operator based on log-space prefix sum.
    Reduces memory overhead from O(C^2) to O(C), eliminating OOM.
    """
    b, l, d_inner = x.shape
    d_state = A.shape[1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        x, dt, B, C = [F.pad(t, (0, 0, 0, pad_len)) for t in [x, dt, B, C]]

    # 1. Discretization: dA = exp(dt * A), dBx = (dt * x) * B
    log_dA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(1) # [B, L, D, N]
    dBx = (dt * x).unsqueeze(-1) * B.unsqueeze(2)           # [B, L, D, N]

    # 2. Core: use prefix sum formula h_t = exp(L_t) * (h_0 + sum(dBx_j * exp(-L_j)))
    # This approach completely avoids constructing [C, C] interaction matrix
    log_dA_cumsum = torch.cumsum(log_dA, dim=1)
    
    # For numerical stability, subtract max value to prevent exp overflow
    # h_t = dA_t * h_{t-1} + dBx_t
    # We process in chunks to match your past_key_values logic
    log_dA_chunk = rearrange(log_dA_cumsum, 'b (n c) d k -> b n c d k', c=chunk_size)
    dBx_chunk = rearrange(dBx, 'b (n c) d k -> b n c d k', c=chunk_size)
    x_chunk = rearrange(x, 'b (n c) d -> b n c d', c=chunk_size)
    C_chunk = rearrange(C, 'b (n c) k -> b n c k', c=chunk_size)
    
    num_chunks = log_dA_chunk.shape[1]
    h = initial_state if initial_state is not None else torch.zeros(b, d_inner, d_state, device=x.device)
    y_chunks = []

    for i in range(num_chunks):
        # Current chunk's relative log decay
        # We need intra-chunk relative offset: L_rel = L_total - L_at_start_of_chunk
        l_offset = log_dA_cumsum[:, i*chunk_size : (i+1)*chunk_size]
        if i > 0:
            l_prev = log_dA_cumsum[:, i*chunk_size - 1].unsqueeze(1)
            l_rel = l_offset - l_prev
        else:
            l_rel = l_offset
            
        # [White-box formula] h_t = exp(L_rel_t) * h_0 + sum_{j <= t} (dBx_j * exp(L_rel_t - L_rel_j))
        exp_l_rel = torch.exp(l_rel)
        
        # Compute intra-chunk state
        # The trick here: sum_{j <= t} (dBx_j / exp(L_rel_j)) * exp(L_rel_t)
        # Add epsilon to prevent division by zero, though exp is always > 0
        dBx_weighted = dBx_chunk[:, i] / (exp_l_rel + 1e-8)
        cum_dBx = torch.cumsum(dBx_weighted, dim=1)
        
        # Current chunk all timestep states
        current_states = exp_l_rel * (h.unsqueeze(1) + cum_dBx)
        
        # Compute output
        y_i = torch.sum(current_states * C_chunk[:, i].unsqueeze(2), dim=-1) + D * x_chunk[:, i]
        y_chunks.append(y_i)
        
        # Update cross-chunk state
        h = current_states[:, -1]
        # SNN stability clamp
        h = torch.clamp(h, -10.0, 10.0)

    y = torch.cat(y_chunks, dim=1)
    return (y[:, :l, :], h) if pad_len > 0 else (y, h)

# ==================== SpikingMamba Standard Wrapper ====================

class SpikingMamba(nn.Module):
    def __init__(self, d_model: int=1024, d_state: int=16, d_conv: int=4, expand: int=2, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = (self.d_inner + 15) // 16

        surrogate = piecewise_quadratic_surrogate()

        self.in_proj = SpikingLinear(nn.Linear(d_model, self.d_inner * 2, bias=False), neuron_type, surrogate)
        self.conv1d = ShortConvolution(self.d_inner, kernel_size=d_conv, activation='silu')

        self.x_proj = SpikingLinear(nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False), neuron_type, surrogate)
        self.dt_proj = SpikingLinear(nn.Linear(self.dt_rank, self.d_inner, bias=True), neuron_type, surrogate)

        # Initialize A matrix
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = SpikingLinear(nn.Linear(self.d_inner, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}

        x_and_z, s_in = self.in_proj(x)
        all_spikes['in_proj'] = s_in
        z, x = x_and_z.chunk(2, dim=-1)

        conv_state = None
        if past_key_values is not None and self.layer_idx in past_key_values:
             conv_state = past_key_values[self.layer_idx].get('conv_state')

        x, new_conv_state = self.conv1d(x, cache=conv_state, output_final_state=True)

        x_dbl, s_x = self.x_proj(x)
        all_spikes['x_proj'] = s_x
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt, s_dt = self.dt_proj(dt)
        all_spikes['dt_proj'] = s_dt
        dt = F.softplus(dt) 

        last_state = None
        if past_key_values is not None and self.layer_idx in past_key_values:
            last_state = past_key_values[self.layer_idx].get('recurrent_state')

        A = -torch.exp(self.A_log.float())
        y, final_state = mamba_chunk_scan_parallel(x, dt, A, B, C, self.D, initial_state=last_state)

        if past_key_values is not None:
            past_key_values[self.layer_idx] = {'recurrent_state': final_state, 'conv_state': new_conv_state}

        y = y * F.silu(z)
        y, s_out = self.out_proj(y)
        all_spikes['out_proj'] = s_out

        return y, None, all_spikes

# [SpikingMLP and SpikingFLABlock]

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
        self.attn = SpikingMamba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
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