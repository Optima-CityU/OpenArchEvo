"""Implements a SpikingFLABlock with enforced spiking MLP and re-introduced spike activity regularization.
"""
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ==================== SNN Related Imports and Helper Classes ====================
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
# --- End of Placeholders ---

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D) # Binary spikes
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

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
    
    for i in range(q.shape[2]): # Iterate through chunks
        q_chunk = q[:, :, i] # [B, H, C, M]
        k_chunk = k[:, :, i] 
        v_chunk = v[:, :, i] # [B, H, C, D]
        f_chunk = f[:, :, i] # [B, H, C, M]
        
        # --- 1. Intra-Chunk (Factorized Form) ---
        # Cumsum of log-decay
        f_cumsum = torch.cumsum(f_chunk, dim=-2) # [B, H, C, M]
        
        # Decompose Decay: exp(Ci - Cj) = exp(Ci) * exp(-Cj)
        decay_exp = torch.exp(f_cumsum)      # [B, H, C, M] (Decay up to current element)
        decay_inv_exp = torch.exp(-f_cumsum) # [B, H, C, M] (Inverse decay up to current element)
        
        # Modulated Q and K
        q_mod = q_chunk * decay_exp
        k_mod = k_chunk * decay_inv_exp
        
        # Attention Matrix: (Q_mod @ K_mod^T)
        # [B, H, C, M] @ [B, H, M, C] -> [B, H, C, C]
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
        # S_new = S * Total_Decay_for_chunk + Sum(K * V * Decay_from_t_to_end_of_chunk)
        
        # Total decay for the whole chunk
        # The decay is applied *after* the chunk processing, so it decays the previous state. 
        # The decay for the whole chunk is exp(sum of all f_chunk values for this chunk).
        # f_cumsum[:, :, -1, :] contains the total cumulative log decay for the chunk.
        total_decay_per_slot = torch.exp(f_cumsum[:, :, -1, :]).unsqueeze(-2) # [B, H, 1, M]
        
        # We need the sum of k_i * v_i for the chunk, modulated by decay to end of chunk.
        # This needs careful calculation. Let's re-examine the definition in the paper.
        # The original paper (FLA's DeltaNet, which GSA is based on) usually uses a simple update.
        # Re-using the logic from the provided Version 2, the state update is:
        # S = S * total_decay + chunk_update
        # where chunk_update is sum(k_mod_element * v_element) with appropriate decays.
        
        # For each element t in chunk: k_t_mod = k_t * exp(-F_t)
        # For the state update, we need sum_{t in chunk} (k_t * exp(F_chunk_end - F_t) * v_t)
        # k_mod already has k_t * exp(-F_t)
        # So we need to multiply by exp(F_chunk_end)
        
        c_end_per_slot = f_cumsum[:, :, -1, :].unsqueeze(-2) # [B, H, 1, M]
        decay_to_end_const_per_slot = torch.exp(c_end_per_slot) # [B, H, 1, M]
        
        # k_weighted = K_mod * exp(F_chunk_end) = (K * exp(-F_t)) * exp(F_chunk_end) 
        k_weighted_for_state_update = k_mod * decay_to_end_const_per_slot
        
        # Update: [B, H, M, C] @ [B, H, C, D] -> [B, H, M, D]
        chunk_update = k_weighted_for_state_update.transpose(-1, -2) @ v_chunk
        
        # S_new = S_old * total_decay_of_current_chunk + (sum of contributions from current chunk)
        # The total decay of current chunk on the *previous* state `S` is exp(sum of f_chunk elements).
        # The `total_decay` from Version 2 calculation was: `torch.exp(f_chunk.sum(dim=-2)).unsqueeze(-1)`
        # This `f_chunk.sum(dim=-2)` corresponds to the total log-decay across the sequence length dimension (C) for each slot (M).
        # So `total_decay` is [B, H, M, 1]

        # Let's verify the `total_decay` variable from Version 2. It was `f_chunk.sum(dim=-2)` which is a sum over `c` dimension for each `m`.
        # This `total_decay` has shape `[B, H, M]`. 
        # In `S = S * total_decay + chunk_update`, `S` is [B, H, M, D], `total_decay` is [B, H, M, 1]. So `total_decay` applies element-wise for each `m` across all `d`.

        # Version 2's total_decay calculation:
        # total_decay = torch.exp(f_chunk.sum(dim=-2)).unsqueeze(-1) # [B, H, M, 1]
        # This is correct if f_chunk is [B,H,C,M] and sum over C gives [B,H,M]. Then unsqueeze(-1) gives [B,H,M,1].
        # This `total_decay` is a vector of M values (per head per batch) that decays the M slots of the state S.
        
        current_chunk_total_decay_factor = torch.exp(f_chunk.sum(dim=-2)).unsqueeze(-1) # [B, H, M, 1]

        # Calculate new state WITHOUT in-place modification
        S = S * current_chunk_total_decay_factor + chunk_update
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :] # Remove padding
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

        # All projections use SpikingLinear as per requirements
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
            # ShortConvolution layers are not spiking as per hybrid design allowance
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)
            f, _ = self.f_conv1d(f)

        # Rearrange for multi-head processing
        q = rearrange(q, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        k = rearrange(k, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        f = rearrange(f, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)

        last_state = None
        if past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 # Extract recurrent state from cache
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
        o = o * F.silu(g) # Gating mechanism with SiLU activation

        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o
        return o, None, all_spikes # Returns analog output, None for reg_loss, and collected spikes

# [SpikingFLABlock Wrapper]

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        # Ensures that both linear layers are spiking-enabled.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x)
        x = F.gelu(x) # Analog activation function after the first spiking linear layer
        x, spike2 = self.fc2(self.dropout(x))
        return x, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', spike_reg_weight: float=0.01, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # GSA also uses SpikingLinear layers internally.
        self.attn = GSA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        # SpikingMLP is now mandatory, ensuring all linear layers here are spiking-enabled.
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

        # Spike regularization weight for the block, used for all collected spikes.
        self.spike_reg_weight = spike_reg_weight

    def forward(self, x, **kwargs):
        all_spikes_list: List[torch.Tensor] = []
        total_reg_loss = None # Initialize, will be computed if spike_reg_weight > 0

        x_res = x
        if self.prenorm: 
            x = self.norm1(x)

        # GSA returns analog output and a dictionary of collected spikes.
        x_attn, _, attn_spikes = self.attn(x, **kwargs)

        # Collect spikes from the attention mechanism
        if attn_spikes and isinstance(attn_spikes, dict):
            for spike_tensor in attn_spikes.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        x = x_res + self.dropout1(x_attn)

        x_res = x
        if self.prenorm: 
            x = self.norm2(x)

        # SpikingMLP returns its final analog output and individual spike tensors for its two layers.
        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x)

        # Collect spikes from the MLP
        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)

        x = x_res + self.dropout2(x_mlp)

        final_spikes = None
        # Calculate spike activity regularization for the entire block based on all collected spikes
        if all_spikes_list:
            # Concatenate all collected binary spike tensors into a single flattened tensor.
            # Ensure each tensor in all_spikes_list is not empty before flattening and concatenating.
            non_empty_spikes = [s.flatten() for s in all_spikes_list if s is not None and s.numel() > 0]
            if non_empty_spikes:
                final_spikes = torch.cat(non_empty_spikes)

                # Apply block-level spike regularization if weight > 0
                if self.spike_reg_weight > 0 and final_spikes.numel() > 0:
                    spike_activity_mean = torch.mean(final_spikes.float())
                    total_reg_loss = spike_activity_mean * self.spike_reg_weight

        return x, total_reg_loss, final_spikes