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
    # neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None
# --- End of Placeholders ---

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_for_neuron = rearrange(x, 'b l d -> b d l') if x.dim() == 3 else x.unsqueeze(-1).transpose(-1, -2) # (B,L,D) -> (B,D,L) or (B,D) -> (B,D,1)

        spikes_from_neuron = self.neuron(x_for_neuron) # Spikes will be (B, D_in, L) or (B, D_in, 1)

        # The linear layer needs (..., D_in) for input.
        # If spikes_from_neuron is (B, D_in, L), rearrange to (B, L, D_in)
        # If spikes_from_neuron is (B, D_in, 1), squeeze to (B, D_in)
        if x.dim() == 3:
            spikes_for_linear = rearrange(spikes_from_neuron, 'b d l -> b l d') # (B, D, L) -> (B, L, D) # Binary spikes
        else: # original x was (B, D_in)
            spikes_for_linear = spikes_from_neuron.squeeze(-1) # (B, D_in, 1) -> (B, D_in)

        output = self.linear(spikes_for_linear) # The internal nn.Linear receives binary spikes
        return output, spikes_for_linear # output is analog, spikes are binary

# ================================================================

def softmax(x):
    return F.softmax(x, dim=-1)

def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    
    padded_len = l + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        # Use non-inplace operations to ensure autograd safety
        attn_intra = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn_intra @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')

    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

"""==================== White-box Operator (Mamba2 SSD) ===================="""

def mamba2_ssd_parallel_stable(q, k, v, a, chunk_size=64, initial_state=None):
    b, h, l, d = q.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v]]
        a = F.pad(a, (0, pad_len), value=0.0)
    
    q, k, v = [rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) for x in [q, k, v]]
    a_chunk = rearrange(a, 'b h (n c) -> b h n c', c=chunk_size)
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d, d, device=q.device)
    o = torch.zeros_like(v)
    
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, a_i = q[:, :, i], k[:, :, i], v[:, :, i], a_chunk[:, :, i]
        
        # 1. Intra-chunk decay: a_i is already [B, H, C]
        a_i_cumsum = torch.cumsum(a_i, dim=-1)
        rel_cumsum = a_i_cumsum.unsqueeze(-1) - a_i_cumsum.unsqueeze(-2)
        
        # Numerical protection: clamp(max=0) completely prevents exp(inf)
        decay_rel = torch.exp(rel_cumsum.clamp(max=0.0)) * mask_tril # [B, H, C, C]
        
        # [Fix] Directly utilize existing B, H dimensions for broadcasting
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_rel
        o_intra = attn_intra @ v_i
        
        # 2. Inter-chunk reading
        # Add minimal value protection to prevent gradient vanishing at exp(very small negative)
        decay_from_start = torch.exp(a_i_cumsum).unsqueeze(-1) # [B, H, C, 1]
        o_inter = (q_i * (decay_from_start + 1e-8)) @ S
        
        o[:, :, i] = o_inter + o_intra
        
        # 3. State update
        decay_total = torch.exp(a_i.sum(dim=-1, keepdim=True)).unsqueeze(-1) # [B, H, 1, 1]
        decay_to_end = torch.exp(a_i_cumsum[..., -1].unsqueeze(-1) - a_i_cumsum).unsqueeze(-1) # [B, H, C, 1]
        
        delta_S = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * (decay_total + 1e-8) + delta_S
        
        # Dynamic range protection
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l, :], S) if pad_len > 0 else (o, S)

"""==================== SpikingMamba2 Layer ===================="""

class SpikingMamba2(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand: int=2, d_state: int=64, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_inner = int(expand * d_model)
        self.num_heads = num_heads 
        self.d_head = self.d_inner // num_heads

        self.proj_out_dim = self.d_inner * 4 + num_heads
        surrogate = piecewise_quadratic_surrogate()

        self.in_proj = SpikingLinear(nn.Linear(d_model, self.proj_out_dim, bias=False), neuron_type, surrogate)
        self.conv1d = ShortConvolution(self.d_inner, kernel_size=4, activation='silu')

        # [NASWOT Fix]: Initialize dt_bias as negative to ensure initial system has long memory (exp(a) close to 1)
        self.dt_bias = nn.Parameter(torch.ones(num_heads) * -2.0)
        self.D = nn.Parameter(torch.ones(num_heads))

        # REMOVED: dt_spike_mod_proj and dt_scale_factor from Version 2.
        # NEW: Spike-driven multiplicative temporal gating for dt
        self.dt_gate_proj = SpikingLinear(nn.Linear(1, self.num_heads, bias=False), neuron_type, surrogate)
        self.dt_activation = nn.Sigmoid() # to keep gate between 0 and 1
        self.dt_epsilon = 1e-6 # For stability to avoid zero scaling

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = SpikingLinear(nn.Linear(self.d_inner, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}
        projected, s_in = self.in_proj(x)
        all_spikes['in_proj'] = s_in

        # Split projected output into z, x_in, B, C, dt_base
        z, x_in, B, C, dt_base = torch.split(projected, [self.d_inner, self.d_inner, self.d_inner, self.d_inner, self.num_heads], dim=-1)

        conv_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
            conv_state = past_key_values[self.layer_idx].get('conv_state')
        x_in, new_conv_state = self.conv1d(x_in, cache=conv_state, output_final_state=True)

        v = rearrange(x_in, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(B, 'b l (h d) -> b h l d', h=self.num_heads)
        q = rearrange(C, 'b l (h d) -> b h l d', h=self.num_heads)

        # Calculate mean spike activity from the input projection spikes
        mean_spike_activity = s_in.float().mean(dim=-1, keepdim=True) # (B, L, 1)

        # NEW: Spike-driven multiplicative temporal gating for dt
        # Project mean spike activity through a SpikingLinear layer to get dt gate
        dt_gate_analog, s_dt_gate = self.dt_gate_proj(mean_spike_activity)
        # Apply sigmoid to create a multiplicative factor between 0+epsilon and 1+epsilon
        dt_gate_factor = self.dt_activation(dt_gate_analog) + self.dt_epsilon # (B, L, num_heads)
        all_spikes['dt_gate_proj'] = s_dt_gate

        # Apply gate factor to dt_base, then add dt_bias
        # This makes the base decay rate dynamically scaled by input spike activity
        dt = (dt_base * dt_gate_factor) + self.dt_bias

        # Aligned with Mamba2: a = -softplus(dt + bias) # dt_bias is already included in dt above
        a = -F.softplus(dt)
        a = rearrange(a, 'b l h -> b h l')

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
            last_state = past_key_values[self.layer_idx].get('recurrent_state')

        y, final_state = mamba2_ssd_parallel_stable(q, k, v, a, initial_state=last_state)

        if past_key_values is not None and self.layer_idx is not None:
            # Ensure the cache is initialized if not present
            if len(past_key_values) <= self.layer_idx:
                past_key_values.append({})
            past_key_values[self.layer_idx]['recurrent_state'] = final_state
            past_key_values[self.layer_idx]['conv_state'] = new_conv_state

        y = y + (self.D.view(1, -1, 1, 1) * v) # self.D needs to be reshaped for broadcasting
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.norm(y)
        y = y * F.silu(z) 

        y, s_out = self.out_proj(y)
        all_spikes['out_proj'] = s_out

        return y, None, all_spikes

"""[SpikingMLP]"""

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x)
        # REMOVED: F.gelu(x) to emphasize spiking non-linearity and simplify MLP
        x, spike2 = self.fc2(self.dropout(x))
        return x, spike1, spike2

# New SpikingResidualScalarGate module (simplified from SpikingResidualGate)
# This version condenses the two-layer SpikingLinear MLP into a single SpikingLinear for efficiency.

class SpikingResidualScalarGate(nn.Module):
    def __init__(self, neuron_type: str, surrogate_function):
        super().__init__()
        # Single SpikingLinear layer that takes (B, L, 1) float input.
        # Its internal neuron processes this float input and produces binary spikes.
        # The internal nn.Linear then operates on these binary spikes to produce a continuous output (B, L, 1).
        self.gate_proj = SpikingLinear(nn.Linear(1, 1, bias=False), neuron_type, surrogate_function)
        # Learnable bias to control the base level of modulation
        self.bias = nn.Parameter(torch.zeros(1))
        # Add a learnable scale factor to provide more flexibility in modulation range
        self.scale_factor = nn.Parameter(torch.ones(1) * 2.0) # Initialize to 2.0 to allow for boosting up to 2x the residual

    def forward(self, avg_activity: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        raw_gate_output, spikes = self.gate_proj(avg_activity)

        # Apply sigmoid to raw output, then scale by learnable factor
        # This makes the modulation factor range from 0 to `self.scale_factor`
        modulation_factor = torch.sigmoid(raw_gate_output + self.bias) * self.scale_factor

        return modulation_factor, [spikes] # Return a list containing the single spike tensor

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingMamba2(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # Use the simplified SpikingResidualScalarGate for attention block
        self.attn_residual_gate = SpikingResidualScalarGate(neuron_type, piecewise_quadratic_surrogate())

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)
        # Use the simplified SpikingResidualScalarGate for MLP block
        self.mlp_residual_gate = SpikingResidualScalarGate(neuron_type, piecewise_quadratic_surrogate())

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []

        # --- Attention Block (SpikingMamba2) ---
        x_res = x
        if self.prenorm: x = self.norm1(x)

        # Pass past_key_values and layer_idx to SpikingMamba2 for internal state handling
        x_attn, _, attn_spikes_dict = self.attn(x, past_key_values=past_key_values, layer_idx=layer_idx, **kwargs)

        # Collect all spikes from attention block for final output
        if attn_spikes_dict:
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        # Determine the primary spike signal for residual gating for the attention block
        # Using 'in_proj' spikes as they represent the initial input activity to Mamba2
        primary_attn_spike_signal = attn_spikes_dict.get('in_proj') # This should be (B, L, D) or (B, D)

        if primary_attn_spike_signal is not None:
            # Ensure it has a sequence dimension (B, L, D) for mean calculation
            if primary_attn_spike_signal.dim() == 2: # (B, D) case
                primary_attn_spike_signal_reshaped = primary_attn_spike_signal.unsqueeze(1) # (B, 1, D)
            else: # (B, L, D) case
                primary_attn_spike_signal_reshaped = primary_attn_spike_signal

            avg_attn_mean_activity = primary_attn_spike_signal_reshaped.float().mean(dim=-1, keepdim=True) # (B, L, 1)

            # Get modulation factor and spikes from the new SpikingResidualScalarGate
            attn_modulation_factor, attn_gate_spikes = self.attn_residual_gate(avg_attn_mean_activity)
            all_spikes_list.extend(attn_gate_spikes) 

            # The multiplication x_attn * attn_modulation_factor will broadcast correctly if attn_modulation_factor is (B, L, 1)
            x_attn = x_attn * attn_modulation_factor 

        # Original residual connection after modulation (or no modulation if primary_attn_spike_signal is None)
        x = x_res + self.dropout1(x_attn)

        # --- MLP Block ---
        x_res = x
        if self.prenorm: x = self.norm2(x)

        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x)

        # Collect MLP spikes for final output
        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)

        # Determine the primary spike signal for residual gating for the MLP block
        # Using mlp_spike1 as it represents the first layer's activity in the MLP
        primary_mlp_spike_signal = mlp_spike1 

        if primary_mlp_spike_signal is not None:
            # Ensure it has a sequence dimension (B, L, D) for mean calculation
            if primary_mlp_spike_signal.dim() == 2: # (B, D) case
                primary_mlp_spike_signal_reshaped = primary_mlp_spike_signal.unsqueeze(1) # (B, 1, D)
            else: # (B, L, D) case
                primary_mlp_spike_signal_reshaped = primary_mlp_spike_signal

            avg_mlp_mean_activity = primary_mlp_spike_signal_reshaped.float().mean(dim=-1, keepdim=True) # (B, L, 1)

            # Get modulation factor and spikes from the new SpikingResidualScalarGate
            mlp_modulation_factor, mlp_gate_spikes = self.mlp_residual_gate(avg_mlp_mean_activity)
            all_spikes_list.extend(mlp_gate_spikes) 

            x_mlp = x_mlp * mlp_modulation_factor

        x = x_res + self.dropout2(x_mlp)

        # Final spike collection: flatten and concatenate all collected binary spikes
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else torch.empty(0, device=x.device, dtype=torch.bool)

        return x, None, final_spikes