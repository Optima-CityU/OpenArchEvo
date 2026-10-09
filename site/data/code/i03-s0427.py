# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ==================== SNN Related Imports and Helper Classes ====================
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

# ==================== White-box Operator (Mamba2 SSD) ====================

def mamba2_ssd_parallel_stable(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, chunk_size: int=64, initial_state: Optional[torch.Tensor]=None):
    """Parallel Stable State Space Diffusion (SSD) operator, designed to receive binary spike inputs for q, k, v."""
    b, h, l, d = q.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v]]
        a = F.pad(a, (0, pad_len), value=0.0)
    
    q, k, v = [rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) for x in [q, k, v]]
    a_chunk = rearrange(a, 'b h (n c) -> b h n c', c=chunk_size)
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d, d, device=q.device)
    o = torch.zeros_like(v, dtype=torch.float) # Output accumulates analog values
    
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(0, num_chunks):
        q_i, k_i, v_i, a_i = q[:, :, i], k[:, :, i], v[:, :, i], a_chunk[:, :, i]
        
        # 1. Intra-chunk decay: a_i is already [B, H, C]
        a_i_cumsum = torch.cumsum(a_i, dim=-1)
        rel_cumsum = a_i_cumsum.unsqueeze(-1) - a_i_cumsum.unsqueeze(-2)
        
        # Numerical protection: clamp(max=0) completely prevents exp(inf)
        decay_rel = torch.exp(rel_cumsum.clamp(max=0.0)) * mask_tril # [B, H, C, C]
        
        # Attn_intra now computes co-firing patterns of binary spikes (q_i, k_i) scaled by decay
        attn_intra = (q_i.float() @ k_i.float().transpose(-1, -2)) * decay_rel # Cast to float for matmul
        o_intra = attn_intra @ v_i.float() # v_i is binary, cast to float for matmul
        
        # 2. Inter-chunk reading
        # Add minimal value protection to prevent gradient vanishing at exp(very small negative)
        decay_from_start = torch.exp(a_i_cumsum).unsqueeze(-1) # [B, H, C, 1]
        o_inter = (q_i.float() * (decay_from_start + 1e-8)) @ S
        
        o[:, :, i] = o_inter + o_intra
        
        # 3. State update
        decay_total = torch.exp(a_i.sum(dim=-1, keepdim=True)).unsqueeze(-1) # [B, H, 1, 1]
        decay_to_end = torch.exp(a_i_cumsum[..., -1].unsqueeze(-1) - a_i_cumsum).unsqueeze(-1) # [B, H, C, 1]
        
        delta_S = (k_i.float() * decay_to_end).transpose(-1, -2) @ v_i.float() # k_i, v_i are binary, cast to float for matmul
        S = S * (decay_total + 1e-8) + delta_S
        
        # Dynamic range protection
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l, :], S) if pad_len > 0 else (o, S)

# ==================== SpikingMamba2 Layer ====================

class SpikingMamba2(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand: int=2, d_state: int=64, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_inner = int(expand * d_model)
        self.num_heads = num_heads 
        self.d_head = self.d_inner // num_heads

        # Projected dimensions for z, x_in, B, C, dt
        self.proj_out_dim = self.d_inner * 4 + num_heads 
        surrogate = piecewise_quadratic_surrogate()

        self.in_proj = SpikingLinear(nn.Linear(d_model, self.proj_out_dim, bias=False), neuron_type, surrogate)

        # Neuron for input to convolution to ensure it processes spikes (x_in -> spikes -> conv)
        self.x_in_neuron_for_conv = neuron_registry[neuron_type](surrogate_function=surrogate)

        self.conv1d = ShortConvolution(self.d_inner, kernel_size=4, activation='silu')

        # SpikingLinear projections for q, k, v inputs to SSD operator
        # These layers take analog outputs from splits and *produce binary spikes* for `mamba2_ssd_parallel_stable`
        self.q_proj_ssd_spiking = SpikingLinear(nn.Linear(self.d_inner, self.d_inner, bias=False), neuron_type, surrogate)
        self.k_proj_ssd_spiking = SpikingLinear(nn.Linear(self.d_inner, self.d_inner, bias=False), neuron_type, surrogate)
        self.v_proj_ssd_spiking = SpikingLinear(nn.Linear(self.d_inner, self.d_inner, bias=False), neuron_type, surrogate)

        self.dt_bias = nn.Parameter(torch.ones(num_heads) * -2.0)
        self.D = nn.Parameter(torch.ones(num_heads))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = SpikingLinear(nn.Linear(self.d_inner, d_model, bias=False), neuron_type, surrogate)

        self.z_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Dict]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        all_spikes = {}
        projected_analog, s_in = self.in_proj(x)
        all_spikes['in_proj'] = s_in

        # Split analog projected output into components
        z_analog, x_in_analog_pre_conv, B_analog_pre_ssd, C_analog_pre_ssd, dt_analog = torch.split(
            projected_analog, [self.d_inner, self.d_inner, self.d_inner, self.d_inner, self.num_heads], dim=-1
        )

        # Apply neuron before convolution to ensure it processes spikes
        x_in_spikes_for_conv = self.x_in_neuron_for_conv(x_in_analog_pre_conv.transpose(-1, -2)).transpose(-1, -2)
        all_spikes['x_in_pre_conv_spikes'] = x_in_spikes_for_conv

        conv_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
             conv_state = past_key_values[self.layer_idx].get('conv_state')

        # Convolution operates on spike inputs
        x_in_conv_out, new_conv_state = self.conv1d(x_in_spikes_for_conv, cache=conv_state, output_final_state=True)

        # Generate q, k, v *binary spikes* for SSD operator from analog outputs
        # The first return value (analog output of SpikingLinear) is discarded for SSD input
        _, s_q_ssd_spikes = self.q_proj_ssd_spiking(C_analog_pre_ssd)
        _, s_k_ssd_spikes = self.k_proj_ssd_spiking(B_analog_pre_ssd)
        _, s_v_ssd_spikes = self.v_proj_ssd_spiking(x_in_conv_out) # v takes analog output of conv as input to its spiking linear

        all_spikes['q_ssd_spikes'] = s_q_ssd_spikes
        all_spikes['k_ssd_spikes'] = s_k_ssd_spikes
        all_spikes['v_ssd_spikes'] = s_v_ssd_spikes

        # Rearrange binary spike tensors for SSD operator
        v_binary = rearrange(s_v_ssd_spikes, 'b l (h d) -> b h l d', h=self.num_heads)
        k_binary = rearrange(s_k_ssd_spikes, 'b l (h d) -> b h l d', h=self.num_heads)
        q_binary = rearrange(s_q_ssd_spikes, 'b l (h d) -> b h l d', h=self.num_heads)

        a = -F.softplus(dt_analog + self.dt_bias) # Decay rates 'a' remain analog
        a = rearrange(a, 'b l h -> b h l')

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
            last_state = past_key_values[self.layer_idx].get('recurrent_state')

        # SSD operates with binary spike inputs for q, k, v
        y_ssd_out, final_state = mamba2_ssd_parallel_stable(q_binary, k_binary, v_binary, a, initial_state=last_state)

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values[self.layer_idx] = {'recurrent_state': final_state, 'conv_state': new_conv_state}

        # D parameter applied to binary spike 'v_binary' for residual connection. Ensure D*v_binary is float.
        y = y_ssd_out + (self.D.view(1, -1, 1, 1) * v_binary.float())
        y = rearrange(y, 'b h l d -> b l (h d)')

        # Z-gate operates on analog output from projection, produces binary spikes
        z_spikes = self.z_neuron(z_analog.transpose(-1, -2)).transpose(-1, -2)
        all_spikes['z_gate'] = z_spikes

        y = self.norm(y)
        y = y * z_spikes.float() # Apply binary z_spikes as a multiplicative gate. Ensure y * z_spikes is float.

        y_out_proj_analog, s_out = self.out_proj(y)
        all_spikes['out_proj'] = s_out

        return y_out_proj_analog, None, all_spikes

class SpikingMLP(nn.Module):
    """A Multi-Layer Perceptron block with SpikingLinear layers and spike-gated activation."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if surrogate_function is None:
            surrogate_function = piecewise_quadratic_surrogate()

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.activation = nn.GELU() # Re-introduced standard activation for MLP expressivity
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

        # Spike-gated activation modulation within MLP, based on previous MLP spike activity
        # This provides a dynamic, spike-driven modulation of the non-linearity.
        self.mlp_act_gate_proj = SpikingLinear(nn.Linear(1, d_ffn), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_out_analog, spike1 = self.fc1(x) # x_fc1_out_analog is analog, spike1 are binary

        # Calculate mean spike activity from the first MLP layer for gating
        mean_spike_activity_fc1 = spike1.float().mean(dim=-1, keepdim=True) # (B, L, 1)

        # Project this mean spike activity to create a dynamic gating factor for the activation
        mlp_act_gate_analog_output, mlp_act_gate_spikes = self.mlp_act_gate_proj(mean_spike_activity_fc1)

        # Apply sigmoid to normalize the gate output between 0 and 1 for soft modulation
        gating_factor = torch.sigmoid(mlp_act_gate_analog_output)

        # Apply the standard activation (GELU) and then gate its output based on spike activity
        x_act = self.activation(x_fc1_out_analog) * gating_factor
        x_after_dropout = self.dropout(x_act)

        x_fc2_out_analog, spike2 = self.fc2(x_after_dropout) # x_fc2_out_analog is analog, spike2 are binary

        # Return final analog output and all collected binary spikes
        return x_fc2_out_analog, spike1, spike2, mlp_act_gate_spikes

class SpikingFLABlock(nn.Module):
    """A spiking-enhanced FLABlock with Mamba2-style attention, MLP, and comprehensive spike-gating."""

    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # SpikingMamba2 is updated to incorporate deeper spiking integration
        self.attn = SpikingMamba2(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        # SpikingMLP now includes its own activation and internal spike-gating
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

        # Final neuron to generate spikes from the block's overall analog output
        self.final_block_neuron = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())

        # Spike-gating for attention residual connection, enabling spike-driven modulation of skip connections
        self.attn_spike_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, piecewise_quadratic_surrogate())

        # Spike-gating for MLP residual connection, similar to attention
        self.mlp_spike_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, piecewise_quadratic_surrogate())

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Dict]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        reg_loss = None # Initialize reg_loss as None

        x_res = x
        if self.prenorm:
            x = self.norm1(x)

        # SpikingMamba2 returns analog output, optional reg loss, and a dictionary of spikes
        x_attn_out, attn_reg_loss, attn_spikes = self.attn(x, past_key_values=past_key_values, use_cache=use_cache, **kwargs)
        if attn_reg_loss is not None:
            reg_loss = attn_reg_loss

        attn_block_spike_activity = None
        if attn_spikes and isinstance(attn_spikes, dict):
            # Collect all spikes from the attention mechanism
            for spike_tensor in attn_spikes.values():
                if spike_tensor is not None: # Ensure spike_tensor is not None before adding
                    all_spikes_list.append(spike_tensor)

            # Prioritize 'k_ssd_spikes' for attention residual gating, or fall back to any available spike activity
            k_spikes_for_gating = attn_spikes.get('k_ssd_spikes')
            if k_spikes_for_gating is not None:
                attn_block_spike_activity = k_spikes_for_gating.float().mean(dim=-1, keepdim=True) # (B, L, 1)
            else:
                for spike_name, spike_tensor in attn_spikes.items():
                    if spike_tensor is not None:
                        attn_block_spike_activity = spike_tensor.float().mean(dim=-1, keepdim=True)
                        break

        # Apply spike-dependent gating to the attention residual connection
        if attn_block_spike_activity is not None:
            attn_gate_analog_output, attn_gate_binary_spikes = self.attn_spike_gate_proj(attn_block_spike_activity)
            all_spikes_list.append(attn_gate_binary_spikes) # Collect spikes from the gate
            attn_gate = torch.sigmoid(attn_gate_analog_output) # Normalize gate output between 0 and 1
            x_attn_out = x_attn_out * attn_gate # Element-wise gating of attention output

        x = x_res + self.dropout1(x_attn_out)

        x_res = x
        if self.prenorm:
            x = self.norm2(x)

        # SpikingMLP now returns its final analog output and individual spike tensors for its layers, including the activation gate
        x_mlp_out, mlp_spike1, mlp_spike2, mlp_act_gate_spikes = self.mlp(x)

        mlp_spikes_for_gating_means = []
        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
            mlp_spikes_for_gating_means.append(mlp_spike1.float().mean(dim=-1, keepdim=True)) # (B, L, 1)
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)
            mlp_spikes_for_gating_means.append(mlp_spike2.float().mean(dim=-1, keepdim=True)) # (B, L, 1)
        if mlp_act_gate_spikes is not None:
            all_spikes_list.append(mlp_act_gate_spikes)
            mlp_spikes_for_gating_means.append(mlp_act_gate_spikes.float().mean(dim=-1, keepdim=True)) # Include for gating if desired

        # Apply spike-dependent gating to the MLP residual connection
        if mlp_spikes_for_gating_means:
            # Average the mean activities from fc1, fc2, and the activation gate for the MLP gate's input
            mlp_block_spike_activity = torch.mean(torch.stack(mlp_spikes_for_gating_means), dim=0) # (B, L, 1)
            mlp_gate_analog_output, mlp_gate_binary_spikes = self.mlp_spike_gate_proj(mlp_block_spike_activity)
            all_spikes_list.append(mlp_gate_binary_spikes) # Collect spikes from the gate
            mlp_gate = torch.sigmoid(mlp_gate_analog_output) # Normalize gate output between 0 and 1
            x_mlp_out = x_mlp_out * mlp_gate # Element-wise gating of MLP output

        x = x_res + self.dropout2(x_mlp_out)

        # Generate spikes from the final analog output of the block for the overall block's spike signal
        final_block_spikes = self.final_block_neuron(x.transpose(-1, -2)).transpose(-1, -2)
        all_spikes_list.append(final_block_spikes)

        final_spikes = None
        if all_spikes_list:
            # Concatenate all collected binary spike tensors into a single flattened tensor
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

        return x, reg_loss, final_spikes