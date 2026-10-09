"""Implements a SpikingFLABlock with the Comba attention mechanism, enhanced for stricter neuromorphic integration.
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
    """A sequence module containing LIF neurons and a linear layer.
    The neuron spikes the input, and the linear layer operates on these binary spikes.
    The output of the linear layer is analog.
    """

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        # Neuron applies thresholding along the last dimension of the transposed input
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes = self.neuron(x.transpose(-1, -2)).transpose(-1, -2) # Binary spikes
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes, output is analog
        return output, spikes # output is analog, spikes are binary

# ==================== White-box Operator (Comba Closed-Loop) ====================

def comba_chunkwise_parallel(q, k, v, g, feedback_net, chunk_size=64, initial_state=None):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # Pad to be divisible by chunk_size
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len), value=0.0)
    
    # Normalization on analog representations of q and k
    q = l2norm(q)
    k = l2norm(k)
    
    # Rearrange into chunks
    q, k, v, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, g]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)

    all_feedback_spikes_internal = []
    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        
        # Closed-loop feedback
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        s_feat = rearrange(S / s_norm, 'b h d_k d_v -> b h (d_k d_v)').mean(dim=-1, keepdim=True) # [B, H, 1]
        
        # Call feedback_net and collect its internal spikes
        f_i_analog, f_i_spikes_list = feedback_net(s_feat) # returns analog output and a list of spikes
        all_feedback_spikes_internal.extend(f_i_spikes_list) # Collect internal spikes from feedback_net
        f_i = f_i_analog.squeeze(-1) # Use analog output for computation
        
        # Gate numerical computation alignment using logsigmoid for decay factors
        log_decay = F.logsigmoid(g_i + rearrange(f_i, 'b h -> b h 1')) # [B, H, C]
        d_cumsum = torch.cumsum(log_decay, dim=-1)
        
        decay_rel = rearrange(d_cumsum, 'b h c -> b h c 1') - rearrange(d_cumsum, 'b h c -> b h 1 c')
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0)
        
        # Compute output
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        attn_intra = (q_i @ rearrange(k_i, 'b h c d -> b h d c')) * decay_mask
        o_intra = attn_intra @ v_i
        
        o[:, :, i] = o_inter + o_intra
        
        # State update stability enhancement
        decay_total = torch.exp(rearrange(d_cumsum[..., -1].clamp(max=0), 'b h -> b h 1 1'))
        decay_to_end = torch.exp((rearrange(d_cumsum[..., -1], 'b h -> b h 1') - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        update = rearrange(k_i * decay_to_end, 'b h c d -> b h d c') @ v_i
        S = S * decay_total + update
        
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S, all_feedback_spikes_internal) if pad_len > 0 else (o, S, all_feedback_spikes_internal)

# ==================== New: Spiking Feedback Network for Comba ====================

class SpikingFeedbackNet(nn.Module):
    """A spiking network to generate feedback signals for Comba.
    Ensures all linear layers receive binary spikes as input.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, neuron_type: str, surrogate_function):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(input_dim, hidden_dim), neuron_type, surrogate_function)
        # The inter_neuron_fb now directly provides the non-linearity instead of an explicit Tanh.
        self.inter_neuron_fb = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.fc2 = SpikingLinear(nn.Linear(hidden_dim, output_dim), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fc1_out_analog, spikes_fc1_output = self.fc1(x) 

        # Spike the analog output from fc1 before feeding to fc2
        spikes_after_fc1_neuron = self.inter_neuron_fb(fc1_out_analog.transpose(-1, -2)).transpose(-1, -2)

        # final_out_analog is analog, spikes_fc2_output are binary
        final_out_analog, spikes_fc2_output = self.fc2(spikes_after_fc1_neuron) 

        # Return analog output and collected spikes
        return final_out_analog, [spikes_fc1_output, spikes_after_fc1_neuron, spikes_fc2_output]

# ==================== New: Spiking Wrapper for ShortConvolution ====================

class SpikingConvWrapper(nn.Module):
    """A wrapper for ShortConvolution that spikes its analog output.
    The internal ShortConvolution operates on analog signals.
    """

    def __init__(self, conv_layer: ShortConvolution, neuron_type: str, surrogate_function):
        super().__init__()
        self.conv = conv_layer
        # Neuron for spiking the analog output of the convolution
        self.output_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)

    def forward(self, x: torch.Tensor, cache=None, output_final_state=False, cu_seqlens=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        conv_out_analog, conv_cache = self.conv(x, cache=cache, output_final_state=output_final_state, cu_seqlens=cu_seqlens)

        # Spike the analog output of the convolution
        conv_out_spikes = self.output_neuron(conv_out_analog.transpose(-1, -2)).transpose(-1, -2)

        # Return analog output for potential further analog operations (e.g., RMSNorm, l2norm) 
        # and the binary spikes for collection.
        return conv_out_analog, conv_out_spikes, conv_cache

# ==================== SpikingComba Layer ====================

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections: All nn.Linear layers replaced by SpikingLinear as per requirements
        # Inputs to these are spikes (from SpikingFLABlock), outputs are analog.
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        self.feedback_net = SpikingFeedbackNet(
            input_dim=1, # Takes scalar mean of S state
            output_dim=1, # Produces scalar feedback per head
            hidden_dim=8,
            neuron_type=neuron_type,
            surrogate_function=surrogate
        )

        if use_short_conv:
            # Wrapped ShortConvolution with SpikingConvWrapper
            self.q_conv1d = SpikingConvWrapper(ShortConvolution(self.key_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.k_conv1d = SpikingConvWrapper(ShortConvolution(self.key_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.v_conv1d = SpikingConvWrapper(ShortConvolution(self.value_dim, conv_size, activation='silu'), neuron_type, surrogate)
        else:
            self.q_conv1d = None
            self.k_conv1d = None
            self.v_conv1d = None

        self.o_norm = RMSNorm(self.head_v_dim)
        # Neuron after o_norm and before o_proj to ensure spike input to o_proj
        self.pre_o_proj_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Dict]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        all_spikes = {}

        # Projections. Input `x` is binary spikes. Outputs (q_analog, k_analog, v_analog, g_analog) are analog.
        q_analog, spikes_q_proj_output = self.q_proj(x)
        k_analog, spikes_k_proj_output = self.k_proj(x)
        v_analog, spikes_v_proj_output = self.v_proj(x)
        g_analog, spikes_g_proj_output = self.g_proj(x)
        all_spikes.update({
            'q_proj_output': spikes_q_proj_output,
            'k_proj_output': spikes_k_proj_output,
            'v_proj_output': spikes_v_proj_output,
            'g_proj_output': spikes_g_proj_output
        })

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if past_key_values is not None and self.layer_idx is not None:
            if self.layer_idx in past_key_values:
                if 'conv_state' in past_key_values[self.layer_idx]:
                    conv_state_q, conv_state_k, conv_state_v = past_key_values[self.layer_idx]['conv_state']

        if self.q_conv1d is not None:
            # SpikingConvWrapper receives analog input, applies analog conv, and *spikes its output*.
            # The q, k, v used for Comba are the *analog outputs* of the convolutions.
            q_conv_out_analog, spikes_q_conv_output, conv_state_q = self.q_conv1d(x=q_analog, cache=conv_state_q, output_final_state=True, cu_seqlens=kwargs.get('cu_seqlens'))
            k_conv_out_analog, spikes_k_conv_output, conv_state_k = self.k_conv1d(x=k_analog, cache=conv_state_k, output_final_state=True, cu_seqlens=kwargs.get('cu_seqlens'))
            v_conv_out_analog, spikes_v_conv_output, conv_state_v = self.v_conv1d(x=v_analog, cache=conv_state_v, output_final_state=True, cu_seqlens=kwargs.get('cu_seqlens'))
            all_spikes.update({
                'q_conv_output': spikes_q_conv_output,
                'k_conv_output': spikes_k_conv_output,
                'v_conv_output': spikes_v_conv_output
            })
            q = q_conv_out_analog
            k = k_conv_out_analog
            v = v_conv_out_analog
        else:
            q, k, v = q_analog, k_analog, v_analog
        g = g_analog

        # Rearrange for attention computation. q, k, v, g are all analog values here.
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l h -> b h l')

        layer_state = None
        if past_key_values is not None and self.layer_idx is not None:
            if self.layer_idx in past_key_values:
                layer_state = past_key_values[self.layer_idx].get('recurrent_state', None)

        # The core comba_chunkwise_parallel operates on analog signals
        # It now also returns the list of internal spikes from feedback_net
        o_analog, final_state, feedback_spikes_collected = comba_chunkwise_parallel(q=q, k=k, v=v, g=g, feedback_net=self.feedback_net, initial_state=layer_state)
        if feedback_spikes_collected:
            all_spikes['feedback_net_internal_spikes'] = torch.cat([s.flatten() for s in feedback_spikes_collected])

        if past_key_values is not None and self.layer_idx is not None:
            if self.layer_idx not in past_key_values:
                past_key_values[self.layer_idx] = {}
            past_key_values[self.layer_idx]['recurrent_state'] = final_state
            if self.q_conv1d is not None:
                past_key_values[self.layer_idx]['conv_state'] = (conv_state_q, conv_state_k, conv_state_v)

        o_analog = rearrange(o_analog, 'b h l d -> b l h d')
        o_analog = self.o_norm(o_analog) # o_analog is analog output after RMSNorm
        o_analog = rearrange(o_analog, 'b l h d -> b l (h d)') # o_analog is analog

        # Spike the output of o_norm before feeding to o_proj
        spikes_pre_o_proj_output = self.pre_o_proj_neuron(o_analog.transpose(-1, -2)).transpose(-1, -2)
        all_spikes['pre_o_proj_output'] = spikes_pre_o_proj_output

        # o_proj now receives spikes as input
        o_final_analog, spikes_o_proj_output = self.o_proj(spikes_pre_o_proj_output)
        all_spikes['o_proj_output'] = spikes_o_proj_output

        return o_final_analog, None, all_spikes

# [SpikingMLP and SpikingFLABlock]

class SpikingMLP(nn.Module):
    """A spiking MLP where all linear layers receive binary spike inputs.
    All internal activation non-linearities are provided by spiking neurons.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        # Neuron after fc1's analog output to ensure spike input to fc2, provides non-linearity
        self.intermediate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fc1_out_analog, spikes_fc1_output = self.fc1(x) 

        # Convert fc1_out_analog (analog) to spikes for dropout and fc2
        spikes_after_fc1_neuron = self.intermediate_neuron(fc1_out_analog.transpose(-1, -2)).transpose(-1, -2)

        # Apply dropout to the spikes
        dropped_spikes = self.dropout(spikes_after_fc1_neuron)

        # fc2 receives spikes as input. fc2_out_analog is the analog output of its internal nn.Linear.
        fc2_out_analog, spikes_fc2_output = self.fc2(dropped_spikes)

        # Return the final analog output and all collected spike tensors
        return fc2_out_analog, spikes_fc1_output, spikes_after_fc1_neuron, spikes_fc2_output

class SpikingFLABlock(nn.Module):
    """A SpikingFLABlock that integrates the Comba attention mechanism and a SpikingMLP,
    ensuring all linear layers receive binary spike inputs and collecting all generated spikes.

    Key enhancements include:
    1. Strict spiking of inputs to both Comba and SpikingMLP blocks.
    2. Spiking outputs of intermediate analog operations (like ShortConvolution, RMSNorm) 
       before feeding into subsequent SpikingLinear layers or spike-processing stages.
    3. Removal of redundant analog activation functions (e.g., Tanh) where spiking neurons 
       already provide the necessary non-linearity.
    4. Comprehensive collection of all generated binary spikes throughout the block for analysis.
    """

    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Neuron to spike the input *before* the attention block
        self.attn_input_neuron = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())

        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)

        # Neuron to spike the input *before* the MLP block
        self.mlp_input_neuron = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())

        d_ffn = d_model * d_ffn_mult
        # SpikingMLP ensures all its internal linear layers receive spike inputs
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Dict]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        x_res = x # Analog residual connection

        # Prepare input for Attention Block
        if self.prenorm: 
            x_normalized = self.norm1(x) # Analog output from RMSNorm
            # Generate spikes from normalized analog input before attention block
            spikes_for_attn = self.attn_input_neuron(x_normalized.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_attn)
        else:
            # If not prenorm, spike the raw input directly before attention
            spikes_for_attn = self.attn_input_neuron(x.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_attn)

        # Attention block. Its input (spikes_for_attn) is strictly binary spikes.
        # It returns analog output, a possible regularization loss (not used here), and a dict of spikes.
        x_attn_analog_out, _, attn_spikes_dict = self.attn(spikes_for_attn, past_key_values=past_key_values, **kwargs) 
        if attn_spikes_dict: 
            all_spikes_list.extend([s for s in attn_spikes_dict.values() if s is not None])
        x = x_res + self.dropout1(x_attn_analog_out) # Residual connection, x_attn_analog_out is analog

        x_res = x # Analog residual connection

        # Prepare input for MLP Block
        if self.prenorm: 
            x_normalized = self.norm2(x) # Analog output from RMSNorm
            # Generate spikes from normalized analog input before MLP block
            spikes_for_mlp = self.mlp_input_neuron(x_normalized.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_mlp)
        else:
            # If not prenorm, spike the output of the attention block directly before MLP
            spikes_for_mlp = self.mlp_input_neuron(x.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_mlp)

        # MLP block. Its input (spikes_for_mlp) is strictly binary spikes.
        # SpikingMLP returns its final analog output and individual spike tensors.
        x_mlp_analog_out, spikes_mlp_fc1_output, spikes_mlp_intermediate_neuron, spikes_mlp_fc2_output = self.mlp(spikes_for_mlp)

        all_spikes_list.extend([
            s for s in [spikes_mlp_fc1_output, spikes_mlp_intermediate_neuron, spikes_mlp_intermediate_neuron, spikes_mlp_fc2_output] if s is not None
        ])

        x = x_res + self.dropout2(x_mlp_analog_out) # Residual connection, x_mlp_analog_out is analog

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        return x, None, final_spikes