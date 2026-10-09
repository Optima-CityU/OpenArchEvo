# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution
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
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

# ================================================================

def comba_chunkwise_parallel(q, k, v, g, feedback_net, chunk_size=64, initial_state=None):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len), value=0.0)
    
    q = l2norm(q)
    k = l2norm(k)
    
    q, k, v, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, g]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)

    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        
        # --- Closed-loop feedback stability enhancement ---
        # 1. Normalize state features to prevent FeedbackNet input explosion
        # Ensured s_norm is never zero by adding 1e-6
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        s_feat = (S / s_norm).mean(dim=(-1, -2), keepdim=True).flatten(2) # [B, H, 1]
        
        # 2. Limit feedback offset to [-2, 2] to prevent complete override of original gating
        f_i = feedback_net(s_feat).squeeze(-1) # [B, H]
        f_i = torch.tanh(f_i) * 2.0 
        
        # --- Gate numerical computation alignment ---
        # Use more stable logsigmoid for decay calculation
        log_decay = F.logsigmoid(g_i + f_i.unsqueeze(-1)) # [B, H, C]
        d_cumsum = torch.cumsum(log_decay, dim=-1)
        
        # Causal masking for relative decay to prevent future information leakage
        decay_rel = d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0) # Ensures exp exponent not greater than 0
        
        # 3. Compute output
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = attn_intra @ v_i
        
        o[:, :, i] = o_inter + o_intra
        
        # --- State update stability enhancement ---
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        # Limit update step size to prevent excessive single updates and ensure stability
        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update
        
        # Strict renormalization after each chunk update to prevent hidden state explosion.
        # This is critical in SNNs where state values can quickly grow or vanish.
        # Scale down if norm is high, but ensure minimum 1.0 to avoid division by zero or huge scaling.
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections using SpikingLinear to ensure all linear layers receive binary spike inputs
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        # Feedback Network: Process state feedback to modulate gating dynamically.
        # This small ANN acts on a compressed representation of the recurrent state.
        self.feedback_net = nn.Sequential(
            nn.Linear(1, 8), # Input dimension 1 for mean(S/norm).flatten(2)
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        self.use_short_conv = use_short_conv
        if use_short_conv:
            # ShortConvolution layers are non-spiking, acting as local feature extractors in a hybrid SNN-ANN design
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs):
        all_spikes = {}

        # Apply SpikingLinear layers, collecting binary spikes
        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q_proj':s_q, 'k_proj':s_k, 'v_proj':s_v, 'g_proj':s_g})

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
             last_state = past_key_values[self.layer_idx]
             if last_state is not None:
                 if 'conv_state' in last_state and self.use_short_conv:
                     conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

        if self.use_short_conv:
            # cu_seqlens=None as x is either original batch or a single unpadded sequence
            q, conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=None)
            k, conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=None)
            v, conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=None)

        # Reshape for multi-head processing using einops
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l h -> b h l')

        recurrent_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            if last_state is not None and 'recurrent_state' in last_state:
                recurrent_state = last_state['recurrent_state']

        # Execute core parallel Comba operator for efficient chunk-wise processing
        o, final_recurrent_state = comba_chunkwise_parallel(q, k, v, g, self.feedback_net, initial_state=recurrent_state)

        # Update cache for recurrent and convolutional states
        if use_cache and past_key_values is not None and self.layer_idx is not None:
            cache_data = {
                'recurrent_state': final_recurrent_state,
                'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None
            }
            past_key_values[self.layer_idx] = cache_data

        # Reshape output and apply final normalization and projection
        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o_proj'] = s_o

        return o, None, all_spikes # None for reg_loss, handled at block level

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        # SpikingLinear's internal nn.Linear receives binary spikes. Its forward() can take analog input.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x_analog_out, spike1 = self.fc1(x)
        # Apply dropout directly to the analog output of the first layer.
        # Removed F.gelu here, relying on spiking neuron for non-linearity.
        x_after_dropout = self.dropout(x_analog_out)
        # x_after_dropout (analog) is input to fc2
        x_final_analog_out, spike2 = self.fc2(x_after_dropout)
        return x_final_analog_out, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', spike_regularization_weight: float=0.001, target_firing_rate: float=0.05, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Comba attention mechanism, designed for sub-quadratic complexity and SNN integration
        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        # MLP always uses SpikingMLP to ensure all linear layers are spiking-neuron-mediated
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

        self.spike_regularization_weight = spike_regularization_weight
        self.target_firing_rate = target_firing_rate # New parameter for target rate regularization

    def forward(self, x, attention_mask: Optional[torch.Tensor]=None, past_key_values=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs):
        all_spikes_list: List[torch.Tensor] = []

        # Handle padding for variable sequence lengths if attention_mask is provided
        # This logic is lifted to SpikingFLABlock for consistent processing of input 'x'
        batch_size_orig, q_len_orig = x.shape[0], x.shape[1]
        indices, cu_seqlens = None, None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len_orig:])
            # Rearrange and index to get the unpadded sequence
            x_processed = index_first_axis(rearrange(x, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        else:
            x_processed = x # If no mask, process the input as is

        # --- Attention Sub-block ---
        x_res = x
        if self.prenorm: 
            x_normalized_attn = self.norm1(x_processed)
        else:
            x_normalized_attn = x_processed

        # attn_output, reg_loss_from_attn, attn_spikes_dict
        x_attn, _, attn_spikes_dict = self.attn(x_normalized_attn, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, **kwargs)

        # Collect spikes from the attention mechanism
        if attn_spikes_dict and isinstance(attn_spikes_dict, dict):
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        # Re-pad attn_output if input was unpadded
        if attention_mask is not None:
            x_attn = pad_input(x_attn.squeeze(0), indices, batch_size_orig, q_len_orig)

        x = x_res + self.dropout1(x_attn)

        # --- MLP Sub-block ---
        x_res = x
        if self.prenorm: 
            # Apply normalization to the unpadded version of `x_res` before feeding to MLP
            if attention_mask is not None:
                x_mlp_input = self.norm2(index_first_axis(rearrange(x_res, "b s ... -> (b s) ..."), indices).unsqueeze(0))
            else:
                x_mlp_input = self.norm2(x_res)
        else: 
            # If no prenorm, just use the unpadded x_res
            if attention_mask is not None:
                x_mlp_input = index_first_axis(rearrange(x_res, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            else:
                x_mlp_input = x_res

        # SpikingMLP returns its final analog output and individual spike tensors for its two layers.
        x_mlp, mlp_spikes1, mlp_spikes2 = self.mlp(x_mlp_input)

        # Collect spikes from the MLP
        if mlp_spikes1 is not None:
            all_spikes_list.append(mlp_spikes1)
        if mlp_spikes2 is not None:
            all_spikes_list.append(mlp_spikes2)

        # Re-pad mlp_output if input was unpadded
        if attention_mask is not None:
            x_mlp = pad_input(x_mlp.squeeze(0), indices, batch_size_orig, q_len_orig)

        x = x_res + self.dropout2(x_mlp)

        final_spikes = None
        current_reg_loss = None
        # Concatenate all collected binary spike tensors into a single flattened tensor.
        if all_spikes_list:
            # Filter out any None values that might have crept in
            all_spikes_list = [s for s in all_spikes_list if s is not None]
            if all_spikes_list: # Check again in case all were None
                final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

                # Apply target firing rate regularization to encourage stable and sparse spike activity
                if self.spike_regularization_weight > 0 and final_spikes.numel() > 0:
                    spike_activity = final_spikes.float().mean()
                    current_reg_loss = self.spike_regularization_weight * (spike_activity - self.target_firing_rate).pow(2)

        return x, current_reg_loss, final_spikes