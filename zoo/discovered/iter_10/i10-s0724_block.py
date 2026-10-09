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

# ================================================================

# ==================== White-box Operator (Comba Closed-Loop) ====================

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
        
        # --- [Fix 1] Closed-loop feedback stability enhancement ---
        # 1. Normalize state features to prevent FeedbackNet input explosion
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True)
        s_norm_clamp = torch.clamp(s_norm, min=1e-6) # prevent division by zero
        s_feat = (S / s_norm_clamp).mean(dim=(-1, -2), keepdim=True).flatten(2) # [B, H, 1]
        
        # 2. Limit feedback offset to [-2, 2] to prevent complete override of original gating
        f_i = feedback_net(s_feat).squeeze(-1) # [B, H]
        f_i = torch.tanh(f_i) * 2.0 
        
        # --- [Fix 2] Gate numerical computation alignment ---
        # Use more stable logsigmoid
        log_decay = F.logsigmoid(g_i + f_i.unsqueeze(-1)) # [B, H, C]
        d_cumsum = torch.cumsum(log_decay, dim=-1)
        
        decay_rel = d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)
        # Ensure exp exponent is not greater than 0
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0)
        
        # 3. Compute output
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = attn_intra @ v_i
        
        o[:, :, i] = o_inter + o_intra
        
        # --- [Fix 3] State update stability enhancement ---
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        # Limit update step size to prevent excessive single updates
        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update
        
        # Strict renormalization after each chunk update
        # This is critical in SNNs to prevent hidden state steady-state gain exceeding 1
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingComba Layer ====================

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        # Feedback Network: Process state feedback (state features -> feedback bias)
        # This remains an ANN as its input is an analog feature derived from the state, not directly spikes.
        self.feedback_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs):
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q':s_q, 'k':s_k, 'v':s_v, 'g':s_g})

        if hasattr(self, 'q_conv1d'):
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l h -> b h l')

        layer_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
            layer_state = past_key_values[self.layer_idx]

        # Execute core parallel operator
        o, final_state = comba_chunkwise_parallel(q, k, v, g, self.feedback_net, initial_state=layer_state)

        if use_cache and past_key_values is not None and self.layer_idx is not None:
            past_key_values[self.layer_idx] = final_state

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        # First linear layer, wrapped in SpikingLinear.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        # Removed F.gelu(x) to simplify the hybrid nature and rely solely on spiking neurons for non-linearity.
        self.dropout = nn.Dropout(dropout)

        # Second linear layer, also wrapped in SpikingLinear.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_out, spike1 = self.fc1(x) 

        # Apply dropout directly to the analog output of the first layer.
        x_after_dropout = self.dropout(x_fc1_out)

        # The (analog) output after dropout becomes the input to the second SpikingLinear layer.
        # x_fc2_out is analog output, spike2 are binary spikes from fc2's internal neuron
        x_fc2_out, spike2 = self.fc2(x_after_dropout) 

        # Return the final analog output of the MLP and the collected binary spikes from its layers.
        return x_fc2_out, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', spike_reg_coeff: float=0.01, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Using Comba (from Version 2) as the attention mechanism
        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        # Always use SpikingMLP, removing the use_spiking_mlp flag
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)
        self.spike_reg_coeff = spike_reg_coeff # Hyperparameter for spike regularization

    def forward(self, x: torch.Tensor, past_key_values: Optional[Dict]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        # Initialize reg_loss as a tensor with device matching x
        reg_loss = torch.tensor(0.0, device=x.device)

        x_res = x
        if self.prenorm:
            x = self.norm1(x)

        # Comba returns analog output, a placeholder for reg_loss (None), and a dictionary of collected spikes.
        # The `past_key_values` dict is passed directly to `self.attn` for state management.
        x_attn, attn_reg_loss, attn_spikes = self.attn(x, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, **kwargs)

        # Comba currently returns None for reg_loss, but keeping the aggregation for future extensibility.
        if attn_reg_loss is not None:
             reg_loss = reg_loss + attn_reg_loss

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
        # Concatenate all collected binary spike tensors into a single flattened tensor.
        if all_spikes_list:
            # Filter out any None values that might have crept in
            all_spikes_list = [s for s in all_spikes_list if s is not None]
            if all_spikes_list: # Check again in case all were None
                final_spikes = torch.cat([s.flatten() for s in all_spikes_list])
                # Calculate spike regularization loss: mean squared firing rate
                # This encourages sparse firing, promoting energy efficiency.
                mean_firing_rate = final_spikes.float().mean()
                reg_loss = reg_loss + self.spike_reg_coeff * (mean_firing_rate ** 2)

        # Return the final output, reg_loss, and the concatenated final_spikes.
        return x, reg_loss, final_spikes