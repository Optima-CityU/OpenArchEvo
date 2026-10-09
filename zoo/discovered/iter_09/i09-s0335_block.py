# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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

# ==================== White-box Operator (Comba Closed-Loop) ====================

def comba_chunkwise_parallel(q, k, v, g, feedback_net, chunk_size=64, initial_state=None):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # Calculate padding length
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        # Pad q, k, v along the sequence dimension
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        # Pad g along the sequence dimension with a suitable default (e.g., 0.0 for log_decay)
        g = F.pad(g, (0, pad_len), value=0.0)
    
    # Apply L2 normalization to query and key
    q = l2norm(q)
    k = l2norm(k)
    
    # Rearrange tensors into chunks
    q, k, v, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, g]
    )
    
    num_chunks = q.shape[2]
    # Initialize state S (if not provided)
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    # Create an upper triangular mask for intra-chunk attention
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)

    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        
        # Closed-loop feedback stability enhancement
        # Normalize state features to prevent FeedbackNet input explosion
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        s_feat = (S / s_norm).mean(dim=(-1, -2), keepdim=True).flatten(2) # [B, H, 1]
        
        # Limit feedback offset to [-2, 2] to prevent complete override of original gating
        f_i = feedback_net(s_feat).squeeze(-1) # [B, H]
        f_i = torch.tanh(f_i) * 2.0 
        
        # Gate numerical computation alignment using logsigmoid for stability
        log_decay = F.logsigmoid(g_i + f_i.unsqueeze(-1)) # [B, H, C]
        d_cumsum = torch.cumsum(log_decay, dim=-1)
        
        decay_rel = d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)
        # Ensure exp exponent is not greater than 0
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0)
        
        # Compute inter-chunk output from global state S
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        # Compute intra-chunk output
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = attn_intra @ v_i
        
        o[:, :, i] = o_inter + o_intra
        
        # State update stability enhancement
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        # Limit update step size to prevent excessive single updates
        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update
        
        # Strict renormalization after each chunk update
        # This is critical in SNNs to prevent hidden state steady-state gain exceeding 1
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 

    # Rearrange output back from chunks
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    
    # Remove padding if it was added
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

# ==================== SpikingComba Layer ====================

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        # Get surrogate function for spiking neurons
        surrogate = piecewise_quadratic_surrogate()

        # Projections using SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        # Feedback Network: Process state feedback (state features -> feedback bias)
        # This is an ANN component that operates on the aggregated normalized state features
        self.feedback_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        # Optional ShortConvolution for local feature extraction
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor, past_key_values=None, **kwargs) -> Tuple[torch.Tensor, None, Dict[str, torch.Tensor]]:
        all_spikes = {}

        # Apply spiking projections and collect spikes
        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q_proj':s_q, 'k_proj':s_k, 'v_proj':s_v, 'g_proj':s_g})

        # Apply ShortConvolution if enabled
        if hasattr(self, 'q_conv1d'):
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

        # Rearrange for multi-head processing
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l h -> b h l')

        layer_state = None
        if past_key_values is not None and self.layer_idx is not None:
            # Ensure safe access to past_key_values
            layer_state = past_key_values.get(self.layer_idx, None)

        # Execute core parallel Comba operator
        o, final_state = comba_chunkwise_parallel(q, k, v, g, self.feedback_net, initial_state=layer_state)

        # Update cache if use_cache is enabled
        if past_key_values is not None and self.layer_idx is not None:
            past_key_values[self.layer_idx] = final_state

        # Rearrange output and apply normalization
        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')

        # Final output projection
        o, s_o = self.o_proj(o)
        all_spikes['o_proj'] = s_o

        return o, None, all_spikes # return output, regularization_loss (None for Comba's internal), all_spikes

# [SpikingMLP & SpikingFLABlock]

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        # All linear layers are SpikingLinear
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        # The analog output from fc1 is activated by GELU, then fed to fc2 (which internally converts to spikes)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_analog_out_1, spike1 = self.fc1(x)
        # Apply GELU to the analog output of the first spiking linear layer
        x_analog_activated = F.gelu(x_analog_out_1) 
        x_analog_dropout = self.dropout(x_analog_activated)
        # The second spiking linear layer receives this analog input and internally converts it to spikes
        x_analog_out_2, spike2 = self.fc2(x_analog_dropout)
        return x_analog_out_2, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', spike_regularization_weight: float=0.01, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        # Comba attention module as in Version 2
        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)

        d_ffn = d_model * d_ffn_mult

        # Always use SpikingMLP, consistent with "inputs of ALL linear layers are binary signals"
        # and "enabled by default" for new features (full SNN block).
        # Removed 'use_spiking_mlp' parameter and conditional logic for simplification.
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)
        # Re-introducing spike regularization for SNNs to encourage sparsity
        self.spike_regularization_weight = spike_regularization_weight

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        total_reg_loss = None # Initialize total regularization loss

        # Attention block
        x_res = x
        if self.prenorm:
            x = self.norm1(x)

        # Comba returns output, None (for internal reg loss), and a dict of spikes
        x_attn, _, attn_spikes_dict = self.attn(x, **kwargs) # attn_reg_loss from Comba is always None

        # Collect spikes from the attention mechanism for the final_spikes output and regularization
        if attn_spikes_dict:
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        # Calculate attention-related spike regularization loss if weight > 0
        if self.spike_regularization_weight > 0:
            if all_spikes_list:
                # Concatenate all spikes from attention and calculate mean firing rate for regularization
                attn_spikes_for_reg = torch.cat([s.flatten() for s in all_spikes_list])
                attn_reg_loss = attn_spikes_for_reg.float().mean() * self.spike_regularization_weight
                total_reg_loss = attn_reg_loss

        x = x_res + self.dropout1(x_attn)

        # FFN block
        x_res = x
        if self.prenorm:
            x = self.norm2(x)

        # SpikingMLP returns analog output and two spike tensors
        x_mlp_out, ffn_spike1, ffn_spike2 = self.mlp(x)
        all_spikes_list.extend([ffn_spike1, ffn_spike2]) # Collect spikes from MLP

        # Calculate FFN-specific regularization loss if weight > 0
        if self.spike_regularization_weight > 0:
            # Concatenate all spikes from FFN and calculate mean firing rate for regularization
            ffn_spikes_for_reg = torch.cat([ffn_spike1.flatten(), ffn_spike2.flatten()])
            ffn_reg_loss = ffn_spikes_for_reg.float().mean() * self.spike_regularization_weight

            if total_reg_loss is None:
                total_reg_loss = ffn_reg_loss
            else:
                total_reg_loss += ffn_reg_loss

        x = x_res + self.dropout2(x_mlp_out)

        final_spikes = None
        if all_spikes_list:
            # Concatenate ALL collected binary spike tensors into a single flattened tensor.
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

        return x, total_reg_loss, final_spikes