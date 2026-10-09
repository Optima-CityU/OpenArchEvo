from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fla.modules import RMSNorm

# ==================== SNN Imports and Helpers ====================
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

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2) # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed) # Spikes from the internal neuron
        spikes = spikes_transposed.transpose(-1, -2) # (B, D, L) -> (B, L, D) # Binary spikes
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

# ==================== White-box Operator (LightNet) ====================

def lightnet_chunk_step(q_i, k_i, v_i, g_i, S_i, mask):
    """Helper function for a single chunk calculation to be checkpointed."""
    kv_i = k_i * v_i
    
    g_cumsum = torch.cumsum(g_i, dim=-2)
    
    # Efficient Decay Mask Calculation
    # shape: (B, H, C, C, D_block)
    decay_matrix = g_cumsum.unsqueeze(-2) - g_cumsum.unsqueeze(-3)
    
    # Apply Mask (Passed in to avoid recreating)
    decay_matrix = decay_matrix.masked_fill(mask.unsqueeze(-1), -float('inf'))
    decay_mask = torch.exp(decay_matrix)
    
    # Output Calculation
    o_intra = torch.einsum('b h i j d, b h j d -> b h i d', decay_mask, kv_i)
    o_intra = o_intra * q_i
    
    # 2. Inter-Chunk
    decay_from_start = torch.exp(g_cumsum)
    o_inter = S_i.unsqueeze(2) * decay_from_start * q_i
    
    o_chunk = o_intra + o_inter
    
    # 3. Compute New State
    # Only need the last row relative to current positions for state update
    total_decay = torch.exp(g_i.sum(dim=-2))
    
    # decay_to_end: decay from position j to the end of the chunk
    decay_to_end = torch.exp(g_cumsum[:, :, -1].unsqueeze(-2) - g_cumsum)
    current_chunk_contrib = (kv_i * decay_to_end).sum(dim=-2)
    
    S_new = S_i * total_decay + current_chunk_contrib
    
    return o_chunk, S_new

def lightnet_chunkwise(q, k, v, g, chunk_size=64, initial_state=None):
    """Optimized LightNet Chunkwise Logic.
    Optimizations:
    1. Gradient Checkpointing: Drastically reduces memory from O(L) to O(chunk_size).
    2. Lazy KV Computation: Avoids storing full KV tensor.
    3. Pre-allocated Mask: Reduces overhead.
    """
    b, h, l, d = q.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        # Ensure padding value and dtype are consistent
        q, k, v, g = [F.pad(x, (0, 0, 0, pad_len), value=0.) for x in [q, k, v, g]]
        
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    g = rearrange(g, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, d, device=q.device, dtype=q.dtype) # Match dtype of q
        
    o_list = []
    
    # [Optimization] Pre-compute mask once
    mask_template = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    
    # Feature Block Size
    block_size = 32 
    
    for i in range(q.shape[2]): # Chunk Loop
        q_chunk = q[:, :, i]
        k_chunk = k[:, :, i] 
        v_chunk = v[:, :, i]
        g_chunk = g[:, :, i]
        
        o_chunk_slices = []
        S_new_slices = []
        
        for start_idx in range(0, d, block_size):
            end_idx = min(start_idx + block_size, d)
            
            # Slices (Views)
            q_i = q_chunk[..., start_idx:end_idx]
            k_i = k_chunk[..., start_idx:end_idx]
            v_i = v_chunk[..., start_idx:end_idx]
            g_i = g_chunk[..., start_idx:end_idx]
            S_i = S[..., start_idx:end_idx]
            
            # [Optimization] Use Checkpoint
            if torch.is_grad_enabled():
                o_slice, S_new_slice = checkpoint(
                    lightnet_chunk_step, 
                    q_i, k_i, v_i, g_i, S_i, mask_template,
                    use_reentrant=False 
                )
            else:
                o_slice, S_new_slice = lightnet_chunk_step(q_i, k_i, v_i, g_i, S_i, mask_template)
            
            o_chunk_slices.append(o_slice)
            S_new_slices.append(S_new_slice)
            
        # Concatenate slices
        S = torch.cat(S_new_slices, dim=-1)
        o_chunk = torch.cat(o_chunk_slices, dim=-1)
        o_list.append(o_chunk)
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    
    if pad_len > 0: 
        o = o[..., :l, :]
    
    return o, S

class LightNet(nn.Module):
    def __init__(self, d_model, num_heads, expand_ratio=2, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = int(d_model * expand_ratio)
        self.head_dim = self.hidden_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        self.q_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)

        self.o_proj = SpikingLinear(nn.Linear(self.hidden_dim, d_model, bias=False), neuron_type, surrogate)
        self.norm = RMSNorm(self.hidden_dim)

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'g': s_g})

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l (h d) -> b h l d', h=self.num_heads)

        g = F.logsigmoid(g) 

        last_state = None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = lightnet_chunkwise(q, k, v, g, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                past_key_values.append({'recurrent_state': final_state})
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_state

        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.norm(o)
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

# ==================== Block Wrapper ====================

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)

        # [NEW] Intermediate spiking neuron to enforce spike-driven internal activation
        self.intermediate_spike_activation = neuron_registry[neuron_type](surrogate_function=surrogate_function)

        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x_fc1_out_analog, spike1 = self.fc1(x)

        # 2. Intermediate Spiking Activation
        #   - The analog output `x_fc1_out_analog` now drives an *explicit* intermediate spiking neuron.
        #   - Transpose for consistency with neuron_registry's typical (B, D, L) processing.
        x_intermediate_input_transposed = rearrange(x_fc1_out_analog, 'b l d -> b d l')
        intermediate_spikes_transposed = self.intermediate_spike_activation(x_intermediate_input_transposed)
        intermediate_spikes = rearrange(intermediate_spikes_transposed, 'b d l -> b l d') # (B, L, D_ffn) - These are binary spikes

        # 3. Dropout applied to the intermediate binary spikes
        #    Dropout on binary spikes results in sparse binary spikes.
        x_after_dropout_spikes = self.dropout(intermediate_spikes)

        # 4. Second SpikingLinear layer
        #   - Takes `x_after_dropout_spikes` (binary) as its input `x`.
        #   - Internally `fc2.neuron` converts `x_after_dropout_spikes` (binary) into `spike2` (binary) for `fc2.linear`.
        #   - `x_fc2_out_analog` is the analog output of `fc2.linear`.
        x_fc2_out_analog, spike2 = self.fc2(x_after_dropout_spikes)

        # Return the final analog output and all collected binary spikes
        return x_fc2_out_analog, spike1, intermediate_spikes, spike2 # Return all three spike tensors

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp=True, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = LightNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)

        d_ffn = d_model * d_ffn_mult
        # The 'use_spiking_mlp' parameter is kept for signature compatibility but its logic effect is unified.
        # The MLP is always SpikingMLP with an intermediate spiking neuron.
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

        # New: Spike-gating for attention residual connection
        # It takes a scalar summary of spike activity and projects it to d_model for gating.
        self.attn_spike_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, piecewise_quadratic_surrogate())
        # New: Learnable scaling factor for the attention spike gate.
        self.attn_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

        # New: Spike-gating for MLP residual connection
        # Similar to attention, takes (B, L, 1) mean spike activity and projects to d_model.
        self.mlp_spike_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, piecewise_quadratic_surrogate())
        # New: Learnable scaling factor for the MLP spike gate.
        self.mlp_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

    def forward(self, x, **kwargs):
        all_spikes_list = []

        x_res = x
        if self.prenorm: 
            x = self.norm1(x)

        # LightNet returns analog output, None for reg_loss, and a dictionary of collected spikes.
        x_attn, _, attn_spikes_dict = self.attn(x, **kwargs)

        if attn_spikes_dict: 
            all_spikes_list.extend(attn_spikes_dict.values()) # Collect all spikes from attention block

        # === Attention Residual Spike Gating ===
        attn_block_spike_activity = None
        if 'k' in attn_spikes_dict and attn_spikes_dict['k'] is not None:
            # Prioritize 'k' spikes from LightNet, as they are central to the interaction
            attn_block_spike_activity = attn_spikes_dict['k'].float().mean(dim=-1, keepdim=True) # (B, L, 1)
        else:
            # Fallback: if 'k' spikes are not available, take the mean activity of the first available spike tensor
            for spike_name, spike_tensor in attn_spikes_dict.items():
                if spike_tensor is not None:
                    attn_block_spike_activity = spike_tensor.float().mean(dim=-1, keepdim=True)
                    break

        x_attn_gated = x_attn
        if attn_block_spike_activity is not None:
            attn_gate_analog_output, attn_gate_binary_spikes = self.attn_spike_gate_proj(attn_block_spike_activity)
            all_spikes_list.append(attn_gate_binary_spikes) # Collect spikes from the gate
            attn_gate = self.attn_gate_scale * torch.sigmoid(attn_gate_analog_output)
            x_attn_gated = x_attn * attn_gate # Element-wise gating

        x = x_res + self.dropout1(x_attn_gated)

        x_res = x
        if self.prenorm: 
            x = self.norm2(x)

        # SpikingMLP returns its final analog output and three spike tensors:
        # spike1 (input to fc1.linear), intermediate_spikes (after fc1.linear, before fc2.neuron), spike2 (input to fc2.linear)
        x_mlp, mlp_spike1, mlp_intermediate_spikes, mlp_spike2 = self.mlp(x)

        all_spikes_list.extend([mlp_spike1, mlp_intermediate_spikes, mlp_spike2]) # Collect all spikes from MLP

        # === MLP Residual Spike Gating ===
        mlp_spikes_for_gating_means = []
        if mlp_spike1 is not None: mlp_spikes_for_gating_means.append(mlp_spike1.float().mean(dim=-1, keepdim=True))
        if mlp_intermediate_spikes is not None: mlp_spikes_for_gating_means.append(mlp_intermediate_spikes.float().mean(dim=-1, keepdim=True))
        if mlp_spike2 is not None: mlp_spikes_for_gating_means.append(mlp_spike2.float().mean(dim=-1, keepdim=True))

        x_mlp_gated = x_mlp
        if mlp_spikes_for_gating_means:
            # Average the mean activities from fc1, intermediate, and fc2 for the MLP gate's input
            mlp_block_spike_activity = torch.mean(torch.stack(mlp_spikes_for_gating_means), dim=0) # (B, L, 1)
            mlp_gate_analog_output, mlp_gate_binary_spikes = self.mlp_spike_gate_proj(mlp_block_spike_activity)
            all_spikes_list.append(mlp_gate_binary_spikes) # Collect spikes from the gate
            mlp_gate = self.mlp_gate_scale * torch.sigmoid(mlp_gate_analog_output)
            x_mlp_gated = x_mlp * mlp_gate # Element-wise gating

        x = x_res + self.dropout2(x_mlp_gated) 

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None]) if all_spikes_list else None

        # Ensure final_spikes is always a tensor, even if empty. Explicitly set dtype to bool.
        if final_spikes is None:
            final_spikes = torch.empty(0, device=x.device, dtype=torch.bool)


        return x, None, final_spikes