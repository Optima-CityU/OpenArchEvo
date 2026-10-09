"""Improved SpikingLightNet with dual spike-gated MLP hidden state"""
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
    # Fallback for environments without src.models.spike
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
        # The neuron takes continuous input, converts it to spikes.
        # The linear layer then operates on these binary spikes.
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2) 
        spikes_transposed = self.neuron(x_transposed) # Binary spikes
        spikes = spikes_transposed.transpose(-1, -2) # Revert transpose for linear layer
        output = self.linear(spikes) # Linear layer receives binary spikes
        return output, spikes

# ==================== White-box Operator (LightNet) ====================

def lightnet_chunk_step(q_i: torch.Tensor, k_i: torch.Tensor, v_i: torch.Tensor, g_i: torch.Tensor, S_i: torch.Tensor, mask: torch.Tensor):
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

def lightnet_chunkwise(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, chunk_size: int=64, initial_state: Optional[torch.Tensor]=None):
    """Optimized LightNet Chunkwise Logic with Gradient Checkpointing."""
    b, h, l, d = q.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v, g = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v, g]]
        
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    g = rearrange(g, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, d, device=q.device)
        
    o_list = []
    
    # Pre-compute mask once
    mask_template = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    
    # Feature Block Size for memory efficiency with checkpointing
    block_size = 32 # Default, can be tuned
    
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
            
            # Use gradient checkpointing for memory efficiency
            o_slice, S_new_slice = checkpoint(
                lightnet_chunk_step, 
                q_i, k_i, v_i, g_i, S_i, mask_template,
                use_reentrant=False 
            )
            
            o_chunk_slices.append(o_slice)
            S_new_slices.append(S_new_slice)
            
        # Concatenate slices
        S = torch.cat(S_new_slices, dim=-1)
        o_chunk = torch.cat(o_chunk_slices, dim=-1)
        o_list.append(o_chunk)
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    
    # Remove padding if applied
    if pad_len > 0: o = o[..., :l, :]
    
    return o, S

class LightNet(nn.Module):
    def __init__(self, d_model: int, num_heads: int, expand_ratio: int=2, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = int(d_model * expand_ratio)
        self.head_dim = self.hidden_dim // num_heads # Note: In LightNet, d_head is usually d_model // num_heads if expansion happens later. Here it's hidden_dim // num_heads after expansion.
        self.layer_idx = layer_idx

        surrogate = piecewise_quadratic_surrogate()

        # All linear projections are now SpikingLinear, taking continuous input and internally spiking
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)

        self.o_proj = SpikingLinear(nn.Linear(self.hidden_dim, d_model, bias=False), neuron_type, surrogate)
        self.norm = RMSNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, layer_idx: Optional[int]=None, **kwargs):
        all_spikes = {}

        # Inputs to SpikingLinear are continuous, outputs are (continuous_linear_output, binary_spikes)
        q_cont, s_q = self.q_proj(x)
        k_cont, s_k = self.k_proj(x)
        v_cont, s_v = self.v_proj(x)
        g_cont, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'g': s_g})

        q = rearrange(q_cont, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k_cont, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v_cont, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g_cont, 'b l (h d) -> b h l d', h=self.num_heads)

        # Apply log-sigmoid as per LightNet paper for decay factor 'g'
        g = F.logsigmoid(g) 

        last_state = None
        current_layer_idx = self.layer_idx if layer_idx is None else layer_idx
        if past_key_values is not None and current_layer_idx is not None:
            if len(past_key_values) > current_layer_idx:
                last_state = past_key_values[current_layer_idx].get('recurrent_state')

        o_cont, final_state = lightnet_chunkwise(q, k, v, g, initial_state=last_state)

        if past_key_values is not None and current_layer_idx is not None:
            if len(past_key_values) <= current_layer_idx:
                past_key_values.append({'recurrent_state': final_state})
            else:
                past_key_values[current_layer_idx]['recurrent_state'] = final_state

        o_cont = rearrange(o_cont, 'b h l d -> b l (h d)')
        o_norm = self.norm(o_cont) # RMSNorm output is continuous

        # Final output projection is also SpikingLinear
        o_final_cont, s_o = self.o_proj(o_norm)
        all_spikes['o'] = s_o

        return o_final_cont, None, all_spikes

# ==================== SpikingMLP with Dual Gated Hidden State ====================

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        # fc1 and fc2 are SpikingLinear, ensuring internal linear operations are on spikes
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

        # New: A gating neuron to introduce spike-based sparsity before GELU
        self.pre_gelu_gating_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)

        # Existing: A gating neuron to introduce spike-based sparsity after GELU
        # Renamed for clarity in the context of dual gating
        self.post_gelu_gating_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_out_cont, spike1 = self.fc1(x) # spike1 is binary

        # --- New Pre-GELU Gating ---
        # Apply a gating neuron to the continuous output of fc1
        # This produces spikes of shape (B, L, d_ffn)
        x_fc1_out_cont_transposed = x_fc1_out_cont.transpose(-1, -2) # (B, d_ffn, L)
        pre_gelu_spikes_transposed = self.pre_gelu_gating_neuron(x_fc1_out_cont_transposed) # (B, d_ffn, L), binary
        pre_gelu_spikes = pre_gelu_spikes_transposed.transpose(-1, -2) # (B, L, d_ffn), binary

        # Gate the fc1 output using these spikes BEFORE GELU
        # This introduces sparsity earlier in the FFN path
        x_fc1_gated_cont = x_fc1_out_cont * pre_gelu_spikes # Sparse-continuous signal

        # Apply non-linear activation (GELU) to the pre-gated output
        x_gelu_cont = F.gelu(x_fc1_gated_cont) # Output is continuous

        # --- Existing Post-GELU Gating (renamed) ---
        # Apply the gating neuron to the continuous output of GELU
        x_gelu_cont_transposed = x_gelu_cont.transpose(-1, -2) # (B, d_ffn, L)
        post_gelu_spikes_transposed = self.post_gelu_gating_neuron(x_gelu_cont_transposed) # (B, d_ffn, L), binary
        post_gelu_spikes = post_gelu_spikes_transposed.transpose(-1, -2) # (B, L, d_ffn), binary

        # Gate the GELU output using the newly generated spikes.
        # This provides a second layer of sparsity control
        x_gated_sparse_cont = x_gelu_cont * post_gelu_spikes

        x_fc2_in = self.dropout(x_gated_sparse_cont) # Input to fc2 is sparse-continuous
        x_fc2_out_cont, spike2 = self.fc2(x_fc2_in) # spike2 is binary

        # Return the final continuous output and all collected spikes
        return x_fc2_out_cont, spike1, spike2, pre_gelu_spikes, post_gelu_spikes

# ==================== SpikingFLABlock (Main Block) ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Attention block uses LightNet, which already leverages SpikingLinear
        self.attn = LightNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        # MLP block uses the enhanced SpikingMLP with dual gated hidden state
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        all_spikes_list = []

        # --- Attention Block (LightNet) ---
        x_res = x
        if self.prenorm: 
            x_norm1 = self.norm1(x) # RMSNorm output is continuous
        else:
            x_norm1 = x

        # attn returns (continuous_output, None, dict_of_binary_spikes)
        x_attn_out_cont, _, attn_spikes_dict = self.attn(x_norm1, past_key_values=past_key_values, **kwargs)

        if attn_spikes_dict: 
            all_spikes_list.extend(attn_spikes_dict.values()) # Collect all spikes from LightNet

        x = x_res + self.dropout1(x_attn_out_cont) # Residual connection with continuous output

        # --- MLP Block ---
        x_res = x
        if self.prenorm: 
            x_norm2 = self.norm2(x) # RMSNorm output is continuous
        else:
            x_norm2 = x

        # mlp returns (continuous_output, binary_spike1, binary_spike2, pre_gelu_spikes, post_gelu_spikes)
        x_mlp_out_cont, s1, s2, s_pre_gelu, s_post_gelu = self.mlp(x_norm2)
        all_spikes_list.extend([s1, s2, s_pre_gelu, s_post_gelu]) # Collect all spikes from SpikingMLP

        x = x_res + self.dropout2(x_mlp_out_cont) # Residual connection with continuous output

        # Final spike collection: flatten and concatenate all collected binary spikes
        # The resulting tensor will be of dtype=torch.bool, representing binary spikes.
        if all_spikes_list:
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])
        else:
            # Handle case of no spikes (though with current design, this should not happen)
            final_spikes = torch.empty(0, device=x.device, dtype=torch.bool)

        # The function signature requires returning (output, None, final_spikes)
        return x, None, final_spikes