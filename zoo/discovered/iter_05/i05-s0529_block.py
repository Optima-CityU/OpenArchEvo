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
    # Placeholder definitions if SNN modules are not available
    class DummySurrogate:
        @staticmethod
        def apply(x, *args, **kwargs):
            # Simple thresholding for dummy surrogate
            return (x > 0).float() 

    class DummyNeuron:
        def __init__(self, surrogate_function=None):
            self.surrogate_function = surrogate_function if surrogate_function else DummySurrogate.apply
        def __call__(self, x):
            # The neuron expects input in (B, D, L) format, transpose for dummy
            # Dummy just thresholds, no state, so transpose is for consistency
            return self.surrogate_function(x)

    neuron_registry = {'sdn': DummyNeuron}
    piecewise_quadratic_surrogate = lambda: DummySurrogate.apply # Return the static method as the surrogate function
# --- End of Placeholders ---

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed) # Neuron converts input to binary spikes
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        output = self.linear(spikes) # Linear layer receives binary spikes
        return output, spikes # output is continuous, spikes are binary input to linear layer

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
    # q, k are expected to be binary spikes here. l2norm is still applicable.
    # If q and k are binary spikes (0 or 1), l2norm effectively scales the 1s based on sparsity.
    q = l2norm(q) 
    k = l2norm(k)
    v = v * beta[..., None] # v is binary spike scaled by continuous beta
    k_beta = k * beta[..., None] # k is normalized binary spike scaled by continuous beta
    
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    # The matrix multiplications below now operate with (normalized) binary q, k and continuous v, beta.
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

class DeltaNet(nn.Module):
    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> DeltaNet:
        super().__init__()

        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")

        self.neuron_type = neuron_type
        self.surrogate_function = piecewise_quadratic_surrogate()

        self.mode = mode
        self.qk_activation = qk_activation 
        self.qk_norm = qk_norm
        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity'] 
        assert self.qk_norm in ['l2', 'sum']
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        # All linear projections are now SpikingLinear, meaning their internal nn.Linear receives spikes
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), self.neuron_type, self.surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), self.neuron_type, self.surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), self.neuron_type, self.surrogate_function)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), self.neuron_type, self.surrogate_function)

        if use_short_conv:
            self.conv_size = conv_size
            # ShortConvolution operates on continuous signals; no internal spiking here.
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation=None 
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation=None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation=None
            )

        # Spiking neurons for q, k, v after projections/convolutions, ensuring inputs to delta_rule_chunkwise are binary spikes
        self.post_proj_q_spike = neuron_registry[self.neuron_type](surrogate_function=self.surrogate_function)
        self.post_proj_k_spike = neuron_registry[self.neuron_type](surrogate_function=self.surrogate_function)
        self.post_proj_v_spike = neuron_registry[self.neuron_type](surrogate_function=self.surrogate_function)

        if use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), self.neuron_type, self.surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps) # RMSNorm operates on continuous values
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps) # RMSNorm operates on continuous values

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), self.neuron_type, self.surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: Unpack[Dict]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        _local_spikes_list: List[torch.Tensor] = [] # Collect all spike tensors locally

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "Expected attention_mask as a 0-1 matrix"

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        # hidden_states (input to q_proj, k_proj, v_proj) is now expected to be spikes from SpikingFLABlock
        q_proj_output, spikes_q_proj_linear_input = self.q_proj(hidden_states)
        k_proj_output, spikes_k_proj_linear_input = self.k_proj(hidden_states)
        v_proj_output, spikes_v_proj_linear_input = self.v_proj(hidden_states)

        _local_spikes_list.append(spikes_q_proj_linear_input.flatten())
        _local_spikes_list.append(spikes_k_proj_linear_input.flatten())
        _local_spikes_list.append(spikes_v_proj_linear_input.flatten())

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            # ShortConvolution operates on continuous signals derived from the SpikingLinear outputs
            q_conv_output, conv_state_q = self.q_conv1d(x=q_proj_output, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k_conv_output, conv_state_k = self.k_conv1d(x=k_proj_output, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v_conv_output, conv_state_v = self.v_conv1d(x=v_proj_output, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

            # Apply spiking neuron *after* convolution for q, k, v. These are the spikes used by delta_rule_chunkwise.
            q_attention_input_spikes = self.post_proj_q_spike(q_conv_output.transpose(-1, -2)).transpose(-1, -2)
            k_attention_input_spikes = self.post_proj_k_spike(k_conv_output.transpose(-1, -2)).transpose(-1, -2)
            v_attention_input_spikes = self.post_proj_v_spike(v_conv_output.transpose(-1, -2)).transpose(-1, -2)

            _local_spikes_list.append(q_attention_input_spikes.flatten())
            _local_spikes_list.append(k_attention_input_spikes.flatten())
            _local_spikes_list.append(v_attention_input_spikes.flatten())

        else:
            # Apply spiking neuron directly to projected outputs for q, k, v. These are the spikes used by delta_rule_chunkwise.
            q_attention_input_spikes = self.post_proj_q_spike(q_proj_output.transpose(-1, -2)).transpose(-1, -2)
            k_attention_input_spikes = self.post_proj_k_spike(k_proj_output.transpose(-1, -2)).transpose(-1, -2)
            v_attention_input_spikes = self.post_proj_v_spike(v_proj_output.transpose(-1, -2)).transpose(-1, -2)

            _local_spikes_list.append(q_attention_input_spikes.flatten())
            _local_spikes_list.append(k_attention_input_spikes.flatten())
            _local_spikes_list.append(v_attention_input_spikes.flatten())

        # Prepare q, k, v for chunkwise attention calculation
        q_attn = rearrange(q_attention_input_spikes, '... (h d) -> ... h d', d=self.head_k_dim)
        k_attn = rearrange(k_attention_input_spikes, '... (h d) -> ... h d', d=self.head_k_dim)
        v_attn = rearrange(v_attention_input_spikes, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.use_beta:
            # hidden_states (input to b_proj) is now expected to be spikes from SpikingFLABlock
            beta_output, spikes_beta_linear_input = self.b_proj(hidden_states)
            beta = beta_output.sigmoid() # beta itself is continuous, but its calculation involved spikes as input to b_proj
            _local_spikes_list.append(spikes_beta_linear_input.flatten())
        else:
            beta = torch.ones_like(q_attn[..., 0])

        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state.get('recurrent_state', None) if last_state is not None else None
        q_attn = rearrange(q_attn, 'b l h d -> b h l d')
        k_attn = rearrange(k_attn, 'b l h d -> b h l d')
        v_attn = rearrange(v_attn, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')

        o, recurrent_state = delta_rule_chunkwise(q=q_attn, k=k_attn, v=v_attn, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')

        if past_key_values is not None:
            cache_data = {
                'recurrent_state': recurrent_state,
                'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None
            }
            if self.layer_idx is not None:
                past_key_values[self.layer_idx] = cache_data

        if self.use_gate:
            # hidden_states (input to g_proj) is now expected to be spikes from SpikingFLABlock
            g_output, spikes_g_linear_input = self.g_proj(hidden_states)
            g = rearrange(g_output, '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g) # o_norm is an ANN operation, inputs are continuous from previous steps
            _local_spikes_list.append(spikes_g_linear_input.flatten())
        else:
            o = self.o_norm(o) # o_norm is an ANN operation, inputs are continuous

        o = rearrange(o, 'b t h d -> b t (h d)')

        # o here is continuous. spikes_o_linear_input are generated by self.o_proj.neuron from this continuous 'o'.
        o_final_output, spikes_o_linear_input = self.o_proj(o) 
        _local_spikes_list.append(spikes_o_linear_input.flatten())

        o = o_final_output # The output of the block is the continuous output from the final linear layer

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, _local_spikes_list

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")
        # Input 'x' to fc1 is now expected to be spikes from SpikingFLABlock
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        # Removed the explicit inter_neuron. SpikingLinear(self.fc2) will handle spiking its input.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _local_spikes_list: List[torch.Tensor] = []

        # x (input) is expected to be spikes from SpikingFLABlock
        x_fc1_output_continuous, spikes1_fc1_linear_input = self.fc1(x) 
        _local_spikes_list.append(spikes1_fc1_linear_input.flatten())

        # Dropout operates on the continuous output of the first spiking linear layer.
        x_dropped_out_continuous = self.dropout(x_fc1_output_continuous)

        # Removed the explicit inter_neuron. Now fc2 (which is SpikingLinear) directly receives the continuous
        # output after dropout. Its internal neuron will then convert this continuous signal into spikes
        # before feeding it to fc2.linear.
        x_fc2_output_continuous, spikes2_fc2_linear_input = self.fc2(x_dropped_out_continuous)
        _local_spikes_list.append(spikes2_fc2_linear_input.flatten())

        return x_fc2_output_continuous, _local_spikes_list

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model) # RMSNorm operates on continuous values
        # New: Neuron to spike the input before the attention sub-block
        self.spike_input_attn = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model) # RMSNorm operates on continuous values
        # New: Neuron to spike the input before the MLP sub-block
        self.spike_input_mlp = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())
        d_ffn = d_model * d_ffn_mult
        print(f"INFO: Layer uses SpikingMLP with {neuron_type} neuron.")
        # SpikingMLP is now initialized with the chosen neuron type and surrogate function
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # --- Attention Sub-block --- (Processes x to produce x_post_attn_continuous)
        x_res_attn = x # Continuous input from previous layer/block

        # Normalize and then explicitly spike the input before feeding to DeltaNet
        if self.prenorm: 
            x_norm1_continuous = self.norm1(x_res_attn)
            # Transpose for neuron, then transpose back to (B, L, D)
            x_spikes_to_attn = self.spike_input_attn(x_norm1_continuous.transpose(-1, -2)).transpose(-1, -2)
        else: # If no pre-norm, spike the raw input x
            x_spikes_to_attn = self.spike_input_attn(x_res_attn.transpose(-1, -2)).transpose(-1, -2)
        all_spikes_list.append(x_spikes_to_attn.flatten()) # Collect spikes from this new neuron

        # DeltaNet now receives binary spikes as its 'hidden_states' input
        # attn_spikes is now a List[torch.Tensor]
        x_attn_output_continuous, reg_loss, attn_spikes = self.attn(x_spikes_to_attn, **kwargs)

        # Collect all spike tensors returned by DeltaNet
        if attn_spikes: # attn_spikes is already a list of flattened tensors
            all_spikes_list.extend(attn_spikes) 

        # Residual connection: summation of continuous values (voltage accumulation before next spike)
        x_post_attn_continuous = x_res_attn + self.dropout1(x_attn_output_continuous)

        # --- MLP Sub-block --- (Processes x_post_attn_continuous to produce x_final_output_continuous)
        x_res_mlp = x_post_attn_continuous # Continuous output from attention sub-block

        # Normalize and then explicitly spike the input before feeding to SpikingMLP
        if self.prenorm: 
            x_norm2_continuous = self.norm2(x_res_mlp)
            # Transpose for neuron, then transpose back to (B, L, D)
            x_spikes_to_mlp = self.spike_input_mlp(x_norm2_continuous.transpose(-1, -2)).transpose(-1, -2)
        else: # If no pre-norm, spike the raw input x_post_attn_continuous
            x_spikes_to_mlp = self.spike_input_mlp(x_res_mlp.transpose(-1, -2)).transpose(-1, -2)
        all_spikes_list.append(x_spikes_to_mlp.flatten()) # Collect spikes from this new neuron

        # SpikingMLP now receives binary spikes as its input
        # mlp_output_spikes is now a List[torch.Tensor]
        x_mlp_output_continuous, mlp_output_spikes = self.mlp(x_spikes_to_mlp)

        # Collect all spike tensors returned by SpikingMLP
        if mlp_output_spikes: # mlp_output_spikes is already a list of flattened tensors
            all_spikes_list.extend(mlp_output_spikes)

        # Residual connection: summation of continuous values (final output before next block)
        x_final_output_continuous = x_res_mlp + self.dropout2(x_mlp_output_continuous)

        final_spikes = None
        if all_spikes_list:
            # Concatenate all collected flattened spike tensors into a single tensor
            final_spikes = torch.cat(all_spikes_list)

        # The block always returns a continuous output 'x', an optional 'reg_loss', and the 'final_spikes' tensor.
        return x_final_output_continuous, reg_loss, final_spikes