"""Implements a SpikingFLABlock with feature-wise adaptive MLP gating driven by attention spikes.
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
    
    # Pre-calculate `attn`, `u`, `w` based on the given algorithm structure
    # Mask for `attn` as defined in the provided `delta_rule_chunkwise`
    mask_outer = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    
    # Reshape for chunking (used by `attn` precomputation)
    q_reshaped, k_reshaped, v_reshaped, k_beta_reshaped = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    
    # Original `attn` computation from the provided code
    attn = -(k_beta_reshaped @ k_reshaped.transpose(-1, -2)).masked_fill(mask_outer, 0)
    for i in range(1, chunk_size):
        # Using clone to avoid in-place operations that might interfere with autograd
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v_reshaped
    w = attn @ k_beta_reshaped

    S = k.new_zeros(b, h, d_k, d_v) # State for recurrent dynamics, same shape as initial k
    o = torch.zeros_like(v_reshaped)

    # Mask for `attn_intra` inside the loop, as defined in the provided `delta_rule_chunkwise`
    mask_inner = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)

    # Main loop for chunk-wise processing
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q_reshaped[:, :, i], k_reshaped[:, :, i] # q_i, k_i for current chunk
        attn_intra = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask_inner, 0)
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
    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> 'DeltaNet':
        super().__init__()

        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")

        surrogate_function = piecewise_quadratic_surrogate()

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

        # All nn.Linear layers are replaced by SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), neuron_type, surrogate_function)
            # New: Spike-based modulator for beta
            # It takes the concatenated mean spike activity from q, k, v (3 features) and projects it to num_heads
            self.spike_beta_mod_proj = SpikingLinear(nn.Linear(3, self.num_heads, bias=True), neuron_type, surrogate_function)


        # ShortConvolution layers remain non-spiking as per instructions and SNN-ANN hybrid design
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None # Only apply silu in conv if qk_activation is silu
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None # Only apply silu in conv if qk_activation is silu
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu' # v_conv1d always uses silu as a default non-linearity
            )
        else:
            # If short conv is not used, provide an activation for v
            self.v_activation = nn.SiLU()


        if use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional['Cache']=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: 'Unpack[Dict]') -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional['Cache']]:
        all_spikes = {} # Used to collect all spikes

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "Expected attention_mask as a 0-1 matrix"

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None and hidden_states.dim() == 3: # Only unpad if the input is already padded (B, S, D)
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states_unpadded = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        else:
            hidden_states_unpadded = hidden_states

        # SpikingLinear layers take analog input and produce analog output and binary spikes
        q, spikes_q_proj = self.q_proj(hidden_states_unpadded)
        k, spikes_k_proj = self.k_proj(hidden_states_unpadded)
        v, spikes_v_proj = self.v_proj(hidden_states_unpadded)

        all_spikes['q_proj'] = spikes_q_proj
        all_spikes['k_proj'] = spikes_k_proj
        all_spikes['v_proj'] = spikes_v_proj

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            q, conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            # Apply qk_activation to q and k only if ShortConvolution is NOT used.
            # v also gets its default activation here.
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
                v = self.v_activation(v)
            elif self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
                v = self.v_activation(v)
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
                v = self.v_activation(v)
            elif self.qk_activation == 'identity':
                v = self.v_activation(v)


        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.qk_norm == 'l2':
            q = l2norm(q)
            k = l2norm(k)
        elif self.qk_norm == 'sum':
            q = sum_norm(q)
            k = sum_norm(k)


        if self.use_beta:
            beta, spikes_beta = self.b_proj(hidden_states_unpadded)
            beta = beta.sigmoid()
            all_spikes['b_proj'] = spikes_beta

            # Integrate spike activity for beta modulation
            # Calculate mean spike activity of q, k, v across the feature dimension
            mean_spikes_q = spikes_q_proj.float().mean(dim=-1, keepdim=True) 
            mean_spikes_k = spikes_k_proj.float().mean(dim=-1, keepdim=True)
            mean_spikes_v = spikes_v_proj.float().mean(dim=-1, keepdim=True)

            # Concatenate these mean spike activities to form the input for modulation
            # Resulting shape: (N, 3) or (B, L, 3)
            aggregated_mean_spikes = torch.cat([mean_spikes_q, mean_spikes_k, mean_spikes_v], dim=-1)

            # Project the aggregated mean spike activity to the number of heads using a SpikingLinear layer
            # Resulting shape: (N, num_heads) or (B, L, num_heads)
            beta_spike_mod, spikes_beta_mod = self.spike_beta_mod_proj(aggregated_mean_spikes)
            all_spikes['spike_beta_mod'] = spikes_beta_mod

            # Apply a simpler, bounded modulation (sigmoid instead of tanh) and then scale
            beta = beta * (beta_spike_mod.sigmoid() * 2) # Use sigmoid for direct gating-like modulation

        else:
            # Initialize beta to ones if not using beta projection. Shape depends on q's dimensions.
            if q.dim() == 4: # (B, L, H, D_head) 
                beta = torch.ones(q.shape[0], q.shape[1], q.shape[2], device=q.device) # (B, L, H)
            elif q.dim() == 3: # (N, H, D_head) 
                beta = torch.ones(q.shape[0], q.shape[1], device=q.device) # (N, H)
            else:
                raise ValueError(f"Unexpected q dimension for beta initialization: {q.dim()}")

        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state.get('recurrent_state', None) if last_state is not None else None

        # Ensure q, k, v, beta are in (B, L, H, D) or (B, L, H) format for rearrange
        # If input was unpadded to (N,D), then q is (N,H,D_head). We need to re-pad for chunkwise processing which expects (B,L,H,D)
        if attention_mask is not None and hidden_states.dim() == 3: # Original input was (B,S,D) and unpadded
             # q, k, v, beta are currently (N, H, D_head) or (N, H) after operations on `hidden_states_unpadded`
             # Need to re-pad them before sending to `delta_rule_chunkwise` which expects (B,H,L,D)

            # First rearrange to get 'L' dimension in the correct place for padding
            q_re = rearrange(q, 'n h d -> (n) h d') # placeholder for (N,H,D)
            k_re = rearrange(k, 'n h d -> (n) h d')
            v_re = rearrange(v, 'n h d -> (n) h d')
            beta_re = rearrange(beta, 'n h -> (n) h')

            # Apply pad_input to bring them back to (B, L, ...) shape. pad_input expects (N,D) and returns (B,L,D).
            # Here D would be (H*D_head) or H.
            q_padded = pad_input(rearrange(q_re, 'n h d -> n (h d)'), indices, batch_size, q_len)
            k_padded = pad_input(rearrange(k_re, 'n h d -> n (h d)'), indices, batch_size, q_len)
            v_padded = pad_input(rearrange(v_re, 'n h d -> n (h d)'), indices, batch_size, q_len)
            beta_padded = pad_input(beta_re, indices, batch_size, q_len) # beta is (B, L, H)

            # Rearrange back to (B, H, L, D) for delta_rule_chunkwise
            q = rearrange(q_padded, 'b l (h d) -> b h l d', h=self.num_heads)
            k = rearrange(k_padded, 'b l (h d) -> b h l d', h=self.num_heads)
            v = rearrange(v_padded, 'b l (h d) -> b h l d', h=self.num_heads)
            beta = rearrange(beta_padded, 'b l h -> b h l')
        else: # Input was already (B, L, D) or (N, D) that was not unpadded
            q = rearrange(q, 'b l h d -> b h l d')
            k = rearrange(k, 'b l h d -> b h l d')
            v = rearrange(v, 'b l h d -> b h l d')
            beta = rearrange(beta, 'b l h -> b h l')


        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')

        if past_key_values is not None:
            cache_data = {
                'recurrent_state': recurrent_state,
                'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None
            }
            if self.layer_idx is not None:
                past_key_values[self.layer_idx] = cache_data

        if self.use_gate:
            g, spikes_g = self.g_proj(hidden_states_unpadded)
            g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
            all_spikes['g_proj'] = spikes_g
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')

        o, spikes_o = self.o_proj(o)
        all_spikes['o_proj'] = spikes_o

        if attention_mask is not None and hidden_states.dim() == 3: # Only pad if original input was 3D and unpadded
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, neuron_type: str='sdn', surrogate_function=None, num_attn_spike_sources: int=7):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        # First linear layer, wrapped in SpikingLinear. Takes analog input, outputs analog, and provides spikes.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)

        # Spike-driven feature-wise gating based on attention spikes
        self.num_attn_spike_sources = num_attn_spike_sources
        if self.num_attn_spike_sources > 0:
            self.attn_spike_gating_proj = SpikingLinear(nn.Linear(self.num_attn_spike_sources, d_ffn, bias=True), neuron_type, surrogate_function)
        else:
            self.attn_spike_gating_proj = None

        # Second linear layer, also wrapped in SpikingLinear. Takes analog input, outputs analog, and provides spikes.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor, attention_spike_context: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        spikes_gating: Optional[torch.Tensor] = None

        x_fc1_out, spike1 = self.fc1(x) # x_fc1_out is analog output, spike1 are binary spikes from fc1's internal neuron

        # Apply spike-driven feature-wise gating
        if self.attn_spike_gating_proj is not None and attention_spike_context is not None:
            # attention_spike_context shape: (B, L, num_attn_spike_sources) or (N, num_attn_spike_sources)
            gating_signal_raw, spikes_gating = self.attn_spike_gating_proj(attention_spike_context)
            gating_signal = gating_signal_raw.sigmoid() # Bounded gate
            x_fc1_out = x_fc1_out * gating_signal # Feature-wise multiplication

        # The (analog) output after gating becomes the input to the second SpikingLinear layer.
        x_fc2_out, spike2 = self.fc2(x_fc1_out) # x_fc2_out is analog output, spike2 are binary spikes from fc2's internal neuron

        # Return the final analog output of the MLP and the collected binary spikes from its layers.
        return x_fc2_out, spike1, spike2, spikes_gating

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        # DeltaNet also uses SpikingLinear layers internally.
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        # The maximum number of attention spike sources from DeltaNet is 7:
        # q_proj, k_proj, v_proj (always 3)
        # b_proj, spike_beta_mod (if use_beta, +2)
        # g_proj (if use_gate, +1)
        # o_proj (always +1)
        self.max_attn_spike_sources = 7 
        # Dropping out the `dropout` parameter for SpikingMLP as a simplification
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate(), num_attn_spike_sources=self.max_attn_spike_sources)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        reg_loss = None # DeltaNet returns None for reg_loss

        x_res = x
        if self.prenorm:
            x = self.norm1(x)

        # DeltaNet returns analog output, None for regularization loss, and a dictionary of collected spikes.
        x_attn, _, attn_spikes_dict = self.attn(x, **kwargs) 

        # Prepare attention_spike_context for SpikingMLP
        attention_spike_context: Optional[torch.Tensor] = None
        attn_spike_context_tensors: List[torch.Tensor] = []

        # List of expected spike keys from DeltaNet. Order matters for consistent concatenation and padding.
        expected_attn_spike_keys = ['q_proj', 'k_proj', 'v_proj', 'b_proj', 'spike_beta_mod', 'g_proj', 'o_proj']

        # Determine batch_size and seq_len/total_tokens from x.shape for creating zero tensors
        # The input `x` to `SpikingFLABlock` is always `(B, L, D_model)`.
        batch_size, seq_len = x.shape[0], x.shape[1] 

        for key in expected_attn_spike_keys:
            spike_tensor = attn_spikes_dict.get(key)
            if spike_tensor is not None: # If spike_tensor exists, it means the projection was active in DeltaNet
                all_spikes_list.append(spike_tensor)
                # The spike_tensor from DeltaNet could be (N, D_proj) or (B, L, D_proj)
                # If (N, D_proj), need to re-pad to (B, L, D_proj) first before mean.
                # However, the mean over dim=-1 will give (N,1) or (B,L,1) respectively.
                # The important part is that the output of .mean(dim=-1, keepdim=True) is either (N,1) or (B,L,1).
                # We need it to be (B, L, 1) for consistent concatenation.
                # DeltaNet's internal unpadding means `spikes_..._proj` are often `(N, D_proj)`. We need to re-pad them.

                # If spike_tensor is 2D (N, D_proj), it means DeltaNet processed unpadded data.
                # We need to re-pad it to (B, L, D_proj) before taking mean to get a (B, L, 1) context.
                if spike_tensor.dim() == 2: # (N, D_proj)
                    # Use indices and q_len from kwargs if available for re-padding
                    indices = kwargs.get('indices', None)
                    if indices is not None:
                        # Rearrange to (N, 1, D_proj) to fit pad_input's expected (N,D) to (B,L,D) mapping.
                        # Then take mean to (B,L,1)
                        padded_spike_tensor = pad_input(spike_tensor, indices, batch_size, seq_len) # (B,L,D_proj)
                        mean_spike_activity_per_source = padded_spike_tensor.float().mean(dim=-1, keepdim=True)
                    else:
                        # Fallback for unexpected case, assume it was (B*L, D) and simply average over D
                        mean_spike_activity_per_source = spike_tensor.float().mean(dim=-1, keepdim=True).view(batch_size, seq_len, 1) 
                else: # (B, L, D_proj)
                    mean_spike_activity_per_source = spike_tensor.float().mean(dim=-1, keepdim=True)

                attn_spike_context_tensors.append(mean_spike_activity_per_source)
            else:
                # If a spike source is not present (e.g., if use_beta=False for b_proj), append a zero tensor.
                zero_tensor = torch.zeros(batch_size, seq_len, 1, device=x.device, dtype=torch.float)
                attn_spike_context_tensors.append(zero_tensor)

        if attn_spike_context_tensors: # This list will always have `self.max_attn_spike_sources` elements now.
            attention_spike_context = torch.cat(attn_spike_context_tensors, dim=-1) # Shape (B, L, max_attn_spike_sources)
        else: # Should not happen with max_attn_spike_sources > 0, but for robustness
            attention_spike_context = torch.zeros(batch_size, seq_len, self.max_attn_spike_sources, device=x.device, dtype=torch.float)

        # Residual connection for attention output
        x = x_res + x_attn

        x_res = x
        if self.prenorm:
            x = self.norm2(x)

        # SpikingMLP now accepts attention_spike_context and returns an additional spike tensor.
        x_mlp, mlp_spike1, mlp_spike2, attn_spike_gate_spikes = self.mlp(x, attention_spike_context=attention_spike_context)

        # Collect spikes from the MLP and its gating
        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)
        if attn_spike_gate_spikes is not None:
            all_spikes_list.append(attn_spike_gate_spikes)

        # Dropout layers were removed to embrace spike-driven dynamic sparsity/regularization
        x = x_res + x_mlp

        final_spikes = None
        if all_spikes_list:
            # Concatenate all collected binary spike tensors into a single flattened tensor.
            final_spikes = torch.cat([s.flatten().float() for s in all_spikes_list])

        return x, reg_loss, final_spikes