# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from typing import Tuple

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
        # The neuron should be instantiated here using the registry
        # The registry entry is expected to return a function/module that applies the spiking nonlinearity
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)
        # Apply neuron: output spikes are binary
        spikes_transposed = self.neuron(x_transposed)
        # Transpose spikes back to (B, L, D)
        spikes = spikes_transposed.transpose(-1, -2)
        # Apply linear layer to the spikes
        output = self.linear(spikes)
        return output, spikes # Return real-valued output and binary spikes

# ==================== White-box Operator (Gated Delta Product) ====================

def gated_delta_chunkwise_parallel(q, k, v, beta, decay, chunk_size=64, initial_state=None):
    """White-box parallel Gated Delta Product operator.
    Uses Wy transform approximation to handle intra-chunk dependencies, ensuring mathematical logic is visible to Sampler.
    Note: q, k, v are expected to be binary spikes for enhanced SNN nature.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len), value=0.0)
        decay = F.pad(decay, (0, pad_len), value=1.0)
    
    # 1. Normalization and data reshaping (q, k are already spikes, so their norm is effectively sqrt(count of 1s))
    # For spikes, a simple L2 norm might not be ideal as it just scales by number of active spikes.
    # However, to match the original DeltaProduct's expectations of normalized vectors for QK similarity,
    # we retain this, understanding it will be `sqrt(N_spikes)` if spikes are 1s and 0s.
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    
    # padded_len = l + pad_len # unused variable, removed to keep code clean
    q, k, v, beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, beta, decay]
    )
    
    num_chunks = q.shape[2]
    # S shape: [B, H, Dk, Dv] (aligned with Delta Rule: mapping Key space to Value space)
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, b_i, d_i = q[:, :, i], k[:, :, i], v[:, :, i], beta[:, :, i], decay[:, :, i]
        
        # [Step 1] Read historical state
        o_inter = q_i @ S
        
        # [Step 2] Intra-chunk parallel update logic (Wy Transform approximation)
        v_error = (v_i - (k_i @ S)) * b_i.unsqueeze(-1) # v_i is now spikes
        
        attn_intra = (k_i @ k_i.transpose(-1, -2)) * mask_tril
        v_error_intra = v_error - (attn_intra @ v_error) * 0.5 
        
        # [Step 3] Combined output
        o_intra = q_i @ (k_i.transpose(-1, -2) @ v_error_intra) # q_i, k_i are now spikes
        o[:, :, i] = o_inter + o_intra
        
        # [Step 4] Recurrent state update
        d_mean = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1)
        S = d_mean * S + (k_i.transpose(-1, -2) @ v_error_intra) # k_i is now spikes
        
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingGatedDeltaProduct Layer ====================

class SpikingGatedDeltaProduct(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.use_short_conv = use_short_conv

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections (6-way projection: Q, K, V, Beta, Decay, Gate)
        # These SpikingLinear layers take real-valued input and produce real-valued output and input spikes.
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        self.w_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        if use_short_conv:
            # ShortConvolution layers operate on the *real-valued* outputs of the initial projections.
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            self.g_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        # NEW: Spiking neurons applied AFTER ShortConvolution outputs (or direct projection outputs) for q, k, v streams
        # to ensure binary inputs to the core gated_delta_chunkwise_parallel operator. The input to these neurons is real-valued,
        # and their output is binary spikes.
        self.q_core_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.k_core_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.v_core_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        # For beta, w, gate, their transformation is non-spiking (sigmoid, silu), so no additional neuron is applied here.
        # The original 's_b', 's_w', 's_g' already capture the spikes *into* their respective linear layers.

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}

        # 1. Spike projection (outputs `_real` are real-valued outputs of the linear layer, `s_` are the binary spikes *fed into* the linear layer)
        q_real, s_q_proj = self.q_proj(x)
        k_real, s_k_proj = self.k_proj(x)
        v_real, s_v_proj = self.v_proj(x)
        beta_real, s_b_proj = self.b_proj(x)
        w_real, s_w_proj = self.w_proj(x)
        gate_real, s_g_proj = self.g_proj(x)
        all_spikes.update({'q_proj_in_spikes':s_q_proj, 'k_proj_in_spikes':s_k_proj, 'v_proj_in_spikes':s_v_proj, 'beta_proj_in_spikes':s_b_proj, 'w_proj_in_spikes':s_w_proj, 'g_proj_in_spikes':s_g_proj})

        # 2. Local feature extraction and subsequent spiking for q, k, v streams
        if self.use_short_conv:
            # ShortConvolution operates on real-valued outputs of initial projections.
            q_conv_out, _ = self.q_conv1d(q_real)
            k_conv_out, _ = self.k_conv1d(k_real)
            v_conv_out, _ = self.v_conv1d(v_real)
            g_conv_out, _ = self.g_conv1d(gate_real)

            # Apply spiking neurons to the real-valued output of convolutions for q, k, v
            # Transpose for neuron operation, then transpose back to (B, L, D)
            q_core_input = self.q_core_neuron(q_conv_out.transpose(-1,-2)).transpose(-1,-2)
            k_core_input = self.k_core_neuron(k_conv_out.transpose(-1,-2)).transpose(-1,-2)
            v_core_input = self.v_core_neuron(v_conv_out.transpose(-1,-2)).transpose(-1,-2)
            # Collect these spikes
            all_spikes.update({'q_core_spikes':q_core_input, 'k_core_spikes':k_core_input, 'v_core_spikes':v_core_input})

            # Use the real-valued conv output for the gate signal (F.silu is real-valued)
            gate_signal = F.silu(g_conv_out)

        else:
            # If no short conv, directly spike the real-valued projection outputs for q, k, v before core operator
            q_core_input = self.q_core_neuron(q_real.transpose(-1,-2)).transpose(-1,-2)
            k_core_input = self.k_core_neuron(k_real.transpose(-1,-2)).transpose(-1,-2)
            v_core_input = self.v_core_neuron(v_real.transpose(-1,-2)).transpose(-1,-2)
            # Collect these spikes
            all_spikes.update({'q_core_spikes':q_core_input, 'k_core_spikes':k_core_input, 'v_core_spikes':v_core_input})

            # Use the real-valued projection output for the gate signal (F.silu is real-valued)
            gate_signal = F.silu(gate_real)

        # 3. Activation function alignment for control signals (beta, decay)
        # These are derived from real-valued outputs of SpikingLinear and remain real-valued for modulation.
        beta_val = torch.sigmoid(beta_real) * 0.5 # Learning rate step mapping
        decay_val = torch.sigmoid(w_real) # State decay mapping (0, 1)

        # 4. Data reshaping
        q_core_input = rearrange(q_core_input, 'b l (h d) -> b h l d', h=self.num_heads)
        k_core_input = rearrange(k_core_input, 'b l (h d) -> b h l d', h=self.num_heads)
        v_core_input = rearrange(v_core_input, 'b l (h d) -> b h l d', h=self.num_heads)
        beta_val = rearrange(beta_val, 'b l h -> b h l')
        decay_val = rearrange(decay_val, 'b l h -> b h l')

        # 5. State management
        layer_state = None
        if past_key_values is not None and self.layer_idx in past_key_values:
            layer_state = past_key_values[self.layer_idx]

        # 6. Execute core operator (Parallel) with explicitly spiked q, k, v inputs
        o, final_state = gated_delta_chunkwise_parallel(
            q_core_input, k_core_input, v_core_input, beta_val, decay_val, initial_state=layer_state
        )

        if past_key_values is not None:
            past_key_values[self.layer_idx] = final_state

        # 7. Gate merging and final projection (aligned with FLA GatedDeltaProduct)
        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)

        # Apply output gate: [B, L, H, D] * [B, L, H, D]
        # Gate signal is derived from real-valued output for smoother modulation
        gate_signal = rearrange(gate_signal, 'b l (h d) -> b l h d', h=self.num_heads)
        o = o * gate_signal

        o = rearrange(o, 'b l h d -> b l (h d)')
        o_final_out, s_o_proj = self.o_proj(o)
        all_spikes['o_proj_in_spikes'] = s_o_proj

        return o_final_out, None, all_spikes

# [SpikingMLP and SpikingFLABlock]

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x) # x is real-valued output of linear(spikes), spike1 is the binary input to fc1.linear
        x = F.gelu(x) 
        x_dropped = self.dropout(x) # dropout operates on real-valued activation
        x_out, spike2 = self.fc2(x_dropped) # x_dropped is real-valued, spike2 is the binary input to fc2.linear
        return x_out, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingGatedDeltaProduct(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            # Pass the surrogate function directly to SpikingMLP
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        all_spikes_list = []
        x_res = x
        if self.prenorm: 
            x = self.norm1(x)
        x_attn, _, attn_spikes_dict = self.attn(x, **kwargs) # attn_spikes is now a dictionary of spike tensors
        if attn_spikes_dict: 
            # Extend with all spike tensors from the dictionary
            all_spikes_list.extend(attn_spikes_dict.values())

        x = x_res + self.dropout1(x_attn)
        x_res = x
        if self.prenorm: 
            x = self.norm2(x)

        if isinstance(self.mlp, SpikingMLP):
            x_mlp, s1, s2 = self.mlp(x)
            all_spikes_list.extend([s1, s2])
        else:
            x_mlp = self.mlp(x)

        x = x_res + x_mlp # Retained Version 2's residual connection for MLP output

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None
        return x, None, final_spikes