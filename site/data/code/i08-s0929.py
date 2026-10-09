""" spiking_fla_explicit_spike_gating.py """
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

from typing import Optional, Tuple, List, Dict

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    # Dummy classes for environments where SNN modules are not available
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return input
        @staticmethod
        def backward(ctx, grad_output): return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None
    print("Warning: Using placeholder SNN modules due to import error.")

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer.

    The neuron applies to the input 'x', producing spikes.
    The internal 'linear' layer then processes these spikes.
    Returns the continuous output of the linear layer and the binary spikes.
    """

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes = self.neuron(x) # Directly pass x to neuron, no transpose logic
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

# ==================== White-box Operator (Gated Delta Product) ====================

def gated_delta_chunkwise_parallel(q, k, v, beta, decay, accum_gate, forget_gate, chunk_size=64, initial_state: Optional[torch.Tensor]=None):
    """White-box parallel Gated Delta Product operator with Spike-driven Accumulation and Forget Gates.
    Uses Wy transform approximation to handle intra-chunk dependencies, ensuring mathematical logic is visible to Sampler.
    The 'accum_gate' dynamically controls the integration of new information into the recurrent state 'S'.
    The 'forget_gate' dynamically controls the retention of previous state information in 'S'.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # Pad to ensure divisibility by chunk_size
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len)) # pad (L) dimension
        k = F.pad(k, (0, 0, 0, pad_len)) # pad (L) dimension
        v = F.pad(v, (0, 0, 0, pad_len)) # pad (L) dimension
        # Beta, decay, accum_gate, and forget_gate are head-wise but token-dependent, so pad along sequence length
        # Using 0.0 for beta/accum and 1.0 for decay/forget as default for padded tokens (no effect)
        beta = F.pad(beta, (0, pad_len), value=0.0) # pad L
        decay = F.pad(decay, (0, pad_len), value=1.0) # pad L
        accum_gate = F.pad(accum_gate, (0, pad_len), value=0.0) # pad L
        forget_gate = F.pad(forget_gate, (0, pad_len), value=1.0) # pad L
    
    padded_l = l + pad_len

    # 1. Standard normalization for Q and K
    q = l2norm(q) # Using fla.modules.l2norm
    k = l2norm(k) # Using fla.modules.l2norm
    
    # Reshape for chunk processing
    q, k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v])
    beta, decay, accum_gate, forget_gate = map(lambda x: rearrange(x, 'b h (n c) -> b h n c', c=chunk_size), [beta, decay, accum_gate, forget_gate])
    
    num_chunks = q.shape[2]
    # S shape: [B, H, Dk, Dv] (aligned with Delta Rule: mapping Key space to Value space)
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device, dtype=q.dtype)
    o = torch.zeros_like(v)
    
    # Pre-prepare Bool mask for intra-chunk attention (lower triangular without diagonal)
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, b_i, d_i, a_i, f_i = q[:, :, i], k[:, :, i], v[:, :, i], beta[:, :, i], decay[:, :, i], accum_gate[:, :, i], forget_gate[:, :, i]
        
        # [Step 1] Read historical state: contribution from previous chunks
        o_inter = q_i @ S
        
        # [Step 2] Intra-chunk parallel update logic (Wy Transform approximation)
        # Calculate error signal: difference between current value and state-predicted value
        v_error = (v_i - (k_i @ S)) * b_i.unsqueeze(-1)
        
        # Intra-chunk interaction (lower triangular part of K @ K.T)
        attn_intra = (k_i @ k_i.transpose(-1, -2)) * mask_tril
        
        # Apply intra-chunk interactions to error signal.
        # 0.5 is a balancing factor for stability in approximation, preventing runaway gradients.
        v_error_intra = v_error - (attn_intra @ v_error) * 0.5 
        
        # [Step 3] Combined output: sum of inter-chunk (historical) and intra-chunk contributions
        o_intra = q_i @ (k_i.transpose(-1, -2) @ v_error_intra)
        o[:, :, i] = o_inter + o_intra
        
        # [Step 4] Recurrent state update with explicit spike-driven accumulation and forget gates
        
        # Effective accumulation: apply accum_gate to the incoming error signal (per-token accumulation)
        new_information_to_accumulate = k_i.transpose(-1, -2) @ (v_error_intra * a_i.unsqueeze(-1))
        
        # Effective decay factor: combine base decay and forget_gate, averaged over the chunk length
        # This provides a chunk-level multiplicative factor for the entire state S.
        effective_decay_factor = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1) * f_i.mean(dim=-1, keepdim=True).unsqueeze(-1)
        
        # Update recurrent state S
        S = effective_decay_factor * S + new_information_to_accumulate
        
        # Clamp protection for ZCP computation (Zero-Crossing Point/Stability)
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingGatedDeltaProduct Layer ====================

class SpikingGatedDeltaProduct(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_k_dim = self.key_dim // num_heads # Ensure head_k_dim is defined for consistency
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections (Q, K, V, Beta, Decay, Accum Gate, Forget Gate)
        # All nn.Linear layers are wrapped in SpikingLinear as per requirements
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Beta (for Gating V_error)
        self.w_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Weight for decay factor

        # Spike-aware modulation for Beta and Decay based on key spikes (from Version 2)
        self.beta_spike_mod_proj = SpikingLinear(nn.Linear(1, num_heads, bias=True), neuron_type, surrogate)
        self.decay_spike_mod_proj = SpikingLinear(nn.Linear(1, num_heads, bias=True), neuron_type, surrogate)

        # Learnable scaling factors for the spike-aware modulation
        self.beta_mod_scale = nn.Parameter(torch.ones(1))
        self.decay_mod_scale = nn.Parameter(torch.ones(1))

        # Query spike-aware modulation for Decay (from Version 2)
        self.q_spike_mod_proj = SpikingLinear(nn.Linear(1, num_heads, bias=True), neuron_type, surrogate)
        self.q_mod_scale = nn.Parameter(torch.ones(1))

        # NEW: Projection for the Spike-driven Accumulation Gate
        self.accum_gate_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        # NEW: Spike-aware modulation for the Accumulation Gate, based on Query spikes
        self.accum_q_spike_mod_proj = SpikingLinear(nn.Linear(1, num_heads, bias=True), neuron_type, surrogate)
        self.accum_mod_scale = nn.Parameter(torch.ones(1)) # Learnable scaling factor for accumulation modulation

        # NEW: Projection for the Spike-driven Forget Gate
        self.forget_gate_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        # NEW: Spike-aware modulation for the Forget Gate, based on Query spikes
        self.forget_q_spike_mod_proj = SpikingLinear(nn.Linear(1, num_heads, bias=True), neuron_type, surrogate)
        self.forget_mod_scale = nn.Parameter(torch.ones(1)) # Learnable scaling factor for forget modulation

        # ShortConvolution layers remain non-spiking as per instructions and SNN-ANN hybrid design
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
        else:
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict]]=None, use_cache: Optional[bool]=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        all_spikes = {} # Collects spikes from all SpikingLinear layers

        # SpikingLinear layers take real-valued input 'x', apply a spiking neuron internally,
        # and then a linear layer to the resulting spikes. They return the real-valued
        # output of the linear layer and the binary spikes produced by the neuron.
        q_linear_out, s_q = self.q_proj(x)
        k_linear_out, s_k = self.k_proj(x)
        v_linear_out, s_v = self.v_proj(x)
        beta_linear_out, s_b = self.b_proj(x)
        w_linear_out, s_w = self.w_proj(x)
        accum_gate_linear_out, s_accum_gate = self.accum_gate_proj(x) # Accumulation gate projection
        forget_gate_linear_out, s_forget_gate = self.forget_gate_proj(x) # Forget gate projection

        all_spikes.update({
            'q_proj': s_q, 'k_proj': s_k, 'v_proj': s_v,
            'beta_proj': s_b, 'w_proj': s_w,
            'accum_gate_proj': s_accum_gate,
            'forget_gate_proj': s_forget_gate
        })

        q, k, v = q_linear_out, k_linear_out, v_linear_out
        beta, w = beta_linear_out, w_linear_out
        accum_gate, forget_gate = accum_gate_linear_out, forget_gate_linear_out

        if not isinstance(self.q_conv1d, nn.Identity):
            # ShortConvolution handles internal rearrangement from (B, L, D) to (B, D, L) and back.
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

        # Apply sigmoid to beta, w (decay), accum_gate, and forget_gate projections
        beta = torch.sigmoid(beta) * 0.5 # Scale down for stability and gating strength
        decay = torch.sigmoid(w) # Decay factor, typically between 0 and 1
        accum_gate = torch.sigmoid(accum_gate) # Accumulation gate, between 0 and 1
        forget_gate = torch.sigmoid(forget_gate) # Forget gate, between 0 and 1

        # Spike-aware modulation for Beta and Decay based on key spikes
        # Resulting shape: (B, L, 1)
        k_spike_mean_activity = s_k.float().mean(dim=-1, keepdim=True)

        # Project the mean spike activity for beta modulation
        beta_mod_analog_out, s_beta_mod = self.beta_spike_mod_proj(k_spike_mean_activity)
        all_spikes['beta_spike_mod_proj'] = s_beta_mod
        beta = beta * (1 + self.beta_mod_scale * beta_mod_analog_out.tanh()) # Scale by tanh(output of spike proj)

        # Project the mean spike activity for decay modulation
        decay_mod_analog_out, s_decay_mod = self.decay_spike_mod_proj(k_spike_mean_activity)
        all_spikes['decay_spike_mod_proj'] = s_decay_mod
        decay = decay * (1 + self.decay_mod_scale * decay_mod_analog_out.tanh()) # Scale by tanh(output of spike proj)

        # Query spike-aware modulation for Decay
        q_spike_mean_activity = s_q.float().mean(dim=-1, keepdim=True) # (B, L, 1)
        q_mod_analog_out, s_q_mod = self.q_spike_mod_proj(q_spike_mean_activity)
        all_spikes['q_spike_mod_proj'] = s_q_mod
        decay = decay * (1 + self.q_mod_scale * q_mod_analog_out.tanh()) # Scale by tanh(output of spike proj)

        # Spike-aware modulation for the Accumulation Gate, based on Query spikes
        accum_q_mod_analog_out, s_accum_q_mod = self.accum_q_spike_mod_proj(q_spike_mean_activity)
        all_spikes['accum_q_spike_mod_proj'] = s_accum_q_mod
        accum_gate = accum_gate * (1 + self.accum_mod_scale * accum_q_mod_analog_out.tanh()) # Scale by tanh(output of spike proj)

        # Spike-aware modulation for the Forget Gate, based on Query spikes
        forget_q_mod_analog_out, s_forget_q_mod = self.forget_q_spike_mod_proj(q_spike_mean_activity)
        all_spikes['forget_q_spike_mod_proj'] = s_forget_q_mod
        forget_gate = forget_gate * (1 + self.forget_mod_scale * forget_q_mod_analog_out.tanh()) # Scale by tanh(output of spike proj)

        # Ensure beta, decay, accum_gate, and forget_gate are within reasonable bounds after modulation
        beta = torch.clamp(beta, 0.0, 1.0)
        decay = torch.clamp(decay, 0.0, 1.0)
        accum_gate = torch.clamp(accum_gate, 0.0, 1.0)
        forget_gate = torch.clamp(forget_gate, 0.0, 1.0)

        # Reshape for multi-head processing
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        beta = rearrange(beta, 'b l h -> b h l')
        decay = rearrange(decay, 'b l h -> b h l')
        accum_gate = rearrange(accum_gate, 'b l h -> b h l')
        forget_gate = rearrange(forget_gate, 'b l h -> b h l')

        layer_state = None
        if use_cache and past_key_values is not None and self.layer_idx is not None and self.layer_idx < len(past_key_values):
            if past_key_values[self.layer_idx] is not None and 'recurrent_state' in past_key_values[self.layer_idx]:
                layer_state = past_key_values[self.layer_idx]['recurrent_state']

        # Call the chunkwise parallel operator with the new accum_gate and forget_gate
        o, final_state_matrix = gated_delta_chunkwise_parallel(q, k, v, beta, decay, accum_gate, forget_gate, initial_state=layer_state)

        # Update cache for future timesteps
        if use_cache and self.layer_idx is not None:
            if past_key_values is None:
                past_key_values = [None] * (self.layer_idx + 1) # Initialize if None
            while len(past_key_values) <= self.layer_idx:
                past_key_values.append(None)
            past_key_values[self.layer_idx] = {'recurrent_state': final_state_matrix}

        o = rearrange(o, 'b h l d -> b l h d') # Revert to (B, L, H, D_v/H)
        o = self.o_norm(o) # Apply RMSNorm

        o = rearrange(o, 'b l h d -> b l (h d)') # Concatenate heads
        o_linear_out, s_o = self.o_proj(o) # Final linear projection
        all_spikes['o_proj'] = s_o

        return o_linear_out, None, all_spikes # No regularization loss from this block

# [SpikingMLP] - Modified for spike-gated GELU activation (from Version 2, unchanged)

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        # fc1 takes real-valued input, produces real-valued output and binary spikes.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)

        # NEW: Spike-gating for GELU activation
        self.gelu_gate_proj = SpikingLinear(nn.Linear(d_ffn, d_ffn, bias=False), neuron_type, surrogate_function)
        self.gelu_gate_scale = nn.Parameter(torch.ones(1))

        # fc2 takes the real-valued output (after dropout and spike-gated GELU) of fc1.
        # It will internally spike this input before applying its linear layer.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_linear_out1, spike1 = self.fc1(x)

        gelu_input_for_gate = x_linear_out1 # Use continuous output of fc1 for gate
        gelu_out = F.gelu(self.dropout(x_linear_out1)) # Apply GELU after dropout, as is common

        # NEW: Apply spike-gated GELU activation
        _, gelu_gate_spikes = self.gelu_gate_proj(gelu_input_for_gate)
        gated_gelu_out = gelu_out * (1 + self.gelu_gate_scale * gelu_gate_spikes)

        x_final_linear_out, spike2 = self.fc2(gated_gelu_out)

        # Return all collected spikes, including the new GELU gating spikes
        return x_final_linear_out, spike1, spike2, gelu_gate_spikes

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        surrogate_function = piecewise_quadratic_surrogate() # Ensure surrogate is initialized for all SpikingLinear

        self.norm1 = RMSNorm(d_model)
        # SpikingGatedDeltaProduct internally uses SpikingLinear layers.
        # The layer_idx is now passed for cache management.
        self.attn = SpikingGatedDeltaProduct(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # Spiking Gate for attention output (from Version 2) - now for adaptive modulation
        self.attn_output_gate_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.attn_gate_scale = nn.Parameter(torch.ones(1)) # Learnable scaling factor

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        # The MLP is exclusively a SpikingMLP to maximize spiking nature.
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.dropout2 = nn.Dropout(dropout)

        # Hard Spiking Gate for FFN output (analogous to attention gate) - now for adaptive modulation
        self.ffn_output_gate_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.ffn_gate_scale = nn.Parameter(torch.ones(1)) # Learnable scaling factor

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict]]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        x_res_attn = x

        # --- Attention Block ---
        if self.prenorm:
            x_norm_attn = self.norm1(x)
        else:
            x_norm_attn = x

        # x_attn_output is real-valued from SpikingGatedDeltaProduct's o_proj.
        # attn_spikes_dict contains spikes from all internal SpikingLinear layers of SpikingGatedDeltaProduct.
        # Pass layer_idx to attn for cache management
        x_attn_output, _, attn_spikes_dict = self.attn(
            x_norm_attn, past_key_values=past_key_values, use_cache=use_cache, layer_idx=layer_idx, **kwargs
        )
        if attn_spikes_dict:
            # Iterating through dict values is safer than .items() if only values are needed.
            # Convert dict_values to list for concatenation.
            all_spikes_list.extend(list(attn_spikes_dict.values()))


        # Apply adaptive Spiking Gate to the attention output.
        # attn_output_gate_proj takes the real-valued x_attn_output, internally spikes it,
        # then returns real-valued linear output (unused) and binary spikes (attn_gate_spikes).
        _, attn_gate_spikes = self.attn_output_gate_proj(x_attn_output)
        all_spikes_list.append(attn_gate_spikes)

        # Adaptive gating of the attention output using binary spikes.
        # This modulates the signal instead of performing a hard element-wise multiplication,
        # allowing for more nuanced, spike-driven control over information flow in the residual path.
        x_attn_modulated = x_attn_output * (1 + self.attn_gate_scale * attn_gate_spikes)

        # Residual connection with the adaptively modulated attention output.
        x = x_res_attn + self.dropout1(x_attn_modulated)

        # --- FFN Block ---
        x_res_ffn = x 

        if self.prenorm:
            x_norm_ffn = self.norm2(x_res_ffn)
        else:
            x_norm_ffn = x_res_ffn

        # MLP processing: x_mlp_out is real-valued.
        # s_mlp1 and s_mlp2 are the binary spikes from the two SpikingLinear layers within SpikingMLP.
        # NEW: Also collect s_mlp_gelu_gate
        x_mlp_out, s_mlp1, s_mlp2, s_mlp_gelu_gate = self.mlp(x_norm_ffn)
        all_spikes_list.extend([s_mlp1, s_mlp2, s_mlp_gelu_gate]) # Collect all spikes, including the new GELU gating spikes

        # Apply adaptive Spiking Gate to the FFN output.
        # Similar to the attention gate, this modulates the FFN output based on spikes.
        _, ffn_gate_spikes = self.ffn_output_gate_proj(x_mlp_out)
        all_spikes_list.append(ffn_gate_spikes)

        # Adaptive gating of the FFN output.
        x_ffn_modulated = x_mlp_out * (1 + self.ffn_gate_scale * ffn_gate_spikes)

        # Residual connection with the adaptively modulated FFN output.
        x = x_res_ffn + self.dropout2(x_ffn_modulated)

        # Concatenate all collected binary spikes into a single flat tensor.
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None])

        # The second return value `reg_loss` is always None for this architecture.
        return x, None, final_spikes