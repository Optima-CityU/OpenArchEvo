import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, List

from fla.modules import RMSNorm, ShortConvolution

# ==================== SNN Imports ====================
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
        x_transposed = x.transpose(-1, -2) 
        spikes = self.neuron(x_transposed).transpose(-1, -2) # Spikes are binary (0 or 1)
        output = self.linear(spikes) # Linear layer operates on binary spikes
        return output, spikes # output is continuous, spikes are binary

# ==================== White-box Operator (GSA) ====================

def gsa_chunkwise(q, k, v, f, chunk_size=64, initial_state=None):
    """GSA Parallel Chunkwise Logic using Factorized Attention.
    Significantly reduces memory by avoiding the materialization of the 5D decay mask.
    Formula:
        Score_ij = Sum_m (Q_im * exp(C_im)) * (K_jm * exp(-C_jm))
    """
    b, h, l, d = v.shape
    num_slots = q.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v, f = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v, f]]
        
    q = rearrange(q, 'b h (n c) m -> b h n c m', c=chunk_size)
    k = rearrange(k, 'b h (n c) m -> b h n c m', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    f = rearrange(f, 'b h (n c) m -> b h n c m', c=chunk_size)
    
    # Softmax addressing
    k = torch.softmax(k, dim=-1)
    q = torch.softmax(q, dim=-1)
    
    # Log-space decay
    f = F.logsigmoid(f)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, num_slots, d, device=q.device, dtype=q.dtype) # Initialize state with correct dtype
        
    o_list = []
    
    for i in range(q.shape[2]):
        q_chunk = q[:, :, i] # [B, H, C, M]
        k_chunk = k[:, :, i] 
        v_chunk = v[:, :, i] # [B, H, C, D]
        f_chunk = f[:, :, i] # [B, H, C, M]
        
        # --- 1. Intra-Chunk (Factorized Form) ---
        # Cumsum of log-decay
        f_cumsum = torch.cumsum(f_chunk, dim=-2) # [B, H, C, M]
        
        decay_exp = torch.exp(f_cumsum)      # [B, H, C, M] (Decay)
        decay_inv_exp = torch.exp(-f_cumsum) # [B, H, C, M] (Growth)
        
        # Modulated Q and K
        q_mod = q_chunk * decay_exp
        k_mod = k_chunk * decay_inv_exp
        
        # Attention Matrix: (Q_mod @ K_mod^T)
        # [B, H, C, M] @ [B, H, M, C] -> [B, H, C, C]
        attn_score = q_mod @ k_mod.transpose(-1, -2)
        
        # Apply Causal Mask: explicit unsqueeze for robust broadcasting
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        attn_score = attn_score.masked_fill(mask.unsqueeze(0).unsqueeze(0), 0)
        
        # O_intra = Attn @ V
        o_intra = attn_score @ v_chunk
        
        # --- 2. Inter-Chunk ---
        # O_inter = (Q * Decay) @ S
        q_decayed = q_chunk * decay_exp
        
        # [B, H, C, M] @ [B, H, M, D] -> [B, H, C, D]
        o_inter = q_decayed @ S
        
        # Output accumulation
        o_list.append(o_intra + o_inter)
        
        # --- 3. Update State ---
        # S_new = S * Total_Decay + Sum(K * V * Decay_to_end)
        
        # Total decay for the whole chunk
        total_decay = torch.exp(f_chunk.sum(dim=-2)).unsqueeze(-1) # [B, H, M, 1]
        
        # Decay from t to end: exp(C_end - C_t) = exp(C_end) * exp(-C_t)
        c_end = f_cumsum[:, :, -1, :].unsqueeze(-2) # [B, H, 1, M]
        decay_to_end_const = torch.exp(c_end)
        
        # k_weighted = K * exp(-C_t) * exp(C_end)
        k_weighted = k_mod * decay_to_end_const
        
        # Update: [B, H, M, C] @ [B, H, C, D] -> [B, H, M, D]
        chunk_update = k_weighted.transpose(-1, -2) @ v_chunk
        
        # Calculate new state
        S = S * total_decay + chunk_update
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :] 
    return o, S

# ==================== Main Layer ====================

class GSA(nn.Module):
    def __init__(self, d_model, num_heads, num_slots=64, expand_v=1.0, use_short_conv=True, conv_size=4, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.value_dim = int(d_model * expand_v)
        surrogate = piecewise_quadratic_surrogate()

        # Projections, all wrapped by SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.f_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate) 
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate) # This is a gating for the final output

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.value_dim // num_heads)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            self.f_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')
        else: # Ensure identity layers are present when short conv is off for consistent forward calls
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()
            self.f_conv1d = nn.Identity()

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {} # Collect spikes from projections

        # All projections are SpikingLinear, so they return (continuous_output, binary_spikes)
        q_cont, s_q = self.q_proj(x)
        k_cont, s_k = self.k_proj(x)
        v_cont, s_v = self.v_proj(x)
        f_cont, s_f = self.f_proj(x)
        g_cont, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'f': s_f, 'g': s_g})

        # ShortConvolution for each projected component. Caching for conv state is handled by ShortConvolution internally.
        q_conv_out, _ = self.q_conv1d(q_cont)
        k_conv_out, _ = self.k_conv1d(k_cont)
        v_conv_out, _ = self.v_conv1d(v_cont)
        f_conv_out, _ = self.f_conv1d(f_cont)

        # Rearrange for multi-head attention
        q_re = rearrange(q_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        k_re = rearrange(k_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        v_re = rearrange(v_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        f_re = rearrange(f_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)

        last_state = None
        # Handle cache for recurrent_state (S in GSA)
        if past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 state_dict = past_key_values[layer_idx]
                 if isinstance(state_dict, dict) and 'recurrent_state' in state_dict:
                    last_state = state_dict['recurrent_state']

        # Core GSA computation
        o_raw_attn_output, final_state = gsa_chunkwise(q_re, k_re, v_re, f_re, chunk_size=64, initial_state=last_state)

        # Update cache with new recurrent state
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx: # If layer_idx is new, append
                past_key_values.append({'recurrent_state': final_state})
            else: # Otherwise, update existing
                past_key_values[layer_idx]['recurrent_state'] = final_state

        # Apply gating mechanism
        # Rearrange output for normalization per head-dimension
        o = rearrange(o_raw_attn_output, 'b h l d -> b l h d')
        o = self.g_norm(o) # RMSNorm on the (h d) dimension

        # Apply silu gate using g_cont (gating is not passed through conv here in GSA)
        # Rearrange g_cont to match o's dimensions before element-wise multiplication
        g_re = rearrange(g_cont, 'b l (h d) -> b l h d', h=self.num_heads) 
        o = o * F.silu(g_re) # Element-wise gating

        # Rearrange back to (b, l, d_model)
        o = rearrange(o, 'b l h d -> b l (h d)')

        # Final output projection
        o_cont, s_o = self.o_proj(o)
        all_spikes['o'] = s_o
        return o_cont, None, all_spikes # attn_reg_loss is None for this component

# [SpikingMLP Wrapper]

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout_mid = nn.Dropout(dropout) 
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x_cont_1, spike1 = self.fc1(x)
        x_gelu = F.gelu(x_cont_1) # GELU operates on continuous output of SpikingLinear
        x_drop = self.dropout_mid(x_gelu)
        # fc2 gets continuous input, produces continuous output and binary spikes
        x_cont_2, spike2 = self.fc2(x_drop)
        return x_cont_2, [spike1, spike2] # Return continuous output and list of spike tensors

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', use_gate_conv: bool=True, gate_conv_size: int=3, use_multi_scale_gate_conv: bool=True, gate_conv_sizes: Optional[List[int]]=None, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        surrogate_function = piecewise_quadratic_surrogate() 

        self.attn = GSA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.use_multi_scale_gate_conv = use_multi_scale_gate_conv
        if self.use_multi_scale_gate_conv:
            if gate_conv_sizes is None:
                # Default multi-scale kernels: original and a slightly larger one
                self.gate_conv_sizes = [gate_conv_size, gate_conv_size * 2 - 1]
            else:
                self.gate_conv_sizes = gate_conv_sizes

            # Create ModuleList for multi-scale ShortConvolution layers
            self.post_attn_residual_gate_conv_add = nn.ModuleList([
                ShortConvolution(d_model, k, activation='silu') for k in self.gate_conv_sizes
            ])
            self.post_mlp_residual_gate_conv_add = nn.ModuleList([
                ShortConvolution(d_model, k, activation='silu') for k in self.gate_conv_sizes
            ])
        else:
            # Original single ShortConvolution for gating
            self.post_attn_residual_gate_conv_add = ShortConvolution(d_model, gate_conv_size, activation='silu')
            self.post_mlp_residual_gate_conv_add = ShortConvolution(d_model, gate_conv_size, activation='silu')

        self.post_attn_residual_gate_add = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)

        self.post_mlp_residual_gate_add = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict]]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []
        total_reg_loss = None # Always None as per Version 2 simplification

        x_res1 = x
        x_norm1 = self.norm1(x) if self.prenorm else x 

        x_attn_output_continuous, _, attn_spikes_dict = self.attn(x_norm1, past_key_values=past_key_values, use_cache=use_cache, layer_idx=layer_idx, **kwargs)
        if attn_spikes_dict: 
            all_spikes_list.extend(attn_spikes_dict.values())

        # Residual connection after attention block with additive gating
        branch_output_attn = self.dropout1(x_attn_output_continuous)

        # Multi-scale convolutional gating for attention block
        if self.use_multi_scale_gate_conv:
            gate_input_attn_conv_outputs = []
            for conv_layer in self.post_attn_residual_gate_conv_add:
                conv_out, _ = conv_layer(branch_output_attn)
                gate_input_attn_conv_outputs.append(conv_out)
            gate_input_attn_conv_add = sum(gate_input_attn_conv_outputs)
        else:
            gate_input_attn_conv_add, _ = self.post_attn_residual_gate_conv_add(branch_output_attn)

        gate_add_cont_attn, s_gate_add_attn = self.post_attn_residual_gate_add(gate_input_attn_conv_add)
        all_spikes_list.append(s_gate_add_attn)

        gating_factor_add_attn = torch.sigmoid(gate_add_cont_attn)

        x_after_attn_spike_gated = x_res1 + branch_output_attn * gating_factor_add_attn

        x_res2 = x_after_attn_spike_gated 
        x_norm2 = self.norm2(x_after_attn_spike_gated) if self.prenorm else x_after_attn_spike_gated 

        x_mlp_output_continuous, mlp_all_spikes = self.mlp(x_norm2)
        all_spikes_list.extend(mlp_all_spikes)

        # Residual connection after MLP block with additive gating
        branch_output_mlp = self.dropout2(x_mlp_output_continuous)

        # Multi-scale convolutional gating for MLP block
        if self.use_multi_scale_gate_conv:
            gate_input_mlp_conv_outputs = []
            for conv_layer in self.post_mlp_residual_gate_conv_add:
                conv_out, _ = conv_layer(branch_output_mlp)
                gate_input_mlp_conv_outputs.append(conv_out)
            gate_input_mlp_conv_add = sum(gate_input_mlp_conv_outputs)
        else:
            gate_input_mlp_conv_add, _ = self.post_mlp_residual_gate_conv_add(branch_output_mlp)

        gate_add_cont_mlp, s_gate_add_mlp = self.post_mlp_residual_gate_add(gate_input_mlp_conv_add)
        all_spikes_list.append(s_gate_add_mlp)

        gating_factor_add_mlp = torch.sigmoid(gate_add_cont_mlp)

        x_final_output_gated = x_res2 + branch_output_mlp * gating_factor_add_mlp

        # Concatenate all collected spikes into a single flat tensor
        if all_spikes_list:
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])
        else:
            # Return an empty tensor if no spikes are collected, matching the expected type
            # and ensuring batch-size independence if list could be empty.
            # Use dtype=torch.bool for binary spikes.
            final_spikes = torch.empty(0, device=x.device, dtype=torch.bool) 

        return x_final_output_gated, total_reg_loss, final_spikes