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
        x_transposed = x.transpose(-1, -2) # Neuron expects (..., dim, seq_len)
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
        S = torch.zeros(b, h, num_slots, d, device=q.device)
        
    o_list = []
    
    for i in range(q.shape[2]):
        q_chunk = q[:, :, i] # [B, H, C, M]
        k_chunk = k[:, :, i] 
        v_chunk = v[:, :, i] # [B, H, C, D]
        f_chunk = f[:, :, i] # [B, H, C, M]
        
        # --- 1. Intra-Chunk (Factorized Form) ---
        # Cumsum of log-decay
        f_cumsum = torch.cumsum(f_chunk, dim=-2) # [B, H, C, M]
        
        # Decompose Decay: exp(Ci - Cj) = exp(Ci) * exp(-Cj)
        # To ensure numerical stability (prevent overflow of exp(-Cj)), we can center C.
        # However, for chunk_size=64, values are usually within range.
        # We proceed with direct decomposition.
        
        decay_exp = torch.exp(f_cumsum)      # [B, H, C, M] (Decay)
        decay_inv_exp = torch.exp(-f_cumsum) # [B, H, C, M] (Growth)
        
        # Modulated Q and K
        q_mod = q_chunk * decay_exp
        k_mod = k_chunk * decay_inv_exp
        
        # Attention Matrix: (Q_mod @ K_mod^T)
        # [B, H, C, M] @ [B, H, M, C] -> [B, H, C, C]
        attn_score = q_mod @ k_mod.transpose(-1, -2)
        
        # Apply Causal Mask
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        attn_score = attn_score.masked_fill(mask, 0)
        
        # O_intra = Attn @ V
        o_intra = attn_score @ v_chunk
        
        # --- 2. Inter-Chunk ---
        # Decay from start: exp(F_cumsum) = decay_exp
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
        # We already have exp(-C_t) as decay_inv_exp.
        c_end = f_cumsum[:, :, -1, :].unsqueeze(-2) # [B, H, 1, M]
        decay_to_end_const = torch.exp(c_end)
        
        # k_weighted = K * exp(-C_t) * exp(C_end)
        k_weighted = k_mod * decay_to_end_const
        
        # Update: [B, H, M, C] @ [B, H, C, D] -> [B, H, M, D]
        chunk_update = k_weighted.transpose(-1, -2) @ v_chunk
        
        # Calculate new state WITHOUT in-place modification
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

        # Projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.f_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate) 
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate) 

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
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        f, s_f = self.f_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'f': s_f, 'g': s_g})

        q_conv_out, _ = self.q_conv1d(q)
        k_conv_out, _ = self.k_conv1d(k)
        v_conv_out, _ = self.v_conv1d(v)
        f_conv_out, _ = self.f_conv1d(f)

        q_re = rearrange(q_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        k_re = rearrange(k_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        v_re = rearrange(v_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        f_re = rearrange(f_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)

        last_state = None
        # Handle Cache (list of dicts) for compatibility with SpikingFLABlock
        if past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 state_dict = past_key_values[layer_idx]
                 if isinstance(state_dict, dict) and 'recurrent_state' in state_dict:
                    last_state = state_dict['recurrent_state']

        o_raw_attn_output, final_state = gsa_chunkwise(q_re, k_re, v_re, f_re, chunk_size=64, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                past_key_values.append({'recurrent_state': final_state})
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_state

        o = rearrange(o_raw_attn_output, 'b h l d -> b l h d')
        o = self.g_norm(o)

        g_re = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads) # Rearrange g after potential conv, but g has no conv in GSA
        o = o * F.silu(g_re) # Apply activation after rearrangement

        o = rearrange(o, 'b l h d -> b l (h d)')
        o_cont, s_o = self.o_proj(o)
        all_spikes['o'] = s_o
        return o_cont, None, all_spikes

# [SpikingFLABlock Wrapper]

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout_mid = nn.Dropout(dropout) 
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x_cont_1, spike1 = self.fc1(x)
        x_gelu = F.gelu(x_cont_1) 
        x_drop = self.dropout_mid(x_gelu)
        x_cont_2, spike2 = self.fc2(x_drop)
        return x_cont_2, [spike1, spike2] # Return output and list of spike tensors

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', use_gate_conv: bool=True, gate_conv_size: int=3, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        # NEW: Spiking Feature Gating layers for attention input modulation
        surrogate_function = piecewise_quadratic_surrogate() 
        self.g_attn_input_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.m_attn_input_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.use_gate_conv = use_gate_conv # Re-using existing flag for new convs
        if self.use_gate_conv:
            self.g_attn_input_conv = ShortConvolution(d_model, gate_conv_size, activation='silu')
            self.m_attn_input_conv = ShortConvolution(d_model, gate_conv_size, activation='silu')
        else:
            self.g_attn_input_conv = nn.Identity()
            self.m_attn_input_conv = nn.Identity()

        self.attn = GSA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.post_attn_residual_gate_conv_add = ShortConvolution(d_model, gate_conv_size, activation='silu') if use_gate_conv else nn.Identity()
        self.post_attn_residual_gate_conv_mult = ShortConvolution(d_model, gate_conv_size, activation='silu') if use_gate_conv else nn.Identity()
        self.post_mlp_residual_gate_conv_add = ShortConvolution(d_model, gate_conv_size, activation='silu') if use_gate_conv else nn.Identity()
        self.post_mlp_residual_gate_conv_mult = ShortConvolution(d_model, gate_conv_size, activation='silu') if use_gate_conv else nn.Identity()


        self.post_attn_residual_gate_add = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.post_attn_residual_gate_mult = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        # Always use SpikingMLP, the flag 'use_spiking_mlp' is removed.
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)

        self.post_mlp_residual_gate_add = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.post_mlp_residual_gate_mult = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict]]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []

        x_res1 = x
        x_norm1 = self.norm1(x) if self.prenorm else x 

        # NEW: Spiking Feature Gating before Attention
        # Apply separate short convolutions for the 'gate' and 'modulator' inputs
        gate_input_g_attn_conv, _ = self.g_attn_input_conv(x_norm1)
        gate_input_m_attn_conv, _ = self.m_attn_input_conv(x_norm1)

        # Apply learnable spiking gates
        g_attn_input_cont, s_g_attn_input = self.g_attn_input_proj(gate_input_g_attn_conv)
        m_attn_input_cont, s_m_attn_input = self.m_attn_input_proj(gate_input_m_attn_conv)
        all_spikes_list.extend([s_g_attn_input, s_m_attn_input])

        gating_factor_attn_input = torch.sigmoid(g_attn_input_cont)

        # Modulate the input to attention: x_norm1 + modulated_info * gating_factor
        x_modulated_for_attn = x_norm1 + m_attn_input_cont * gating_factor_attn_input
        # END NEW

        x_attn_output_continuous, _, attn_spikes_dict = self.attn(x_modulated_for_attn, past_key_values=past_key_values, use_cache=use_cache, layer_idx=layer_idx, **kwargs)
        if attn_spikes_dict: 
            all_spikes_list.extend(attn_spikes_dict.values())

        # Residual connection after attention block with dual gating
        branch_output_attn = self.dropout1(x_attn_output_continuous)

        # Apply separate short convolutions for the 'add' and 'mult' gate inputs
        # The 'mult' gate typically acts on the residual itself (x_res1)
        gate_input_attn_conv_add, _ = self.post_attn_residual_gate_conv_add(branch_output_attn)
        gate_input_attn_conv_mult, _ = self.post_attn_residual_gate_conv_mult(x_res1) 

        # Apply learnable spiking dual gates to the post-attention residual path
        gate_add_cont_attn, s_gate_add_attn = self.post_attn_residual_gate_add(gate_input_attn_conv_add)
        gate_mult_cont_attn, s_gate_mult_attn = self.post_attn_residual_gate_mult(gate_input_attn_conv_mult)
        all_spikes_list.extend([s_gate_add_attn, s_gate_mult_attn])

        gating_factor_add_attn = torch.sigmoid(gate_add_cont_attn)
        gating_factor_mult_attn = torch.sigmoid(gate_mult_cont_attn)

        # Dual gate application: x_res * (1 - forget_factor) + new_info * input_factor
        x_after_attn_spike_gated = x_res1 * (1.0 - gating_factor_mult_attn) + branch_output_attn * gating_factor_add_attn

        x_res2 = x_after_attn_spike_gated 
        x_norm2 = self.norm2(x_after_attn_spike_gated) if self.prenorm else x_after_attn_spike_gated 

        # MLP returns list of spike tensors
        x_mlp_output_continuous, mlp_all_spikes = self.mlp(x_norm2)
        all_spikes_list.extend(mlp_all_spikes)

        # Residual connection after MLP block with dual gating
        branch_output_mlp = self.dropout2(x_mlp_output_continuous)

        # Apply separate short convolutions for the 'add' and 'mult' gate inputs
        # The 'mult' gate typically acts on the residual itself (x_res2)
        gate_input_mlp_conv_add, _ = self.post_mlp_residual_gate_conv_add(branch_output_mlp)
        gate_input_mlp_conv_mult, _ = self.post_mlp_residual_gate_conv_mult(x_res2) 

        # Apply learnable spiking dual gates to the post-MLP residual path
        gate_add_cont_mlp, s_gate_add_mlp = self.post_mlp_residual_gate_add(gate_input_mlp_conv_add)
        gate_mult_cont_mlp, s_gate_mult_mlp = self.post_mlp_residual_gate_mult(gate_input_mlp_conv_mult)
        all_spikes_list.extend([s_gate_add_mlp, s_gate_mult_mlp])

        gating_factor_add_mlp = torch.sigmoid(gate_add_cont_mlp)
        gating_factor_mult_mlp = torch.sigmoid(gate_mult_cont_mlp)

        # Dual gate application: x_res * (1 - forget_factor) + new_info * input_factor
        x_final_output_gated = x_res2 * (1.0 - gating_factor_mult_mlp) + branch_output_mlp * gating_factor_add_mlp

        # Concatenate all collected spikes into a single flat tensor
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        return x_final_output_gated, None, final_spikes