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

class SpikingShortConvolution(nn.Module):
    """A sequence module containing LIF neurons and a ShortConvolution layer, where the convolution operates on spikes."""

    def __init__(self, short_conv_layer: ShortConvolution, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.conv = short_conv_layer

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor]=None, output_final_state: Optional[bool]=False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_transposed_for_neuron = x.transpose(-1, -2) # (B, L, D) -> (B, D, L)
        spikes = self.neuron(x_transposed_for_neuron).transpose(-1, -2) # (B, D, L) -> (B, L, D) spikes

        # self.conv (ShortConvolution) takes spikes (binary) as input.
        # Its output `output_conv` will be continuous.
        conv_output = self.conv(spikes, cache=cache, output_final_state=output_final_state)

        if output_final_state:
            output_conv, new_conv_state = conv_output
            return output_conv, spikes, new_conv_state
        else:
            output_conv = conv_output
            return output_conv, spikes, None # Consistent 3-tuple output

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
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.f_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate) 
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate) 

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.value_dim // num_heads)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            # Replaced ShortConvolution with SpikingShortConvolution
            self.q_conv1d = SpikingShortConvolution(ShortConvolution(num_heads * num_slots, conv_size, activation='silu'), neuron_type, surrogate)
            self.k_conv1d = SpikingShortConvolution(ShortConvolution(num_heads * num_slots, conv_size, activation='silu'), neuron_type, surrogate)
            self.v_conv1d = SpikingShortConvolution(ShortConvolution(self.value_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.f_conv1d = SpikingShortConvolution(ShortConvolution(num_heads * num_slots, conv_size, activation='silu'), neuron_type, surrogate)
        else: # Ensure identity layers are present when short conv is off for consistent forward calls
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()
            self.f_conv1d = nn.Identity()

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}

        q_cont, s_q = self.q_proj(x)
        k_cont, s_k = self.k_proj(x)
        v_cont, s_v = self.v_proj(x)
        f_cont, s_f = self.f_proj(x)
        g_cont, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'f': s_f, 'g': s_g})

        # Capture spikes from SpikingShortConvolution layers
        if isinstance(self.q_conv1d, SpikingShortConvolution):
            q_conv_out, s_q_conv, q_conv_state = self.q_conv1d(q_cont, output_final_state=True)
            all_spikes['q_conv'] = s_q_conv
        else:
            q_conv_out = self.q_conv1d(q_cont)
            s_q_conv, q_conv_state = None, None # For consistency

        if isinstance(self.k_conv1d, SpikingShortConvolution):
            k_conv_out, s_k_conv, k_conv_state = self.k_conv1d(k_cont, output_final_state=True)
            all_spikes['k_conv'] = s_k_conv
        else:
            k_conv_out = self.k_conv1d(k_cont)
            s_k_conv, k_conv_state = None, None

        if isinstance(self.v_conv1d, SpikingShortConvolution):
            v_conv_out, s_v_conv, v_conv_state = self.v_conv1d(v_cont, output_final_state=True)
            all_spikes['v_conv'] = s_v_conv
        else:
            v_conv_out = self.v_conv1d(v_cont)
            s_v_conv, v_conv_state = None, None

        if isinstance(self.f_conv1d, SpikingShortConvolution):
            f_conv_out, s_f_conv, f_conv_state = self.f_conv1d(f_cont, output_final_state=True)
            all_spikes['f_conv'] = s_f_conv
        else:
            f_conv_out = self.f_conv1d(f_cont)
            s_f_conv, f_conv_state = None, None

        q_re = rearrange(q_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        k_re = rearrange(k_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        v_re = rearrange(v_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        f_re = rearrange(f_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)

        last_state = None
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

            # Also store conv states, if they are active
            if isinstance(self.q_conv1d, SpikingShortConvolution):
                past_key_values[layer_idx]['q_conv_state'] = q_conv_state
            if isinstance(self.k_conv1d, SpikingShortConvolution):
                past_key_values[layer_idx]['k_conv_state'] = k_conv_state
            if isinstance(self.v_conv1d, SpikingShortConvolution):
                past_key_values[layer_idx]['v_conv_state'] = v_conv_state
            if isinstance(self.f_conv1d, SpikingShortConvolution):
                past_key_values[layer_idx]['f_conv_state'] = f_conv_state

        o = rearrange(o_raw_attn_output, 'b h l d -> b l h d')
        o = self.g_norm(o)

        g_re = rearrange(g_cont, 'b l (h d) -> b l h d', h=self.num_heads) 
        o = o * F.silu(g_re) 

        o = rearrange(o, 'b l h d -> b l (h d)')
        o_cont, s_o = self.o_proj(o)
        all_spikes['o'] = s_o
        return o_cont, None, all_spikes

# [SpikingMLP Wrapper]

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

# New SpikingResidualModulator

class SpikingResidualModulator(nn.Module):
    def __init__(self, d_model: int, neuron_type: str, surrogate_function):
        super().__init__()
        # This module will receive mean spike activity per token (B, L, 1)
        # It needs to project this into a learnable modulation factor.
        self.proj_in = SpikingLinear(nn.Linear(1, d_model // 2, bias=False), neuron_type, surrogate_function)
        self.proj_out = SpikingLinear(nn.Linear(d_model // 2, 1, bias=False), neuron_type, surrogate_function)

        # Learnable parameters to control the range and bias of modulation
        # This will allow the modulation factor to deviate from 1.0 based on spike activity.
        self.bias = nn.Parameter(torch.zeros(1)) # Initial bias for tanh input
        self.scale = nn.Parameter(torch.ones(1) * 2.0) # Scale the tanh output, allowing modulation range

    def forward(self, avg_spike_activity: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_list = []

        # Project the mean spike activity through spiking layers
        x_cont_proj, spikes1 = self.proj_in(avg_spike_activity)
        spikes_list.append(spikes1)

        # Apply non-linear activation (continuous)
        x_act = F.gelu(x_cont_proj)

        # Project to a single raw modulation value
        mod_raw_cont, spikes2 = self.proj_out(x_act)
        spikes_list.append(spikes2)

        # Apply tanh to ensure the modulation factor is centered around 1.0, with a learnable scale
        # A factor of 1.0 means no change; tanh allows both positive and negative deviation
        # `1.0 + self.scale * torch.tanh(mod_raw_cont + self.bias)` ensures a base of 1.0
        modulation_factor = 1.0 + self.scale * torch.tanh(mod_raw_cont + self.bias)

        return modulation_factor, spikes_list

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', use_gate_conv: bool=True, gate_conv_size: int=3, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        surrogate_function = piecewise_quadratic_surrogate() 

        # Removed previous input gating to simplify the architecture and improve coherence.
        # This reduces redundant gating mechanisms and focuses on spike-activity-driven residual modulation.

        self.attn = GSA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # New: SpikingResidualModulator for the attention block's residual path.
        # This modulator dynamically scales the residual connection based on the actual spike activity
        # generated by the attention block itself, making the residual more 'spike-aware'.
        self.attn_residual_modulator = SpikingResidualModulator(d_model, neuron_type, surrogate_function)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)

        # New: SpikingResidualModulator for the MLP block's residual path.
        # Similar to the attention modulator, this adjusts the MLP residual based on its own spike activity.
        self.mlp_residual_modulator = SpikingResidualModulator(d_model, neuron_type, surrogate_function)

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict]]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []

        x_res1 = x
        x_norm1 = self.norm1(x) if self.prenorm else x 

        # --- Attention Block (GSA) ---
        # Direct input to attention, without complex pre-attention gating.
        x_attn_output_continuous, _, attn_spikes_dict = self.attn(x_norm1, past_key_values=past_key_values, use_cache=use_cache, layer_idx=layer_idx, **kwargs)

        attn_block_spikes_for_modulator = []
        if attn_spikes_dict:
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)
                    # Calculate mean spike activity per token across feature dimension for modulator input
                    attn_block_spikes_for_modulator.append(rearrange(spike_tensor.float().mean(dim=-1), 'b l -> b l 1')) # (B, L, 1)

        branch_output_attn = self.dropout1(x_attn_output_continuous)

        # Apply Spike-Activity Modulated Residual for Attention
        if attn_block_spikes_for_modulator:
            # Average spike activity across all spike tensors within the attention block to drive the modulator
            avg_attn_mean_activity = torch.mean(torch.stack(attn_block_spikes_for_modulator), dim=0)
            attn_modulation_factor, attn_modulator_spikes = self.attn_residual_modulator(avg_attn_mean_activity)
            all_spikes_list.extend(attn_modulator_spikes)
            # Modulate the residual branch: x_res + modulated_branch
            x_after_attn_spike_gated = x_res1 + branch_output_attn * attn_modulation_factor
        else:
            # Fallback to standard residual connection if no spikes were collected (e.g., during initialization/edge cases)
            x_after_attn_spike_gated = x_res1 + branch_output_attn 

        # --- MLP Block ---
        x_res2 = x_after_attn_spike_gated 
        x_norm2 = self.norm2(x_after_attn_spike_gated) if self.prenorm else x_after_attn_spike_gated 

        x_mlp_output_continuous, mlp_all_spikes = self.mlp(x_norm2)

        mlp_block_spikes_for_modulator = []
        if mlp_all_spikes:
            for spike_tensor in mlp_all_spikes:
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)
                    mlp_block_spikes_for_modulator.append(rearrange(spike_tensor.float().mean(dim=-1), 'b l -> b l 1')) # (B, L, 1)

        branch_output_mlp = self.dropout2(x_mlp_output_continuous)

        # Apply Spike-Activity Modulated Residual for MLP
        if mlp_block_spikes_for_modulator:
            avg_mlp_mean_activity = torch.mean(torch.stack(mlp_block_spikes_for_modulator), dim=0)
            mlp_modulation_factor, mlp_modulator_spikes = self.mlp_residual_modulator(avg_mlp_mean_activity)
            all_spikes_list.extend(mlp_modulator_spikes)
            x_final_output_gated = x_res2 + branch_output_mlp * mlp_modulation_factor
        else:
            x_final_output_gated = x_res2 + branch_output_mlp # Fallback to standard residual

        # Concatenate all collected binary spikes into a single flat tensor as required.
        final_spikes_filtered = [s.flatten() for s in all_spikes_list if s is not None]
        final_spikes = torch.cat(final_spikes_filtered) if final_spikes_filtered else torch.empty(0, device=x.device, dtype=torch.bool)

        return x_final_output_gated, None, final_spikes