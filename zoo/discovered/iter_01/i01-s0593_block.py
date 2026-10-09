import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, List

from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from fla.models.utils import Cache

# ==================== SNN Base Components (Consistent with DeltaNet) ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder SNN modules.")
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return input # Placeholder for continuous passthrough
        @staticmethod
        def backward(ctx, grad_output): return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_for_neuron = x.transpose(-1, -2) # (..., D_features, L_sequence)
        spikes = self.neuron(x_for_neuron).transpose(-1, -2) # Apply neuron along L, then transpose back

        # The linear layer then operates on these binary spikes
        output = self.linear(spikes)
        return output, spikes

# ==================== White-box Operator (HGRN2) ====================

def hgrn2_chunkwise(g: torch.Tensor, x: torch.Tensor, chunk_size: int=64, initial_state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """HGRN2 Parallel Logic. Identical to HGRN (Element-wise Recurrence)."""
    b, h, l, d = x.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        g, x = [F.pad(t, (0, 0, 0, pad_len)) for t in [g, x]]
        
    g = rearrange(g, 'b h (n c) d -> b h n c d', c=chunk_size)
    x = rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, d, device=x.device, dtype=x.dtype)
        
    o_list = []
    block_size = 32 
    
    mask_template = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device), diagonal=1)
    
    for i in range(g.shape[2]): # Iterate over chunks
        g_chunk = g[:, :, i] 
        x_chunk = x[:, :, i] 
        
        o_chunk_slices = []
        S_new_slices = []
        
        for start_idx in range(0, d, block_size):
            end_idx = min(start_idx + block_size, d)
            
            g_i = g_chunk[..., start_idx:end_idx] 
            x_i = x_chunk[..., start_idx:end_idx] 
            S_i = S[..., start_idx:end_idx]       
            
            g_cumsum = torch.cumsum(g_i, dim=-2) 
            
            decay_matrix = g_cumsum.unsqueeze(-2) - g_cumsum.unsqueeze(-3)
            
            decay_matrix = decay_matrix.masked_fill(mask_template.unsqueeze(-1), -float('inf'))
            decay_mask = torch.exp(decay_matrix) 
            
            o_intra = torch.einsum('b h i j d, b h j d -> b h i d', decay_mask, x_i)
            
            decay_from_start = torch.exp(g_cumsum) 
            o_inter = S_i.unsqueeze(2) * decay_from_start 
            
            o_chunk_slices.append(o_intra + o_inter)
            
            total_decay = torch.exp(g_i.sum(dim=-2)) 
            
            g_cumsum_end = g_cumsum[:, :, -1, :].unsqueeze(-2) 
            decay_to_end = torch.exp(g_cumsum_end - g_cumsum) 
            
            current_chunk_contrib = (x_i * decay_to_end).sum(dim=-2) 
            S_new_slices.append(S_i * total_decay + current_chunk_contrib)
            
        S = torch.cat(S_new_slices, dim=-1) 
        o_chunk = torch.cat(o_chunk_slices, dim=-1) 
        o_list.append(o_chunk)
        
    o = torch.stack(o_list, dim=2) 
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    return o, S

# ==================== Main Layer (SpikingHGRN_ContextualGate) ====================

class SpikingHGRN_ContextualGate(nn.Module):
    def __init__(self, d_model: int, num_heads: Optional[int]=None, expand_ratio: float=1.0, use_short_conv: bool=True, neuron_type: str='sdn', conv_size: int=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        if num_heads is None: num_heads = d_model // 64
        self.num_heads = num_heads
        self.hidden_dim = int(d_model * expand_ratio)
        self.use_short_conv = use_short_conv

        surrogate = piecewise_quadratic_surrogate()

        # All linear layers are now SpikingLinear
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.i_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)

        # New/Reintroduced: Contextual Modulation and Reset Gate Projections
        self.reset_gate_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.ctx_mod_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)

        if use_short_conv:
            self.conv_g = ShortConvolution(self.hidden_dim, kernel_size=conv_size, activation=None)
            self.conv_i = ShortConvolution(self.hidden_dim, kernel_size=conv_size, activation=None)
            self.conv_reset_gate = ShortConvolution(self.hidden_dim, kernel_size=conv_size, activation=None) # New
            self.conv_ctx_mod = ShortConvolution(self.hidden_dim, kernel_size=conv_size, activation=None)     # New

        # Pre-normalization layers from Version 2, applied after convolutions
        self.norm_g = RMSNorm(self.hidden_dim)
        self.norm_i = RMSNorm(self.hidden_dim)
        self.norm_reset_gate = RMSNorm(self.hidden_dim) # New
        self.norm_ctx_mod = RMSNorm(self.hidden_dim)     # New

        self.o_proj = SpikingLinear(nn.Linear(self.hidden_dim, d_model, bias=False), neuron_type, surrogate)
        self.norm = RMSNorm(self.hidden_dim)

        # Reintroduced from Version 1, now fully dynamic and spike-derived
        self.dynamic_decay_proj = SpikingLinear(nn.Linear(d_model, self.num_heads), neuron_type, surrogate)

        # Learnable parameters for contextual modulation of g (decay) and i (input)
        # Using 1 for 'l' dimension as these are applied across the sequence per head-dim
        self.ctx_mod_gain_g = nn.Parameter(torch.ones(1, self.num_heads, 1, self.hidden_dim // self.num_heads))
        self.ctx_mod_gain_i = nn.Parameter(torch.ones(1, self.num_heads, 1, self.hidden_dim // self.num_heads))

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, None, Dict[str, torch.Tensor]]:
        all_spikes: Dict[str, torch.Tensor] = {} 

        # Outputs of SpikingLinear are (continuous_output, spikes)
        g_proj_out_cont, s_g = self.g_proj(x)
        i_proj_out_cont, s_i = self.i_proj(x)
        reset_gate_proj_out_cont, s_reset_gate = self.reset_gate_proj(x)
        ctx_mod_proj_out_cont, s_ctx_mod = self.ctx_mod_proj(x)
        dynamic_decay_out_cont, s_dynamic_decay = self.dynamic_decay_proj(x)

        all_spikes.update({'g': s_g, 'i': s_i, 'reset_gate': s_reset_gate, 'ctx_mod': s_ctx_mod, 'dynamic_decay_gate': s_dynamic_decay})

        conv_state_g, conv_state_i, conv_state_reset_gate, conv_state_ctx_mod = None, None, None, None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                states = past_key_values[layer_idx]
                conv_state_g = states.get('conv_state_g')
                conv_state_i = states.get('conv_state_i')
                conv_state_reset_gate = states.get('conv_state_reset_gate')
                conv_state_ctx_mod = states.get('conv_state_ctx_mod')

        if self.use_short_conv:
            g_proj_out_cont, conv_state_g = self.conv_g(g_proj_out_cont, cache=conv_state_g, output_final_state=True)
            i_proj_out_cont, conv_state_i = self.conv_i(i_proj_out_cont, cache=conv_state_i, output_final_state=True)
            reset_gate_proj_out_cont, conv_state_reset_gate = self.conv_reset_gate(reset_gate_proj_out_cont, cache=conv_state_reset_gate, output_final_state=True)
            ctx_mod_proj_out_cont, conv_state_ctx_mod = self.conv_ctx_mod(ctx_mod_proj_out_cont, cache=conv_state_ctx_mod, output_final_state=True)

        # Apply pre-normalization after short convolutions
        g_normed = self.norm_g(g_proj_out_cont)
        i_normed = self.norm_i(i_proj_out_cont)
        reset_gate_normed = self.norm_reset_gate(reset_gate_proj_out_cont)
        ctx_mod_normed = self.norm_ctx_mod(ctx_mod_proj_out_cont)

        # Reshape to multi-head format
        g_reshaped = rearrange(g_normed, 'b l (h d) -> b h l d', h=self.num_heads)
        i_reshaped = rearrange(i_normed, 'b l (h d) -> b h l d', h=self.num_heads)
        reset_gate_signal_reshaped = rearrange(reset_gate_normed, 'b l (h d) -> b h l d', h=self.num_heads)
        ctx_mod_signal_reshaped = rearrange(ctx_mod_normed, 'b l (h d) -> b h l d', h=self.num_heads)

        # Reshape and activate dynamic decay gain
        dynamic_decay_gain = rearrange(dynamic_decay_out_cont, 'b l h -> b h l 1')
        dynamic_decay_gain = F.sigmoid(dynamic_decay_gain) # Bound dynamic gain between 0 and 1

        # Apply sigmoid to contextual modulation signal for bounded influence
        ctx_mod_activated = F.sigmoid(ctx_mod_signal_reshaped)

        # Apply contextual modulation to g and i before their final non-linearities
        g_pre_act = g_reshaped + (self.ctx_mod_gain_g * ctx_mod_activated)
        i_pre_act = i_reshaped + (self.ctx_mod_gain_i * ctx_mod_activated)

        # Apply reset gate to input 'i'
        reset_gate_activated = F.sigmoid(reset_gate_signal_reshaped)

        # Apply final non-linearities and dynamic/gating factors
        g = F.logsigmoid(g_pre_act) * dynamic_decay_gain 
        i = F.silu(i_pre_act) * reset_gate_activated

        last_state = None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = hgrn2_chunkwise(g, i, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            new_cache = {
                'recurrent_state': final_state,
                'conv_state_g': conv_state_g,
                'conv_state_i': conv_state_i,
                'conv_state_reset_gate': conv_state_reset_gate,
                'conv_state_ctx_mod': conv_state_ctx_mod,
            }
            if len(past_key_values) <= layer_idx:
                past_key_values.append(new_cache)
            else:
                past_key_values[layer_idx].update(new_cache) # Use update to preserve existing keys

        o = rearrange(o, 'b h l d -> b l (h d)')
        o_normed = self.norm(o)
        o_proj_out_cont, s_o = self.o_proj(o_normed) 
        all_spikes['o'] = s_o
        return o_proj_out_cont, None, all_spikes

# ==================== Spiking MLP with Gated Linear Unit (GLU) ====================
# Keeping SpikingMLP from Version 2 as it's a good pattern and correctly uses SpikingLinear.

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if surrogate_function is None: 
            surrogate_function = piecewise_quadratic_surrogate() 

        self.proj_in = SpikingLinear(nn.Linear(d_model, 2 * d_ffn), neuron_type, surrogate_function)
        self.gate_act = nn.SiLU()
        self.norm_ffn = RMSNorm(d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj_in_out_continuous, s_proj_in = self.proj_in(x) 

        gate_continuous, value_continuous = proj_in_out_continuous.chunk(2, dim=-1)

        gated_value_continuous = self.gate_act(gate_continuous) * value_continuous

        x_normed_continuous = self.norm_ffn(gated_value_continuous)
        x_dropped_continuous = self.dropout(x_normed_continuous)

        final_mlp_out_continuous, s_proj_out = self.proj_out(x_dropped_continuous)

        return final_mlp_out_continuous, s_proj_in, s_proj_out

# ==================== SpikingFLABlock (Overall Block) ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', conv_size: int=4, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        # Use the modified SpikingHGRN_ContextualGate
        self.attn = SpikingHGRN_ContextualGate(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, conv_size=conv_size, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        surrogate_function = piecewise_quadratic_surrogate() 

        # NEW: Spike-Activated Residual Gate for post-attention residual (reintroduced from V1)
        self.post_attn_residual_gate = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        # Post-attention convolution from Version 2
        self.post_attn_conv = ShortConvolution(d_model, kernel_size=conv_size, activation='silu') # Kept from V2

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        # Use the SpikingMLP from Version 2
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.dropout2 = nn.Dropout(dropout)

        # NEW: Spike-Activated Residual Gate for post-MLP residual (reintroduced from V1)
        self.post_mlp_residual_gate = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        x_res_attn = x # Input for the attention residual connection
        if self.prenorm: 
            x_attn_input = self.norm1(x)
        else:
            x_attn_input = x 

        # Process through SpikingHGRN_ContextualGate, collecting spikes
        x_attn_output_continuous, _, attn_spikes_dict = self.attn(x_attn_input, past_key_values=past_key_values, layer_idx=layer_idx, **kwargs)
        if attn_spikes_dict: 
            all_spikes_list.extend([s for s in attn_spikes_dict.values() if s is not None])

        # Apply post-attention ShortConvolution with state handling
        conv_state_post_attn = None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                conv_state_post_attn = past_key_values[layer_idx].get('conv_state_post_attn')

        x_post_attn_conv_out_continuous, new_conv_state_post_attn = self.post_attn_conv(x_attn_output_continuous, cache=conv_state_post_attn, output_final_state=True)

        # Update cache for post_attn_conv (ensure past_key_values[layer_idx] exists first)
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx: # HGRN2 should have created it, but just in case
                past_key_values.append({})
            past_key_values[layer_idx]['conv_state_post_attn'] = new_conv_state_post_attn

        # Residual connection after attention block, gated by spike-activated gate
        residual_attn_candidate = self.dropout1(x_post_attn_conv_out_continuous)

        gate_attn_cont, s_post_attn_res_gate = self.post_attn_residual_gate(residual_attn_candidate)
        all_spikes_list.append(s_post_attn_res_gate)
        gating_factor_attn = torch.sigmoid(gate_attn_cont)
        x_after_attn_spike_gated = x_res_attn + (residual_attn_candidate * gating_factor_attn)

        x_res_mlp = x_after_attn_spike_gated 
        if self.prenorm: 
            x_mlp_input = self.norm2(x_after_attn_spike_gated)
        else:
            x_mlp_input = x_after_attn_spike_gated 

        # Process through SpikingMLP, collecting spikes
        x_mlp_output_continuous, s_proj_in, s_proj_out = self.mlp(x_mlp_input)
        all_spikes_list.extend([s_proj_in, s_proj_out])

        # Residual connection after MLP block, gated by spike-activated gate
        residual_mlp_candidate = self.dropout2(x_mlp_output_continuous)

        gate_mlp_cont, s_post_mlp_res_gate = self.post_mlp_residual_gate(residual_mlp_candidate)
        all_spikes_list.append(s_post_mlp_res_gate)
        gating_factor_mlp = torch.sigmoid(gate_mlp_cont)
        x_final_output_gated = x_res_mlp + (residual_mlp_candidate * gating_factor_mlp)

        # Concatenate all collected spikes into a single flat tensor. Handle empty list gracefully.
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None]) if all_spikes_list else torch.empty(0, device=x.device)

        return x_final_output_gated, None, final_spikes