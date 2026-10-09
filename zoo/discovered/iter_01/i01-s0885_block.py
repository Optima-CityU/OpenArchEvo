import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, List

from fla.modules import RMSNorm, ShortConvolution
from fla.models.utils import Cache

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
        x_transposed = x.transpose(-1, -2)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)
        output = self.linear(spikes)
        return output, spikes

# ==================== White-box Operator (HGRN2 - unchanged from Version 2) ====================

def hgrn2_chunkwise(g, x, chunk_size: int=64, initial_state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
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
    block_size = 32  # Processing the d_model dimension in blocks. Keep for consistency with original.

    mask_template = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device), diagonal=1)

    for i in range(g.shape[2]):  # Chunk Loop
        g_chunk = g[:, :, i]
        x_chunk = x[:, :, i]

        o_chunk_slices = []
        S_new_slices = []

        for start_idx in range(0, d, block_size):
            end_idx = min(start_idx + block_size, d)

            g_i = g_chunk[..., start_idx:end_idx]
            x_i = x_chunk[..., start_idx:end_idx]
            S_i = S[..., start_idx:end_idx]

            # 1. Intra-Chunk
            g_cumsum = torch.cumsum(g_i, dim=-2)
            decay_matrix = g_cumsum.unsqueeze(-2) - g_cumsum.unsqueeze(-3)

            decay_matrix = decay_matrix.masked_fill(mask_template.unsqueeze(-1), -float('inf'))
            decay_mask = torch.exp(decay_matrix)

            o_intra = torch.einsum('b h i j d, b h j d -> b h i d', decay_mask, x_i)

            # 2. Inter-Chunk
            decay_from_start = torch.exp(g_cumsum)
            o_inter = S_i.unsqueeze(2) * decay_from_start

            o_chunk_slices.append(o_intra + o_inter)

            # 3. Compute New State
            total_decay = torch.exp(g_i.sum(dim=-2))
            decay_to_end = torch.exp(g_cumsum[:, :, -1].unsqueeze(-2) - g_cumsum)
            current_chunk_contrib = (x_i * decay_to_end).sum(dim=-2)
            S_new_slices.append(S_i * total_decay + current_chunk_contrib)

        S = torch.cat(S_new_slices, dim=-1)
        o_chunk = torch.cat(o_chunk_slices, dim=-1)
        o_list.append(o_chunk)

    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')

    if pad_len > 0: o = o[..., :l, :]

    return o, S

# ==================== Main Layer (SpikingHGRN_Adaptive - MODIFIED) ====================

class SpikingHGRN_Adaptive(nn.Module):
    def __init__(self, d_model: int, num_heads: Optional[int]=None, expand_ratio: float=1.0, use_short_conv: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        if num_heads is None: num_heads = d_model // 64
        self.num_heads = num_heads
        self.hidden_dim = int(d_model * expand_ratio)
        self.use_short_conv = use_short_conv

        surrogate = piecewise_quadratic_surrogate()

        # Projections for main recurrence
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)
        self.i_proj = SpikingLinear(nn.Linear(d_model, self.hidden_dim, bias=False), neuron_type, surrogate)

        if use_short_conv:
            self.conv_g = ShortConvolution(self.hidden_dim, kernel_size=4, activation=None)
            self.conv_i = ShortConvolution(self.hidden_dim, kernel_size=4, activation=None)

        self.norm_g = RMSNorm(self.hidden_dim)
        self.norm_i = RMSNorm(self.hidden_dim)

        self.o_proj = SpikingLinear(nn.Linear(self.hidden_dim, d_model, bias=False), neuron_type, surrogate)
        self.norm = RMSNorm(self.hidden_dim)

        # Learnable initial decay gain
        self.head_decay_gain = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

        # Original Adaptive Factor (decay modulator)
        self.decay_modulator_proj = SpikingLinear(nn.Linear(d_model, self.num_heads), neuron_type, surrogate)
        if use_short_conv:
            # NEW: Short convolution for decay modulator
            self.conv_decay_modulator = ShortConvolution(self.num_heads, kernel_size=4, activation=None)
        self.decay_modulator_norm = RMSNorm(self.num_heads)
        self.decay_modulator_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

        # NEW: Contextual Decay Modulator (for multi-scale decay control)
        self.context_decay_proj = SpikingLinear(nn.Linear(d_model, self.num_heads), neuron_type, surrogate)
        if use_short_conv:
            self.conv_context_decay = ShortConvolution(self.num_heads, kernel_size=4, activation=None)
        self.context_decay_norm = RMSNorm(self.num_heads)
        self.context_decay_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

        # NEW: Adaptive Input Gate (input modulator)
        self.input_modulator_proj = SpikingLinear(nn.Linear(d_model, self.num_heads), neuron_type, surrogate)
        if use_short_conv:
            # NEW: Short convolution for input modulator
            self.conv_input_modulator = ShortConvolution(self.num_heads, kernel_size=4, activation=None)
        self.input_modulator_norm = RMSNorm(self.num_heads)
        self.input_modulator_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

        # NEW: Adaptive Output Gate (output modulator)
        # Project to hidden_dim for per-head, per-dimension modulation
        self.output_modulator_proj = SpikingLinear(nn.Linear(d_model, self.num_heads * (self.hidden_dim // self.num_heads)), neuron_type, surrogate)
        self.output_modulator_norm = RMSNorm(self.hidden_dim // self.num_heads)
        self.output_modulator_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, layer_idx: Optional[int]=None, use_cache: Optional[bool]=False, **kwargs) -> Tuple[torch.Tensor, None, Dict[str, torch.Tensor]]:
        all_spikes: Dict[str, torch.Tensor] = {}

        # Cache states for convolutions
        conv_state_g, conv_state_i = None, None
        conv_state_decay_modulator, conv_state_input_modulator, conv_state_context_decay = None, None, None

        if use_cache and past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                states = past_key_values[layer_idx]
                conv_state_g = states.get('conv_state_g')
                conv_state_i = states.get('conv_state_i')
                conv_state_decay_modulator = states.get('conv_state_decay_modulator')
                conv_state_input_modulator = states.get('conv_state_input_modulator')
                conv_state_context_decay = states.get('conv_state_context_decay')

        # Decay Modulator (original adaptive factor)
        decay_modulator_raw, s_decay_modulator = self.decay_modulator_proj(x)
        all_spikes['decay_modulator'] = s_decay_modulator
        if self.use_short_conv:
            decay_modulator_raw, conv_state_decay_modulator = self.conv_decay_modulator(decay_modulator_raw, cache=conv_state_decay_modulator, output_final_state=True)

        decay_modulator_normed = self.decay_modulator_norm(decay_modulator_raw)
        decay_modulator = rearrange(decay_modulator_normed, 'b l h -> b h l 1')
        decay_modulator = F.sigmoid(decay_modulator) * self.decay_modulator_scale

        # NEW: Contextual Decay Modulator
        context_decay_modulator_raw, s_context_decay = self.context_decay_proj(x)
        all_spikes['context_decay_modulator'] = s_context_decay
        if self.use_short_conv:
            context_decay_modulator_raw, conv_state_context_decay = self.conv_context_decay(context_decay_modulator_raw, cache=conv_state_context_decay, output_final_state=True)
        context_decay_modulator_normed = self.context_decay_norm(context_decay_modulator_raw)
        context_decay_modulator = rearrange(context_decay_modulator_normed, 'b l h -> b h l 1')
        context_decay_modulator = F.sigmoid(context_decay_modulator) * self.context_decay_scale

        # NEW: Input Modulator (adaptive gate for input `i`)
        input_modulator_raw, s_input_modulator = self.input_modulator_proj(x)
        all_spikes['input_modulator'] = s_input_modulator
        if self.use_short_conv:
            input_modulator_raw, conv_state_input_modulator = self.conv_input_modulator(input_modulator_raw, cache=conv_state_input_modulator, output_final_state=True)

        input_modulator_normed = self.input_modulator_norm(input_modulator_raw)
        input_modulator = rearrange(input_modulator_normed, 'b l h -> b h l 1')
        input_modulator = F.sigmoid(input_modulator) * self.input_modulator_scale

        # NEW: Output Modulator
        output_modulator_raw, s_output_modulator = self.output_modulator_proj(x)
        all_spikes['output_modulator'] = s_output_modulator
        # No short conv for output_modulator for now, keeping it simpler.

        output_modulator_normed = self.output_modulator_norm(rearrange(output_modulator_raw, 'b l (h d) -> b l h d', h=self.num_heads))
        output_modulator = rearrange(output_modulator_normed, 'b l h d -> b h l d')
        output_modulator = F.sigmoid(output_modulator) * self.output_modulator_scale  # Scale applied after sigmoid

        # Spiking linear projections for g and i
        g_proj_out, s_g = self.g_proj(x)
        i_proj_out, s_i = self.i_proj(x)
        all_spikes.update({'g': s_g, 'i': s_i})

        if self.use_short_conv:
            g_proj_out, conv_state_g = self.conv_g(g_proj_out, cache=conv_state_g, output_final_state=True)
            i_proj_out, conv_state_i = self.conv_i(i_proj_out, cache=conv_state_i, output_final_state=True)

        g_normed = self.norm_g(g_proj_out)
        i_normed = self.norm_i(i_proj_out)

        g = rearrange(g_normed, 'b l (h d) -> b h l d', h=self.num_heads)
        i = rearrange(i_normed, 'b l (h d) -> b h l d', h=self.num_heads)

        # Apply non-linearities and modulate with adaptive factors
        # Apply the new contextual_decay_modulator to g
        g = F.logsigmoid(g) * self.head_decay_gain * (1. + decay_modulator) * (1. + context_decay_modulator)
        # Apply the new input_modulator to the input 'i'
        i = F.silu(i) * input_modulator  # Multiplicative gating of input

        last_state = None
        if use_cache and past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        o, final_state = hgrn2_chunkwise(g, i, initial_state=last_state)

        if use_cache and past_key_values is not None and layer_idx is not None:
            new_cache = {
                'recurrent_state': final_state,
                'conv_state_g': conv_state_g,
                'conv_state_i': conv_state_i,
                'conv_state_decay_modulator': conv_state_decay_modulator,
                'conv_state_input_modulator': conv_state_input_modulator,
                'conv_state_context_decay': conv_state_context_decay
            }
            if len(past_key_values) <= layer_idx:
                past_key_values.append(new_cache)
            else:
                past_key_values[layer_idx] = new_cache

        # NEW: Apply output modulator
        o = o * output_modulator  # Apply modulator after recurrence, before final norm/proj

        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.norm(o)
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o
        return o, None, all_spikes

# ==================== GatedSpikingMLP (MODIFIED) ====================

class GatedSpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None, use_mlp_conv: bool=True, mlp_conv_size: int=4):
        super().__init__()
        # Ensure surrogate_function is always passed and correctly initialized
        if surrogate_function is None:
            surrogate_function = piecewise_quadratic_surrogate()

        self.d_model = d_model # Store d_model for gate projection
        self.d_ffn = d_ffn  # Store d_ffn to split the output later
        self.use_mlp_conv = use_mlp_conv

        # Unified Spiking Linear layer for both gate and up projections
        self.unified_gate_up_proj = SpikingLinear(nn.Linear(d_model, 2 * d_ffn), neuron_type, surrogate_function)

        self.dropout = nn.Dropout(dropout)

        # NEW: Internal spiking layer after GLU and dropout, before conv/down_proj
        self.mlp_internal_spike_proj = SpikingLinear(nn.Linear(d_ffn, d_ffn), neuron_type, surrogate_function)

        # Spiking Linear layer before MLP ShortConvolution
        if self.use_mlp_conv:
            self.mlp_pre_conv_proj = SpikingLinear(nn.Linear(d_ffn, d_ffn), neuron_type, surrogate_function)
            self.mlp_conv = ShortConvolution(d_ffn, mlp_conv_size, activation='silu')

            # NEW: Spiking Adaptive Conv Gate
            self.mlp_conv_gate_proj = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function) # Project to d_ffn for per-dimension gate
            self.mlp_conv_gate_norm = RMSNorm(d_ffn)
            self.mlp_conv_gate_scale = nn.Parameter(torch.ones(1, 1, d_ffn)) # Learnable scale for the gate, broadcastable

        self.down_proj = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_mlp_spikes = []

        # Unified projection and split
        unified_output, s_unified = self.unified_gate_up_proj(x)
        gate_out, up_out = torch.split(unified_output, [self.d_ffn, self.d_ffn], dim=-1)
        all_mlp_spikes.append(s_unified)

        # GLU-style gating with SiLU activation for the up_out path
        gated_hidden_continuous = gate_out * F.silu(up_out)
        gated_hidden_continuous = self.dropout(gated_hidden_continuous)

        # NEW: Apply internal spiking layer
        gated_hidden_spiking_input, s_internal_spike = self.mlp_internal_spike_proj(gated_hidden_continuous)
        all_mlp_spikes.append(s_internal_spike)

        # Apply spiking pre-convolution layer and then short convolution if enabled
        if self.use_mlp_conv:
            spiking_conv_input_raw, s_gated_hidden_pre_conv = self.mlp_pre_conv_proj(gated_hidden_spiking_input)
            all_mlp_spikes.append(s_gated_hidden_pre_conv)

            # Compute Spiking Adaptive Conv Gate
            # Use the original input `x` to the MLP block for contextual gating
            gate_signal_raw, s_mlp_conv_gate = self.mlp_conv_gate_proj(x)
            all_mlp_spikes.append(s_mlp_conv_gate)
            gate_signal_normed = self.mlp_conv_gate_norm(gate_signal_raw)
            gate_signal = F.sigmoid(gate_signal_normed) * self.mlp_conv_gate_scale # Shape (B, L, d_ffn)

            # Apply gate signal to conv input
            spiking_conv_input = spiking_conv_input_raw * gate_signal # (B, L, d_ffn) * (B, L, d_ffn)

            # Conv state not explicitly tracked for MLP conv within block, assuming stateless for MLP.
            gated_hidden_convolved, _ = self.mlp_conv(spiking_conv_input)
            gated_hidden_final = gated_hidden_convolved
        else:
            gated_hidden_final = gated_hidden_spiking_input

        # Final projection layer
        output, s_down = self.down_proj(gated_hidden_final)
        all_mlp_spikes.append(s_down)
        return output, all_mlp_spikes

# ==================== SpikingFLABlock (Overall Block with Adaptive HGRN and GatedMLP) ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        # Use the adaptive HGRN module
        self.attn = SpikingHGRN_Adaptive(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)

        d_ffn = d_model * d_ffn_mult
        # Use GatedSpikingMLP instead of SpikingMLP
        self.mlp = GatedSpikingMLP(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            neuron_type=neuron_type,
            surrogate_function=piecewise_quadratic_surrogate(),
            use_mlp_conv=True  # Default to true for enhanced local processing
        )
        self.dropout2 = nn.Dropout(dropout)

        # Final norm and spiking projection for architectural completeness and output stabilization.
        self.final_norm = RMSNorm(d_model)
        self.final_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, piecewise_quadratic_surrogate())

    def forward(self, x: torch.Tensor, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, None, torch.Tensor]:
        all_spikes_list: List[torch.Tensor] = []

        # ATTENTION/RECURRENCE BLOCK
        x_res_attn = x
        if self.prenorm: x_attn_input = self.norm1(x)
        else: x_attn_input = x

        # Pass kwargs to attn, which includes past_key_values, use_cache, layer_idx
        x_attn_output, _, attn_spikes_dict = self.attn(x_attn_input, past_key_values=past_key_values, use_cache=use_cache, layer_idx=layer_idx, **kwargs)
        if attn_spikes_dict:
            all_spikes_list.extend([s for s in attn_spikes_dict.values() if s is not None])

        x = x_res_attn + self.dropout1(x_attn_output)

        # MLP BLOCK (using GatedSpikingMLP)
        x_res_mlp = x
        if self.prenorm: x_mlp_input = self.norm2(x)
        else: x_mlp_input = x

        x_mlp_output, mlp_spikes_list = self.mlp(x_mlp_input)
        all_spikes_list.extend(mlp_spikes_list)

        x = x_res_mlp + self.dropout2(x_mlp_output)

        # Final normalization and spiking projection for the whole block output.
        x_final_norm = self.final_norm(x)
        x_final_proj_output, s_final_proj = self.final_proj(x_final_norm)
        all_spikes_list.append(s_final_proj)
        x = x_final_proj_output  # The final output of the block is this analog signal.

        # Ensure final_spikes is a flat, concatenated tensor
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None]) if all_spikes_list else torch.empty(0, device=x.device)

        return x, None, final_spikes