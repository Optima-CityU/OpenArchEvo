import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

# ==================== SNN Related Imports and Helper Classes ====================
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
# --- End of Placeholders ---

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed) # Apply neuron to get spikes
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        output = self.linear(spikes) # Linear layer processes spikes
        return output, spikes

# ==================== White-box Operator (GLA) ====================

def gla_chunkwise(q, k, v, g, chunk_size=64, initial_state=None):
    """Gated Linear Attention Chunkwise Parallel Logic.
    
    Formula:
        S_t = S_{t-1} * exp(-g_t) + K_t^T V_t
        O_t = Q_t S_t
    
    Parallelization:
        We treat this as a "Weighted Linear Attention".
        Since 'g' is scalar per head (B, H, L, 1), the decay mask is (C, C).
        This allows full parallelization via matrix multiplication without OOM.

    Inputs q, k, v are expected to be binary spikes, g is a continuous decay factor.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # 1. Padding
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v, g = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v, g]]
        
    # 2. Reshape to Chunks
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    g = rearrange(g, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    # Initialize Matrix State S: [B, H, Dk, Dv]
    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = torch.zeros(b, h, d_k, d_v, device=q.device, dtype=q.dtype) # Ensure dtype consistency
        
    o = torch.zeros_like(v)
    
    # Pre-calculate cumulative decay for efficient mask construction
    # g is log-decay.
    # cumsum_g[t] = sum(g_0...g_t)
    
    for i in range(q.shape[2]): # Chunk Loop
        q_i = q[:, :, i] # [B, H, C, Dk] - Spikes
        k_i = k[:, :, i] # [B, H, C, Dk] - Spikes
        v_i = v[:, :, i] # [B, H, C, Dv] - Spikes
        g_i = g[:, :, i] # [B, H, C, 1] - Continuous
        
        # --- 1. Intra-Chunk (Parallel Attention) ---
        # Construct Decay Mask D_{ij} = exp(Sum_{k=j+1}^i g_k) = exp(Cumsum_i - Cumsum_j)
        g_cumsum = torch.cumsum(g_i, dim=-2) # [B, H, C, 1]
        
        # Mask shape: [B, H, C, C] (Broadcasting over last dim 1 is automatic)
        decay_matrix = g_cumsum - rearrange(g_cumsum, 'b h c 1 -> b h 1 c')
        
        # Causal Mask
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        # Apply mask to decay_matrix. For upper triangle (future), set to -inf before exp.
        decay_matrix = decay_matrix.masked_fill(mask.unsqueeze(0).unsqueeze(0), -float('inf'))
        decay_mask = torch.exp(decay_matrix)
        
        # Attention Score: A = (Q @ K^T) * Mask
        # [B, H, C, Dk] @ [B, H, Dk, C] -> [B, H, C, C]
        attn = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        
        # Output Intra: O = A @ V
        o_intra = attn @ v_i
        
        # --- 2. Inter-Chunk (Recurrence) ---
        decay_from_start = torch.exp(g_cumsum) # [B, H, C, 1]
        
        # Broadcast multiply: Q scaled by decay
        # [B, H, C, Dk] * [B, H, C, 1] -> [B, H, C, Dk]
        q_decayed = q_i * decay_from_start
        
        # [B, H, C, Dk] @ [B, H, Dk, Dv] -> [B, H, C, Dv]
        o_inter = q_decayed @ S
        
        o[:, :, i] = o_intra + o_inter
        
        # --- 3. Update State S ---
        total_decay = torch.exp(g_i.sum(dim=-2, keepdim=True)) # [B, H, 1, 1]
        
        # Decay from t to End: exp(Cumsum_End - Cumsum_t)
        decay_to_end = torch.exp(rearrange(g_cumsum[:, :, -1, :], 'b h 1 -> b h 1 1') - g_cumsum) # [B, H, C, 1]
        
        # Weighted KV: (K * decay)
        k_decayed = k_i * decay_to_end
        
        # Update: S = S * total_decay + K_decayed^T @ V
        # [B, H, Dk, C] @ [B, H, C, Dv] -> [B, H, Dk, Dv]
        current_chunk_contrib = k_decayed.transpose(-1, -2) @ v_i
        
        S = S * total_decay + current_chunk_contrib
        
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    return o, S

class SpikingGLA(nn.Module):
    def __init__(self, d_model, num_heads, expand_k=0.5, expand_v=1.0, gate_low_rank_dim=16, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads
        self.head_k_dim = self.key_dim // num_heads 

        surrogate = piecewise_quadratic_surrogate()

        # All initial projections are SpikingLinear, ensuring their internal nn.Linear layers process spikes.
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)

        self.use_short_conv = True 
        if self.use_short_conv:
             # ShortConvolution operates on the real-valued outputs of the SpikingLinear layers
             self.conv1d_group = ShortConvolution(self.key_dim * 2 + self.value_dim + num_heads, kernel_size=4, activation='silu')

        # NEW: Core spiking neurons for Q, K, V streams *after* initial projections/convolutions
        self.q_core_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.k_core_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.v_core_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)

        # Output projection is also SpikingLinear.
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.head_v_dim)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, layer_idx: Optional[int]=None, use_cache: Optional[bool]=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        all_spikes = {}

        # Apply SpikingLinear projections. `q_out, k_out, v_out, g_out` are real-valued outputs
        # from the internal nn.Linear, while `s_q, s_k, s_v, s_g` are the spikes fed to those nn.Linear layers.
        q_out, s_q = self.q_proj(x)
        k_out, s_k = self.k_proj(x)
        v_out, s_v = self.v_proj(x)
        g_out, s_g = self.g_proj(x)
        all_spikes.update({'q_proj_spikes': s_q, 'k_proj_spikes': s_k, 'v_proj_spikes': s_v, 'g_proj_spikes': s_g})

        # Short Convolution operates on the real-valued outputs of the initial linear projections.
        # It's not an nn.Linear itself.
        if self.use_short_conv:
            combined = torch.cat([q_out, k_out, v_out, g_out], dim=-1)
            conv_state = None
            if past_key_values is not None and layer_idx is not None:
                 if len(past_key_values) > layer_idx:
                    conv_state = past_key_values[layer_idx].get('conv_state')

            combined, new_conv_state = self.conv1d_group(combined, cache=conv_state, output_final_state=use_cache, cu_seqlens=kwargs.get('cu_seqlens'))
            q_pre_spike, k_pre_spike, v_pre_spike, g_pre_sigmoid = torch.split(combined, [self.key_dim, self.key_dim, self.value_dim, self.num_heads], dim=-1)
        else:
            q_pre_spike, k_pre_spike, v_pre_spike, g_pre_sigmoid = q_out, k_out, v_out, g_out

        # NEW: Apply core spiking neurons to q, k, v streams to get binary spikes
        q_spikes_gla = self.q_core_neuron(q_pre_spike.transpose(-1,-2)).transpose(-1,-2)
        k_spikes_gla = self.k_core_neuron(k_pre_spike.transpose(-1,-2)).transpose(-1,-2)
        v_spikes_gla = self.v_core_neuron(v_pre_spike.transpose(-1,-2)).transpose(-1,-2)
        all_spikes.update({'q_core_gla_spikes': q_spikes_gla, 'k_core_gla_spikes': k_spikes_gla, 'v_core_gla_spikes': v_spikes_gla})

        # Gate activation for g: LogSigmoid -> Log Space Decay. `g` remains real-valued.
        # g_pre_sigmoid (continuous) from projections/convolution is passed through logsigmoid.
        g_decay_rates = F.logsigmoid(g_pre_sigmoid)

        # Reshape for multi-head attention (now using the spiked tensors for q, k, v)
        q = rearrange(q_spikes_gla, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k_spikes_gla, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v_spikes_gla, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g_decay_rates, 'b l h -> b h l 1') # [B, H, L, 1] - continuous decay factor

        # State Management for recurrence
        last_state = None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        # Chunkwise Parallel Op. Inputs q, k, v are now binary spikes, g is continuous.
        o_gla_output, final_state = gla_chunkwise(q, k, v, g, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            state_dict = {'recurrent_state': final_state}
            if self.use_short_conv: state_dict['conv_state'] = new_conv_state

            if len(past_key_values) <= layer_idx:
                past_key_values.append(state_dict)
            else:
                past_key_values[layer_idx] = state_dict

        # Output reshaping and normalization
        o = rearrange(o_gla_output, 'b h l d -> b l h d')
        o = self.g_norm(o) # g_norm operates on continuous output of GLA
        o = rearrange(o, 'b l h d -> b l (h d)')

        # Final output projection through SpikingLinear
        o_final_out, s_o_proj = self.o_proj(o) # o_proj takes continuous output of GLA (after norm), produces spikes
        all_spikes['o_proj_spikes'] = s_o_proj

        return o_final_out, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        # First linear layer is SpikingLinear, ensuring its internal nn.Linear processes spikes.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)

        # Introduce an intermediate neuron to spike the output of fc1 BEFORE feeding to fc2.
        # This ensures the input to fc2's internal nn.Linear is also binary spikes.
        self.intermediate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)

        # Second linear layer is SpikingLinear.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_out, spike1 = self.fc1(x) # spike1 are spikes fed to fc1's linear, x_fc1_out is continuous output

        # Apply intermediate neuron to the real-valued output of fc1 to get intermediate spikes.
        # These `spikes_mid` will then be fed into fc2, meaning fc2's internal nn.Linear processes `spikes_mid`.
        spikes_mid_transposed = self.intermediate_neuron(x_fc1_out.transpose(-1, -2))
        spikes_mid = spikes_mid_transposed.transpose(-1, -2)

        # x_fc2_out is the real-valued output of fc2's internal linear layer.
        # spike2 are the binary spikes that fed into fc2's internal linear layer.
        # Note: dropout is applied *before* SpikingLinear's neuron, so spike2 captures spikes *after* dropout.
        x_fc2_out, spike2 = self.fc2(self.dropout(spikes_mid))

        # Return final output and all collected spike tensors.
        # spike1: spikes input to fc1's internal linear
        # spikes_mid: spikes from intermediate neuron (pre-dropout for fc2's input)
        # spike2: spikes input to fc2's internal linear (post-dropout)
        return x_fc2_out, spike1, spikes_mid, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        self.attn = SpikingGLA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        surrogate_function = piecewise_quadratic_surrogate()

        # Gate projection for attention branch residual connection using SpikingLinear
        self.gate_attn_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        # Inhibition gate for attention branch using SpikingLinear
        self.attn_inhibition_gate = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)

        self.dropout2 = nn.Dropout(dropout)

        # Gate projection for MLP branch residual connection using SpikingLinear
        self.gate_mlp_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        # Inhibition gate for MLP branch using SpikingLinear
        self.mlp_inhibition_gate = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # --- Attention Block with Spiking Gating and Inhibition ---
        x_res1 = x
        if self.prenorm: 
            x_norm1 = self.norm1(x)
        else:
            x_norm1 = x 

        x_attn_output_continuous, _, attn_spikes_dict = self.attn(x_norm1, past_key_values=past_key_values, layer_idx=kwargs.get('layer_idx', None), use_cache=use_cache, **kwargs)

        if attn_spikes_dict:
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        # Apply dropout to the attention output branch (continuous)
        attn_branch = self.dropout1(x_attn_output_continuous)

        # Generate gating and inhibition factors. The linear layers internally process spikes
        # from x_norm1, and we explicitly use the *binary spikes* from their neuron output for gating.
        _, s_gate_attn = self.gate_attn_proj(x_norm1) # s_gate_attn are binary spikes
        _, s_inhib_attn = self.attn_inhibition_gate(x_norm1) # s_inhib_attn are binary spikes
        all_spikes_list.extend([s_gate_attn, s_inhib_attn])

        # Ensure spikes are binary (0 or 1) for effective multiplicative gating
        s_gate_attn_binary = (s_gate_attn > 0).float()
        s_inhib_attn_binary = (s_inhib_attn > 0).float()

        # Adaptive residual connection with spiking inhibition:
        # residual + (continuous_branch_output * binary_gating_factor) * (1 - binary_inhibition_factor)
        # This effectively switches the branch contribution on/off or reduces it based on binary spikes.
        x = x_res1 + attn_branch * s_gate_attn_binary * (1.0 - s_inhib_attn_binary)

        # --- MLP Block with Spiking Gating and Inhibition ---
        x_res2 = x
        if self.prenorm: 
            x_norm2 = self.norm2(x)
        else:
            x_norm2 = x 

        # SpikingMLP returns its output (continuous) and three spike tensors.
        x_mlp_output_continuous, mlp_spike_fc1, mlp_spike_intermediate, mlp_spike_fc2 = self.mlp(x_norm2)
        all_spikes_list.extend([mlp_spike_fc1, mlp_spike_intermediate, mlp_spike_fc2])

        # Apply dropout to the MLP output branch (continuous)
        mlp_branch = self.dropout2(x_mlp_output_continuous)

        # Generate gating and inhibition factors for MLP branch using SpikingLinear
        _, s_gate_mlp = self.gate_mlp_proj(x_norm2) # s_gate_mlp are binary spikes
        _, s_inhib_mlp = self.mlp_inhibition_gate(x_norm2) # s_inhib_mlp are binary spikes
        all_spikes_list.extend([s_gate_mlp, s_inhib_mlp])

        # Ensure spikes are binary (0 or 1) for effective multiplicative gating
        s_gate_mlp_binary = (s_gate_mlp > 0).float()
        s_inhib_mlp_binary = (s_inhib_mlp > 0).float()

        # Adaptive residual connection with spiking inhibition for MLP
        x_final_output_gated = x_res2 + mlp_branch * s_gate_mlp_binary * (1.0 - s_inhib_mlp_binary)

        final_spikes = None
        if all_spikes_list:
            # Concatenate all collected spike tensors into a single flat tensor as required.
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])
            final_spikes = (final_spikes > 0).float() # Ensure binary output

        return x_final_output_gated, None, final_spikes