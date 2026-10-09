import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, List

from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ==================== SNN Base Components (Consistent with DeltaNet) ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return (input > 0).float() # CRITICAL FIX: Ensure binary spikes
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
        x_transposed = x.transpose(-1, -2) # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2) # (B, D, L) -> (B, L, D)
        output = self.linear(spikes) # Linear layer receives spikes
        return output, spikes

# ==================== White-box Operator (Gated Delta Product) ====================

def gated_delta_chunkwise_parallel(q, k, v, beta, decay, chunk_size=64, initial_state=None):
    """White-box parallel Gated Delta Product operator.
    Uses Wy transform approximation to handle intra-chunk dependencies, ensuring mathematical logic is visible to Sampler.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        # Beta is (B, H, L), Decay is (B, H, L)
        beta = F.pad(beta, (0, pad_len), value=0.0) # Pad L dimension
        decay = F.pad(decay, (0, pad_len), value=1.0) # Pad L dimension
    
    # 1. Standard normalization and data reshaping
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    
    # Rearrange for chunk processing
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    beta = rearrange(beta, 'b h (n c) -> b h n c', c=chunk_size)
    decay = rearrange(decay, 'b h (n c) -> b h n c', c=chunk_size)
    
    num_chunks = q.shape[2]
    # S shape: [B, H, Dk, Dv] (aligned with Delta Rule: mapping Key space to Value space)
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    # Pre-prepare Bool mask
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, b_i, d_i = q[:, :, i], k[:, :, i], v[:, :, i], beta[:, :, i], decay[:, :, i]
        
        # [Step 1] Read historical state
        o_inter = q_i @ S
        
        # [Step 2] Intra-chunk parallel update logic (Wy Transform approximation)
        # b_i is now binary (0 or 1), making the error computation sparse.
        v_error = (v_i - (k_i @ S)) * b_i.unsqueeze(-1) # (B, H, C, Dv)
        
        # Consider the cancellation effect of previous k in the chunk on current update (Associative Memory correction)
        attn_intra_mask = (k_i @ k_i.transpose(-1, -2)) * mask_tril.unsqueeze(0).unsqueeze(0)
        # 0.5 is to balance second-order gradients and maintain SNN stability
        v_error_intra = v_error - (attn_intra_mask @ v_error) * 0.5 # (B, H, C, C) @ (B, H, C, Dv) -> (B, H, C, Dv)
        
        # [Step 3] Combined output
        o_intra = q_i @ (k_i.transpose(-1, -2) @ v_error_intra)
        o[:, :, i] = o_inter + o_intra
        
        # [Step 4] Recurrent state update
        d_mean = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1) # (B, H, 1, 1)
        S = d_mean * S + (k_i.transpose(-1, -2) @ v_error_intra)
        
        # Clamp protection for ZCP computation
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== Spiking Wrapper for ShortConvolution (re-introduced from V1, for uniformity) ====================

class SpikingConvWrapper(nn.Module):
    def __init__(self, conv_layer: ShortConvolution, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.conv = conv_layer

    def forward(self, x):
        spikes_input = self.neuron(x.transpose(-1, -2)).transpose(-1, -2)
        # Convolution now receives spikes (ShortConvolution expects analog float input, but here it gets binary spikes from the neuron)
        conv_out_analog, _ = self.conv(spikes_input) # ShortConvolution's internal activation will operate on these, making output analog.
        # Return analog output and generated spikes
        return conv_out_analog, spikes_input

# ==================== SpikingGatedDeltaProduct Layer ====================

class SpikingGatedDeltaProduct(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.use_short_conv = use_short_conv # Store this flag

        # Dimension forced conversion
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections (6-way projection: Q, K, V, Beta, Decay, Gate)
        # All inputs to the internal nn.Linear layers of SpikingLinear are binary spikes
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        self.w_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        # NEW: A standalone spiking neuron for the output gate
        self.gate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)

        # NEW: Neurons to make beta and decay factors binary
        self.beta_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.decay_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)

        if use_short_conv:
            # Wrapped ShortConvolution instances with SpikingConvWrapper
            self.q_conv1d = SpikingConvWrapper(ShortConvolution(self.key_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.k_conv1d = SpikingConvWrapper(ShortConvolution(self.key_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.v_conv1d = SpikingConvWrapper(ShortConvolution(self.value_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.g_conv1d = SpikingConvWrapper(ShortConvolution(self.value_dim, conv_size, activation='silu'), neuron_type, surrogate)
        else:
            self.q_conv1d = None
            self.k_conv1d = None
            self.v_conv1d = None
            self.g_conv1d = None


        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}

        # 1. Spike projection: Input `x` is real-valued, outputs `q, k, v, beta_raw, w_raw, gate_raw_val` are real-valued
        # and `s_*` are the actual binary spikes fed into the internal Linear layers.
        q, s_q_proj_in = self.q_proj(x)
        k, s_k_proj_in = self.k_proj(x)
        v, s_v_proj_in = self.v_proj(x)
        beta_raw, s_b_proj_in = self.b_proj(x)
        w_raw, s_w_proj_in = self.w_proj(x)
        gate_raw_val, s_g_proj_in = self.g_proj(x)

        all_spikes.update({'q_proj_input':s_q_proj_in, 'k_proj_input':s_k_proj_in, 'v_proj_input':s_v_proj_in, 'beta_proj_input':s_b_proj_in, 'w_proj_input':s_w_proj_in, 'g_proj_input':s_g_proj_in})

        # 2. Local feature extraction (Convolution operates on real-valued outputs of SpikingLinear, but now wrapped with a neuron)
        if self.use_short_conv: # Use the flag to conditionally apply convolutions
            q_conv_out, s_q_conv_in = self.q_conv1d(q)
            k_conv_out, s_k_conv_in = self.k_conv1d(k)
            v_conv_out, s_v_conv_in = self.v_conv1d(v)
            gate_conv_out, s_g_conv_in = self.g_conv1d(gate_raw_val) # Conv on real-valued gate projection

            all_spikes.update({'q_conv_input':s_q_conv_in, 'k_conv_input':s_k_conv_in, 'v_conv_input':s_v_conv_in, 'g_conv_input':s_g_conv_in})

            q = q_conv_out
            k = k_conv_out
            v = v_conv_out
            gate_raw_val = gate_conv_out
        # If not use_short_conv, q,k,v,gate_raw_val remain as output from proj layers (analog outputs from SpikingLinear).

        # 3. Activation function alignment (based on SNN stability optimization) and NEW binary modulation factors
        beta_analog = torch.sigmoid(beta_raw) * 0.5 # Analog value (0, 0.5)
        decay_analog = torch.sigmoid(w_raw) # Analog value (0, 1)

        # NEW: Produce binary spikes for beta and decay factors
        beta_spikes = self.beta_neuron(beta_analog.transpose(-1, -2)).transpose(-1, -2) # Now beta is binary (0 or 1)
        decay_spikes = self.decay_neuron(decay_analog.transpose(-1, -2)).transpose(-1, -2) # Now decay is binary (0 or 1)
        all_spikes.update({'beta_neuron_output': beta_spikes, 'decay_neuron_output': decay_spikes})

        # NEW: Produce binary spikes for the gate directly
        # The gate_raw_val is output of g_proj.linear (real-valued, potentially convolved). Feed it to a neuron.
        gate_final_spikes = self.gate_neuron(gate_raw_val.transpose(-1, -2)).transpose(-1, -2)
        all_spikes['g_proj_output'] = gate_final_spikes # Collect these spikes
        gate_signal = gate_final_spikes # Use the spikes as the gate signal

        # 4. Data reshaping
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        beta = rearrange(beta_spikes, 'b l h -> b h l') # Beta is now binary
        decay = rearrange(decay_spikes, 'b l h -> b h l') # Decay is now binary

        # 5. State management
        layer_state = None
        if past_key_values is not None:
            if isinstance(past_key_values, dict) and self.layer_idx in past_key_values:
                layer_state = past_key_values[self.layer_idx]
            elif isinstance(past_key_values, list) and self.layer_idx is not None and self.layer_idx < len(past_key_values):
                layer_state = past_key_values[self.layer_idx]

        # 6. Execute core operator (Parallel) - beta and decay are binary here.
        o, final_state = gated_delta_chunkwise_parallel(q, k, v, beta, decay, initial_state=layer_state)

        if past_key_values is not None:
            if isinstance(past_key_values, dict):
                past_key_values[self.layer_idx] = final_state
            elif isinstance(past_key_values, list) and self.layer_idx is not None:
                if self.layer_idx >= len(past_key_values):
                    past_key_values.extend([None] * (self.layer_idx - len(past_key_values) + 1))
                past_key_values[self.layer_idx] = final_state


        # 7. Gate merging and final projection (aligned with FLA GatedDeltaProduct)
        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)

        # Apply output gate: [B, L, H, D] * [B, L, H, D]
        # Use the binary gate_signal (spikes) here
        gate_signal = rearrange(gate_signal, 'b l (h d) -> b l h d', h=self.num_heads)
        o = o * gate_signal # o is multiplied by binary gate spikes

        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o_proj_in = self.o_proj(o) # Output projection, also spiking.
        all_spikes['o_proj_input'] = s_o_proj_in

        return o, None, all_spikes

# ==================== SpikingMLP and new SpikingFFN (for uniform SpikingLinear usage) ====================

class SpikingFFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.gelu = nn.GELU() # Activation applied to real-valued output of SpikingLinear
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x_out_fc1_analog, s_fc1_in = self.fc1(x) # x is real-valued output, spike1 are spikes that fed into fc1.linear
        x_act_analog = self.gelu(x_out_fc1_analog)
        x_dropped_analog = self.dropout(x_act_analog)
        x_out_fc2_analog, s_fc2_in = self.fc2(x_dropped_analog) # x is real-valued output, spike2 are spikes that fed into fc2.linear
        return x_out_fc2_analog, s_fc1_in, s_fc2_in

# ==================== SpikingFLABlock (Unified Interface) ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp=True, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', spike_regularization_weight: float=0.001, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        surrogate_function = piecewise_quadratic_surrogate()

        # Attention Block Components
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingGatedDeltaProduct(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # NEW: Spiking Residual Gating for Attention Block
        self.attn_residual_gate_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        # Add a dedicated neuron for converting the analog output of attn_residual_gate_proj to binary spikes
        self.attn_residual_gate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)


        # MLP Block Components
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        self.mlp = SpikingFFN(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.dropout2 = nn.Dropout(dropout)

        # NEW: Spiking Residual Gating for MLP Block
        self.mlp_residual_gate_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        # Add a dedicated neuron for converting the analog output of mlp_residual_gate_proj to binary spikes
        self.mlp_residual_gate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)


        self.spike_regularization_weight = spike_regularization_weight

    def forward(self, x: torch.Tensor, past_key_values: Optional[Dict]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # === Attention Block ===
        x_res1 = x # Store residual connection input
        x_attn_in = self.norm1(x) if self.prenorm else x

        x_attn_analog_out, reg_loss_attn, attn_spikes_dict = self.attn(x_attn_in, past_key_values=past_key_values, **kwargs)
        if attn_spikes_dict:
            # Flatten all spikes from the dict values and extend the main list
            for key in attn_spikes_dict:
                if attn_spikes_dict[key] is not None:
                    all_spikes_list.append(attn_spikes_dict[key].flatten())


        # NEW: Spiking Residual Gating for Attention Block
        gate_attn_out_analog, s_attn_residual_gate_proj_in = self.attn_residual_gate_proj(x_res1)
        all_spikes_list.append(s_attn_residual_gate_proj_in.flatten())

        # Now, pass the analog output through a neuron to get binary gating spikes
        # Transpose for neuron processing and then transpose back
        gating_factor_attn_spikes = self.attn_residual_gate_neuron(gate_attn_out_analog.transpose(-1, -2)).transpose(-1, -2)
        all_spikes_list.append(gating_factor_attn_spikes.flatten()) # Collect these spikes too

        # Apply binary gating factor
        branch_output_attn = self.dropout1(x_attn_analog_out) * gating_factor_attn_spikes
        x_after_attn = x_res1 + branch_output_attn # Add gated branch output to residual input

        # === MLP Block ===
        x_res2 = x_after_attn # Store residual connection input for MLP
        x_mlp_in = self.norm2(x_after_attn) if self.prenorm else x_after_attn

        x_mlp_analog_out, s_mlp_fc1_in, s_mlp_fc2_in = self.mlp(x_mlp_in)
        all_spikes_list.extend([s_mlp_fc1_in.flatten(), s_mlp_fc2_in.flatten()])

        # NEW: Spiking Residual Gating for MLP Block
        gate_mlp_out_analog, s_mlp_residual_gate_proj_in = self.mlp_residual_gate_proj(x_res2)
        all_spikes_list.append(s_mlp_residual_gate_proj_in.flatten())

        # Now, pass the analog output through a neuron to get binary gating spikes
        # Transpose for neuron processing and then transpose back
        gating_factor_mlp_spikes = self.mlp_residual_gate_neuron(gate_mlp_out_analog.transpose(-1, -2)).transpose(-1, -2)
        all_spikes_list.append(gating_factor_mlp_spikes.flatten()) # Collect these spikes too

        # Apply binary gating factor
        branch_output_mlp = self.dropout2(x_mlp_analog_out) * gating_factor_mlp_spikes
        x_final_output = x_res2 + branch_output_mlp # Add gated branch output to residual input

        # 3. Spikes Collection and Regularization
        valid_spikes = [s for s in all_spikes_list if s is not None]
        final_spikes = torch.cat(valid_spikes) if valid_spikes else None

        current_reg_loss = reg_loss_attn if reg_loss_attn is not None else 0.0
        # Spike regularization will be computed on the entire final_spikes tensor,
        # which now includes the new binary gating spikes.
        if final_spikes is not None and self.spike_regularization_weight > 0:
            current_reg_loss = current_reg_loss + self.spike_regularization_weight * final_spikes.mean()

        # The forward signature must return (output, None, final_spikes).
        return x_final_output, None, final_spikes