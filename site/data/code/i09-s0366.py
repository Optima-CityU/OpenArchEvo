import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

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
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D) # Binary spikes
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

# ==================== White-box Operator (Gated Delta Product) ====================

def gated_delta_chunkwise_parallel(q, k, v, beta, decay, chunk_size=64, initial_state=None):
    """White-box parallel Gated Delta Product operator.
    Uses Wy transform approximation to handle intra-chunk dependencies.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        # Use F.pad with value 0 for q, k, v, beta and 1 for decay as specified
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len), value=0.0)
        decay = F.pad(decay, (0, pad_len), value=1.0) # Decay should be 1.0 for padded elements to not affect state
    
    # 1. Standard normalization (applied to continuous inputs of the operator)
    q = l2norm(q)
    k = l2norm(k)
    
    # Data reshaping for chunk processing
    q, k, v = [rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) for x in [q, k, v]]
    beta, decay = [rearrange(x, 'b h (n c) -> b h n c', c=chunk_size) for x in [beta, decay]]
    
    num_chunks = q.shape[2]
    # S shape: [B, H, Dk, Dv] (aligned with Delta Rule: mapping Key space to Value space)
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device, dtype=q.dtype) # Ensure dtype matches q
    o = torch.zeros_like(v)
    
    # Pre-prepare Causal mask
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, b_i, d_i = q[:, :, i], k[:, :, i], v[:, :, i], beta[:, :, i], decay[:, :, i]
        
        # [Step 1] Read historical state (inter-chunk dependency)
        o_inter = q_i @ S # [B, H, C, Dk] @ [B, H, Dk, Dv] -> [B, H, C, Dv]
        
        # [Step 2] Intra-chunk parallel update logic (Wy Transform approximation)
        # v_error = (v - k@S) * beta
        v_error = (v_i - (k_i @ S)) * b_i.unsqueeze(-1) # b_i [B, H, C] -> [B, H, C, 1]
        
        # Associative Memory correction for intra-chunk updates
        # (k_i @ k_i.transpose(-1, -2)) -> [B, H, C, C]
        attn_intra = (k_i @ k_i.transpose(-1, -2)) * mask_tril
        v_error_intra = v_error - (attn_intra @ v_error) * 0.5 
        
        # [Step 3] Combined output
        o_intra = q_i @ (k_i.transpose(-1, -2) @ v_error_intra) # [B, H, C, Dk] @ ([B, H, Dk, C] @ [B, H, C, Dv]) -> [B, H, C, Dv]
        o[:, :, i] = o_inter + o_intra
        
        # [Step 4] Recurrent state update
        d_mean = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1) # d_i [B, H, C] -> [B, H, 1] -> [B, H, 1, 1]
        S = d_mean * S + (k_i.transpose(-1, -2) @ v_error_intra) # [B, H, Dk, Dv]
        
        # --- MODIFICATION: Dynamic State Normalization ---
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 
        # --- END MODIFICATION ---

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    # Remove padding if it was added
    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

# ==================== SpikingGatedDeltaProduct Layer ====================

class SpikingGatedDeltaProduct(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # All projections use SpikingLinear as required
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Beta for learning rate
        self.w_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Weight for decay

        # Original recurrent output gate, renamed for clarity
        self.recurrent_gate_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate) 

        # Direct path projection and its gate for output fusion
        self.direct_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.direct_gate_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)


        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            # Convolutions for the direct path and gates
            self.recurrent_gate_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            self.direct_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            self.direct_gate_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
        else:
            # Explicitly set to None if not used for clearer hasattr checks
            self.q_conv1d = None 
            self.k_conv1d = None
            self.v_conv1d = None
            self.recurrent_gate_conv1d = None
            self.direct_conv1d = None
            self.direct_gate_conv1d = None


        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes_dict = {}

        # 1. Spiking projections
        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        beta, s_b = self.b_proj(x)
        w, s_w = self.w_proj(x)

        # GATES AND DIRECT PROJECTION
        recurrent_gate_raw, s_rg = self.recurrent_gate_proj(x)
        direct_val_raw, s_dv = self.direct_proj(x)
        direct_gate_raw, s_dg = self.direct_gate_proj(x)

        all_spikes_dict.update({'q':s_q, 'k':s_k, 'v':s_v, 'beta':s_b, 'w':s_w, 
                                'recurrent_gate':s_rg, 'direct_val':s_dv, 'direct_gate':s_dg})

        # 2. Local feature extraction via ShortConvolution
        if self.q_conv1d is not None: # Check against None
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v_conv, _ = self.v_conv1d(v) # v_conv for input to GDP

            # Convolutions for gates and direct path
            recurrent_gate, _ = self.recurrent_gate_conv1d(recurrent_gate_raw)
            direct_val, _ = self.direct_conv1d(direct_val_raw)
            direct_gate, _ = self.direct_gate_conv1d(direct_gate_raw)
        else:
            v_conv = v
            recurrent_gate = recurrent_gate_raw
            direct_val = direct_val_raw
            direct_gate = direct_gate_raw

        # 3. Activation functions
        beta = torch.sigmoid(beta) * 0.5 
        decay = torch.sigmoid(w) 

        # GATE ACTIVATIONS
        recurrent_gate = torch.sigmoid(recurrent_gate) 
        direct_gate = torch.sigmoid(direct_gate) 

        # 4. Data reshaping for core operator
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v_in_gdp = rearrange(v_conv, 'b l (h d) -> b h l d', h=self.num_heads) # v_conv is the input to GDP
        beta = rearrange(beta, 'b l h -> b h l')
        decay = rearrange(decay, 'b l h -> b h l')

        # 5. State management for recurrent connections
        layer_state = None
        if past_key_values is not None:
            # Safely get state using .get() to handle cases where layer_idx might not be present yet
            layer_state = past_key_values.get(self.layer_idx, None)


        # 6. Execute core operator
        o_recurrent, final_state = gated_delta_chunkwise_parallel(q, k, v_in_gdp, beta, decay, initial_state=layer_state)

        if past_key_values is not None:
            past_key_values[self.layer_idx] = final_state

        # 7. Output fusion (recurrent path + direct path)
        # Reshape recurrent_gate, direct_gate, and direct_val to match o_recurrent's head-split format
        recurrent_gate_rearranged = rearrange(recurrent_gate, 'b l (h d) -> b h l d', h=self.num_heads)
        direct_gate_rearranged = rearrange(direct_gate, 'b l (h d) -> b h l d', h=self.num_heads)
        direct_val_rearranged = rearrange(direct_val, 'b l (h d) -> b h l d', h=self.num_heads)

        # Combine recurrent and direct path with their respective gates
        y_fused = (recurrent_gate_rearranged * o_recurrent) + (direct_gate_rearranged * direct_val_rearranged)

        # 8. Normalization and final projection
        y_fused = rearrange(y_fused, 'b h l d -> b l h d') # Reshape to apply o_norm per head correctly
        y_fused = self.o_norm(y_fused)

        y_fused = rearrange(y_fused, 'b l h d -> b l (h d)') # Reshape back to flat d_model for final projection
        o, s_o = self.o_proj(y_fused)
        all_spikes_dict['o'] = s_o

        return o, None, all_spikes_dict

# SpikingMLP class: always uses SpikingLinear

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x)
        # F.gelu(x) removed here to promote more direct spiking signal flow, minimizing intermediate analog computations.
        x = self.dropout(x) # Apply dropout after first layer's analog output
        x, spike2 = self.fc2(x) # Feed directly to second spiking linear
        return x, spike1, spike2

# ContextualSpikeActivityGate to modulate residuals based on aggregated spike activity from multiple sources

class ContextualSpikeActivityGate(nn.Module):
    def __init__(self, num_spike_features: int, neuron_type: str, surrogate_function):
        super().__init__()
        # Takes (B, L, num_spike_features) and projects to (B, L, 1) modulation factor.
        # The linear layer receives binary spikes, and the neuron converts the analog input
        # (the concatenated mean activities) into binary spikes before the linear projection.
        self.proj = SpikingLinear(nn.Linear(num_spike_features, 1, bias=False), neuron_type, surrogate_function)
        self.bias = nn.Parameter(torch.zeros(1)) # Learnable bias for sigmoid

    def forward(self, aggregated_mean_spike_activities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        modulated_val_continuous, spikes_from_gate_proj = self.proj(aggregated_mean_spike_activities) 

        # Use a scaled sigmoid to produce a positive modulation factor
        # E.g., a factor between 1.0 and 2.0. This prevents complete gating (0)
        # and allows boosting (up to 2x) based on spike activity.
        modulation_factor = 1.0 + torch.sigmoid(modulated_val_continuous + self.bias)

        return modulation_factor, spikes_from_gate_proj

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingGatedDeltaProduct(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # Determine number of spike features for the attention gate
        # q, k, v, o from SpikingGatedDeltaProduct
        num_attn_spike_features = 4 # q, k, v, o (from attn block itself)
        self.attn_spike_gate = ContextualSpikeActivityGate(num_attn_spike_features, neuron_type, piecewise_quadratic_surrogate())

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)

        # Determine number of spike features for the MLP gate
        # fc1, fc2 from SpikingMLP
        num_mlp_spike_features = 2 # fc1, fc2
        self.mlp_spike_gate = ContextualSpikeActivityGate(num_mlp_spike_features, neuron_type, piecewise_quadratic_surrogate())

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []
        reg_loss = None # Keep for signature compatibility

        # --- Attention Block ---
        x_res_attn = x # Store residual for attention
        if self.prenorm: x = self.norm1(x)
        x_attn, _, attn_spikes_dict = self.attn(x, **kwargs)

        # Collect spikes from attention block and prepare for contextual spike-gating
        attn_spike_features = []

        # Define the spike sources we want to use for the attention gate
        attn_gate_spike_sources = ['q', 'k', 'v', 'o'] # from SpikingGatedDeltaProduct

        for source_name in attn_gate_spike_sources:
            spike_tensor = attn_spikes_dict.get(source_name)
            if spike_tensor is not None and spike_tensor.numel() > 0:
                all_spikes_list.append(spike_tensor)
                # Ensure the mean is taken across the feature dimension, resulting in (B, L, 1)
                attn_spike_features.append(spike_tensor.float().mean(dim=-1, keepdim=True))
            else:
                # If a source is missing or empty, append a zero tensor of the correct shape
                # This ensures the concatenated tensor always has the expected num_spike_features
                if x_attn.shape[0] > 0 and x_attn.shape[1] > 0: # Check if B, L are valid
                    zero_feature = torch.zeros(x_attn.shape[0], x_attn.shape[1], 1, device=x.device, dtype=torch.float)
                    attn_spike_features.append(zero_feature)
                else: # Handle edge case of empty sequence length, etc.
                    # For a truly empty input sequence, we might get 0-dimensional tensors.
                    # In this case, we need to ensure the gate input is also 0-dimensional or handled by the gate's robustness.
                    # For now, let's return an empty tensor that will be caught by numel() > 0 check later.
                    attn_spike_features.append(torch.empty(0, 0, 1, device=x.device, dtype=torch.float))


        # Apply Contextual Spike-Activity modulated residual for Attention
        attn_modulation_factor = 1.0 # Default to no modulation
        attn_gate_spikes_collected = None
        if attn_spike_features:
            # Concatenate features along the last dimension to create input for ContextualSpikeActivityGate
            # Filter out potentially empty tensors if they arose from edge cases, so torch.cat doesn't fail
            valid_attn_spike_features = [f for f in attn_spike_features if f.numel() > 0]

            if valid_attn_spike_features:
                aggregated_attn_mean_activities = torch.cat(valid_attn_spike_features, dim=-1) # (B, L, num_attn_spike_features)

                # Only apply gate if the aggregated_attn_mean_activities tensor is not empty
                if aggregated_attn_mean_activities.numel() > 0:
                    attn_modulation_factor, attn_gate_spikes_collected = self.attn_spike_gate(aggregated_attn_mean_activities)
                    if attn_gate_spikes_collected is not None and attn_gate_spikes_collected.numel() > 0:
                        all_spikes_list.append(attn_gate_spikes_collected)

        x_attn_modulated = x_attn * attn_modulation_factor 
        x = x_res_attn + self.dropout1(x_attn_modulated) # Apply residual connection

        # --- MLP Block ---
        x_res_mlp = x # Store residual for MLP (output of attention block)
        if self.prenorm: x = self.norm2(x)
        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x)

        # Collect spikes from MLP block and prepare for contextual spike-gating
        mlp_spike_features = []

        # Define the spike sources we want to use for the MLP gate
        mlp_gate_spike_sources = [mlp_spike1, mlp_spike2] # from SpikingMLP

        for spike_tensor in mlp_gate_spike_sources:
            if spike_tensor is not None and spike_tensor.numel() > 0:
                all_spikes_list.append(spike_tensor)
                mlp_spike_features.append(spike_tensor.float().mean(dim=-1, keepdim=True))
            else:
                # If a source is missing or empty, append a zero tensor of the correct shape
                if x_mlp.shape[0] > 0 and x_mlp.shape[1] > 0:
                    zero_feature = torch.zeros(x_mlp.shape[0], x_mlp.shape[1], 1, device=x.device, dtype=torch.float)
                    mlp_spike_features.append(zero_feature)
                else:
                    mlp_spike_features.append(torch.empty(0, 0, 1, device=x.device, dtype=torch.float))


        # Apply Contextual Spike-Activity modulated residual for MLP
        mlp_modulation_factor = 1.0 # Default to no modulation
        mlp_gate_spikes_collected = None
        if mlp_spike_features:
            # Filter out potentially empty tensors before concatenation
            valid_mlp_spike_features = [f for f in mlp_spike_features if f.numel() > 0]

            if valid_mlp_spike_features:
                aggregated_mlp_mean_activities = torch.cat(valid_mlp_spike_features, dim=-1) # (B, L, num_mlp_spike_features)

                # Only apply gate if the aggregated_mlp_mean_activities tensor is not empty
                if aggregated_mlp_mean_activities.numel() > 0:
                    mlp_modulation_factor, mlp_gate_spikes_collected = self.mlp_spike_gate(aggregated_mlp_mean_activities)
                    if mlp_gate_spikes_collected is not None and mlp_gate_spikes_collected.numel() > 0:
                        all_spikes_list.append(mlp_gate_spikes_collected)

        x_mlp_modulated = x_mlp * mlp_modulation_factor
        x = x_res_mlp + self.dropout2(x_mlp_modulated) # Apply residual connection

        # Final spike collection: flatten and concatenate all collected binary spikes
        # Ensure that `all_spikes_list` is not empty before concatenating.
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else torch.empty(0, device=x.device, dtype=torch.bool)


        return x, reg_loss, final_spikes