import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, List

from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ==================== SNN Related Imports and Helper Classes ====================
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
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x):
        x_transposed = x.transpose(-1, -2)
        spikes = self.neuron(x_transposed).transpose(-1, -2) # Binary spikes
        return self.linear(spikes), spikes # output is analog, spikes are binary

# ==================== White-box Operator (KDA Chunkwise) ====================

def kda_chunkwise_parallel(q, k, v, beta, decay, alpha, chunk_size=32, initial_state=None):
    """KDA Chunkwise parallel operator.
    Core logic: S_t = decay * S_{t-1} + beta * (alpha * v_t - S_{t-1} @ k_t) @ k_t^T
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len), value=0.0)
        decay = F.pad(decay, (0, pad_len), value=0.0) 
        alpha = F.pad(alpha, (0, pad_len), value=0.0)
        
    # 1. Robust normalization
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    v = v * alpha.unsqueeze(-1) # Input Gate

    # Reshape to Chunks
    q, k, v, beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, beta, decay]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)

    # Pre-generate Bool type mask to prevent errors
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, b_i, d_i = q[:, :, i], k[:, :, i], v[:, :, i], beta[:, :, i], decay[:, :, i]
        
        # [Step 1] Compute decay matrix (Log-space)
        d_cumsum = torch.cumsum(d_i, dim=-1) 
        decay_rel = (d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)) 
        decay_mask = torch.exp(decay_rel).masked_fill(mask_triu, 0) # Keep only causal part
        
        # [Step 2] Compute prediction error (Delta Rule + Intra-chunk Wy approximation)
        # v_pred_inter is the prediction from historical state S
        v_pred_inter = k_i @ S 
        v_error = (v_i - v_pred_inter) * b_i.unsqueeze(-1)
        
        # Intra-chunk parallel compensation: handle the effect of previous k in this chunk on current update
        # Aligned with FLA's Wy transform logic
        attn_intra = (k_i @ k_i.transpose(-1, -2)) * decay_mask * mask_tril
        v_error_intra = v_error - (attn_intra @ v_error) * 0.5 # 0.5 is stability factor
        
        # [Step 3] Historical reading and intra-chunk incremental output
        decay_from_start = torch.exp(d_cumsum).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        q_decay_k = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = q_decay_k @ v_error_intra
        
        o[:, :, i] = o_inter + o_intra
        
        # [Step 4] State update
        # Decay ratio to end of chunk
        d_mean = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1)
        delta_S = k_i.transpose(-1, -2) @ v_error_intra
        S = d_mean.exp() * S + delta_S
        
        # Clamp to prevent overflow, ensure SNN ZCP computation stability
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingKDA Layer ====================

class SpikingKDA(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=1.0, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        # Consolidate base control projections (beta, decay, alpha) into one SpikingLinear layer
        self.control_proj = SpikingLinear(nn.Linear(d_model, num_heads * 3, bias=False), neuron_type, surrogate) 

        # NEW: State-adaptive control modulation using SpikingLinear
        # Takes a scalar feature derived from the global state 'S' and projects to num_heads * 3
        self.state_control_proj = SpikingLinear(nn.Linear(1, num_heads * 3, bias=True), neuron_type, surrogate)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, initial_state: Optional[torch.Tensor]=None, **kwargs):
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        all_spikes.update({'q':s_q, 'k':s_k, 'v':s_v})

        # Generate base control signals from hidden_states
        control_signals_base, s_control_base = self.control_proj(x)
        all_spikes['control_base'] = s_control_base

        # Prepare and compute state-derived feature for modulation
        if initial_state is None:
            # Dynamically infer batch size from x
            batch_size = x.shape[0]
            S_for_state_feat = torch.zeros(
                batch_size, self.num_heads, self.key_dim // self.num_heads, self.head_v_dim,
                device=x.device
            )
        else:
            S_for_state_feat = initial_state

        # Compute a scalar feature per batch from the state S
        # Norm per head, then mean across heads to get a global feature [B, 1]
        s_norm_state = S_for_state_feat.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6 # [B, H, 1, 1]
        state_feature = (S_for_state_feat / s_norm_state).mean(dim=(-1, -2)).mean(dim=1, keepdim=True) # [B, 1]

        # Project this feature through SpikingLinear to get modulation signals
        state_control_modulation_analog, s_state_control = self.state_control_proj(state_feature) # [B, num_heads*3]
        all_spikes['state_control_mod'] = s_state_control

        # Expand state_control_modulation to match the sequence length
        # It acts as a global modulation for the current chunk based on previous state
        state_control_modulation_expanded = state_control_modulation_analog.unsqueeze(1).expand(-1, x.shape[1], -1) # [B, L, num_heads*3]

        # Combine base and state-modulated signals
        combined_control_signals = control_signals_base + state_control_modulation_expanded

        # Split the consolidated control signals into beta, w (for decay), and alpha.
        control_signals_reshaped = rearrange(combined_control_signals, 'b l (h n_params) -> b l h n_params', h=self.num_heads, n_params=3)
        beta_raw, w_raw, alpha_raw = control_signals_reshaped.chunk(3, dim=-1)

        # Squeeze the last dimension which is 1 after chunking, to get (b, l, h) for each control signal.
        beta_raw = beta_raw.squeeze(-1)
        w_raw = w_raw.squeeze(-1)
        alpha_raw = alpha_raw.squeeze(-1)


        if hasattr(self, 'q_conv1d'):
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

        # Activation function stability handling for control parameters
        beta = torch.sigmoid(beta_raw) * 0.5 # Limit step size to [0, 0.5]
        alpha = torch.sigmoid(alpha_raw) # Limit to [0, 1]
        decay = F.logsigmoid(w_raw) # Log space decay, always negative

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        beta = rearrange(beta, 'b l h -> b h l')
        decay = rearrange(decay, 'b l h -> b h l')
        alpha = rearrange(alpha, 'b l h -> b h l')

        # Handle recurrent state for KDA
        layer_state = None
        if past_key_values is not None and self.layer_idx is not None:
             if len(past_key_values) > self.layer_idx:
                 state_dict = past_key_values[self.layer_idx]
                 if isinstance(state_dict, dict) and 'recurrent_state' in state_dict:
                    layer_state = state_dict['recurrent_state']

        o, final_state = kda_chunkwise_parallel(q, k, v, beta, decay, alpha, initial_state=layer_state)

        # Update cache with KDA's final state
        if past_key_values is not None and self.layer_idx is not None:
            if len(past_key_values) <= self.layer_idx:
                past_key_values.append({'recurrent_state': final_state}) # Add if not present
            else:
                past_key_values[self.layer_idx]['recurrent_state'] = final_state # Update existing


        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o) # RMSNorm on the combined head dimension
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, final_state, all_spikes # Changed return signature to match FLABlock's needs

# [SpikingMLP and SpikingFLABlock]

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        # Ensure SpikingLinear is used for all linear layers
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x)
        x = F.gelu(x)
        # x after dropout is analog input to fc2's neuron, output of fc2 is analog
        x, spike2 = self.fc2(self.dropout(x))
        return x, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Initialize SpikingKDA, passing neuron_type and all kwargs
        self.attn = SpikingKDA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        # Always use SpikingMLP for full SNN integration
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_values: Optional[list]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []
        x_res1 = x
        if self.prenorm: 
            x_norm1 = self.norm1(x)
        else:
            x_norm1 = x 

        # Handle recurrent state for KDA (passed as initial_state to attn)
        layer_state = None
        if past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 state_dict = past_key_values[layer_idx]
                 if isinstance(state_dict, dict) and 'recurrent_state' in state_dict:
                    layer_state = state_dict['recurrent_state']

        # SpikingKDA returns output, final_state, and a dictionary of spikes
        x_attn_output_continuous, final_attn_state, attn_spikes_dict = self.attn(x_norm1, initial_state=layer_state, **kwargs)
        if attn_spikes_dict: 
            # Collect all spike tensors from the dictionary
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None: # Ensure spike_tensor is not None before adding
                    all_spikes_list.append(spike_tensor)

        # Update cache with KDA's final state
        if past_key_values is not None and layer_idx is not None and use_cache:
            if len(past_key_values) <= layer_idx:
                # Use a dictionary to store the state, consistent with FLA's Cache structure
                past_key_values.append({'recurrent_state': final_attn_state})
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_attn_state

        # Residual connection after attention block
        x_after_attn = x_res1 + self.dropout1(x_attn_output_continuous)

        x_res2 = x_after_attn
        if self.prenorm: 
            x_norm2 = self.norm2(x_after_attn)
        else:
            x_norm2 = x_after_attn

        # SpikingMLP returns output, and two spike tensors
        x_mlp_output_continuous, mlp_spike1, mlp_spike2 = self.mlp(x_norm2)
        all_spikes_list.extend([mlp_spike1, mlp_spike2]) # Add both spike tensors from MLP

        # Residual connection after MLP block
        x_final_output = x_res2 + self.dropout2(x_mlp_output_continuous)

        # Concatenate all collected spikes into a single flattened tensor
        valid_spikes = [s for s in all_spikes_list if s is not None]
        final_spikes = torch.cat([s.flatten() for s in valid_spikes]) if valid_spikes else None

        return x_final_output, None, final_spikes