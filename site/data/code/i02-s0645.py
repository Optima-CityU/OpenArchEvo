import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, List

from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from fla.models.utils import Cache

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

# ==================== White-box Operator (MesaNet / TTT Chunkwise) ====================

def mesa_chunkwise_parallel(q, k, v, eta, decay, chunk_size=64, initial_state=None):
    """MesaNet / TTT Chunkwise parallel operator.
    Converts online gradient descent to intra-chunk parallel matrix operations.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        eta = F.pad(eta, (0, pad_len), value=0.0)
        decay = F.pad(decay, (0, pad_len), value=1.0)
    
    # MesaNet core: K must be L2Norm to stabilize TTT gradient step size
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    
    q, k, v, eta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, eta, decay]
    )
    
    num_chunks = q.shape[2]
    # S shape: [B, H, Dk, Dv]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    for i in range(num_chunks):
        q_i, k_i, v_i, e_i, d_i = q[:, :, i], k[:, :, i], v[:, :, i], eta[:, :, i], decay[:, :, i]
        
        # [Causal Fix Step 1] Base residual: use pre-chunk state S
        v_residual = v_i - (k_i @ S)
        
        # [Causal Fix Step 2] Intra-chunk parallel gradient accumulation
        inner_interaction = (q_i @ k_i.transpose(-1, -2)) * e_i.unsqueeze(-1)
        mask_strict = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device), diagonal=-1)
        inner_interaction = inner_interaction * mask_strict
        
        o_intra = inner_interaction @ v_residual
        
        # [Causal Fix Step 3] Output calculation
        o[:, :, i] = q_i @ S + o_intra
        
        # [Causal Fix Step 4] State update (cross-chunk propagation)
        d_block = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1)
        delta_S = k_i.transpose(-1, -2) @ (v_residual * e_i.unsqueeze(-1))
        S = d_block * S + delta_S
        
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingMesaNet Layer ====================

class SpikingMesaNet(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
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
        self.eta_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Dynamic learning rate
        self.decay_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Dynamic regularization term

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

    def forward(self, x, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs):
        all_spikes = {}

        q_cont, s_q = self.q_proj(x)
        k_cont, s_k = self.k_proj(x)
        v_cont, s_v = self.v_proj(x)
        eta_cont, s_e = self.eta_proj(x)
        decay_cont, s_d = self.decay_proj(x)
        all_spikes.update({'q':s_q, 'k':s_k, 'v':s_v, 'eta':s_e, 'decay':s_d})

        q, _ = self.q_conv1d(q_cont)
        k, _ = self.k_conv1d(k_cont)
        v, _ = self.v_conv1d(v_cont)

        # Activation function alignment (optimized for TTT)
        # eta typically maps to a small positive range, simulating conservative gradient updates
        eta = torch.sigmoid(eta_cont) * 0.1 
        decay = torch.sigmoid(decay_cont) # Regularization coefficient (0, 1)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        eta = rearrange(eta, 'b l h -> b h l')
        decay = rearrange(decay, 'b l h -> b h l')

        layer_state = None
        # Handle Cache (list of dicts) for compatibility with SpikingFLABlock
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                state_dict = past_key_values[layer_idx]
                if isinstance(state_dict, dict) and 'recurrent_state' in state_dict:
                    layer_state = state_dict['recurrent_state']

        o_raw_attn_output, final_state = mesa_chunkwise_parallel(q, k, v, eta, decay, initial_state=layer_state)

        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                past_key_values.append({'recurrent_state': final_state}) # Add if not present
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_state # Update existing

        o = rearrange(o_raw_attn_output, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o_cont, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o_cont, None, all_spikes

# SpikingMLP (modified to be more SNN-centric by replacing GELU with RMSNorm on continuous output)

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        # Apply RMSNorm to the continuous output for SNN-friendly normalization before the next spiking layer
        self.norm_mid = RMSNorm(d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_cont_1, spike1 = self.fc1(x)
        x_normed = self.norm_mid(x_cont_1)
        x_cont_2_in = self.dropout(x_normed)
        x_cont_2, spike2 = self.fc2(x_cont_2_in)
        return x_cont_2, spike1, spike2 # Return all spikes produced

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', use_gate_conv: bool=True, gate_conv_size: int=3, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        self.attn = SpikingMesaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        surrogate_function = piecewise_quadratic_surrogate() 

        self.use_gate_conv = use_gate_conv
        if self.use_gate_conv:
            # ShortConvolution for features derived from the residual path
            self.attn_res_conv = ShortConvolution(d_model, gate_conv_size, activation='silu')
            # ShortConvolution for features derived from the attention branch output
            self.attn_branch_conv = ShortConvolution(d_model, gate_conv_size, activation='silu')

            self.mlp_res_conv = ShortConvolution(d_model, gate_conv_size, activation='silu')
            self.mlp_branch_conv = ShortConvolution(d_model, gate_conv_size, activation='silu')
        else:
            self.attn_res_conv = nn.Identity()
            self.attn_branch_conv = nn.Identity()
            self.mlp_res_conv = nn.Identity()
            self.mlp_branch_conv = nn.Identity()

        # Simplified Gating: Instead of separate intermediate projections,
        # the forget/input gates directly process concatenated, convolved features.
        # This reduces the number of SpikingLinear layers for gating.

        # Learnable Spiking Forget and Input Gates for post-attention residual
        # Each takes [convolved_res_features, convolved_branch_features] => 2*d_model input
        self.attn_forget_gate_proj = SpikingLinear(nn.Linear(2 * d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.attn_input_gate_proj = SpikingLinear(nn.Linear(2 * d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)

        # Learnable Spiking Forget and Input Gates for post-MLP residual
        # Each takes [convolved_res_features, convolved_branch_features] => 2*d_model input
        self.mlp_forget_gate_proj = SpikingLinear(nn.Linear(2 * d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.mlp_input_gate_proj = SpikingLinear(nn.Linear(2 * d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []

        x_res1 = x
        if self.prenorm: 
            x_norm1 = self.norm1(x)
        else:
            x_norm1 = x 

        x_attn_output_continuous, _, attn_spikes_dict = self.attn(x_norm1, past_key_values=past_key_values, use_cache=use_cache, layer_idx=layer_idx, **kwargs)
        if attn_spikes_dict: 
            all_spikes_list.extend(attn_spikes_dict.values())

        # Residual connection after attention block
        branch_output_attn = self.dropout1(x_attn_output_continuous)

        # --- Spiking Gated Recurrent Fusion (SGRF) for Attention Residual ---
        # Apply convolutions to residual and branch paths
        conv_res_attn, _ = self.attn_res_conv(x_res1)
        conv_branch_attn, _ = self.attn_branch_conv(branch_output_attn)

        # Concatenate convolved features for gate inputs
        gate_input_attn = torch.cat([conv_res_attn, conv_branch_attn], dim=-1)

        # Forget Gate (processes combined features)
        forget_gate_cont_attn, s_forget_gate_attn = self.attn_forget_gate_proj(gate_input_attn)
        forget_factor_attn = torch.sigmoid(forget_gate_cont_attn)
        all_spikes_list.append(s_forget_gate_attn)

        # Input Gate (processes combined features)
        input_gate_cont_attn, s_input_gate_attn = self.attn_input_gate_proj(gate_input_attn)
        input_factor_attn = torch.sigmoid(input_gate_cont_attn)
        all_spikes_list.append(s_input_gate_attn)

        # Apply GRU-like update rule (using continuous gate factors)
        x_after_attn_spike_gated = x_res1 * forget_factor_attn + branch_output_attn * input_factor_attn
        # --- End SGRF for Attention Residual ---


        x_res2 = x_after_attn_spike_gated 
        if self.prenorm: 
            x_norm2 = self.norm2(x_after_attn_spike_gated)
        else:
            x_norm2 = x_after_attn_spike_gated 

        # MLP returns 2 spike tensors
        x_mlp_output_continuous, s_mlp1, s_mlp2 = self.mlp(x_norm2)
        all_spikes_list.extend([s_mlp1, s_mlp2])

        # Residual connection after MLP block
        branch_output_mlp = self.dropout2(x_mlp_output_continuous)

        # --- Spiking Gated Recurrent Fusion (SGRF) for MLP Residual ---
        # Apply convolutions to residual and branch paths
        conv_res_mlp, _ = self.mlp_res_conv(x_res2)
        conv_branch_mlp, _ = self.mlp_branch_conv(branch_output_mlp)

        # Concatenate convolved features for gate inputs
        gate_input_mlp = torch.cat([conv_res_mlp, conv_branch_mlp], dim=-1)

        # Forget Gate (processes combined features)
        forget_gate_cont_mlp, s_forget_gate_mlp = self.mlp_forget_gate_proj(gate_input_mlp)
        forget_factor_mlp = torch.sigmoid(forget_gate_cont_mlp)
        all_spikes_list.append(s_forget_gate_mlp)

        # Input Gate (processes combined features)
        input_gate_cont_mlp, s_input_gate_mlp = self.mlp_input_gate_proj(gate_input_mlp)
        input_factor_mlp = torch.sigmoid(input_gate_cont_mlp)
        all_spikes_list.append(s_input_gate_mlp)

        # Apply GRU-like update rule (using continuous gate factors)
        x_final_output_gated = x_res2 * forget_factor_mlp + branch_output_mlp * input_factor_mlp
        # --- End SGRF for MLP Residual ---

        # Concatenate all collected spikes into a single flat tensor
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        return x_final_output_gated, None, final_spikes