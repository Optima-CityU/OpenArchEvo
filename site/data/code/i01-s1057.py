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
        x_transposed = x.transpose(-1, -2)
        spikes = self.neuron(x_transposed).transpose(-1, -2)
        output = self.linear(spikes)
        return output, spikes

# ==================== White-box Operator (KDA Chunkwise) ====================

def kda_chunkwise_parallel(q, k, v, beta, decay, alpha, chunk_size=32, initial_state=None):
    """KDA Chunkwise parallel operator.
    Core logic: S_t = decay * S_{t-1} + beta * (alpha * v_t - S_{t-1} @ k_t) @ k_t^T
    Modified to use dimensional `beta` and `alpha` gates.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        # beta and alpha need to be padded along the sequence dimension as well
        # Decay is per head, per sequence position (b, h, l), so pad it similarly
        beta = F.pad(beta, (0, 0, 0, pad_len), value=0.0)
        decay = F.pad(decay, (0, pad_len), value=0.0) 
        alpha = F.pad(alpha, (0, 0, 0, pad_len), value=0.0)
        
    # 1. Robust normalization
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-5)

    # Reshape to Chunks
    q, k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v])
    # Beta and Alpha are now dimensional, so they also need `d` in rearrange
    # Decay is (b, h, l), so rearrange to (b, h, n, c)
    beta, decay, alpha = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [beta, decay, alpha]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)

    # Pre-generate Bool type mask to prevent errors
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        b_i, d_i, a_i = beta[:, :, i], decay[:, :, i], alpha[:, :, i]
        
        # Apply dimensional alpha gate to v_i
        v_i = v_i * a_i # Alpha gate is now per-dimension

        # [Step 1] Compute decay matrix (Log-space)
        d_cumsum = torch.cumsum(d_i, dim=-1) 
        decay_rel = (d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)) 
        decay_mask = torch.exp(decay_rel).masked_fill(mask_triu, 0) # Keep only causal part
        
        # [Step 2] Compute prediction error (Delta Rule + Intra-chunk Wy approximation)
        # v_pred_inter is the prediction from historical state S
        v_pred_inter = k_i @ S 
        # Apply dimensional beta gate to v_error
        v_error = (v_i - v_pred_inter) * b_i # Beta gate is now per-dimension
        
        # Intra-chunk parallel compensation: handle the effect of previous k in this chunk on current update
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
        d_mean = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1) # Still uses scalar mean for decay
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
        self.head_v_dim = self.value_dim // num_heads # Dimension per head for value/gates
        self.head_k_dim = self.key_dim // num_heads # Dimension per head for key

        surrogate = piecewise_quadratic_surrogate()

        # Projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        # Beta and Alpha are now dimensional, matching value_dim per head
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads * self.head_v_dim, bias=False), neuron_type, surrogate) 
        self.w_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) 
        self.a_proj = SpikingLinear(nn.Linear(d_model, num_heads * self.head_v_dim, bias=False), neuron_type, surrogate) 

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

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)

        beta_raw, s_b = self.b_proj(x) 
        w_raw, s_w = self.w_proj(x) 
        alpha_raw, s_a = self.a_proj(x) 
        all_spikes.update({'q':s_q, 'k':s_k, 'v':s_v, 'beta':s_b, 'w':s_w, 'alpha':s_a})

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if use_cache and past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                states = past_key_values[layer_idx]
                conv_state_q = states.get('kda_conv_state_q')
                conv_state_k = states.get('kda_conv_state_k')
                conv_state_v = states.get('kda_conv_state_v')

        if hasattr(self, 'q_conv1d') and not isinstance(self.q_conv1d, nn.Identity):
            q_conv_out, conv_state_q = self.q_conv1d(q, cache=conv_state_q, output_final_state=True)
            k_conv_out, conv_state_k = self.k_conv1d(k, cache=conv_state_k, output_final_state=True)
            v_conv_out, conv_state_v = self.v_conv1d(v, cache=conv_state_v, output_final_state=True)
        else:
            q_conv_out, k_conv_out, v_conv_out = q, k, v

        beta = rearrange(beta_raw, 'b l (h d) -> b h l d', h=self.num_heads)
        alpha = rearrange(alpha_raw, 'b l (h d) -> b h l d', h=self.num_heads)

        beta = torch.sigmoid(beta) * 0.5 
        alpha = torch.sigmoid(alpha) 
        decay = F.logsigmoid(w_raw) 

        q = rearrange(q_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        decay = rearrange(decay, 'b l h -> b h l') 

        layer_state = None
        if use_cache and past_key_values is not None and layer_idx is not None:
             if len(past_key_values) > layer_idx:
                 state_dict = past_key_values[layer_idx]
                 if isinstance(state_dict, dict) and 'recurrent_state' in state_dict:
                    layer_state = state_dict['recurrent_state']

        o, final_recurrent_state = kda_chunkwise_parallel(q, k, v, beta, decay, alpha, initial_state=layer_state)

        if use_cache and past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                past_key_values.append({'recurrent_state': final_recurrent_state})
            else:
                past_key_values[layer_idx]['recurrent_state'] = final_recurrent_state

            # Update conv states if short conv is used
            if hasattr(self, 'q_conv1d') and not isinstance(self.q_conv1d, nn.Identity):
                past_key_values[layer_idx]['kda_conv_state_q'] = conv_state_q
                past_key_values[layer_idx]['kda_conv_state_k'] = conv_state_k
                past_key_values[layer_idx]['kda_conv_state_v'] = conv_state_v

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

# Redesigned SpikingMLP with GLU-like gating, ensuring all linear layers are SpikingLinear

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        # Two parallel SpikingLinear layers for the GLU-like gating mechanism
        if surrogate_function is None:
            surrogate_function = piecewise_quadratic_surrogate()
        self.fc1_gate = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.fc1_up = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)

        # RMSNorm now applied after gating, before the final linear layer (reinstated from V1)
        self.norm_gated = RMSNorm(d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_spikes_mlp = []

        # Project input to two branches using spiking layers
        x_gate_cont, spike_gate = self.fc1_gate(x)
        x_up_cont, spike_up = self.fc1_up(x)
        all_spikes_mlp.extend([spike_gate, spike_up])

        # Gating mechanism: Apply sigmoid to the gate path's continuous output
        # and element-wise multiply with the 'up' path's continuous output.
        gated_output = x_up_cont * torch.sigmoid(x_gate_cont)

        # Normalize the gated continuous signal before dropout and final projection
        normed_gated_output = self.norm_gated(gated_output)

        # Apply dropout to the normalized gated output
        dropped_output = self.dropout(normed_gated_output)

        # Final projection using a spiking linear layer
        final_mlp_out_cont, spike_fc2 = self.fc2(dropped_output)
        all_spikes_mlp.append(spike_fc2)

        return final_mlp_out_cont, all_spikes_mlp

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', conv_size: int=4, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        self.attn = SpikingKDA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, conv_size=conv_size, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.surrogate = piecewise_quadratic_surrogate()

        # Post-attention short convolution for the branch output
        self.post_attn_branch_conv = ShortConvolution(d_model, kernel_size=conv_size, activation='silu')

        # Short convolution for the gating context (combining residual + branch)
        self.attn_gating_context_conv = ShortConvolution(d_model, kernel_size=conv_size, activation='silu')

        # Dual Gating for Attention Residual
        self.attn_residual_forget_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, self.surrogate)
        self.attn_residual_add_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, self.surrogate)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=self.surrogate)
        self.dropout2 = nn.Dropout(dropout)

        # Post-MLP short convolution for the branch output
        self.post_mlp_branch_conv = ShortConvolution(d_model, kernel_size=conv_size, activation='silu')

        # Short convolution for the MLP gating context (combining residual + branch)
        self.mlp_gating_context_conv = ShortConvolution(d_model, kernel_size=conv_size, activation='silu')

        # Dual Gating for MLP Residual
        self.mlp_residual_forget_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, self.surrogate)
        self.mlp_residual_add_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, self.surrogate)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        reg_loss_attn = None

        current_layer_cache: Dict[str, torch.Tensor] = {}
        if use_cache and past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                current_layer_cache = past_key_values[layer_idx]
            else:
                # Grow cache for intermediate layers if needed
                for _ in range(layer_idx - len(past_key_values) + 1):
                    past_key_values.append({})
                current_layer_cache = past_key_values[layer_idx]

        # --- Attention Block ---
        x_res_attn = x

        if self.prenorm: 
            x_attn_input = self.norm1(x)
        else:
            x_attn_input = x 

        x_attn_output_continuous, reg_loss_attn, attn_spikes_dict = self.attn(
            x_attn_input, 
            past_key_values=past_key_values, 
            use_cache=use_cache, 
            layer_idx=layer_idx, 
            **kwargs
        )
        if attn_spikes_dict: 
            all_spikes_list.extend([s for s in attn_spikes_dict.values() if s is not None])

        # Apply post-attention branch convolution
        conv_state_post_attn_branch = current_layer_cache.get('conv_state_post_attn_branch')
        branch_output_attn_conv, new_conv_state_post_attn_branch = self.post_attn_branch_conv(
            x_attn_output_continuous, cache=conv_state_post_attn_branch, output_final_state=True
        )
        if use_cache and past_key_values is not None and layer_idx is not None:
            past_key_values[layer_idx]['conv_state_post_attn_branch'] = new_conv_state_post_attn_branch

        branch_output_attn = self.dropout1(branch_output_attn_conv)

        # Contextual Dual Gating for Attention Residual
        # Combine residual and branch output for gating context
        raw_gating_input_attn = x_res_attn + branch_output_attn

        # Apply short convolution for temporal context to the gating input
        conv_state_attn_gating_context = current_layer_cache.get('conv_state_attn_gating_context')
        gating_context_attn_conv, new_conv_state_attn_gating_context = self.attn_gating_context_conv(
            raw_gating_input_attn, cache=conv_state_attn_gating_context, output_final_state=True
        )
        if use_cache and past_key_values is not None and layer_idx is not None:
            past_key_values[layer_idx]['conv_state_attn_gating_context'] = new_conv_state_attn_gating_context

        # Generate forget and add factors from the convolved context using SpikingLinear
        forget_raw_attn, s_forget_attn = self.attn_residual_forget_proj(gating_context_attn_conv)
        forget_factor_attn = torch.sigmoid(forget_raw_attn)
        all_spikes_list.append(s_forget_attn)

        add_raw_attn, s_add_attn = self.attn_residual_add_proj(gating_context_attn_conv)
        add_factor_attn = torch.sigmoid(add_raw_attn)
        all_spikes_list.append(s_add_attn)

        # Apply dual gating to the residual connection
        x_after_attn_gated = x_res_attn * forget_factor_attn + branch_output_attn * add_factor_attn

        # --- MLP Block ---
        x_res_mlp = x_after_attn_gated 
        if self.prenorm: 
            x_mlp_input = self.norm2(x_after_attn_gated)
        else:
            x_mlp_input = x_after_attn_gated 

        x_mlp_output_continuous, mlp_all_spikes = self.mlp(x_mlp_input) # SpikingMLP returns (output, list_of_spikes)
        all_spikes_list.extend(mlp_all_spikes)

        # Apply post-MLP branch convolution
        conv_state_post_mlp_branch = current_layer_cache.get('conv_state_post_mlp_branch')
        branch_output_mlp_conv, new_conv_state_post_mlp_branch = self.post_mlp_branch_conv(
            x_mlp_output_continuous, cache=conv_state_post_mlp_branch, output_final_state=True
        )
        if use_cache and past_key_values is not None and layer_idx is not None:
            past_key_values[layer_idx]['conv_state_post_mlp_branch'] = new_conv_state_post_mlp_branch

        branch_output_mlp = self.dropout2(branch_output_mlp_conv)

        # Contextual Dual Gating for MLP Residual
        raw_gating_input_mlp = x_res_mlp + branch_output_mlp

        conv_state_mlp_gating_context = current_layer_cache.get('conv_state_mlp_gating_context')
        gating_context_mlp_conv, new_conv_state_mlp_gating_context = self.mlp_gating_context_conv(
            raw_gating_input_mlp, cache=conv_state_mlp_gating_context, output_final_state=True
        )
        if use_cache and past_key_values is not None and layer_idx is not None:
            past_key_values[layer_idx]['conv_state_mlp_gating_context'] = new_conv_state_mlp_gating_context

        forget_raw_mlp, s_forget_mlp = self.mlp_residual_forget_proj(gating_context_mlp_conv)
        forget_factor_mlp = torch.sigmoid(forget_raw_mlp)
        all_spikes_list.append(s_forget_mlp)

        add_raw_mlp, s_add_mlp = self.mlp_residual_add_proj(gating_context_mlp_conv)
        add_factor_mlp = torch.sigmoid(add_raw_mlp)
        all_spikes_list.append(s_add_mlp)

        x_final_output_gated = x_res_mlp * forget_factor_mlp + branch_output_mlp * add_factor_mlp 

        # Concatenate all collected spikes into a single flat tensor. Handle empty list gracefully.
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None]) if all_spikes_list else torch.empty(0, device=x.device)

        return x_final_output_gated, reg_loss_attn, final_spikes