import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, List

# Import from fla.modules
from fla.modules import FusedRMSNormGated, RMSNorm
from fla.modules.l2norm import l2norm
from fla.models.utils import Cache
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input # Added for DeltaNet compatibility

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
        spikes_transposed = self.neuron(x_transposed) # Generate spikes (binary signal)
        spikes = spikes_transposed.transpose(-1, -2) # (B, D, L) -> (B, L, D)
        output = self.linear(spikes) # Linear layer operates on binary spikes
        return output, spikes # output is continuous, spikes are binary

# ==================== White-box Operator (Delta-rule Chunkwise) ====================

def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    """Delta-rule Chunkwise parallel operator."""
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    
    padded_len = l + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        # Use non-inplace operations to ensure autograd safety
        attn_intra = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn_intra @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')

    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

# ==================== DeltaNet (Spiking Attention) ====================

class DeltaNet(nn.Module):
    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=False, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> 'DeltaNet':
        super().__init__()

        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")

        surrogate_function = piecewise_quadratic_surrogate()

        self.mode = mode
        self.qk_activation = qk_activation # Kept for signature compatibility, but functionally ignored for SNN purity
        self.qk_norm = qk_norm
        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv # Maintain the parameter, but ensure it's False if removed
        if self.use_short_conv:
             # ShortConvolution is removed for SNN purity as per analysis.
             # If `use_short_conv` is True, it will raise an error now, ensuring SNN purity.
            raise ValueError("ShortConvolution is not supported in this SNN-pure DeltaNet implementation.")

        self.conv_size = conv_size # Kept for signature compatibility
        self.conv_bias = conv_bias # Kept for signature compatibility
        self.allow_neg_eigval = allow_neg_eigval
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        # All linear layers are replaced with SpikingLinear, which internally applies a spiking neuron.
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), neuron_type, surrogate_function)

        # ShortConvolution layers are removed to enhance SNN purity.

        if use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        all_spikes: Dict[str, torch.Tensor] = {} # Used to collect all spikes from SpikingLinear layers

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "Expected attention_mask as a 0-1 matrix"

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        # Inputs to SpikingLinear are analog, outputs are analog (from linear layer) and binary spikes
        q, spikes_q_proj = self.q_proj(hidden_states)
        k, spikes_k_proj = self.k_proj(hidden_states)
        v, spikes_v_proj = self.v_proj(hidden_states)

        all_spikes['q_proj'] = spikes_q_proj
        all_spikes['k_proj'] = spikes_k_proj
        all_spikes['v_proj'] = spikes_v_proj

        # ShortConvolution processing and qk_activation functions are removed for SNN purity.

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.use_beta:
            beta_out_analog, spikes_beta = self.b_proj(hidden_states)
            beta = beta_out_analog.sigmoid()
            all_spikes['b_proj'] = spikes_beta
        else:
            beta = torch.ones_like(q[..., 0])

        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state.get('recurrent_state', None) if last_state is not None else None
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')

        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')

        if past_key_values is not None:
            cache_data = {
                'recurrent_state': recurrent_state,
            }
            if self.layer_idx is not None:
                past_key_values[self.layer_idx] = cache_data

        if self.use_gate:
            g_out_analog, spikes_g = self.g_proj(hidden_states)
            g = rearrange(g_out_analog, '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
            all_spikes['g_proj'] = spikes_g
        else:
            o = self.o_norm(o) # RMSNorm operates on analog values

        o = rearrange(o, 'b t h d -> b t (h d)')

        o_final_analog, spikes_o = self.o_proj(o) # Final output projection on analog values
        all_spikes['o_proj'] = spikes_o

        if attention_mask is not None:
            o_final_analog = pad_input(o_final_analog.squeeze(0), indices, batch_size, q_len)

        return o_final_analog, None, all_spikes

# ==================== SpikingMLP ====================

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        # fc1 is a SpikingLinear layer, which means it will apply a neuron internally
        # before its linear transformation, and return both its analog output and the spikes.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)

        # fc2 is also a SpikingLinear layer, taking the (analog) output from fc1 (after dropout)
        # and applying its own internal neuron before its linear transformation.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_analog_out, spike1 = self.fc1(x)
        x_after_dropout = self.dropout(x_fc1_analog_out)
        x_fc2_analog_out, spike2 = self.fc2(x_after_dropout)
        return x_fc2_analog_out, spike1, spike2

# ==================== SpikingFLABlock (Improved with Binary Gated Residual Modulation) ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_spiking_mlp: bool=True, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model) # RMSNorm operates on analog signals

        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        surrogate_function = piecewise_quadratic_surrogate() 

        # Spiking Gated Residual for Attention branch
        # This SpikingLinear layer will derive a gating factor from the residual path itself.
        # Its internal linear layer will operate on spikes generated from the analog residual input.
        self.attn_gate_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        self.norm2 = RMSNorm(d_model) # RMSNorm operates on analog signals
        d_ffn = d_model * d_ffn_mult

        if use_spiking_mlp:
            # SpikingMLP, which contains SpikingLinear layers internally.
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)
        else:
             raise ValueError("use_spiking_mlp must be True for SpikingFLABlock to ensure SNN nature.")

        self.dropout2 = nn.Dropout(dropout)

        # Spiking Gated Residual for MLP branch
        # Similar to the attention gate, this modulates the MLP branch's contribution.
        self.mlp_gate_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Cache]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # --- Attention Block ---
        x_res1 = x # Analog residual connection from block input
        if self.prenorm: 
            x_norm1 = self.norm1(x) # Normalization on analog signals
        else:
            x_norm1 = x 

        x_attn_output_continuous, reg_loss, attn_spikes_dict = self.attn(x_norm1, past_key_values=past_key_values, use_cache=use_cache, layer_idx=layer_idx, **kwargs)

        # Collect spikes from DeltaNet's internal SpikingLinear layers
        if attn_spikes_dict and isinstance(attn_spikes_dict, dict):
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        branch_output_attn = self.dropout1(x_attn_output_continuous)

        # --- Spiking Binary Gated Residual Modulation after Attention ---
        # The gating factor is derived from the *residual path* through a SpikingLinear.
        # The SpikingLinear returns both continuous output and binary spikes. 
        # We will use the *binary spikes* directly as the gating factor.
        gating_factor_cont_attn, s_gate_attn = self.attn_gate_proj(x_res1)
        all_spikes_list.append(s_gate_attn) # Collect spikes from the gate projection

        # Apply the binary spikes directly as the gating factor.
        # This ensures the gate is truly spike-driven and sparse (0 or 1).
        gating_factor_attn = s_gate_attn.float() # Ensure it's float for multiplication

        # Apply the Gated Residual mechanism: residual + (gating_factor * branch_output)
        # The attention branch's contribution is now scaled by binary spikes, promoting sparsity
        # and event-driven modulation of the residual pathway.
        x_after_attn_gated = x_res1 + (branch_output_attn * gating_factor_attn)

        # --- MLP Block ---
        x_res2 = x_after_attn_gated # Analog residual connection from previous stage output
        if self.prenorm: 
            x_norm2 = self.norm2(x_after_attn_gated) # Normalization on analog signals
        else:
            x_norm2 = x_after_attn_gated 

        x_mlp_output_continuous, s_mlp1, s_mlp2 = self.mlp(x_norm2)
        all_spikes_list.extend([s_mlp1, s_mlp2])

        branch_output_mlp = self.dropout2(x_mlp_output_continuous)

        # --- Spiking Binary Gated Residual Modulation after MLP ---
        # Similar binary gating for the MLP branch.
        gating_factor_cont_mlp, s_gate_mlp = self.mlp_gate_proj(x_res2)
        all_spikes_list.append(s_gate_mlp) # Collect spikes from the gate projection

        # Apply the binary spikes directly as the gating factor.
        gating_factor_mlp = s_gate_mlp.float() # Ensure it's float for multiplication

        # Apply the Gated Residual mechanism: residual + (gating_factor * branch_output)
        # The MLP branch's contribution is scaled by binary spikes, similar to the attention branch.
        x_final_output_gated = x_res2 + (branch_output_mlp * gating_factor_mlp)

        # Concatenate all collected spikes into a single flat tensor
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        return x_final_output_gated, reg_loss, final_spikes