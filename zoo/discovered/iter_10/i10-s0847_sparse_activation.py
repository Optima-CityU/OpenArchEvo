import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

# ==================== SNN Related Imports and Helper Classes ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    # Placeholder for environments where SNN modules are not available
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
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D) # Binary spikes
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

"""==================== White-box Operator (Mamba2 SSD) ===================="""

def mamba2_ssd_parallel_stable(q, k, v, a, chunk_size=64, initial_state=None):
    b, h, l, d = q.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v]]
        a = F.pad(a, (0, pad_len), value=0.0) # padding with 0.0 for 'a' means no decay for padded elements
    
    q, k, v = [rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) for x in [q, k, v]]
    a_chunk = rearrange(a, 'b h (n c) -> b h n c', c=chunk_size)
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d, d, device=q.device)
    o = torch.zeros_like(v)
    
    # CRITICAL: Keeping Version 2's mask_tril with diagonal=-1 for strict causality in intra-chunk attention
    # The 'D' term in SpikingMamba2 handles the direct self-connection for 'v'.
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, a_i = q[:, :, i], k[:, :, i], v[:, :, i], a_chunk[:, :, i]
        
        # 1. Intra-chunk decay
        a_i_cumsum = torch.cumsum(a_i, dim=-1) # [B, H, C]
        
        # rel_cumsum[t_end, t_start] = sum_{k=t_start+1}^{t_end} a_i[k]
        rel_cumsum = a_i_cumsum.unsqueeze(-1) - a_i_cumsum.unsqueeze(-2) # [B, H, C, C]

        # Clamp max=0 and then apply mask_tril for causal decay
        decay_rel = torch.exp(rel_cumsum.clamp(max=0.0)) * mask_tril # [B, H, C, C]
        
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_rel
        o_intra = attn_intra @ v_i
        
        # 2. Inter-chunk reading
        # Scaling `q_i[c]` by `exp(sum_{s=0}^c a_i[s])`
        # Added numerical stability (1e-8) from Version 2
        decay_from_start = torch.exp(a_i_cumsum).unsqueeze(-1) # [B, H, C, 1]
        o_inter = (q_i * (decay_from_start + 1e-8)) @ S
        
        o[:, :, i] = o_inter + o_intra
        
        # 3. State update
        # `decay_total` is `exp(sum_{s=0}^{C-1} a_s)`
        # Using a_i.sum for decay_total from Version 2
        decay_total = torch.exp(a_i.sum(dim=-1, keepdim=True)).unsqueeze(-1) # [B, H, 1, 1]
        
        # `decay_to_end` is `exp(sum_{s=c+1}^{C-1} a_s)`
        decay_to_end = torch.exp(a_i_cumsum[..., -1].unsqueeze(-1) - a_i_cumsum).unsqueeze(-1) # [B, H, C, 1]
        
        delta_S = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * (decay_total + 1e-8) + delta_S
        
        # Dynamic range protection
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l, :], S) if pad_len > 0 else (o, S)

"""==================== SpikingMamba2 Layer ===================="""

class SpikingMamba2(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand: int=2, d_state: int=64, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_inner = int(expand * d_model)
        self.num_heads = num_heads 
        self.d_head = self.d_inner // num_heads

        # The output dimension for in_proj needs to be (z + x_in + B + C) + dt.
        self.proj_out_dim = self.d_inner * 4 + num_heads
        surrogate = piecewise_quadratic_surrogate()

        self.in_proj = SpikingLinear(nn.Linear(d_model, self.proj_out_dim, bias=False), neuron_type, surrogate)
        self.conv1d = ShortConvolution(self.d_inner, kernel_size=4, activation='silu')

        self.dt_bias = nn.Parameter(torch.ones(num_heads) * -2.0) # Initialized negative for long memory
        self.D = nn.Parameter(torch.ones(num_heads))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = SpikingLinear(nn.Linear(self.d_inner, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}
        projected, s_in = self.in_proj(x)
        all_spikes['in_proj'] = s_in

        # Split into z, x_in, B_param (k), C_param (q), dt_param (dt)
        z, x_in, B_param, C_param, dt_param = torch.split(projected, [self.d_inner, self.d_inner, self.d_inner, self.d_inner, self.num_heads], dim=-1)

        conv_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
             conv_state = past_key_values[self.layer_idx].get('conv_state')
        x_in, new_conv_state = self.conv1d(x_in, cache=conv_state, output_final_state=True)

        v = rearrange(x_in, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(B_param, 'b l (h d) -> b h l d', h=self.num_heads)
        q = rearrange(C_param, 'b l (h d) -> b h l d', h=self.num_heads)

        # Aligned with Mamba2: a = -softplus(dt + bias) 
        a = -F.softplus(dt_param + self.dt_bias) 
        a = rearrange(a, 'b l h -> b h l')

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
            last_state = past_key_values[self.layer_idx].get('recurrent_state')

        y, final_state = mamba2_ssd_parallel_stable(q, k, v, a, initial_state=last_state)

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values[self.layer_idx] = {'recurrent_state': final_state, 'conv_state': new_conv_state}

        y = y + (self.D.view(1, -1, 1, 1) * v) 
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.norm(y)
        y = y * F.silu(z) # z is an analog output, F.silu is fine here as part of the gating

        y, s_out = self.out_proj(y)
        all_spikes['out_proj'] = s_out

        return y, None, all_spikes

"""[SpikingMLP]"""

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x_analog_out_1, spike1 = self.fc1(x)
        # CRITICAL: Removed F.gelu here to enhance spiking network nature,
        # relying on the non-linearity of the spiking neurons.
        x_analog_dropout = self.dropout(x_analog_out_1)
        x_analog_out_2, spike2 = self.fc2(x_analog_dropout)
        return x_analog_out_2, spike1, spike2

# New SpikingResidualGate module with contextual modulation

class SpikingResidualGate(nn.Module):
    def __init__(self, d_model: int, neuron_type: str, surrogate_function):
        super().__init__()
        self.intermediate_dim = d_model // 4 
        if self.intermediate_dim == 0: self.intermediate_dim = 1 

        self.proj_in = SpikingLinear(nn.Linear(d_model, self.intermediate_dim, bias=False), neuron_type, surrogate_function)
        self.proj_out = SpikingLinear(nn.Linear(self.intermediate_dim, 1, bias=False), neuron_type, surrogate_function)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_list = []

        x_cont_mid_analog, spikes1 = self.proj_in(hidden_states)
        spikes_list.append(spikes1)

        # CRITICAL: Removed F.gelu here to enhance spiking network nature,
        # relying on the non-linearity of the spiking neurons.
        mod_factor_raw, spikes2 = self.proj_out(x_cont_mid_analog)
        spikes_list.append(spikes2)

        modulation_factor = 1.0 + 2.0 * torch.sigmoid(mod_factor_raw + self.bias)

        return modulation_factor, spikes_list

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', spike_reg_weight: float=0.001, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingMamba2(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.attn_residual_gate = SpikingResidualGate(d_model, neuron_type, piecewise_quadratic_surrogate())

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)
        self.mlp_residual_gate = SpikingResidualGate(d_model, neuron_type, piecewise_quadratic_surrogate())

        # CRITICAL: Re-added spike regularization weight for controlling firing rates
        self.spike_reg_weight = spike_reg_weight

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []

        # --- Attention Block (SpikingMamba2) ---
        x_res = x
        x_for_attn = self.norm1(x) if self.prenorm else x

        x_attn, _, attn_spikes_dict = self.attn(x_for_attn, **kwargs)

        if attn_spikes_dict:
            for spike_tensor_name, spike_tensor in attn_spikes_dict.items():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        attn_modulation_factor, attn_gate_spikes = self.attn_residual_gate(x_for_attn)
        all_spikes_list.extend(attn_gate_spikes)
        x_attn = x_attn * attn_modulation_factor 

        x = x_res + self.dropout1(x_attn)

        # --- MLP Block ---
        x_res = x
        x_for_mlp = self.norm2(x) if self.prenorm else x

        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x_for_mlp)

        if mlp_spike1 is not None: all_spikes_list.append(mlp_spike1)
        if mlp_spike2 is not None: all_spikes_list.append(mlp_spike2)

        mlp_modulation_factor, mlp_gate_spikes = self.mlp_residual_gate(x_for_mlp)
        all_spikes_list.extend(mlp_gate_spikes)
        x_mlp = x_mlp * mlp_modulation_factor

        x = x_res + self.dropout2(x_mlp)

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else torch.empty(0, device=x.device, dtype=torch.bool)

        # CRITICAL: Calculate regularization loss based on spike activity
        reg_loss = None
        if self.spike_reg_weight > 0 and final_spikes.numel() > 0:
            # L1 regularization on the mean firing rate of all collected spikes
            reg_loss = self.spike_reg_weight * final_spikes.float().mean() 

        return x, reg_loss, final_spikes