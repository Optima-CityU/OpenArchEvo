import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

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
        spikes = self.neuron(x_transposed).transpose(-1, -2)
        return self.linear(spikes), spikes

"""==================== White-box Operator (Mamba2 SSD) ===================="""

def mamba2_ssd_parallel_stable(q, k, v, a, chunk_size=64, initial_state=None):
    b, h, l, d = q.shape
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v]]
        a = F.pad(a, (0, pad_len), value=0.0)
    
    q, k, v = [rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) for x in [q, k, v]]
    a_chunk = rearrange(a, 'b h (n c) -> b h n c', c=chunk_size)
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype) # Ensure dtype consistency
    o = torch.zeros_like(v)
    
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, a_i = q[:, :, i], k[:, :, i], v[:, :, i], a_chunk[:, :, i]
        
        # 1. Intra-chunk decay: a_i is already [B, H, C]
        a_i_cumsum = torch.cumsum(a_i, dim=-1)
        rel_cumsum = a_i_cumsum.unsqueeze(-1) - a_i_cumsum.unsqueeze(-2)
        
        # Numerical protection: clamp(max=0) completely prevents exp(inf)
        decay_rel = torch.exp(rel_cumsum.clamp(max=0.0)) * mask_tril # [B, H, C, C]
        
        # [Fix] Directly utilize existing B, H dimensions for broadcasting
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_rel
        o_intra = attn_intra @ v_i
        
        # 2. Inter-chunk reading
        # Add minimal value protection to prevent gradient vanishing at exp(very small negative)
        decay_from_start = torch.exp(a_i_cumsum).unsqueeze(-1) # [B, H, C, 1]
        o_inter = (q_i * (decay_from_start + 1e-8)) @ S
        
        o[:, :, i] = o_inter + o_intra
        
        # 3. State update
        decay_total = torch.exp(a_i.sum(dim=-1, keepdim=True)).unsqueeze(-1) # [B, H, 1, 1]
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

        self.proj_out_dim = self.d_inner * 4 + num_heads # Corrected: B, C, D, dt are d_inner, q is d_inner // 4, k is d_inner // 4, v is d_inner // 2 for Mamba-1-like. For Mamba2 it is (d_inner, d_inner, d_inner, d_inner, num_heads) for (Z, X, B, C, dt)
        # However, the split is `z, x_in, B, C, dt`, where x_in becomes v. B, C are q, k.
        # So we need 3 * d_inner for z, x_in, B, C, and num_heads for dt.
        # Looking at original Mamba2 it's 2*D + D_proj, for our case it's 2*d_inner (x, z) and then 2*d_inner + num_heads for (B, C, dt)
        # Total: 4 * d_inner + num_heads. This matches the existing code.

        surrogate = piecewise_quadratic_surrogate()

        self.in_proj = SpikingLinear(nn.Linear(d_model, self.proj_out_dim, bias=False), neuron_type, surrogate)
        self.conv1d = ShortConvolution(self.d_inner, kernel_size=4, activation='silu')

        # [NASWOT Fix]: Initialize dt_bias as negative to ensure initial system has long memory (exp(a) close to 1)
        self.dt_bias = nn.Parameter(torch.ones(num_heads) * -2.0)
        self.D = nn.Parameter(torch.ones(num_heads))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = SpikingLinear(nn.Linear(self.d_inner, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, use_cache=False, **kwargs):
        all_spikes = {}
        projected, s_in = self.in_proj(x)
        all_spikes['in_proj'] = s_in

        # Ensure split aligns with proj_out_dim
        # z (d_inner), x_in (d_inner), B (d_inner), C (d_inner), dt (num_heads)
        z, x_in, B, C, dt = torch.split(projected, [self.d_inner, self.d_inner, self.d_inner, self.d_inner, self.num_heads], dim=-1)

        conv_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
             conv_state = past_key_values[self.layer_idx].get('conv_state')

        x_in, new_conv_state = self.conv1d(x_in, cache=conv_state, output_final_state=use_cache) # Pass use_cache to conv1d

        v = rearrange(x_in, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(B, 'b l (h d) -> b h l d', h=self.num_heads)
        q = rearrange(C, 'b l (h d) -> b h l d', h=self.num_heads)

        # Aligned with Mamba2: a = -softplus(dt + bias)
        a = -F.softplus(dt + self.dt_bias)
        a = rearrange(a, 'b l h -> b h l')

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx].get('recurrent_state')

        # Ensure q, k, v, a have the same dtype for mamba2_ssd_parallel_stable
        # a is already float. q,k,v should come from projected, which is from x.
        # x is assumed to be float.
        # S should be initialized with the same dtype as q.
        y, final_state = mamba2_ssd_parallel_stable(q, k, v, a, initial_state=last_state)

        if past_key_values is not None and self.layer_idx is not None:
            if self.layer_idx not in past_key_values:
                past_key_values[self.layer_idx] = {} # Initialize if not present
            past_key_values[self.layer_idx]['recurrent_state'] = final_state
            past_key_values[self.layer_idx]['conv_state'] = new_conv_state

        y = y + (self.D.view(1, -1, 1, 1) * v) # self.D needs to be reshaped for broadcasting
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.norm(y)
        y = y * F.silu(z) 

        y, s_out = self.out_proj(y)
        all_spikes['out_proj'] = s_out

        return y, None, all_spikes # attn_reg_loss is None

"""[SpikingMLP]"""

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        # Check if neuron_registry or surrogate_function is None, raise error
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available in SpikingMLP.")

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x_gelu, spike1 = self.fc1(x) # Output of fc1 is analog, spikes are binary
        x_dropout = self.dropout(F.gelu(x_gelu)) # Apply GELU to analog output then dropout
        x_final, spike2 = self.fc2(x_dropout) # Input to fc2 is analog
        return x_final, spike1, spike2

# New SpikingResidualGate module

class SpikingResidualGate(nn.Module):
    def __init__(self, d_model: int, neuron_type: str, surrogate_function):
        super().__init__()
        # Check if neuron_registry or surrogate_function is None, raise error
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available in SpikingResidualGate.")

        # Input is (B, L, 1), so the linear layer maps 1 -> d_model // 2
        self.proj_in = SpikingLinear(nn.Linear(1, d_model // 2, bias=False), neuron_type, surrogate_function)
        self.proj_out = SpikingLinear(nn.Linear(d_model // 2, 1, bias=False), neuron_type, surrogate_function)
        # Learnable bias to control the base level of modulation
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, avg_activity: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_list = []

        # First SpikingLinear layer: Input avg_activity is float (B, L, 1)
        x_cont, spikes1 = self.proj_in(avg_activity) # x_cont is analog (B, L, d_model // 2), spikes1 is binary (B, L, d_model // 2)
        spikes_list.append(spikes1)

        # Apply non-linear activation to the continuous output
        x_act = F.gelu(x_cont) # x_act is analog (B, L, d_model // 2)

        # Second SpikingLinear layer: Input x_act is analog
        mod_factor_raw, spikes2 = self.proj_out(x_act) # mod_factor_raw is analog (B, L, 1), spikes2 is binary (B, L, 1)
        spikes_list.append(spikes2)

        # Scale and shift the output through a sigmoid to create a modulation factor
        # This ensures the factor is positive and provides a learnable range for modulation.
        # The factor will be between 1.0 and 3.0 (1.0 + 2.0*0 to 1.0 + 2.0*1)
        modulation_factor = 1.0 + 2.0 * torch.sigmoid(mod_factor_raw + self.bias)

        return modulation_factor, spikes_list

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', spike_regularization_weight: float=0.01, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingMamba2(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # New SpikingResidualGate for attention block
        self.attn_residual_gate = SpikingResidualGate(d_model, neuron_type, piecewise_quadratic_surrogate())

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)
        # New SpikingResidualGate for MLP block
        self.mlp_residual_gate = SpikingResidualGate(d_model, neuron_type, piecewise_quadratic_surrogate())
        self.spike_regularization_weight = spike_regularization_weight # Store the weight

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        total_reg_loss = None

        # --- Attention Block (SpikingMamba2) ---
        x_res = x
        if self.prenorm: x = self.norm1(x)

        # SpikingMamba2 returns analog output, None for reg_loss, and a dictionary of collected spikes.
        x_attn, _, attn_spikes_dict = self.attn(x, **kwargs)

        attn_block_mean_spike_activities = []
        if attn_spikes_dict:
            for spike_tensor_name, spike_tensor in attn_spikes_dict.items():
                if spike_tensor is not None:
                    # Append raw binary spikes
                    all_spikes_list.append(spike_tensor)
                    # Calculate mean spike activity per token across feature dimension (B, L, 1) for the gate
                    attn_block_mean_spike_activities.append(spike_tensor.float().mean(dim=-1, keepdim=True))

        # Apply Spike-Activity Modulated Residual for Attention via SpikingResidualGate
        if attn_block_mean_spike_activities:
            # Average mean activities across different spiking layers within the attention block
            avg_attn_mean_activity = torch.mean(torch.stack(attn_block_mean_spike_activities), dim=0)
            # Get modulation factor and spikes from the new SpikingResidualGate
            attn_modulation_factor, attn_gate_spikes = self.attn_residual_gate(avg_attn_mean_activity)
            all_spikes_list.extend(attn_gate_spikes)
            x_attn = x_attn * attn_modulation_factor 

        x = x_res + self.dropout1(x_attn)

        # --- MLP Block ---
        x_res = x
        if self.prenorm: x = self.norm2(x)

        # SpikingMLP returns analog output and individual spike tensors for its two layers.
        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x)

        mlp_block_mean_spike_activities = []
        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
            mlp_block_mean_spike_activities.append(mlp_spike1.float().mean(dim=-1, keepdim=True))
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)
            mlp_block_mean_spike_activities.append(mlp_spike2.float().mean(dim=-1, keepdim=True))

        # Apply Spike-Activity Modulated Residual for MLP via SpikingResidualGate
        if mlp_block_mean_spike_activities:
            # Average mean activities across different spiking layers within the MLP block
            avg_mlp_mean_activity = torch.mean(torch.stack(mlp_block_mean_spike_activities), dim=0)
            # Get modulation factor and spikes from the new SpikingResidualGate
            mlp_modulation_factor, mlp_gate_spikes = self.mlp_residual_gate(avg_mlp_mean_activity)
            all_spikes_list.extend(mlp_gate_spikes)
            x_mlp = x_mlp * mlp_modulation_factor

        x = x_res + self.dropout2(x_mlp)

        # Final spike collection: flatten and concatenate all collected binary spikes
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else torch.empty(0, device=x.device, dtype=torch.bool)

        # Calculate and add firing rate regularization loss if weight > 0
        if self.spike_regularization_weight > 0 and final_spikes.numel() > 0:
            firing_rate_loss = final_spikes.float().mean() * self.spike_regularization_weight
            total_reg_loss = firing_rate_loss # Assign directly as per signature

        return x, total_reg_loss, final_spikes