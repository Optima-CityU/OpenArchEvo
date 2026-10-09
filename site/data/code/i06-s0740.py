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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L) for neuron processing
        spikes_transposed = self.neuron(x_transposed) # Neuron output is binary spikes
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D) # Binary spikes

        # The internal nn.Linear layer receives these binary spikes.
        output = self.linear(spikes) # output is analog (weights multiplied by binary spikes)
        return output, spikes # output is analog, spikes are binary

"""==================== White-box Operator (Comba Closed-Loop) ===================="""

def comba_chunkwise_parallel(q, k, v, g, feedback_net, chunk_size=64, initial_state=None):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len), value=0.0)
    
    q = l2norm(q)
    k = l2norm(k)
    
    q, k, v, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, g]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)

    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        
        # --- Closed-loop feedback stability enhancement ---
        # 1. Normalize state features to prevent FeedbackNet input explosion
        # Calculate L2 norm across the last two dimensions (d_k, d_v) for each head and batch item
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        # Average across the last two dimensions and flatten the feature into a single scalar per head per batch
        s_feat = (S / s_norm).mean(dim=(-1, -2), keepdim=True).flatten(2) # [B, H, 1]
        
        # 2. Limit feedback offset to [-2, 2] to prevent complete override of original gating
        f_i = feedback_net(s_feat).squeeze(-1) # [B, H]
        f_i = torch.tanh(f_i) * 2.0 
        
        # --- Gate numerical computation alignment ---
        # Use more stable logsigmoid
        log_decay = F.logsigmoid(g_i + f_i.unsqueeze(-1)) # [B, H, C]
        d_cumsum = torch.cumsum(log_decay, dim=-1)
        
        decay_rel = d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)
        # Ensure exp exponent is not greater than 0
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0)
        
        # 3. Compute output
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = attn_intra @ v_i
        
        o[:, :, i] = o_inter + o_intra
        
        # --- State update stability enhancement ---
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        # Limit update step size to prevent excessive single updates
        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update
        
        # Strict renormalization after each chunk update
        # This is critical in SNNs to prevent hidden state steady-state gain exceeding 1
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

"""==================== SpikingComba Layer ===================="""

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections using SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        # Feedback Network: Process state feedback (state features -> feedback bias)
        self.feedback_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(), # Analog non-linearity
            nn.Linear(8, 1)
        )

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
        else:
            self.q_conv1d = None
            self.k_conv1d = None
            self.v_conv1d = None

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes_dict = {} # Use dict for clarity during collection

        # All initial projections are now SpikingLinear, producing binary spikes as input to their internal nn.Linear
        q_analog, s_q = self.q_proj(x)
        k_analog, s_k = self.k_proj(x)
        v_analog, s_v = self.v_proj(x)
        g_analog, s_g = self.g_proj(x)
        all_spikes_dict.update({'q': s_q, 'k': s_k, 'v': s_v, 'g': s_g})

        # Short convolutions operate on the analog outputs of the spiking linear layers
        if self.q_conv1d:
            q_conv, _ = self.q_conv1d(q_analog)
            k_conv, _ = self.k_conv1d(k_analog)
            v_conv, _ = self.v_conv1d(v_analog)
        else:
            q_conv, k_conv, v_conv = q_analog, k_analog, v_analog

        q_re = rearrange(q_conv, 'b l (h d) -> b h l d', h=self.num_heads)
        k_re = rearrange(k_conv, 'b l (h d) -> b h l d', h=self.num_heads)
        v_re = rearrange(v_conv, 'b l (h d) -> b h l d', h=self.num_heads)
        g_re = rearrange(g_analog, 'b l h -> b h l') # g is already dimension h

        layer_state = None
        if past_key_values is not None and isinstance(past_key_values, dict) and self.layer_idx in past_key_values:
            layer_state = past_key_values[self.layer_idx]

        # Execute core parallel operator
        # The feedback_net here is the analog network within Comba which takes analog state features
        o_analog, final_state = comba_chunkwise_parallel(q_re, k_re, v_re, g_re, self.feedback_net, initial_state=layer_state)

        if past_key_values is not None:
            if not isinstance(past_key_values, dict):
                past_key_values = {} # Create an empty dict if it's not one, for robust write
            past_key_values[self.layer_idx] = final_state

        o_re = rearrange(o_analog, 'b h l d -> b l h d') # Revert head dimension for RMSNorm
        o_normed = self.o_norm(o_re)
        o_out = rearrange(o_normed, 'b l h d -> b l (h d)') # Flatten heads back

        # Final output projection is also SpikingLinear
        final_output, s_o = self.o_proj(o_out)
        all_spikes_dict['o'] = s_o

        # Comba does not produce a regularization loss by itself in its forward.
        reg_loss = None 
        return final_output, reg_loss, all_spikes_dict

"""[SpikingMLP]"""

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()

        # Two parallel SpikingLinear layers for a gated mechanism (e.g., SwiGLU-like)
        # Both take the original 'x' (analog) as input, produce spikes, and then analog output for activation.
        self.fc1_value = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.fc1_gate = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)

        # Analog activations applied to the analog outputs of the spiking linear layers
        self.activation_value = nn.GELU()
        self.activation_gate = nn.Sigmoid() # Sigmoid for gating factor between 0 and 1

        self.dropout = nn.Dropout(dropout)

        # Second SpikingLinear layer takes the gated analog signal as input
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_value_analog_out, spike_value = self.fc1_value(x)
        x_value_activated = self.activation_value(x_value_analog_out)

        # Process through gate path
        x_gate_analog_out, spike_gate = self.fc1_gate(x)
        x_gate_activated = self.activation_gate(x_gate_analog_out)

        # Apply spike-driven gating: element-wise product of value and gate paths
        x_gated_analog = x_value_activated * x_gate_activated

        # Apply dropout to the combined analog signal
        x_after_dropout_analog = self.dropout(x_gated_analog)

        # Final linear layer, which takes the analog gated signal as input,
        # produces spikes internally, and then analog output.
        x_final_analog_out, spike_final = self.fc2(x_after_dropout_analog)

        # Return final analog output and all collected binary spikes
        return x_final_analog_out, spike_value, spike_gate, spike_final

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        # Comba already has all its internal linear layers wrapped in SpikingLinear
        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        # Use the updated SpikingMLP which now has a gated structure
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        total_reg_loss = None # Initialize for potential regularization loss collection

        # --- Attention Block (Comba) ---
        x_res = x
        if self.prenorm: 
            x = self.norm1(x)

        # Comba returns analog output, optional regularization loss (currently None), and a dictionary of collected spikes.
        x_attn, reg_loss_attn, attn_spikes_dict = self.attn(x, **kwargs)

        if reg_loss_attn is not None:
            total_reg_loss = reg_loss_attn # Currently Comba does not return reg_loss, so this will be None

        # Collect spikes from the attention mechanism
        if attn_spikes_dict:
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        x = x_res + self.dropout1(x_attn)

        # --- MLP Block ---
        x_res = x
        if self.prenorm: 
            x = self.norm2(x)

        # SpikingMLP now returns its final analog output and individual spike tensors for its three SpikingLinear layers.
        x_mlp, mlp_spike_value, mlp_spike_gate, mlp_spike_final = self.mlp(x)

        # Collect spikes from the MLP
        if mlp_spike_value is not None:
            all_spikes_list.append(mlp_spike_value)
        if mlp_spike_gate is not None:
            all_spikes_list.append(mlp_spike_gate)
        if mlp_spike_final is not None:
            all_spikes_list.append(mlp_spike_final)

        x = x_res + self.dropout2(x_mlp)

        final_spikes = None
        if all_spikes_list:
            # Concatenate all collected binary spike tensors into a single flattened tensor as required.
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

        # The signature requires returning (x, Optional[torch.Tensor], Optional[torch.Tensor])
        # The second element is typically for regularization loss, which Comba currently does not output,
        # but the placeholder in SpikingFLABlock is `None` for now.
        return x, total_reg_loss, final_spikes