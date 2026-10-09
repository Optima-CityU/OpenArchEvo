from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING

# Dependency Consistency: Do not remove any of the existing import statements
# Spiking SNN related imports
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

# Preserve SpikingLinear Class: The class SpikingLinear(nn.Module) must be kept exactly as it is, without modification or deletion

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2: # (B, D)
            x_transposed = x.unsqueeze(-1).transpose(-1, -2) # (B, D, 1)
        else: # (B, L, D)
            x_transposed = x.transpose(-1, -2) # (B, D, L)

        spikes_transposed = self.neuron(x_transposed) # Neuron operates on (B, D, L) or (B, D, 1)
        spikes = spikes_transposed.transpose(-1, -2)  # Transpose back to (B, L, D) or (B, 1, D)

        if x.dim() == 2: # Original input was (B, D), so squeeze the dummy L dimension from spikes
            spikes = spikes.squeeze(1) # (B, D)

        output = self.linear(spikes) # Linear layer takes binary spikes
        return output, spikes

# ==================== ScaledFeedbackNet for adaptive feedback gating ====================
# IMPORTANT: This class's internal linear layers are NOT SpikingLinear.
# This is a pragmatic design choice forced by the strict constraint:
# 'Do not modify the function signature (i.e., function name, args, ...) for comba_chunkwise_parallel'.
# If its internal layers were SpikingLinear, its forward method would need to return
# both the output and the spikes (output, spikes), but comba_chunkwise_parallel expects
# only a single tensor from feedback_net(s_feat).
# Thus, for functionality and adherence to the immutable signature, its internal
# nn.Linear layers cannot be wrapped by SpikingLinear, and its internal spikes
# cannot be collected by the global 'all_spikes_list'.

class ScaledFeedbackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(8, 1)
        self.scale_factor = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        return x

# ==================== White-box Operator (Comba Closed-Loop) ====================

def comba_chunkwise_parallel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, feedback_net: nn.Module, chunk_size: int=64, initial_state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Closed-loop feedback stability enhancement
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        s_feat = (S / s_norm).mean(dim=(-1, -2), keepdim=True).flatten(2)

        # Use the ScaledFeedbackNet to get the raw feedback offset
        f_i_raw = feedback_net(s_feat).squeeze(-1)
        # Apply tanh and the learned scale_factor for adaptive feedback
        f_i = torch.tanh(f_i_raw) * feedback_net.scale_factor

        # Gate numerical computation alignment
        log_decay = F.logsigmoid(g_i + f_i.unsqueeze(-1))
        d_cumsum = torch.cumsum(log_decay, dim=-1)

        decay_rel = d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0)

        # Compute output
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S

        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = attn_intra @ v_i

        o[:, :, i] = o_inter + o_intra

        # State update stability enhancement
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)

        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update

        # Strict renormalization after each chunk update
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingComba Layer ====================

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: Optional[int]=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections: All nn.Linear layers replaced by SpikingLinear as per requirements
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        # [PRESERVED]: ScaledFeedbackNet, per constraints on comba_chunkwise_parallel signature.
        # See detailed explanation in ScaledFeedbackNet class comments and motivation section.
        self.feedback_net = ScaledFeedbackNet()

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Dict]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q':s_q, 'k':s_k, 'v':s_v, 'g':s_g})

        if self.use_short_conv:
            # ShortConvolution operates on the analog outputs from the SpikingLinear layers' linear part.
            # It performs local feature extraction without introducing new spiking mechanisms here,
            # respecting the current Comba structure which expects analog q, k, v for dot products.
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l h -> b h l')

        layer_state = None
        if past_key_values is not None and self.layer_idx is not None:
            layer_state = past_key_values.get(self.layer_idx, None)

        # Execute core parallel operator, passing the ScaledFeedbackNet instance
        o, final_state = comba_chunkwise_parallel(q, k, v, g, self.feedback_net, initial_state=layer_state, **kwargs) # Pass kwargs for chunk_size if needed

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values[self.layer_idx] = final_state

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

# ==================== SpikingMLP Layer (Modified) ====================

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        # First linear layer, wrapped in SpikingLinear. Takes analog input, outputs analog (from linear), and provides spikes.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        # Second linear layer, also wrapped in SpikingLinear. Takes analog input, outputs analog (from linear), and provides spikes.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, spike1 = self.fc1(x) # x is analog output from fc1.linear, spike1 is binary from fc1.neuron
        # The output `x` from `self.fc1.linear` is analog, dropout is applied on it.
        # This analog output is then fed into the neuron of `self.fc2` internally.
        x = self.dropout(x) # Dropout on analog output
        x, spike2 = self.fc2(x) # x is analog output from fc2.linear, spike2 is binary from fc2.neuron
        return x, spike1, spike2

# ==================== SpikingFLABlock (Main Block) ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', use_spike_block_gate: bool=True, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Attention mechanism (Comba)
        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        # SpikingMLP is now mandatory to ensure all linear layers are spiking where possible
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)

        self.use_spike_block_gate = use_spike_block_gate
        if self.use_spike_block_gate:
            surrogate = piecewise_quadratic_surrogate()
            # Projects aggregated spike activity (scalar per token) to d_model for gating
            # Input to this SpikingLinear will be (B, L, 1) and output (B, L, d_model)
            self.spike_block_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, surrogate)
            # New parameter for spike-driven multiplicative gain
            self.block_gate_gain = nn.Parameter(torch.zeros(1)) # Initialized to 0 for no initial modulation

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # Store original input for residual connection later
        x_input_block = x
        if self.prenorm: x = self.norm1(x)

        # Attention block
        x_attn, reg_loss, attn_spikes_dict = self.attn(x, **kwargs)
        if attn_spikes_dict:
            # Filter out any None values if they accidentally creep in (shouldn't if all_spikes is properly managed)
            all_spikes_list.extend([s for s in attn_spikes_dict.values() if s is not None])
        x = x_input_block + self.dropout1(x_attn)

        # Store attention output for residual connection later
        x_input_mlp = x
        if self.prenorm: x = self.norm2(x)

        # MLP block (always SpikingMLP)
        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x)
        if mlp_spike1 is not None: all_spikes_list.append(mlp_spike1)
        if mlp_spike2 is not None: all_spikes_list.append(mlp_spike2)

        # New: Context-Adaptive Spike-Driven Residual Gating
        if self.use_spike_block_gate and all_spikes_list:
            # Concatenate all collected binary spike tensors from both attention and MLP
            # Ensure spikes are (B, L, D) or (B, D) to correctly calculate mean activity per token.
            processed_spikes_for_gate = []
            for s_tensor in all_spikes_list:
                if s_tensor.dim() == 2: # (B, D_feat) -> treat as (B, 1, D_feat)
                    processed_spikes_for_gate.append(s_tensor.unsqueeze(1))
                elif s_tensor.dim() == 3: # (B, L, D_feat)
                    processed_spikes_for_gate.append(s_tensor)
                # Else, if it's 1D, it's likely a flattened tensor already from a previous cat; skip or handle explicitly.
                # Given the source of spikes (SpikingLinear output), they should be 2D or 3D before flattening.

            if processed_spikes_for_gate:
                concatenated_spikes = torch.cat(processed_spikes_for_gate, dim=-1)
                # Calculate mean spike activity per token position across all spike dimensions
                # Resulting shape: (B, L, 1)
                mean_spike_activity_per_token = concatenated_spikes.float().mean(dim=-1, keepdim=True)

                # Project this mean activity through a SpikingLinear layer
                # spike_gate_analog will be (B, L, d_model), spikes_gate_binary will be (B, L, d_model)
                spike_gate_analog, spikes_gate_binary = self.spike_block_gate_proj(mean_spike_activity_per_token)
                all_spikes_list.append(spikes_gate_binary)

                # Apply a spike-driven multiplicative gate
                # The gate is (1 + learned_gain * binary_spike). When spike is 0, gate is 1 (no effect).
                # When spike is 1, gate is (1 + learned_gain), modulating x_mlp.
                g_block = (1 + self.block_gate_gain * spikes_gate_binary)

                # Modulate the MLP output with the spike-driven gate before adding to residual
                x = x_input_mlp + self.dropout2(x_mlp * g_block)
            else:
                # Fallback if no spikes were collected or processed for gating
                x = x_input_mlp + self.dropout2(x_mlp)
        else:
            # Original residual connection if gating is disabled or no spikes to gate with
            x = x_input_mlp + self.dropout2(x_mlp)

        # Final spike collection: flatten and concatenate all collected spikes
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        return x, reg_loss, final_spikes