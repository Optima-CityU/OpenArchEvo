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
        # Pad eta and decay to match sequence length
        eta = F.pad(eta, (0, pad_len), value=0.0)
        decay = F.pad(decay, (0, pad_len), value=1.0) # Decay should typically be 1.0 when padding

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

    # Pre-allocate mask for efficiency
    mask_strict = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, e_i, d_i = q[:, :, i], k[:, :, i], v[:, :, i], eta[:, :, i], decay[:, :, i]

        # [Causal Fix Step 1] Base residual: use pre-chunk state S
        v_residual = v_i - (k_i @ S)

        # [Causal Fix Step 2] Intra-chunk parallel gradient accumulation
        inner_interaction = (q_i @ k_i.transpose(-1, -2)) * e_i.unsqueeze(-1)
        inner_interaction = inner_interaction.masked_fill(~mask_strict, 0.) # Apply strict lower triangular mask

        o_intra = inner_interaction @ v_residual

        # [Causal Fix Step 3] Output calculation
        o[:, :, i] = q_i @ S + o_intra

        # [Causal Fix Step 4] State update (cross-chunk propagation)
        d_block = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1) # Averaging decay across sequence elements within chunk
        delta_S = k_i.transpose(-1, -2) @ (v_residual * e_i.unsqueeze(-1))
        S = d_block * S + delta_S

        S = torch.clamp(S, -10.0, 10.0) # Clamp to prevent state explosion

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingMesaNet Layer ====================

class SpikingMesaNet(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads # Used for output normalization

        surrogate = piecewise_quadratic_surrogate()

        # All projections use SpikingLinear as per requirements
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

        # NEW: Spike-driven modulation for eta and decay based on aggregated spike activity
        # Input features: 3 (for aggregated Q, K, V spike counts per token)
        # Output features: num_heads (to modulate per-head eta/decay)
        self.spike_eta_mod_proj = SpikingLinear(nn.Linear(3, num_heads, bias=False), neuron_type, surrogate)
        self.spike_decay_mod_proj = SpikingLinear(nn.Linear(3, num_heads, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor, layer_cache: Optional[Dict[str, torch.Tensor]]=None, use_cache: Optional[bool]=False, **kwargs) -> Tuple[torch.Tensor, None, Dict[str, torch.Tensor]]:
        all_spikes = {}
        b, l, d_model = x.shape

        # Projections and spike generation for base signals
        q_cont, s_q = self.q_proj(x)
        k_cont, s_k = self.k_proj(x)
        v_cont, s_v = self.v_proj(x)
        eta_cont_base, s_e_base = self.eta_proj(x)
        decay_cont_base, s_d_base = self.decay_proj(x)
        all_spikes.update({'q':s_q, 'k':s_k, 'v':s_v, 'eta_base':s_e_base, 'decay_base':s_d_base})

        # Retrieve convolutional states and recurrent state from cache
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if layer_cache is not None:
            conv_state_q = layer_cache.get('conv_state_q')
            conv_state_k = layer_cache.get('conv_state_k')
            conv_state_v = layer_cache.get('conv_state_v')
            recurrent_state = layer_cache.get('recurrent_state')

        # Apply short convolutions to continuous outputs with state caching
        q, conv_state_q_new = self.q_conv1d(q_cont, cache=conv_state_q, output_final_state=True)
        k, conv_state_k_new = self.k_conv1d(k_cont, cache=conv_state_k, output_final_state=True)
        v, conv_state_v_new = self.v_conv1d(v_cont, cache=conv_state_v, output_final_state=True)

        # Update convolutional states in cache if using cache
        if use_cache and layer_cache is not None:
             layer_cache['conv_state_q'] = conv_state_q_new
             layer_cache['conv_state_k'] = conv_state_k_new
             layer_cache['conv_state_v'] = conv_state_v_new

        # --- Spike-Driven Modulation for Eta and Decay ---
        # Aggregate spike counts per token for Q, K, V
        # s_q, s_k, s_v are (B, L, feature_dim). Sum along feature dim to get (B, L, 1) per type
        s_q_sum_per_token = s_q.sum(dim=-1, keepdim=True).float()
        s_k_sum_per_token = s_k.sum(dim=-1, keepdim=True).float()
        s_v_sum_per_token = s_v.sum(dim=-1, keepdim=True).float()
        # Concatenate these into a single input for spike-driven projections
        spike_context_for_eta_decay = torch.cat([s_q_sum_per_token, s_k_sum_per_token, s_v_sum_per_token], dim=-1) # (B, L, 3)

        # Project spike context to modulation signals
        spike_eta_mod_cont, s_se_mod = self.spike_eta_mod_proj(spike_context_for_eta_decay) # (B, L, num_heads)
        spike_decay_mod_cont, s_sd_mod = self.spike_decay_mod_proj(spike_context_for_eta_decay) # (B, L, num_heads)
        all_spikes.update({'spike_eta_mod':s_se_mod, 'spike_decay_mod':s_sd_mod})

        # Combine base projections with spike-driven modulation
        # Use sigmoid to keep eta positive and limited, and for decay within (0,1)
        eta = torch.sigmoid(eta_cont_base + spike_eta_mod_cont) * 0.1 # Conservative learning rate
        decay = torch.sigmoid(decay_cont_base + spike_decay_mod_cont) # Regularization coefficient (0, 1)

        # Rearrange for multi-head processing
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        eta = rearrange(eta, 'b l h -> b h l')
        decay = rearrange(decay, 'b l h -> b h l')

        # Core recurrent operation
        o_raw_attn_output, final_recurrent_state = mesa_chunkwise_parallel(q, k, v, eta, decay, initial_state=recurrent_state)

        # Update recurrent state in cache if using cache
        if use_cache and layer_cache is not None:
            layer_cache['recurrent_state'] = final_recurrent_state

        # Post-processing for output
        o = rearrange(o_raw_attn_output, 'b h l d -> b l h d')
        o = self.o_norm(o) # Normalize per-head
        o = rearrange(o, 'b l h d -> b l (h d)')

        # Final output projection
        o_cont, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o_cont, None, all_spikes

# Redesigned SpikingMLP based on Version 2's successful structure, but strictly Spiking

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        # Ensure all internal linear layers are SpikingLinear
        _surrogate = surrogate_function if surrogate_function is not None else piecewise_quadratic_surrogate()

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn, bias=False), neuron_type, _surrogate)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model, bias=False), neuron_type, _surrogate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_cont_1, s_fc1 = self.fc1(x) # s_fc1 are spikes from the first layer

        # Apply GELU to the continuous output of the first spiking linear layer
        x_act_1 = F.gelu(x_cont_1) 

        # Apply dropout to the activated continuous signal
        x_drop_1 = self.dropout(x_act_1)

        # Second spiking linear layer operates on the continuous, dropped-out signal
        x_cont_2, s_fc2 = self.fc2(x_drop_1) # s_fc2 are spikes from the second layer

        return x_cont_2, s_fc1, s_fc2 # Return continuous output and individual spike tensors

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', conv_size: int=4, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        surrogate_function = piecewise_quadratic_surrogate() # Get surrogate once

        self.norm1 = RMSNorm(d_model)
        # Pass kwargs to SpikingMesaNet init for potential future compatibility
        self.attn = SpikingMesaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, conv_size=conv_size, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        # Always use SpikingMLP as per requirements "replacing all nn.Linear() by SpikingLinear()"
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.dropout2 = nn.Dropout(dropout)

        # Keep Version 2's final projection (it improved performance)
        self.final_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        # NEW: Spike-Activated Residual Gates with context from their own branch's output spikes
        # attn_res_gate_proj: Input will be (d_model + 1) if using average spike count (1 feature)
        self.attn_res_gate_proj = SpikingLinear(nn.Linear(d_model + 1, d_model, bias=False), neuron_type, surrogate_function)
        # mlp_res_gate_proj: Input will be (d_model + 1) if using average spike count (1 feature)
        self.mlp_res_gate_proj = SpikingLinear(nn.Linear(d_model + 1, d_model, bias=False), neuron_type, surrogate_function)

        # NEW: Spiking Attention Output Modulation Gate (from V2, refined input)
        # Input features: 3 (for aggregated Q, K, V spike counts per token)
        self.attn_output_spike_gate_proj = SpikingLinear(nn.Linear(3, d_model, bias=False), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict[str, torch.Tensor]]]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []
        b, l, d_model = x.shape

        # Handle past_key_values for the current layer
        current_layer_cache: Dict[str, torch.Tensor] = {}
        if past_key_values is not None and layer_idx is not None:
            # Ensure the list is long enough for the current layer_idx
            while len(past_key_values) <= layer_idx:
                past_key_values.append({}) # Initialize new dict for the layer if it doesn't exist
            current_layer_cache = past_key_values[layer_idx]

        x_res1 = x # Residual connection input for attention block

        # --- Attention Block ---
        x_norm1 = self.norm1(x) if self.prenorm else x

        # Process through SpikingMesaNet, passing the specific layer's cache
        x_attn_output_continuous, _, attn_spikes_dict = self.attn(
            x_norm1,
            layer_cache=current_layer_cache, # Pass the specific layer's cache dictionary
            use_cache=use_cache,
            **kwargs
        )
        if attn_spikes_dict:
            # Flatten spikes from attn_spikes_dict and add to all_spikes_list
            for key, val in attn_spikes_dict.items():
                if val is not None:
                    all_spikes_list.append(val)

        # Aggregate spikes from Q, K, V for the attention output modulation gate
        # s_q, s_k, s_v are (B, L, feature_dim). Sum along feature dim to get (B, L, 1) per type
        s_q_sum_per_token = attn_spikes_dict['q'].sum(dim=-1, keepdim=True).float()
        s_k_sum_per_token = attn_spikes_dict['k'].sum(dim=-1, keepdim=True).float()
        s_v_sum_per_token = attn_spikes_dict['v'].sum(dim=-1, keepdim=True).float()
        spike_context_for_attn_gate = torch.cat([s_q_sum_per_token, s_k_sum_per_token, s_v_sum_per_token], dim=-1) # (B, L, 3)

        # Pass the aggregated spike context through a SpikingLinear layer to get a continuous gate signal
        spike_gate_output_cont, s_attn_spike_gate = self.attn_output_spike_gate_proj(spike_context_for_attn_gate)
        all_spikes_list.append(s_attn_spike_gate)
        # Apply sigmoid to normalize this continuous output into a gate factor (0 to 1)
        attn_modulation_gate = torch.sigmoid(spike_gate_output_cont)

        # Apply the spike-driven gate to the continuous attention output
        gated_x_attn_output_continuous = x_attn_output_continuous * attn_modulation_gate

        # Residual connection after attention block, gated by a spike-activated gate
        residual_attn_candidate = self.dropout1(gated_x_attn_output_continuous)

        # NEW: Spike-Activated Residual Gate for Attention. Input is a concatenation of the continuous
        # residual candidate and the average spikes from the attention's output.
        # s_o (spikes from the final output projection of MesaNet) has shape (B, L, D_model).
        # Average it along the feature dimension to get a single spike-frequency proxy (B, L, 1).
        s_o_avg = attn_spikes_dict['o'].float().mean(dim=-1, keepdim=True) # (B, L, 1)
        attn_gate_input = torch.cat([residual_attn_candidate, s_o_avg], dim=-1) # Concatenate: (B, L, d_model + 1)

        gate_attn_cont, s_g_attn_res = self.attn_res_gate_proj(attn_gate_input)
        all_spikes_list.append(s_g_attn_res)
        gating_factor_attn = torch.sigmoid(gate_attn_cont)
        x_after_attn = x_res1 + (residual_attn_candidate * gating_factor_attn)

        # --- MLP Block ---
        x_res2 = x_after_attn # Residual connection input for MLP

        x_norm2 = self.norm2(x_after_attn) if self.prenorm else x_after_attn

        # Process through SpikingMLP, collecting spikes
        x_mlp_output_continuous, s_mlp_fc1, s_mlp_fc2 = self.mlp(x_norm2)
        all_spikes_list.extend([s_mlp_fc1, s_mlp_fc2])

        # Residual connection after MLP block, gated by a spike-activated gate
        residual_mlp_candidate = self.dropout2(x_mlp_output_continuous)

        # NEW: Spike-Activated Residual Gate for MLP. Input is a concatenation of the continuous
        # residual candidate and the average spikes from the MLP's second FC layer.
        # s_mlp_fc2 (spikes from the second FC layer of MLP) has shape (B, L, d_model).
        # Average it along the feature dimension to get a single spike-frequency proxy (B, L, 1).
        s_mlp_fc2_avg = s_mlp_fc2.float().mean(dim=-1, keepdim=True) # (B, L, 1)
        mlp_gate_input = torch.cat([residual_mlp_candidate, s_mlp_fc2_avg], dim=-1) # Concatenate: (B, L, d_model + 1)

        gate_mlp_cont, s_g_mlp_res = self.mlp_res_gate_proj(mlp_gate_input)
        all_spikes_list.append(s_g_mlp_res)
        gating_factor_mlp = torch.sigmoid(gate_mlp_cont)
        x_combined_output = x_res2 + (residual_mlp_candidate * gating_factor_mlp)

        # Final spiking projection after residual connections and dropouts (from Version 2)
        x_final_output_continuous, s_final_proj = self.final_proj(x_combined_output)
        all_spikes_list.append(s_final_proj)

        # Concatenate all collected spikes into a single flat tensor. Handle empty list gracefully.
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None]) if all_spikes_list else torch.empty(0, device=x.device, dtype=torch.float)

        return x_final_output_continuous, None, final_spikes