import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

# Original FLA imports (assuming these are available in the real environment)
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

# ==================== SNN Related Imports and Helper Classes ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder SNN modules.")
    class DummySurrogate:
        @staticmethod
        def apply(v_mem, threshold, d_ref=None, spk_cnt=None, **kwargs):
            return (v_mem > threshold).float()

    class DummyNeuron:
        def __init__(self, surrogate_function, **kwargs):
            self.surrogate_function = surrogate_function
            self.v_mem = None # Membrane potential
            self.threshold = 1.0 # Simple threshold

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            if self.v_mem is None or self.v_mem.shape != x.shape:
                self.v_mem = torch.zeros_like(x, device=x.device)

            # Integrate input (simple integrate-and-fire for dummy)
            self.v_mem = self.v_mem + x

            # Generate spikes
            spikes = self.surrogate_function.apply(self.v_mem, self.threshold)

            # Reset membrane potential for spiking neurons (simple hard reset)
            self.v_mem = self.v_mem * (1 - spikes)

            return spikes

    neuron_registry = {'sdn': lambda surrogate_function: DummyNeuron(surrogate_function)}
    piecewise_quadratic_surrogate = lambda: DummySurrogate

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

# ================================================================

def softmax(x):
    return F.softmax(x, dim=-1)

def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    # Padding to ensure sequence length is a multiple of chunk_size
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len)) # beta is (b h l) so pad last dim

    padded_len = l + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Intra-chunk computation setup
    mask_intra = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q_chunks, k_chunks, v_chunks, k_beta_chunks = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])

    attn_matrices = -(k_beta_chunks @ k_chunks.transpose(-1, -2))
    attn_matrices = attn_matrices.masked_fill(mask_intra, 0) # Fill upper triangle with 0, including diagonal

    for i in range(1, chunk_size):
        # This is a specific recurrence for the delta rule, kept as is.
        attn_matrices[..., i, :i] = attn_matrices[..., i, :i] + (attn_matrices[..., i, :, None].clone() * attn_matrices[..., :, :i].clone()).sum(-2)
    attn_matrices = attn_matrices + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u_chunks = attn_matrices @ v_chunks
    w_chunks = attn_matrices @ k_beta_chunks

    # Inter-chunk recurrence
    S = k.new_zeros(b, h, d_k, d_v)
    o_rearranged = torch.zeros_like(v_chunks)

    mask_inter = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q_chunks[:, :, i], k_chunks[:, :, i]
        
        # Intra-chunk attention for this current chunk
        attn_intra = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask_inter, 0) # Fill upper triangle with 0, excluding diagonal

        # Update u_i based on current chunk and global state
        u_i = u_chunks[:, :, i] - w_chunks[:, :, i] @ S
        
        # Calculate inter-chunk output
        o_inter = q_i @ S
        o_rearranged[:, :, i] = o_inter + attn_intra @ u_i
        
        # Update global state S for next chunk
        S = S + k_i.transpose(-1, -2) @ u_i
        
    o = rearrange(o_rearranged, 'b h n c d -> b h (n c) d')

    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

class DeltaNet(nn.Module):
    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> 'DeltaNet':
        super().__init__()

        if neuron_registry is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")

        surrogate_function = piecewise_quadratic_surrogate()

        self.mode = mode
        self.qk_activation = qk_activation
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
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        # New: Self-Modulating Spiking Input (SMSI) components
        # Projects original hidden_states to generate context spikes
        self.context_proj = SpikingLinear(nn.Linear(hidden_size, hidden_size), neuron_type, surrogate_function)
        # Projects the mean spike activity from context_proj to generate dynamic input modulation
        self.input_mod_proj = SpikingLinear(nn.Linear(1, hidden_size), neuron_type, surrogate_function)


        # All nn.Linear layers are replaced by SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), neuron_type, surrogate_function)
            # Spike-based modulator for beta (from V2)
            self.spike_beta_mod_proj = SpikingLinear(nn.Linear(1, self.num_heads, bias=True), neuron_type, surrogate_function)


        # ShortConvolution layers remain non-spiking as per instructions and SNN-ANN hybrid design
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )

        if use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional['Cache']=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: 'Unpack[Dict]') -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional['Cache']]:
        all_spikes = {}

        # CRITICAL FIX: Removed unpadding/re-padding logic to ensure batch size independence.
        # The `delta_rule_chunkwise` expects (B, H, L, D) and handles its own internal padding.
        # `hidden_states` is already assumed to be (B, L, D) where L is max sequence length (padded).
        # The `attention_mask` parameter is primarily for compatibility with transformer interfaces.
        # `cu_seqlens` can still be passed to `ShortConvolution` if applicable and generated externally.

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None) # Still pass cu_seqlens to ShortConvolution if provided

        # --- Self-Modulating Spiking Input (SMSI) ---
        context_proj_out_analog, context_spikes = self.context_proj(hidden_states)
        all_spikes['context_proj'] = context_spikes

        mean_context_spike_activity = context_spikes.float().mean(dim=-1, keepdim=True)

        mod_signal_analog, mod_spikes = self.input_mod_proj(mean_context_spike_activity)
        all_spikes['input_mod_proj'] = mod_spikes

        modulated_hidden_states = hidden_states * (1 + mod_signal_analog.tanh())
        # --- End SMSI ---

        q, spikes_q_proj = self.q_proj(modulated_hidden_states)
        k, spikes_k_proj = self.k_proj(modulated_hidden_states)
        v, spikes_v_proj = self.v_proj(modulated_hidden_states)

        all_spikes['q_proj'] = spikes_q_proj
        all_spikes['k_proj'] = spikes_k_proj
        all_spikes['v_proj'] = spikes_v_proj

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            # ShortConvolution can accept cu_seqlens for efficient packed sequence processing
            q, conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        q, k = map(lambda x: rearrange(x, 'b l (h d) -> b l h d', h=self.num_heads), (q, k))
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.num_heads)

        if self.qk_norm == 'l2':
            q = l2norm(q)
            k = l2norm(k)
        elif self.qk_norm == 'sum':
            q = sum_norm(q)
            k = sum_norm(k)


        if self.use_beta:
            beta, spikes_beta = self.b_proj(modulated_hidden_states)
            beta = beta.sigmoid()
            all_spikes['b_proj'] = spikes_beta

            if spikes_k_proj is not None:
                # Use k_proj spikes for beta modulation for more dynamic control
                k_spike_mean_activity = spikes_k_proj.float().mean(dim=-1, keepdim=True)
                beta_spike_mod_analog, spikes_beta_mod_binary = self.spike_beta_mod_proj(k_spike_mean_activity)
                all_spikes['spike_beta_mod'] = spikes_beta_mod_binary
                beta = beta * (1 + beta_spike_mod_analog.tanh()) # Modulate beta based on spike activity
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
                'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None
            }
            if self.layer_idx is not None:
                past_key_values[self.layer_idx] = cache_data

        if self.use_gate:
            g, spikes_g = self.g_proj(modulated_hidden_states)
            g = rearrange(g, 'b t (h d) -> b t h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
            all_spikes['g_proj'] = spikes_g
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')

        o, spikes_o = self.o_proj(o)
        all_spikes['o_proj'] = spikes_o

        # Removed `pad_input` as unpadding was removed for batch-size independence.
        # The output `o` is already the correct shape (B, L, D) corresponding to the input `hidden_states`.

        return o, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_out, spike1 = self.fc1(x)
        x_after_dropout = self.dropout(x_fc1_out)
        x_fc2_out, spike2 = self.fc2(x_after_dropout)
        return x_fc2_out, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm

        # Norm before attention
        self.norm1 = RMSNorm(d_model)
        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # NEW: Spike-Gated Residual for Attention Output (SGRC-Attn)
        # This layer generates a spike-driven gate from the attention output.
        self.attn_output_gate_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, piecewise_quadratic_surrogate())

        # Norm before MLP
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

        # NEW: Spike-Gated Residual for MLP Output (SGRC-MLP)
        # This layer generates a spike-driven gate from the MLP output.
        self.mlp_output_gate_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, piecewise_quadratic_surrogate())

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # --- Attention Sub-block ---
        x_res_attn = x # Preserve residual for attention

        x_for_attn = self.norm1(x) if self.prenorm else x # Apply pre-norm if enabled

        x_attn_out_analog, reg_loss, attn_spikes_dict = self.attn(x_for_attn, **kwargs)

        if attn_spikes_dict:
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        # NEW: Apply Spike-Gated Residual for Attention Output
        # The gate itself is produced by a spiking linear layer, providing SNN nature.
        attn_gate_raw_analog, attn_gate_spikes = self.attn_output_gate_proj(x_attn_out_analog)
        all_spikes_list.append(attn_gate_spikes) # Collect spikes from the gate projection

        # Use a sigmoid activation to constrain gate values between 0 and 1, modulating the attention output.
        attn_gate = attn_gate_raw_analog.sigmoid() 
        gated_x_attn = x_attn_out_analog * attn_gate # Element-wise gating

        # Residual connection: input + gated_attention_output
        x = x_res_attn + self.dropout1(gated_x_attn)
        if not self.prenorm: # If post-norm, apply norm here after residual for attention output
            x = self.norm1(x)


        # --- MLP Sub-block ---
        x_res_mlp = x # Preserve residual for MLP

        x_for_mlp = self.norm2(x) if self.prenorm else x # Apply pre-norm if enabled

        x_mlp_out_analog, mlp_spike1, mlp_spike2 = self.mlp(x_for_mlp)

        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)

        # NEW: Apply Spike-Gated Residual for MLP Output
        mlp_gate_raw_analog, mlp_gate_spikes = self.mlp_output_gate_proj(x_mlp_out_analog)
        all_spikes_list.append(mlp_gate_spikes) # Collect spikes from the gate projection

        # Use a sigmoid activation to constrain gate values between 0 and 1, modulating the MLP output.
        mlp_gate = mlp_gate_raw_analog.sigmoid()
        gated_x_mlp = x_mlp_out_analog * mlp_gate # Element-wise gating

        # Residual connection: input + gated_mlp_output
        x = x_res_mlp + self.dropout2(gated_x_mlp)
        if not self.prenorm: # If post-norm, apply norm here after residual for MLP output
            x = self.norm2(x)

        final_spikes = None
        if all_spikes_list:
            # Flatten all collected binary spike tensors and concatenate them into a single tensor
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

        return x, reg_loss, final_spikes