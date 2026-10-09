import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution, FusedRMSNormGated
from fla.modules.l2norm import l2norm
from typing import Optional, Tuple, List, Dict

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    # Fallback for SNN modules: A simple step function neuron and dummy surrogate.
    # This ensures the code remains runnable even if SNN specific libraries are not installed.
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return input
        @staticmethod
        def backward(ctx, grad_output): return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: lambda x: (x > 0).float()} # Simple step function (binary output)
    piecewise_quadratic_surrogate = lambda: DummySurrogate.apply # Provide a callable surrogate

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

def softmax(x):
    return F.softmax(x, dim=-1)

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
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
        attn_intra = (q_i @ k_i.transpose(-1, -2)).masked_fill(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn_intra @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')

    if pad_len > 0:
        o = o[:, :, :l]
    return o, S

class DeltaNet(nn.Module):
    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> 'DeltaNet':
        super().__init__()

        # Ensure surrogate_function is callable for SpikingLinear
        surrogate_function = piecewise_quadratic_surrogate()
        if not callable(surrogate_function):
             raise ValueError("`piecewise_quadratic_surrogate` must return a callable function for the SNN module.")

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

        # All nn.Linear layers are replaced by SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(hidden_size, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(hidden_size, self.num_heads, bias=False), neuron_type, surrogate_function)
            # New: Spike-based modulator for beta
            # It takes the mean spike activity (a scalar per token) and projects it to num_heads
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
        else:
            # If short conv is not used, provide an activation for v, similar to v_conv1d's internal activation
            self.v_activation = nn.SiLU()


        if use_gate:
            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, hidden_size, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Dict]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, **kwargs: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        all_spikes = {} # Used to collect all spikes

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

        # SpikingLinear layers take analog input and produce analog output and binary spikes
        q, spikes_q_proj = self.q_proj(hidden_states)
        k, spikes_k_proj = self.k_proj(hidden_states)
        v, spikes_v_proj = self.v_proj(hidden_states)

        all_spikes['q_proj'] = spikes_q_proj
        all_spikes['k_proj'] = spikes_k_proj
        all_spikes['v_proj'] = spikes_v_proj

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            q, conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            # Apply activation for v if short conv is not used, to match v_conv1d behavior
            v = self.v_activation(v)


        # Apply qk_activation only if not 'identity', centralized here for consistency
        if self.qk_activation == 'silu':
            q, k = F.silu(q), F.silu(k)
        elif self.qk_activation == 'relu':
            q, k = q.relu(), k.relu()
        elif self.qk_activation == 'elu':
            q, k = elu_p1(q), elu_p1(k)
        # If self.qk_activation == 'identity', no additional activation applied to q, k here.


        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.qk_norm == 'l2':
            q = l2norm(q)
            k = l2norm(k)
        elif self.qk_norm == 'sum':
            q = sum_norm(q)
            k = sum_norm(k)


        beta = None
        if self.use_beta:
            beta_analog_output, spikes_beta_proj = self.b_proj(hidden_states)
            beta = beta_analog_output.sigmoid()
            all_spikes['b_proj'] = spikes_beta_proj

            # Integrate spike activity for beta modulation
            if spikes_k_proj is not None:
                # Calculate mean spike activity of keys across the feature dimension
                # Resulting shape: (B, L, 1)
                k_spike_mean_activity = spikes_k_proj.float().mean(dim=-1, keepdim=True)
                # Project the mean spike activity to the number of heads using a SpikingLinear layer
                # Resulting shape: (B, L, num_heads)
                beta_spike_mod_analog_output, spikes_beta_mod = self.spike_beta_mod_proj(k_spike_mean_activity)
                all_spikes['spike_beta_mod'] = spikes_beta_mod
                # Apply a soft modulation (e.g., tanh) and then multiply
                beta = beta * (1 + beta_spike_mod_analog_output.tanh()) # Using tanh to keep modulation around 1 (can be slightly >1 or <1)
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

        reg_loss = None # DeltaNet does not return reg_loss

        if self.use_gate:
            g_analog_output, spikes_g = self.g_proj(hidden_states)
            g = rearrange(g_analog_output, '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
            all_spikes['g_proj'] = spikes_g
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')

        o_analog_output, spikes_o_proj = self.o_proj(o)
        all_spikes['o_proj'] = spikes_o_proj

        if attention_mask is not None:
            o_analog_output = pad_input(o_analog_output.squeeze(0), indices, batch_size, q_len)

        return o_analog_output, reg_loss, all_spikes # return o_analog_output for consistency

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        # Ensure surrogate_function is callable
        if not callable(surrogate_function):
             raise ValueError("`surrogate_function` must be callable for SpikingMLP.")

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_out_analog, spike1 = self.fc1(x)
        x_fc2_out_analog, spike2 = self.fc2(x_fc1_out_analog)
        return x_fc2_out_analog, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)

        surrogate_function = piecewise_quadratic_surrogate()
        if not callable(surrogate_function):
             raise ValueError("`piecewise_quadratic_surrogate` must return a callable function for the SNN module.")

        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.dropout2 = nn.Dropout(dropout)

        # Spike-gating for attention residual connection
        self.attn_spike_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, surrogate_function)
        self.attn_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

        # Spike-gating for MLP residual connection
        self.mlp_spike_gate_proj = SpikingLinear(nn.Linear(1, d_model), neuron_type, surrogate_function)
        self.mlp_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

        # Temporal convolution for spike activity before gating projections
        self.use_spike_conv_for_gate = kwargs.pop('use_spike_conv_for_gate', True)
        self.spike_gate_conv_size = kwargs.pop('spike_gate_conv_size', 3)

        if self.use_spike_conv_for_gate:
            self.attn_spike_gate_conv = ShortConvolution(
                hidden_size=1, # Input dimension is 1 (mean spike activity)
                kernel_size=self.spike_gate_conv_size,
                activation='silu'
            )
            self.mlp_spike_gate_conv = ShortConvolution(
                hidden_size=1,
                kernel_size=self.spike_gate_conv_size,
                activation='silu'
            )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        x_res = x
        if self.prenorm:
            x = self.norm1(x)

        x_attn, reg_loss, attn_spikes_dict = self.attn(x, **kwargs)

        attn_block_spike_activity = None
        if attn_spikes_dict and isinstance(attn_spikes_dict, dict):
            k_spikes_for_gating = attn_spikes_dict.get('k_proj')
            if k_spikes_for_gating is not None:
                attn_block_spike_activity = k_spikes_for_gating.float().mean(dim=-1, keepdim=True) # (B, L, 1)
            else:
                for spike_name, spike_tensor in attn_spikes_dict.items():
                    if spike_tensor is not None:
                        attn_block_spike_activity = spike_tensor.float().mean(dim=-1, keepdim=True)
                        break

            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        # Apply temporal convolution to spike activity if enabled
        if attn_block_spike_activity is not None and self.use_spike_conv_for_gate:
            # The output of ShortConvolution is analog, which then feeds into SpikingLinear
            attn_block_spike_activity, _ = self.attn_spike_gate_conv(attn_block_spike_activity)

        if attn_block_spike_activity is not None:
            attn_gate_analog_output, attn_gate_binary_spikes = self.attn_spike_gate_proj(attn_block_spike_activity)
            all_spikes_list.append(attn_gate_binary_spikes)
            attn_gate = self.attn_gate_scale * torch.sigmoid(attn_gate_analog_output) # Use sigmoid as in Version 2
            x_attn = x_attn * attn_gate

        x = x_res + self.dropout1(x_attn)

        x_res = x
        if self.prenorm:
            x = self.norm2(x)

        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x)

        mlp_spikes_for_gating_means = []
        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
            mlp_spikes_for_gating_means.append(mlp_spike1.float().mean(dim=-1, keepdim=True))
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)
            mlp_spikes_for_gating_means.append(mlp_spike2.float().mean(dim=-1, keepdim=True))

        mlp_block_spike_activity = None
        if mlp_spikes_for_gating_means:
            mlp_block_spike_activity = torch.mean(torch.stack(mlp_spikes_for_gating_means), dim=0) # (B, L, 1)

        # Apply temporal convolution to spike activity if enabled
        if mlp_block_spike_activity is not None and self.use_spike_conv_for_gate:
            # The output of ShortConvolution is analog, which then feeds into SpikingLinear
            mlp_block_spike_activity, _ = self.mlp_spike_gate_conv(mlp_block_spike_activity)

        if mlp_block_spike_activity is not None:
            mlp_gate_analog_output, mlp_gate_binary_spikes = self.mlp_spike_gate_proj(mlp_block_spike_activity)
            all_spikes_list.append(mlp_gate_binary_spikes)
            mlp_gate = self.mlp_gate_scale * torch.sigmoid(mlp_gate_analog_output) # Use sigmoid as in Version 2
            x_mlp = x_mlp * mlp_gate

        x = x_res + self.dropout2(x_mlp)

        final_spikes = None
        if all_spikes_list:
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

        return x, reg_loss, final_spikes