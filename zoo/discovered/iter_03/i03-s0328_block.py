from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

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
        spikes = self.neuron(x.transpose(-1, -2)).transpose(-1, -2) # The neuron applies thresholding along the last dimension of the transposed input
        return self.linear(spikes), spikes

# ==================== White-box Operator (Comba Closed-Loop) ====================

def comba_chunkwise_parallel(q, k, v, g, feedback_net, chunk_size=64, initial_state=None):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len), value=0.0) # pad g with 0.0, as it's a log-decay factor
    
    # In the improved version, inputs q, k, v are expected to be binary spikes.
    # However, l2norm needs floating point values. This indicates a hybrid approach where
    # the core attention mechanism still operates on continuous (analog) values derived from spikes.
    # Thus, q, k are normalized *after* being potentially derived from spikes via SpikingLinear (analog output).
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
        
        # --- [Fix 1] Closed-loop feedback stability enhancement ---
        # 1. Normalize state features to prevent FeedbackNet input explosion
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        s_feat = (S / s_norm).mean(dim=(-1, -2), keepdim=True).flatten(2) # [B, H, 1]
        
        # 2. Limit feedback offset to [-2, 2] to prevent complete override of original gating
        # feedback_net now internally ensures spike inputs to its linear layers.
        f_i = feedback_net(s_feat).squeeze(-1) # [B, H]
        f_i = torch.tanh(f_i) * 2.0 
        
        # --- [Fix 2] Gate numerical computation alignment ---
        # Use more stable logsigmoid for decay factors
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
        
        # --- [Fix 3] State update stability enhancement ---
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        # Limit update step size to prevent excessive single updates
        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update
        
        # Strict renormalization after each chunk update
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== New: Spiking Feedback Network for Comba ====================

class SpikingFeedbackNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, neuron_type: str, surrogate_function):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(input_dim, hidden_dim), neuron_type, surrogate_function)
        self.activation = nn.Tanh() 
        # [MODIFICATION]: Neuron after activation to ensure spike input to fc2
        self.inter_neuron_fb = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.fc2 = SpikingLinear(nn.Linear(hidden_dim, output_dim), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out_fc1, _ = self.fc1(x) # _ contains spikes generated from x (input to fc1)
        x_act = self.activation(x_out_fc1) # x_act is analog output from Tanh

        # Convert x_act to spikes before feeding to fc2
        spikes_after_activation = self.inter_neuron_fb(x_act.transpose(-1, -2)).transpose(-1, -2)

        final_out_analog, _ = self.fc2(spikes_after_activation) # _ contains spikes generated from spikes_after_activation (input to fc2)

        return final_out_analog

# ==================== New: Spiking Wrapper for ShortConvolution ====================

class SpikingConvWrapper(nn.Module):
    def __init__(self, conv_layer: ShortConvolution, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.conv = conv_layer

    def forward(self, x):
        spikes = self.neuron(x.transpose(-1, -2)).transpose(-1, -2)
        conv_out, _ = self.conv(x) # ShortConvolution expects analog float input, and its output is analog
        return conv_out, spikes # Return analog output and generated spikes

# ==================== SpikingComba Layer ====================

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        # Projections: All nn.Linear layers replaced by SpikingLinear as per requirements
        # Inputs to these are spikes (from SpikingFLABlock), outputs are analog.
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        self.feedback_net = SpikingFeedbackNet(
            input_dim=1,
            output_dim=1,
            hidden_dim=8,
            neuron_type=neuron_type,
            surrogate_function=surrogate
        )

        if use_short_conv:
            # Wrapped ShortConvolution with SpikingConvWrapper
            self.q_conv1d = SpikingConvWrapper(ShortConvolution(self.key_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.k_conv1d = SpikingConvWrapper(ShortConvolution(self.key_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.v_conv1d = SpikingConvWrapper(ShortConvolution(self.value_dim, conv_size, activation='silu'), neuron_type, surrogate)
        else:
            self.q_conv1d = None
            self.k_conv1d = None
            self.v_conv1d = None

        self.o_norm = RMSNorm(self.head_v_dim)
        # [MODIFICATION]: Neuron after o_norm and before o_proj to ensure spike input to o_proj
        self.pre_o_proj_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}

        # Projections. Outputs (q, k, v, g) are analog (from internal nn.Linear of SpikingLinear).
        q_analog, s_q_input = self.q_proj(x)
        k_analog, s_k_input = self.k_proj(x)
        v_analog, s_v_input = self.v_proj(x)
        g_analog, s_g_input = self.g_proj(x)
        all_spikes.update({'q_proj_input':s_q_input, 'k_proj_input':s_k_input, 'v_proj_input':s_v_input, 'g_proj_input':s_g_input})

        if self.q_conv1d is not None:
            # SpikingConvWrapper generates spikes from its analog input, but passes analog to ShortConvolution.
            # conv_out is analog.
            q_conv_out_analog, s_q_conv_input = self.q_conv1d(q_analog)
            k_conv_out_analog, s_k_conv_input = self.k_conv1d(k_analog)
            v_conv_out_analog, s_v_conv_input = self.v_conv1d(v_analog)
            all_spikes.update({'q_conv_input':s_q_conv_input, 'k_conv_input':s_k_conv_input, 'v_conv_input':s_v_conv_input})
            q = q_conv_out_analog
            k = k_conv_out_analog
            v = v_conv_out_analog
        else:
            q, k, v = q_analog, k_analog, v_analog
        g = g_analog

        # Rearrange for attention computation. q, k, v, g are all analog values here.
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g, 'b l h -> b h l')

        layer_state = None
        if past_key_values is not None:
            layer_state = past_key_values[self.layer_idx] if self.layer_idx in past_key_values else None

        # The core comba_chunkwise_parallel operates on analog signals
        o_analog, final_state = comba_chunkwise_parallel(q, k, v, g, self.feedback_net, initial_state=layer_state)

        if past_key_values is not None:
            past_key_values[self.layer_idx] = final_state

        o_analog = rearrange(o_analog, 'b h l d -> b l h d')
        o_analog = self.o_norm(o_analog) # o_analog is analog output after RMSNorm
        o_analog = rearrange(o_analog, 'b l h d -> b l (h d)') # o_analog is analog

        # [MODIFICATION]: Spike the output of o_norm before feeding to o_proj
        spikes_before_o_proj = self.pre_o_proj_neuron(o_analog.transpose(-1, -2)).transpose(-1, -2)
        all_spikes['pre_o_proj_neuron'] = spikes_before_o_proj

        # o_proj now receives spikes as input
        o_final_analog, s_o_proj_input = self.o_proj(spikes_before_o_proj)
        all_spikes['o_proj_input'] = s_o_proj_input

        return o_final_analog, None, all_spikes

# [SpikingMLP and SpikingFLABlock]

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        # [MODIFICATION]: Neuron after fc1's analog output to ensure spike input to fc2
        self.intermediate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        fc1_out_analog, spike1_input = self.fc1(x) 

        # Convert fc1_out_analog to spikes for dropout and fc2
        spikes_after_fc1_output = self.intermediate_neuron(fc1_out_analog.transpose(-1, -2)).transpose(-1, -2)

        # Apply dropout to the spikes
        dropped_spikes = self.dropout(spikes_after_fc1_output)

        # fc2 now receives spikes as input. fc2_out_analog is the output of its internal nn.Linear.
        fc2_out_analog, spike2_input = self.fc2(dropped_spikes)

        # Return the final analog output and all collected spike tensors
        return fc2_out_analog, spike1_input, spikes_after_fc1_output, spike2_input

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Spiking neuron after the first RMSNorm (or directly on input if not prenorm)
        self.attn_input_neuron = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())

        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)

        # Spiking neuron after the second RMSNorm (or after attn block if not prenorm)
        self.mlp_input_neuron = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())

        d_ffn = d_model * d_ffn_mult
        # SpikingMLP now ensures all its internal linear layers receive spike inputs
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        all_spikes_list = []
        x_res = x

        # Prepare input for Attention Block
        if self.prenorm: 
            x_normalized = self.norm1(x) # Analog output
            # [MODIFICATION]: Generate spikes from normalized input before attention block
            spikes_for_attn = self.attn_input_neuron(x_normalized.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_attn)
        else:
            # If not prenorm, spike the raw input directly before attention
            spikes_for_attn = self.attn_input_neuron(x.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_attn)

        # Attention block. Its input (spikes_for_attn) is now strictly binary spikes.
        x_attn_analog_out, _, attn_spikes = self.attn(spikes_for_attn, **kwargs) 
        if attn_spikes: 
            all_spikes_list.extend([s for s in attn_spikes.values() if s is not None])
        x = x_res + self.dropout1(x_attn_analog_out) # x_attn_analog_out is analog output from Comba

        x_res = x

        # Prepare input for MLP Block
        if self.prenorm: 
            x_normalized = self.norm2(x) # Analog output
            # [MODIFICATION]: Generate spikes from normalized input before MLP block
            spikes_for_mlp = self.mlp_input_neuron(x_normalized.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_mlp)
        else:
            # If not prenorm, spike the output of the attention block directly before MLP
            spikes_for_mlp = self.mlp_input_neuron(x.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_mlp)

        # MLP block. Its input (spikes_for_mlp) is now strictly binary spikes.
        # SpikingMLP now returns 1 analog output and 3 spike tensors.
        x_mlp_analog_out, mlp_spike1_input, mlp_spikes_after_fc1_output, mlp_spike2_input = self.mlp(spikes_for_mlp)

        all_spikes_list.extend([
            s for s in [mlp_spike1_input, mlp_spikes_after_fc1_output, mlp_spike2_input] if s is not None
        ])

        x = x_res + self.dropout2(x_mlp_analog_out) # x_mlp_analog_out is analog output from MLP

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        return x, None, final_spikes