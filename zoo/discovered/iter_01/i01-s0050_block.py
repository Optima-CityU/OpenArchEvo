import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder SNN modules.")

    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        output = self.linear(spikes)
        return output, spikes

def kda_chunkwise_parallel(q, k, v, beta, decay, alpha, chunk_size=32, initial_state=None):
    """KDA Chunkwise parallel operator.
    Core logic: S_t = decay * S_{t-1} + beta * (alpha * v_t - S_{t-1} @ k_t) @ k_t^T
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len), value=0.0)
        decay = F.pad(decay, (0, pad_len), value=0.0) 
        alpha = F.pad(alpha, (0, pad_len), value=0.0)
        
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    v = v * alpha.unsqueeze(-1)

    q, k, v = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v]
    )
    beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [beta, decay]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)

    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i, b_i, d_i = q[:, :, i], k[:, :, i], v[:, :, i], beta[:, :, i], decay[:, :, i]
        
        d_cumsum = torch.cumsum(d_i, dim=-1) 
        decay_rel = (d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)) 
        decay_mask = torch.exp(decay_rel).masked_fill(mask_triu, 0)
        
        v_pred_inter = k_i @ S 
        v_error = (v_i - v_pred_inter) * b_i.unsqueeze(-1)
        
        attn_intra = (k_i @ k_i.transpose(-1, -2)) * decay_mask * mask_tril
        v_error_intra = v_error - (attn_intra @ v_error) * 0.5
        
        decay_from_start = torch.exp(d_cumsum).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        q_decay_k = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = q_decay_k @ v_error_intra
        
        o[:, :, i] = o_inter + o_intra
        
        d_mean = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1)
        delta_S = k_i.transpose(-1, -2) @ v_error_intra
        S = d_mean.exp() * S + delta_S
        
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

class SpikingKDA(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=1.0, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) 
        self.w_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) 
        self.a_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) 

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
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        beta, s_b = self.b_proj(x)
        w, s_w = self.w_proj(x)
        alpha, s_a = self.a_proj(x)
        all_spikes.update({'q':s_q, 'k':s_k, 'v':s_v, 'beta':s_b, 'w':s_w, 'alpha':s_a})

        if self.q_conv1d is not None:
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

        beta = torch.sigmoid(beta) * 0.5
        alpha = torch.sigmoid(alpha) 
        decay = F.logsigmoid(w)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        beta = rearrange(beta, 'b l h -> b h l')
        decay = rearrange(decay, 'b l h -> b h l')
        alpha = rearrange(alpha, 'b l h -> b h l')

        layer_state = None
        if past_key_values is not None and self.layer_idx is not None and self.layer_idx in past_key_values:
            layer_state = past_key_values[self.layer_idx]

        o, final_state = kda_chunkwise_parallel(q, k, v, beta, decay, alpha, initial_state=layer_state)

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values[self.layer_idx] = final_state

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x) 
        x_dropped = self.dropout(x)
        x_final, spike2 = self.fc2(x_dropped) 
        return x_final, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp=True, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingKDA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)

        self.final_norm = RMSNorm(d_model)
        self.final_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, piecewise_quadratic_surrogate())

    def forward(self, x, **kwargs):
        all_spikes_list: list[torch.Tensor] = [] 
        x_res = x

        if self.prenorm: 
            x = self.norm1(x)

        x_attn, _, attn_spikes = self.attn(x, **kwargs)

        if attn_spikes:
            for spike_tensor in attn_spikes.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        x = x_res + self.dropout1(x_attn)

        x_res = x

        if self.prenorm: 
            x = self.norm2(x)

        x_mlp, s1, s2 = self.mlp(x)
        all_spikes_list.extend([s1, s2])

        x = x_res + self.dropout2(x_mlp)

        x = self.final_norm(x)
        x_final_proj, s_final_proj = self.final_proj(x)
        all_spikes_list.append(s_final_proj)
        x = x_final_proj

        final_spikes = None
        if all_spikes_list:
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

        return x, None, final_spikes