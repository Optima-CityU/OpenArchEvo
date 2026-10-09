from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

from typing import Optional, Tuple, List, Dict

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
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)
        output = self.linear(spikes)
        return output, spikes

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
        
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        s_feat = (S / s_norm).mean(dim=(-1, -2), keepdim=True).flatten(2)
        
        f_i = feedback_net(s_feat).squeeze(-1)
        f_i = torch.tanh(f_i) * 2.0 
        
        log_decay = F.logsigmoid(g_i + f_i.unsqueeze(-1))
        d_cumsum = torch.cumsum(log_decay, dim=-1)
        
        decay_rel = d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0)
        
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = attn_intra @ v_i
        
        o[:, :, i] = o_inter + o_intra
        
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update
        
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()
        if neuron_registry is None or surrogate is None:
            raise ImportError("Spiking neuron modules are not available. Please check your installation.")

        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)

        self.feedback_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Dict[int, torch.Tensor]]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        all_spikes: Dict[str, torch.Tensor] = {}

        q_out, s_q = self.q_proj(x)
        k_out, s_k = self.k_proj(x)
        v_out, s_v = self.v_proj(x)
        g_out, s_g = self.g_proj(x)
        all_spikes.update({'q_proj':s_q, 'k_proj':s_k, 'v_proj':s_v, 'g_proj':s_g})

        if self.use_short_conv:
            q_conv_out, _ = self.q_conv1d(q_out)
            k_conv_out, _ = self.k_conv1d(k_out)
            v_conv_out, _ = self.v_conv1d(v_out)
        else:
            q_conv_out, k_conv_out, v_conv_out = q_out, k_out, v_out

        q = rearrange(q_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g_out, 'b l h -> b h l')

        layer_state = None
        if past_key_values is not None and self.layer_idx is not None:
            layer_state = past_key_values.get(self.layer_idx, None)

        o, final_state = comba_chunkwise_parallel(q, k, v, g, self.feedback_net, initial_state=layer_state)

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values[self.layer_idx] = final_state

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')

        o_final, s_o = self.o_proj(o)
        all_spikes['o_proj'] = s_o

        return o_final, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError("Spiking neuron modules are not available.")

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)

        self.intermediate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)

        self.dropout = nn.Dropout(dropout) 

        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_fc1_out_analog, spike1 = self.fc1(x) 

        x_gelu_analog = F.gelu(x_fc1_out_analog)

        intermediate_spikes_transposed = self.intermediate_neuron(x_gelu_analog.transpose(-1, -2))
        intermediate_spikes = intermediate_spikes_transposed.transpose(-1, -2)

        spikes_after_dropout = self.dropout(intermediate_spikes)

        x_fc2_out_analog, spike2 = self.fc2(spikes_after_dropout) 

        return x_fc2_out_analog, spike1, intermediate_spikes, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        x_res = x
        if self.prenorm:
            x = self.norm1(x)

        x_attn, reg_loss, attn_spikes_dict = self.attn(x, **kwargs)

        if attn_spikes_dict and isinstance(attn_spikes_dict, dict):
            for spike_tensor in attn_spikes_dict.values():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)

        x = x_res + self.dropout1(x_attn)

        x_res = x
        if self.prenorm:
            x = self.norm2(x)

        x_mlp_out_analog, mlp_spike1, mlp_intermediate_spikes, mlp_spike2 = self.mlp(x)

        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
        if mlp_intermediate_spikes is not None:
            all_spikes_list.append(mlp_intermediate_spikes)
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)

        x = x_res + self.dropout2(x_mlp_out_analog)

        final_spikes = None
        if all_spikes_list:
            final_spikes = torch.cat([s.flatten() for s in all_spikes_list])

        return x, reg_loss, final_spikes