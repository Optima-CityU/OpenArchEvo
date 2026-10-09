"""Example program: a spiking retention block (decayed linear attention).

This file shows what the search evolves: one self-contained module that defines
``SpikingFLABlock(d_model, num_heads, **kwargs)`` whose ``forward(x)`` returns
``(y, reg_loss, spikes)``. Every parameter-dominant projection is a
``SpikingLinear`` (binary spikes in, real values out); state updates and the
token-mixer arithmetic stay continuous. Pure PyTorch, runs on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from openarchevo.models import SpikingLinear, piecewise_quadratic_surrogate


def chunked_retention(q, k, v, log_decay, chunk_size=64):
    """Causal retention o_t = sum_{s<=t} decay^(t-s) (q_t . k_s) v_s, computed chunk by chunk.

    q, k, v: (B, H, L, Dh); log_decay: (H,), negative. Cost is linear in L.
    """
    b, h, length, d_v = v.shape
    pad = (-length) % chunk_size
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    n = q.shape[2] // chunk_size
    q, k, v = (t.reshape(b, h, n, chunk_size, -1) for t in (q, k, v))

    idx = torch.arange(chunk_size, device=q.device, dtype=q.dtype)
    lg = log_decay.view(h, 1, 1)
    causal = idx[:, None] >= idx[None, :]
    intra_decay = torch.exp(lg * (idx[:, None] - idx[None, :]).clamp(min=0)) * causal
    q_decay = torch.exp(lg * (idx + 1).view(-1, 1))
    k_decay = torch.exp(lg * (chunk_size - 1 - idx).view(-1, 1))
    chunk_decay = torch.exp(lg * chunk_size)

    state = q.new_zeros(b, h, q.shape[-1], d_v)
    outs = []
    for i in range(n):
        qi, ki, vi = q[:, :, i], k[:, :, i], v[:, :, i]
        intra = ((qi @ ki.transpose(-1, -2)) * intra_decay) @ vi
        inter = (qi * q_decay) @ state
        outs.append(intra + inter)
        state = chunk_decay * state + (ki * k_decay).transpose(-1, -2) @ vi
    return torch.stack(outs, dim=2).reshape(b, h, n * chunk_size, d_v)[:, :, :length]


class SpikingRetention(nn.Module):
    def __init__(self, d_model, num_heads, neuron_type="bptt", surrogate_function=None, chunk_size=64):
        super().__init__()
        self.num_heads, self.chunk_size = num_heads, chunk_size
        self.q_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.o_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.head_norm = nn.RMSNorm(d_model // num_heads, eps=1e-5)
        # Multi-scale decays as in RetNet: 1 - 2^(-5 - h).
        decay = 1.0 - 2.0 ** (-5.0 - torch.arange(num_heads, dtype=torch.float32))
        self.register_buffer("log_decay", torch.log(decay))

    def forward(self, x):
        b, length, d = x.shape
        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        q, k, v = (t.view(b, length, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))
        k = k * (k.shape[-1] ** -0.5)
        o = chunked_retention(q, k, v, self.log_decay, self.chunk_size)
        o = self.head_norm(o).transpose(1, 2).reshape(b, length, d)
        o, s_o = self.o_proj(o)
        return o, [s_q, s_k, s_v, s_o]


class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, neuron_type="bptt", surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        h, s1 = self.fc1(x)
        y, s2 = self.fc2(h)
        return y, [s1, s2]


class SpikingFLABlock(nn.Module):
    """Pre-norm residual block: spiking retention, then a spiking MLP."""

    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, neuron_type="bptt", **kwargs):
        super().__init__()
        surrogate_function = piecewise_quadratic_surrogate()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.attn = SpikingRetention(d_model, num_heads, neuron_type, surrogate_function)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.mlp = SpikingMLP(d_model, d_model * d_ffn_mult, neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        h, attn_spikes = self.attn(self.norm1(x))
        x = x + self.dropout(h)
        h, mlp_spikes = self.mlp(self.norm2(x))
        x = x + self.dropout(h)
        spikes = torch.cat([s.flatten() for s in attn_spikes + mlp_spikes])
        return x, None, spikes
