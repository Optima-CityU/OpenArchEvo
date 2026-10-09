"""Example program: a spiking diagonal state-space block (S4D-style, real decays).

Each channel applies a long causal convolution whose kernel is a sum of
learnable exponential decays, evaluated with an FFT in O(L log L). The input and
output projections are spiking.
"""

import math

import torch
import torch.nn as nn

from openarchevo.models import SpikingLinear, piecewise_quadratic_surrogate


class SpikingDiagonalSSM(nn.Module):
    def __init__(self, d_model, d_state=16, neuron_type="bptt", surrogate_function=None):
        super().__init__()
        self.in_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, surrogate_function)
        self.out_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, surrogate_function)
        rates = torch.linspace(math.log(1e-3), math.log(0.5), d_state)
        self.log_rate = nn.Parameter(rates.repeat(d_model, 1))            # (D, N): decay rates
        self.coef = nn.Parameter(torch.randn(d_model, d_state) / d_state)  # (D, N): mixing weights
        self.skip = nn.Parameter(torch.ones(d_model))

    def kernel(self, length):
        t = torch.arange(length, device=self.coef.device, dtype=self.coef.dtype)
        decay = torch.exp(-torch.exp(self.log_rate)[..., None] * t)        # (D, N, L)
        return (self.coef[..., None] * decay).sum(dim=1)                    # (D, L)

    def forward(self, x):
        u, s_in = self.in_proj(x)
        length = u.shape[1]
        u_f = torch.fft.rfft(u.transpose(1, 2), n=2 * length)
        k_f = torch.fft.rfft(self.kernel(length), n=2 * length)
        y = torch.fft.irfft(u_f * k_f, n=2 * length)[..., :length].transpose(1, 2)
        y = torch.nn.functional.gelu(y + self.skip * u)
        o, s_out = self.out_proj(y)
        return o, [s_in, s_out]


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
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, neuron_type="bptt", **kwargs):
        super().__init__()
        surrogate_function = piecewise_quadratic_surrogate()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.ssm = SpikingDiagonalSSM(d_model, neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.mlp = SpikingMLP(d_model, d_model * d_ffn_mult, neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        h, ssm_spikes = self.ssm(self.norm1(x))
        x = x + self.dropout(h)
        h, mlp_spikes = self.mlp(self.norm2(x))
        x = x + self.dropout(h)
        spikes = torch.cat([s.flatten() for s in ssm_spikes + mlp_spikes])
        return x, None, spikes
