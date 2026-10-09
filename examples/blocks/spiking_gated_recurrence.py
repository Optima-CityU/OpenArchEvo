"""Example program: a gated linear recurrence with spike-activity control.

The forget gate of an element-wise linear recurrence (HGRN-style) is shifted by
the block's own spike activity -- a small instance of the
"spike statistic -> learned control -> state update" dependency found in the
discovered architectures. An activity penalty is returned as ``reg_loss``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from openarchevo.models import SpikingLinear, piecewise_quadratic_surrogate


def gated_scan(forget, inputs, chunk_size=32):
    """h_t = f_t * h_{t-1} + (1 - f_t) * i_t over dim 1, chunked in log space. f in (0.5, 1)."""
    b, length, d = inputs.shape
    pad = (-length) % chunk_size
    if pad:
        forget = F.pad(forget, (0, 0, 0, pad), value=1.0)
        inputs = F.pad(inputs, (0, 0, 0, pad))
    h = inputs.new_zeros(b, d)
    outs = []
    for f, x in zip(forget.split(chunk_size, dim=1), inputs.split(chunk_size, dim=1)):
        log_cum = torch.cumsum(torch.log(f), dim=1)                     # sum_{r<=t} log f_r
        acc = torch.cumsum(torch.exp(-log_cum) * (1 - f) * x, dim=1)    # bounded since f > 0.5
        out = torch.exp(log_cum) * (acc + h[:, None])
        outs.append(out)
        h = out[:, -1]
    return torch.cat(outs, dim=1)[:, :length]


class SpikingGatedRecurrence(nn.Module):
    def __init__(self, d_model, neuron_type="bptt", surrogate_function=None, target_rate=0.1):
        super().__init__()
        self.target_rate = target_rate
        self.in_proj = SpikingLinear(nn.Linear(d_model, 3 * d_model), neuron_type, surrogate_function)
        self.out_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, surrogate_function)
        self.activity_gain = nn.Parameter(torch.full((d_model,), 0.5))
        self.norm = nn.RMSNorm(d_model, eps=1e-5)

    def forward(self, x):
        z, s_in = self.in_proj(x)
        i, f_logit, g = z.chunk(3, dim=-1)
        # Spike statistic -> control: tokens whose input fires more keep more memory.
        rate = s_in.mean(dim=-1, keepdim=True)
        f_logit = f_logit + self.activity_gain * (rate - self.target_rate)
        forget = 0.5 + 0.5 * torch.sigmoid(f_logit)
        h = gated_scan(forget, F.silu(i))
        o, s_out = self.out_proj(self.norm(h) * torch.sigmoid(g))
        reg = (s_in.mean() - self.target_rate) ** 2
        return o, reg, [s_in, s_out]


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
        self.mixer = SpikingGatedRecurrence(d_model, neuron_type, surrogate_function)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.mlp = SpikingMLP(d_model, d_model * d_ffn_mult, neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.reg_weight = 0.01

    def forward(self, x, **kwargs):
        h, reg, mixer_spikes = self.mixer(self.norm1(x))
        x = x + self.dropout(h)
        h, mlp_spikes = self.mlp(self.norm2(x))
        x = x + self.dropout(h)
        spikes = torch.cat([s.flatten() for s in mixer_spikes + mlp_spikes])
        return x, self.reg_weight * reg, spikes
