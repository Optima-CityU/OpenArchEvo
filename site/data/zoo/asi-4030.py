# -*- coding: utf-8 -*-
"""
spiking_fla_pyramid
===================
A radically different **hierarchical temporal-pyramid** block that departs from
all earlier *multi-branch gating* and *delta-rule* paradigms.  Instead of
blending parallel memory paths via soft/ hard gates, information is aggregated
through a **causal coarse-to-fine pyramid** that recursively downsamples the
sequence and broadcasts coarse context back to fine resolution.  This yields an
O(N · log N) computation and introduces a new SNN/ANN interaction pattern:

1.  **Hierarchical aggregation** –  Each stride-`S` level summarises the **first
    token** of every causal window of length `S` (changed from *last token* to
    guarantee strict causality), processes it with a dedicated `SpikingLinear`
    projection, then repeats (upsamples) the coarse features so every fine-scale
    token receives its matching **past** coarse context.
2.  **Cross-scale residual fusion** –  Learnable per-level scalar `alpha` merges
    coarse signals with the fine-level `SpikingLinear` projection.  No
    competitive gating or softmax is used, completely breaking away from the
    repeated gate patterns.
3.  **Universal spiking linear layers** –  Every projection (fine as well as
    all coarse levels and the FFN) is implemented with `SpikingLinear`; all
    inputs are binary spikes.  Spikes from **every** layer are gathered and
    concatenated into the mandatory `final_spikes` tensor returned by
    `forward()`.
4.  **Sub-quadratic & causal** –  The down-/upsampling operations are
    element-wise and strictly causal (each summary only sees present & past
    tokens).  Complexity is `O(N log N)`.
5.  **Einops everywhere** –  All reshaping is performed with `einops.rearrange`
    or `einops.repeat`.
6.  **Interface & naming preserved** –  Class name `SpikingFLABlock`, standard
    constructor params, and forward signature are fully retained.

The resulting architecture opens a brand-new design avenue where spiking
modules provide **multi-scale summaries** rather than acting as per-token gates
or linear replacements inside monolithic paths.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import RMSNorm
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError:  # pragma: no cover
    raise ImportError("Spiking neuron modules (src.models.spike) not found.")

# ---------------------------------------------------------------------------
# SpikingLinear helper (MUST stay identical to previous definition)
# ---------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear Module preceded by a spiking neuron."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B, L, D) → neuron expects (B, D, L)
        spikes_t = self.neuron(x.transpose(-1, -2))
        spikes = spikes_t.transpose(-1, -2)  # back to (B, L, D)
        out = self.linear(spikes)
        return out, spikes

# ---------------------------------------------------------------------------
# Simple 2-layer Spiking Feed-Forward Network
# ---------------------------------------------------------------------------
class SpikingFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float, *, neuron_type: str):
        super().__init__()
        surrogate = piecewise_quadratic_surrogate()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn, bias=False), neuron_type, surrogate)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model, bias=False), neuron_type, surrogate)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes: List[torch.Tensor] = []
        x, s1 = self.fc1(x)
        spikes.append(s1)
        x = self.act(x)
        x = self.drop1(x)
        x, s2 = self.fc2(x)
        spikes.append(s2)
        x = self.drop2(x)
        return x, spikes

# ---------------------------------------------------------------------------
# Hierarchical Temporal Pyramid Mixer
# ---------------------------------------------------------------------------
class TemporalPyramidMixer(nn.Module):
    """Causal coarse-to-fine pyramid aggregation with spiking projections."""

    def __init__(self, d_model: int, strides: Tuple[int, ...], *, neuron_type: str):
        super().__init__()
        surrogate = piecewise_quadratic_surrogate()
        self.strides = strides
        # fine-scale projection
        self.fine_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
        # one spiking projection per coarse level
        self.coarse_projs = nn.ModuleList([
            SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate)
            for _ in strides
        ])
        # learnable residual weights (initialised small)
        self.alpha = nn.Parameter(torch.zeros(len(strides)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return fused representation and list of spike tensors."""
        B, L, _ = x.shape
        spikes_collected: List[torch.Tensor] = []

        # Fine-scale processing
        fine, s_fine = self.fine_proj(x)
        spikes_collected.append(s_fine)
        out = fine

        # Iterate over pyramid levels
        for idx, S in enumerate(self.strides):
            if L < S:  # skip levels larger than sequence
                continue
            # causal down-sampling: take the **first** token in each window of length S
            n_windows = math.ceil(L / S)
            win_idx = torch.arange(n_windows, device=x.device)
            token_idx = win_idx * S  # first index in each window (strictly <= current position)
            token_idx = token_idx.clamp_max(L - 1)  # ensure valid indices

            # gather tokens
            coarse_tokens = rearrange(x, "b l d -> b d l")  # easier gather along last dim
            coarse_tokens = torch.gather(
                coarse_tokens,
                -1,
                token_idx[None, None, :].expand(B, x.shape[-1], -1),
            )
            coarse_tokens = rearrange(coarse_tokens, "b d n -> b n d")  # (B, N, D)

            # spiking projection on coarse sequence
            coarse_feat, s_c = self.coarse_projs[idx](coarse_tokens)
            spikes_collected.append(s_c)

            # upsample back – repeat each coarse token S times, then trim to L
            coarse_up = repeat(coarse_feat, "b n d -> b (n S) d", S=S)
            coarse_up = coarse_up[:, :L, :]

            out = out + self.alpha[idx] * coarse_up

        return out, spikes_collected

# ---------------------------------------------------------------------------
# Main Block – preserves external interface
# ---------------------------------------------------------------------------
class SpikingFLABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,  # retained for signature compatibility (unused)
        *,
        strides: Tuple[int, ...] = (2, 4, 8, 16),
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        **kwargs,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.mixer = TemporalPyramidMixer(d_model, strides, neuron_type=neuron_type)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SpikingFFN(d_model, d_model * d_ffn_mult, dropout, neuron_type=neuron_type)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs):  # noqa: D401
        spikes_total: List[torch.Tensor] = []

        residual = x
        if self.prenorm:
            x = self.norm1(x)
        x, lvl_spikes = self.mixer(x)
        spikes_total.extend(lvl_spikes)
        x = residual + self.drop1(x)

        residual = x
        if self.prenorm:
            x = self.norm2(x)
        x_ffn, ffn_spikes = self.ffn(x)
        spikes_total.extend(ffn_spikes)
        x = residual + self.drop2(x_ffn)

        # concatenate all spike tensors into a flat 1-D tensor
        final_spikes = torch.cat([s.flatten() for s in spikes_total], dim=0)
        return x, None, final_spikes
