"""``SpikingLinear``: the unit that enforces the spiking-projection constraint.

Parameter-dominant projections must receive binary spikes. ``SpikingLinear``
passes its input through a spiking neuron (time = sequence axis) and then a
linear layer, and returns both the projection and the spikes so that the block
can report them (firing rate, SWSP).

The constructor signature matches the one that evolved programs were asked to
keep unchanged: ``SpikingLinear(nn.Linear(d_in, d_out), neuron_type, surrogate_function)``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from openarchevo.models.neurons import build_neuron


class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str = "bptt",
                 surrogate_function: Optional[Callable] = None):
        super().__init__()
        self.neuron = build_neuron(neuron_type, surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``(B, L, D_in)`` -> ``(output (B, L, D_out), spikes (B, L, D_in))``."""
        spikes = self.neuron(x.transpose(-1, -2)).transpose(-1, -2)
        return self.linear(spikes), spikes
