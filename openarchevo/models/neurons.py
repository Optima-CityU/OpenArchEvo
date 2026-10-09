# Adapted from the spiking-neuron code of SpikingSSMs / SDN (Shen et al., AAAI 2025),
# https://github.com/shenshuaijie/SDN. The upstream copyright and license terms apply
# to the adapted parts; see THIRD_PARTY_NOTICES.md.
"""Spiking neurons and the surrogate gradient.

All neurons take a tensor whose **last** dimension is time (sequence position),
e.g. ``(B, D, L)``, and return binary spikes of the same shape.

* ``bptt`` - hard-reset LIF unrolled over time, gradients through the membrane:
  ``v~_t = lambda * v_{t-1} + I_t;  s_t = H(v~_t - theta);  v_t = (1 - s_t) v~_t``.
* ``sltt`` - the same neuron with the membrane detached between steps (cheaper).
* ``sdn``  - the surrogate dynamic network of SpikingSSM (Shen et al., 2025), a
  small pretrained TorchScript model that predicts LIF membrane potentials for a
  whole sequence in parallel. Used during search and evaluation in the paper.
  The checkpoint is not bundled: obtain it from the SpikingSSM authors' release
  (https://github.com/shenshuaijie/SDN) and set ``OPENARCHEVO_SDN_PATH``.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from torch import nn


class PiecewiseQuadratic(torch.autograd.Function):
    """Heaviside forward, triangular (piecewise-quadratic) surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.requires_grad:
            ctx.save_for_backward(x)
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        x_abs = x.abs()
        return (grad_output * (1.0 - x_abs)).masked_fill_(x_abs > 1, 0)


def piecewise_quadratic_surrogate() -> Callable[[torch.Tensor], torch.Tensor]:
    return PiecewiseQuadratic.apply


class LIFNeuron(nn.Module):
    """Hard-reset LIF neuron, unrolled over the last dimension."""

    detach_membrane = False

    def __init__(self, surrogate_function: Optional[Callable] = None, decay: float = 0.125,
                 threshold: float = 1.0, reset: float = 0.0):
        super().__init__()
        self.surrogate_function = surrogate_function or piecewise_quadratic_surrogate()
        self.decay, self.threshold, self.reset = decay, threshold, reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.zeros_like(x[..., 0])
        spikes = []
        for t in range(x.size(-1)):
            v = (v.detach() if self.detach_membrane else v) * self.decay + x[..., t]
            s = self.surrogate_function(v - self.threshold)
            spikes.append(s)
            s_ = s.detach()
            v = (1 - s_) * v + s_ * self.reset
        return torch.stack(spikes, dim=-1)


class BPTTNeuron(LIFNeuron):
    detach_membrane = False


class SLTTNeuron(LIFNeuron):
    detach_membrane = True


class SDNNeuron(nn.Module):
    """LIF neuron whose membrane potentials are predicted by a pretrained SDN."""

    def __init__(self, surrogate_function: Optional[Callable] = None):
        super().__init__()
        path = os.environ.get("OPENARCHEVO_SDN_PATH")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                "neuron_type='sdn' needs the SpikingSSM SDN TorchScript checkpoint; set OPENARCHEVO_SDN_PATH "
                "(see https://github.com/shenshuaijie/SDN) or use neuron_type='bptt'."
            )
        self.surrogate_function = surrogate_function or piecewise_quadratic_surrogate()
        self.model = torch.jit.load(path).eval()

    @torch.no_grad()
    def predict_membrane(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        return self.model(x.detach().contiguous().view(-1, 1, length)).view(x.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.surrogate_function(self.predict_membrane(x) + x - 1.0)


#: ``registry[name](surrogate_function=...)`` builds a neuron.
registry: dict[str, type[nn.Module]] = {"bptt": BPTTNeuron, "sltt": SLTTNeuron, "sdn": SDNNeuron}


def build_neuron(name: str, surrogate_function: Optional[Callable] = None) -> nn.Module:
    if name not in registry:
        raise KeyError(f"Unknown neuron type '{name}'. Available: {sorted(registry)}")
    return registry[name](surrogate_function=surrogate_function)
