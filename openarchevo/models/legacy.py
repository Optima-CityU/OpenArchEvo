"""Shims for the ``src.models.*`` modules that archived programs import.

Programs from the original search import a handful of helpers from the research
code base (a fork of the SpikingSSM / state-spaces ``src`` tree). This module
provides drop-in replacements so that archived programs in ``zoo/`` load
unchanged; :func:`openarchevo.models.program.install_legacy_spike_alias`
registers them under their original module paths:

========================================  ==========================================
original import                           provided by
========================================  ==========================================
``src.models.spike.neuron.registry``      :data:`openarchevo.models.neurons.registry`
``src.models.spike.surrogate.*``          :mod:`openarchevo.models.neurons`
``src.models.nn.dropout.DropoutNd``       :class:`DropoutNd` (below)
``src.models.spike.nn.SpikingLinear``     :class:`LegacySpikingLinear` (opt-in, below)
========================================  ==========================================

``src.models.spike.nn`` did not exist in the research code base that evaluated
the archive, so the seven archived programs that import it ran their own
in-file fallback definitions. It is therefore *not* registered by default; pass
``legacy_spike_nn=True`` to :func:`~openarchevo.models.program.load_block_class`
to run those programs with the package's neurons instead.
"""

# DropoutNd is adapted from the state-spaces S4 repository
# (https://github.com/state-spaces/s4, src/models/nn/dropout.py), Apache-2.0,
# as vendored by SpikingSSMs (MIT). See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import nn

from openarchevo.models.spiking_linear import SpikingLinear


class DropoutNd(nn.Module):
    """Dropout whose mask is tied across the sequence axis (S4 ``DropoutNd``).

    ``X``: ``(batch, dim, lengths...)`` if ``transposed`` else ``(batch, lengths..., dim)``.
    """

    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0, 1), but got {p}")
        self.p, self.tie, self.transposed = p, tie, transposed

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return X
        if not self.transposed:
            X = X.movedim(-1, 1)
        mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
        mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
        X = X * mask * (1.0 / (1 - self.p))
        if not self.transposed:
            X = X.movedim(1, -1)
        return X


class LegacySpikingLinear(SpikingLinear):
    """``SpikingLinear`` that also accepts the ``(in_features, out_features, ...)`` form.

    Archived programs call it either as ``SpikingLinear(nn.Linear(...), neuron_type,
    surrogate_function)`` or as ``SpikingLinear(d_in, d_out, bias=..., neuron_type=...,
    surrogate_function=...)``. Both return ``(output, spikes)``.
    """

    def __init__(self, linear_layer: Union[nn.Linear, int], *args, bias: bool = True,
                 neuron_type: Optional[str] = None, surrogate_function: Optional[Callable] = None,
                 **kwargs):
        if isinstance(linear_layer, int):
            out_features, *rest = args
            linear_layer = nn.Linear(linear_layer, out_features, bias=bias)
            args = tuple(rest)
        if args:
            neuron_type = args[0]
        if len(args) > 1:
            surrogate_function = args[1]
        super().__init__(linear_layer, neuron_type or "sdn", surrogate_function)
