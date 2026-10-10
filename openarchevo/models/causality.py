"""Adapt the unchanged experiment causality checker to configured block constructors."""

from __future__ import annotations

from typing import Any, Optional

import torch

from openarchevo.causality import assert_block_class_causal


def check_block_causality(block_cls, *, device: str | torch.device, label: str,
                          d_model: int = 256, num_heads: int = 8, d_ffn_mult: int = 4,
                          neuron_type: str = "sdn", extra_block_kwargs: Optional[dict[str, Any]] = None) -> float:
    """Run the service's probes, adapting only device and block constructor settings.

    Search uses the experiment's LM-sized defaults. Training supplies its model
    configuration. The checker always disables dropout and restores a fresh
    block for each forward, without touching the model that will be trained.
    """
    def build_block(**kwargs):
        kwargs.update(num_heads=num_heads, d_ffn_mult=d_ffn_mult, neuron_type=neuron_type)
        kwargs.update(extra_block_kwargs or {})
        return block_cls(**kwargs)

    return assert_block_class_causal(build_block, device=torch.device(device), label=label, d_model=d_model)
