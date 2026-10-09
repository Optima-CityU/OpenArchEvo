"""The program contract: what every architecture program must provide.

This is the single place that defines the search space's interface. Everything
that checks, builds or runs a program imports it from here:

* ``inner_loop/search.py`` (``static_check``) and ``surrogate/fake.py`` check
  that the class exists;
* ``models/program.py`` (``load_block_class``) loads it;
* ``models/wrapper.py`` (the probing / training model) and
  ``fingerprint/extract.py`` (the single-block trace) build it with
  :func:`block_kwargs`;
* ``models/wrapper.py``, ``fingerprint/probes.py`` and ``training/`` consume
  the forward output described below.

The contract used in the paper::

    class SpikingFLABlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ffn_mult, dropout, neuron_type,
                     prenorm=True, use_spiking_mlp=True, layer_idx=None, **kwargs): ...

        def forward(self, x):            # x: (B, L, d_model)
            return y, reg_loss, spikes
            # y:        (B, L, d_model)
            # reg_loss: scalar tensor added to the training loss, or None
            # spikes:   1-D tensor of all binary spikes emitted in the block, or None

To change the search space, edit this module, then the prompt templates in
``llm/templates/`` (which describe the contract to the LLM in words) and, if
the forward output changes, ``SequenceModel.forward`` in ``models/wrapper.py``.
This module has no dependencies, so torch-free code can import it.
"""

from __future__ import annotations

from typing import Any, Optional

#: Name of the class every program must define.
BLOCK_CLASS = "SpikingFLABlock"

#: Constructor arguments that are fixed for every program in this search space.
FIXED_KWARGS: dict[str, Any] = {"prenorm": True, "use_spiking_mlp": True}

#: Number of values returned by ``forward``: ``(y, reg_loss, spikes)``.
FORWARD_OUTPUTS = ("y", "reg_loss", "spikes")


def block_kwargs(d_model: int, num_heads: int, d_ffn_mult: int, dropout: float, neuron_type: str,
                 layer_idx: Optional[int] = None, **extra: Any) -> dict[str, Any]:
    """Constructor keyword arguments for one block (``layer_idx`` only when given)."""
    kwargs: dict[str, Any] = dict(d_model=d_model, num_heads=num_heads, d_ffn_mult=d_ffn_mult, dropout=dropout,
                                  neuron_type=neuron_type, **FIXED_KWARGS)
    if layer_idx is not None:
        kwargs["layer_idx"] = layer_idx
    kwargs.update(extra)
    return kwargs
