# -*- coding: utf-8 -*-
"""
spiking_fla_gateconv – Causal Gate-Conv Spiking FLA Block
========================================================
Identifier: spiking_fla_gateconv

This evolution stream-lines the former DeltaNet based designs into a **fully
spiking, linear-time Gate-Conv block** that obeys every hard constraint of the
mission:

1. **Every nn.Linear is wrapped by SpikingLinear** – inputs are binary spikes.
2. **Flat `final_spikes` tensor** is returned for *all* spiking neurons in the
   block (gate/value projections + 2-layer FFN).
3. **Causal, sub-quadratic O(N) complexity** – sequence mixing is achieved by a
   *depth-wise* causal convolution (kernel≪N) plus element-wise gating.
4. **Batch-size agnostic & einops-driven** – shapes are inferred at runtime and
   all reshapes use `einops.rearrange`.
5. **Class & interface preserved** – the public API (`SpikingFLABlock.__init__`,
   `forward`) is unchanged, so the layer remains a drop-in replacement for
   previous variants.

Conceptually, the block performs two stages:
------------------------------------------------
1. **Gate-Conv Mixing (stateful memory)**
   Hidden state is split into a *gate* (sigmoid) and *value* path, both produced
   by `SpikingLinear`.  The gate modulates the value, which is *added* to a
   depth-wise causal convolution of the *original* hidden state.  This yields a
   lightweight, recurrent-like memory similar to Mamba/Hyena but fully linear
   in \(N\) and compatible with binary spikes.

2. **Spiking Feed-Forward Network**
   A standard 2-layer SiLU FFN implemented with `SpikingLinear` layers provides
   extra non-linearity and capacity.  All spike tensors from stage 1 & 2 are
   concatenated into the required `final_spikes` output.

No attention matrix, soft-max or quadratic operation is used.  The design keeps
parameter count minimal yet addresses the *path-starvation* and *overly dense
firing* issues observed in previous experiments by employing explicit gating
and depth-wise convolution for local context while maintaining global identity
residuals.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm  # type: ignore

try:
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Spiking neuron modules are required but not found. "
        "Ensure `src.models.spike` is on PYTHONPATH." 
    ) from exc

# ---------------------------------------------------------------------------
# SpikingLinear (must stay exactly identical to legacy definition) ----------
# ---------------------------------------------------------------------------

class SpikingLinear(nn.Module):
    """A Linear Module preceded by a spiking neuron layer.

    The wrapped `linear_layer` receives *binary* spikes as its input.  Both the
    linear output and the raw spike tensor are returned so callers can collect
    spike statistics.
    """

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor):  # -> Tuple[Tensor, Tensor]
        # transpose to make features the channel dimension for the spiking neuron
        spikes_t = self.neuron(x.transpose(-1, -2))  # (B, D, L)
        spikes = spikes_t.transpose(-1, -2)          # (B, L, D)
        out = self.linear(spikes)                    # (B, L, *)
        return out, spikes

# ---------------------------------------------------------------------------
# Spiking FLA Block ---------------------------------------------------------
# ---------------------------------------------------------------------------

class SpikingFLABlock(nn.Module):
    """Fully-spiking Gate-Conv Feed-Forward residual block (O(N)).

    Parameters
    ----------
    d_model: int
        Model dimensionality.
    num_heads: int
        *Unused* but kept for API compatibility.
    use_spiking_mlp: bool
        *Unused* – the block is intrinsically spiking; kept for compatibility.
    d_ffn_mult: int
        Expansion factor for the hidden layer of the FFN.
    dropout: float
        Drop-out probability applied after each residual branch.
    prenorm: bool
        If *True*, RMSNorm is applied *before* each sub-layer (Post-norm if
        False).
    neuron_type: str
        Spiking neuron type registered in `src.models.spike.neuron.registry`.
    **kwargs: Any
        Extra keyword arguments are accepted for forward-compatibility and are
        ignored.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,  # pylint: disable=unused-argument
        use_spiking_mlp: bool = True,  # kept for interface parity
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        depthwise_kernel: int = 5,
        **kwargs,  # noqa: D401, pylint: disable=unused-argument
    ) -> None:
        super().__init__()

        surrogate_fn = piecewise_quadratic_surrogate()
        self.prenorm = bool(prenorm)

        # ------ Stage 0 : Normalisations ---------------------------------
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # ------ Stage 1 : Gate-Conv Mixer --------------------------------
        self.gate_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_fn)
        self.value_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_fn)

        # Depth-wise causal convolution (groups=d_model)  – O(N * k)
        self.dw_kernel = int(depthwise_kernel)
        padding = self.dw_kernel - 1  # left padding for causality
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=self.dw_kernel,
            groups=d_model,
            bias=False,
            padding=0,  # we pad manually to keep strictly causal behaviour
        )

        # ------ Stage 2 : Spiking Feed-Forward ---------------------------
        hidden_ff = d_model * d_ffn_mult
        self.ff1 = SpikingLinear(nn.Linear(d_model, hidden_ff, bias=False), neuron_type, surrogate_fn)
        self.ff2 = SpikingLinear(nn.Linear(hidden_ff, d_model, bias=False), neuron_type, surrogate_fn)

        # ------ Dropouts --------------------------------------------------
        self.drop_resid = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):  # pylint: disable=unused-argument
        """Forward pass.

        Returns
        -------
        Tuple[Tensor, None, Tensor]
            • Output hidden states `(B, L, D)`
            • `reg_loss` placeholder (None)
            • `final_spikes` – flat tensor concatenating *all* spike outputs
        """
        spike_collector = []  # collect spike tensors for the final output

        # ============ Stage 1 : Gate-Conv Mixer ============
        residual = x
        if self.prenorm:
            x = self.norm1(x)

        # Spiking projections
        gate_logits, sp_gate = self.gate_proj(x)   # (B, L, D)
        value_act, sp_value = self.value_proj(x)   # (B, L, D)
        spike_collector.extend([sp_gate, sp_value])

        # Depth-wise causal convolution
        x_t = rearrange(x, "b l d -> b d l")                # (B, D, L)
        x_pad = F.pad(x_t, (self.dw_kernel - 1, 0))          # left pad (K-1, 0)
        conv_out = self.dw_conv(x_pad)                       # (B, D, L)
        conv_out = rearrange(conv_out, "b d l -> b l d")    # back to (B, L, D)

        mixer_out = torch.sigmoid(gate_logits) * value_act + conv_out
        x = residual + self.drop_resid(mixer_out)

        # ============ Stage 2 : Spiking Feed-Forward ============
        residual_ff = x
        if self.prenorm:
            x = self.norm2(x)

        hidden_ff, sp_ff1 = self.ff1(x)
        hidden_ff = F.silu(hidden_ff)
        ff_out, sp_ff2 = self.ff2(hidden_ff)
        spike_collector.extend([sp_ff1, sp_ff2])

        x = residual_ff + self.drop_ff(ff_out)

        # ============ Spike aggregation ============
        # Flatten & concatenate all collected spike tensors into one vector.
        final_spikes = torch.cat([s.flatten() for s in spike_collector], dim=0)

        return x, None, final_spikes
