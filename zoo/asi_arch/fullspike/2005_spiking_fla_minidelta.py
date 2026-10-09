# -*- coding: utf-8 -*-
"""
spiking_fla_minidelta – A fully-spiking, chunk-wise associative block with binary-spike
              projections and simplified mixture routing.

Key innovations
---------------
1. Ubiquitous SpikingLinear
   Every affine projection (q/k/v, beta, o, gate, MLP) is constructed by
   wrapping an `nn.Linear` with the provided `SpikingLinear` shim.  The wrapper
   inserts a spiking neuron in front of the dense operator so **all linear
   layers consume binary spikes** as mandated.

2. Spike Trace Propagation
   All intermediate spike tensors are collected during the forward pass.  The
   block outputs a single flat `final_spikes` tensor which concatenates every
   binary spike produced inside the module – guaranteeing compliance with the
   specified output contract.

3. Lightweight Delta-Memory Mixer
   The original DeltaNet variants use sophisticated gated mixtures with many
   dense layers.  To keep this reference implementation concise while still
   adhering to sub-quadratic complexity, we retain the efficient `delta_rule_
   chunkwise` associative memory and fuse it with the direct value path through
   a **learnable per-head softmax gate** (two trainable parameters per head).
   This removes several dense layers yet preserves the critical ability to
   trade off local (value) vs. global (delta memory) information.

4. End-to-End Spiking MLP
   A compact two-layer feed-forward network is implemented using `SpikingLinear`
   at each layer so that even the non-memory portion of the block adheres to
   the binary-input constraint.

5. Strict Causality & O(N) Complexity
   The chunked Delta rule is identical to the proven O(N) algorithm used in
   earlier experiments.  No softmax attention is introduced, preserving the
   desired sub-quadratic asymptotic behaviour and causal information flow.

This implementation purposefully focuses on the **minimal viable fully-spiking
Delta block** – a solid foundation for subsequent research iterations where
more elaborate gating or multi-scale paths can be added without violating the
binary-input guarantee.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules.l2norm import l2norm  # efficient unit-norm helper
from fla.modules import RMSNorm  # normalisation remains analogue

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "Spiking neuron modules (src.models.spike) must be available for "
        "spiking_fla_minidelta."
    ) from _exc

# --------------------------------------------------
#  Base spiking wrapper (MUST stay unmodified)
# --------------------------------------------------
class SpikingLinear(nn.Module):
    """A Linear module with a pre-synaptic spiking neuron."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert features to spikes along the channel dimension
        spikes = self.neuron(x.transpose(-1, -2))  # (B,D,L)
        spikes_t = spikes.transpose(-1, -2)  # back to (B,L,D)
        out = self.linear(spikes_t)
        return out, spikes_t  # spikes in (B,L,D) format for convenience


# --------------------------------------------------
#  Chunk-wise causal Δ-rule (unchanged O(N) kernel)
# --------------------------------------------------
# NOTE: We purposefully keep the kernel identical to prior versions to
#       guarantee efficiency and correctness.

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B,H,L,Dk)
    k: torch.Tensor,  # (B,H,L,Dk)
    v: torch.Tensor,  # (B,H,L,Dv)
    beta: torch.Tensor,  # (B,H,L)
    *,
    chunk_size: int = 32,
):
    """Efficient associative retrieval with linear-time complexity."""
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    q = l2norm(q)
    k = l2norm(k)

    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_full = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_full, 0)
    for i in range(1, chunk_size):
        inv[..., i, :i] += (inv[..., i, :, None].clone() * inv[..., :, :i].clone()).sum(-2)
    inv = inv + torch.eye(chunk_size, dtype=inv.dtype, device=q.device)

    u = inv @ v
    w = inv @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S


# --------------------------------------------------
#  Fully-Spiking Delta memory block (attention replacement)
# --------------------------------------------------
class MiniSpikeDelta(nn.Module):
    """A minimal DeltaNet variant with fully-spiking projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_beta: bool = True,
        neuron_type: str = "sdn",
        surrogate_function=piecewise_quadratic_surrogate(),
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.use_beta = bool(use_beta)

        self.key_dim = int(d_model * expand_k)
        self.val_dim = int(d_model * expand_v)
        assert self.key_dim % self.num_heads == 0 and self.val_dim % self.num_heads == 0
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.val_dim // self.num_heads

        # Spiking projections ------------------------------------------------
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate_function)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.val_dim, bias=False), neuron_type, surrogate_function)
        if self.use_beta:
            self.b_proj = SpikingLinear(nn.Linear(d_model, self.num_heads, bias=False), neuron_type, surrogate_function)

        # simple two-path learnable gate (delta vs. value) -------------------
        self.gate_logits = nn.Parameter(torch.zeros(self.num_heads, 2))

        # Output projection --------------------------------------------------
        self.o_proj = SpikingLinear(nn.Linear(self.val_dim, d_model, bias=False), neuron_type, surrogate_function)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return (output, list_of_spike_tensors)."""
        spike_bank: List[torch.Tensor] = []

        # Projections (with spike capture) -----------------------------------
        q_lin, q_spk = self.q_proj(hidden_states)
        k_lin, k_spk = self.k_proj(hidden_states)
        v_lin, v_spk = self.v_proj(hidden_states)
        spike_bank.extend([q_spk, k_spk, v_spk])

        if self.use_beta:
            beta_raw, beta_spk = self.b_proj(hidden_states)
            beta = beta_raw.sigmoid()
            spike_bank.append(beta_spk)
        else:
            beta = hidden_states.new_ones((*hidden_states.shape[:2], self.num_heads))

        # Head split ---------------------------------------------------------
        q = rearrange(q_lin, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k_lin, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v_lin, "b l (h d) -> b h l d", h=self.num_heads)
        beta_h = rearrange(beta, "b l h -> b h l")

        # Delta memory -------------------------------------------------------
        delta_out, _ = delta_rule_chunkwise(q, k, v, beta_h)
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # Direct value path --------------------------------------------------
        value_out = rearrange(v, "b h l d -> b l h d")  # same shape as delta_out

        # Per-head softmax gate ---------------------------------------------
        w = torch.softmax(self.gate_logits, dim=-1)  # (H,2)
        w_delta = w[:, 0].view(1, 1, self.num_heads, 1)
        w_value = w[:, 1].view(1, 1, self.num_heads, 1)

        out_mix = w_delta * delta_out + w_value * value_out  # (B,L,H,Dv)
        out_mix_flat = rearrange(out_mix, "b l h d -> b l (h d)")

        # Output projection --------------------------------------------------
        out_final, o_spk = self.o_proj(out_mix_flat)
        spike_bank.append(o_spk)

        return out_final, spike_bank


# --------------------------------------------------
#  Fully-Spiking two-layer MLP
# --------------------------------------------------
class SpikingMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
        surrogate_function=piecewise_quadratic_surrogate(),
    ) -> None:
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spike_bank: List[torch.Tensor] = []
        h1, spk1 = self.fc1(x)
        h1_act = F.gelu(h1)
        h1_act = self.dropout(h1_act)
        h2, spk2 = self.fc2(h1_act)
        spike_bank.extend([spk1, spk2])
        return h2, spike_bank


# --------------------------------------------------
#  Spiking FLA Block (public entry point)
# --------------------------------------------------
class SpikingFLABlock(nn.Module):
    """Fully-spiking FLA block combining Mini-Delta memory and MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        use_spiking_mlp: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.prenorm = bool(prenorm)
        self.norm1 = RMSNorm(d_model)
        self.attn = MiniSpikeDelta(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs):  # type: ignore[override]
        spike_bank: List[torch.Tensor] = []

        # ---- Attention / Delta memory ------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm1(x)
        attn_out, attn_spikes = self.attn(x)
        spike_bank.extend(attn_spikes)
        x = residual + self.dropout1(attn_out)

        # ---- Feed-forward --------------------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm2(x)
        mlp_out, mlp_spikes = self.mlp(x)
        spike_bank.extend(mlp_spikes)
        x = residual + self.dropout2(mlp_out)

        # ---- Flatten spikes ----------------------------------------------
        if spike_bank:
            final_spikes = torch.cat([s.flatten() for s in spike_bank], dim=0)
        else:  # fallback (should not happen)
            final_spikes = torch.empty(0, device=x.device, dtype=x.dtype)

        # No regularisation loss in this minimal variant
        reg_loss = None
        return x, reg_loss, final_spikes


# --------------------------------------------------
#  Optional typing helper for external static analysers
# --------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache
