# -*- coding: utf-8 -*-
"""
spiking_fla_recdeltanet
=======================
A minimal yet effective fully-spiking sequence block derived from the DeltaNet
family.  Key goals of this evolutionary step are:

1. *Pure-Spiking Linear Layers* – every affine transform is wrapped by
   `SpikingLinear`, guaranteeing binary spike inputs to the weight matrix and
   satisfying the SNN design constraints.
2. *Sub-Quadratic Recurrent Memory* – global dependencies are handled by the
   proven **delta-rule associative scan** (`delta_rule_chunkwise`) which runs in
   **O(N) time** thanks to fixed-size chunking.
3. *Lean Architecture* – we remove super-fluous branches (HSM, large
   convolutions, multi-path gates) to keep the implementation concise while
   still demonstrating long-range context handling through the delta pathway
   and local detail preservation via the identity/value path.  A per-head
   learnable gate balances the two.
4. *Unified Spike Reporting* – every spike tensor generated inside the block
   (Q/K/V/O projections + MLP) is gathered and flattened into a single
   `final_spikes` tensor returned by `SpikingFLABlock.forward` as required.

The new block keeps full interface compatibility – `SpikingFLABlock` remains the
entry point and its forward method still accepts arbitrary `**kwargs`.  Internal
operations use `einops.rearrange` exclusively for shape manipulations ensuring
full batch/sequence agnosticism.  No operation introduces super-linear
complexity.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------------------------------------------------------
# External utilities from the existing code base (imports preserved as required)
# -----------------------------------------------------------------------------
from fla.modules import RMSNorm  # type: ignore
from fla.modules.l2norm import l2norm  # type: ignore

try:
    from src.models.spike.neuron import registry as neuron_registry  # type: ignore
    from src.models.spike.surrogate import piecewise_quadratic_surrogate  # type: ignore
except ImportError:
    # Soft-fail so the file can still be imported (e.g. for documentation or
    # type-checking on systems without the spiking modules).  Instantiation will
    # raise a clear error later if a user actually tries to build the model
    # without the required dependencies.
    print("Warning: Spiking neuron modules (src.models.spike) not found.")
    neuron_registry = None
    piecewise_quadratic_surrogate = None  # type: ignore

# -----------------------------------------------------------------------------
# SpikingLinear – MUST remain identical to the original definition (minus the
# surrogate-initialisation safeguard handled outside).
# -----------------------------------------------------------------------------

class SpikingLinear(nn.Module):
    """A Linear layer preceded by a spiking neuron."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function: Callable):
        super().__init__()
        if neuron_registry is None or surrogate_function is None:
            raise ImportError(
                "Spiking neuron modules are not available – make sure `src.models.spike` "
                "is on the PYTHONPATH and all dependencies are installed."
            )
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Incoming activations -> spikes (B,L,D) → (B,D,L) for neuron, then back.
        spikes_t = self.neuron(x.transpose(-1, -2))  # (B,D,L)
        spikes = spikes_t.transpose(-1, -2)  # (B,L,D)
        out = self.linear(spikes)
        return out, spikes

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:  # shifted ELU(+1)
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def _sum_norm(x: torch.Tensor) -> torch.Tensor:  # simple sum-normalisation
    return (x / x.sum(dim=-1, keepdim=True)).to(x)

# -----------------------------------------------------------------------------
# Chunk-wise delta rule – identical numerics to previous experiments
# -----------------------------------------------------------------------------

def delta_rule_chunkwise(
    q: torch.Tensor,  # (B H L Dk)
    k: torch.Tensor,  # (B H L Dk)
    v: torch.Tensor,  # (B H L Dv)
    beta: torch.Tensor,  # (B H L)
    *,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal associative scan with fixed chunk size – O(N) complexity.

    NOTE: The original implementation was wrapped with `@torch.no_grad()`,
    which unintentionally disabled gradient propagation through the delta-path.
    The decorator has been removed to ensure that the model remains fully
    trainable while keeping the exact arithmetic intact.
    """

    b, h, L, d_k = q.shape
    # ------------------------------------------------------------------
    # Pad so that sequence length is multiple of chunk_size
    # ------------------------------------------------------------------
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad_cfg = (0, 0, 0, pad_len)
        q, k, v = (F.pad(t, pad_cfg) for t in (q, k, v))
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len

    # ------------------------------------------------------------------
    # Normalisation & preparation
    # ------------------------------------------------------------------
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape to (B H N C D)
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )

    tri_mask_inc = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )

    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri_mask_inc, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] += (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=q.dtype, device=q.device)

    u = attn @ v
    w = attn @ k_beta

    S = k.new_zeros(b, h, d_k, v.shape[-1])
    o = torch.zeros_like(v)

    tri_mask_exc = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )

    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_mask_exc, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        o[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    o = rearrange(o, "b h n c d -> b h (n c) d")
    if pad_len:
        o = o[:, :, :L]
    return o, S

# -----------------------------------------------------------------------------
# Minimal DeltaNet variant with fully-spiking projections and a per-head gate
# -----------------------------------------------------------------------------

class DeltaNetSpike(nn.Module):
    """Simple two-path (delta + identity) DeltaNet block with spiking projections."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        neuron_type: str = "sdn",
        surrogate_function: Optional[Callable] = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.chunk_size = chunk_size

        # ------------------------------------------------------------------
        # Resolve surrogate function – fall back to default implementation if
        # the caller did not provide one.
        # ------------------------------------------------------------------
        if surrogate_function is None:
            if piecewise_quadratic_surrogate is None:
                raise ImportError(
                    "`piecewise_quadratic_surrogate` could not be imported and no "
                    "custom surrogate_function was provided."
                )
            surrogate_function = piecewise_quadratic_surrogate()

        # Spiking projections -------------------------------------------------
        self.q_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function
        )
        self.k_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function
        )
        self.v_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function
        )
        self.o_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function
        )

        # Learnable per-head gate (sigmoid) weight for delta path
        self.gate_logit = nn.Parameter(torch.zeros(num_heads))

        # Placeholder for spikes produced during the forward pass
        self.last_spikes: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,D)
        self.last_spikes.clear()

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        self.last_spikes.extend([s_q, s_k, s_v])

        # Head split ---------------------------------------------------
        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads)

        # Activation / normalisation ----------------------------------
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = _elu_plus_one(q), _elu_plus_one(k)
        if self.qk_norm == "sum":
            q, k = _sum_norm(q), _sum_norm(k)

        # Beta = 1 (no scaling) ---------------------------------------
        beta = torch.ones_like(q[..., 0])

        # Delta rule global memory ------------------------------------
        q_g = rearrange(q, "b l h d -> b h l d")
        k_g = rearrange(k, "b l h d -> b h l d")
        v_g = rearrange(v, "b l h d -> b h l d")
        beta_g = rearrange(beta, "b l h -> b h l")
        delta_out, _ = delta_rule_chunkwise(
            q_g, k_g, v_g, beta_g, chunk_size=self.chunk_size
        )
        delta_out = rearrange(delta_out, "b h l d -> b l h d")

        # Identity path (value)
        v_direct = v

        # Per-head gate ------------------------------------------------
        gate = torch.sigmoid(self.gate_logit).view(1, 1, self.num_heads, 1)
        out = gate * delta_out + (1.0 - gate) * v_direct

        # Merge heads and output projection ---------------------------
        out = rearrange(out, "b l h d -> b l (h d)")
        out, s_o = self.o_proj(out)
        self.last_spikes.append(s_o)
        return out

# -----------------------------------------------------------------------------
# Fully-spiking MLP (2-layer, GELU) – minimal and efficient
# -----------------------------------------------------------------------------

class SpikingMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        neuron_type: str = "sdn",
        surrogate_function: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        # Resolve surrogate exactly like in DeltaNetSpike
        if surrogate_function is None:
            if piecewise_quadratic_surrogate is None:
                raise ImportError(
                    "`piecewise_quadratic_surrogate` could not be imported and no "
                    "custom surrogate_function was provided."
                )
            surrogate_function = piecewise_quadratic_surrogate()

        self.fc1 = SpikingLinear(
            nn.Linear(d_model, d_ff, bias=False), neuron_type, surrogate_function
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(
            nn.Linear(d_ff, d_model, bias=False), neuron_type, surrogate_function
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_collector: List[torch.Tensor] = []
        x, s1 = self.fc1(x)
        spikes_collector.append(s1)
        x = self.act(x)
        x = self.dropout(x)
        x, s2 = self.fc2(x)
        spikes_collector.append(s2)
        return x, spikes_collector

# -----------------------------------------------------------------------------
# SpikingFLABlock – public facing block keeping interface intact
# -----------------------------------------------------------------------------

class SpikingFLABlock(nn.Module):
    """A transformer-like block with fully-spiking internals."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_spiking_mlp: bool = True,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        prenorm: bool = True,
        neuron_type: str = "sdn",
        **kwargs,
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = DeltaNetSpike(
            d_model=d_model, num_heads=num_heads, neuron_type=neuron_type
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = SpikingMLP(
                d_model=d_model, d_ff=d_ffn, dropout=dropout, neuron_type=neuron_type
            )
        else:
            raise ValueError("use_spiking_mlp must be True for fully-spiking block")
        self.dropout2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward – returns (hidden, reg_loss=None, final_spikes)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs):
        all_spikes: List[torch.Tensor] = []

        # --- Attention / memory -------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm1(x)
        x_attn = self.attn(x)
        all_spikes.extend(self.attn.last_spikes)
        x = residual + self.dropout1(x_attn)

        # --- Feed-forward -------------------------------------------
        residual = x
        if self.prenorm:
            x = self.norm2(x)
        x_mlp, mlp_spikes = self.mlp(x)
        all_spikes.extend(mlp_spikes)
        x = residual + self.dropout2(x_mlp)

        # --- Assemble final spikes tensor ---------------------------
        final_spikes = (
            torch.cat([s.flatten() for s in all_spikes])
            if all_spikes
            else torch.empty(0, device=x.device)
        )

        return x, None, final_spikes
