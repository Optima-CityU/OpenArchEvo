"""The fixed sequence-model wrapper shared by probing and proxy training.

``embedding -> n_layers x block -> RMSNorm -> head``. Language modelling
applies the head per token; classification (ListOps) mean-pools over the
sequence first. Only the block is searched; everything else is fixed. The block
is built and called as defined in :mod:`openarchevo.contract`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from openarchevo.contract import block_kwargs


class SequenceModel(nn.Module):
    def __init__(self, block_cls: type[nn.Module], vocab_size: int, n_outputs: int, d_model: int,
                 n_layers: int, num_heads: int, d_ffn_mult: int = 2, dropout: float = 0.1,
                 neuron_type: str = "bptt", pad_token_id: int = 0, is_lm: bool = False,
                 extra_block_kwargs: Optional[dict[str, Any]] = None):
        super().__init__()
        self.is_lm = is_lm
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([
            block_cls(**block_kwargs(d_model, num_heads, d_ffn_mult, dropout, neuron_type, layer_idx=i,
                                     **(extra_block_kwargs or {})))
            for i in range(n_layers)])
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, n_outputs)
        self.features: Optional[torch.Tensor] = None  # last hidden states (used by NASWOT)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor], list[torch.Tensor]]:
        """Returns ``(logits, regularisation loss or None, list of per-block spike tensors)``."""
        x = self.embedding(tokens)
        reg_loss: Optional[torch.Tensor] = None
        spikes: list[torch.Tensor] = []
        for block in self.blocks:
            x, block_reg, block_spikes = block(x)
            if self.training and block_reg is not None:
                reg_loss = block_reg if reg_loss is None else reg_loss + block_reg
            if block_spikes is not None:
                spikes.append(block_spikes)
        self.features = x.detach()
        x = self.norm(x)
        if not self.is_lm:
            x = x.mean(dim=1)
        return self.head(x), reg_loss, spikes
