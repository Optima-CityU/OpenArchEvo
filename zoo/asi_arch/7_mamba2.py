# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

if TYPE_CHECKING:
    from fla.models.utils import Cache


# @torch.compile
def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
    
    padded_len = l + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, padded_len // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')

    if pad_len > 0:
        o = o[:, :, :l]
    return o, S


class DeltaNet(nn.Module):
    """
    Final optimized architectural replacement for Mamba2, based on the DeltaNet structure.
    This module uses the 'convolve-then-project' data flow for high performance and is fully implemented in pure PyTorch.
    """
    def __init__(
        self,
        d_model: int = None,
        hidden_size: int = 1024,
        num_heads: int = 8,
        head_dim: int = 128,
        expand: int = 2,
        conv_kernel: int = 4,
        use_conv_bias: bool = False,
        hidden_act: str = "silu",
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__()

        if d_model is not None:
            hidden_size = d_model
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.key_dim = self.num_heads * self.head_k_dim
        
        self.num_kv_heads = num_heads
        
        # Mamba2's intermediate_size is mapped to value_dim via expand factor
        self.value_dim = int(hidden_size * expand)
        self.head_v_dim = self.value_dim // self.num_heads
        assert self.value_dim % self.num_heads == 0, "value_dim must be divisible by num_heads"

        self.layer_idx = layer_idx
        self.use_output_gate = True
        
        # Optimized: Fused projection and convolution
        self.in_proj_dim = self.key_dim + self.key_dim + self.value_dim
        self.in_proj = nn.Linear(hidden_size, self.in_proj_dim, bias=False)

        self.conv_size = conv_kernel
        self.conv_bias = use_conv_bias
        self.conv1d = ShortConvolution(self.in_proj_dim, kernel_size=self.conv_size, bias=self.conv_bias, activation='silu')
        
        # Projections for gate and beta still act on original hidden_states
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=True)

        if hidden_act in ["swish", "silu"]:
            self.gate_fn = F.silu
        elif hidden_act == "gelu":
            self.gate_fn = F.gelu
        else:
            self.gate_fn = F.silu

        self.o_norm = RMSNorm(self.head_v_dim, elementwise_affine=True, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        
        batch_size, q_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            
        # Unpadding logic
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states_unpadded = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        else:
            hidden_states_unpadded = hidden_states
            cu_seqlens = None

        # Optimized data flow: convolve-then-split
        x_qkv = self.in_proj(hidden_states_unpadded)
        conv_state = last_state.get('conv_state', None) if last_state is not None else None
        x_qkv, conv_state = self.conv1d(x_qkv, cache=conv_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        q, k, v = torch.split(x_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.num_heads)
        
        beta = self.b_proj(hidden_states_unpadded).sigmoid()

        recurrent_state = last_state.get('recurrent_state', None) if last_state is not None else None
        
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')
            
        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')

        if use_cache and past_key_values is not None:
            new_state = {
                'recurrent_state': recurrent_state,
                'conv_state': conv_state
            }
            past_key_values.update(new_state, self.layer_idx, q_len)
        
        g = self.g_proj(hidden_states_unpadded)
        g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
        o = self.o_norm(o) * self.gate_fn(g)
        
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)

        # Padding logic
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values