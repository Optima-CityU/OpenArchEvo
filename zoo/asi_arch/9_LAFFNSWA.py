# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

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


class LinearAttention(nn.Module):
    """
    The Linear Attention module, based on the Delta Rule. This serves as the LA component of the DeltaNetBlock.
    """
    def __init__(self, d_model: int, num_heads: int = 8, expand_v: float = 2.0, conv_size: int = 4, layer_idx: int = None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        
        self.key_dim = d_model
        self.value_dim = int(d_model * expand_v)
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.b_proj = nn.Linear(d_model, self.num_heads, bias=False)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        
        self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation='silu')
        self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, activation='silu')
        self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, activation='silu')
        self.o_norm = RMSNorm(self.head_v_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Cache] = None, use_cache: Optional[bool] = False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # Unpadding logic might not be needed here if handled by the block, but kept for standalone use
        batch_size, q_len, _ = hidden_states.shape
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask)
            hidden_states = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)
        else:
            cu_seqlens = None

        last_state = None
        if past_key_values is not None and self.layer_idx is not None:
            last_state = past_key_values.get(self.layer_idx, None)

        # ###################### 核心修复：正确处理卷积输出和缓存 ######################
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state.get('conv_state', (None, None, None))
        
        q = self.q_proj(hidden_states)
        q, conv_state_q = self.q_conv1d(q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        
        k = self.k_proj(hidden_states)
        k, conv_state_k = self.k_conv1d(k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        
        v = self.v_proj(hidden_states)
        v, conv_state_v = self.v_conv1d(v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        # ##########################################################################

        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.num_heads)

        beta = self.b_proj(hidden_states).sigmoid()
        
        q, k, v = map(lambda x: rearrange(x, 'b l h d -> b h l d'), (q, k, v))
        beta = rearrange(beta, 'b l h -> b h l')
            
        recurrent_state = last_state.get('recurrent_state', None) if last_state is not None else None
        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')
        
        if use_cache:
            new_state = {'recurrent_state': recurrent_state, 'conv_state': (conv_state_q, conv_state_k, conv_state_v)}
            # This assumes `past_key_values` is a dictionary-like object that can be updated.
            if past_key_values is not None:
                past_key_values[self.layer_idx] = new_state

        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


class SlidingWindowAttention(nn.Module):
    """
    Memory-optimized Sliding Window Attention using torch.nn.functional.scaled_dot_product_attention.
    """
    def __init__(self, d_model: int, num_heads: int = 8, window_size: int = 128, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, None, None]:
        batch_size, seq_len, _ = hidden_states.shape
        
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Create a sliding window mask. True means the position is MASKED.
        # This creates a causal mask with a limited window size.
        q_indices = torch.arange(seq_len, device=hidden_states.device)
        k_indices = torch.arange(seq_len, device=hidden_states.device)
        rel_indices = q_indices[:, None] - k_indices[None, :]
        
        # Mask positions that are either in the future (causal) or too far in the past (window).
        mask = ~((rel_indices >= 0) & (rel_indices < self.window_size))
        
        if attention_mask is not None:
            # The attention_mask from the training loop is typically (B, L), 1 for kept, 0 for padded
            # We need to convert it to a broadcastable shape (B, 1, 1, L) and a boolean type where True means MASK
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
            mask = mask.unsqueeze(0).unsqueeze(1) | padding_mask # (B, 1, L, L)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        
        output = self.o_proj(rearrange(attn_output, 'b h l d -> b l (h d)'))
        return output, None, None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn_mult: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * d_ffn_mult, bias=False)
        self.act = nn.SiLU()
        self.w2 = nn.Linear(d_model * d_ffn_mult, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class DeltaNet(nn.Module):
    """
    A complete Transformer-style block with the LA -> FFN -> SWA structure.
    The SWA module is optimized to be memory-efficient.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        expand_v: float = 2.0,
        conv_size: int = 4,
        window_size: int = 128,
        d_ffn_mult: int = 2,
        dropout: float = 0.1,
        layer_idx: int = None,
        **kwargs
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.la = LinearAttention(d_model=d_model, num_heads=num_heads, expand_v=expand_v, conv_size=conv_size, layer_idx=layer_idx, **kwargs)
        self.ffn = FeedForward(d_model=d_model, d_ffn_mult=d_ffn_mult, dropout=dropout, **kwargs)
        self.swa = SlidingWindowAttention(d_model=d_model, num_heads=num_heads, window_size=window_size, **kwargs)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        
        # 1. Linear Attention
        residual = hidden_states
        x, _, updated_cache = self.la(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        hidden_states = residual + x

        # 2. Feed-Forward Network
        residual = hidden_states
        x = self.ffn(self.norm2(hidden_states))
        hidden_states = residual + x
        
        # 3. Sliding Window Attention
        residual = hidden_states
        x, _, _ = self.swa(self.norm3(hidden_states), attention_mask=attention_mask)
        hidden_states = residual + x
        
        return hidden_states, None, updated_cache