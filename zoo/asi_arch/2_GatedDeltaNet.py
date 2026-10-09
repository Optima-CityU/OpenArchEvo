# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from torch.nn import functional as F
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

def softmax(x):
    return F.softmax(x, dim=-1)

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

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class DeltaNet(nn.Module):
    def __init__(
        self,
        d_model: int = None,
        hidden_size: int = 1024,
        num_heads: int = 8,
        num_v_heads: int = None,
        expand_v: float = 2.0,
        mode: str = 'chunk1',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs
    ) -> DeltaNet:
        super().__init__()
        
        if d_model is not None:
            hidden_size = d_model

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.expand_v = expand_v
        self.mode = mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx
        
        # ### 核心修改：所有维度都从hidden_size派生 ###
        self.head_k_dim = self.hidden_size // self.num_heads
        self.key_dim = self.hidden_size # key_dim 和 hidden_size 保持一致
        self.head_v_dim = int(self.head_k_dim * self.expand_v)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads"
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(f"num_v_heads must be divisible by num_heads.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, activation='silu')
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        
        # 移除了之前的断言，因为新的__init__设计从根本上解决了维度问题
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "attention_mask must be a 2D tensor."

        batch_size, q_len, _ = hidden_states.shape
        
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.num_v_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.num_v_heads)

        if self.num_v_heads < self.num_heads:
            k = repeat(k, 'b l h d -> b l (h g) d', g=self.num_heads // self.num_v_heads)
            v = repeat(v, 'b l h d -> b l (h g) d', g=self.num_heads // self.num_v_heads)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        beta = rearrange(beta, 'b l h -> b h l')
            
        o, recurrent_state = delta_rule_chunkwise(q=q, k=k, v=v, beta=beta)
        o = rearrange(o, 'b h l d -> b l h d')

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
            
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values