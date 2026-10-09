# -*- coding: utf-8 -*-
# A DeltaNet module refactored to function as a high-performance
# Sliding Window Attention (SWA) block, while maintaining the user-provided API format.

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

# --------------------------------------------------------------------------------
# Dependency Placeholders & Imports
# --------------------------------------------------------------------------------

# Check for FlashAttention library
try:
    from flash_attn import flash_attn_func
    IS_FLASH_ATTN_2_AVAILABLE = True
except ImportError:
    IS_FLASH_ATTN_2_AVAILABLE = False

# The user's template requires these modules. We provide placeholders if they are not available.
# It is recommended to have these dependencies installed for full compatibility.
try:
    from fla.modules import RMSNorm, ShortConvolution
    from fla.layers.utils import get_unpad_data, pad_input
    # FusedRMSNormGated is a specialized module, we'll use a functional equivalent if not found.
    class FusedRMSNormGated(RMSNorm):
        def forward(self, x, g):
            # Functional equivalent of the gated RMSNorm
            normed_x = super().forward(x)
            return normed_x * F.silu(g)
except ImportError:
    print("Warning: `fla` library not found. Using fallback implementations for required modules.")
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps
        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight * hidden_states).to(input_dtype)
    class FusedRMSNormGated(RMSNorm):
        def forward(self, x, g):
            normed_x = super().forward(x)
            return normed_x * F.silu(g)
    # The template requires ShortConvolution, but SWA does not use it. This is a dummy class.
    class ShortConvolution(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x, **kwargs): return x, None
    # Dummy functions for API compatibility
    def get_unpad_data(x): return None, None, None
    def pad_input(x, *args): return x

# This is a generic cache placeholder, adapt to your project's Cache object if needed.
class Cache:
    def __init__(self): self.key_value = None
    def update(self, key, value, layer_idx): self.key_value = (key, value)
    def get(self, layer_idx): return self.key_value

# --------------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) Helpers
# --------------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000, device=None):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --------------------------------------------------------------------------------
# The Refactored DeltaNet Module (now functioning as SWA)
# --------------------------------------------------------------------------------

class DeltaNet(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = False,
        use_gate: bool = True,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs
    ) -> DeltaNet:
        super().__init__()
        
        if d_model is not None:
            hidden_size = d_model
            
        # SWA specific parameters, extracted from kwargs
        self.window_size = kwargs.get('window_size', 1024)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 8192)
        self.dropout = kwargs.get('dropout', 0.0)
        
        # Print warnings for unused parameters to avoid confusion
        if use_short_conv:
            print(f"Warning: `use_short_conv=True` is not applicable for the SWA implementation of DeltaNet and will be ignored.")
        if mode != 'chunk':
            print(f"Warning: `mode='{mode}'` is not applicable for SWA and will be ignored.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_gate = use_gate
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings
        )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_norm = FusedRMSNormGated(self.hidden_size, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.hidden_size, eps=norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        
        B, L, _ = hidden_states.shape

        # 1. Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        # 2. Caching logic for auto-regressive decoding
        kv_seq_len = L
        if past_key_values is not None:
            # Resuse kv from last generated token
            past_k, past_v = past_key_values.get(self.layer_idx)
            kv_seq_len += past_k.shape[1]
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        if use_cache and past_key_values is not None:
            past_key_values.update(k, v, self.layer_idx)
            
        position_ids = torch.arange(kv_seq_len - L, kv_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        # 3. Apply RoPE
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin, position_ids)
        v = v.transpose(1, 2)
        
        # 4. Sliding Window Attention
        if IS_FLASH_ATTN_2_AVAILABLE and hidden_states.device.type == 'cuda':
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                window_size=(self.window_size, 0)
            )
        else:
            # Fallback for non-cuda devices or if flash_attn is not installed
            # Create a combined causal, sliding window, and padding attention mask
            causal_sliding_mask = self._create_causal_sliding_mask(L, kv_seq_len, hidden_states.device)
            if attention_mask is not None:
                # Invert padding mask (1 for attend, 0 for ignore) and combine
                padding_mask = attention_mask[:, None, -L:, :kv_seq_len].bool()
                final_mask = causal_sliding_mask & padding_mask
            else:
                final_mask = causal_sliding_mask
            
            # PyTorch SDPA expects mask where `True` is masked
            final_mask = ~final_mask

            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=final_mask,
                dropout_p=self.dropout if self.training else 0.0
            )

        # 5. Output projection and gating
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        
        if self.use_gate:
            g = self.g_proj(hidden_states)
            attn_output = self.o_norm(attn_output, g)
        else:
            attn_output = self.o_norm(attn_output)
            
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_values

    def _create_causal_sliding_mask(self, q_len, kv_len, device):
        # Creates a boolean mask that is both causal and respects the sliding window
        q_indices = torch.arange(q_len, device=device).view(-1, 1) + (kv_len - q_len)
        k_indices = torch.arange(kv_len, device=device).view(1, -1)
        
        # Attend if key is in the past AND within the window
        mask = (k_indices <= q_indices) & (q_indices - k_indices < self.window_size)
        return mask