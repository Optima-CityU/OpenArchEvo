""" SpikingFLABlock_AdaptiveSpikeRouting """
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Tuple, List, Optional, Dict

# --- Start of required SpikingLinear and dummy imports ---

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
    from src.models.nn.dropout import DropoutNd
    from fla.modules import RMSNorm # Re-introduce ShortConvolution for GSA
    registry = neuron_registry
except (ImportError, ModuleNotFoundError):
    # Dummy placeholders for when src imports are not available
    print("Warning: Using placeholder SNN modules and FLA modules.")
    class DummySurrogate:
        @staticmethod
        def apply(x, *args, **kwargs):
            return (x > 0).float()

    def piecewise_quadratic_surrogate():
        return DummySurrogate

    registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}

    class DropoutNd(nn.Identity):
        pass

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            input_dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (self.weight * x).to(input_dtype)

# NOTE: Original ShortConvolution from FLA is no longer used directly.
# It is replaced by SpikingShortConvolution below.

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x):
        spikes = self.neuron(x) # Apply neuron to continuous input to generate spikes
        return self.linear(spikes), spikes # Apply linear layer to spikes, return continuous current and spikes

# NEW: SpikingConv1d class to ensure Conv1d processes spikes

class SpikingConv1d(nn.Module):
    """A module containing a LIF neuron and a Conv1d layer.

    The neuron applies to the input 'x', producing spikes.
    The internal 'conv1d' layer then processes these spikes.
    Returns the continuous output of the conv1d layer and the binary spikes.
    """

    def __init__(self, conv1d_layer: nn.Conv1d, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = registry[neuron_type](surrogate_function=surrogate_function)
        self.conv1d = conv1d_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes = self.neuron(x)
        output = self.conv1d(spikes) # The internal Conv1d receives binary spikes
        return output, spikes

# MODIFIED: SpikingShortConvolution using SpikingConv1d

class SpikingShortConvolution(nn.Module):
    def __init__(self, d_model, kernel_size, activation='silu', neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        # Use SpikingConv1d instead of raw nn.Conv1d
        self.spiking_conv1d = SpikingConv1d(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=kernel_size - 1, bias=False),
            neuron_type,
            surrogate_function
        )
        self.activation = nn.SiLU() if activation == 'silu' else nn.Identity()

    def forward(self, x):
        x_transposed = rearrange(x, 'b l d -> b d l')

        # Apply SpikingConv1d, which internally spikes then convolves
        conv_output_continuous, conv_spikes = self.spiking_conv1d(x_transposed)

        # Truncate padding and transpose back
        conv_output_continuous = conv_output_continuous[..., :x_transposed.shape[-1]] 
        conv_output_continuous = rearrange(conv_output_continuous, 'b d l -> b l d')

        activated_output = self.activation(conv_output_continuous)
        return activated_output, conv_spikes # Return continuous output and spikes

# --- End of required SpikingLinear and dummy imports ---


# =============================================================================
# ===== Start of SpikingSSM Components (S4DKernel, SS4D) (FROM VERSION 2) =====
# =============================================================================

class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, channels=1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(channels, H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("chn, hnl -> chl", C, torch.exp(K)).real
        return K, None

    def register(self, name, tensor, lr=None):
        if lr == 0.0: self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class SS4D(nn.Module):
    def __init__(self, d_model, neuron='sdn', learnable_vth=True, shared_vth=False, d_state=64, dropout=0.0, transposed=True, bidirectional=False, channels=1, trainable_B=False, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.transposed = transposed
        self.learnable_vth = learnable_vth
        self.bidirectional = bidirectional
        self.D = nn.Parameter(torch.randn(channels, self.h))
        if learnable_vth:
            self.ln_vth = nn.Parameter(torch.zeros(d_model, 1) if not shared_vth else torch.zeros(1))
        if bidirectional: channels *= 2
        self.kernel = S4DKernel(self.h, N=self.n, channels=channels, **kernel_args)
        self.neuron = registry[neuron](surrogate_function=piecewise_quadratic_surrogate())
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # SS4D's output_linear takes spikes as input, so the internal Conv1d is already processing spikes.
        # No need for SpikingConv1d wrapper here as the neuron is applied before this whole sequence.
        self.output_linear = nn.Sequential(nn.Conv1d(self.h, 2 * self.h, kernel_size=1), nn.GLU(dim=-2))

    def forward(self, u, **kwargs):
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        k, _ = self.kernel(L=L)
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.einsum("bhl,chl->bchl", u_f, k_f)
        y = torch.fft.irfft(y, n=2 * L)[..., :L]
        y = y + torch.einsum("bhl,ch->bchl", u, self.D)
        y = rearrange(y, "b c h l -> b (c h) l") # y is now (B, C*H, L)
        if self.learnable_vth: y = y / torch.exp(self.ln_vth)

        spikes = self.neuron(y) # Spikes are generated here, shape (B, C*H, L)
        y_dropout_spikes = self.dropout(spikes) # Dropout on spikes
        y = self.output_linear(y_dropout_spikes) # output_linear takes spikes as input

        if not self.transposed: y = y.transpose(-1, -2)
        return y, spikes # y is continuous, spikes are binary

# =============================================================================
# ===== NEW: SpikingMLP (FROM VERSION 2, unchanged) ===========================
# =============================================================================

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, neuron_type: str='sdn', dropout: float=0.0, **kwargs):
        super().__init__()
        self.surrogate_function = piecewise_quadratic_surrogate()
        self.neuron_type = neuron_type

        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), self.neuron_type, self.surrogate_function)
        self.norm1 = nn.LayerNorm(d_ffn)
        self.dropout_layer = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), self.neuron_type, self.surrogate_function)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_spikes_in_mlp = []

        current_1, spikes_1 = self.fc1(x)
        all_spikes_in_mlp.append(spikes_1)

        normed_current_1 = self.norm1(current_1)
        dropped_current_1 = self.dropout_layer(normed_current_1)

        current_2, spikes_2 = self.fc2(dropped_current_1)
        all_spikes_in_mlp.append(spikes_2)

        normed_current_2 = self.norm2(current_2)
        return normed_current_2, all_spikes_in_mlp

# =============================================================================
# ===== NEW: gsa_chunkwise (FROM VERSION 1, ADAPTED) ========================== NO CHANGES
# =============================================================================

def gsa_chunkwise(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, f: torch.Tensor, chunk_size: int=64, initial_state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """GSA Parallel Chunkwise Logic using Factorized Attention.
    Significantly reduces memory by avoiding the materialization of the 5D decay mask.
    Formula:
        Score_ij = Sum_m (Q_im * exp(C_im)) * (K_jm * exp(-C_jm))
    
    Args:
        q (torch.Tensor): Query tensor of shape (B, H, L, M)
        k (torch.Tensor): Key tensor of shape (B, H, L, M)
        v (torch.Tensor): Value tensor of shape (B, H, L, D_v)
        f (torch.Tensor): Decay factor tensor of shape (B, H, L, M)
        chunk_size (int): Size of chunks for processing.
        initial_state (Optional[torch.Tensor]): Initial recurrent state for inter-chunk computation.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (B, H, L, D_v) and final recurrent state.
    """
    b, h, l_orig, d_v = v.shape
    num_slots = q.shape[-1]
    
    # Pad to be divisible by chunk_size
    pad_len = (chunk_size - l_orig % chunk_size) % chunk_size
    if pad_len > 0:
        q_padded = F.pad(q, (0, 0, 0, pad_len))
        k_padded = F.pad(k, (0, 0, 0, pad_len))
        v_padded = F.pad(v, (0, 0, 0, pad_len))
        f_padded = F.pad(f, (0, 0, 0, pad_len))
    else:
        q_padded, k_padded, v_padded, f_padded = q, k, v, f

    # Rearrange into chunks
    q_chunks = rearrange(q_padded, 'b h (n c) m -> b h n c m', c=chunk_size)
    k_chunks = rearrange(k_padded, 'b h (n c) m -> b h n c m', c=chunk_size)
    v_chunks = rearrange(v_padded, 'b h (n c) d_v -> b h n c d_v', c=chunk_size)
    f_chunks = rearrange(f_padded, 'b h (n c) m -> b h n c m', c=chunk_size)
    
    # Softmax addressing (applied to continuous values from SpikingLinear output)
    k_chunks = torch.softmax(k_chunks, dim=-1)
    q_chunks = torch.softmax(q_chunks, dim=-1)
    
    # Log-space decay
    f_chunks = F.logsigmoid(f_chunks) # log(sigma(f))
    
    if initial_state is not None:
        S = initial_state # (B, H, M, D_v)
    else:
        S = torch.zeros(b, h, num_slots, d_v, device=q.device, dtype=q.dtype)
        
    o_list = []
    
    num_chunks = q_chunks.shape[2]
    for i in range(num_chunks):
        q_chunk = q_chunks[:, :, i] # [B, H, C, M]
        k_chunk = k_chunks[:, :, i] 
        v_chunk = v_chunks[:, :, i] # [B, H, C, D_v]
        f_chunk = f_chunks[:, :, i] # [B, H, C, M]
        
        # --- 1. Intra-Chunk (Factorized Form) ---
        # Cumsum of log-decay (decay within chunk)
        f_cumsum = torch.cumsum(f_chunk, dim=-2) # [B, H, C, M]
        
        # Decompose Decay: exp(Ci - Cj) = exp(Ci) * exp(-Cj)
        decay_exp = torch.exp(f_cumsum)      # [B, H, C, M] (Decay up to current pos i)
        decay_inv_exp = torch.exp(-f_cumsum) # [B, H, C, M] (Inverse decay up to current pos i)
        
        # Modulated Q and K
        q_mod = q_chunk * decay_exp
        k_mod = k_chunk * decay_inv_exp
        
        # Attention Matrix: (Q_mod @ K_mod^T)
        # [B, H, C, M] @ [B, H, M, C] -> [B, H, C, C]
        attn_score = q_mod @ k_mod.transpose(-1, -2)
        
        # Apply Causal Mask (Lower triangular for causal, upper triangular filled with 0) 
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        attn_score = attn_score.masked_fill(mask, 0)
        
        # O_intra = Attn @ V
        o_intra = attn_score @ v_chunk
        
        # --- 2. Inter-Chunk --- (Q_current_chunk_decayed @ S_previous_chunk)
        # O_inter = (Q * Decay) @ S
        q_decayed = q_chunk * decay_exp # Q modulated by decay from start of current sequence (not just chunk)
        
        # [B, H, C, M] @ [B, H, M, D_v] -> [B, H, C, D_v]
        o_inter = q_decayed @ S
        
        # Output accumulation
        o_list.append(o_intra + o_inter)
        
        # --- 3. Update State --- (S_new = S_old * Total_Decay_over_chunk + Chunk_Update)
        # Total decay for the whole chunk: exp(sum(f_chunk))
        total_decay_over_chunk = torch.exp(f_chunk.sum(dim=-2)).unsqueeze(-1) # [B, H, M, 1]
        
        c_end_of_chunk = f_cumsum[:, :, -1, :].unsqueeze(-2) # [B, H, 1, M] (last cumulative decay of chunk)
        decay_to_end_const = torch.exp(c_end_of_chunk) # exp(C_end_of_chunk)
        
        k_weighted_for_state_update = k_mod * decay_to_end_const # [B, H, C, M]
        
        # Update: [B, H, M, C] @ [B, H, C, D_v] -> [B, H, M, D_v]
        chunk_update_state = k_weighted_for_state_update.transpose(-1, -2) @ v_chunk
        
        S = S * total_decay_over_chunk + chunk_update_state
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d_v -> b h (n c) d_v')
    if pad_len > 0: o = o[..., :l_orig, :]
    return o, S

# =============================================================================
# ===== MODIFIED: SpikingGSA (Wrapper for gsa_chunkwise) ===========================
# =============================================================================

class SpikingGSA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_slots: int=64, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.value_dim = int(d_model * expand_v)
        self.surrogate_function = piecewise_quadratic_surrogate()
        self.neuron_type = neuron_type

        # Projections using SpikingLinear
        self.q_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, self.surrogate_function)
        self.k_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, self.surrogate_function)
        # The value_dim is for d_model * expand_v, but each head gets d_model // num_heads * expand_v
        self.v_proj = SpikingLinear(nn.Linear(d_model, num_heads * (self.value_dim // num_heads), bias=False), neuron_type, self.surrogate_function) 
        self.f_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, self.surrogate_function)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads * (self.value_dim // num_heads), bias=False), neuron_type, self.surrogate_function) # For output gating

        self.o_proj = SpikingLinear(nn.Linear(num_heads * (self.value_dim // num_heads), d_model, bias=False), neuron_type, self.surrogate_function)
        self.g_norm = RMSNorm(self.value_dim // num_heads) # RMSNorm operates on continuous current

        self.use_short_conv = use_short_conv
        if use_short_conv:
            # Use SpikingShortConvolution instead of original ShortConvolution
            self.q_conv1d = SpikingShortConvolution(num_heads * num_slots, conv_size, activation='silu', neuron_type=neuron_type, surrogate_function=self.surrogate_function)
            self.k_conv1d = SpikingShortConvolution(num_heads * num_slots, conv_size, activation='silu', neuron_type=neuron_type, surrogate_function=self.surrogate_function)
            self.v_conv1d = SpikingShortConvolution(num_heads * (self.value_dim // num_heads), conv_size, activation='silu', neuron_type=neuron_type, surrogate_function=self.surrogate_function) # Correct dim for value
            self.f_conv1d = SpikingShortConvolution(num_heads * num_slots, conv_size, activation='silu', neuron_type=neuron_type, surrogate_function=self.surrogate_function)
        else:
            self.q_conv1d = nn.Identity()
            self.k_conv1d = nn.Identity()
            self.v_conv1d = nn.Identity()
            self.f_conv1d = nn.Identity()

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict]]=None, layer_idx: Optional[int]=None, chunk_size: int=64, **kwargs) -> Tuple[torch.Tensor, None, Dict[str, torch.Tensor]]:
        all_gsa_spikes = {} # Collects all spikes from GSA components

        # Apply SpikingLinear projections
        q_cont, s_q = self.q_proj(x)
        k_cont, s_k = self.k_proj(x)
        v_cont, s_v = self.v_proj(x)
        f_cont, s_f = self.f_proj(x)
        g_cont, s_g = self.g_proj(x) # Output gate
        all_gsa_spikes.update({'q_proj': s_q, 'k_proj': s_k, 'v_proj': s_v, 'f_proj': s_f, 'g_proj': s_g})

        # Apply SpikingShortConvolution (operates on continuous current, but internally spikes before Conv1d)
        # Collect spikes from these conv layers
        q_conv_out, s_q_conv = self.q_conv1d(q_cont)
        k_conv_out, s_k_conv = self.k_conv1d(k_cont)
        v_conv_out, s_v_conv = self.v_conv1d(v_cont)
        f_conv_out, s_f_conv = self.f_conv1d(f_cont)
        all_gsa_spikes.update({
            'q_conv': s_q_conv, 'k_conv': s_k_conv, 'v_conv': s_v_conv, 'f_conv': s_f_conv
        })

        # Rearrange for multi-head GSA
        q_re = rearrange(q_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        k_re = rearrange(k_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        v_re = rearrange(v_conv_out, 'b l (h d) -> b h l d', h=self.num_heads)
        f_re = rearrange(f_conv_out, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)

        last_state = None
        if past_key_values is not None and layer_idx is not None and len(past_key_values) > layer_idx:
            state_dict = past_key_values[layer_idx]
            if isinstance(state_dict, dict) and 'gsa_recurrent_state' in state_dict:
                last_state = state_dict['gsa_recurrent_state']

        # Apply chunkwise GSA logic (operates on continuous current)
        o_raw_attn_output, final_state = gsa_chunkwise(q_re, k_re, v_re, f_re, chunk_size=chunk_size, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) <= layer_idx:
                if len(past_key_values) == layer_idx:
                     past_key_values.append({'gsa_recurrent_state': final_state})
                elif len(past_key_values) < layer_idx:
                    raise ValueError(f"past_key_values length {len(past_key_values)} is less than layer_idx {layer_idx}")
            else:
                past_key_values[layer_idx]['gsa_recurrent_state'] = final_state

        # Rearrange and normalize output
        o = rearrange(o_raw_attn_output, 'b h l d -> b l h d')
        o = self.g_norm(o) # RMSNorm on continuous current

        # Apply output gating (g_proj is SpikingLinear, g_cont is continuous output)
        g_re = rearrange(g_cont, 'b l (h d) -> b l h d', h=self.num_heads) # Rearrange g_cont after its proj
        o = o * F.silu(g_re) # Apply activation after rearrangement

        o = rearrange(o, 'b l h d -> b l (h d)')

        # Final output projection using SpikingLinear
        o_cont, s_o = self.o_proj(o)
        all_gsa_spikes['o_proj'] = s_o

        return o_cont, None, all_gsa_spikes # Return continuous output, None for reg_loss, and all collected spikes

# =============================================================================
# ===== NEW: SpikingFLABlock (Parallel Gated SS4D, GSA, and MLP) ============== 
# =============================================================================

class SpikingFLABlock(nn.Module):
    """This class implements a novel hybrid SpikingFLABlock with parallel, dynamically
    routed SS4D, SpikingGSA, and SpikingMLP branches. A central Spiking Router
    determines the contribution of each branch's output based on input context via
    direct spike-gating.
    """

    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=4, dropout: float=0.1, prenorm: bool=True, use_spiking_mlp: Optional[bool]=True, neuron_type: Optional[str]='sdn', gsa_num_slots: int=64, gsa_expand_v: float=1.0, gsa_conv_size: int=4, gsa_use_short_conv: bool=True, chunk_size: int=64, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.neuron_type = neuron_type
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.surrogate_function = piecewise_quadratic_surrogate()

        self.use_spiking_mlp = use_spiking_mlp if use_spiking_mlp is not None else True

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        # Pre-normalization for the main input before branching
        self.norm_input = RMSNorm(d_model)

        # Dynamic Spiking Router components
        # Each router projection generates spikes to gate its branch's contribution
        # The output dimension of the router is d_model, so it can gate element-wise.
        self.router_ss4d_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, self.surrogate_function)
        self.router_ss4d_gate_scale = nn.Parameter(torch.ones(1)) # Learnable scale for spikes

        self.router_gsa_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, self.surrogate_function)
        self.router_gsa_gate_scale = nn.Parameter(torch.ones(1)) # Learnable scale for spikes

        if self.use_spiking_mlp:
            self.router_mlp_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, self.surrogate_function)
            self.router_mlp_gate_scale = nn.Parameter(torch.ones(1)) # Learnable scale for spikes

        # --- SS4D Branch Components ---
        ss4d_kwargs = kwargs.get('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs.setdefault('neuron', neuron_type)
        self.ss4d_module = SS4D(d_model, dropout=dropout, transposed=True, **ss4d_kwargs)
        self.dropout_ss4d_out = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # --- GSA Branch Components ---
        self.gsa_module = SpikingGSA(
            d_model=d_model,
            num_heads=num_heads,
            num_slots=gsa_num_slots,
            expand_v=gsa_expand_v,
            use_short_conv=gsa_use_short_conv,
            conv_size=gsa_conv_size,
            neuron_type=neuron_type,
            **kwargs 
        )
        self.dropout_gsa_out = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # --- Spiking MLP Branch Components ---
        if self.use_spiking_mlp:
            d_ffn = d_model * d_ffn_mult
            self.mlp_module = SpikingMLP(d_model, d_ffn, neuron_type=neuron_type, dropout=dropout)
            self.dropout_mlp_out = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # Final normalization for post-norm architecture (if applicable)
        self.norm_final = RMSNorm(d_model) if not prenorm else nn.Identity()
        self.dropout_final = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, past_key_values: Optional[List[Dict]]=None, use_cache: Optional[bool]=False, layer_idx: Optional[int]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        all_spikes_list = []
        x_res = x  # Initial continuous residual stream

        # Apply pre-normalization if configured
        x_input_for_branches = self.norm_input(x_res) if self.prenorm else x_res

        # ==================== Dynamic Spiking Router ====================
        # Each router SpikingLinear projects the input and produces binary spikes.
        # These spikes are then scaled by a learnable parameter to act as adaptive gates.

        # SS4D Router
        _, ss4d_router_spikes = self.router_ss4d_proj(x_input_for_branches)
        all_spikes_list.append(ss4d_router_spikes)
        ss4d_mod_factor = (1 + self.router_ss4d_gate_scale * ss4d_router_spikes)

        # GSA Router
        _, gsa_router_spikes = self.router_gsa_proj(x_input_for_branches)
        all_spikes_list.append(gsa_router_spikes)
        gsa_mod_factor = (1 + self.router_gsa_gate_scale * gsa_router_spikes)

        # MLP Router (if MLP branch is enabled)
        mlp_mod_factor = None
        if self.use_spiking_mlp:
            _, mlp_router_spikes = self.router_mlp_proj(x_input_for_branches)
            all_spikes_list.append(mlp_router_spikes)
            mlp_mod_factor = (1 + self.router_mlp_gate_scale * mlp_router_spikes)

        # ==================== SS4D Branch ====================
        # SS4D module expects transposed input (B, D, L)
        x_ss4d_input_transposed = rearrange(x_input_for_branches, 'b l d -> b d l')
        ss4d_output_continuous, ss4d_spikes = self.ss4d_module(x_ss4d_input_transposed)
        all_spikes_list.append(ss4d_spikes)
        ss4d_output_continuous = rearrange(ss4d_output_continuous, 'b d l -> b l d') # Transpose back to (B, L, D)
        ss4d_output_continuous = self.dropout_ss4d_out(ss4d_output_continuous)

        # ==================== GSA Branch ====================
        # GSA module expects (B, L, D) input
        gsa_output_continuous, _, gsa_spikes_dict = self.gsa_module(
            x_input_for_branches, # Input directly from initial norm
            past_key_values=past_key_values,
            use_cache=use_cache,
            layer_idx=layer_idx,
            chunk_size=self.chunk_size,
            **kwargs
        )
        # Extend with all collected spikes from the GSA module
        all_spikes_list.extend(list(gsa_spikes_dict.values()))
        gsa_output_continuous = self.dropout_gsa_out(gsa_output_continuous)

        # ==================== Spiking MLP Branch ====================
        mlp_output_current = None
        if self.use_spiking_mlp:
            # MLP module expects (B, L, D) input
            mlp_output_current, mlp_internal_spikes = self.mlp_module(x_input_for_branches)
            all_spikes_list.extend(mlp_internal_spikes)
            mlp_output_current = self.dropout_mlp_out(mlp_output_current)

        # ==================== Combine Branch Outputs with Spike-Driven Gates ====================
        # The contribution of each branch is modulated by the binary spikes from its router.
        current_output_stream = x_res # Start with the residual input

        current_output_stream = current_output_stream + (ss4d_output_continuous * ss4d_mod_factor)
        current_output_stream = current_output_stream + (gsa_output_continuous * gsa_mod_factor)

        if self.use_spiking_mlp:
            current_output_stream = current_output_stream + (mlp_output_current * mlp_mod_factor)

        # ==================== Finalization ====================
        # Apply final post-normalization if prenorm is false, otherwise return current stream
        output = self.norm_final(current_output_stream) if not self.prenorm else current_output_stream
        output = self.dropout_final(output) # Apply final dropout

        reg_loss = None # No explicit regularization loss implemented in this block

        # Concatenate all collected binary spikes into a single flat tensor.
        # Ensure only non-None tensors are concatenated.
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None])

        return output, reg_loss, final_spikes