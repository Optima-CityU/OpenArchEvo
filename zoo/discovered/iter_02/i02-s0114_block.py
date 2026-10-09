import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution

# ==================== SNN Imports ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder SNN modules.")
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return input
        @staticmethod
        def backward(ctx, grad_output): return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x):
        x_transposed = x.transpose(-1, -2)
        spikes = self.neuron(x_transposed).transpose(-1, -2)
        return self.linear(spikes), spikes

# ==================== White-box Operator (GLA) ====================

def gla_chunkwise(q, k, v, g, chunk_size=64, initial_state=None):
    """Gated Linear Attention Chunkwise Parallel Logic.
    
    Formula:
        S_t = S_{t-1} * exp(-g_t) + K_t^T V_t
        O_t = Q_t S_t
    
    Parallelization:
        We treat this as a "Weighted Linear Attention".
        Since 'g' is scalar per head (B, H, L, 1), the decay mask is (C, C).
        This allows full parallelization via matrix multiplication without OOM.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    # 1. Padding
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v, g = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v, g]]
        
    # 2. Reshape to Chunks
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    g = rearrange(g, 'b h (n c) d -> b h n c d', c=chunk_size)
    
    # Initialize Matrix State S: [B, H, Dk, Dv]
    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = torch.zeros(b, h, d_k, d_v, device=q.device, dtype=q.dtype)
        
    o = torch.zeros_like(v)
    
    # Pre-calculate cumulative decay for efficient mask construction
    for i in range(q.shape[2]): # Chunk Loop
        q_i = q[:, :, i] # [B, H, C, Dk]
        k_i = k[:, :, i] # [B, H, C, Dk]
        v_i = v[:, :, i] # [B, H, C, Dv]
        g_i = g[:, :, i] # [B, H, C, 1]
        
        # --- 1. Intra-Chunk (Parallel Attention) ---
        # Construct Decay Mask D_{ij} = exp(Sum_{k=j+1}^i g_k) = exp(Cumsum_i - Cumsum_j)
        g_cumsum = torch.cumsum(g_i, dim=-2) # [B, H, C, 1]
        
        # Mask shape: [B, H, C, C] (Broadcasting over last dim 1 is automatic)
        decay_matrix = g_cumsum - rearrange(g_cumsum, 'b h c 1 -> b h 1 c') # This creates a (B,H,C,C) matrix
        
        # Causal Mask
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        decay_matrix = decay_matrix.masked_fill(mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        decay_mask = torch.exp(decay_matrix)
        
        # Attention Score: A = (Q @ K^T) * Mask
        # [B, H, C, Dk] @ [B, H, Dk, C] -> [B, H, C, C]
        attn = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        
        # Output Intra: O = A @ V
        o_intra = attn @ v_i
        
        # --- 2. Inter-Chunk (Recurrence) ---
        # Decay from start of chunk to current time step t
        # exp(g_cumsum[t])
        decay_from_start = torch.exp(g_cumsum) # [B, H, C, 1]
        
        # Broadcast multiply: Q scaled by decay
        # [B, H, C, Dk] * [B, H, C, 1] -> [B, H, C, Dk]
        q_decayed = q_i * decay_from_start
        
        # [B, H, C, Dk] @ [B, H, Dk, Dv] -> [B, H, C, Dv]
        o_inter = q_decayed @ S
        
        o[:, :, i] = o_intra + o_inter
        
        # --- 3. Update State S ---
        # S_new = S_old * Total_Decay + Sum(K^T * V * Decay_to_End)
        
        # Total Decay for the whole chunk
        # exp(Sum(g_all))
        total_decay = torch.exp(g_i.sum(dim=-2, keepdim=True)) # [B, H, 1, 1]
        
        # Contribution from current chunk
        # Decay from t to End: exp(Cumsum_End - Cumsum_t)
        # g_cumsum[:, :, -1, :].unsqueeze(-2) -> [B, H, 1, 1] (last cumulative sum)
        decay_to_end = torch.exp(g_cumsum[:, :, -1, :].unsqueeze(-2) - g_cumsum) # [B, H, C, 1]
        
        # Weighted KV: (K * decay)
        k_decayed = k_i * decay_to_end
        
        # Update: S = S * total_decay + current_chunk_contrib
        # [B, H, Dk, C] @ [B, H, C, Dv] -> [B, H, Dk, Dv]
        current_chunk_contrib = k_decayed.transpose(-1, -2) @ v_i
        
        S = S * total_decay + current_chunk_contrib
        
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0: o = o[..., :l, :]
    return o, S

# ==================== Main Layer ====================

class GLA(nn.Module):
    def __init__(self, d_model, num_heads, expand_k=0.5, expand_v=1.0, gate_low_rank_dim=16, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads

        surrogate = piecewise_quadratic_surrogate()

        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)

        # Short Conv (Optional but standard in FLA)
        self.use_short_conv = True 
        if self.use_short_conv:
             self.conv1d_group = ShortConvolution(self.key_dim * 2 + self.value_dim + num_heads, kernel_size=4, activation='silu')

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.head_v_dim)

    def forward(self, x, past_key_values=None, layer_idx=None, **kwargs):
        all_spikes = {}
        q_out, s_q = self.q_proj(x)
        k_out, s_k = self.k_proj(x)
        v_out, s_v = self.v_proj(x)
        g_out, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'g': s_g})

        # Short Conv
        if self.use_short_conv:
            combined = torch.cat([q_out, k_out, v_out, g_out], dim=-1)
            conv_state = None
            if past_key_values and layer_idx is not None:
                 if len(past_key_values) > layer_idx:
                    conv_state = past_key_values[layer_idx].get('conv_state')
            combined, new_conv_state = self.conv1d_group(combined, cache=conv_state, output_final_state=True)
            q_out, k_out, v_out, g_out = torch.split(combined, [self.key_dim, self.key_dim, self.value_dim, self.num_heads], dim=-1)

        # Reshape
        q = rearrange(q_out, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k_out, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v_out, 'b l (h d) -> b h l d', h=self.num_heads)
        g = rearrange(g_out, 'b l h -> b h l 1') # [B, H, L, 1]

        # Gate activation: LogSigmoid -> Log Space Decay
        g = F.logsigmoid(g)

        # State Management
        last_state = None
        if past_key_values is not None and layer_idx is not None:
            if len(past_key_values) > layer_idx:
                last_state = past_key_values[layer_idx].get('recurrent_state')

        # Chunkwise Parallel Op
        o, final_state = gla_chunkwise(q, k, v, g, initial_state=last_state)

        if past_key_values is not None and layer_idx is not None:
            state_dict = {'recurrent_state': final_state}
            if self.use_short_conv: state_dict['conv_state'] = new_conv_state

            if len(past_key_values) <= layer_idx:
                past_key_values.append(state_dict)
            else:
                past_key_values[layer_idx] = state_dict

        # Output
        o = rearrange(o, 'b h l d -> b l h d')
        o = self.g_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')

        o_out, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o_out, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        # Introduce an intermediate neuron to replace GELU and ensure spike input to fc2
        self.intermediate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x_fc1, spike1 = self.fc1(x)

        # Apply the intermediate neuron directly on x_fc1 to get intermediate spikes.
        spikes_mid_transposed = self.intermediate_neuron(x_fc1.transpose(-1, -2))
        spikes_mid = spikes_mid_transposed.transpose(-1, -2)

        # Dropout on the intermediate spikes. The second SpikingLinear now receives these intermediate spikes.
        x_fc2_out, spike2 = self.fc2(self.dropout(spikes_mid))

        return x_fc2_out, spike1, spikes_mid, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = GLA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        self.use_spiking_mlp = use_spiking_mlp
        if use_spiking_mlp:
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        all_spikes_list = []
        x_res = x
        if self.prenorm: x = self.norm1(x)

        x_attn, _, attn_spikes_dict = self.attn(x, **kwargs)
        if attn_spikes_dict: 
            all_spikes_list.extend(attn_spikes_dict.values())
        x = x_res + self.dropout1(x_attn)

        x_res = x
        if self.prenorm: x = self.norm2(x)

        if self.use_spiking_mlp:
            x_mlp, s1, s_mid, s2 = self.mlp(x)
            all_spikes_list.extend([s1, s_mid, s2])
        else:
            x_mlp = self.mlp(x)

        x = x + self.dropout2(x_mlp)

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        return x, None, final_spikes