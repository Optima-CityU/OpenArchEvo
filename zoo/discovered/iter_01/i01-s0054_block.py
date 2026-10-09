import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
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
        # The neuron converts the real-valued input to binary spikes
        spikes = self.neuron(x_transposed).transpose(-1, -2)
        # The linear layer operates on the binary spikes
        return self.linear(spikes), spikes

# ==================== White-box Operator (KDA Chunkwise) ====================

def kda_chunkwise_parallel(q, k, v, beta, decay, alpha, chunk_size=32, initial_state=None):
    """KDA Chunkwise parallel operator.
    Core logic: S_t = decay * S_{t-1} + beta * (alpha * v_t - S_{t-1} @ k_t) @ k_t^T
    Modified to use dimensional `beta` and `alpha` gates.
    """
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        # beta and alpha need to be padded along the sequence dimension as well
        beta = F.pad(beta, (0, 0, 0, pad_len), value=0.0)
        decay = F.pad(decay, (0, pad_len), value=0.0) 
        alpha = F.pad(alpha, (0, 0, 0, pad_len), value=0.0)
        
    # 1. Robust normalization
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-5)
    k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-5)

    # Reshape to Chunks
    q, k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v])
    # Beta and Alpha are now dimensional, so they also need `d` in rearrange
    beta, decay, alpha = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [beta, decay, alpha]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)

    # Pre-generate Bool type mask to prevent errors
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)
    mask_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=-1)

    for i in range(num_chunks):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        b_i, d_i, a_i = beta[:, :, i], decay[:, :, i], alpha[:, :, i]
        
        # Apply dimensional alpha gate to v_i
        v_i = v_i * a_i # Alpha gate is now per-dimension

        # [Step 1] Compute decay matrix (Log-space)
        d_cumsum = torch.cumsum(d_i, dim=-1) 
        decay_rel = (d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)) 
        decay_mask = torch.exp(decay_rel).masked_fill(mask_triu, 0) # Keep only causal part
        
        # [Step 2] Compute prediction error (Delta Rule + Intra-chunk Wy approximation)
        # v_pred_inter is the prediction from historical state S
        v_pred_inter = k_i @ S 
        # Apply dimensional beta gate to v_error
        v_error = (v_i - v_pred_inter) * b_i # Beta gate is now per-dimension
        
        # Intra-chunk parallel compensation: handle the effect of previous k in this chunk on current update
        attn_intra = (k_i @ k_i.transpose(-1, -2)) * decay_mask * mask_tril
        v_error_intra = v_error - (attn_intra @ v_error) * 0.5 # 0.5 is stability factor
        
        # [Step 3] Historical reading and intra-chunk incremental output
        decay_from_start = torch.exp(d_cumsum).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        q_decay_k = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = q_decay_k @ v_error_intra
        
        o[:, :, i] = o_inter + o_intra
        
        # [Step 4] State update
        # Decay ratio to end of chunk
        d_mean = d_i.mean(dim=-1, keepdim=True).unsqueeze(-1) # Still uses scalar mean for decay
        delta_S = k_i.transpose(-1, -2) @ v_error_intra
        S = d_mean.exp() * S + delta_S
        
        # Clamp to prevent overflow, ensure SNN ZCP computation stability
        S = torch.clamp(S, -10.0, 10.0)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== SpikingKDA Layer ====================

class SpikingKDA(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=1.0, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: int=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads # Dimension per head for value/gates
        self.head_k_dim = self.key_dim // num_heads # Dimension per head for key

        surrogate = piecewise_quadratic_surrogate()

        # Projections
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        # Beta and Alpha are now dimensional, matching value_dim per head
        self.b_proj = SpikingLinear(nn.Linear(d_model, num_heads * self.head_v_dim, bias=False), neuron_type, surrogate) 
        self.w_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) 
        self.a_proj = SpikingLinear(nn.Linear(d_model, num_heads * self.head_v_dim, bias=False), neuron_type, surrogate) 

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

    def forward(self, x, past_key_values=None, **kwargs):
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)

        beta_raw, s_b = self.b_proj(x) # Raw output before activation
        w_raw, s_w = self.w_proj(x) # Raw output before activation
        alpha_raw, s_a = self.a_proj(x) # Raw output before activation
        all_spikes.update({'q':s_q, 'k':s_k, 'v':s_v, 'beta':s_b, 'w':s_w, 'alpha':s_a})

        if hasattr(self, 'q_conv1d'):
            # ShortConvolution already handles (real_value, _) output
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)

        # Activation function stability handling for real-valued outputs of SpikingLinear
        # Beta and Alpha are now dimensional per head_v_dim
        beta = rearrange(beta_raw, 'b l (h d) -> b h l d', h=self.num_heads)
        alpha = rearrange(alpha_raw, 'b l (h d) -> b h l d', h=self.num_heads)

        beta = torch.sigmoid(beta) * 0.5 # Limit step size, now dimensional
        alpha = torch.sigmoid(alpha) # Now dimensional
        decay = F.logsigmoid(w_raw) # Still scalar per head per position

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        decay = rearrange(decay, 'b l h -> b h l') # Scalar per head per position
        # beta and alpha are already rearranged to b h l d

        layer_state = None
        if past_key_values is not None:
            layer_state = past_key_values[self.layer_idx] if self.layer_idx in past_key_values else None

        o, final_state = kda_chunkwise_parallel(q, k, v, beta, decay, alpha, initial_state=layer_state)

        if past_key_values is not None:
            past_key_values[self.layer_idx] = final_state

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o

        return o, None, all_spikes

# [SpikingMLP and SpikingFLABlock]

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        # Removed F.gelu to ensure a more strictly spiking internal flow
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x)
        # Removed F.gelu to maintain spiking nature: x (real output of fc1) directly feeds to dropout and then fc2
        x, spike2 = self.fc2(self.dropout(x))
        return x, spike1, spike2

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, use_spiking_mlp, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SpikingKDA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult
        if use_spiking_mlp:
            self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model)) # Kept for non-spiking fallback
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        all_spikes_list = []
        x_res = x
        if self.prenorm: x = self.norm1(x)
        x_attn, _, attn_spikes = self.attn(x, **kwargs)
        if attn_spikes: all_spikes_list.extend(attn_spikes.values()) # attn_spikes is a dict
        x = x_res + self.dropout1(x_attn)
        x_res = x
        if self.prenorm: x = self.norm2(x)
        if isinstance(self.mlp, SpikingMLP):
            x_mlp, s1, s2 = self.mlp(x)
            all_spikes_list.extend([s1, s2])
        else:
            x_mlp = self.mlp(x)
        x = x_res + self.dropout2(x_mlp)
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None
        return x, None, final_spikes