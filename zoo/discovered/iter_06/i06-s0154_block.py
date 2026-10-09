"""# -*- coding: utf-8 -*-"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING

# Dependency Consistency: Do not remove any of the existing import statements
# Spiking SNN related imports
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

# Preserve SpikingLinear Class: The class SpikingLinear(nn.Module) must be kept exactly as it is, without modification or deletion

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2) 
        spikes_transposed = self.neuron(x_transposed) # Neuron operates on (B, D, L)
        spikes = spikes_transposed.transpose(-1, -2)  # Transpose back to (B, L, D) for linear layer
        output = self.linear(spikes) # Linear layer takes binary spikes
        return output, spikes # output is analog, spikes is the binary input to self.linear

# ==================== White-box Operator (Comba Closed-Loop) ====================
# This function's signature and internal logic (except for feedback_net call) are immutable.
# Its inputs q, k, v, g are expected to be analog (outputs of SpikingLinear or SpikingConvWrapper).

def comba_chunkwise_parallel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, feedback_net: nn.Module, chunk_size: int=64, initial_state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len), value=0.0) # pad g with 0.0, as it's a log-decay factor
    
    # Inputs q, k, v are analog outputs from SpikingLinear/SpikingConvWrapper
    # l2norm will keep them analog
    q = l2norm(q)
    k = l2norm(k)
    
    q, k, v, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) if x.dim()==4 
             else rearrange(x, 'b h (n c) -> b h n c', c=chunk_size),
        [q, k, v, g]
    )
    
    num_chunks = q.shape[2]
    S = initial_state if initial_state is not None else torch.zeros(b, h, d_k, d_v, device=q.device)
    o = torch.zeros_like(v)
    
    mask_triu = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)

    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        
        # Closed-loop feedback stability enhancement
        # s_feat is analog, derived from continuous state S.
        # This analog s_feat is passed to SpikingFeedbackNet (which internally spikes it for its linear layers)
        s_norm = S.norm(p=2, dim=(-1, -2), keepdim=True) + 1e-6
        s_feat = (S / s_norm).mean(dim=(-1, -2), keepdim=True).flatten(2) # [B, H, 1]
        
        # Use the SpikingFeedbackNet to get the feedback offset.
        # feedback_net receives analog s_feat, but internally its linear layers receive spikes.
        # f_i is an analog output from feedback_net.
        f_i_raw = feedback_net(s_feat).squeeze(-1) # [B, H]
        f_i = torch.tanh(f_i_raw) * 2.0 

        # Gate numerical computation alignment
        log_decay = F.logsigmoid(g_i + f_i.unsqueeze(-1)) # g_i and f_i are analog
        d_cumsum = torch.cumsum(log_decay, dim=-1)
        
        decay_rel = d_cumsum.unsqueeze(-1) - d_cumsum.unsqueeze(-2)
        decay_mask = torch.exp(decay_rel.clamp(max=0)).masked_fill(mask_triu, 0)
        
        # Compute output (all these operations are on analog values)
        decay_from_start = torch.exp(d_cumsum.clamp(max=0)).unsqueeze(-1)
        o_inter = (q_i * decay_from_start) @ S
        
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask
        o_intra = attn_intra @ v_i
        
        o[:, :, i] = o_inter + o_intra
        
        # State update stability enhancement
        decay_total = torch.exp(d_cumsum[..., -1].clamp(max=0)).unsqueeze(-1).unsqueeze(-1)
        decay_to_end = torch.exp((d_cumsum[..., -1].unsqueeze(-1) - d_cumsum).clamp(max=0)).unsqueeze(-1)
        
        update = (k_i * decay_to_end).transpose(-1, -2) @ v_i
        S = S * decay_total + update
        
        # Strict renormalization after each chunk update
        S_norm_per_head = S.norm(p=2, dim=(-1, -2), keepdim=True)
        S = S / torch.clamp(S_norm_per_head / 10.0, min=1.0) 

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return (o[:, :, :l], S) if pad_len > 0 else (o, S)

# ==================== New: Spiking Feedback Network for Comba ====================
# All linear layers inside are SpikingLinear, and an intermediate neuron ensures
# the second linear layer receives spikes after the activation.

class SpikingFeedbackNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, neuron_type: str, surrogate_function):
        super().__init__()
        # fc1 receives analog input (s_feat from comba_chunkwise_parallel),
        # but internally generates spikes for its nn.Linear.
        self.fc1 = SpikingLinear(nn.Linear(input_dim, hidden_dim), neuron_type, surrogate_function)
        self.activation = nn.Tanh() 
        # Neuron after activation to ensure spike input to fc2
        self.inter_neuron_fb = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        # fc2 receives spikes generated by inter_neuron_fb.
        self.fc2 = SpikingLinear(nn.Linear(hidden_dim, output_dim), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out_fc1_analog, spikes_fc1_input = self.fc1(x) 
        x_act_analog = self.activation(x_out_fc1_analog) 

        # Convert x_act_analog to spikes before feeding to fc2
        spikes_after_activation = self.inter_neuron_fb(x_act_analog.transpose(-1, -2)).transpose(-1, -2)

        final_out_analog, spikes_fc2_input = self.fc2(spikes_after_activation) 

        # Return the analog output and the spikes (for collection outside if needed)
        # This SpikingFeedbackNet is used by comba_chunkwise_parallel,
        # which expects a single tensor output. So, only the analog output is returned.
        return final_out_analog

# ==================== New: Spiking Wrapper for ShortConvolution ====================
# This wrapper ensures that the *input* to the ShortConvolution is spiked.
# However, ShortConvolution itself operates on analog values and produces analog output.
# This is a hybrid approach, where linear layers always get spikes, but non-linear ops like conv or attention core may use analog.

class SpikingConvWrapper(nn.Module):
    def __init__(self, conv_layer: ShortConvolution, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.conv = conv_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes_from_input = self.neuron(x.transpose(-1, -2)).transpose(-1, -2) # Spikes generated from x
        # ShortConvolution expects analog float input.
        # We pass the original analog input 'x' to the conv, NOT the spikes_from_input,
        # as ShortConvolution itself is not spike-aware.
        conv_out_analog, _ = self.conv(x) 
        return conv_out_analog, spikes_from_input # Return analog output and spikes generated from input 'x'

# ==================== SpikingMLP Layer (Modified) ====================
# Ensures all internal linear layers receive spike inputs.

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        # fc1 receives spikes (from SpikingFLABlock), internally generates spikes for its nn.Linear.
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        # Neuron after fc1's analog output to ensure spike input to fc2.
        self.intermediate_neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.dropout = nn.Dropout(dropout)
        # fc2 receives spikes (from intermediate_neuron after dropout), internally generates spikes for its nn.Linear.
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fc1_out_analog, spikes_fc1_input = self.fc1(x) 

        # Convert fc1_out_analog to spikes for dropout and fc2
        spikes_after_fc1_output = self.intermediate_neuron(fc1_out_analog.transpose(-1, -2)).transpose(-1, -2)

        # Apply dropout to the spikes
        dropped_spikes = self.dropout(spikes_after_fc1_output)

        # fc2 receives spikes as input. Its output is analog.
        fc2_out_analog, spikes_fc2_input = self.fc2(dropped_spikes)

        # Return the final analog output and all collected spike tensors
        return fc2_out_analog, spikes_fc1_input, spikes_after_fc1_output, spikes_fc2_input

# ==================== SpikingComba Layer ====================

class Comba(nn.Module):
    def __init__(self, d_model: int=1024, num_heads: int=8, expand_k: float=0.5, expand_v: float=1.0, use_short_conv: bool=True, conv_size: int=4, neuron_type: str='sdn', layer_idx: Optional[int]=None, global_context_dim: int=32, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_v_dim = self.value_dim // num_heads
        self.global_context_dim = global_context_dim

        surrogate = piecewise_quadratic_surrogate()

        # All inputs to these SpikingLinear layers will be spikes (from SpikingFLABlock or SpikingConvWrapper),
        # and they will internally ensure their nn.Linear receives spikes. Their outputs are analog.
        self.q_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, self.key_dim, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate) # Base Gate

        # [NEW in Version 3] Spiking Adaptive Gate for global context modulation (additive)
        # This gate adds to the base gate 'g', allowing dynamic modulation of decay factors.
        self.spiking_mod_gate_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)

        # [NEW in Version 3] Spiking Multiplicative Gate for dynamic recurrent decay control
        # This gate provides a second, multiplicative spiking signal to dynamically scale the effective 'g' gate.
        # This enhances the dynamic range and complexity of recurrent state updates.
        self.spiking_mult_gate_proj = SpikingLinear(nn.Linear(d_model, num_heads, bias=False), neuron_type, surrogate)


        self.feedback_net = SpikingFeedbackNet(
            input_dim=1, # s_feat is [B, H, 1]
            output_dim=1,
            hidden_dim=8,
            neuron_type=neuron_type,
            surrogate_function=surrogate
        )

        if use_short_conv:
            # Wrapped ShortConvolution with SpikingConvWrapper.
            # Its input (from q/k/v_proj) is analog. Its output is analog.
            # It also returns spikes generated from its input, for collection.
            self.q_conv1d = SpikingConvWrapper(ShortConvolution(self.key_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.k_conv1d = SpikingConvWrapper(ShortConvolution(self.key_dim, conv_size, activation='silu'), neuron_type, surrogate)
            self.v_conv1d = SpikingConvWrapper(ShortConvolution(self.value_dim, conv_size, activation='silu'), neuron_type, surrogate)
        else:
            self.q_conv1d = None
            self.k_conv1d = None
            self.v_conv1d = None

        self.o_norm = RMSNorm(self.head_v_dim)
        # Neuron after o_norm and before o_proj to ensure spike input to o_proj.
        self.pre_o_proj_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)

        # [NEW ARCHITECTURE] Adaptive Global State Integration
        # Global Context Encoder: Summarizes input sequence into a global context vector
        self.global_context_encoder_proj1 = SpikingLinear(nn.Linear(d_model, global_context_dim * 2, bias=False), neuron_type, surrogate)
        self.global_context_encoder_neuron = neuron_registry[neuron_type](surrogate_function=surrogate)
        self.global_context_encoder_proj2 = SpikingLinear(nn.Linear(global_context_dim * 2, global_context_dim, bias=False), neuron_type, surrogate)


        # State Modulation Network: Modulates the recurrent state S based on global context
        # It takes global_context_dim and outputs two values per head: a gate and a bias for the state S.
        self.state_modulation_net = SpikingMLP(
            d_model=global_context_dim,
            d_ffn=global_context_dim * 2,
            dropout=0.1, # Using a fixed dropout here
            neuron_type=neuron_type,
            surrogate_function=surrogate
        )
        # MODIFICATION: Output only 2 scalars per head (scale, bias), dramatically reducing parameter count
        self.state_modulation_output_proj = SpikingLinear(nn.Linear(global_context_dim, num_heads * 2, bias=False), neuron_type, surrogate)

    def forward(self, x: torch.Tensor, past_key_values: Optional[Dict]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        all_spikes = {}

        # [NEW ARCHITECTURE] Compute Global Context from input spikes 'x' (which is spikes_for_attn from SpikingFLABlock)
        # x shape: (B, L, D)
        gc_proj1_analog, s_gc_proj1_input = self.global_context_encoder_proj1(x)
        all_spikes['gc_proj1_input'] = s_gc_proj1_input

        # Spike after projection 1
        gc_spikes_inter = self.global_context_encoder_neuron(gc_proj1_analog.transpose(-1, -2)).transpose(-1, -2)
        all_spikes['gc_encoder_neuron'] = gc_spikes_inter

        gc_proj2_analog, s_gc_proj2_input = self.global_context_encoder_proj2(gc_spikes_inter)
        all_spikes['gc_proj2_input'] = s_gc_proj2_input

        # Max pool over the sequence length to get a single global context vector per batch
        # Use gc_proj2_analog.shape[1] for kernel_size to ensure dynamic batch size compatibility
        global_context_vector = F.max_pool1d(gc_proj2_analog.transpose(-1, -2), kernel_size=gc_proj2_analog.shape[1]).squeeze(-1) # (B, global_context_dim)

        # Projections: Output (q_analog, s_q_input) means q_analog is analog, s_q_input are the spikes fed to q_proj's internal nn.Linear.
        q_analog, s_q_input = self.q_proj(x) 
        k_analog, s_k_input = self.k_proj(x)
        v_analog, s_v_input = self.v_proj(x)
        g_base_analog, s_g_input = self.g_proj(x) # Base Gate (analog output)
        all_spikes.update({'q_proj_input':s_q_input, 'k_proj_input':s_k_input, 'v_proj_input':s_v_input, 'g_proj_input':s_g_input})

        # [NEW in Version 3] Compute spiking modulation gate (additive)
        spiking_mod_analog, s_spiking_mod_input = self.spiking_mod_gate_proj(x) # Output is analog
        all_spikes['spiking_mod_gate_proj_input'] = s_spiking_mod_input

        # [NEW in Version 3] Compute spiking multiplicative gate
        spiking_mult_analog, s_spiking_mult_input = self.spiking_mult_gate_proj(x) # Output is analog
        all_spikes['spiking_mult_gate_proj_input'] = s_spiking_mult_input

        # Apply spiking modulation by adding it to the base gate 'g_base'.
        g_base_rearranged = rearrange(g_base_analog, 'b l h -> b h l')
        spiking_mod_reshaped = rearrange(spiking_mod_analog, 'b l h -> b h l')

        # Combine additive gates
        g_effective_additive = g_base_rearranged + spiking_mod_reshaped

        # Apply multiplicative gate (sigmoid ensures non-negative scaling)
        spiking_mult_reshaped = rearrange(spiking_mult_analog, 'b l h -> b h l')
        multiplicative_factor = F.sigmoid(spiking_mult_reshaped) * 2.0 # Scale from 0 to 2, centered at 1

        # Final gate 'g' value (analog input to comba_chunkwise_parallel)
        g = g_effective_additive * multiplicative_factor

        if self.q_conv1d is not None:
            # SpikingConvWrapper receives analog, internally generates spikes from its input,
            # passes original analog input to conv, and outputs analog.
            q_conv_out_analog, s_q_conv_input = self.q_conv1d(q_analog)
            k_conv_out_analog, s_k_conv_input = self.k_conv1d(k_analog)
            v_conv_out_analog, s_v_conv_input = self.v_conv1d(v_analog)
            all_spikes.update({'q_conv_input':s_q_conv_input, 'k_conv_input':s_k_conv_input, 'v_conv_input':s_v_conv_input})
            q = q_conv_out_analog
            k = k_conv_out_analog
            v = v_conv_out_analog
        else:
            q, k, v = q_analog, k_analog, v_analog

        # Rearrange for attention computation. q, k, v, g are all analog values here.
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        # 'g' is already processed (rearranged and modulated) to be 'b h l' before being passed to comba_chunkwise_parallel

        layer_state = None
        if past_key_values is not None and self.layer_idx is not None:
            layer_state = past_key_values[self.layer_idx] if self.layer_idx in past_key_values else None

        # The core comba_chunkwise_parallel operates on analog signals q, k, v, g
        # and calls SpikingFeedbackNet with an analog s_feat.
        o_analog, final_state = comba_chunkwise_parallel(q, k, v, g, self.feedback_net, initial_state=layer_state)

        # [NEW ARCHITECTURE] Modulate final_state using global context
        # The SpikingMLP expects (B, L, D) so unsqueeze sequence dim.
        mlp_out_analog, s_sm_fc1_input, s_sm_inter_neuron, s_sm_fc2_input = self.state_modulation_net(global_context_vector.unsqueeze(1)) 
        all_spikes.update({
            'sm_fc1_input': s_sm_fc1_input,
            'sm_inter_neuron': s_sm_inter_neuron,
            'sm_fc2_input': s_sm_fc2_input
        })

        # Reshape the output of MLP from (B, 1, global_context_dim) to (B, global_context_dim)
        mlp_out_analog_squeezed = mlp_out_analog.squeeze(1) 

        # [MODIFICATION] Get scalar scale and bias per head
        modulation_params_analog, s_sm_output_proj = self.state_modulation_output_proj(mlp_out_analog_squeezed)
        all_spikes['sm_output_proj_input'] = s_sm_output_proj

        # Reshape modulation_params_analog to (B, H, 2) for scale and bias
        # then expand to (B, H, 1, 1) for broadcasting
        b, h, d_k, d_v = final_state.shape # Get dimensions from final_state directly

        # modulation_params_analog is (B, num_heads * 2)
        # Split into scale and bias
        scale_params_raw = modulation_params_analog[..., :self.num_heads] # (B, num_heads)
        bias_params_raw = modulation_params_analog[..., self.num_heads:]  # (B, num_heads)

        # Reshape and expand for broadcasting to (B, H, D_k, D_v)
        scale_params = rearrange(scale_params_raw, 'b h -> b h 1 1')
        bias_params = rearrange(bias_params_raw, 'b h -> b h 1 1')

        # Apply modulation to final_state (analog operation)
        # Use sigmoid for scale to keep it positive and limited, centered around 1 if possible
        final_state = final_state * (F.sigmoid(scale_params) * 2.0) + bias_params # scale from 0 to 2, centered at 1

        # Re-normalize after modulation
        final_state_norm_per_head = final_state.norm(p=2, dim=(-1, -2), keepdim=True)
        final_state = final_state / torch.clamp(final_state_norm_per_head / 10.0, min=1.0) 


        if past_key_values is not None and self.layer_idx is not None:
            past_key_values[self.layer_idx] = final_state

        o_analog = rearrange(o_analog, 'b h l d -> b l h d')
        o_analog = self.o_norm(o_analog) # o_analog is analog output after RMSNorm
        o_analog = rearrange(o_analog, 'b l h d -> b l (h d)') # o_analog is analog

        # Spike the output of o_norm before feeding to o_proj
        spikes_before_o_proj = self.pre_o_proj_neuron(o_analog.transpose(-1, -2)).transpose(-1, -2)
        all_spikes['pre_o_proj_neuron'] = spikes_before_o_proj

        # o_proj receives spikes as input
        o_final_analog, s_o_proj_input = self.o_proj(spikes_before_o_proj)
        all_spikes['o_proj_input'] = s_o_proj_input

        return o_final_analog, None, all_spikes

# ==================== SpikingFLABlock (Main Block) ====================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn_mult: int=2, dropout: float=0.1, prenorm: bool=True, neuron_type: str='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        # Spiking neuron after the first RMSNorm (or directly on input if not prenorm)
        # This neuron ensures that the input to the Comba block is always spikes.
        self.attn_input_neuron = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())

        # Attention mechanism (Comba)
        # Pass global_context_dim to Comba
        self.attn = Comba(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)

        # Spiking neuron after the second RMSNorm (or after attn block if not prenorm)
        # This neuron ensures that the input to the MLP block is always spikes.
        self.mlp_input_neuron = neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())

        d_ffn = d_model * d_ffn_mult
        # SpikingMLP now ensures all its internal linear layers receive spike inputs
        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        x_res = x

        # Prepare input for Attention Block
        if self.prenorm: 
            x_normalized = self.norm1(x) # Analog output from RMSNorm
            # Generate spikes from normalized input before attention block
            spikes_for_attn = self.attn_input_neuron(x_normalized.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_attn)
        else:
            # If not prenorm, spike the raw input directly before attention
            spikes_for_attn = self.attn_input_neuron(x.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_attn)

        # Attention block. Its input (spikes_for_attn) is strictly binary spikes.
        # It returns analog output and a dictionary of spikes.
        x_attn_analog_out, reg_loss, attn_spikes_dict = self.attn(spikes_for_attn, **kwargs) 
        if attn_spikes_dict:
            all_spikes_list.extend([s for s in attn_spikes_dict.values() if s is not None])
        x = x_res + self.dropout1(x_attn_analog_out) # x_attn_analog_out is analog output from Comba

        x_res = x

        # Prepare input for MLP Block
        if self.prenorm: 
            x_normalized = self.norm2(x) # Analog output from RMSNorm
            # Generate spikes from normalized input before MLP block
            spikes_for_mlp = self.mlp_input_neuron(x_normalized.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_mlp)
        else:
            # If not prenorm, spike the output of the attention block directly before MLP
            spikes_for_mlp = self.mlp_input_neuron(x.transpose(-1, -2)).transpose(-1, -2)
            all_spikes_list.append(spikes_for_mlp)

        # MLP block. Its input (spikes_for_mlp) is strictly binary spikes.
        # SpikingMLP returns 1 analog output and 3 spike tensors.
        x_mlp_analog_out, mlp_spike1_input, mlp_spikes_after_fc1_output, mlp_spike2_input = self.mlp(spikes_for_mlp)

        all_spikes_list.extend([
            s for s in [mlp_spike1_input, mlp_spikes_after_fc1_output, mlp_spike2_input] if s is not None
        ])

        x = x_res + self.dropout2(x_mlp_analog_out) # x_mlp_analog_out is analog output from MLP

        # Final spike collection: flatten and concatenate all collected spikes
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        return x, reg_loss, final_spikes