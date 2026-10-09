import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Tuple, List, Optional, Dict, TYPE_CHECKING

# Ensure all necessary imports are present for the SNN modules
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
    from src.models.nn.dropout import DropoutNd
except (ImportError, ModuleNotFoundError):
    # Dummy implementations for local testing if imports fail
    class DummySurrogate:
        @staticmethod
        def apply(x, *args, **kwargs):
            return (x > 0).float()
    class DummyNeuron(nn.Module):
        def __init__(self, surrogate_function=None, **kwargs):
            super().__init__()
            self.surrogate_function = surrogate_function if surrogate_function else DummySurrogate.apply
        def forward(self, x):
            return self.surrogate_function(x)
    neuron_registry = {'sdn': lambda surrogate_function: DummyNeuron(surrogate_function)}
    piecewise_quadratic_surrogate = lambda: None
    DropoutNd = nn.Dropout # Fallback for DropoutNd

# FLA Specific Imports
from fla.modules import ShortConvolution


# SpikingLinear (MUST BE KEPT AS IS - copied from Version 1 description)
# ==================== SpikingLinear (MUST BE KEPT AS IS) ====================

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D) # Binary spikes
        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
        return output, spikes # output is analog, spikes are binary

# ==================== White-box Operator (S4DKernel, SS4D) ====================

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

        self.channels = channels

        d_ffn_mult_ss4d = 2
        self.d_ffn_ss4d = int(self.h * d_ffn_mult_ss4d)

        # Spiking Gating Unit: Takes the SS4D input (after initial transpose if transposed is True)
        # and generates spikes to gate the processed features.
        # Input to gate_proj will be (B*L, H), output is (B*L, C*H) spikes
        self.gate_proj = SpikingLinear(
            nn.Linear(self.h, self.channels * self.h, bias=False),
            neuron_type=neuron, surrogate_function=piecewise_quadratic_surrogate()
        )

        # First MLP layer: input to SpikingLinear should be analog. SpikingLinear spikes internally.
        self.output_mlp_linear1 = SpikingLinear(
            nn.Linear(self.channels * self.h, self.d_ffn_ss4d, bias=True),
            neuron_type=neuron, surrogate_function=piecewise_quadratic_surrogate()
        )
        self.output_mlp_dropout1 = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # Second MLP layer: input to SpikingLinear should be analog. SpikingLinear spikes internally.
        self.output_mlp_linear2 = SpikingLinear(
            nn.Linear(self.d_ffn_ss4d, self.h, bias=True),
            neuron_type=neuron, surrogate_function=piecewise_quadratic_surrogate()
        )
        self.output_mlp_dropout2 = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, u, **kwargs):
        all_ss4d_spikes = []

        # u is expected to be (B, D, L) if self.transposed is True
        # If transposed is False, u is (B, L, D), and it gets transposed to (B, D, L) here.
        if not self.transposed:
            u_in_core = u.transpose(-1, -2) # (B, D, L)
        else:
            u_in_core = u # (B, D, L)

        b, h_dim, L = u_in_core.shape # b, self.h, L

        # 1. Core SS4D Kernel computation
        k, _ = self.kernel(L=L)
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u_in_core, n=2 * L)

        y_fft = k_f.unsqueeze(0) * u_f.unsqueeze(1) # (B, C, H, L_fft)
        y_fft = torch.fft.irfft(y_fft, n=2 * L)[..., :L]
        y_direct = torch.einsum("bhl,ch->bchl", u_in_core, self.D)

        y_continuous_pre_mlp = y_fft + y_direct # (B, C, H, L)
        y_continuous_pre_mlp = rearrange(y_continuous_pre_mlp, "b c h l -> b (c h) l") # (B, C*H, L)

        if self.learnable_vth:
            # Apply learnable threshold scaling. This is a continuous operation.
            y_continuous_pre_mlp = y_continuous_pre_mlp / torch.exp(self.ln_vth)

        # 2. Spiking Gating Unit
        # Input to gate_proj: u_in_core (B, H, L)
        # Reshape u_in_core to (B*L, H) for SpikingLinear input
        u_reshaped_for_gate_input = rearrange(u_in_core, 'b h l -> (b l) h')
        _, gate_spikes_flat = self.gate_proj(u_reshaped_for_gate_input)
        all_ss4d_spikes.append(gate_spikes_flat.flatten()) # Append flattened spikes

        # Reshape gate_spikes_flat from (B*L, C*H) back to (B, C*H, L) to apply element-wise
        gate_spikes = rearrange(gate_spikes_flat, '(b l) ch -> b ch l', b=b, l=L)

        # Apply the spiking gate: multiply with the continuous features
        y_continuous_gated = y_continuous_pre_mlp * gate_spikes # Element-wise multiplication, introducing sparsity

        # 3. Output MLPs (already use SpikingLinear internally)
        # Reshape for MLPs: (B, C*H, L) -> (B*L, C*H)
        x_reshaped_for_mlp_input = rearrange(y_continuous_gated, 'b ch l -> (b l) ch')

        # First MLP Linear layer (SpikingLinear takes analog input, internally spikes it)
        x_out1_continuous, spikes_from_linear1_internal_neuron = self.output_mlp_linear1(x_reshaped_for_mlp_input)
        all_ss4d_spikes.append(spikes_from_linear1_internal_neuron.flatten()) # Append flattened spikes
        x_out1_continuous = self.output_mlp_dropout1(x_out1_continuous)

        # Second MLP Linear layer (SpikingLinear takes analog input, internally spikes it)
        x_out2_continuous, spikes_from_linear2_internal_neuron = self.output_mlp_linear2(x_out1_continuous)
        all_ss4d_spikes.append(spikes_from_linear2_internal_neuron.flatten()) # Append flattened spikes
        x_out2_continuous = self.output_mlp_dropout2(x_out2_continuous)

        # Reshape back to (B, H, L)
        y_final_continuous = rearrange(x_out2_continuous, '(b l) h -> b h l', b=b, l=L)

        if not self.transposed: y_final_continuous = y_final_continuous.transpose(-1, -2) # (B, L, D)

        # Concatenate all collected spikes or return empty if none
        combined_ss4d_spikes = torch.cat([s.flatten() for s in all_ss4d_spikes if s is not None and s.numel() > 0]) if all_ss4d_spikes else torch.empty(0, device=u.device)

        return y_final_continuous, combined_ss4d_spikes

# ==================== SpikingResidualGate (Modified from Version 1) ====================
# This class is reused to introduce spiking dynamics into the residual connection.

class SpikingResidualGate(nn.Module):
    def __init__(self, d_model: int, neuron_type: str, surrogate_function, use_conv: bool=True, conv_size: int=3, **kwargs):
        super().__init__()
        self.use_conv = use_conv
        # All linear layers must be SpikingLinear and receive binary spikes internally
        self.forget_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)
        self.input_proj = SpikingLinear(nn.Linear(d_model, d_model, bias=False), neuron_type, surrogate_function)

        if use_conv:
            # ShortConvolution takes (B, L, D) and returns (B, L, D)
            self.forget_conv = ShortConvolution(d_model, conv_size, activation='silu')
            self.input_conv = ShortConvolution(d_model, conv_size, activation='silu')
        else:
            self.forget_conv = nn.Identity()
            self.input_conv = nn.Identity()

    def forward(self, x_res: torch.Tensor, branch_output: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_spikes = []

        # Inputs x_res and branch_output are (B, L, D)
        forget_conv_out, _ = self.forget_conv(x_res)
        input_conv_out, _ = self.input_conv(branch_output)

        # SpikingLinear takes analog input, produces analog output and binary spikes
        forget_gate_cont, s_forget_gate = self.forget_proj(forget_conv_out)
        input_gate_cont, s_input_gate = self.input_proj(input_conv_out)

        all_spikes.extend([s_forget_gate, s_input_gate])

        # Sigmoid activation to create gating factors
        forget_gate_factor = torch.sigmoid(forget_gate_cont)
        input_gate_factor = torch.sigmoid(input_gate_cont)

        # Apply the gated residual connection
        gated_output = x_res * (1.0 - forget_gate_factor) + branch_output * input_gate_factor

        return gated_output, all_spikes

# ==================== SpikingFLABlock (Enhanced with SpikingResidualGate) ====================

class SpikingFLABlock(nn.Module):
    """This class implements an enhanced SpikingFLABlock, leveraging a Spiking State Space Model (SS4D)
    as its core computational unit. It ensures all internal linear operations utilize SpikingLinear
    layers, which internally receive binary spike inputs.
    
    Key enhancements:
    1. **SS4D Core**: Retains the efficient SS4D for long-range dependency modeling.
    2. **Spiking Residual Gate**: Introduces a `SpikingResidualGate` after the SS4D output.
       This gate uses spiking neurons to dynamically control the information flow between the
       main SS4D branch and the residual connection, adding a neuromorphic gating mechanism
       to the fundamental residual pathway. This promotes sparsity and event-driven processing
       in a critical architectural component.
    3. **Parameter Streamlining**: Removes unused parameters from the `__init__` signature for clarity.
    4. **Comprehensive Spike Collection**: Gathers spikes from both the SS4D and the new Residual Gate
       for accurate firing rate regularization and analysis.
    
    The block maintains sub-quadratic complexity, supports chunked processing (inherently by SS4D),
    and ensures batch-size agnostic operations.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, neuron_type: Optional[str]=None, spike_regularization_weight: float=0.01, use_residual_gate: bool=True, residual_gate_conv_size: int=3, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.neuron_type = neuron_type if neuron_type is not None else 'sdn' # Default neuron type
        self.spike_regularization_weight = spike_regularization_weight
        self.use_residual_gate = use_residual_gate

        # Normalization layer
        self.norm = nn.LayerNorm(d_model) # Preserving LayerNorm as per Version 2
        self.dropout_out = DropoutNd(dropout)

        # Initialize the SS4D layer with appropriate parameters
        ss4d_kwargs = kwargs.get('layer', {}) # Allows passing SS4D specific args via 'layer' dict in kwargs
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001)) # Default LR for SS4D parameters
        ss4d_kwargs.setdefault('neuron', self.neuron_type)
        ss4d_kwargs.setdefault('dropout', dropout)
        ss4d_kwargs.setdefault('learnable_vth', True) # Enable adaptive thresholding by default
        ss4d_kwargs.setdefault('d_state', 64) # Default S4 state dimension

        self.layer = SS4D(d_model, transposed=True, **ss4d_kwargs)
        self.gain = nn.Parameter(torch.ones(1)) # Learnable gain for normalized input

        # Initialize the SpikingResidualGate to introduce spiking dynamics into the residual path
        if self.use_residual_gate:
            surrogate_function = piecewise_quadratic_surrogate()
            self.post_ss4d_residual_gate = SpikingResidualGate(
                d_model=d_model,
                neuron_type=self.neuron_type,
                surrogate_function=surrogate_function,
                use_conv=True, # Use ShortConvolution within the gate for local temporal context
                conv_size=residual_gate_conv_size
            )
        else:
            self.post_ss4d_residual_gate = nn.Identity() # Placeholder if gate is disabled

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for the SpikingFLABlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            **kwargs: Additional keyword arguments for compatibility.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                - Continuous output tensor of shape (B, L, D).
                - Total regularization loss (Optional[torch.Tensor]).
                - Concatenated flattened binary spikes from all spiking neurons (Optional[torch.Tensor]).
        """
        x_res = x # Store input for residual connection (B, L, D)
        all_spikes_list: List[torch.Tensor] = [] # List to collect all spike tensors

        # Pre-normalization step
        if self.prenorm:
            x_normalized = self.norm(x)
        else:
            x_normalized = x

        x_scaled = x_normalized * self.gain # Apply learnable gain

        # Prepare input for SS4D: (B, L, D) -> (B, D, L)
        x_ss4d_input = rearrange(x_scaled, 'b l d -> b d l')

        # Apply the core SS4D layer
        x_ss4d_output_continuous_transposed, ss4d_all_spikes = self.layer(x_ss4d_input)
        all_spikes_list.append(ss4d_all_spikes) # Collect spikes from SS4D

        # Apply dropout to the continuous output of SS4D
        x_ss4d_output_continuous_transposed = self.dropout_out(x_ss4d_output_continuous_transposed)

        # Transpose SS4D output back to standard shape: (B, D, L) -> (B, L, D)
        x_ss4d_output_branch = rearrange(x_ss4d_output_continuous_transposed, 'b d l -> b l d')

        # Apply SpikingResidualGate to integrate SS4D output with the residual connection
        if self.use_residual_gate:
            output, residual_gate_spikes = self.post_ss4d_residual_gate(x_res, x_ss4d_output_branch)
            all_spikes_list.extend(residual_gate_spikes) # Collect spikes from the residual gate
        else:
            # If residual gate is disabled, use standard residual addition
            output = x_res + x_ss4d_output_branch

        # Post-normalization step (if prenorm is False)
        if not self.prenorm:
            output = self.norm(output)

        total_reg_loss = None

        # Concatenate all collected spikes into a single flattened tensor
        # Handle cases where all_spikes_list might be empty or contain empty tensors
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None and s.numel() > 0]) if all_spikes_list and any(s is not None and s.numel() > 0 for s in all_spikes_list) else torch.empty(0, device=x.device)

        # Calculate firing rate regularization loss
        if self.spike_regularization_weight > 0 and final_spikes.numel() > 0:
            # Ensure final_spikes is float for mean calculation
            firing_rate_loss = final_spikes.float().mean() * self.spike_regularization_weight
            total_reg_loss = firing_rate_loss
        elif self.spike_regularization_weight > 0: # If weight is > 0 but no spikes were generated
            total_reg_loss = torch.tensor(0.0, device=x.device)

        return output, total_reg_loss, final_spikes