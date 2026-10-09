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


# SpikingLinear (MUST BE KEPT AS IS - copied from Version 1 description)
# ==================== SpikingLinear (MUST BE KEPT AS IS) ====================

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (..., D, L)
        spikes_transposed = self.neuron(x_transposed) # Generate spikes (binary signal)
        spikes = spikes_transposed.transpose(-1, -2)  # (..., D, L) -> (..., L, D) # Binary spikes
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

        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        k, _ = self.kernel(L=L)
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)

        y_fft = torch.einsum("bhl,chl->bchl", u_f, k_f)
        y_fft = torch.fft.irfft(y_fft, n=2 * L)[..., :L]
        y_direct = torch.einsum("bhl,ch->bchl", u, self.D)

        y_continuous_pre_mlp = y_fft + y_direct # (B, C, H, L)
        y_continuous_pre_mlp = rearrange(y_continuous_pre_mlp, "b c h l -> b (c h) l") # (B, C*H, L) and continuous

        if self.learnable_vth: y_continuous_pre_mlp = y_continuous_pre_mlp / torch.exp(self.ln_vth) # (B, C*H, L) and continuous

        # Reshape to (Batch*SeqLen, Features) for SpikingLinear
        b, ch_dim, l = y_continuous_pre_mlp.shape
        x_reshaped_for_linear_input = rearrange(y_continuous_pre_mlp, 'b ch l -> (b l) ch') # Analog input for SpikingLinear

        # First MLP Linear layer (SpikingLinear takes analog input, internally spikes it)
        x_out1_continuous, spikes_from_linear1_internal_neuron = self.output_mlp_linear1(x_reshaped_for_linear_input)
        all_ss4d_spikes.append(spikes_from_linear1_internal_neuron.flatten()) # Collect spikes generated *inside* SpikingLinear
        x_out1_continuous = self.output_mlp_dropout1(x_out1_continuous) # Apply dropout to continuous output

        # Second MLP Linear layer (SpikingLinear takes analog input, internally spikes it)
        x_out2_continuous, spikes_from_linear2_internal_neuron = self.output_mlp_linear2(x_out1_continuous)
        all_ss4d_spikes.append(spikes_from_linear2_internal_neuron.flatten()) # Collect spikes generated *inside* SpikingLinear
        x_out2_continuous = self.output_mlp_dropout2(x_out2_continuous) # Apply dropout to continuous output

        # Reshape back to (B, H, L)
        y_final_continuous = rearrange(x_out2_continuous, '(b l) h -> b h l', b=b, l=l)

        if not self.transposed: y_final_continuous = y_final_continuous.transpose(-1, -2)

        combined_ss4d_spikes = torch.cat(all_ss4d_spikes)

        return y_final_continuous, combined_ss4d_spikes

# ==================== SpikingFLABlock ====================

class SpikingFLABlock(nn.Module):
    """This class is a single layer/block that integrates the Spiking State Space Model (SS4D)
    as its core, ensuring all internal linear operations are replaced by SpikingLinear
    and receive binary inputs generated by spiking neurons. It adheres to a standard
    (pre-norm, residual) structure and includes spike firing rate regularization.
    It introduces a Spiking Gated Residual mechanism, where the SS4D output
    is modulated by binary spikes derived from the normalized input.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, neuron_type: Optional[str]=None, spike_regularization_weight: float=0.01, use_spiking_gate: bool=True, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.neuron_type = neuron_type if neuron_type is not None else 'sdn' # Default neuron type
        self.spike_regularization_weight = spike_regularization_weight
        self.use_spiking_gate = use_spiking_gate

        # Surrogate function for spiking neurons
        self.surrogate_function = piecewise_quadratic_surrogate()

        # Normalization layer
        self.norm = nn.LayerNorm(d_model) # Operates on continuous signals
        self.dropout_out = DropoutNd(dropout) # Renamed to avoid clash if SS4D also has a dropout attribute

        # The main SS4D layer
        ss4d_kwargs = kwargs.get('layer', {}) # Get SS4D specific kwargs
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs.setdefault('neuron', self.neuron_type) # Propagate neuron type to SS4D
        ss4d_kwargs.setdefault('dropout', dropout) # Pass dropout to SS4D for its internal MLPs
        ss4d_kwargs.setdefault('learnable_vth', True) # Ensure adaptive thresholding is enabled by default

        self.layer = SS4D(d_model, transposed=True, **ss4d_kwargs)
        self.gain = nn.Parameter(torch.ones(1)) # For potential scaling after norm

        if self.use_spiking_gate:
            # SpikingLinear for the gating mechanism
            # It takes analog input and produces analog output + binary spikes
            self.spiking_gate_proj = SpikingLinear(
                nn.Linear(d_model, d_model, bias=False), # Projects to d_model for element-wise gating
                neuron_type=self.neuron_type,
                surrogate_function=self.surrogate_function
            )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Matches the forward signature `(output, reg_loss, final_spikes)`.
        Input `x` is expected to be of shape (B, L, D).
        """
        all_block_spikes_list: List[torch.Tensor] = []
        x_res = x # Residual connection

        # Pre-normalization (operates on continuous signals)
        if self.prenorm:
            x_normalized = self.norm(x) # continuous (B, L, D)
        else:
            x_normalized = x # continuous (B, L, D)

        x_scaled = x_normalized * self.gain # continuous (B, L, D)

        # Transpose for SS4D's expected input shape: (B, L, D) -> (B, D, L)
        x_ss4d_input = rearrange(x_scaled, 'b l d -> b d l') # (B, D, L) and continuous

        # Apply the main SS4D layer. It returns (continuous_output, concatenated_spikes)
        # SS4D handles internal spiking within its MLP layers via SpikingLinear.
        x_ss4d_output_continuous, ss4d_all_spikes = self.layer(x_ss4d_input) # x_ss4d_output_continuous is (B, D, L) continuous

        # Add spikes from SS4D to the list, if any
        if ss4d_all_spikes is not None and ss4d_all_spikes.numel() > 0:
            all_block_spikes_list.append(ss4d_all_spikes)

        # Transpose SS4D output back to standard shape: (B, D, L) -> (B, L, D)
        x_ss4d_output_transposed = rearrange(x_ss4d_output_continuous, 'b d l -> b l d') # (B, L, D) and continuous

        # Apply dropout to the continuous output of SS4D
        x_ss4d_output_dropped = self.dropout_out(x_ss4d_output_transposed) # (B, L, D) continuous

        # Apply Spiking Gated Residual mechanism if enabled
        if self.use_spiking_gate:
            # Input to the gate projection is the normalized input `x_normalized`
            # Reshape for SpikingLinear: (B, L, D) -> (B*L, D)
            gate_input_reshaped = rearrange(x_normalized, 'b l d -> (b l) d')

            # The SpikingLinear layer internally applies a neuron and then a linear transformation.
            # We care about the binary spikes produced by the internal neuron.
            gating_factor_analog_out, s_gate_spikes = self.spiking_gate_proj(gate_input_reshaped)

            # Add spikes from the gate projection to the list, if any
            if s_gate_spikes is not None and s_gate_spikes.numel() > 0:
                all_block_spikes_list.append(s_gate_spikes.flatten()) # Ensure it's flattened for concatenation

            # Use the binary spikes as the gating factor (0 or 1)
            # Reshape back to (B, L, D) for element-wise multiplication
            gating_factor_binary = rearrange(s_gate_spikes.float(), '(b l) d -> b l d', b=x.shape[0], l=x.shape[1])

            # Modulate the SS4D branch output with the binary spiking gate
            branch_output = x_ss4d_output_dropped * gating_factor_binary
        else:
            branch_output = x_ss4d_output_dropped # No gating, just the dropout output

        # Add residual connection
        output = x_res + branch_output # final output is (B, L, D) and continuous

        # Post-normalization (if prenorm is False, this is where the main norm would apply)
        if not self.prenorm:
            output = self.norm(output)

        total_reg_loss = None
        # Concatenate all collected spikes into a single flat tensor. If no spikes, it's an empty tensor.
        final_spikes = torch.cat(all_block_spikes_list) if all_block_spikes_list else torch.empty(0, device=x.device)

        # Calculate firing rate regularization loss
        if self.spike_regularization_weight > 0 and final_spikes.numel() > 0:
            # Mean firing rate for regularization
            firing_rate_loss = final_spikes.float().mean() * self.spike_regularization_weight
            total_reg_loss = firing_rate_loss
        elif final_spikes.numel() == 0 and self.spike_regularization_weight > 0:
            # If no spikes are generated but regularization is active, loss is 0.
            total_reg_loss = torch.tensor(0.0, device=x.device)

        return output, total_reg_loss, final_spikes