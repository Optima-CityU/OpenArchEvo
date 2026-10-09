# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Tuple, List, Optional

# --- Start of required SpikingLinear and dummy imports ---

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
    from src.models.nn.dropout import DropoutNd
    registry = neuron_registry
except (ImportError, ModuleNotFoundError):
    # Dummy placeholders for when src imports are not available
    class DummySurrogate:
        @staticmethod
        def apply(x, *args, **kwargs):
            return (x > 0).float()

    def piecewise_quadratic_surrogate():
        return DummySurrogate

    registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}

    class DropoutNd(nn.Identity):
        pass

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x):
        spikes = self.neuron(x) # Apply neuron to continuous input to generate spikes
        return self.linear(spikes), spikes # Apply linear layer to spikes, return continuous current and spikes

# --- End of required SpikingLinear and dummy imports ---


# =============================================================================
# ===== Start of SpikingSSM Components (S4DKernel, SS4D) ======================
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
        # output_linear's input is spikes, so this is consistent with the general SNN requirement
        # nn.Conv1d is treated as a non-nn.Linear layer, consuming spikes and outputting current.
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
# ===== NEW: SpikingMLP =======================================================
# =============================================================================

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, neuron_type: str='sdn', dropout: float=0.0, **kwargs):
        super().__init__()
        self.surrogate_function = piecewise_quadratic_surrogate()
        self.neuron_type = neuron_type

        # fc1 takes continuous input, spikes internally, outputs continuous current
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), self.neuron_type, self.surrogate_function)
        self.norm1 = nn.LayerNorm(d_ffn) # Normalize continuous current after fc1
        self.dropout_layer = DropoutNd(dropout) if dropout > 0.0 else nn.Identity() # Dropout on continuous current

        # fc2 takes continuous input, spikes internally, outputs continuous current
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), self.neuron_type, self.surrogate_function)
        self.norm2 = nn.LayerNorm(d_model) # Final norm on continuous current after fc2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_spikes_in_mlp = []

        current_1, spikes_1 = self.fc1(x) # SpikingLinear generates spikes internally
        all_spikes_in_mlp.append(spikes_1) # Collect internal spikes

        normed_current_1 = self.norm1(current_1)
        dropped_current_1 = self.dropout_layer(normed_current_1) # Apply dropout on continuous current

        current_2, spikes_2 = self.fc2(dropped_current_1) # Input is continuous
        all_spikes_in_mlp.append(spikes_2) # Collect internal spikes

        normed_current_2 = self.norm2(current_2) # Final normalization for the continuous output
        return normed_current_2, all_spikes_in_mlp

# =============================================================================
# ===== UPDATED: SpikingFLABlock (Evolution) with SS4D Spiking Gating =========
# =============================================================================

class SpikingFLABlock(nn.Module):
    """This class is a single layer/block that uses the Spiking State Space Model (SS4D)
    as its core, augmented with an explicit Spiking Multi-Layer Perceptron (MLP).
    It is designed as a drop-in replacement for other blocks,
    adhering to the standard (pre-norm, residual) structure.

    This evolved version enhances the spiking control by introducing a spiking gating
    mechanism for the SS4D branch, making the information flow modulation
    more neuromorphic. All nn.Linear layers are explicitly wrapped by SpikingLinear,
    maintains a continuous residual stream, and ensures all generated spikes are collected.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, use_spiking_mlp: Optional[bool]=True, d_ffn_mult: Optional[int]=4, neuron_type: Optional[str]='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.neuron_type = neuron_type
        # Ensure use_spiking_mlp is enabled by default as per requirements
        self.use_spiking_mlp = use_spiking_mlp if use_spiking_mlp is not None else True

        # --- SS4D Branch Components ---
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_ss4d = DropoutNd(dropout) if dropout > 0.0 else nn.Identity() # Applied to continuous output

        ss4d_kwargs = kwargs.get('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs.setdefault('neuron', neuron_type) # Ensure SS4D uses the specified neuron type
        self.layer = SS4D(d_model, dropout=dropout, transposed=True, **ss4d_kwargs)

        # NEW: Spiking Gating Mechanism for SS4D branch
        # Takes continuous input, spikes internally, outputs continuous current for gating factor
        self.ss4d_gate_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, piecewise_quadratic_surrogate())
        self.dropout_ss4d_gate_proj = DropoutNd(dropout) if dropout > 0.0 else nn.Identity() # Dropout for gate current


        # --- Spiking MLP Branch Components ---
        if self.use_spiking_mlp:
            # Normalization before the MLP, applied to the output of the SS4D residual stream
            self.norm2 = nn.LayerNorm(d_model)

            # EXISTING: Spiking Gating Mechanism for MLP branch
            # Takes continuous input, spikes internally, outputs continuous current for gating factor
            self.mlp_gate_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, piecewise_quadratic_surrogate())
            self.dropout_mlp_gate_proj = DropoutNd(dropout) if dropout > 0.0 else nn.Identity() # Dropout for gate current

            d_ffn = d_model * (d_ffn_mult if d_ffn_mult is not None else 4)
            self.mlp = SpikingMLP(d_model, d_ffn, neuron_type=neuron_type, dropout=dropout)
            self.dropout_mlp_out = DropoutNd(dropout) if dropout > 0.0 else nn.Identity() # Dropout for MLP output current

        # Final normalization for post-norm architecture (if applicable)
        if not self.prenorm:
            self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Matches the forward signature `(output, reg_loss, spikes)`.
        Input `x` is expected to be of shape (B, L, D).
        The main residual stream is maintained as a continuous signal.
        """
        x_res_ss4d = x  # Residual connection for the SS4D path (continuous)
        all_spikes_list = []

        # === SS4D Branch ===
        x_ss4d_input = self.norm1(x_res_ss4d) if self.prenorm else x_res_ss4d

        # NEW: Apply spiking gate to SS4D input
        ss4d_gate_current, ss4d_gate_spikes = self.ss4d_gate_proj(x_ss4d_input)
        all_spikes_list.append(ss4d_gate_spikes) # Collect spikes from the SS4D gate projection
        dropped_ss4d_gate_current = self.dropout_ss4d_gate_proj(ss4d_gate_current)
        ss4d_gate_factor = torch.sigmoid(dropped_ss4d_gate_current) # Continuous gating factor
        x_ss4d_input = x_ss4d_input * ss4d_gate_factor # Apply learned gating factor

        # Transpose for Conv1d operations inside SS4D: (B, L, D) -> (B, D, L)
        x_ss4d_input = rearrange(x_ss4d_input, 'b l d -> b d l')

        # Apply the main SS4D layer
        ss4d_output_continuous, spikes_ss4d = self.layer(x_ss4d_input)
        all_spikes_list.append(spikes_ss4d) # Collect spikes from SS4D (binary)

        # Transpose back to standard shape: (B, D, L) -> (B, L, D)
        ss4d_output_continuous = rearrange(ss4d_output_continuous, 'b d l -> b l d')

        # Apply dropout on the continuous output of SS4D
        ss4d_output_continuous = self.dropout_ss4d(ss4d_output_continuous)

        # Add SS4D residual connection: current input + SS4D output
        x_after_ss4d_residual = x_res_ss4d + ss4d_output_continuous # Continuous

        # === Spiking MLP Branch with Gating ===
        output = x_after_ss4d_residual # Initialize output with the SS4D residual stream

        if self.use_spiking_mlp:
            x_res_mlp = x_after_ss4d_residual # Residual connection for the MLP path (continuous)

            # --- MLP Gating Mechanism ---
            # The gate takes the continuous signal, spikes it, applies a linear transform, outputs current
            gate_current, gate_spikes = self.mlp_gate_proj(x_res_mlp)
            all_spikes_list.append(gate_spikes) # Collect spikes from the gate projection

            # Apply dropout to the gate current
            dropped_gate_current = self.dropout_mlp_gate_proj(gate_current)

            # Sigmoid activation to produce a gating factor between 0 and 1
            mlp_gate_factor = torch.sigmoid(dropped_gate_current) # Continuous gating factor

            # --- MLP Core ---
            # Normalize input to the MLP branch (continuous)
            x_mlp_input_normed = self.norm2(x_res_mlp) if self.prenorm else x_res_mlp

            # Apply Spiking MLP. MLP expects continuous input and returns continuous output and internal spikes.
            mlp_output_current, spikes_mlp_internal = self.mlp(x_mlp_input_normed)
            all_spikes_list.extend(spikes_mlp_internal) # Collect internal MLP spikes

            # Apply dropout to the MLP output current
            mlp_output_current = self.dropout_mlp_out(mlp_output_current)

            # Apply the learned gate to the MLP output
            gated_mlp_output = mlp_output_current * mlp_gate_factor # Element-wise gating

            # Add the gated MLP output (continuous) as residual to the continuous stream
            output = x_res_mlp + gated_mlp_output

        # === Finalization ===
        # Post-normalization if prenorm is false
        if not self.prenorm:
            output = self.norm_final(output)

        reg_loss = None # No explicit regularization loss implemented in this block

        # Concatenate all collected spikes into a single flat tensor as required
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None])

        return output, reg_loss, final_spikes