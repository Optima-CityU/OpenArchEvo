import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Tuple, List, Optional, Union

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

# NOTE: This SpikingLinear class is provided by the problem statement and MUST NOT be modified.

class SpikingLinear(nn.Module):
    """A sequence module containing LIF neurons and a linear layer."""

    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transposed = x.transpose(-1, -2) # (B, L, D) -> (B, D, L)
        spikes_transposed = self.neuron(x_transposed) # Apply neuron to continuous current
        spikes = spikes_transposed.transpose(-1, -2) # (B, D, L) -> (B, L, D) - binary spikes
        output = self.linear(spikes) # Linear layer receives binary spikes, outputs continuous current
        return output, spikes

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
        if not self.transposed: u = u.transpose(-1, -2) # Ensure (B, C*H, L) if transposed is False

        L = u.size(-1)
        k, _ = self.kernel(L=L)
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L) # u_f operates on continuous current
        y = torch.einsum("bhl,chl->bchl", u_f, k_f)
        y = torch.fft.irfft(y, n=2 * L)[..., :L]
        y = y + torch.einsum("bhl,ch->bchl", u, self.D)
        y = rearrange(y, "b c h l -> b (c h) l") # y is now (B, C*H, L)
        if self.learnable_vth: y = y / torch.exp(self.ln_vth)

        spikes = self.neuron(y) # Spikes are generated here, input 'y' is (B, C*H, L) and is continuous
        y_dropout_spikes = self.dropout(spikes) # Dropout on spikes
        y = self.output_linear(y_dropout_spikes) # output_linear takes spikes as input, produces continuous current

        if not self.transposed: y = y.transpose(-1, -2) # Transpose back to (B, L, D) if needed
        return y, spikes # y is continuous, spikes are binary

# =============================================================================
# ===== NEW: SpikingMLP (Re-introduced for Modularity) ========================
# =============================================================================

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, neuron_type: str='sdn', dropout: float=0.0, **kwargs):
        super().__init__()
        self.surrogate_function = piecewise_quadratic_surrogate()
        self.neuron_type = neuron_type

        # fc1 takes continuous input, outputs (continuous current, spikes)
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), self.neuron_type, self.surrogate_function)
        self.norm1 = nn.LayerNorm(d_ffn)
        self.dropout_layer = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # fc2 takes continuous input (from norm1, dropout), outputs (continuous current, spikes)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), self.neuron_type, self.surrogate_function)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_spikes_in_mlp = []

        # x is (B, L, D_model), continuous
        current_1, spikes_1 = self.fc1(x) # current_1 is (B, L, D_ffn), spikes_1 is (B, L, D_ffn)
        all_spikes_in_mlp.append(spikes_1)

        normed_current_1 = self.norm1(current_1)
        dropped_current_1 = self.dropout_layer(normed_current_1)

        # dropped_current_1 is (B, L, D_ffn), continuous
        current_2, spikes_2 = self.fc2(dropped_current_1) # current_2 is (B, L, D_model), spikes_2 is (B, L, D_model)
        all_spikes_in_mlp.append(spikes_2)

        return current_2, all_spikes_in_mlp

# =============================================================================
# ===== NEW: SpikingFLABlock (Evolution) with Learnable Gating Scales =====
# =============================================================================

class SpikingFLABlock(nn.Module):
    """This class is a single layer/block that uses the Spiking State Space Model (SS4D)
    as its core, augmented with an explicitly spike-driven Multi-Layer Perceptron (MLP).
    It is designed as a drop-in replacement for other blocks,
    adhering to the standard (pre-norm, residual) structure.

    This evolved version introduces **learnable multi-scale weighting** for all its
    spike-convolutional gating mechanisms. Instead of simply summing the contributions
    from different temporal receptive fields (kernel sizes) when forming a gating signal,
    the model can now learn to assign specific weights to each scale.
    This enhances the neuromorphic control by making information flow modulation directly
    sensitive to local spike patterns and history across different temporal receptive fields
    in a more adaptive and optimized manner.

    All `nn.Linear` layers are explicitly wrapped by `SpikingLinear`, maintaining a
    continuous residual stream, and ensuring all generated spikes are collected.
    It also utilizes adaptive spike activity regularization towards a target firing rate.

    **Innovation in this version**: Learnable weights are applied to the outputs of parallel
    multi-scale depthwise convolutions on binary spikes before combination. This allows
    dynamic prioritization of different temporal scales in the gating signals.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, use_spiking_mlp: Optional[bool]=True, d_ffn_mult: Optional[int]=4, neuron_type: Optional[str]='sdn', spike_reg_weight: float=0.01, target_spike_rate: float=0.1, gate_kernel_sizes: Union[int, List[int], None]=None, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.neuron_type = neuron_type
        self.spike_reg_weight = spike_reg_weight
        self.target_spike_rate = target_spike_rate
        self.surrogate_function = piecewise_quadratic_surrogate()

        # Handle multi-scale gate_kernel_sizes
        if gate_kernel_sizes is None:
            self.gate_kernel_sizes = [3, 5, 7] # Default to more diverse multi-scale
        elif isinstance(gate_kernel_sizes, int):
            self.gate_kernel_sizes = [gate_kernel_sizes]
        else:
            self.gate_kernel_sizes = gate_kernel_sizes
        if not self.gate_kernel_sizes: # Fallback if an empty list was passed
            self.gate_kernel_sizes = [3]

        self.use_spiking_mlp = use_spiking_mlp if use_spiking_mlp is not None else True

        # --- SS4D Branch Components ---
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_ss4d = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        ss4d_kwargs = kwargs.get('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs.setdefault('neuron', neuron_type)
        self.layer = SS4D(d_model, dropout=dropout, transposed=True, **ss4d_kwargs)

        # Multi-Scale Spike-Convolutional Gating Mechanism for SS4D input
        self.ss4d_gate_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, self.surrogate_function)
        self.ss4d_gate_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=0, groups=d_model, bias=False)
            for k in self.gate_kernel_sizes
        ])
        # NEW: Learnable weights for combining multi-scale SS4D gate conv outputs
        self.ss4d_gate_scale_weights = nn.Parameter(torch.ones(len(self.gate_kernel_sizes)))
        self.ss4d_gate_norm = nn.LayerNorm(d_model)
        self.ss4d_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

        # --- Spiking MLP Branch Components ---
        if self.use_spiking_mlp:
            self.norm2 = nn.LayerNorm(d_model)

            # Multi-Scale Spike-Convolutional Gating Mechanism for MLP branch (base gate)
            self.mlp_gate_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, self.surrogate_function)
            self.mlp_gate_convs = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=0, groups=d_model, bias=False)
                for k in self.gate_kernel_sizes
            ])
            # NEW: Learnable weights for combining multi-scale MLP base gate conv outputs
            self.mlp_base_gate_scale_weights = nn.Parameter(torch.ones(len(self.gate_kernel_sizes)))
            self.mlp_base_gate_norm = nn.LayerNorm(d_model)
            self.mlp_base_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

            # NEW: Multi-Scale Spike-Convolutional Cross-Modulation for MLP gate from SS4D spikes
            self.ss4d_spikes_to_mlp_gate_convs = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=0, groups=d_model, bias=False)
                for k in self.gate_kernel_sizes
            ])
            # NEW: Learnable weights for combining multi-scale SS4D-to-MLP cross-mod conv outputs
            self.ss4d_to_mlp_gate_scale_weights = nn.Parameter(torch.ones(len(self.gate_kernel_sizes)))
            self.ss4d_to_mlp_gate_norm = nn.LayerNorm(d_model)
            self.ss4d_to_mlp_gate_scale = nn.Parameter(torch.ones(1, 1, d_model))

            d_ffn = d_model * (d_ffn_mult if d_ffn_mult is not None else 4)
            self.mlp = SpikingMLP(d_model, d_ffn, neuron_type=neuron_type, dropout=dropout)
            self.dropout_mlp_out = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        if not self.prenorm:
            self.norm_final = nn.LayerNorm(d_model)

    def _apply_multi_scale_spike_conv_gating(self, spikes: torch.Tensor, conv_layers: nn.ModuleList, scale_weights: nn.Parameter, gate_norm: nn.LayerNorm, gate_scale_factor: nn.Parameter) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Helper function to apply multi-scale spike-convolutional gating."""
        spikes_transposed = rearrange(spikes, 'b l d -> b d l')

        currents_per_scale = []
        for i, conv_layer in enumerate(conv_layers):
            k_size = self.gate_kernel_sizes[i]
            padding_val = k_size - 1
            # Causal padding on the *sequence dimension* (last dimension for Conv1d)
            padded_input = F.pad(spikes_transposed, (padding_val, 0))
            current = conv_layer(padded_input) # Output is (B, D, L)
            currents_per_scale.append(current)

        # NEW: Apply learnable weights before summing
        weighted_currents = [c * scale_weights[i] for i, c in enumerate(currents_per_scale)]

        # Combine currents from different scales (weighted sum)
        combined_gating_current = torch.sum(torch.stack(weighted_currents), dim=0)

        # Transpose back to (B, L, D) for LayerNorm and element-wise multiplication
        combined_gating_current = rearrange(combined_gating_current, 'b d l -> b l d')

        # Normalize the combined gating current before activation
        normalized_gating_current = gate_norm(combined_gating_current)
        gating_factor = torch.sigmoid(gate_scale_factor * normalized_gating_current)

        return gating_factor

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Matches the forward signature `(output, reg_loss, spikes)`.
        Input `x` is expected to be of shape (B, L, D).
        The main residual stream is maintained as a continuous signal.
        """
        x_res_ss4d = x
        all_spikes_list = []
        total_reg_loss = torch.tensor(0.0, device=x.device)

        # === SS4D Branch ===
        x_ss4d_input_prenorm = self.norm1(x_res_ss4d) if self.prenorm else x_res_ss4d

        # Multi-Scale Spike-Convolutional Gating for SS4D input
        _, ss4d_gate_spikes = self.ss4d_gate_proj(x_ss4d_input_prenorm) # ss4d_gate_spikes is (B, L, D)
        all_spikes_list.append(ss4d_gate_spikes)

        ss4d_gating_factor = self._apply_multi_scale_spike_conv_gating(
            ss4d_gate_spikes,
            self.ss4d_gate_convs,
            self.ss4d_gate_scale_weights,
            self.ss4d_gate_norm,
            self.ss4d_gate_scale
        )

        x_ss4d_input_gated = x_ss4d_input_prenorm * ss4d_gating_factor

        # SS4D expects input in (B, D, L) format
        x_ss4d_input_processed = rearrange(x_ss4d_input_gated, 'b l d -> b d l')

        ss4d_output_continuous, spikes_ss4d = self.layer(x_ss4d_input_processed) # spikes_ss4d is (B, D, L)
        all_spikes_list.append(rearrange(spikes_ss4d, 'b d l -> b l d')) # Store as (B, L, D) for consistency

        # Transpose back to (B, L, D) for residual connection
        ss4d_output_continuous = rearrange(ss4d_output_continuous, 'b d l -> b l d')

        ss4d_output_continuous = self.dropout_ss4d(ss4d_output_continuous)

        x_after_ss4d_residual = x_res_ss4d + ss4d_output_continuous

        # === Spiking MLP Branch with Multi-Scale Gating and Cross-Modulation ===
        output = x_after_ss4d_residual

        if self.use_spiking_mlp:
            x_res_mlp = x_after_ss4d_residual

            # --- MLP Base Gating Mechanism (Multi-Scale Spike Convolutional) ---
            _, mlp_base_gate_spikes = self.mlp_gate_proj(x_res_mlp) # (B, L, D)
            all_spikes_list.append(mlp_base_gate_spikes)

            mlp_base_gating_current_normalized = self._apply_multi_scale_spike_conv_gating(
                mlp_base_gate_spikes,
                self.mlp_gate_convs,
                self.mlp_base_gate_scale_weights,
                self.mlp_base_gate_norm,
                torch.ones_like(self.mlp_base_gate_scale) # Placeholder, will combine with cross-mod later
            ) # Returns factor, but here we want the normalized current before sigmoid
            # The `_apply_multi_scale_spike_conv_gating` returns the final factor, 
            # but for MLP, we need the *normalized current* to combine with cross-mod, 
            # then apply final scale and sigmoid. So, let's manually inline this part or modify helper.

            # Re-inlining the MLP base gate processing to get the normalized current directly
            mlp_base_gate_spikes_transposed = rearrange(mlp_base_gate_spikes, 'b l d -> b d l')
            currents_per_scale_mlp_base = []
            for i, conv_layer in enumerate(self.mlp_gate_convs):
                k_size = self.gate_kernel_sizes[i]
                padding_val = k_size - 1
                padded_input = F.pad(mlp_base_gate_spikes_transposed, (padding_val, 0))
                currents_per_scale_mlp_base.append(conv_layer(padded_input))

            weighted_currents_mlp_base = [c * self.mlp_base_gate_scale_weights[i] for i, c in enumerate(currents_per_scale_mlp_base)]
            mlp_base_gating_current_combined = torch.sum(torch.stack(weighted_currents_mlp_base), dim=0)
            mlp_base_gating_current_combined = rearrange(mlp_base_gating_current_combined, 'b d l -> b l d')
            mlp_base_gating_current_normalized = self.mlp_base_gate_norm(mlp_base_gating_current_combined)

            # --- Multi-Scale Spike-Convolutional Cross-Modulation for MLP gate from SS4D spikes ---
            cross_mod_spikes_for_mlp_gate = spikes_ss4d # (B, D, L) from self.layer

            currents_per_scale_ss4d_to_mlp = []
            for i, conv_layer in enumerate(self.ss4d_spikes_to_mlp_gate_convs):
                k_size = self.gate_kernel_sizes[i]
                padding_val = k_size - 1
                padded_input = F.pad(cross_mod_spikes_for_mlp_gate, (padding_val, 0))
                currents_per_scale_ss4d_to_mlp.append(conv_layer(padded_input))

            weighted_currents_ss4d_to_mlp = [c * self.ss4d_to_mlp_gate_scale_weights[i] for i, c in enumerate(currents_per_scale_ss4d_to_mlp)]
            ss4d_to_mlp_mod_current_combined = torch.sum(torch.stack(weighted_currents_ss4d_to_mlp), dim=0)
            ss4d_to_mlp_mod_current_combined = rearrange(ss4d_to_mlp_mod_current_combined, 'b d l -> b l d')
            ss4d_to_mlp_mod_current_normalized = self.ss4d_to_mlp_gate_norm(ss4d_to_mlp_mod_current_combined)
            ss4d_to_mlp_mod_current_scaled = self.ss4d_to_mlp_gate_scale * ss4d_to_mlp_mod_current_normalized

            # Combine base MLP gate current with cross-modulation current before final sigmoid
            combined_mlp_gate_input = mlp_base_gating_current_normalized + ss4d_to_mlp_mod_current_scaled
            mlp_gating_factor = torch.sigmoid(self.mlp_base_gate_scale * combined_mlp_gate_input)

            # --- MLP Core ---
            x_mlp_input_normed = self.norm2(x_res_mlp) if self.prenorm else x_res_mlp

            mlp_output_current, spikes_mlp_internal = self.mlp(x_mlp_input_normed) # mlp_output_current is (B, L, D), spikes_mlp_internal is list of (B, L, D) tensors
            all_spikes_list.extend(spikes_mlp_internal)

            mlp_output_current = self.dropout_mlp_out(mlp_output_current)

            gated_mlp_output = mlp_output_current * mlp_gating_factor

            output = x_res_mlp + gated_mlp_output

        # === Finalization ===
        if not self.prenorm:
            output = self.norm_final(output)

        final_spikes = None
        if all_spikes_list:
            # Filter out any potential None or empty tensors before concatenating
            valid_spikes = [s.flatten() for s in all_spikes_list if s is not None and s.numel() > 0]
            if valid_spikes:
                final_spikes = torch.cat(valid_spikes)

                # Re-introduce V1's target spike rate regularization
                if self.spike_reg_weight > 0 and final_spikes.numel() > 0 and self.target_spike_rate is not None:
                    spike_activity_mean = torch.mean(final_spikes.float()) # Ensure float for mean calculation
                    total_reg_loss = self.spike_reg_weight * torch.abs(spike_activity_mean - self.target_spike_rate)

        return output, total_reg_loss, final_spikes