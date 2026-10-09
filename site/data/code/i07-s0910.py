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
    print("Warning: Using placeholder SNN modules.")
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
        spikes = self.neuron(x.float()) # Apply neuron to current input to generate spikes (binary)
        return self.linear(spikes), spikes # Apply linear layer to spikes, return continuous current and spikes

# --- End of required SpikingLinear and dummy imports ---


# =============================================================================
# ===== Start of SpikingSSM Components (S4DKernel, SS4D) - From Version 2 =====
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
        # The neuron here means SS4D internally converts its current-like output to spikes
        self.neuron = registry[neuron](surrogate_function=piecewise_quadratic_surrogate())
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # self.output_linear consists of nn.Conv1d and nn.GLU, which are not nn.Linear.
        # Thus, they are not wrapped by SpikingLinear, and they explicitly take binary spikes as input.
        self.output_linear = nn.Sequential(nn.Conv1d(self.h, 2 * self.h, kernel_size=1, bias=False), nn.GLU(dim=-2))

    def forward(self, u, **kwargs):
        if not self.transposed: u = u.transpose(-1, -2) # ensure u is (B, D, L)

        L = u.size(-1) # Sequence length

        k, _ = self.kernel(L=L) # k is (channels*H, L)

        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))

        # Handle different channel configurations for k_f and u_f
        # If channels=1, k is (H, L). k_f is (H, L').
        # If u is (B, H, L), u_f is (B, H, L').
        # The einsum should be `b h l, h l -> b h l`.
        # If channels > 1, k is (C, H, L). k_f is (C, H, L').
        # Then u needs to be (B, C, H, L) or handled differently. Current usage is (B, D, L) for u.
        # Assuming channels=1 as per current SS4D init default.
        if self.D.shape[0] == 1 and k.dim() == 3 and k.shape[0] == 1: # if channels was 1 (1, H, L)
            k = k.squeeze(0) # (H, L)

        u_f = torch.fft.rfft(u, n=2 * L) # (B, H, L')
        k_f = torch.fft.rfft(k, n=2 * L) # (H, L')

        # Perform convolution in frequency domain
        y = torch.einsum("bhl,hl->bhl", u_f, k_f) # (B, H, L')
        y = torch.fft.irfft(y, n=2 * L)[..., :L] # (B, H, L) - continuous current

        # Direct feedthrough D term
        y = y + torch.einsum("bhl,h->bhl", u, self.D.squeeze(0)) # self.D is (C=1, H) -> (H)

        if self.learnable_vth: y = y / torch.exp(self.ln_vth) # Apply learnable threshold if enabled

        spikes = self.neuron(y) # Spikes are generated here from continuous 'y', shape (B, H, L) - binary
        y_dropout_spikes_float = self.dropout(spikes.float()) # Dropout on binary spikes (as float)

        # output_linear expects (B, D, L) as input for Conv1d, which matches y_dropout_spikes_float
        y_final_output_current = self.output_linear(y_dropout_spikes_float) # outputs continuous current

        if not self.transposed: y_final_output_current = y_final_output_current.transpose(-1, -2) # (B, L, D)
        return y_final_output_current, spikes # y_final_output_current is continuous, spikes are binary

# =============================================================================
# ===== UPDATED: SpikingMLP (With more explicit SNN flow) =====================
# =============================================================================

class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, neuron_type: str='sdn', dropout: float=0.0, **kwargs):
        super().__init__()
        self.surrogate_function = piecewise_quadratic_surrogate()
        self.neuron_type = neuron_type

        # fc1 takes float input (analog current or float spikes), outputs continuous current and its internal spikes
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn, bias=False), self.neuron_type, self.surrogate_function)
        self.norm1 = nn.LayerNorm(d_ffn) # Normalize continuous current

        # Explicit spike activation after norm1 to convert analog current to spikes
        self.intermediate_spike_activation = registry[neuron_type](surrogate_function=self.surrogate_function)
        self.dropout_layer = DropoutNd(dropout) if dropout > 0.0 else nn.Identity() # Apply dropout to intermediate spikes (float)

        # fc2 takes float input (should be float spikes), outputs continuous current and its internal spikes
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model, bias=False), self.neuron_type, self.surrogate_function)
        self.norm2 = nn.LayerNorm(d_model) # Final norm on continuous current

    def _apply_spike_activation_helper(self, x_analog: torch.Tensor, spike_activation_module) -> torch.Tensor:
        """Helper for MLP to apply spiking activation, ensuring correct dimension handling.
        Input x_analog should be (B, L, D). Output will be (B, L, D) binary spikes.
        """
        x_transposed_for_spike = rearrange(x_analog, 'b l d -> b d l')
        spiked_x_transposed = spike_activation_module(x_transposed_for_spike)
        spiked_x = rearrange(spiked_x_transposed, 'b d l -> b l d') # (B, L, D) - Binary spikes
        return spiked_x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_spikes_in_mlp = []

        # fc1 takes x, outputs current_1_analog (continuous) and spikes_1 (binary)
        current_1_analog, spikes_1 = self.fc1(x)
        all_spikes_in_mlp.append(spikes_1)

        normed_current_1_analog = self.norm1(current_1_analog) # continuous

        # Explicitly spike the normalized analog current before feeding to dropout/fc2
        intermediate_spikes_bool = self._apply_spike_activation_helper(normed_current_1_analog, self.intermediate_spike_activation)
        all_spikes_in_mlp.append(intermediate_spikes_bool)

        # Dropout on spikes (convert to float for dropout)
        dropped_spikes_float = self.dropout_layer(intermediate_spikes_bool.float())

        # fc2 takes these dropped spikes (float) as input
        current_2_analog, spikes_2 = self.fc2(dropped_spikes_float) # current_2_analog is continuous, spikes_2 are binary
        all_spikes_in_mlp.append(spikes_2)

        normed_current_2_analog = self.norm2(current_2_analog) # continuous
        return normed_current_2_analog, all_spikes_in_mlp

# =============================================================================
# ===== NEW HELPER MODULES (adapted from V1 ideas) ============================
# =============================================================================

class CausalConv1d(nn.Module):
    """1D Causal Convolution for temporal feature aggregation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b l c -> b c l') # (B, C, L) for Conv1d
        x = self.conv(x)
        if self.padding > 0:
            x = x[..., :-self.padding] # Remove padding to ensure causality
        x = rearrange(x, 'b c l -> b l c') # (B, L, C)
        return x

class SpikingAdaptiveScaler(nn.Module):
    """A dedicated spiking MLP that takes input statistics (mean/std) and
    produces a dynamic scaling factor for the analog input to a subsequent
    spiking neuron. This modulates the effective firing rate, with an adaptive
    internal spike generation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, neuron_type: str, surrogate_function, **kwargs):
        super().__init__()
        # proj_in takes analog input, outputs continuous current and internal spikes
        self.proj_in = SpikingLinear(nn.Linear(input_dim, hidden_dim, bias=False), neuron_type, surrogate_function)
        self.norm = nn.LayerNorm(hidden_dim)

        # threshold_predictor takes analog current, outputs continuous current (used for modulation factor) and internal spikes
        self.threshold_predictor = SpikingLinear(nn.Linear(hidden_dim, 1, bias=False), neuron_type, surrogate_function)

        # intermediate_spike_activation takes analog current, outputs binary spikes
        self.intermediate_spike_activation = registry[neuron_type](surrogate_function=surrogate_function)

        # proj_out takes float spikes, outputs continuous current and internal spikes
        self.proj_out = SpikingLinear(nn.Linear(hidden_dim, output_dim, bias=False), neuron_type, surrogate_function)

    def _apply_spike_activation_helper(self, x_analog: torch.Tensor, spike_activation_module) -> torch.Tensor:
        """Helper to apply spiking activation, ensuring correct dimension handling.
        Input x_analog should be (B, L, D). Output will be (B, L, D) binary spikes.
        """
        x_transposed_for_spike = rearrange(x_analog, 'b l d -> b d l')
        spiked_x_transposed = spike_activation_module(x_transposed_for_spike)
        spiked_x = rearrange(spiked_x_transposed, 'b d l -> b l d') # (B, L, D) - Binary spikes
        return spiked_x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_list = []

        x_proj_in_analog, spikes1 = self.proj_in(x) # spikes1 are binary
        spikes_list.append(spikes1)

        x_norm_analog = self.norm(x_proj_in_analog) # continuous

        # Predict adaptive threshold factor from the normalized continuous current
        threshold_factor_current_analog, threshold_spikes = self.threshold_predictor(x_norm_analog) # threshold_spikes are binary
        spikes_list.append(threshold_spikes)

        # Use softplus to ensure factor is positive and provides a minimum base. Reshape for broadcastable multiplication (B, L, 1)
        adaptive_modulation_factor_analog = (F.softplus(threshold_factor_current_analog) + 0.5) # continuous

        # Intermediate Spiking: Apply the adaptive modulation factor to the continuous input current of the intermediate neuron
        gated_x_intermediate_input_analog = x_norm_analog * adaptive_modulation_factor_analog # continuous

        # Explicitly spike the modulated analog current
        intermediate_spikes_bool = self._apply_spike_activation_helper(gated_x_intermediate_input_analog, self.intermediate_spike_activation)
        spikes_list.append(intermediate_spikes_bool)

        # proj_out takes float spikes as input
        output_analog, spikes2 = self.proj_out(intermediate_spikes_bool.float()) # spikes2 are binary
        spikes_list.append(spikes2)

        return output_analog, spikes_list # Return continuous output and all internal spikes

class SpikingAdaptiveGate(nn.Module):
    """A dedicated spiking MLP for generating modulation factors for residual gates.
    It processes combined mean activity, sparsity, and their temporal features.
    It can also take an extra input (e.g., from a previous gate's output).
    It includes an adaptive internal spike generation mechanism.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, neuron_type: str, surrogate_function, extra_input_dim: int=0, **kwargs):
        super().__init__()
        # proj_in takes analog input, outputs continuous current and internal spikes
        self.proj_in = SpikingLinear(nn.Linear(input_dim + extra_input_dim, hidden_dim, bias=False), neuron_type, surrogate_function)
        self.norm = nn.LayerNorm(hidden_dim)

        # threshold_predictor takes analog current, outputs continuous current (used for modulation factor) and internal spikes
        self.threshold_predictor = SpikingLinear(nn.Linear(hidden_dim, 1, bias=False), neuron_type, surrogate_function)

        # intermediate_spike_activation takes analog current, outputs binary spikes
        self.intermediate_spike_activation = registry[neuron_type](surrogate_function=surrogate_function)

        # proj_out takes float spikes, outputs continuous current and internal spikes
        self.proj_out = SpikingLinear(nn.Linear(hidden_dim, output_dim, bias=False), neuron_type, surrogate_function)

    def _apply_spike_activation_helper(self, x_analog: torch.Tensor, spike_activation_module) -> torch.Tensor:
        """Helper to apply spiking activation, ensuring correct dimension handling.
        Input x_analog should be (B, L, D). Output will be (B, L, D) binary spikes.
        """
        x_transposed_for_spike = rearrange(x_analog, 'b l d -> b d l')
        spiked_x_transposed = spike_activation_module(x_transposed_for_spike)
        spiked_x = rearrange(spiked_x_transposed, 'b d l -> b l d') # (B, L, D) - Binary spikes
        return spiked_x

    def forward(self, x: torch.Tensor, extra_input: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_list = []

        if extra_input is not None:
            x = torch.cat([x, extra_input], dim=-1) # x here is analog (features), extra_input is analog current

        x_proj_in_analog, spikes1 = self.proj_in(x) # spikes1 are binary
        spikes_list.append(spikes1)

        x_norm_analog = self.norm(x_proj_in_analog) # continuous

        # Predict adaptive threshold factor from the normalized continuous current
        threshold_factor_current_analog, threshold_spikes = self.threshold_predictor(x_norm_analog) # threshold_spikes are binary
        spikes_list.append(threshold_spikes)
        adaptive_modulation_factor_analog = (F.softplus(threshold_factor_current_analog) + 0.5) # continuous

        # Intermediate Spiking: Apply the adaptive modulation factor to the continuous input current of the intermediate neuron
        gated_x_intermediate_input_analog = x_norm_analog * adaptive_modulation_factor_analog # continuous

        # Explicitly spike the modulated analog current
        intermediate_spikes_bool = self._apply_spike_activation_helper(gated_x_intermediate_input_analog, self.intermediate_spike_activation)
        spikes_list.append(intermediate_spikes_bool)

        # proj_out takes float spikes as input
        output_analog, spikes2 = self.proj_out(intermediate_spikes_bool.float()) # spikes2 are binary
        spikes_list.append(spikes2)

        return output_analog, spikes_list # Return continuous output and all internal spikes

# =============================================================================
# ===== Evolution: SpikingFLABlock ============================================
# =============================================================================

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, use_spiking_mlp: Optional[bool]=True, d_ffn_mult: Optional[int]=4, neuron_type: Optional[str]='sdn', **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.neuron_type = neuron_type
        self.surrogate = piecewise_quadratic_surrogate()
        self.use_spiking_mlp = use_spiking_mlp if use_spiking_mlp is not None else True

        # --- Spike Activations for Feature Generation & Pre-MLP ---
        # Used to convert analog currents to spikes for feature extraction or as input to SpikingMLP
        self.ss4d_feature_spike_activation = registry[neuron_type](surrogate_function=self.surrogate)
        self.mlp_feature_spike_activation = registry[neuron_type](surrogate_function=self.surrogate)
        self.mlp_pre_spike_activation = registry[neuron_type](surrogate_function=self.surrogate) # For input to SpikingMLP

        # Causal Conv1d for extracting multi-scale temporal features from input spike activity
        self.causal_conv_mean_k1 = CausalConv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.causal_conv_mean_k3 = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3)
        self.causal_conv_sparsity_k1 = CausalConv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.causal_conv_sparsity_k3 = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3)

        # --- SS4D Branch Components ---
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_ss4d_output = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        ss4d_kwargs = kwargs.get('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs.setdefault('neuron', neuron_type)
        self.layer = SS4D(d_model, dropout=dropout, transposed=True, **ss4d_kwargs)

        # SpikingAdaptiveScaler for SS4D input (modulates continuous current before spiking for features)
        scaler_input_dim = 2 # mean, std of the continuous input
        scaler_hidden_dim = d_model // 8
        self.ss4d_input_scaler = SpikingAdaptiveScaler(input_dim=scaler_input_dim, hidden_dim=scaler_hidden_dim, output_dim=d_model, neuron_type=neuron_type, surrogate_function=self.surrogate)

        # SpikingAdaptiveGate for SS4D branch (modulates continuous current)
        gate_input_dim = 6 # (mean, conv_mean_k1, conv_mean_k3, sparsity, conv_sparsity_k1, conv_sparsity_k3)
        gate_hidden_dim = d_model // 4
        self.ss4d_gate_mlp = SpikingAdaptiveGate(input_dim=gate_input_dim, hidden_dim=gate_hidden_dim, output_dim=d_model, neuron_type=neuron_type, surrogate_function=self.surrogate)
        self.dropout_ss4d_gate_mlp = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        self.ss4d_gate_bias = nn.Parameter(torch.zeros(1, 1, d_model)) # Homeostatic bias for the gate output


        # --- Spiking MLP Branch Components ---
        if self.use_spiking_mlp:
            self.norm2 = nn.LayerNorm(d_model)
            d_ffn = d_model * (d_ffn_mult if d_ffn_mult is not None else 4)
            self.mlp = SpikingMLP(d_model, d_ffn, neuron_type=neuron_type, dropout=dropout)
            self.dropout_mlp_out = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

            # SpikingAdaptiveScaler for MLP input (modulates continuous current before spiking for features)
            self.mlp_input_scaler = SpikingAdaptiveScaler(input_dim=scaler_input_dim, hidden_dim=scaler_hidden_dim, output_dim=d_model, neuron_type=neuron_type, surrogate_function=self.surrogate)

            # SpikingAdaptiveGate for MLP branch (modulates continuous current)
            mlp_gate_extra_input_dim = d_model # From ss4d_gate_analog_output_raw
            self.mlp_gate_mlp = SpikingAdaptiveGate(input_dim=gate_input_dim, hidden_dim=gate_hidden_dim, output_dim=d_model, neuron_type=neuron_type, surrogate_function=self.surrogate, extra_input_dim=mlp_gate_extra_input_dim)
            self.dropout_mlp_gate_mlp = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
            self.mlp_gate_bias = nn.Parameter(torch.zeros(1, 1, d_model)) # Homeostatic bias for the gate output

        # Final normalization for post-norm architecture (if applicable)
        if not self.prenorm:
            self.norm_final = nn.LayerNorm(d_model)

    def _apply_spike_activation_helper(self, x_analog: torch.Tensor, spike_activation_module) -> torch.Tensor:
        """Helper to apply spiking activation, ensuring correct dimension handling.
        Input x_analog should be (B, L, D). Output will be (B, L, D) binary spikes.
        """
        x_transposed_for_spike = rearrange(x_analog, 'b l d -> b d l')
        spiked_x_transposed = spike_activation_module(x_transposed_for_spike)
        spiked_x = rearrange(spiked_x_transposed, 'b d l -> b l d') # (B, L, D) - Binary spikes
        return spiked_x

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        x_res_ss4d = x # (B, L, D) - initial input is continuous
        all_spikes_list = []
        reg_loss = None # Unused in this version, kept for signature compliance
        ss4d_gate_analog_output_raw = None # To store raw gate output for MLP gate

        # === SS4D Branch ===
        x_norm1_analog = self.norm1(x_res_ss4d) if self.prenorm else x_res_ss4d # (B, L, D) - continuous

        # --- Dynamic Input Scaling for SS4D ---
        mean_x1 = x_norm1_analog.mean(dim=-1, keepdim=True) # (B, L, 1)
        std_x1 = x_norm1_analog.std(dim=-1, keepdim=True).clamp(min=1e-5) # (B, L, 1)
        stats1_analog = torch.cat([mean_x1, std_x1], dim=-1) # (B, L, 2) - analog features

        # ss4d_input_scaler takes analog stats, outputs analog scale_raw_analog and its internal spikes
        scale1_raw_analog, scale1_spikes = self.ss4d_input_scaler(stats1_analog) # scale1_spikes are binary
        all_spikes_list.extend(scale1_spikes)
        scale1_analog = F.softplus(scale1_raw_analog) + 0.5 # Ensure scaling factor is always >= 0.5 (B, L, D) - analog
        x_ss4d_scaled_analog = x_norm1_analog * scale1_analog # (B, L, D) - analog current

        # --- Feature Generation for SS4D Gate (from dedicated input spikes) ---
        # Convert the scaled continuous input to explicit binary spikes for gate feature extraction
        spiked_x_for_ss4d_features_bool = self._apply_spike_activation_helper(x_ss4d_scaled_analog, self.ss4d_feature_spike_activation) # (B, L, D) - binary spikes
        all_spikes_list.append(spiked_x_for_ss4d_features_bool)

        # Calculate mean activity and sparsity from these feature spikes (converted to float for calculations)
        ss4d_mean_activity_analog = spiked_x_for_ss4d_features_bool.float().mean(dim=-1, keepdim=True) # (B, L, 1) - analog
        ss4d_sparsity_analog = (spiked_x_for_ss4d_features_bool.float().sum(dim=-1, keepdim=True) / spiked_x_for_ss4d_features_bool.shape[-1]) # (B, L, 1) - analog

        # Extract multi-scale temporal features from the analog mean/sparsity
        ss4d_temporal_mean_k1_analog = self.causal_conv_mean_k1(ss4d_mean_activity_analog)
        ss4d_temporal_mean_k3_analog = self.causal_conv_mean_k3(ss4d_mean_activity_analog)
        ss4d_temporal_sparsity_k1_analog = self.causal_conv_sparsity_k1(ss4d_sparsity_analog)
        ss4d_temporal_sparsity_k3_analog = self.causal_conv_sparsity_k3(ss4d_sparsity_analog)

        # Concatenate for richer gate input (6 features) - analog
        ss4d_gate_features_analog = torch.cat([
            ss4d_mean_activity_analog, ss4d_temporal_mean_k1_analog, ss4d_temporal_mean_k3_analog,
            ss4d_sparsity_analog, ss4d_temporal_sparsity_k1_analog, ss4d_temporal_sparsity_k3_analog
        ], dim=-1) # (B, L, 6) - analog features

        # SS4D gate MLP takes analog features, outputs raw analog gate signals and its internal spikes
        ss4d_gate_analog_output_raw, ss4d_gate_binary_spikes = self.ss4d_gate_mlp(ss4d_gate_features_analog) # ss4d_gate_binary_spikes are binary
        all_spikes_list.extend(ss4d_gate_binary_spikes)

        dropped_ss4d_gate_current_analog = self.dropout_ss4d_gate_mlp(ss4d_gate_analog_output_raw)
        ss4d_gate_factor_analog = torch.sigmoid(dropped_ss4d_gate_current_analog + self.ss4d_gate_bias) # (B, L, D) - analog gating factor

        # Apply learned gating factor to the continuous input to SS4D core
        x_ss4d_gated_input_analog = x_ss4d_scaled_analog * ss4d_gate_factor_analog # (B, L, D) - continuous current

        # Transpose for Conv1d operations inside SS4D: (B, L, D) -> (B, D, L)
        x_ss4d_gated_input_transposed = rearrange(x_ss4d_gated_input_analog, 'b l d -> b d l')

        # Apply the main SS4D layer. SS4D expects continuous input `u` (B, D, L)
        ss4d_output_continuous, spikes_ss4d = self.layer(x_ss4d_gated_input_transposed) # ss4d_output_continuous is continuous, spikes_ss4d are binary
        all_spikes_list.append(spikes_ss4d)

        # Transpose back to standard shape: (B, D, L) -> (B, L, D)
        ss4d_output_continuous = rearrange(ss4d_output_continuous, 'b d l -> b l d')

        ss4d_output_continuous = self.dropout_ss4d_output(ss4d_output_continuous)
        x_after_ss4d_residual_analog = x_res_ss4d + ss4d_output_continuous # (B, L, D) - continuous residual

        # === Spiking MLP Branch with Gating ===
        output_analog = x_after_ss4d_residual_analog

        if self.use_spiking_mlp:
            x_res_mlp_analog = x_after_ss4d_residual_analog # (B, L, D) - continuous

            x_norm2_analog = self.norm2(x_res_mlp_analog) if self.prenorm else x_res_mlp_analog # (B, L, D) - continuous

            # --- Dynamic Input Scaling for MLP ---
            mean_x2 = x_norm2_analog.mean(dim=-1, keepdim=True)
            std_x2 = x_norm2_analog.std(dim=-1, keepdim=True).clamp(min=1e-5)
            stats2_analog = torch.cat([mean_x2, std_x2], dim=-1) # (B, L, 2) - analog features

            # mlp_input_scaler takes analog stats, outputs analog scale_raw_analog and its internal spikes
            scale2_raw_analog, scale2_spikes = self.mlp_input_scaler(stats2_analog) # scale2_spikes are binary
            all_spikes_list.extend(scale2_spikes)
            scale2_analog = F.softplus(scale2_raw_analog) + 0.5 # (B, L, D) - analog
            x_mlp_scaled_analog = x_norm2_analog * scale2_analog # (B, L, D) - analog current

            # --- Feature Generation for MLP Gate (from dedicated input spikes) ---
            # Convert the scaled continuous input to explicit binary spikes for gate feature extraction
            spiked_x_for_mlp_features_bool = self._apply_spike_activation_helper(x_mlp_scaled_analog, self.mlp_feature_spike_activation) # (B, L, D) - binary spikes
            all_spikes_list.append(spiked_x_for_mlp_features_bool)

            # Calculate mean activity and sparsity from these feature spikes (converted to float for calculations)
            mlp_mean_activity_analog = spiked_x_for_mlp_features_bool.float().mean(dim=-1, keepdim=True) # (B, L, 1) - analog
            mlp_sparsity_analog = (spiked_x_for_mlp_features_bool.float().sum(dim=-1, keepdim=True) / spiked_x_for_mlp_features_bool.shape[-1]) # (B, L, 1) - analog

            # Extract multi-scale temporal features from the analog mean/sparsity
            mlp_temporal_mean_k1_analog = self.causal_conv_mean_k1(mlp_mean_activity_analog)
            mlp_temporal_mean_k3_analog = self.causal_conv_mean_k3(mlp_mean_activity_analog)
            mlp_temporal_sparsity_k1_analog = self.causal_conv_sparsity_k1(mlp_sparsity_analog)
            mlp_temporal_sparsity_k3_analog = self.causal_conv_sparsity_k3(mlp_sparsity_analog)

            # Concatenate for richer gate input (6 features) - analog
            mlp_gate_features_analog = torch.cat([
                mlp_mean_activity_analog, mlp_temporal_mean_k1_analog, mlp_temporal_mean_k3_analog,
                mlp_sparsity_analog, mlp_temporal_sparsity_k1_analog, mlp_temporal_sparsity_k3_analog
            ], dim=-1) # (B, L, 6) - analog features

            # MLP gate MLP takes analog features + SS4D gate output (analog), outputs raw analog gate signals and its internal spikes
            # Pass ss4d_gate_analog_output_raw (if available and not None)
            mlp_gate_analog_output_raw, mlp_gate_binary_spikes = self.mlp_gate_mlp(mlp_gate_features_analog, extra_input=ss4d_gate_analog_output_raw) # mlp_gate_binary_spikes are binary
            all_spikes_list.extend(mlp_gate_binary_spikes)

            dropped_mlp_gate_current_analog = self.dropout_mlp_gate_mlp(mlp_gate_analog_output_raw)
            mlp_gate_factor_analog = torch.sigmoid(dropped_mlp_gate_current_analog + self.mlp_gate_bias) # (B, L, D) - analog gating factor

            # Apply learned gating factor to the continuous input to MLP core
            x_mlp_gated_input_analog = x_mlp_scaled_analog * mlp_gate_factor_analog # (B, L, D) - continuous current

            # --- Explicit Spiking before MLP ---
            # Convert the final gated analog input to explicit binary spikes before feeding to SpikingMLP
            spiked_x_mlp_input_bool = self._apply_spike_activation_helper(x_mlp_gated_input_analog, self.mlp_pre_spike_activation) # (B, L, D) - binary spikes
            all_spikes_list.append(spiked_x_mlp_input_bool)

            # MLP takes float spikes as input
            mlp_output_current_analog, spikes_mlp_internal = self.mlp(spiked_x_mlp_input_bool.float()) # mlp_output_current_analog is continuous, spikes_mlp_internal are binary
            all_spikes_list.extend(spikes_mlp_internal)

            mlp_output_current_analog = self.dropout_mlp_out(mlp_output_current_analog)
            output_analog = x_res_mlp_analog + mlp_output_current_analog # (B, L, D) - continuous residual

        # === Finalization ===
        if not self.prenorm:
            output_analog = self.norm_final(output_analog)

        # Concatenate all collected spikes into a single flat tensor of binary spikes (torch.bool)
        # Ensure all tensors are converted to bool before cat for consistent type.
        # Filter out None values just in case (though they should all be tensors here)
        final_spikes = torch.cat([s.flatten().to(dtype=torch.bool) for s in all_spikes_list if s is not None])

        return output_analog, reg_loss, final_spikes