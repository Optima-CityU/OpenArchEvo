import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Tuple, List, Optional, Dict

# ==================== SNN Imports and Helpers ====================
try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
    from src.models.spike.nn import SpikingLinear
    from src.models.nn.dropout import DropoutNd
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder SNN modules.")
    class DummySurrogate:
        @staticmethod
        def apply(x, *args, **kwargs):
            # Simple thresholding for dummy surrogate
            return (x > 0).float()
    class DummyNeuron:
        def __init__(self, surrogate_function=None, **kwargs):
            self.surrogate_function = surrogate_function if surrogate_function is not None else DummySurrogate.apply
            # Default threshold for the dummy neuron, can be adjusted
            self.threshold = 1.0
        def __call__(self, x):
            # Apply threshold and then surrogate
            return self.surrogate_function(x - self.threshold)

    neuron_registry = {'sdn': lambda surrogate_function=None, **kwargs: DummyNeuron(surrogate_function, **kwargs)}
    piecewise_quadratic_surrogate = lambda: DummySurrogate.apply

    class SpikingLinear(nn.Module):
        """A sequence module containing LIF neurons and a linear layer."""
        def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
            super().__init__()
            self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
            self.linear = linear_layer

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x_transposed = x.transpose(-1, -2)
            spikes_transposed = self.neuron(x_transposed)
            spikes = spikes_transposed.transpose(-1, -2)
            output = self.linear(spikes)
            return output, spikes

    class DropoutNd(nn.Dropout):
        def __init__(self, p: float = 0.5, inplace: bool = False):
            super().__init__(p=p, inplace=inplace)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(x)

# RMSNorm import, also with a placeholder
try:
    from fla.modules import RMSNorm
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder RMSNorm.")
    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.scale = dim ** -0.5
            self.gamma = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return F.normalize(x, dim=-1) * self.scale * self.gamma

# ==================== New: SpikingPostNormActivation ====================

class SpikingPostNormActivation(nn.Module):
    def __init__(self, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if surrogate_function is None:
            surrogate_function = piecewise_quadratic_surrogate()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_transposed = x.transpose(-1, -2)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2) # Transpose back to (batch, seq_len, dim)
        return spikes

# =============================================================================
# ===== Start of SpikingSSM Components (S4DKernel, SS4D) - UNCHANGED ============
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
        # Using rearrange for tensor operations
        K = rearrange(dtA, 'h n -> h n 1') * torch.arange(L, device=A.device)
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
        self.learnable_vth = learnable_vth # This will be set to False from SpikingFLABlock
        self.bidirectional = bidirectional
        self.channels = channels
        self.D = nn.Parameter(torch.randn(self.channels, self.h))
        if learnable_vth: # Only create if learnable_vth is True
            self.ln_vth = nn.Parameter(torch.zeros(d_model, 1) if not shared_vth else torch.zeros(1))
        else:
            self.register_buffer('ln_vth', torch.zeros(1)) # Placeholder if not learnable

        kernel_channels_for_s4d = self.channels * 2 if bidirectional else self.channels
        self.kernel = S4DKernel(self.h, N=self.n, channels=kernel_channels_for_s4d, **kernel_args)

        self.dropout_layer = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        self.output_linear_proj = SpikingLinear(
            nn.Linear(self.h * self.channels, self.h, bias=False),
            neuron,
            piecewise_quadratic_surrogate()
        )

    def forward(self, u, **kwargs):
        if not self.transposed: u = rearrange(u, 'b l d -> b d l') # (B, D, L)
        L = u.size(-1)
        k, _ = self.kernel(L=L)

        if self.bidirectional:
            k_parts = rearrange(k, "(s c) h l -> s c h l", s=2, c=self.channels)
            k_combined = F.pad(k_parts[0], (0, L)) + F.pad(k_parts[1].flip(-1), (L, 0))
            k = k_combined

        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.einsum("bhl,chl->bchl", u_f, k_f)

        y = torch.fft.irfft(y, n=2 * L)[..., :L]
        y = y + torch.einsum("bhl,ch->bchl", u, self.D)

        y = rearrange(y, "b c h l -> b (c h) l")

        # Apply SS4D's internal learnable_vth ONLY IF it's enabled for SS4D.
        # In this design, it will be disabled by SpikingFLABlock.
        if self.learnable_vth:
            y = y / torch.exp(rearrange(self.ln_vth, 'd 1 -> 1 d 1') if self.ln_vth.dim() == 2 else rearrange(self.ln_vth, 'd -> 1 d 1'))

        B_dim, D_in_dim, L_dim = y.shape
        y_reshaped_for_linear = rearrange(y, 'b d l -> (b l) d')

        y_real_valued_output, ssm_proj_spikes = self.output_linear_proj(y_reshaped_for_linear)

        y_real_valued_output_dropped = self.dropout_layer(y_real_valued_output)

        y_final_ssm_output = rearrange(y_real_valued_output_dropped, '(b l) d -> b d l', b=B_dim, l=L_dim)

        if not self.transposed: y_final_ssm_output = rearrange(y_final_ssm_output, 'b d l -> b l d')

        return y_final_ssm_output, ssm_proj_spikes

class SpikingFLABlock(nn.Module):
    """SpikingFLABlock with Spiking Co-Activation Unit (SCU) and FFN Local Residual Gating (FLRG).

    This evolved architecture introduces several innovations to enhance efficiency, spiking nature,
    and information flow control:

    1.  Adaptive Dynamic Potential Threshold (ADPT) (Retained): A learnable base threshold is dynamically
        offset by an activity-dependent modulation derived from the block's input. This provides
        a unified, adaptive control over the excitability of spiking linear layers within the block.
        All subsequent SpikingLinear inputs are scaled by this dynamic threshold.

    2.  Adaptive Unified Spiking Gate (AUSG) (Modified):
        The AUSG now primarily focuses on the SS4D sub-block's residual gating. It generates:
        -   `ss4d_input_spikes`: Dynamically modulates the input to the SS4D kernel.
        -   `residual_forget_spikes`: Controls the proportion of the SS4D residual pathway to "forget" or suppress.
        -   `residual_update_spikes`: Controls the proportion of the SS4D branch output to "update" or add to the residual.

    3.  Spiking Co-Activation Unit (SCU) (New):
        Redesigns the interaction within the SCM-FFN. Instead of a simple element-wise multiplication
        of `main_spikes * mod_spikes`, this unit takes the concatenated spike trains from the
        main and modulation FFN branches, passes them through a SpikingLinear layer, and then through
        a SpikingPostNormActivation. This allows the network to learn more complex, non-linear,
        and dynamic combinations of the two spike streams, providing a richer feature interaction.

    4.  FFN Local Residual Gating (FLRG) (New):
        Introduces a dedicated, local spiking gating mechanism for the FFN's residual connection.
        This gate is generated from the FFN's pre-normalized input and produces `ffn_local_forget_spikes`
        and `ffn_local_update_spikes`. This enables the FFN to independently and adaptively control
        its residual information flow, enhancing modularity and context-specific processing.

    5.  Spike-Activated Dropout (SAD) in SCM-FFN (Retained): Dropout is applied directly to the binary
        spikes generated by the parallel FFN branches, enforcing sparsity and promoting robust
        spike-based information processing.

    6.  Dynamic Branch-Specific Thresholding (DBST) (Retained): Each branch of the
        SCM-FFN retains its own learnable threshold offset, which is added to the block-level ADPT,
        allowing fine-grained control over the excitability of individual FFN pathways.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, d_ffn_mult: int=2, neuron_type: str='sdn', use_spiking_mlp: bool=True, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.use_spiking_mlp = use_spiking_mlp
        self.d_ffn = d_model * d_ffn_mult

        surrogate_function = piecewise_quadratic_surrogate()

        # Adaptive Dynamic Potential Threshold (ADPT) components (Retained)
        self.base_log_vth = nn.Parameter(torch.zeros(d_model)) # Base learnable log-threshold per feature dimension
        self.activity_mod_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False), # Input is (B, L, D), projects to (B, L, D) for per-dim modulation
            neuron_type,
            surrogate_function
        )

        # --- Adaptive Unified Spiking Gate (AUSG) --- (Now primarily for SS4D residual and SS4D input)
        # Projects pre-normalized input to generate global gating signals
        # Outputs: SS4D Input Modulator, Residual Forget (SS4D), Residual Update (SS4D)
        # Each is d_model, so total 3 * d_model
        self.ausg_proj = SpikingLinear(
            nn.Linear(d_model, 3 * d_model, bias=False),
            neuron_type,
            surrogate_function
        )
        self.ausg_ss4d_input_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.ausg_residual_forget_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.ausg_residual_update_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

        # --- SS4D Sub-block Components --- (Retained)
        self.norm1 = RMSNorm(d_model)
        self.dropout1 = DropoutNd(dropout) # Dropout for SS4D output branch

        ss4d_kwargs = kwargs.pop('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs['neuron'] = neuron_type
        # CRITICAL: Disable SS4D's internal learnable_vth to centralize DPT
        ss4d_kwargs['learnable_vth'] = False
        self.layer = SS4D(d_model, transposed=True, dropout=dropout, **ss4d_kwargs)

        # --- Spiking Cross-Modulation Feed-Forward Network (SCM-FFN) Sub-block --- (Modified)
        if self.use_spiking_mlp:
            self.norm2 = RMSNorm(d_model)

            # SCM-FFN: Two parallel spiking branches for main and modulation
            self.ffn_main_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_mod_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.spiking_ffn_main_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
            self.spiking_ffn_mod_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

            # Spike-Activated Dropout (SAD) for FFN branches
            self.ffn_main_dropout = DropoutNd(dropout)
            self.ffn_mod_dropout = DropoutNd(dropout)

            # NEW: Spiking Co-Activation Unit (SCU)
            self.ffn_co_act_proj = SpikingLinear(
                nn.Linear(2 * self.d_ffn, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.spiking_ffn_co_act_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

            # Integrator neuron for modulated spikes
            self.spiking_ffn_integrator_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

            self.ffn_down_proj = SpikingLinear(
                nn.Linear(self.d_ffn, d_model, bias=False),
                neuron_type,
                surrogate_function
            )

            # Dynamic Branch-Specific Thresholding (DBST) parameters for SCM-FFN branches
            self.ffn_main_log_vth_offset = nn.Parameter(torch.zeros(d_model))
            self.ffn_mod_log_vth_offset = nn.Parameter(torch.zeros(d_model))

            # NEW: FFN Local Residual Gating (FLRG)
            self.ffn_local_gating_proj = SpikingLinear(
                nn.Linear(d_model, 2 * d_model, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_local_forget_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
            self.ffn_local_update_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # --- Adaptive Dynamic Potential Threshold (ADPT) Calculation ---
        activity_offset_potential, s_activity_offset = self.activity_mod_proj(x)
        all_spikes_list.append(s_activity_offset)

        # Apply tanh to the potential to bound the modulation, then add to base log threshold
        activity_offset = activity_offset_potential.tanh() # Bounded offset for thresholds

        # Combined DPT factor: base learnable threshold + activity-dependent offset
        shared_vth_factor = torch.exp(rearrange(self.base_log_vth, 'd -> 1 1 d') + activity_offset)

        x_res1 = x # Residual connection for SS4D sub-block (analog potential)
        x_norm1 = self.norm1(x) if self.prenorm else x

        # Apply ADPT to normalize potentials entering the block's main processing path
        x_dpt_scaled = x_norm1 / shared_vth_factor

        # --- Adaptive Unified Spiking Gate (AUSG) Calculation ---
        # Input to AUSG is the ADPT-scaled pre-normalized input
        ausg_potentials, s_ausg_proj = self.ausg_proj(x_dpt_scaled)
        all_spikes_list.append(s_ausg_proj)

        # Split potentials into SS4D input modulator, residual forget (SS4D), residual update (SS4D)
        ss4d_input_mod_pot, residual_forget_pot, residual_update_pot = torch.chunk(ausg_potentials, 3, dim=-1)

        ss4d_input_spikes = self.ausg_ss4d_input_act(ss4d_input_mod_pot)
        residual_forget_spikes = self.ausg_residual_forget_act(residual_forget_pot)
        residual_update_spikes = self.ausg_residual_update_act(residual_update_pot)
        all_spikes_list.extend([ss4d_input_spikes, residual_forget_spikes, residual_update_spikes])

        # --- Gated SS4D Sub-block ---
        # Apply AUSG's ss4d_input_spikes to modulate SS4D input
        x_ss4d_input = x_dpt_scaled * ss4d_input_spikes # Element-wise binary channel mask

        # SS4D computation
        x_ssm_input_reshaped = rearrange(x_ss4d_input, 'b l d -> b d l')
        x_ssm_output_potential, ssm_proj_spikes = self.layer(x_ssm_input_reshaped)
        all_spikes_list.append(ssm_proj_spikes)
        x_ssm_output_potential = rearrange(x_ssm_output_potential, 'b d l -> b l d')

        # Apply dropout to the main SS4D output branch (analog potential)
        x_ssm_output_dropped = self.dropout1(x_ssm_output_potential)

        # --- Apply AUSG's residual gating for SS4D sub-block ---
        # x_res1 is original pre-skip analog potential
        x_after_ss4d_gated = x_res1 * residual_forget_spikes + x_ssm_output_dropped * residual_update_spikes

        # --- Spiking Cross-Modulation Feed-Forward Network (SCM-FFN) Sub-block ---
        if self.use_spiking_mlp:
            x_res2 = x_after_ss4d_gated # Residual connection for FFN sub-block (analog potential)
            x_norm2 = self.norm2(x_after_ss4d_gated) if self.prenorm else x_after_ss4d_gated

            # NEW: FFN Local Residual Gating (FLRG) calculation
            ffn_gating_potentials, s_ffn_gate_proj = self.ffn_local_gating_proj(x_norm2)
            all_spikes_list.append(s_ffn_gate_proj)
            ffn_local_forget_pot, ffn_local_update_pot = torch.chunk(ffn_gating_potentials, 2, dim=-1)

            ffn_local_forget_spikes = self.ffn_local_forget_act(ffn_local_forget_pot)
            ffn_local_update_spikes = self.ffn_local_update_act(ffn_local_update_pot)
            all_spikes_list.extend([ffn_local_forget_spikes, ffn_local_update_spikes])

            # Apply Dynamic Branch-Specific Thresholding (DBST) for SCM-FFN branches
            main_branch_vth_factor = torch.exp(rearrange(self.base_log_vth + self.ffn_main_log_vth_offset, 'd -> 1 1 d') + activity_offset)
            mod_branch_vth_factor = torch.exp(rearrange(self.base_log_vth + self.ffn_mod_log_vth_offset, 'd -> 1 1 d') + activity_offset)

            x_main_dpt_scaled = x_norm2 / main_branch_vth_factor
            x_mod_dpt_scaled = x_norm2 / mod_branch_vth_factor

            # SCM-FFN: Two parallel spiking branches for main and modulation
            main_potential, s_main_proj = self.ffn_main_proj(x_main_dpt_scaled)
            mod_potential, s_mod_proj = self.ffn_mod_proj(x_mod_dpt_scaled)
            all_spikes_list.extend([s_main_proj, s_mod_proj])

            # Convert potentials to spikes
            main_spikes_raw = self.spiking_ffn_main_act(main_potential)
            mod_spikes_raw = self.spiking_ffn_mod_act(mod_potential)

            # Spike-Activated Dropout (SAD) applied directly to spikes
            main_spikes_dropped = self.ffn_main_dropout(main_spikes_raw)
            mod_spikes_dropped = self.ffn_mod_dropout(mod_spikes_raw)
            all_spikes_list.extend([main_spikes_raw, mod_spikes_raw, main_spikes_dropped, mod_spikes_dropped])

            # NEW: Spiking Co-Activation Unit (SCU)
            combined_input = torch.cat([main_spikes_dropped, mod_spikes_dropped], dim=-1)
            co_activation_potential, s_co_act_proj = self.ffn_co_act_proj(combined_input)
            all_spikes_list.append(s_co_act_proj)
            co_activated_spikes = self.spiking_ffn_co_act_act(co_activation_potential)
            all_spikes_list.append(co_activated_spikes)

            # Integrator neuron for co-activated spikes
            integrated_spikes = self.spiking_ffn_integrator_act(co_activated_spikes)
            all_spikes_list.append(integrated_spikes)

            # FFN down-projection. Input to this SpikingLinear is the combined sparse spike train.
            ffn_final_output_potential, s_down_proj = self.ffn_down_proj(integrated_spikes)
            all_spikes_list.append(s_down_proj)

            # --- Apply FFN Local Residual Gating (FLRG) ---
            # x_res2 is previous residual (analog potential)
            output = x_res2 * ffn_local_forget_spikes + ffn_final_output_potential * ffn_local_update_spikes
        else:
            output = x_after_ss4d_gated

        # Post-normalization if not pre-norm
        if not self.prenorm:
            output = self.norm1(output) # Assuming final norm for the whole block

        # Collect all spikes into a single flattened tensor
        # Ensure final_spikes is binary float tensor (already handled by neuron outputs, but explicit for robustness)
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None]) if all_spikes_list else None
        if final_spikes is not None:
            final_spikes = (final_spikes > 0).float()

        reg_loss = None # Keep for compatibility

        return output, reg_loss, final_spikes