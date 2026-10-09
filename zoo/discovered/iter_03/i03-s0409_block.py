"""Implements a SpikingFLABlock with Adaptive DPT, R-ESGFN, and SCRG for enhanced neuromorphic integration.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Tuple, List, Optional

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

# =============================================================================
# ===== SpikingFLABlock with Adaptive DPT, R-ESGFN, and SCRG ============ 
# =============================================================================

class SpikingFLABlock(nn.Module):
    """SpikingFLABlock with Adaptive Dynamic Potential Threshold (ADPT),
    Refined Enhanced Spiking Gated Feed-Forward Network (R-ESGFN),
    and Spike-Controlled Residual Gating (SCRG).

    This evolved architecture introduces several innovations to enhance the efficiency and spiking nature:
    1.  Adaptive Dynamic Potential Threshold (ADPT): A learnable base threshold is dynamically
        offset by an activity-dependent modulation derived from the block's input. This provides
        a unified, adaptive control over the excitability of spiking linear layers within the block.
    2.  Refined Enhanced Spiking Gated FFN (R-ESGFN): The Feed-Forward Network is redesigned
        with a three-branch parallel spiking structure where the contributions of the main and
        first gate branch are scaled by learnable weights before being multiplicatively gated
        by the second gate branch. This allows more flexible weighting of spiking pathways
        while maintaining computational efficiency.
    3.  Spike-Controlled Residual Gating (SCRG): Residual connections are robustly modulated
        by binary spikes from forget and update gates, enforcing a more explicit spiking control flow.
    4.  Dynamic Branch-Specific Thresholding (DBST): Each branch of the ESGFN now has its
        own learnable threshold offset, which is added to the block-level ADPT, allowing
        fine-grained control over the excitability of individual FFN pathways.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, d_ffn_mult: int=2, neuron_type: str='sdn', use_spiking_mlp: bool=True, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.use_spiking_mlp = use_spiking_mlp
        self.d_ffn = d_model * d_ffn_mult

        surrogate_function = piecewise_quadratic_surrogate()

        # Adaptive Dynamic Potential Threshold (ADPT) components
        # Base learnable log-threshold, applied per feature dimension
        self.base_log_vth = nn.Parameter(torch.zeros(d_model))
        # SpikingLinear to project input mean activity for dynamic threshold offset
        self.activity_mod_proj = SpikingLinear(
            nn.Linear(1, d_model), # Input is mean activity (B, L, 1), output modulates d_model thresholds
            neuron_type,
            surrogate_function
        )

        # --- Gated SS4D Sub-block Components ---
        self.norm1 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) # Dropout for SS4D output branch

        ss4d_kwargs = kwargs.pop('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs['neuron'] = neuron_type
        # CRITICAL: Disable SS4D's internal learnable_vth to centralize DPT
        ss4d_kwargs['learnable_vth'] = False
        self.layer = SS4D(d_model, transposed=True, dropout=dropout, **ss4d_kwargs)

        self.norm_gate = RMSNorm(d_model)
        self.gate_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False),
            neuron_type,
            surrogate_function
        )
        self.gain = nn.Parameter(torch.ones(1))

        # NEW: Consolidated Post-SS4D Residual Gating (SCRG)
        self.ss4d_residual_gate_proj = SpikingLinear(nn.Linear(d_model, 2 * d_model, bias=False), neuron_type, surrogate_function)
        self.ss4d_residual_forget_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.ss4d_residual_update_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

        # --- Refined Enhanced Spiking Gated Feed-Forward Network (R-ESGFN) Sub-block ---
        if self.use_spiking_mlp:
            self.norm2 = RMSNorm(d_model)
            # FFN components, all using SpikingLinear and SpikingPostNormActivation for intermediate spikes
            self.ffn_main_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_gate1_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_gate2_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.spiking_ffn_main_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
            self.spiking_ffn_gate1_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
            self.spiking_ffn_gate2_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

            self.ffn_down_proj = SpikingLinear(
                nn.Linear(self.d_ffn, d_model, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_dropout_internal = nn.Dropout(dropout) # Dropout for FFN internal potentials before spiking

            # Learnable weights for R-ESGFN combination
            self.main_spike_weight = nn.Parameter(torch.ones(1))
            self.gate1_spike_weight = nn.Parameter(torch.ones(1))

            # NEW: Dynamic Branch-Specific Thresholding (DBST) parameters for ESGFN branches
            self.ffn_main_log_vth_offset = nn.Parameter(torch.zeros(d_model))
            self.ffn_gate1_log_vth_offset = nn.Parameter(torch.zeros(d_model))
            self.ffn_gate2_log_vth_offset = nn.Parameter(torch.zeros(d_model))

            # NEW: Consolidated Post-MLP Residual Gating (SCRG)
            self.mlp_residual_gate_proj = SpikingLinear(nn.Linear(d_model, 2 * d_model, bias=False), neuron_type, surrogate_function)
            self.mlp_residual_forget_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
            self.mlp_residual_update_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # --- Adaptive Dynamic Potential Threshold (ADPT) Calculation ---
        # Calculate mean activity of the input for dynamic threshold modulation
        input_mean_activity = x.float().mean(dim=-1, keepdim=True) # Shape (B, L, 1)

        # Project mean activity to d_model dimensions via SpikingLinear
        activity_offset_potential, s_activity_offset = self.activity_mod_proj(input_mean_activity)
        all_spikes_list.append(s_activity_offset)

        # Apply tanh to the potential to bound the modulation, then add to base log threshold
        activity_offset = activity_offset_potential.tanh() # Bounded offset for thresholds

        # Combined DPT factor: base learnable threshold + activity-dependent offset
        shared_vth_factor = torch.exp(rearrange(self.base_log_vth, 'd -> 1 1 d') + activity_offset)

        x_res1 = x
        x_norm1 = self.norm1(x) if self.prenorm else x

        # Apply ADPT to normalize potentials entering the SS4D sub-block
        x_dpt_scaled = x_norm1 / shared_vth_factor

        # --- Gated SS4D Sub-block ---
        x_scaled = x_dpt_scaled * self.gain # Apply gain to the ADPT-scaled input

        # Gating Pathway for SS4D output
        # Input to gate_proj (SpikingLinear) is the ADPT-scaled potential
        gate_raw_potential, gate_spikes = self.gate_proj(x_scaled)
        all_spikes_list.append(gate_spikes)

        # SS4D computation
        x_ssm_input = rearrange(x_scaled, 'b l d -> b d l') # Input to SS4D is ADPT-scaled potential
        x_ssm_output_potential, ssm_proj_spikes = self.layer(x_ssm_input)
        all_spikes_list.append(ssm_proj_spikes)
        x_ssm_output_potential = rearrange(x_ssm_output_potential, 'b d l -> b l d')

        # Apply dropout to the main SS4D output branch (analog potential)
        x_ssm_output_dropped = self.dropout1(x_ssm_output_potential)

        # Apply the learned gate (binary spikes) to the SS4D output (analog potential)
        gated_ssm_output = gate_spikes * x_ssm_output_dropped

        # NEW: Post-SS4D Spike-Controlled Residual Gating (SCRG)
        # Input to ss4d_residual_gate_proj (SpikingLinear) is the original residual (potential)
        ss4d_gate_potentials, s_ss4d_residual_gate = self.ss4d_residual_gate_proj(x_res1)
        all_spikes_list.append(s_ss4d_residual_gate) # Collect spikes from the gate projection

        forget_gate_ss4d_cont, update_gate_ss4d_cont = torch.chunk(ss4d_gate_potentials, 2, dim=-1)

        forget_gate_ss4d_spikes = self.ss4d_residual_forget_act(forget_gate_ss4d_cont)
        update_gate_ss4d_spikes = self.ss4d_residual_update_act(update_gate_ss4d_cont)
        all_spikes_list.extend([forget_gate_ss4d_spikes, update_gate_ss4d_spikes])

        # Residual connection gated by spikes
        x = x_res1 * forget_gate_ss4d_spikes + gated_ssm_output * update_gate_ss4d_spikes

        # --- Refined Enhanced Spiking Gated Feed-Forward Network (R-ESGFN) Sub-block ---
        if self.use_spiking_mlp:
            x_res2 = x
            x_norm2 = self.norm2(x) if self.prenorm else x

            # Apply Dynamic Branch-Specific Thresholding (DBST) for ESGFN branches
            # Each FFN branch gets its own DPT factor, derived from base_log_vth + activity_offset + branch_specific_offset
            main_branch_vth_factor = torch.exp(rearrange(self.base_log_vth + self.ffn_main_log_vth_offset, 'd -> 1 1 d') + activity_offset)
            gate1_branch_vth_factor = torch.exp(rearrange(self.base_log_vth + self.ffn_gate1_log_vth_offset, 'd -> 1 1 d') + activity_offset)
            gate2_branch_vth_factor = torch.exp(rearrange(self.base_log_vth + self.ffn_gate2_log_vth_offset, 'd -> 1 1 d') + activity_offset)

            x_main_dpt = x_norm2 / main_branch_vth_factor
            x_gate1_dpt = x_norm2 / gate1_branch_vth_factor
            x_gate2_dpt = x_norm2 / gate2_branch_vth_factor

            # R-ESGFN: Three parallel spiking branches
            # Inputs to ffn_main_proj, ffn_gate1_proj, ffn_gate2_proj (SpikingLinear) are branch-specific DPT-scaled potentials
            main_potential, s_main = self.ffn_main_proj(x_main_dpt)
            gate1_potential, s_gate1 = self.ffn_gate1_proj(x_gate1_dpt)
            gate2_potential, s_gate2 = self.ffn_gate2_proj(x_gate2_dpt)

            # Apply dropout to continuous potentials before spiking
            main_potential_dropped = self.ffn_dropout_internal(main_potential)
            gate1_potential_dropped = self.ffn_dropout_internal(gate1_potential)
            gate2_potential_dropped = self.ffn_dropout_internal(gate2_potential)

            # Convert potentials to spikes using SpikingPostNormActivation
            main_spikes = self.spiking_ffn_main_act(main_potential_dropped)
            gate1_spikes = self.spiking_ffn_gate1_act(gate1_potential_dropped)
            gate2_spikes = self.spiking_ffn_gate2_act(gate2_potential_dropped)
            all_spikes_list.extend([s_main, s_gate1, s_gate2, main_spikes, gate1_spikes, gate2_spikes])

            # Refined combination of spikes for intermediate potential
            # (weighted sum of main and gate1 spikes) * gate2 spikes
            inter_potential = (self.main_spike_weight * main_spikes + self.gate1_spike_weight * gate1_spikes) * gate2_spikes

            # FFN down-projection. Input to this SpikingLinear is the combined spike-derived potential.
            ffn_final_output_potential, s_down = self.ffn_down_proj(inter_potential)
            all_spikes_list.append(s_down)

            # NEW: Post-MLP Spike-Controlled Residual Gating (SCRG)
            # Input to mlp_residual_gate_proj (SpikingLinear) is the previous residual (potential)
            mlp_gate_potentials, s_mlp_residual_gate = self.mlp_residual_gate_proj(x_res2)
            all_spikes_list.append(s_mlp_residual_gate)

            forget_gate_mlp_cont, update_gate_mlp_cont = torch.chunk(mlp_gate_potentials, 2, dim=-1)

            forget_gate_mlp_spikes = self.mlp_residual_forget_act(forget_gate_mlp_cont)
            update_gate_mlp_spikes = self.mlp_residual_update_act(update_gate_mlp_cont)
            all_spikes_list.extend([forget_gate_mlp_spikes, update_gate_mlp_spikes])

            # Residual connection gated by spikes
            output = x_res2 * forget_gate_mlp_spikes + ffn_final_output_potential * update_gate_mlp_spikes
        else:
            output = x

        # Post-normalization if not pre-norm
        if not self.prenorm:
            output = self.norm1(output)

        # Collect all spikes into a single flattened tensor
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None
        reg_loss = None # Keep for compatibility

        # Ensure final_spikes is binary float tensor
        if final_spikes is not None:
            final_spikes = (final_spikes > 0).float()

        return output, reg_loss, final_spikes