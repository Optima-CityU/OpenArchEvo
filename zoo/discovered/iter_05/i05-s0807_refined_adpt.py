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
            # x is analog input potential
            x_transposed = x.transpose(-1, -2) # (B, L, D) -> (B, D, L) for neuron processing
            spikes_transposed = self.neuron(x_transposed) # Spikes (binary)
            spikes = spikes_transposed.transpose(-1, -2) # (B, D, L) -> (B, L, D)
            output = self.linear(spikes) # The internal nn.Linear receives binary spikes
            return output, spikes # output is analog, spikes are binary

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
        self.learnable_vth = learnable_vth
        self.bidirectional = bidirectional
        self.channels = channels
        self.D = nn.Parameter(torch.randn(self.channels, self.h))
        if learnable_vth:
            self.ln_vth = nn.Parameter(torch.zeros(d_model, 1) if not shared_vth else torch.zeros(1))
        else:
            self.register_buffer('ln_vth', torch.zeros(1))

        kernel_channels_for_s4d = self.channels * 2 if bidirectional else self.channels
        self.kernel = S4DKernel(self.h, N=self.n, channels=kernel_channels_for_s4d, **kernel_args)

        self.dropout_layer = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        self.output_linear_proj = SpikingLinear(
            nn.Linear(self.h * self.channels, self.h, bias=False),
            neuron,
            piecewise_quadratic_surrogate()
        )

    def forward(self, u, **kwargs):
        if not self.transposed: u = rearrange(u, 'b l d -> b d l')
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

        if self.learnable_vth:
            # Replaced manual view/reshape with rearrange for consistency and robustness
            y = y / torch.exp(rearrange(self.ln_vth, 'd 1 -> 1 d 1') if self.ln_vth.dim() == 2 else rearrange(self.ln_vth, 'd -> 1 d 1'))

        B_dim, D_in_dim, L_dim = y.shape
        y_reshaped_for_linear = rearrange(y, 'b d l -> (b l) d')

        y_real_valued_output, ssm_proj_spikes = self.output_linear_proj(y_reshaped_for_linear)

        y_real_valued_output_dropped = self.dropout_layer(y_real_valued_output)

        y_final_ssm_output = rearrange(y_real_valued_output_dropped, '(b l) d -> b d l', b=B_dim, l=L_dim)

        if not self.transposed: y_final_ssm_output = rearrange(y_final_ssm_output, 'b d l -> b l d')

        return y_final_ssm_output, ssm_proj_spikes

# =============================================================================
# ===== SpikingFLABlock with Refined Adaptive DPT (R-ADPT), S-ESGFN, and SCRG ============
# =============================================================================

class SpikingFLABlock(nn.Module):
    """SpikingFLABlock with Refined Adaptive Dynamic Potential Threshold (R-ADPT),
    Simplified Enhanced Spiking Gated Feed-Forward Network (S-ESGFN),
    and Spike-Controlled Residual Gating (SCRG).

    This evolved architecture introduces several innovations to enhance the efficiency and spiking nature:
    1.  Refined Adaptive Dynamic Potential Threshold (R-ADPT): The adaptive threshold mechanism is
        modified to calculate a block-global activity signal (mean across sequence length) to modulate
        the base potential threshold. This provides a more stable and efficient global adaptation
        of neuron excitability, rather than a per-token modulation.
    2.  Simplified Enhanced Spiking Gated FFN (S-ESGFN): The Feed-Forward Network is redesigned
        with a two-branch parallel spiking structure (content and gate). The content branch's
        spikes are directly and multiplicatively gated by the gate branch's spikes, offering a
        more direct and interpretable spike-based information flow compared to weighted sums,
        aligning better with neuromorphic principles of direct spike communication.
    3.  Spike-Controlled Residual Gating (SCRG): Residual connections are robustly modulated
        by binary spikes from forget and update gates, enforcing a more explicit spiking control flow.
    4.  Dynamic Branch-Specific Thresholding (DBST): Each branch of the S-ESGFN now has its
        own learnable threshold offset, which is added to the block-level R-ADPT, allowing
        fine-grained control over the excitability of individual FFN pathways.
    5.  Spike Activity Regularization: An optional regularization term is introduced to
        encourage sparsity in the overall spike activity, which is crucial for the efficiency
        and biological plausibility of Spiking Neural Networks.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, d_ffn_mult: int=2, neuron_type: str='sdn', use_spiking_mlp: bool=True, spike_loss_weight: float=0.0, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.use_spiking_mlp = use_spiking_mlp
        self.d_ffn = d_model * d_ffn_mult
        self.spike_loss_weight = spike_loss_weight # Added spike regularization weight

        surrogate_function = piecewise_quadratic_surrogate()

        # Refined Adaptive Dynamic Potential Threshold (R-ADPT) components
        # base_log_vth is a learnable per-feature base threshold.
        self.base_log_vth = nn.Parameter(torch.zeros(d_model))
        # activity_mod_proj now takes d_model (global mean activity per feature) and projects to d_model
        self.activity_mod_proj = SpikingLinear(
            nn.Linear(d_model, d_model), # Input to linear is (B, D), output is (B, D)
            neuron_type,
            surrogate_function
        )

        # --- Gated SS4D Sub-block Components ---
        self.norm1 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        ss4d_kwargs = kwargs.pop('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs['neuron'] = neuron_type
        ss4d_kwargs['learnable_vth'] = False # CRITICAL: Disable SS4D's internal learnable_vth
        self.layer = SS4D(d_model, transposed=True, dropout=dropout, **ss4d_kwargs)

        # Preserved from v2, although not used explicitly in this forward pass in v2. Kept for compatibility
        self.norm_gate = RMSNorm(d_model)
        self.gate_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False),
            neuron_type,
            surrogate_function
        )
        self.gain = nn.Parameter(torch.ones(1))

        # SCRG for SS4D residual
        self.ss4d_residual_gate_proj = SpikingLinear(nn.Linear(d_model, 2 * d_model, bias=False), neuron_type, surrogate_function)
        self.ss4d_residual_forget_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
        self.ss4d_residual_update_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

        # --- Simplified Enhanced Spiking Gated Feed-Forward Network (S-ESGFN) Sub-block ---
        if self.use_spiking_mlp:
            self.norm2 = RMSNorm(d_model)
            # S-ESGFN: Two parallel branches (content and gate)
            self.ffn_content_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_gate_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.spiking_ffn_content_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
            self.spiking_ffn_gate_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

            self.ffn_down_proj = SpikingLinear(
                nn.Linear(self.d_ffn, d_model, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_dropout_internal = nn.Dropout(dropout)

            # DBST parameters for S-ESGFN branches
            self.ffn_content_log_vth_offset = nn.Parameter(torch.zeros(d_model))
            self.ffn_gate_log_vth_offset = nn.Parameter(torch.zeros(d_model))

            # SCRG for MLP residual
            self.mlp_residual_gate_proj = SpikingLinear(nn.Linear(d_model, 2 * d_model, bias=False), neuron_type, surrogate_function)
            self.mlp_residual_forget_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
            self.mlp_residual_update_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # --- Refined Adaptive Dynamic Potential Threshold (R-ADPT) Calculation ---
        # Calculate a global mean activity for the block input
        # Average across sequence length L, resulting in (B, D)
        global_input_activity = x.float().mean(dim=1) # (B, L, D) -> (B, D)

        # Project this global activity to modulate the threshold
        # activity_mod_proj expects (B, D) and outputs (B, D)
        activity_offset_potential, s_activity_offset = self.activity_mod_proj(global_input_activity)
        all_spikes_list.append(s_activity_offset)

        # Apply activation to get the final offset
        activity_offset = activity_offset_potential.tanh() # (B, D)

        # The shared_vth_factor is now (B, 1, d_model), broadcastable over L
        # It combines the learnable base_log_vth (per-feature) with the global activity offset (per-batch, per-feature)
        shared_vth_factor = torch.exp(rearrange(self.base_log_vth, 'd -> 1 1 d') + rearrange(activity_offset, 'b d -> b 1 d'))

        x_res1 = x
        x_norm1 = self.norm1(x) if self.prenorm else x
        # Apply the R-ADPT threshold scaling
        x_dpt_scaled = x_norm1 / shared_vth_factor

        # --- Gated SS4D Sub-block ---
        x_scaled = x_dpt_scaled * self.gain

        gate_raw_potential, gate_spikes = self.gate_proj(x_scaled)
        all_spikes_list.append(gate_spikes)

        x_ssm_input = rearrange(x_scaled, 'b l d -> b d l')
        x_ssm_output_potential, ssm_proj_spikes = self.layer(x_ssm_input)
        all_spikes_list.append(ssm_proj_spikes)
        x_ssm_output_potential = rearrange(x_ssm_output_potential, 'b d l -> b l d')

        x_ssm_output_dropped = self.dropout1(x_ssm_output_potential)
        gated_ssm_output = gate_spikes * x_ssm_output_dropped

        # Post-SS4D Spike-Controlled Residual Gating (SCRG)
        ss4d_gate_potentials, s_ss4d_residual_gate = self.ss4d_residual_gate_proj(x_res1)
        all_spikes_list.append(s_ss4d_residual_gate)

        forget_gate_ss4d_cont, update_gate_ss4d_cont = torch.chunk(ss4d_gate_potentials, 2, dim=-1)

        forget_gate_ss4d_spikes = self.ss4d_residual_forget_act(forget_gate_ss4d_cont)
        update_gate_ss4d_spikes = self.ss4d_residual_update_act(update_gate_ss4d_cont)
        all_spikes_list.extend([forget_gate_ss4d_spikes, update_gate_ss4d_spikes])

        x = x_res1 * forget_gate_ss4d_spikes + gated_ssm_output * update_gate_ss4d_spikes

        # --- Simplified Enhanced Spiking Gated Feed-Forward Network (S-ESGFN) Sub-block ---
        if self.use_spiking_mlp:
            x_res2 = x
            x_norm2 = self.norm2(x) if self.prenorm else x

            # Dynamic Branch-Specific Thresholding (DBST) for S-ESGFN branches
            # These offsets are also combined with the R-ADPT global offset
            content_branch_vth_factor = torch.exp(rearrange(self.base_log_vth + self.ffn_content_log_vth_offset, 'd -> 1 1 d') + rearrange(activity_offset, 'b d -> b 1 d'))
            gate_branch_vth_factor = torch.exp(rearrange(self.base_log_vth + self.ffn_gate_log_vth_offset, 'd -> 1 1 d') + rearrange(activity_offset, 'b d -> b 1 d'))

            x_content_dpt = x_norm2 / content_branch_vth_factor
            x_gate_dpt = x_norm2 / gate_branch_vth_factor

            # S-ESGFN: Two parallel spiking branches
            content_potential, s_content_proj = self.ffn_content_proj(x_content_dpt)
            gate_potential, s_gate_proj = self.ffn_gate_proj(x_gate_dpt)

            content_potential_dropped = self.ffn_dropout_internal(content_potential)
            gate_potential_dropped = self.ffn_dropout_internal(gate_potential)

            content_spikes = self.spiking_ffn_content_act(content_potential_dropped)
            gate_spikes = self.spiking_ffn_gate_act(gate_potential_dropped)
            all_spikes_list.extend([s_content_proj, s_gate_proj, content_spikes, gate_spikes])

            # Direct multiplicative gating of spikes
            # The output of this operation is still a binary-like tensor (0 or 1 where content_spikes is 1 AND gate_spikes is 1)
            inter_spikes_gated = content_spikes * gate_spikes

            ffn_final_output_potential, s_down = self.ffn_down_proj(inter_spikes_gated)
            all_spikes_list.append(s_down)

            # Post-MLP Spike-Controlled Residual Gating (SCRG)
            mlp_gate_potentials, s_mlp_residual_gate = self.mlp_residual_gate_proj(x_res2)
            all_spikes_list.append(s_mlp_residual_gate)

            forget_gate_mlp_cont, update_gate_mlp_cont = torch.chunk(mlp_gate_potentials, 2, dim=-1)

            forget_gate_mlp_spikes = self.mlp_residual_forget_act(forget_gate_mlp_cont)
            update_gate_mlp_spikes = self.mlp_residual_update_act(update_gate_mlp_cont)
            all_spikes_list.extend([forget_gate_mlp_spikes, update_gate_mlp_spikes])

            output = x_res2 * forget_gate_mlp_spikes + ffn_final_output_potential * update_gate_mlp_spikes
        else:
            output = x

        if not self.prenorm:
            output = self.norm1(output)

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None

        reg_loss = None
        if final_spikes is not None and self.spike_loss_weight > 0.0:
            # Encourage sparsity by penalizing total spikes (mean spike activity)
            reg_loss = self.spike_loss_weight * final_spikes.mean()

        # Ensure final_spikes is binary float tensor
        if final_spikes is not None:
            final_spikes = (final_spikes > 0).float() # Ensure it's strictly binary 0/1

        return output, reg_loss, final_spikes