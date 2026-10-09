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
            return (x > 0).float()
    class DummyNeuron:
        def __init__(self, surrogate_function=None, **kwargs): pass
        def __call__(self, x): return (x > 0).float()

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

# ==================== New: SpikingPostNormActivation (from Version 1) ====================

class SpikingPostNormActivation(nn.Module):
    def __init__(self, neuron_type: str='sdn', surrogate_function=None):
        super().__init__()
        if surrogate_function is None:
            surrogate_function = piecewise_quadratic_surrogate()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_transposed = x.transpose(-1, -2)
        spikes_transposed = self.neuron(x_transposed)
        spikes = spikes_transposed.transpose(-1, -2)
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

        if self.learnable_vth: y = y / torch.exp(rearrange(self.ln_vth, 'd 1 -> 1 d 1') if self.ln_vth.dim() == 2 else rearrange(self.ln_vth, 'd -> 1 d 1'))

        B_dim, D_in_dim, L_dim = y.shape
        y_reshaped_for_linear = rearrange(y, 'b d l -> (b l) d')

        y_real_valued_output, ssm_proj_spikes = self.output_linear_proj(y_reshaped_for_linear)

        y_real_valued_output_dropped = self.dropout_layer(y_real_valued_output)

        y_final_ssm_output = rearrange(y_real_valued_output_dropped, '(b l) d -> b d l', b=B_dim, l=L_dim)

        if not self.transposed: y_final_ssm_output = rearrange(y_final_ssm_output, 'b d l -> b l d')

        return y_final_ssm_output, ssm_proj_spikes

# =============================================================================
# ===== SpikingFLABlock with Dynamic Potential Threshold (DPT) ============
# =============================================================================

class SpikingFLABlock(nn.Module):
    """SpikingFLABlock with Dynamic Potential Threshold (DPT) combining a Gated Spiking State Space Model (SS4D)
    with a Gated Spiking Feed-Forward Network (GSFFN).
    This architecture introduces a learnable, per-dimension dynamic threshold applied to the analog potentials
    before feeding them into spiking linear layers, aiming to regulate neuronal excitability and firing rates
    across the block. All linear layers operate on binary spikes, leveraging SpikingLinear for
    analog-to-spike conversion or direct processing of binary spike inputs.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, d_ffn_mult: int=2, neuron_type: str='sdn', use_spiking_mlp: bool=True, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.use_spiking_mlp = use_spiking_mlp
        self.d_ffn = d_model * d_ffn_mult

        surrogate_function = piecewise_quadratic_surrogate()

        # Dynamic Potential Threshold (DPT) for this block
        # Learnable log-threshold, applied per feature dimension to control excitability
        self.block_log_vth = nn.Parameter(torch.zeros(d_model))

        # --- Gated SS4D Sub-block Components ---
        self.norm1 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) # Dropout for SS4D output branch

        ss4d_kwargs = kwargs.pop('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs['neuron'] = neuron_type
        self.layer = SS4D(d_model, transposed=True, dropout=dropout, **ss4d_kwargs)

        self.norm_gate = RMSNorm(d_model)
        self.gate_proj = SpikingLinear(
            nn.Linear(d_model, d_model, bias=False),
            neuron_type,
            surrogate_function
        )
        self.gain = nn.Parameter(torch.ones(1))

        # --- Gated Spiking Feed-Forward Network (GSFFN) Sub-block Components ---
        if self.use_spiking_mlp:
            self.norm2 = RMSNorm(d_model)
            self.dropout2 = nn.Dropout(dropout) # Dropout for FFN down-projection

            # Gated FFN components, all using SpikingLinear
            self.ffn_up_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_gate_proj = SpikingLinear(
                nn.Linear(d_model, self.d_ffn, bias=False),
                neuron_type,
                surrogate_function
            )
            # Spiking activations for FFN internal gating to ensure binary signals for multiplication
            self.spiking_ffn_gate_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)
            self.spiking_ffn_up_act = SpikingPostNormActivation(neuron_type=neuron_type, surrogate_function=surrogate_function)

            self.ffn_down_proj = SpikingLinear(
                nn.Linear(self.d_ffn, d_model, bias=False),
                neuron_type,
                surrogate_function
            )
            self.ffn_dropout = nn.Dropout(dropout) # Dropout for FFN internal potentials before spiking

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
        all_spikes_list: List[torch.Tensor] = []

        # Apply Dynamic Potential Threshold (DPT) after initial normalization/scaling
        # and before any subsequent spiking operations within the block.
        # This scales the potentials, making neurons harder/easier to spike.
        block_vth_factor = torch.exp(rearrange(self.block_log_vth, 'd -> 1 1 d'))

        # --- Gated SS4D Sub-block ---
        x_res1 = x
        x_norm1 = self.norm1(x) if self.prenorm else x
        x_scaled = x_norm1 * self.gain

        # Apply DPT to the SS4D input potential
        x_scaled_dpt = x_scaled / block_vth_factor

        # Gating Pathway for SS4D output
        # Apply DPT to the gate input potential
        gate_input_potential_dpt = self.norm_gate(x_scaled) / block_vth_factor
        gate_raw_potential, gate_spikes = self.gate_proj(gate_input_potential_dpt)
        all_spikes_list.append(gate_spikes)

        # SS4D computation
        x_ssm_input = rearrange(x_scaled_dpt, 'b l d -> b d l')
        x_ssm_output_real, ssm_proj_spikes = self.layer(x_ssm_input)
        all_spikes_list.append(ssm_proj_spikes)
        x_ssm_output_real = rearrange(x_ssm_output_real, 'b d l -> b l d')

        # Apply dropout to the main SS4D output branch (analog potential)
        x_ssm_output_dropped = self.dropout1(x_ssm_output_real)

        # Apply the learned gate (binary spikes) to the SS4D output (analog potential)
        gated_ssm_output = gate_spikes * x_ssm_output_dropped

        # First residual connection
        x = x_res1 + gated_ssm_output

        # --- Gated Spiking Feed-Forward Network (GSFFN) Sub-block ---
        if self.use_spiking_mlp:
            x_res2 = x
            x_norm2 = self.norm2(x) if self.prenorm else x

            # Apply DPT to the FFN input potential
            x_norm2_dpt = x_norm2 / block_vth_factor

            # FFN up-projection path
            ffn_up_potential, s_up = self.ffn_up_proj(x_norm2_dpt)
            all_spikes_list.append(s_up)

            # FFN gate path
            ffn_gate_potential, s_gate = self.ffn_gate_proj(x_norm2_dpt)
            all_spikes_list.append(s_gate)

            # Apply dropout to continuous potentials before spiking
            ffn_up_potential_dropped = self.ffn_dropout(ffn_up_potential)
            ffn_gate_potential_dropped = self.ffn_dropout(ffn_gate_potential)

            # Convert potentials to spikes to enable binary multiplication
            gated_ffn_spikes = self.spiking_ffn_gate_act(ffn_gate_potential_dropped) # Analog to spikes
            up_ffn_spikes = self.spiking_ffn_up_act(ffn_up_potential_dropped)       # Analog to spikes

            # Element-wise multiplication of binary spikes results in binary spikes
            inter_spikes = gated_ffn_spikes * up_ffn_spikes
            all_spikes_list.extend([gated_ffn_spikes, up_ffn_spikes]) # Collect the intermediate spikes

            # FFN down-projection. Input to this SpikingLinear is now binary spikes.
            ffn_final_output, s_down = self.ffn_down_proj(inter_spikes)
            all_spikes_list.append(s_down)

            # Second residual connection
            output = x_res2 + ffn_final_output
        else:
            output = x

        # Post-normalization if not pre-norm
        if not self.prenorm:
            output = self.norm1(output)

        # Collect all spikes into a single flattened tensor
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else None
        reg_loss = None # Keep for compatibility

        return output, reg_loss, final_spikes