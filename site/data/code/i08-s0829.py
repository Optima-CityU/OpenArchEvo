import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Tuple, List, Optional, Dict

# --- Start of required SpikingLinear and dummy imports ---

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
    from src.models.nn.dropout import DropoutNd
    registry = neuron_registry
except (ImportError, ModuleNotFoundError):
    # Dummy placeholders for when src imports are not available
    print("Warning: Using placeholder SNN modules.")
    class DummySurrogate:
        @staticmethod
        def apply(x, *args, **kwargs):
            # Simple threshold for dummy spike generation
            return (x > 0).float() 

    def piecewise_quadratic_surrogate():
        return DummySurrogate

    registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}

    class DropoutNd(nn.Identity):
        pass

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        # Ensure neuron_registry is available from the context
        if 'registry' not in globals() or registry is None:
            raise ImportError("Spiking neuron registry is not available.")
        self.neuron = registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spikes = self.neuron(x) 
        # Apply linear layer to binary spikes, return continuous current and spikes
        return self.linear(spikes), spikes

# --- End of required SpikingLinear and dummy imports ---


# =============================================================================
# ===== Start of SpikingSSM Components (S4DKernel, SS4D) (FROM VERSION 2, with fix) =====
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
        self.channels = channels # Store channels for correct output_linear dimension
        self.D = nn.Parameter(torch.randn(channels, self.h))
        if learnable_vth:
            self.ln_vth = nn.Parameter(torch.zeros(d_model, 1) if not shared_vth else torch.zeros(1))
        if bidirectional: channels *= 2
        self.kernel = S4DKernel(self.h, N=self.n, channels=channels, **kernel_args)
        self.neuron = registry[neuron](surrogate_function=piecewise_quadratic_surrogate())
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        # FIX: Adjusted output_linear input dimension to match (C*H, L) input from rearrange
        self.output_linear = nn.Sequential(nn.Conv1d(self.h * self.channels, 2 * self.h * self.channels, kernel_size=1), nn.GLU(dim=-2))

    def forward(self, u, **kwargs):
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        k, _ = self.kernel(L=L)
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)

        # Ensure proper broadcasting for u_f * k_f.
        # u_f: (B, H, L//2+1), k_f: (C, H, L//2+1)
        # Result should be (B, C, H, L//2+1)
        y_f = u_f.unsqueeze(1) * k_f.unsqueeze(0) 

        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]

        # Ensure proper broadcasting for y + D * u_expanded
        # y: (B, C, H, L), D: (C, H), u: (B, H, L)
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * u.unsqueeze(1)

        y = rearrange(y, "b c h l -> b (c h) l") # y is now (B, C*H, L)
        if self.learnable_vth: y = y / torch.exp(self.ln_vth)

        spikes = self.neuron(y) # Spikes are generated here, shape (B, C*H, L)
        y_dropout_spikes = self.dropout(spikes) # Dropout on spikes
        y = self.output_linear(y_dropout_spikes) # output_linear takes spikes as input

        if not self.transposed: y = y.transpose(-1, -2)
        return y, spikes # y is continuous, spikes are binary

# =============================================================================
# ===== NEW: SpikingFLABlock (Evolution with Inlined Spiking FFN & Spike-Gated FFN) =========
# =============================================================================

class SpikingFLABlock(nn.Module):
    """This class implements a SpikingFLABlock with a Spiking State Space Model (SS4D)
    backbone followed by an inlined, spike-gated Spiking Feedforward Network (FFN).
    It features explicit spiking gating mechanisms for both SS4D and FFN pathways,
    collects all generated spikes, and maintains a continuous residual stream.
    Spike activity regularization is reintroduced for better SNN efficiency.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, use_spiking_mlp: Optional[bool]=True, d_ffn_mult: Optional[int]=4, neuron_type: Optional[str]='sdn', spike_reg_weight: float=0.01, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.neuron_type = neuron_type
        # Ensure use_spiking_mlp is enabled by default as per requirements.
        # Renamed to use_spiking_ffn for clarity, but keeps the behavior.
        self.use_spiking_ffn = use_spiking_mlp if use_spiking_mlp is not None else True
        self.surrogate_function = piecewise_quadratic_surrogate()
        self.spike_reg_weight = spike_reg_weight # Reintroduced regularization weight

        # --- SS4D Branch Components ---
        self.norm1_ss4d = nn.LayerNorm(d_model) # Applied if prenorm
        self.dropout_ss4d_out = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        ss4d_kwargs = kwargs.get('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs.setdefault('neuron', neuron_type)
        self.ss4d_module = SS4D(d_model, dropout=dropout, transposed=True, **ss4d_kwargs)

        # Spiking Gating Mechanism for SS4D branch input
        # The gate itself will be a continuous projection from input, then spiked.
        self.ss4d_gate_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, self.surrogate_function)
        self.dropout_ss4d_gate_proj = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # --- Inlined Spiking Feedforward Network (FFN) Components ---
        if self.use_spiking_ffn:
            self.norm_ffn = nn.LayerNorm(d_model) # Applied if prenorm

            # Spiking Gating Mechanism for FFN input, now directly using spikes for gating
            self.ffn_gate_proj = SpikingLinear(nn.Linear(d_model, d_model), neuron_type, self.surrogate_function)
            self.dropout_ffn_gate_proj = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

            d_ffn = d_model * (d_ffn_mult if d_ffn_mult is not None else 4)
            self.ffn_expand = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, self.surrogate_function)
            self.ffn_contract = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, self.surrogate_function)
            self.dropout_ffn_out = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # Final normalization for post-norm architecture (if applicable)
        self.norm_final = nn.LayerNorm(d_model) if not prenorm else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Matches the forward signature `(output, reg_loss, spikes)`.
        Input `x` is expected to be of shape (B, L, D).
        The main residual stream is maintained as a continuous signal.
        """
        x_res = x  # Initial continuous residual stream
        all_spikes_list = []

        # ==================== SS4D Branch ====================
        # Pre-normalization for SS4D
        x_ss4d_input_prenormed = self.norm1_ss4d(x_res) if self.prenorm else x_res

        # Spiking Gating for SS4D input
        ss4d_gate_current, ss4d_gate_spikes = self.ss4d_gate_proj(x_ss4d_input_prenormed)
        all_spikes_list.append(ss4d_gate_spikes)
        dropped_ss4d_gate_current = self.dropout_ss4d_gate_proj(ss4d_gate_current)
        # Using sigmoid on continuous current for SS4D gating factor, consistent with V2
        ss4d_gate_factor = torch.sigmoid(dropped_ss4d_gate_current) 
        x_ss4d_gated_input = x_ss4d_input_prenormed * ss4d_gate_factor

        # Transpose for SS4D (expects B, D, L)
        x_ss4d_gated_input_transposed = rearrange(x_ss4d_gated_input, 'b l d -> b d l')

        # Apply SS4D module
        ss4d_output_continuous_transposed, spikes_ss4d = self.ss4d_module(x_ss4d_gated_input_transposed)
        all_spikes_list.append(spikes_ss4d)

        # Transpose back (to B, L, D)
        ss4d_output_continuous = rearrange(ss4d_output_continuous_transposed, 'b d l -> b l d')

        # Apply dropout on the continuous output of SS4D
        ss4d_output_continuous = self.dropout_ss4d_out(ss4d_output_continuous)

        # Add SS4D output to the residual stream
        x_after_ss4d_residual = x_res + ss4d_output_continuous
        current_output_stream = x_after_ss4d_residual

        # ==================== Inlined Spiking Feedforward Network (FFN) Branch ====================
        if self.use_spiking_ffn:
            # Pre-normalization for FFN
            x_ffn_input_prenormed = self.norm_ffn(current_output_stream) if self.prenorm else current_output_stream

            # Spiking Gating for FFN input, now directly using binary spikes
            ffn_gate_current, ffn_gate_spikes = self.ffn_gate_proj(x_ffn_input_prenormed)
            all_spikes_list.append(ffn_gate_spikes)
            # Dropout is applied to the continuous current before it's potentially used or for conceptual consistency
            dropped_ffn_gate_current = self.dropout_ffn_gate_proj(ffn_gate_current) 

            # Use the actual binary spikes for gating the FFN pathway
            # This makes the FFN computation conditional on the FFN gate spikes
            x_ffn_gated_input = x_ffn_input_prenormed * ffn_gate_spikes.float()

            # FFN Core: Expansion via SpikingLinear
            expanded_current, expanded_spikes = self.ffn_expand(x_ffn_gated_input) # Input is now spike-gated
            all_spikes_list.append(expanded_spikes)

            # FFN Core: Contraction via SpikingLinear
            contracted_current, contracted_spikes = self.ffn_contract(expanded_current)
            all_spikes_list.append(contracted_spikes)

            # Apply dropout on the continuous output of FFN
            ffn_output_continuous = self.dropout_ffn_out(contracted_current)

            # Add FFN output to the residual stream
            current_output_stream = current_output_stream + ffn_output_continuous

        # ==================== Finalization ====================
        # Apply final post-normalization if not pre-normed
        output = self.norm_final(current_output_stream) if not self.prenorm else current_output_stream

        # Concatenate all collected binary spike tensors into a single flattened tensor.
        # Filter out None values in case some spike tensors are conditionally produced
        final_spikes = torch.cat([s.flatten() for s in all_spikes_list if s is not None])

        # Calculate spike activity regularization for the entire block
        reg_loss = None 
        if self.spike_reg_weight > 0 and final_spikes.numel() > 0:
            # Target a sparse mean firing rate (e.g., 0.1) to encourage sparsity
            target_mean_activity = 0.1 
            current_mean_activity = torch.mean(final_spikes.float())
            # Use L2 regularization to encourage activity close to the target
            reg_loss = self.spike_reg_weight * ((current_mean_activity - target_mean_activity) ** 2)

        return output, reg_loss, final_spikes