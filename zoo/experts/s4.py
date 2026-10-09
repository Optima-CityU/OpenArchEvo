import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Tuple, List, Optional

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
    from src.models.nn.dropout import DropoutNd
    registry = neuron_registry
except (ImportError, ModuleNotFoundError):
    # registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None
# --- End of Placeholders ---


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
        y = rearrange(y, "b c h l -> b (c h) l")
        if self.learnable_vth: y = y / torch.exp(self.ln_vth)

        spikes = self.neuron(y)
        y = self.dropout(spikes)
        y = self.output_linear(y)

        if not self.transposed: y = y.transpose(-1, -2)
        return y, spikes

# =============================================================================
# ===== NEW: Single-Layer SpikingSSM Block with SpikingFLABlock Interface =======
# =============================================================================

class SpikingFLABlock(nn.Module):
    """This class is a single layer/block that uses the Spiking State Space Model (SS4D)
    as its core. It is designed as a drop-in replacement for other blocks,
    adhering to the standard (pre-norm, residual) structure.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, use_spiking_mlp: Optional[bool]=None, d_ffn_mult: Optional[int]=None, neuron_type: Optional[str]=None, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model

        # Core layer components for a single block
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # The main SS4D layer
        # Extract layer-specific arguments for SS4D from kwargs
        ss4d_kwargs = kwargs.get('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        self.layer = SS4D(d_model, dropout=dropout, transposed=True, **ss4d_kwargs)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Matches the forward signature `(output, reg_loss, spikes)`.
        Input `x` is expected to be of shape (B, L, D).
        """
        x_res = x

        # Pre-normalization
        if self.prenorm:
            x = self.norm(x)
        x = x * self.gain
        # Transpose for Conv1d operations inside SS4D: (B, L, D) -> (B, D, L)
        x = x.transpose(-1, -2)
        # Apply the main SS4D layer
        x, spikes = self.layer(x)
        # Apply dropout
        x = self.dropout(x)
        # Transpose back to standard shape: (B, D, L) -> (B, L, D)
        x = x.transpose(-1, -2)
        # Add residual connection
        output = x_res + x

        # Post-normalization
        if not self.prenorm:
            output = self.norm(output)

        # Align with the target return format: (output, reg_loss, spikes)
        # We return 'spikes' in a list to be consistent with multi-layer spike collection
        reg_loss = None

        return output, reg_loss, spikes.flatten()