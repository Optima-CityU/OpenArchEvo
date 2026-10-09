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
    from src.models.nn.dropout import DropoutNd
    from src.models.spike.nn import SpikingLinear
    # From fla library, consistent with Version 1 LightNet
    from fla.modules import RMSNorm 
    registry = neuron_registry
except (ImportError, ModuleNotFoundError):
    print("Warning: Using placeholder SNN modules and RMSNorm.")
    class DummySurrogate:
        @staticmethod
        def apply(x, *args, **kwargs):
            return (x > 0).float()
    class DummyNeuron(nn.Module):
        def __init__(self, surrogate_function=None, **kwargs):
            super().__init__()
            self.surrogate_function = surrogate_function if surrogate_function else DummySurrogate.apply
        def forward(self, x):
            # Ensure output is binary for the dummy
            return (self.surrogate_function(x) > 0).float().to(x.dtype)
    class DummySpikingLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=True, neuron_type='sdn', surrogate_function=None, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.neuron = DummyNeuron(surrogate_function=surrogate_function) # Internal neuron for output spikes

        def forward(self, x_binary): # Input to SpikingLinear is expected to be binary spikes
            # Ensure input to self.linear is float, as spikes are float 0/1
            out_continuous = self.linear(x_binary.float()) 
            spikes = self.neuron(out_continuous) # Spikes from the neuron AFTER the linear layer
            return out_continuous, spikes # output is analog, spikes are binary
    neuron_registry = {'sdn': lambda surrogate_function: DummyNeuron(surrogate_function)}
    piecewise_quadratic_surrogate = lambda: None
    SpikingLinear = DummySpikingLinear # Use dummy for local testing
    registry = neuron_registry
    DropoutNd = nn.Dropout # Fallback for DropoutNd

    # Dummy RMSNorm if fla.modules is not available
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        def forward(self, x):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


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

        d_ffn_mult_ss4d = 2 
        self.d_ffn_ss4d = int(self.h * d_ffn_mult_ss4d)

        self.output_spike_linear1 = SpikingLinear(channels * self.h, self.d_ffn_ss4d, bias=True, neuron_type=neuron, surrogate_function=piecewise_quadratic_surrogate())
        self.output_neuron2 = registry[neuron](surrogate_function=piecewise_quadratic_surrogate()) 
        self.output_spike_linear2 = SpikingLinear(self.d_ffn_ss4d, channels * self.h, bias=True, neuron_type=neuron, surrogate_function=piecewise_quadratic_surrogate())

    def forward(self, u, **kwargs):
        if not self.transposed: u = u.transpose(-1, -2) # u: (B, L, H) -> (B, H, L)
        L = u.size(-1)
        k, _ = self.kernel(L=L) # k is continuous kernel

        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))

        # k is continuous, u is binary (from SpikingFLABlock's pre-neuron)
        k_f = torch.fft.rfft(k.float(), n=2 * L) 
        u_f = torch.fft.rfft(u.float(), n=2 * L) # FFT on binary spikes

        # Frequency domain multiplication (element-wise for convolution)
        # u_f: (B, H, L_freq), k_f: (C, H, L_freq)
        # Output: (B, C, H, L_freq) as h is common and not summed over
        y_conv = torch.einsum("bhl,chl->bchl", u_f, k_f)
        y_conv = torch.fft.irfft(y_conv, n=2 * L)[..., :L]

        # The D term operates on the binary input 'u'
        # u: (B, H, L), self.D: (C, H)
        # einsum("bhl,ch->bchl", u, self.D) performs u[b,h,l] * D[c,h] element-wise, resulting in (B,C,H,L)
        # This acts as a linear projection on the binary input 'u' but keeps the d_model dimension for subsequent sum
        y_direct = torch.einsum("bhl,ch->bchl", u, self.D) 

        y = y_conv + y_direct # y is now continuous

        y = rearrange(y, "b c h l -> b (c h) l") # y is still continuous

        if self.learnable_vth: y = y / torch.exp(self.ln_vth)

        # First internal neuron in SS4D, converts continuous 'y' to spikes
        spikes_from_ss4d_core = self.neuron(y)
        all_ss4d_spikes = [spikes_from_ss4d_core.flatten()]

        # dropout is applied on binary spikes, which is fine
        y_for_output_linear = self.dropout(spikes_from_ss4d_core)

        b, ch, l = y_for_output_linear.shape

        # Rearrange for linear layer, input must be binary spikes (already is)
        x_reshaped_for_linear = rearrange(y_for_output_linear, 'b ch l -> (b l) ch')

        # SpikingLinear 1: receives binary, outputs continuous and spikes
        x_out1_continuous, spikes_out1 = self.output_spike_linear1(x_reshaped_for_linear)
        all_ss4d_spikes.append(spikes_out1.flatten())

        # Internal neuron 2: converts continuous to spikes
        x_out1_spikes = self.output_neuron2(x_out1_continuous) 
        all_ss4d_spikes.append(x_out1_spikes.flatten())

        # SpikingLinear 2: receives binary, outputs continuous and spikes
        x_out2_continuous, spikes_out2 = self.output_spike_linear2(x_out1_spikes)
        all_ss4d_spikes.append(spikes_out2.flatten())

        y = rearrange(x_out2_continuous, '(b l) ch -> b ch l', b=b, l=l)

        if not self.transposed: y = y.transpose(-1, -2)

        return y, torch.cat(all_ss4d_spikes)

# =============================================================================
# ===== NEW: Single-Layer SpikingSSM Block with SpikingFLABlock Interface =======
# =============================================================================

class SpikingFLABlock(nn.Module):
    """This class is a single layer/block that uses the Spiking State Space Model (SS4D)
    as its core, enhanced with a Spiking Gated Feed-Forward Network (Gated SNN-FFN)
    and a Spiking Input Gate for the SS4D block.
    It is designed as a drop-in replacement for other blocks, adhering to the standard
    (pre-norm, residual) structure. All internal linear layers are replaced with
    SpikingLinear and strictly operate on binary signals produced by spike neurons.
    
    Evolution changes from Version 2:
    - Introduced a `ss4d_pre_neuron` to explicitly convert the input to the SS4D block
      into binary spikes, ensuring all linear-like operations within SS4D (like the D term)
      receive binary inputs, consistent with SNN principles.
    - Verified and ensured all `SpikingLinear` layers receive binary spike inputs.
    - **New in Version 3: Spike-driven SS4D input gating.** The continuous `sigmoid`-based
      modulation factor for the SS4D input has been replaced with a direct element-wise
      multiplication by binary spikes generated from the gating path. This makes the gate
      fully neuromorphic and more computationally efficient.
    """

    def __init__(self, d_model: int, dropout: float=0.1, prenorm: bool=True, num_heads: Optional[int]=None, use_spiking_mlp: Optional[bool]=None, d_ffn_mult: Optional[int]=None, neuron_type: Optional[str]=None, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        self.neuron_type = neuron_type if neuron_type is not None else 'sdn'

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout) 

        # New: Spiking Input Gate for SS4D
        self.use_ss4d_input_gate = kwargs.pop('use_ss4d_input_gate', True)
        if self.use_ss4d_input_gate:
            self.ss4d_gate_pre_neuron = registry[self.neuron_type](surrogate_function=piecewise_quadratic_surrogate())
            # SpikingLinear's internal neuron generates spikes for modulation
            self.ss4d_gate_linear = SpikingLinear(d_model, d_model, bias=True, neuron_type=self.neuron_type, surrogate_function=piecewise_quadratic_surrogate())
            # Removed self.ss4d_gate_scale as it's no longer needed for binary modulation

        # NEW: Neuron to convert input to SS4D block into spikes
        # This ensures the 'u' argument of SS4D.forward() is always binary
        self.ss4d_pre_neuron = registry[self.neuron_type](surrogate_function=piecewise_quadratic_surrogate())

        ss4d_kwargs = kwargs.get('layer', {})
        ss4d_kwargs.setdefault('lr', kwargs.get('lr', 0.001))
        ss4d_kwargs.setdefault('neuron', self.neuron_type)

        self.layer = SS4D(d_model, dropout=dropout, transposed=True, **ss4d_kwargs)
        self.gain = nn.Parameter(torch.ones(1)) # Keep for initial scaling of the normalized input

        self.use_spiking_mlp = use_spiking_mlp if use_spiking_mlp is not None else True
        if self.use_spiking_mlp:
            self.d_ffn_mult = d_ffn_mult if d_ffn_mult is not None else 4
            self.d_ffn = int(d_model * self.d_ffn_mult)

            self.mlp_norm = RMSNorm(d_model)

            self.mlp_pre_gate_neuron = registry[self.neuron_type](surrogate_function=piecewise_quadratic_surrogate())

            self.mlp_linear_value = SpikingLinear(d_model, self.d_ffn, bias=True, neuron_type=self.neuron_type, surrogate_function=piecewise_quadratic_surrogate())
            self.mlp_linear_gate = SpikingLinear(d_model, self.d_ffn, bias=True, neuron_type=self.neuron_type, surrogate_function=piecewise_quadratic_surrogate())

            self.mlp_linear_out = SpikingLinear(self.d_ffn, d_model, bias=True, neuron_type=self.neuron_type, surrogate_function=piecewise_quadratic_surrogate())
            self.mlp_dropout_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        all_spikes_list = []
        x_res = x

        if self.prenorm:
            x_normalized = self.norm(x)
        else:
            x_normalized = x

        x_scaled = x_normalized * self.gain

        if self.use_ss4d_input_gate:
            # Input to neuron can be continuous, output must be binary spikes
            gate_input_spikes = self.ss4d_gate_pre_neuron(x_normalized)
            all_spikes_list.append(gate_input_spikes.flatten())

            # SpikingLinear now processes binary spikes. We need the spikes *from* this layer's neuron
            # to perform direct binary modulation.
            # The `SpikingLinear` returns (analog_output, binary_spikes_from_internal_neuron).
            _, ss4d_gate_spikes_for_modulation = self.ss4d_gate_linear(gate_input_spikes)
            all_spikes_list.append(ss4d_gate_spikes_for_modulation.flatten())

            # Direct binary modulation: element-wise multiplication with binary spikes
            # This acts as an ON/OFF switch for the input features based on spike activity.
            x_scaled = x_scaled * ss4d_gate_spikes_for_modulation 

        # NEW: Convert the (potentially modulated) input to SS4D into binary spikes
        x_ss4d_pre_neuron_spikes = self.ss4d_pre_neuron(x_scaled)
        all_spikes_list.append(x_ss4d_pre_neuron_spikes.flatten())

        x_ss4d_input_for_layer = rearrange(x_ss4d_pre_neuron_spikes, 'b l d -> b d l')

        # The SS4D block now receives binary spikes as its primary input 'u'
        x_ss4d_output_continuous, ss4d_spikes = self.layer(x_ss4d_input_for_layer)
        all_spikes_list.append(ss4d_spikes)

        x_ss4d_output = rearrange(x_ss4d_output_continuous, 'b d l -> b l d')

        output = x_res + self.dropout(x_ss4d_output)

        if self.use_spiking_mlp:
            x_mlp_res = output

            x_mlp_normalized = self.mlp_norm(output)

            # Input to neuron can be continuous, output must be binary spikes
            mlp_pre_gate_spikes = self.mlp_pre_gate_neuron(x_mlp_normalized)
            all_spikes_list.append(mlp_pre_gate_spikes.flatten())

            # Input to SpikingLinear must be binary spikes
            val_out_continuous, val_spikes = self.mlp_linear_value(mlp_pre_gate_spikes)
            all_spikes_list.append(val_spikes.flatten())

            # Input to SpikingLinear must be binary spikes
            gate_out_continuous, gate_spikes = self.mlp_linear_gate(mlp_pre_gate_spikes)
            all_spikes_list.append(gate_spikes.flatten())

            # Product of binary spikes is also binary
            gated_binary_input_to_final_linear = val_spikes * gate_spikes

            # Input to SpikingLinear must be binary spikes
            final_mlp_out_continuous, final_mlp_spikes = self.mlp_linear_out(gated_binary_input_to_final_linear)
            all_spikes_list.append(final_mlp_spikes.flatten())
            final_mlp_out_continuous = self.mlp_dropout_out(final_mlp_out_continuous)

            output = x_mlp_res + final_mlp_out_continuous

        if not self.prenorm:
            # This applies norm after the residual connection, if prenorm is False
            output = self.norm(output)

        reg_loss = None
        # Ensure final_spikes is a single concatenated tensor
        final_spikes = torch.cat(all_spikes_list)

        return output, reg_loss, final_spikes