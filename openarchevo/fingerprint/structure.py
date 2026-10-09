"""Fourteen module and structural statistics of a single block (requires PyTorch).

One block (``d_model=128``) is run on a fixed-shape input while forward hooks
record every leaf module's input/output shapes, parameters and MACs:

* module statistics: parameter count, ``SpikingLinear`` count, Conv1d count,
  fraction of spiking projections whose name contains ``gate``;
* dimension flow over Linear/Conv layers in execution order: depth (number of
  such layers), expansion / contraction steps, bottleneck ratio, largest
  expansion factor;
* FFN statistics from expand -> contract patterns: mean expansion ratio and the
  FFN share of parameters and MACs;
* compute density (MACs per parameter) and the size of the largest
  ``torch.compile`` (TorchDynamo) subgraph relative to all captured graphs.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

_TRANSFORMS = (nn.Linear, nn.Conv1d)


def _leaf_macs(module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> int:
    if isinstance(module, nn.Linear):
        return inp[..., 0].numel() * module.in_features * module.out_features
    if isinstance(module, nn.Conv1d):
        k = module.kernel_size[0]
        return out.numel() * k * (module.in_channels // module.groups)
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
        return 2 * out.numel()
    if isinstance(module, (nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.ReLU)):
        return out.numel()
    return 0


def _trace(block: nn.Module, x: torch.Tensor) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def hook(name: str):
        def fn(module: nn.Module, args: tuple, output: Any) -> None:
            inp = args[0] if args and isinstance(args[0], torch.Tensor) else None
            out = output[0] if isinstance(output, tuple) else output
            if inp is None or not isinstance(out, torch.Tensor):
                return
            records.append({"name": name, "module": module, "in": tuple(inp.shape), "out": tuple(out.shape),
                            "params": sum(p.numel() for p in module.parameters()),
                            "macs": _leaf_macs(module, inp, out)})
        return fn

    hooks = [m.register_forward_hook(hook(n)) for n, m in block.named_modules()
             if n and not list(m.children()) and not isinstance(m, torch.jit.ScriptModule)]
    try:
        with torch.no_grad():
            block(x)
    finally:
        for h in hooks:
            h.remove()
    return records


def _largest_subgraph_ratio(block: nn.Module, x: torch.Tensor) -> float:
    sizes: list[int] = []

    def backend(gm: torch.fx.GraphModule, _inputs: Any):
        sizes.append(len(gm.graph.nodes))
        return gm.forward

    try:
        torch._dynamo.reset()
        with torch.no_grad():
            torch.compile(block, backend=backend, fullgraph=False)(x)
    except Exception:
        return 0.0
    return max(sizes) / sum(sizes) if sizes else 0.0


def _ffn_layers(seq: list[dict[str, Any]]) -> tuple[list[float], set[int]]:
    """Find expand -> (same-width)* -> contract patterns within a window of 5 layers."""
    ratios, members, i = [], set(), 0
    while i < len(seq):
        d_in, d_out = seq[i]["d_in"], seq[i]["d_out"]
        if d_out <= 1.1 * d_in:
            i += 1
            continue
        group = [i]
        for j in range(i + 1, min(i + 5, len(seq))):
            dj_in, dj_out = seq[j]["d_in"], seq[j]["d_out"]
            if dj_in == d_out and dj_out < 0.85 * dj_in:
                ratios.append(d_out / d_in)
                members.update(group + [j])
                i = j
                break
            if (dj_in, dj_out) in {(d_in, d_out), (d_out, d_out)}:
                group.append(j)
            else:
                break
        i += 1
    return ratios, members


def structural_features(block: nn.Module, x: torch.Tensor) -> dict[str, float]:
    """Compute the 14 ``arch_*`` features of ``block`` on input ``x`` of shape ``(1, L, d_model)``."""
    block.eval()
    d_model = x.shape[-1]
    named = dict(block.named_modules())
    spiking = [n for n, m in named.items() if n and hasattr(m, "neuron") and hasattr(m, "linear")]
    n_gate = sum("gate" in n.lower() for n in spiking)
    n_conv1d = sum(isinstance(m, nn.Conv1d) for m in named.values())

    records = _trace(block, x)
    total_params = sum(p.numel() for p in block.parameters())
    total_macs = sum(r["macs"] for r in records)

    seq = []
    for r in records:
        if not isinstance(r["module"], _TRANSFORMS):
            continue
        conv = isinstance(r["module"], nn.Conv1d) and len(r["out"]) >= 3
        seq.append({"d_in": r["in"][1] if conv else r["in"][-1], "d_out": r["out"][1] if conv else r["out"][-1],
                    "params": r["params"], "macs": r["macs"]})
    expand = [s["d_out"] / s["d_in"] for s in seq if s["d_out"] > s["d_in"]]
    ffn_ratios, ffn_idx = _ffn_layers(seq)

    return {
        "arch_total_params": float(total_params),
        "arch_depth_proxy": float(len(seq)),
        "arch_num_spiking_linear": float(len(spiking)),
        "arch_num_conv1d": float(n_conv1d),
        "arch_gate_layer_ratio": min(n_gate / max(len(spiking), 1), 1.0),
        "arch_num_expand": float(len(expand)),
        "arch_num_contract": float(sum(s["d_out"] < s["d_in"] for s in seq)),
        "arch_bottleneck_ratio": min((s["d_out"] for s in seq), default=d_model) / d_model,
        "arch_max_expand_ratio": max(expand, default=1.0),
        "arch_avg_ffn_expansion": sum(ffn_ratios) / len(ffn_ratios) if ffn_ratios else 1.0,
        "arch_ffn_param_ratio": min(sum(seq[i]["params"] for i in ffn_idx) / (total_params + 1e-8), 1.0),
        "arch_ffn_macs_ratio": min(sum(seq[i]["macs"] for i in ffn_idx) / (total_macs + 1e-8), 1.0),
        "arch_macs_per_param": total_macs / (total_params + 1e-8),
        "arch_largest_subgraph_ratio": _largest_subgraph_ratio(block, x),
    }
