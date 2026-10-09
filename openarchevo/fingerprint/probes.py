"""Seven initialization-time probes on the wrapped model (requires PyTorch).

All probes use one fixed ListOps-style minibatch, a fixed initialization seed and
the fixed wrapper, so differences come from the block alone:

* SynFlow, GradNorm, SNIP, Jacobian covariance (condition number), NASWOT: the
  usual zero-cost proxies, adapted to the ``(logits, reg_loss, spikes)`` wrapper.
* SWSP: number of distinct *sample-wise* binary patterns over all recorded
  neuron x position responses (SWAP adapted to spikes).
* FireRate: mean spike activity across the blocks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from openarchevo.models.wrapper import SequenceModel


def synflow(model: SequenceModel, tokens: torch.Tensor) -> float:
    signs = {}
    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data.abs_()
    model.zero_grad()
    logits, _, _ = model(tokens)
    logits.sum().backward()
    score = sum((p.grad * p).abs().sum().item() for p in model.parameters() if p.grad is not None)
    for name, p in model.named_parameters():
        p.data.mul_(signs[name])
    model.zero_grad()
    return score


def gradnorm_and_snip(model: SequenceModel, tokens: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    model.zero_grad()
    logits, _, _ = model(tokens)
    F.cross_entropy(logits, targets).backward()
    grads = [(p, p.grad) for p in model.parameters() if p.grad is not None]
    gradnorm = sum(g.pow(2).sum().item() for _, g in grads) ** 0.5
    snip = sum((g * p).abs().sum().item() for p, g in grads)
    model.zero_grad()
    return gradnorm, snip


def jacob_cov(model: SequenceModel, tokens: torch.Tensor) -> float:
    """Condition number of the input-output Jacobian for one sample."""
    emb = model.embedding(tokens[:1]).detach().requires_grad_(True)

    def head_of(x: torch.Tensor) -> torch.Tensor:
        for block in model.blocks:
            x = block(x)[0]
        return model.head(model.norm(x).mean(dim=1))

    jac = torch.autograd.functional.jacobian(head_of, emb).reshape(model.head.out_features, -1)
    s = torch.linalg.svdvals(jac)
    s_min = s[s > 0].min() if torch.any(s > 0) else torch.tensor(1e-6)
    return (s.max() / (s_min + 1e-6)).item()


@torch.no_grad()
def naswot(model: SequenceModel, tokens: torch.Tensor) -> float:
    model(tokens)
    feats = model.features.mean(dim=1)
    kernel = feats @ feats.t() + 1e-5 * torch.eye(len(feats), device=feats.device)
    return torch.slogdet(kernel)[1].item()


def _is_neuron(module: nn.Module) -> bool:
    name = type(module).__name__.lower()
    return "neuron" in name or "lif" in name or "sdn" in name


@torch.no_grad()
def swsp(model: SequenceModel, tokens: torch.Tensor) -> float:
    """Distinct sample-wise spike patterns over all neuron outputs."""
    outputs: list[torch.Tensor] = []
    hooks = [m.register_forward_hook(lambda _m, _i, o: outputs.append((o[0] if isinstance(o, tuple) else o).detach()))
             for m in model.modules() if _is_neuron(m)]
    try:
        model(tokens)
    finally:
        for h in hooks:
            h.remove()
    if not outputs:
        return 0.0
    patterns = torch.cat([o.reshape(tokens.size(0), -1) for o in outputs], dim=1)  # (samples, responses)
    return float(torch.unique(torch.sign(patterns).T, dim=0).size(0))


@torch.no_grad()
def fire_rate(model: SequenceModel, tokens: torch.Tensor) -> float:
    _, _, spikes = model(tokens)
    if not spikes:
        raise ValueError("The block returned no spikes.")
    return torch.stack([s.float().mean() for s in spikes]).mean().item()


def compute_probes(model: SequenceModel, tokens: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    model.eval()
    gradnorm, snip = gradnorm_and_snip(model, tokens, targets)
    return {
        "zcp_synflow": synflow(model, tokens[:1]),
        "zcp_gradnorm": gradnorm,
        "zcp_snip": snip,
        "zcp_jacob_cov": jacob_cov(model, tokens),
        "zcp_naswot": naswot(model, tokens),
        "zcp_swap": swsp(model, tokens),
        "zcp_fire_rate": fire_rate(model, tokens),
    }
