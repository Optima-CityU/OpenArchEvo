from __future__ import annotations

import importlib.util
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np

# Needed for deterministic CuBLAS behavior on CUDA >= 10.2.
# Keep caller-provided value if already set.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn


DEFAULT_VALUE_TOL = 1e-7
DEFAULT_GRAD_TOL = 1e-7
DEFAULT_STRESS_BIASES = (0.5, 1.0)
DEFAULT_STRESS_RUNS = 1
DEFAULT_BASE_RUNS = 0
DEFAULT_SEQ_LEN = 128


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except Exception:
            value = default
    if min_value is not None:
        return max(min_value, value)
    return value


def _env_float(name: str, default: float, *, min_value: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except Exception:
            value = default
    if min_value is not None:
        return max(min_value, value)
    return value


def _env_biases(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    if not toks:
        return ()
    vals = []
    for t in toks:
        try:
            vals.append(float(t))
        except Exception:
            continue
    return tuple(vals)


def _extract_tensor_output(output: object, label: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        t = output[0]
        if isinstance(t, torch.Tensor):
            return t
    raise RuntimeError(f"[{label}] Unexpected output type in causality check: {type(output)}")


def _pivot_list(seq_len: int, *, include_early_cut: bool = False) -> list[int]:
    pivots = {
        0,
        1,
        max(0, seq_len // 4),
        max(0, seq_len // 3),
        max(0, seq_len // 2),
        max(0, (2 * seq_len) // 3),
        max(0, seq_len - 2),
    }
    if include_early_cut:
        # Sample the early part of the gap before the first length-scaled
        # cut (L/4), adding at most one backward pass per probe input.
        pivots.add(seq_len // 16)
    return sorted(p for p in pivots if p < seq_len - 1)


@contextmanager
def _deterministic(seed: int, device: torch.device):
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)

    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if use_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if use_cuda:
            target_index = device.index if device.index is not None else torch.cuda.current_device()
            with torch.cuda.device(target_index):
                torch.cuda.manual_seed(seed)

        prev_bench = torch.backends.cudnn.benchmark
        prev_det = torch.backends.cudnn.deterministic
        prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
        prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        prev_use_det = torch.are_deterministic_algorithms_enabled()
        prev_matmul_precision = torch.get_float32_matmul_precision()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_float32_matmul_precision("highest")
        try:
            yield
        finally:
            torch.backends.cudnn.benchmark = prev_bench
            torch.backends.cudnn.deterministic = prev_det
            torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
            torch.use_deterministic_algorithms(prev_use_det, warn_only=True)
            torch.set_float32_matmul_precision(prev_matmul_precision)
            random.setstate(py_state)
            np.random.set_state(np_state)


@contextmanager
def _forked_init_rng(device: torch.device, seed: int = 7777):
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if use_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if use_cuda:
            target_index = device.index if device.index is not None else torch.cuda.current_device()
            with torch.cuda.device(target_index):
                torch.cuda.manual_seed(seed)
        yield


def _build_block(block_cls, device: torch.device, d_model: int, num_heads: int, d_ffn_mult: int, dropout: float):
    block = block_cls(
        d_model=d_model,
        num_heads=num_heads,
        d_ffn_mult=d_ffn_mult,
        dropout=dropout,
        prenorm=True,
        layer_idx=0,
        use_spiking_mlp=True,
        neuron_type="sdn",
    ).to(device)
    block.eval()
    return block


class _RestoredForward:
    """Rebuild block from canonical in-memory CPU state for each forward."""

    def __init__(self, block_cls, device: torch.device, d_model: int, num_heads: int, d_ffn_mult: int, dropout: float):
        self.block_cls = block_cls
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn_mult = d_ffn_mult
        self.dropout = dropout
        self.restore_calls = 0

        with _forked_init_rng(device, seed=7777):
            block = _build_block(
                block_cls,
                device,
                d_model=d_model,
                num_heads=num_heads,
                d_ffn_mult=d_ffn_mult,
                dropout=dropout,
            )
        self.canonical_state_cpu = {k: v.detach().cpu().clone() for k, v in block.state_dict().items()}
        del block

    def _new_restored_block(self) -> nn.Module:
        with _forked_init_rng(self.device, seed=7777):
            block = _build_block(
                self.block_cls,
                self.device,
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ffn_mult=self.d_ffn_mult,
                dropout=self.dropout,
            )
        block.load_state_dict(self.canonical_state_cpu, strict=True)
        for p in block.parameters():
            p.requires_grad_(False)
        block.eval()
        self.restore_calls += 1
        return block

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        block = self._new_restored_block()
        try:
            return _extract_tensor_output(block(x), label="SpikingFLABlock")
        finally:
            del block


def gradient_dependency_test(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    d_model: int,
    seq_len: int,
    runs: int,
    *,
    input_bias: float = 0.0,
) -> float:
    max_future_grad = 0.0
    # Extra gradient cuts reuse the graph without adding perturbation forwards.
    pivots = _pivot_list(seq_len, include_early_cut=True)
    for i in range(runs):
        with _deterministic(10000 + i, device):
            x = (torch.randn(1, seq_len, d_model, device=device, dtype=torch.float32) + input_bias).requires_grad_(True)
            y = forward_fn(x)
            for t in pivots:
                # Probe every prefix output, including positions between pivots.
                # Random weights reduce cancellation between their gradients.
                prefix = y[:, : t + 1, :]
                proj = torch.randn_like(prefix)
                loss = (prefix * proj).sum()
                g = torch.autograd.grad(loss, x, retain_graph=True, create_graph=False)[0]
                if t + 1 < seq_len:
                    future = g[:, t + 1 :, :]
                    if future.numel() > 0:
                        max_future_grad = max(max_future_grad, float(future.abs().max().item()))
    return max_future_grad


def single_future_token_perturbation_test(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    d_model: int,
    seq_len: int,
    runs: int,
    *,
    input_bias: float = 0.0,
) -> float:
    max_prefix_change = 0.0
    pivots = _pivot_list(seq_len)
    for i in range(runs):
        with _deterministic(13000 + i, device):
            x = torch.randn(1, seq_len, d_model, device=device, dtype=torch.float32) + input_bias
            y = forward_fn(x)
            for t in pivots:
                for j in sorted({t + 1, seq_len - 1}):
                    if j <= t:
                        continue
                    x2 = x.clone()
                    x2[:, j, :] = torch.randn_like(x2[:, j, :])
                    y2 = forward_fn(x2)
                    max_prefix_change = max(
                        max_prefix_change,
                        float((y2[:, : t + 1, :] - y[:, : t + 1, :]).abs().max().item()),
                    )
    return max_prefix_change


def evaluate_block_causality(
    block_cls,
    *,
    device: torch.device,
    d_model: int,
    seq_len: int,
    runs: int,
    value_tol: float,
    grad_tol: float,
    stress_biases: tuple[float, ...],
    stress_runs: int,
) -> dict:
    restored_forward = _RestoredForward(
        block_cls,
        device=device,
        d_model=d_model,
        num_heads=8,
        d_ffn_mult=4,
        dropout=0.0,
    )

    base_runs = max(0, int(runs))
    grad_future = 0.0
    single_future_change = 0.0
    if base_runs > 0:
        grad_future = gradient_dependency_test(restored_forward, device, d_model, seq_len, base_runs)
        single_future_change = single_future_token_perturbation_test(restored_forward, device, d_model, seq_len, base_runs)

    biased_grad_future = 0.0
    biased_single_future_change = 0.0
    active_biases = tuple(float(b) for b in stress_biases if float(b) != 0.0)
    if active_biases:
        sruns = max(1, int(stress_runs))
        for bias in active_biases:
            biased_grad_future = max(
                biased_grad_future,
                gradient_dependency_test(
                    restored_forward,
                    device,
                    d_model,
                    seq_len,
                    sruns,
                    input_bias=bias,
                ),
            )
            biased_single_future_change = max(
                biased_single_future_change,
                single_future_token_perturbation_test(
                    restored_forward,
                    device,
                    d_model,
                    seq_len,
                    sruns,
                    input_bias=bias,
                ),
            )

    hard_fail_reasons = []
    if single_future_change > value_tol:
        hard_fail_reasons.append("single_future_perturbation")
    if grad_future > grad_tol:
        hard_fail_reasons.append("gradient_future_dependency")
    if biased_single_future_change > value_tol:
        hard_fail_reasons.append("biased_single_future_perturbation")
    if biased_grad_future > grad_tol:
        hard_fail_reasons.append("biased_gradient_future_dependency")

    return {
        "pass": len(hard_fail_reasons) == 0,
        "hard_fail_reasons": hard_fail_reasons,
        "metrics": {
            "gradient_max_abs_future_grad": float(grad_future),
            "single_future_token_prefix_max_abs_change": float(single_future_change),
            "biased_gradient_max_abs_future_grad": float(biased_grad_future),
            "biased_single_future_token_prefix_max_abs_change": float(biased_single_future_change),
        },
        "state_restore": {
            "mode": "in_memory_cpu_state_dict + fresh_instance_per_forward",
            "restore_calls": int(restored_forward.restore_calls),
        },
    }


def load_spiking_fla_block_class(file_path: str, label: str):
    module_name = f"surrogate_submit_{Path(file_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"[{label}] Unable to load module spec for causality check.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    block_cls = getattr(module, "SpikingFLABlock", None)
    if block_cls is None:
        raise RuntimeError(f"[{label}] SpikingFLABlock not found for causality check.")
    return block_cls


def _resolve_runtime_config(
    *,
    runs: int,
    seq_len: int,
    grad_tol: float,
    value_tol: float,
) -> dict:
    return {
        "runs": _env_int("SURROGATE_CAUSALITY_RUNS", runs, min_value=0),
        "seq_len": _env_int("SURROGATE_CAUSALITY_SEQ_LEN", seq_len, min_value=4),
        "grad_tol": _env_float("SURROGATE_CAUSALITY_GRAD_TOL", grad_tol, min_value=0.0),
        "value_tol": _env_float("SURROGATE_CAUSALITY_VALUE_TOL", value_tol, min_value=0.0),
        "stress_runs": _env_int("SURROGATE_CAUSALITY_STRESS_RUNS", DEFAULT_STRESS_RUNS, min_value=1),
        "stress_biases": _env_biases("SURROGATE_CAUSALITY_STRESS_BIASES", DEFAULT_STRESS_BIASES),
    }


def assert_block_class_causal(
    block_cls,
    *,
    device: torch.device,
    label: str,
    d_model: int = 256,
    seq_len: int = DEFAULT_SEQ_LEN,
    runs: int = DEFAULT_BASE_RUNS,
    tol: float = DEFAULT_GRAD_TOL,
) -> float:
    cfg = _resolve_runtime_config(
        runs=runs,
        seq_len=seq_len,
        grad_tol=tol,
        value_tol=DEFAULT_VALUE_TOL,
    )
    try:
        verdict = evaluate_block_causality(
            block_cls,
            device=device,
            d_model=d_model,
            seq_len=cfg["seq_len"],
            runs=cfg["runs"],
            value_tol=cfg["value_tol"],
            grad_tol=cfg["grad_tol"],
            stress_biases=cfg["stress_biases"],
            stress_runs=cfg["stress_runs"],
        )
    except Exception as exc:
        raise RuntimeError(f"[{label}] Failed to run causality check: {exc}") from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if not verdict["pass"]:
        m = verdict["metrics"]
        raise RuntimeError(
            f"[{label}] Causality check failed: {','.join(verdict['hard_fail_reasons'])}; "
            f"grad={m['gradient_max_abs_future_grad']:.3e}, "
            f"single={m['single_future_token_prefix_max_abs_change']:.3e}, "
            f"bgrad={m['biased_gradient_max_abs_future_grad']:.3e}, "
            f"bsingle={m['biased_single_future_token_prefix_max_abs_change']:.3e}."
        )

    return float(
        max(
            verdict["metrics"]["gradient_max_abs_future_grad"],
            verdict["metrics"]["biased_gradient_max_abs_future_grad"],
            verdict["metrics"]["single_future_token_prefix_max_abs_change"],
            verdict["metrics"]["biased_single_future_token_prefix_max_abs_change"],
        )
    )
