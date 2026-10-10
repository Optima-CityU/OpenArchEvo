"""CPU regressions for experiment-compatible causality admission checks."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from openarchevo.causality import (  # noqa: E402
    DEFAULT_GRAD_TOL,
    DEFAULT_STRESS_BIASES,
    assert_block_class_causal,
    evaluate_block_causality,
    gradient_dependency_test,
)


@pytest.fixture(autouse=True)
def clean_causality_environment(monkeypatch):
    for name in ("RUNS", "SEQ_LEN", "GRAD_TOL", "VALUE_TOL", "STRESS_RUNS", "STRESS_BIASES"):
        monkeypatch.delenv(f"SURROGATE_CAUSALITY_{name}", raising=False)


class CausalBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x.cumsum(dim=1), None, None


class GlobalMeanBlock(CausalBlock):
    def forward(self, x):
        return x + x.mean(dim=1, keepdim=True), None, None


class DetachedLeakBlock(CausalBlock):
    def forward(self, x):
        return x + x.mean(dim=1, keepdim=True).detach(), None, None


class StatefulCausalBlock(CausalBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("step", torch.tensor(0.0))
        self.calls = 0

    def forward(self, x):
        self.step += 1
        self.calls += 1
        return x.cumsum(dim=1) + self.step.clone() + self.calls, None, None


def evaluate(block_cls, **overrides):
    kwargs = dict(device=torch.device("cpu"), d_model=8, seq_len=32, runs=1,
                  value_tol=1e-7, grad_tol=1e-7, stress_biases=(), stress_runs=0)
    kwargs.update(overrides)
    return evaluate_block_causality(block_cls, **kwargs)


def test_default_gate_accepts_causal_block_and_rejects_future_information():
    assert assert_block_class_causal(CausalBlock, device=torch.device("cpu"), label="causal", d_model=8) == 0.0
    with pytest.raises(RuntimeError, match="[Cc]ausality"):
        assert_block_class_causal(GlobalMeanBlock, device=torch.device("cpu"), label="leaky", d_model=8)


def test_differentiable_leak_detected_by_both_probes():
    verdict = evaluate(GlobalMeanBlock)
    assert not verdict["pass"]
    assert set(verdict["hard_fail_reasons"]) == {"gradient_future_dependency", "single_future_perturbation"}


def test_perturbation_detects_leak_hidden_from_autograd():
    verdict = evaluate(DetachedLeakBlock)
    assert not verdict["pass"]
    assert verdict["metrics"]["gradient_max_abs_future_grad"] == 0.0
    assert verdict["metrics"]["single_future_token_prefix_max_abs_change"] > 1e-7
    assert verdict["hard_fail_reasons"] == ["single_future_perturbation"]


def test_stress_bias_exposes_inactive_spiking_path():
    class GatedLeakBlock(CausalBlock):
        def forward(self, x):
            spike = (x.mean() > 0.75).to(x.dtype)
            return x + spike * x.mean(dim=1, keepdim=True), None, None

    assert evaluate(GatedLeakBlock)["pass"]
    verdict = evaluate(GatedLeakBlock, runs=0, stress_biases=DEFAULT_STRESS_BIASES, stress_runs=1)
    assert not verdict["pass"]
    assert "biased_gradient_future_dependency" in verdict["hard_fail_reasons"]
    assert "biased_single_future_perturbation" in verdict["hard_fail_reasons"]


@pytest.mark.parametrize("seq_len", [32, 64, 128, 256])
def test_gradient_probe_covers_early_outputs_between_sampled_positions(seq_len):
    destination, source = 3 * seq_len // 128, 12 * seq_len // 128

    def forward(x):
        y = x.clone()
        y[:, destination, :] += x[:, source, :]
        return y

    leak = gradient_dependency_test(forward, torch.device("cpu"), d_model=8, seq_len=seq_len,
                                    runs=1, input_bias=DEFAULT_STRESS_BIASES[0])
    assert leak > DEFAULT_GRAD_TOL


def test_opposite_channel_dependencies_do_not_cancel():
    def forward(x):
        y = x.clone()
        y[:, 3, 0] += x[:, 12, 0]
        y[:, 3, 1] -= x[:, 12, 0]
        return y

    assert gradient_dependency_test(forward, torch.device("cpu"), d_model=8, seq_len=128,
                                    runs=1, input_bias=0.5) > DEFAULT_GRAD_TOL


def test_fresh_instances_isolate_buffers_and_python_state_between_forwards():
    verdict = evaluate(StatefulCausalBlock)
    assert verdict["pass"], verdict
    assert verdict["state_restore"]["restore_calls"] > 1
    assert all(value == 0.0 for value in verdict["metrics"].values())


LEAKY_PROGRAM = '''
import torch
from torch import nn
class SpikingFLABlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, x):
        return x + x.mean(dim=1, keepdim=True), None, None
'''


def test_fingerprint_rejects_noncausal_program_before_probes(monkeypatch):
    from openarchevo.config import FingerprintConfig
    from openarchevo.fingerprint.extract import FingerprintError, extract_fingerprint
    from openarchevo.fingerprint import probes

    def unexpected_probe(*args, **kwargs):
        pytest.fail("Fingerprint probes ran before causality admission")

    monkeypatch.setattr(probes, "compute_probes", unexpected_probe)
    monkeypatch.setenv("SURROGATE_CAUSALITY_SEQ_LEN", "16")
    with pytest.raises(FingerprintError, match="[Cc]ausality"):
        extract_fingerprint(LEAKY_PROGRAM, FingerprintConfig(), "cpu")


def test_fingerprint_checks_experiment_dimensions_and_selected_neuron(monkeypatch):
    from openarchevo.models import causality
    from openarchevo.config import FingerprintConfig
    from openarchevo.fingerprint.extract import FingerprintError, extract_fingerprint

    checked = []

    def gate(block_cls, *, device, label, d_model=256, num_heads=8, d_ffn_mult=4, neuron_type="sdn", **kwargs):
        checked.append((str(device), d_model, num_heads, d_ffn_mult, neuron_type))
        raise RuntimeError("causality test sentinel")

    monkeypatch.setattr(causality, "check_block_causality", gate)
    with pytest.raises(FingerprintError, match="causality test sentinel"):
        extract_fingerprint(LEAKY_PROGRAM, FingerprintConfig(neuron_type="bptt"), "cpu")
    assert checked == [("cpu", 256, 8, 4, "bptt")]


def test_lm_training_rejects_noncausal_block_before_model_construction(monkeypatch, tmp_path):
    from training import common

    program = tmp_path / "leaky.py"
    program.write_text(LEAKY_PROGRAM)

    def unexpected_model(*args, **kwargs):
        pytest.fail("Training model was constructed before causality admission")

    monkeypatch.setattr(common, "SequenceModel", unexpected_model)
    monkeypatch.setenv("SURROGATE_CAUSALITY_SEQ_LEN", "16")
    with pytest.raises(RuntimeError, match="[Cc]ausality"):
        common.build_model(program, dict(d_model=8, num_heads=2, n_layers=1),
                           vocab_size=16, n_outputs=16, pad_token_id=0, is_lm=True)


@pytest.mark.parametrize("is_lm", [True, False])
def test_training_gate_uses_model_settings_and_only_applies_to_lm(monkeypatch, tmp_path, is_lm):
    from training import common

    calls = []
    block_cls = object()
    model = object()
    model_cfg = dict(d_model=12, num_heads=3, d_ffn_mult=2, n_layers=1, neuron_type="bptt")

    def gate(cls, **kwargs):
        assert cls is block_cls
        calls.append(("gate", kwargs))

    def build(cls, **kwargs):
        assert cls is block_cls
        calls.append(("model", kwargs))
        return model

    monkeypatch.setattr(common, "load_block_class", lambda path: block_cls)
    monkeypatch.setattr(common, "check_block_causality", gate)
    monkeypatch.setattr(common, "SequenceModel", build)
    assert common.build_model(tmp_path / "candidate.py", model_cfg, 16, 16, 0,
                              is_lm=is_lm, device="cpu") is model
    assert [kind for kind, kwargs in calls] == (["gate", "model"] if is_lm else ["model"])
    if is_lm:
        kwargs = calls[0][1]
        assert str(kwargs["device"]) == "cpu"
        assert {key: kwargs[key] for key in ("d_model", "num_heads", "d_ffn_mult", "neuron_type")} == {
            key: model_cfg[key] for key in ("d_model", "num_heads", "d_ffn_mult", "neuron_type")}


def test_adapter_passes_constructor_settings_to_every_fresh_block(monkeypatch):
    from openarchevo.models.causality import check_block_causality

    constructors = []

    class ConfiguredBlock(CausalBlock):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.d_model = kwargs["d_model"]
            constructors.append(kwargs)

        def forward(self, x):
            assert x.shape[-1] == self.d_model
            return super().forward(x)

    monkeypatch.setenv("SURROGATE_CAUSALITY_SEQ_LEN", "16")
    assert check_block_causality(ConfiguredBlock, device="cpu", label="configured", d_model=12,
                                 num_heads=3, d_ffn_mult=2, neuron_type="bptt",
                                 extra_block_kwargs={"custom_option": 7}) == 0.0
    assert len(constructors) > 1
    for kwargs in constructors:
        assert kwargs["d_model"] == 12
        assert kwargs["num_heads"] == 3
        assert kwargs["d_ffn_mult"] == 2
        assert kwargs["neuron_type"] == "bptt"
        assert kwargs["custom_option"] == 7
