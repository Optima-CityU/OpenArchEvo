"""Torch-free tests: fitness, selection, novelty, configs."""

import math
from pathlib import Path

import numpy as np
import pytest

from openarchevo.config import ServiceConfig, from_dict, load_search_config, load_service_config
from openarchevo.fitness import fitness_fn
from openarchevo.novelty import novelty_within
from openarchevo.nsga2 import crowding_distance, non_dominated_fronts, nsga2_select

ROOT = Path(__file__).resolve().parents[1]


def test_fitness_anchor_and_bounds():
    f = fitness_fn("composite")
    assert f({"wikitext2_ppl": 58.28, "listops_acc": 53.0}) == pytest.approx(0.5)
    lo, hi = 1 / (1 + 3), 1 / (1 + math.exp(-3))  # sigma(-ln 3), sigma(3)
    assert f({"wikitext2_ppl": 1e6, "listops_acc": 0.0}) == pytest.approx(lo)
    assert f({"wikitext2_ppl": 1.0, "listops_acc": 100.0}) == pytest.approx(hi)
    assert fitness_fn("wt2_only")({"wikitext2_ppl": 58.28}) == pytest.approx(0.5)
    with pytest.raises(KeyError):
        f({"wikitext2_ppl": 50.0})


def test_nsga2():
    obj = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.2, 0.2], [0.1, 0.1]])
    assert non_dominated_fronts(obj)[0] == [0, 1, 2]
    assert sorted(nsga2_select(obj, 3)) == [0, 1, 2]
    assert sorted(nsga2_select(obj, 2)) == [0, 1]  # boundary points have infinite crowding distance
    assert np.isinf(crowding_distance(obj[:3])[[0, 1]]).all()


def test_novelty_excludes_self():
    sim = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    assert novelty_within(sim) == pytest.approx([0.7, 0.6, 0.5])


def test_configs_load():
    cfg = load_search_config(ROOT / "configs" / "search.yaml")
    assert cfg.outer_loop.batch_size == 16 and cfg.inner_loop.pool_size == 1080
    dry = load_search_config(ROOT / "configs" / "search.yaml", dry_run=True)
    assert dry.novelty.code_backend == "difflib" and dry.inner_loop.pool_size < cfg.inner_loop.pool_size
    assert isinstance(load_service_config(ROOT / "configs" / "surrogate.yaml"), ServiceConfig)
    with pytest.raises(KeyError):
        from_dict(ServiceConfig, {"prot": 1})


def test_fake_surrogate_follows_fitness_mode():
    from openarchevo.fitness import FITNESS_MODES
    from openarchevo.surrogate import FakeSurrogate

    code = "class SpikingFLABlock:\n    def forward(self, x):\n        return x * 0.5\n"
    for mode, transforms in FITNESS_MODES.items():
        fake = FakeSurrogate(transforms)
        assert set(fake.evaluate(code).metrics) == set(transforms)
        assert set(fake.train(code, "seed")) == set(transforms)
        fitness_fn(mode)(fake.train(code, "seed"))  # all required metrics present


def test_surrogate_refuses_remote_host_without_opt_in():
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "SURROGATE_TOKEN"}
    script = str(ROOT / "scripts" / "start_surrogate.py")
    out = subprocess.run([sys.executable, script, "--host", "0.0.0.0"], capture_output=True, text=True, env=env)
    assert out.returncode != 0 and "Refusing" in out.stderr
    out = subprocess.run([sys.executable, script, "--host", "0.0.0.0", "--allow-remote"],
                         capture_output=True, text=True, env=env)
    assert out.returncode != 0 and "SURROGATE_TOKEN" in out.stderr


def test_surrogate_token():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from openarchevo.surrogate.service import create_app

    class Stub:
        version, models = 0, {}

    client = TestClient(create_app(ServiceConfig(), predictor=Stub(), token="s3cret"))
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200
