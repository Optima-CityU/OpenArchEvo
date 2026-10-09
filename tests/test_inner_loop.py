"""Torch-free unit tests of the inner-loop pieces: islands, prompts, proposal parsing."""

import numpy as np
import pytest

from openarchevo.candidate import Candidate
from openarchevo.inner_loop.islands import IslandModel
from openarchevo.llm.openai_compat import parse_proposal
from openarchevo.llm.prompts import build_user_prompt
from openarchevo.novelty.similarity import SimilarityModel


def _cand(i: int, yhat: float, measured: float | None = None) -> Candidate:
    rng = np.random.default_rng(i)
    return Candidate(id=f"c{i}", code=f"class SpikingFLABlock:\n    k = {i}\n", rationale=f"idea {i}",
                     fingerprint=rng.uniform(0, 1, 21).tolist(), embedding=rng.normal(size=8).tolist(),
                     predicted_fitness=yhat, fitness=measured)


def test_islands_rank_by_predicted_fitness_and_trim():
    model = IslandModel(SimilarityModel("difflib"), n_islands=1, capacity=2, reset_interval_s=None)
    model.seed_islands([_cand(0, yhat=0.9, measured=0.1), _cand(1, yhat=0.2, measured=0.99)])
    assert model.register(_cand(2, yhat=0.5)) == 0
    assert model.register(_cand(3, yhat=0.1)) == 0  # island reaches 2 * capacity -> survival
    survivors = {c.id for c in model.islands[0]}
    assert len(survivors) == 2 and "c0" in survivors  # best by yhat survives, not best measured
    assert model.register(_cand(0, yhat=0.9)) is None  # near-duplicate (same id/code) is rejected


def test_reset_reseeds_weak_islands():
    model = IslandModel(SimilarityModel("difflib"), n_islands=2, capacity=None, reset_interval_s=0.0)
    model.islands = [[_cand(0, 0.9), _cand(1, 0.8)], [_cand(2, 0.1)]]
    assert model.maybe_reset()
    assert [c.id for c in model.islands[1]] == ["c0"] and len(model.islands[0]) == 2


def test_prompt_orders_weakest_first():
    prompt = build_user_prompt([_cand(0, 0.9), _cand(1, 0.3)])
    assert prompt.index("k = 1") < prompt.index("k = 0")
    assert "outperforms [Version 1]" in prompt


def test_parse_proposal():
    p = parse_proposal('```json\n{"name": "n", "motivation": "why", "code": "x = 1\\n"}\n```')
    assert (p.name, p.rationale, p.code) == ("n", "why", "x = 1\n")
    with pytest.raises(ValueError):
        parse_proposal('{"name": "n", "motivation": "", "code": "x"}')


def _inner_loop(generator, surrogate=None, max_errors=5):
    from openarchevo.config import InnerLoopConfig
    from openarchevo.fitness import fitness_fn
    from openarchevo.inner_loop import InnerLoop
    from openarchevo.llm import HashEmbedder
    from openarchevo.surrogate import FakeSurrogate

    cfg = InnerLoopConfig(pool_size=4, n_islands=1, island_capacity=None, num_workers=2,
                          reset_interval_s=None, max_consecutive_errors=max_errors)
    return InnerLoop(cfg, generator, HashEmbedder(), surrogate or FakeSurrogate(), SimilarityModel("difflib"),
                     fitness_fn("composite"), round_index=3)


def _roots():
    from openarchevo.surrogate import FakeSurrogate

    roots = []
    for i in range(2):
        code = f'class SpikingFLABlock:\n    """root {i}"""\n    def forward(self, x):\n        return x * {i}.5\n'
        roots.append(Candidate(id=f"root{i}", code=code, rationale=f"root {i}",
                               fingerprint=FakeSurrogate().fingerprint(code)))
    return roots


def test_candidate_ids_are_sequential():
    from openarchevo.llm import FakeLLM

    inner = _inner_loop(FakeLLM(seed=1))
    pool = inner.run(_roots()).pool
    assert pool and all(c.id == f"r03-s{c.sample:04d}" for c in pool)


def test_persistent_llm_errors_stop_the_loop():
    class Broken:
        def propose(self, system_prompt, user_prompt):
            raise ConnectionError("401 Unauthorized")

    inner = _inner_loop(Broken())
    with pytest.raises(RuntimeError, match="llm_errors in a row"):
        inner.run(_roots())
    assert inner.stats["llm_error"] < 20  # stopped early, not after pool_size * max_attempts_factor
