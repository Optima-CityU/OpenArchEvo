"""Assemble the pluggable components of a search from its configuration.

``dry_run=True`` swaps the network-backed parts for offline fakes (LLM,
embedder, surrogate), so the whole pipeline runs on a CPU without PyTorch,
API keys or a surrogate service.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from openarchevo.config import SearchConfig
from openarchevo.fitness import FITNESS_MODES, fitness_fn, required_metrics
from openarchevo.llm import FakeLLM, HashEmbedder, OpenAICompatibleEmbedder, OpenAICompatibleLLM
from openarchevo.llm.base import ProgramGenerator, TextEmbedder
from openarchevo.novelty.similarity import SimilarityModel
from openarchevo.surrogate import FakeSurrogate, Surrogate, SurrogateClient


@dataclass
class Components:
    generator: ProgramGenerator
    embedder: TextEmbedder
    surrogate: Surrogate
    similarity: SimilarityModel
    fitness: Callable[[Mapping[str, float]], float]
    required_metrics: list[str]


def build_similarity(cfg: SearchConfig, cache_path: Optional[Path] = None) -> SimilarityModel:
    n = cfg.novelty
    return SimilarityModel(n.code_backend, n.duplicate_eps, cache_path, n.num_workers)


def build_components(cfg: SearchConfig, cache_path: Optional[Path] = None, dry_run: bool = False) -> Components:
    if dry_run:
        generator: ProgramGenerator = FakeLLM(seed=cfg.seed)
        embedder: TextEmbedder = HashEmbedder()
        surrogate: Surrogate = FakeSurrogate(FITNESS_MODES[cfg.fitness_mode])
    else:
        generator = OpenAICompatibleLLM(cfg.llm.model, cfg.llm.base_url_env, cfg.llm.api_key_env,
                                        cfg.llm.max_tokens, cfg.llm.temperature, cfg.llm.timeout_s)
        embedder = OpenAICompatibleEmbedder(cfg.embedder.model, cfg.embedder.base_url_env,
                                            cfg.embedder.api_key_env, cfg.embedder.dimensions)
        surrogate = SurrogateClient(cfg.surrogate)
    return Components(generator, embedder, surrogate, build_similarity(cfg, cache_path),
                      fitness_fn(cfg.fitness_mode), required_metrics(cfg.fitness_mode))


def check_endpoints(comps: Components) -> None:
    """Preflight for a real run: the surrogate service and the embedding endpoint must answer.

    Raises ``RuntimeError`` with a hint instead of letting the inner loop fail
    thousands of times. (The LLM is not called here; persistent LLM errors stop
    the inner loop after ``inner_loop.max_consecutive_errors``.)
    """
    status = getattr(comps.surrogate, "status", None)
    if status is not None:
        url = getattr(comps.surrogate, "base_url", "the surrogate service")
        try:
            status()
        except Exception as exc:
            raise RuntimeError(f"Surrogate service at {url} does not answer ({type(exc).__name__}: {exc}). "
                               "Start it with scripts/start_surrogate.py, check SURROGATE_URL / the SSH tunnel "
                               "and SURROGATE_TOKEN.") from exc
    try:
        comps.embedder.embed("preflight")
    except Exception as exc:
        raise RuntimeError(f"Embedding endpoint failed ({type(exc).__name__}: {exc}). "
                           "Check the embedder: section of configs/search.yaml and its environment variables.") from exc
