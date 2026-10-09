"""Typed configuration loaded from the YAML files in ``configs/``.

Unknown keys raise an error so that typos do not silently fall back to defaults.
Defaults below are the paper's main-search settings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


# ----------------------------------------------------------------------------- search

@dataclass
class LLMConfig:
    """Program generator (OpenAI-compatible chat endpoint)."""

    model: str = "gemini-2.5-flash"
    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 32768
    temperature: Optional[float] = None
    timeout_s: float = 600.0


@dataclass
class EmbedderConfig:
    """Rationale embedder (OpenAI-compatible embeddings endpoint)."""

    model: str = "text-embedding-v4"
    dimensions: Optional[int] = 2048
    base_url_env: str = "EMBEDDING_BASE_URL"
    api_key_env: str = "EMBEDDING_API_KEY"


@dataclass
class NoveltyConfig:
    code_backend: str = "codebleu"   # "codebleu" (paper) or "difflib" (no extra deps)
    duplicate_eps: float = 1e-6      # near-duplicate if any view similarity >= 1 - eps
    num_workers: int = 16            # processes for CodeBLEU; 0 = compute in-thread


@dataclass
class SurrogateClientConfig:
    url_env: str = "SURROGATE_URL"
    default_url: str = "http://127.0.0.1:9000"
    timeout_s: float = 600.0
    token_env: str = "SURROGATE_TOKEN"   # shared secret sent as a bearer token, if set


@dataclass
class InnerLoopConfig:
    """Island-based program evolution (Algorithm 2)."""

    pool_size: int = 1080              # accepted candidates per outer iteration (n_pool)
    n_islands: int = 10                # M
    island_capacity: int = 20          # c_isl; survival triggers at 2 * c_isl
    inter_island_prob: float = 0.5     # parents from different islands vs. one island
    num_parents: int = 2               # programs shown in one prompt
    samples_per_prompt: int = 2        # LLM calls per sampled prompt
    include_rationale: bool = True     # show parents' rationales in the prompt
    num_workers: int = 16              # concurrent proposal workers
    reset_interval_s: float = 3600.0   # weak-half island reset period
    selection_pressure: float = 1.0    # alpha in rank-based selection p ~ rank^-alpha
    fitness_rank_weight: float = 0.2   # weight of fitness rank vs. dominance-dissimilarity rank
    max_attempts_factor: int = 10      # stop after pool_size * factor proposals
    max_consecutive_errors: int = 20   # stop after this many LLM/embedding/surrogate errors in a row


@dataclass
class WarmStartConfig:
    enabled: bool = False              # first round: one island, small pool, no survival
    pool_size: int = 48


@dataclass
class OuterLoopConfig:
    """Training allocation and archive updates (Algorithm 1)."""

    rounds: int = 10                   # T
    batch_size: int = 16               # K
    budget: int = 160                  # B, new evaluations excluding D_0
    shortlist_top: int = 30            # |P_top|
    shortlist_dissim: int = 30         # |P_dissim|
    best_per_island: int = 2
    stepping_stones: int = 5
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    #: Shell templates run by ``run_round.py --auto`` for each selected program.
    #: Run from the repository root. Placeholders: {python}, {repo}, {program}, {id}, {results_dir}.
    train_commands: list[str] = field(default_factory=list)


@dataclass
class SearchConfig:
    fitness_mode: str = "composite"    # see openarchevo.fitness.FITNESS_MODES
    seed: int = 0
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    surrogate: SurrogateClientConfig = field(default_factory=SurrogateClientConfig)
    inner_loop: InnerLoopConfig = field(default_factory=InnerLoopConfig)
    outer_loop: OuterLoopConfig = field(default_factory=OuterLoopConfig)


# ----------------------------------------------------------------------------- surrogate service

@dataclass
class ProbeConfig:
    """Fixed probe minibatch for the initialization-time measurements."""

    batch_path: Optional[str] = None   # torch file with (inputs, targets); None -> synthetic batch
    batch_size: int = 32
    seq_len: int = 2000
    vocab_size: int = 20
    n_classes: int = 10


@dataclass
class WrapperConfig:
    """Fixed outer wrapper used when probing a candidate block."""

    n_layers: int = 2
    d_model: int = 128
    num_heads: int = 4
    d_ffn_mult: int = 2
    dropout: float = 0.1


@dataclass
class FingerprintConfig:
    seed: int = 42
    neuron_type: str = "sdn"
    timeout_s: float = 360.0
    devices: list[str] = field(default_factory=lambda: ["cuda:0"])
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    wrapper: WrapperConfig = field(default_factory=WrapperConfig)
    #: Sequence length of the single-block structural trace. MAC/Param scales with
    #: it; 5767 matches the fixed similarity bounds in fingerprint.schema.
    structure_seq_len: int = 5767


@dataclass
class TabPFNConfig:
    model_path_env: str = "TABPFN_MODEL_PATH"
    device: str = "cuda:0"
    random_state: int = 42
    log_targets: list[str] = field(default_factory=lambda: ["wikitext2_ppl"])


@dataclass
class ServiceConfig:
    host: str = "127.0.0.1"            # loopback only; see scripts/start_surrogate.py --allow-remote
    port: int = 9000
    token_env: str = "SURROGATE_TOKEN"   # if set, every request must carry this bearer token
    tabpfn: TabPFNConfig = field(default_factory=TabPFNConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)


# ----------------------------------------------------------------------------- loading

def from_dict(cls: type[T], data: Optional[dict[str, Any]]) -> T:
    """Build a (nested) dataclass from a dict, rejecting unknown keys."""
    data = dict(data or {})
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        hint = hints[key]
        kwargs[key] = from_dict(hint, value) if dataclasses.is_dataclass(hint) else value
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_search_config(path: Path, dry_run: bool = False) -> SearchConfig:
    """Load ``configs/search.yaml``; with ``dry_run`` the ``dry_run:`` section is merged on top."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    overrides = raw.pop("dry_run", {}) or {}
    if dry_run:
        raw = _deep_merge(raw, overrides)
    return from_dict(SearchConfig, raw)


def load_service_config(path: Path) -> ServiceConfig:
    return from_dict(ServiceConfig, yaml.safe_load(Path(path).read_text()) or {})
