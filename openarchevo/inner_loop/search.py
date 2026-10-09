"""Surrogate-guided program evolution (Algorithm 2 of the paper).

Each worker thread repeats::

    parents  <- islands.sample_parents(k)                 # within or across islands
    prompt   <- build_user_prompt(parents)                # versioned context
    proposal <- LLM(system_prompt, prompt)                # name, rationale, code
    reject if the program fails static checks             # feasibility, part 1
    fingerprint, metrics <- surrogate.evaluate(code)      # executes the program (part 2)
    reject if the fingerprint nearly matches a parent
    y_hat    <- fitness(metrics);  embedding <- embed(rationale)
    island   <- islands.register(candidate)               # rejects near-redundant programs
    add to pool;  islands.maybe_reset()

until ``pool_size`` candidates are accepted. Accepted candidates are appended to
``pool.jsonl`` as they arrive.

Candidate ids are ``r{t:02d}-s{n:04d}``: round ``t`` and the program's running
number ``n`` in this inner loop. ``n`` is assigned when a program reaches island
registration, so programs rejected there as near-duplicates leave gaps.

Errors are not silent: the first few LLM, embedding and surrogate errors are
logged at WARNING, and ``max_consecutive_errors`` failures of one kind in a row
(for example a wrong API key or an unreachable surrogate) stop the loop with a
``RuntimeError``.
"""

from __future__ import annotations

import ast
import json
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from openarchevo.candidate import Candidate
from openarchevo.config import InnerLoopConfig
from openarchevo.contract import BLOCK_CLASS
from openarchevo.inner_loop.islands import IslandModel
from openarchevo.llm.base import ProgramGenerator, TextEmbedder
from openarchevo.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from openarchevo.novelty.similarity import SimilarityModel
from openarchevo.surrogate.base import InfeasibleProgram, Surrogate

log = logging.getLogger("openarchevo.inner_loop")


def static_check(code: str, class_name: str = BLOCK_CLASS) -> Optional[str]:
    """Cheap feasibility test before execution; returns a reason or ``None`` if OK."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            if any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in node.body):
                return None
            return f"{class_name} has no forward()"
    return f"no class {class_name}"


@dataclass
class InnerLoopResult:
    pool: list[Candidate]
    stats: dict[str, int]


class InnerLoop:
    """Runs one inner loop for outer iteration ``round_index``.

    Args:
        cfg: Inner-loop settings.
        generator / embedder / surrogate / similarity: Pluggable components.
        fitness: Maps a metrics dict to fitness (``openarchevo.fitness.fitness_fn``).
        round_index: Outer iteration ``t``; used in candidate ids.
        warm_start_pool: If set, run the warm-start stage instead: one island, no
            survival, and this many accepted candidates.
        pool_path: If given, accepted candidates are appended here as JSON lines.
    """

    def __init__(self, cfg: InnerLoopConfig, generator: ProgramGenerator, embedder: TextEmbedder,
                 surrogate: Surrogate, similarity: SimilarityModel,
                 fitness: Callable[[Mapping[str, float]], float], round_index: int, seed: int = 0,
                 warm_start_pool: Optional[int] = None, pool_path: Optional[Path] = None):
        self.cfg = cfg
        self.generator, self.embedder, self.surrogate = generator, embedder, surrogate
        self.similarity, self.fitness = similarity, fitness
        self.round_index = round_index
        self.pool_size = warm_start_pool or cfg.pool_size
        self.max_attempts = self.pool_size * cfg.max_attempts_factor
        self.islands = IslandModel(
            similarity, 1 if warm_start_pool else cfg.n_islands, None if warm_start_pool else cfg.island_capacity,
            inter_island_prob=cfg.inter_island_prob, selection_pressure=cfg.selection_pressure,
            fitness_rank_weight=cfg.fitness_rank_weight, reset_interval_s=cfg.reset_interval_s, seed=seed)
        self.pool_path = pool_path
        self.pool: list[Candidate] = []
        self.stats: Counter = Counter()
        self._streak: Counter = Counter()  # consecutive errors per kind
        self._fatal: Optional[str] = None
        self._next_number = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ------------------------------------------------------------------ driver
    def run(self, roots: Sequence[Candidate]) -> InnerLoopResult:
        """Seed islands with ``roots`` (scored by the current surrogate) and evolve."""
        if not roots:
            raise ValueError("The inner loop needs at least one root program.")
        seeds = [Candidate.from_dict(r.to_dict()) for r in roots]
        for seed, pred in zip(seeds, self.surrogate.predict([s.fingerprint for s in seeds])):
            seed.predicted, seed.predicted_fitness = pred, self.fitness(pred)
        self.islands.seed_islands(seeds)
        log.info("Round %d: seeded %d roots into islands %s", self.round_index, len(seeds), self.islands.sizes())
        if self.pool_path:
            self.pool_path.parent.mkdir(parents=True, exist_ok=True)
            self.pool_path.write_text("")
        with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
            for f in [ex.submit(self._worker) for _ in range(self.cfg.num_workers)]:
                f.result()
        if self._fatal:
            raise RuntimeError(self._fatal)
        log.info("Round %d: accepted %d candidates; stats %s", self.round_index, len(self.pool), dict(self.stats))
        return InnerLoopResult(pool=list(self.pool), stats=dict(self.stats))

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                parents = self.islands.sample_parents(self.cfg.num_parents)
                prompt = build_user_prompt(parents, include_rationale=self.cfg.include_rationale)
                for _ in range(self.cfg.samples_per_prompt):
                    if self._stop.is_set():
                        break
                    self._propose(prompt, parents)
            except Exception as exc:  # keep the worker alive, but stop if failures persist
                self._error("worker_error", exc)

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _error(self, kind: str, exc: Exception) -> None:
        """Count an error; warn about the first few; stop after too many in a row."""
        limit = self.cfg.max_consecutive_errors
        with self._lock:
            self.stats[kind] += 1
            self._streak[kind] += 1
            n, streak = self.stats[kind], self._streak[kind]
            if limit and streak >= limit and self._fatal is None:
                self._fatal = (f"Inner loop stopped: {streak} {kind}s in a row (last: {type(exc).__name__}: {exc}). "
                               "Check the endpoint, API key and service named in configs/search.yaml.")
                self._stop.set()
            if n > self.max_attempts:
                self._stop.set()
        if n <= 3:
            log.warning("%s #%d: %s: %s", kind, n, type(exc).__name__, exc, exc_info=kind == "worker_error")
            if n == 3:
                log.warning("Further %ss are counted in the round statistics but not logged.", kind)

    def _ok(self, kind: str) -> None:
        if self._streak[kind]:
            with self._lock:
                self._streak[kind] = 0

    # ------------------------------------------------------------------ one proposal
    def _propose(self, prompt: str, parents: list[Candidate]) -> None:
        with self._lock:
            self.stats["proposals"] += 1
            if self.stats["proposals"] > self.max_attempts:
                log.warning("Reached max attempts (%d); stopping inner loop.", self.max_attempts)
                self._stop.set()
                return
        try:
            proposal = self.generator.propose(SYSTEM_PROMPT, prompt)
        except Exception as exc:  # network, HTTP or parse errors
            return self._error("llm_error", exc)
        self._ok("llm_error")
        if static_check(proposal.code):
            return self._count("infeasible_static")
        try:
            evaluation = self.surrogate.evaluate(proposal.code)
        except InfeasibleProgram:
            self._ok("surrogate_error")
            return self._count("infeasible_runtime")
        except Exception as exc:  # service unreachable, not fitted, ...
            return self._error("surrogate_error", exc)
        self._ok("surrogate_error")
        fp = np.asarray(evaluation.fingerprint)
        if any(p.fingerprint is not None and np.allclose(fp, p.fingerprint, rtol=1e-6, atol=1e-8) for p in parents):
            return self._count("parent_fingerprint_match")
        try:
            embedding = self.embedder.embed(proposal.rationale)
        except Exception as exc:
            return self._error("embedding_error", exc)
        self._ok("embedding_error")
        with self._lock:
            self._next_number += 1
            number = self._next_number
        cand = Candidate(
            id=f"r{self.round_index:02d}-s{number:04d}", sample=number, code=proposal.code,
            rationale=proposal.rationale, name=proposal.name, fingerprint=list(map(float, fp)),
            embedding=embedding, predicted=evaluation.metrics, predicted_fitness=self.fitness(evaluation.metrics),
            round=self.round_index, parent_ids=[p.id for p in parents])
        if self.islands.register(cand) is None:
            return self._count("redundant")
        with self._lock:
            if len(self.pool) >= self.pool_size:
                self._stop.set()
                return
            self.pool.append(cand)
            if self.pool_path:
                with self.pool_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(cand.to_dict()) + "\n")
            if len(self.pool) >= self.pool_size:
                self._stop.set()
        if self.islands.maybe_reset():
            self._count("island_resets")
