"""Candidate architecture records and the evaluated archive.

A *candidate* is one architecture program together with its three views
(code, design rationale, behavioural fingerprint), its surrogate predictions and,
once trained, its measured task metrics. The *archive* ``D_t`` is the set of
candidates with measured outcomes; it is stored as one JSON object per line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


@dataclass
class Candidate:
    """One architecture program and everything the search knows about it.

    Attributes:
        id: Stable identifier. Inner-loop programs get ``r{t:02d}-s{n:04d}``, e.g.
            ``r03-s0335`` (round 3, program number 335 of that inner loop); the
            initial archive uses the program file stem (e.g. ``deltanet``). The
            paper's zoo uses the same pattern with ``i`` (``i03-s0335``).
        code: Python source defining the block class (``openarchevo.contract.BLOCK_CLASS``).
        rationale: Natural-language design rationale produced with the code.
        name: Short name proposed by the LLM (informational only).
        fingerprint: 21-dim behavioural fingerprint (see ``openarchevo.fingerprint.schema``).
        embedding: Rationale embedding used for rationale similarity.
        predicted: Surrogate-predicted task metrics, e.g. ``{"wikitext2_ppl": 41.2}``.
        measured: Task metrics measured by real (proxy) training.
        predicted_fitness: Fitness of ``predicted`` (``\\hat y``).
        fitness: Fitness of ``measured`` (``y``); ``None`` until trained.
        round: Outer iteration that produced the candidate (0 = initial archive).
        parent_ids: Programs shown to the LLM, in prompt order.
        island: Island the candidate was registered to in the inner loop.
        sample: Program number ``n`` inside its inner loop (the ``sNNNN`` part of ``id``).
        meta: Free-form extras (e.g. shortlist group, training logs).
    """

    id: str
    code: str
    rationale: str = ""
    name: str = ""
    fingerprint: Optional[list[float]] = None
    embedding: Optional[list[float]] = None
    predicted: dict[str, float] = field(default_factory=dict)
    measured: dict[str, float] = field(default_factory=dict)
    predicted_fitness: Optional[float] = None
    fitness: Optional[float] = None
    round: Optional[int] = None
    parent_ids: list[str] = field(default_factory=list)
    island: Optional[int] = None
    sample: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def yhat(self) -> float:
        """Predicted fitness, the score that ranks programs inside the inner loop.

        Roots are re-scored by the current surrogate, so measured and predicted
        fitness are never mixed within an island (``-inf`` if not yet predicted).
        """
        return self.predicted_fitness if self.predicted_fitness is not None else float("-inf")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def read_jsonl(path: Path) -> list[Candidate]:
    """Read candidates from a JSON-lines file (missing file -> empty list)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Candidate.from_dict(json.loads(line)) for line in f if line.strip()]


def write_jsonl(path: Path, candidates: Iterable[Candidate]) -> None:
    """Write candidates to a JSON-lines file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict()) + "\n")
    tmp.replace(path)


class Archive:
    """The evaluated archive ``D_t``: candidates with measured task outcomes."""

    def __init__(self, candidates: Optional[Iterable[Candidate]] = None):
        self._items: list[Candidate] = list(candidates or [])

    @classmethod
    def load(cls, path: Path) -> "Archive":
        return cls(read_jsonl(path))

    def save(self, path: Path) -> None:
        write_jsonl(path, self._items)

    def add(self, candidate: Candidate) -> None:
        if candidate.fitness is None:
            raise ValueError(f"Candidate {candidate.id} has no measured fitness.")
        if candidate.id in self.ids():
            raise ValueError(f"Candidate {candidate.id} is already in the archive.")
        self._items.append(candidate)

    def ids(self) -> set[str]:
        return {c.id for c in self._items}

    def initial(self) -> list[Candidate]:
        """The initial archive ``D_0`` (round 0)."""
        return [c for c in self._items if c.round == 0]

    def new_evaluations(self) -> int:
        """Number of evaluated architectures beyond ``D_0`` (counts towards budget B)."""
        return sum(1 for c in self._items if c.round != 0)

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Candidate:
        return self._items[index]
