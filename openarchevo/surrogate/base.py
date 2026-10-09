"""Surrogate interface used by the search loops.

A surrogate does two things for an untrained program:

1. extract its behavioural fingerprint ``x(a)`` (executes the program), and
2. predict its task metrics from ``x(a)`` with a model fitted on the archive.

The search turns predicted metrics into ``\\hat y(a)`` with the same fitness
function used for measured outcomes (``openarchevo.fitness``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class InfeasibleProgram(Exception):
    """The program could not be instantiated or run (C(a) = 0)."""


@dataclass
class Evaluation:
    fingerprint: list[float]
    metrics: dict[str, float]


class Surrogate(Protocol):
    def fit(self, features: Sequence[Sequence[float]], targets: dict[str, Sequence[float]]) -> None:
        """Refresh the predictor on labelled context (archive fingerprints and metrics)."""

    def fingerprint(self, code: str) -> list[float]:
        """Extract the 21-dim fingerprint; raises :class:`InfeasibleProgram`."""

    def predict(self, features: Sequence[Sequence[float]]) -> list[dict[str, float]]:
        """Predict task metrics for fingerprints."""

    def evaluate(self, code: str) -> Evaluation:
        """Fingerprint + prediction in one call; raises :class:`InfeasibleProgram`."""
