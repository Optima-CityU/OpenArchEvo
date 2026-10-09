"""Fitness: map measured or predicted task metrics to a scalar in (0, 1).

This is the paper's mapping (Appendix "Fitness Definition"). Each task metric is
turned into a relative improvement over an anchor model (SpikingDeltaNet),
clipped to [-1, 1] and squashed by an asymmetric sigmoid; the composite fitness
averages the task scores. The anchor maps to exactly 0.5.

The same function scores surrogate predictions (``\\hat y``) and measured
outcomes (``y``), so the two are on one scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class TaskTransform:
    """Asymmetric sigmoid transform of one task metric.

    Attributes:
        anchor: Metric value of the anchor model; maps to 0.5.
        tol_gain: Relative improvement that saturates the gain side (T+).
        tol_loss: Relative degradation that saturates the loss side (T-).
        slope_gain: Sigmoid slope for improvements (kappa+).
        slope_loss: Sigmoid slope for degradations (kappa-).
        higher_is_better: True for accuracy, False for perplexity.
    """

    anchor: float
    tol_gain: float
    tol_loss: float
    slope_gain: float
    slope_loss: float
    higher_is_better: bool

    def __call__(self, x: float) -> float:
        delta = (x - self.anchor) if self.higher_is_better else (self.anchor - x)
        tol = self.tol_gain if delta >= 0 else self.tol_loss
        r = max(-1.0, min(1.0, delta / (self.anchor * tol)))
        slope = self.slope_gain if r >= 0 else self.slope_loss
        return 1.0 / (1.0 + math.exp(-slope * r))


#: Main search: WikiText-2 perplexity + ListOps accuracy (percentage points).
COMPOSITE = {
    "wikitext2_ppl": TaskTransform(58.28, 0.02, 0.25, 3.0, math.log(3.0), higher_is_better=False),
    "listops_acc": TaskTransform(53.0, 0.02, 0.70, 3.0, math.log(3.0), higher_is_better=True),
}

#: Single-task example: WikiText-2 only (used by the paper's ablations and method comparisons).
WT2_ONLY = {
    "wikitext2_ppl": TaskTransform(58.28, 0.15, 0.25, 3.0, 3.0, higher_is_better=False),
}

FITNESS_MODES: dict[str, Mapping[str, TaskTransform]] = {
    "composite": COMPOSITE,
    "wt2_only": WT2_ONLY,
}


def fitness_fn(mode: str = "composite") -> Callable[[Mapping[str, float]], float]:
    """Return ``f(metrics) -> fitness`` for a fitness mode.

    ``metrics`` must contain every metric of the mode (e.g. ``wikitext2_ppl`` and
    ``listops_acc``); the result is the mean of the per-task scores.
    """
    transforms = FITNESS_MODES[mode]

    def fitness(metrics: Mapping[str, float]) -> float:
        missing = [k for k in transforms if metrics.get(k) is None]
        if missing:
            raise KeyError(f"Missing metrics for fitness mode '{mode}': {missing}")
        scores = [t(float(metrics[k])) for k, t in transforms.items()]
        return sum(scores) / len(scores)

    return fitness


def required_metrics(mode: str = "composite") -> list[str]:
    """Metric names that a fitness mode needs (and that training must report)."""
    return list(FITNESS_MODES[mode])
