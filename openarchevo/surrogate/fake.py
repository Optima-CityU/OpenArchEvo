"""A CPU-only, deterministic surrogate for dry runs and tests.

The fake fingerprint depends on the program's *skeleton* (source with numeric
literals removed) and, more weakly, on its constants, so variants of one parent
have similar fingerprints. Fake metrics are a fixed smooth function of the
fingerprint; :meth:`FakeSurrogate.train` adds noise to imitate real training.

The metric names come from the fitness mode (``openarchevo.fitness``): each
metric gets a value between 40 % worse and 20 % better than its anchor, in the direction
given by ``higher_is_better``. A new fitness mode therefore works in dry runs
without changes here.
"""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Mapping, Optional, Sequence

import numpy as np

from openarchevo.contract import BLOCK_CLASS
from openarchevo.fingerprint.schema import LOWER, NUM_FEATURES, UPPER
from openarchevo.fitness import FITNESS_MODES, TaskTransform
from openarchevo.surrogate.base import Evaluation, InfeasibleProgram

_NUMBER = re.compile(r"\d+\.?\d*")


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)


class FakeSurrogate:
    """Offline stand-in for the surrogate service and for proxy training.

    Args:
        transforms: Metric name -> :class:`~openarchevo.fitness.TaskTransform`
            of the fitness mode (default: the ``composite`` mode).
        block_class: Class name every program must define.
    """

    def __init__(self, transforms: Optional[Mapping[str, TaskTransform]] = None, block_class: str = BLOCK_CLASS):
        self.transforms = dict(transforms or FITNESS_MODES["composite"])
        self.block_class = block_class
        self.context_size = 0

    def fit(self, features: Sequence[Sequence[float]], targets: dict[str, Sequence[float]]) -> None:
        self.context_size = len(features)

    def fingerprint(self, code: str) -> list[float]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise InfeasibleProgram(f"SyntaxError: {exc}") from exc
        if not any(isinstance(n, ast.ClassDef) and n.name == self.block_class for n in ast.walk(tree)):
            raise InfeasibleProgram(f"No class {self.block_class}")
        base = np.random.default_rng(_seed(_NUMBER.sub("#", code))).uniform(0.15, 0.85, NUM_FEATURES)
        consts = sum(float(x) for x in _NUMBER.findall(code))
        jitter = np.random.default_rng(_seed(f"{consts:.6f}")).uniform(-0.15, 0.15, NUM_FEATURES)
        u = np.clip(base + jitter, 0.0, 1.0)
        return (LOWER + u * (UPPER - LOWER)).tolist()

    @staticmethod
    def _value(t: TaskTransform, rel: float) -> float:
        """Metric value at relative improvement ``rel`` over the anchor."""
        return t.anchor * (1.0 + rel) if t.higher_is_better else t.anchor * (1.0 - rel)

    def predict(self, features: Sequence[Sequence[float]]) -> list[dict[str, float]]:
        out = []
        blocks = np.array_split(np.arange(NUM_FEATURES), len(self.transforms))  # features per metric
        for f in features:
            u = (np.asarray(f) - LOWER) / (UPPER - LOWER)
            pred = {}
            for i, (name, t) in enumerate(self.transforms.items()):
                score = 0.8 * u[blocks[i]].mean() + 0.2 * np.sin(5.0 * u[blocks[i][0]]) ** 2  # in [0, 1]
                pred[name] = float(self._value(t, -0.4 + 0.6 * score))
            out.append(pred)
        return out

    def evaluate(self, code: str) -> Evaluation:
        fp = self.fingerprint(code)
        return Evaluation(fingerprint=fp, metrics=self.predict([fp])[0])

    def train(self, code: str, seed_text: str) -> dict[str, float]:
        """Imitate proxy training: the fake metrics plus seeded noise (about 2 % of the anchor)."""
        metrics = self.evaluate(code).metrics
        rng = np.random.default_rng(_seed(seed_text))
        return {k: float(max(1e-6, v + rng.normal(0, 0.02 * abs(self.transforms[k].anchor))))
                for k, v in metrics.items()}
