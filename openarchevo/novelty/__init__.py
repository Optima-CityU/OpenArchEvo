"""Novelty and three-view similarity.

Novelty of ``a`` relative to a fixed reference set ``R`` is its mean
dissimilarity to the other members of ``R``::

    N(a; R) = mean_{a' in R, a' != a} d(a, a'),   d = 1 - s
"""

from __future__ import annotations

import numpy as np

from openarchevo.novelty.similarity import SimilarityModel, cosine


def novelty_within(sim: np.ndarray) -> np.ndarray:
    """Novelty of every member of a reference set from its square similarity matrix.

    The diagonal (self-similarity) is excluded, as in ``N(a; R \\ {a})``.
    """
    sim = np.asarray(sim, dtype=np.float64)
    n = len(sim)
    if n < 2:
        return np.zeros(n)
    off_diag_sum = sim.sum(axis=1) - np.diag(sim)
    return 1.0 - off_diag_sum / (n - 1)


__all__ = ["SimilarityModel", "cosine", "novelty_within"]
