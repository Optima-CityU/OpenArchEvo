"""NSGA-II environmental selection on two maximised objectives.

Used twice per outer iteration: to choose inner-loop roots on
``[y(a), N(a; D)]`` and to choose the training batch on ``[\\hat y(a), N(a; P_f)]``.
"""

from __future__ import annotations

import numpy as np


def non_dominated_fronts(objectives: np.ndarray) -> list[list[int]]:
    """Partition rows into Pareto fronts (all objectives maximised).

    Args:
        objectives: Array of shape ``(n, m)``.

    Returns:
        A list of fronts, each a list of row indices; front 0 is non-dominated.
    """
    obj = np.asarray(objectives, dtype=float)
    n = len(obj)
    if n == 0:
        return []
    no_worse = np.all(obj[:, None, :] >= obj[None, :, :], axis=2)
    better = np.any(obj[:, None, :] > obj[None, :, :], axis=2)
    dominates = no_worse & better  # dominates[i, j]: i dominates j
    counts = dominates.sum(axis=0)
    assigned = np.zeros(n, dtype=bool)
    fronts: list[list[int]] = []
    current = np.flatnonzero(counts == 0)
    while current.size:
        fronts.append(current.tolist())
        assigned[current] = True
        counts = counts - dominates[current].sum(axis=0)
        current = np.flatnonzero((counts == 0) & ~assigned)
    return fronts


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Crowding distance of the rows of one front (boundary points get ``inf``)."""
    obj = np.asarray(objectives, dtype=float)
    n, m = obj.shape
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(obj[:, j], kind="stable")
        lo, hi = obj[order[0], j], obj[order[-1], j]
        dist[order[0]] = dist[order[-1]] = np.inf
        if hi == lo:
            continue
        gaps = (obj[order[2:], j] - obj[order[:-2], j]) / (hi - lo)
        dist[order[1:-1]] += gaps
    return dist


def nsga2_select(objectives: np.ndarray, k: int) -> list[int]:
    """Select ``k`` row indices by non-dominated sorting and crowding truncation."""
    obj = np.asarray(objectives, dtype=float)
    if k >= len(obj):
        return list(range(len(obj)))
    selected: list[int] = []
    for front in non_dominated_fronts(obj):
        if len(selected) + len(front) <= k:
            selected.extend(front)
            continue
        dist = crowding_distance(obj[front])
        order = np.argsort(-dist, kind="stable")
        selected.extend(front[i] for i in order[: k - len(selected)])
        break
    return selected


def nondominated(objectives: np.ndarray) -> list[int]:
    """Indices of the non-dominated rows."""
    fronts = non_dominated_fronts(objectives)
    return fronts[0] if fronts else []
