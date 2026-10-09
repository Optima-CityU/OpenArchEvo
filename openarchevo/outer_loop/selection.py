"""Selection steps of the outer loop.

* :func:`select_roots`    NSGA-II on ``[y(a), N(a; D_{t-1})]`` -> inner-loop roots.
* :func:`top_candidates`  island bests, stepping stones, then global best by ``\\hat y``.
* :func:`most_dissimilar` remaining candidates with the lowest mean similarity to the top set.
* :func:`screen_against`  drop near-duplicates of already evaluated architectures.
* :func:`select_batch`    NSGA-II on ``[\\hat y(a), N(a; P_f)]`` -> training batch.
* :func:`archive_front`   non-dominated archive members under ``[y, N(a; D)]``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from openarchevo.candidate import Candidate
from openarchevo.novelty import novelty_within
from openarchevo.novelty.similarity import SimilarityModel
from openarchevo.nsga2 import nondominated, nsga2_select


def _with_novelty(cands: Sequence[Candidate], similarity: SimilarityModel, use_measured: bool) -> np.ndarray:
    nov = novelty_within(similarity.matrix(list(cands)))
    fit = [c.fitness if use_measured else c.predicted_fitness for c in cands]
    for c, n in zip(cands, nov):
        c.meta["novelty"] = float(n)
    return np.column_stack([fit, nov])


def select_roots(archive: Sequence[Candidate], similarity: SimilarityModel, k: int) -> list[Candidate]:
    """Roots ``P_t0``: ``k`` archive members by measured fitness and archive novelty."""
    cands = list(archive)
    return [cands[i] for i in nsga2_select(_with_novelty(cands, similarity, use_measured=True), k)]


def top_candidates(pool: Sequence[Candidate], k: int, best_per_island: int = 2,
                   stepping_stones: int = 5) -> list[Candidate]:
    """``P_top``: per-island bests, stepping stones that raised the best-so-far ``\\hat y``,
    then the globally best candidates, up to ``k``."""
    by_score = sorted(pool, key=lambda c: c.predicted_fitness, reverse=True)
    if len(pool) <= k:
        return by_score
    island_best: list[Candidate] = []
    for island in sorted({c.island for c in pool}, key=str):
        island_best += [c for c in by_score if c.island == island][:best_per_island]
    stones, best = [], -np.inf
    for c in sorted(pool, key=lambda c: c.sample or 0):
        if c.predicted_fitness > best:
            best = c.predicted_fitness
            stones.append(c)
    stones = sorted(stones, key=lambda c: c.predicted_fitness, reverse=True)[:stepping_stones]

    chosen: dict[str, Candidate] = {}
    for c in sorted(island_best, key=lambda c: c.predicted_fitness, reverse=True) + stones + by_score:
        if len(chosen) >= k:
            break
        chosen.setdefault(c.id, c)
    return sorted(chosen.values(), key=lambda c: c.predicted_fitness, reverse=True)


def most_dissimilar(pool: Sequence[Candidate], refs: Sequence[Candidate], similarity: SimilarityModel,
                    k: int) -> list[Candidate]:
    """``P_dissim``: ``k`` non-reference candidates with the lowest mean similarity to ``refs``."""
    ref_ids = {c.id for c in refs}
    rest = [c for c in pool if c.id not in ref_ids]
    if k <= 0 or not rest or not refs:
        return []
    mean_sim = similarity.matrix(rest, list(refs)).mean(axis=1)
    return [rest[i] for i in np.argsort(mean_sim, kind="stable")[:k]]


def screen_against(cands: Sequence[Candidate], evaluated: Sequence[Candidate],
                   similarity: SimilarityModel) -> list[Candidate]:
    """Remove candidates that nearly duplicate an evaluated architecture."""
    return [c for c in cands if not similarity.has_near_duplicate(c, evaluated)]


def drop_near_duplicates(cands: Sequence[Candidate], similarity: SimilarityModel) -> list[Candidate]:
    """Greedy de-duplication: keep a candidate only if it is no near-duplicate of one kept before it."""
    kept: list[Candidate] = []
    for c in cands:
        if not similarity.has_near_duplicate(c, kept):
            kept.append(c)
    return kept


def select_batch(cands: Sequence[Candidate], similarity: SimilarityModel, k: int) -> list[Candidate]:
    """Training batch ``Q_t``: ``k`` candidates by predicted fitness and novelty within ``P_f``."""
    if k <= 0 or not cands:
        return []
    objectives = _with_novelty(cands, similarity, use_measured=False)
    return [cands[i] for i in nsga2_select(objectives, k)]


def archive_front(archive: Sequence[Candidate], similarity: SimilarityModel) -> list[Candidate]:
    """``ND_D(D)``: archive members non-dominated in measured fitness and archive novelty."""
    cands = list(archive)
    if not cands:
        return []
    return [cands[i] for i in nondominated(_with_novelty(cands, similarity, use_measured=True))]
