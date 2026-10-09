"""Island populations for the inner loop.

* **Seeding**: roots are spectrally clustered into ``M`` islands with the
  three-view similarity as affinity.
* **Parent sampling**: with probability ``inter_island_prob`` one parent from
  each of ``k`` islands, otherwise ``k`` parents from one island. Within an
  island, parents are drawn by rank-based selection (``p ~ rank^-alpha``) on a
  score that mixes the predicted-fitness (``yhat``) rank with a
  dominance-dissimilarity rank (MEoH-style): being dominated by a near-clone is
  penalised more than being dominated by a dissimilar program. The island best
  is always the top choice.
* **Registration**: a new program joins the island with the greatest mean
  similarity (empty islands first); near-duplicates of any active member are
  rejected.
* **Survival**: when an island reaches ``2 * capacity`` it keeps ``capacity``
  members by the same mixed score (its best member always survives).
* **Reset / migration**: every ``reset_interval_s`` the weaker half of the
  islands (by best predicted fitness) is cleared and each cleared island is
  re-seeded with the best member of a randomly chosen surviving island.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Optional, Sequence

import numpy as np

from openarchevo.candidate import Candidate
from openarchevo.novelty.similarity import SimilarityModel


def _norm_rank(values: np.ndarray) -> np.ndarray:
    """Ranks scaled to [0, 1], 1 = best (largest value)."""
    n = len(values)
    if n <= 1:
        return np.ones(n)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return (n - 1 - ranks) / (n - 1)


def spectral_assign(sim: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Cluster labels from a similarity matrix (negative similarities clipped to 0)."""
    if n_clusters <= 1 or len(sim) <= n_clusters:
        return np.arange(len(sim)) % max(n_clusters, 1)
    from sklearn.cluster import SpectralClustering

    affinity = np.clip((sim + sim.T) / 2, 0.0, None)
    return SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans",
                              random_state=seed).fit_predict(affinity)


class IslandModel:
    def __init__(self, similarity: SimilarityModel, n_islands: int, capacity: Optional[int],
                 inter_island_prob: float = 0.5, selection_pressure: float = 1.0,
                 fitness_rank_weight: float = 0.2, reset_interval_s: Optional[float] = 3600.0, seed: int = 0):
        self.similarity = similarity
        self.capacity = capacity
        self.inter_island_prob = inter_island_prob
        self.alpha = selection_pressure
        self.w_fit = fitness_rank_weight
        self.reset_interval_s = reset_interval_s
        self.seed = seed
        self.islands: list[list[Candidate]] = [[] for _ in range(n_islands)]
        self._home: dict[str, int] = {}
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        self._last_reset = time.monotonic()

    # ------------------------------------------------------------------ seeding
    def seed_islands(self, roots: Sequence[Candidate]) -> None:
        labels = spectral_assign(self.similarity.matrix(list(roots)), len(self.islands), self.seed)
        for cand, label in zip(roots, labels):
            self._add(cand, int(label))

    def _add(self, cand: Candidate, island: int) -> None:
        cand.island = island
        self.islands[island].append(cand)
        self._home[cand.id] = island

    # ------------------------------------------------------------------ scoring
    def _mixed_scores(self, members: list[Candidate]) -> np.ndarray:
        """Fitness rank mixed with the dominance-dissimilarity rank; island best -> +inf."""
        scores = np.array([c.yhat for c in members])
        best = int(np.argmax(scores))
        sim = self.similarity.matrix(members)
        to_best = sim[best]
        # i dominates j: higher fitness AND less similar to the island best.
        dominates = (scores[:, None] > scores[None, :]) & (to_best[:, None] < to_best[None, :])
        penalty = np.where(dominates, -sim, 0.0).sum(axis=0)
        mixed = self.w_fit * _norm_rank(scores) + (1 - self.w_fit) * _norm_rank(penalty)
        mixed[best] = np.inf
        return mixed

    def _select_within(self, members: list[Candidate], k: int) -> list[Candidate]:
        k = min(k, len(members))
        mixed = self._mixed_scores(members)
        order = np.argsort(-mixed, kind="stable")
        rank = np.empty(len(members))
        rank[order] = np.arange(1, len(members) + 1)
        weights = rank ** (-self.alpha)
        chosen: list[Candidate] = []
        pool = list(range(len(members)))
        for _ in range(k):
            w = np.array([weights[i] for i in pool])
            pick = self._rng.choices(pool, weights=w / w.sum())[0]
            chosen.append(members[pick])
            pool.remove(pick)
        return chosen

    # ------------------------------------------------------------------ public API
    def sample_parents(self, k: int) -> list[Candidate]:
        """Sample ``k`` parents within one island or across ``k`` islands."""
        with self._lock:
            nonempty = [list(isl) for isl in self.islands if isl]
            across = len(nonempty) > 1 and self._rng.random() < self.inter_island_prob
            picked_islands = self._rng.sample(nonempty, min(k, len(nonempty))) if across else [self._rng.choice(nonempty)]
        if across:
            return [self._select_within(isl, 1)[0] for isl in picked_islands]
        return self._select_within(picked_islands[0], k)

    def register(self, cand: Candidate) -> Optional[int]:
        """Add ``cand`` to its most similar island; ``None`` if it is a near-duplicate."""
        with self._lock:
            snapshot = [list(isl) for isl in self.islands]
        members = [m for isl in snapshot for m in isl]
        if self.similarity.has_near_duplicate(cand, members):
            return None
        empty = [i for i, isl in enumerate(snapshot) if not isl]
        if empty:
            target = self._rng.choice(empty)
        else:
            means = np.array([self.similarity.matrix([cand], isl).mean() for isl in snapshot])
            ties = np.flatnonzero(means == means.max()).tolist()
            from_parents = [i for i in ties if i in {self._home.get(p) for p in cand.parent_ids}]
            target = self._rng.choice(from_parents or ties)
        with self._lock:
            # re-check against members that other workers added since the snapshot
            seen = {id(m) for m in members}
            if self.similarity.has_near_duplicate(cand, [m for isl in self.islands for m in isl if id(m) not in seen]):
                return None
            self._add(cand, target)
            if self.capacity is not None and len(self.islands[target]) >= 2 * self.capacity:
                self._survive(target)
        return target

    def _survive(self, island: int) -> None:
        members = self.islands[island]
        keep = np.argsort(-self._mixed_scores(members), kind="stable")[: self.capacity]
        self.islands[island] = [members[i] for i in keep]

    def maybe_reset(self) -> bool:
        """Clear the weaker half of the islands if the reset interval has elapsed."""
        if self.reset_interval_s is None or len(self.islands) < 2:
            return False
        with self._lock:
            if time.monotonic() - self._last_reset < self.reset_interval_s:
                return False
            self._last_reset = time.monotonic()
            best = [max((c.yhat for c in isl), default=-np.inf) for isl in self.islands]
            order = np.argsort([b + 1e-9 * self._rng.random() for b in best])  # random tie-break
            n_reset = len(self.islands) // 2
            weak, strong = order[:n_reset].tolist(), order[n_reset:].tolist()
            for i in weak:
                donor = self.islands[self._rng.choice(strong)]
                self.islands[i] = []
                if donor:
                    founder = max(donor, key=lambda c: c.yhat)
                    self.islands[i].append(founder)
            return True

    def sizes(self) -> list[int]:
        with self._lock:
            return [len(isl) for isl in self.islands]
