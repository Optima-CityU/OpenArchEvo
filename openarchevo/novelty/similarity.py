"""Three-view similarity between architecture programs.

    s(a1, a2) = 1/3 * [ s_code + s_rationale + s_fingerprint ]
    d(a1, a2) = 1 - s(a1, a2)

* ``s_code``: CodeBLEU (Python, equal weights on its four components), averaged
  over both reference/candidate directions. A ``difflib`` token-ratio backend is
  available for dependency-free dry runs.
* ``s_rationale``: cosine similarity of rationale embeddings.
* ``s_fingerprint``: cosine similarity of the scaled 21-dim fingerprints
  (``openarchevo.fingerprint.schema.scale``); zero if either vector is zero.

Pairwise components are cached by candidate id, optionally on disk, because
CodeBLEU is the dominant cost of a round.
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import multiprocessing
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from openarchevo.candidate import Candidate
from openarchevo.fingerprint.schema import scale

VIEWS = ("code", "rationale", "fingerprint")


def cosine(u: Optional[Sequence[float]], v: Optional[Sequence[float]]) -> float:
    """Cosine similarity; 0 when a vector is missing or has zero norm."""
    if u is None or v is None:
        return 0.0
    a, b = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if a.size == 0 or a.shape != b.shape:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def fingerprint_similarity(fp1: Optional[Sequence[float]], fp2: Optional[Sequence[float]]) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return cosine(scale(np.asarray(fp1)), scale(np.asarray(fp2)))


def codebleu_similarity(code1: str, code2: str) -> float:
    """Symmetric CodeBLEU (requires ``pip install codebleu tree-sitter-python``)."""
    from codebleu import calc_codebleu

    weights = (0.25, 0.25, 0.25, 0.25)
    s12 = calc_codebleu([code1], [code2], lang="python", weights=weights)["codebleu"]
    s21 = calc_codebleu([code2], [code1], lang="python", weights=weights)["codebleu"]
    return float((s12 + s21) / 2)


_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|\S")


def difflib_similarity(code1: str, code2: str) -> float:
    """Token-level ``SequenceMatcher`` ratio: a dependency-free stand-in for CodeBLEU."""
    return difflib.SequenceMatcher(None, _TOKEN.findall(code1), _TOKEN.findall(code2), autojunk=False).ratio()


CODE_BACKENDS: dict[str, Callable[[str, str], float]] = {
    "codebleu": codebleu_similarity,
    "difflib": difflib_similarity,
}


def _code_similarity(backend: str, code1: str, code2: str) -> float:
    """Module-level so that it can run in worker processes."""
    return float(CODE_BACKENDS[backend](code1, code2))


class SimilarityModel:
    """Computes and caches three-view similarities between candidates.

    Args:
        code_backend: ``"codebleu"`` (paper) or ``"difflib"`` (dry runs).
        duplicate_eps: Two programs are near-duplicates if any single view has
            similarity ``>= 1 - duplicate_eps``.
        cache_path: Optional JSON file for persisting pairwise components.
        num_workers: Processes for batched code similarity (CodeBLEU is
            CPU-bound); ``0`` computes in the calling thread.
    """

    def __init__(self, code_backend: str = "codebleu", duplicate_eps: float = 1e-6,
                 cache_path: Optional[Path] = None, num_workers: int = 0):
        if code_backend == "codebleu" and importlib.util.find_spec("codebleu") is None:
            raise ImportError("code_backend='codebleu' needs `pip install codebleu==0.7.0 tree-sitter-python==0.21.0`; "
                              "use code_backend='difflib' for a dependency-free run.")
        self.code_backend = code_backend
        self.duplicate_eps = duplicate_eps
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._pool = (ProcessPoolExecutor(num_workers, mp_context=multiprocessing.get_context("spawn"))
                      if num_workers > 1 else None)
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    @staticmethod
    def _key(a: Candidate, b: Candidate) -> str:
        return "|".join(sorted((a.id, b.id)))

    def _fill(self, pairs: Sequence[tuple[Candidate, Candidate]]) -> None:
        """Compute and cache the components of all uncached pairs (code views in parallel)."""
        with self._lock:
            todo = {self._key(a, b): (a, b) for a, b in pairs if a.id != b.id and self._key(a, b) not in self._cache}
        if not todo:
            return
        items = list(todo.items())
        args = ([self.code_backend] * len(items), [a.code for _, (a, _) in items], [b.code for _, (_, b) in items])
        if self._pool is not None and len(items) > 1:
            code = list(self._pool.map(_code_similarity, *args, chunksize=max(1, len(items) // 64)))
        else:
            code = list(map(_code_similarity, *args))
        with self._lock:
            for (key, (a, b)), s_code in zip(items, code):
                self._cache[key] = [s_code, cosine(a.embedding, b.embedding),
                                    fingerprint_similarity(a.fingerprint, b.fingerprint)]

    def components(self, a: Candidate, b: Candidate) -> tuple[float, float, float]:
        """``(s_code, s_rationale, s_fingerprint)`` for a pair of candidates."""
        if a.id == b.id:
            return (1.0, 1.0, 1.0)
        self._fill([(a, b)])
        with self._lock:
            return tuple(self._cache[self._key(a, b)])  # type: ignore[return-value]

    def similarity(self, a: Candidate, b: Candidate) -> float:
        """Equal-weight three-view similarity ``s(a, b)``."""
        return float(np.mean(self.components(a, b)))

    def is_near_duplicate(self, a: Candidate, b: Candidate) -> bool:
        """True if any single view says the two programs are (almost) identical."""
        return max(self.components(a, b)) >= 1.0 - self.duplicate_eps

    def has_near_duplicate(self, a: Candidate, others: Sequence[Candidate]) -> bool:
        """True if ``a`` nearly duplicates any of ``others``."""
        self._fill([(a, b) for b in others])
        return any(self.is_near_duplicate(a, b) for b in others)

    def matrix(self, left: Sequence[Candidate], right: Optional[Sequence[Candidate]] = None) -> np.ndarray:
        """Similarity matrix ``S[i, j] = s(left[i], right[j])`` (``right`` defaults to ``left``)."""
        right = left if right is None else right
        self._fill([(a, b) for a in left for b in right])
        return np.array([[self.similarity(a, b) for b in right] for a in left]).reshape(len(left), len(right))

    def save(self, keep_ids: Optional[set[str]] = None) -> None:
        """Persist the cache; with ``keep_ids``, only pairs whose two ids are both in it."""
        if self.cache_path is None:
            return
        with self._lock:
            cache = self._cache if keep_ids is None else {
                k: v for k, v in self._cache.items() if all(i in keep_ids for i in k.split("|"))}
            payload = json.dumps(cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(payload)
