"""Offline stand-ins for the LLM and the embedder (used by ``--dry-run``).

``FakeLLM`` reads the strongest program from the prompt (the last ``python``
block) and perturbs one or two floating-point constants, which is enough to
exercise parsing, feasibility checks, de-duplication, islands and selection
without any network access.
"""

from __future__ import annotations

import hashlib
import random
import re
import threading

import numpy as np

from openarchevo.llm.base import Proposal

_CODE_BLOCK = re.compile(r"```python\n(.*?)\n```", re.S)
_FLOAT = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")
_IDEAS = [
    "sharpen the spike-driven gate", "slow down the state decay", "rebalance the residual branch",
    "lower the firing threshold of the controller", "widen the recurrent memory", "damp the output gate",
    "add activity-dependent control of the update", "stabilise the chunked recurrence",
]


class FakeLLM:
    """Deterministic (seeded) mutation of the best parent program."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._count = 0

    def propose(self, system_prompt: str, user_prompt: str) -> Proposal:
        blocks = _CODE_BLOCK.findall(user_prompt)
        if not blocks:
            raise ValueError("No program found in prompt.")
        code = blocks[-1]
        with self._lock:
            self._count += 1
            n, rng_seed = self._count, self._rng.random()
        rng = random.Random(rng_seed)
        floats = list(_FLOAT.finditer(code))
        changes = []
        for m in rng.sample(floats, k=min(len(floats), rng.choice([1, 2]))):
            old = float(m.group(1))
            new = round(old * rng.uniform(0.5, 1.5) + rng.uniform(0.0, 0.05), 4)
            changes.append((m.span(1), f"{new}"))
        for (start, end), text in sorted(changes, reverse=True):
            code = code[:start] + text + code[end:]
        if not changes:
            code += f"\n# variant {n}\n"
        idea = rng.choice(_IDEAS)
        rationale = f"Variant {n}: {idea}; adjusted {len(changes)} constant(s) of the strongest parent."
        return Proposal(name=f"spiking_fla_variant_{n}", rationale=rationale, code=code)


class HashEmbedder:
    """Signed feature-hashing bag-of-words embedding (deterministic, no network)."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dimensions)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dimensions] += 1.0 if (h >> 64) & 1 else -1.0
        return vec.tolist()
