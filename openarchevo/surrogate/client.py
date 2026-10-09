"""HTTP client for the surrogate service (``scripts/start_surrogate.py``)."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import httpx

from openarchevo.config import SurrogateClientConfig
from openarchevo.surrogate.base import Evaluation, InfeasibleProgram


class SurrogateClient:
    """Talks to ``/fit``, ``/fingerprint``, ``/predict`` and ``/evaluate``."""

    def __init__(self, cfg: Optional[SurrogateClientConfig] = None):
        cfg = cfg or SurrogateClientConfig()
        self.base_url = os.environ.get(cfg.url_env, cfg.default_url).rstrip("/")
        token = os.environ.get(cfg.token_env)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(timeout=cfg.timeout_s, headers=headers)

    def _post(self, path: str, payload: dict) -> dict:
        resp = self._client.post(f"{self.base_url}{path}", json=payload)
        if resp.status_code == 422:
            raise InfeasibleProgram(resp.json().get("detail", "infeasible program"))
        resp.raise_for_status()
        return resp.json()

    def fit(self, features: Sequence[Sequence[float]], targets: dict[str, Sequence[float]]) -> None:
        self._post("/fit", {"features": [list(map(float, f)) for f in features],
                            "targets": {k: list(map(float, v)) for k, v in targets.items()}})

    def fingerprint(self, code: str) -> list[float]:
        return self._post("/fingerprint", {"code": code})["fingerprint"]

    def predict(self, features: Sequence[Sequence[float]]) -> list[dict[str, float]]:
        return self._post("/predict", {"features": [list(map(float, f)) for f in features]})["predictions"]

    def evaluate(self, code: str) -> Evaluation:
        out = self._post("/evaluate", {"code": code})
        return Evaluation(fingerprint=out["fingerprint"], metrics=out["metrics"])

    def status(self) -> dict:
        resp = self._client.get(f"{self.base_url}/status")
        resp.raise_for_status()
        return resp.json()
