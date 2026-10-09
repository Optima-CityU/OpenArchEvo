"""A small FastAPI surrogate service.

Endpoints (JSON in / JSON out):

* ``POST /fit``          ``{"features": [[21 floats], ...], "targets": {"listops_acc": [...], "wikitext2_ppl": [...]}}``
* ``POST /fingerprint``  ``{"code": "..."}`` -> ``{"fingerprint": [...]}``
* ``POST /predict``      ``{"features": [[...], ...]}`` -> ``{"predictions": [{"listops_acc": ..}, ..]}``
* ``POST /evaluate``     ``{"code": "..."}`` -> ``{"fingerprint": [...], "metrics": {...}}``
* ``GET  /status``

Fingerprint extraction executes untrusted LLM-written code, so each request runs
in a fresh subprocess with a timeout, one at a time per configured device.
A program that fails to build or run is reported with HTTP 422.

**Security.** The subprocess is not a sandbox: anyone who can reach the service
can run arbitrary code on its machine. Bind it to 127.0.0.1 (the default) and
use an SSH tunnel from other machines. If a token is given (``SURROGATE_TOKEN``),
every request must send ``Authorization: Bearer <token>``. The service holds one
fitted model, so run one service per search.

Start with ``python scripts/start_surrogate.py --config configs/surrogate.yaml``.
"""

from __future__ import annotations

import hmac
import queue
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openarchevo.config import ServiceConfig
from openarchevo.fingerprint.schema import NUM_FEATURES
from openarchevo.surrogate.tabpfn_model import TabPFNPredictor


class FitRequest(BaseModel):
    features: list[list[float]]
    targets: dict[str, list[float]]


class CodeRequest(BaseModel):
    code: str


class PredictRequest(BaseModel):
    features: list[list[float]]


def create_app(cfg: ServiceConfig, predictor: Optional[TabPFNPredictor] = None,
               token: Optional[str] = None) -> FastAPI:
    from openarchevo.fingerprint.extract import FingerprintError, extract_in_subprocess

    app = FastAPI(title="OpenArchEvo surrogate")

    if token:
        expected = f"Bearer {token}"

        @app.middleware("http")
        async def require_token(request: Request, call_next):
            if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
                return JSONResponse({"detail": "missing or wrong bearer token"}, status_code=401)
            return await call_next(request)

    predictor = predictor or TabPFNPredictor(cfg.tabpfn)
    devices: "queue.Queue[str]" = queue.Queue()
    for d in cfg.fingerprint.devices:
        devices.put(d)
    fit_lock = threading.Lock()

    def _check(features: list[list[float]]) -> None:
        if any(len(f) != NUM_FEATURES for f in features):
            raise HTTPException(status_code=400, detail=f"Each fingerprint must have {NUM_FEATURES} values.")

    def _fingerprint(code: str) -> list[float]:
        device = devices.get()
        try:
            return extract_in_subprocess(code, cfg.fingerprint, device)
        except FingerprintError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            devices.put(device)

    def _predict(features: list[list[float]]) -> list[dict[str, float]]:
        with fit_lock:
            try:
                return predictor.predict(features)
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/fit")
    def fit(req: FitRequest) -> dict:
        _check(req.features)
        if any(len(v) != len(req.features) for v in req.targets.values()):
            raise HTTPException(status_code=400, detail="Each target needs one value per fingerprint.")
        with fit_lock:
            predictor.fit(req.features, req.targets)
        return {"version": predictor.version, "n": len(req.features), "targets": sorted(req.targets)}

    @app.post("/fingerprint")
    def fingerprint(req: CodeRequest) -> dict:
        return {"fingerprint": _fingerprint(req.code)}

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        _check(req.features)
        return {"predictions": _predict(req.features)}

    @app.post("/evaluate")
    def evaluate(req: CodeRequest) -> dict:
        fp = _fingerprint(req.code)
        return {"fingerprint": fp, "metrics": _predict([fp])[0]}

    @app.get("/status")
    def status() -> dict:
        return {"model_version": predictor.version, "targets": sorted(predictor.models),
                "devices": cfg.fingerprint.devices}

    return app
