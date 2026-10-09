"""TabPFN-2.5 regressors that predict task metrics from fingerprints.

One regressor per task metric is fitted *in context* on the archive (TabPFN does
not train weights; ``fit`` stores the labelled context). Perplexity is modelled
in log space. The TabPFN checkpoint is **not** shipped: download the TabPFN-2.5
regressor weights from the official source (https://github.com/PriorLabs/TabPFN),
accept its license, and point ``TABPFN_MODEL_PATH`` at the file.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from openarchevo.config import TabPFNConfig


class TabPFNPredictor:
    def __init__(self, cfg: TabPFNConfig):
        self.cfg = cfg
        self.model_path = os.environ.get(cfg.model_path_env)
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Set {cfg.model_path_env} to a TabPFN regressor checkpoint downloaded from the "
                "official TabPFN distribution (weights are not redistributed with OpenArchEvo)."
            )
        self.models: dict = {}
        self.version = 0

    def fit(self, features: Sequence[Sequence[float]], targets: dict[str, Sequence[float]]) -> None:
        from tabpfn import TabPFNRegressor

        x = np.asarray(features, dtype=np.float64)
        models = {}
        for name, values in targets.items():
            y = np.asarray(values, dtype=np.float64)
            if name in self.cfg.log_targets:
                if (y <= 0).any():
                    raise ValueError(f"{name} must be positive for the log transform.")
                y = np.log(y)
            model = TabPFNRegressor(model_path=self.model_path, device=self.cfg.device,
                                    random_state=self.cfg.random_state)
            model.fit(x, y)
            models[name] = model
        self.models = models
        self.version += 1

    def predict(self, features: Sequence[Sequence[float]]) -> list[dict[str, float]]:
        if not self.models:
            raise RuntimeError("Surrogate has not been fitted; call /fit first.")
        x = np.asarray(features, dtype=np.float64)
        columns = {}
        for name, model in self.models.items():
            pred = np.asarray(model.predict(x), dtype=np.float64)
            columns[name] = np.exp(pred) if name in self.cfg.log_targets else pred
        return [{name: float(col[i]) for name, col in columns.items()} for i in range(len(x))]
