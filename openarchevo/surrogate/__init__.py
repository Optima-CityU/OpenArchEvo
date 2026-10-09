"""Surrogate: fingerprint extraction + TabPFN prediction of task metrics.

``client.SurrogateClient`` and ``fake.FakeSurrogate`` are light; the service
(``service.create_app``) needs FastAPI, PyTorch and TabPFN.
"""

from openarchevo.surrogate.base import Evaluation, InfeasibleProgram, Surrogate
from openarchevo.surrogate.client import SurrogateClient
from openarchevo.surrogate.fake import FakeSurrogate

__all__ = ["Evaluation", "FakeSurrogate", "InfeasibleProgram", "Surrogate", "SurrogateClient"]
