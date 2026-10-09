"""OpenArchEvo: LLM-driven, surrogate-assisted evolutionary architecture discovery.

The top-level package is deliberately light: importing it needs only NumPy and
PyYAML. PyTorch (``openarchevo.models``, ``openarchevo.fingerprint``) and the
surrogate service dependencies (FastAPI, TabPFN) are imported lazily by the
modules that need them.
"""

from openarchevo.candidate import Archive, Candidate
from openarchevo.fitness import fitness_fn

__version__ = "0.1.0"
__all__ = ["Archive", "Candidate", "fitness_fn", "__version__"]
