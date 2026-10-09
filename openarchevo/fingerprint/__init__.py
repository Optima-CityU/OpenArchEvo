"""Behavioural fingerprint (21 features).

``schema`` is torch-free. ``extract`` (zero-cost probes + structural statistics)
needs PyTorch and is imported explicitly by the surrogate service::

    from openarchevo.fingerprint.extract import extract_fingerprint
"""

from openarchevo.fingerprint.schema import FEATURE_KEYS, NUM_FEATURES, scale, to_vector

__all__ = ["FEATURE_KEYS", "NUM_FEATURES", "scale", "to_vector"]
