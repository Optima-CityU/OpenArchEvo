"""The 21-dimensional behavioural fingerprint: feature order and scaling bounds.

Torch-free so that similarity, the surrogate client and dry runs can import it.

Order matters: the surrogate is fitted on vectors in exactly this order, and the
fixed scaling bounds below are aligned with it. The bounds ``[l_j, u_j]`` were
calibrated once (5th/95th percentiles over a reference set of paired
architectures) and are kept fixed during search; fingerprint similarity maps
each feature with ``c * (2 * (x - l) / (u - l) - 1)``, ``c = 0.9``, without
clipping, before taking cosine similarity.
"""

from __future__ import annotations

import numpy as np

#: Seven initialization-time probes (forward / forward+backward on a fixed batch).
PROBE_KEYS = [
    "zcp_synflow",    # SynFlow
    "zcp_gradnorm",   # GradNorm
    "zcp_snip",       # SNIP
    "zcp_jacob_cov",  # Jacobian covariance (condition-number form)
    "zcp_naswot",     # NASWOT log-determinant
    "zcp_swap",       # SWSP: sample-wise spiking patterns
    "zcp_fire_rate",  # mean spike activity
]

#: Four module statistics and ten structural statistics.
ARCH_KEYS = [
    "arch_total_params",
    "arch_depth_proxy",
    "arch_num_spiking_linear",
    "arch_num_conv1d",
    "arch_gate_layer_ratio",
    "arch_num_expand",
    "arch_num_contract",
    "arch_bottleneck_ratio",
    "arch_max_expand_ratio",
    "arch_avg_ffn_expansion",
    "arch_ffn_param_ratio",
    "arch_ffn_macs_ratio",
    "arch_macs_per_param",
    "arch_largest_subgraph_ratio",
]

FEATURE_KEYS = PROBE_KEYS + ARCH_KEYS
NUM_FEATURES = len(FEATURE_KEYS)  # 21

#: Fixed lower / upper scaling bounds, aligned with FEATURE_KEYS.
LOWER = np.array([
    117.794808, 0.791154, 1.756847, 1.496996, -107.093773, 23218.0, 0.033485,
    60907.6, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.03125, 2.0, 1.547368, 0.18435,
    0.185301, 5589.004253, 0.153261,
])
UPPER = np.array([
    130.987782, 10.19703, 45.04537, 52.011767, -13.112961, 56936.4, 0.118608,
    216662.4, 13.0, 9.0, 4.0, 0.25, 3.35, 5.0, 1.0, 5.4, 4.0, 0.99325,
    1.0, 6392.949196, 0.971021,
])

SCALE = 0.9


def to_vector(features: dict[str, float]) -> np.ndarray:
    """Order a feature dict into the canonical 21-dim vector."""
    return np.array([float(features[k]) for k in FEATURE_KEYS], dtype=np.float64)


def scale(x: np.ndarray) -> np.ndarray:
    """Fixed feature-wise affine map of ``[LOWER, UPPER]`` to ``[-SCALE, SCALE]``."""
    x = np.asarray(x, dtype=np.float64)
    return SCALE * (2.0 * (x - LOWER) / (UPPER - LOWER) - 1.0)
