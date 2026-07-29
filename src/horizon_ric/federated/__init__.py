"""Federated layer — security-focused: Byzantine-robust aggregation (poisoning
defence) and Shamir secure aggregation (privacy of individual updates).

Both are torch-free numpy / pure-Python so they run anywhere.
"""

from horizon_ric.federated.robust import (
    KrumResult,
    aggregate,
    coordinate_median,
    fedavg,
    flatten_state,
    fltrust,
    krum,
    trimmed_mean,
    unflatten_state,
)
from horizon_ric.federated.secure import (
    PRIME,
    ShamirSecretSharing,
    secure_mean,
)

__all__ = [
    "KrumResult",
    "aggregate",
    "coordinate_median",
    "fedavg",
    "flatten_state",
    "fltrust",
    "krum",
    "trimmed_mean",
    "unflatten_state",
    "PRIME",
    "ShamirSecretSharing",
    "secure_mean",
]
