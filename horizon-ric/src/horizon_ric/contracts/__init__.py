"""Abstract base contracts that the platform core depends on.

These define the public API surface that any vertical adapter (NTN+AI-RAN
today; power-grid, fleet, etc. later) must implement.
"""

from horizon_ric.contracts.constraint_layer import ConstraintLayer, ConstraintViolation
from horizon_ric.contracts.domain_adapter import DomainAdapter
from horizon_ric.contracts.metric_suite import MetricSuite

__all__ = [
    "ConstraintLayer",
    "ConstraintViolation",
    "DomainAdapter",
    "MetricSuite",
]
