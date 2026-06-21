"""O-RAN integration: R1 + A1 + O1 adapters + rApp lifecycle.

R1 = rApp ↔ SMO/Non-RT RIC (registration, service exposure).
A1 = Non-RT → Near-RT RIC (ML policy delivery, PolicyStatus, A1-EI).
O1 = SMO ↔ E2 nodes (NETCONF/YANG, ncclient).

All three adapters honour the WG11 security spec via `AuthConfig`
(mTLS + OAuth2). Health + Prometheus endpoints are bound by the
lifecycle on boot — see `horizon_ric.rapp.health`.
"""

from horizon_ric.rapp.a1_adapter import DEFAULT_POLICY_TYPES, A1Adapter, A1AdapterConfig
from horizon_ric.rapp.auth import AuthConfig, build_secure_async_client
from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle, RAppState
from horizon_ric.rapp.r1_adapter import R1Adapter, R1AdapterConfig

__all__ = [
    "A1Adapter",
    "A1AdapterConfig",
    "AuthConfig",
    "DEFAULT_POLICY_TYPES",
    "HorizonRAppLifecycle",
    "R1Adapter",
    "R1AdapterConfig",
    "RAppState",
    "build_secure_async_client",
]

# O1 is gated on ncclient being importable.
try:
    from horizon_ric.rapp.o1_adapter import (  # noqa: F401  (re-exported)
        O1Adapter,
        O1AdapterConfig,
    )

    __all__.extend(["O1Adapter", "O1AdapterConfig"])
except ImportError:  # pragma: no cover
    pass
