"""E2 interface companions — real E2SM service-model ingest for Horizon.

This package bridges measurements produced by a real near-RT RIC E2
termination (FlexRIC's nearRT-RIC + E2 agents, or any O-RAN compliant
E2 node) into the Horizon telemetry bus. It deliberately contains NO
E2AP/SCTP transport of its own: the near-RT RIC owns the E2
association; Horizon consumes the E2SM payloads it surfaces.
"""

from horizon_ric.e2.kpm_bridge import (
    KpmBridgeError,
    KpmMeasurementBridge,
    e2_to_pipeline,
)

__all__ = [
    "KpmBridgeError",
    "KpmMeasurementBridge",
    "e2_to_pipeline",
]
