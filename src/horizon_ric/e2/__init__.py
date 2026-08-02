"""E2 interface companions — real E2SM service-model ingest for Horizon.

This package bridges measurements produced by a real near-RT RIC E2
termination (FlexRIC's nearRT-RIC + E2 agents, or any O-RAN compliant
E2 node) into the Horizon telemetry bus. It deliberately contains NO
E2AP/SCTP transport of its own: the near-RT RIC owns the E2
association; Horizon consumes the E2SM payloads it surfaces.

The E2SM-RC side (:mod:`horizon_ric.e2.rc_control`) keeps that same
boundary: it *constructs* RIC Control Header / Control Message payloads
from a Shield disposition — failing closed on a refused action — and
leaves their delivery to the near-RT RIC's own E2 termination.
"""

from horizon_ric.e2.kpm_bridge import (
    KpmBridgeError,
    KpmMeasurementBridge,
    e2_to_pipeline,
)
from horizon_ric.e2.rc_control import (
    RcControlError,
    RcControlRequest,
    control_from_disposition,
    control_message_parameters,
    decode_control_header,
    decode_control_message,
    encode_control_header,
    encode_control_message,
    rc_spec,
)

__all__ = [
    "KpmBridgeError",
    "KpmMeasurementBridge",
    "RcControlError",
    "RcControlRequest",
    "control_from_disposition",
    "control_message_parameters",
    "decode_control_header",
    "decode_control_message",
    "e2_to_pipeline",
    "encode_control_header",
    "encode_control_message",
    "rc_spec",
]
