"""Standard I/O contracts and connector framework.

This package is the **stable extension surface** for PreceptualAI. To add a
new data source, modality, sink, or transport, write a class that
implements one of the abstract bases here and register it in the
`registry` — every other component (encoder, world model, A1 emitter,
audit log) reads/writes through these interfaces and never against
provider-specific shapes.

Layered design:

    Schemas (Pydantic v2)         what data flows
        TelemetryEvent            one observation from any source
        FeatureFrame              encoder-ready batch of events
        PolicyAction              actor output to a Near-RT RIC
        AuditRecord               evidence-store payload

    Connector ABCs (sync + async)  how data flows
        Source                    pulls/streams TelemetryEvents
        Sink                      consumes any of the above
        BiConnector               bidirectional (e.g. gRPC, Kafka req/reply)

    Registry                       look-up by name from config
        register_source(name, cls), register_sink(name, cls)
        get_source(name, **opts) -> Source

Reference connectors (`horizon_ric.io.connectors`):
    file       JSONL on disk (telemetry replay)
    http       REST polling source / webhook sink
    grpc       streaming RPC source/sink
    kafka      Kafka consumer/producer (optional `confluent-kafka`)

Plugin discovery: connectors registered via the
`horizon_ric.connectors` setuptools entry-point are auto-loaded at
import time, so an external package can ship a connector without
touching this codebase.
"""

from horizon_ric.io.connector import (
    BiConnector,
    Connector,
    ConnectorConfig,
    ConnectorConfigError,
    ConnectorError,
    ConnectorIOError,
    ConnectorState,
    Sink,
    Source,
    with_retry,
)
from horizon_ric.io.registry import (
    ConnectorRegistry,
    get_sink,
    get_source,
    list_connectors,
    register_sink,
    register_source,
    registry,
)
from horizon_ric.io.schemas import (
    AuditRecord,
    FeatureFrame,
    Modality,
    PolicyAction,
    TelemetryEvent,
)

__all__ = [
    "AuditRecord",
    "BiConnector",
    "Connector",
    "ConnectorConfig",
    "ConnectorConfigError",
    "ConnectorError",
    "ConnectorIOError",
    "ConnectorRegistry",
    "ConnectorState",
    "FeatureFrame",
    "Modality",
    "PolicyAction",
    "Sink",
    "Source",
    "TelemetryEvent",
    "get_sink",
    "get_source",
    "list_connectors",
    "register_sink",
    "register_source",
    "registry",
    "with_retry",
]
