"""OpenTelemetry tracing for Horizon-RIC.

One span per decision; attributes mirror the TS 28.105 §8 inference
report fields and the rApp SLO contract:

  decision_id            (TS 28.105 §8.2 — required for traceability)
  latency_ms             (operator SLO — see deploy/SLO.md)
  sla_breach_count       (TS 28.552 §6.3.1 — KPI emission)
  constraint_violations  (policy guard count, per emit_guards.py)

The OTLP exporter is configured via `OTEL_EXPORTER_OTLP_ENDPOINT`
(default: stdout exporter when unset, so dev runs are visible without
a collector). Span timing uses real wall-clock — no mocks.
"""
from horizon_ric.observability.tracing import (
    DecisionSpan,
    decision_span,
    get_tracer,
    init_tracing,
    shutdown_tracing,
)

__all__ = [
    "DecisionSpan",
    "decision_span",
    "get_tracer",
    "init_tracing",
    "shutdown_tracing",
]
