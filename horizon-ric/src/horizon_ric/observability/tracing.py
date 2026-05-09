"""Real OpenTelemetry tracing wired into the rApp's decision loop.

Public surface:

* `init_tracing(service_name, otlp_endpoint=None, in_memory=False)` —
  installs a real `TracerProvider` with a real `BatchSpanProcessor`
  feeding either:
    - the OTLP gRPC exporter (production / Tempo / Jaeger / OTel
      Collector — chosen when `OTEL_EXPORTER_OTLP_ENDPOINT` is set),
    - the console exporter (default in dev), or
    - the in-memory exporter (tests; pass `in_memory=True`).

* `decision_span(decision_id, **attrs)` — context manager that opens a
  span named "rapp.decision" with the TS 28.105 §8.2 mandatory
  `decision_id` attribute set up-front, computes wall-clock latency on
  exit and writes it to `latency_ms`. Extra attributes (`sla_breach_count`,
  `constraint_violations`, anything else) are forwarded.

* `DecisionSpan` — the same as a class, for callers that want to update
  attributes mid-span.

Tests at `tests/test_otel.py` use `init_tracing(in_memory=True)` to
inspect emitted spans without needing a collector running.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

_PROVIDER: TracerProvider | None = None
_IN_MEMORY_EXPORTER: InMemorySpanExporter | None = None


def init_tracing(
    service_name: str = "horizon-ric",
    *,
    service_version: str = "0.2.0",
    otlp_endpoint: str | None = None,
    in_memory: bool = False,
) -> TracerProvider:
    """Install a TracerProvider. Idempotent; second call replaces the
    provider so tests can swap in the in-memory exporter cleanly.
    """
    global _PROVIDER, _IN_MEMORY_EXPORTER

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "service.namespace": "oran.rapp",
        }
    )
    provider = TracerProvider(resource=resource)

    if in_memory:
        _IN_MEMORY_EXPORTER = InMemorySpanExporter()
        # Use the simple processor so spans are flushed synchronously —
        # critical for tests that read them immediately on exit.
        provider.add_span_processor(SimpleSpanProcessor(_IN_MEMORY_EXPORTER))
    else:
        endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        exporter: SpanExporter
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        else:
            exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))

    # OpenTelemetry refuses to override the global provider once set.
    # In tests we want to swap exporters between cases, so we forcibly
    # reset the underlying proxy. This is documented in the SDK as the
    # supported test-reset approach.
    try:
        trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    except Exception:
        pass
    trace.set_tracer_provider(provider)
    _PROVIDER = provider
    return provider


def shutdown_tracing() -> None:
    """Flush + tear down the active provider. Safe to call repeatedly."""
    global _PROVIDER, _IN_MEMORY_EXPORTER
    if _PROVIDER is not None:
        _PROVIDER.shutdown()
        _PROVIDER = None
    _IN_MEMORY_EXPORTER = None


def get_tracer():
    return trace.get_tracer("horizon_ric.observability")


def get_in_memory_exporter() -> InMemorySpanExporter | None:
    """Test hook: returns the InMemorySpanExporter when init_tracing was
    called with in_memory=True; otherwise None.
    """
    return _IN_MEMORY_EXPORTER


class DecisionSpan:
    """Class-form decision span for callers that want to mutate
    attributes throughout the body of a decision.

    Example::

        with DecisionSpan("dec-0001") as span:
            ...
            span.set("sla_breach_count", 2)
            ...
    """

    def __init__(self, decision_id: str, **attrs: Any) -> None:
        self.decision_id = decision_id
        self._attrs = dict(attrs)
        self._span = None
        self._t0 = 0.0

    def __enter__(self) -> "DecisionSpan":
        self._t0 = time.perf_counter()
        tracer = get_tracer()
        self._span_cm = tracer.start_as_current_span(
            "rapp.decision", attributes={"decision_id": self.decision_id}
        )
        self._span = self._span_cm.__enter__()
        # Defaults so every span carries the contracted attributes even
        # if the caller never sets them.
        self._span.set_attribute("sla_breach_count", 0)
        self._span.set_attribute("constraint_violations", 0)
        for k, v in self._attrs.items():
            self._span.set_attribute(k, v)
        return self

    def set(self, key: str, value: Any) -> None:
        if self._span is not None:
            self._span.set_attribute(key, value)

    def __exit__(self, exc_type, exc, tb) -> None:
        latency_ms = (time.perf_counter() - self._t0) * 1000.0
        if self._span is not None:
            self._span.set_attribute("latency_ms", latency_ms)
            if exc is not None:
                self._span.record_exception(exc)
                from opentelemetry.trace import Status, StatusCode

                self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        if hasattr(self, "_span_cm"):
            self._span_cm.__exit__(exc_type, exc, tb)


@contextmanager
def decision_span(decision_id: str, **attrs: Any) -> Iterator[DecisionSpan]:
    """Function-form alias for `DecisionSpan(...)`."""
    with DecisionSpan(decision_id, **attrs) as ds:
        yield ds
