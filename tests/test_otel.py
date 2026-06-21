"""OpenTelemetry tracing tests for Horizon-RIC.

Verifies that:

1. `init_tracing(in_memory=True)` installs a real provider (no mocks).
2. `decision_span(...)` emits a span named "rapp.decision" with the
   contracted TS 28.105 §8.2 / SLO attributes.
3. The latency_ms attribute is populated from a real perf_counter
   measurement (>= 0).
4. Spans flow through the FastAPI submit_policy path end-to-end.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from horizon_ric.observability import (
    decision_span,
    init_tracing,
    shutdown_tracing,
)
from horizon_ric.observability.tracing import get_in_memory_exporter


@pytest.fixture()
def memory_tracing():
    """Fresh in-memory provider per test."""
    init_tracing(service_name="horizon-ric-test", in_memory=True)
    yield get_in_memory_exporter()
    shutdown_tracing()


def test_init_tracing_installs_real_provider(memory_tracing) -> None:
    """The provider is real (not the no-op default)."""
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    # The default no-op provider is `ProxyTracerProvider` with no
    # `add_span_processor`. Ours is the SDK class.
    assert hasattr(provider, "add_span_processor"), (
        f"expected SDK TracerProvider, got {type(provider).__name__}"
    )


def test_decision_span_emits_named_span(memory_tracing) -> None:
    with decision_span("dec-0001") as ds:
        ds.set("sla_breach_count", 3)
    spans = memory_tracing.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "rapp.decision"


def test_decision_span_attributes_match_contract(memory_tracing) -> None:
    with decision_span("dec-0002") as ds:
        ds.set("sla_breach_count", 2)
        ds.set("constraint_violations", 1)
    span = memory_tracing.get_finished_spans()[0]
    attrs = dict(span.attributes)
    assert attrs["decision_id"] == "dec-0002"
    assert attrs["sla_breach_count"] == 2
    assert attrs["constraint_violations"] == 1
    assert "latency_ms" in attrs and attrs["latency_ms"] >= 0.0


def test_decision_span_latency_is_real(memory_tracing) -> None:
    """The latency_ms must reflect actual elapsed wall-clock — not zero."""
    with decision_span("dec-0003"):
        time.sleep(0.005)  # 5 ms
    span = memory_tracing.get_finished_spans()[0]
    latency = float(span.attributes["latency_ms"])
    assert latency >= 4.0, f"expected ~5ms, got {latency}"


def test_decision_span_records_exception_status(memory_tracing) -> None:
    with pytest.raises(ValueError):
        with decision_span("dec-err"):
            raise ValueError("boom")
    spans = memory_tracing.get_finished_spans()
    assert len(spans) == 1
    sp = spans[0]
    # OTel sets status to ERROR + records the exception event.
    assert any(ev.name == "exception" for ev in sp.events)


def test_submit_policy_end_to_end_emits_span(memory_tracing) -> None:
    """Posting a policy through the real FastAPI app emits a real span
    with all four contracted attributes populated.
    """
    from horizon_ric.rapp.api_v1 import build_api, issue_token

    app = build_api()
    client = TestClient(app)
    token = issue_token("op", scopes=["read", "write"])
    rsp = client.post(
        "/api/v1/policies",
        headers={"Authorization": f"Bearer {token}"},
        json={"action_type": "reroute", "params": {"gateway": "G2"}},
    )
    assert rsp.status_code == 201, rsp.text
    spans = memory_tracing.get_finished_spans()
    rapp_spans = [s for s in spans if s.name == "rapp.decision"]
    assert len(rapp_spans) == 1
    attrs = dict(rapp_spans[0].attributes)
    for required in (
        "decision_id",
        "latency_ms",
        "sla_breach_count",
        "constraint_violations",
    ):
        assert required in attrs, f"missing {required}: {attrs}"
    assert attrs["decision_id"].startswith("dec-")
