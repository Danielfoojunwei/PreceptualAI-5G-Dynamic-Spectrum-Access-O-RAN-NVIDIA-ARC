"""Tests for the Alertmanager v2 webhook client.

Uses `httpx.MockTransport` to intercept the POST and assert the JSON
body matches the published v2 OpenAPI schema.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from horizon_ric.sla.alertmanager import (
    AlertManagerClient,
    format_breach_for_alertmanager,
)
from horizon_ric.sla.policy import SLABreachEvent


def _breach(severity: str = "critical") -> SLABreachEvent:
    return SLABreachEvent(
        id="breach-test-0001",
        sla_id="sla-test-1",
        ts_utc=datetime(2026, 5, 6, 12, 34, 56, 789000, tzinfo=timezone.utc),
        severity=severity,
        observed_value=25.0,
        target_threshold=10.0,
        decision_id_at_breach="dec-abc123",
        source_metric="latency_p99_ms",
        lasting_s=42.5,
    )


def test_payload_matches_v2_schema():
    """Per Alertmanager v2 OpenAPI: postableAlert requires labels (map),
    annotations (map), startsAt (date-time), endsAt (date-time)."""
    breach = _breach()
    alert = format_breach_for_alertmanager(breach, sla_name="latency", tenant="t1")
    # required keys per v2 spec
    assert set(alert.keys()) >= {"labels", "annotations", "startsAt", "endsAt"}
    # labels are <name,string>
    for k, v in alert["labels"].items():
        assert isinstance(k, str) and isinstance(v, str)
    for k, v in alert["annotations"].items():
        assert isinstance(k, str) and isinstance(v, str)
    # alertname is required by Alertmanager
    assert alert["labels"]["alertname"] == "SLABreach_latency_p99_ms"
    assert alert["labels"]["severity"] == "critical"
    assert alert["labels"]["sla_id"] == "sla-test-1"
    assert alert["labels"]["tenant"] == "t1"
    # startsAt is RFC-3339 with Z
    assert alert["startsAt"].endswith("Z")
    assert alert["startsAt"] == "2026-05-06T12:34:56.789Z"
    # endsAt is the v2 zero-time sentinel for "open"
    assert alert["endsAt"] == "0001-01-01T00:00:00Z"


def test_post_to_alertmanager_uses_v2_path():
    """The client POSTs to /api/v2/alerts."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        captured["method"] = req.method
        captured["body"] = json.loads(req.content)
        captured["content_type"] = req.headers.get("content-type")
        return httpx.Response(200, json={"status": "success"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    am = AlertManagerClient("http://am.example:9093", client=client)
    resp = am.route_breach_to_alertmanager(_breach())
    assert resp.status_code == 200
    assert captured["url"] == "http://am.example:9093/api/v2/alerts"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    # body must be a list of alert objects (v2 spec)
    assert isinstance(captured["body"], list) and len(captured["body"]) == 1


def test_payload_carries_decision_id_annotation():
    """`decision_id` in annotations enables linking to the audit chain."""
    alert = format_breach_for_alertmanager(_breach())
    assert alert["annotations"]["decision_id"] == "dec-abc123"
    assert alert["annotations"]["observed_value"] == "25.0"
    assert alert["annotations"]["threshold"] == "10.0"


def test_batch_post():
    """Multiple breaches go in a single POST."""
    posted: list = []

    def handler(req: httpx.Request) -> httpx.Response:
        posted.extend(json.loads(req.content))
        return httpx.Response(200)

    am = AlertManagerClient(
        "http://am:9093", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    breaches = [_breach() for _ in range(3)]
    resp = am.route_breaches(breaches)
    assert resp is not None and resp.status_code == 200
    assert len(posted) == 3


def test_severity_passes_through_labels():
    """Severity label round-trips for warning / info."""
    for sev in ("info", "warning", "critical"):
        alert = format_breach_for_alertmanager(_breach(severity=sev))
        assert alert["labels"]["severity"] == sev


def test_tenant_label_only_when_set():
    """When the client has no tenant configured, no `tenant` label leaks."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    am = AlertManagerClient(
        "http://am:9093",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        tenant=None,
    )
    am.route_breach_to_alertmanager(_breach())
    labels = captured["body"][0]["labels"]
    assert "tenant" not in labels
    # alertname is mandatory
    assert "alertname" in labels


def test_generator_url_is_optional_and_passes_through():
    """generatorURL field is optional but spec-supported."""
    alert = format_breach_for_alertmanager(
        _breach(), generator_url="http://horizon-ric/audit/dec-abc123"
    )
    assert alert["generatorURL"] == "http://horizon-ric/audit/dec-abc123"
    # Without it, the field is omitted (cleaner payload).
    alert2 = format_breach_for_alertmanager(_breach())
    assert "generatorURL" not in alert2
