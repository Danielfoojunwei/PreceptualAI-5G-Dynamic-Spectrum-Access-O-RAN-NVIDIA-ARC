"""Tests for the NIS2 24h/72h/1m incident reporter (Devil-A Fix #10).

Validates:
  * Manual ``trigger_incident`` POSTs to operator + CSIRT and persists
    to the evidence sink.
  * Audit-chain integrity auto-trigger fires only when ``verify()``
    returns non-(-1).
  * Critical-SLA-breach auto-trigger fires only above threshold.
  * Auth-failure-rate auto-trigger fires only when ratio > threshold
    over the 5-minute window.
  * Milestones are computed correctly (+24h/+72h/+30d UTC).
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import httpx
import pytest

from horizon_ric.security.nis2_reporter import (
    NIS2IncidentReporter,
    NIS2Milestones,
)


def _frozen_clock(when: dt.datetime):
    return lambda: when


class _RecordingTransport(httpx.MockTransport):
    """MockTransport that records every call."""

    def __init__(self, status: int = 200) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode()) if request.content else {}
            self.calls.append((str(request.url), body))
            return httpx.Response(status, json={"ok": True})

        super().__init__(handler)


def _build_reporter(
    *,
    when: dt.datetime,
    transport: httpx.MockTransport,
    evidence: list[dict[str, Any]] | None = None,
) -> NIS2IncidentReporter:
    return NIS2IncidentReporter(
        operator_endpoint="https://operator.example/nis2",
        csirt_endpoint="https://csirt.example/nis2",
        client=httpx.Client(transport=transport),
        evidence_sink=(evidence.append if evidence is not None else None),
        clock=_frozen_clock(when),
    )


# ─── milestones ─────────────────────────────────────────────────────────


class TestMilestones:
    def test_milestones_at_24_72_720_hours(self):
        t0 = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        m = NIS2Milestones.from_trigger(t0)
        assert (m.early_warning_due - t0).total_seconds() == 24 * 3600
        assert (m.incident_notification_due - t0).total_seconds() == 72 * 3600
        assert (m.final_report_due - t0).total_seconds() == 30 * 24 * 3600

    def test_naive_datetime_is_treated_as_utc(self):
        naive = dt.datetime(2026, 5, 6, 12, 0, 0)
        m = NIS2Milestones.from_trigger(naive)
        assert m.triggered_at.tzinfo is not None


# ─── manual trigger ─────────────────────────────────────────────────────


class TestManualTrigger:
    def test_trigger_posts_to_both_endpoints(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        tr = _RecordingTransport()
        ev: list[dict[str, Any]] = []
        rep = _build_reporter(when=when, transport=tr, evidence=ev)
        inc = rep.trigger_incident(
            severity="critical",
            scope="A1 emit pipeline",
            evidence={"reason": "test"},
            trigger_kind="manual",
        )
        # Both endpoints called.
        urls = {u for u, _ in tr.calls}
        assert "https://operator.example/nis2" in urls
        assert "https://csirt.example/nis2" in urls
        # Audit chain captured the payload at least once.
        assert len(ev) >= 1
        assert ev[0]["incident_id"] == inc.incident_id
        # Milestones present in the payload.
        body = tr.calls[0][1]
        assert "milestones" in body
        assert body["milestones"]["incident_notification_due"].endswith("+00:00")

    def test_trigger_persists_delivery_failure(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)

        def failing(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"err": "boom"})

        tr = httpx.MockTransport(failing)
        ev: list[dict[str, Any]] = []
        rep = _build_reporter(when=when, transport=tr, evidence=ev)
        rep.trigger_incident(
            severity="high",
            scope="x",
            evidence={},
        )
        # 1 incident + 2 delivery failures (operator + CSIRT)
        kinds = [e.get("kind") for e in ev]
        assert kinds.count("nis2.delivery_failure") == 2


# ─── audit-chain auto trigger ───────────────────────────────────────────


class _FakeStore:
    def __init__(self, verify_result: int) -> None:
        self._r = verify_result

    def verify(self) -> int:
        return self._r


class TestAuditChainAutoTrigger:
    def test_no_trigger_when_chain_intact(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        tr = _RecordingTransport()
        rep = _build_reporter(when=when, transport=tr)
        out = rep.auto_check_audit_chain(_FakeStore(verify_result=-1))
        assert out is None
        assert tr.calls == []

    def test_trigger_when_chain_broken(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        tr = _RecordingTransport()
        rep = _build_reporter(when=when, transport=tr)
        out = rep.auto_check_audit_chain(_FakeStore(verify_result=42))
        assert out is not None
        assert out.severity == "critical"
        assert out.trigger_kind == "audit_chain_integrity"
        assert out.evidence["first_invalid_index"] == 42
        assert len(tr.calls) == 2  # operator + CSIRT


# ─── SLA breach auto trigger ────────────────────────────────────────────


class TestSlaBreachAutoTrigger:
    def test_below_threshold_does_not_fire(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        tr = _RecordingTransport()
        rep = _build_reporter(when=when, transport=tr)
        out = rep.auto_check_sla_breach(
            breach_probability=0.5, slice_id="slice-1"
        )
        assert out is None

    def test_above_threshold_fires(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        tr = _RecordingTransport()
        rep = _build_reporter(when=when, transport=tr)
        out = rep.auto_check_sla_breach(
            breach_probability=0.99, slice_id="slice-1"
        )
        assert out is not None
        assert out.trigger_kind == "sla_breach"


# ─── auth-failure-rate auto trigger ─────────────────────────────────────


class TestAuthFailureRateAutoTrigger:
    def test_below_threshold_does_not_fire(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        tr = _RecordingTransport()
        rep = _build_reporter(when=when, transport=tr)
        for _ in range(99):
            rep.record_auth_event(success=True, when=when)
        rep.record_auth_event(success=False, when=when)  # 1% exactly
        # Threshold is "> 1%"; exactly 1.0% does not fire.
        out = rep.auto_check_auth_failure_rate()
        assert out is None

    def test_above_threshold_fires(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        tr = _RecordingTransport()
        rep = _build_reporter(when=when, transport=tr)
        for _ in range(95):
            rep.record_auth_event(success=True, when=when)
        for _ in range(5):
            rep.record_auth_event(success=False, when=when)  # 5% failure
        out = rep.auto_check_auth_failure_rate()
        assert out is not None
        assert out.trigger_kind == "auth_failure_rate"
        assert out.evidence["failure_rate_pct"] > 1.0

    def test_too_few_samples_does_not_fire(self):
        when = dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc)
        tr = _RecordingTransport()
        rep = _build_reporter(when=when, transport=tr)
        for _ in range(3):
            rep.record_auth_event(success=False, when=when)
        out = rep.auto_check_auth_failure_rate()
        assert out is None  # noise floor
