"""Tests for the ITU-T X.733 alarm-record schema and integrations.

Devil-A Finding #20 / Solver-2 Fix #20.

Covers:
  * Pydantic model construction + YANG notification render
  * Severity / alarm-type vocab
  * map_event_to_alarm — every shipped horizon.* event has a mapping
  * X733AlarmBus emit/subscribe + clear-on-clearing
  * Integration: circuit_breaker emits cb.opened / cb.closed
  * Integration: graceful_degradation emits degraded.entered / .recovered
  * Integration: evidence/store.verify() emits tamper_detected
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from horizon_ric.observability.x733_alarms import (
    X733Alarm,
    X733AlarmBus,
    X733AlarmType,
    X733Severity,
    default_bus,
    map_event_to_alarm,
)

# ─── pydantic model + YANG render ───────────────────────────────────────


def test_alarm_pydantic_round_trip_yang() -> None:
    a = X733Alarm(
        alarm_type=X733AlarmType.QOS,
        probable_cause="thresholdCrossed",
        perceived_severity=X733Severity.MAJOR,
        specific_problem="SLA breach predicted",
        additional_text="horizon.sla.breach",
    )
    y = a.to_yang_notification()
    assert y["alarm-type"] == "qualityOfServiceAlarm"
    assert y["perceived-severity"] == "major"
    assert y["probable-cause"] == "thresholdCrossed"
    assert y["alarm-id"] == a.alarm_id
    # event-time renders as ISO-8601 UTC
    parsed = dt.datetime.fromisoformat(y["event-time"])
    assert parsed.tzinfo is not None


def test_severity_enum_complete() -> None:
    assert {s.value for s in X733Severity} == {
        "cleared",
        "indeterminate",
        "critical",
        "major",
        "minor",
        "warning",
    }


def test_alarm_type_enum_complete() -> None:
    assert {a.value for a in X733AlarmType} == {
        "communicationsAlarm",
        "qualityOfServiceAlarm",
        "processingErrorAlarm",
        "equipmentAlarm",
        "environmentalAlarm",
    }


# ─── map_event_to_alarm ─────────────────────────────────────────────────


def test_map_event_to_alarm_unknown_raises() -> None:
    with pytest.raises(KeyError):
        map_event_to_alarm("horizon.does.not.exist")


def test_map_event_to_alarm_known_events() -> None:
    a = map_event_to_alarm(
        "horizon.cb.opened",
        additional_information={"breaker": "test"},
    )
    assert a.alarm_type == X733AlarmType.PROCESSING_ERROR
    assert a.perceived_severity == X733Severity.MAJOR
    assert a.additional_information["breaker"] == "test"
    # Cleared variant
    a2 = map_event_to_alarm("horizon.cb.closed")
    assert a2.perceived_severity == X733Severity.CLEARED


# ─── X733AlarmBus ───────────────────────────────────────────────────────


def test_bus_subscribe_and_emit_event() -> None:
    bus = X733AlarmBus()
    received: list[X733Alarm] = []
    bus.subscribe(received.append)
    bus.emit_event(
        "horizon.sla.breach",
        additional_information={"slice_id": "s-1"},
    )
    assert len(received) == 1
    assert received[0].alarm_type == X733AlarmType.QOS
    assert received[0].additional_information["slice_id"] == "s-1"


def test_bus_active_and_clearing_alarm() -> None:
    bus = X733AlarmBus()
    bus.emit_event("horizon.cb.opened", additional_information={"breaker": "smo"})
    assert len(bus.active_alarms()) == 1
    # Same managed-object/type/probable-cause cleared:
    bus.emit_event("horizon.cb.closed")
    assert len(bus.active_alarms()) == 0


# ─── integration: circuit breaker ───────────────────────────────────────


def test_circuit_breaker_emits_x733_on_open(monkeypatch) -> None:
    """Force the breaker into OPEN state and assert an X.733 alarm fires."""
    import asyncio

    import httpx

    from horizon_ric.runtime.circuit_breaker import (
        AsyncCircuitBreaker,
        BreakerConfig,
    )

    bus = X733AlarmBus()
    received: list[X733Alarm] = []
    bus.subscribe(received.append)

    monkeypatch.setattr(
        "horizon_ric.observability.x733_alarms._DEFAULT_BUS",
        bus,
    )

    async def boom() -> None:
        raise httpx.ConnectError("upstream unreachable")

    cb = AsyncCircuitBreaker(BreakerConfig(name="test", fail_max=2))

    async def drive() -> None:
        for _ in range(3):
            try:
                await cb.call(boom)
            except Exception:
                pass

    asyncio.run(drive())

    opened = [a for a in received if "horizon.cb.opened" in a.additional_text]
    assert opened, f"expected horizon.cb.opened, got {[a.additional_text for a in received]}"
    assert opened[0].alarm_type == X733AlarmType.PROCESSING_ERROR
    assert opened[0].perceived_severity == X733Severity.MAJOR


# ─── integration: graceful degradation ──────────────────────────────────


def test_graceful_degradation_emits_x733(monkeypatch) -> None:
    from horizon_ric.runtime.graceful_degradation import (
        DegradationController,
        FailureMode,
    )

    bus = X733AlarmBus()
    received: list[X733Alarm] = []
    bus.subscribe(received.append)
    monkeypatch.setattr(
        "horizon_ric.observability.x733_alarms._DEFAULT_BUS", bus
    )

    ctl = DegradationController()
    ctl.enter(FailureMode.SMO_UNREACHABLE, reason="SMO ping timeout")
    ctl.recover(reason="SMO healthy")

    texts = [a.additional_text for a in received]
    assert "horizon.degraded.entered" in texts
    assert "horizon.degraded.recovered" in texts
    entered = next(a for a in received if a.additional_text == "horizon.degraded.entered")
    cleared = next(a for a in received if a.additional_text == "horizon.degraded.recovered")
    assert entered.perceived_severity == X733Severity.MAJOR
    assert cleared.perceived_severity == X733Severity.CLEARED


# ─── integration: evidence-store tamper detection ───────────────────────


def test_evidence_store_emits_tamper_alarm(tmp_path, monkeypatch) -> None:
    from datetime import datetime, timezone

    from horizon_ric.evidence.schema import (
        DecisionRecord,
        ModelVersions,
        PredictedOutcome,
    )
    from horizon_ric.evidence.store import JsonlEvidenceStore

    bus = X733AlarmBus()
    received: list[X733Alarm] = []
    bus.subscribe(received.append)
    monkeypatch.setattr(
        "horizon_ric.observability.x733_alarms._DEFAULT_BUS", bus
    )

    def _rec(did: str) -> DecisionRecord:
        return DecisionRecord(
            decision_id=did,
            timestamp=datetime.now(timezone.utc),
            rapp_instance_id="rapp-x733",
            state_hash="0" * 64,
            chosen_action={"a": 1},
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=0.1, sla_risk_1min=0.1, sla_risk_5min=0.1
            ),
            rejected_alternatives=[],
            model_versions=ModelVersions(
                encoder="v",
                risk_heads="v",
                dyna="v",
                policy="v",
                constraint_layer="v",
                rapp="v",
            ),
        )

    p = tmp_path / "ev.jsonl"
    store = JsonlEvidenceStore(p)
    store.append(_rec("d-1"))
    store.append(_rec("d-2"))

    # Verify intact -> no alarm.
    assert store.verify() == -1
    assert not received

    # Tamper: rewrite the file with a bogus hash on line 1.
    lines = p.read_text().splitlines()
    import json as _json

    obj = _json.loads(lines[1])
    obj["hash"] = "00" * 32
    lines[1] = _json.dumps(obj)
    p.write_text("\n".join(lines) + "\n")

    bad = store.verify()
    assert bad >= 0
    tamper_alarms = [
        a for a in received if a.additional_text == "horizon.evidence.tamper_detected"
    ]
    assert tamper_alarms, "tamper detection must emit an X.733 alarm"
    assert tamper_alarms[0].perceived_severity == X733Severity.CRITICAL
    assert tamper_alarms[0].alarm_type == X733AlarmType.PROCESSING_ERROR
