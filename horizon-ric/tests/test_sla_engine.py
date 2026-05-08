"""Tests for the SLA evaluation engine + persistence round-trip."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine

from horizon_ric.sla.engine import SLAEvaluator
from horizon_ric.sla.policy import (
    SLA,
    SLABreachEvent,
    SLOTarget,
    create_schema,
    list_breaches,
    load_slas,
    persist_breach,
    persist_sla,
)


def _sla_latency(window: int = 5) -> SLA:
    return SLA(
        id="sla-test-latency",
        name="latency p99",
        targets=[
            SLOTarget(
                metric="latency_p99_ms",
                comparison="<=",
                threshold=10.0,
                window_s=window,
            )
        ],
        severity_levels={"critical": {"metrics": ["latency_p99_ms"]}},
    )


def test_single_breach_after_window():
    """Breach event fires only after sustained `window_s` seconds."""
    ev = SLAEvaluator(slas=[_sla_latency(window=5)])
    # Tick at t=0 — over threshold but no window elapsed yet.
    out0 = ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=0.0)
    assert out0 == []
    # Tick at t=4 — still pending.
    out1 = ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=4.0)
    assert out1 == []
    # Tick at t=6 — window elapsed, ONE event fires.
    out2 = ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=6.0)
    assert len(out2) == 1
    assert out2[0].sla_id == "sla-test-latency"
    assert out2[0].severity == "critical"
    assert out2[0].source_metric == "latency_p99_ms"
    assert out2[0].observed_value == 25.0


def test_sustained_breach_dedups_to_one_event():
    """A sustained breach yields the SAME breach id across ticks."""
    ev = SLAEvaluator(slas=[_sla_latency(window=2)])
    ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=0.0)
    out1 = ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=3.0)
    out2 = ev.evaluate({"latency_p99_ms": 27.0}, None, now_mono=10.0)
    out3 = ev.evaluate({"latency_p99_ms": 30.0}, None, now_mono=20.0)
    assert len(out1) == len(out2) == len(out3) == 1
    assert out1[0].id == out2[0].id == out3[0].id
    assert out3[0].lasting_s >= 20.0
    assert out3[0].observed_value == 30.0  # latest sample


def test_recovery_clears_breach_state():
    """After recovery, the next breach is a NEW event id."""
    ev = SLAEvaluator(slas=[_sla_latency(window=1)])
    ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=0.0)
    first = ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=2.0)
    assert len(first) == 1
    first_id = first[0].id
    # Recover — observation passes.
    out_ok = ev.evaluate({"latency_p99_ms": 5.0}, None, now_mono=3.0)
    assert out_ok == []
    assert ev.active_breaches() == []
    # Re-breach.
    ev.evaluate({"latency_p99_ms": 50.0}, None, now_mono=4.0)
    second = ev.evaluate({"latency_p99_ms": 50.0}, None, now_mono=10.0)
    assert len(second) == 1
    assert second[0].id != first_id


def test_multi_sla_evaluation():
    """Two SLAs, two metrics — both can fire independently."""
    sla_a = _sla_latency(window=1)
    sla_b = SLA(
        id="sla-test-epfd",
        name="EPFD margin",
        targets=[
            SLOTarget(
                metric="epfd_margin_dB",
                comparison=">=",
                threshold=5.0,
                window_s=0,  # instantaneous
            )
        ],
        severity_levels={"critical": {"metrics": ["epfd_margin_dB"]}},
    )
    ev = SLAEvaluator(slas=[sla_a, sla_b])
    ev.evaluate({"latency_p99_ms": 50.0, "epfd_margin_dB": 2.0}, None, now_mono=0.0)
    out = ev.evaluate(
        {"latency_p99_ms": 50.0, "epfd_margin_dB": 2.0}, None, now_mono=2.0
    )
    sla_ids = sorted(b.sla_id for b in out)
    assert sla_ids == ["sla-test-epfd", "sla-test-latency"]


def test_severity_resolution_default_warning():
    """If `severity_levels` doesn't list the metric, default to 'warning'."""
    sla = SLA(
        id="sla-warn",
        name="default sev",
        targets=[
            SLOTarget(
                metric="throughput_mbps",
                comparison=">=",
                threshold=100.0,
                window_s=0,
            )
        ],
    )
    ev = SLAEvaluator(slas=[sla])
    ev.evaluate({"throughput_mbps": 50.0}, None, now_mono=0.0)
    out = ev.evaluate({"throughput_mbps": 50.0}, None, now_mono=1.0)
    assert len(out) == 1
    assert out[0].severity == "warning"


def test_decision_id_threaded_into_breach():
    """The breach event carries the decision id at breach time."""
    from horizon_ric.evidence.schema import (
        DecisionRecord,
        ModelVersions,
        PredictedOutcome,
    )

    rec = DecisionRecord.new(
        decision_id="dec-abc123",
        rapp_instance_id="rapp-test",
        state_hash="0" * 64,
        chosen_action={"a": 1},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.05, sla_risk_1min=0.06, sla_risk_5min=0.07
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="e",
            risk_heads="r",
            dyna="d",
            policy="p",
            constraint_layer="c",
            rapp="0",
        ),
    )
    ev = SLAEvaluator(slas=[_sla_latency(window=0)])
    out = ev.evaluate({"latency_p99_ms": 99.0}, rec, now_mono=0.0)
    assert len(out) == 1
    assert out[0].decision_id_at_breach == "dec-abc123"


def test_persistence_round_trip(tmp_path):
    """SLA + breach round-trip cleanly through SQLite."""
    db = tmp_path / "sla.db"
    engine = create_engine(f"sqlite:///{db}", future=True)
    create_schema(engine)
    sla = _sla_latency(window=5)
    persist_sla(engine, sla)
    loaded = load_slas(engine)
    assert len(loaded) == 1
    assert loaded[0].id == sla.id
    assert loaded[0].targets[0].threshold == 10.0
    assert loaded[0].targets[0].comparison == "<="

    breach = SLABreachEvent(
        sla_id=sla.id,
        severity="critical",
        observed_value=25.0,
        target_threshold=10.0,
        source_metric="latency_p99_ms",
        lasting_s=12.0,
    )
    persist_breach(engine, breach)
    rows = list_breaches(engine, sla_id=sla.id)
    assert len(rows) == 1
    assert rows[0].id == breach.id
    assert rows[0].observed_value == 25.0
    # Update the same breach (sustained) — still one row.
    breach.lasting_s = 30.0
    breach.observed_value = 27.0
    persist_breach(engine, breach)
    rows2 = list_breaches(engine, sla_id=sla.id)
    assert len(rows2) == 1
    assert rows2[0].lasting_s == 30.0
    assert rows2[0].observed_value == 27.0


def test_evaluator_from_db(tmp_path):
    """SLAEvaluator.from_db loads at boot."""
    db = tmp_path / "sla.db"
    engine = create_engine(f"sqlite:///{db}", future=True)
    create_schema(engine)
    persist_sla(engine, _sla_latency(window=0))
    ev = SLAEvaluator.from_db(engine)
    assert len(ev.slas) == 1
    out = ev.evaluate({"latency_p99_ms": 30.0}, None, now_mono=0.0)
    assert len(out) == 1


def test_pending_breach_does_not_emit():
    """During the pending window, no events emit."""
    ev = SLAEvaluator(slas=[_sla_latency(window=10)])
    for t in (0.0, 1.0, 5.0, 9.9):
        out = ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=t)
        assert out == []


def test_metric_missing_from_observation_no_state_change():
    """A tick that doesn't include the SLA metric must not affect state."""
    ev = SLAEvaluator(slas=[_sla_latency(window=1)])
    # Trigger a pending breach.
    ev.evaluate({"latency_p99_ms": 25.0}, None, now_mono=0.0)
    # Tick without that metric — state preserved.
    ev.evaluate({"throughput_mbps": 100.0}, None, now_mono=0.5)
    out = ev.evaluate({"latency_p99_ms": 26.0}, None, now_mono=2.0)
    assert len(out) == 1


def test_sla_requires_at_least_one_target():
    """Pydantic validator rejects an SLA with empty targets."""
    with pytest.raises(Exception):
        SLA(name="empty", targets=[])
