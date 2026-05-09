"""Tests for breach context annotation on DecisionRecord + chain integrity."""

from __future__ import annotations

from datetime import datetime, timezone

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import SqliteEvidenceStore
from horizon_ric.sla.breach_annotation import (
    annotate_decision_record,
    extract_breach_context,
)
from horizon_ric.sla.policy import SLABreachEvent


def _versions() -> ModelVersions:
    return ModelVersions(
        encoder="e",
        risk_heads="r",
        dyna="d",
        policy="p",
        constraint_layer="c",
        rapp="0",
    )


def _record(decision_id: str = "dec-1") -> DecisionRecord:
    return DecisionRecord.new(
        decision_id=decision_id,
        rapp_instance_id="rapp-test",
        state_hash="0" * 64,
        chosen_action={"act": "x"},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.05, sla_risk_1min=0.06, sla_risk_5min=0.07
        ),
        rejected_alternatives=[],
        model_versions=_versions(),
    )


def _breach(metric: str = "latency_p99_ms") -> SLABreachEvent:
    return SLABreachEvent(
        sla_id="sla-x",
        ts_utc=datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc),
        severity="critical",
        observed_value=20.0,
        target_threshold=10.0,
        decision_id_at_breach="dec-1",
        source_metric=metric,
        lasting_s=15.0,
    )


def test_annotation_adds_breach_dicts():
    """annotate_decision_record populates `sla_breach_context`."""
    rec = _record()
    annotate_decision_record(rec, [_breach()])
    assert rec.sla_breach_context is not None
    assert len(rec.sla_breach_context) == 1
    assert rec.sla_breach_context[0]["source_metric"] == "latency_p99_ms"
    # Round-trip rebuild typed objects.
    parsed = extract_breach_context(rec)
    assert len(parsed) == 1
    assert parsed[0].sla_id == "sla-x"
    assert parsed[0].severity == "critical"


def test_annotation_with_no_breaches_clears_field():
    """An empty breach list clears the context (None on the record)."""
    rec = _record()
    annotate_decision_record(rec, [])
    assert rec.sla_breach_context is None


def test_breach_context_in_evidence_chain(tmp_path):
    """Annotated DecisionRecord round-trips through the SQLite evidence store
    and `verify()` returns -1 (chain intact)."""
    db = tmp_path / "ev.db"
    store = SqliteEvidenceStore(f"sqlite:///{db}")
    rec1 = _record("dec-a")
    annotate_decision_record(rec1, [_breach()])
    store.append(rec1)
    rec2 = _record("dec-b")
    annotate_decision_record(rec2, [_breach(metric="epfd_margin_dB")])
    store.append(rec2)

    assert store.verify() == -1
    # Read back and confirm breach context preserved.
    rows = list(store)
    assert len(rows) == 2
    assert rows[0][0].sla_breach_context is not None
    assert rows[0][0].sla_breach_context[0]["source_metric"] == "latency_p99_ms"
    assert rows[1][0].sla_breach_context[0]["source_metric"] == "epfd_margin_dB"


def test_tamper_with_breach_context_breaks_chain(tmp_path):
    """Editing the breach context after-the-fact must break the hash chain."""
    import json

    from sqlalchemy import select, update

    db = tmp_path / "ev.db"
    store = SqliteEvidenceStore(f"sqlite:///{db}")
    rec = _record("dec-tamper")
    annotate_decision_record(rec, [_breach()])
    store.append(rec)

    # Tamper: rewrite the payload row's JSON to remove the breach context.
    with store._engine.begin() as conn:  # type: ignore[attr-defined]
        row = conn.execute(select(store._table.c.seq, store._table.c.payload_json)).first()  # type: ignore[attr-defined]
        seq, payload = row[0], row[1]
        obj = json.loads(payload)
        obj["sla_breach_context"] = None
        new_payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        conn.execute(
            update(store._table).where(store._table.c.seq == seq).values(  # type: ignore[attr-defined]
                payload_json=new_payload
            )
        )
    # verify() must now return 0 (the first record is corrupt).
    assert store.verify() == 0
