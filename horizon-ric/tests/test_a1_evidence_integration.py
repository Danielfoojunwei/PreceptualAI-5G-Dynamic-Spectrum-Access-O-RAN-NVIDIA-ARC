"""A1 emit_policy persists DecisionRecord to attached evidence store."""

from pathlib import Path

import httpx
import pytest

from horizon_ric.evidence import (
    DecisionRecord,
    JsonlEvidenceStore,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig


def _record(decision_id: str) -> DecisionRecord:
    return DecisionRecord.new(
        decision_id=decision_id,
        rapp_instance_id="rapp-1",
        state_hash="abc",
        chosen_action={"slice_id": "s1", "priority": 5},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.05, sla_risk_1min=0.05, sla_risk_5min=0.05
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="enc-0.1.0",
            risk_heads="rh-0.1.0",
            dyna="dy-0.1.0",
            policy="pol-0.1.0",
            constraint_layer="cl-0.1.0",
            rapp="horizon-ric-0.1.0",
        ),
    )


@pytest.mark.asyncio
async def test_emit_persists_evidence_on_success(tmp_path: Path):
    captured: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(201)

    cfg = A1AdapterConfig()
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
    adapter.attach_evidence_store(store)

    rec = _record("d-emit-1")
    pid, status = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
        decision_record=rec,
    )
    assert status == 201
    assert pid

    assert len(store) == 1
    persisted, _ = next(iter(store))
    assert persisted.decision_id == "d-emit-1"

    await adapter.close()


@pytest.mark.asyncio
async def test_emit_does_not_persist_on_failure(tmp_path: Path):
    async def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    cfg = A1AdapterConfig()
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
    adapter.attach_evidence_store(store)

    with pytest.raises(httpx.HTTPError):
        await adapter.emit_policy(
            "horizon.qos.priority",
            {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
            decision_record=_record("d-fail-1"),
        )
    # Nothing persisted on failure
    assert len(store) == 0

    await adapter.close()


@pytest.mark.asyncio
async def test_policy_schemas_are_strict_per_type():
    """Each declared type must produce a tight, type-specific JSON schema."""
    adapter = A1Adapter()
    for type_name in adapter.cfg.policy_types:
        schema = adapter._policy_create_schema(type_name)
        assert schema["additionalProperties"] is False
        assert "required" in schema
    await adapter.close()


@pytest.mark.asyncio
async def test_unknown_policy_type_schema_raises():
    adapter = A1Adapter()
    with pytest.raises(ValueError):
        adapter._policy_create_schema("not.a.real.type")
    await adapter.close()
