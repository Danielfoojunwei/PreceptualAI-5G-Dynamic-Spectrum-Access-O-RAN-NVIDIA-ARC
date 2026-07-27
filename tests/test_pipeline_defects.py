"""Regression tests for the confirmed pipeline runtime defects.

Covers, against REAL loopback sockets (no httpx.MockTransport for the wire):

  * breaker-open emits land on the audit chain (CircuitBreakerError caught);
  * EvidencePersistError — policy live on the RIC, store append failed →
    accepted=True + evidence_persist_failed=True, distinct audit handling;
  * enforcement-poll tolerance — non-JSON body / non-dict JSON never fail an
    already-accepted emit;
  * HORIZON_A1_DRY_RUN kill switch — no A1 traffic, auditable dry-run row;
  * HORIZON_A1_REQUIRE_CERT emit gate;
  * honest rules-based model_versions provenance;
  * Ed25519-signed certificates flowing into the audit record;
  * decision latency excludes the enforcement-poll wait.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request, Response

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.io.schemas import TelemetryEvent
from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig, EvidencePersistError
from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig
from horizon_ric.runtime.circuit_breaker import CircuitBreakerError
from tests.test_pipeline_wiring import _build_a1_app, _event, _new_state, _serve


def _adapter(port: int) -> A1Adapter:
    return A1Adapter(
        A1AdapterConfig(near_rt_ric_base_url=f"http://127.0.0.1:{port}")
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HORIZON_A1_DRY_RUN", "HORIZON_A1_REQUIRE_CERT", "HORIZON_CERT_SIGNING_KEY_PATH"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# 1. Breaker-open emits must not vanish from the audit chain
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_breaker_open_emit_lands_on_audit_chain(tmp_path: Path):
    # Nothing listens on this port: every PUT raises ConnectError, which
    # counts as a breaker failure. After fail_max=5 the breaker opens and
    # the 6th call raises CircuitBreakerError instead of an httpx error —
    # the pipeline must record it as emit_failed, not crash.
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]

    adapter = _adapter(dead_port)
    store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
    adapter.attach_evidence_store(store)
    pipeline = DecisionPipeline(adapter, store, PipelineConfig())
    try:
        results = []
        for i in range(6):
            results.append(
                await pipeline.process_event(_event(event_id=f"evt-cb-{i}"))
            )

        assert adapter.circuit_breaker.state == "open"
        # Prove the breaker really rejects with CircuitBreakerError now
        # (raised before any socket I/O while OPEN).
        with pytest.raises(CircuitBreakerError):
            await adapter.emit_policy(
                "horizon.qos.priority",
                {"scope": {"slice_id": "s"}, "qos_objectives": {"priority": 1}},
            )
    finally:
        await adapter.close()

    # Every decision failed the emit, and EVERY one is on the audit chain —
    # including the breaker-rejected ones after the circuit opened.
    assert all(r.emit_failed and not r.accepted for r in results)
    assert len(store) == 6
    for record, _ in store:
        assert record.chosen_action["emit_failed"] is True


# ---------------------------------------------------------------------------
# 2. EvidencePersistError — policy live, store append failed
# ---------------------------------------------------------------------------
class _FailingStore(JsonlEvidenceStore):
    """A real JSONL store whose appends fail like a full disk."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.append_attempts = 0

    def append(self, record):  # noqa: D102
        self.append_attempts += 1
        raise OSError(28, "No space left on device")


@pytest.mark.asyncio
async def test_evidence_persist_failure_is_distinct_from_emit_failure(tmp_path: Path):
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        store = _FailingStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(
            adapter, store, PipelineConfig(status_poll_attempts=1, status_poll_interval_s=0.01)
        )
        try:
            result = await pipeline.process_event(_event(event_id="evt-df"))
        finally:
            await adapter.close()

    # The policy IS live on the RIC and counted as emitted...
    assert len(state["policies"]) == 1
    assert adapter.policies_emitted_count() == 1
    # ...and the pipeline reports accepted-but-unaudited, not emit_failed.
    assert result.accepted is True
    assert result.emit_failed is False
    assert result.evidence_persist_failed is True
    assert result.policy_id == state["policies"][0]["policy_id"]
    assert result.http_status == 202
    # Both the adapter append and the pipeline's best-effort retry ran.
    assert store.append_attempts == 2


@pytest.mark.asyncio
async def test_adapter_raises_evidence_persist_error_with_policy_id(tmp_path: Path):
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        store = _FailingStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        try:
            with pytest.raises(EvidencePersistError) as excinfo:
                await adapter.emit_policy(
                    "horizon.qos.priority",
                    {
                        "scope": {"slice_id": "slice-x"},
                        "qos_objectives": {"priority": 5},
                    },
                    decision_record=_minimal_record(),
                )
        finally:
            await adapter.close()

    err = excinfo.value
    assert err.policy_id
    assert err.policy_type == "horizon.qos.priority"
    assert err.http_status == 202
    # The policy really shipped before the store blew up.
    assert state["policies"][0]["policy_id"] == err.policy_id


def _minimal_record():
    from horizon_ric.evidence.schema import (
        DecisionRecord,
        ModelVersions,
        PredictedOutcome,
    )

    return DecisionRecord.new(
        decision_id="dec-min",
        rapp_instance_id="test",
        state_hash="0" * 64,
        chosen_action={"policy_type": "horizon.qos.priority"},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.05, sla_risk_1min=0.05, sla_risk_5min=0.05
        ),
        rejected_alternatives=[],
        constraint_corrections=[],
        model_versions=ModelVersions(
            encoder="none",
            risk_heads="risk_rules_v1",
            dyna="none",
            policy="risk_band_rules_v1",
            constraint_layer="shield_terrestrial_v1",
            rapp="0.2.0",
        ),
        random_seed=1,
    )


# ---------------------------------------------------------------------------
# 3. Enforcement-poll tolerance — never fail an accepted emit
# ---------------------------------------------------------------------------
def _app_with_status(status_body: bytes, media_type: str, state) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.put("/A1-P/v2/policytypes/{ptid}")
    async def put_type(ptid: int, request: Request) -> Response:
        return Response(status_code=201)

    @app.put("/A1-P/v2/policytypes/{ptid}/policies/{pid}")
    async def put_policy(ptid: int, pid: str, request: Request) -> Response:
        state["policies"].append({"policy_id": pid})
        return Response(status_code=202)

    @app.get("/A1-P/v2/policytypes/{ptid}/policies/{pid}/status")
    async def get_status(ptid: int, pid: str) -> Response:
        state["status_calls"].append(pid)
        return Response(content=status_body, media_type=media_type)

    return app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,media_type",
    [
        (b"ENFORCED", "text/plain"),  # non-JSON body → json.JSONDecodeError
        (b'["ENFORCED"]', "application/json"),  # JSON but not an object
        (b"null", "application/json"),  # JSON null → not a dict
    ],
)
async def test_poll_never_fails_accepted_emit(tmp_path: Path, body: bytes, media_type: str):
    state = _new_state()
    with _serve(_app_with_status(body, media_type, state)) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(
            adapter,
            store,
            PipelineConfig(status_poll_attempts=2, status_poll_interval_s=0.01),
        )
        try:
            result = await pipeline.process_event(_event(event_id="evt-status"))
        finally:
            await adapter.close()

    assert result.accepted is True
    assert result.emit_failed is False
    assert result.enforcement_status is None
    assert result.enforced is None
    assert len(state["status_calls"]) == 2  # both attempts ran, tolerated
    assert len(store) == 1  # adapter-persisted evidence intact


# ---------------------------------------------------------------------------
# 4. Latency excludes the enforcement-poll wait
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_latency_excludes_enforcement_poll(tmp_path: Path):
    state = _new_state()
    slow_s = 0.35

    app = FastAPI(docs_url=None, redoc_url=None)

    @app.put("/A1-P/v2/policytypes/{ptid}/policies/{pid}")
    async def put_policy(ptid: int, pid: str, request: Request) -> Response:
        state["policies"].append({"policy_id": pid})
        return Response(status_code=202)

    @app.get("/A1-P/v2/policytypes/{ptid}/policies/{pid}/status")
    async def get_status(ptid: int, pid: str) -> dict[str, str]:
        await asyncio.sleep(slow_s)
        return {"enforceStatus": "ENFORCED"}

    with _serve(app) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(adapter, store, PipelineConfig())
        try:
            result = await pipeline.process_event(_event(event_id="evt-lat"))
        finally:
            await adapter.close()

    assert result.accepted is True
    assert result.enforced is True
    # The poll took at least `slow_s`; the decision latency must not
    # include it, and the poll duration is reported separately.
    assert result.poll_duration_ms >= slow_s * 1000.0 * 0.9
    assert result.latency_ms < slow_s * 1000.0 * 0.9


# ---------------------------------------------------------------------------
# 5. HORIZON_A1_DRY_RUN kill switch
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dry_run_kill_switch(tmp_path: Path, monkeypatch):
    from horizon_ric.rapp.pipeline import HORIZON_DRY_RUN_DECISIONS

    before = HORIZON_DRY_RUN_DECISIONS._value.get()
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(adapter, store, PipelineConfig())
        monkeypatch.setenv("HORIZON_A1_DRY_RUN", "true")
        try:
            result = await pipeline.process_event(_event(event_id="evt-dry"))
        finally:
            await adapter.close()

    # No A1 traffic at all — not even a status poll.
    assert state["policies"] == []
    assert state["status_calls"] == []
    assert adapter.policies_emitted_count() == 0

    assert result.dry_run is True
    assert result.accepted is False
    assert result.blocked is False
    assert result.emit_failed is False
    assert result.policy_id is None

    # The suppressed decision is still auditable.
    assert len(store) == 1
    record, _ = next(iter(store))
    assert record.chosen_action["dry_run"] is True
    assert HORIZON_DRY_RUN_DECISIONS._value.get() == before + 1


# ---------------------------------------------------------------------------
# 6. HORIZON_A1_REQUIRE_CERT emit gate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_require_cert_gate_refuses_uncertified_emits(monkeypatch):
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        monkeypatch.setenv("HORIZON_A1_REQUIRE_CERT", "1")
        payload = {
            "scope": {"slice_id": "slice-x"},
            "qos_objectives": {"priority": 5},
        }
        try:
            with pytest.raises(ValueError, match="no safety_certificate"):
                await adapter.emit_policy("horizon.qos.priority", payload)

            import dataclasses as _dc

            from tests.test_shield_signing import _certificate

            unsafe = _dc.replace(_certificate(), safe=False, emit_blocked=True)
            with pytest.raises(ValueError, match="not clean"):
                await adapter.emit_policy(
                    "horizon.qos.priority", payload, safety_certificate=unsafe
                )

            ok_cert = _certificate()
            policy_id, status = await adapter.emit_policy(
                "horizon.qos.priority", payload, safety_certificate=ok_cert
            )
        finally:
            await adapter.close()

    # Only the certified emit reached the RIC.
    assert [p["policy_id"] for p in state["policies"]] == [policy_id]
    assert status == 202


@pytest.mark.asyncio
async def test_pipeline_passes_certificate_through_gate(tmp_path: Path, monkeypatch):
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(
            adapter, store, PipelineConfig(status_poll_interval_s=0.01)
        )
        monkeypatch.setenv("HORIZON_A1_REQUIRE_CERT", "1")
        try:
            result = await pipeline.process_event(_event(event_id="evt-gate"))
        finally:
            await adapter.close()

    assert result.accepted is True
    assert len(state["policies"]) == 1


# ---------------------------------------------------------------------------
# 7. Honest model_versions provenance
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_model_versions_are_honest_rules_identifiers(tmp_path: Path):
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(
            adapter, store, PipelineConfig(status_poll_interval_s=0.01)
        )
        try:
            await pipeline.process_event(_event(event_id="evt-mv"))
        finally:
            await adapter.close()

    record, _ = next(iter(store))
    mv = record.model_versions
    assert mv.encoder == "none"
    assert mv.risk_heads == "risk_rules_v1"
    assert mv.dyna == "none"
    assert mv.policy == "risk_band_rules_v1"
    assert mv.constraint_layer == "shield_terrestrial_v1"
    assert mv.rapp == PipelineConfig().rapp_version
    # No phantom learned-model tags anywhere in the record.
    dumped = record.model_dump_json()
    for phantom in ("jepa", "tdmpc", "dyna_v0"):
        assert phantom not in dumped


# ---------------------------------------------------------------------------
# 8. Ed25519-signed certificates in the pipeline
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_signs_certificates_when_key_configured(tmp_path: Path, monkeypatch):
    from horizon_ric.shield.signing import generate_signing_key

    key_path = tmp_path / "signing.pem"
    key = generate_signing_key(key_path)
    monkeypatch.setenv("HORIZON_CERT_SIGNING_KEY_PATH", str(key_path))

    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(
            adapter, store, PipelineConfig(status_poll_interval_s=0.01)
        )
        try:
            result = await pipeline.process_event(_event(event_id="evt-signed"))
        finally:
            await adapter.close()

    assert result.accepted is True
    record, _ = next(iter(store))
    cert_view = record.chosen_action["certificate"]
    assert cert_view["signature"]
    from horizon_ric.shield.signing import key_fingerprint

    assert cert_view["signing_key_fingerprint"] == key_fingerprint(key.public_key())


def test_pipeline_fails_loudly_on_unreadable_signing_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "HORIZON_CERT_SIGNING_KEY_PATH", str(tmp_path / "missing.pem")
    )
    adapter = A1Adapter(A1AdapterConfig(near_rt_ric_base_url="http://127.0.0.1:9"))
    store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
    try:
        with pytest.raises(RuntimeError, match="cannot read Ed25519"):
            DecisionPipeline(adapter, store, PipelineConfig())
    finally:
        asyncio.run(adapter.close())
