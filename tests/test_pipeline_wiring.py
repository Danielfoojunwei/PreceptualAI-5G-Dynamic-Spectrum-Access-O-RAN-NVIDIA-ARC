"""End-to-end wiring tests for the rApp decision pipeline.

These tests run against a REAL Near-RT RIC stub — a FastAPI app served by
uvicorn on a loopback socket (no httpx.MockTransport, no unittest.mock) —
implementing the legacy A1AP surface:

    PUT /A1-P/v2/policytypes/{ptid}                     -> 201
    PUT /A1-P/v2/policytypes/{ptid}/policies/{pid}      -> 202 (body recorded)
    GET /A1-P/v2/policytypes/{ptid}/policies/{pid}/status
                                                        -> instance_status IN EFFECT

Covered paths: happy path (emit + enforcement), Shield/guard refusal (no A1
traffic, auditable refusal record), emit failure (503, audit row, no emit
count), and the daemon's --once smoke run against the same stub.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest
import uvicorn
from fastapi import FastAPI, Request, Response

from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.io.schemas import TelemetryEvent
from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Real-socket Near-RT RIC stub
# ---------------------------------------------------------------------------
def _build_a1_app(state: dict[str, Any], policy_put_status: int = 202) -> FastAPI:
    """Legacy-dialect A1AP surface backed by a shared recording dict."""
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.put("/A1-P/v2/policytypes/{ptid}")
    async def put_policy_type(ptid: int, request: Request) -> Response:
        state["types"][ptid] = await request.json()
        return Response(status_code=201)

    @app.put("/A1-P/v2/policytypes/{ptid}/policies/{pid}")
    async def put_policy(ptid: int, pid: str, request: Request) -> Response:
        body = await request.json()
        state["policies"].append(
            {"policy_type_id": ptid, "policy_id": pid, "body": body}
        )
        return Response(status_code=policy_put_status)

    @app.get("/A1-P/v2/policytypes/{ptid}/policies/{pid}/status")
    async def get_policy_status(ptid: int, pid: str) -> dict[str, str]:
        state["status_calls"].append(pid)
        return {"instance_status": "IN EFFECT"}

    return app


@contextlib.contextmanager
def _serve(app: FastAPI) -> Iterator[int]:
    """Run ``app`` under uvicorn on an ephemeral loopback port (real socket)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 15.0
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("uvicorn server thread died during startup")
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn server failed to start within 15 s")
        time.sleep(0.01)
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def _new_state() -> dict[str, Any]:
    return {"types": {}, "policies": [], "status_calls": []}


def _event(event_id: str = "evt-0001", sla_risk: float = 0.05) -> TelemetryEvent:
    return TelemetryEvent(
        event_id=event_id,
        modality="kpm_5g",
        source_id="test-src",
        ts_utc=datetime.now(timezone.utc),
        sequence=1,
        payload={"sla_risk_30s": sla_risk},
    )


def _adapter(port: int) -> A1Adapter:
    return A1Adapter(
        A1AdapterConfig(near_rt_ric_base_url=f"http://127.0.0.1:{port}")
    )


# ---------------------------------------------------------------------------
# (a) happy path — planner → Shield → guards → A1 emit → enforcement poll
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_happy_path_emits_and_polls_enforcement(tmp_path: Path):
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(
            adapter, store, PipelineConfig(status_poll_interval_s=0.05)
        )
        try:
            result = await pipeline.process_event(_event(sla_risk=0.05))
        finally:
            await adapter.close()

    assert result.accepted is True
    assert result.blocked is False
    assert result.emit_failed is False
    assert result.http_status == 202
    assert result.policy_type == "horizon.qos.priority"  # risk 0.05 < 0.3 band
    assert result.policy_id
    assert result.enforcement_status == "IN EFFECT"
    assert result.enforced is True
    assert result.latency_ms > 0.0

    # The Near-RT RIC saw exactly one policy PUT with a schema-conformant body
    # (only keys declared in A1Adapter._policy_create_schema, additionalProperties
    # False, plus the rapp_metadata decision link and the assurance envelope).
    assert len(state["policies"]) == 1
    put = state["policies"][0]
    assert put["policy_type_id"] == 20001
    assert put["policy_id"] == result.policy_id
    body = put["body"]
    assert set(body) == {"scope", "qos_objectives", "rapp_metadata", "assurance"}
    assert set(body["scope"]) == {"slice_id"}
    assert body["scope"]["slice_id"].startswith("slice-")
    assert set(body["qos_objectives"]) == {"priority"}
    assert 1 <= body["qos_objectives"]["priority"] <= 15
    assert body["rapp_metadata"] == {"decision_id": result.decision_id}
    # The certificate crosses the wire as a digest + verdict, not as a copy.
    # Unsigned here (HORIZON_CERT_SIGNING_KEY_PATH is not set), so no signature
    # fields. See tests/test_a1_assurance_envelope.py for the full contract.
    assurance = body["assurance"]
    assert set(assurance) == {
        "certificate_digest",
        "safe",
        "projected",
        "violated_ids",
        "min_margin_dB",
        "profile_digest",
    }
    assert assurance["safe"] is True
    assert assurance["violated_ids"] == []
    assert len(assurance["certificate_digest"]) == 64
    assert state["status_calls"] == [result.policy_id]

    # Evidence: exactly one record, appended by the adapter on emit success.
    assert len(store) == 1
    record, _ = next(iter(store))
    assert record.decision_id == result.decision_id
    assert record.chosen_action["policy_type"] == result.policy_type
    assert record.chosen_action["source_event"] == "evt-0001"
    assert record.chosen_action["certificate"]["safe"] is True
    assert len(record.rejected_alternatives) == 2
    assert adapter.policies_emitted_count() == 1


# ---------------------------------------------------------------------------
# (b) shield-block path — unfixable EIRP ceiling, zero A1 traffic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_blocks_on_unfixable_eirp_without_a1_traffic(tmp_path: Path):
    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        # A -100 dBm EIRP ceiling admits no implementable transmit power:
        # the only "safe" Tx sits far below the hardware floor, so the
        # pipeline must fail closed before any A1 call.
        pipeline = DecisionPipeline(
            adapter, store, PipelineConfig(max_eirp_dbm=-100.0)
        )
        try:
            result = await pipeline.process_event(_event(event_id="evt-blocked"))
        finally:
            await adapter.close()

    assert result.blocked is True
    assert result.accepted is False
    assert result.policy_id is None
    assert result.block_reasons  # at least one guard id recorded
    assert "tx_power_below_hw_floor" in result.block_reasons

    # Zero policy traffic reached the Near-RT RIC.
    assert state["policies"] == []
    assert state["status_calls"] == []
    assert adapter.policies_emitted_count() == 0

    # The refusal itself is an auditable evidence record.
    assert len(store) == 1
    record, _ = next(iter(store))
    assert record.chosen_action["emit_blocked"] is True
    assert record.chosen_action["block_reasons"] == result.block_reasons
    assert record.chosen_action["source_event"] == "evt-blocked"


# ---------------------------------------------------------------------------
# (c) emit-failure path — Near-RT RIC refuses the policy (503)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_records_emit_failure_on_503(tmp_path: Path):
    state = _new_state()
    with _serve(_build_a1_app(state, policy_put_status=503)) as port:
        adapter = _adapter(port)
        store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
        adapter.attach_evidence_store(store)
        pipeline = DecisionPipeline(adapter, store, PipelineConfig())
        try:
            result = await pipeline.process_event(_event(event_id="evt-503"))
        finally:
            await adapter.close()

    assert result.accepted is False
    assert result.blocked is False
    assert result.emit_failed is True
    assert result.http_status == 503
    assert result.policy_id is None
    assert result.enforcement_status is None

    # The adapter never counted an emit; the pipeline appended the audit row.
    assert adapter.policies_emitted_count() == 0
    assert len(store) == 1
    record, _ = next(iter(store))
    assert record.chosen_action["emit_failed"] is True
    assert record.chosen_action["source_event"] == "evt-503"


# ---------------------------------------------------------------------------
# (d) --once smoke — full daemon loop against the real server
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_daemon_once_reports_accepted_policies(tmp_path: Path, monkeypatch):
    from scripts.run_horizon_rapp import run_daemon

    # A clean environment: no auth, legacy dialect, deterministic knobs.
    for name in (
        "A1_CLIENT_TOKEN",
        "HORIZON_A1_OAUTH_TOKEN_URL",
        "HORIZON_A1_OAUTH_CLIENT_ID",
        "HORIZON_A1_OAUTH_CLIENT_SECRET",
        "HORIZON_A1_CLIENT_CERT_PATH",
        "HORIZON_A1_CLIENT_KEY_PATH",
        "HORIZON_A1_DIALECT",
        "HORIZON_SHIELD_MAX_EIRP_DBM",
        "HORIZON_DECISION_BUDGET_MS",
    ):
        monkeypatch.delenv(name, raising=False)

    events_file = tmp_path / "replay.jsonl"
    events_file.write_text(
        "\n".join(
            _event(event_id=f"evt-once-{i}", sla_risk=0.05).model_dump_json()
            for i in range(3)
        )
        + "\n"
    )
    report_path = tmp_path / "report.json"
    config_path = tmp_path / "daemon.yaml"

    state = _new_state()
    with _serve(_build_a1_app(state)) as port:
        config_path.write_text(
            json.dumps(
                {
                    "source": {"type": "file", "path": str(events_file)},
                    "sink": {"audit_path": str(tmp_path / "audit.jsonl")},
                    "health": {"port": _free_port()},
                    "metrics": {"port": _free_port()},
                    "state": {
                        "checkpoint_path": str(tmp_path / "state.json"),
                        "interval_seconds": 30,
                    },
                }
            )  # JSON is valid YAML
        )
        # _config_from_env() must honour these — the daemon builds its A1
        # adapter from the environment, not from hardcoded defaults.
        monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("HORIZON_ONCE_REQUIRE_ACCEPTED", "1")
        monkeypatch.setenv("HORIZON_A1_STATUS_POLL_INTERVAL_S", "0.05")

        rc = run_daemon(
            source_config=str(config_path), once=True, report_json=str(report_path)
        )

    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["events"] == 3
    assert report["accepted"] == 3
    assert report["blocked"] == 0
    assert report["emit_failed"] == 0
    assert report["enforced"] == 3
    assert len(report["policy_ids"]) == 3
    assert report["audit_chain_length"] == 3
    assert report["audit_verify_first_broken_index"] == -1

    # The stub saw one policy PUT per event, and (after the degraded R1 boot)
    # the daemon still registered the A1 policy types directly.
    assert len(state["policies"]) == 3
    assert set(state["types"]) == {20001, 20002, 20003, 20004}
