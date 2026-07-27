"""Tests for the operator dashboard REST API.

Covers all 7 documented endpoints + the auth/token issuance route, and
verifies that JWT bearer auth is enforced on every protected endpoint.

Layered on top of a real ``JsonlEvidenceStore`` written to a per-test
temp dir so the audit-chain semantics are exercised end-to-end (no
mocks, no in-memory shims).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# JWT secret must be present BEFORE importing the dashboard API module —
# the module-level secret loader is lazy, but tests using TestClient need
# a deterministic secret across the suite.
os.environ.setdefault("HORIZON_API_JWT_SECRET", "test-secret-do-not-use-in-prod")
os.environ.setdefault("HORIZON_API_USERS", "alice:wonderland:operator,bob:builder:viewer")
# These fixtures use dev-only plaintext credentials; production mode (the
# default) refuses them. tests/test_dashboard_api_security.py covers the
# hashed-credential + production-mode paths.
os.environ.setdefault("HORIZON_PRODUCTION_MODE", "false")

from horizon_ric.evidence.schema import (  # noqa: E402
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
    RejectedAlternative,
    RejectionReasonMachine,
)
from horizon_ric.evidence.store import JsonlEvidenceStore  # noqa: E402
from horizon_ric.rapp.dashboard_api import (  # noqa: E402
    build_dashboard_api,
    issue_token,
)

# ── fixtures ────────────────────────────────────────────────────────────────


def _make_record(
    decision_id: str,
    *,
    timestamp: datetime,
    sla_30s: float = 0.05,
) -> DecisionRecord:
    versions = ModelVersions(
        encoder="enc-1.0.0",
        risk_heads="rh-1.0.0",
        dyna="dyna-1.0.0",
        policy="pol-1.0.0",
        constraint_layer="cl-1.0.0",
        rapp="0.2.0",
    )
    chosen_pred = PredictedOutcome(
        sla_risk_30s=sla_30s,
        sla_risk_1min=sla_30s + 0.01,
        sla_risk_5min=sla_30s + 0.02,
        energy_kwh=0.12,
    )
    rejected = RejectedAlternative(
        rank=1,
        action={"action_type": "reroute", "params": {"gateway": "G3"}},
        predicted_outcome=PredictedOutcome(
            sla_risk_30s=0.21, sla_risk_1min=0.18, sla_risk_5min=0.15
        ),
        rejection_reason_machine=RejectionReasonMachine(
            primary_cause="gateway_overload",
            primary_metric="gateway_load_G3",
            predicted_value=0.92,
            threshold=0.80,
            horizon="30s",
        ),
        rejection_reason_human="G3 predicted to exceed 80% load within 30s.",
        random_seed=0,
    )
    return DecisionRecord(
        decision_id=decision_id,
        timestamp=timestamp,
        rapp_instance_id="rapp-test-0",
        state_hash="0" * 64,
        chosen_action={"action_type": "reroute", "params": {"gateway": "G1"}},
        predicted_outcome_chosen=chosen_pred,
        rejected_alternatives=[rejected],
        model_versions=versions,
    )


@pytest.fixture
def evidence_store(tmp_path: Path) -> JsonlEvidenceStore:
    """A real JSONL evidence store with three pre-populated decisions."""
    store = JsonlEvidenceStore(tmp_path / "horizon.jsonl")
    base = datetime.now(timezone.utc) - timedelta(minutes=5)
    for i in range(3):
        rec = _make_record(
            f"dec-{i:04d}",
            timestamp=base + timedelta(seconds=30 * i),
            sla_30s=0.05 + 0.01 * i,
        )
        store.append(rec)
    return store


@pytest.fixture
def client(evidence_store: JsonlEvidenceStore) -> TestClient:
    """A TestClient bound to a dashboard API with the real evidence store."""
    app = build_dashboard_api(lifecycle=None, evidence_store=evidence_store)
    return TestClient(app)


@pytest.fixture
def auth_header() -> dict[str, str]:
    token = issue_token(sub="alice", role="operator")
    return {"Authorization": f"Bearer {token}"}


# ── auth ────────────────────────────────────────────────────────────────────


def test_auth_required_on_state(client: TestClient) -> None:
    """Every protected endpoint must reject anonymous requests."""
    response = client.get("/api/v1/state")
    assert response.status_code == 401
    assert "missing bearer token" in response.json()["detail"]


def test_auth_required_on_all_endpoints(client: TestClient) -> None:
    """Every documented endpoint enforces JWT bearer auth."""
    cases = [
        ("GET", "/api/v1/state"),
        ("GET", "/api/v1/policies"),
        ("GET", "/api/v1/policies/dec-0000"),
        ("POST", "/api/v1/audit/verify"),
        ("GET", "/api/v1/sla/timeline"),
        ("GET", "/api/v1/circuit-breakers"),
        ("GET", "/api/v1/connectors"),
    ]
    for method, path in cases:
        resp = client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} did not 401"


def test_auth_rejects_garbage_token(client: TestClient) -> None:
    response = client.get(
        "/api/v1/state",
        headers={"Authorization": "Bearer not-a-valid-jwt"},
    )
    assert response.status_code == 401
    assert "invalid token" in response.json()["detail"]


def test_auth_token_issuance_real_credentials(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/token",
        json={"username": "alice", "password": "wonderland"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["role"] == "operator"
    assert isinstance(body["access_token"], str) and len(body["access_token"]) > 50

    # The minted token must work against /state.
    auth = {"Authorization": f"Bearer {body['access_token']}"}
    state_resp = client.get("/api/v1/state", headers=auth)
    assert state_resp.status_code == 200


def test_auth_token_rejects_bad_password(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/token",
        json={"username": "alice", "password": "WRONG"},
    )
    assert response.status_code == 401


# ── /state ──────────────────────────────────────────────────────────────────


def test_state_returns_live_evidence_chain(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.get("/api/v1/state", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    assert body["rapp_state"] == "stopped"  # no lifecycle attached
    assert body["evidence_chain_head"] is not None
    assert body["last_decision_id"] == "dec-0002"
    assert body["uptime_seconds"] >= 0
    assert body["watchdog_interval_s"] > 0


# ── /policies ───────────────────────────────────────────────────────────────


def test_policies_lists_recent(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.get("/api/v1/policies?limit=10", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    # Ordered most recent first.
    ids = [item["decision_id"] for item in body["items"]]
    assert ids == ["dec-0002", "dec-0001", "dec-0000"]
    # Hash-chain field is real (64-hex sha256).
    assert all(len(item["chain_hash"]) == 64 for item in body["items"])


def test_policy_detail_returns_full_record(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.get("/api/v1/policies/dec-0001", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    assert body["record"]["decision_id"] == "dec-0001"
    assert body["record"]["rejected_alternatives"]
    assert (
        body["record"]["rejected_alternatives"][0]["rejection_reason_machine"][
            "primary_cause"
        ]
        == "gateway_overload"
    )
    assert len(body["chain_hash"]) == 64


def test_policy_detail_404(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.get("/api/v1/policies/does-not-exist", headers=auth_header)
    assert resp.status_code == 404


# ── /audit/verify ───────────────────────────────────────────────────────────


def test_audit_verify_intact(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.post("/api/v1/audit/verify", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    assert body["intact"] is True
    assert body["broken_at_index"] == -1
    assert body["chain_length"] == 3
    assert body["head_hash"] is not None


def test_audit_verify_detects_tamper(
    tmp_path: Path, auth_header: dict[str, str]
) -> None:
    """Tamper a record byte and confirm verify() returns the broken index."""
    store = JsonlEvidenceStore(tmp_path / "tampered.jsonl")
    base = datetime.now(timezone.utc)
    for i in range(2):
        store.append(_make_record(f"dec-{i}", timestamp=base + timedelta(seconds=i)))

    # Flip a single character in the first record's payload to break the chain.
    p = tmp_path / "tampered.jsonl"
    raw = p.read_text()
    # The JSONL line stores the canonical (sorted, no spaces) record
    # under "record"; the outer wrapper uses pretty-style spacing. Patch
    # the inner reroute string so the canonical-JSON re-hash diverges.
    tampered = raw.replace('"reroute"', '"REWRITE"', 1)
    assert tampered != raw, "test setup: substitution did not match"
    p.write_text(tampered)

    app = build_dashboard_api(lifecycle=None, evidence_store=store)
    client = TestClient(app)
    resp = client.post("/api/v1/audit/verify", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    assert body["intact"] is False
    assert body["broken_at_index"] == 0


# ── /sla/timeline ───────────────────────────────────────────────────────────


def test_sla_timeline_returns_per_horizon_samples(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    resp = client.get("/api/v1/sla/timeline", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizons"] == ["30s", "60s", "300s"]
    # 3 records × 3 horizons = 9 samples.
    assert len(body["items"]) == 9
    # Risks must be in [0,1] and tagged with the originating decision.
    for item in body["items"]:
        assert 0.0 <= item["sla_risk"] <= 1.0
        assert item["decision_id"].startswith("dec-")
        assert item["horizon"] in {"30s", "60s", "300s"}


def test_sla_timeline_filters_window(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    # Window in the future — should be empty.
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    resp = client.get(
        "/api/v1/sla/timeline",
        headers=auth_header,
        params={"from": future, "to": future},
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == []


# ── /circuit-breakers ───────────────────────────────────────────────────────


def test_circuit_breakers_returns_provided(
    evidence_store: JsonlEvidenceStore, auth_header: dict[str, str]
) -> None:
    """When breakers dict is provided, it is surfaced verbatim."""
    class _CB:
        state = "closed"
        fail_counter = 0

    breakers = {"horizon.r1": _CB(), "horizon.a1": _CB(), "horizon.o1": _CB()}
    app = build_dashboard_api(
        lifecycle=None, evidence_store=evidence_store, breakers=breakers
    )
    client = TestClient(app)
    resp = client.get("/api/v1/circuit-breakers", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    names = sorted(b["name"] for b in body["breakers"])
    assert names == ["horizon.a1", "horizon.o1", "horizon.r1"]
    for cb in body["breakers"]:
        assert cb["state"] == "closed"
        assert cb["fail_counter"] == 0


# ── /connectors ─────────────────────────────────────────────────────────────


def test_connectors_lists_registered(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    """Built-in HTTP / Kafka / file connectors register themselves at import time."""
    # Force-import the connectors package so the registry is populated.
    import horizon_ric.io.connectors  # noqa: F401

    resp = client.get("/api/v1/connectors", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["sources"], list)
    assert isinstance(body["sinks"], list)


# ── lifecycle integration smoke ────────────────────────────────────────────


def test_state_with_real_lifecycle(
    evidence_store: JsonlEvidenceStore, auth_header: dict[str, str]
) -> None:
    """When a real `HorizonRAppLifecycle` is attached, /state surfaces its enum."""
    from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle

    lc = HorizonRAppLifecycle(state_path=Path("/tmp/horizon-lifecycle-test.json"))
    app = build_dashboard_api(lifecycle=lc, evidence_store=evidence_store)
    client = TestClient(app)
    resp = client.get("/api/v1/state", headers=auth_header)
    assert resp.status_code == 200
    body = resp.json()
    assert body["rapp_state"] == "init"
    assert body["degraded_state"] == "normal"
