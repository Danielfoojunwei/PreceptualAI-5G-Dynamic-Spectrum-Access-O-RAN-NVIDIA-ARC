"""Security tests for the hardened operator dashboard API.

Covers the audit findings against ``horizon_ric.rapp.dashboard_api``:

  * PBKDF2-hashed credentials in ``HORIZON_API_USERS`` (stdlib
    ``hashlib.pbkdf2_hmac`` + ``hmac.compare_digest``).
  * Plaintext credentials refused when ``HORIZON_PRODUCTION_MODE`` is
    truthy (the default); still accepted — with a loud warning — when it
    is explicitly falsy (dev).
  * Per-endpoint role enforcement: read endpoints accept any
    authenticated role, the privileged ``POST /api/v1/audit/verify`` is
    admin/operator only (viewer → 403).
  * JWT tampering (signature corruption, payload mutation, re-signing
    with an attacker secret) is rejected.
  * The default evidence path matches the runner's audit chain
    (``/var/lib/horizon/audit.jsonl``), overridable via
    ``HORIZON_EVIDENCE_PATH``.

All tests run a real FastAPI ``TestClient`` over a real
``JsonlEvidenceStore`` — no mocks.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jose import jwt as jose_jwt

# Deterministic secret for standalone runs of this file; the full suite
# shares the same value via tests/test_api.py.
os.environ.setdefault("HORIZON_API_JWT_SECRET", "test-secret-do-not-use-in-prod")

from horizon_ric.evidence.schema import (  # noqa: E402
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore  # noqa: E402
from horizon_ric.rapp.dashboard_api import (  # noqa: E402
    DEFAULT_EVIDENCE_PATH,
    DEFAULT_JWT_AUDIENCE,
    DEFAULT_JWT_ISSUER,
    PRIVILEGED_ROLES,
    build_dashboard_api,
    hash_password,
    issue_token,
)

# Low iteration count keeps the suite fast; the entry format carries the
# count, so production entries use the 600k default independently.
_TEST_ITERATIONS = 5_000


def _make_record(decision_id: str, *, timestamp: datetime) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        timestamp=timestamp,
        rapp_instance_id="rapp-sec-test-0",
        state_hash="0" * 64,
        chosen_action={"action_type": "reroute", "params": {"gateway": "G1"}},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.05, sla_risk_1min=0.06, sla_risk_5min=0.07
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="enc-1.0.0",
            risk_heads="rh-1.0.0",
            dyna="dyna-1.0.0",
            policy="pol-1.0.0",
            constraint_layer="cl-1.0.0",
            rapp="0.2.0",
        ),
    )


@pytest.fixture
def evidence_store(tmp_path: Path) -> JsonlEvidenceStore:
    store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
    base = datetime.now(timezone.utc) - timedelta(minutes=1)
    for i in range(2):
        store.append(_make_record(f"dec-{i:04d}", timestamp=base + timedelta(seconds=i)))
    return store


@pytest.fixture
def client(evidence_store: JsonlEvidenceStore) -> TestClient:
    app = build_dashboard_api(lifecycle=None, evidence_store=evidence_store)
    return TestClient(app)


def _login(client: TestClient, username: str, password: str):
    return client.post(
        "/api/v1/auth/token", json={"username": username, "password": password}
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── hashed credentials ──────────────────────────────────────────────────────


def test_login_with_hashed_credentials_succeeds_in_production(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PBKDF2 entries authenticate even with production mode at its default."""
    monkeypatch.delenv("HORIZON_PRODUCTION_MODE", raising=False)
    entry = hash_password("wonderland", iterations=_TEST_ITERATIONS)
    assert entry.startswith("pbkdf2_sha256$")
    monkeypatch.setenv("HORIZON_API_USERS", f"alice:{entry}:admin")

    resp = _login(client, "alice", "wonderland")
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "admin"
    assert body["token_type"] == "bearer"

    # The minted token is a working bearer credential.
    state = client.get("/api/v1/state", headers=_bearer(body["access_token"]))
    assert state.status_code == 200


def test_login_with_wrong_password_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HORIZON_PRODUCTION_MODE", raising=False)
    entry = hash_password("wonderland", iterations=_TEST_ITERATIONS)
    monkeypatch.setenv("HORIZON_API_USERS", f"alice:{entry}:admin")

    assert _login(client, "alice", "WRONG").status_code == 401
    assert _login(client, "alice", "wonderland ").status_code == 401
    assert _login(client, "nobody", "wonderland").status_code == 401


def test_hash_entries_survive_colon_delimited_role_parsing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The $-separated hash coexists with the user:secret:role format."""
    monkeypatch.delenv("HORIZON_PRODUCTION_MODE", raising=False)
    a = hash_password("pw-a", iterations=_TEST_ITERATIONS)
    b = hash_password("pw-b", iterations=_TEST_ITERATIONS)
    monkeypatch.setenv("HORIZON_API_USERS", f"ada:{a}:viewer,bob:{b}:operator")

    ra = _login(client, "ada", "pw-a")
    rb = _login(client, "bob", "pw-b")
    assert (ra.status_code, rb.status_code) == (200, 200)
    assert ra.json()["role"] == "viewer"
    assert rb.json()["role"] == "operator"


# ── plaintext credential policy ─────────────────────────────────────────────


def test_plaintext_refused_in_production_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HORIZON_PRODUCTION_MODE unset ⇒ production ⇒ plaintext refused."""
    monkeypatch.delenv("HORIZON_PRODUCTION_MODE", raising=False)
    monkeypatch.setenv("HORIZON_API_USERS", "alice:wonderland:admin")
    resp = _login(client, "alice", "wonderland")
    assert resp.status_code == 401


def test_plaintext_refused_when_production_mode_explicit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HORIZON_PRODUCTION_MODE", "true")
    monkeypatch.setenv("HORIZON_API_USERS", "alice:wonderland:admin")
    assert _login(client, "alice", "wonderland").status_code == 401


def test_plaintext_allowed_in_dev_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit HORIZON_PRODUCTION_MODE=false keeps the legacy dev path."""
    monkeypatch.setenv("HORIZON_PRODUCTION_MODE", "false")
    monkeypatch.setenv("HORIZON_API_USERS", "alice:wonderland:operator")
    resp = _login(client, "alice", "wonderland")
    assert resp.status_code == 200
    assert resp.json()["role"] == "operator"


# ── role enforcement ────────────────────────────────────────────────────────


def test_viewer_gets_403_on_privileged_endpoint(client: TestClient) -> None:
    token = issue_token(sub="bob", role="viewer")
    resp = client.post("/api/v1/audit/verify", headers=_bearer(token))
    assert resp.status_code == 403
    assert "viewer" in resp.json()["detail"]


def test_admin_passes_privileged_endpoint(client: TestClient) -> None:
    token = issue_token(sub="root", role="admin")
    resp = client.post("/api/v1/audit/verify", headers=_bearer(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["intact"] is True
    assert body["chain_length"] == 2


def test_operator_retains_privileged_access_backcompat(client: TestClient) -> None:
    """Operator tokens predate the hardening; they keep audit-verify access."""
    assert set(PRIVILEGED_ROLES) == {"admin", "operator"}
    token = issue_token(sub="ops", role="operator")
    assert client.post("/api/v1/audit/verify", headers=_bearer(token)).status_code == 200


def test_viewer_can_use_read_only_endpoints(client: TestClient) -> None:
    token = issue_token(sub="bob", role="viewer")
    for path in (
        "/api/v1/state",
        "/api/v1/policies",
        "/api/v1/sla/timeline",
        "/api/v1/circuit-breakers",
        "/api/v1/connectors",
    ):
        resp = client.get(path, headers=_bearer(token))
        assert resp.status_code == 200, f"GET {path} -> {resp.status_code}"


def test_missing_role_claim_gets_403_on_privileged_endpoint(
    client: TestClient,
) -> None:
    """A structurally valid token without a role claim must not pass."""
    import time as _time

    now = int(_time.time())
    payload = {
        "iss": DEFAULT_JWT_ISSUER,
        "aud": DEFAULT_JWT_AUDIENCE,
        "sub": "ghost",
        "iat": now,
        "exp": now + 60,
    }
    token = jose_jwt.encode(
        payload, os.environ["HORIZON_API_JWT_SECRET"], algorithm="HS256"
    )
    assert client.post("/api/v1/audit/verify", headers=_bearer(token)).status_code == 403


# ── JWT tampering ───────────────────────────────────────────────────────────


def test_jwt_signature_corruption_rejected(client: TestClient) -> None:
    token = issue_token(sub="root", role="admin")
    corrupted = token[:-3] + ("AAA" if not token.endswith("AAA") else "BBB")
    resp = client.get("/api/v1/state", headers=_bearer(corrupted))
    assert resp.status_code == 401


def test_jwt_payload_mutation_rejected(client: TestClient) -> None:
    """Flipping role viewer→admin in the payload invalidates the signature."""
    token = issue_token(sub="bob", role="viewer")
    header_b64, payload_b64, sig_b64 = token.split(".")
    raw = base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4))
    assert b'"viewer"' in raw
    forged_payload = (
        base64.urlsafe_b64encode(raw.replace(b'"viewer"', b'"admin"'))
        .rstrip(b"=")
        .decode("ascii")
    )
    forged = f"{header_b64}.{forged_payload}.{sig_b64}"
    resp = client.post("/api/v1/audit/verify", headers=_bearer(forged))
    assert resp.status_code == 401


def test_jwt_resigned_with_attacker_secret_rejected(client: TestClient) -> None:
    import time as _time

    now = int(_time.time())
    forged = jose_jwt.encode(
        {
            "iss": DEFAULT_JWT_ISSUER,
            "aud": DEFAULT_JWT_AUDIENCE,
            "sub": "mallory",
            "role": "admin",
            "iat": now,
            "exp": now + 3600,
        },
        "attacker-controlled-secret",
        algorithm="HS256",
    )
    assert client.get("/api/v1/state", headers=_bearer(forged)).status_code == 401


# ── hash-password CLI helper ────────────────────────────────────────────────


def test_hash_password_cli_generates_working_entry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "horizon_ric.rapp.dashboard_api",
            "--hash-password",
            "s3cret-cli",
            "--iterations",
            str(_TEST_ITERATIONS),
            "--user",
            "carol",
            "--role",
            "admin",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    entry = proc.stdout.strip()
    assert entry.startswith("carol:pbkdf2_sha256$")
    assert entry.endswith(":admin")

    monkeypatch.delenv("HORIZON_PRODUCTION_MODE", raising=False)  # production
    monkeypatch.setenv("HORIZON_API_USERS", entry)
    resp = _login(client, "carol", "s3cret-cli")
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"
    assert _login(client, "carol", "not-the-password").status_code == 401


# ── evidence path alignment ─────────────────────────────────────────────────


def test_default_evidence_path_matches_runner_audit_chain() -> None:
    assert DEFAULT_EVIDENCE_PATH == "/var/lib/horizon/audit.jsonl"


def test_evidence_path_env_override_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "override" / "audit.jsonl"
    monkeypatch.setenv("HORIZON_EVIDENCE_PATH", str(override))
    build_dashboard_api(lifecycle=None)
    # JsonlEvidenceStore materialises its backing file on construction.
    assert override.exists()
