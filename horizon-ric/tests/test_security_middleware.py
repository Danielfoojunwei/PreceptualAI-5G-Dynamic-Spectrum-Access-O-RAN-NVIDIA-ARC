"""Security middleware tests — verify JWT extraction + RBAC enforcement
on a real FastAPI app."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
import structlog
from fastapi.testclient import TestClient

from horizon_ric.rapp.api import build_app
from horizon_ric.security.jwt import JWTManager
from horizon_ric.security.rbac import (
    DEFAULT_MODEL_PATH,
    DEFAULT_POLICY_PATH,
    Casbin,
)

_FIXTURE_KEY = Path(__file__).resolve().parent / "fixtures" / "test_jwt_signing.pem"


@pytest.fixture()
def app_setup(tmp_path: Path):
    pol = tmp_path / "rbac_policy.csv"
    shutil.copyfile(DEFAULT_POLICY_PATH, pol)
    rbac = Casbin(model_path=DEFAULT_MODEL_PATH, policy_path=pol)

    signing = tmp_path / "signing.pem"
    signing.write_bytes(_FIXTURE_KEY.read_bytes())
    jwt_mgr = JWTManager(signing_key_path=signing, issuer="iss", audience="aud")
    app = build_app(jwt_mgr, rbac)
    return app, jwt_mgr, rbac


@pytest.fixture()
def client(app_setup):
    app, _, _ = app_setup
    return TestClient(app)


@pytest.fixture()
def capture_audit(caplog):
    """Capture structlog events so tests can assert on auth.* names.

    We bridge structlog → stdlib logging via configure() — cheap and
    test-only; production config lives elsewhere."""
    structlog.configure(
        processors=[structlog.stdlib.render_to_log_kwargs],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    caplog.set_level(logging.INFO, logger="horizon_ric.security")
    yield caplog
    structlog.reset_defaults()


# ---------------------------------------------------------------------------
# 1. Missing Authorization → 401
# ---------------------------------------------------------------------------
def test_missing_authorization(client: TestClient, capture_audit) -> None:
    r = client.get("/policies")
    assert r.status_code == 401
    assert any("auth.token_invalid" in rec.message
               or "auth.token_invalid" == getattr(rec, "event", None)
               for rec in capture_audit.records)


# ---------------------------------------------------------------------------
# 2. Invalid token → 401
# ---------------------------------------------------------------------------
def test_invalid_token(client: TestClient, capture_audit) -> None:
    r = client.get(
        "/policies", headers={"Authorization": "Bearer notarealtoken"}
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# 3. Valid token, no role → 403
# ---------------------------------------------------------------------------
def test_valid_token_no_role(app_setup, capture_audit) -> None:
    app, jwt_mgr, _ = app_setup
    # Token but the user has no role bindings → enforce returns False.
    tok = jwt_mgr.mint_token("nobody", "default", roles=[])
    c = TestClient(app)
    r = c.get("/policies", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403
    assert any(
        getattr(rec, "event", None) == "auth.denied"
        or "auth.denied" in rec.getMessage()
        for rec in capture_audit.records
    )


# ---------------------------------------------------------------------------
# 4. Valid token + role → 200
# ---------------------------------------------------------------------------
def test_valid_token_with_role(app_setup, capture_audit) -> None:
    app, jwt_mgr, rbac = app_setup
    rbac.add_role("alice", "operator", "default")
    tok = jwt_mgr.mint_token("alice", "default", roles=["operator"])
    c = TestClient(app)
    r = c.get("/policies", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant"] == "default"
    assert body["subject"] == "alice"
    # auth.granted must have been logged.
    assert any(
        getattr(rec, "event", None) == "auth.granted"
        or "auth.granted" in rec.getMessage()
        for rec in capture_audit.records
    )


# ---------------------------------------------------------------------------
# 5. Tenant mismatch via X-Tenant header → 403
# ---------------------------------------------------------------------------
def test_tenant_mismatch_denied(app_setup) -> None:
    app, jwt_mgr, rbac = app_setup
    rbac.add_role("alice", "admin", "default")
    rbac._enforcer.add_policy("admin", "default", "*", "*")
    tok = jwt_mgr.mint_token("alice", "default", roles=["admin"])
    c = TestClient(app)
    r = c.get(
        "/policies",
        headers={
            "Authorization": f"Bearer {tok}",
            "X-Tenant": "other_tenant",
        },
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# 6. require_role enforces per-route extra roles (operator-only emit)
# ---------------------------------------------------------------------------
def test_require_role_blocks_api_user(app_setup) -> None:
    app, jwt_mgr, rbac = app_setup
    rbac.add_role("bot", "api-user", "default")
    tok = jwt_mgr.mint_token("bot", "default", roles=["api-user"])
    c = TestClient(app)
    # POST /policies/{id} requires operator/admin per build_app.
    # api-user shouldn't be allowed; first the global Casbin enforce
    # may already block (api-user has no `emit` on `policies/*`),
    # which still produces 403.
    r = c.post(
        "/policies/abc", headers={"Authorization": f"Bearer {tok}"}
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# 7. Public path bypasses auth
# ---------------------------------------------------------------------------
def test_healthz_public(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ---------------------------------------------------------------------------
# 8. Operator can emit policies
# ---------------------------------------------------------------------------
def test_operator_can_emit(app_setup) -> None:
    app, jwt_mgr, rbac = app_setup
    rbac.add_role("op", "operator", "default")
    tok = jwt_mgr.mint_token("op", "default", roles=["operator"])
    c = TestClient(app)
    r = c.post(
        "/policies/p1", headers={"Authorization": f"Bearer {tok}"}
    )
    assert r.status_code == 200
    assert r.json()["emitted_by"] == "op"
