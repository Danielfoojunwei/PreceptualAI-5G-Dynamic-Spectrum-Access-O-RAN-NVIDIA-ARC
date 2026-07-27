"""Env-driven A1 auth wiring for lifecycle._config_from_env().

Covers the credential groups the Helm chart injects — static bearer token
(``A1_CLIENT_TOKEN``), OAuth2 client-credentials triple, and the mTLS pair —
plus the verify-TLS / production-mode flag parsing and the "empty string is
unset" rule (Helm ``b64enc`` of "" produces an empty value).

The end-to-end check uses a REAL socket: a uvicorn-served FastAPI stub records
the ``Authorization`` header that ``A1Adapter.register_policy_types()`` sends
when the adapter is built from ``_config_from_env()``.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
import uvicorn
from fastapi import FastAPI, Request, Response

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
from horizon_ric.rapp.lifecycle import _config_from_env

# Every env var _config_from_env() reads for the A1 adapter — cleared before
# each test so developer machines / CI can't leak configuration in.
_ENV_VARS = (
    "HORIZON_SMO_URL",
    "HORIZON_RAPP_ID",
    "HORIZON_NEAR_RT_RIC_URL",
    "HORIZON_A1_DIALECT",
    "HORIZON_A1_TIMEOUT_S",
    "HORIZON_A1_RIC_ID",
    "HORIZON_A1_OSC_RIC_ID",
    "HORIZON_A1_EIAP_RIC_ID",
    "HORIZON_A1_MANTARAY_RIC_ID",
    "HORIZON_A1_SERVICE_ID",
    "A1_CLIENT_TOKEN",
    "HORIZON_A1_OAUTH_TOKEN_URL",
    "HORIZON_A1_OAUTH_CLIENT_ID",
    "HORIZON_A1_OAUTH_CLIENT_SECRET",
    "HORIZON_A1_OAUTH_SCOPE",
    "HORIZON_A1_CLIENT_CERT_PATH",
    "HORIZON_A1_CLIENT_KEY_PATH",
    "HORIZON_A1_CA_BUNDLE_PATH",
    "HORIZON_A1_VERIFY_TLS",
    "HORIZON_PRODUCTION_MODE",
    "HORIZON_HEALTH_HOST",
    "HORIZON_HEALTH_PORT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _a1_config() -> A1AdapterConfig:
    _, a1, _, _ = _config_from_env()
    return a1


# ---------------------------------------------------------------------------
# No credentials → plain HTTP (OSC sandbox path)
# ---------------------------------------------------------------------------
def test_no_auth_env_yields_auth_none():
    a1 = _a1_config()
    assert a1.auth is None
    assert a1.timeout_seconds == A1AdapterConfig.timeout_seconds
    assert a1.osc_service_id == a1.rapp_id


def test_empty_string_env_counts_as_unset(monkeypatch):
    for name in (
        "A1_CLIENT_TOKEN",
        "HORIZON_A1_OAUTH_TOKEN_URL",
        "HORIZON_A1_OAUTH_CLIENT_ID",
        "HORIZON_A1_OAUTH_CLIENT_SECRET",
        "HORIZON_A1_CLIENT_CERT_PATH",
        "HORIZON_A1_CLIENT_KEY_PATH",
    ):
        monkeypatch.setenv(name, "")  # Helm b64enc of "" injects empty values
    assert _a1_config().auth is None


# ---------------------------------------------------------------------------
# Static bearer token (the Helm secret's exact env name)
# ---------------------------------------------------------------------------
def test_static_bearer_token_builds_auth(monkeypatch):
    monkeypatch.setenv("A1_CLIENT_TOKEN", "tok123")
    auth = _a1_config().auth
    assert auth is not None
    assert auth.static_bearer_token == "tok123"
    assert auth.has_static_token()
    assert not auth.has_oauth()
    assert not auth.has_mtls()
    # WG11 defaults: verify certs, production mode on.
    assert auth.verify_tls is True
    assert auth.production_mode is True


# ---------------------------------------------------------------------------
# OAuth2 client-credentials triple
# ---------------------------------------------------------------------------
def test_oauth_triple_builds_auth(monkeypatch):
    monkeypatch.setenv("HORIZON_A1_OAUTH_TOKEN_URL", "https://smo/token")
    monkeypatch.setenv("HORIZON_A1_OAUTH_CLIENT_ID", "horizon")
    monkeypatch.setenv("HORIZON_A1_OAUTH_CLIENT_SECRET", "s3cret")
    monkeypatch.setenv("HORIZON_A1_OAUTH_SCOPE", "a1:write")
    auth = _a1_config().auth
    assert auth is not None
    assert auth.has_oauth()
    assert auth.token_url == "https://smo/token"
    assert auth.client_id == "horizon"
    assert auth.client_secret == "s3cret"
    assert auth.scope == "a1:write"
    assert auth.static_bearer_token is None


def test_partial_oauth_triple_is_ignored(monkeypatch):
    monkeypatch.setenv("HORIZON_A1_OAUTH_TOKEN_URL", "https://smo/token")
    # No client id / secret — an incomplete group must not build auth.
    assert _a1_config().auth is None


# ---------------------------------------------------------------------------
# mTLS pair
# ---------------------------------------------------------------------------
def test_mtls_pair_builds_auth(monkeypatch):
    monkeypatch.setenv("HORIZON_A1_CLIENT_CERT_PATH", "/etc/tls/client.crt")
    monkeypatch.setenv("HORIZON_A1_CLIENT_KEY_PATH", "/etc/tls/client.key")
    monkeypatch.setenv("HORIZON_A1_CA_BUNDLE_PATH", "/etc/tls/ca.pem")
    auth = _a1_config().auth
    assert auth is not None
    assert auth.has_mtls()
    assert auth.client_cert_path == Path("/etc/tls/client.crt")
    assert auth.client_key_path == Path("/etc/tls/client.key")
    assert auth.ca_bundle_path == Path("/etc/tls/ca.pem")
    assert auth.static_bearer_token is None
    assert not auth.has_oauth()


def test_mtls_cert_without_key_is_ignored(monkeypatch):
    monkeypatch.setenv("HORIZON_A1_CLIENT_CERT_PATH", "/etc/tls/client.crt")
    assert _a1_config().auth is None


# ---------------------------------------------------------------------------
# Flag parsing: HORIZON_A1_VERIFY_TLS / HORIZON_PRODUCTION_MODE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["0", "false", "no", "False", "NO"])
def test_verify_tls_falsey_values(monkeypatch, raw: str):
    monkeypatch.setenv("A1_CLIENT_TOKEN", "tok")
    monkeypatch.setenv("HORIZON_A1_VERIFY_TLS", raw)
    monkeypatch.setenv("HORIZON_PRODUCTION_MODE", raw)
    auth = _a1_config().auth
    assert auth is not None
    assert auth.verify_tls is False
    assert auth.production_mode is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "TRUE"])
def test_verify_tls_truthy_values(monkeypatch, raw: str):
    monkeypatch.setenv("A1_CLIENT_TOKEN", "tok")
    monkeypatch.setenv("HORIZON_A1_VERIFY_TLS", raw)
    monkeypatch.setenv("HORIZON_PRODUCTION_MODE", raw)
    auth = _a1_config().auth
    assert auth is not None
    assert auth.verify_tls is True
    assert auth.production_mode is True


# ---------------------------------------------------------------------------
# Partial credential groups are dropped LOUDLY (structured warning)
# ---------------------------------------------------------------------------
def test_partial_oauth_group_logs_missing_vars(monkeypatch, capsys):
    monkeypatch.setenv("HORIZON_A1_OAUTH_TOKEN_URL", "https://smo/token")
    assert _a1_config().auth is None
    out = capsys.readouterr().out
    assert "config.auth.partial_credential_group" in out
    assert "HORIZON_A1_OAUTH_CLIENT_ID" in out
    assert "HORIZON_A1_OAUTH_CLIENT_SECRET" in out


def test_partial_mtls_group_logs_missing_vars(monkeypatch, capsys):
    monkeypatch.setenv("HORIZON_A1_CLIENT_CERT_PATH", "/etc/tls/client.crt")
    assert _a1_config().auth is None
    out = capsys.readouterr().out
    assert "config.auth.partial_credential_group" in out
    assert "HORIZON_A1_CLIENT_KEY_PATH" in out


def test_complete_groups_log_no_partial_warning(monkeypatch, capsys):
    monkeypatch.setenv("A1_CLIENT_TOKEN", "tok")
    assert _a1_config().auth is not None
    assert "partial_credential_group" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# WG11 fail-fast: verify_tls off + production on is refused at startup,
# with or without a complete credential group.
# ---------------------------------------------------------------------------
def test_verify_tls_off_in_production_fails_fast_without_credentials(monkeypatch):
    monkeypatch.setenv("HORIZON_A1_VERIFY_TLS", "false")
    # HORIZON_PRODUCTION_MODE defaults to True.
    with pytest.raises(ValueError, match="WG11"):
        _config_from_env()


def test_verify_tls_off_in_explicit_production_fails_fast_with_token(monkeypatch):
    monkeypatch.setenv("A1_CLIENT_TOKEN", "tok")
    monkeypatch.setenv("HORIZON_A1_VERIFY_TLS", "0")
    monkeypatch.setenv("HORIZON_PRODUCTION_MODE", "true")
    with pytest.raises(ValueError, match="WG11"):
        _config_from_env()


def test_verify_tls_off_allowed_outside_production(monkeypatch):
    monkeypatch.setenv("HORIZON_A1_VERIFY_TLS", "false")
    monkeypatch.setenv("HORIZON_PRODUCTION_MODE", "false")
    # Flags are parsed even with zero credential groups; dev posture is OK.
    assert _a1_config().auth is None


# ---------------------------------------------------------------------------
# HORIZON_A1_TIMEOUT_S reaches the authenticated client too
# ---------------------------------------------------------------------------
def test_timeout_flows_into_auth_config(monkeypatch):
    monkeypatch.setenv("A1_CLIENT_TOKEN", "tok")
    monkeypatch.setenv("HORIZON_A1_TIMEOUT_S", "3.5")
    a1 = _a1_config()
    assert a1.timeout_seconds == 3.5
    assert a1.auth is not None
    # Previously stuck at the AuthConfig default (30 s) — the env timeout
    # must govern the authenticated httpx client as well.
    assert a1.auth.timeout_seconds == 3.5


# ---------------------------------------------------------------------------
# Timeout / RIC ids / service id
# ---------------------------------------------------------------------------
def test_timeout_ric_and_service_ids(monkeypatch):
    monkeypatch.setenv("HORIZON_A1_TIMEOUT_S", "2.5")
    monkeypatch.setenv("HORIZON_A1_RIC_ID", "ric-shared")
    monkeypatch.setenv("HORIZON_A1_OSC_RIC_ID", "ric-osc-only")
    monkeypatch.setenv("HORIZON_A1_SERVICE_ID", "svc-42")
    a1 = _a1_config()
    assert a1.timeout_seconds == 2.5
    assert a1.osc_ric_id == "ric-osc-only"  # per-dialect override wins
    assert a1.eiap_ric_id == "ric-shared"
    assert a1.mantaray_ric_id == "ric-shared"
    assert a1.osc_service_id == "svc-42"


def test_service_id_defaults_to_rapp_id(monkeypatch):
    monkeypatch.setenv("HORIZON_RAPP_ID", "custom-rapp")
    a1 = _a1_config()
    assert a1.rapp_id == "custom-rapp"
    assert a1.osc_service_id == "custom-rapp"


# ---------------------------------------------------------------------------
# Real-socket check: the bearer token actually reaches the wire
# ---------------------------------------------------------------------------
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


@pytest.mark.asyncio
async def test_static_token_arrives_as_bearer_header(monkeypatch):
    state: dict[str, Any] = {"auth_headers": [], "types": []}
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.put("/A1-P/v2/policytypes/{ptid}")
    async def put_policy_type(ptid: int, request: Request) -> Response:
        state["auth_headers"].append(request.headers.get("authorization"))
        state["types"].append(ptid)
        return Response(status_code=201)

    with _serve(app) as port:
        monkeypatch.setenv("HORIZON_NEAR_RT_RIC_URL", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("A1_CLIENT_TOKEN", "tok123")
        monkeypatch.setenv("HORIZON_PRODUCTION_MODE", "1")
        # verify_tls stays at its True default: _build_ssl_context returns an
        # SSLContext (cipher allowlist applied) which plain http never uses.
        _, a1_cfg, _, _ = _config_from_env()
        assert a1_cfg.auth is not None and a1_cfg.auth.production_mode is True

        adapter = A1Adapter(a1_cfg)
        try:
            accepted = await adapter.register_policy_types()
        finally:
            await adapter.close()

    assert sorted(accepted) == [20001, 20002, 20003, 20004]
    assert len(state["auth_headers"]) == 4
    assert set(state["auth_headers"]) == {"Bearer tok123"}
