"""A1 multi-dialect wire-contract tests — two tiers.

Tier 1 — OFFLINE CONTRACT TESTS (``httpx.MockTransport``).
    For each of the 4 supported A1 dialects (legacy, osc, eiap, mantaray)
    these tests drive emit_policy + get_policy_status + rollback_policy
    through an httpx.MockTransport that records the URLs the adapter would
    hit on a real vendor stack, then assert the URL shape matches the
    public spec for that vendor:
        * legacy   — historical near-RT-RIC A1AP mirror (/A1-P/v2/...)
        * osc      — OSC NONRTRIC PMS reference (/a1-policy/v2/...)
        * eiap     — Ericsson EIAP rApp SDK (/A1-PolicyManagement/v2/...)
        * mantaray — Nokia MantaRay SMO via SDN-R (/sdn-r/api/v1/...)
    No socket is opened; these are pure in-process contract checks.

Tier 2 — REAL-SOCKET WIRE-CONTRACT TESTS (``TestRealSocketVendorContracts``).
    A genuine FastAPI app is served by uvicorn in a daemon thread on
    127.0.0.1 with an ephemeral port, and the unmodified A1Adapter client
    talks to it over a real TCP socket. No ``httpx.MockTransport`` and no
    ``unittest.mock`` anywhere in this tier. These tests pin the exact
    JSON bodies and ``Authorization`` headers an Ericsson EIAP
    (/A1-PolicyManagement/v2/...) or Nokia MantaRay (/sdn-r/api/v1/...)
    endpoint would receive, including the static-bearer path and a full
    OAuth2 client-credentials round-trip against a real /oauth/token
    endpoint served by the same server.

Honest scope: neither tier is proof of deployment interoperability with
commercial Ericsson or Nokia stacks — that requires vendor tenant access
(see deploy/onboarding/ and docs/VENDOR_ONBOARDING.md). Direct live
interoperability with the official OSC A1 simulator is exercised
separately by ``.github/workflows/osc-a1-integration.yml``.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Callable

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
from horizon_ric.rapp.auth import AuthConfig

# Per-dialect URL/body expectations. policy_type_id 20001 corresponds to
# the default "horizon.qos.priority" type defined in DEFAULT_POLICY_TYPES.
DIALECTS: dict[str, dict[str, str | Callable[[str], str]]] = {
    "legacy": {
        "create_url": "/A1-P/v2/policytypes/20001/policies",
        "instance_url": lambda pid: f"/A1-P/v2/policytypes/20001/policies/{pid}",
        "status_url": lambda pid: f"/A1-P/v2/policytypes/20001/policies/{pid}/status",
        "create_method": "PUT",
    },
    "osc": {
        "create_url": "/a1-policy/v2/policies",
        "instance_url": lambda pid: f"/a1-policy/v2/policies/{pid}",
        "status_url": lambda pid: f"/a1-policy/v2/policies/{pid}/status",
        "create_method": "PUT",
    },
    "eiap": {
        "create_url": "/A1-PolicyManagement/v2/policies",
        "instance_url": lambda pid: f"/A1-PolicyManagement/v2/policies/{pid}",
        "status_url": lambda pid: f"/A1-PolicyManagement/v2/policies/{pid}/status",
        "create_method": "PUT",
    },
    "mantaray": {
        "create_url": "/sdn-r/api/v1/policies",
        "instance_url": lambda pid: f"/sdn-r/api/v1/policies/{pid}",
        "status_url": lambda pid: f"/sdn-r/api/v1/policies/{pid}/status",
        "create_method": "PUT",
    },
}


async def _make_adapter(dialect: str, handler):
    cfg = A1AdapterConfig(dialect=dialect)
    adapter = A1Adapter(cfg)
    # Replace the live AsyncClient with a MockTransport-bound one.
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )
    return adapter


# ===========================================================================
# Tier 1 — offline contract tests (httpx.MockTransport; no socket opened).
# ===========================================================================

# -- legacy (offline contract) -----------------------------------------------


@pytest.mark.asyncio
async def test_legacy_emit_policy_url_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        return httpx.Response(201)

    adapter = await _make_adapter("legacy", handler)
    pid, status = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
        policy_id="legacy-pid-001",
    )
    assert status == 201
    method, path = captured[-1]
    assert method == DIALECTS["legacy"]["create_method"]
    assert path == DIALECTS["legacy"]["instance_url"]("legacy-pid-001")
    await adapter.close()


@pytest.mark.asyncio
async def test_legacy_get_policy_status_url_matches_spec():
    async def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(200, json={"enforceStatus": "ENFORCED"})
        return httpx.Response(201)

    adapter = await _make_adapter("legacy", handler)
    body = await adapter.get_policy_status(
        "horizon.qos.priority", "legacy-pid-001"
    )
    assert body["enforceStatus"] == "ENFORCED"
    await adapter.close()


@pytest.mark.asyncio
async def test_legacy_rollback_policy_url_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(201)

    adapter = await _make_adapter("legacy", handler)
    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
        policy_id="legacy-rb",
    )
    rb = await adapter.rollback_policy("horizon.qos.priority", pid)
    assert rb == 204
    method, path = captured[-1]
    assert method == "DELETE"
    assert path == DIALECTS["legacy"]["instance_url"](pid)
    await adapter.close()


# -- osc (offline contract) --------------------------------------------------


@pytest.mark.asyncio
async def test_osc_emit_policy_url_and_body_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        captured.append((req.method, req.url.path, body))
        return httpx.Response(201)

    adapter = await _make_adapter("osc", handler)
    pid, status = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
        policy_id="osc-pid-001",
    )
    assert status == 201
    method, path, body = captured[-1]
    assert method == "PUT"
    assert path == DIALECTS["osc"]["create_url"]
    # OSC PMS uses snake_case keys per pms-api.json schema.
    for k in ("policy_id", "policytype_id", "ric_id", "policy_data"):
        assert k in body, f"OSC body missing {k!r}: {body}"
    assert body["policy_id"] == pid
    await adapter.close()


@pytest.mark.asyncio
async def test_osc_get_policy_status_url_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "GET":
            return httpx.Response(200, json={"enforceStatus": "ENFORCED"})
        return httpx.Response(201)

    adapter = await _make_adapter("osc", handler)
    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    await adapter.get_policy_status("horizon.qos.priority", pid)
    method, path = captured[-1]
    assert method == "GET"
    assert path == DIALECTS["osc"]["status_url"](pid)
    await adapter.close()


@pytest.mark.asyncio
async def test_osc_rollback_policy_url_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(201)

    adapter = await _make_adapter("osc", handler)
    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    rb = await adapter.rollback_policy("horizon.qos.priority", pid)
    assert rb == 204
    method, path = captured[-1]
    assert method == "DELETE"
    assert path == DIALECTS["osc"]["instance_url"](pid)
    await adapter.close()


# -- eiap (offline contract) -------------------------------------------------


@pytest.mark.asyncio
async def test_eiap_emit_policy_url_and_body_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        captured.append((req.method, req.url.path, body))
        return httpx.Response(201)

    adapter = await _make_adapter("eiap", handler)
    pid, status = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    assert status == 201
    method, path, body = captured[-1]
    assert method == "PUT"
    assert path == DIALECTS["eiap"]["create_url"]
    # EIAP rApp SDK uses camelCase keys.
    for k in ("policyId", "policyTypeId", "ricId", "policyData"):
        assert k in body, f"EIAP body missing {k!r}: {body}"
    assert body["policyId"] == pid
    await adapter.close()


@pytest.mark.asyncio
async def test_eiap_get_policy_status_url_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "GET":
            return httpx.Response(200, json={"enforceStatus": "ENFORCED"})
        return httpx.Response(201)

    adapter = await _make_adapter("eiap", handler)
    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    await adapter.get_policy_status("horizon.qos.priority", pid)
    method, path = captured[-1]
    assert method == "GET"
    assert path == DIALECTS["eiap"]["status_url"](pid)
    await adapter.close()


@pytest.mark.asyncio
async def test_eiap_rollback_policy_url_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(201)

    adapter = await _make_adapter("eiap", handler)
    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    rb = await adapter.rollback_policy("horizon.qos.priority", pid)
    assert rb == 204
    method, path = captured[-1]
    assert method == "DELETE"
    assert path == DIALECTS["eiap"]["instance_url"](pid)
    await adapter.close()


# -- mantaray (offline contract) --------------------------------------------


@pytest.mark.asyncio
async def test_mantaray_emit_policy_url_and_body_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        captured.append((req.method, req.url.path, body))
        return httpx.Response(201)

    adapter = await _make_adapter("mantaray", handler)
    pid, status = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    assert status == 201
    method, path, body = captured[-1]
    assert method == "PUT"
    assert path == DIALECTS["mantaray"]["create_url"]
    # MantaRay carries rappId on top of the EIAP-shape camelCase keys.
    for k in ("policyId", "policyTypeId", "ricId", "rappId", "policyData"):
        assert k in body, f"MantaRay body missing {k!r}: {body}"
    assert body["policyId"] == pid
    await adapter.close()


@pytest.mark.asyncio
async def test_mantaray_get_policy_status_url_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "GET":
            return httpx.Response(200, json={"enforceStatus": "ENFORCED"})
        return httpx.Response(201)

    adapter = await _make_adapter("mantaray", handler)
    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    await adapter.get_policy_status("horizon.qos.priority", pid)
    method, path = captured[-1]
    assert method == "GET"
    assert path == DIALECTS["mantaray"]["status_url"](pid)
    await adapter.close()


@pytest.mark.asyncio
async def test_mantaray_rollback_policy_url_matches_spec():
    captured = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(201)

    adapter = await _make_adapter("mantaray", handler)
    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    rb = await adapter.rollback_policy("horizon.qos.priority", pid)
    assert rb == 204
    method, path = captured[-1]
    assert method == "DELETE"
    assert path == DIALECTS["mantaray"]["instance_url"](pid)
    await adapter.close()


# ===========================================================================
# Tier 2 — real-socket wire-contract tests.
#
# A real FastAPI app served by uvicorn in a daemon thread on 127.0.0.1
# with an ephemeral port. The A1Adapter's own httpx.AsyncClient (built by
# A1Adapter.__init__ / build_secure_async_client — NOT swapped out) talks
# to it over a genuine TCP socket. No httpx.MockTransport, no
# unittest.mock anywhere below this line.
# ===========================================================================


def _build_recording_app(request_log: list[dict[str, Any]]) -> FastAPI:
    """FastAPI app mimicking the EIAP + MantaRay A1 URL surfaces.

    Every request is appended to ``request_log`` as
    {method, path, body, authorization} before a canned success response
    is returned, so tests can assert the exact bytes-on-the-wire contract.
    """
    app = FastAPI()

    async def _record(request: Request) -> None:
        raw = await request.body()
        try:
            body: Any = json.loads(raw) if raw else None
        except ValueError:
            # e.g. the x-www-form-urlencoded OAuth2 token request.
            body = raw.decode("utf-8", errors="replace")
        request_log.append(
            {
                "method": request.method,
                "path": request.url.path,
                "body": body,
                "authorization": request.headers.get("authorization"),
            }
        )

    # -- Ericsson EIAP A1 PolicyManagement surface --------------------------

    @app.put("/A1-PolicyManagement/v2/policy-types/{policy_type_id}")
    async def eiap_put_policy_type(policy_type_id: int, request: Request) -> JSONResponse:
        await _record(request)
        return JSONResponse({"policyTypeId": str(policy_type_id)}, status_code=201)

    @app.put("/A1-PolicyManagement/v2/policies")
    async def eiap_put_policy(request: Request) -> JSONResponse:
        await _record(request)
        return JSONResponse({"result": "created"}, status_code=201)

    @app.get("/A1-PolicyManagement/v2/policies/{policy_id}/status")
    async def eiap_policy_status(policy_id: str, request: Request) -> JSONResponse:
        await _record(request)
        return JSONResponse({"enforceStatus": "ENFORCED"})

    @app.delete("/A1-PolicyManagement/v2/policies/{policy_id}")
    async def eiap_delete_policy(policy_id: str, request: Request) -> Response:
        await _record(request)
        return Response(status_code=204)

    # -- Nokia MantaRay SDN-R surface ---------------------------------------

    @app.put("/sdn-r/api/v1/policy-types/{policy_type_id}")
    async def mantaray_put_policy_type(policy_type_id: int, request: Request) -> JSONResponse:
        await _record(request)
        return JSONResponse({"policyTypeId": str(policy_type_id)}, status_code=201)

    @app.put("/sdn-r/api/v1/policies")
    async def mantaray_put_policy(request: Request) -> JSONResponse:
        await _record(request)
        return JSONResponse({"result": "created"}, status_code=201)

    @app.get("/sdn-r/api/v1/policies/{policy_id}/status")
    async def mantaray_policy_status(policy_id: str, request: Request) -> JSONResponse:
        await _record(request)
        return JSONResponse({"enforceStatus": "ENFORCED"})

    @app.delete("/sdn-r/api/v1/policies/{policy_id}")
    async def mantaray_delete_policy(policy_id: str, request: Request) -> Response:
        await _record(request)
        return Response(status_code=204)

    # -- Keycloak-style OAuth2 token endpoint (client-credentials) ----------

    @app.post("/oauth/token")
    async def oauth_token(request: Request) -> JSONResponse:
        await _record(request)
        return JSONResponse(
            {"access_token": "oauth-tok", "expires_in": 3600, "token_type": "Bearer"}
        )

    return app


class _LiveVendorServer:
    """uvicorn server on 127.0.0.1:<ephemeral> recording every request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        # Reserve an ephemeral port by binding to port 0, then hand the
        # concrete port number to uvicorn.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            self.port: int = probe.getsockname()[1]
        self._server = uvicorn.Server(
            uvicorn.Config(
                _build_recording_app(self.requests),
                host="127.0.0.1",
                port=self.port,
                log_level="error",
            )
        )
        self._thread = threading.Thread(
            target=self._server.run, name="a1-vendor-test-server", daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 15.0
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("uvicorn server thread died before startup completed")
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn server did not start within 15s")
            time.sleep(0.01)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)

    def requests_for(self, path: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"] == path]


@pytest.fixture(scope="module")
def live_server():
    server = _LiveVendorServer()
    server.start()
    yield server
    server.stop()


QOS_PAYLOAD = {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}}


class TestRealSocketVendorContracts:
    """EIAP / MantaRay wire contracts over a real local HTTP socket."""

    @pytest.fixture(autouse=True)
    def _fresh_request_log(self, live_server):
        live_server.requests.clear()
        yield

    @staticmethod
    def _adapter(live_server, dialect: str, **cfg_kwargs) -> A1Adapter:
        cfg = A1AdapterConfig(
            near_rt_ric_base_url=live_server.base_url,
            dialect=dialect,
            **cfg_kwargs,
        )
        return A1Adapter(cfg)

    # -- (a) EIAP emit: exact camelCase body over a real socket -------------

    async def test_eiap_emit_sends_exact_camelcase_body(self, live_server):
        adapter = self._adapter(live_server, "eiap", eiap_ric_id="ric-eiap-01")
        try:
            pid, status = await adapter.emit_policy(
                "horizon.qos.priority", QOS_PAYLOAD, policy_id="eiap-real-001"
            )
        finally:
            await adapter.close()
        assert (pid, status) == ("eiap-real-001", 201)
        (req,) = live_server.requests_for("/A1-PolicyManagement/v2/policies")
        assert req["method"] == "PUT"
        assert req["body"] == {
            "policyId": "eiap-real-001",
            "policyTypeId": "20001",
            "ricId": "ric-eiap-01",
            "policyData": QOS_PAYLOAD,
        }

    async def test_eiap_register_policy_types_over_real_socket(self, live_server):
        adapter = self._adapter(live_server, "eiap")
        try:
            accepted = await adapter.register_policy_types()
        finally:
            await adapter.close()
        assert sorted(accepted) == [20001, 20002, 20003, 20004]
        seen_paths = {r["path"] for r in live_server.requests if r["method"] == "PUT"}
        for type_id in (20001, 20002, 20003, 20004):
            assert f"/A1-PolicyManagement/v2/policy-types/{type_id}" in seen_paths

    # -- (b) MantaRay emit: rappId present over a real socket ---------------

    async def test_mantaray_emit_includes_rapp_id(self, live_server):
        adapter = self._adapter(
            live_server, "mantaray", mantaray_ric_id="ric-mantaray-01"
        )
        try:
            pid, status = await adapter.emit_policy(
                "horizon.qos.priority", QOS_PAYLOAD, policy_id="mantaray-real-001"
            )
        finally:
            await adapter.close()
        assert (pid, status) == ("mantaray-real-001", 201)
        (req,) = live_server.requests_for("/sdn-r/api/v1/policies")
        assert req["method"] == "PUT"
        assert req["body"] == {
            "policyId": "mantaray-real-001",
            "policyTypeId": "20001",
            "ricId": "ric-mantaray-01",
            "rappId": "horizon-ric-rapp",
            "policyData": QOS_PAYLOAD,
        }

    async def test_mantaray_register_policy_types_over_real_socket(self, live_server):
        adapter = self._adapter(live_server, "mantaray")
        try:
            accepted = await adapter.register_policy_types()
        finally:
            await adapter.close()
        assert sorted(accepted) == [20001, 20002, 20003, 20004]
        seen_paths = {r["path"] for r in live_server.requests if r["method"] == "PUT"}
        for type_id in (20001, 20002, 20003, 20004):
            assert f"/sdn-r/api/v1/policy-types/{type_id}" in seen_paths

    # -- (c) static bearer token reaches the server -------------------------

    async def test_static_bearer_token_reaches_server(self, live_server):
        # AuthConfig constructed directly and deliberately NOT via any env
        # parsing (the lifecycle env layer is covered elsewhere).
        auth = AuthConfig(static_bearer_token="tok-eiap", production_mode=True)
        adapter = self._adapter(
            live_server, "eiap", eiap_ric_id="ric-eiap-01", auth=auth
        )
        try:
            _, status = await adapter.emit_policy(
                "horizon.qos.priority", QOS_PAYLOAD, policy_id="eiap-auth-001"
            )
        finally:
            await adapter.close()
        assert status == 201
        (req,) = live_server.requests_for("/A1-PolicyManagement/v2/policies")
        assert req["authorization"] == "Bearer tok-eiap"

    # -- (d) OAuth2 client-credentials round-trip ---------------------------

    async def test_oauth2_client_credentials_token_flow(self, live_server):
        auth = AuthConfig(
            token_url=f"{live_server.base_url}/oauth/token",
            client_id="horizon-ric-rapp",
            client_secret="rapp-secret",
            production_mode=True,
        )
        adapter = self._adapter(
            live_server, "mantaray", mantaray_ric_id="ric-mantaray-01", auth=auth
        )
        try:
            _, status = await adapter.emit_policy(
                "horizon.qos.priority", QOS_PAYLOAD, policy_id="mantaray-auth-001"
            )
        finally:
            await adapter.close()
        assert status == 201
        # The adapter really fetched a token from the local /oauth/token
        # endpoint (client-credentials grant) before emitting.
        token_reqs = live_server.requests_for("/oauth/token")
        assert len(token_reqs) == 1
        assert token_reqs[0]["method"] == "POST"
        assert "grant_type=client_credentials" in token_reqs[0]["body"]
        (req,) = live_server.requests_for("/sdn-r/api/v1/policies")
        assert req["authorization"] == "Bearer oauth-tok"
