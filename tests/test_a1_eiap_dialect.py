"""Locks the A1Adapter EIAP (Ericsson Intelligent Automation Platform)
dialect to the published EIAP rApp SDK contract.

Reference (public surface):
    Ericsson Developer Studio / Open Source O-RAN dev portal documents
    the A1 PolicyManagement v2 surface used by the EIAP rApp SDK:

        Base path:        /A1-PolicyManagement/v2/
        Policy create:    PUT  /A1-PolicyManagement/v2/policies
                          body: {policyId, policyTypeId, ricId, policyData}
        Policy status:    GET  /A1-PolicyManagement/v2/policies/{id}/status
        Policy delete:    DELETE /A1-PolicyManagement/v2/policies/{id}

The full reference schema lives behind an Ericsson Developer login —
see docs/SMO_INTEGRATION.md for the link and the honest blocker. These
tests pin the publicly-documented portion of the contract via
httpx.MockTransport so a regression cannot silently change the wire
format the EIAP SMO sees.
"""

from __future__ import annotations

import json

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig


def _wire_mock_transport(adapter: A1Adapter, handler) -> None:
    # Replace the httpx.AsyncClient on the adapter with one bound to a
    # MockTransport so each test asserts on real HTTP request objects.
    import asyncio as _asyncio

    _asyncio.get_event_loop().run_until_complete(adapter._client.aclose())
    adapter._client = httpx.AsyncClient(
        base_url=adapter.cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_eiap_register_policy_types_url_path():
    captured: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(201)

    cfg = A1AdapterConfig(dialect="eiap")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )
    accepted = await adapter.register_policy_types()
    assert len(accepted) == 4, "all 4 default PreceptualAI types should register"
    for req in captured:
        assert req.url.path.startswith("/A1-PolicyManagement/v2/policy-types/"), (
            f"EIAP policy-type URL must live under /A1-PolicyManagement/v2/, "
            f"got: {req.url.path}"
        )
    await adapter.close()


@pytest.mark.asyncio
async def test_eiap_emit_policy_uses_flat_url_and_camelcase_body():
    captured: list[tuple[str, str, dict]] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        captured.append((req.method, req.url.path, body))
        if req.method == "PUT":
            return httpx.Response(201)
        return httpx.Response(404)

    cfg = A1AdapterConfig(dialect="eiap", eiap_ric_id="ric-eiap-east-1")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    pid, status = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    assert status == 201
    method, path, body = captured[-1]
    assert method == "PUT"
    assert path == "/A1-PolicyManagement/v2/policies"
    # EIAP body shape — camelCase keys per Ericsson rApp SDK.
    for k in ("policyId", "policyTypeId", "ricId", "policyData"):
        assert k in body, f"EIAP body missing required key {k!r}: {body}"
    assert body["policyId"] == pid
    assert body["policyTypeId"] == "20001"
    assert body["ricId"] == "ric-eiap-east-1"
    # Crucially, EIAP does NOT use snake_case OSC keys.
    assert "policy_id" not in body
    assert "policy_data" not in body
    await adapter.close()


@pytest.mark.asyncio
async def test_eiap_get_policy_status_url():
    captured: list[str] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req.url.path)
        if req.method == "PUT":
            return httpx.Response(201)
        if req.method == "GET":
            return httpx.Response(
                200, json={"enforceStatus": "ENFORCED"}
            )
        return httpx.Response(404)

    cfg = A1AdapterConfig(dialect="eiap")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    body = await adapter.get_policy_status("horizon.qos.priority", pid)
    assert body["enforceStatus"] == "ENFORCED"
    assert captured[-1] == f"/A1-PolicyManagement/v2/policies/{pid}/status"
    await adapter.close()


@pytest.mark.asyncio
async def test_eiap_rollback_policy_url():
    captured: list[tuple[str, str]] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "PUT":
            return httpx.Response(201)
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    cfg = A1AdapterConfig(dialect="eiap")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    pid, _ = await adapter.emit_policy(
        "horizon.qos.priority",
        {"scope": {"slice_id": "s1"}, "qos_objectives": {"priority": 5}},
    )
    rb = await adapter.rollback_policy("horizon.qos.priority", pid)
    assert rb == 204
    method, path = captured[-1]
    assert method == "DELETE"
    assert path == f"/A1-PolicyManagement/v2/policies/{pid}"
    await adapter.close()


@pytest.mark.asyncio
async def test_eiap_list_policies_uses_query_param():
    captured: list[httpx.URL] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req.url)
        if req.method == "GET":
            return httpx.Response(200, json=["p-1", "p-2"])
        return httpx.Response(404)

    cfg = A1AdapterConfig(dialect="eiap")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    ids = await adapter.list_policies("horizon.qos.priority")
    assert ids == ["p-1", "p-2"]
    last = captured[-1]
    assert last.path == "/A1-PolicyManagement/v2/policies"
    assert last.params.get("policyTypeId") == "20001"
    await adapter.close()
