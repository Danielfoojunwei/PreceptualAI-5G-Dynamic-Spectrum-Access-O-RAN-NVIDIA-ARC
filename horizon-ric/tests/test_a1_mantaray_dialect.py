"""Locks the A1Adapter MantaRay (Nokia MantaRay SMO) dialect.

Reference (public surface):
    Nokia MantaRay SMO exposes the A1/R1 surface through SDN-R:

        Base path:        /sdn-r/api/v1/policies
        rApp catalogue:   /sdn-r/api/v1/rapps/
        Policy create:    PUT /sdn-r/api/v1/policies
                          body: {policyId, policyTypeId, ricId, rappId,
                                 policyData}
        Auth:             keycloak-issued JWT (handled by auth.py).

The full MantaRay administrator guide lives behind a Nokia DAC login —
see docs/SMO_INTEGRATION.md for the link and the honest blocker. These
tests pin the publicly-documented portion of the contract via
httpx.MockTransport.
"""

from __future__ import annotations

import json

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig


@pytest.mark.asyncio
async def test_mantaray_register_policy_types_url_path():
    captured: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(201)

    cfg = A1AdapterConfig(dialect="mantaray")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )
    accepted = await adapter.register_policy_types()
    assert len(accepted) == 4
    for req in captured:
        assert req.url.path.startswith("/sdn-r/api/v1/policy-types/"), (
            f"MantaRay policy-type URL must live under /sdn-r/api/v1/, "
            f"got: {req.url.path}"
        )
    await adapter.close()


@pytest.mark.asyncio
async def test_mantaray_emit_policy_uses_sdn_r_url_and_camelcase_body():
    captured: list[tuple[str, str, dict]] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        captured.append((req.method, req.url.path, body))
        if req.method == "PUT":
            return httpx.Response(201)
        return httpx.Response(404)

    cfg = A1AdapterConfig(
        dialect="mantaray",
        mantaray_ric_id="ric-mantaray-eu-1",
        rapp_id="horizon-ric-rapp",
    )
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
    assert path == "/sdn-r/api/v1/policies"
    for k in ("policyId", "policyTypeId", "ricId", "rappId", "policyData"):
        assert k in body, f"MantaRay body missing required key {k!r}: {body}"
    assert body["policyId"] == pid
    assert body["policyTypeId"] == "20001"
    assert body["ricId"] == "ric-mantaray-eu-1"
    assert body["rappId"] == "horizon-ric-rapp"
    # MantaRay differs from EIAP by carrying rappId — make sure the
    # EIAP-shaped body wouldn't pass.
    assert "policy_id" not in body
    await adapter.close()


@pytest.mark.asyncio
async def test_mantaray_get_policy_status_url():
    captured: list[str] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req.url.path)
        if req.method == "PUT":
            return httpx.Response(201)
        if req.method == "GET":
            return httpx.Response(200, json={"enforceStatus": "ENFORCED"})
        return httpx.Response(404)

    cfg = A1AdapterConfig(dialect="mantaray")
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
    assert captured[-1] == f"/sdn-r/api/v1/policies/{pid}/status"
    await adapter.close()


@pytest.mark.asyncio
async def test_mantaray_rollback_policy_url():
    captured: list[tuple[str, str]] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append((req.method, req.url.path))
        if req.method == "PUT":
            return httpx.Response(201)
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    cfg = A1AdapterConfig(dialect="mantaray")
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
    assert path == f"/sdn-r/api/v1/policies/{pid}"
    await adapter.close()


@pytest.mark.asyncio
async def test_mantaray_list_policies_uses_query_param():
    captured: list[httpx.URL] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req.url)
        if req.method == "GET":
            return httpx.Response(200, json=["mr-p-1", "mr-p-2", "mr-p-3"])
        return httpx.Response(404)

    cfg = A1AdapterConfig(dialect="mantaray")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    ids = await adapter.list_policies("horizon.qos.priority")
    assert ids == ["mr-p-1", "mr-p-2", "mr-p-3"]
    last = captured[-1]
    assert last.path == "/sdn-r/api/v1/policies"
    assert last.params.get("policyTypeId") == "20001"
    await adapter.close()
