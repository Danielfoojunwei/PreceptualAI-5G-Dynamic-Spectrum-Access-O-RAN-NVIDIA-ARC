"""Locks the A1Adapter OSC dialect to the upstream pms-api.json contract.

The OSC ``nonrtric-plt-a1policymanagementservice`` reference exposes
``/a1-policy/v2/policies`` (flat) with a ``policy_info`` body — see
``deploy/osc_specs/pms-api.json``. These tests pin the URL paths and
body shape so we don't regress back to the legacy nested URL.
"""

from __future__ import annotations

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig


@pytest.mark.asyncio
async def test_osc_register_policy_types_url_path():
    captured: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(201)

    cfg = A1AdapterConfig(dialect="osc")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )
    accepted = await adapter.register_policy_types()
    assert len(accepted) == 4
    for req in captured:
        assert req.url.path.startswith("/a1-policy/v2/policy-types/"), req.url.path
    await adapter.close()


@pytest.mark.asyncio
async def test_osc_emit_policy_uses_flat_url_and_policy_info_body():
    captured: list[tuple[str, str, dict]] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        body = {}
        if req.content:
            import json

            body = json.loads(req.content)
        captured.append((req.method, req.url.path, body))
        if req.method == "PUT":
            return httpx.Response(201)
        if req.method == "DELETE":
            return httpx.Response(204)
        if req.method == "GET":
            return httpx.Response(200, json={"status": {"enforceStatus": "ENFORCED"}})
        return httpx.Response(404)

    cfg = A1AdapterConfig(dialect="osc")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    pid, status = await adapter.emit_policy(
        "horizon.qos.priority",
        {
            "scope": {"slice_id": "s1"},
            "qos_objectives": {"priority": 5},
        },
    )
    assert status == 201
    method, path, body = captured[-1]
    assert method == "PUT"
    assert path == "/a1-policy/v2/policies"
    # Real OSC body shape — policy_info schema (pms-api.json).
    for k in ("policy_id", "policytype_id", "ric_id", "policy_data"):
        assert k in body, f"OSC policy_info body missing {k!r}: {body}"
    assert body["policy_id"] == pid
    assert body["policytype_id"] == "20001"

    psi = await adapter.get_policy_status("horizon.qos.priority", pid)
    assert psi["status"]["enforceStatus"] == "ENFORCED"
    assert captured[-1][1] == f"/a1-policy/v2/policies/{pid}/status"

    rb = await adapter.rollback_policy("horizon.qos.priority", pid)
    assert rb == 204
    assert captured[-1][1] == f"/a1-policy/v2/policies/{pid}"

    await adapter.close()
