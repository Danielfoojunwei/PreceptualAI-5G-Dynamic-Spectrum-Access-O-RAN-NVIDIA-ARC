"""Contract tests for the direct O-RAN-SC A1 2.1.0 interface.

Unlike the ``osc`` dialect, which addresses the Non-RT RIC Policy Management
Service, ``osc_a1`` addresses the nested API implemented by the official
``o-ran-sc/sim-a1-interface`` Near-RT RIC simulator. The live container test
is in ``.github/workflows/osc-a1-integration.yml``; this file keeps the wire
contract deterministic in the fast test pack.
"""

from __future__ import annotations

import json

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig


@pytest.mark.asyncio
async def test_osc_a1_full_policy_lifecycle_matches_upstream_contract():
    captured: list[tuple[str, str, dict]] = []
    policy_ids: list[str] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        captured.append((req.method, req.url.path, body))

        if req.method == "PUT" and "/policies/" not in req.url.path:
            return httpx.Response(201)
        if req.method == "PUT":
            policy_ids[:] = [req.url.path.rsplit("/", 1)[-1]]
            return httpx.Response(202)
        if req.method == "GET" and req.url.path.endswith("/status"):
            return httpx.Response(200, json={"enforceStatus": "ENFORCED"})
        if req.method == "GET":
            return httpx.Response(200, json=policy_ids)
        if req.method == "DELETE":
            policy_ids.clear()
            return httpx.Response(202)
        return httpx.Response(405)

    cfg = A1AdapterConfig(dialect="osc_a1")
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    accepted = await adapter.register_policy_types()
    assert accepted == [20001, 20002, 20003, 20004]
    assert captured[0][1] == "/a1-p/policytypes/20001"
    assert set(captured[0][2]) >= {
        "name",
        "description",
        "policy_type_id",
        "create_schema",
    }

    payload = {
        "scope": {"slice_id": "slice-1"},
        "qos_objectives": {"priority": 5},
    }
    policy_id, status = await adapter.emit_policy(
        "horizon.qos.priority",
        payload,
        policy_id="horizon-live-smoke",
    )
    assert status == 202
    assert captured[-1] == (
        "PUT",
        "/a1-p/policytypes/20001/policies/horizon-live-smoke",
        payload,
    )

    assert await adapter.list_policies("horizon.qos.priority") == [policy_id]
    assert captured[-1][1] == "/a1-p/policytypes/20001/policies"

    status_body = await adapter.get_policy_status(
        "horizon.qos.priority", policy_id
    )
    assert status_body["enforceStatus"] == "ENFORCED"
    assert (
        captured[-1][1]
        == "/a1-p/policytypes/20001/policies/horizon-live-smoke/status"
    )

    assert await adapter.rollback_policy("horizon.qos.priority", policy_id) == 202
    assert not await adapter.list_policies("horizon.qos.priority")
    await adapter.close()


def test_unknown_a1_dialect_fails_closed():
    with pytest.raises(ValueError, match="unsupported A1 dialect"):
        A1Adapter(A1AdapterConfig(dialect="typo"))
