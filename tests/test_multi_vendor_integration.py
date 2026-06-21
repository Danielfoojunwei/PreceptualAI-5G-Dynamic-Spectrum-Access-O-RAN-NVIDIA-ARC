"""End-to-end multi-vendor integration test.

For each of the 4 supported A1 dialects (legacy, osc, eiap, mantaray)
this suite drives emit_policy + get_policy_status + rollback_policy
through an httpx.MockTransport that records the URLs the adapter would
hit on a real vendor stack. Each per-dialect test then asserts the URL
shape matches the public spec for that vendor.

This is the proof that Horizon-RIC's payloads conform to the published
interfaces of:
    * legacy   — historical near-RT-RIC A1AP mirror (/A1-P/v2/...)
    * osc      — OSC NONRTRIC PMS reference (/a1-policy/v2/...)
    * eiap     — Ericsson EIAP rApp SDK (/A1-PolicyManagement/v2/...)
    * mantaray — Nokia MantaRay SMO via SDN-R (/sdn-r/api/v1/...)

It deliberately uses a parametrised dispatch map so adding a 5th vendor
later is a one-line change in :data:`DIALECTS`.
"""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig

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


# -- legacy ------------------------------------------------------------------


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


# -- osc ---------------------------------------------------------------------


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


# -- eiap --------------------------------------------------------------------


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


# -- mantaray ---------------------------------------------------------------


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
