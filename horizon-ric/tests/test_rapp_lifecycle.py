"""rApp lifecycle + R1/A1 adapter tests.

These tests use httpx's MockTransport so we never touch a real SMO or Near-RT
RIC during CI. The wire format we assert is the one we send to those systems
in production — not a fake. A passing test means we'd send standards-shaped
requests if a server were on the other end.
"""

from __future__ import annotations

import json

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import (
    DEFAULT_POLICY_TYPES,
    A1Adapter,
    A1AdapterConfig,
)
from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle, RAppState
from horizon_ric.rapp.r1_adapter import R1Adapter, R1AdapterConfig

# ─── R1 adapter ───────────────────────────────────────────────────────────


class TestR1Adapter:
    def test_default_payload_shape(self):
        cfg = R1AdapterConfig()
        adapter = R1Adapter(cfg)
        payload = adapter._registration_payload()
        # Required top-level fields per O-RAN.WG2.R1AP §5.3.1
        for k in (
            "rapp_id",
            "rapp_version",
            "rapp_name",
            "rapp_description",
            "services_produced",
            "services_consumed",
            "rapp_metadata",
        ):
            assert k in payload
        # Each produced service has id + version
        for svc in payload["services_produced"]:
            assert "service_id" in svc and "version" in svc
        # Spec-compliance metadata is non-empty
        assert payload["rapp_metadata"]["spec_compliance"]

    @pytest.mark.asyncio
    async def test_register_then_deregister_against_mock(self):
        captured: list[httpx.Request] = []

        async def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            if req.method == "POST":
                return httpx.Response(201, json={"status": "registered"})
            if req.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(404)

        cfg = R1AdapterConfig()
        adapter = R1Adapter(cfg)
        # Replace the real client with one bound to our mock transport.
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url=cfg.smo_base_url,
            transport=httpx.MockTransport(handler),
        )

        result = await adapter.register()
        assert result == {"status": "registered"}
        assert adapter.is_registered() is True
        assert captured[0].method == "POST"
        assert captured[0].url.path == "/r1/registration/v1/registration"
        body = json.loads(captured[0].content)
        assert body["rapp_id"] == cfg.rapp_id

        await adapter.deregister()
        assert adapter.is_registered() is False
        assert captured[-1].method == "DELETE"
        assert cfg.rapp_id in captured[-1].url.path

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deregister_noop_when_not_registered(self):
        adapter = R1Adapter()
        # Should not raise, should not hit the network.
        await adapter.deregister()
        await adapter.close()


# ─── A1 adapter ───────────────────────────────────────────────────────────


class TestA1Adapter:
    def test_default_policy_types_well_formed(self):
        for name, spec in DEFAULT_POLICY_TYPES.items():
            assert isinstance(name, str)
            for k in ("policy_type_id", "name", "description", "schema_v"):
                assert k in spec
            # IDs in the operator-extension range (>= 20000)
            assert spec["policy_type_id"] >= 20000

    @pytest.mark.asyncio
    async def test_register_policy_types_emits_one_put_per_type(self):
        captured: list[httpx.Request] = []

        async def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            return httpx.Response(201)

        cfg = A1AdapterConfig()
        adapter = A1Adapter(cfg)
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url=cfg.near_rt_ric_base_url,
            transport=httpx.MockTransport(handler),
        )

        accepted = await adapter.register_policy_types()
        assert len(accepted) == len(DEFAULT_POLICY_TYPES)
        assert all(req.method == "PUT" for req in captured)
        # All paths under /A1-P/v2/policytypes/{id}
        for req in captured:
            assert req.url.path.startswith("/A1-P/v2/policytypes/")
        await adapter.close()

    @pytest.mark.asyncio
    async def test_emit_policy_then_rollback(self):
        captured: list[httpx.Request] = []

        async def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            if req.method == "PUT":
                return httpx.Response(201)
            if req.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(404)

        cfg = A1AdapterConfig()
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
        assert isinstance(pid, str) and len(pid) >= 8
        assert status == 201
        assert adapter.policies_emitted_count() == 1

        rollback_status = await adapter.rollback_policy(
            "horizon.qos.priority", pid
        )
        assert rollback_status == 204
        assert any(
            req.method == "DELETE" and pid in req.url.path for req in captured
        )
        await adapter.close()

    @pytest.mark.asyncio
    async def test_emit_unknown_type_raises(self):
        adapter = A1Adapter()
        with pytest.raises(ValueError):
            await adapter.emit_policy("not.a.real.type", {})
        await adapter.close()


# ─── Lifecycle ────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_initial_state_is_init(self):
        lc = HorizonRAppLifecycle()
        assert lc.state == RAppState.INIT

    @pytest.mark.asyncio
    async def test_boot_then_shutdown_transitions(self):
        lc = HorizonRAppLifecycle()

        async def r1_handler(req: httpx.Request) -> httpx.Response:
            if req.method == "POST":
                return httpx.Response(201, json={"status": "ok"})
            return httpx.Response(204)

        async def a1_handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(201)

        # Swap clients for mock transports.
        await lc._r1._client.aclose()
        lc._r1._client = httpx.AsyncClient(
            base_url=lc._r1.cfg.smo_base_url,
            transport=httpx.MockTransport(r1_handler),
        )
        await lc._a1._client.aclose()
        lc._a1._client = httpx.AsyncClient(
            base_url=lc._a1.cfg.near_rt_ric_base_url,
            transport=httpx.MockTransport(a1_handler),
        )

        result = await lc.boot()
        assert lc.state == RAppState.RUNNING
        assert "registration" in result
        assert len(result["policy_types"]) == len(DEFAULT_POLICY_TYPES)

        await lc.shutdown()
        assert lc.state == RAppState.STOPPED
