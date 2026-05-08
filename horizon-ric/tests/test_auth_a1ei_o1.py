"""Auth + A1 PolicyStatus + A1-EI + O1 adapter tests."""

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
from horizon_ric.rapp.auth import AuthConfig, build_secure_async_client
from horizon_ric.rapp.r1_adapter import R1Adapter, R1AdapterConfig


# ─── AuthConfig + secure-client construction ────────────────────────────


class TestAuthConfig:
    def test_no_auth_returns_plain_client(self):
        cfg = AuthConfig()
        assert not cfg.has_mtls()
        assert not cfg.has_oauth()
        assert not cfg.has_static_token()
        client = build_secure_async_client(cfg, base_url="http://example")
        assert isinstance(client, httpx.AsyncClient)

    def test_static_bearer_attaches_header(self):
        cfg = AuthConfig(static_bearer_token="mysecret")
        assert cfg.has_static_token()
        client = build_secure_async_client(cfg, base_url="http://example")
        # `auth` is set to a custom Auth flow.
        assert client.auth is not None

    def test_oauth_flow_set_when_configured(self):
        cfg = AuthConfig(
            token_url="https://idp/token",
            client_id="rapp",
            client_secret="shh",
        )
        assert cfg.has_oauth()
        client = build_secure_async_client(cfg, base_url="http://example")
        assert client.auth is not None


class TestWG11AuthHardening:
    """WG11 §6 compliance: cert-skip ban + cipher allow-list."""

    def test_production_mode_with_verify_tls_false_raises(self):
        # production_mode is True by default; verify_tls=False must refuse.
        cfg = AuthConfig(verify_tls=False)
        with pytest.raises(ValueError, match="WG11"):
            build_secure_async_client(cfg, base_url="https://example")

    def test_dev_mode_allows_verify_tls_false(self):
        # production_mode=False is the only legitimate way to disable TLS verify.
        cfg = AuthConfig(verify_tls=False, production_mode=False)
        client = build_secure_async_client(cfg, base_url="https://example")
        assert isinstance(client, httpx.AsyncClient)

    def test_cipher_allowlist_default_present(self):
        # AuthConfig accepts and exposes the cipher allow-list field with the
        # WG11 default suites.
        cfg = AuthConfig()
        assert "TLS_AES_256_GCM_SHA384" in cfg.cipher_allowlist
        assert "TLS_CHACHA20_POLY1305_SHA256" in cfg.cipher_allowlist
        assert "TLS_AES_128_GCM_SHA256" in cfg.cipher_allowlist
        # Caller can override.
        custom = AuthConfig(cipher_allowlist=["TLS_AES_256_GCM_SHA384"])
        assert custom.cipher_allowlist == ["TLS_AES_256_GCM_SHA384"]
        # Building a client with the allow-list does not crash.
        client = build_secure_async_client(custom, base_url="https://example")
        assert isinstance(client, httpx.AsyncClient)


class TestR1AuthIntegration:
    def test_auth_plumbed_into_client(self):
        cfg = R1AdapterConfig(auth=AuthConfig(static_bearer_token="abc"))
        adapter = R1Adapter(cfg)
        # The injected client carries the auth flow.
        assert adapter._client.auth is not None


class TestA1AuthIntegration:
    def test_auth_plumbed_into_client(self):
        cfg = A1AdapterConfig(auth=AuthConfig(static_bearer_token="abc"))
        adapter = A1Adapter(cfg)
        assert adapter._client.auth is not None


# ─── A1 PolicyStatus + A1-EI ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_policy_status():
    captured: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"enforceStatus": "ENFORCED"})

    cfg = A1AdapterConfig()
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )

    status = await adapter.get_policy_status("horizon.qos.priority", "pid-1")
    assert status == {"enforceStatus": "ENFORCED"}
    assert captured[0].method == "GET"
    assert "/policies/pid-1/status" in captured[0].url.path
    await adapter.close()


@pytest.mark.asyncio
async def test_list_policies_handles_array_response():
    async def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["p-1", "p-2", "p-3"])

    cfg = A1AdapterConfig()
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )
    pids = await adapter.list_policies("horizon.qos.priority")
    assert pids == ["p-1", "p-2", "p-3"]
    await adapter.close()


@pytest.mark.asyncio
async def test_create_ei_job():
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
    status = await adapter.create_ei_job(
        "ue.mobility.predict", "ei-1",
        ei_job_data={"interval_s": 10},
        target_uri="http://horizon-ric:8081/ei",
    )
    assert status == 201
    assert captured[0].method == "PUT"
    assert "/A1-EI/v1/eijobs/ei-1" in captured[0].url.path
    await adapter.close()


@pytest.mark.asyncio
async def test_delete_ei_job():
    async def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    cfg = A1AdapterConfig()
    adapter = A1Adapter(cfg)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url,
        transport=httpx.MockTransport(handler),
    )
    status = await adapter.delete_ei_job("ei-99")
    assert status == 204
    await adapter.close()


@pytest.mark.asyncio
async def test_unknown_policy_type_status_raises():
    adapter = A1Adapter()
    with pytest.raises(ValueError):
        await adapter.get_policy_status("not.a.real.type", "p")
    await adapter.close()


# ─── O1 adapter (no live server — tests config + import path) ───────────


def test_o1_adapter_importable():
    """O1Adapter should be importable when ncclient is present."""
    try:
        from horizon_ric.rapp.o1_adapter import O1Adapter, O1AdapterConfig
    except ImportError:
        pytest.skip("ncclient not installed; O1 adapter unavailable")

    cfg = O1AdapterConfig(host="127.0.0.1", port=830, username="test")
    adapter = O1Adapter(cfg)
    assert adapter.cfg.host == "127.0.0.1"
    assert adapter.is_connected is False
