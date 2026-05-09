"""Tests for the NVIDIA Aerial Cloud-Native RAN (ARC) management-plane
integration. Pins the wire shape that an ARC deployment will see when
PreceptualAI uploads inference checkpoints and reads PM KPIs.

Reference (public surface):
    NVIDIA Aerial Cloud RAN public docs describe a REST surface mounted
    under /api/v1/ with multipart model uploads and JSON KPI reads. The
    full administrator surface lives behind an NVIDIA Developer login;
    these tests pin the publicly-documented portion via
    httpx.MockTransport so a regression cannot silently change the shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from horizon_ric.integrations.nvidia_arc import ARCClient, ARCClientConfig


@pytest.mark.asyncio
async def test_arc_submit_inference_model_url_and_auth_header(tmp_path: Path):
    captured: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            201,
            json={"model_id": "m-123", "sha256": "abc", "status": "registered"},
        )

    cfg = ARCClientConfig(
        base_url="https://aerial-arc.local", api_key="test-key-xyz"
    )
    client = ARCClient(
        config=cfg, transport=httpx.MockTransport(handler)
    )

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"fake-torch-bytes")
    body = await client.submit_inference_model(
        ckpt, {"model_id": "horizon-ltc", "framework": "torch"}
    )

    assert body["model_id"] == "m-123"
    req = captured[-1]
    # ARC management URL lives under /api/v1/models
    assert req.url.path == "/api/v1/models"
    assert req.method == "POST"
    # Auth header per ARC spec.
    assert req.headers.get("X-NVIDIA-API-KEY") == "test-key-xyz"
    # multipart upload — model + model_card parts.
    ctype = req.headers.get("content-type", "")
    assert ctype.startswith("multipart/form-data"), ctype
    raw = req.content
    assert b"model.pt" in raw
    assert b"model_card" in raw
    await client.close()


@pytest.mark.asyncio
async def test_arc_submit_inference_model_injects_sha256(tmp_path: Path):
    received_card: dict = {}

    async def handler(req: httpx.Request) -> httpx.Response:
        # Multipart parsing minimal — just look for the JSON card blob.
        raw = req.content.decode("utf-8", errors="ignore")
        # Find the card json blob between the JSON content-type marker
        # and the next boundary.
        marker = "application/json"
        idx = raw.find(marker)
        if idx >= 0:
            tail = raw[idx + len(marker):]
            # Skip CRLFs then read until next "--" boundary.
            tail = tail.lstrip("\r\n")
            end = tail.find("\r\n--")
            blob = tail[:end] if end > 0 else tail
            try:
                received_card.update(json.loads(blob.strip()))
            except json.JSONDecodeError:
                pass
        return httpx.Response(
            201, json={"model_id": "m-1", "status": "registered"}
        )

    cfg = ARCClientConfig(api_key="k")
    client = ARCClient(
        config=cfg, transport=httpx.MockTransport(handler)
    )

    ckpt = tmp_path / "weights.bin"
    payload = b"X" * 1024
    ckpt.write_bytes(payload)
    await client.submit_inference_model(ckpt, {"model_id": "x"})

    # ARC requires the registry to verify the artifact SHA-256 matches.
    assert "artifact_sha256" in received_card
    assert len(received_card["artifact_sha256"]) == 64
    assert received_card["artifact_filename"] == "weights.bin"
    assert received_card["artifact_bytes"] == 1024
    await client.close()


@pytest.mark.asyncio
async def test_arc_get_pm_kpis_url_and_query():
    captured: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={
                "cell_id": "cell-7",
                "window": {"start": "2026-05-06T00:00:00Z", "end": "2026-05-06T00:01:00Z"},
                "kpis": [
                    {
                        "name": "DRB.UEThpDl",
                        "unit": "kbps",
                        "samples": [1200.0, 1230.5, 1180.2],
                    }
                ],
            },
        )

    cfg = ARCClientConfig(api_key="k")
    client = ARCClient(
        config=cfg, transport=httpx.MockTransport(handler)
    )

    body = await client.get_pm_kpis(
        "cell-7",
        ("2026-05-06T00:00:00Z", "2026-05-06T00:01:00Z"),
    )
    assert body["cell_id"] == "cell-7"
    assert body["kpis"][0]["name"] == "DRB.UEThpDl"
    req = captured[-1]
    assert req.method == "GET"
    assert req.url.path == "/api/v1/pm/kpis/cell-7"
    assert req.url.params.get("start") == "2026-05-06T00:00:00Z"
    assert req.url.params.get("end") == "2026-05-06T00:01:00Z"
    await client.close()


@pytest.mark.asyncio
async def test_arc_submit_inference_model_missing_path_raises(tmp_path: Path):
    cfg = ARCClientConfig(api_key="k")

    async def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = ARCClient(
        config=cfg, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(FileNotFoundError):
        await client.submit_inference_model(
            tmp_path / "does-not-exist.pt", {"model_id": "x"}
        )
    await client.close()


@pytest.mark.asyncio
async def test_arc_get_pm_kpis_propagates_http_error():
    async def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "backend overloaded"})

    cfg = ARCClientConfig(api_key="k")
    client = ARCClient(
        config=cfg, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(httpx.HTTPError):
        await client.get_pm_kpis(
            "cell-1", ("2026-05-06T00:00:00Z", "2026-05-06T00:01:00Z")
        )
    await client.close()
