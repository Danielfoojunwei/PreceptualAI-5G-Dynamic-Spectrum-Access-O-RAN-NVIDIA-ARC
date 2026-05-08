"""Tests for VIAVI D4AI sandbox adapters.

Verifies:

* TM500 (functional) — fixture replay yields 5 schema-valid TelemetryEvents,
  events round-trip through Pydantic JSON, ``.mode == "functional"``,
  and the connect/close lifecycle works against a mocked httpx transport.
* PowerMeter / XhaulAdvisor / TeraVMCore (documented-only) — ``.mode``
  is ``"documented-only"`` and ``connect()`` raises
  ``NotImplementedError("contact VIAVI ops")`` so ops can grep for it.

Connect/close lifecycle tests are marked ``integration`` (skipped from
the fast pack) because they spin up an httpx mock-transport client and
exercise the async streaming path — heavier than the pure-fixture path.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from horizon_ric.integrations.viavi_d4ai import (
    PowerMeterAdapter,
    TeraVMCoreAdapter,
    TM500Adapter,
    XhaulAdvisorAdapter,
)
from horizon_ric.io.schemas import TelemetryEvent

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tm500_sample.jsonl"


# ─── TM500 fixture-replay path ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_tm500_replay_fixture_yields_five_events() -> None:
    """fixture replay yields exactly 5 TelemetryEvent instances."""

    adapter = TM500Adapter(base_url="http://tm500.local")
    events: list[TelemetryEvent] = []
    async for ev in await adapter.replay_fixture(FIXTURE_PATH):
        events.append(ev)
    assert len(events) == 5
    for ev in events:
        assert isinstance(ev, TelemetryEvent)
        assert ev.modality == "ue_qos"
        assert ev.source_id == "tm500_ue_emu_01"
        assert ev.ts_utc.tzinfo is not None  # tz-aware


@pytest.mark.asyncio
async def test_tm500_replay_fixture_schema_roundtrip() -> None:
    """Every yielded event survives a Pydantic JSON round-trip."""

    adapter = TM500Adapter(base_url="http://tm500.local")
    async for ev in await adapter.replay_fixture(FIXTURE_PATH):
        raw = ev.model_dump_json()
        restored = TelemetryEvent.model_validate_json(raw)
        assert restored.event_id == ev.event_id
        assert restored.payload == ev.payload
        assert restored.tags == ev.tags
        assert restored.ts_utc == ev.ts_utc


@pytest.mark.asyncio
async def test_tm500_replay_fixture_missing_file_raises() -> None:
    adapter = TM500Adapter(base_url="http://tm500.local")
    with pytest.raises(FileNotFoundError):
        async for _ev in await adapter.replay_fixture(
            Path("/nonexistent/tm500_missing.jsonl")
        ):
            pass


# ─── Adapter-mode flags ─────────────────────────────────────────────────


def test_tm500_mode_is_functional() -> None:
    assert TM500Adapter.mode == "functional"
    inst = TM500Adapter(base_url="http://tm500.local")
    assert inst.mode == "functional"


def test_power_meter_mode_is_documented_only() -> None:
    assert PowerMeterAdapter.mode == "documented-only"


def test_xhaul_advisor_mode_is_documented_only() -> None:
    assert XhaulAdvisorAdapter.mode == "documented-only"


def test_teravm_core_mode_is_documented_only() -> None:
    assert TeraVMCoreAdapter.mode == "documented-only"


# ─── Documented-only NotImplementedError ────────────────────────────────


@pytest.mark.asyncio
async def test_power_meter_connect_raises_with_ops_hint() -> None:
    adapter = PowerMeterAdapter(host="ru-bench-1.lab")
    with pytest.raises(NotImplementedError) as ei:
        await adapter.connect()
    # Helpful, greppable hint.
    assert "contact VIAVI ops" in str(ei.value)


@pytest.mark.asyncio
async def test_xhaul_advisor_connect_raises_with_ops_hint() -> None:
    adapter = XhaulAdvisorAdapter(capture_iface="eth4")
    with pytest.raises(NotImplementedError) as ei:
        await adapter.connect()
    assert "contact VIAVI ops" in str(ei.value)


@pytest.mark.asyncio
async def test_teravm_core_connect_raises_with_ops_hint() -> None:
    adapter = TeraVMCoreAdapter(amf_url="http://amf.lab:8080")
    with pytest.raises(NotImplementedError) as ei:
        await adapter.connect()
    assert "contact VIAVI ops" in str(ei.value)


# ─── TM500 connect/close lifecycle (mocked httpx) ───────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tm500_connect_and_close_lifecycle() -> None:
    """connect() opens an httpx client; close() releases it; both idempotent."""

    adapter = TM500Adapter(base_url="http://tm500.local", timeout_s=2.0)
    assert adapter._client is None
    await adapter.connect()
    assert isinstance(adapter._client, httpx.AsyncClient)
    # Idempotent connect.
    first = adapter._client
    await adapter.connect()
    assert adapter._client is first
    await adapter.close()
    assert adapter._client is None
    # Idempotent close.
    await adapter.close()
    assert adapter._client is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tm500_stream_against_mock_transport() -> None:
    """stream() yields TelemetryEvents from a mocked /telemetry endpoint."""

    # Build a JSONL response from the fixture so the test is self-contained.
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
    expected_event_ids = [
        json.loads(line)["event_id"]
        for line in fixture_text.splitlines()
        if line.strip()
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/telemetry"
        return httpx.Response(200, text=fixture_text)

    transport = httpx.MockTransport(handler)
    adapter = TM500Adapter(base_url="http://tm500.local")
    # Inject the mock-transport client directly so connect() is a no-op.
    adapter._client = httpx.AsyncClient(
        base_url="http://tm500.local",
        transport=transport,
    )
    try:
        seen_ids: list[str] = []
        async for ev in adapter.stream():
            assert isinstance(ev, TelemetryEvent)
            seen_ids.append(ev.event_id)
        assert seen_ids == expected_event_ids
    finally:
        await adapter.close()
