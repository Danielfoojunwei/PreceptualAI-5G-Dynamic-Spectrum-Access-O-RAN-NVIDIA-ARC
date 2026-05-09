"""Tests for the NVIDIA ARC-OTA real-time I/Q feed consumer.

Pins the wire shape an ARC-OTA HTTP bridge sees: the consumer polls
``/arc-ota/poll``, translates each capture message into a
``TelemetryEvent``, exposes connect/disconnect state via the
``arc_ota_connected`` Prometheus gauge, and reconnects on failure with
exponential backoff. The legacy ``nvidia_arc`` module is checked for
its ``DeprecationWarning`` and that the new symbols re-export cleanly.
"""

from __future__ import annotations

import asyncio
import importlib
import warnings
from typing import Any

import httpx
import pytest

from horizon_ric.integrations.nvidia_arc_ota import (
    ARCOTAConfig,
    ARCOTAConsumer,
    arc_ota_connected,
    arc_ota_msgs_total,
)
from horizon_ric.io.schemas import TelemetryEvent


def _make_handler(scripted_responses: list[Any]):
    """Build a MockTransport handler that walks a script of responses.

    Each scripted entry is either an ``httpx.Response`` (returned
    verbatim) or an ``Exception`` (raised on the matching call). The
    final entry is repeated forever once exhausted.
    """

    state = {"i": 0}

    async def handler(req: httpx.Request) -> httpx.Response:
        idx = min(state["i"], len(scripted_responses) - 1)
        entry = scripted_responses[idx]
        state["i"] += 1
        if isinstance(entry, Exception):
            raise entry
        return entry

    return handler


@pytest.mark.asyncio
async def test_connect_sets_gauge_and_is_connected_true():
    """connect() flips the gauge to 1 and is_connected() returns True."""

    arc_ota_connected.set(0)
    cfg = ARCOTAConfig(endpoint_url="https://arc-ota.local")
    handler = _make_handler([httpx.Response(200, json={"messages": [], "done": True})])
    consumer = ARCOTAConsumer(cfg, transport=httpx.MockTransport(handler))

    await consumer.connect()
    try:
        assert consumer.is_connected() is True
        # Snapshot the gauge value via the prometheus_client public API.
        # Gauge._value.get() is the documented test-time accessor.
        assert arc_ota_connected._value.get() == 1.0
    finally:
        await consumer.close()
    assert consumer.is_connected() is False


@pytest.mark.asyncio
async def test_translate_yields_valid_telemetry_event():
    """One ARC-OTA wire message round-trips into a schema-valid event."""

    arc_msg = {
        "capture_type": "uplink_iq",
        "ingest_ts_ns": 1_700_000_000_000_000_000,
        "event_id": "arc-ota-fixture-1",
        "cell_id": "cell-7",
        "payload": {"iq_len": 1024, "rssi_dbm": -85.0},
    }
    handler = _make_handler(
        [httpx.Response(200, json={"messages": [arc_msg], "done": True})]
    )
    cfg = ARCOTAConfig(endpoint_url="https://arc-ota.local")
    consumer = ARCOTAConsumer(cfg, transport=httpx.MockTransport(handler))
    await consumer.connect()
    try:
        events: list[TelemetryEvent] = []
        async for ev in consumer.stream():
            events.append(ev)
    finally:
        await consumer.close()

    assert len(events) == 1
    ev = events[0]
    # Schema-validity: round-trip through pydantic to confirm.
    revalidated = TelemetryEvent.model_validate(ev.model_dump())
    assert revalidated.event_id == "arc-ota-fixture-1"
    assert revalidated.modality == "spectrum_iq"
    assert revalidated.source_id == "nvidia_arc_ota"
    assert revalidated.tags["arc_ota_capture_type"] == "uplink_iq"
    assert revalidated.payload["iq_len"] == 1024
    # cell_id was lifted from the top-level wire message into the payload.
    assert revalidated.payload["cell_id"] == "cell-7"
    assert revalidated.ts_utc.tzinfo is not None


@pytest.mark.asyncio
async def test_disconnect_then_reconnect_flips_gauge():
    """Transport error → gauge to 0 → next poll succeeds → gauge back to 1."""

    arc_ota_connected.set(0)
    arc_msg = {
        "capture_type": "fapi_l1",
        "ingest_ts_ns": 1_700_000_000_000_000_000,
        "payload": {"mcs": 12},
    }
    # First call raises (server hung up); second call returns the message
    # and a "done" sentinel so the stream can exit.
    script = [
        httpx.ConnectError("server hung up"),
        httpx.Response(200, json={"messages": [arc_msg], "done": True}),
    ]
    handler = _make_handler(script)
    cfg = ARCOTAConfig(
        endpoint_url="https://arc-ota.local",
        reconnect_backoff_max_s=0.01,  # keep the test fast
    )
    consumer = ARCOTAConsumer(cfg, transport=httpx.MockTransport(handler))

    gauge_observations: list[float] = []
    await consumer.connect()
    try:
        async for ev in consumer.stream():
            # We observe at least one event despite the initial failure.
            gauge_observations.append(arc_ota_connected._value.get())
            assert ev.modality == "kpm_5g"
    finally:
        await consumer.close()

    # The gauge was 1 at yield-time (post-recovery), proving the
    # reconnect path raised it back up after the disconnect dropped it.
    assert any(v == 1.0 for v in gauge_observations)
    # And close() drops it back to 0.
    assert arc_ota_connected._value.get() == 0.0


def test_legacy_module_emits_deprecation_warning_and_reexports():
    """Importing the legacy module raises DeprecationWarning; new symbols
    are reachable from both old and new module paths."""

    import horizon_ric.integrations.nvidia_arc as legacy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Force a fresh module load so the module-level warnings.warn fires.
        importlib.reload(legacy)

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected a DeprecationWarning on legacy import"
    assert "nvidia_arc_ota" in str(deprecations[0].message)

    # Re-exports work from both module paths.
    from horizon_ric.integrations.nvidia_arc import ARCOTAConfig as LegacyConfig
    from horizon_ric.integrations.nvidia_arc import ARCOTAConsumer as LegacyConsumer

    assert LegacyConfig is ARCOTAConfig
    assert LegacyConsumer is ARCOTAConsumer

    # Legacy management-plane symbols still exist (they are NOT a stub).
    assert hasattr(legacy, "ARCClient")
    assert hasattr(legacy, "ARCClientConfig")


@pytest.mark.asyncio
async def test_close_cancels_background_stream_task_cleanly():
    """A running stream task is cancelled cleanly when close() is awaited."""

    # Build a handler that returns one batch then blocks indefinitely on
    # subsequent calls — this gives us a stream that's still inside an
    # await when close() arrives.
    block_event = asyncio.Event()
    delivered = asyncio.Event()

    async def handler(req: httpx.Request) -> httpx.Response:
        if not delivered.is_set():
            delivered.set()
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "capture_type": "kpm",
                            "ingest_ts_ns": 1_700_000_000_000_000_000,
                            "payload": {"counter": 1},
                        }
                    ]
                },
            )
        # Subsequent polls hang until the test releases them, simulating
        # a long-poll the consumer is waiting on at close() time.
        await block_event.wait()
        return httpx.Response(200, json={"messages": [], "done": True})

    cfg = ARCOTAConfig(
        endpoint_url="https://arc-ota.local", grpc_timeout_s=10.0
    )
    consumer = ARCOTAConsumer(cfg, transport=httpx.MockTransport(handler))
    await consumer.connect()

    received: list[TelemetryEvent] = []

    async def drain() -> None:
        async for ev in consumer.stream():
            received.append(ev)

    task = asyncio.create_task(drain())
    consumer._stream_task = task
    # Wait until the consumer has actually drained the first batch.
    await delivered.wait()
    # Give the task a tick to yield the event.
    for _ in range(10):
        if received:
            break
        await asyncio.sleep(0)

    await consumer.close()
    # After close, the stream task is finished (cancelled or completed).
    assert task.done()
    assert consumer.is_connected() is False
    # We did receive at least the first event before close.
    assert len(received) >= 1
    # Unblock anything still waiting (defensive — task should be done already).
    block_event.set()
