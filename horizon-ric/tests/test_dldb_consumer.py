"""Tests for `DLDBLiveConsumer` — the NVIDIA Aerial Data Lake live bridge.

We spin a tiny in-process Starlette server bound to a free port via
uvicorn-in-a-thread. The server exposes `/dldb/stream` returning a JSON
list of capture-point events drawn from a server-side queue. The
consumer connects over real loopback HTTP, exercising the actual httpx
transport, queue backpressure, capture-point filter, and reconnect
logic.

If `starlette`/`uvicorn` are unavailable on a host (parquet-only
deployment), the live tests are skipped — the consumer code itself
still imports.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import pytest

starlette = pytest.importorskip("starlette")
uvicorn = pytest.importorskip("uvicorn")

from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from horizon_ric.data.aerial import (  # noqa: E402
    DLDB_DROPPED_EVENTS_TOTAL,
    DLDBLiveConsumer,
)
from horizon_ric.io.schemas import TelemetryEvent  # noqa: E402

# ─── fake DLDB server ────────────────────────────────────────────────────


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeDLDB:
    """In-process Starlette + uvicorn fake DLDB.

    Each call to `/dldb/stream` drains and returns whatever events the
    test has pushed via :meth:`push`. The server lives in a daemon
    thread; :meth:`stop` requests a graceful shutdown.
    """

    def __init__(self) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.poll_count = 0

        async def stream(_request: Any) -> JSONResponse:
            self.poll_count += 1
            with self._lock:
                batch = self._buffer
                self._buffer = []
            return JSONResponse(batch)

        app = Starlette(routes=[Route("/dldb/stream", stream)])
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port,
            log_level="warning", lifespan="off",
        )
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        # Wait for the socket to actually accept connections.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with contextlib.closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as s:
                if s.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.02)
        raise RuntimeError(f"FakeDLDB did not come up on port {self.port}")

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5.0)

    def push(self, events: list[dict[str, Any]]) -> None:
        with self._lock:
            self._buffer.extend(events)


@pytest.fixture()
def fake_dldb() -> Iterator[FakeDLDB]:
    server = FakeDLDB()
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _make_event(idx: int, capture_point: str = "l2_fapi") -> dict[str, Any]:
    return {
        "capture_point": capture_point,
        "event_id": f"dldb-evt-{idx:04d}",
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "sequence": idx,
        "payload": {"rnti": 1000 + idx, "mcs": idx % 28},
    }


# ─── tests ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_round_trip(fake_dldb: FakeDLDB) -> None:
    """100-event streaming round-trip preserves order and yields all events."""
    events = [_make_event(i) for i in range(100)]
    fake_dldb.push(events)

    consumer = DLDBLiveConsumer(
        dldb_endpoint=fake_dldb.url,
        capture_points=["l2_fapi", "l1_fapi", "ul_iq"],
        buffer_size=512,
    )
    received: list[TelemetryEvent] = []
    started = time.perf_counter()
    try:
        async for ev in consumer.stream():
            received.append(ev)
            if len(received) == 100:
                break
    finally:
        await consumer.close()
    elapsed = time.perf_counter() - started

    assert len(received) == 100
    assert [ev.event_id for ev in received] == [
        f"dldb-evt-{i:04d}" for i in range(100)
    ]
    # Throughput sanity: must clear at least 50 events/sec on loopback.
    rate = 100 / max(elapsed, 1e-6)
    print(f"\n[dldb-roundtrip] 100 events in {elapsed*1000:.1f} ms = {rate:.0f} ev/s")
    assert rate > 50, f"throughput too low: {rate:.0f} ev/s"


@pytest.mark.asyncio
async def test_backpressure_increments_drop_counter(fake_dldb: FakeDLDB) -> None:
    """A small queue + slow consumer must drop events and bump the counter."""
    fake_dldb.push([_make_event(i) for i in range(200)])

    before = DLDB_DROPPED_EVENTS_TOTAL._value.get()  # type: ignore[attr-defined]

    consumer = DLDBLiveConsumer(
        dldb_endpoint=fake_dldb.url,
        capture_points=["l2_fapi"],
        buffer_size=4,  # tiny → forces drops as poll loop outpaces consumer
    )
    received = 0
    try:
        async for _ev in consumer.stream():
            received += 1
            # Slow consumer: yield long enough for poll loop to refill +
            # overflow the queue several times.
            await asyncio.sleep(0.05)
            if received >= 4:
                break
    finally:
        await consumer.close()

    after = DLDB_DROPPED_EVENTS_TOTAL._value.get()  # type: ignore[attr-defined]
    assert after - before > 0, (
        f"expected drops to increment; before={before} after={after}"
    )


@pytest.mark.asyncio
async def test_capture_point_filter_excludes_others(fake_dldb: FakeDLDB) -> None:
    """Subscribing to ``ul_iq`` must skip non-IQ events."""
    mixed = []
    for i in range(20):
        mixed.append(_make_event(i, capture_point="ul_iq"))
        mixed.append(_make_event(100 + i, capture_point="l2_fapi"))
        mixed.append(_make_event(200 + i, capture_point="cell_ue_context"))
    fake_dldb.push(mixed)

    consumer = DLDBLiveConsumer(
        dldb_endpoint=fake_dldb.url,
        capture_points=["ul_iq"],
        buffer_size=256,
    )
    received: list[TelemetryEvent] = []
    try:
        deadline = asyncio.get_event_loop().time() + 3.0
        async for ev in consumer.stream():
            received.append(ev)
            if len(received) >= 20 or asyncio.get_event_loop().time() > deadline:
                break
    finally:
        await consumer.close()

    assert len(received) == 20
    for ev in received:
        assert ev.tags.get("capture_point") == "ul_iq"
        assert ev.modality == "spectrum_iq"


@pytest.mark.asyncio
async def test_reconnect_after_server_restart() -> None:
    """Killing the server mid-stream must be recovered within 5 s."""
    server_a = FakeDLDB()
    server_a.start()
    port = server_a.port

    consumer = DLDBLiveConsumer(
        dldb_endpoint=server_a.url,
        capture_points=["l2_fapi"],
        buffer_size=64,
        reconnect_backoff_max_s=0.5,
    )

    server_a.push([_make_event(i) for i in range(5)])
    received: list[TelemetryEvent] = []

    async def collect() -> None:
        async for ev in consumer.stream():
            received.append(ev)

    task = asyncio.create_task(collect())
    try:
        # Wait for the first batch.
        for _ in range(50):
            if len(received) >= 5:
                break
            await asyncio.sleep(0.05)
        assert len(received) >= 5

        # Kill the server. The consumer will see ConnectError on next poll.
        server_a.stop()
        await asyncio.sleep(0.3)  # let backoff fire at least once

        # Bring up a new server on the SAME port (drains by design — fresh queue).
        server_b = FakeDLDB()
        server_b.port = port
        server_b.url = f"http://127.0.0.1:{port}"
        # Re-build the uvicorn config bound to the same port.
        async def stream(_request: Any) -> JSONResponse:
            with server_b._lock:  # type: ignore[attr-defined]
                batch = server_b._buffer  # type: ignore[attr-defined]
                server_b._buffer = []  # type: ignore[attr-defined]
            return JSONResponse(batch)
        app = Starlette(routes=[Route("/dldb/stream", stream)])
        config = uvicorn.Config(
            app, host="127.0.0.1", port=port,
            log_level="warning", lifespan="off",
        )
        server_b.server = uvicorn.Server(config)
        server_b._thread = threading.Thread(target=server_b.server.run, daemon=True)
        server_b.start()

        try:
            server_b.push([_make_event(1000 + i) for i in range(3)])
            deadline = time.time() + 5.0
            while time.time() < deadline and len(received) < 8:
                await asyncio.sleep(0.05)
            assert len(received) >= 8, (
                f"reconnect failed; received={len(received)}"
            )
        finally:
            server_b.stop()
    finally:
        await consumer.close()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


@pytest.mark.asyncio
async def test_schema_parity_with_telemetry_event(fake_dldb: FakeDLDB) -> None:
    """Emitted events must be valid TelemetryEvent instances with all canonical fields."""
    fake_dldb.push([_make_event(0, capture_point="l2_fapi")])

    consumer = DLDBLiveConsumer(
        dldb_endpoint=fake_dldb.url,
        capture_points=["l2_fapi"],
        buffer_size=8,
    )
    try:
        async for ev in consumer.stream():
            break
    finally:
        await consumer.close()

    assert isinstance(ev, TelemetryEvent)
    # Round-trip via JSON proves the event matches the canonical wire schema.
    blob = ev.model_dump_json()
    restored = TelemetryEvent.model_validate_json(blob)
    assert restored.event_id == ev.event_id
    assert restored.modality == ev.modality
    assert restored.ts_utc == ev.ts_utc
    assert restored.payload == ev.payload
    # Canonical required fields all present.
    for field in ("event_id", "modality", "source_id", "ts_utc", "schema_version"):
        assert getattr(ev, field) is not None


@pytest.mark.asyncio
async def test_clean_close_cancels_polling_task(fake_dldb: FakeDLDB) -> None:
    """``close()`` must cancel the polling task and not leak it."""
    consumer = DLDBLiveConsumer(
        dldb_endpoint=fake_dldb.url,
        capture_points=["l2_fapi"],
        buffer_size=8,
    )

    # Trigger task creation by reading one tick.
    fake_dldb.push([_make_event(0)])
    async for _ev in consumer.stream():
        break
    poll_task = consumer._poll_task
    assert poll_task is not None and not poll_task.done()

    await consumer.close()

    assert poll_task.done(), "polling task should be done after close()"
    assert consumer._client is None, "http client should be released"
    # Idempotent: a second close() must not raise.
    await consumer.close()
