"""Tests for the production circuit breaker.

Uses a real httpx.AsyncClient with `httpx.MockTransport` to simulate
upstream behaviour — the breaker itself is the real `pybreaker`-backed
implementation. Logs are captured via a structlog test processor so we
can assert on event names without coupling to formatting.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import structlog
from structlog.testing import capture_logs

from horizon_ric.runtime.circuit_breaker import (
    AsyncCircuitBreaker,
    BreakerConfig,
    CircuitBreakerError,
)


@pytest.fixture
def captured_logs():
    """Capture structlog events emitted during the test (real capture, not mock)."""
    with capture_logs() as events:
        yield events


def _make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://upstream.test",
    )


@pytest.mark.asyncio
async def test_five_consecutive_5xx_trips_breaker(captured_logs):
    cb = AsyncCircuitBreaker(BreakerConfig(name="test.cb", fail_max=5, reset_timeout=30))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"err": "down"})

    async with _make_client(handler) as client:
        async def call():
            r = await client.get("/r1/registration/v1/registration")
            r.raise_for_status()

        # Five consecutive failures must trip the breaker.
        for _ in range(5):
            with pytest.raises(httpx.HTTPStatusError):
                await cb.call(call)

        assert cb.state == "open"

        # Sixth call: rejected immediately, never hits the upstream.
        with pytest.raises(CircuitBreakerError):
            await cb.call(call)

    # Stable event names are emitted.
    names = [e.get("event") for e in captured_logs]
    assert "horizon.cb.opened" in names
    assert "horizon.cb.rejected" in names
    assert "horizon.cb.failure" in names


@pytest.mark.asyncio
async def test_4xx_does_not_trip_breaker(captured_logs):
    cb = AsyncCircuitBreaker(BreakerConfig(name="cb.4xx", fail_max=3))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"err": "not found"})

    async with _make_client(handler) as client:
        async def call():
            r = await client.get("/missing")
            r.raise_for_status()

        for _ in range(10):
            with pytest.raises(httpx.HTTPStatusError):
                await cb.call(call)

    assert cb.state == "closed"
    assert cb.fail_counter == 0


@pytest.mark.asyncio
async def test_timeout_counts_as_failure(captured_logs):
    cb = AsyncCircuitBreaker(BreakerConfig(name="cb.timeout", fail_max=2))

    async def call():
        raise httpx.TimeoutException("simulated")

    for _ in range(2):
        with pytest.raises(httpx.TimeoutException):
            await cb.call(call)

    assert cb.state == "open"


@pytest.mark.asyncio
async def test_half_open_then_closed_on_success(captured_logs):
    # Short reset_timeout so we don't wait minutes in tests.
    cb = AsyncCircuitBreaker(BreakerConfig(name="cb.recovery", fail_max=2, reset_timeout=0.5))

    failing = True

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503 if failing else 200, json={"ok": not failing})

    async with _make_client(handler) as client:
        async def call():
            r = await client.get("/x")
            r.raise_for_status()

        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                await cb.call(call)
        assert cb.state == "open"

        # Wait past reset_timeout
        await asyncio.sleep(0.6)

        # Now the upstream is healthy
        failing = False
        # First call after timeout: triggers half-open transition,
        # succeeds, breaker closes.
        await cb.call(call)
        assert cb.state == "closed"

    names = [e.get("event") for e in captured_logs]
    assert "horizon.cb.opened" in names
    # half_open and closed transitions should both fire
    assert "horizon.cb.half_open" in names
    assert "horizon.cb.closed" in names


@pytest.mark.asyncio
async def test_breaker_open_returns_immediately_no_upstream(captured_logs):
    """When OPEN, the breaker must short-circuit before the http call."""
    cb = AsyncCircuitBreaker(BreakerConfig(name="cb.short", fail_max=1, reset_timeout=60))

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(503)

    async with _make_client(handler) as client:
        async def call():
            r = await client.get("/x")
            r.raise_for_status()

        with pytest.raises(httpx.HTTPStatusError):
            await cb.call(call)
        assert call_count == 1
        assert cb.state == "open"

        start = time.monotonic()
        with pytest.raises(CircuitBreakerError):
            await cb.call(call)
        elapsed = time.monotonic() - start
        # Must be near-instant (no network round-trip)
        assert elapsed < 0.05
        assert call_count == 1, "upstream must not have been hit"


@pytest.mark.asyncio
async def test_connect_error_counts_as_failure():
    cb = AsyncCircuitBreaker(BreakerConfig(name="cb.connect", fail_max=2))

    async def call():
        raise httpx.ConnectError("nodename nor servname provided")

    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await cb.call(call)
    assert cb.state == "open"
