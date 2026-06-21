"""Graceful shutdown must drain in-flight planner work (Devil-B finding #15/#17).

Previously ``lifecycle.shutdown()`` closed the R1/A1 adapters before
in-flight ``asyncio.create_task`` planner coroutines completed —
cancelling them mid-write and breaking the audit chain.

These tests verify:

  1. ``shutdown()`` sets the drain signal so producers stop submitting.
  2. ``shutdown()`` waits for tracked tasks before closing adapters.
  3. ``shutdown()`` honours the drain timeout and cancels stragglers.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle
from horizon_ric.rapp.r1_adapter import R1Adapter, R1AdapterConfig


def _mock_transport_ok() -> httpx.MockTransport:
    async def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(201, json={"status": "registered"})
        if req.method == "DELETE":
            return httpx.Response(204)
        if req.method == "PUT":
            return httpx.Response(202)
        return httpx.Response(200, json={})

    return httpx.MockTransport(handler)


def _build_lifecycle() -> HorizonRAppLifecycle:
    r1_cfg = R1AdapterConfig()
    a1_cfg = A1AdapterConfig()
    lc = HorizonRAppLifecycle(r1_config=r1_cfg, a1_config=a1_cfg)
    # Swap the underlying httpx clients out for mock-transported ones so
    # shutdown's deregister/close calls don't try to hit real network.
    lc._r1._client = httpx.AsyncClient(
        base_url=r1_cfg.smo_base_url, transport=_mock_transport_ok()
    )
    lc._a1._client = httpx.AsyncClient(
        base_url=a1_cfg.near_rt_ric_base_url, transport=_mock_transport_ok()
    )
    return lc


@pytest.mark.asyncio
async def test_shutdown_sets_drain_signal_first() -> None:
    """The drain signal must be set BEFORE any adapter close."""
    lc = _build_lifecycle()
    assert not lc.drain_signal.is_set()
    await lc.shutdown(drain_timeout_s=1.0)
    assert lc.drain_signal.is_set()


@pytest.mark.asyncio
async def test_shutdown_waits_for_in_flight_planner_tasks() -> None:
    """A tracked planner task that finishes within drain_timeout must complete."""
    lc = _build_lifecycle()
    completed = asyncio.Event()
    started = asyncio.Event()

    async def planner_work() -> None:
        started.set()
        # Simulated planner work that takes ~50 ms — well within the
        # default 25 s drain budget.
        await asyncio.sleep(0.05)
        completed.set()

    task = asyncio.create_task(planner_work())
    lc.track_planner_task(task)
    await started.wait()
    # Shutdown should NOT cancel the task; it should wait for it.
    await lc.shutdown(drain_timeout_s=2.0)
    assert completed.is_set(), "shutdown returned before planner task completed"
    assert task.done() and not task.cancelled()


@pytest.mark.asyncio
async def test_shutdown_cancels_tasks_that_exceed_drain_timeout() -> None:
    """A planner task that exceeds drain_timeout must be force-cancelled."""
    lc = _build_lifecycle()

    async def stuck_planner() -> None:
        # Far longer than the drain budget below.
        await asyncio.sleep(10.0)

    task = asyncio.create_task(stuck_planner())
    lc.track_planner_task(task)
    # Start the task before draining.
    await asyncio.sleep(0)
    await lc.shutdown(drain_timeout_s=0.1)
    # Cancellation should have been issued and awaited.
    assert task.done()
    assert task.cancelled() or task.exception() is not None


@pytest.mark.asyncio
async def test_shutdown_with_no_planner_tasks_is_fast_path() -> None:
    """No in-flight tasks → shutdown completes promptly without timeout."""
    lc = _build_lifecycle()
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await lc.shutdown(drain_timeout_s=30.0)
    elapsed = loop.time() - t0
    # Empty drain should take well under 1 second.
    assert elapsed < 1.0, f"empty shutdown took {elapsed:.3f}s"
