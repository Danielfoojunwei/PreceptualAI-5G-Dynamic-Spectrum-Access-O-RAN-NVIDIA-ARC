"""Tests for the bounded backpressure queue."""

from __future__ import annotations

import asyncio

import pytest

from horizon_ric.runtime.backpressure import (
    BackpressureQueue,
    QueueFullDropped,
)


@pytest.mark.asyncio
async def test_oldest_policy_evicts_head_when_full():
    q = BackpressureQueue[int]("test.oldest", max_depth=3, drop_policy="oldest")
    for i in range(3):
        await q.put(i)
    assert q.qsize() == 3

    # Fourth put evicts 0, accepts 3.
    accepted = await q.put(3)
    assert accepted is True
    assert q.qsize() == 3
    assert q.dropped == 1

    # Drain — should be 1, 2, 3.
    out = [await q.get() for _ in range(3)]
    assert out == [1, 2, 3]


@pytest.mark.asyncio
async def test_newest_policy_rejects_new_when_full():
    q = BackpressureQueue[int]("test.newest", max_depth=2, drop_policy="newest")
    await q.put(1)
    await q.put(2)
    accepted = await q.put(99)
    assert accepted is False
    assert q.dropped == 1
    out = [await q.get() for _ in range(2)]
    assert out == [1, 2]


@pytest.mark.asyncio
async def test_block_policy_blocks_until_drained():
    q = BackpressureQueue[int]("test.block", max_depth=2, drop_policy="block")
    await q.put(1)
    await q.put(2)

    put_done = asyncio.Event()

    async def producer():
        await q.put(3)
        put_done.set()

    task = asyncio.create_task(producer())
    await asyncio.sleep(0.05)
    assert not put_done.is_set(), "block policy must block when full"

    val = await q.get()
    assert val == 1
    await asyncio.wait_for(put_done.wait(), timeout=1.0)
    await task

    rest = [await q.get() for _ in range(2)]
    assert rest == [2, 3]


@pytest.mark.asyncio
async def test_get_drains_at_fixed_rate_no_drops():
    q = BackpressureQueue[int]("test.drain", max_depth=10, drop_policy="oldest")

    async def producer():
        for i in range(20):
            await q.put(i)
            await asyncio.sleep(0.01)

    async def consumer():
        seen = []
        for _ in range(20):
            seen.append(await q.get())
        return seen

    p = asyncio.create_task(producer())
    c = asyncio.create_task(consumer())
    seen = await c
    await p
    assert seen == list(range(20))
    assert q.dropped == 0


def test_invalid_max_depth_rejected():
    with pytest.raises(ValueError):
        BackpressureQueue[int]("bad", max_depth=0)
    with pytest.raises(ValueError):
        BackpressureQueue[int]("bad", max_depth=-1)


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        BackpressureQueue[int]("bad", max_depth=10, drop_policy="ignore")  # type: ignore[arg-type]


def test_put_nowait_newest_raises_when_full():
    q = BackpressureQueue[int]("nw.newest", max_depth=1, drop_policy="newest")
    q.put_nowait(1)
    with pytest.raises(QueueFullDropped):
        q.put_nowait(2)


def test_put_nowait_oldest_evicts():
    q = BackpressureQueue[int]("nw.oldest", max_depth=2, drop_policy="oldest")
    q.put_nowait(1)
    q.put_nowait(2)
    q.put_nowait(3)
    assert q.qsize() == 2
    assert q.get_nowait() == 2
    assert q.get_nowait() == 3


@pytest.mark.asyncio
async def test_depth_metric_is_updated():
    """The Prometheus gauge is set on each put/get."""
    from horizon_ric.runtime.backpressure import QUEUE_DEPTH

    q = BackpressureQueue[int]("metric.test", max_depth=5, drop_policy="oldest")
    await q.put(1)
    await q.put(2)
    # Read back the gauge value.
    val = QUEUE_DEPTH.labels(name="metric.test")._value.get()  # type: ignore[attr-defined]
    assert val == 2
    await q.get()
    val = QUEUE_DEPTH.labels(name="metric.test")._value.get()  # type: ignore[attr-defined]
    assert val == 1


@pytest.mark.asyncio
async def test_drops_metric_increments_on_eviction():
    from horizon_ric.runtime.backpressure import QUEUE_DROPS

    q = BackpressureQueue[int]("metric.drops", max_depth=2, drop_policy="oldest")
    await q.put(1)
    await q.put(2)
    await q.put(3)
    val = QUEUE_DROPS.labels(name="metric.drops", policy="oldest")._value.get()  # type: ignore[attr-defined]
    assert val == 1
