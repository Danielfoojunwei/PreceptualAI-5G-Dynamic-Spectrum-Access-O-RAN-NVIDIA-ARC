"""Atomic A→B model promotion tests."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from horizon_ric.runtime.atomic_promotion import AtomicPromoter, PromotionEvent

pytestmark = pytest.mark.asyncio


class TestBasicSwap:
    async def test_promote_swaps_active_pointer(self):
        p = AtomicPromoter("model-A", "sha-A")
        assert p.active_sha() == "sha-A"
        ev = await p.promote("model-B", "sha-B")
        assert p.active_sha() == "sha-B"
        assert ev.from_sha == "sha-A"
        assert ev.to_sha == "sha-B"
        assert ev.kind == "promote"

    async def test_get_returns_active_model(self):
        p = AtomicPromoter("model-A", "sha-A")
        m, s = p.get()
        assert m == "model-A"
        assert s == "sha-A"
        p.release(s)
        await p.promote("model-B", "sha-B")
        m, s = p.get()
        assert m == "model-B"
        p.release(s)


class TestRollback:
    async def test_rollback_returns_to_previous(self):
        p = AtomicPromoter("model-A", "sha-A")
        await p.promote("model-B", "sha-B")
        ev = await p.rollback()
        assert ev.kind == "rollback"
        assert p.active_sha() == "sha-A"

    async def test_rollback_without_previous_raises(self):
        p = AtomicPromoter("model-A", "sha-A")
        with pytest.raises(RuntimeError):
            await p.rollback()

    async def test_rollback_under_drain_timeout(self):
        p = AtomicPromoter("model-A", "sha-A", drain_timeout_s=0.5)
        await p.promote("model-B", "sha-B")
        t0 = time.monotonic()
        await p.rollback()
        assert (time.monotonic() - t0) < 0.6  # rollback completes within slot


class TestConcurrentReadsUnderPromote:
    async def test_readers_always_see_complete_model(self):
        """Spin readers; perform 5 promotions; assert no torn reads."""
        p = AtomicPromoter("M0", "sha0")
        seen_shas: list[str] = []
        stop = asyncio.Event()

        async def reader():
            while not stop.is_set():
                m, s = p.get()
                # Validate the (model, sha) pair is consistent: the
                # model literal must equal "M" + the sha digit.
                expected_idx = s.replace("sha", "")
                assert m == f"M{expected_idx}", f"torn read: {m=} {s=}"
                seen_shas.append(s)
                p.release(s)
                await asyncio.sleep(0)

        readers = [asyncio.create_task(reader()) for _ in range(8)]
        await asyncio.sleep(0.01)
        for i in range(1, 6):
            await p.promote(f"M{i}", f"sha{i}")
            await asyncio.sleep(0.01)
        stop.set()
        for r in readers:
            await r
        # We should have observed multiple distinct shas across the run
        assert len(set(seen_shas)) >= 2


class TestDrainTimeout:
    async def test_drain_holds_until_reader_releases(self):
        p = AtomicPromoter("M0", "sha0", drain_timeout_s=2.0)

        # Acquire but don't release yet
        m, s = p.get()
        assert s == "sha0"

        promote_task = asyncio.create_task(p.promote("M1", "sha1"))
        # Give promote time to swap pointer + start draining
        await asyncio.sleep(0.05)
        assert p.active_sha() == "sha1"  # pointer already moved

        # Promote_task is awaiting drain. Release.
        p.release(s)
        ev = await promote_task
        assert ev.drain_duration_s < 1.5

    async def test_drain_timeout_expires_gracefully(self):
        p = AtomicPromoter("M0", "sha0", drain_timeout_s=0.1)
        # Acquire but never release
        _m, s = p.get()
        ev = await p.promote("M1", "sha1")
        # Drain should have timed out at ~0.1 s
        assert 0.08 < ev.drain_duration_s < 0.5
        # But pointer is on the new model
        assert p.active_sha() == "sha1"
        p.release(s)  # cleanup


class TestSink:
    async def test_sink_invoked_with_event(self):
        captured: list[PromotionEvent] = []
        p = AtomicPromoter("M0", "sha0", sink=captured.append)
        await p.promote("M1", "sha1")
        await p.promote("M2", "sha2")
        await p.rollback()
        assert len(captured) == 3
        assert captured[0].kind == "promote"
        assert captured[2].kind == "rollback"

    async def test_sink_failure_recorded_in_event_error(self):
        def raising(_e):
            raise RuntimeError("audit chain unreachable")

        p = AtomicPromoter("M0", "sha0", sink=raising)
        ev = await p.promote("M1", "sha1")
        assert ev.error is not None
        assert "audit chain" in ev.error
        # Pointer still moved despite sink failure
        assert p.active_sha() == "sha1"


class TestEventSerialisation:
    async def test_promotion_event_to_dict(self):
        p = AtomicPromoter("M0", "sha0")
        ev = await p.promote("M1", "sha1")
        d = ev.to_dict()
        assert d["kind"] == "promote"
        assert d["from_sha"] == "sha0"
        assert d["to_sha"] == "sha1"
        assert "T" in d["timestamp"]
        assert d["slot_number"] == 1
