"""Lease-based leader-election tests (Devil-B finding #12).

Uses ``FakeLeaseBackend`` (a real implementation of the LeaseBackend ABC,
backed by an in-memory store with optimistic-concurrency semantics).
The election logic itself is real — only the storage is faked.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from horizon_ric.runtime.leader_election import (
    DEFAULT_LEASE_DURATION_S,
    FakeLeaseBackend,
    FileLeaseBackend,
    LeaderElector,
    LeaseConflictError,
    LeaseRecord,
)


def _clock(start: datetime) -> tuple:
    """Return (now_fn, advance_fn) sharing one mutable epoch."""
    state = {"now": start}

    def now() -> datetime:
        return state["now"]

    def advance(seconds: float) -> None:
        state["now"] = state["now"] + timedelta(seconds=seconds)

    return now, advance


@pytest.mark.asyncio
async def test_first_replica_wins_lease() -> None:
    """A single replica with no existing lease becomes leader."""
    backend = FakeLeaseBackend()
    started: list[bool] = []
    elector = LeaderElector(
        identity="pod-a",
        backend=backend,
        on_started_leading=lambda: started.append(True),
    )
    became = await elector.try_acquire_or_renew()
    assert became is True
    assert elector.is_leader is True
    assert started == [True]
    rec = backend.get()
    assert rec is not None
    assert rec.holder_identity == "pod-a"


@pytest.mark.asyncio
async def test_second_replica_loses_to_active_leader() -> None:
    """Two replicas; the second sees a fresh lease and stays follower."""
    backend = FakeLeaseBackend()
    leader = LeaderElector(identity="pod-a", backend=backend)
    follower_started: list[bool] = []
    follower = LeaderElector(
        identity="pod-b",
        backend=backend,
        on_started_leading=lambda: follower_started.append(True),
    )
    await leader.try_acquire_or_renew()
    await follower.try_acquire_or_renew()
    assert leader.is_leader is True
    assert follower.is_leader is False
    assert follower_started == []
    # Lease still belongs to pod-a.
    assert backend.get().holder_identity == "pod-a"


@pytest.mark.asyncio
async def test_follower_takes_over_after_lease_expires() -> None:
    """When the leader stops renewing, a follower acquires the lease."""
    start = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
    now_fn, advance = _clock(start)
    backend = FakeLeaseBackend()
    started_b: list[bool] = []
    leader = LeaderElector(
        identity="pod-a",
        backend=backend,
        lease_duration_s=15.0,
        _now=now_fn,
    )
    follower = LeaderElector(
        identity="pod-b",
        backend=backend,
        lease_duration_s=15.0,
        on_started_leading=lambda: started_b.append(True),
        _now=now_fn,
    )
    # pod-a acquires.
    await leader.try_acquire_or_renew()
    assert leader.is_leader
    # pod-a goes silent (e.g. SIGKILL); pod-b polls but lease still fresh.
    advance(5.0)
    await follower.try_acquire_or_renew()
    assert follower.is_leader is False
    # Lease expires (>15s without renew).
    advance(20.0)
    await follower.try_acquire_or_renew()
    assert follower.is_leader is True
    assert started_b == [True]
    rec = backend.get()
    assert rec.holder_identity == "pod-b"
    # Transition counter incremented (pod-a → pod-b).
    assert rec.lease_transitions == 1


@pytest.mark.asyncio
async def test_leader_renew_extends_lease_lifetime() -> None:
    """Successful renew updates renewTime so lease does NOT expire."""
    start = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
    now_fn, advance = _clock(start)
    backend = FakeLeaseBackend()
    leader = LeaderElector(
        identity="pod-a",
        backend=backend,
        lease_duration_s=15.0,
        _now=now_fn,
    )
    follower = LeaderElector(
        identity="pod-b",
        backend=backend,
        lease_duration_s=15.0,
        _now=now_fn,
    )
    await leader.try_acquire_or_renew()
    # Tick every 5s, renewing.
    for _ in range(10):
        advance(5.0)
        await leader.try_acquire_or_renew()
        await follower.try_acquire_or_renew()
        assert leader.is_leader is True
        assert follower.is_leader is False
    # After 50 simulated seconds the leader still holds the lease.
    assert backend.get().holder_identity == "pod-a"


@pytest.mark.asyncio
async def test_optimistic_concurrency_prevents_split_brain() -> None:
    """Concurrent updates: only one wins, the other becomes follower."""
    backend = FakeLeaseBackend()
    a = LeaderElector(identity="pod-a", backend=backend)
    b = LeaderElector(identity="pod-b", backend=backend)
    # Both see "no lease exists"; both attempt create. Only one wins.
    res_a, res_b = await asyncio.gather(
        a.try_acquire_or_renew(), b.try_acquire_or_renew()
    )
    assert sum([res_a, res_b]) == 1
    rec = backend.get()
    assert rec.holder_identity in ("pod-a", "pod-b")


@pytest.mark.asyncio
async def test_on_started_and_stopped_callbacks_fire() -> None:
    """on_started_leading runs once on acquire; on_stopped_leading on loss."""
    start = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
    now_fn, advance = _clock(start)
    backend = FakeLeaseBackend()
    started: list[bool] = []
    stopped: list[bool] = []
    leader = LeaderElector(
        identity="pod-a",
        backend=backend,
        lease_duration_s=15.0,
        on_started_leading=lambda: started.append(True),
        on_stopped_leading=lambda: stopped.append(True),
        _now=now_fn,
    )
    challenger = LeaderElector(
        identity="pod-b", backend=backend, lease_duration_s=15.0, _now=now_fn
    )
    await leader.try_acquire_or_renew()
    assert started == [True]
    # Simulate pod-a getting cut off long enough that pod-b steals the lease.
    advance(20.0)
    await challenger.try_acquire_or_renew()
    assert challenger.is_leader
    # Now pod-a returns and tries to renew; its resource version is stale,
    # so it observes the loss and fires on_stopped_leading.
    await leader.try_acquire_or_renew()
    assert leader.is_leader is False
    assert stopped == [True]


def test_file_backend_round_trips_lease(tmp_path: Path) -> None:
    """``FileLeaseBackend`` persists and re-reads a lease on the same node."""
    backend = FileLeaseBackend(tmp_path / "lease.json")
    rec = LeaseRecord(
        holder_identity="pod-a",
        lease_duration_seconds=15,
        acquire_time=datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc),
        renew_time=datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc),
        lease_transitions=0,
    )
    created = backend.create(rec)
    fetched = backend.get()
    assert fetched is not None
    assert fetched.holder_identity == "pod-a"
    # Optimistic concurrency: stale resource version is rejected.
    with pytest.raises(LeaseConflictError):
        backend.update(rec, expected_resource_version="999")
