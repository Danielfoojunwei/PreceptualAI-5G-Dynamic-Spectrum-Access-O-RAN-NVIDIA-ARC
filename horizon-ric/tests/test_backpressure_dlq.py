"""Dead-letter queue for BackpressureQueue (Devil-B finding #14).

When ``drop_policy="oldest"`` evicts an event to make room, the evicted
event must be FLUSHED to a DLQ file — silent data loss is an audit-chain
break.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from horizon_ric.runtime.backpressure import BackpressureQueue, QueueFullDropped


@pytest.mark.asyncio
async def test_oldest_drop_writes_to_dlq(tmp_path: Path) -> None:
    """A 'oldest' drop must append the evicted item to the DLQ JSONL."""
    dlq = tmp_path / "queue.dlq.jsonl"
    q: BackpressureQueue[dict] = BackpressureQueue(
        name="test", max_depth=2, drop_policy="oldest", dlq_path=dlq
    )
    await q.put({"seq": 0, "v": "first"})
    await q.put({"seq": 1, "v": "second"})
    # Queue is full; this put evicts seq=0.
    accepted = await q.put({"seq": 2, "v": "third"})
    assert accepted is True
    assert q.dropped == 1

    # The evicted seq=0 must now be in the DLQ.
    lines = dlq.read_text().strip().splitlines()
    assert len(lines) == 1, lines
    entry = json.loads(lines[0])
    assert entry["queue"] == "test"
    assert entry["policy"] == "oldest"
    assert entry["item"] == {"seq": 0, "v": "first"}
    assert isinstance(entry["ts"], float)


@pytest.mark.asyncio
async def test_newest_drop_writes_to_dlq(tmp_path: Path) -> None:
    """A 'newest' drop must record the rejected new item to the DLQ.

    For policy='newest' it is the *new* item that is refused, not an
    existing one — but the data is equally lost without a DLQ.
    """
    dlq = tmp_path / "queue.dlq.jsonl"
    q: BackpressureQueue[int] = BackpressureQueue(
        name="newest_q", max_depth=1, drop_policy="newest", dlq_path=dlq
    )
    accepted = await q.put(100)
    assert accepted is True
    refused = await q.put(200)  # Should be dropped, not enqueued.
    assert refused is False
    assert q.dropped == 1

    lines = dlq.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["item"] == 200
    assert entry["policy"] == "newest"


@pytest.mark.asyncio
async def test_replay_dlq_drains_and_truncates(tmp_path: Path) -> None:
    """``replay_dlq()`` yields all entries and empties the file."""
    dlq = tmp_path / "queue.dlq.jsonl"
    q: BackpressureQueue[int] = BackpressureQueue(
        name="replay_q", max_depth=1, drop_policy="oldest", dlq_path=dlq
    )
    # Force three drops: enqueue 0, evict-on-1, evict-on-2, evict-on-3.
    await q.put(0)
    await q.put(1)
    await q.put(2)
    await q.put(3)
    assert q.dropped == 3

    replayed = list(q.replay_dlq())
    assert [e["item"] for e in replayed] == [0, 1, 2]
    # After replay, the file should be truncated; a second replay yields nothing.
    assert dlq.read_text() == ""
    assert list(q.replay_dlq()) == []


def test_dlq_disabled_when_path_is_none(tmp_path: Path) -> None:
    """No dlq_path → silent legacy behaviour (back-compat)."""
    q: BackpressureQueue[int] = BackpressureQueue(
        name="no_dlq", max_depth=1, drop_policy="oldest", dlq_path=None
    )
    asyncio.get_event_loop().run_until_complete(q.put(0))
    asyncio.get_event_loop().run_until_complete(q.put(1))
    asyncio.get_event_loop().run_until_complete(q.put(2))
    assert q.dropped == 2
    assert q.dlq_path is None
    # No DLQ file was created anywhere.
    assert list(tmp_path.iterdir()) == []
