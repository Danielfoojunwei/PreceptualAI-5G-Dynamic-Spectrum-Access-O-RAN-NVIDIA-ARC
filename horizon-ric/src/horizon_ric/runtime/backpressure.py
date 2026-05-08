"""Bounded async queue with explicit backpressure policies.

Wraps `asyncio.Queue` with three drop policies suitable for different
ingest paths in the rApp:

  * ``oldest``  — drop the oldest enqueued item when full.
                  Right for live telemetry: prefer a fresh sample over
                  a stale one. This is the default for the encoder feed.
  * ``newest``  — drop the new item when full (refuse).
                  Right when downstream consumers must see a coherent
                  sliding window — e.g. evidence-store batching.
  * ``block``   — block the producer until space is available.
                  Right for batch ingest where we'd rather slow down
                  than lose data (e.g. replay from disk).

Every queue exports a Prometheus gauge:
    horizon_queue_depth{name="<queue-name>"}
and a counter:
    horizon_queue_drops_total{name="<queue-name>", policy="oldest|newest"}

Stable structlog event names:

    horizon.bp.queue_full   — drop occurred (per drop, sampled at info)
    horizon.bp.put          — debug: depth crossed warn threshold
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Final, Generic, Iterable, Literal, TypeVar

import structlog

logger = structlog.get_logger(__name__)


# Lazily import prometheus metrics so unit tests of the queue itself don't
# require prometheus_client just to import this module. (It's already a
# project dep, so this is mostly for clarity.)
try:
    from prometheus_client import Counter, Gauge

    QUEUE_DEPTH = Gauge(
        "horizon_queue_depth",
        "Current depth of a PreceptualAI backpressure queue.",
        labelnames=("name",),
    )
    QUEUE_DROPS = Counter(
        "horizon_queue_drops_total",
        "Items dropped because the backpressure queue was full.",
        labelnames=("name", "policy"),
    )
    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    _METRICS_AVAILABLE = False
    QUEUE_DEPTH = None  # type: ignore[assignment]
    QUEUE_DROPS = None  # type: ignore[assignment]


DropPolicy = Literal["oldest", "newest", "block"]
DEFAULT_MAX_DEPTH: Final[int] = 10_000

T = TypeVar("T")


class QueueFullDropped(Exception):
    """Raised when policy='newest' and the queue is full.

    Callers can catch this if they want to take a different action than
    silently logging.
    """


class BackpressureQueue(Generic[T]):
    """Bounded queue with explicit drop policy and metrics.

    Use `put()` from producers and `get()` from consumers. Both are
    async; `put()` is non-blocking under `oldest`/`newest` policies.
    """

    def __init__(
        self,
        name: str,
        max_depth: int = DEFAULT_MAX_DEPTH,
        drop_policy: DropPolicy = "oldest",
        dlq_path: Path | str | None = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {max_depth}")
        if drop_policy not in ("oldest", "newest", "block"):
            raise ValueError(f"unknown drop_policy: {drop_policy!r}")
        self.name = name
        self.max_depth = max_depth
        self.drop_policy = drop_policy
        self._q: asyncio.Queue[T] = asyncio.Queue(maxsize=max_depth)
        self._dropped = 0
        # Dead-letter queue (JSONL on disk). When set, every dropped
        # event is appended to this file so the chain remains a complete
        # record of what was *observed* — even if it is "we observed
        # that we elided this event" (Devil-B finding #6/#14).
        self._dlq_path: Path | None = Path(dlq_path) if dlq_path else None
        self._dlq_lock = threading.Lock()
        if self._dlq_path is not None:
            self._dlq_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._dlq_path.exists():
                self._dlq_path.touch()
        self._update_depth_gauge()

    # ---- introspection ----
    def qsize(self) -> int:
        return self._q.qsize()

    def full(self) -> bool:
        return self._q.full()

    def empty(self) -> bool:
        return self._q.empty()

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def dlq_path(self) -> Path | None:
        return self._dlq_path

    # ---- DLQ ----
    def _flush_to_dlq(self, item: Any, *, policy: str) -> None:
        """Append a dropped item to the DLQ file as a JSON line.

        The line carries:

          * ``ts``     — wall-clock seconds since epoch
          * ``queue``  — queue name
          * ``policy`` — drop policy that produced the drop
          * ``item``   — the original payload, JSON-serialised when
                         possible; otherwise the ``repr()`` is recorded
                         so a non-JSON event still leaves a trace.

        Failures to write are logged but never raised — the producer's
        critical path must continue. (If the DLQ disk is full, the
        evidence store's disk-full alarm fires upstream.)
        """
        if self._dlq_path is None:
            return
        try:
            payload: Any
            try:
                payload = json.loads(json.dumps(item, default=str))
            except (TypeError, ValueError):
                payload = repr(item)
            line = json.dumps(
                {
                    "ts": time.time(),
                    "queue": self.name,
                    "policy": policy,
                    "item": payload,
                },
                sort_keys=True,
            )
            with self._dlq_lock:
                with self._dlq_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
        except OSError as exc:
            logger.error(
                "horizon.bp.dlq.write_failed",
                queue=self.name,
                error=str(exc),
            )

    def replay_dlq(self) -> Iterable[dict[str, Any]]:
        """Drain the DLQ file: yield every recorded drop, then truncate.

        Returns an iterator over the JSON-decoded entries (the same
        dicts that were written by ``_flush_to_dlq``). After every line
        has been yielded the file is truncated to zero bytes so a
        re-call doesn't double-replay.

        Use as::

            for entry in queue.replay_dlq():
                await queue.put(entry["item"])
        """
        if self._dlq_path is None or not self._dlq_path.exists():
            return iter(())

        with self._dlq_lock:
            with self._dlq_path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
            self._dlq_path.write_text("")  # truncate

        def _gen() -> Iterable[dict[str, Any]]:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "horizon.bp.dlq.bad_line",
                        queue=self.name,
                        line=line[:200],
                    )

        return _gen()

    # ---- producer side ----
    async def put(self, item: T) -> bool:
        """Enqueue `item`. Returns True if accepted, False if dropped.

        Under ``oldest`` policy, an existing item is dropped to make room
        for the new one (this returns True — the new item was accepted).
        """
        if self.drop_policy == "block":
            await self._q.put(item)
            self._update_depth_gauge()
            return True

        if not self._q.full():
            self._q.put_nowait(item)
            self._update_depth_gauge()
            return True

        # Queue is full.
        if self.drop_policy == "newest":
            self._dropped += 1
            self._inc_drop_counter("newest")
            self._flush_to_dlq(item, policy="newest")
            logger.warning(
                "horizon.bp.queue_full",
                queue=self.name,
                policy="newest",
                action="rejected_new",
                depth=self._q.qsize(),
                total_dropped=self._dropped,
            )
            return False

        # policy == "oldest": evict one and enqueue the new.
        evicted: Any = None
        try:
            evicted = self._q.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover — race
            pass
        self._dropped += 1
        self._inc_drop_counter("oldest")
        if evicted is not None:
            self._flush_to_dlq(evicted, policy="oldest")
        self._q.put_nowait(item)
        self._update_depth_gauge()
        logger.warning(
            "horizon.bp.queue_full",
            queue=self.name,
            policy="oldest",
            action="evicted_oldest",
            depth=self._q.qsize(),
            total_dropped=self._dropped,
        )
        return True

    def put_nowait(self, item: T) -> bool:
        """Synchronous flavour. Raises QueueFullDropped on policy=newest."""
        if self.drop_policy == "block":
            # block doesn't make sense in nowait — be strict.
            self._q.put_nowait(item)
            self._update_depth_gauge()
            return True
        if not self._q.full():
            self._q.put_nowait(item)
            self._update_depth_gauge()
            return True
        if self.drop_policy == "newest":
            self._dropped += 1
            self._inc_drop_counter("newest")
            self._flush_to_dlq(item, policy="newest")
            raise QueueFullDropped(self.name)
        # oldest
        evicted: Any = None
        try:
            evicted = self._q.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover
            pass
        self._dropped += 1
        self._inc_drop_counter("oldest")
        if evicted is not None:
            self._flush_to_dlq(evicted, policy="oldest")
        self._q.put_nowait(item)
        self._update_depth_gauge()
        return True

    # ---- consumer side ----
    async def get(self) -> T:
        item = await self._q.get()
        self._update_depth_gauge()
        return item

    def get_nowait(self) -> T:
        item = self._q.get_nowait()
        self._update_depth_gauge()
        return item

    # ---- metrics ----
    def _update_depth_gauge(self) -> None:
        if _METRICS_AVAILABLE:
            try:
                QUEUE_DEPTH.labels(name=self.name).set(self._q.qsize())
            except Exception:  # pragma: no cover
                pass

    def _inc_drop_counter(self, policy: str) -> None:
        if _METRICS_AVAILABLE:
            try:
                QUEUE_DROPS.labels(name=self.name, policy=policy).inc()
            except Exception:  # pragma: no cover
                pass


__all__ = [
    "BackpressureQueue",
    "DropPolicy",
    "QueueFullDropped",
    "DEFAULT_MAX_DEPTH",
]
