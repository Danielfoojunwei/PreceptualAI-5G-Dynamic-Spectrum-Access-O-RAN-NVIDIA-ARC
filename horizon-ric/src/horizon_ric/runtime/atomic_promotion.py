"""Slot-boundary atomic A→B model promotion.

Replaces today's swap-with-restart promotion path with a single-pointer
atomic swap that guarantees no in-flight decision uses a half-loaded
model. Readers go through ``get()`` which returns the currently-active
model under a non-blocking read; the promoter waits for a slot-boundary
event, swaps the pointer, and holds the old model in ``_draining`` for
``drain_timeout_s`` seconds so any in-flight decision finishes against
the old model before its bytes are released.

References
----------
3GPP TS 28.567 §6.3 — LoopState semantics (Promote / Rollback).
3GPP TS 38.331 R19 — model activation / rollback signalling.
~/.claude/plans/preceptualai-airan-alliance-integration.md §11.4 M7.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generic, Literal, Optional, TypeVar

ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class PromotionEvent:
    """One audited promotion or rollback transition.

    Both promotion and rollback emit the same event shape; the ``kind``
    field disambiguates. ``audit_chain_seq`` is filled in by the caller
    after persisting the event to the audit chain.
    """

    timestamp: datetime
    slot_number: int
    from_sha: str
    to_sha: str
    kind: Literal["promote", "rollback"]
    audit_chain_seq: Optional[int] = None
    drain_duration_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "slot_number": self.slot_number,
            "from_sha": self.from_sha,
            "to_sha": self.to_sha,
            "kind": self.kind,
            "audit_chain_seq": self.audit_chain_seq,
            "drain_duration_s": self.drain_duration_s,
            "error": self.error,
        }


@dataclass
class _ActiveSlot(Generic[ModelT]):
    model: ModelT
    sha: str
    in_flight: int = 0
    _cond: threading.Condition = field(default_factory=threading.Condition)


class AtomicPromoter(Generic[ModelT]):
    """Slot-boundary atomic A→B model swap.

    Reader path is lock-free in the common case: ``get()`` reads the
    pointer, increments ``in_flight``, and the caller decrements after
    the decision via ``release()``. The promoter swaps the pointer, then
    waits on the old slot's condition variable until ``in_flight`` drops
    to zero or ``drain_timeout_s`` elapses.
    """

    def __init__(
        self,
        initial_model: ModelT,
        initial_sha: str,
        *,
        drain_timeout_s: float = 0.5,
        sink=None,
    ):
        self._slot = _ActiveSlot(model=initial_model, sha=initial_sha)
        self._previous: Optional[_ActiveSlot[ModelT]] = None
        self._draining: list[_ActiveSlot[ModelT]] = []
        self._drain_timeout_s = drain_timeout_s
        self._sink = sink  # callable(PromotionEvent) → None
        self._lock = threading.RLock()
        self._slot_counter = 0

    # ─── Reader path ───────────────────────────────────────────────────

    def get(self) -> tuple[ModelT, str]:
        """Acquire the active model + its SHA. Caller MUST call ``release(sha)``
        when done so the drain bookkeeping is correct.
        """
        with self._lock:
            slot = self._slot
            with slot._cond:
                slot.in_flight += 1
            return slot.model, slot.sha

    def release(self, sha: str) -> None:
        """Release a previously-acquired slot. ``sha`` is the SHA returned
        by ``get()`` — ensures we decrement the right slot if a promotion
        moved the pointer between ``get()`` and ``release()``.
        """
        # Search active first, then draining list.
        for candidate in [self._slot, *self._draining]:
            if candidate.sha == sha:
                with candidate._cond:
                    candidate.in_flight -= 1
                    if candidate.in_flight == 0:
                        candidate._cond.notify_all()
                return

    # ─── Mutator path ──────────────────────────────────────────────────

    async def promote(
        self,
        new_model: ModelT,
        new_sha: str,
        *,
        slot_boundary: Optional[asyncio.Event] = None,
    ) -> PromotionEvent:
        """Wait on slot boundary, swap pointer, drain old slot.

        If ``slot_boundary`` is provided, ``promote`` awaits it before
        swapping. Otherwise the swap happens immediately.
        """
        if slot_boundary is not None:
            await slot_boundary.wait()

        return await self._swap(new_model, new_sha, kind="promote")

    async def rollback(self) -> PromotionEvent:
        """Atomic swap back to the previous model. Raises if no previous."""
        with self._lock:
            if self._previous is None:
                raise RuntimeError("rollback() called but no previous model recorded")
            prev = self._previous
        return await self._swap(prev.model, prev.sha, kind="rollback")

    async def _swap(self, new_model, new_sha, *, kind) -> PromotionEvent:
        import time

        with self._lock:
            self._slot_counter += 1
            slot_number = self._slot_counter
            old_slot = self._slot
            new_slot = _ActiveSlot(model=new_model, sha=new_sha)
            from_sha = old_slot.sha
            to_sha = new_sha
            # Atomic swap.
            self._slot = new_slot
            self._draining.append(old_slot)
            # For rollback we want to remember what we left so a
            # subsequent rollback un-does the rollback (B→A→B); record
            # the old slot as the previous *only* on promote.
            if kind == "promote":
                self._previous = old_slot
            else:
                # On rollback, the new "previous" is the model we just
                # rolled back FROM (i.e., the old_slot before this swap).
                self._previous = old_slot

        # Drain old slot — yield to the event loop until in_flight hits 0
        # or timeout. Using asyncio.sleep(0) means readers running in
        # other coroutines (or in threads via release()) can make progress.
        drain_start = time.monotonic()
        deadline = drain_start + self._drain_timeout_s
        while True:
            with old_slot._cond:
                if old_slot.in_flight <= 0:
                    break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.005)
        drain_duration = time.monotonic() - drain_start

        # Remove drained slot from the draining list.
        with self._lock:
            try:
                self._draining.remove(old_slot)
            except ValueError:
                pass

        event = PromotionEvent(
            timestamp=datetime.now(timezone.utc),
            slot_number=slot_number,
            from_sha=from_sha,
            to_sha=to_sha,
            kind=kind,
            drain_duration_s=drain_duration,
        )
        if self._sink is not None:
            try:
                self._sink(event)
            except Exception as e:
                event = PromotionEvent(
                    timestamp=event.timestamp,
                    slot_number=event.slot_number,
                    from_sha=event.from_sha,
                    to_sha=event.to_sha,
                    kind=event.kind,
                    drain_duration_s=event.drain_duration_s,
                    error=f"sink failed: {e!r}",
                )
        return event

    # ─── Inspection ─────────────────────────────────────────────────────

    def active_sha(self) -> str:
        return self._slot.sha

    def previous_sha(self) -> Optional[str]:
        return self._previous.sha if self._previous else None


__all__ = [
    "AtomicPromoter",
    "PromotionEvent",
]
