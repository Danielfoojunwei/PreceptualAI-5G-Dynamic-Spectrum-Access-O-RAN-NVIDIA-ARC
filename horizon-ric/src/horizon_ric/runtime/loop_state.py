"""3GPP TS 28.567 LoopState machine — closed-loop AI/ML lifecycle governance.

Every model promotion / rollback / retrain transitions a deterministic
state machine:

    Idle → Retrain → Validate → Promote → Monitor
                              ↘ Idle      ↓     ↘ Rollback → Idle
                                                ↘ Retrain (re-tune)

Allowed edges are the ONLY edges; an attempt to take any other edge raises
``InvalidTransitionError``. Every transition is appended to a thread-safe
in-memory history and (optionally) emitted to the audit chain via the
provided ``EvidenceStore``-shaped sink.

References
----------
3GPP TS 28.567 §6.3 — Closed-loop automation, LoopState semantics.
3GPP TS 28.105 §7.4 — AI/ML inference function lifecycle.
ETSI ZSM 009-1 / 009-2 — Closed-loop governance.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional


class LoopState(str, Enum):
    """3GPP TS 28.567 §6.3 LoopState enumeration.

    Stored as ``str`` so the value round-trips through JSON without
    custom serialisers.
    """

    IDLE = "Idle"
    RETRAIN = "Retrain"
    VALIDATE = "Validate"
    PROMOTE = "Promote"
    MONITOR = "Monitor"
    ROLLBACK = "Rollback"


# Allowed edges. Any (from, to) pair NOT in this set is illegal.
# Encodes the lifecycle from `~/.claude/plans/preceptualai-airan-alliance-integration.md` §11.4 M9.
_ALLOWED_EDGES: frozenset[tuple[LoopState, LoopState]] = frozenset(
    {
        (LoopState.IDLE, LoopState.RETRAIN),
        (LoopState.RETRAIN, LoopState.VALIDATE),
        (LoopState.VALIDATE, LoopState.PROMOTE),
        (LoopState.VALIDATE, LoopState.IDLE),  # rejected at validation
        (LoopState.PROMOTE, LoopState.MONITOR),
        (LoopState.MONITOR, LoopState.ROLLBACK),
        (LoopState.MONITOR, LoopState.RETRAIN),  # drift triggers re-tune
        (LoopState.ROLLBACK, LoopState.IDLE),
    }
)


class InvalidTransitionError(RuntimeError):
    """Raised on an attempt to take an edge not in ``_ALLOWED_EDGES``."""


@dataclass(frozen=True)
class StateTransition:
    """One audited LoopState change.

    Persisted to history and (optionally) emitted to the audit chain.
    """

    from_state: LoopState
    to_state: LoopState
    timestamp: datetime
    reason: str
    actor: str = "system"
    model_sha: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "actor": self.actor,
            "model_sha": self.model_sha,
        }


# Sink callable shape: takes a StateTransition, returns whatever (we ignore).
SinkFn = Callable[[StateTransition], None]


class LoopStateMachine:
    """Thread-safe TS 28.567 LoopState machine.

    Construct with an initial state (default ``IDLE``) and an optional
    ``sink`` that receives every applied transition (so callers can wire
    the audit chain). The history is append-only; ``current_state`` is
    derived from the last transition.
    """

    def __init__(
        self,
        initial: LoopState = LoopState.IDLE,
        *,
        sink: Optional[SinkFn] = None,
        actor: str = "system",
    ):
        self._state: LoopState = initial
        self._history: list[StateTransition] = []
        self._sink: Optional[SinkFn] = sink
        self._actor: str = actor
        self._lock = threading.RLock()

    # ─── Accessors ─────────────────────────────────────────────────────

    def current_state(self) -> LoopState:
        with self._lock:
            return self._state

    def history(self) -> list[StateTransition]:
        """Return a *copy* of history. Append-only contract preserved."""
        with self._lock:
            return list(self._history)

    # ─── Mutations ─────────────────────────────────────────────────────

    def transition(
        self,
        new_state: LoopState,
        reason: str,
        *,
        actor: Optional[str] = None,
        model_sha: Optional[str] = None,
    ) -> StateTransition:
        """Apply a transition. Raises ``InvalidTransitionError`` if illegal.

        The transition is recorded to history and the sink is invoked
        BEFORE the lock is released, so a sink that raises will roll the
        transition back (history will not contain it; current state
        unchanged).
        """
        if not isinstance(new_state, LoopState):
            raise TypeError(f"new_state must be LoopState, got {type(new_state)}")

        with self._lock:
            edge = (self._state, new_state)
            if edge not in _ALLOWED_EDGES:
                raise InvalidTransitionError(
                    f"illegal transition {self._state.value} → {new_state.value} "
                    f"(allowed from {self._state.value}: "
                    f"{sorted({e[1].value for e in _ALLOWED_EDGES if e[0] == self._state})})"
                )
            t = StateTransition(
                from_state=self._state,
                to_state=new_state,
                timestamp=datetime.now(timezone.utc),
                reason=reason,
                actor=actor or self._actor,
                model_sha=model_sha,
            )
            # Sink emit FIRST — if it raises, we roll back without recording.
            if self._sink is not None:
                self._sink(t)
            self._history.append(t)
            self._state = new_state
            return t

    def replay(self) -> LoopState:
        """Recompute the current state by replaying history from IDLE.

        Useful for verifying the machine's invariants after persistence.
        Returns the state that history alone implies.
        """
        with self._lock:
            state = LoopState.IDLE
            for t in self._history:
                if (state, t.to_state) not in _ALLOWED_EDGES:
                    raise InvalidTransitionError(
                        f"history is corrupt: edge {state.value}→{t.to_state.value} "
                        f"is not allowed"
                    )
                state = t.to_state
            return state


__all__ = [
    "InvalidTransitionError",
    "LoopState",
    "LoopStateMachine",
    "StateTransition",
]
