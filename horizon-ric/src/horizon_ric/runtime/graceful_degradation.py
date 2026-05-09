"""Graceful-degradation state machine for the rApp.

Maps a small fixed set of failure modes to documented degraded
behaviours. The state machine is owned by `DegradationController`
and observed via:

  * `controller.state` — current `DegradedState`
  * `controller.is_serving()` — drives `/readyz` (returns 503 when
    the answer is False)
  * `controller.snapshot()` — for the metrics endpoint and ops dumps

Failure-mode → degraded-mode mapping (also in RELIABILITY.md):

    SMO unreachable         → KEEP_LAST_GOOD   (continue emitting last
                                                A1 policies, audit local
                                                only, /readyz=503)
    Evidence store down     → BUFFER_TO_TMP    (write to /tmp/evidence,
                                                page operator, /readyz=503)
    Telemetry feed dropped  → HOLD_OUTPUT      (stop emitting new policies,
                                                hold previous, /readyz=503)
    Healthy                 → NORMAL           (/readyz=200)

When the underlying dependency recovers, callers invoke `recover()`
to flip back to NORMAL. Recovery is gated on a single boolean (the
caller has confirmed the dep is back) — we don't try to probe from
inside this module.

Stable structlog event names:

    horizon.degraded.entered  — transition to a degraded mode
    horizon.degraded.recover  — transition back to NORMAL
    horizon.degraded.refused  — recovery rejected (e.g., still failing)
    horizon.degraded.action   — recorded a policy emit / buffer write
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class DegradedState(str, Enum):
    NORMAL = "normal"
    KEEP_LAST_GOOD = "keep_last_good"     # SMO unreachable
    BUFFER_TO_TMP = "buffer_to_tmp"       # Evidence store down
    HOLD_OUTPUT = "hold_output"           # Telemetry feed dropped


# Failure modes a caller can declare. The mapping to DegradedState is fixed.
class FailureMode(str, Enum):
    SMO_UNREACHABLE = "smo_unreachable"
    EVIDENCE_STORE_DOWN = "evidence_store_down"
    TELEMETRY_DROPPED = "telemetry_dropped"


_FAILURE_TO_STATE: dict[FailureMode, DegradedState] = {
    FailureMode.SMO_UNREACHABLE: DegradedState.KEEP_LAST_GOOD,
    FailureMode.EVIDENCE_STORE_DOWN: DegradedState.BUFFER_TO_TMP,
    FailureMode.TELEMETRY_DROPPED: DegradedState.HOLD_OUTPUT,
}


@dataclass
class DegradationSnapshot:
    state: str
    failure_mode: str | None
    entered_at: float | None
    last_good_policies: int
    buffered_evidence_count: int
    held_decisions: int


class DegradationController:
    """Thread-safe controller for the degraded-mode FSM.

    The rApp's main loop calls `enter(failure_mode)` when it detects a
    failure (e.g. CircuitBreakerError on R1) and `recover()` once the
    dep is healthy again. Helper methods (`record_last_good_policy`,
    `buffer_evidence`, etc.) record the *behaviour* required for each
    degraded state — the caller is responsible for choosing them.
    """

    def __init__(self, tmp_buffer_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._state: DegradedState = DegradedState.NORMAL
        self._failure_mode: FailureMode | None = None
        self._entered_at: float | None = None
        self._last_good_policies: dict[str, dict[str, Any]] = {}
        self._held_decisions: list[dict[str, Any]] = []
        self._buffered_evidence: list[dict[str, Any]] = []
        self._tmp_buffer_path = tmp_buffer_path or Path("/tmp/horizon-evidence-buffer.jsonl")

    # ---- state ----
    @property
    def state(self) -> DegradedState:
        with self._lock:
            return self._state

    @property
    def failure_mode(self) -> FailureMode | None:
        with self._lock:
            return self._failure_mode

    def is_serving(self) -> bool:
        """True when /readyz should return 200. False in any degraded mode."""
        with self._lock:
            return self._state == DegradedState.NORMAL

    def snapshot(self) -> DegradationSnapshot:
        with self._lock:
            return DegradationSnapshot(
                state=self._state.value,
                failure_mode=self._failure_mode.value if self._failure_mode else None,
                entered_at=self._entered_at,
                last_good_policies=len(self._last_good_policies),
                buffered_evidence_count=len(self._buffered_evidence),
                held_decisions=len(self._held_decisions),
            )

    # ---- transitions ----
    def enter(self, mode: FailureMode, reason: str = "") -> DegradedState:
        with self._lock:
            new = _FAILURE_TO_STATE[mode]
            if self._state == new and self._failure_mode == mode:
                return new  # idempotent
            prev = self._state
            self._state = new
            self._failure_mode = mode
            self._entered_at = time.time()
        logger.warning(
            "horizon.degraded.entered",
            from_state=prev.value,
            to_state=new.value,
            failure_mode=mode.value,
            reason=reason,
        )
        # Emit X.733 alarm for O-RAN.WG10 OAM §6 fault management.
        from horizon_ric.observability.x733_alarms import default_bus

        default_bus().emit_event(
            "horizon.degraded.entered",
            additional_information={
                "from_state": prev.value,
                "to_state": new.value,
                "failure_mode": mode.value,
                "reason": reason,
            },
        )
        return new

    def recover(self, reason: str = "dep_healthy") -> DegradedState:
        with self._lock:
            if self._state == DegradedState.NORMAL:
                return self._state
            prev = self._state
            prev_mode = self._failure_mode
            self._state = DegradedState.NORMAL
            self._failure_mode = None
            self._entered_at = None
            held_count = len(self._held_decisions)
            buf_count = len(self._buffered_evidence)
        logger.info(
            "horizon.degraded.recover",
            from_state=prev.value,
            failure_mode=prev_mode.value if prev_mode else None,
            reason=reason,
            held_decisions=held_count,
            buffered_evidence=buf_count,
        )
        # Clearing alarm so the SMO sees a matched cleared/major pair.
        from horizon_ric.observability.x733_alarms import default_bus

        default_bus().emit_event(
            "horizon.degraded.recovered",
            additional_information={
                "from_state": prev.value,
                "failure_mode": prev_mode.value if prev_mode else None,
                "reason": reason,
            },
        )
        return DegradedState.NORMAL

    # ---- per-mode behaviour helpers ----

    def record_last_good_policy(
        self, policy_type: str, policy_id: str, payload: dict[str, Any]
    ) -> None:
        """Cache a successful policy emit so KEEP_LAST_GOOD can replay it."""
        with self._lock:
            self._last_good_policies[policy_type] = {
                "policy_id": policy_id,
                "payload": payload,
                "ts": time.time(),
            }

    def get_last_good_policy(self, policy_type: str) -> dict[str, Any] | None:
        with self._lock:
            return self._last_good_policies.get(policy_type)

    def buffer_evidence(self, record: dict[str, Any]) -> Path:
        """Append a record to the local /tmp buffer. Returns the buffer path."""
        with self._lock:
            self._buffered_evidence.append(record)
            self._tmp_buffer_path.parent.mkdir(parents=True, exist_ok=True)
            with self._tmp_buffer_path.open("a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(record, sort_keys=True))
                f.write("\n")
        logger.info(
            "horizon.degraded.action",
            mode=self._state.value,
            action="buffered_evidence",
            buffer_path=str(self._tmp_buffer_path),
        )
        return self._tmp_buffer_path

    def hold_decision(self, decision: dict[str, Any]) -> None:
        """Hold a fresh decision instead of emitting (HOLD_OUTPUT mode)."""
        with self._lock:
            self._held_decisions.append(decision)
        logger.info(
            "horizon.degraded.action",
            mode=self._state.value,
            action="held_decision",
            held_total=len(self._held_decisions),
        )

    def drain_buffered_evidence(self) -> list[dict[str, Any]]:
        """Return and clear the buffered-evidence list (called on recover)."""
        with self._lock:
            out = list(self._buffered_evidence)
            self._buffered_evidence.clear()
        return out

    def drain_held_decisions(self) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._held_decisions)
            self._held_decisions.clear()
        return out


# Module-level convenience: a singleton, optional. Tests construct their
# own controller instances; production wires one into `lifecycle`.
_default: DegradationController | None = None


def get_default_controller() -> DegradationController:
    global _default
    if _default is None:
        _default = DegradationController()
    return _default


def enter_degraded_mode(mode: FailureMode, reason: str = "") -> DegradedState:
    """Convenience wrapper for the default controller."""
    return get_default_controller().enter(mode, reason=reason)


def recover_from_degraded(reason: str = "dep_healthy") -> DegradedState:
    return get_default_controller().recover(reason=reason)


__all__ = [
    "DegradedState",
    "FailureMode",
    "DegradationController",
    "DegradationSnapshot",
    "enter_degraded_mode",
    "recover_from_degraded",
    "get_default_controller",
]
