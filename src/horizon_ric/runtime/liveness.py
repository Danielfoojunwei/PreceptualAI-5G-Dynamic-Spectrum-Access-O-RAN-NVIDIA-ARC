"""Process-liveness signals exposed to the `/healthz` probe.

`/healthz` must fail (503) when any of the following is unhealthy:

  * **Watchdog tick freshness** — the sd_notify watchdog loop pings every
    ``DEFAULT_WATCHDOG_INTERVAL_S`` (15s). If the latest tick is older
    than ``watchdog_max_age_s`` (default 30s = 2 × interval) the asyncio
    loop is wedged.
  * **Asyncio loop lag** — the gap between two successive ``loop.time()``
    samples taken ``probe_period_s`` (1s) apart should be ≈ probe_period.
    Lag > ``loop_lag_max_s`` (default 0.1s) means a coroutine is hogging
    the loop.
  * **Planner-task heartbeat** — the planner emits a heartbeat each
    decision; if the gap exceeds ``planner_max_silence_s`` (default
    ``decision_sla_s × max_missed_intervals``) the planner is stalled.

`/readyz` adds three further checks:

  * Connector state RUNNING (the rApp lifecycle has finished bootstrap).
  * Evidence store writable (the underlying file/DB accepts a probe write).
  * R1 registered with the SMO.

This module deliberately keeps **no global state**. The lifecycle wires a
single `LivenessRegistry` instance to the FastAPI handler. Tests
construct their own registry, drive the signals manually, and assert
the resulting status.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import structlog

logger = structlog.get_logger(__name__)

# 2 × DEFAULT_WATCHDOG_INTERVAL_S (= 30 s) — same window systemd uses
# before WatchdogSec=30 fires.
DEFAULT_WATCHDOG_MAX_AGE_S: float = 30.0
DEFAULT_LOOP_LAG_MAX_S: float = 0.100
DEFAULT_LOOP_PROBE_PERIOD_S: float = 1.0
# Planner emits decisions on a per-source SLA. Default tolerated silence
# is the SLA period × max_missed.
DEFAULT_PLANNER_SLA_S: float = 5.0
DEFAULT_PLANNER_MAX_MISSED: int = 3


@dataclass
class LivenessReport:
    """Structured result returned by `LivenessRegistry.evaluate()`."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    details: dict[str, float | str | bool] = field(default_factory=dict)

    def to_response_body(self) -> dict[str, object]:
        return {
            "status": "ok" if self.ok else "unhealthy",
            "checks": self.details,
            "failed": self.failures,
        }


@dataclass
class ReadinessReport:
    """Structured result returned by `LivenessRegistry.evaluate_readiness()`."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    def to_response_body(self) -> dict[str, object]:
        return {
            "status": "ready" if self.ok else "not_ready",
            "checks": self.details,
            "failed": self.failures,
        }


class LivenessRegistry:
    """Tracks the runtime signals consulted by `/healthz` and `/readyz`.

    The registry is `monotonic`-time-based: producers call
    `mark_watchdog_tick()`, `mark_planner_heartbeat()`, and a background
    coroutine `loop_lag_probe_task()` measures asyncio-loop lag. The
    consumer (`/healthz` route) calls `evaluate()` which compares the
    latest timestamps to the configured thresholds.

    Tests inject `time_source` (defaults to `time.monotonic`) so they
    can fast-forward the clock without `asyncio.sleep`.
    """

    def __init__(
        self,
        *,
        watchdog_max_age_s: float = DEFAULT_WATCHDOG_MAX_AGE_S,
        loop_lag_max_s: float = DEFAULT_LOOP_LAG_MAX_S,
        loop_probe_period_s: float = DEFAULT_LOOP_PROBE_PERIOD_S,
        planner_sla_s: float = DEFAULT_PLANNER_SLA_S,
        planner_max_missed: int = DEFAULT_PLANNER_MAX_MISSED,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._t = time_source
        self._watchdog_max_age_s = watchdog_max_age_s
        self._loop_lag_max_s = loop_lag_max_s
        self._loop_probe_period_s = loop_probe_period_s
        self._planner_max_silence_s = planner_sla_s * planner_max_missed
        # Initialise to "now" so a freshly-booted process is healthy until
        # its first probe interval passes; otherwise the very first
        # /healthz scrape after boot would 503.
        now = self._t()
        self._last_watchdog_tick: float = now
        self._last_planner_heartbeat: float = now
        self._loop_lag_s: float = 0.0
        self._planner_seen_heartbeat: bool = False
        # Optional readiness probes.
        self._connector_state_provider: Optional[Callable[[], str]] = None
        self._evidence_writable_probe: Optional[Callable[[], bool]] = None
        self._r1_registered_provider: Optional[Callable[[], bool]] = None

    # ---- producer-side updates ----

    def mark_watchdog_tick(self) -> None:
        self._last_watchdog_tick = self._t()

    def mark_planner_heartbeat(self) -> None:
        self._last_planner_heartbeat = self._t()
        self._planner_seen_heartbeat = True

    def record_loop_lag(self, lag_s: float) -> None:
        self._loop_lag_s = max(0.0, lag_s)

    # ---- readiness providers ----

    def set_connector_state_provider(self, fn: Callable[[], str]) -> None:
        self._connector_state_provider = fn

    def set_evidence_writable_probe(self, fn: Callable[[], bool]) -> None:
        self._evidence_writable_probe = fn

    def set_r1_registered_provider(self, fn: Callable[[], bool]) -> None:
        self._r1_registered_provider = fn

    # ---- background probe ----

    async def loop_lag_probe_task(self) -> None:
        """Background task: continuously samples asyncio loop lag.

        Sleeps for `loop_probe_period_s` then computes the actual elapsed
        wall-clock; the difference between expected and actual is the
        loop lag. Cancellation-safe.
        """
        try:
            while True:
                t0 = self._t()
                await asyncio.sleep(self._loop_probe_period_s)
                t1 = self._t()
                actual = t1 - t0
                lag = max(0.0, actual - self._loop_probe_period_s)
                self.record_loop_lag(lag)
        except asyncio.CancelledError:
            raise

    # ---- evaluation (consumer side) ----

    def evaluate(self) -> LivenessReport:
        """Snapshot the signals and produce a 200/503 verdict for `/healthz`."""
        now = self._t()
        watchdog_age = now - self._last_watchdog_tick
        planner_age = now - self._last_planner_heartbeat
        details: dict[str, float | str | bool] = {
            "watchdog_age_s": round(watchdog_age, 3),
            "watchdog_max_age_s": self._watchdog_max_age_s,
            "loop_lag_s": round(self._loop_lag_s, 4),
            "loop_lag_max_s": self._loop_lag_max_s,
            "planner_age_s": round(planner_age, 3),
            "planner_max_silence_s": self._planner_max_silence_s,
            "planner_seen_heartbeat": self._planner_seen_heartbeat,
        }
        failures: list[str] = []

        if watchdog_age > self._watchdog_max_age_s:
            failures.append("watchdog_stale")
        if self._loop_lag_s > self._loop_lag_max_s:
            failures.append("loop_lag_high")
        # Planner: if we've never seen a heartbeat, do not 503 immediately
        # (the lifecycle is still booting). Once we've seen at least one,
        # the silence threshold applies.
        if self._planner_seen_heartbeat and planner_age > self._planner_max_silence_s:
            failures.append("planner_stalled")

        return LivenessReport(ok=not failures, failures=failures, details=details)

    def evaluate_readiness(self) -> ReadinessReport:
        """Liveness + connector + evidence-writable + R1-registered."""
        liveness = self.evaluate()
        details: dict[str, object] = dict(liveness.details)
        failures: list[str] = list(liveness.failures)

        if self._connector_state_provider is not None:
            try:
                conn_state = self._connector_state_provider()
            except Exception as exc:  # noqa: BLE001
                conn_state = f"error:{type(exc).__name__}"
            details["connector_state"] = conn_state
            if conn_state != "RUNNING":
                failures.append("connector_not_running")
        else:
            details["connector_state"] = "unknown"

        if self._evidence_writable_probe is not None:
            try:
                writable = bool(self._evidence_writable_probe())
            except Exception as exc:  # noqa: BLE001
                writable = False
                details["evidence_probe_error"] = type(exc).__name__
            details["evidence_writable"] = writable
            if not writable:
                failures.append("evidence_not_writable")
        else:
            details["evidence_writable"] = "unknown"

        if self._r1_registered_provider is not None:
            try:
                registered = bool(self._r1_registered_provider())
            except Exception:  # noqa: BLE001
                registered = False
            details["r1_registered"] = registered
            if not registered:
                failures.append("r1_not_registered")
        else:
            details["r1_registered"] = "unknown"

        return ReadinessReport(
            ok=not failures, failures=failures, details=details
        )


__all__ = [
    "LivenessRegistry",
    "LivenessReport",
    "ReadinessReport",
    "DEFAULT_WATCHDOG_MAX_AGE_S",
    "DEFAULT_LOOP_LAG_MAX_S",
    "DEFAULT_LOOP_PROBE_PERIOD_S",
    "DEFAULT_PLANNER_SLA_S",
    "DEFAULT_PLANNER_MAX_MISSED",
]
