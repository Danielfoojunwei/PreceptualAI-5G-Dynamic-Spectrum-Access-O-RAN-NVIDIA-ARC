"""Circuit breaker for SMO/Near-RT-RIC HTTP calls.

Wraps the existing httpx.AsyncClient calls in R1Adapter, A1Adapter and
O1Adapter with a real `pybreaker.CircuitBreaker`. When the breaker is
OPEN the wrapped call raises `pybreaker.CircuitBreakerError` immediately
instead of hammering a downed SMO.

The breaker uses pybreaker (Apache-2.0) — battle-tested, ~10 years old.
Pybreaker's bundled async surface (`call_async`) requires Tornado, which
we don't ship; instead we drive its state machine directly through the
documented `state.before_call` / `_handle_error` / `_handle_success`
methods so an asyncio coroutine integrates cleanly.

Every state transition emits a structlog event with a stable name so
observability can alert on it:

    horizon.cb.opened    — failure threshold crossed
    horizon.cb.half_open — reset_timeout elapsed; one trial call allowed
    horizon.cb.closed    — trial call succeeded; back to normal traffic
    horizon.cb.rejected  — call rejected because the breaker was OPEN
    horizon.cb.failure   — a tracked failure was counted

Defaults (fail_max=5, reset_timeout=30s) come from the O-RAN.WG2
expected RIC reliability profile and are documented in RELIABILITY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

import httpx
import pybreaker
import structlog

logger = structlog.get_logger(__name__)


# Re-export the canonical error type so callers don't import pybreaker.
CircuitBreakerError = pybreaker.CircuitBreakerError


T = TypeVar("T")


@dataclass(frozen=True)
class BreakerConfig:
    """Tuning parameters for the circuit breaker.

    Defaults align with the O-RAN.WG2 RIC reliability profile:
    five consecutive failures within the rolling window flip the breaker
    OPEN, and a 30-second reset window keeps the SMO unburdened while
    still recovering inside the A1 cadence (~1 min).
    """

    name: str = "horizon.cb"
    fail_max: int = 5
    reset_timeout: float = 30.0


class _BreakerLogger(pybreaker.CircuitBreakerListener):
    """Translate pybreaker state changes into structlog events.

    These event names are part of the public observability contract —
    alerts and dashboards depend on them. Do not rename without updating
    the alerting rules.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def state_change(
        self,
        cb: pybreaker.CircuitBreaker,
        old_state: pybreaker.CircuitBreakerState | None,
        new_state: pybreaker.CircuitBreakerState,
    ) -> None:
        new = new_state.name
        old = old_state.name if old_state is not None else "init"
        # X.733 alarm emission (Devil-A Finding #20 / WG10 OAM §6).
        # Importing lazily so the alarm bus is loaded only when a CB
        # actually transitions — avoids import cycles at boot.
        from horizon_ric.observability.x733_alarms import default_bus

        bus = default_bus()
        if new == "open":
            logger.warning(
                "horizon.cb.opened",
                breaker=self._name,
                from_state=old,
                fail_counter=cb.fail_counter,
            )
            bus.emit_event(
                "horizon.cb.opened",
                additional_information={
                    "breaker": self._name,
                    "from_state": old,
                    "fail_counter": cb.fail_counter,
                },
            )
        elif new == "half-open":
            logger.info(
                "horizon.cb.half_open",
                breaker=self._name,
                from_state=old,
            )
            bus.emit_event(
                "horizon.cb.half_open",
                additional_information={"breaker": self._name, "from_state": old},
            )
        elif new == "closed":
            logger.info(
                "horizon.cb.closed",
                breaker=self._name,
                from_state=old,
            )
            bus.emit_event(
                "horizon.cb.closed",
                additional_information={"breaker": self._name, "from_state": old},
            )

    def failure(
        self, cb: pybreaker.CircuitBreaker, exc: BaseException
    ) -> None:
        logger.warning(
            "horizon.cb.failure",
            breaker=self._name,
            fail_counter=cb.fail_counter,
            exc_type=type(exc).__name__,
        )

    def success(self, cb: pybreaker.CircuitBreaker) -> None:
        logger.debug("horizon.cb.success", breaker=self._name)


def _is_failure(exc: BaseException) -> bool:
    """Return True if *exc* should count as a circuit failure.

    HTTP 5xx, timeouts, connection errors → failures.
    HTTP 4xx → NOT failures (client error, server is healthy).
    """

    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPError):
        return True
    return False


class AsyncCircuitBreaker:
    """Async wrapper over pybreaker.CircuitBreaker.

    pybreaker's `call_async` is Tornado-specific. We drive the state
    machine ourselves:

      1. `state.before_call(...)` — raises `CircuitBreakerError` if OPEN
         and the reset_timeout hasn't elapsed; otherwise transitions
         OPEN -> HALF_OPEN automatically.
      2. The user coroutine is awaited outside any pybreaker frame.
      3. The result is reported back via `_handle_success` /
         `_handle_error`, which fires the standard listener callbacks
         (so state transitions still get logged).

    Only failures matching `_is_failure` count toward `fail_max`; a 4xx
    response is propagated to the caller without tripping the breaker.
    """

    def __init__(self, config: BreakerConfig | None = None) -> None:
        self.cfg = config or BreakerConfig()
        self._listener = _BreakerLogger(self.cfg.name)
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=self.cfg.fail_max,
            reset_timeout=self.cfg.reset_timeout,
            listeners=[self._listener],
            name=self.cfg.name,
            # Re-raise the original exception when threshold is crossed
            # (instead of swallowing it inside CircuitBreakerError) — that
            # way the caller still sees the underlying httpx error from
            # the *last* failing call. Subsequent calls (while OPEN) are
            # rejected with CircuitBreakerError as expected.
            throw_new_error_on_trip=False,
            # `exclude` is checked via `is_system_error`; we do our own
            # filtering with `_is_failure` for finer 4xx vs 5xx control.
        )

    @property
    def state(self) -> str:
        """Current state name: 'closed', 'open', or 'half-open'."""
        return self._breaker.current_state

    @property
    def fail_counter(self) -> int:
        return self._breaker.fail_counter

    @property
    def name(self) -> str:
        return self.cfg.name

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Invoke `fn(*args, **kwargs)` through the breaker.

        Raises:
            CircuitBreakerError: when the breaker is OPEN.
            Any exception raised by `fn` (including HTTPError) — failures
            matching `_is_failure` are first reported to the breaker.
        """
        # 1. Gatekeeping. pybreaker's `state.before_call` invokes the
        # function synchronously when transitioning OPEN -> HALF_OPEN —
        # which doesn't work for an async coroutine factory. So we
        # implement the OPEN/HALF_OPEN gate ourselves and only delegate
        # the legal-call decision to pybreaker.
        if self._breaker.current_state == "open":
            from datetime import datetime, timedelta, timezone

            opened_at = self._breaker._state_storage.opened_at
            timeout = timedelta(seconds=self._breaker.reset_timeout)
            if opened_at and datetime.now(timezone.utc) < opened_at + timeout:
                logger.warning(
                    "horizon.cb.rejected",
                    breaker=self.cfg.name,
                    state="open",
                )
                raise CircuitBreakerError(
                    "Timeout not elapsed yet, circuit breaker still open"
                )
            # Reset window has elapsed — transition to HALF_OPEN and
            # allow this call to be the trial call.
            self._breaker.half_open()
        # Notify listeners (informational).
        for listener in self._breaker.listeners:
            listener.before_call(self._breaker, fn, *args, **kwargs)

        # 2. Run the coroutine outside pybreaker's frame.
        try:
            result = await fn(*args, **kwargs)
        except BaseException as exc:
            # 3a. Report failure / non-failure exceptions.
            if _is_failure(exc):
                # Drives counter, listeners.failure, possibly opens circuit.
                # `on_failure` re-raises (that's pybreaker's contract); we
                # suppress that here because the caller's original
                # exception is the one that matters. The state has
                # already been updated by the time on_failure raises.
                try:
                    self._breaker.state._handle_error(exc, reraise=False)
                except BaseException:
                    pass
            else:
                # 4xx etc — server is fine, treat as success.
                self._breaker.state._handle_success()
            raise
        else:
            # 3b. Report success — closes HALF_OPEN, resets counter.
            self._breaker.state._handle_success()
            return result

    def reset(self) -> None:
        """Force the breaker back to CLOSED (for tests / operator override)."""
        self._breaker.close()


__all__ = [
    "AsyncCircuitBreaker",
    "BreakerConfig",
    "CircuitBreakerError",
]
