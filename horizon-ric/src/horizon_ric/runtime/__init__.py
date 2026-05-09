"""PreceptualAI runtime reliability layer.

Provides the production-grade plumbing required for 99.999 % uptime:
circuit breakers, systemd watchdog, atomic state checkpointing,
backpressured queues, graceful-degradation FSM, and a real chaos
test harness.

Consumers should import from this package, not the submodules
directly, so that internal layout can evolve. The exports below are
the stable API.
"""

from horizon_ric.runtime.backpressure import (
    BackpressureQueue,
    DropPolicy,
    QueueFullDropped,
)
from horizon_ric.runtime.circuit_breaker import (
    AsyncCircuitBreaker,
    BreakerConfig,
    CircuitBreakerError,
)
from horizon_ric.runtime.graceful_degradation import (
    DegradationController,
    DegradationSnapshot,
    DegradedState,
    FailureMode,
    enter_degraded_mode,
    get_default_controller,
    recover_from_degraded,
)
from horizon_ric.runtime.state_recovery import (
    DEFAULT_CHECKPOINT_INTERVAL_S,
    load_state,
    save_state,
    with_periodic_checkpoint,
)
from horizon_ric.runtime.watchdog import (
    DEFAULT_WATCHDOG_INTERVAL_S,
    notify_ready,
    notify_stopping,
    notify_watchdog,
    watchdog_loop,
)


def chaos_test_main() -> int:
    """Re-export for `python -m horizon_ric.runtime.chaos_test` parity."""
    from horizon_ric.runtime.chaos_test import main as _main

    return _main()


__all__ = [
    # circuit breaker
    "AsyncCircuitBreaker",
    "BreakerConfig",
    "CircuitBreakerError",
    # watchdog
    "DEFAULT_WATCHDOG_INTERVAL_S",
    "notify_ready",
    "notify_stopping",
    "notify_watchdog",
    "watchdog_loop",
    # state recovery
    "DEFAULT_CHECKPOINT_INTERVAL_S",
    "load_state",
    "save_state",
    "with_periodic_checkpoint",
    # backpressure
    "BackpressureQueue",
    "DropPolicy",
    "QueueFullDropped",
    # graceful degradation
    "DegradationController",
    "DegradationSnapshot",
    "DegradedState",
    "FailureMode",
    "enter_degraded_mode",
    "recover_from_degraded",
    "get_default_controller",
    # chaos test
    "chaos_test_main",
]
