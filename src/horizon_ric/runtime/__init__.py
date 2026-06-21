"""Runtime reliability layer.

Production plumbing the rApp daemon depends on: circuit breakers, systemd
watchdog, atomic state checkpointing, graceful-degradation FSM, plus the
content-addressed artefact vault, atomic A→B promotion, loop-state machine,
and shadow executor used by the model-lifecycle / trust path.

Consumers should import from this package, not the submodules directly.
"""

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

__all__ = [
    "AsyncCircuitBreaker",
    "BreakerConfig",
    "CircuitBreakerError",
    "DEFAULT_WATCHDOG_INTERVAL_S",
    "notify_ready",
    "notify_stopping",
    "notify_watchdog",
    "watchdog_loop",
    "DEFAULT_CHECKPOINT_INTERVAL_S",
    "load_state",
    "save_state",
    "with_periodic_checkpoint",
    "DegradationController",
    "DegradationSnapshot",
    "DegradedState",
    "FailureMode",
    "enter_degraded_mode",
    "recover_from_degraded",
    "get_default_controller",
]
