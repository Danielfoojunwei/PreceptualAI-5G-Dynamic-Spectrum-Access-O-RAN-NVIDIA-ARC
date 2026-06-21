"""rApp lifecycle — boot, register, serve, deregister.

Top-level orchestration of PreceptualAI rApp instance.
Implements the lifecycle states from O-RAN.WG2.NON-RT-RIC-ARCH §rApp Lifecycle.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
from horizon_ric.rapp.r1_adapter import R1Adapter, R1AdapterConfig
from horizon_ric.runtime.graceful_degradation import DegradationController
from horizon_ric.runtime.liveness import LivenessRegistry
from horizon_ric.runtime.state_recovery import (
    DEFAULT_CHECKPOINT_INTERVAL_S,
    load_state,
    with_periodic_checkpoint,
)
from horizon_ric.runtime.watchdog import (
    DEFAULT_WATCHDOG_INTERVAL_S,
    notify_ready,
    notify_watchdog,
)

logger = structlog.get_logger(__name__)


def _default_state_path() -> Path:
    return Path(os.environ.get("HORIZON_STATE_PATH", "/var/lib/horizon/state.json"))


class RAppState(str, Enum):
    INIT = "init"
    REGISTERING = "registering"
    RUNNING = "running"
    DEGRADED = "degraded"
    DEREGISTERING = "deregistering"
    STOPPED = "stopped"


class HorizonRAppLifecycle:
    """Top-level rApp lifecycle manager."""

    def __init__(
        self,
        r1_config: R1AdapterConfig | None = None,
        a1_config: A1AdapterConfig | None = None,
        health_host: str = "0.0.0.0",
        health_port: int = 8081,
        state_path: Path | None = None,
        checkpoint_interval_s: float = DEFAULT_CHECKPOINT_INTERVAL_S,
        watchdog_interval_s: float = DEFAULT_WATCHDOG_INTERVAL_S,
    ):
        self._state = RAppState.INIT
        self._r1 = R1Adapter(r1_config)
        self._a1 = A1Adapter(a1_config)
        self._stop_signal = asyncio.Event()
        self._health_host = health_host
        self._health_port = health_port
        self._health_server_task: asyncio.Task | None = None
        # Reliability layer wiring (see horizon_ric.runtime).
        self._state_path = state_path or _default_state_path()
        self._checkpoint_interval_s = checkpoint_interval_s
        self._watchdog_interval_s = watchdog_interval_s
        self._degradation = DegradationController()
        self._a1_emitted_count = 0
        self._last_decision_id: str | None = None
        self._evidence_chain_head: str | None = None
        # Liveness registry exposed to /healthz + /readyz. Producers are
        # wired in `run_forever()` (watchdog tick, planner heartbeat,
        # loop-lag probe). For test/embedded use the registry is created
        # eagerly so probes against an in-process app see a healthy
        # process from t=0.
        self._liveness = LivenessRegistry()
        # In-flight planner tasks tracked for graceful drain on shutdown.
        self._planner_tasks: set[asyncio.Task] = set()
        self._drain_signal = asyncio.Event()
        # Restore from previous crash, if a state file is present.
        self._restore_from_disk()

    @property
    def degradation(self) -> DegradationController:
        return self._degradation

    @property
    def liveness(self) -> LivenessRegistry:
        """Liveness registry consulted by /healthz and /readyz."""
        return self._liveness

    def track_planner_task(self, task: asyncio.Task) -> None:
        """Register a planner asyncio.Task so graceful shutdown can drain it.

        Callers in the planner / decision-emit path do
        ``task = asyncio.create_task(...); lifecycle.track_planner_task(task)``.
        Tasks remove themselves on completion via a done-callback.
        """
        self._planner_tasks.add(task)
        task.add_done_callback(self._planner_tasks.discard)

    @property
    def drain_signal(self) -> asyncio.Event:
        """Set on shutdown so producers stop accepting new work."""
        return self._drain_signal

    def _restore_from_disk(self) -> None:
        """Load checkpointed state from `self._state_path` if present.

        Missing or corrupt files are tolerated — the rApp boots from
        defaults and emits a `horizon.state.missing` / `.corrupt` event.
        """
        prior = load_state(self._state_path)
        if prior is None:
            return
        self._a1_emitted_count = int(prior.get("a1_emitted_count", 0))
        self._last_decision_id = prior.get("last_decision_id")
        self._evidence_chain_head = prior.get("evidence_chain_head")
        rapp_state = prior.get("rapp_state")
        logger.info(
            "lifecycle.state.restored",
            a1_emitted_count=self._a1_emitted_count,
            last_decision_id=self._last_decision_id,
            evidence_chain_head=self._evidence_chain_head,
            rapp_state=rapp_state,
        )

    def _state_snapshot(self) -> dict[str, Any]:
        """Provider for the periodic checkpointer."""
        return {
            "rapp_state": self._state.value,
            "a1_emitted_count": self._a1_emitted_count
            or self._a1.policies_emitted_count(),
            "last_decision_id": self._last_decision_id,
            "evidence_chain_head": self._evidence_chain_head,
            "degradation": self._degradation.snapshot().__dict__,
        }

    @property
    def state(self) -> RAppState:
        return self._state

    @property
    def a1(self) -> A1Adapter:
        return self._a1

    @property
    def r1(self) -> R1Adapter:
        return self._r1

    async def boot(self) -> dict[str, Any]:
        """Boot sequence: register with R1, register A1 policy types.

        On a registration failure we transition to DEGRADED rather than
        crashing — the supervisor (k8s, systemd) can decide to restart.
        """
        self._state = RAppState.REGISTERING
        logger.info("lifecycle.boot.start")
        try:
            registration = await self._r1.register()
            accepted_types = await self._a1.register_policy_types()
        except Exception as exc:
            self._state = RAppState.DEGRADED
            logger.error("lifecycle.boot.failed", error=str(exc))
            raise
        self._state = RAppState.RUNNING

        # Mark static info gauge once we have ids.
        try:
            from horizon_ric.rapp.health import RAPP_INFO

            RAPP_INFO.labels(
                rapp_id=self._r1.cfg.rapp_id,
                version=self._r1.cfg.rapp_version,
            ).set(1.0)
        except Exception:  # pragma: no cover — metrics are best-effort
            pass

        logger.info(
            "lifecycle.boot.complete",
            registration=registration,
            policy_types_registered=len(accepted_types),
        )
        return {"registration": registration, "policy_types": accepted_types}

    async def shutdown(self, drain_timeout_s: float = 25.0) -> None:
        """Graceful shutdown with planner-task drain.

        Order of operations:

          1. Set ``_drain_signal`` so producers stop submitting new work.
          2. ``asyncio.wait`` on every tracked planner task with a deadline
             of ``drain_timeout_s`` (default 25s — fits inside k8s'
             ``terminationGracePeriodSeconds: 30``).
          3. Cancel any task that didn't drain in time and await its
             cancellation.
          4. Deregister R1, close adapters.

        This closes Devil-B finding #17 (in-flight planner work was
        previously cancelled out from under its own audit-write).
        """
        self._state = RAppState.DEREGISTERING
        logger.info(
            "lifecycle.shutdown.start",
            in_flight_planner_tasks=len(self._planner_tasks),
            drain_timeout_s=drain_timeout_s,
        )
        # 1. Signal producers to stop accepting new work.
        self._drain_signal.set()

        # 2. Drain in-flight planner work with a deadline.
        in_flight = [t for t in self._planner_tasks if not t.done()]
        if in_flight:
            logger.info("lifecycle.shutdown.drain_start", count=len(in_flight))
            done, pending = await asyncio.wait(
                in_flight, timeout=drain_timeout_s
            )
            logger.info(
                "lifecycle.shutdown.drain_complete",
                drained=len(done),
                still_pending=len(pending),
            )
            # 3. Force-cancel the rest.
            for t in pending:
                t.cancel()
            for t in pending:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

        # 4. Now safe to close adapters — no in-flight emits racing the close.
        await self._r1.deregister()
        await self._r1.close()
        await self._a1.close()
        if self._health_server_task and not self._health_server_task.done():
            self._health_server_task.cancel()
        self._state = RAppState.STOPPED
        logger.info("lifecycle.shutdown.complete")

    def mark_degraded(self, reason: str) -> None:
        """Flip to DEGRADED state (e.g. on persistent R1/A1 failure)."""
        prev = self._state
        self._state = RAppState.DEGRADED
        logger.warning("lifecycle.degraded", reason=reason, previous_state=prev.value)

    async def _serve_health(self) -> None:
        """Run the FastAPI health server until cancelled."""
        import uvicorn

        from horizon_ric.rapp.health import build_health_app

        app = build_health_app(self)
        config = uvicorn.Config(
            app,
            host=self._health_host,
            port=self._health_port,
            log_level="warning",
            lifespan="off",
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except asyncio.CancelledError:
            await server.shutdown()
            raise

    async def _serve_api(self) -> None:
        """Run the operator dashboard FastAPI server (port 8083) until cancelled.

        Builds the dashboard API with this lifecycle instance attached so
        ``/api/v1/state`` returns live `RAppState` and counters, and
        ``/api/v1/circuit-breakers`` enumerates the R1/A1/O1 adapter
        breakers. Disabled when ``HORIZON_API_DISABLE`` is truthy or
        ``HORIZON_API_JWT_SECRET`` is unset (the dashboard API refuses
        to start without a signing secret).
        """
        if os.environ.get("HORIZON_API_DISABLE"):
            logger.info("lifecycle.api.disabled", reason="HORIZON_API_DISABLE")
            return
        if not os.environ.get("HORIZON_API_JWT_SECRET"):
            logger.warning(
                "lifecycle.api.disabled",
                reason="HORIZON_API_JWT_SECRET unset; refusing to start API",
            )
            return

        import uvicorn

        from horizon_ric.rapp.dashboard_api import build_dashboard_api

        api_host = os.environ.get("HORIZON_API_HOST", "0.0.0.0")
        api_port = int(os.environ.get("HORIZON_API_PORT", "8083"))
        app = build_dashboard_api(self)
        config = uvicorn.Config(
            app,
            host=api_host,
            port=api_port,
            log_level="warning",
            lifespan="off",
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except asyncio.CancelledError:
            await server.shutdown()
            raise

    async def run_forever(self) -> None:
        """Run until SIGTERM/SIGINT.

        Wires up the production reliability layer:

          * sd_notify watchdog (WATCHDOG=1 every 15 s by default)
          * state checkpointer (atomic snapshot every 30 s by default)
          * health server (FastAPI on :8081)
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._stop_signal.set)
        await self.boot()
        # READY=1 to systemd once the boot sequence has completed.
        notify_ready()
        # Spin up the health server, watchdog, and checkpoint loop.
        self._health_server_task = asyncio.create_task(self._serve_health())
        # Operator dashboard REST API on port 8083.
        self._api_server_task = asyncio.create_task(self._serve_api())

        # Wire the liveness registry to the watchdog: each watchdog tick
        # also marks the registry, so /healthz can detect a wedged loop
        # via stale watchdog timestamps.
        async def _watchdog_with_liveness() -> None:
            try:
                while True:
                    notify_watchdog()
                    self._liveness.mark_watchdog_tick()
                    await asyncio.sleep(self._watchdog_interval_s)
            except asyncio.CancelledError:
                from horizon_ric.runtime.watchdog import notify_stopping
                notify_stopping()
                raise

        watchdog_task = asyncio.create_task(_watchdog_with_liveness())
        loop_lag_task = asyncio.create_task(self._liveness.loop_lag_probe_task())
        try:
            async with with_periodic_checkpoint(
                self._state_snapshot,
                self._state_path,
                interval_s=self._checkpoint_interval_s,
            ):
                await self._stop_signal.wait()
        finally:
            for t, name in (
                (watchdog_task, "watchdog_task"),
                (loop_lag_task, "loop_lag_task"),
            ):
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        f"lifecycle.{name}.exit_failed",
                        error=str(exc),
                        exc_type=type(exc).__name__,
                    )
            api_task = getattr(self, "_api_server_task", None)
            if api_task is not None and not api_task.done():
                api_task.cancel()
                try:
                    await api_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "lifecycle.api_task.exit_failed",
                        error=str(exc),
                        exc_type=type(exc).__name__,
                    )
            await self.shutdown()


def _config_from_env() -> tuple[R1AdapterConfig, A1AdapterConfig, str, int]:
    """Build configs from environment variables (Docker/K8s deployment)."""
    r1 = R1AdapterConfig(
        smo_base_url=os.environ.get("HORIZON_SMO_URL", R1AdapterConfig.smo_base_url),
        rapp_id=os.environ.get("HORIZON_RAPP_ID", R1AdapterConfig.rapp_id),
    )
    a1 = A1AdapterConfig(
        near_rt_ric_base_url=os.environ.get(
            "HORIZON_NEAR_RT_RIC_URL", A1AdapterConfig.near_rt_ric_base_url
        ),
        rapp_id=r1.rapp_id,
    )
    host = os.environ.get("HORIZON_HEALTH_HOST", "0.0.0.0")
    port = int(os.environ.get("HORIZON_HEALTH_PORT", "8081"))
    return r1, a1, host, port


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="horizon-rapp",
        description=(
            "PreceptualAI rApp daemon — registers with the SMO over R1, "
            "registers A1 policy types with the Near-RT RIC, and serves "
            "/healthz, /readyz, /metrics until SIGTERM/SIGINT."
        ),
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Boot, run a single readiness check, then shut down. Useful for "
             "smoke tests, k8s pre-deploy job, or systemd ConditionalStart.",
    )
    p.add_argument(
        "--source-config",
        type=str,
        default=None,
        help="Optional path to a connector config YAML. When provided, the "
             "production daemon entrypoint (scripts.run_horizon_rapp) is used "
             "instead of the bare lifecycle loop.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"horizon-rapp {os.environ.get('HORIZON_VERSION', '0.2.0')} "
                f"(sha={os.environ.get('HORIZON_GIT_SHA', 'unknown')})",
    )
    return p


async def _run_once(lifecycle: "HorizonRAppLifecycle") -> int:
    """Boot + verify readiness + shutdown.

    Returns 0 on RUNNING, 0 on DEGRADED (smoke is about the daemon being
    able to *come up* — DEGRADED is a legitimate steady state when the SMO
    is unreachable, e.g. in CI), 1 on hard crash.
    """
    try:
        await lifecycle.boot()
    except Exception as exc:
        logger.warning("lifecycle.run_once.degraded", error=str(exc))
        lifecycle.mark_degraded(reason=str(exc))
    acceptable = (RAppState.RUNNING, RAppState.DEGRADED)
    state_ok = lifecycle.state in acceptable
    await lifecycle.shutdown()
    return 0 if state_ok else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for `horizon-rapp` console script."""
    args = _build_parser().parse_args(argv)

    # Delegate to the production daemon when a source config is supplied.
    if args.source_config:
        from scripts.run_horizon_rapp import run_daemon

        return run_daemon(source_config=args.source_config, once=args.once)

    r1_cfg, a1_cfg, host, port = _config_from_env()
    lifecycle = HorizonRAppLifecycle(
        r1_config=r1_cfg, a1_config=a1_cfg, health_host=host, health_port=port
    )
    if args.once:
        return asyncio.run(_run_once(lifecycle))
    asyncio.run(lifecycle.run_forever())
    return 0


if __name__ == "__main__":
    sys.exit(main())
