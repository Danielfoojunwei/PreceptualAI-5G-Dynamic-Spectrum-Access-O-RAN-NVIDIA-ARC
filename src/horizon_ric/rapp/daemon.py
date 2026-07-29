"""Production daemon for the Horizon-RIC rApp (packaged entrypoint).

Differs from the bare ``horizon-rapp`` console script in that it wires the
*full* runtime: a telemetry source, the decision pipeline (risk-band planner →
Decision Safety Shield → pre-emit guards → A1 emitter → enforcement poll →
evidence store), with checkpoint-based crash recovery, sd_notify readiness /
watchdog integration, and a Prometheus metrics server. See
``horizon_ric.rapp.pipeline`` for the per-event contract.

``scripts/run_horizon_rapp.py`` is a thin re-export shim over this module so
existing paths, docs and CI keep working; the packaged wheel ships this module
so ``horizon-rapp --source-config ...`` works from a plain ``pip install``.

Usage::

    python -m horizon_ric.rapp.daemon --source-config local-dev.yaml
    python -m horizon_ric.rapp.daemon --once --report-json report.json

The ``--once`` mode boots, drains a single batch from the source, emits the
corresponding A1 policies, persists DecisionRecords, and exits cleanly. We
use it as the post-install smoke-test in Helm and as a CI gate; with
``HORIZON_ONCE_REQUIRE_ACCEPTED=1`` the exit code is 1 unless at least one
policy was accepted by the Near-RT RIC.

Source config (YAML)::

    source:
      type: file              # file | kafka | synthetic
      path: data/replay.jsonl # file only
      # synthetic only:
      # count: 0              # events to emit; 0 = unbounded
      # interval_seconds: 10
      # risk_profile: cycle   # optional: cycle sla_risk_30s 0.05 → 0.9
      # kafka only: bootstrap_servers, topic, group_id, [SASL knobs]
    sink:
      audit_path: /var/lib/horizon/audit.jsonl
    health:
      port: 8081
    metrics:
      port: 8082
    state:
      checkpoint_path: /var/lib/horizon/state.json
      interval_seconds: 30
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import structlog
import yaml
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = structlog.get_logger("horizon.daemon")


class DaemonStartupError(RuntimeError):
    """Fatal misconfiguration detected at daemon startup (fail loudly)."""


# ---------------------------------------------------------------------------
# Daemon-scoped Prometheus metrics (separate registry-friendly names from the
# rApp lifecycle metrics in horizon_ric.rapp.health).
# ---------------------------------------------------------------------------
DECISION_LATENCY = Histogram(
    "horizon_decision_latency_seconds",
    "End-to-end decision latency from telemetry ingest to A1 emit acceptance "
    "(excludes the enforcement-status poll).",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)
EVENTS_INGESTED = Counter(
    "horizon_events_ingested_total",
    "Telemetry events drained from the source connector.",
)
QUEUE_DEPTH = Gauge(
    "horizon_telemetry_queue_depth",
    "Current depth of the planner input queue (backpressure indicator).",
)
AUDIT_CHAIN_LENGTH = Gauge(
    "horizon_audit_chain_length",
    "Number of DecisionRecords persisted in the evidence chain.",
)
MALFORMED_EVENTS = Counter(
    "horizon_daemon_malformed_events_total",
    "Source lines/messages that failed schema validation and were skipped.",
)


@dataclass
class DaemonConfig:
    source_type: str = "file"
    source_path: str = "data/replay.jsonl"
    synthetic_count: int = 0
    """`source.type: synthetic` only — events to emit; 0 = unbounded."""
    synthetic_interval_s: float = 10.0
    synthetic_risk_profile: str | None = None
    """Optional: ``cycle`` cycles sla_risk_30s 0.05 → 0.9 across events."""
    audit_path: str = "/var/lib/horizon/audit.jsonl"
    health_port: int = 8081
    metrics_port: int = 8082
    state_path: str = "/var/lib/horizon/state.json"
    state_interval_s: float = 30.0
    once: bool = False
    once_max_events: int = 10
    report_json: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DaemonConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"source config not found: {p}")
        with p.open() as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        src = raw.get("source", {}) or {}
        cfg.source_type = src.get("type", cfg.source_type)
        cfg.source_path = src.get("path", cfg.source_path)
        cfg.synthetic_count = int(src.get("count", cfg.synthetic_count))
        cfg.synthetic_interval_s = float(
            src.get("interval_seconds", cfg.synthetic_interval_s)
        )
        cfg.synthetic_risk_profile = src.get(
            "risk_profile", cfg.synthetic_risk_profile
        )
        sink = raw.get("sink", {})
        cfg.audit_path = sink.get("audit_path", cfg.audit_path)
        cfg.health_port = int(raw.get("health", {}).get("port", cfg.health_port))
        cfg.metrics_port = int(raw.get("metrics", {}).get("port", cfg.metrics_port))
        st = raw.get("state", {})
        cfg.state_path = st.get("checkpoint_path", cfg.state_path)
        cfg.state_interval_s = float(st.get("interval_seconds", cfg.state_interval_s))
        cfg.extra = raw
        return cfg


# ---------------------------------------------------------------------------
# Crash-recovery checkpoint
# ---------------------------------------------------------------------------
@dataclass
class StateCheckpoint:
    """Resumable state snapshot persisted to disk every state_interval_s."""

    last_event_id: str | None = None
    events_processed: int = 0
    decisions_emitted: int = 0
    last_checkpoint_utc: str = ""

    @classmethod
    def load(cls, path: Path) -> "StateCheckpoint":
        if not path.exists():
            return cls()
        try:
            with path.open() as f:
                obj = json.load(f)
            return cls(
                last_event_id=obj.get("last_event_id"),
                events_processed=int(obj.get("events_processed", 0)),
                decisions_emitted=int(obj.get("decisions_emitted", 0)),
                last_checkpoint_utc=obj.get("last_checkpoint_utc", ""),
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("state.load.corrupt", error=str(exc), path=str(path))
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_utc = datetime.now(timezone.utc).isoformat()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(self.__dict__, f, sort_keys=True)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Health / metrics server (own thread so it survives planner stalls)
# ---------------------------------------------------------------------------
def _build_admin_app(state_ref: dict[str, Any]):
    """Build a minimal ASGI app exposing /healthz, /readyz, /metrics."""
    from fastapi import FastAPI, Response

    app = FastAPI(title="Horizon-RIC Daemon", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(response: Response):
        if state_ref.get("ready", False):
            return {"status": "ready"}
        response.status_code = 503
        return {"status": "not_ready"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


# ---------------------------------------------------------------------------
# Source / sink wiring
# ---------------------------------------------------------------------------
# sla_risk_30s values cycled by `risk_profile: cycle` — covers all three
# planner risk bands (QoS < 0.3 ≤ steering < 0.6 ≤ admission).
_RISK_CYCLE = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9)


async def _synthetic_events(
    count: int,
    interval_s: float,
    risk_profile: str | None,
) -> AsyncIterator[Any]:
    """Explicit synthetic telemetry generator (``source.type: synthetic``).

    ``count == 0`` streams unbounded; ``risk_profile: cycle`` walks
    ``sla_risk_30s`` through ``_RISK_CYCLE`` so demo installs exercise every
    planner band.
    """
    from horizon_ric.io.schemas import TelemetryEvent

    i = 0
    while count == 0 or i < count:
        if risk_profile == "cycle":
            risk = _RISK_CYCLE[i % len(_RISK_CYCLE)]
        else:
            risk = 0.05 + 0.01 * (i % 25)
        yield TelemetryEvent(
            event_id=f"synthetic-{i:06d}",
            modality="synthetic",
            source_id="synthetic",
            ts_utc=datetime.now(timezone.utc),
            sequence=i,
            payload={"sla_risk_30s": risk},
        )
        i += 1
        if interval_s > 0 and (count == 0 or i < count):
            await asyncio.sleep(interval_s)


async def _tolerant_file_events(
    path: Path,
    resume_after: str | None,
    counters: dict[str, int],
) -> AsyncIterator[Any]:
    """Line-tolerant JSONL telemetry reader with checkpoint resume.

    Unlike :class:`~horizon_ric.io.connectors.file_connector.FileSource`
    (which stays strict for other callers), a malformed line here logs
    ``daemon.event.malformed`` + increments a counter and the stream
    continues — one bad row must not kill the daemon.

    When ``resume_after`` (the checkpoint's ``last_event_id``) is set, events
    up to AND INCLUDING the matching event_id are skipped so a restart does
    not re-emit duplicate policies. If the id never matches (source file
    changed), everything is processed and a warning is logged.
    """
    from horizon_ric.io.schemas import TelemetryEvent

    loop = asyncio.get_running_loop()

    def _read_lines() -> list[str]:
        with path.open("r", encoding="utf-8") as f:
            return [line for line in f if line.strip()]

    lines = await loop.run_in_executor(None, _read_lines)
    events: list[Any] = []
    for lineno, line in enumerate(lines, start=1):
        try:
            obj = json.loads(line)
            # Match the envelope FileSink writes: {"schema": tag, "data": ...};
            # tolerate raw payloads too (older files without an envelope).
            if isinstance(obj, dict) and "schema" in obj and "data" in obj:
                if obj["schema"] != "telemetry":
                    # Non-telemetry rows (audit, feature frames) are simply
                    # not telemetry — skip silently like FileSource does.
                    continue
                payload = obj["data"]
            else:
                payload = obj
            events.append(TelemetryEvent.model_validate(payload))
        except Exception as exc:
            counters["malformed"] = counters.get("malformed", 0) + 1
            MALFORMED_EVENTS.inc()
            logger.warning(
                "daemon.event.malformed",
                path=str(path),
                line_number=lineno,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    start = 0
    if resume_after:
        for idx, ev in enumerate(events):
            if ev.event_id == resume_after:
                start = idx + 1
                break
        if start:
            logger.info(
                "daemon.resume.skipped_replayed_events",
                skipped=start,
                last_event_id=resume_after,
            )
        else:
            logger.warning(
                "daemon.resume.checkpoint_event_not_found",
                last_event_id=resume_after,
                note="processing ALL events — source file likely changed "
                "since the checkpoint was written",
            )

    for ev in events[start:]:
        yield ev


async def _open_source(
    cfg: DaemonConfig,
    checkpoint: "StateCheckpoint",
    counters: dict[str, int],
) -> tuple[AsyncIterator[Any], Any]:
    """Open the configured telemetry source.

    Returns ``(async_iterator, source_obj_or_None)``; the source object (when
    not None) must be closed by the caller at shutdown. Misconfiguration or a
    missing optional dependency raises :class:`DaemonStartupError` — the
    daemon fails loudly at startup, never silently.
    """
    if cfg.source_type == "file":
        path = Path(cfg.source_path)
        if not path.exists():
            if cfg.once:
                # --once smoke mode only: a synthetic batch keeps CI/Helm
                # test hooks working on a fresh checkout — but LOUDLY.
                logger.warning(
                    "daemon.source.file_missing_synthetic_fallback",
                    path=str(path),
                    events=cfg.once_max_events,
                    note="--once mode: telemetry file missing, emitting a "
                    "SYNTHETIC batch. Never rely on this in production — "
                    "use an explicit `source.type: synthetic` instead.",
                )
                return (
                    _synthetic_events(
                        count=cfg.once_max_events, interval_s=0.0, risk_profile=None
                    ),
                    None,
                )
            raise DaemonStartupError(
                f"telemetry source file not found: {path} — refusing to start "
                "in daemon mode (a missing file must not silently fabricate "
                "telemetry). For demo installs configure an explicit "
                "`source: {type: synthetic, count: 0, interval_seconds: 10}`."
            )
        return (
            _tolerant_file_events(
                path, resume_after=checkpoint.last_event_id, counters=counters
            ),
            None,
        )
    if cfg.source_type == "synthetic":
        logger.info(
            "daemon.source.synthetic",
            count=cfg.synthetic_count,
            interval_seconds=cfg.synthetic_interval_s,
            risk_profile=cfg.synthetic_risk_profile,
        )
        return (
            _synthetic_events(
                count=cfg.synthetic_count,
                interval_s=cfg.synthetic_interval_s,
                risk_profile=cfg.synthetic_risk_profile,
            ),
            None,
        )
    if cfg.source_type == "kafka":
        from horizon_ric.io.connector import ConnectorError
        from horizon_ric.io.connectors.kafka_connector import KafkaSource

        try:
            src = KafkaSource.from_dict(cfg.extra.get("source", {}))
        except ConnectorError as exc:
            raise DaemonStartupError(f"Kafka source misconfigured: {exc}") from exc
        try:
            await src.connect()
        except ConnectorError as exc:
            # Covers KafkaDependencyError (aiokafka not installed) and
            # broker/connectivity failures — all fatal at startup.
            raise DaemonStartupError(f"Kafka source failed to connect: {exc}") from exc
        return src.stream(), src
    raise DaemonStartupError(f"unknown source type: {cfg.source_type!r}")


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------
async def _daemon_main(cfg: DaemonConfig) -> int:
    from horizon_ric.evidence.store import JsonlEvidenceStore
    from horizon_ric.rapp.lifecycle import (
        HorizonRAppLifecycle,
        RAppState,
        _config_from_env,
    )
    from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig
    from horizon_ric.runtime.watchdog import notify_ready, watchdog_loop

    state_ref: dict[str, Any] = {"ready": False}
    chk_path = Path(cfg.state_path)
    checkpoint = StateCheckpoint.load(chk_path)
    logger.info(
        "daemon.start",
        events_processed=checkpoint.events_processed,
        decisions_emitted=checkpoint.decisions_emitted,
        once=cfg.once,
        source_type=cfg.source_type,
    )

    audit_path = Path(cfg.audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(audit_path)

    # Honour HORIZON_NEAR_RT_RIC_URL / HORIZON_A1_DIALECT / auth env — the
    # same knobs the Helm chart and the bare console script use.
    r1_cfg, a1_cfg, health_host, _ = _config_from_env()
    lifecycle = HorizonRAppLifecycle(
        r1_config=r1_cfg,
        a1_config=a1_cfg,
        health_host=health_host,
        health_port=cfg.health_port,
    )

    # Boot the lifecycle. R1/A1 may not be reachable in --once smoke mode;
    # we tolerate that and continue in DEGRADED.
    boot_degraded = False
    try:
        await lifecycle.boot()
    except Exception as exc:
        logger.warning("daemon.lifecycle.boot_degraded", error=str(exc))
        lifecycle.mark_degraded(reason=str(exc))
        boot_degraded = True

    # Evidence store attaches to the A1 adapter so accepted emits persist
    # their DecisionRecord atomically with the policy delivery.
    lifecycle.a1.attach_evidence_store(store)

    if boot_degraded:
        # boot() registers R1 before A1, so a missing SMO aborts the A1 policy
        # type registration too. A1 lives on the Near-RT RIC, not the SMO —
        # retry it directly so policies still flow without an R1 registration.
        try:
            accepted_types = await lifecycle.a1.register_policy_types()
            logger.info("daemon.a1.policy_types_registered", accepted=accepted_types)
        except Exception as exc:
            logger.warning("daemon.a1.policy_types_failed", error=str(exc))

    counters: dict[str, int] = {"malformed": 0}

    # Pipeline construction can fail loudly (e.g. HORIZON_CERT_SIGNING_KEY_PATH
    # set but unreadable) — that's a startup error; shut the lifecycle down.
    try:
        pipeline = DecisionPipeline(lifecycle.a1, store, PipelineConfig.from_env())
        # Open the source BEFORE serving traffic so misconfiguration
        # (missing telemetry file in daemon mode, bad Kafka config,
        # missing aiokafka) fails fast with a clear error.
        stream, source_obj = await _open_source(cfg, checkpoint, counters)
    except Exception:
        with contextlib.suppress(Exception):
            await lifecycle.shutdown()
        raise

    state_ref["ready"] = lifecycle.state in (RAppState.RUNNING, RAppState.DEGRADED)

    # Start the admin server in a background task.
    import uvicorn

    admin_cfg = uvicorn.Config(
        _build_admin_app(state_ref),
        host="0.0.0.0",
        port=cfg.metrics_port,
        log_level="warning",
        lifespan="off",
    )
    admin_server = uvicorn.Server(admin_cfg)
    admin_task = asyncio.create_task(admin_server.serve())

    stop = asyncio.Event()

    def _on_signal():
        logger.info("daemon.signal.received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:  # pragma: no cover — Windows
            pass

    # systemd Type=notify: READY=1 once boot completed, then periodic
    # WATCHDOG=1 pings (watchdog_loop sends STOPPING=1 on cancellation).
    notify_ready()
    watchdog_task = asyncio.create_task(watchdog_loop())

    last_checkpoint_at = time.monotonic()
    n_events = 0
    n_accepted = 0
    n_blocked = 0
    n_failed = 0
    n_errors = 0
    n_dry_run = 0
    n_enforced = 0
    policy_ids: list[str] = []
    chain_index = -1
    # O(n) not O(n²): read the (file-validating) len(store) ONCE at startup,
    # then track appended records locally per event.
    audit_len = len(store)
    AUDIT_CHAIN_LENGTH.set(audit_len)

    try:
        stream_iter = stream.__aiter__()
        stop_wait_task: asyncio.Task | None = None
        try:
            while not stop.is_set():
                # Race the next-event read against the stop signal so a
                # SIGTERM during a slow/blocked source read still shuts
                # down promptly instead of hanging on __anext__.
                next_task: asyncio.Task = asyncio.ensure_future(
                    stream_iter.__anext__()
                )
                if stop_wait_task is None or stop_wait_task.done():
                    stop_wait_task = asyncio.ensure_future(stop.wait())
                done, _ = await asyncio.wait(
                    {next_task, stop_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if next_task not in done:
                    next_task.cancel()
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await next_task
                    break
                try:
                    event = next_task.result()
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    # A malformed message surfaced through the source
                    # iterator (e.g. Kafka parse failure) — log, count,
                    # keep the daemon alive.
                    counters["malformed"] = counters.get("malformed", 0) + 1
                    MALFORMED_EVENTS.inc()
                    logger.warning(
                        "daemon.event.malformed",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    continue

                EVENTS_INGESTED.inc()
                n_events += 1
                QUEUE_DEPTH.set(0)  # this loop drains as it ingests

                try:
                    result = await pipeline.process_event(event)
                except Exception as exc:
                    logger.error(
                        "daemon.decision.error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        event_id=event.event_id,
                    )
                    n_errors += 1
                    continue

                # Decision latency excludes the enforcement-poll wait
                # (PipelineResult.poll_duration_ms carries it separately).
                DECISION_LATENCY.observe(result.latency_ms / 1000.0)
                audit_len += 1
                AUDIT_CHAIN_LENGTH.set(audit_len)
                if result.blocked:
                    n_blocked += 1
                elif result.dry_run:
                    n_dry_run += 1
                elif result.emit_failed:
                    n_failed += 1
                else:
                    n_accepted += 1
                    checkpoint.decisions_emitted += 1
                    if result.policy_id:
                        policy_ids.append(result.policy_id)
                    if result.enforced:
                        n_enforced += 1
                logger.info(
                    "daemon.decision",
                    event_id=event.event_id,
                    decision_id=result.decision_id,
                    policy_type=result.policy_type,
                    policy_id=result.policy_id,
                    accepted=result.accepted,
                    blocked=result.blocked,
                    block_reasons=result.block_reasons or None,
                    emit_failed=result.emit_failed,
                    dry_run=result.dry_run or None,
                    evidence_persist_failed=result.evidence_persist_failed or None,
                    enforcement=result.enforcement_status,
                    latency_ms=round(result.latency_ms, 2),
                    poll_duration_ms=round(result.poll_duration_ms, 2),
                )

                checkpoint.last_event_id = event.event_id
                checkpoint.events_processed += 1

                now = time.monotonic()
                if now - last_checkpoint_at > cfg.state_interval_s:
                    checkpoint.save(chk_path)
                    last_checkpoint_at = now

                if cfg.once and n_events >= cfg.once_max_events:
                    break
        finally:
            if stop_wait_task is not None:
                stop_wait_task.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await stop_wait_task
    finally:
        # Every step below is individually guarded so lifecycle.shutdown()
        # ALWAYS runs — an admin-server (or report) failure must not leak
        # the R1/A1 clients or skip deregistration.
        try:
            checkpoint.save(chk_path)
        except Exception as exc:
            logger.error("daemon.checkpoint.save_failed", error=str(exc))
        try:
            chain_index = store.verify()
        except Exception as exc:  # pragma: no cover
            logger.warning("daemon.audit.verify_failed", error=str(exc))
        try:
            audit_len = len(store)
        except Exception as exc:  # pragma: no cover
            logger.warning("daemon.audit.len_failed", error=str(exc))
        AUDIT_CHAIN_LENGTH.set(audit_len)
        logger.info(
            "daemon.shutdown",
            events=n_events,
            accepted=n_accepted,
            blocked=n_blocked,
            failed=n_failed,
            errors=n_errors,
            dry_run=n_dry_run,
            malformed=counters.get("malformed", 0),
            enforced=n_enforced,
            audit_chain_length=audit_len,
            audit_verify_first_broken_index=chain_index,
        )
        if cfg.report_json:
            report = {
                "events": n_events,
                "accepted": n_accepted,
                "blocked": n_blocked,
                "emit_failed": n_failed,
                "errors": n_errors,
                "dry_run": n_dry_run,
                "malformed": counters.get("malformed", 0),
                "enforced": n_enforced,
                "policy_ids": policy_ids,
                "audit_chain_length": audit_len,
                "audit_verify_first_broken_index": chain_index,
            }
            try:
                report_path = Path(cfg.report_json)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, sort_keys=True, indent=2))
                logger.info("daemon.report.written", path=str(report_path))
            except OSError as exc:
                logger.error("daemon.report.write_failed", error=str(exc))
        watchdog_task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await watchdog_task
        if source_obj is not None:
            try:
                await source_obj.close()
            except Exception as exc:
                logger.warning("daemon.source.close_failed", error=str(exc))
        try:
            admin_server.should_exit = True
            await asyncio.wait_for(admin_task, timeout=5.0)
        except asyncio.TimeoutError:
            admin_task.cancel()
        except Exception as exc:
            logger.error("daemon.admin.shutdown_failed", error=str(exc))
        finally:
            # The one shutdown that must never be skipped.
            await lifecycle.shutdown()

    if (
        cfg.once
        and os.environ.get("HORIZON_ONCE_REQUIRE_ACCEPTED") == "1"
        and n_accepted < 1
    ):
        logger.error("daemon.once.no_accepted_policies", accepted=n_accepted)
        return 1
    return 0


def run_daemon(
    source_config: str | None = None,
    once: bool = False,
    report_json: str | None = None,
) -> int:
    """Synchronous entrypoint used by ``horizon-rapp --source-config ...``."""
    if source_config:
        cfg = DaemonConfig.from_yaml(source_config)
    else:
        cfg = DaemonConfig()
    cfg.once = once or cfg.once
    cfg.report_json = report_json or cfg.report_json
    try:
        return asyncio.run(_daemon_main(cfg))
    except DaemonStartupError as exc:
        logger.error("daemon.startup.failed", error=str(exc))
        print(f"horizon-rapp daemon: startup failed: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_horizon_rapp",
        description="Horizon-RIC rApp production daemon.",
    )
    p.add_argument(
        "--source-config",
        type=str,
        default=os.environ.get("HORIZON_SOURCE_CONFIG"),
        help="Path to a YAML connector config (see scripts/local-dev.yaml).",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Drain N events and exit (smoke-test mode, default N=10).",
    )
    p.add_argument(
        "--once-max-events",
        type=int,
        default=10,
        help="Number of events to drain in --once mode.",
    )
    p.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Write a JSON run report (events/accepted/blocked/policy ids/"
             "audit chain state) to this path on shutdown.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.source_config:
        cfg = DaemonConfig.from_yaml(args.source_config)
    else:
        cfg = DaemonConfig()
    cfg.once = args.once
    cfg.once_max_events = args.once_max_events
    cfg.report_json = args.report_json
    try:
        return asyncio.run(_daemon_main(cfg))
    except DaemonStartupError as exc:
        logger.error("daemon.startup.failed", error=str(exc))
        print(f"horizon-rapp daemon: startup failed: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "AUDIT_CHAIN_LENGTH",
    "DECISION_LATENCY",
    "EVENTS_INGESTED",
    "MALFORMED_EVENTS",
    "QUEUE_DEPTH",
    "DaemonConfig",
    "DaemonStartupError",
    "StateCheckpoint",
    "main",
    "run_daemon",
]


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    sys.exit(main())
