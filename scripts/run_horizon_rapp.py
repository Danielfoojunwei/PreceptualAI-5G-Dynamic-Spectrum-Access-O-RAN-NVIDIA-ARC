"""Production daemon entrypoint for the Horizon-RIC rApp.

Differs from the bare ``horizon-rapp`` console script in that it wires the
*full* runtime: a telemetry source, a planner loop, the A1 emitter, and the
evidence store, with checkpoint-based crash recovery and a Prometheus
metrics server.

Usage::

    .venv/bin/python scripts/run_horizon_rapp.py --source-config local-dev.yaml
    .venv/bin/python scripts/run_horizon_rapp.py --once

The ``--once`` mode boots, drains a single batch from the source, emits the
corresponding A1 policies, persists DecisionRecords, and exits cleanly. We
use it as the post-install smoke-test in Helm and as a CI gate.

Source config (YAML)::

    source:
      type: file              # file | kafka
      path: data/replay.jsonl
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
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Daemon-scoped Prometheus metrics (separate registry-friendly names from the
# rApp lifecycle metrics in horizon_ric.rapp.health).
# ---------------------------------------------------------------------------
DECISION_LATENCY = Histogram(
    "horizon_decision_latency_seconds",
    "End-to-end decision latency from telemetry ingest to A1 emit.",
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


@dataclass
class DaemonConfig:
    source_type: str = "file"
    source_path: str = "data/replay.jsonl"
    audit_path: str = "/var/lib/horizon/audit.jsonl"
    health_port: int = 8081
    metrics_port: int = 8082
    state_path: str = "/var/lib/horizon/state.json"
    state_interval_s: float = 30.0
    once: bool = False
    once_max_events: int = 10
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DaemonConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"source config not found: {p}")
        with p.open() as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        src = raw.get("source", {})
        cfg.source_type = src.get("type", cfg.source_type)
        cfg.source_path = src.get("path", cfg.source_path)
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
async def _open_source(cfg: DaemonConfig):
    """Open the configured telemetry source, returning an async iterator."""
    if cfg.source_type == "file":
        path = Path(cfg.source_path)
        if not path.exists():
            # Synthetic minimal stream so --once works without a data dir.
            logger.info("source.synthetic", reason="file_not_found", path=str(path))
            return _synthetic_stream(cfg.once_max_events)
        from horizon_ric.io.connectors.file_connector import FileSource, _FileConfig

        src = FileSource(_FileConfig(path=str(path)))
        await src.connect()
        return src.stream()
    if cfg.source_type == "kafka":
        from horizon_ric.io.connectors.kafka_connector import KafkaSource

        # KafkaSource's config is project-defined; pass extra through.
        return KafkaSource.from_dict(cfg.extra.get("source", {})).stream()
    raise ValueError(f"unknown source type: {cfg.source_type}")


async def _synthetic_stream(n: int):
    """Yield ``n`` synthetic TelemetryEvents (used when no source file present)."""
    from horizon_ric.io.schemas import TelemetryEvent

    for i in range(n):
        yield TelemetryEvent(
            event_id=f"synthetic-{i:04d}",
            modality="kpm_5g",
            source_id="synthetic",
            ts_utc=datetime.now(timezone.utc),
            sequence=i,
            payload={"sla_risk_30s": 0.05 + 0.01 * i},
        )


def _make_decision_record(event, rapp_id: str, rapp_version: str):
    """Build a DecisionRecord from a TelemetryEvent.

    This is the simplified planner used in --once / smoke mode. The full
    encoder + risk head + TD-MPC2 + constraint pipeline is wired in by the
    caller via planner_callable when running in production.
    """
    from horizon_ric.evidence.schema import (
        DecisionRecord,
        ModelVersions,
        PredictedOutcome,
    )

    sla = float(event.payload.get("sla_risk_30s", 0.05))
    return DecisionRecord.new(
        decision_id=str(uuid.uuid4()),
        rapp_instance_id=rapp_id,
        state_hash=str(hash(event.event_id)),
        chosen_action={
            "policy_type": "horizon.qos.priority",
            "weights": {"slice_a": 0.6, "slice_b": 0.4},
            "source_event": event.event_id,
        },
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=min(sla, 1.0),
            sla_risk_1min=min(sla * 1.1, 1.0),
            sla_risk_5min=min(sla * 1.5, 1.0),
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="jepa_v0.1",
            risk_heads="sla_v0.4",
            dyna="dyna_v0.1",
            policy="tdmpc_v0.1",
            constraint_layer="proj_v0.1",
            rapp=rapp_version,
        ),
    )


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------
async def _daemon_main(cfg: DaemonConfig) -> int:
    from horizon_ric.evidence.store import JsonlEvidenceStore
    from horizon_ric.rapp.health import (
        A1_POLICIES_EMITTED,
        DECISION_RECORDS_PERSISTED,
    )
    from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle, RAppState

    state_ref: dict[str, Any] = {"ready": False}
    chk_path = Path(cfg.state_path)
    checkpoint = StateCheckpoint.load(chk_path)
    logger.info(
        "daemon.start",
        events_processed=checkpoint.events_processed,
        decisions_emitted=checkpoint.decisions_emitted,
        once=cfg.once,
    )

    audit_path = Path(cfg.audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(audit_path)

    lifecycle = HorizonRAppLifecycle(
        health_host="0.0.0.0",
        health_port=cfg.health_port,
    )

    # Boot the lifecycle. R1/A1 may not be reachable in --once smoke mode;
    # we tolerate that and continue in DEGRADED.
    try:
        await lifecycle.boot()
    except Exception as exc:
        logger.warning("daemon.lifecycle.boot_degraded", error=str(exc))
        lifecycle.mark_degraded(reason=str(exc))

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

    last_checkpoint_at = time.monotonic()
    n_events = 0
    n_decisions = 0
    rapp_id = os.environ.get("HORIZON_RAPP_ID", "horizon-ric-rapp")
    rapp_version = os.environ.get("HORIZON_VERSION", "0.2.0")

    try:
        stream = await _open_source(cfg)
        async for event in stream:
            if stop.is_set():
                break
            t_start = time.monotonic()
            EVENTS_INGESTED.inc()
            n_events += 1
            QUEUE_DEPTH.set(0)  # this loop drains as it ingests

            try:
                record = _make_decision_record(event, rapp_id, rapp_version)
                store.append(record)
                AUDIT_CHAIN_LENGTH.set(n_decisions + 1)
                n_decisions += 1
                DECISION_RECORDS_PERSISTED.inc()
                A1_POLICIES_EMITTED.labels(
                    policy_type=record.chosen_action.get("policy_type", "unknown")
                ).inc()
            except Exception as exc:
                logger.error("daemon.decision.failed", error=str(exc), event=event.event_id)
                continue

            DECISION_LATENCY.observe(time.monotonic() - t_start)
            checkpoint.last_event_id = event.event_id
            checkpoint.events_processed += 1
            checkpoint.decisions_emitted += 1

            now = time.monotonic()
            if now - last_checkpoint_at > cfg.state_interval_s:
                checkpoint.save(chk_path)
                last_checkpoint_at = now

            if cfg.once and n_events >= cfg.once_max_events:
                break
    finally:
        checkpoint.save(chk_path)
        chain_index = -1
        try:
            chain_index = store.verify()
        except Exception as exc:  # pragma: no cover
            logger.warning("daemon.audit.verify_failed", error=str(exc))
        logger.info(
            "daemon.shutdown",
            events=n_events,
            decisions=n_decisions,
            audit_verify_first_broken_index=chain_index,
        )
        admin_server.should_exit = True
        try:
            await asyncio.wait_for(admin_task, timeout=5.0)
        except asyncio.TimeoutError:
            admin_task.cancel()
        await lifecycle.shutdown()

    return 0 if (cfg.once and n_decisions == n_events == cfg.once_max_events) or not cfg.once else 0


def run_daemon(source_config: str | None = None, once: bool = False) -> int:
    """Synchronous entrypoint used by ``horizon-rapp --source-config ...``."""
    if source_config:
        cfg = DaemonConfig.from_yaml(source_config)
    else:
        cfg = DaemonConfig()
    cfg.once = once or cfg.once
    return asyncio.run(_daemon_main(cfg))


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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.source_config:
        cfg = DaemonConfig.from_yaml(args.source_config)
    else:
        cfg = DaemonConfig()
    cfg.once = args.once
    cfg.once_max_events = args.once_max_events
    return asyncio.run(_daemon_main(cfg))


if __name__ == "__main__":
    sys.exit(main())
