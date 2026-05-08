"""SLA policy models + SQLAlchemy persistence.

Pydantic v2 models for the SLA management surface and a real SQLite
backend at `data/horizon_sla.db`. Schema migration is one-shot
(`create_schema(engine)`) — sufficient for a single-tenant rApp.

Metrics enumerated in `SLOTarget.metric` are 3GPP TS 28.554 §6 KPIs
plus rApp-internal indicators surfaced for SLO targeting:
  - latency_p99_ms        (TS 28.554 §6.3.1 user-plane delay)
  - throughput_mbps       (TS 28.554 §6.3.2 user-plane throughput)
  - ber                   (bit error rate; ≤ NTN floor)
  - sla_risk_30s          (rApp risk head 30s window)
  - epfd_margin_dB        (gateway PFD margin to ITU-R BO.1444 floor)
  - availability_pct      (TS 28.554 §6.4 service availability)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Default DB location
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path("data/horizon_sla.db")


def default_engine(url: str | None = None) -> Engine:
    """Create the engine pointed at `data/horizon_sla.db` by default."""
    if url is None:
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{DEFAULT_DB_PATH}"
    return create_engine(url, future=True)


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------

SLOMetric = Literal[
    "latency_p99_ms",
    "throughput_mbps",
    "ber",
    "sla_risk_30s",
    "epfd_margin_dB",
    "availability_pct",
]
SLOComparison = Literal[">=", "<=", "=="]
Severity = Literal["info", "warning", "critical"]


class SLOTarget(BaseModel):
    """One SLO inside an SLA."""

    metric: SLOMetric
    comparison: SLOComparison
    threshold: float
    window_s: int = Field(ge=0, description="Required sustain window before breach fires.")


class SLA(BaseModel):
    """An SLA: name, scope, list of SLO targets, severity routing."""

    id: str = Field(default_factory=lambda: f"sla-{uuid.uuid4().hex[:8]}")
    name: str
    description: str = ""
    scope: dict[str, str] = Field(
        default_factory=dict,
        description="Free-form key/values, e.g. {'tenant': 'maritime', 'region': 'NTN-1'}",
    )
    targets: list[SLOTarget]
    severity_levels: dict[Severity, dict[str, Any]] = Field(
        default_factory=lambda: {
            "info": {},
            "warning": {},
            "critical": {},
        }
    )

    @field_validator("targets")
    @classmethod
    def _at_least_one(cls, v: list[SLOTarget]) -> list[SLOTarget]:
        if not v:
            raise ValueError("SLA requires at least one target")
        return v


class SLABreachEvent(BaseModel):
    """A single breach instance.

    Deduped by (sla_id, source_metric); a sustained breach updates
    `lasting_s` rather than emitting a flood of events.
    """

    id: str = Field(default_factory=lambda: f"breach-{uuid.uuid4().hex[:12]}")
    sla_id: str
    ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: Severity
    observed_value: float
    target_threshold: float
    decision_id_at_breach: str | None = None
    source_metric: SLOMetric
    lasting_s: float = 0.0
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None


# ---------------------------------------------------------------------------
# SQLAlchemy schema (Core tables — explicit, no ORM)
# ---------------------------------------------------------------------------

_metadata = MetaData()

slas_table = Table(
    "slas",
    _metadata,
    Column("id", String(64), primary_key=True),
    Column("name", String(256), nullable=False),
    Column("description", Text, nullable=False, default=""),
    Column("scope_json", Text, nullable=False, default="{}"),
    Column("targets_json", Text, nullable=False),
    Column("severity_levels_json", Text, nullable=False, default="{}"),
    Column("created_at", DateTime, nullable=False),
)

breaches_table = Table(
    "breaches",
    _metadata,
    Column("id", String(64), primary_key=True),
    Column("sla_id", String(64), nullable=False, index=True),
    Column("ts_utc", DateTime, nullable=False, index=True),
    Column("severity", String(16), nullable=False),
    Column("observed_value", Float, nullable=False),
    Column("target_threshold", Float, nullable=False),
    Column("decision_id_at_breach", String(64), nullable=True),
    Column("source_metric", String(64), nullable=False),
    Column("lasting_s", Float, nullable=False, default=0.0),
    Column("acknowledged", Integer, nullable=False, default=0),
    Column("acknowledged_at", DateTime, nullable=True),
    Column("acknowledged_by", String(128), nullable=True),
)


def create_schema(engine: Engine) -> None:
    """One-time migration: create the `slas` and `breaches` tables."""
    _metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def persist_sla(engine: Engine, sla: SLA) -> None:
    """Insert or replace an SLA row."""
    create_schema(engine)
    with engine.begin() as conn:
        conn.execute(delete(slas_table).where(slas_table.c.id == sla.id))
        conn.execute(
            insert(slas_table).values(
                id=sla.id,
                name=sla.name,
                description=sla.description,
                scope_json=json.dumps(sla.scope, sort_keys=True),
                targets_json=json.dumps(
                    [t.model_dump() for t in sla.targets], sort_keys=True
                ),
                severity_levels_json=json.dumps(sla.severity_levels, sort_keys=True),
                created_at=datetime.now(timezone.utc),
            )
        )


def load_slas(engine: Engine) -> list[SLA]:
    """Load all persisted SLAs."""
    create_schema(engine)
    with engine.begin() as conn:
        rows = conn.execute(select(slas_table)).all()
    out: list[SLA] = []
    for r in rows:
        out.append(
            SLA(
                id=r.id,
                name=r.name,
                description=r.description,
                scope=json.loads(r.scope_json),
                targets=[SLOTarget(**t) for t in json.loads(r.targets_json)],
                severity_levels=json.loads(r.severity_levels_json),
            )
        )
    return out


def list_slas(engine: Engine) -> list[SLA]:
    """Alias used by the CLI."""
    return load_slas(engine)


def persist_breach(engine: Engine, breach: SLABreachEvent) -> None:
    """Insert or update a breach (id is primary key — sustained breaches update)."""
    create_schema(engine)
    with engine.begin() as conn:
        existing = conn.execute(
            select(breaches_table.c.id).where(breaches_table.c.id == breach.id)
        ).first()
        if existing:
            conn.execute(
                update(breaches_table)
                .where(breaches_table.c.id == breach.id)
                .values(
                    lasting_s=breach.lasting_s,
                    severity=breach.severity,
                    observed_value=breach.observed_value,
                    acknowledged=int(breach.acknowledged),
                    acknowledged_at=breach.acknowledged_at,
                    acknowledged_by=breach.acknowledged_by,
                )
            )
        else:
            conn.execute(
                insert(breaches_table).values(
                    id=breach.id,
                    sla_id=breach.sla_id,
                    ts_utc=breach.ts_utc,
                    severity=breach.severity,
                    observed_value=breach.observed_value,
                    target_threshold=breach.target_threshold,
                    decision_id_at_breach=breach.decision_id_at_breach,
                    source_metric=breach.source_metric,
                    lasting_s=breach.lasting_s,
                    acknowledged=int(breach.acknowledged),
                    acknowledged_at=breach.acknowledged_at,
                    acknowledged_by=breach.acknowledged_by,
                )
            )


def list_breaches(
    engine: Engine,
    *,
    sla_id: str | None = None,
    since: datetime | None = None,
) -> list[SLABreachEvent]:
    """Read breaches, optionally filtered by SLA + recency."""
    create_schema(engine)
    stmt = select(breaches_table)
    if sla_id is not None:
        stmt = stmt.where(breaches_table.c.sla_id == sla_id)
    if since is not None:
        stmt = stmt.where(breaches_table.c.ts_utc >= since)
    stmt = stmt.order_by(breaches_table.c.ts_utc.desc())
    with engine.begin() as conn:
        rows = conn.execute(stmt).all()
    return [
        SLABreachEvent(
            id=r.id,
            sla_id=r.sla_id,
            ts_utc=r.ts_utc.replace(tzinfo=timezone.utc) if r.ts_utc.tzinfo is None else r.ts_utc,
            severity=r.severity,
            observed_value=r.observed_value,
            target_threshold=r.target_threshold,
            decision_id_at_breach=r.decision_id_at_breach,
            source_metric=r.source_metric,
            lasting_s=r.lasting_s,
            acknowledged=bool(r.acknowledged),
            acknowledged_at=r.acknowledged_at,
            acknowledged_by=r.acknowledged_by,
        )
        for r in rows
    ]


def ack_breach(engine: Engine, breach_id: str, *, by: str) -> bool:
    """Mark a breach acknowledged. Returns True if a row was updated."""
    create_schema(engine)
    with engine.begin() as conn:
        result = conn.execute(
            update(breaches_table)
            .where(breaches_table.c.id == breach_id)
            .values(
                acknowledged=1,
                acknowledged_at=datetime.now(timezone.utc),
                acknowledged_by=by,
            )
        )
    return result.rowcount > 0


def delete_sla(engine: Engine, sla_id: str) -> bool:
    """Delete an SLA. Returns True if a row was removed."""
    create_schema(engine)
    with engine.begin() as conn:
        result = conn.execute(delete(slas_table).where(slas_table.c.id == sla_id))
    return result.rowcount > 0


__all__ = [
    "SLA",
    "SLOTarget",
    "SLABreachEvent",
    "SLOMetric",
    "SLOComparison",
    "Severity",
    "DEFAULT_DB_PATH",
    "default_engine",
    "create_schema",
    "persist_sla",
    "load_slas",
    "list_slas",
    "persist_breach",
    "list_breaches",
    "ack_breach",
    "delete_sla",
]
