"""Stable I/O schemas (Pydantic v2).

Every connector, encoder, world model, planner, and sink consumes and
produces values typed against these schemas. They are the wire contract
between modules. Adding a new modality means adding a `Modality` literal
plus a Pydantic submodel — never editing existing fields (use additive,
backward-compatible changes only).

Versioning: schemas carry a top-level `schema_version` string. Major
bumps must be coordinated; minor additions are additive.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Canonical modality identifiers. Extend this list when adding new sources;
# downstream code dispatches on it via `match` / dict lookup.
Modality = Literal[
    "kpm_5g",            # 3GPP TS 28.552 KPI streams
    "kpm_ntn",           # NTN-specific metrics from gateway / payload
    "spectrum_iq",       # raw or compressed I/Q spectrum samples
    "spectrum_psd",      # PSD/STFT frames
    "tle",               # Two-Line-Element ephemeris updates
    "ephemeris_oem",     # CCSDS OEM ephemeris files
    "weather_grib",      # ERA5 / GFS rain rate, wet refractivity
    "thermal",           # edge accelerator thermal/power telemetry
    "spectrum_lbt",      # WiFi / 5G NR-U LBT occupancy events
    "ue_qos",            # per-UE QoS history (latency, throughput, BLER)
    "isac_radar",        # Rel-19 sensing returns
    "user_event",        # operator-issued event (override, plan switch)
    "synthetic",         # generator output (maritime, etc.)
]


_SCHEMA_VERSION = "0.1.0"


class _Base(BaseModel):
    """Shared config for all I/O schemas.

    `extra='allow'` lets new producers add forward-compatible fields
    without breaking older consumers; they are preserved on round-trip.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        arbitrary_types_allowed=False,
    )


class TelemetryEvent(_Base):
    """One observation from any source. The atomic unit on the bus.

    `payload` is intentionally a `dict[str, Any]` — strict per-modality
    schemas live in dedicated submodels under `horizon_ric.io.payloads`.
    Validation of payloads happens in the consumer, keyed off `modality`.
    """

    schema_version: str = Field(default=_SCHEMA_VERSION)
    event_id: str
    modality: Modality
    source_id: str
    """Stable identifier of the producing connector instance."""
    ts_utc: datetime
    """Event timestamp; UTC, timezone-aware required."""
    monotonic_ns: int | None = None
    """Optional monotonic clock for ordering when wall-clock skews."""
    sequence: int | None = None
    """Optional per-source sequence number; gaps detect drops."""
    payload: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("ts_utc")
    @classmethod
    def _ts_must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("ts_utc must be timezone-aware (UTC preferred)")
        return v.astimezone(timezone.utc)


class FeatureFrame(_Base):
    """Encoder-ready batch built from many TelemetryEvents.

    The frame stores tokenised + aligned data ready for the world model.
    Lists are kept structured (no flattened tensors) so different encoders
    can pick what they need.
    """

    schema_version: str = Field(default=_SCHEMA_VERSION)
    frame_id: str
    window_start_utc: datetime
    window_end_utc: datetime
    modalities_present: list[Modality]
    n_events: int
    payload: dict[str, Any] = Field(default_factory=dict)
    """Encoder-specific payload (tokens, masks, timestamps)."""

    @classmethod
    def empty(cls, frame_id: str, t0: datetime, t1: datetime) -> Self:
        return cls(
            frame_id=frame_id,
            window_start_utc=t0,
            window_end_utc=t1,
            modalities_present=[],
            n_events=0,
        )


class PolicyAction(_Base):
    """Actor output destined for the A1 adapter / Near-RT RIC.

    `policy_type` matches the A1 type registered in `a1_adapter.py`.
    `payload` must validate against that type's JSON schema; the A1
    adapter performs that check before emit.
    """

    schema_version: str = Field(default=_SCHEMA_VERSION)
    action_id: str
    policy_type: str
    policy_payload: dict[str, Any]
    decision_id: str | None = None
    """Optional link to a DecisionRecord in the evidence store."""
    issued_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AuditRecord(_Base):
    """Audit/explanation record persisted to the evidence store.

    Wraps `evidence.schema.DecisionRecord` for the bus; the consumer
    re-validates as the strict pydantic class before persisting.
    """

    schema_version: str = Field(default=_SCHEMA_VERSION)
    record_id: str
    decision_record: dict[str, Any]
    """Raw DecisionRecord JSON (consumer revalidates)."""


__all__ = [
    "AuditRecord",
    "FeatureFrame",
    "Modality",
    "PolicyAction",
    "TelemetryEvent",
]
