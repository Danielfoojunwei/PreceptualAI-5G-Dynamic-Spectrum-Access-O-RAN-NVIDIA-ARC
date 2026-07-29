"""E2SM-KPM → Horizon TelemetryEvent bridge.

Turns real E2SM-KPM measurement reports — the RIC Indication payloads a
near-RT RIC E2 termination (e.g. FlexRIC) receives from E2 nodes — into
:class:`horizon_ric.io.schemas.TelemetryEvent` objects (modality
``"kpm_5g"``) that drive the existing rApp decision pipeline.

Two ingestion paths:

1. :meth:`KpmMeasurementBridge.from_flexric_kpm_json` — one JSON record
   as harvested from FlexRIC's KPM monitor xApp output (its stdout log
   or sqlite3 database rows re-serialised as JSON).
2. :meth:`KpmMeasurementBridge.from_e2sm_kpm_indication` — raw ASN.1
   aligned-PER bytes of an ``E2SM-KPM-IndicationMessage`` (the OCTET
   STRING carried in the E2AP RIC Indication), decoded with
   ``asn1tools`` against the O-RAN E2SM-KPM v3.00 spec committed under
   ``horizon_ric/e2/asn1/`` (see ``PROVENANCE.md`` there). Optionally
   the matching ``E2SM-KPM-IndicationHeader`` bytes supply the
   collection timestamp.

Derived risk (``sla_risk_30s``)
-------------------------------
The pipeline's risk-band planner keys off ``payload["sla_risk_30s"]``.
The bridge derives it from real 3GPP TS 28.552 KPM measurements with a
documented, deterministic mapping (``kpm_risk_v1``); each metric maps
to a normalised risk contribution in [0, 1] and the FINAL risk is the
maximum contribution (worst signal wins — conservative by design):

===============================  ================================================
metric (TS 28.552)               contribution
===============================  ================================================
``RRU.PrbTotDl``                 DL PRB utilisation in percent → ``prb / 100``
``DRB.RlcSduDelayDl``            DL RLC SDU delay in ms → ``delay / 50`` (50 ms
                                 budget)
``DRB.UEThpDl``                  DL UE throughput in kbps → ``1 - thp / 10_000``
                                 (10 Mbps SLA floor; higher throughput = lower
                                 risk)
``DRB.PdcpSduVolumeDL`` /        DL volume in kbit per granularity period →
``DRB.RlcSduTransmittedVolumeDL``  ``vol / 100_000`` (100 Mbit reference load)
===============================  ================================================

All contributions are clamped into [0, 1]. When none of the mapped
metrics is present the risk floor ``0.05`` is used and flagged in
``payload["risk"]["metric"] = None``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import asn1tools
import structlog

from horizon_ric.io.schemas import TelemetryEvent

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# ASN.1 spec (E2SM-KPM v3.00, aligned PER) — committed with provenance.
# ---------------------------------------------------------------------------
ASN1_DIR = Path(__file__).resolve().parent / "asn1"
KPM_ASN1_FILE = ASN1_DIR / "e2sm_kpm_v03.00_standard.asn1"

_spec_cache: Any | None = None


def kpm_spec() -> Any:
    """Compile (once) and return the asn1tools E2SM-KPM v3.00 spec.

    ``codec="per"`` is asn1tools' ALIGNED PER — the transfer syntax
    E2SM-KPM mandates (FlexRIC encodes with ``ATS_ALIGNED_BASIC_PER``).
    """
    global _spec_cache
    if _spec_cache is None:
        _spec_cache = asn1tools.compile_files([str(KPM_ASN1_FILE)], codec="per")
        logger.info(
            "kpm_bridge.asn1.compiled",
            file=str(KPM_ASN1_FILE),
            modality="kpm_5g",
        )
    return _spec_cache


class KpmBridgeError(ValueError):
    """Raised when a KPM record/PDU cannot be converted to telemetry."""


# ---------------------------------------------------------------------------
# Risk mapping (kpm_risk_v1) — see module docstring for the table.
# ---------------------------------------------------------------------------
RISK_MAPPING_ID = "kpm_risk_v1"

# 50 ms DL RLC SDU delay budget: delays at/above this map to risk 1.0.
DELAY_BUDGET_MS = 50.0
# 10 Mbps SLA throughput floor: 0 kbps → risk 1.0, >= 10_000 kbps → 0.0.
THROUGHPUT_FLOOR_KBPS = 10_000.0
# 100 Mbit reference DL volume per granularity period.
VOLUME_REFERENCE_KBIT = 100_000.0
# Risk assigned when no mapped metric is present in the record.
RISK_FLOOR = 0.05


def _clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def _risk_contributions(measurements: dict[str, float]) -> dict[str, float]:
    """Per-metric normalised risk contributions (kpm_risk_v1)."""
    out: dict[str, float] = {}
    v = measurements.get("RRU.PrbTotDl")
    if v is not None:
        out["RRU.PrbTotDl"] = _clamp01(float(v) / 100.0)
    v = measurements.get("DRB.RlcSduDelayDl")
    if v is not None:
        out["DRB.RlcSduDelayDl"] = _clamp01(float(v) / DELAY_BUDGET_MS)
    v = measurements.get("DRB.UEThpDl")
    if v is not None:
        out["DRB.UEThpDl"] = _clamp01(1.0 - float(v) / THROUGHPUT_FLOOR_KBPS)
    for name in ("DRB.PdcpSduVolumeDL", "DRB.RlcSduTransmittedVolumeDL"):
        v = measurements.get(name)
        if v is not None:
            out[name] = _clamp01(float(v) / VOLUME_REFERENCE_KBIT)
    return out


def derive_sla_risk(measurements: dict[str, float]) -> tuple[float, str | None, dict[str, float]]:
    """Map real KPM measurements onto ``sla_risk_30s`` ∈ [0, 1].

    Returns ``(risk, dominant_metric, contributions)``; the risk is the
    maximum contribution (worst signal wins). With no mapped metric the
    floor ``RISK_FLOOR`` is returned and ``dominant_metric`` is None.
    """
    contributions = _risk_contributions(measurements)
    if not contributions:
        return RISK_FLOOR, None, {}
    metric = max(contributions, key=lambda k: contributions[k])
    return contributions[metric], metric, contributions


# ---------------------------------------------------------------------------
# ASN.1 helpers.
# ---------------------------------------------------------------------------
def _decode_timestamp(ts_octets: bytes) -> datetime | None:
    """Decode the KPM ``TimeStamp`` (OCTET STRING SIZE(8)).

    The value is uint64 microseconds since the Unix epoch
    (FlexRIC fills it from ``time_now_us()``). Two wire layouts are
    accepted:

    * straight big-endian (the natural network order);
    * the layout FlexRIC actually emits — two big-endian 32-bit words
      with the LOW word first. Its encoder applies ``htonll`` and then
      ``int_64_to_octet_string`` (which swaps again on little-endian
      hosts), so the words come out swapped; observed empirically on
      captured PDUs and matched here so real captures carry their real
      collection time.

    A plausibility window (~2001..2286 in µs) arbitrates between the
    two; values plausible in neither layout yield None rather than a
    nonsense timestamp — the emulator may fill random octets.
    """
    if len(ts_octets) != 8:
        return None
    candidates = (
        int.from_bytes(ts_octets, "big"),
        (int.from_bytes(ts_octets[4:], "big") << 32) | int.from_bytes(ts_octets[:4], "big"),
    )
    for us in candidates:
        if 1_000_000_000_000_000 <= us < 10_000_000_000_000_000_000:
            try:
                return datetime.fromtimestamp(us / 1e6, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):  # pragma: no cover
                return None
    return None


def _meas_value(record_item: tuple[str, Any]) -> float | None:
    """MeasurementRecordItem CHOICE → float (``noValue`` → None)."""
    kind, value = record_item
    if kind == "integer":
        return float(value)
    if kind == "real":
        return float(value)
    return None  # noValue


def _format1_measurements(fmt1: dict[str, Any]) -> tuple[dict[str, float], int | None]:
    """Extract ``{metric_name: value}`` from IndicationMessage-Format1.

    ``measInfoList`` names the columns; ``measData`` rows are one
    granularity period each with ``measRecord`` aligned column-wise.
    The LATEST row wins (matching how a live monitor consumes the
    stream); ``noValue`` entries are skipped. Nameless columns (some
    E2 nodes send ``measID`` instead) are keyed ``measID.<n>``.
    """
    info_list = fmt1.get("measInfoList") or []
    names: list[str] = []
    for item in info_list:
        kind, value = item["measType"]
        names.append(value if kind == "measName" else f"measID.{value}")

    meas_data = fmt1.get("measData") or []
    measurements: dict[str, float] = {}
    if meas_data:
        latest = meas_data[-1]
        for idx, record_item in enumerate(latest.get("measRecord") or []):
            value = _meas_value(record_item)
            if value is None:
                continue
            name = names[idx] if idx < len(names) else f"column.{idx}"
            measurements[name] = value
    return measurements, fmt1.get("granulPeriod")


def _ue_id_summary(ue_id: tuple[str, Any]) -> dict[str, Any]:
    """UEID CHOICE → JSON-safe summary dict (ids kept, bytes hexed)."""
    kind, body = ue_id

    def _jsonable(v: Any) -> Any:
        if isinstance(v, bytes):
            return v.hex()
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
            return {v[0]: _jsonable(v[1])}  # nested CHOICE
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], bytes):
            return {"bits": v[0].hex(), "n_bits": v[1]}  # BIT STRING
        if isinstance(v, dict):
            return {k: _jsonable(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_jsonable(x) for x in v]
        return v

    return {"type": kind, **{k: _jsonable(v) for k, v in body.items()}}


# ---------------------------------------------------------------------------
# The bridge.
# ---------------------------------------------------------------------------
@dataclass
class KpmMeasurementBridge:
    """Convert E2SM-KPM measurement records into TelemetryEvents.

    One instance per E2 node / xApp feed; ``source_id`` stamps every
    event so downstream sequencing and audit trails stay per-source.
    """

    source_id: str = "flexric-kpm"
    e2sm_version: str = "v3.00"
    _sequence: int = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _build_event(
        self,
        measurements: dict[str, float],
        *,
        ts_utc: datetime | None,
        ingest: str,
        granularity_period_ms: int | None = None,
        ue_id: dict[str, Any] | None = None,
        collect_start_time_utc: str | None = None,
        sender_name: str | None = None,
    ) -> TelemetryEvent:
        risk, metric, contributions = derive_sla_risk(measurements)
        payload: dict[str, Any] = {
            "measurements": measurements,
            "sla_risk_30s": risk,
            "risk": {
                "mapping": RISK_MAPPING_ID,
                "metric": metric,
                "contributions": contributions,
            },
        }
        if granularity_period_ms is not None:
            payload["granularity_period_ms"] = granularity_period_ms
        if ue_id is not None:
            payload["ue_id"] = ue_id
        if collect_start_time_utc is not None:
            payload["collect_start_time_utc"] = collect_start_time_utc
        if sender_name is not None:
            payload["sender_name"] = sender_name

        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            modality="kpm_5g",
            source_id=self.source_id,
            ts_utc=ts_utc or datetime.now(timezone.utc),
            sequence=self._next_sequence(),
            payload=payload,
            tags={
                "e2sm": f"kpm-{self.e2sm_version}",
                "ingest": ingest,
            },
        )
        logger.info(
            "kpm_bridge.event",
            event_id=event.event_id,
            ingest=ingest,
            n_measurements=len(measurements),
            sla_risk_30s=round(risk, 4),
            risk_metric=metric,
        )
        return event

    # -- path (a): FlexRIC KPM xApp JSON record -------------------------
    def from_flexric_kpm_json(self, record: dict[str, Any]) -> TelemetryEvent:
        """One FlexRIC KPM xApp record (JSON dict) → one TelemetryEvent.

        Accepted shapes (tolerant on purpose — the xApp's stdout log and
        its sqlite rows serialise slightly differently):

        * ``{"measurements": {name: value, ...}, ...}`` — canonical;
        * flat dicts where every key containing a dot (``DRB.UEThpDl``)
          is treated as a measurement.

        Optional keys: ``ts_utc``/``timestamp`` (ISO 8601 or epoch
        seconds), ``ue_id``, ``granularity_period_ms``/``granulPeriod``,
        ``sender_name``.
        """
        if not isinstance(record, dict):
            raise KpmBridgeError(f"expected dict record, got {type(record).__name__}")

        raw = record.get("measurements")
        if raw is None:
            raw = {k: v for k, v in record.items() if isinstance(k, str) and "." in k}
        if not isinstance(raw, dict) or not raw:
            raise KpmBridgeError("record carries no KPM measurements")
        measurements: dict[str, float] = {}
        for name, value in raw.items():
            try:
                measurements[str(name)] = float(value)
            except (TypeError, ValueError) as exc:
                raise KpmBridgeError(f"non-numeric measurement {name!r}: {value!r}") from exc

        ts_utc: datetime | None = None
        raw_ts = record.get("ts_utc") or record.get("timestamp")
        if isinstance(raw_ts, (int, float)):
            ts_utc = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        elif isinstance(raw_ts, str):
            try:
                parsed = datetime.fromisoformat(raw_ts)
            except ValueError as exc:
                raise KpmBridgeError(f"unparseable timestamp {raw_ts!r}") from exc
            ts_utc = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

        gran = record.get("granularity_period_ms", record.get("granulPeriod"))
        ue_id = record.get("ue_id")
        return self._build_event(
            measurements,
            ts_utc=ts_utc,
            ingest="flexric_json",
            granularity_period_ms=int(gran) if gran is not None else None,
            ue_id=ue_id if isinstance(ue_id, dict) else None,
            sender_name=record.get("sender_name"),
        )

    # -- path (b): raw E2SM-KPM RIC Indication ASN.1 bytes --------------
    def from_e2sm_kpm_indication(
        self,
        indication_bytes: bytes,
        *,
        header_bytes: bytes | None = None,
    ) -> list[TelemetryEvent]:
        """Decode a real ``E2SM-KPM-IndicationMessage`` (aligned PER).

        ``indication_bytes`` is the OCTET STRING content of the E2AP
        RIC Indication's *IndicationMessage* IE; ``header_bytes``
        (optional) the *IndicationHeader* IE, used for the collection
        timestamp and sender name.

        Format 1/2 yield one event; Format 3 (REPORT Style 4,
        UE-level) yields one event per UE measurement report.
        """
        spec = kpm_spec()
        try:
            decoded = spec.decode("E2SM-KPM-IndicationMessage", indication_bytes)
        except Exception as exc:
            raise KpmBridgeError(f"E2SM-KPM IndicationMessage decode failed: {exc}") from exc

        collect_ts: datetime | None = None
        collect_iso: str | None = None
        sender_name: str | None = None
        if header_bytes is not None:
            try:
                hdr = spec.decode("E2SM-KPM-IndicationHeader", header_bytes)
            except Exception as exc:
                raise KpmBridgeError(f"E2SM-KPM IndicationHeader decode failed: {exc}") from exc
            hdr_kind, hdr_body = hdr["indicationHeader-formats"]
            if hdr_kind == "indicationHeader-Format1":
                collect_ts = _decode_timestamp(hdr_body.get("colletStartTime", b""))
                collect_iso = collect_ts.isoformat() if collect_ts else None
                sender_name = hdr_body.get("senderName")

        fmt_kind, fmt_body = decoded["indicationMessage-formats"]
        events: list[TelemetryEvent] = []
        if fmt_kind in ("indicationMessage-Format1", "indicationMessage-Format2"):
            measurements, gran = _format1_measurements(fmt_body)
            events.append(
                self._build_event(
                    measurements,
                    ts_utc=collect_ts,
                    ingest="asn1_aper",
                    granularity_period_ms=gran,
                    collect_start_time_utc=collect_iso,
                    sender_name=sender_name,
                )
            )
        elif fmt_kind == "indicationMessage-Format3":
            for report in fmt_body.get("ueMeasReportList") or []:
                measurements, gran = _format1_measurements(report["measReport"])
                events.append(
                    self._build_event(
                        measurements,
                        ts_utc=collect_ts,
                        ingest="asn1_aper",
                        granularity_period_ms=gran,
                        ue_id=_ue_id_summary(report["ueID"]),
                        collect_start_time_utc=collect_iso,
                        sender_name=sender_name,
                    )
                )
        else:  # pragma: no cover — spec has exactly three formats
            raise KpmBridgeError(f"unsupported indication format {fmt_kind!r}")
        return events


# ---------------------------------------------------------------------------
# Pipeline runner — bridged events drive real Shield-gated A1 decisions.
# ---------------------------------------------------------------------------
async def e2_to_pipeline(
    events: Iterable[TelemetryEvent] | AsyncIterator[TelemetryEvent],
    pipeline: Any,
) -> list[Any]:
    """Run bridged KPM TelemetryEvents through the decision pipeline.

    ``pipeline`` is a :class:`horizon_ric.rapp.pipeline.DecisionPipeline`
    (duck-typed on ``process_event`` so tests can substitute doubles).
    Per-event pipeline failures are logged and re-raised — an E2-fed
    decision that errors must be visible, not swallowed.
    """
    results: list[Any] = []

    async def _process(event: TelemetryEvent) -> None:
        result = await pipeline.process_event(event)
        results.append(result)
        logger.info(
            "e2_to_pipeline.decision",
            event_id=event.event_id,
            decision_id=getattr(result, "decision_id", None),
            accepted=getattr(result, "accepted", None),
            blocked=getattr(result, "blocked", None),
            policy_type=getattr(result, "policy_type", None),
        )

    if hasattr(events, "__aiter__"):
        async for event in events:  # type: ignore[union-attr]
            await _process(event)
    else:
        for event in events:  # type: ignore[union-attr]
            await _process(event)
    return results


__all__ = [
    "ASN1_DIR",
    "KPM_ASN1_FILE",
    "KpmBridgeError",
    "KpmMeasurementBridge",
    "RISK_MAPPING_ID",
    "derive_sla_risk",
    "e2_to_pipeline",
    "kpm_spec",
]
