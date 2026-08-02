"""OCUDU's JSON metrics as telemetry the trust gate can admit.

Horizon's benchmarks are driven by ray-traced propagation, which is measured
data but not *RAN* data: it says what the channel does, not what a scheduler
did with it. OCUDU emits real per-cell and per-UE scheduler metrics as JSON,
and it is the same gNB the E2 join proof used — so this closes the loop from a
real RAN into the enforcement path instead of stopping at the channel.

The interesting part is not the parsing. It is that ``telemetry_trust`` will
not admit any of it on trust: a metrics blob from a gNB is exactly the shared,
cross-domain state whose unconditional trust the agentic layer was built to
remove. So this converts OCUDU's output into
:class:`~horizon_agentic.telemetry_trust.TelemetryRecord` and leaves every
check to the gate — the gNB is a source like any other, subject to
authentication, freshness, bounds and corroboration.

Two things it deliberately does **not** do.

It does not sign on OCUDU's behalf. OCUDU emits unauthenticated JSON; there is
no key, and inventing one here would produce a record that looks authenticated
and is not. :func:`records_from_metrics` returns unsigned records, and a
deployment must either put them behind a collector that holds a real key or
accept that the source-authentication check will refuse them. Refusing is the
correct default and is what happens if nobody decides.

It does not invent fields. Only metrics OCUDU actually emits are mapped, under
names taken from its own JSON generators. A field Horizon would like — say a
per-slice PRB share — is absent here because OCUDU does not report it at this
level, and a plausible-looking substitute would be worse than the gap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from horizon_agentic.telemetry_trust import FieldBound, TelemetryRecord

__all__ = [
    "CELL_FIELDS",
    "UE_FIELDS",
    "OCUDU_BOUNDS",
    "CellMetrics",
    "parse_metrics",
    "records_from_metrics",
]

# Field names as OCUDU's own JSON generators emit them
# (apps/helpers/metrics/json_generators/du_high/scheduler.cpp). Kept verbatim so
# a reader can grep for them in OCUDU rather than trusting a rename.
CELL_FIELDS: tuple[str, ...] = (
    "average_latency",
    "error_indication_count",
    "failed_dl_pdcch",
    "failed_ul_pdcch",
    "late_dl_harqs",
    "late_ul_harqs",
    "max_latency",
)
UE_FIELDS: tuple[str, ...] = (
    "cqi",
    "dl_brate",
    "dl_mcs",
    "dl_nof_ok",
    "dl_nof_nok",
    "dl_ri",
    "last_phr",
    "bsr",
)

# Physical bounds for the trust gate. Deliberately generous where the quantity
# is unbounded in principle (rates, counters) and tight where physics or the
# standard bounds it: CQI is 0..15, MCS 0..31, rank 1..8, PHR is a 3GPP report
# range. A bound nobody can justify is a bound that only refuses honest data.
OCUDU_BOUNDS: Mapping[str, FieldBound] = {
    "cqi": FieldBound(0.0, 15.0, ""),
    "dl_mcs": FieldBound(0.0, 31.0, ""),
    "dl_ri": FieldBound(1.0, 8.0, ""),
    "last_phr": FieldBound(-23.0, 40.0, "dB"),
    "bsr": FieldBound(0.0, 1e12, "bytes"),
    "dl_brate": FieldBound(0.0, 1e12, "bit/s"),
    "dl_nof_ok": FieldBound(0.0, 1e12, ""),
    "dl_nof_nok": FieldBound(0.0, 1e12, ""),
    "average_latency": FieldBound(0.0, 1e9, "us"),
    "max_latency": FieldBound(0.0, 1e9, "us"),
    "error_indication_count": FieldBound(0.0, 1e12, ""),
    "failed_dl_pdcch": FieldBound(0.0, 1e12, ""),
    "failed_ul_pdcch": FieldBound(0.0, 1e12, ""),
    "late_dl_harqs": FieldBound(0.0, 1e12, ""),
    "late_ul_harqs": FieldBound(0.0, 1e12, ""),
}


@dataclass(frozen=True)
class CellMetrics:
    """One cell's scheduler metrics from one OCUDU report."""

    pci: int
    timestamp: float
    cell: Mapping[str, float]
    ues: tuple[Mapping[str, Any], ...] = ()

    @property
    def source_id(self) -> str:
        """One source per cell, not per gNB.

        The trust gate's replay defence is per-source, and two cells report
        independently — sharing a source id between them would make one cell's
        sequence number silence the other.
        """
        return f"ocudu-pci-{self.pci}"


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def parse_metrics(payload: str | Mapping[str, Any]) -> tuple[CellMetrics, ...]:
    """Parse one OCUDU JSON metrics report.

    The envelope OCUDU writes is ``{"timestamp": ..., "du": {"du_high":
    {"mac": {"timestamp": ..., "cells": [{"cell_metrics": {...},
    "ue_list": [...]}]}}}}``. This tolerates the ``mac`` block being reached
    either through the full ``du/du_high`` path or supplied directly, because
    the split and monolithic apps nest it differently — but it does not invent
    a report when the structure is absent.
    """
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)

    mac: Any = data
    for key in ("du", "du_high", "mac"):
        if isinstance(mac, Mapping) and key in mac:
            mac = mac[key]
    if not isinstance(mac, Mapping) or "cells" not in mac:
        return ()

    timestamp = _number(mac.get("timestamp")) or _number(data.get("timestamp")) or 0.0
    out: list[CellMetrics] = []
    for entry in mac.get("cells") or ():
        if not isinstance(entry, Mapping):
            continue
        cell_block = entry.get("cell_metrics")
        if not isinstance(cell_block, Mapping):
            continue
        pci = _number(cell_block.get("pci"))
        fields = {
            name: value
            for name in CELL_FIELDS
            if (value := _number(cell_block.get(name))) is not None
        }
        ues = tuple(u for u in (entry.get("ue_list") or ()) if isinstance(u, Mapping))
        out.append(
            CellMetrics(
                pci=int(pci) if pci is not None else -1,
                timestamp=timestamp,
                cell=fields,
                ues=ues,
            )
        )
    return tuple(out)


def _aggregate_ues(ues: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Reduce a UE list to per-cell numbers the invariants can use.

    The *worst* value is taken for quality metrics rather than the mean. A cell
    whose average CQI is comfortable can still contain a UE at the edge of
    coverage, and an enforcement layer that reasons about the average would
    approve an action that harms exactly the user it should protect.
    """
    out: dict[str, float] = {}
    for name in UE_FIELDS:
        values = [
            v for u in ues if (v := _number(u.get(name))) is not None
        ]
        if not values:
            continue
        out[name] = min(values) if name in {"cqi", "dl_mcs", "dl_ri", "last_phr"} else sum(values)
    return out


def records_from_metrics(
    metrics: Iterable[CellMetrics], *, start_sequence: Mapping[str, int] | None = None
) -> tuple[TelemetryRecord, ...]:
    """Convert parsed metrics into unsigned telemetry records, one per cell.

    Unsigned on purpose — see the module docstring. The caller must sign with a
    key the trust policy knows, or the gate will refuse them, which is the
    correct outcome for a source nobody has decided to trust.
    """
    seq = dict(start_sequence or {})
    records: list[TelemetryRecord] = []
    for cell in metrics:
        fields = {**cell.cell, **_aggregate_ues(cell.ues)}
        if not fields:
            # A record with no fields is refused by the gate anyway; emitting
            # one would only turn a quiet gNB into a stream of refusals.
            continue
        source = cell.source_id
        seq[source] = seq.get(source, 0) + 1
        records.append(
            TelemetryRecord(
                source_id=source,
                observed_at=cell.timestamp,
                sequence=seq[source],
                fields=fields,
            )
        )
    return tuple(records)
