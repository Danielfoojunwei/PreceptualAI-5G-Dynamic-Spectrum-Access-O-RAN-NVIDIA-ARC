# horizon_ric.e2 — E2SM-KPM ingest for the Horizon rApp

Bridges **real E2SM-KPM measurement reports** — the RIC Indication
payloads a near-RT RIC E2 termination (FlexRIC, or any O-RAN WG3
compliant E2 node) receives from the RAN — into
`horizon_ric.io.schemas.TelemetryEvent` objects that drive the
existing `DecisionPipeline` (planner → Shield → guards → A1 →
evidence).

This package owns **no transport**: the near-RT RIC owns the SCTP/E2AP
association; Horizon is an rApp and consumes the E2SM payloads the RIC
surfaces (xApp output, or raw indication OCTET STRINGs).

## Components

- `kpm_bridge.KpmMeasurementBridge`
  - `from_flexric_kpm_json(record)` — one JSON record harvested from
    FlexRIC's KPM monitor xApp output → one `TelemetryEvent`.
  - `from_e2sm_kpm_indication(indication_bytes, header_bytes=None)` —
    decodes a raw aligned-PER `E2SM-KPM-IndicationMessage` (the OCTET
    STRING inside the E2AP RIC Indication) with **asn1tools** against
    the committed O-RAN E2SM-KPM v3.00 spec; Format 1/2 yield one
    event, Format 3 (UE-level Style 4) one event per UE report.
- `kpm_bridge.e2_to_pipeline(events, pipeline)` — async runner that
  feeds bridged events into `DecisionPipeline.process_event`, so a
  real KPM measurement produces a real Shield-gated A1 decision.
- `asn1/e2sm_kpm_v03.00_standard.asn1` — the spec text, byte-identical
  to the file FlexRIC's own asn1c wire codec was generated from
  (provenance + sha256 in `asn1/PROVENANCE.md`).

## Derived risk — `kpm_risk_v1`

The pipeline's risk-band planner keys off `payload["sla_risk_30s"]`.
The bridge computes it from real TS 28.552 metrics; each present
metric contributes a normalised risk in [0, 1] and the **maximum**
contribution wins (conservative). Mapping (documented constants in
`kpm_bridge.py`):

| metric | contribution |
| --- | --- |
| `RRU.PrbTotDl` (% PRB used) | `prb / 100` |
| `DRB.RlcSduDelayDl` (ms) | `delay / 50` |
| `DRB.UEThpDl` (kbps) | `1 − thp / 10000` |
| `DRB.PdcpSduVolumeDL`, `DRB.RlcSduTransmittedVolumeDL` (kbit) | `vol / 100000` |

No mapped metric present → floor `0.05` with `risk.metric = null`.
The full contribution set is recorded in `payload["risk"]` for audit.

## Tests

`tests/test_e2_kpm_bridge.py` runs without any FlexRIC process: the
fixtures under `tests/fixtures/e2sm_kpm/` are **live-captured** wire
PDUs from a real FlexRIC E2 association (see their `PROVENANCE.md`),
and the decode path is self-contained (committed spec + asn1tools).
The end-to-end harness and the live-run proof live in
`deploy/e2-companion/`.
