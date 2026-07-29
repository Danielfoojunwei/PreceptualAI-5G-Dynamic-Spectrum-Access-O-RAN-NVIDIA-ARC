"""E2SM-KPM bridge tests — real FlexRIC-encoded PDUs → TelemetryEvents → pipeline.

The binary fixtures under ``tests/fixtures/e2sm_kpm/`` were produced by
FlexRIC's own emulator-agent indication callback and its production
asn1c aligned-PER wire encoder (see the fixtures' ``PROVENANCE.md``) —
decoding them here with ``asn1tools`` against the committed O-RAN
E2SM-KPM v3.00 spec is a genuine cross-implementation check. No FlexRIC
process needs to be running: the decode path is self-contained.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from horizon_ric.e2.kpm_bridge import (
    KPM_ASN1_FILE,
    RISK_FLOOR,
    RISK_MAPPING_ID,
    KpmBridgeError,
    KpmMeasurementBridge,
    derive_sla_risk,
    e2_to_pipeline,
    kpm_spec,
)
from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.io.schemas import TelemetryEvent
from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig
from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig

FIXTURES = Path(__file__).parent / "fixtures" / "e2sm_kpm"
IND_MSG = (FIXTURES / "indication_msg.aper.bin").read_bytes()
IND_HDR = (FIXTURES / "indication_hdr.aper.bin").read_bytes()

# The 3GPP TS 28.552 measurement set the FlexRIC gNB emulator advertises
# for REPORT Style 4 (examples/emulator/agent/sm_kpm.c, NGRAN_GNB).
GNB_STYLE4_METRICS = [
    "DRB.PdcpSduVolumeDL",
    "DRB.PdcpSduVolumeUL",
    "DRB.RlcSduDelayDl",
    "DRB.UEThpDl",
    "DRB.UEThpUl",
    "RRU.PrbTotDl",
    "RRU.PrbTotUl",
]


# ---------------------------------------------------------------------------
# ASN.1 spec provenance + raw decode of the captured PDUs
# ---------------------------------------------------------------------------
def test_committed_asn1_spec_matches_flexric_provenance():
    digest = hashlib.sha256(KPM_ASN1_FILE.read_bytes()).hexdigest()
    assert digest == "ab473a9cc60bc3e3b1e32d032ce3b7158175c210282a006d662b1c48bc8a573a"
    provenance = (KPM_ASN1_FILE.parent / "PROVENANCE.md").read_text()
    assert digest in provenance


def test_spec_compiles_and_exposes_indication_types():
    spec = kpm_spec()
    for name in (
        "E2SM-KPM-IndicationMessage",
        "E2SM-KPM-IndicationHeader",
        "E2SM-KPM-ActionDefinition",
    ):
        assert name in spec.types
    assert kpm_spec() is spec  # cached singleton


def test_decode_real_flexric_indication_message():
    spec = kpm_spec()
    decoded = spec.decode("E2SM-KPM-IndicationMessage", IND_MSG)
    kind, body = decoded["indicationMessage-formats"]
    assert kind == "indicationMessage-Format3"  # REPORT Style 4, UE-level
    reports = body["ueMeasReportList"]
    assert len(reports) == 6
    for report in reports:
        fmt1 = report["measReport"]
        names = [item["measType"][1] for item in fmt1["measInfoList"]]
        assert names == GNB_STYLE4_METRICS
        row = fmt1["measData"][-1]["measRecord"]
        assert len(row) == len(names)
        assert all(item[0] in {"integer", "real"} for item in row)


def test_decode_real_flexric_indication_header():
    spec = kpm_spec()
    decoded = spec.decode("E2SM-KPM-IndicationHeader", IND_HDR)
    kind, body = decoded["indicationHeader-formats"]
    assert kind == "indicationHeader-Format1"
    assert body["senderName"] == "My OAI-MONO"
    assert body["vendorName"] == "OAI"
    assert len(body["colletStartTime"]) == 8


# ---------------------------------------------------------------------------
# Risk mapping (kpm_risk_v1)
# ---------------------------------------------------------------------------
def test_risk_prb_utilisation_dominates():
    risk, metric, contributions = derive_sla_risk(
        {"RRU.PrbTotDl": 87.0, "DRB.UEThpDl": 9_000.0}
    )
    assert metric == "RRU.PrbTotDl"
    assert risk == pytest.approx(0.87)
    assert contributions["DRB.UEThpDl"] == pytest.approx(0.1)


def test_risk_delay_and_throughput_mappings():
    risk, metric, _ = derive_sla_risk({"DRB.RlcSduDelayDl": 25.0})
    assert (risk, metric) == (pytest.approx(0.5), "DRB.RlcSduDelayDl")
    risk, metric, _ = derive_sla_risk({"DRB.UEThpDl": 2_500.0})
    assert (risk, metric) == (pytest.approx(0.75), "DRB.UEThpDl")
    risk, _, _ = derive_sla_risk({"DRB.PdcpSduVolumeDL": 250_000.0})
    assert risk == 1.0  # clamped


def test_risk_floor_without_mapped_metrics():
    risk, metric, contributions = derive_sla_risk({"CARR.PDSCHMCSDist": 12.0})
    assert (risk, metric, contributions) == (RISK_FLOOR, None, {})


# ---------------------------------------------------------------------------
# Ingestion path (b): real ASN.1 indication bytes
# ---------------------------------------------------------------------------
def test_bridge_converts_real_indication_to_events():
    bridge = KpmMeasurementBridge(source_id="flexric-emu-gnb")
    events = bridge.from_e2sm_kpm_indication(IND_MSG, header_bytes=IND_HDR)
    assert len(events) == 6
    for sequence, event in enumerate(events, start=1):
        assert isinstance(event, TelemetryEvent)
        assert event.modality == "kpm_5g"
        assert event.source_id == "flexric-emu-gnb"
        assert event.sequence == sequence
        assert event.tags["ingest"] == "asn1_aper"
        assert event.tags["e2sm"] == "kpm-v3.00"
        payload = event.payload
        assert set(payload["measurements"]) == set(GNB_STYLE4_METRICS)
        assert 0.0 <= payload["sla_risk_30s"] <= 1.0
        assert payload["risk"]["mapping"] == RISK_MAPPING_ID
        assert payload["risk"]["metric"] in GNB_STYLE4_METRICS
        assert payload["ue_id"]["type"] == "gNB-UEID"
        assert payload["sender_name"] == "My OAI-MONO"
        # Header collectStartTime (FlexRIC word-swapped µs) decoded to a
        # real, tz-aware capture time — not a fallback now().
        assert event.ts_utc.tzinfo is not None
        assert payload["collect_start_time_utc"] == event.ts_utc.isoformat()
        assert datetime(2026, 1, 1, tzinfo=timezone.utc) <= event.ts_utc
        assert event.ts_utc <= datetime(2027, 1, 1, tzinfo=timezone.utc)
        # Events survive the bus round-trip contract.
        TelemetryEvent.model_validate_json(event.model_dump_json())


def test_bridge_rejects_garbage_indication_bytes():
    bridge = KpmMeasurementBridge()
    with pytest.raises(KpmBridgeError, match="IndicationMessage decode failed"):
        bridge.from_e2sm_kpm_indication(b"\x00\x01\x02garbage")
    with pytest.raises(KpmBridgeError, match="IndicationHeader decode failed"):
        bridge.from_e2sm_kpm_indication(IND_MSG, header_bytes=b"\xff")


def test_round_trip_format1_with_same_spec():
    """Spec-conformant SYNTHETIC vector (encode→decode with asn1tools).

    Clearly labelled: this Format 1 PDU is built by the same asn1tools
    spec that decodes it — it proves the Format 1/2 bridge path, not
    cross-implementation encoding (the captured fixtures above do that).
    """
    spec = kpm_spec()
    fmt1 = {
        "measData": [
            {"measRecord": [("integer", 42), ("real", 12.5), ("noValue", None)]},
            {"measRecord": [("integer", 58), ("real", 44.0), ("noValue", None)]},
        ],
        "measInfoList": [
            {"measType": ("measName", "RRU.PrbTotDl"), "labelInfoList": [{"measLabel": {"noLabel": "true"}}]},
            {"measType": ("measName", "DRB.RlcSduDelayDl"), "labelInfoList": [{"measLabel": {"noLabel": "true"}}]},
            {"measType": ("measName", "DRB.UEThpUl"), "labelInfoList": [{"measLabel": {"noLabel": "true"}}]},
        ],
        "granulPeriod": 1000,
    }
    encoded = spec.encode(
        "E2SM-KPM-IndicationMessage",
        {"indicationMessage-formats": ("indicationMessage-Format1", fmt1)},
    )
    events = KpmMeasurementBridge().from_e2sm_kpm_indication(encoded)
    assert len(events) == 1
    payload = events[0].payload
    # Latest measData row wins; noValue column dropped.
    assert payload["measurements"] == {"RRU.PrbTotDl": 58.0, "DRB.RlcSduDelayDl": 44.0}
    assert payload["granularity_period_ms"] == 1000
    assert payload["risk"]["metric"] == "DRB.RlcSduDelayDl"  # 44/50 > 58/100
    assert payload["sla_risk_30s"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Ingestion path (a): FlexRIC KPM xApp JSON records
# ---------------------------------------------------------------------------
def test_from_flexric_json_canonical_shape():
    bridge = KpmMeasurementBridge(source_id="xapp-kpm-moni")
    event = bridge.from_flexric_kpm_json(
        {
            "measurements": {"RRU.PrbTotDl": 42, "DRB.UEThpDl": 6_000.0},
            "ts_utc": "2026-07-27T08:40:27+00:00",
            "granularity_period_ms": 1000,
            "ue_id": {"type": "gNB-UEID", "amf-UE-NGAP-ID": 112358132134},
            "sender_name": "My OAI-MONO",
        }
    )
    assert event.tags["ingest"] == "flexric_json"
    assert event.ts_utc == datetime(2026, 7, 27, 8, 40, 27, tzinfo=timezone.utc)
    assert event.payload["measurements"] == {"RRU.PrbTotDl": 42.0, "DRB.UEThpDl": 6_000.0}
    assert event.payload["sla_risk_30s"] == pytest.approx(0.42)
    assert event.payload["ue_id"]["amf-UE-NGAP-ID"] == 112358132134


def test_from_flexric_json_flat_shape_and_epoch_ts():
    event = KpmMeasurementBridge().from_flexric_kpm_json(
        {"DRB.RlcSduDelayDl": 10.0, "RRU.PrbTotUl": 3, "tstamp_raw": "ignored-no-dot"}
    )
    assert event.payload["measurements"] == {"DRB.RlcSduDelayDl": 10.0, "RRU.PrbTotUl": 3.0}
    epoch = KpmMeasurementBridge().from_flexric_kpm_json(
        {"measurements": {"RRU.PrbTotDl": 5}, "timestamp": 1_785_141_627.0}
    )
    assert epoch.ts_utc == datetime(2026, 7, 27, 8, 40, 27, tzinfo=timezone.utc)


def test_from_flexric_json_rejects_bad_records():
    bridge = KpmMeasurementBridge()
    with pytest.raises(KpmBridgeError, match="no KPM measurements"):
        bridge.from_flexric_kpm_json({"note": "empty"})
    with pytest.raises(KpmBridgeError, match="non-numeric measurement"):
        bridge.from_flexric_kpm_json({"measurements": {"DRB.UEThpDl": "fast"}})
    with pytest.raises(KpmBridgeError, match="unparseable timestamp"):
        bridge.from_flexric_kpm_json(
            {"measurements": {"DRB.UEThpDl": 1.0}, "ts_utc": "yesterday"}
        )
    with pytest.raises(KpmBridgeError, match="expected dict"):
        bridge.from_flexric_kpm_json(["not", "a", "dict"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# e2_to_pipeline: a REAL captured KPM record drives a Shield-gated A1 decision
# ---------------------------------------------------------------------------
def _mock_a1_adapter(state: dict) -> A1Adapter:
    """A1Adapter (legacy dialect) against a recording httpx.MockTransport."""

    async def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if req.method == "PUT" and "/policies/" in path:
            state["policies"].append(
                {"path": path, "body": json.loads(req.content)}
            )
            return httpx.Response(202)
        if req.method == "PUT":
            return httpx.Response(201)
        if req.method == "GET" and path.endswith("/status"):
            return httpx.Response(200, json={"instance_status": "IN EFFECT"})
        return httpx.Response(404)

    cfg = A1AdapterConfig()
    adapter = A1Adapter(cfg)
    adapter._client = httpx.AsyncClient(
        base_url=cfg.near_rt_ric_base_url, transport=httpx.MockTransport(handler)
    )
    return adapter


@pytest.mark.asyncio
async def test_real_kpm_indication_drives_pipeline_decision(tmp_path: Path):
    state: dict = {"policies": []}
    adapter = _mock_a1_adapter(state)
    store = JsonlEvidenceStore(tmp_path / "audit.jsonl")
    adapter.attach_evidence_store(store)
    pipeline = DecisionPipeline(
        adapter, store, PipelineConfig(status_poll_interval_s=0.01)
    )

    bridge = KpmMeasurementBridge(source_id="flexric-emu-gnb")
    events = bridge.from_e2sm_kpm_indication(IND_MSG, header_bytes=IND_HDR)
    try:
        results = await e2_to_pipeline(events[:1], pipeline)
    finally:
        await adapter.close()

    assert len(results) == 1
    result = results[0]
    assert result.accepted is True
    assert result.blocked is False
    assert result.enforced is True
    # Risk band chosen from the REAL KPM-derived sla_risk_30s.
    risk = events[0].payload["sla_risk_30s"]
    expected_type = (
        "horizon.qos.priority"
        if risk < 0.3
        else "horizon.traffic.steering" if risk < 0.6 else "horizon.admission.control"
    )
    assert result.policy_type == expected_type
    # The A1 wire saw the policy PUT; the evidence chain has the decision.
    assert len(state["policies"]) == 1
    assert len(store) == 1
    record, _ = next(iter(store))
    assert record.decision_id == result.decision_id
    assert record.chosen_action["source_event"] == events[0].event_id


@pytest.mark.asyncio
async def test_e2_to_pipeline_accepts_async_iterators(tmp_path: Path):
    class _Recorder:
        def __init__(self) -> None:
            self.seen: list[str] = []

        async def process_event(self, event: TelemetryEvent) -> str:
            self.seen.append(event.event_id)
            return f"done-{event.event_id}"

    bridge = KpmMeasurementBridge()
    events = bridge.from_e2sm_kpm_indication(IND_MSG)

    async def _aiter():
        for event in events:
            yield event

    recorder = _Recorder()
    results = await e2_to_pipeline(_aiter(), recorder)
    assert results == [f"done-{e.event_id}" for e in events]
    assert recorder.seen == [e.event_id for e in events]
