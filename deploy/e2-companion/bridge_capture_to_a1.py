#!/usr/bin/env python3
"""Bridge REAL E2SM-KPM indications into the Horizon pipeline over a real A1.

Input (one of):
  * ``--e2ap-capture capture.jsonl`` — sniffed E2AP-over-loopback datagrams
    (one JSON object per line with ``payload_hex``, as written by
    ``sniff_e2ap.py``); the newest RICindication (procedureCode 5) is used
    and the E2SM-KPM header/message OCTET STRINGs are carved out of it
    (APER length-determinant scan + exact re-encode verification);
  * ``--indication-msg FILE [--indication-hdr FILE]`` — raw aligned-PER
    E2SM-KPM PDU files (defaults: the live-captured fixtures under
    ``tests/fixtures/e2sm_kpm/``).

The PDUs are decoded with ``horizon_ric.e2.kpm_bridge`` (asn1tools,
E2SM-KPM v3.00), bridged to TelemetryEvents (modality ``kpm_5g`` with the
documented ``kpm_risk_v1`` SLA-risk mapping) and driven through the REAL
:class:`~horizon_ric.rapp.pipeline.DecisionPipeline` — planner → Shield →
guard chain → A1 emit → enforcement poll — against the A1 endpoint given
by ``--a1-url`` / ``--dialect``. Every decision lands in the evidence
store; a JSON report is written for the proof document.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from horizon_ric.e2.kpm_bridge import (  # noqa: E402
    KpmMeasurementBridge,
    e2_to_pipeline,
    kpm_spec,
)
from horizon_ric.evidence.store import JsonlEvidenceStore  # noqa: E402
from horizon_ric.rapp.a1_adapter import A1Adapter, A1AdapterConfig  # noqa: E402
from horizon_ric.rapp.pipeline import DecisionPipeline, PipelineConfig  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "e2sm_kpm"


def carve(pdu: bytes, typename: str, lmin: int, lmax: int) -> list[tuple[int, bytes]]:
    """Find APER length-prefixed slices of ``pdu`` that decode as ``typename``
    and re-encode byte-identically (exact round-trip)."""
    spec = kpm_spec()
    hits: list[tuple[int, bytes]] = []
    for i in range(len(pdu)):
        candidates = []
        if pdu[i] < 0x80:
            candidates.append((i + 1, pdu[i]))
        elif (pdu[i] & 0xC0) == 0x80 and i + 1 < len(pdu):
            candidates.append((i + 2, ((pdu[i] & 0x3F) << 8) | pdu[i + 1]))
        for start, length in candidates:
            if not (lmin <= length <= lmax) or start + length > len(pdu):
                continue
            chunk = pdu[start : start + length]
            try:
                decoded = spec.decode(typename, chunk)
                if spec.encode(typename, decoded) == chunk:
                    hits.append((start, chunk))
            except Exception:
                continue
    return hits


def load_from_capture(path: Path) -> tuple[bytes, bytes | None]:
    """Latest RICindication in the sniffer capture → (msg_bytes, hdr_bytes)."""
    indications = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            payload = bytes.fromhex(row["payload_hex"])
            # E2AP initiatingMessage (0x00) + procedureCode 5 = RICindication
            if row.get("dst_port") == 36421 and len(payload) > 2 and payload[0] == 0x00 and payload[1] == 5:
                indications.append(payload)
    if not indications:
        raise SystemExit(f"no E2AP RICindication datagrams in {path}")
    # Newest first; skip indications whose payload is not a carvable KPM
    # message (e.g. the emulator's randomly-formatted second subscription).
    for pdu in reversed(indications):
        msgs = carve(pdu, "E2SM-KPM-IndicationMessage", 100, 16000)
        if not msgs:
            continue
        hdrs = carve(pdu, "E2SM-KPM-IndicationHeader", 10, 400)
        print(
            f"carved from live E2AP RICindication ({len(pdu)} B): "
            f"msg@{msgs[0][0]} ({len(msgs[0][1])} B), "
            f"hdr@{hdrs[0][0] if hdrs else '-'}"
        )
        return msgs[0][1], hdrs[0][1] if hdrs else None
    raise SystemExit("no carvable E2SM-KPM IndicationMessage in any RICindication PDU")


async def run(args: argparse.Namespace) -> int:
    if args.e2ap_capture:
        msg_bytes, hdr_bytes = load_from_capture(Path(args.e2ap_capture))
    else:
        msg_bytes = Path(args.indication_msg).read_bytes()
        hdr_bytes = Path(args.indication_hdr).read_bytes() if args.indication_hdr else None

    bridge = KpmMeasurementBridge(source_id=args.source_id)
    events = bridge.from_e2sm_kpm_indication(msg_bytes, header_bytes=hdr_bytes)
    if args.max_events > 0:
        events = events[: args.max_events]
    print(f"bridged {len(events)} TelemetryEvent(s) from the KPM indication")

    adapter = A1Adapter(
        A1AdapterConfig(near_rt_ric_base_url=args.a1_url, dialect=args.dialect)
    )
    store = JsonlEvidenceStore(Path(args.evidence))
    adapter.attach_evidence_store(store)
    pipeline = DecisionPipeline(adapter, store, PipelineConfig.from_env())

    try:
        registered = await adapter.register_policy_types()
        print(f"A1 policy types registered: {registered}")
        results = await e2_to_pipeline(events, pipeline)
    finally:
        await adapter.close()

    report = {
        "a1_url": args.a1_url,
        "dialect": args.dialect,
        "indication_msg_sha256": __import__("hashlib").sha256(msg_bytes).hexdigest(),
        "n_events": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "ts_utc": e.ts_utc.isoformat(),
                "sla_risk_30s": e.payload["sla_risk_30s"],
                "risk": e.payload["risk"],
                "measurements": e.payload["measurements"],
                "ue_id": e.payload.get("ue_id"),
            }
            for e in events
        ],
        "results": [asdict(r) for r in results],
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2, default=str))
    print(f"report -> {args.report_json}")
    ok = all(r.accepted or r.blocked for r in results)
    for r in results:
        print(
            f"decision {r.decision_id}: accepted={r.accepted} blocked={r.blocked} "
            f"policy_type={r.policy_type} policy_id={r.policy_id} "
            f"enforcement={r.enforcement_status}"
        )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2ap-capture", help="sniffer JSONL with live E2AP datagrams")
    parser.add_argument(
        "--indication-msg", default=str(FIXTURES / "indication_msg.aper.bin")
    )
    parser.add_argument(
        "--indication-hdr", default=str(FIXTURES / "indication_hdr.aper.bin")
    )
    parser.add_argument("--a1-url", default="http://127.0.0.1:10000")
    parser.add_argument("--dialect", default="legacy")
    parser.add_argument("--source-id", default="flexric-emu-gnb")
    parser.add_argument("--max-events", type=int, default=1)
    parser.add_argument("--evidence", default="/tmp/e2-companion-evidence.jsonl")
    parser.add_argument("--report-json", default="/tmp/e2-companion-report.json")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
