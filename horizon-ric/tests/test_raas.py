"""Tests for the rApps-as-a-Service (RaaS) endpoint.

A single PreceptualAI instance is shared by many tenants. The RaaS layer:
  * meters every successful A1 emission per tenant (billing counter)
  * exposes an aggregate /api/v1/billing/usage view
  * exports a signed audit-trail tarball per (tenant, since_ts)

These tests pin both the metering API and the export tarball/signature
shape so a regression cannot silently break either contract.
"""

from __future__ import annotations

import json
import os
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.integrations.raas import RaaSEndpoint
from horizon_ric.security.tenant import TenantScope


def _make_record(decision_id: str, ts: datetime | None = None) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        timestamp=ts or datetime.now(timezone.utc),
        rapp_instance_id="horizon-ric-rapp@x",
        state_hash="deadbeef",
        chosen_action={"policy": "qos", "value": 5},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.1, sla_risk_1min=0.2, sla_risk_5min=0.3
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="e@1",
            risk_heads="r@1",
            dyna="d@1",
            policy="p@1",
            constraint_layer="c@1",
            rapp="rapp@1",
        ),
    )


def test_raas_meter_decision_increments_per_tenant(tmp_path: Path):
    raas = RaaSEndpoint(rapp_id="horizon-ric", export_root=tmp_path)
    assert raas.meter_decision("tenant-a") == 1
    assert raas.meter_decision("tenant-a") == 2
    # Second tenant has its own counter.
    assert raas.meter_decision("tenant-b") == 1
    a = raas.usage("tenant-a")
    b = raas.usage("tenant-b")
    assert a.decisions == 2
    assert b.decisions == 1
    assert a.rapp_id == "horizon-ric"


def test_raas_meter_rejects_empty_tenant(tmp_path: Path):
    raas = RaaSEndpoint(rapp_id="horizon-ric", export_root=tmp_path)
    with pytest.raises(ValueError):
        raas.meter_decision("")


def test_raas_all_usage_returns_every_tenant(tmp_path: Path):
    raas = RaaSEndpoint(rapp_id="horizon-ric", export_root=tmp_path)
    raas.meter_decision("t1")
    raas.meter_decision("t2")
    raas.meter_decision("t2")
    raas.meter_decision("t3")
    rows = {r.tenant_id: r.decisions for r in raas.all_usage()}
    assert rows == {"t1": 1, "t2": 2, "t3": 1}


def test_raas_export_audit_trail_filters_by_tenant(tmp_path: Path):
    # Force HMAC mode so the test does not depend on cosign being
    # installed on the CI runner.
    os.environ["HORIZON_RAAS_FORCE_HMAC"] = "1"
    try:
        store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
        with TenantScope("tenant-a"):
            store.append(_make_record("d-a-1"))
            store.append(_make_record("d-a-2"))
        with TenantScope("tenant-b"):
            store.append(_make_record("d-b-1"))

        raas = RaaSEndpoint(
            rapp_id="horizon-ric",
            evidence_store=store,
            export_root=tmp_path / "exports",
        )
        export = raas.export_audit_trail("tenant-a", since_ts=0.0)

        assert export.tenant_id == "tenant-a"
        assert export.record_count == 2
        assert export.archive_path.exists()
        assert export.signature_path.exists()
        assert export.signature_mode == "hmac-sha256"

        # Open the tarball — must contain audit.jsonl + manifest.json.
        with tarfile.open(export.archive_path, "r:gz") as tar:
            names = sorted(m.name for m in tar.getmembers())
            assert names == ["audit.jsonl", "manifest.json"]
            jsonl = tar.extractfile("audit.jsonl").read().decode("utf-8")
            manifest = json.loads(
                tar.extractfile("manifest.json").read().decode("utf-8")
            )
        # JSONL contains exactly the 2 tenant-a records.
        lines = [l for l in jsonl.splitlines() if l]
        assert len(lines) == 2
        for line in lines:
            row = json.loads(line)
            assert row["tenant_id"] == "tenant-a"
        assert manifest["tenant_id"] == "tenant-a"
        assert manifest["record_count"] == 2
        assert manifest["rapp_id"] == "horizon-ric"
    finally:
        os.environ.pop("HORIZON_RAAS_FORCE_HMAC", None)


def test_raas_export_audit_trail_signature_binds_archive(tmp_path: Path):
    os.environ["HORIZON_RAAS_FORCE_HMAC"] = "1"
    try:
        store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
        with TenantScope("tenant-a"):
            store.append(_make_record("d-1"))

        raas = RaaSEndpoint(
            rapp_id="horizon-ric",
            evidence_store=store,
            export_root=tmp_path / "exports",
        )
        export = raas.export_audit_trail("tenant-a", since_ts=0.0)

        sig = json.loads(export.signature_path.read_text())
        # The signature payload binds rApp id + archive sha256.
        assert sig["mode"] == "hmac-sha256"
        assert sig["rapp_id"] == "horizon-ric"
        import hashlib

        archive_digest = hashlib.sha256(export.archive_path.read_bytes()).hexdigest()
        assert sig["archive_sha256"] == archive_digest
        # And the signature is hex.
        assert len(sig["signature"]) == 64
        int(sig["signature"], 16)  # raises if not hex
    finally:
        os.environ.pop("HORIZON_RAAS_FORCE_HMAC", None)


def test_raas_export_with_no_evidence_store_yields_empty_archive(tmp_path: Path):
    os.environ["HORIZON_RAAS_FORCE_HMAC"] = "1"
    try:
        raas = RaaSEndpoint(
            rapp_id="horizon-ric",
            evidence_store=None,
            export_root=tmp_path / "exports",
        )
        export = raas.export_audit_trail("tenant-z", since_ts=0.0)
        assert export.record_count == 0
        # Archive is still produced and signed.
        assert export.archive_path.exists()
        assert export.signature_path.exists()
    finally:
        os.environ.pop("HORIZON_RAAS_FORCE_HMAC", None)


def test_raas_export_filters_by_since_ts(tmp_path: Path):
    os.environ["HORIZON_RAAS_FORCE_HMAC"] = "1"
    try:
        store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
        old = datetime(2026, 1, 1, tzinfo=timezone.utc)
        new = datetime(2026, 5, 6, tzinfo=timezone.utc)
        with TenantScope("tenant-a"):
            store.append(_make_record("d-old", ts=old))
            store.append(_make_record("d-new", ts=new))

        raas = RaaSEndpoint(
            rapp_id="horizon-ric",
            evidence_store=store,
            export_root=tmp_path / "exports",
        )
        # since_ts mid-year ⇒ only the new record is exported.
        cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()
        export = raas.export_audit_trail("tenant-a", since_ts=cutoff)
        assert export.record_count == 1
        with tarfile.open(export.archive_path, "r:gz") as tar:
            jsonl = tar.extractfile("audit.jsonl").read().decode("utf-8")
        rows = [json.loads(l) for l in jsonl.splitlines() if l]
        assert rows[0]["decision_id"] == "d-new"
    finally:
        os.environ.pop("HORIZON_RAAS_FORCE_HMAC", None)


def test_raas_export_archive_is_deterministic_for_mtime(tmp_path: Path):
    """Tarball entries use mtime=0 so two exports of the same data have
    identical content hashes (modulo the gzip header timestamp)."""
    os.environ["HORIZON_RAAS_FORCE_HMAC"] = "1"
    try:
        store = JsonlEvidenceStore(tmp_path / "ev.jsonl")
        with TenantScope("tenant-a"):
            store.append(
                _make_record(
                    "d-1", ts=datetime(2026, 5, 6, tzinfo=timezone.utc)
                )
            )
        raas = RaaSEndpoint(
            rapp_id="horizon-ric",
            evidence_store=store,
            export_root=tmp_path / "exports",
        )
        export = raas.export_audit_trail("tenant-a", since_ts=0.0)
        with tarfile.open(export.archive_path, "r:gz") as tar:
            for m in tar.getmembers():
                assert m.mtime == 0, (
                    f"tarball entry {m.name!r} has wall-clock mtime — "
                    f"audit exports must be deterministic"
                )
    finally:
        os.environ.pop("HORIZON_RAAS_FORCE_HMAC", None)
