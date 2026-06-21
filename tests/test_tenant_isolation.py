"""Tenant isolation tests for the evidence store.

Proves that:
  * Each tenant has its own SHA-256 hash chain (writes from tenant A
    don't shift tenant B's chain).
  * Iterating inside a `TenantScope` only yields that tenant's records.
  * Concurrent writes from different tenants don't interleave chains.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import (
    JsonlEvidenceStore,
    SqliteEvidenceStore,
)
from horizon_ric.security.tenant import TenantScope


def _make_record(decision_id: str) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        timestamp=datetime.now(timezone.utc),
        rapp_instance_id="rapp-1",
        state_hash="0" * 64,
        chosen_action={"a": 1},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.1, sla_risk_1min=0.1, sla_risk_5min=0.1
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="v", risk_heads="v", dyna="v",
            policy="v", constraint_layer="v", rapp="v",
        ),
    )


def test_jsonl_per_tenant_chains(tmp_path: Path) -> None:
    store = JsonlEvidenceStore(tmp_path / "ev.jsonl")

    with TenantScope("tenant_a"):
        store.append(_make_record("a-1"))
        store.append(_make_record("a-2"))
    with TenantScope("tenant_b"):
        store.append(_make_record("b-1"))
        store.append(_make_record("b-2"))
    with TenantScope("tenant_a"):
        store.append(_make_record("a-3"))

    # Iterating with no scope yields all 5 records, but verify() must
    # walk per-tenant chains.
    assert store.verify() == -1

    # An auditor scoped to tenant_a sees only their 3 records.
    with TenantScope("tenant_a"):
        ids = [r.decision_id for r, _ in store]
    assert ids == ["a-1", "a-2", "a-3"]

    # Tenant B can't see tenant A's records.
    with TenantScope("tenant_b"):
        ids_b = [r.decision_id for r, _ in store]
    assert ids_b == ["b-1", "b-2"]


def test_sqlite_per_tenant_chains(tmp_path: Path) -> None:
    db = f"sqlite:///{tmp_path / 'ev.db'}"
    store = SqliteEvidenceStore(db)

    with TenantScope("tenant_a"):
        store.append(_make_record("a-1"))
    with TenantScope("tenant_b"):
        store.append(_make_record("b-1"))
    with TenantScope("tenant_a"):
        store.append(_make_record("a-2"))

    assert store.verify() == -1

    with TenantScope("tenant_a"):
        recs = [r.decision_id for r, _ in store]
    assert recs == ["a-1", "a-2"]
    with TenantScope("tenant_b"):
        recs = [r.decision_id for r, _ in store]
    assert recs == ["b-1"]


@pytest.mark.asyncio
async def test_concurrent_tenant_writes_dont_interleave(tmp_path: Path) -> None:
    store = JsonlEvidenceStore(tmp_path / "ev.jsonl")

    async def writer(tenant: str, n: int) -> None:
        for i in range(n):
            with TenantScope(tenant):
                store.append(_make_record(f"{tenant}-{i}"))
            await asyncio.sleep(0)

    await asyncio.gather(writer("tenant_a", 10), writer("tenant_b", 10))

    # Both chains should remain valid.
    assert store.verify() == -1

    with TenantScope("tenant_a"):
        a_ids = [r.decision_id for r, _ in store]
    with TenantScope("tenant_b"):
        b_ids = [r.decision_id for r, _ in store]

    # Each tenant sees exactly its own ordered records, no leakage.
    assert all(x.startswith("tenant_a-") for x in a_ids)
    assert all(x.startswith("tenant_b-") for x in b_ids)
    assert len(a_ids) == 10
    assert len(b_ids) == 10
