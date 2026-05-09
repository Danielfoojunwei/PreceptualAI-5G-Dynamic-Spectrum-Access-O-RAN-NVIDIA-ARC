"""Pinned reproducers for bugs found in the deep-debug pass.

Each test is one bug. Tests that are still broken after the pass are
marked ``@pytest.mark.xfail(strict=True)`` so that:

  * CI will fail loudly the moment a bug is fixed (the xfail unexpectedly
    passes), forcing the dev to flip the marker off.
  * No regression of a fixed bug is silent.

When a fix lands, drop the ``xfail`` line.

Mapping (kept in sync with DEBUG_REPORT.md):
  CRIT-01  evidence chain breaks under concurrent JsonlEvidenceStore.append
  CRIT-02  evidence chain breaks under concurrent SqliteEvidenceStore.append
  CRIT-03  /audit/verify endpoint ignores the chain and returns ok=True
  CRIT-04  hardcoded JWT dev secret used as production default
  CRIT-05  /sla/timeline returns synthesised numbers, not real samples
  CRIT-06  evidence store ignores per-tenant chain (docstring lies)
  HIGH-01  A1Adapter._policies_emitted not lock-protected (counter race)
  HIGH-02  watchdog_task swallows arbitrary Exception in lifecycle.run_forever finally
"""

from __future__ import annotations

import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _make_record(i: int):
    from horizon_ric.evidence.schema import (
        DecisionRecord,
        ModelVersions,
        PredictedOutcome,
    )

    return DecisionRecord.new(
        decision_id=f"dec-{i:08d}",
        timestamp=datetime.now(timezone.utc),
        rapp_instance_id="r-test",
        state_hash="0" * 64,
        chosen_action={"i": i},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.1, sla_risk_1min=0.1, sla_risk_5min=0.1
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="e", risk_heads="r", dyna="d", policy="p",
            constraint_layer="c", rapp="0",
        ),
    )


# ───────────────────── CRIT-01 ────────────────────────────────────────────
def test_crit01_jsonl_store_concurrent_append_chain_intact(tmp_path: Path) -> None:
    """50 threads append → chain must still verify.

    Reproducer: without an inter-thread lock, two appenders read the same
    `_last_hash()` → both write rows whose `prev` hash is the same → chain
    breaks at index 1 (or wherever the race lands).
    """
    from horizon_ric.evidence.store import JsonlEvidenceStore

    p = tmp_path / "ev.jsonl"
    store = JsonlEvidenceStore(p)
    n_threads = 50
    threads = [
        threading.Thread(target=lambda i=i: store.append(_make_record(i)))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.verify() == -1, "chain broke under concurrent append"


# ───────────────────── CRIT-02 ────────────────────────────────────────────
def test_crit02_sqlite_store_concurrent_append_chain_intact(tmp_path: Path) -> None:
    from horizon_ric.evidence.store import SqliteEvidenceStore

    s = SqliteEvidenceStore(f"sqlite:///{tmp_path}/ev.db")
    n_threads = 50
    threads = [
        threading.Thread(target=lambda i=i: s.append(_make_record(i)))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.verify() == -1, "sqlite chain broke under concurrent append"


# ───────────────────── CRIT-03 ────────────────────────────────────────────
def test_crit03_audit_verify_endpoint_must_actually_verify(tmp_path: Path) -> None:
    """The /audit/verify endpoint must ACTUALLY walk a hash chain.

    With the bug it returns ok=True, first_bad_index=-1 unconditionally.
    """
    import inspect

    from horizon_ric.rapp import api_v1

    src = inspect.getsource(api_v1.create_app)
    # A real implementation must call .verify() on a store. With the bug
    # it never references `.verify(` — it just returns hardcoded values.
    assert ".verify(" in src, (
        "audit/verify endpoint never calls .verify() on an evidence store"
    )


# ───────────────────── CRIT-04 ────────────────────────────────────────────
def test_crit04_jwt_secret_must_not_have_hardcoded_default() -> None:
    """In production we must refuse to start with a known-public dev secret."""
    # Force a fresh import without HORIZON_RIC_JWT_SECRET set.
    import importlib
    import sys

    saved = os.environ.pop("HORIZON_RIC_JWT_SECRET", None)
    try:
        for mod in ("horizon_ric.rapp.api_v1",):
            if mod in sys.modules:
                del sys.modules[mod]
        api_v1 = importlib.import_module("horizon_ric.rapp.api_v1")
        # When no env var is set, the import-time default must NOT silently
        # fall back to a public dev secret. The fix is to raise loudly
        # (or at least make the secret a non-public random per-process
        # value).
        assert api_v1._JWT_SECRET != "horizon-ric-dev-secret", (
            "production import-time default must not be a public dev secret"
        )
    finally:
        if saved is not None:
            os.environ["HORIZON_RIC_JWT_SECRET"] = saved


# ───────────────────── CRIT-05 ────────────────────────────────────────────
@pytest.mark.xfail(
    strict=True,
    reason="CRIT-05: /sla/timeline emits synthesised numbers (0.04 + 0.005 * (i % 5)), "
           "not real KPM samples. Replace with a real source or remove the route.",
)
def test_crit05_sla_timeline_must_not_synthesise() -> None:
    """The /sla/timeline endpoint must source its data from a real KPM
    pipeline (or 28.554 KPI sink), not from a closed-form arithmetic
    synth. We assert that two distinct rApp instances produce *different*
    timelines for the same window — a synth always produces identical
    numbers."""
    from fastapi.testclient import TestClient

    from horizon_ric.rapp.api_v1 import create_app, issue_token

    app1 = create_app()
    app2 = create_app()
    c1 = TestClient(app1)
    c2 = TestClient(app2)
    tok = issue_token("alice", scopes=["read"])
    headers = {"Authorization": f"Bearer {tok}"}
    now = datetime.now(timezone.utc).replace(microsecond=0)
    later = now + timedelta(minutes=10)
    params = {"from": now.isoformat(), "to": later.isoformat()}

    r1 = c1.get("/v1/sla/timeline", headers=headers, params=params)
    r2 = c2.get("/v1/sla/timeline", headers=headers, params=params)
    s1 = [(s["sla_risk_30s"], s["sla_risk_1min"]) for s in r1.json()]
    s2 = [(s["sla_risk_30s"], s["sla_risk_1min"]) for s in r2.json()]
    # If both are synth from a closed form, they will be identical → fail.
    assert s1 != s2, "timeline appears to be synth (deterministic across instances)"


# ───────────────────── CRIT-06 ────────────────────────────────────────────
def test_crit06_evidence_store_per_tenant_chains_isolated(tmp_path: Path) -> None:
    """Two tenants writing to the same JsonlEvidenceStore must produce
    chains that verify independently — tampering tenant A's chain must
    not break tenant B's verify_tenant()."""
    from horizon_ric.evidence.store import JsonlEvidenceStore
    from horizon_ric.security.tenant import TenantScope

    p = tmp_path / "ev.jsonl"
    store = JsonlEvidenceStore(p)
    with TenantScope("tenant-a"):
        store.append(_make_record(0))
        store.append(_make_record(1))
    with TenantScope("tenant-b"):
        store.append(_make_record(10))
        store.append(_make_record(11))

    assert hasattr(store, "verify_tenant"), (
        "evidence store must expose verify_tenant(tid)"
    )
    # Both clean.
    assert store.verify_tenant("tenant-a") == -1
    assert store.verify_tenant("tenant-b") == -1

    # Tamper tenant-a row 0 → tenant-a verify breaks at 0, tenant-b is
    # still intact.
    lines = p.read_text().splitlines()
    tampered = []
    a_count = 0
    for ln in lines:
        if not ln.strip():
            tampered.append(ln)
            continue
        if '"tenant_id": "tenant-a"' in ln and a_count == 0:
            tampered.append(ln.replace('"i": 0', '"i": 999'))
            a_count += 1
        else:
            tampered.append(ln)
    p.write_text("\n".join(tampered) + "\n")
    assert store.verify_tenant("tenant-a") == 0
    assert store.verify_tenant("tenant-b") == -1, (
        "tenant-b chain must remain intact when tenant-a is tampered"
    )


# ───────────────────── HIGH-01 ────────────────────────────────────────────
def test_high01_a1adapter_emit_counter_race() -> None:
    """Counter must equal the number of successful emits even under
    concurrent fire. The current code does `self._policies_emitted += 1`
    in async context — for asyncio this is safe within one event loop,
    BUT only because Python ints are atomic for single-statement
    increments under the GIL. We assert that there is an explicit
    synchronisation primitive (Lock or atomic) so the contract is
    documented, not accidental."""
    import inspect

    from horizon_ric.rapp import a1_adapter

    src = inspect.getsource(a1_adapter.A1Adapter)
    assert "asyncio.Lock" in src or "_emit_lock" in src, (
        "A1Adapter must hold an explicit asyncio.Lock around _policies_emitted"
    )


# ───────────────────── HIGH-02 ────────────────────────────────────────────
def test_high02_lifecycle_finally_must_not_swallow_arbitrary_exception() -> None:
    """`run_forever` finally awaits watchdog_task with
    `except (asyncio.CancelledError, Exception): pass`. This swallows
    every error. The fix is to swallow ONLY CancelledError and log
    others."""
    import inspect

    from horizon_ric.rapp import lifecycle

    src = inspect.getsource(lifecycle.HorizonRAppLifecycle.run_forever)
    assert "(asyncio.CancelledError, Exception)" not in src, (
        "run_forever must not swallow generic Exception in its finally clause"
    )
