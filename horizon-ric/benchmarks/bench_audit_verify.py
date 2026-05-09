"""Benchmark the SHA-256 chained audit-verify operation on N records.

Runs append + verify across the JsonlEvidenceStore. Intended to support the
BENCHMARK_HEAD_TO_HEAD claim that audit-chain integrity verification is
sub-second on commodity hardware.

Usage:
    .venv/bin/python benchmarks/bench_audit_verify.py
"""
from __future__ import annotations

import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore


def _make_record(i: int) -> DecisionRecord:
    return DecisionRecord.new(
        decision_id=f"dec-{i:08d}",
        timestamp=datetime.now(timezone.utc),
        rapp_instance_id="bench-rapp",
        state_hash="0" * 64,
        chosen_action={"action_type": "noop", "i": i},
        predicted_outcome_chosen=PredictedOutcome(
            sla_risk_30s=0.1, sla_risk_1min=0.1, sla_risk_5min=0.1
        ),
        rejected_alternatives=[],
        model_versions=ModelVersions(
            encoder="e", risk_heads="r", dyna="d",
            policy="p", constraint_layer="c", rapp="0",
        ),
    )


def bench(n_records: int, n_iters: int = 5, n_warmup: int = 2) -> dict:
    append_ms: list[float] = []
    verify_ms: list[float] = []

    for i in range(n_warmup + n_iters):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ev.jsonl"
            store = JsonlEvidenceStore(p)

            t0 = time.perf_counter()
            for k in range(n_records):
                store.append(_make_record(k))
            t1 = time.perf_counter()

            t2 = time.perf_counter()
            result = store.verify()
            t3 = time.perf_counter()

            assert result == -1, f"chain broken at {result}"
            if i >= n_warmup:
                append_ms.append((t1 - t0) * 1000.0)
                verify_ms.append((t3 - t2) * 1000.0)

    append_ms.sort()
    verify_ms.sort()
    p95_idx = max(0, int(len(append_ms) * 0.95) - 1)
    return {
        "n_records": n_records,
        "n_iters": len(append_ms),
        "append_median_ms": statistics.median(append_ms),
        "append_p95_ms": append_ms[p95_idx],
        "verify_median_ms": statistics.median(verify_ms),
        "verify_p95_ms": verify_ms[p95_idx],
        "verify_per_record_us": (statistics.median(verify_ms) * 1000.0) / n_records,
    }


if __name__ == "__main__":
    for n in (100, 1000):
        r = bench(n_records=n, n_iters=3, n_warmup=1)
        print(
            f"N={r['n_records']:>6d}  "
            f"append median={r['append_median_ms']:>8.2f} ms (p95 {r['append_p95_ms']:>8.2f})  "
            f"verify median={r['verify_median_ms']:>8.2f} ms (p95 {r['verify_p95_ms']:>8.2f})  "
            f"per-record {r['verify_per_record_us']:>6.2f} µs"
        )
