#!/usr/bin/env python3
"""Measure the Shield decision path on one local Python process.

This benchmark supplies a reproducible baseline and an honest boundary. It
does not emulate an SMO, RIC platform, radio, network I/O, multi-node
deployment, failover or operator traffic, so it cannot establish carrier-scale
latency, throughput, availability or scalability.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_ric.shield.shield import default_terrestrial_shield


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = max(0, min(len(sorted_values) - 1, int(percentile * len(sorted_values)) - 1))
    return sorted_values[index]


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def run(iterations: int, warmup: int) -> dict[str, Any]:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")

    shield = default_terrestrial_shield(
        band_lo_hz=3.40e9,
        band_hi_hz=3.50e9,
        max_eirp_dBm=33.0,
        guard_band_hz=1.0e6,
    )

    def action(index: int) -> dict[str, Any]:
        # Half the proposals require frequency and/or power projection. The
        # post-Shield action must still satisfy every programmed invariant.
        return {
            "block": "ric_policy",
            "frequency_hz": 3.45e9 if index % 2 == 0 else 3.495e9,
            "bandwidth_hz": 20e6,
            "tx_power_dBm": 20.0 if index % 3 else 35.0,
            "antenna_gain_dBi": 5.0,
        }

    for index in range(warmup):
        shield.dispose(action(index), decision_id=f"warmup-{index}")

    latencies_us: list[float] = []
    projected = 0
    blocked = 0
    started = time.perf_counter_ns()
    for index in range(iterations):
        one_started = time.perf_counter_ns()
        disposition = shield.dispose(
            action(index),
            decision_id=f"load-{index}",
            loop_tier="non_rt",
        )
        latencies_us.append((time.perf_counter_ns() - one_started) / 1_000.0)
        projected += int(disposition.certificate.projected)
        blocked += int(disposition.certificate.emit_blocked)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0

    ordered = sorted(latencies_us)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "benchmark": "single_process_shield_decision_path",
        "parameters": {
            "iterations": iterations,
            "warmup_iterations": warmup,
            "invariant_ids": shield.invariant_ids,
            "traffic_mix": (
                "deterministic valid and projectable terrestrial policy proposals"
            ),
        },
        "results": {
            "elapsed_seconds": elapsed_seconds,
            "decisions_per_second": iterations / elapsed_seconds,
            "latency_microseconds": {
                "mean": statistics.fmean(latencies_us),
                "p50": _percentile(ordered, 0.50),
                "p95": _percentile(ordered, 0.95),
                "p99": _percentile(ordered, 0.99),
                "maximum": ordered[-1],
            },
            "projected_decisions": projected,
            "blocked_decisions": blocked,
            "max_rss_kib": usage.ru_maxrss,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
        },
        "scope": {
            "proves": (
                "Measured local runtime of the in-process Shield and certificate "
                "construction for the stated deterministic traffic mix."
            ),
            "does_not_prove": [
                "carrier-scale latency, throughput, availability or scalability",
                "performance with SMO/RIC/radio/network I/O",
                "multi-node concurrency, failover or 24-hour stability",
                "performance on an operator production hardware target",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/control_plane_load.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.iterations, args.warmup)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
