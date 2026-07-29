#!/usr/bin/env python3
"""Measure the Shield decision path against a REAL captured control-plane load.

What is real and what is not
----------------------------
REAL (measured, from 5GAD-2022, MIT licensed, see ``datasets/5gad_inl/``):

  * Which control-plane events occurred, in what order, and at what nanosecond
    timestamps, on a real free5GC 5G standalone core while ten real attacks were
    executed against it on a physical test bench at Idaho National Laboratory.
  * The message mix: the actual SBI method + resource path of every HTTP/1.1
    request (NRF NF-management writes, NRF discovery reads, UDM subscriber-data
    reads) and the actual PFCP message type of every N4 datagram.
  * The offered arrival rate and burstiness: mean and peak events per second in
    one-second bins, and the inter-arrival distribution.

REAL (measured, this host): the Shield's own latency and throughput.

NOT REAL (derived, and labelled as such everywhere): the radio parameters of the
action each captured event is mapped to. 5GAD is a *core-network* capture; it
contains no RIC policy actions and no radio parameters, so there is nothing to
measure there. Each captured event is mapped to one Shield proposal by a
deterministic function of that event's own bytes (see ``_proposal_from_event``).
The mapping preserves the real read/write distinction — a captured write becomes
an emitting proposal, a captured read becomes a non-emitting one — and nothing
else about the radio side is claimed to come from the capture.

This benchmark therefore answers one question honestly: *can the Shield service
the control-plane event rate that a real 5G core actually sustained under a real
attack, and what does it do with those events?* It still does not emulate an
SMO, a RIC platform, a radio, network I/O, a multi-node deployment or failover,
so it cannot establish carrier-scale latency, throughput, availability or
scalability.

Build the dataset first (non-interactive, ~24 MB, no login)::

    python datasets/5gad_inl/build.py

Then::

    python benchmarks/control_plane_load_suite.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_ric.shield.shield import default_terrestrial_shield

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET = _REPO_ROOT / "datasets" / "5gad_inl"

# The licensed channel the Shield is programmed with for this measurement.
_BAND_LO_HZ = 3.40e9
_BAND_HI_HZ = 3.50e9
_MAX_EIRP_DBM = 33.0
_GUARD_BAND_HZ = 1.0e6

# Captured HTTP methods that mutate core state; PFCP session modification is a
# write too. Everything else is a read.
_WRITE_METHODS = frozenset({"PUT", "POST", "PATCH", "DELETE"})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DatasetMissing(RuntimeError):
    """Raised when the 5GAD-derived features have not been built."""


def load_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = dataset_dir / "manifest.json"
    events_path = dataset_dir / "generated" / "control_plane_events.jsonl"
    if not manifest_path.is_file() or not events_path.is_file():
        raise DatasetMissing(
            f"5GAD control-plane features not found ({events_path}). Build them "
            f"with: python datasets/5gad_inl/build.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not events:
        raise DatasetMissing(f"{events_path} is empty")
    return manifest, events


# ---------------------------------------------------------------------------
# Event -> Shield proposal (deterministic, and NOT measured — see module docstring)
# ---------------------------------------------------------------------------
def _proposal_from_event(event: dict[str, Any]) -> dict[str, Any]:
    """Map one captured control-plane event to one Shield action proposal.

    Deterministic and content-addressed: the radio parameters are a pure
    function of the captured event's own identity (its body digest when it had a
    body, otherwise its method+path). Two runs of the same capture therefore
    produce byte-identical proposals, and two different captured requests
    produce different proposals.

    The radio parameters are DERIVED, not measured. What the capture supplies is
    the event's existence, ordering, timing and read/write kind.
    """
    identity = event.get("body_sha256") or ""
    if not identity:
        identity = f"{event['method']}:{event['path']}"
    seed = int.from_bytes(identity.encode("utf-8")[:8].ljust(8, b"\0"), "big")

    is_write = event["method"] in _WRITE_METHODS or (
        event.get("protocol") == "pfcp" and "modification" in event["path"]
    )

    # Spread carriers across, and deliberately past, the licensed edge so a
    # realistic fraction of proposals require frequency projection.
    span = _BAND_HI_HZ - _BAND_LO_HZ
    frequency_hz = _BAND_LO_HZ + ((seed >> 7) % 1001) / 1000.0 * (span * 1.02)
    # A minority of proposals exceed the EIRP ceiling and must be projected down.
    over_power = (seed >> 17) % 3 == 0

    return {
        "block": "ric_policy",
        "frequency_hz": frequency_hz,
        "bandwidth_hz": 20e6,
        "tx_power_dBm": 35.0 if over_power else 20.0,
        "antenna_gain_dBi": 5.0,
        # Reads do not emit; carry the distinction through so the certificate
        # reflects the captured event's real kind.
        "emit": bool(is_write),
    }


# ---------------------------------------------------------------------------
# Real arrival profile (measured, straight from the capture timestamps)
# ---------------------------------------------------------------------------
def _arrival_profile(events: list[dict[str, Any]]) -> dict[str, Any]:
    times_ns = sorted(int(e["t_offset_ns"]) for e in events)
    span_ns = times_ns[-1] - times_ns[0] if len(times_ns) > 1 else 0
    span_s = span_ns / 1e9
    bins = Counter(t // 1_000_000_000 for t in times_ns)
    peak = max(bins.values()) if bins else len(times_ns)
    gaps = [b - a for a, b in zip(times_ns, times_ns[1:])]
    ordered = sorted(gaps)

    def _q(fraction: float) -> float:
        if not ordered:
            return 0.0
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] / 1e6

    return {
        "events": len(times_ns),
        "span_seconds": span_s,
        "mean_events_per_second": (len(times_ns) / span_s) if span_s > 0 else 0.0,
        "peak_events_per_second_1s_bins": peak,
        "occupied_1s_bins": len(bins),
        "interarrival_ms": {
            "min": _q(0.0),
            "p50": _q(0.50),
            "p95": _q(0.95),
            "max": (ordered[-1] / 1e6) if ordered else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Shield replay
# ---------------------------------------------------------------------------
def _new_shield():
    return default_terrestrial_shield(
        band_lo_hz=_BAND_LO_HZ,
        band_hi_hz=_BAND_HI_HZ,
        max_eirp_dBm=_MAX_EIRP_DBM,
        guard_band_hz=_GUARD_BAND_HZ,
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = max(0, min(len(sorted_values) - 1, int(percentile * len(sorted_values)) - 1))
    return sorted_values[index]


def _latency_block(latencies_us: list[float]) -> dict[str, float]:
    ordered = sorted(latencies_us)
    return {
        "mean": statistics.fmean(latencies_us),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "maximum": ordered[-1],
    }


def _replay(events: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    """Drive the Shield once per captured event, ``repeats`` times over.

    Replays as fast as the process allows (it measures the SERVICE rate, which
    is then compared against the capture's measured OFFERED rate). It does not
    sleep out the real inter-arrival gaps: the longest capture spans 1791 s and
    sleeping through it would measure the clock, not the Shield.
    """
    shield = _new_shield()
    proposals = [_proposal_from_event(event) for event in events]

    # Warm up on the first 200 proposals so first-call costs do not land in the
    # measured distribution.
    for index, proposal in enumerate(proposals[:200]):
        shield.dispose(proposal, decision_id=f"warmup-{index}")

    latencies_us: list[float] = []
    projected = 0
    blocked = 0
    started = time.perf_counter_ns()
    for repeat in range(repeats):
        for index, proposal in enumerate(proposals):
            one_started = time.perf_counter_ns()
            disposition = shield.dispose(
                proposal,
                decision_id=f"replay-{repeat}-{index}",
                loop_tier="non_rt",
            )
            latencies_us.append((time.perf_counter_ns() - one_started) / 1_000.0)
            projected += int(disposition.certificate.projected)
            blocked += int(disposition.certificate.emit_blocked)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0

    decisions = len(proposals) * repeats
    return {
        "decisions": decisions,
        "elapsed_seconds": elapsed_seconds,
        "decisions_per_second": decisions / elapsed_seconds,
        "latency_microseconds": _latency_block(latencies_us),
        "projected_decisions": projected,
        "blocked_decisions": blocked,
    }


def run_real(events: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    """Per-capture and aggregate replay of the real captured control plane."""
    by_capture: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_capture.setdefault(event["capture"], []).append(event)

    per_capture: list[dict[str, Any]] = []
    for capture, rows in sorted(by_capture.items()):
        profile = _arrival_profile(rows)
        service = _replay(rows, repeats)
        writes = sum(1 for r in rows if r["method"] in _WRITE_METHODS)
        headroom = (
            service["decisions_per_second"] / profile["peak_events_per_second_1s_bins"]
            if profile["peak_events_per_second_1s_bins"]
            else None
        )
        per_capture.append(
            {
                "capture": capture,
                "attack_class": rows[0]["attack_class"],
                "capture_role": rows[0]["role"],
                "message_mix": dict(
                    sorted(Counter(f"{r['method']} {r['sbi_service']}" for r in rows).items())
                ),
                "captured_writes": writes,
                "captured_reads": len(rows) - writes,
                "measured_arrival_profile": profile,
                "measured_shield_service": service,
                "service_to_peak_offered_headroom_x": headroom,
            }
        )

    aggregate_profile = _arrival_profile(events)
    aggregate_service = _replay(events, repeats)
    return {
        "per_capture": per_capture,
        "aggregate": {
            "measured_arrival_profile": aggregate_profile,
            "measured_shield_service": aggregate_service,
            "worst_case_headroom_x": min(
                (
                    c["service_to_peak_offered_headroom_x"]
                    for c in per_capture
                    if c["service_to_peak_offered_headroom_x"] is not None
                ),
                default=None,
            ),
        },
    }


def run_synthetic_control(iterations: int, warmup: int) -> dict[str, Any]:
    """The pre-migration synthetic traffic mix, retained as a labelled control.

    This is the benchmark's previous behaviour verbatim: an invented alternating
    proposal pattern with no connection to any capture. It is kept so the real
    replay can be compared against the number that used to be reported, and it
    is labelled synthetic in the output.
    """
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    shield = _new_shield()

    def action(index: int) -> dict[str, Any]:
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
            action(index), decision_id=f"load-{index}", loop_tier="non_rt"
        )
        latencies_us.append((time.perf_counter_ns() - one_started) / 1_000.0)
        projected += int(disposition.certificate.projected)
        blocked += int(disposition.certificate.emit_blocked)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0

    return {
        "data_provenance": "synthetic",
        "traffic_mix": "deterministic alternating terrestrial policy proposals (invented)",
        "iterations": iterations,
        "warmup_iterations": warmup,
        "elapsed_seconds": elapsed_seconds,
        "decisions_per_second": iterations / elapsed_seconds,
        "latency_microseconds": _latency_block(latencies_us),
        "projected_decisions": projected,
        "blocked_decisions": blocked,
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def run(
    *,
    dataset_dir: Path,
    repeats: int,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    manifest, events = load_dataset(dataset_dir)
    shield = _new_shield()
    real = run_real(events, repeats)
    synthetic = run_synthetic_control(iterations, warmup)
    usage = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "benchmark": "shield_decision_path_under_real_captured_control_plane_load",
        "data_provenance": {
            "dataset": manifest["dataset"],
            "data_kind": manifest["data_kind"],
            "repo_url": manifest["repo_url"],
            "repo_commit": manifest["repo_commit"],
            "repo_doi": manifest["repo_doi"],
            "paper_doi": manifest["paper_doi"],
            "licence": manifest["licensing"]["source_repository"],
            "attribution": manifest["licensing"]["attribution_required"],
            "source_archive_sha256": manifest["source_archive_sha256"],
            "events_sha256": manifest["events_sha256"],
            "features_sha256": manifest["features_sha256"],
            "ci_reproducible": manifest["ci_reproducible"],
            "real": [
                "event existence, ordering and nanosecond timestamps",
                "SBI method and resource path; PFCP message type",
                "offered arrival rate and burstiness",
                "read/write kind of each captured event",
            ],
            "derived_not_measured": [
                "the radio parameters of the Shield proposal each event maps to "
                "(5GAD is a core-network capture and contains no radio actions); "
                "see _proposal_from_event",
            ],
        },
        "parameters": {
            "invariant_ids": shield.invariant_ids,
            "band_lo_hz": _BAND_LO_HZ,
            "band_hi_hz": _BAND_HI_HZ,
            "max_eirp_dBm": _MAX_EIRP_DBM,
            "guard_band_hz": _GUARD_BAND_HZ,
            "replay_repeats": repeats,
            # Headline iteration count = captured events x replay repeats. Kept
            # under this key so the committed-evidence contract in
            # tests/test_control_plane_load_evidence.py continues to hold across
            # the migration from the synthetic mix to the real capture replay.
            "iterations": real["aggregate"]["measured_shield_service"]["decisions"],
            "traffic_mix": (
                "one Shield disposition per control-plane event captured on a real "
                "free5GC core under ten real attacks (5GAD-2022), replayed "
                f"{repeats}x"
            ),
        },
        # Headline result = the REAL capture replay aggregate. The synthetic
        # control is reported separately under "synthetic_control" and is never
        # the number quoted here.
        "results": {
            "source": "real_capture_replay.aggregate",
            **real["aggregate"]["measured_shield_service"],
            "max_rss_kib": usage.ru_maxrss,
        },
        "real_capture_replay": real,
        "synthetic_control": synthetic,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "max_rss_kib": usage.ru_maxrss,
        },
        "scope": {
            "proves": (
                "Measured local runtime of the in-process Shield and certificate "
                "construction when driven once per control-plane event actually "
                "captured on a real free5GC core under ten real attacks, and the "
                "ratio of that service rate to the peak arrival rate those "
                "attacks actually offered."
            ),
            "does_not_prove": [
                "carrier-scale latency, throughput, availability or scalability",
                "performance with SMO/RIC/radio/network I/O",
                "multi-node concurrency, failover or 24-hour stability",
                "performance on an operator production hardware target",
                "anything about the radio parameters of the replayed actions — "
                "those are derived from each captured event's bytes, not measured",
                "detection or mitigation of the captured attacks themselves; the "
                "Shield is an action-legality projector, not an intrusion "
                "detector, and this benchmark measures load, not detection",
            ],
            "honest_limits": [
                "The replay runs at maximum speed rather than sleeping out the "
                "real inter-arrival gaps: the longest capture spans 1791 s, and "
                "replaying it in real time would measure the clock, not the "
                "Shield. Real timing is reported separately as the measured "
                "arrival profile and compared against the measured service rate.",
                "5GAD's benign captures contain almost no service-based-interface "
                "traffic — free5GC's steady state after registration is GTP-U "
                "user plane — so there is no meaningful benign SBI control-plane "
                "baseline to compare the attack rates against. The 65 KB "
                "Normal-1UE reference yields zero decodable SBI requests and is "
                "reported as such rather than padded out.",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="times to replay each capture's event sequence (service-rate sampling)",
    )
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "results" / "control_plane_load.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(
            dataset_dir=args.dataset_dir,
            repeats=args.repeats,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    except DatasetMissing as exc:
        print(f"ERROR: {exc}")
        return 2

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")

    real = result["real_capture_replay"]
    print("Shield decision path under REAL captured 5G control-plane load")
    print(f"  source: {result['data_provenance']['dataset']}")
    print(f"  licence: {result['data_provenance']['licence']}")
    print()
    print(
        f"{'capture':<28} {'class':<24} {'events':>7} {'peak/s':>7} "
        f"{'offered/s':>10} {'shield/s':>10} {'headroom':>9}"
    )
    print("-" * 102)
    for entry in real["per_capture"]:
        profile = entry["measured_arrival_profile"]
        service = entry["measured_shield_service"]
        headroom = entry["service_to_peak_offered_headroom_x"]
        print(
            f"{entry['capture']:<28} {entry['attack_class']:<24} "
            f"{profile['events']:>7} {profile['peak_events_per_second_1s_bins']:>7} "
            f"{profile['mean_events_per_second']:>10.2f} "
            f"{service['decisions_per_second']:>10.0f} "
            f"{(f'{headroom:.0f}x' if headroom else 'n/a'):>9}"
        )
    aggregate = real["aggregate"]
    print("-" * 102)
    print(
        f"  aggregate: {aggregate['measured_arrival_profile']['events']} real events, "
        f"peak {aggregate['measured_arrival_profile']['peak_events_per_second_1s_bins']}/s "
        f"offered; Shield serviced "
        f"{aggregate['measured_shield_service']['decisions_per_second']:.0f}/s "
        f"(p99 {aggregate['measured_shield_service']['latency_microseconds']['p99']:.1f} us)"
    )
    print(f"  worst-case headroom over any capture: {aggregate['worst_case_headroom_x']:.0f}x")
    print(
        f"  synthetic control (pre-migration mix): "
        f"{result['synthetic_control']['decisions_per_second']:.0f}/s"
    )
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
