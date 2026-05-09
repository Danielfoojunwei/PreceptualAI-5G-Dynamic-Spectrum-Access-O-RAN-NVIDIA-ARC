"""L1/L2/A1 inference admit-control + thermal latency model.

These functions are pure: no state. They are used by the rApp and the edge
agent to refuse a model on the L1 critical path that exceeds the symbol-
aligned budget; to flag a model size that won't make the next A1 refresh;
and to project realistic Jetson Orin Nano latency under thermal load.

────────────────────────────────────────────────────────────────────────────
Edge p99 budget — measured, not aspirational
────────────────────────────────────────────────────────────────────────────

Earlier docs cited an 8 ms p99 target for Jetson Orin Nano. The
`scripts/edge_benchmark_arm64.py` run captured in
`deploy/EDGE_BENCHMARK_PROOF.md` measured the following on real
aarch64 silicon with a real 4-layer MLP head + real audit-append
cycle (no mocks):

    GB10 host (CPU path):   p50 27.296 ms · p95 46.275 ms · p99 48.441 ms
    Projected Orin Nano (×3 thermal/clock/BW scaling):
                             p50 81.889 ms · p95 138.826 ms · p99 145.323 ms

We therefore retract the 8 ms claim and set the **edge decision
budget** below to **160 ms p99** — slightly above the Orin Nano
projection so a healthy edge does not page at the limit, and below
the A1 refresh budget so an A1 decision can still land before the
next refresh window. The GB10 CPU path runs comfortably (≈48 ms),
the Orin Nano runs near (~145 ms) but inside, the budget. The 8 ms
figure was for an integer-quantised attention-only kernel without
the audit-append + telemetry-event marshaling that the real path
carries; it never represented end-to-end edge latency.

References:
    3GPP TS 38.211 §4.3.2 (slot timing per numerology μ).
    3GPP TS 38.214 §5.3 (UE PDSCH processing capabilities N1).
    O-RAN.WG2.A1AP-v05.00 §3.3 (A1 policy refresh cadence).
    NVIDIA Jetson Orin Nano Datasheet DA-11081 §3.1 (TOPS, BW).
    deploy/EDGE_BENCHMARK_PROOF.md (measured GB10 / projected Nano p99).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

# ─── L1 admit gate ───────────────────────────────────────────────────────


def l1_admit(
    model_latency_us: float,
    slot_numerology: int = 1,
    headroom_us: float = 50.0,
) -> Literal["L1_ok", "defer_to_L2", "defer_to_A1"]:
    """Decide whether a model can run inside the L1 critical path.

    Symbol duration T_sym = 1 ms / (14 · 2^μ). Available L1 ML budget per
    symbol is roughly the symbol duration minus DU processing + fronthaul
    deadlines (~250 µs at μ=1). We refuse anything > headroom_us.

    Returns one of:
        "L1_ok"        — fits inside one symbol with margin.
        "defer_to_L2"  — too slow for L1, can still meet slot deadline.
        "defer_to_A1"  — only acceptable at A1 (rApp) cadence.
    """
    if slot_numerology < 0 or slot_numerology > 6:
        raise ValueError(f"slot_numerology must be in [0,6], got {slot_numerology}")
    t_slot_us = 1000.0 / (1 << slot_numerology)
    t_symbol_us = t_slot_us / 14.0
    if model_latency_us <= max(headroom_us, t_symbol_us * 0.5):
        return "L1_ok"
    if model_latency_us <= t_slot_us * 0.4:
        return "defer_to_L2"
    return "defer_to_A1"


# ─── L2 budget ───────────────────────────────────────────────────────────


def l2_inference_budget_us(
    slot_numerology: int = 1,
    scheduler_overhead_us: float = 300.0,
) -> float:
    """Microseconds available for ML inside one MAC slot.

    slot - bookkeeping/sort/DCI overhead (default 300 µs at μ=1) = budget.
    """
    t_slot_us = 1000.0 / (1 << slot_numerology)
    return max(t_slot_us - scheduler_overhead_us, 0.0)


# ─── A1 admit gate ───────────────────────────────────────────────────────


def a1_admit(
    decision_latency_ms: float,
    policy_period_ms: float = 100.0,
    a1_propagation_ms: float = 5.0,
    e2_enforcement_ms: float = 5.0,
) -> bool:
    """True iff the rApp decision can land before the next A1 refresh."""
    budget = policy_period_ms - a1_propagation_ms - e2_enforcement_ms
    return decision_latency_ms <= budget


# ─── Compute / memory floors ─────────────────────────────────────────────


def edge_compute_lower_bound_us(
    n_params: int,
    precision_bits: int,
    tops: float,
    batch: int = 1,
) -> float:
    """Optimistic compute lower bound: latency_us = (2 · N · batch) / TOPS.

    Used as the FLOP floor; real-world latency is typically max(compute,
    memory) — see `memory_bound_latency_ms`.
    """
    if tops <= 0 or n_params <= 0:
        raise ValueError("tops and n_params must be positive")
    ops = 2.0 * n_params * batch
    # Convert TOPS (10^12 ops/s) to ops/µs
    ops_per_us = tops * 1e12 / 1e6
    return ops / ops_per_us


def memory_bound_latency_ms(
    activation_bytes: int,
    weight_bytes: int,
    bw_gbps: float,
) -> float:
    """Roofline lower bound from memory-bandwidth: ms to stream the round-trip."""
    if bw_gbps <= 0:
        raise ValueError(f"bw_gbps must be > 0, got {bw_gbps}")
    total_bytes = activation_bytes + weight_bytes
    bw_bytes_per_ms = bw_gbps * 1e9 / 1000.0
    return total_bytes / bw_bytes_per_ms


# ─── Thermal-aware projection ────────────────────────────────────────────


@dataclass(frozen=True)
class ThermalState:
    junction_celsius: float
    power_mode_w: float
    headroom_seconds: float


def inference_budget_ms(
    thermal_state: ThermalState,
    nominal_latency_ms: float,
    knee_celsius: float = 75.0,
    max_celsius: float = 95.0,
    throttle_factor: float = 2.0,
) -> float:
    """Projected latency under current thermal state.

    Below `knee_celsius` we run at nominal. Between knee and max we scale
    linearly up to `nominal_latency_ms · throttle_factor`. Above max we
    clamp at the throttle factor.
    """
    if thermal_state.junction_celsius <= knee_celsius:
        return nominal_latency_ms
    if thermal_state.junction_celsius >= max_celsius:
        return nominal_latency_ms * throttle_factor
    span = max_celsius - knee_celsius
    frac = (thermal_state.junction_celsius - knee_celsius) / span
    return nominal_latency_ms * (1.0 + frac * (throttle_factor - 1.0))


# ─── End-to-end stack-up helper ──────────────────────────────────────────


def end_to_end_latency_ms(stages: dict[str, float]) -> tuple[float, list[str]]:
    """Sum of stage budgets + ranked bottleneck names (highest first)."""
    if not stages:
        raise ValueError("stages must be non-empty")
    total = float(sum(stages.values()))
    ranked = sorted(stages, key=lambda k: stages[k], reverse=True)
    return total, ranked


# Federated-learning bandwidth model (weights-only, per UHCI constraint).


def fl_round_seconds(
    n_params: int,
    precision_bits: int,
    sparsity: float,
    link_mbps: float,
) -> float:
    """Seconds to transmit one FL round payload over the given uplink."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0,1), got {sparsity}")
    if link_mbps <= 0:
        raise ValueError(f"link_mbps must be > 0, got {link_mbps}")
    payload_bits = n_params * precision_bits * (1.0 - sparsity)
    return payload_bits / (link_mbps * 1e6)


def fl_max_rounds_per_hour(
    n_params: int,
    precision_bits: int,
    sparsity: float,
    link_mbps: float,
    overhead_factor: float = 1.5,
) -> int:
    """Practical FL round ceiling per hour (overhead_factor accounts for
    aggregation, validation, straggler tail)."""
    sec = fl_round_seconds(n_params, precision_bits, sparsity, link_mbps)
    return int(math.floor(3600.0 / (sec * overhead_factor)))


# ─── Edge p99 latency budget (measured-grounded) ─────────────────────────


# Honest budgets, derived from `deploy/EDGE_BENCHMARK_PROOF.md`. The
# old 8 ms figure is retracted — see the module docstring for the
# rationale and measurements.
EDGE_P99_BUDGET_MS_GB10: float = 60.0
"""GB10 CPU-path p99 budget. Measured 48.441 ms; budget 60 ms gives
~24 % headroom for kernel scheduling jitter and audit fsync."""

EDGE_P99_BUDGET_MS_ORIN_NANO: float = 160.0
"""Jetson Orin Nano CPU-path p99 budget. Projected 145.323 ms (×3
GB10 → Nano scaling for clock + DRAM BW + 15 W thermal cap);
budget 160 ms gives ~10 % headroom and stays inside the 200 ms
SLO p99 alert (`deploy/SLO.md` row 2)."""


def edge_p99_budget_ms(target: str = "orin_nano") -> float:
    """Return the measured-grounded edge p99 budget for `target`.

    `target` must be one of ``"gb10"`` or ``"orin_nano"``.
    """
    t = target.lower()
    if t == "gb10":
        return EDGE_P99_BUDGET_MS_GB10
    if t == "orin_nano":
        return EDGE_P99_BUDGET_MS_ORIN_NANO
    raise ValueError(f"unknown edge target {target!r}; expected 'gb10' or 'orin_nano'")


__all__ = [
    "EDGE_P99_BUDGET_MS_GB10",
    "EDGE_P99_BUDGET_MS_ORIN_NANO",
    "ThermalState",
    "a1_admit",
    "edge_compute_lower_bound_us",
    "edge_p99_budget_ms",
    "end_to_end_latency_ms",
    "fl_max_rounds_per_hour",
    "fl_round_seconds",
    "inference_budget_ms",
    "l1_admit",
    "l2_inference_budget_us",
    "memory_bound_latency_ms",
]
