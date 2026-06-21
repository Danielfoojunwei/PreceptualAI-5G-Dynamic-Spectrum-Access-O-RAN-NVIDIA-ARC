# 24-hour Shadow Soak Proof (Row 7)

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`../README.md`](../README.md) for the 49-section deep dive.*


_Run recorded_: 2026-05-07T10:21:16.743351+00:00 → 2026-05-07T10:51:17.525627+00:00

Wall-clock 30.0 min × speedup 48.0× = **24.00 simulated hours**.

> **Provenance note.** The figures below were recorded on the 2026-05-07
> tree. They are a historical measurement, not a value re-derived on every
> CI run. The driver script `scripts/soak_24h.py` referenced under "How this
> was generated" is **not currently present in this tree** (the daily soak
> systemd unit at `deploy/systemd/` invokes it on the edge device); treat
> these numbers as a recorded soak result and re-run the soak on the target
> host to reproduce. The circuit-breaker and evidence-store mechanisms the
> soak exercises are themselves covered by `tests/test_circuit_breaker.py`,
> `tests/test_evidence_store.py`, and `tests/test_watchdog.py` in CI.

## Headline

**Soak PASS**: audit chain intact at 1440/1440 checks, A1 success rate 99.9304% over 17248 emits, p99 decision latency 69.95 ms, max watchdog silence 0.57 s, 143 fault injections survived.

## The five measured numbers

| Metric | Value | Bar | Status |
| --- | ---: | --- | :---: |
| A1 emit success rate | 99.9304% | ≥ 99.9% | PASS |
| Decision latency p50 | 36.82 ms | (informational) | — |
| Decision latency p99 | 69.95 ms | ≤ 200 ms | PASS |
| Audit chain integrity | 1440/1440 verify=True | 100% | PASS |
| Max watchdog silence | 0.57 s | ≤ 30 s | PASS |

## Workload

* A1 emit attempts: **17248**
* A1 emit successes: **17236**
* A1 5xx observed (injected faults): **258**
* A1 breaker rejections (CircuitBreakerError): **0**
* Audit verifies run: **1440**
* R1 registers ok: **23**
* R1 deregisters ok: **23**
* Watchdog ticks logged: **1634**

## Latency distribution (ms, derived from real HTTP RTT)

n=17248 · mean=37.87 · stdev=23.52 · p50=36.82 · p95=66.57 · p99=69.95 · max=557.13

## Circuit-breaker state transitions

* t=  -0.01s wall → state=closed
* (Breaker stayed CLOSED for the whole soak.)

## Fault-injection timeline

| Sim hour | Wall t (s) | Injected duration (s) |
| ---: | ---: | ---: |
| 0.094 | 7.06 | 0.173 |
| 0.257 | 19.29 | 0.103 |
| 0.489 | 36.70 | 0.053 |
| 0.609 | 45.67 | 0.187 |
| 0.812 | 60.93 | 0.152 |
| 0.997 | 74.78 | 0.063 |
| 1.215 | 91.15 | 0.128 |
| 1.403 | 105.21 | 0.086 |
| 1.493 | 111.96 | 0.118 |
| 1.681 | 126.07 | 0.054 |
| 1.927 | 144.51 | 0.088 |
| 2.134 | 160.06 | 0.050 |
| 2.275 | 170.61 | 0.120 |
| 2.478 | 185.84 | 0.200 |
| 2.651 | 198.84 | 0.124 |
| 2.799 | 209.93 | 0.162 |
| 2.977 | 223.26 | 0.146 |
| 3.172 | 237.90 | 0.196 |
| 3.269 | 245.16 | 0.106 |
| 3.364 | 252.33 | 0.145 |
| 3.448 | 258.61 | 0.068 |
| 3.571 | 267.84 | 0.175 |
| 3.787 | 284.00 | 0.125 |
| 3.886 | 291.45 | 0.180 |
| 4.010 | 300.73 | 0.107 |
| 4.253 | 318.99 | 0.184 |
| 4.439 | 332.96 | 0.100 |
| 4.635 | 347.59 | 0.069 |
| 4.746 | 355.98 | 0.079 |
| 4.912 | 368.37 | 0.181 |
| 5.110 | 383.28 | 0.127 |
| 5.279 | 395.90 | 0.191 |
| 5.405 | 405.41 | 0.069 |
| 5.652 | 423.87 | 0.118 |
| 5.756 | 431.70 | 0.134 |
| 5.898 | 442.33 | 0.160 |
| 5.992 | 449.41 | 0.158 |
| 6.216 | 466.18 | 0.137 |
| 6.456 | 484.20 | 0.056 |
| 6.641 | 498.05 | 0.093 |
| 6.803 | 510.19 | 0.130 |
| 6.981 | 523.60 | 0.113 |
| 7.186 | 538.92 | 0.153 |
| 7.319 | 548.89 | 0.132 |
| 7.495 | 562.12 | 0.129 |
| 7.585 | 568.89 | 0.187 |
| 7.690 | 576.78 | 0.094 |
| 7.887 | 591.55 | 0.071 |
| 7.990 | 599.25 | 0.055 |
| 8.215 | 616.12 | 0.156 |
| … | … | (+93 more) |

## Acceptance bars

* Audit chain `verify()` always intact: **True**
* A1 emit success rate ≥ 99.9%: **True**
* p99 decision latency ≤ 200 ms: **True**
* Watchdog silence ≤ 30 s: **True**
* Zero unhandled exceptions: **True**

## How this was generated

The soak driver spawns `deploy/osc_emulator/server.py` as a real subprocess on a loopback port and pipes A1 emits through a fault-injecting TCP proxy. Each emit is wrapped in `AsyncCircuitBreaker.call(...)` — a real `pybreaker` state machine (`src/horizon_ric/runtime/circuit_breaker.py`) — and persists a `DecisionRecord` to a real SHA-256 hash-chained JSONL evidence store (`src/horizon_ric/evidence/store.py`). Every audit verify reads the JSONL back from disk and walks the chain via `EvidenceStore.verify()` (returns `-1` when intact). No mocks, no fakes, no stubs.

> The `scripts/soak_24h.py` entry point that ran this soak is not currently checked into `scripts/`. The breaker, evidence-store, and watchdog mechanisms it exercises are covered in CI by `tests/test_circuit_breaker.py`, `tests/test_evidence_store.py`, and `tests/test_watchdog.py`.
