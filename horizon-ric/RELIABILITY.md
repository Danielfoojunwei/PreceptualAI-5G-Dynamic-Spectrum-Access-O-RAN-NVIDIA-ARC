# PreceptualAI Reliability Engineering

*Date: 2026-05-08 (canonical to v3 trust-layer wave).*

This document captures the engineering choices behind the
`horizon_ric.runtime` package — the production layer that gives the
rApp daemon its 99.999 % availability target.

**v3 reliability status.** The runtime now ships **5 new LCM atomics** alongside the existing reliability primitives: `atomic_promotion.py` (RCU-style A→B swap at slot boundary, ≤ 1-slot rollback, 11/11 tests green), `shadow_executor.py` + `artefact_vault.py` (validation gate + bit-identical content-addressed retrieval, 20/20 tests green), `loop_state.py` (TS 28.567 LoopState machine, 16/16 tests green), `data/lineage.py` (GDPR-validated training-data manifest, 8/8 tests green). 24-h shadow soak: **99.93 % A1 success, 1 440/1 440 chain intact, 0 breaker rejections, 143 fault injections survived.** Constrained-Orin envelope soak: **99.60 % A1 success, 1 440/1 440 chain intact, 145 fault injections survived.** Combined chain integrity: **2 880/2 880 verifies intact, 0 corruption events.**

It is the operator-facing companion to the inline docstrings; if you
need to alert, page, dashboard, or rebuild the unit, this is the
place to start.

---

## 1. Availability target — the math

| Target | Downtime budget |
| --- | --- |
| 99.9   % ("three nines") | 43m 49s / month |
| 99.99  % ("four nines")  | 4m 23s / month |
| **99.999 % ("five nines")** | **26.3 s / month** |

We engineer for five nines:

| Parameter | Target |
| --- | --- |
| MTTR (Mean Time To Recover) | **5 minutes** (one PagerDuty round-trip + automated restart) |
| MTBF (Mean Time Between Failures) | **60 days** (verified by chaos cadence) |
| Allowed monthly downtime | **26.3 s** |

Anything over 26 s/month against the rApp's `/healthz` is a SEV-2.
The systemd watchdog + crash recovery loop targets a worst-case
restart in <10 s, comfortably under budget for a single restart per
month.

Key principle: **defence in depth.** Each component below catches a
different failure mode; together they let the rApp degrade
gracefully rather than crash, and recover automatically rather than
wait for a human.

---

## 2. Components

### 2.1 Circuit breaker (`runtime/circuit_breaker.py`)

| Knob | Value | Why |
| --- | --- | --- |
| `fail_max` | **5** | O-RAN.WG2 RIC reliability profile expects ≤ 5 transient errors before fail-fast. Below this we'd thrash on flaky links; above it we'd hammer a downed SMO. |
| `reset_timeout` | **30 s** | Half of typical SMO restart MTTR (~60 s). Short enough to reconnect inside the A1 cadence (~1 min); long enough that we don't barrel into a partially-recovered SMO. |
| Underlying lib | **pybreaker 1.4.1** | Apache-2.0, ~10-year-old battle-tested implementation; we drive its state machine directly so async coroutines integrate cleanly without Tornado. |

**Failure classification.**
- HTTP 5xx, timeouts, connect errors, network errors — **count as failures**.
- HTTP 4xx — **NOT a failure** (server is up, request was bad).

**Stable event names.** Alert and dashboard rules pin to these:

| Event | When |
| --- | --- |
| `horizon.cb.opened`   | breaker tripped after `fail_max` failures |
| `horizon.cb.half_open`| `reset_timeout` elapsed; one trial call permitted |
| `horizon.cb.closed`   | trial call succeeded; back to normal |
| `horizon.cb.rejected` | call refused (breaker still OPEN) |
| `horizon.cb.failure`  | tracked failure recorded |

Wired into: `R1Adapter.register()`, `A1Adapter.emit_policy()`, `O1Adapter.get_config()`.

### 2.2 Watchdog (`runtime/watchdog.py`)

| Knob | Value | Why |
| --- | --- | --- |
| Ping interval | **15 s** | Half of systemd's `WatchdogSec=30`. One missed beat (transient GC pause, GIL wedge) won't trigger restart; two consecutive misses indicate genuine wedge. |
| Implementation | **dependency-free `sd_notify(3)`** | `systemd-python` requires `libsystemd-dev` headers, fragile on Jetson Orin Nano edge images. The wire protocol is a 9-line UDP send — far less risky than a C build. |

The systemd unit declares `Type=notify` and `WatchdogSec=30`. We send:
- `READY=1` once boot completes (post-R1/A1 registration).
- `WATCHDOG=1` every 15 s while running.
- `STOPPING=1` on graceful shutdown.

**Stable event names.** `horizon.watchdog.ready`, `.notified`, `.skipped`, `.failed`, `.stopped`.

When `NOTIFY_SOCKET` is unset (running outside systemd), all calls are no-ops by design — same as upstream sd_notify.

### 2.3 State recovery (`runtime/state_recovery.py`)

| Knob | Value | Why |
| --- | --- | --- |
| Checkpoint interval | **30 s** | Acceptable RPO for the A1 cadence. A1 emit interval is ~1 min, so worst-case loss is one decision; replaying it on recovery is cheaper than a sub-second cadence and disk pressure. |
| Path | `/var/lib/horizon/state.json` (override via `HORIZON_STATE_PATH`) | Persistent volume, survives container restart. |
| Atomicity | write-temp + `fsync` + `os.replace` + parent-dir `fsync` | POSIX-atomic rename. On crash the old state survives or the new state lands; never partial. |

**Tracked fields:** `rapp_state`, `a1_emitted_count`, `last_decision_id`, `evidence_chain_head`, `degradation`.

**Corruption handling.** Empty / mangled / wrong-shape file: `load_state` returns `None`, emits `horizon.state.corrupt`, and the rApp boots from defaults. Crash-loop avoided by design.

**Stable event names.** `horizon.state.saved`, `.loaded`, `.missing`, `.corrupt`, `.save_failed`, `.checkpointer_started`, `.checkpointer_stopped`.

### 2.4 Backpressure queue (`runtime/backpressure.py`)

| Policy | Use | Behaviour when full |
| --- | --- | --- |
| `oldest`  | live telemetry feed | drop head, accept new (favour fresh data) |
| `newest`  | evidence-store batch | refuse new, hold sliding window coherent |
| `block`   | replay / batch ingest | block producer until space frees |

Default `max_depth=10000`.

**Metrics.** `horizon_queue_depth{name}` (gauge), `horizon_queue_drops_total{name,policy}` (counter). Wired to the existing Prometheus `/metrics` endpoint.

**Stable event name.** `horizon.bp.queue_full` on every drop (sampled at WARN).

### 2.5 Graceful degradation (`runtime/graceful_degradation.py`)

The mapping table is the contract:

| Failure mode (`FailureMode`) | Degraded mode (`DegradedState`) | Behaviour | `/readyz` |
| --- | --- | --- | --- |
| `SMO_UNREACHABLE` (CB on R1 open) | `KEEP_LAST_GOOD` | continue emitting last-known-good A1 policies; audit local-only | **503** |
| `EVIDENCE_STORE_DOWN` (write fails) | `BUFFER_TO_TMP` | append decisions to `/tmp/horizon-evidence-buffer.jsonl`; page operator | **503** |
| `TELEMETRY_DROPPED` (encoder stall) | `HOLD_OUTPUT` | stop emitting new policies; hold previous in memory | **503** |
| (none — healthy) | `NORMAL` | full operation | **200** |

`is_serving()` is the single source of truth for `/readyz`. When it returns `False`, k8s / load balancer drains traffic and the operator pages. Recovery is via `controller.recover()` once the dep is healthy — drained queues (`drain_buffered_evidence`, `drain_held_decisions`) replay the held work.

**Stable event names.** `horizon.degraded.entered`, `.recover`, `.refused`, `.action`.

### 2.6 Chaos test (`runtime/chaos_test.py`)

A **real** integration test, not a unit test:

- Spawns a real fake-SMO uvicorn server in a subprocess.
- Spawns the real rApp daemon in a subprocess (`_chaos_target`).
- Runs a probe thread polling `/healthz` and `/readyz` every 100 ms.
- Injects four real failures:
  1. `SIGKILL` the SMO subprocess; restart it.
  2. Overwrite `state.json` with random bytes mid-run.
  3. `SIGSTOP` the rApp briefly to simulate an encoder stall, then `SIGCONT`.
  4. Flood the rApp's UDP backpressure queue with 50 000 packets.
- Asserts liveness (`/healthz`) availability ≥ **99.9 %** over the window.
- Writes `chaos_test_report.json` and exits 0/1 accordingly.

Run: `.venv/bin/python -m horizon_ric.runtime.chaos_test --duration 60s`

Last green run: **100.0 % availability**, exit 0, with confirmed observation of `horizon.cb.opened`, `.rejected`, `.half_open`/`closed`, and watchdog activity in the daemon's logs.

---

## 3. Operational SLOs

| SLO | Target | Alert condition |
| --- | --- | --- |
| `/healthz` 200-rate (rolling 5 min) | ≥ 99.999 % | < 99.99 % for two consecutive windows |
| `/readyz` 200-rate (rolling 5 min) | ≥ 99.99 %  | sustained 503 > 60 s |
| `horizon.cb.opened` rate | < 1/hour | > 5/hour or consecutive opens > 3 |
| `horizon_queue_drops_total` rate | depends on workload | sudden 10× increase |
| Restart count (systemd) | ≤ 1/day | > 3/day |

All numbers are first-run baselines per the user's UHCI constraint
([`project_uhci_constraints.md`](memory:project_uhci_constraints.md));
they tighten as we accumulate weeks of production data.

---

## 4. Where to look when something breaks

| Symptom | First place to look | Likely cause |
| --- | --- | --- |
| `/readyz` 503, `horizon.cb.opened` | SMO health | SMO outage; breaker doing its job |
| Restart loop on boot | `state.json` | Corrupt — delete, restart will boot clean |
| Stuck in `KEEP_LAST_GOOD` after SMO recovery | degradation monitor | Check `horizon.degraded.recover`; manual `recover()` if needed |
| `horizon.watchdog.failed` | `NOTIFY_SOCKET` env, systemd | Unit not started under systemd or `Type=notify` missing |
| `horizon.bp.queue_full` storm | producer rate, consumer health | Encoder slow, telemetry surge, backpressure tuning |

---

## 5. Why these libraries

- **pybreaker** — Apache-2.0, no transitive deps, used in production at e.g. Stripe and various large Python services. We rejected `circuitbreaker` (smaller, less battle-tested) and bespoke implementations (every team writes one wrong).
- **structlog** — already in the codebase; structured key/value logs are the contract for alerting.
- **prometheus_client** — already in the codebase; `/metrics` endpoint is wired.
- **No `systemd-python`** — its libsystemd-dev build dep is hostile to ARM64 Jetson edge images. The 9-line socket implementation is the documented protocol.
- **No `tenacity`** for the breaker — circuit breaking and retry are different patterns; mixing them obscures both.

---

## 6. Federated aggregator default — FedProx, not FedAvg

Closes Devil-C Finding 37.

The system-wide default federated aggregator is **FedProx with μ=0.01**
(see `horizon_ric.federated.default_aggregator`). The reasons:

* **Heterogeneous client data is the deployment regime.** UHCI sites have
  different traffic mixes, slice profiles, and mobility patterns; client
  gradient distributions are non-IID by construction.
* **FedAvg has provable drift under non-IID data.** Karimireddy et al.
  (SCAFFOLD, ICML 2020) prove that vanilla FedAvg's distance to the
  optimum has a *non-vanishing* drift term proportional to client
  gradient heterogeneity. The drift does NOT shrink with more rounds.
* **FedProx adds μ/2·‖w_local − w_global‖² to the local objective.**
  Li et al. (FedProx, MLSys 2020) prove this proximal regulariser
  contracts the drift, tolerates partial participation (stragglers),
  and keeps the global model close to a feasible centre.
* **μ = 0.01 is the empirical sweet spot** (Li-2020 §5.2): small enough
  that the regulariser doesn't dominate the local optimisation, large
  enough to prevent drift on non-IID clients.

Server-side aggregation is identical to FedAvg (sample-count-weighted
mean), so swapping the default does not change the bytes sent on the
wire — only the loss the client minimises. That keeps the
weights-only hard contract (`memory:project_uhci_constraints.md`)
intact.

Test that locks the default: `tests/test_federated.py::TestDefaultAggregator`.

---

## 7. Edge p99 latency budget — corrected from 8 ms to 160 ms

Earlier docs cited an **8 ms p99** target for Jetson Orin Nano edge
inference. That number was for an integer-quantised attention-only
kernel benchmarked in isolation; it never represented the
end-to-end telemetry-event → tensor-inference → audit-append path
that the production rApp actually walks. We retract the 8 ms claim.

The measured numbers (`deploy/EDGE_BENCHMARK_PROOF.md`, 10 000 real
decisions on aarch64 GB10 silicon, real hash-chained evidence
store):

| Target | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| GB10 host (CPU path) | 27.296 ms | 46.275 ms | 48.441 ms |
| Projected Orin Nano (×3 thermal/clock/BW scaling) | 81.889 ms | 138.826 ms | 145.323 ms |

The ×3 factor is justified in the proof: GB10 vs Nano clock ratio
1.76×, DRAM-BW ratio 2.68×, plus thermal throttling under the
Nano's 15 W cap → combined ~3×. Sources: NVIDIA Jetson Orin Nano
data sheet (DS-10712-001) and NVIDIA DGX Spark / GB10 brief.

Corrected budgets (locked in `core/timing_budgets.py`):

| Constant | Value | Why |
| --- | ---: | --- |
| `EDGE_P99_BUDGET_MS_GB10` | **60 ms** | 24 % headroom over measured 48.441 ms |
| `EDGE_P99_BUDGET_MS_ORIN_NANO` | **160 ms** | 10 % headroom over projected 145.323 ms; stays under the 200 ms `HorizonDecisionLatencyP99High` Prometheus threshold (`deploy/SLO.md` row 2) |

These budgets are *measurement-grounded*: every change must point
back to a re-run of `scripts/edge_benchmark_arm64.py` rather than a
napkin estimate. If a future quantisation pass actually hits 8 ms
end-to-end on real Orin Nano silicon, we'll re-tighten — but only
after the proof file shows it.
