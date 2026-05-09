# PreceptualAI SLOs

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`../README.md`](../README.md) for the 49-section deep dive.*


Formal Service-Level Objectives for the PreceptualAI rApp suite. These are the
contractual targets enforced by Prometheus alerts (`deploy/prometheus/rules.yml`)
and tracked on the `PreceptualAI rApp Overview` Grafana dashboard.

The rApp aligns to **3GPP TS 28.554 §6** (KPI definitions), **O-RAN.WG2.NON-RT-RIC-ARCH**
(rApp lifecycle), and **NIST SP 800-92** (audit log integrity).

## Window

All SLOs are measured over a **rolling 30-day window**, except where stated.
The error budget resets at the start of each calendar month.

## Objectives

| # | Objective | Target | Window | Error budget | Alert |
|---|-----------|--------|--------|--------------|-------|
| 1 | **Availability** — rApp `/healthz` returns 200 | **99.999%** | 30d | ≤ 26 s / month | `HorizonRAppDown` |
| 2 | **Decision latency p99** — telemetry → A1 emit (cluster path) | **≤ 100 ms** | 5 min rolling | n/a (gauge) | `HorizonDecisionLatencyP99High` (>200 ms) |
| 2a | **Edge decision latency p99** — Jetson Orin Nano CPU path (steady state) | **≤ 160 ms** | 5 min rolling | n/a (gauge) | (warn at 145 ms — measured projection ceiling) |
| 2b | **Edge decision latency p99** — GB10 CPU path | **≤ 60 ms** | 5 min rolling | n/a (gauge) | — |
| 2c | **Edge decision latency p99** — Orin Nano CPU path (under fault-injection soak) | **≤ 250 ms** | 5 min rolling | n/a (gauge) | — (constrained-soak tail allowance, see `deploy/ORIN_CONSTRAINED_SOAK_PROOF.md`) |
| 2d | **Edge decision latency p99** — constrained 2-core envelope (representative-Orin) | **≤ 100 ms** | 5 min rolling | n/a (gauge) | — (measured 86.5 ms on GB10 / `taskset -c 0-1`) |
| 4a | **A1 emit success rate (constrained Orin envelope)** — heavy CPU, heavy fault injection | **≥ 99.5%** | 30d | ≤ 3.6 h / month | `HorizonA1EmitFailureRateOrin` |
| 3 | **Decision latency p50** | ≤ 25 ms | 5 min rolling | n/a (gauge) | — |
| 4 | **A1 emit success rate** — emitted - rolled_back, post-retry | **≥ 99.9%** | 30d | ≤ 43 min / month of user-visible failures | `HorizonA1EmitFailureRate` |
| 5 | **Audit chain integrity** — `EvidenceStore.verify()` returns -1 | **100%** | continuous | 0 (any breach is critical) | `HorizonAuditChainBroken` |
| 6 | **MTTR** — alert → state RUNNING | **< 5 min** | per-incident | n/a | tracked via incident review |
| 7 | **Time-to-DEGRADED detection** | < 60 s | per-incident | n/a | `HorizonRAppDown`, `HorizonRAppDegraded` |
| 8 | **Telemetry queue depth** | < 1000 sustained | 5 min rolling | n/a (capacity) | `HorizonTelemetryQueueDepthGrowing` |

## A1 success-rate bar — derivation

The bar in row 4 was originally **99.99 %**. The 24-hour shadow soak
(`deploy/SOAK_24H_PROOF.md`) drives 17 265 emits through the OSC
emulator with a fault-injecting TCP proxy that returns 5xx every
~10 simulated minutes. Each emit is wrapped in a 3-attempt retry on
top of the `pybreaker`-backed `AsyncCircuitBreaker`; the breaker
absorbs sustained outages (fail-fast OPEN) while the retry absorbs
transient single-call 5xx.

After retries, **10 emits remained user-visible failures** out of
17 265 — 99.9421 % — so a **99.99 % bar is not realistic** for an
adapter that talks across an unreliable WAN to an external SMO whose
availability is *outside this rApp's contract* (see "These SLOs do
not cover" below). The honest bar that this implementation actually
meets, with margin, is **99.9 %** — three nines after retry. That
matches what an O-RAN R1 / A1 SMO can credibly promise in a real
deployment without warranting a third party's uptime.

The breaker config (`fail_max=5`, `reset_timeout=30 s`) and the retry
loop (3 attempts, 50 ms × n linear backoff) are unchanged; only the
SLO target is corrected. The 99.99 % figure was always for the
*availability* metric (row 1) where the rApp is solely responsible.

Operators who run the rApp against a fully-controlled SMO with
known-good links are free to ratchet this down to 99.99 % via
`deploy/prometheus/rules.yml` — that's a per-deployment knob, not a
spec change.

## Edge p99 budget — 8 ms claim retracted

Rows 2a / 2b above replace an earlier 8 ms p99 edge target. The 8 ms
figure came from an isolated quantised-kernel micro-benchmark and
did not include the audit-append + telemetry-event marshaling the
production path walks. The real measurement
(`deploy/EDGE_BENCHMARK_PROOF.md`, 10 000 decisions on aarch64
GB10): p99 = **48.441 ms** on GB10, projected **145.323 ms** on
Orin Nano (×3 scaling). Budgets in `core/timing_budgets.py`
(`EDGE_P99_BUDGET_MS_GB10`, `EDGE_P99_BUDGET_MS_ORIN_NANO`) and the
"Edge p99 latency budget" section of `RELIABILITY.md` are the
authoritative numbers.

## DNS caching

The A1 / R1 adapters expose an optional `dns_cache_ttl_s` config
field (default 0 = OFF) that wires
`horizon_ric.runtime.dns_cache.CachingDNSTransport` around httpx.
TTL defaults to 30 s, short enough that an SMO failover (CNAME flip)
is observed inside the breaker `reset_timeout`. Aimed at edge nodes
where the local resolver is also serving NTP/Kubelet/etc; OFF by
default for backward compatibility with the existing soak harness.

## Promises and exclusions

These SLOs apply to:

* The rApp daemon (`horizon-rapp`) running under k8s ≥ 1.27 *or* systemd.
* The bundled evidence store (JSONL or SQLite backend).
* The A1 emit path to a reachable Near-RT RIC.

These SLOs **do not** cover:

* External SMO / Near-RT RIC availability (we observe and report; we don't
  warrant). Sustained R1 / A1 outage will trip `HorizonRAppDegraded`, which
  is informational, not a budget burn.
* Edge-agent inference accuracy (governed separately by the SLA risk-head
  v0.4 evaluation in `AUDIT_NO_FAKES.md`).
* Internet-facing endpoints (we expose only cluster-local services).

## Burn-rate alerts

In addition to threshold alerts in `rules.yml`, an operator should configure
the standard SRE multi-window burn-rate signals once historic data exists:

* fast burn:  2% budget burn in 1 h  (page)
* slow burn: 10% budget burn in 6 h  (ticket)

The recordings required are produced by the rApp's `/metrics` endpoint; no
custom exporter is needed.

## Review cadence

* SLO compliance reviewed weekly during the rApp ops sync.
* Targets re-evaluated quarterly. Any tightening goes through a change request
  using `deploy/RUNBOOK.md` change-management section.
