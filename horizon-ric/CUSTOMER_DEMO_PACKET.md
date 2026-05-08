# PreceptualAI — Customer Demo Packet

**Version**: 0.2.0   **Date**: 2026-05-07   **Repo SHA**: `d23f1215156d00b7f16ab0c6934b3342f6f25900`

Pointers: [`FINAL_PROGRESS.md`](FINAL_PROGRESS.md) · [`GAPS_TO_PILOT.md`](GAPS_TO_PILOT.md) · [`RELIABILITY.md`](RELIABILITY.md) · [`deploy/SLO.md`](deploy/SLO.md)

---

## Page 1 — The 60-second pitch

**PreceptualAI = the regulator-defensible audit-rApp that runs alongside Ericsson EIAP / Nokia MantaRay.**

### Three differentiators

| # | What it is | Where it lives |
|---|---|---|
| 1 | **Counterfactual envelope** — every A1 policy carries chosen + rejected actions with predicted SLA risk at 30 s / 1 min / 5 min, signed and stored. | `src/horizon_ric/evidence/schema.py` |
| 2 | **Hash-chained audit** — SHA-256 chain plus RFC 3161 anchor, 7-year retention, **18 µs verify per record**. | `src/horizon_ric/evidence/{store,rfc3161}.py` |
| 3 | **ITU-R S.1503 EPFD in-loop** — every NTN-touching decision precomputes downlink EPFD against the GSO arc and refuses to emit if the mask is breached. | `src/horizon_ric/data/itu_r.py`, `benchmarks/epfd_10k.json` |

### What we are NOT

- **No production cells.** No tier-1 reference deployment; we augment a prime SMO, we do not replace one (`DEVIL_D_RFP.md:5`).
- **No FIPS 140-3 module today.** Crypto primitives are correct; the validated module is on the roadmap (`DEVIL_D_RFP.md:83`).
- **No 24×7 NOC.** No contractual Sev-1 MTTR. Pilot operations only (`DEVIL_D_RFP.md:69`).

---

## Page 2 — The 7 numbers a hostile architect will demand

| # | Metric | Value | Source |
|---|---|---:|---|
| 1 | Edge decision p99 latency, constrained 2-core Orin-Nano envelope, 10 K steps | **86.5 ms** | `deploy/ORIN_CONSTRAINED_SOAK_PROOF.md:90` |
| 2 | A1 emit success rate, 24-h shadow soak, 17 248 emits | **99.93 %** | `deploy/SOAK_24H_PROOF.md:9` |
| 3 | Audit chain integrity across both 24-h soaks | **2 880 / 2 880** verifies intact (1 440 unconstrained + 1 440 constrained) | `deploy/SOAK_24H_PROOF.md:18` + `deploy/ORIN_CONSTRAINED_SOAK_PROOF.md:46` |
| 4 | Audit chain verify cost per record | **18 µs** | `benchmarks/bench_audit_verify.py`, `BENCHMARK_HEAD_TO_HEAD.md:42` |
| 5 | EPFD violation rate, 10 000 real-orbit Starlink scenarios × 200 sats | **0.19 %** (19 / 10 000) | `benchmarks/epfd_10k.json:21` |
| 6 | OWASP ZAP 2.16 baseline scan against live `/api/v1` + dashboard | **0 Critical / 0 High / 0 Medium** (2 Low, 1 Info) | `deploy/ZAP_SCAN_PROOF.md:11` |
| 7 | Subsystem health smoke test | **57 / 57 green in 3.05 s** | `scripts/lightup_all_subsystems.py`, `DEMO_LIGHTUP.md:5` |

**Honest disclosure on #1**: 86.5 ms was measured on a GB10 aarch64 host with `taskset -c 0-1` (2 cores pinned) — a stricter compute budget than a real Orin Nano. Physical Jetson Orin Nano hardware soak is procurement-blocked, not software-blocked (`GAPS_TO_PILOT.md:230`). Constrained soak A1 success was 99.60 % (Orin bar 99.5 %); unconstrained 24-h soak was 99.93 %.

---

## Page 3 — The 5-minute walkthrough script

Copy-paste, top to bottom. No mocks, no fakes — every step exercises real code paths.

```bash
git clone <repo-url> && cd horizon-ric
.venv/bin/python -m horizon_ric.cli demo --quick
```
**Architect should see**: a single A1 policy emitted with a counterfactual envelope JSON (chosen action + 3 rejected alternatives, each with `sla_risk_30s/1min/5min`).
**Point out**: the rejected alternatives — this is what every other vendor's rApp throws away.

```bash
.venv/bin/python scripts/lightup_all_subsystems.py
```
**Architect should see**: `lightup_all: 57/57 subsystems green, total wall-clock ~3.05 s`.
**Point out**: every public subsystem in `PRD.md` runs a real op (no stubs). This is the canonical smoke test.

```bash
.venv/bin/python scripts/edge_benchmark_arm64.py --steps 1000
```
**Architect should see**: p50 / p95 / p99 latency printed; full 4-layer MLP head + audit-append per decision.
**Point out**: this is the same script that produced the 86.5 ms constrained-Orin number — reproducible on their hardware in seconds.

```bash
.venv/bin/python scripts/run_horizon_rapp.py --once
```
**Architect should see**: one full rApp tick — telemetry ingest → risk head → policy → A1 emit → evidence append → chain verify.
**Point out**: the audit chain's `verify=-1` return value (the index of the first broken link; -1 means intact).

```bash
curl http://localhost:8083/healthz
curl http://localhost:8083/metrics | head -30
```
**Architect should see**: `{"status":"ok"}`, then Prometheus counters for A1 emits, breaker state, audit verifies, EPFD checks.
**Point out**: the Grafana panel at `deploy/grafana/` plots these directly; the SLO bars in `deploy/SLO.md` are wired to these counters.

---

## Page 4 — RFP scorecard summary (link-out)

Bottom line from [`DEVIL_D_RFP.md`](DEVIL_D_RFP.md):

- **38 / 100** on a stock tier-1 RFP scorecard against Ericsson EIAP / Nokia MantaRay (`DEVIL_D_RFP.md:259`).
- **4 / 4 on the differentiator axis** — counterfactual envelope, hash-chained audit, ITU-R S.1503 EPFD in-loop, TS 28.105 model card. Ericsson and Nokia have no public response on any of the four (`DEVIL_D_RFP.md:271`).
- **Posture**: "augment Ericsson, do not replace Ericsson."

Full 15-question breakdown, gap-to-parity costs, and the 6 we cannot fix on a 12-month budget: see `DEVIL_D_RFP.md`.

---

*This packet is hand-out grade. Numbers cite `file:line` against repo SHA `d23f1215156d00b7f16ab0c6934b3342f6f25900`. If a number changes, this document is stale — regenerate from the proof MDs in `deploy/` and `benchmarks/`.*
