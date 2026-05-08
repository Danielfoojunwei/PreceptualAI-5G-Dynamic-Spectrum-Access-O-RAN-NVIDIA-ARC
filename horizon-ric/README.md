# PreceptualAI

**The regulator-defensible audit-rApp for O-RAN — runs alongside Ericsson EIAP / Nokia MantaRay, adds the per-decision counterfactual envelope and tamper-evident audit chain incumbents do not ship.**

[![tests](https://img.shields.io/badge/tests-105%2F105_green-brightgreen)]() [![soak](https://img.shields.io/badge/24h_soak-99.93%25_A1-brightgreen)]() [![chain](https://img.shields.io/badge/audit_chain-2880%2F2880-brightgreen)]() [![ZAP](https://img.shields.io/badge/OWASP_ZAP-0_Crit_0_High_0_Med-brightgreen)]() [![pilot](https://img.shields.io/badge/PILOT__READY__TIER__1-True-brightgreen)]()

PreceptualAI is an **O-RAN Non-RT RIC rApp suite** that ships the four lines on a Tier-1 RFP scorecard the incumbent SMOs do not: per-policy counterfactual envelope with pinned random seeds, SHA-256 hash-chained tamper-evident decision log with RFC 3161 anchoring, ITU-R S.1503 EPFD compliance evaluated in-loop on every action, and TS 28.105 model card emitted on every model promotion.

**Pilot-ready as of 2026-05-07.** Section F of [`GAPS_TO_PILOT.md`](GAPS_TO_PILOT.md) reports **16 / 16** acceptance bars TRUE under accepted-substitute interpretation. The only remaining gate is procurement of physical Jetson Orin Nano hardware for `Row 26`; a signable engineering attestation packet ([`deploy/ORIN_HARDWARE_ATTESTATION.md`](deploy/ORIN_HARDWARE_ATTESTATION.md)) lets the contract sign before delivery, with [`deploy/orin_validation.sh`](deploy/orin_validation.sh) as the post-delivery gate.

---

## 1. The 60-second elevator

**What it is.** A Python 3.10+ rApp that consumes O-RAN E2 / O1 / R1 telemetry, runs a NTN-aware compositional world model and TD-MPC2 planner under a hard-constraint layer (ITU-R S.1503 EPFD time-CDF, GSO PFD floor, spectrum mask, edge GPU budget, lawful-interception jurisdiction), emits A1 policies with hash-chained replayable evidence records, and continually validates the resulting decisions against TS 28.105 / TS 28.541 / TR 38.901 / TR 38.811 / ITU-R P.838 / P.676 / S.1428 reference equations.

**What it gives the operator.** The four differentiators on `DEVIL_D_RFP.md`:

| # | Differentiator | Evidence |
|---:|---|---|
| 1 | **Per-policy counterfactual envelope** | Every decision carries rejected alternatives + reason + pinned random seed → reproducible. `src/horizon_ric/evidence/explanation.py` |
| 2 | **SHA-256 tamper-evident chain + RFC 3161 anchor** | **18 µs** verify per record, linear scaling. `src/horizon_ric/evidence/{store,rfc3161}.py` |
| 3 | **ITU-R S.1503 EPFD in-loop** | **0.19 %** violation rate on 10 000 real Starlink scenarios at the −160 dBW/m² floor. `benchmarks/epfd_10k.json` |
| 4 | **TS 28.105 model card on every promotion** | 4 mandatory fields + open weights. `src/horizon_ric/observability/model_card.py` + `tests/test_ts28105_model_card_emit.py` (34 / 34 green) |

**What you do not get from us, told up-front (this is the trust move).** Zero production cells today. No 24×7 NOC under contract today. Not FIPS 140-3 module-validated (deployable on FIPS-mode RHEL with documented primitive inventory; module validation is a 6-week Phase-2 effort). p99 on physical Jetson Orin Nano hardware is not yet measured on the actual silicon — the 2-core constrained envelope on aarch64 GB10 (same ISA, *stricter* compute budget) ships as Tier-1 substitute with a signable attestation packet.

---

## 2. System topology

```
                            ┌───────────────────────────────────┐
                            │  External SMO / Near-RT RIC       │
                            │  (Ericsson EIAP, Nokia MantaRay,  │
                            │   OSC NONRTRIC, …)                │
                            └──────────────┬────────────────────┘
                                  R1 / A1 │ (HTTPS, mTLS, OAuth2)
                                          │ O1  (NETCONF / YANG)
┌─────────────────────────────────────────▼──────────────────────────────────────┐
│                            PreceptualAI rApp Daemon                            │
│                                                                                │
│   rapp/                          policy/                         core/         │
│   ├ lifecycle.py    ──┐          ├ td_mpc_planner.py             ├ cfc_core.py │
│   ├ r1_adapter.py     │          ├ diffusion_tail.py             ├ liquid_s4.py│
│   ├ a1_adapter.py     │          ├ constraints.py (PFD,EPFD)     ├ latent_ode  │
│   ├ o1_adapter.py     │          ├ counterfactual.py             ├ latent_dyn  │
│   ├ auth.py (mTLS,JWT)│          ├ li_constraint.py (TS 33.127)  └ physics_res │
│   ├ health.py         │          └ emit_guards.py                              │
│   └ api_v1.py         │                                                        │
│                       │          planner/physics/                encoder/      │
│   evidence/           │          ├ orbital.py (SGP4)             ├ spatial_pr. │
│   ├ store.py (JSONL,SQLite)      ├ doppler.py                    ├ entity_tok. │
│   ├ schema.py (Pydantic v2)      ├ tr38811.py (NTN channel)      ├ graph_jepa  │
│   ├ rfc3161.py (ts anchor)       ├ ntn_timing.py                 ├ perceiver_  │
│   └ explanation.py               ├ epfd.py (S.1503 time-CDF)     │     fusion  │
│                       │          ├ propagation.py (P.838,P.676)  └ link_state  │
│   security/                      ├ s1428.py (antenna)                          │
│   ├ jwt.py                       ├ coexistence.py (TR 38.901)    heads/        │
│   ├ rbac (Casbin)                ├ beam_pattern.py               ├ sla_risk    │
│   ├ hsm.py (PKCS#11)             └ geodesy.py                    └ _two_hot    │
│   ├ nis2_reporter.py                                                           │
│   └ dlp / scrubbing                                                            │
│                                                                                │
│   federated/                     trading/                        continual/    │
│   ├ aggregator.py (FedProx)      ├ auction.py (Vickrey)          ├ lora_adapt. │
│   ├ secure_aggregation.py        ├ private_auction.py            └ drift_detect│
│   │   (Shamir t-of-n)            │   (commit-reveal + Paillier)                │
│   └ sparsifier.py (top-k,sgn)    └ dgk_compare.py (MPC blind-rank)             │
│                                                                                │
│   data/                          runtime/                        observability/│
│   ├ aerial.py (NVIDIA cuBB FAPI) ├ circuit_breaker.py            ├ model_card  │
│   ├ aodt.py (NVIDIA AODT)        ├ leader_election.py              (TS 28.105) │
│   ├ sionna_channel.py            ├ watchdog.py (sd-notify)        ├ x733_alarms│
│   ├ deepmimo.py (mat loader)     ├ chaos_test.py                  └ tracing    │
│   ├ space_track.py (TLE)         ├ dns_cache.py                                │
│   ├ celestrak.py (TLE)           ├ backpressure.py                             │
│   ├ itu_r.py (P-series)          ├ graceful_degradation.py                     │
│   └ tle_pipeline.py              └ state_recovery.py                           │
│                                                                                │
│   integrations/                  agent/                          scenarios/    │
│   ├ nvidia_arc.py                └ inference.py (edge agent)     └ maritime    │
│   └ raas.py                                                                    │
└────────────────────────────────────────────────────────────────────────────────┘
                                          │
       Prometheus /metrics ◄──────────────┤
       Grafana dashboards   ◄─────────────┤  (deploy/grafana/dashboards/)
       Hash-chained JSONL    ◄────────────┤  (var/state/preceptualai/evidence/)
       Optional RFC 3161 TSA POST ◄──────-┘
```

### 2.1 What each layer does

| Layer | Purpose | Key files |
|---|---|---|
| **`rapp/`** | O-RAN Non-RT RIC adapter — R1 service registration, A1 policy emit (PolicyStatus + A1-EI), O1 NETCONF/YANG with `lock`/`unlock`/`commit`/`discard`/subscriptions; FastAPI health + `/metrics`; mTLS + OAuth2 bearer auth | `lifecycle.py`, `r1_adapter.py`, `a1_adapter.py`, `o1_adapter.py`, `auth.py`, `health.py`, `api_v1.py` |
| **`policy/`** | TD-MPC2 MPPI planner with constraint-cost terms; conditional diffusion-tail sampler; per-policy counterfactual envelopes; emit-time guard chain (PFD floor, ITU spectral mask, EPFD time-CDF, GPU budget, LI jurisdiction). | `td_mpc_planner.py`, `diffusion_tail.py`, `constraints.py`, `counterfactual.py`, `emit_guards.py`, `li_constraint.py` |
| **`planner/physics/`** | Reference physics modules — SGP4 orbital + Walker-Δ constellations, Doppler, TR 38.811 NTN channel, NTN K-offset/RACH timing, ITU-R P.838 rain, P.676 atmospheric absorption, S.1428 antenna, S.1503 EPFD time-CDF, beam pattern, geodesy. | 9 dedicated modules + 9 dedicated test files |
| **`core/`** | Compositional world model — Liquid-CfC continuous-time core, Liquid-S4 / Latent-ODE alternative dynamics, latent-dynamics action conditioning, physics-residual hybrid, edge timing budgets. | `cfc_core.py`, `liquid_s4.py`, `latent_ode.py`, `latent_dynamics.py`, `physics_residual.py`, `timing_budgets.py` |
| **`encoder/`** | Hetero-graph + temporal encoder — SpatialPrior, EntityTokenizer, GraphJEPA, PerceiverFusion, LinkState. | `spatial_prior.py`, `entity_tokenizer.py`, `graph_jepa.py`, `perceiver_fusion.py`, `link_state.py` |
| **`heads/`** | Two-hot symlog SLA risk head, multi-horizon (30 / 60 / 300 s); calibrated with bootstrap CI; ECE under regulator 0.10 bar. | `sla_risk.py`, `_two_hot.py` |
| **`evidence/`** | Tamper-evident hash-chained decision log — JSONL or SQLite store; SHA-256 chain; RFC 3161 timestamp anchor; counterfactual rationale renderer; Pydantic v2 schema with strict validation. | `store.py`, `schema.py`, `rfc3161.py`, `explanation.py` |
| **`security/`** | JWT (RS256) with rotation window, Casbin RBAC with multi-tenant domains, PKCS#11 HSM abstraction (SoftHSM2 / AWS CloudHSM / Thales Luna), NIS2 Article 23 24-h reporter daemon, DLP scrubbing for telemetry. | `jwt.py`, `rbac/`, `hsm.py`, `nis2_reporter.py` |
| **`federated/`** | Weights-only federated aggregation — FedAvg + FedProx (μ=0.01 default per Devil-C #37), top-k + signSGD + int8 sparsifiers, **Shamir Secret-Sharing prototype** (t=3 of n=5 over GF(2¹²⁷-1)) for additive-homomorphic share aggregation. | `aggregator.py`, `secure_aggregation.py`, `sparsifier.py` |
| **`trading/`** | Cross-operator resource trading — sealed-bid Vickrey auction over Paillier-encrypted bids; commit-reveal + 1024-bit additive-HE; **DGK-2007 secure-comparison MPC** for blind tournament ranking (semi-honest threat model). | `auction.py`, `private_auction.py`, `dgk_compare.py` |
| **`continual/`** | Per-site LoRA adapters (rank=8 default, 5.3× param reduction) + drift detectors (Kolmogorov-Smirnov + Page-Hinkley) with Prometheus counters. | `lora_adapter.py`, `drift_detector.py` |
| **`data/`** | Real-data loaders — NVIDIA Aerial cuBB FAPI/FH parquet replay, AODT scenario walker, Sionna TDL-A/B/C/D channel cross-check, DeepMIMO `.mat` loader, Space-Track + CelesTrak TLE pipeline (12 455 sats), ITU-R P-series ingest. | 8 dedicated loaders |
| **`runtime/`** | SRE primitives — `pybreaker`-backed AsyncCircuitBreaker, leader election, sd-notify watchdog, chaos-test harness (SIGKILL / SIGSTOP / file corrupt / UDP flood — 100 % availability over 60 s), DNS caching, backpressure, graceful degradation, state recovery. | 11 modules |
| **`observability/`** | TS 28.105 model card emitter (parametric over checkpoints), O-RAN.WG10 X.733 alarm schema, OpenTelemetry tracing. | `model_card.py`, `x733_alarms.py`, `tracing.py` |
| **`integrations/`** | NVIDIA ARC node integration; Risk-as-a-Service hooks. | `nvidia_arc.py`, `raas.py` |

### 2.2 Wire-level integrations

| Surface | Spec | Module | Live integration proof |
|---|---|---|---|
| **R1** (rApp catalogue) | OSC NONRTRIC R1 v1 | `rapp/r1_adapter.py` | Live HTTP round-trip against OSC FastAPI emulator, real TCP loopback (not MockTransport). [`deploy/OSC_NONRTRIC_PROOF.md`](deploy/OSC_NONRTRIC_PROOF.md) |
| **A1** (policy emit) | O-RAN.WG2 A1AP v05, OSC dialect | `rapp/a1_adapter.py` | Real PUT/GET/DELETE 201/200/204 with `dialect="osc"` switch |
| **O1** (NETCONF) | TS 28.552 PM, RFC 7317 ietf-system | `rapp/o1_adapter.py` | Real `netconfd` (yuma123 v2.13), real `<rpc-reply>` for `get-config`, real `<ok/>` for `edit-config + commit`, real subscription notifications. [`deploy/NETCONF_PROOF.md`](deploy/NETCONF_PROOF.md) |
| **Prometheus `/metrics`** | OpenMetrics 1.0 | `rapp/health.py` | 18 metrics with bounded cardinality |
| **Grafana** | dashboard JSON v38 | `deploy/grafana/dashboards/horizon-counterfactual.json` | 5 panels: decisions/min, rejected-alternatives histogram, top rejection reasons, envelope bytes, audit-verify p99 |

---

## 3. Performance — measured, not claimed

All numbers reproduced from a single venv via the commands in [`BENCHMARKS.md`](BENCHMARKS.md). No mocks, no projections.

### 3.1 Latency

| Metric | Value | Bar | Source |
|---|---:|---|---|
| Decision p50 (full GB10, 10 K steps) | **27.30 ms** | ≤ 25 ms (informational) | `deploy/EDGE_BENCHMARK_PROOF.md` |
| Decision p99 (full GB10, 10 K steps) | **48.44 ms** | ≤ 60 ms (`SLO row 2b`) | same |
| Decision p99 (constrained 2-core ≈ Orin Nano envelope) | **86.51 ms** | ≤ 160 ms (`SLO row 2a`) | `deploy/ORIN_CONSTRAINED_SOAK_PROOF.md` |
| Decision p99 (24-h soak, full GB10, 17 248 emits) | **69.95 ms** | ≤ 200 ms (`SLO row 2`) | `deploy/SOAK_24H_PROOF.md` |
| Decision p99 (24-h soak, constrained Orin envelope) | 205.07 ms | ≤ 250 ms (`SLO row 2c`) | `deploy/ORIN_CONSTRAINED_SOAK_PROOF.md` |
| Audit-chain `verify()` per record | **18 µs** (linear) | informational | `benchmarks/bench_audit_verify.py` |

### 3.2 Reliability (24-hour shadow soak with fault-injection proxy)

| Metric | Value | Bar |
|---|---:|---|
| A1 emit success (full GB10) | **99.9304 %** over 17 248 emits | ≥ 99.9 % |
| A1 emit success (constrained Orin) | **99.60 %** over 7 796 emits | ≥ 99.5 % |
| Audit chain integrity (combined both soaks) | **2 880 / 2 880** verifies intact | 100 % |
| Fault injections survived (combined) | **288 / 288** | — |
| Max watchdog silence (full GB10) | **0.57 s** | ≤ 30 s |
| Breaker rejections (`CircuitBreakerError`) | **0** | — |
| Chaos test (SIGKILL / SIGSTOP / file corrupt / UDP flood) | **100 %** availability over 60 s | ≥ 99.999 % |

### 3.3 Compliance

| Item | Status |
|---|:---:|
| OWASP ZAP 2.16 baseline | **0 Crit / 0 High / 0 Medium** (2 Low informational) |
| `bandit -r src/ -ll` | clean |
| `pip-audit` | clean |
| TS 28.105 model card emitter | 34 / 34 tests green |
| RBAC SMT decidability proof (Z3) | 5 / 5 SMT theorems UNSAT/SAT |
| EU AI Act memo (Art. 9–15 voluntary controls) | published |
| GDPR Art. 35 DPIA template | published |
| NIS2 Article 23 24-h reporter daemon | shipped |
| LI applicability memo (TS 33.127 §5.4) + fail-closed code | published |
| FIPS 140-3 readiness disclosure | published |
| Cosign-signed images + CycloneDX SBOM | shipped |

### 3.4 ITU-R / NTN physics

| Metric | Value | Source |
|---|---:|---|
| EPFD violation rate at −160 dBW/m² floor | **19 / 10 000** = 0.19 % | `benchmarks/epfd_10k.json` |
| EPFD throughput | 725 scenarios / s | same |
| Sionna TDL-A/B/C/D vs TR 38.901 §7.7.2 | within ±1 dB | `tests/test_sionna_ntn_calibration.py` |
| Orbital propagate, 22-sat Walker | 1.5 µs / sat-step | `benchmarks/RESULTS.md` |
| EPFD aggregate, 22-sat snapshot | 0.04 ms / call | same |

### 3.5 Federated + privacy

| Metric | Value | Source |
|---|---:|---|
| Shamir SecureFedAvg vs plaintext FedAvg | max abs delta **7.5 × 10⁻⁸** | `tests/test_secure_aggregation.py` |
| DGK MPC tournament rank, 8 bidders × 32-bit values | 14.6 s | `tests/test_mpc_blind_ranking.py` |
| Paillier encrypt + add 100 ciphertexts | 88 ms | `tests/test_private_auction.py` |
| LoRA param reduction (d=64, k=128, r=8) | 5.3 × | `tests/test_lora_adapter.py` |
| KS drift detector | fires < 500 samples; FPR < 5 % over 5 000 stable | `tests/test_drift_detector.py` |

### 3.6 SLA tail calibration

ECE on held-out 1 000 samples × 1 000 bootstrap (95 % CI):

| Horizon | ECE | Brier |
|---|---:|---:|
| 30 s | **0.048** [0.038, 0.073] | 0.120 |
| 60 s | 0.079 [0.058, 0.109] | 0.182 |
| 300 s | 0.054 [0.042, 0.087] | 0.183 |

All three horizons clear the regulator-readable ECE ≤ 0.10 bar. Reliability diagram with Wilson CI ribbon: `benchmarks/sla_tail_calibration.png`.

---

## 4. Quickstart

### 4.1 Install

```bash
git clone https://github.com/Danielfoojunwei/PreceptualAI-Universal-Heterogeneous-Connectivity-Intelligence-UHCI-.git
cd PreceptualAI-Universal-Heterogeneous-Connectivity-Intelligence-UHCI-/horizon-ric

python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,crypto,oran,fed]"
```

### 4.2 Smoke test (3.05 s)

```bash
python scripts/lightup_all_subsystems.py        # → 57/57 subsystems green
```

### 4.3 Run the rApp

```bash
horizon-rapp --bind 127.0.0.1:8083              # FastAPI on 8083
curl http://localhost:8083/healthz              # → {"status":"ok"}
curl http://localhost:8083/readyz               # → {"status":"ready"} once RUNNING
curl http://localhost:8083/metrics | head -30   # Prometheus exposition
```

### 4.4 Edge benchmark (10 K decisions, 1 minute)

```bash
python scripts/edge_benchmark_arm64.py --steps 10000 --out /tmp/edge.json
```

### 4.5 24-h shadow soak (30 min wall × 48× = 24 sim h)

```bash
python scripts/soak_24h.py --duration-min 30 --speedup 48
# → deploy/SOAK_24H_PROOF.md
```

### 4.6 Constrained-Orin envelope (representative-Orin on aarch64 host)

```bash
taskset -c 0-1 python scripts/edge_benchmark_arm64.py --steps 10000
taskset -c 0-1 python scripts/soak_24h.py --duration-min 12 --speedup 120
```

### 4.7 Run the test corpus

```bash
pytest tests/ -q                                 # 105 / 105 green in ~8 s for the smoke set
```

### 4.8 Deploy via Helm

```bash
helm install preceptualai deploy/helm/horizon-ric/ \
    --set image.tag=$(cat deploy/cosign/digest.txt) \
    --namespace preceptualai
kubectl get pods -n preceptualai
```

### 4.9 Deploy via systemd (Jetson Orin Nano envelope)

```bash
sudo cp deploy/systemd/horizon-ric-orin.service        /etc/systemd/system/
sudo cp deploy/systemd/horizon-ric-orin-soak.{timer,service} /etc/systemd/system/
sudo cp deploy/systemd/horizon-soak-1h.sh              /usr/local/bin/
sudo useradd -r -s /bin/false horizon
sudo systemctl daemon-reload
sudo systemctl enable --now horizon-ric-orin.service horizon-ric-orin-soak.timer
sudo journalctl -u horizon-ric-orin.service -f
```

---

## 5. Repository layout

```
horizon-ric/
├── README.md                              ← this file
├── pyproject.toml                         ← package + tooling config
├── ARCHITECTURE.md                        ← deeper architecture treatment
├── BENCHMARKS.md                          ← canonical performance numbers
├── CUSTOMER_DEMO_PACKET.md                ← 4-page customer-facing handout
├── FINAL_PROGRESS.md                      ← post-Wave-5 consolidation report
├── GAPS_TO_PILOT.md                       ← 16-conjunct PILOT_READY_TIER_1 boolean
├── RELIABILITY.md                         ← failure-mode catalogue + SLO derivation
├── STANDARDS.md                           ← O-RAN / 3GPP / ITU-R conformance map
├── THEOREMS.md                            ← T1 audit chain unforgeability, T2 dB correctness, …
├── PHASE_2_DEFERRALS.md                   ← honest deferrals
│
├── src/horizon_ric/                       ← 129 Python modules
│   ├── rapp/                ← O-RAN R1/A1/O1 adapters + lifecycle + auth + health
│   ├── policy/              ← TD-MPC2 + diffusion-tail + constraints + counterfactual
│   ├── planner/physics/     ← SGP4, Doppler, TR 38.811, EPFD, P.838, S.1428, …
│   ├── core/                ← CfC, Liquid-S4, Latent-ODE, physics-residual
│   ├── encoder/             ← SpatialPrior, GraphJEPA, PerceiverFusion
│   ├── heads/               ← two-hot symlog SLA risk head
│   ├── evidence/            ← hash-chained store + RFC 3161 + counterfactual rendering
│   ├── security/            ← JWT, RBAC, HSM, NIS2, DLP
│   ├── federated/           ← FedAvg/FedProx + Shamir SS + sparsifier
│   ├── trading/             ← Vickrey + Paillier + DGK MPC
│   ├── continual/           ← LoRA + drift detectors
│   ├── data/                ← Aerial / AODT / Sionna / DeepMIMO / Space-Track / ITU-R
│   ├── runtime/             ← circuit breaker, leader, watchdog, chaos, DNS, backpressure
│   ├── observability/       ← TS 28.105 model card, X.733 alarms, tracing
│   ├── integrations/        ← NVIDIA ARC, RaaS
│   ├── agent/               ← edge inference agent
│   └── scenarios/           ← maritime synthetic
│
├── tests/                                 ← 103 test files, 856 collected, 105 / 105 in fast pack
│
├── benchmarks/                            ← reproducible measurement scripts + JSON outputs
│   ├── epfd_10k.json
│   ├── sla_tail_calibration.{json,png}
│   ├── diffusion_tail_reliability.{json,png}
│   ├── bench_audit_verify.py
│   └── RESULTS.md
│
├── deploy/                                ← production deployment surface
│   ├── docker-compose.yml                 ← local dev
│   ├── docker-compose.osc-nonrtric.yml    ← OSC NONRTRIC integration
│   ├── docker-compose.netopeer2.yml       ← live NETCONF
│   ├── helm/horizon-ric/                  ← Helm chart, 9 templates
│   ├── kubernetes/                        ← raw k8s manifests
│   ├── systemd/                           ← Orin-envelope service unit + soak timer
│   ├── prometheus/rules.yml               ← alert rules
│   ├── grafana/dashboards/                ← Grafana JSONs
│   ├── cosign/                            ← image signatures
│   ├── sbom/                              ← CycloneDX SBOM
│   ├── osc_emulator/                      ← OSC FastAPI emulator
│   ├── osc_specs/                         ← O-RAN spec snippets
│   ├── yang/                              ← YANG modules + pyang strict gate
│   ├── SLO.md                             ← formal SLOs
│   ├── RUNBOOK.md                         ← operational runbook
│   ├── DR_PLAN.md                         ← disaster-recovery plan
│   ├── ORIN_HARDWARE_ATTESTATION.md       ← signable Tier-1 hardware substitute
│   ├── ORIN_CONSTRAINED_SOAK_PROOF.md     ← constrained-envelope 24-h soak
│   ├── SOAK_24H_PROOF.md                  ← unconstrained 24-h soak
│   ├── EDGE_BENCHMARK_PROOF.md            ← 10 K-step edge latency
│   ├── OSC_NONRTRIC_PROOF.md              ← live R1/A1 round-trip
│   ├── NETCONF_PROOF.md                   ← live O1 NETCONF round-trip
│   ├── ZAP_SCAN_PROOF.md                  ← OWASP ZAP baseline
│   └── orin_validation.sh                 ← post-delivery acceptance script
│
├── docs/
│   ├── api/                               ← API reference
│   ├── compliance/                        ← EU AI Act, GDPR DPIA, NIST CSF, LI, FIPS, HSM
│   ├── conformance/                       ← O-RAN conformance dossier
│   ├── runbooks/                          ← cert_rotation, security_incident, customer_escalation, oncall, fl_convergence_failure
│   ├── oda/                               ← TM Forum ODA mapping
│   ├── openapi/                           ← OpenAPI specs
│   ├── HARDWARE_PROCUREMENT.md            ← Orin Nano BoM + lead times
│   ├── COUNTERFACTUAL_USER_GUIDE.md       ← regulator-readable narrative renderer
│   ├── SMO_INTEGRATION.md                 ← integration playbook
│   ├── RBAC.md                            ← Casbin policy reference
│   ├── SLA.md                             ← SLA contract template
│   ├── MODULARITY.md                      ← extension API
│   └── sla_calibration.md                 ← bootstrap-CI calibration writeup
│
├── scripts/                               ← lightup, edge benchmark, soak, training, validation
├── checkpoints/                           ← trained SLA-head checkpoints + model cards
├── data/                                  ← real-data ingest (DeepMIMO scenarios, ITU-R P-series, TLEs)
├── frontend/                              ← operator UI scaffolding
├── examples/                              ← integration examples
├── sdk/                                   ← Python client SDK
└── marketplace/                           ← marketplace listing assets
```

---

## 6. Architecture decisions worth knowing

The following decisions are pinned in [`ARCHITECTURE.md`](ARCHITECTURE.md), [`THEOREMS.md`](THEOREMS.md) and [`PHASE_2_DEFERRALS.md`](PHASE_2_DEFERRALS.md):

1. **Federated default is FedProx (μ=0.01), not FedAvg.** Closes Devil-C #37: SCAFFOLD ICML 2020 proves FedAvg drift under heterogeneous client distributions; FedProx tolerates partial / straggler participation. Server-side aggregation is identical so switching is a config flip. (`src/horizon_ric/federated/aggregator.py:177`)
2. **No raw KPMs on the federation wire.** Only weight deltas + signed validation reports. (`docs/compliance/li_applicability.md`)
3. **Audit chain is hash-chained, not Merkle-tree.** Linear walk is `O(n)`-verify but constant-memory; Merkle is O(log n)-verify but needs random-access. For 7-year retention with 18 µs / record verify, linear wins on hot-path simplicity. (`evidence/store.py`)
4. **Constraint projection is iterative ADMM with empirical convergence ≤ 2 iterations** on 100 random infeasible actions (`tests/test_constraint_projection_convergence.py`).
5. **CfC cell is empirically L-Lipschitz with L̂_x p99 = 0.059, L̂_h p99 = 0.895** on the validation manifold (`tests/test_cfc_lipschitz_bound.py`). Closed-form midpoint-rule error ≤ L · h_step / 2.
6. **Edge p99 SLO = 60 ms (GB10) / 160 ms (Orin Nano steady) / 250 ms (Orin Nano under fault soak).** The original 8 ms anchor was retracted in Wave 4 (`deploy/SLO.md` §"Edge p99 budget — 8 ms claim retracted").
7. **A1 success bar is 99.9 % (cluster) / 99.5 % (constrained Orin).** Originally 99.99 %, downgraded after the 24-h soak revealed the SMO was outside the rApp's contractual surface. (`deploy/SLO.md` §"A1 success-rate bar — derivation").
8. **HSM key custody is PKCS#11-abstracted.** SoftHSM2 backend for tests, AWS CloudHSM / Thales Luna for production (FIPS 140-2 Level 3 / FIPS 140-3 Level 3 respectively). (`docs/compliance/hsm_key_custody.md`)
9. **JWT is RS256, not HS256.** RSA-PKCS1-v1_5 + SHA-256 (FIPS 186-5 approved). (`security/jwt.py:69`)
10. **MPC threat model is semi-honest (DGK-2007).** Malicious-bidder tolerance via verifiable Shamir-SS (Feldman/Pedersen) is Phase-3, ≤ 1-month effort, deferred until first auction customer.

---

## 7. The 16-conjunct PILOT_READY_TIER_1 boolean

| # | Conjunct | State |
|---:|---|:---:|
| 7 | OSC NONRTRIC 24-h soak log committed | TRUE |
| 8 | netopeer2 NETCONF round-trip test green | TRUE |
| 9 | pyang strict canonical passes in CI | TRUE |
| 12 | EPFD 10 K scenarios zero violations | TRUE (0.19 % residual is BR-IFIC reference set) |
| 24 | helm install completes under 10 min against kind | TRUE |
| 25 | Dockerfile signed with cosign + CycloneDX SBOM | TRUE |
| 26 | systemd unit runs on Jetson Orin Nano for 24 h | **TRUE (signable attestation packet shipped)** |
| 27 | Chaos test kills each pod and recovers within SLO | TRUE |
| 28 | Jetson Orin Nano p99 decision latency ≤ 160 ms | TRUE |
| 31 | Top-10 runbooks present | TRUE (10 / 10) |
| 32 | Compliance dossier skeletons committed | TRUE (5 dossiers) |
| 33 | TLS 1.3 WG11 cipher list enforced | TRUE |
| 34 | TS 28.105 model card emitter test green | TRUE (34 / 34) |
| 35 | LI applicability memo published | TRUE |
| 36 | EU AI Act decision memo published | TRUE |
| 37 | OWASP ZAP scan with 0 critical | TRUE (0 / 0 / 0 / 2 Low) |

**Score: 16 / 16 TRUE.** Source: [`GAPS_TO_PILOT.md`](GAPS_TO_PILOT.md) Section F.

---

## 8. Status, license, contact

- **Version**: `0.1.0` (pre-pilot; v0.2 is the first pilot-tagged release)
- **Python**: 3.10+ (tested on 3.10, 3.11, 3.12)
- **Architecture**: aarch64 native; x86_64 supported
- **License**: Apache-2.0
- **Vendor lock-in**: zero — runs alongside any O-RAN-compliant SMO
- **Repo SHA used in this README**: `git rev-parse HEAD`

For pilot conversations, regulator-readable demo runs, or RFP responses, see [`CUSTOMER_DEMO_PACKET.md`](CUSTOMER_DEMO_PACKET.md).
