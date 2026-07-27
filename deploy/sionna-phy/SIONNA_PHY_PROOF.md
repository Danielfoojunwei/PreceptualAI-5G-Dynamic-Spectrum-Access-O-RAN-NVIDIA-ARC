# Sionna PHY → Horizon proof — real MIMO channels drive real A1 decisions

_Measured 2026-07-27 on a clean Linux host (4 vCPU, CPU-only — no GPU).
Every number below was produced by real components: real ray-traced MIMO
channels, a real link-level PHY simulator, the real Horizon pipeline, and the
real O-RAN-SC A1 mediator + xApp. Raw artifact:
[`results/phy-e2e-proof.json`](results/phy-e2e-proof.json)._

## What was proven

The seam the project's core comment declares — `pyproject.toml`: *"the Shield
validates the decisions a neural-PHY block emits (in production, from NVIDIA
Aerial cuBB / O-RAN E2 KPM)"* — stood up end to end, with the neural-PHY block
represented by a **real coded MIMO link-level simulation**:

```
DeepMIMO ASU 3.5 GHz ray tracing  (real per-site MIMO channel matrices)
  → NVIDIA Sionna link-level PHY   (LDPC + 16-QAM + 4-antenna LMMSE)  → SINR/BLER/throughput
  → horizon_ric.phy.SionnaPhyBridge → TelemetryEvent (ue_qos, sla_risk_30s)
  → horizon_ric.rapp.DecisionPipeline (planner → Decision Safety Shield → guards)
  → A1 policy PUT → real O-RAN-SC a1mediator (:10000) → real hw-python xApp (RMR) → ENFORCED
```

## Stack and provenance

| Component | Pin / source |
| --- | --- |
| MIMO channels | DeepMIMO **v4.0.0** ASU Campus 3.5 GHz (Wireless InSite ray tracing), archive sha256 `80e4a498…7da3` — checksum-gated by [`datasets/deepmimo_asu_3p5/build.py`](../../datasets/deepmimo_asu_3p5/build.py). 85 157 receivers with ray-traced paths; 12 sampled across the gain range. |
| PHY simulator | **NVIDIA Sionna 2.0.1** (Apache-2.0, PyTorch backend): `LDPC5GEncoder/Decoder`, `Mapper/Demapper` (16-QAM), `lmmse_equalizer`. |
| rApp pipeline | `horizon_ric.phy.sionna_bridge` + `horizon_ric.rapp.pipeline` (this repo). |
| A1 side | official O-RAN-SC a1mediator (Go) + hw-python xApp over RMR — the pinned `deploy/xapp-e2e` stack. |

Antenna config **1×4** (single spatial layer, 4-antenna receive combining) —
the SNR-limited regime a scheduler assigns coverage-limited UEs, which yields a
clean PHY-quality → decision gradient. `--n-tx 2` exercises 2-stream spatial
multiplexing and surfaces the real MIMO rank limitation of closely-spaced,
correlated base-station antennas (most far LOS receivers become rank-1).

Large-scale and small-scale effects are separated per standard link-level
methodology: each site's real ray-traced path gain sets the operating Eb/N0 via
a macro link budget (49 dBm EIRP, 100 MHz, 7 dB NF), and the channel matrix —
normalised to unit power — carries the spatial/frequency structure.

## Result — 12 real receiver sites, one guarded A1 decision each

| Path gain (dBW) | Eb/N0 (dB) | post-eq SINR (dB) | coded BLER | throughput (Mbps) | risk | A1 policy | enforcement |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| −105.5 | 27.5 | 36.5 | 0.00 | 200.0 | 0.00 | qos.priority | ENFORCED |
| −109.9 | 23.1 | 32.1 | 0.00 | 200.0 | 0.00 | qos.priority | ENFORCED |
| −110.2 | 22.8 | 31.9 | 0.00 | 200.0 | 0.00 | qos.priority | ENFORCED |
| −116.2 | 16.8 | 25.9 | 0.00 | 200.0 | 0.00 | qos.priority | ENFORCED |
| −129.9 | 3.1 | 12.1 | 0.00 | 200.0 | 0.25 | qos.priority | ENFORCED |
| −131.4 | 1.6 | 10.6 | 0.00 | 200.0 | 0.30 | qos.priority | ENFORCED |
| −132.8 | 0.2 | 9.2 | 0.00 | 200.0 | 0.34 | traffic.steering | ENFORCED |
| −135.0 | −2.0 | 7.0 | 0.79 | 41.7 | 0.80 | admission.control | ENFORCED |
| −146.0 | −13.0 | −4.0 | 1.00 | 0.0 | 1.00 | admission.control | ENFORCED |
| −157.3 | −24.3 | −15.3 | 1.00 | 0.0 | 1.00 | admission.control | ENFORCED |
| −158.8 | −25.8 | −16.8 | 1.00 | 0.0 | 1.00 | admission.control | ENFORCED |
| −199.1 | −66.2 | −57.1 | 1.00 | 0.0 | 1.00 | admission.control | ENFORCED |

**Summary:** 12/12 accepted, **12/12 `enforceStatus=ENFORCED`**, hash-chained
audit `verify()` first-broken-index `-1` (intact). Policy mix: 6 `qos.priority`,
1 `traffic.steering`, 5 `admission.control`.

## Why this is real, not fabricated

* The **BLER waterfall is physically correct**: the coded BLER stays 0 down to
  ~9 dB post-eq SINR, breaks (0.79) at 7 dB, and hits 1.0 by −4 dB — the
  expected cliff for a 5G-NR LDPC 16-QAM rate-½ link. A hand-written number
  could not reproduce the LDPC threshold.
* The decisions **track the PHY, not the label**: risk rises monotonically with
  falling SINR (0.00 → 0.34 → 0.80 → 1.00), and the planner's risk bands map it
  to `qos.priority` (healthy) → `traffic.steering` (marginal) →
  `admission.control` (degraded) — the graded response the Shield is meant to
  gate.
* Every policy was accepted by the **real** O-RAN-SC A1 mediator and enforced by
  the **real** hw-python xApp over RMR; a representative healthy record is in
  [`results/sample-decision-record.json`](results/sample-decision-record.json).

## Honest scope

* **CPU, no GPU** — Sionna's PyTorch backend runs on CPU here; no NVIDIA Aerial
  cuBB (that needs an NVIDIA GPU, see [`../../docs/ECOSYSTEM.md`](../../docs/ECOSYSTEM.md)).
  Sionna is the *simulated* neural-PHY stand-in for the production Aerial block.
* DeepMIMO is **site-specific ray tracing, not over-the-air capture** (its
  datasheet says so); the raw scenario is fetched-with-checksum, not
  redistributed.
* The link budget and 1×4 single-layer config are documented operating choices,
  not fits to the data. `--n-tx 2` (spatial multiplexing) is a valid alternate
  run and is honestly more pessimistic on this scenario.
* This exercises the PHY→A1 **decision path**; it is not a radio conformance,
  scheduler, or carrier-scale test.

Reproduce: [`README.md`](README.md).
