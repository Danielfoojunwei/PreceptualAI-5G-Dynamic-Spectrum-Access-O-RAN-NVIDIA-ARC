# Horizon-RIC — an AI-RAN Security & Trust rApp

> O-RAN Non-RT-RIC **rApp** that makes AI-RAN decisions **safe and auditable by
> construction**: a poisoned, drifted, or adversarial neural-PHY/RIC model still
> cannot emit an unsafe or illegal radio policy.

**Status:** torch-free, installs on stock Python 3.10+ with no accelerator ·
**349 tests green in CI** (ruff + mypy + pytest + docker + yang-strict) ·
benchmark + threat model committed · Apache-2.0.

This README is also the AI-RAN Alliance *Call for Innovation* proposal.

---

## Executive summary

The AI-RAN industry is shipping neural physical-layer blocks (neural receivers,
learned constellations, DPoD) and RIC control models onto live spectrum. The
open problem is **not** capability — it is **trust**: when an AI model is
poisoned, drifted, or adversarially driven, what stops it from steering a cell
out of band, over its power licence, or into an illegal waveform — and how does
an operator *prove* to a regulator and a CFO what the model did?

**Horizon-RIC is a runtime-assurance + audit layer for AI-RAN decisions.** It
does not train or run the model; it sits beside the SMO/Non-RT RIC and wraps
every decision the model emits in four mechanisms:

1. **Decision Safety Shield** — checks each proposed action against RAN-physics,
   spectrum-regulatory, AI-PHY, and lawful-intercept invariants, and **projects
   it to a safe action or a certified classical fallback**, emitting a signed
   `SafetyCertificate`.
2. **At-decision-time evidence** — a SHA-256 hash-chained, RFC-3161-anchored
   `DecisionRecord` with a replayable counterfactual.
3. **Model provenance** — a signature over `(weights ‖ training-manifest)`,
   verified on promotion.
4. **Robust federated aggregation** — Krum / median / trimmed-mean against
   model-poisoning clients, plus Shamir secure aggregation for update privacy.

**Measured result (`benchmarks/results/poisoning_shield.json`):** on a
10,000-decision stream that is 30 % poisoned, an unguarded emit path would put
**2,366 illegal policies** on the air interface; with the Shield, **0**. Under a
Byzantine federation, the global-model distance from the honest mean drops from
**202** (FedAvg) to **8–12** (robust aggregators).

---

## Problem statement and market relevance

**The trust gap is the binding constraint on AI-RAN deployment, not the
technology gap.**

- **Regulatory.** The EU AI Act classifies AI in critical infrastructure as
  high-risk (Art. 6 + Annex III → Art. 9–15 controls); NIS2 Art. 23 mandates
  24-hour incident notification; Ofcom's 2025/26 AI approach requires
  explainability "sufficient to support post-incident regulatory review."
- **Commercial.** Dell'Oro (2025) projects Tier-1s will **under-deploy AI-RAN by
  30–40 % in 2026–2027** because capex follows the *trust* curve, not the
  technology curve. The decision-level evidence layer — counterfactual + tamper-
  evident audit + safe rollback — is empty in today's SMOs.
- **Security.** O-RAN WG11 added ~39 AI/ML-specific threats (data/model
  poisoning, adversarial input, model inversion/extraction, supply-chain) to its
  Threat Model in 2024 (OWASP-ML + ENISA); 3GPP TR 33.898 studies AI/ML security
  for the RAN. These threats currently have **no shipped rApp-level control**.

**Market:** every operator deploying AI-RAN neural-PHY (on NVIDIA Aerial, Nokia
MantaRay, Ericsson, VIAVI) needs this horizontal trust layer to pass procurement,
regulatory sign-off, and CFO attribution. It is complementary to — not
competitive with — the vendors shipping the AI capability.

---

## Innovative solution and technical approach

The core idea is **AI-RAN-native runtime assurance**: bind the AI's *output* to
the RAN's own physical and regulatory invariants, at the RAN's own timescales.

**The Shield** (`src/horizon_ric/shield/`) — `Shield.dispose(action) →
(safe_action, SafetyCertificate)` runs an ordered invariant chain:

| Invariant | Basis | Action on violation |
|---|---|---|
| Lawful intercept | 3GPP TS 33.127, **fail-closed** | refuse emit if LI scope unknown |
| Spectral mask | 3GPP TS 38.104 | clip carrier inside the licensed channel |
| Max EIRP / Tx power | block-edge / licence limit | reduce power to the ceiling |
| Neural-RX envelope | TBLER vs classical LMMSE baseline + demap confidence | **fall back to the certified classical receiver** |
| Constellation legality + PAPR | legal M-QAM orders, PAPR ceiling | snap to nearest legal order / classical QAM |

Because the gate is mathematical and **independent of the model**, a
poisoned/backdoored/adversarial model cannot emit an unsafe or illegal policy;
every disposition yields a certificate recording the invariants checked, the
margin to each bound, and any fallback — *evidence by construction*.

**The trust chain runs model → decision → evidence end-to-end:**
`provenance/` signs and verifies the model artefact (RSA-PSS over weights ‖
manifest, HSM-held key); `federated/robust.py` bounds a poisoning client's
influence (Krum / median / trimmed-mean); `federated/secure.py` hides individual
updates (Shamir `(t,n)` over GF(2¹²⁷−1)); `evidence/` hash-chains every
`DecisionRecord` and anchors the chain head to **real public RFC-3161 TSAs**
(freetsa, DigiCert) with nonce + imprint verification.

What's novel: not the individual crypto, but **composing a verified safety
envelope + at-decision-time, loop-tiered, replayable evidence around the
specific AI-PHY blocks the AI-RAN Alliance is standardising** — the security
control most vendors lack.

---

## Deployment feasibility

- **Drops in beside any O-RAN SMO.** Real R1 (registration), A1 (policy emit,
  OSC/EIAP/MantaRay dialects), and O1 (NETCONF/YANG via `ncclient`) adapters,
  behind circuit breakers, mTLS, OAuth2, and Casbin RBAC with tenant domains.
- **No accelerator, no lock-in.** Torch-free (numpy + pydantic); the Shield
  validates the *data* a neural block emits (in production, from NVIDIA Aerial
  cuBB / O-RAN E2 KPM), so it never needs to run the model. `pip install -e .`
  completes in seconds on a stock host.
- **Runs where AI-RAN runs.** Three shapes: Helm-on-Kubernetes, systemd
  edge-envelope (aarch64), or bare-metal Docker. Prometheus `/metrics`, Grafana
  dashboards, RFC-3161-anchored JSONL/SQLite evidence store.
- **Operationally safe.** Hash-chained audit, atomic A→B model promotion with
  bit-identical rollback, graceful degradation, systemd watchdog, NIS2 reporter.

```bash
pip install -e ".[dev]"
pytest tests/ -m "not integration and not slow" -q          # 349 passed
python benchmarks/poisoning_shield_benchmark.py --decisions 10000 --poison-rate 0.30
horizon-rapp --once                                          # boot + readiness smoke
```

---

## 12-month timeline with milestones

| Months | Milestone | Output |
|---|---|---|
| **M1–2** | Per-band invariant calibration — pin TS 38.104 emission masks and EIRP limits for FR1/FR3 target bands; live R1/A1 against OSC NONRTRIC. | Calibrated Shield configs + conformance run |
| **M3–4** | Strengthen FL security — verifiable secret sharing (Feldman/Pedersen) for *malicious*-server resistance; differential-privacy accountant for membership-inference/inversion. | Upgraded `federated/`; closes two GAP-register items |
| **M5–6** | Live AI-PHY integration — shield a real neural receiver (e.g. HybridDeepRx) and learned constellation on NVIDIA Aerial cuBB / ARC-OTA telemetry. | End-to-end demo against real I/Q |
| **M7–8** | Loop-tiered evidence — dApp (sub-ms sampled), Near-RT (per-decision), Non-RT (full counterfactual); inference-API extraction rate-limiting. | Evidence architecture + extraction defence |
| **M9–10** | Operator/testbed pilot — deploy alongside an SMO; 24-h soak; production PKCS#11 HSM custody; conformance dossier. | Pilot report + signed attestation packet |
| **M11–12** | Standardization + public benchmark — contribute the AI/ML threat→control mapping to O-RAN WG11; release the poisoning/robustness benchmark + dataset. | Standards contribution + public benchmark |

---

## Expected deliverables and impact

**Deliverables**
- A production-grade, open-source **security & trust rApp** (this repo).
- The **Decision Safety Shield** algorithm with safety certificates.
- **Robust + secure federated aggregation** algorithms.
- **Benchmarking-ready code + committed results** (`benchmarks/`).
- A **poisoned-AI-PHY-decision dataset generator** for reproducible evaluation.
- A **threat-model → control mapping** (`docs/THREAT_MODEL.md`) suitable as a
  standardization input.

**Impact**
- **Safety:** illegal air-interface emits from a 30 %-poisoned decision stream
  reduced from 2,366 → **0** (10k-decision benchmark).
- **Poisoning resilience:** Byzantine pull on the global model cut **~20×**.
- **Deployability:** unblocks the 30–40 % AI-RAN under-deployment Dell'Oro
  attributes to the trust gap, by giving operators the regulator- and
  CFO-defensible evidence layer their SMO lacks.

### Expected outputs (mapping to the Call)

| Call output | What we provide |
|---|---|
| **Prototypes** | The runnable rApp + R1/A1/O1 adapters + `horizon-rapp` daemon. |
| **Algorithms or models** | The Shield + invariants; Krum/median/trimmed-mean; Shamir secure aggregation; RSA-PSS model-provenance. |
| **Benchmarking-ready code** | `benchmarks/poisoning_shield_benchmark.py` + committed `results/`. |
| **Datasets** | A synthetic poisoned-AI-PHY-decision trace generator (in the benchmark); consumes real NVIDIA Aerial / E2 telemetry in deployment. |
| **Standardization contributions** | `docs/THREAT_MODEL.md` mapping O-RAN WG11 / OWASP-ML / TR 33.898 threats to controls, with an honest GAP register. |

---

## Honest status and gaps

Real, not mocked: the Shield, provenance signing/verify, robust/secure
aggregation, hash-chained evidence store, RFC-3161 TSA anchoring (real public
TSAs), and the R1/A1/O1 wire adapters are all working code with tests.

Known limitations (also in `docs/THREAT_MODEL.md`): invariant thresholds must be
pinned per licensed band; the malicious-server FL case needs verifiable secret
sharing (M3–4); no DP accountant yet; inference-API extraction rate-limiting is
roadmap; HSM custody ships with in-memory (RSA-2048) and SoftHSM2 (PKCS#11)
backends — production CloudHSM/Luna wire in through the same PKCS#11 path.

## AI-RAN Alliance positioning

An **AI-for-RAN** security & trust contribution: an open innovation + benchmark
against the O-RAN WG11 / OWASP-ML AI/ML threat surface. The normative security
specification is owned by O-RAN WG11 / 3GPP; this project is a reference
implementation and benchmark, not a competing standard.

---

Architecture, controls, and the full threat model: [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).
License: Apache-2.0.
