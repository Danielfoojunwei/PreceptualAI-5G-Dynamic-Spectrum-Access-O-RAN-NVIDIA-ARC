# PreceptualAI — the audit-first, NTN-native rApp that runs alongside your existing SMO

_Marketplace whitepaper, May 2026. Audience: CTO, Network Architect, Regulatory Officer, Procurement._

---

## 1. The 99.999% reliability problem in AI-driven RAN

The AI-RAN industry has spent 2024 and 2025 turning the slide-deck phrase "AI-native network" into deployed software. Ericsson + MasOrange went live with EIAP-driven autonomous operations ([2025 press release](https://www.ericsson.com/en/press-releases/3/2025/ericsson-and-masorange-advance-autonomous-networks-with-ai-driven-automation-platform-and-rapps)). Nokia introduced MantaRay SON to NTT DOCOMO's multi-vendor 5G core in [November 2025](https://www.globenewswire.com/news-release/2025/11/25/3194000/0/en/Nokia-introduces-MantaRay-SON-to-NTT-DOCOMO-s-multi-vendor-5G-network.html), and Nokia + du staged the industry's first autonomous network slicing in [December 2025](https://www.globenewswire.com/news-release/2025/12/03/3198586/0/en/Nokia-and-du-set-new-benchmark-in-5G-innovation-with-autonomous-network-slicing-in-industry-first.html). NVIDIA + Nokia struck a strategic partnership to put GPU-accelerated AI-RAN at cell sites ([Frank Rayal analysis, Nov 2025](https://frankrayal.com/2025/11/03/nvidia-and-nokia-a-strategic-bet-on-edge-ai-ran/)).

The RAN community has, in other words, decided that AI is real. What it has not yet decided is what to do when the AI is wrong, when the regulator asks "why did you make that decision at 03:14 UTC", or when an LEO satellite passes overhead and the operator is suddenly inside the EPFD scope of ITU-R S.1503.

The "five nines" RAN community has 30 years of experience auditing deterministic rule books. It has roughly five years of experience auditing learned policies. The mismatch between what AI rApps emit and what regulators consume is the second-order problem nobody is shipping a product for.

PreceptualAI is that product.

---

## 2. The auditability gap nobody fills

Ericsson's [rApp Directory](https://www.ericsson.com/en/portfolio/cross-portfolio/rapp-directory), Nokia's [rApp Marketplace](https://www.nokia.com/mobile-networks/ran-operations/mantaray-smo/), the [TIP Exchange](https://exchange.telecominfraproject.com/marketplace), and the [O-RAN-SC reference rApps](https://wiki.o-ran-sc.org/) are populated with policies that decide things — load balancing, energy savings, traffic steering, anomaly detection. None of the public datasheets we have searched describe an rApp output that ships with all four of:

1. The chosen action and its predicted KPI envelope.
2. The top-K **rejected alternatives** with structured machine-readable rejection reasons (gateway overload, SLA breach predicted, EPFD constraint violation, …) **and** human-readable text a non-engineer can read.
3. A **SHA-256-chained tamper-evident DecisionRecord log** so a single edited byte is immediately visible.
4. A **3GPP TS 28.105 model card** binding every emitted decision to a versioned, sha256-pinned model artifact that is shippable to a regulator.

That is the gap. PreceptualAI fills it. Every A1 policy emitted by PreceptualAI carries the four-tuple above. The audit chain verifies in **17.66 ms over 1000 records** on commodity hardware (`benchmarks/bench_audit_verify.py`). The model cards are in `checkpoints/*.md`, gated by `tests/test_ts28105_model_card_emit.py`. The counterfactual envelope schema is in `src/horizon_ric/evidence/schema.py`.

When TM Forum's Level-4 autonomous-network certification or a national regulator asks "show me you did not edit the audit log", the answer is a single command. We have not found a competitor system that exposes this as a contract.

---

## 3. NTN-native compliance (ITU-R Article 22, S.1503-3, TR 38.811)

The 3GPP NTN program (Releases 17 → 19) and ITU-R Article 22 / S.1503 EPFD scope put a new class of constraint on top of the rApp: when an LEO satellite emitter is in your radio scene, you have a regulatory obligation, not just an SLA obligation. Existing SMO vendors treat NTN as a separate gNB-layer product line — Mavenir's NTN portfolio sits next to its RIC product line ([Mavenir NTN](https://www.mavenir.com/portfolio/mavair/non-terrestrial-network-ntn/), [Mavenir RIC](https://www.mavenir.com/portfolio/mavscale/ai-analytics/ran-intelligent-controller-ric/)). Ericsson's NTN work is at the [payload-architecture](https://www.ericsson.com/en/blog/2024/10/ntn-payload-architecture) and Qualcomm/Thales [milestone](https://www.ericsson.com/en/press-releases/3/2025/ericsson-qualcomm-thales-achieve-space-connectivity-milestone) level — not at the rApp.

PreceptualAI is NTN-first. Inside the policy decision loop:

- **ITU-R P.525, P.618, P.676, P.838** for free-space loss, slant-path rain attenuation, gaseous absorption — `src/horizon_ric/planner/physics/propagation.py`.
- **ITU-R S.1428** beam-cap pattern for NGSO emitters — `src/horizon_ric/planner/physics/s1428.py`.
- **ITU-R S.1503 Annex 1 Eq.(1) EPFD aggregate** — `src/horizon_ric/planner/physics/epfd.py:91`. Snapshot benched at **0.04 ms / call** (`RESULTS.md:epfd_aggregate_22sat_snapshot`); 1-hour 500-sat Walker time-CDF at **0.19 s** (`RESULTS.md:epfd_time_cdf_500sat_1h`).
- **3GPP TR 38.811 NTN propagation timing** — `tests/test_tr38811_ntn_timing.py`.
- **Doppler-shift pass envelopes** — `src/horizon_ric/planner/orbital/`, benched at 0.80 ms for a 600 s pass (`RESULTS.md:doppler_pass_envelope_600s`).
- **Walker / orbital propagation** — 1.5 µs / sat-step on a 22-sat constellation (`RESULTS.md:orbital_walker_22sat_propagate`).

Reference EPFD validation tooling exists ([Transfinite Visualyse-EPFD](https://www.transfinite.com/content/ValidationSoftwareforRecS1503), [ITU EPFD support](https://www.itu.int/epfdsupport/)) but it is **filing-time** — designed to validate ITU paperwork before a satellite filing. We do not know of a public SMO or RIC that runs S.1503 in-loop as an rApp constraint at policy-decision time. PreceptualAI does.

The O-RAN ALLIANCE NTN deployments whitepaper of [April 2025](https://mediastorage.o-ran.org/ecosystem-resources/O-RAN-2025.04.02.WP.O-RAN_NTN_Deployments-v08.4.pdf) sets out the architecture; PreceptualAI is the rApp surface for that architecture.

---

## 4. Architecture sketch

PreceptualAI is an rApp + Non-RT/Near-RT pair, not a SMO. We bolt onto your SMO over the standard O-RAN interfaces.

```
                    +---------------------------------------------+
                    |  Operator SMO (Ericsson EIAP / Nokia        |
                    |  MantaRay / NVIDIA-coupled SMO / O-RAN-SC)  |
                    +-----+--------------+-----------+------------+
                          |              |           |
                       R1 |           A1 |        O1 | (NETCONF/YANG TS 28.541)
                          v              v           v
                    +---------------------------------------------+
                    |                PreceptualAI rApp              |
                    |                                             |
                    |   Encoder (Perceiver fusion + GraphJEPA)    |
                    |              |                              |
                    |              v                              |
                    |   World model (latent dynamics + diffusion  |
                    |   tail) + physics layers (P.525/P.618/      |
                    |   P.676/P.838/S.1428/S.1503 EPFD)           |
                    |              |                              |
                    |              v                              |
                    |   Planner (TD-MPC) -> chosen action +       |
                    |   top-K rejected alternatives               |
                    |              |                              |
                    |              v                              |
                    |   Constraint layer (hard-project against    |
                    |   ITU-R S.1503 + 3GPP timing budgets)       |
                    |              |                              |
                    |              v                              |
                    |   Counterfactual envelope                   |
                    |   (machine + human reasons)                 |
                    |              |                              |
                    |              v                              |
                    |   SHA-256 chained DecisionRecord store      |
                    |   (per-tenant, JSONL or SQLite)             |
                    +---------------------------------------------+
                                |
                                v
                    +---------------------------------------------+
                    | Signed evidence bundle (cosign + CycloneDX) |
                    | Replayable against TS 28.105 model cards    |
                    +---------------------------------------------+
```

We are SMO-agnostic by design. The R1 surface is in `src/horizon_ric/rapp/r1_adapter.py`, A1 in `a1_adapter.py`, O1 in `o1_adapter.py`. Federated learning across operator sites is **weights-only** — the data never leaves the site (`src/horizon_ric/federated/`).

---

## 5. Performance and benchmarks

All numbers below come from `benchmarks/RESULTS.md` (12-bench sweep, 11 PASS / 1 WATCH / 0 FAIL) and `benchmarks/bench_audit_verify.py`, both reproducible from this repo (`.venv/bin/python benchmarks/run_benchmarks.py`). Methodology: 1–2 warm-up + 3–20 measured iterations, median + p95 reported.

| Capability | Median | p95 |
|---|---|---|
| End-to-end decision (stages 2→4, no HTTP) | 33.36 ms | 33.36 ms |
| EPFD aggregate snapshot (22 NGSO emitters) | 0.04 ms | 0.04 ms |
| EPFD 1-hour time-CDF (500-sat Walker) | 0.19 s | 0.19 s |
| Doppler pass envelope (600 s) | 0.80 ms | 0.80 ms |
| Orbital Walker propagate (22-sat, 1 step) | 1.5 µs | 1.5 µs |
| Constraint full-check + project (22 emitters) | 0.06 ms | 0.06 ms |
| TD-MPC plan (H=12, 64 samples, 3 iters) | 175.41 ms | 182.74 ms |
| Latent-dynamics rollout (B=256, H=12) | 64.38 ms | 78.39 ms |
| Diffusion tail-sample (n=32, 20 steps) | 35.75 ms | 61.54 ms |
| Perceiver fusion forward (256×128) | 10.33 ms | 10.95 ms |
| **Audit verify, 100 records** | **1.78 ms** | 1.78 ms |
| **Audit verify, 1000 records** | **17.66 ms** | 17.66 ms |
| Audit append, 1000 records | 1681.71 ms | — |

Full per-vendor head-to-head and the "what is published / what is not" breakdown is in [BENCHMARK_HEAD_TO_HEAD.md](BENCHMARK_HEAD_TO_HEAD.md). The honest summary: PreceptualAI sits inside the standard Near-RT RIC budget envelope ([STL Partners — 10 ms-1 s](https://stlpartners.com/articles/network-innovation/ric-xapps-rapps-who-are-the-key-players/)), and on the auditability + EPFD axes our published numbers exist where competitor numbers do not.

---

## 6. Integration story — Ericsson EIAP, Nokia MantaRay, NVIDIA Aerial = SMOs; we are an rApp on top

We do not compete with EIAP, MantaRay, or NVIDIA Aerial. They are the SMO and the PHY accelerator; PreceptualAI is an rApp that runs on top.

- **Ericsson EIAP**: PreceptualAI plugs in as an R1-conformant rApp. EIAP brings the rApp lifecycle, the rApp Directory, the operator relationships and the SLA-backed support contract; PreceptualAI brings the audit chain, the counterfactual envelope, and the NTN physics. The two compose. EIAP's [O1 article](https://telecomdrive.com/how-ericsson-is-reinventing-network-management-with-eiap-o1-interface-8679562/) describes the orchestration story we plug into.

- **Nokia MantaRay**: same pattern. MantaRay is now [Open RAN R1-compliant](https://the-mobile-network.com/2025/06/nokia-evolves-mantaray-to-open-ran-smo-solution/). PreceptualAI sits as an rApp instance under MantaRay's directory, with our DecisionRecord chain stored next to MantaRay's own logs. Nokia + HPE Juniper-RIC bundle is one possible deployment ([SDxCentral](https://www.sdxcentral.com/news/nokia-picks-hpes-juniper-ric/)).

- **NVIDIA Aerial / ARC / AI-RAN Alliance**: NVIDIA Aerial is a gNB SDK ([cuBB SDK](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)) and an in-cell-site GPU compute substrate ([NVIDIA ARC-Compact](https://developer.nvidia.com/blog/deploy-ai-ran-at-cell-sites-with-nvidia-arc-compact/)). PreceptualAI does not touch the PHY. We consume FAPI events from cuBB as encoder input (see `data/aerial/` parquet replay path in `src/horizon_ric/data/aerial.py`), and we run our own decision loop on a Jetson Orin Nano next to the cell site.

- **Mavenir RIC, VIAVI TeraVM, Capgemini RATIO**: equivalent integration shape. VIAVI is the test platform that validates the rApp; we expect to run our rApp through TeraVM AI RSG ([2025 announcement](https://www.viavisolutions.com/en-us/news-releases/viavi-introduces-teravm-ai-ran-scenario-generator)) on an operator pre-prod lab.

The integration story is **and**, not **or**. The operator does not throw out their SMO investment to use us; they bolt us on as one rApp instance, get the audit + NTN compliance, and keep every other vendor in place.

---

## 7. 90-day pilot path

`PILOT.md` defines a customer-verifiable 90-day pilot with three contractual outcomes:

1. **SLA-attainment uplift ≥ 8 %** on the pilot cluster vs the rule-based baseline, measured in the customer's own dashboards over the last 30 days of the pilot.
2. **Zero ITU-R / 3GPP constraint violations** across the full pilot window, including a deliberate-bad-action red-team on day 60 — verified by replaying the evidence store and confirming the constraint-layer rejected the action.
3. **End-to-end p99 decision latency ≤ 8 ms on a Jetson Orin Nano** under the pilot's actual traffic mix.

The customer keeps the hash-chained evidence file regardless of outcome. The kill-switch (`src/horizon_ric/rapp/lifecycle.py`) reverts to the existing rule-based RIC in seconds. The pilot pack covers CTO, Network Architect, Regulatory Officer, and Procurement personas.

---

## 8. Open source + open weights advantage

Every named SMO competitor ships proprietary binaries. We ship:

- Source code, this repo.
- Model weights (`checkpoints/*.pt`).
- Model cards (`checkpoints/*.md`, canonical TS 28.105 §7 block + sha256 + training-data manifest + metrics).
- SBOM (CycloneDX 1.5, `deploy/sbom/horizon-ric-sbom.json`).
- Cosign signatures (`deploy/cosign/cosign.pub`, `.github/workflows/build.yml`).
- Reproducible benchmarks (`benchmarks/run_benchmarks.py`).

NVIDIA's [aerial-cuda-accelerated-ran](https://github.com/NVIDIA/aerial-cuda-accelerated-ran) is the open *gNB SDK* — not the AI rApp weights. Capgemini and others contribute to O-RAN-SC reference rApps but do not ship their commercial AI weights. For operators in supply-chain-sensitive jurisdictions, "we can verify exactly what shipped, against the source" is not a marketing line — it is a procurement gate.

---

## 9. One-line marketplace positioning

**PreceptualAI is the audit-first, NTN-native rApp that runs alongside your existing SMO (Ericsson, Nokia, NVIDIA-coupled, or open-source) and ships counterfactuals + EPFD compliance + tamper-evident audit your current rApp catalogue does not.**

---

## 10. References

- [Ericsson Intelligent Automation Platform (EIAP)](https://www.ericsson.com/en/ran/intelligent-ran-automation/intelligent-automation-platform)
- [Ericsson rApp Directory](https://www.ericsson.com/en/portfolio/cross-portfolio/rapp-directory)
- [Ericsson + MasOrange (2025)](https://www.ericsson.com/en/press-releases/3/2025/ericsson-and-masorange-advance-autonomous-networks-with-ai-driven-automation-platform-and-rapps)
- [Ericsson EIAP O1 article](https://telecomdrive.com/how-ericsson-is-reinventing-network-management-with-eiap-o1-interface-8679562/)
- [Ericsson NTN payload architecture](https://www.ericsson.com/en/blog/2024/10/ntn-payload-architecture)
- [Nokia MantaRay SMO](https://www.nokia.com/mobile-networks/ran-operations/mantaray-smo/)
- [Nokia MantaRay → Open RAN](https://the-mobile-network.com/2025/06/nokia-evolves-mantaray-to-open-ran-smo-solution/)
- [Nokia + NTT DOCOMO (Nov 2025)](https://www.globenewswire.com/news-release/2025/11/25/3194000/0/en/Nokia-introduces-MantaRay-SON-to-NTT-DOCOMO-s-multi-vendor-5G-network.html)
- [Nokia + du autonomous slicing (Dec 2025)](https://www.globenewswire.com/news-release/2025/12/03/3198586/0/en/Nokia-and-du-set-new-benchmark-in-5G-innovation-with-autonomous-network-slicing-in-industry-first.html)
- [Nokia + HPE Juniper-RIC (SDxCentral)](https://www.sdxcentral.com/news/nokia-picks-hpes-juniper-ric/)
- [NVIDIA AI-RAN](https://www.nvidia.com/en-us/industries/telecommunications/ai-ran/)
- [NVIDIA Aerial docs](https://docs.nvidia.com/aerial/index.html)
- [NVIDIA aerial-cuda-accelerated-ran (GitHub)](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- [NVIDIA ARC-Compact](https://developer.nvidia.com/blog/deploy-ai-ran-at-cell-sites-with-nvidia-arc-compact/)
- [NVIDIA + Nokia AI-RAN strategic bet (Frank Rayal)](https://frankrayal.com/2025/11/03/nvidia-and-nokia-a-strategic-bet-on-edge-ai-ran/)
- [XAI-on-RAN, arXiv 2511.17514](https://arxiv.org/html/2511.17514)
- [Mavenir RIC](https://www.mavenir.com/portfolio/mavscale/ai-analytics/ran-intelligent-controller-ric/)
- [Mavenir NTN](https://www.mavenir.com/portfolio/mavair/non-terrestrial-network-ntn/)
- [VIAVI RIC test platform](https://www.viavisolutions.com/en-us/ran-intelligent-controller)
- [VIAVI TeraVM AI RSG](https://www.viavisolutions.com/en-us/news-releases/viavi-introduces-teravm-ai-ran-scenario-generator)
- [Capgemini RATIO whitepaper](https://www.capgemini.com/wp-content/uploads/2022/03/ratio-capgemini-ric-for-intelligent-open-ran-operations-whitepaper_-17-february-2022.pdf)
- [Capgemini + Deutsche Telekom (2026)](https://www.capgemini.com/news/press-releases/capgemini-and-deutsche-telekom-engineer-an-open-platform-for-intelligent-ran-automation/)
- [O-RAN ALLIANCE NTN deployments whitepaper (Apr 2025)](https://mediastorage.o-ran.org/ecosystem-resources/O-RAN-2025.04.02.WP.O-RAN_NTN_Deployments-v08.4.pdf)
- [ITU-R S.1503-2 PDF](https://www.itu.int/dms_pubrec/itu-r/rec/s/R-REC-S.1503-2-201312-S!!PDF-E.pdf)
- [ITU EPFD support](https://www.itu.int/epfdsupport/)
- [Transfinite Visualyse-EPFD](https://www.transfinite.com/content/ValidationSoftwareforRecS1503)
- [STL Partners — RIC, xApps, rApps primer](https://stlpartners.com/articles/network-innovation/ric-xapps-rapps-who-are-the-key-players/)
- [TIP Exchange Marketplace](https://exchange.telecominfraproject.com/marketplace)
- [LightReading — rApp marketplaces](https://www.lightreading.com/open-ran/that-s-a-rapp-ran-automation-ecosystems-and-marketplaces)
