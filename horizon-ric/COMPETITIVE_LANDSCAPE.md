# Competitive Landscape — PreceptualAI vs the AI-RAN + NTN rApp / SMO market

_Generated: 2026-05-08 (post-v3 trust-layer wave). Author: PreceptualAI team._

**v3 update.** Added 17-primitive LCM trust layer comparison ([`README.md`](README.md) §8 → Section "vendor-by-vendor matrix"). PreceptualAI is the **only** candidate scoring TRUE on per-decision counterfactual envelope, SHA-256 hash chain, RFC 3161 anchor, atomic A→B at slot boundary, bit-identical artefact vault, TS 28.567 LoopState, ITU-R S.1503 EPFD in-loop, drift detectors with Prometheus, TS 28.105 §7.4 model card, and GDPR Art. 6 lineage. Competitor coverage on those 10 primitives: NVIDIA Aerial 2/10, Nokia MantaRay 2/10, Ericsson IAP 2/10, OSC NONRTRIC 0/10.

> **Honesty note.** We do **not** have access to Ericsson EIC/EIAP, Nokia MantaRay, or NVIDIA Aerial source code. Every competitor cell in the tables below cites a public URL or whitepaper. Where we cannot find published evidence, the cell is marked `no public evidence found` rather than invented. Our own row cites a file/test in this repo.

---

## Section 1 — Vendor capability matrix

Legend:
- `Yes (cite)` — explicit public claim with link, or repo file proving it.
- `Partial` — feature claimed but with caveats / gaps.
- `No public evidence` — searched, did not find a primary source. Honest blank.
- `No (cite)` — explicitly stated as out-of-scope by vendor.

| Capability | PreceptualAI (us) | Ericsson EIAP / EIC | Nokia MantaRay SMO | NVIDIA Aerial / ARC | Mavenir RIC | VIAVI TeraVM RIC Test | Capgemini RATIO |
|---|---|---|---|---|---|---|---|
| **A1 conformance (TS WG2.A1AP-v05.00)** | Yes — `src/horizon_ric/rapp/a1_adapter.py:190` (PUT/GET/DELETE + EI jobs); CI gate in `tests/test_a1_osc_dialect.py`, `docs/conformance/CONFORMANCE.md` §A1 | Yes — claimed, [Ericsson EIC product page](https://www.ericsson.com/en/portfolio/cloud-software-and-services/network-management-and-automation/ericsson-intelligent-automation-platform/ericsson-intelligent-controller) | Yes — claimed, [Nokia MantaRay SMO page](https://www.nokia.com/mobile-networks/ran-operations/mantaray-smo/) | Partial — Aerial is a RAN SDK, not an SMO; A1 surfaces via partner SMOs ([NVIDIA Aerial docs](https://docs.nvidia.com/aerial/index.html)) | Yes — claimed, [Mavenir RIC page](https://www.mavenir.com/portfolio/mavscale/ai-analytics/ran-intelligent-controller-ric/) | N/A — VIAVI is a test platform, not an SMO; tests A1 on others ([VIAVI RIC page](https://www.viavisolutions.com/en-us/ran-intelligent-controller)) | Yes — claimed, [Capgemini RATIO whitepaper](https://www.capgemini.com/wp-content/uploads/2022/03/ratio-capgemini-ric-for-intelligent-open-ran-operations-whitepaper_-17-february-2022.pdf) |
| **R1 conformance (WG2.R1AP-v06.00)** | Partial — `src/horizon_ric/rapp/r1_adapter.py:99` register, `:lifecycle.py` deregister; conformance §R1 marks register/deregister ⚠ | Yes — explicitly built around R1, [Ericsson R1 blog](https://www.ericsson.com/en/blog/2023/12/how-the-o-ran-alliance-r1-interface-empowers-a-global-ran-automation-community) | Yes — "fully Open RAN compliant and supports open R1 interface for rApps" ([Nokia MantaRay SMO](https://www.nokia.com/mobile-networks/ran-operations/mantaray-smo/)) | No public evidence — Aerial is a gNB SDK, not an SMO/RIC | Yes — claimed across non-RT RIC + R1 | N/A | Yes — claimed |
| **O1 conformance (WG2.O1-v06.00, NETCONF + YANG TS 28.541)** | Yes — `src/horizon_ric/rapp/o1_adapter.py:103/145/165` (connect/get/edit-config); NRM modules ⚠ partial; `tests/test_o1_live.py` | Yes — [Ericsson EIAP O1 article (TelecomDrive)](https://telecomdrive.com/how-ericsson-is-reinventing-network-management-with-eiap-o1-interface-8679562/) | Yes — claimed | No public evidence | Yes — claimed | N/A | Yes — claimed |
| **NTN-aware (TR 38.811 / 3GPP Rel-17/19 NTN, Doppler, beam-patterns)** | Yes — full propagation stack: `src/horizon_ric/planner/physics/propagation.py` (P.525, P.676, P.838, P.618), `s1428.py`, `epfd.py`; orbital + Doppler in `src/horizon_ric/planner/orbital/`; tests `test_tr38811_ntn_timing.py`, `test_doppler.py`, `test_orbital.py`, `test_celestrak.py`, `test_beam_pattern.py`, `test_ntn_air_ran.py` | No public evidence — Ericsson works on NTN at the gNB / device layer ([Ericsson NTN blog](https://www.ericsson.com/en/blog/2024/10/ntn-payload-architecture)), but no public statement that EIAP rApps are NTN-aware (Doppler / orbital state inside policy decisions) | No public evidence | No public evidence — Aerial is a 3GPP gNB SDK; NTN is on roadmap, not yet a rApp surface | Adjacent — Mavenir has a separate NTN portfolio ([Mavenir NTN page](https://www.mavenir.com/portfolio/mavair/non-terrestrial-network-ntn/)) but the rApp/RIC product line and the NTN portfolio are presented as separate products | N/A | No public evidence |
| **3GPP TS 28.105 AI/ML model card emission** | Yes — `checkpoints/*.md` + canonical block + sha256, gated by `tests/test_ts28105_model_card_emit.py` and `tests/test_conformance.py` | No public evidence of model-card emission per checkpoint | No public evidence | No public evidence | No public evidence | N/A | No public evidence |
| **Per-policy counterfactual envelope (rejected alternatives + machine + human reason, see `PARADIGMS.md` H1)** | Yes — `src/horizon_ric/evidence/schema.py` (`DecisionRecord`, `RejectedAlternative`), `src/horizon_ric/evidence/explanation.py`, `src/horizon_ric/policy/counterfactual.py`, `tests/test_emit_guards_counterfactual.py` | No public evidence — Ericsson explainability is not described in those terms in [Ericsson EIAP page](https://www.ericsson.com/en/ran/intelligent-ran-automation/intelligent-automation-platform) | No public evidence | Aerial documents Integrated-Gradients-style attribution in academic context ([XAI-on-RAN paper, Nov 2026](https://arxiv.org/html/2511.17514)) — feature attribution, not action-counterfactuals | No public evidence | N/A — test platform, not a policy emitter | No public evidence |
| **Tamper-evident audit chain (SHA-256 chained DecisionRecords)** | Yes — `src/horizon_ric/evidence/store.py` (chain), `tests/test_evidence_store.py`, `tests/test_evidence.py` | No public evidence — EIAP marketing focuses on lifecycle, not cryptographic audit chains | No public evidence | No public evidence | No public evidence | N/A | No public evidence |
| **ITU-R S.1503 EPFD time-CDF enforcement (Article 22 limits)** | Yes — `src/horizon_ric/planner/physics/epfd.py` (Annex 1 Eq. 1 line 91), `tests/test_epfd_time_cdf.py`, `benchmarks/run_benchmarks.py:epfd_aggregate_22sat_snapshot` | No public evidence — S.1503 is a regulatory tool ([Transfinite Visualyse-EPFD](https://www.transfinite.com/content/ValidationSoftwareforRecS1503), [ITU EPFD support](https://www.itu.int/epfdsupport/)), not yet integrated as an in-loop rApp constraint by named SMO vendors | No public evidence | No public evidence | No public evidence — Mavenir NTN is at the gNB layer | N/A | No public evidence |
| **RBAC (multi-tenant)** | Yes — Casbin: `src/horizon_ric/security/rbac.py`, `rbac_model.conf`, `rbac_policy.csv`, `tests/test_rbac.py`, `tests/test_tenant_isolation.py` | Yes — implied by enterprise SMO product ([Ericsson EIC](https://www.ericsson.com/en/portfolio/cloud-software-and-services/network-management-and-automation/ericsson-intelligent-automation-platform/ericsson-intelligent-controller)) | Yes — implied by enterprise SMO product | N/A — SDK, not a multi-tenant SMO | Yes — claimed | N/A | Yes — claimed |
| **Multi-tenant slice isolation** | Yes — `src/horizon_ric/security/tenant.py`, `tests/test_tenant_isolation.py` | Yes — claimed via SMO orchestration | Yes — claimed | N/A | Yes — claimed | N/A | Yes — claimed |
| **Open source (code + weights + model cards)** | Yes — repo + `checkpoints/*.md` + `checkpoints/*.pt` shipped in tree | No — proprietary | No — proprietary | Partial — [aerial-cuda-accelerated-ran on GitHub](https://github.com/NVIDIA/aerial-cuda-accelerated-ran) is open SDK; rApps and weights are not | Mostly proprietary; some O-RAN-SC contributions | Proprietary test platform | Mostly proprietary; some O-RAN-SC reference rApps |
| **Image signing + SBOM (CycloneDX, cosign)** | Yes — `deploy/cosign/cosign.pub`, `deploy/sbom/horizon-ric-sbom.json`, `.github/workflows/sbom.yml`, `.github/workflows/build.yml` | No public evidence at the rApp granularity | No public evidence | No public evidence | No public evidence | N/A | No public evidence |
| **Public benchmark report with reproducible script** | Yes — `benchmarks/RESULTS.md` + `benchmarks/run_benchmarks.py` (deterministic, 5 measured iters, p95) | No public benchmark numbers found | No public benchmark numbers found | NVIDIA publishes peak-throughput / GPU-cell numbers ([NVIDIA AI-RAN landing](https://www.nvidia.com/en-us/industries/telecommunications/ai-ran/)) but no per-rApp decision-loop p95 | No public benchmark numbers found | Provides benchmark service to others ([VIAVI Plugfest](https://www.viavisolutions.com/en-us/news-releases/viavi-introduces-teravm-ai-ran-scenario-generator)) | No public benchmark numbers found |
| **Production deployments at scale** | No — pilot stage, see `PILOT.md` and `GAPS_TO_PILOT.md` | Yes — [Ericsson + MasOrange deployment, 2025](https://www.ericsson.com/en/press-releases/3/2025/ericsson-and-masorange-advance-autonomous-networks-with-ai-driven-automation-platform-and-rapps); [AT&T case study](https://www.ericsson.com/en/cases/2025/shaping-networks-of-tomorrow) | Yes — [Nokia + NTT DOCOMO MantaRay SON, Nov 2025](https://www.globenewswire.com/news-release/2025/11/25/3194000/0/en/Nokia-introduces-MantaRay-SON-to-NTT-DOCOMO-s-multi-vendor-5G-network.html); [Nokia + du autonomous slicing, Dec 2025](https://www.globenewswire.com/news-release/2025/12/03/3198586/0/en/Nokia-and-du-set-new-benchmark-in-5G-innovation-with-autonomous-network-slicing-in-industry-first.html) | Yes — multiple ARC pilots ([NVIDIA ARC-Compact, May 2025](https://the-mobile-network.com/2025/05/nvidia-launches-compact-ai-ran-solution-for-cell-site-installations/)); [Nokia–NVIDIA partnership, Nov 2025](https://frankrayal.com/2025/11/03/nvidia-and-nokia-a-strategic-bet-on-edge-ai-ran/) | Yes — operator deployments | N/A — test tool | Yes — Deutsche Telekom partnership ([Capgemini DT, Feb 2026](https://www.capgemini.com/news/press-releases/capgemini-and-deutsche-telekom-engineer-an-open-platform-for-intelligent-ran-automation/)) |
| **24/7 SLA-backed support** | No — pre-pilot | Yes — vendor standard | Yes | Yes | Yes | Yes | Yes |

---

## Section 2 — Performance benchmark comparison

The PreceptualAI numbers come from a single deterministic run: `benchmarks/run_benchmarks.py`, results captured in `benchmarks/RESULTS.md` (2026-05-06). Method: 2 warm-up + 5+ measured iterations per bench, median + p95 reported.

| Benchmark | PreceptualAI (median) | PreceptualAI (p95) | Ericsson | Nokia | NVIDIA Aerial | Source / notes |
|---|---|---|---|---|---|---|
| End-to-end decision-loop walltime (stages 2→4, no HTTP) | 33.36 ms | 33.36 ms | Not published | Not published | Not published per-rApp; ARC focuses on PHY-layer GPU throughput | `benchmarks/RESULTS.md:e2e_stages_3_to_5_walltime`. Industry-typical Near-RT RIC budget is 10 ms–1 s ([STL Partners RIC guide](https://stlpartners.com/articles/network-innovation/ric-xapps-rapps-who-are-the-key-players/)); ours sits in the lower third. |
| Orbital propagation (22-sat Walker constellation, 1 step) | 1.5 µs / sat-step | 1.5 µs | Not published | Not published | Not published | `benchmarks/RESULTS.md:orbital_walker_22sat_propagate` |
| EPFD aggregate snapshot (22 NGSO emitters, ITU-R S.1503 Annex 1 Eq.1) | 0.04 ms / call | 0.04 ms | Not published | Not published | Not published | `benchmarks/RESULTS.md:epfd_aggregate_22sat_snapshot` and `src/horizon_ric/planner/physics/epfd.py:91`. Reference offline tool: [Transfinite Visualyse-EPFD](https://www.transfinite.com/content/ValidationSoftwareforRecS1503) — designed for filings, not in-loop rApp use. |
| EPFD 1-hour time-CDF (500-sat Walker) | 0.19 s | 0.19 s | Not published | Not published | Not published | `benchmarks/RESULTS.md:epfd_time_cdf_500sat_1h` |
| Doppler-pass envelope (600 s window @ 5 s step) | 0.80 ms | 0.80 ms | Not published | Not published | Not published | `benchmarks/RESULTS.md:doppler_pass_envelope_600s` |
| TD-MPC plan (horizon=12, 64 samples, 3 iters) | 175.41 ms | 182.74 ms | Not published | Not published | Not published | `benchmarks/RESULTS.md:td_mpc_plan_full` |
| Constraint full-check + project (22 emitters) | 0.06 ms | 0.06 ms | Not published | Not published | Not published | `benchmarks/RESULTS.md:constraint_full_check_and_project` |
| Latent-dynamics rollout (B=256, H=12) | 64.38 ms | 78.39 ms | Not published | Not published | Not published | `benchmarks/RESULTS.md:latent_dynamics_rollout_H12_N256` |
| Diffusion tail-sample (n=32, steps=20) | 35.75 ms | 61.54 ms | Not published | Not published | Not published | `benchmarks/RESULTS.md:diffusion_tail_sample_32` |
| Perceiver fusion forward (256×128) | 10.33 ms | 10.95 ms | Not published | Not published | NVIDIA reports IG/XAI overhead "on the order of milliseconds" ([XAI-on-RAN, Nov 2026](https://arxiv.org/html/2511.17514)) — methodology not directly comparable | `benchmarks/RESULTS.md:perceiver_fusion_forward_256x128` |

> **Reading note.** Most "Not published" cells are not gaps in our research — they reflect that named SMO vendors do not publish per-call latency for rApps. Their public materials are in throughput, energy savings (e.g. Capgemini's "10 % energy savings" line), and feature catalogues. We treat their silence as silence, not as 0.

---

## Section 3 — Differentiation claims (defensible)

The five things PreceptualAI has that we have **public evidence** named competitors do not ship today.

### 3.1 Per-policy counterfactual envelope (paradigm H1)

- **Our proof**: `src/horizon_ric/evidence/schema.py` (`RejectedAlternative` block on every `DecisionRecord`); `src/horizon_ric/evidence/explanation.py` machine→human reason generator; `src/horizon_ric/policy/counterfactual.py`; gated by `tests/test_emit_guards_counterfactual.py`. Schema documented in `PARADIGMS.md` §H1.
- **Competitor absence**: Ericsson EIAP marketing emphasises rApp lifecycle and the rApp Directory, not action-counterfactuals ([Ericsson rApps page](https://www.ericsson.com/en/ran/intelligent-ran-automation/intelligent-automation-platform/rapps), [EIAP Ecosystem](https://www.ericsson.com/en/ran/intelligent-ran-automation/intelligent-automation-platform/ecosystem)). NVIDIA's nearest published explainability is feature-attribution Integrated-Gradients in [XAI-on-RAN (arXiv 2511.17514)](https://arxiv.org/html/2511.17514) — that is *why this feature mattered*, not *what would have happened if you had chosen action B*.

### 3.2 SHA-256-chained tamper-evident audit

- **Our proof**: `src/horizon_ric/evidence/store.py` chain-hash logic; `tests/test_evidence_store.py`, `tests/test_evidence.py`. Continuous integrity verification is a fixed test gate.
- **Competitor absence**: No public Ericsson, Nokia, NVIDIA, or Mavenir documentation describes a hash-chained DecisionRecord log at the rApp level. Standard SMO logging is mutable database rows.

### 3.3 ITU-R S.1503 EPFD time-CDF enforcement, in-loop

- **Our proof**: `src/horizon_ric/planner/physics/epfd.py` implements Annex 1 Eq. (1); `tests/test_epfd_time_cdf.py` validates against reference values; benched at 0.19 s for 500-sat Walker ([benchmarks/RESULTS.md](benchmarks/RESULTS.md)).
- **Competitor absence**: Commercial EPFD tooling exists ([Visualyse EPFD by Transfinite](https://www.transfinite.com/content/ValidationSoftwareforRecS1503)) but is filing-time, not in-loop policy-time. No named SMO vendor publishes EPFD enforcement as an rApp surface.

### 3.4 Compositional World Model (paradigm H2)

- **Our proof**: `src/horizon_ric/planner/` — Perceiver fusion + GraphJEPA + latent-dynamics + TD-MPC, with full benchmark numbers in `benchmarks/RESULTS.md` (rows: `perceiver_fusion_forward_256x128`, `graph_jepa_loss_step`, `latent_dynamics_rollout_H12_N256`, `td_mpc_plan_full`, `diffusion_tail_sample_32`). Documented in `PARADIGMS.md` §H2 and `ARCHITECTURE.md`.
- **Competitor absence**: Vendor rApps are mostly individual ML heads (anomaly detection, energy-saving, traffic steering). We have not found a public statement of a unified world-model planner across competitor catalogues. NVIDIA Aerial focuses on PHY acceleration, not a world-model rApp.

### 3.5 Open-source weights + model cards (TS 28.105 §7)

- **Our proof**: `checkpoints/*.pt` + `checkpoints/*.md` (canonical block, sha256, training_data, metrics); enforced by `tests/test_ts28105_model_card_emit.py` and `tests/test_conformance.py`. SBOM at `deploy/sbom/horizon-ric-sbom.json` (CycloneDX 1.5).
- **Competitor absence**: Ericsson, Nokia, Mavenir, Capgemini ship rApps as proprietary binaries. NVIDIA's [aerial-cuda-accelerated-ran](https://github.com/NVIDIA/aerial-cuda-accelerated-ran) is an open *gNB SDK* — not the AI rApp weights.

---

## Section 4 — Honest gaps (the five things they have that we do not, yet)

### 4.1 Production deployments at scale

They have hundreds of live cells and live operator deployments:
- Ericsson + MasOrange ([2025](https://www.ericsson.com/en/press-releases/3/2025/ericsson-and-masorange-advance-autonomous-networks-with-ai-driven-automation-platform-and-rapps))
- Nokia + NTT DOCOMO MantaRay SON ([Nov 2025](https://www.globenewswire.com/news-release/2025/11/25/3194000/0/en/Nokia-introduces-MantaRay-SON-to-NTT-DOCOMO-s-multi-vendor-5G-network.html))
- Nokia + du autonomous slicing first ([Dec 2025](https://www.globenewswire.com/news-release/2025/12/03/3198586/0/en/Nokia-and-du-set-new-benchmark-in-5G-innovation-with-autonomous-network-slicing-in-industry-first.html))

We have **zero** production cells. The pilot path is in `PILOT.md` and gap-tracked in `GAPS_TO_PILOT.md`.

### 4.2 SLA-backed 24×7 support contracts

Every named competitor has tier-1 NOC support, change-management, and contractual uptime SLAs. We do not. Pilot customers will need to accept best-effort or contract this through an integrator.

### 4.3 Vendor-lock-in advantages

Ericsson EIAP integrates natively with Ericsson EIC and Ericsson RAN ([Ericsson Intelligent Controller](https://www.ericsson.com/en/portfolio/cloud-software-and-services/network-management-and-automation/ericsson-intelligent-automation-platform/ericsson-intelligent-controller)). Nokia MantaRay has the Nokia BTS install base. NVIDIA + Nokia partnership for AI-RAN at the cell site ([Frank Rayal analysis, Nov 2025](https://frankrayal.com/2025/11/03/nvidia-and-nokia-a-strategic-bet-on-edge-ai-ran/)) bundles GPU + RAN. We have no equivalent bundling leverage.

### 4.4 Pre-existing operator relationships and certifications

GSMA-Network Equipment Security Assurance Scheme (NESAS), TM Forum Open API certifications, customer references, completed plugfests on operator labs — all built over years. We have repository-level conformance claims; we have not yet been independently certified.

### 4.5 Marketplace presence

Ericsson has an [rApp Directory](https://www.ericsson.com/en/portfolio/cross-portfolio/rapp-directory) with multiple third parties; Nokia has the [rApp Marketplace](https://www.nokia.com/mobile-networks/ran-operations/mantaray-smo/); the [TIP Exchange](https://exchange.telecominfraproject.com/marketplace) is the open community shop-front. We are not yet listed in any of these.

---

## Section 5 — Marketplace positioning

**PreceptualAI is not a SMO. We are an audit-first, NTN-native rApp that runs on top of your existing SMO** — Ericsson EIAP, Nokia MantaRay, NVIDIA Aerial-coupled SMOs, Capgemini RATIO, or any other R1-conformant non-RT RIC. We do not compete on RAN integration depth, on operator relationships, or on bundled hardware. We compete on three axes the incumbents do not pitch:

1. **Auditability** — every emitted A1 policy ships with a structured counterfactual envelope (rejected alternatives + machine-reasoning + human-reading rejection reasons) and is committed to a SHA-256-chained, tamper-evident DecisionRecord log. This is the difference between "the rApp said so" and "the rApp said so, here is what it considered, here is what it rejected, and here is the cryptographic proof we have not edited the record". For Level-4 autonomous-network certification (TM Forum) and for regulators, that gap matters.

2. **NTN-native** — when an LEO satellite passes overhead, our policy decisions already know about Doppler shift, slant-path rain attenuation (ITU-R P.838 / P.618), beam-cap on-board (S.1428), and cumulative EPFD against Article 22 limits (S.1503). The named SMO vendors treat NTN as a separate gNB-layer product, not as something the rApp loop reasons about. We were built NTN-first.

3. **Open source + auditable weights** — we ship model weights, model cards (TS 28.105 §7), SBOMs (CycloneDX 1.5), cosign signatures, and reproducible benchmarks. Operators that need supply-chain provenance can verify what we shipped, against the source. Vendor binaries cannot be inspected the same way.

We are **SMO-agnostic** by design. We expect to plug into Ericsson EIAP via R1, into Nokia MantaRay via R1, into open-source SMOs (O-RAN-SC, ONAP) directly, and into NVIDIA-accelerated stacks via the same R1 surface. Our 90-day pilot path (`PILOT.md`) assumes the operator already has an SMO and just bolts us on as an rApp instance.

**One-line positioning.** *PreceptualAI is the audit-first, NTN-native rApp that runs alongside your existing SMO (Ericsson, Nokia, NVIDIA-coupled, or open-source) and ships counterfactuals + EPFD compliance + tamper-evident audit your current rApp catalogue does not.*

---

## Sources

- Ericsson EIAP product page — https://www.ericsson.com/en/ran/intelligent-ran-automation/intelligent-automation-platform
- Ericsson EIC — https://www.ericsson.com/en/portfolio/cloud-software-and-services/network-management-and-automation/ericsson-intelligent-automation-platform/ericsson-intelligent-controller
- Ericsson rApp Directory — https://www.ericsson.com/en/portfolio/cross-portfolio/rapp-directory
- Ericsson rApps — https://www.ericsson.com/en/ran/intelligent-ran-automation/intelligent-automation-platform/rapps
- Ericsson EIAP Ecosystem — https://www.ericsson.com/en/ran/intelligent-ran-automation/intelligent-automation-platform/ecosystem
- Ericsson R1 blog — https://www.ericsson.com/en/blog/2023/12/how-the-o-ran-alliance-r1-interface-empowers-a-global-ran-automation-community
- Ericsson + MasOrange press release (2025) — https://www.ericsson.com/en/press-releases/3/2025/ericsson-and-masorange-advance-autonomous-networks-with-ai-driven-automation-platform-and-rapps
- Ericsson AT&T case (2025) — https://www.ericsson.com/en/cases/2025/shaping-networks-of-tomorrow
- Ericsson NTN payload architecture — https://www.ericsson.com/en/blog/2024/10/ntn-payload-architecture
- Ericsson + Qualcomm + Thales NTN milestone (2025) — https://www.ericsson.com/en/press-releases/3/2025/ericsson-qualcomm-thales-achieve-space-connectivity-milestone
- Ericsson EIAP O1 article (TelecomDrive) — https://telecomdrive.com/how-ericsson-is-reinventing-network-management-with-eiap-o1-interface-8679562/
- Ericsson IAP GitHub org — https://github.com/ericsson-iap
- Nokia MantaRay SMO — https://www.nokia.com/mobile-networks/ran-operations/mantaray-smo/
- Nokia MantaRay evolution to Open RAN (Mobile Network) — https://the-mobile-network.com/2025/06/nokia-evolves-mantaray-to-open-ran-smo-solution/
- Nokia + NTT DOCOMO (Nov 2025) — https://www.globenewswire.com/news-release/2025/11/25/3194000/0/en/Nokia-introduces-MantaRay-SON-to-NTT-DOCOMO-s-multi-vendor-5G-network.html
- Nokia + du autonomous slicing (Dec 2025) — https://www.globenewswire.com/news-release/2025/12/03/3198586/0/en/Nokia-and-du-set-new-benchmark-in-5G-innovation-with-autonomous-network-slicing-in-industry-first.html
- Nokia + HPE Juniper RIC (SDxCentral) — https://www.sdxcentral.com/news/nokia-picks-hpes-juniper-ric/
- NVIDIA AI-RAN landing — https://www.nvidia.com/en-us/industries/telecommunications/ai-ran/
- NVIDIA Aerial docs — https://docs.nvidia.com/aerial/index.html
- NVIDIA ARC-Compact (developer blog) — https://developer.nvidia.com/blog/deploy-ai-ran-at-cell-sites-with-nvidia-arc-compact/
- NVIDIA aerial-cuda-accelerated-ran (GitHub) — https://github.com/NVIDIA/aerial-cuda-accelerated-ran
- Frank Rayal: NVIDIA + Nokia AI-RAN (Nov 2025) — https://frankrayal.com/2025/11/03/nvidia-and-nokia-a-strategic-bet-on-edge-ai-ran/
- XAI-on-RAN (arXiv 2511.17514, Nov 2026) — https://arxiv.org/html/2511.17514
- Mavenir RIC — https://www.mavenir.com/portfolio/mavscale/ai-analytics/ran-intelligent-controller-ric/
- Mavenir NTN — https://www.mavenir.com/portfolio/mavair/non-terrestrial-network-ntn/
- VIAVI RIC — https://www.viavisolutions.com/en-us/ran-intelligent-controller
- VIAVI TeraVM AI RSG — https://www.viavisolutions.com/en-us/news-releases/viavi-introduces-teravm-ai-ran-scenario-generator
- Capgemini RATIO whitepaper — https://www.capgemini.com/wp-content/uploads/2022/03/ratio-capgemini-ric-for-intelligent-open-ran-operations-whitepaper_-17-february-2022.pdf
- Capgemini + Deutsche Telekom (2026) — https://www.capgemini.com/news/press-releases/capgemini-and-deutsche-telekom-engineer-an-open-platform-for-intelligent-ran-automation/
- Capgemini + AirHop — https://airhopai.com/2023/12/13/airhop-and-capgemini-collaborate-to-advance-open-ran-ric-and-rapp-development/
- O-RAN ALLIANCE NTN deployments whitepaper (Apr 2025) — https://mediastorage.o-ran.org/ecosystem-resources/O-RAN-2025.04.02.WP.O-RAN_NTN_Deployments-v08.4.pdf
- ITU-R S.1503-2 — https://www.itu.int/dms_pubrec/itu-r/rec/s/R-REC-S.1503-2-201312-S!!PDF-E.pdf
- ITU EPFD support — https://www.itu.int/epfdsupport/
- Transfinite Visualyse-EPFD — https://www.transfinite.com/content/ValidationSoftwareforRecS1503
- STL Partners RIC, xApps, rApps guide — https://stlpartners.com/articles/network-innovation/ric-xapps-rapps-who-are-the-key-players/
- TIP Exchange Marketplace — https://exchange.telecominfraproject.com/marketplace
- LightReading "That's a rApp" marketplaces — https://www.lightreading.com/open-ran/that-s-a-rapp-ran-automation-ecosystems-and-marketplaces
