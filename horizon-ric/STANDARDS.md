# Standards Conformance — rApp + xApp

> Complete inventory of standards that govern PreceptualAI (rApp) and its companion xApp deployments (parent Preceptual UHCI). Each architectural component is mapped to specific spec sections. Each conformance gap has a remediation plan.
>
> **Scope**: Production + pilot deployment of an O-RAN compliant rApp/xApp suite handling NTN+terrestrial RAN telemetry and policy.
>
> **Audience**: engineering, compliance, customer architects, regulator-facing reviewers.

---

## 1. Standards Inventory

### 1.1 O-RAN Alliance Specifications

#### Architecture & Cross-Cutting
| Spec | Title | Version | Scope |
|---|---|---|---|
| O-RAN.WG1.O-RAN-Architecture-Description | O-RAN Architecture Description | v13.00 (2024) | Overall reference architecture |
| O-RAN.WG1.OAM-Architecture | OAM Architecture | v12.00 | OAM topology, interfaces |
| O-RAN.WG1.SMO-OAM-Architecture | SMO and OAM Architecture | v01.00 | SMO functions, hosting rApps |
| O-RAN.WG1.Use-Cases-Detailed-Specification | Use Cases Detailed Specification | v15.00 | Reference use cases (incl. NTN flows in 2024+) |
| O-RAN.WG1.AIML-FL | AI/ML Federated Learning Architecture | v01.00 | Federated learning reference |

#### rApp / Non-RT RIC (PreceptualAI)
| Spec | Title | Version | Scope |
|---|---|---|---|
| O-RAN.WG2.NON-RT-RIC-ARCH | Non-RT RIC Architecture | v04.00 | rApp host + framework |
| O-RAN.WG2.R1AP | R1 Application Protocol | v06.00 | rApp ↔ SMO/Non-RT RIC communication |
| O-RAN.WG2.R1GAP | R1 General Aspects and Principles | v06.00 | R1 design principles |
| O-RAN.WG2.A1AP | A1 Application Protocol | v05.00 | Non-RT → Near-RT policy delivery |
| O-RAN.WG2.A1-GAP | A1 General Aspects and Principles | v05.00 | A1 design principles |
| O-RAN.WG2.A1-EI | A1 Enrichment Information | v04.00 | EI types/jobs |
| O-RAN.WG2.A1TD | A1 Transport Data Layer | v04.00 | A1 transport details |
| O-RAN.WG2.AIML | AI/ML Workflow | v04.00 | rApp AI/ML lifecycle in Non-RT RIC |
| O-RAN.WG2.AIMLFW | AI/ML Framework | v01.00 | AI/ML framework architecture |
| O-RAN.WG2.UCR | Use Cases and Requirements | v14.00 | Non-RT RIC use cases |

#### xApp / Near-RT RIC (parent project)
| Spec | Title | Version | Scope |
|---|---|---|---|
| O-RAN.WG3.E2AP | E2 Application Protocol | v05.00 | xApp ↔ E2-Node communication |
| O-RAN.WG3.E2GAP | E2 General Aspects and Principles | v05.00 | E2 design |
| O-RAN.WG3.E2SM-KPM | E2 Service Model — Key Performance Metrics | v05.00 | KPM telemetry consumption |
| O-RAN.WG3.E2SM-RC | E2 Service Model — RAN Control | v04.00 | RC actions (xApp emits control) |
| O-RAN.WG3.E2SM-CCC | E2SM — Cell Configuration and Control | v01.00 | Cell-level config |
| O-RAN.WG3.E2SM-NI | E2SM — Network Interface | v01.00 | Network-level metrics |
| O-RAN.WG3.NEAR-RT-RIC-ARCH | Near-RT RIC Architecture | v04.00 | xApp host + framework |
| O-RAN.WG3.UCR | xApp Use Cases and Requirements | v04.00 | xApp use cases |

#### Common: O1, O2, Open Fronthaul, Cloud
| Spec | Title | Version | Scope |
|---|---|---|---|
| O-RAN.WG10.O1-Interface | O1 Interface specification | v11.00 | NETCONF/YANG O&M |
| O-RAN.WG10.O1-Interoperability | O1 Interoperability | v06.00 | OAM interop |
| O-RAN.WG6.O2-GAP | O2 General Aspects and Principles | v04.00 | SMO ↔ O-Cloud |
| O-RAN.WG6.AAL-GAP | Acceleration Abstraction Layer GAP | v01.00 | Hardware acceleration abstraction |
| O-RAN.WG6.O-Cloud-Notification-Architecture | O-Cloud Notification Architecture | v01.00 | Cloud event model |
| O-RAN.WG4.MP-YANG | M-Plane YANG Models | v14.00 | M-plane YANG schemas |
| O-RAN.WG4.CUS | Control, User and Synchronization plane | v15.00 | Open Fronthaul C/U/S-plane |

#### Security (WG11)
| Spec | Title | Version | Scope |
|---|---|---|---|
| O-RAN.WG11.Security-Requirements | Security Requirements Specification | v07.00 | Cross-cutting security requirements |
| O-RAN.WG11.Security-Protocols | Security Protocols Specifications | v06.00 | Concrete protocol-level security |
| O-RAN.WG11.Security-Tests | Security Tests Specification | v06.00 | Conformance security tests |
| O-RAN.WG11.Threat-Model | Threat Model | v06.00 | Threat surfaces by interface |

#### Conformance Tests (WG5/WG7/etc.)
| Spec | Title | Version | Scope |
|---|---|---|---|
| O-RAN.WG2.A1-IF-Conformance | A1 Interface Conformance Tests | v05.00 | A1 conformance |
| O-RAN.WG2.R1-Conformance | R1 Conformance Tests | v01.00 (drafts) | R1 conformance |
| O-RAN.WG3.E2-IF-Conformance | E2 Interface Conformance Tests | v05.00 | E2 conformance |
| O-RAN.WG10.O1-IF-Conformance | O1 Interface Conformance Tests | v06.00 | O1 conformance |

#### NTN extensions (next-Generation Research Group + WG1)
| Reference | Status |
|---|---|
| O-RAN nGRG NTN study reports (RR-2023-04, RR-2024-01, etc.) | Studies, baseline for spec work |
| O-RAN.WG1 NTN Use Case Detailed Specifications | In v15+, ongoing 2024–2026 |

### 1.2 3GPP Specifications

#### Core 5G Architecture
| Spec | Title | Release | Scope |
|---|---|---|---|
| 3GPP TS 23.501 | System Architecture for the 5G System | Rel-18+ | 5GS architecture, slicing, NEF/NWDAF |
| 3GPP TS 23.502 | Procedures for the 5G System | Rel-18+ | Stage-2 procedures |
| 3GPP TS 23.503 | Policy and Charging Control Framework | Rel-18+ | PCF policies; A1-policy alignment |
| 3GPP TS 23.288 | Architecture Enhancements for 5G System to support Network Data Analytics Services | Rel-18+ | NWDAF (data analytics function) |

#### NG-RAN
| Spec | Title | Release | Scope |
|---|---|---|---|
| 3GPP TS 38.300 | NR Stage 2 — Overall Description | Rel-18+ | NR architecture incl. NTN sections |
| 3GPP TS 38.401 | NG-RAN Architecture Description | Rel-18+ | gNB-CU/DU split, F1, Xn |
| 3GPP TS 38.410 | NG general aspects and principles | Rel-18+ | NG interface |
| 3GPP TS 38.413 | NG Application Protocol (NGAP) | Rel-18+ | Control plane |
| 3GPP TS 38.420 | Xn general aspects and principles | Rel-18+ | Xn between gNBs |
| 3GPP TS 38.470 | F1 general aspects and principles | Rel-18+ | F1 between CU and DU |

#### Management & Orchestration (most relevant for rApp)
| Spec | Title | Release | Scope |
|---|---|---|---|
| 3GPP TS 28.532 | Management Services | Rel-18+ | MnS framework |
| 3GPP TS 28.533 | Management and Orchestration Architecture Framework | Rel-18+ | Management architecture |
| 3GPP TS 28.541 | 5G Network Resource Model (NRM) | Rel-18+ | NRM IRP, YANG models |
| 3GPP TS 28.550 | Performance Management Concept | Rel-18+ | PM concept |
| 3GPP TS 28.552 | Performance Measurements 5G Network | Rel-18+ | PM measurements (we consume) |
| 3GPP TS 28.554 | 5G End-to-End KPIs | Rel-18+ | E2E KPI definitions (we report) |
| 3GPP TS 28.622 | Generic NRM IRP | Rel-18+ | NRM IRP solution set |
| 3GPP TS 32.401 | PM Concept and Requirements | Rel-18+ | PM requirements |

#### AI/ML Management
| Spec | Title | Release | Scope |
|---|---|---|---|
| 3GPP TS 28.105 | AI/ML Management | Rel-18 | AI/ML lifecycle in 3GPP MnS |
| 3GPP TR 28.908 | Study on AI/ML Management | Rel-18 | Study report |
| 3GPP TR 28.910 | Study on Autonomous Network Levels | Rel-18 | AN Levels 0–5 |

#### NTN (this is where our v9/PreceptualAI scope lives)
| Spec | Title | Release | Scope |
|---|---|---|---|
| 3GPP TR 38.811 | Study on NR to support Non-Terrestrial Networks | Rel-15 (study) | NTN channel models |
| 3GPP TR 38.821 | Solutions for NR to support NTN | Rel-16 (study) | NTN architecture solutions |
| 3GPP TS 38.811 | NR; Non-Terrestrial Networks (normative) | Rel-17 | NTN baseline |
| 3GPP TR 38.863 | Solutions for NB-IoT/eMTC support of NTN | Rel-17 | IoT-NTN |
| 3GPP TR 23.700-28 | Study on NTN System Architecture | Rel-18 | NTN system architecture |
| 3GPP TS 22.261 | Service Requirements for the 5G System | Rel-18+ | Service requirements (incl. NTN service classes) |

#### Slicing & QoS
| Spec | Title | Release | Scope |
|---|---|---|---|
| 3GPP TS 28.530 | Network Slicing — Concepts, Use Cases, and Requirements | Rel-18+ | Slicing requirements |
| 3GPP TS 28.531 | Provisioning for Network Slicing | Rel-18+ | Slice lifecycle |
| 3GPP TS 28.541 §6.3 | Slice NRM | Rel-18+ | Slice resource model |

#### Security (5G core/RAN)
| Spec | Title | Release | Scope |
|---|---|---|---|
| 3GPP TS 33.501 | Security Architecture and Procedures for 5G System | Rel-18+ | Security architecture |
| 3GPP TS 33.117 | Catalogue of General Security Assurance Requirements | Rel-18+ | Security assurance baseline |
| 3GPP TS 33.310 | Network Domain Security; Authentication Framework | Rel-18+ | NDS/AF |

### 1.3 ITU & FCC (NTN spectrum compliance)

#### ITU-R Recommendations (mandatory for spectrum operations)
| Recommendation | Title | Scope |
|---|---|---|
| ITU-R S.1503 | Functional description for software for compliance with EPFD limits | EPFD validation methodology — **HARD compliance** |
| ITU-R S.1428 | Reference FSS earth-station antenna radiation pattern | Antenna pattern reference |
| ITU-R S.580 | Radiation diagrams for FSS earth station antennas | Antenna pattern |
| ITU-R S.524 | Maximum permissible levels of off-axis e.i.r.p. density | EIRP off-axis limits |
| ITU-R P.452 | Prediction procedure for the evaluation of microwave interference between stations | Interference prediction (HARD) |
| ITU-R P.676 | Attenuation by atmospheric gases | Path-loss component |
| ITU-R P.837 | Characteristics of precipitation for propagation modelling | Rain rate model |
| ITU-R P.838 | Specific attenuation model for rain for use in prediction methods | Rain attenuation |
| ITU-R P.840 | Attenuation due to clouds and fog | Cloud/fog component |
| ITU-R P.681 | Propagation data required for the design of Earth-space land mobile telecommunications systems | Land mobile-satellite |
| ITU-R F.1336 | Reference radiation patterns for omnidirectional, sectoral, and other antennas | Reference patterns |
| ITU-R RA.769 | Protection criteria used for radio astronomical measurements | Radio astronomy protection |

#### ITU Radio Regulations (binding)
| Article | Title | Scope |
|---|---|---|
| RR Article 5 | Frequency allocations | Spectrum allocation |
| RR Article 21 | Terrestrial and space services sharing frequency bands above 1 GHz | Coexistence |
| RR Article 22 | Space services — EPFD limits | NGSO/GSO coexistence (HARD) |

#### ITU databases (data sources, not specs)
| Database | Scope |
|---|---|
| ITU SRS (Space Network List) | Filed satellite networks |
| ITU MIFR (Master International Frequency Register) | Master frequency register |

#### FCC (US)
| Reference | Scope |
|---|---|
| 47 CFR Part 25 | Satellite Communications | NGSO/GSO licensing, EPFD requirements (US) |
| FCC IBFS | International Bureau Filing System | Public technical filings (Starlink/Kuiper/OneWeb annexes) |

### 1.4 Operator-facing Standards (TM Forum, ETSI, IETF)

#### TM Forum
| API/Spec | Scope |
|---|---|
| TMF921 | Intent Management API |
| TMF640 | Service Activation and Configuration API |
| TMF638 | Service Inventory API |
| TMF641 | Service Ordering API |
| TMF642 | Alarm Management API |
| TMF724 | Open API for Autonomous Networks |
| TMF917 | RAN Sharing |

#### ETSI
| Spec | Scope |
|---|---|
| ETSI GS ENI 005 | ENI System Architecture |
| ETSI GR ENI 011 | ENI Use Cases |
| ETSI GS ZSM 002 | ZSM Reference Architecture |
| ETSI GS ZSM 008 | ZSM Cross-domain orchestration |
| ETSI GS ZSM 009-1 / 009-2 | Closed-loop automation |
| ETSI GS NFV-MAN 001 | NFV management and orchestration |

#### IETF
| RFC | Scope |
|---|---|
| RFC 6241 | NETCONF |
| RFC 6242 | Using SSH for NETCONF |
| RFC 7950 | YANG 1.1 |
| RFC 8040 | RESTCONF |
| RFC 8345 | YANG Data Model for Network Topologies |
| RFC 9020 | YANG Data Model for L2 VPN Service Delivery |
| RFC 8949 | CBOR |

### 1.5 AI-RAN Alliance & TIP (industry consortia)

| Reference | Status | Scope |
|---|---|---|
| AI-RAN Alliance Reference Architecture | In progress 2024–2026 | Reference for AI-RAN deployments |
| AI-RAN Alliance reference workflow | In progress | Reference workflow examples |
| TIP MUST framework | Released | MUST RAN technical priorities |
| TIP NEUTRAL HOST | Released | Neutral host reference |

---

## 2. Component → Spec Mapping (PreceptualAI rApp + xApp)

### 2.1 PreceptualAI (rApp) Components

| Component | Governing Specs | Conformance Tests |
|---|---|---|
| **Edge Telemetry Agent** | O-RAN O1 (telemetry), 3GPP TS 28.552 (PM measurements), TS 28.622 (NRM IRP), RFC 6241 (NETCONF), RFC 7950 (YANG) | O-RAN O1-IF-Conformance |
| **Resource-State JEPA Encoder** | O-RAN.WG2.AIML / AIMLFW (model lifecycle), 3GPP TS 28.105 (AI/ML Mgmt) | AI/ML conformance per WG2.AIML |
| **Multi-Head Risk Model** | Same as encoder + 3GPP TS 28.554 (E2E KPIs definitions) | KPI definitions match TS 28.554 |
| **Compositional World Model (H2)** | Physics modules cite 3GPP TR 38.901, ITU-R P.452/676/837/838/F.1336; closed-form sections must replicate spec equations | Per-module unit tests vs ITU-R reference impls |
| **Dyna Rollout Planner** | O-RAN.WG2.AIMLFW | Behavioural test (no formal conformance) |
| **SAC Policy Optimiser** | Same | Same |
| **Hard Constraint Projection (PFD/ITU mask)** | ITU-R S.1503 EPFD methodology, ITU-R Article 22 RR limits, FCC 47 CFR Part 25 §25.146 (NGSO EPFD) | Stress-test against ITU-R S.1503 reference cases |
| **Counterfactual Explanation Layer (H1)** | 3GPP TR 28.908 §AI/ML transparency, 3GPP TR 28.910 §AN levels, ETSI GS ENI 005 §explainability | Schema validation |
| **Federated Aggregator** | O-RAN.WG1.AIML-FL Federated Learning Architecture, 3GPP TS 28.105 §AI/ML model coordination | Per O-RAN AIML-FL |
| **A1 Policy Compiler** | O-RAN.WG2.A1AP, A1-GAP, A1-EI; aligned to 3GPP TS 23.503 (PCF policies) | O-RAN A1-IF-Conformance |
| **R1 Adapter (rApp ↔ SMO)** | O-RAN.WG2.R1AP, R1GAP | O-RAN R1-Conformance (drafts) |
| **O1 Adapter (telemetry ingest)** | O-RAN.WG10.O1-Interface, RFC 6241 NETCONF | O-RAN O1-IF-Conformance |
| **Cross-Operator Resource Trading (H3)** | TM Forum TMF921 (Intent Mgmt), TMF724 (AN Open API), 3GPP TS 28.530 §slicing roaming | New conformance test (we author) |
| **Evidence Store** | 3GPP TS 28.105 §AI/ML provenance, 3GPP TR 28.908 §audit, GDPR Article 30 (records of processing), NIST SP 800-92 (log management) | Audit-log retention test |
| **rApp Lifecycle Manager** | O-RAN.WG2.NON-RT-RIC-ARCH §rApp lifecycle, TMF724 | rApp registration/deregistration test |

### 2.2 xApp (parent project) Components

| Component | Governing Specs | Conformance Tests |
|---|---|---|
| **xApp Container** | O-RAN.WG3.NEAR-RT-RIC-ARCH §xApp framework | Near-RT RIC conformance suite |
| **E2 KPM Consumer** | O-RAN.WG3.E2AP, E2GAP, E2SM-KPM v05.00 | O-RAN E2-IF-Conformance |
| **E2 RC Actor (control)** | O-RAN.WG3.E2SM-RC v04.00 | O-RAN E2-IF-Conformance |
| **xApp policy receiver (A1 consumer)** | O-RAN.WG2.A1AP from receiving side | A1-IF-Conformance |
| **Encoder + Dynamics + Controller (1.6B-MoE / dense)** | O-RAN.WG2.AIML (model deployment), 3GPP TS 28.105 | AI/ML conformance |
| **Hard Constraint Layer (PFD)** | ITU-R S.1503, ITU-R Article 22, FCC Part 25 | EPFD validation |
| **EventGate inference optimization** | None (implementation detail) | Performance test only |

### 2.3 Common (shared infrastructure)

| Component | Governing Specs | Conformance Tests |
|---|---|---|
| **mTLS between all components** | O-RAN.WG11.Security-Protocols §TLS 1.3 mandates, 3GPP TS 33.310 §NDS/AF | O-RAN security test suite |
| **OAuth 2.0 / OpenID Connect** | O-RAN.WG11.Security-Protocols §authentication | Security test suite |
| **YANG models** | RFC 7950, 3GPP TS 28.541 (5G NRM YANG), O-RAN.WG4.MP-YANG | YANG validation (pyang/yanglint) |
| **Containerised deployment** | O-RAN.WG6.AAL-GAP, O-RAN.WG6.O-Cloud-Notification-Architecture | O-Cloud conformance |
| **Helm charts** | CNCF Helm best practices | Chart linting (helm lint, kubeval) |
| **Logging / observability** | O-RAN.WG10 OAM, ONAP VES 7.x | VES event schema validation |

---

## 3. Conformance Test Plan

Three tiers, each with concrete test artifacts.

### 3.1 Tier 1: Self-test (CI)
Run on every PR. Owns ~70% of conformance assertions.

| Test category | Tooling | Pass criterion |
|---|---|---|
| YANG schema validation | `pyang --strict --canonical` | 0 errors, 0 warnings on all O1/A1 schemas |
| ASN.1 message encoding (E2AP, A1AP) | `asn1c -fcompound-names` | Round-trip encode/decode for all message types in spec |
| OpenAPI schema (R1) | OpenAPI validator | Schema matches O-RAN R1 spec exactly |
| TLS 1.3 cipher suite enforcement | `testssl.sh` | Only WG11-permitted suites enabled |
| AI/ML model card (TS 28.105) | Custom validator | All required fields present per spec |
| Helm chart conformance | `helm lint`, `kubeval`, `polaris` | 0 critical, 0 high findings |
| Container image scanning | `trivy`, `grype` | 0 critical CVEs, ≤5 high CVEs aged < 30 days |
| Secrets scanning | `gitleaks`, `trufflehog` | 0 secrets detected |
| Static analysis | `bandit`, `semgrep` (security rules) | 0 high findings |
| Dependency SBOM | `syft` + `grype` | SBOM emitted in CycloneDX, no unpatched critical |

### 3.2 Tier 2: Integration test (lab)
Run nightly + before each release.

| Test category | Tooling | Pass criterion |
|---|---|---|
| **A1 conformance** | OSC NONRTRIC ric-plt-a1 + O-RAN.WG2.A1-IF-Conformance test cases | All A1 policy types CRUD operations green |
| **R1 conformance** | OSC NONRTRIC + R1 conformance drafts | rApp registers, subscribes, queries |
| **E2 conformance** | OSC RIC sim-e2-interface + WG3.E2-IF-Conformance | KPM subscription + RC control round-trip |
| **O1 conformance** | OSC SMO + WG10.O1-IF-Conformance | NETCONF subscribe to PM measurements per TS 28.552 |
| **AI/ML lifecycle (TS 28.105)** | Synthetic model lifecycle: register → train → deploy → monitor → retire | All states transition correctly |
| **Slicing (TS 28.530/541)** | NRM slice creation/modification/deletion via O1 | YANG validates, slice activates |
| **EPFD compliance (ITU-R S.1503)** | S.1503 reference test cases | All test cases pass within tolerance |
| **VES event schema (ONAP)** | VES schema validator | All emitted events parse |

### 3.3 Tier 3: Plugfest / certification
Run at major release.

| Forum | Frequency | Output |
|---|---|---|
| O-RAN Global PlugFest (Spring + Fall) | Twice yearly | Multi-vendor interop report |
| TIP RAN Lab certification | Yearly | TIP MUST badge |
| Keysight / VIAVI test campaign | Per release | Independent conformance report |
| AI-RAN Alliance reference contribution | Per major version | Reference architecture acceptance |
| Operator-specific lab (Nokia MantaRay App Store, Ericsson EIAP catalog) | Per marketplace listing | Marketplace publication |

---

## 4. Gap Analysis (current state vs. compliance target)

Honest current state of the codebase against standards.

| Area | Current state | Gap | Remediation |
|---|---|---|---|
| **A1 Policy emission** | Stub adapter exists in `xapp/canary.py` | A1AP v05 message types not implemented | Phase 1: implement full A1AP message set + conformance test |
| **R1 rApp registration** | Not implemented | Entire R1 stack missing | Phase 1: build R1 adapter against OSC NONRTRIC reference |
| **O1 telemetry ingest** | None | NETCONF/YANG ingestion missing | Phase 1: add `xapp/o1_adapter.py` (libyang + ncclient) |
| **E2 KPM consumer (xApp)** | Stubbed in `xapp/e2_adapter.py` | E2SM-KPM v05.00 not fully implemented | Parent project Phase 2: extend with full KPM v05.00 ASN.1 |
| **E2 RC actor (xApp)** | Not implemented | Need E2SM-RC v04.00 control message generation | Parent project Phase 2: implement RC actor |
| **YANG models** | None | TS 28.541 NRM YANG not vendored | Phase 1: vendor YANG, validate with pyang |
| **TLS 1.3 enforcement** | Default Python TLS, no audit | Need WG11-compliant cipher suite enforcement | Phase 1: explicit TLS config, conformance test |
| **OAuth 2.0 / mTLS** | Not implemented | Required by WG11 | Phase 1: integrate Vault + OIDC provider |
| **AI/ML model card (TS 28.105)** | None | Missing required fields | Phase 1: emit model card on every promotion |
| **EPFD compliance (S.1503)** | Geometric layer designed (M25 in parent) | Not validated against S.1503 reference test cases | Phase 1: add S.1503 test suite |
| **ITU-R P.452/676/837/838 (CWM physics)** | Not implemented | Need closed-form modules | Phase 1: implement (H2 paradigm), validate against ITU-Rpy |
| **Slicing NRM (TS 28.541)** | None | Slice-aware actions need NRM compliance | Phase 2 |
| **Federated Learning (AIML-FL)** | `federated/split_aggregators.py` exists, not O-RAN AIML-FL conformant | Need conformance test against AIML-FL ref architecture | Phase 2 |
| **Evidence audit (TS 28.105 provenance)** | `xapp/canary.py::PromotionLifecycle` close, missing fields | Add per-spec required provenance fields | Phase 1 |
| **Counterfactual explanation (TR 28.908 transparency)** | Not implemented | New feature (H1 paradigm) | Phase 1 (H1 delivers this) |
| **VES event emission** | None | Event reporting missing | Phase 2 |
| **TMF921 Intent API** | None | No intent ingestion | Phase 3 |
| **GDPR DPIA** | None | Required before pilot with EU operator | Phase 1 (start outline) |
| **NIST CSF mapping** | None | Required for US tier-1 telco pilot | Phase 1 (start outline) |
| **SBOM generation** | None | Required by EO 14028 for federal customers | Phase 1: integrate Syft into CI |

**Summary**: ~22 distinct conformance gaps. None are blocking the architecture. All are addressable within the 30-week PLAN-v1 roadmap (remediation plan defined; conformance test execution begins in Phase 1.5).

---

## 5. NTN-Specific Conformance (the hard parts)

These require the most attention because regulators verify them directly.

### 5.1 EPFD Compliance (ITU-R Article 22, RR-22)

Hard constraint. Any policy that violates EPFD masks **must** be impossible to emit.

**Required artifacts**:
1. Implementation of ITU-R S.1503 EPFD validation methodology (reference C++ available; we can wrap or reimplement)
2. EPFD masks for downlink (S.1503 Table 22-1A) and uplink (Table 22-2)
3. Test suite covering ALL 5 reference scenarios in ITU-R Recommendation
4. Integration into M25 hard constraint projection layer
5. Per-policy compliance certificate logged in evidence store

**Pre-pilot proof**: 100K random scenario test, 0 violations across 5 reference EPFD masks.

### 5.2 Antenna Pattern Compliance (ITU-R S.580, S.524, F.1336)

Required for operator-facing antenna assumptions.

**Required artifacts**:
1. Reference antenna patterns from ITU-R S.580 §3.1
2. Off-axis EIRP density limits per S.524
3. Reference patterns per F.1336 (ITU-R also publishes test code)
4. Validation in `planner/physics/beam_pattern.py`

**Pre-pilot proof**: per-band antenna pattern matches ITU-R reference within ≤0.5 dB across angles 0°–90°.

### 5.3 Propagation Compliance (ITU-R P.452/676/837/838/840)

Required for SLA-risk argumentation that operators trust.

**Required artifacts**:
1. P.452 closed-form (interference between earth stations)
2. P.676 gas attenuation (function of frequency, elevation, water vapor)
3. P.837 rainfall rate climatology (gridded global data)
4. P.838 specific rain attenuation (function of frequency, polarization, rain rate)
5. P.840 cloud/fog attenuation
6. Reference: `ITU-Rpy` python package — vendor or reimplement

**Pre-pilot proof**: closed-form modules match `ITU-Rpy` reference within ≤0.1 dB across the parameter space.

### 5.4 GSO Arc Protection (ITU RR Article 22 + FCC 47 CFR §25.146)

Hard regulatory floor.

**Required artifacts**:
1. GSO catalog ingestion (ITU SRS / FCC IBFS)
2. Geometric calculation of beam pointing → angle to each GSO arc
3. Attenuation lookup → predicted interference level
4. Comparison to EPFD mask → feasibility determination
5. Projection to feasible set if infeasible

**Pre-pilot proof**: 10K random LEO beam configurations tested; **0 violations**. Failure here is a regulatory non-starter.

### 5.5 3GPP NTN Channel Model (TR 38.811 / TS 38.811)

Required for synthetic data generation that operators trust.

**Required artifacts**:
1. NTN-TDL (Tapped Delay Line) channel models from TR 38.811 §6.7
2. NTN-CDL (Clustered Delay Line) channel models from §6.7
3. Doppler frequency model from §6.7.1
4. Atmospheric scintillation from §6.6
5. Implementation in Sionna or our own generator

**Pre-pilot proof**: generated channels match TR 38.811 reference statistics within ≤5%.

### 5.6 NTN System Architecture (TR 23.700-28 / TS 38.300 NTN sections)

Required to claim "NTN-aware" in marketing.

**Required artifacts**:
1. Support for both transparent and regenerative satellite payloads
2. Awareness of feeder link, service link, ISL distinctions
3. Support for the four NTN deployment scenarios (TR 23.700-28 §4.2)
4. Handling of NTN-specific timers (large RTT, beam switching)

**Pre-pilot proof**: scenario-specific Section-7-style metrics measured for each of 4 scenarios.

---

## 6. Compliance Attestation Roadmap

Different deployments require different attestations.

| Attestation | When required | Effort |
|---|---|---|
| **GDPR DPIA** | Any EU pilot | 2 weeks (contracted DPO) |
| **NIST CSF mapping** | Any US enterprise/federal pilot | 2 weeks (security architect) |
| **SOC 2 Type I** | Most enterprise pilots | 12 weeks + audit (~$30-50K) |
| **SOC 2 Type II** | Multi-customer GA | 12 months observation + audit (~$60-100K) |
| **ISO 27001** | EU enterprise pilots | 6-9 months + audit (~$50-100K) |
| **FedRAMP Moderate** | US federal | 12-18 months + 3PAO (~$500K-1M) |
| **NIS2 (EU)** | EU operator pilots | 6 months + audit |
| **PCI DSS** | Only if handling payment | n/a for PreceptualAI |
| **HIPAA** | Only if healthcare deployment | n/a likely |
| **FCC equipment auth** | If we ship hardware | n/a (we're software only) |
| **CE marking** | If we ship hardware | n/a |
| **TIP MUST certification** | For TIP marketplace | 6-8 weeks at TIP RAN Lab |
| **AI-RAN Alliance reference** | For AI-RAN Alliance promotion | per their submission process |

---

## 7. Pilot-Ready Conformance Checklist (Tier 1)

What MUST be true before any pilot can begin (regardless of customer):

### Standards conformance
- [ ] A1AP v05 conformance tests green (at minimum: policy CRUD, EI subscriptions)
- [ ] R1AP / R1GAP rApp registration green
- [ ] O1 NETCONF subscribe to TS 28.552 PM measurements green
- [ ] E2 conformance green (for paired xApp deployment)
- [ ] YANG models pass `pyang --strict`
- [ ] TLS 1.3 with WG11 cipher suite list enforced
- [ ] mTLS between rApp ↔ SMO, rApp ↔ aggregator, rApp ↔ edge agents
- [ ] OAuth 2.0 / OpenID Connect for human access to operator dashboard
- [ ] AI/ML model card emitted on every promotion (TS 28.105 fields)

### NTN compliance
- [ ] ITU-R S.1503 EPFD test cases pass (5/5)
- [ ] GSO arc projection: 10K random scenarios → 0 violations
- [ ] ITU-R P.452/676/837/838 closed-forms validated against ITU-Rpy
- [ ] ITU-R F.1336 / S.580 / S.524 patterns validated
- [ ] FCC IBFS catalog parser produces valid GSO arc dataset
- [ ] 3GPP TR 38.811 channel models implemented + statistically validated

### Security
- [ ] All container images < 30 days CVE-free at critical
- [ ] SBOM emitted in CycloneDX format for all artifacts
- [ ] Pen test passed (0 critical findings, ≤5 medium remediated)
- [ ] Secrets management: HashiCorp Vault integrated, no hardcoded secrets
- [ ] Audit log retention: ≥6 months, write-once storage (S3 Object Lock or equivalent)

### Privacy
- [ ] GDPR DPIA completed and signed by external DPO (if EU pilot)
- [ ] Data residency controls (raw KPMs never leave operator data plane — federation is weights-only)
- [ ] Operator-controlled kill switch (regulator/operator manual override)

### Operations
- [ ] Helm charts for OSC NONRTRIC + at least one vendor SMO (MantaRay or EIAP)
- [ ] Prometheus metrics exporter (latency, error rate, queue length, drift severity)
- [ ] Grafana dashboards for operator visibility
- [ ] Runbook for top 10 failure modes
- [ ] Disaster recovery plan: RPO ≤ 1 hour, RTO ≤ 4 hours

### Documentation
- [ ] OpenAPI 3.1 spec for all rApp APIs
- [ ] Operator deployment guide (Helm install, configuration, validation)
- [ ] Compliance attestations dossier (DPIA, NIST CSF mapping, SBOMs)
- [ ] Counterfactual explanation user guide (per H1 paradigm)
- [ ] Incident response playbook

### Customer-specific
- [ ] Compatibility verified with customer's specific O-RAN vendor stack
- [ ] On-site / lab environment provisioned
- [ ] Customer data integration approved by their security review
- [ ] Acceptance test plan signed by customer technical lead
- [ ] Pilot SLA agreement signed by both legal teams

---

## 8. Standards-Driven Risk Register

| Risk | Standard | Mitigation |
|---|---|---|
| O-RAN spec versions move faster than we can keep up | All | Pin to released versions; track changes via O-RAN Alliance change requests; allow 1-quarter lag on new specs |
| 3GPP NTN spec ambiguity (Rel-18 NTN architecture still evolving) | TR 23.700-28 | Implement against TS 38.811 v17 baseline; add Rel-19 features only after they stabilize |
| ITU-R EPFD validation expensive (S.1503 reference is 1000s of test points) | S.1503 | Start with 5-case minimum; expand iteratively |
| AI/ML conformance is immature in 3GPP TS 28.105 | TS 28.105 | Implement minimum required fields; over-deliver via H1 counterfactual |
| Customer SMO vendor uses non-standard extensions | Vendor-specific | Maintain vendor-shim layer; convert extensions to standard A1AP |
| Federated Learning standards (O-RAN AIML-FL) just emerging | AIML-FL | Implement to current draft; refactor as standard finalises |
| Cross-operator trading (H3) has no precedent in standards | TM Forum / 3GPP | Author reference; submit to AI-RAN Alliance for standardization |

---

## 9. Standards Update Cadence

How we track changes.

| Source | Frequency | Owner |
|---|---|---|
| O-RAN Alliance plenary | Quarterly | Compliance lead |
| 3GPP RAN/SA/CT plenaries | Quarterly | Standards architect |
| ITU-R Study Group meetings | Twice yearly | Spectrum compliance lead |
| FCC technical docket updates | Monthly | Regulatory counsel |
| TM Forum API releases | Twice yearly | Integration architect |
| AI-RAN Alliance + TIP working groups | Monthly | Technical lead |
| Internal review against pinned versions | Sprintly | Engineering |

A spec-change automatically triggers:
1. Impact assessment (≤ 5 days)
2. Backlog grooming for any changed conformance tests
3. Customer notification if breaking change

---

## 10. Honest Summary

**What this document does**: maps every architectural component to specific standards, identifies the 22 distinct conformance gaps in our current code, and gives a tiered conformance test plan.

**What it doesn't do**: turn the architecture into shipping product overnight. The 22 gaps are real engineering work — 18 of them are addressable in PLAN-v1 Phases 1–2 with the existing 30-week roadmap. The remaining 4 (cross-operator trading, advanced O-RAN AIML-FL conformance, FedRAMP, full slicing NRM) are Phase 3+ scope.

**Pilot-ready means**: Tier 1 conformance + Phase 1 deliverables (per PLAN-v1) + the pilot-ready checklist in §7. **8–10 weeks of focused work after Phase 1 ends**. Total 18–20 weeks from start to first pilot launch.

**Production-ready means** (multi-customer GA): Tier 2 conformance + SOC 2 Type II + ISO 27001 + multi-vendor SMO certifications. **18–24 months after first pilot**.

The standards landscape is dense but tractable. Every spec we cite has a clear conformance path. The work is real but bounded. Ship Phase 1, get pilot-ready, demonstrate, then scale.
