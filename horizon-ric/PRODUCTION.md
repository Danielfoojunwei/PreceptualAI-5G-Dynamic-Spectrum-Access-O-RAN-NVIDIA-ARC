# Production & Pilot Readiness

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


> Three-tier checklist: **Pilot-Ready** → **Production-Ready** → **GA-Ready**. Each tier is gated by concrete acceptance criteria. Pair this with `STANDARDS.md` (which covers protocol conformance) for the full picture.

---

## 0. The three tiers

| Tier | Definition | Customer-facing label | Effort beyond Phase 1 |
|---|---|---|---|
| **Tier 1: Pilot-Ready** | Can deploy in operator's lab/staging with their data, under joint operator+vendor supervision | "Lab pilot" / "PoC" / "Trial" | +8–10 weeks after Phase 1 |
| **Tier 2: Production-Ready** | Can deploy in operator's production network handling real subscriber traffic, single tenant | "Limited GA" / "Production trial" | +6–9 months after first pilot success |
| **Tier 3: GA-Ready** | Multi-tenant, multi-region, multi-vendor, marketplace listing | "General Availability" | +12–18 months after Tier 2 |

This document specifies what each tier requires.

---

## 1. Tier 1: Pilot-Ready Checklist

**Goal**: deploy in a single operator's lab + staging environment with their representative telemetry. Joint engineering oversight from both vendor and operator. Not yet handling real subscriber traffic.

### 1.1 Functional acceptance

- [ ] All Phase 1 PLAN-v1 deliverables green (per `PLAN-v1.md` §3 Phase 1)
- [ ] All H1 (counterfactual) and H2 (compositional world model) paradigms shipped per `PARADIGMS.md`
- [ ] All Tier 1 conformance gates green (per `STANDARDS.md` §3.1 + §3.2)
- [ ] Rollback verified: revert to previous policy version completes in ≤30 s
- [ ] Maritime scenario end-to-end demo runnable in 10 minutes from Helm install (currently runnable as `python scripts/e2e_simulation.py`; Helm chart in Phase 1.5)

### 1.2 Standards conformance (Tier 1)

See `STANDARDS.md` §7 — referenced here for completeness:
- [ ] O-RAN A1AP v05, R1AP, O1, E2 conformance tests green
- [ ] ITU-R S.1503 EPFD validation: 5 reference test cases pass
- [ ] GSO arc projection: 10K random scenarios → 0 violations
- [ ] 3GPP TS 28.105 AI/ML model card emitted on every promotion
- [ ] 3GPP TR 38.811 NTN channel models statistically validated

### 1.3 Security baseline

- [ ] **Cryptography**: TLS 1.3 enforced, FIPS 140-3-validated cryptographic modules where required (BoringSSL FIPS, OpenSSL 3.0 FIPS provider, or equivalent)
- [ ] **mTLS**: between every component pair (rApp ↔ SMO, edge agent ↔ aggregator, etc.)
- [ ] **OAuth 2.0 / OIDC**: human access via operator IdP (Keycloak / Okta / Azure AD)
- [ ] **RBAC**: per-component role definitions; least-privilege defaults
- [ ] **Secrets management**: HashiCorp Vault (or equivalent); zero hardcoded secrets in code
- [ ] **Container scanning**: Trivy/Grype in CI; gate at 0 critical CVEs
- [ ] **Image signing**: cosign signatures on all images; verifying admission controller in K8s
- [ ] **SBOM**: CycloneDX SBOM emitted per build (Syft); signed and attached to release
- [ ] **SLSA Level 2** (build provenance for all artifacts; signed builds)
- [ ] **Penetration test**: independent test by qualified vendor; 0 critical findings, ≤5 medium remediated
- [ ] **Secure coding review**: Bandit (Python), gosec (Go), cargo-audit (Rust); all findings triaged
- [ ] **Dependency management**: Renovate or Dependabot enabled; CVE alerts <30 days remediation SLA
- [ ] **Static analysis**: Semgrep security rules in CI; zero high-severity allowed
- [ ] **Audit logging**: structured (JSON) logs for every privileged action; 6-month retention
- [ ] **Audit log integrity**: write-once storage (S3 Object Lock / similar)

### 1.4 Privacy & compliance

- [ ] **GDPR DPIA** (if any EU operator): contracted external DPO, signed report
- [ ] **NIST CSF mapping** (if US operator): documented controls mapping to CSF 2.0
- [ ] **Data residency**: raw KPMs never leave operator data plane; verified in architecture review
- [ ] **Data minimisation**: only metrics specified in TS 28.552 are collected; documented schema
- [ ] **Right-to-erasure** (GDPR Art. 17): per-subject data deletion API documented
- [ ] **Records of processing** (GDPR Art. 30): inventory in compliance dossier
- [ ] **Operator override / kill switch**: regulator-mandated manual override path; tested
- [ ] **Cross-border transfer** (if applicable): SCCs or BCRs in place

### 1.5 Reliability & operations

- [ ] **High Availability**: rApp deployable in active-active configuration with ≥2 replicas
- [ ] **Liveness / readiness probes**: K8s probes for every container
- [ ] **Resource limits**: CPU/memory limits and requests for every pod
- [ ] **Pod Disruption Budget**: PDB defined for stateful components
- [ ] **Backup**: TimescaleDB and MinIO backed up nightly; restore tested monthly
- [ ] **Disaster Recovery**: RPO ≤ 1 hour, RTO ≤ 4 hours; documented + tested
- [ ] **Graceful degradation**: rApp continues operating with degraded predictions if any single component fails
- [ ] **Circuit breakers**: timeout + retry + circuit breaker pattern for every external call
- [ ] **Rate limiting**: protect against runaway feedback loops
- [ ] **Chaos test**: kill any single pod, verify system recovers within SLO

### 1.6 Observability

- [ ] **Prometheus metrics** exported per `monitoring/metrics.py`:
  - Inference latency (p50, p95, p99)
  - Error rate by endpoint
  - Queue length per stage
  - Drift severity per detector
  - Federated round duration
  - SLA risk distribution (histogram)
  - Constraint violation count (must alert at > 0)
  - Active rApp count, A1 policies emitted/sec
- [ ] **OpenTelemetry tracing**: trace ID propagation across rApp ↔ SMO ↔ Near-RT RIC
- [ ] **Structured logging**: JSON logs with trace IDs, severity, redacted PII
- [ ] **Log aggregation**: Loki / ELK / Splunk integration documented
- [ ] **Grafana dashboards**: at least 4 dashboards (operations, drift, decisions, compliance)
- [ ] **Synthetic monitoring**: continuous probe simulating typical operator query
- [ ] **Alerting**: PagerDuty/Opsgenie integration; on-call rotation defined for pilot

### 1.7 Performance

- [ ] **A1 policy emission**: p95 < 2s end-to-end (per PLAN-v1 Phase 1 gate)
- [ ] **Risk prediction**: p95 < 1s
- [ ] **Federated round**: ≤ 5 minutes for 10-site convergence
- [ ] **Throughput**: rApp can process ≥1000 risk-prediction queries/sec on a 4-core 16GB pod
- [ ] **Scalability test**: scale-out tested to 10x baseline replica count

### 1.8 Documentation

- [ ] **Architecture document** (`docs/architecture.md`): deployment topology, data flow, component descriptions
- [ ] **Operator Deployment Guide** (`docs/operator_guide.md`): step-by-step Helm install, configuration, validation
- [ ] **OpenAPI 3.1 spec** (`docs/api/openapi.yaml`): every rApp REST/gRPC endpoint
- [ ] **Configuration reference**: every config field, default, allowed range
- [ ] **Runbooks** (`docs/runbooks/`):
  - rApp pod failure
  - SMO connectivity loss
  - Federated convergence failure
  - Drift alert response
  - Constraint violation alert response
  - Database backup/restore
  - Certificate rotation
  - Security incident response
  - Customer escalation playbook
  - On-call rotation procedure
- [ ] **Incident response playbook** (security + reliability)
- [ ] **Compliance dossier**:
  - GDPR DPIA (if EU)
  - NIST CSF mapping (if US)
  - SBOM
  - Pen test report
  - Conformance test reports
- [ ] **Customer-facing data sheet**: 2-page PDF
- [ ] **Counterfactual explanation user guide** (H1)

### 1.9 Customer-specific

- [ ] **Vendor compatibility**: tested against customer's specific SMO (MantaRay/EIAP/OSC) version
- [ ] **Network compatibility**: tested against customer's specific RAN vendor (Nokia/Ericsson/Samsung/Mavenir)
- [ ] **Data integration**: agreed with customer security review
- [ ] **Lab environment**: provisioned and accessible to joint engineering team
- [ ] **Acceptance test plan**: signed by customer technical lead
- [ ] **Pilot SLA**: signed by both legal teams (uptime targets, support response, escalation)
- [ ] **Joint runbook**: customer + vendor on-call procedures

---

## 2. Tier 2: Production-Ready Checklist (additions over Tier 1)

**Goal**: deploy in operator's production network handling real subscriber traffic. Single tenant. May still be limited geography.

### 2.1 Standards conformance (Tier 2 additions)

- [ ] All Tier 2 conformance gates green (per `STANDARDS.md` §3.2)
- [ ] O-RAN PlugFest certificate (Spring or Fall) within last 12 months
- [ ] Independent third-party conformance report (Keysight or VIAVI)
- [ ] Federated Learning conformance against O-RAN AIML-FL specifications
- [ ] Slicing NRM (3GPP TS 28.541) full conformance for slice-aware actions

### 2.2 Security additions

- [ ] **SOC 2 Type I**: external audit report
- [ ] **SLSA Level 3**: hermetic, reproducible builds
- [ ] **Vulnerability response SLA**: critical CVE patched within 7 days; high within 30 days
- [ ] **Security incident response**: 24/7 on-call security
- [ ] **Bug bounty program**: HackerOne or equivalent, with documented scope
- [ ] **Threat modeling refresh**: STRIDE/PASTA review per major release
- [ ] **Hardware Security Module (HSM)**: keys for signing models stored in HSM (AWS CloudHSM / Azure Key Vault HSM / on-prem)
- [ ] **Zero-trust architecture**: BeyondCorp-style; service-to-service auth via SPIFFE/SPIRE
- [ ] **SIEM integration**: Splunk/Sentinel/Elastic Security; security events forwarded
- [ ] **Penetration testing**: quarterly third-party tests
- [ ] **Red team exercise**: annual

### 2.3 Privacy additions

- [ ] **ISO 27001 certification** (if EU operator or large enterprise)
- [ ] **NIS2 compliance** (if EU)
- [ ] **Privacy Impact Assessment (PIA)** updated per major release
- [ ] **Vendor Risk Assessment** completed by customer

### 2.4 Reliability additions

- [ ] **Multi-region deployment**: active-active across 2 regions or active-passive
- [ ] **RPO ≤ 5 minutes, RTO ≤ 30 minutes** for rApp instance
- [ ] **99.95% availability SLA** (≤ 4.4 hours downtime/year)
- [ ] **Game day exercise**: quarterly
- [ ] **Chaos engineering**: continuous (Chaos Mesh / Litmus / Gremlin)
- [ ] **Capacity planning**: documented projected load, headroom, scaling triggers
- [ ] **Performance regression testing**: every release
- [ ] **Failure mode catalog**: top 50 failure modes with mitigation

### 2.5 Observability additions

- [ ] **Distributed tracing** end-to-end with OpenTelemetry
- [ ] **APM**: Datadog / Dynatrace / New Relic
- [ ] **Log analytics**: ML-based anomaly detection on logs
- [ ] **Synthetic monitoring** from multiple geographic locations
- [ ] **Real User Monitoring (RUM)** of operator dashboard
- [ ] **SLO + error budget**: defined per service, tracked in dashboard

### 2.6 Operations additions

- [ ] **24/7 on-call rotation** (vendor-side)
- [ ] **Incident management process**: ITIL-aligned or SRE-style
- [ ] **Change management**: documented CAB process
- [ ] **Release management**: feature flags for every new capability
- [ ] **Blue/green deployment** or canary deployment for every release
- [ ] **Rollback automation**: one-command revert to last known good
- [ ] **Database migrations**: Flyway/Liquibase, backwards-compatible only
- [ ] **Schema versioning**: API + data schema versioned independently

### 2.7 Documentation additions

- [ ] **Architecture Decision Records** (ADRs) for every major decision
- [ ] **API versioning policy**: documented compatibility guarantees
- [ ] **Customer support documentation**: knowledge base, ticket templates
- [ ] **Compliance attestations**: SOC 2 Type I report, ISO 27001 certificate
- [ ] **Pricing documentation**: contractual basis for billing

---

## 3. Tier 3: GA-Ready Checklist (additions over Tier 2)

**Goal**: multi-tenant, multi-region, multi-vendor, marketplace listing. Anyone can buy and deploy.

### 3.1 Standards conformance (Tier 3 additions)

- [ ] All Tier 3 conformance gates green
- [ ] **TIP MUST badge** (TIP RAN Lab certification)
- [ ] **AI-RAN Alliance reference contribution** accepted
- [ ] Multi-vendor interop: green against ≥3 SMO vendors
- [ ] Multi-vendor interop: green against ≥5 RAN vendors

### 3.2 Security additions

- [ ] **SOC 2 Type II**: 12-month observation period audited
- [ ] **ISO 27017** (cloud security) and **ISO 27018** (PII protection) if cloud-deployed
- [ ] **FedRAMP Moderate** (if US federal target)
- [ ] **CSA STAR** (Cloud Security Alliance)
- [ ] **SLSA Level 4**: two-party review of all build pipelines

### 3.3 Reliability additions

- [ ] **99.99% availability SLA** (≤ 52 minutes/year)
- [ ] **Multi-region active-active** with automatic failover
- [ ] **RPO ≤ 1 minute, RTO ≤ 15 minutes**
- [ ] **Cross-cloud portability**: deployable on AWS, Azure, GCP, on-prem
- [ ] **Hyperscaler partnerships**: AWS Marketplace, Azure Marketplace, GCP Marketplace listings

### 3.4 Operations additions

- [ ] **Multi-tenant isolation**: hard tenant boundaries (separate DBs, separate K8s namespaces, separate KMS keys)
- [ ] **Self-service onboarding**: customer-facing portal for new operator deployment
- [ ] **Usage metering & billing**: per-customer metering of API calls, data volume, etc.
- [ ] **Customer success organization**: CSMs, technical account managers
- [ ] **Tiered support**: T1 / T2 / T3 with documented SLAs
- [ ] **Marketplace listings**: Nokia MantaRay App Store, Ericsson EIAP catalog, AI-RAN Alliance catalog

### 3.5 Ecosystem

- [ ] **Partner certifications**: certified integrations with ≥3 RAN vendors, ≥2 SMO vendors
- [ ] **Reference customers**: ≥3 named operator customers
- [ ] **Case studies**: ≥3 published case studies
- [ ] **Community**: developer documentation, sample rApps, conference presence

---

## 4. Cross-Tier: Open Questions to Resolve Before Phase 1

These are architectural / strategic decisions that block Phase 1 work and require explicit answers.

| # | Question | Default | Decision needed by |
|---|---|---|---|
| 1 | Which SMO platform for first lab integration? | OSC NONRTRIC reference | Phase 1 week 1 |
| 2 | Which operator for first pilot? | Maritime/port operator (most accessible) | Phase 2 |
| 3 | Build proprietary HE library or use SEAL/HElib? | Use Microsoft SEAL (proven, audited) | Phase 3 |
| 4 | Which language for edge agent: Rust or Go? | Rust (memory safety + performance) | Phase 1 week 2 |
| 5 | Which ML framework: PyTorch or JAX? | PyTorch (existing parent codebase) | Already decided |
| 6 | Open-source the platform core? | Yes (Apache 2.0 like parent project) | Strategic decision needed |
| 7 | License model: subscription / per-deployment / open-core? | Open-core (free OSS, paid enterprise) | Strategic decision needed |
| 8 | Single-tenant or multi-tenant from day 1? | Single-tenant; multi-tenant in Tier 3 | Already decided |
| 9 | Which cloud-native stack: vanilla K8s or OpenShift? | Vanilla K8s + Helm; OpenShift charts later | Phase 1 week 1 |
| 10 | Build evidence dashboard from scratch or use existing tool (Grafana + plugins)? | Grafana plugins for Tier 1; custom UI for Tier 2 | Phase 1 week 4 |

---

## 5. Effort & Timeline (concrete numbers)

| Tier | Effort (calendar) | Effort (engineering hours) | Compute | External cost (procurement-driven) |
|---|---|---|---|---|
| **Tier 1: Pilot-Ready** | 18-20 weeks total (Phase 1 + Tier 1 hardening) | ~600 eng-hrs | ~75 GB10-hours | **$0–50K** depending on pilot customer profile (see below) |
| **Tier 2: Production-Ready** | +6-9 months after first pilot success | ~2000 eng-hrs | ~150 GB10-hours/year | **$50K–200K** (SOC 2 Type I + advanced security tools) |
| **Tier 3: GA-Ready** | +12-18 months after Tier 2 | ~5000 eng-hrs (incl. multi-tenant rebuild) | ~500 GB10-hours/year | **$200K–1M+** (FedRAMP, multi-region, ISO certs) |

### External cost is procurement-driven, not engineering-driven

The code, architecture, and conformance tests are the same at $0 external cost as at $50K. What changes is **which customer's vendor-management organization will accept your pilot**.

**Tier 1 external-cost breakdown**:

| Pilot customer | Cost | Components |
|---|---|---|
| Open-source / community pilot, friendly small operator | **$0** | Self-DPIA from CIPP/E or IAPP templates, OWASP ZAP automated scan, template MSA from Y Combinator or Andreessen Horowitz open-source legal templates |
| Mid-tier operator (T2/T3 telco), private 5G enterprise | **$15–25K** | Lightweight DPO consultation ($5K) + HackerOne-managed bug bounty ($10–15K) + internal legal review |
| Tier-1 telco EU procurement (Vodafone, DT, Telefónica, Orange) | **$30–50K** | External DPO contract ($5–15K) + reputable third-party pen test ($20–40K) + external legal review ($5–15K) |
| Tier-1 telco + government/regulated (FirstNet, military) | **$100K+** | Adds FedRAMP-aligned controls, formal threat modeling, supply-chain attestations |

**Tier 2 external-cost breakdown**:

| Item | Cost | When required |
|---|---|---|
| SOC 2 Type I audit | $30–50K | Most enterprise pilots demand |
| HSM (cloud or on-prem) | $30–100K | Required for production model signing |
| Quarterly third-party pen tests | $20–40K each | Per quarter |
| Red team exercise | $20–30K | Annual |
| SIEM platform (Splunk/Sentinel/Elastic) | $20–40K/year | Production observability |
| Bug bounty program management | $10–20K/year | Continuous security |

**Tier 3 external-cost breakdown**:

| Item | Cost | When required |
|---|---|---|
| SOC 2 Type II annual audit | $60–100K | Multi-customer GA |
| ISO 27001 certification | $50–100K initial + $30K/year | EU enterprise GA |
| FedRAMP Moderate ATO | $500K–1M | US federal customer |
| ISO 27017 + 27018 (cloud + PII) | $30–50K | Cloud-native deployments |
| NIS2 audit (EU) | $30–60K/year | EU operator GA |
| Multi-region cloud infrastructure | $100–500K/year | Operating cost |

### How to think about it as a startup

You can ship Tier 1 with **$0 external cost** if your first pilot customer is friendly enough. You can defer all the audits until a customer specifically asks for them. The audits cost money because operators with formal procurement *require third-party validation* — but most pilots can begin without them.

The trap to avoid: spending $50K+ on audits *before* you have a pilot lined up. Audits are valid for 12 months; if you do them too early, you'll pay again.

The recommended sequence:
1. **Phase 1.5** with $0–15K (just enough to be presentable)
2. **Find a pilot customer** — their procurement defines the next round of external cost
3. **Spend on audits** matching that customer's specific requirements
4. **Layer additional certifications** as customer count grows

Real costs vary by team size, customer demands, geography, and regulated industries.

---

## 6. Pilot-Ready Day-1 Activities

When greenlit, the first 14 days should produce:

| Day | Activity | Owner |
|---|---|---|
| 1 | Set up `horizon-ric/src/horizon_ric/` skeleton; CI baseline (lint + test) | Engineering |
| 1-2 | Stand up OSC NONRTRIC reference deployment in lab | Engineering |
| 2-3 | Decide pilot customer profile + start outreach | Sales |
| 3-5 | Implement R1 adapter skeleton (registration only) | Engineering |
| 5 | Verify rApp registers with NONRTRIC | Engineering |
| 5-7 | Begin GDPR DPIA scoping (engage DPO) | Compliance |
| 5-7 | Begin NIST CSF mapping outline | Security |
| 7-10 | Start Phase 1 maritime synthetic generator | Engineering |
| 10-12 | Begin H2 physics modules (`planner/physics/`) | Engineering |
| 12-14 | Begin compliance dossier structure | Compliance + Engineering |

By end of week 2: rApp skeleton runs in OSC NONRTRIC, compliance scope documented, physics modules under construction.

---

## 7. Bottom Line

Pilot-ready in 18-20 weeks total (Phase 1 + 8-10 weeks Tier 1 hardening). Production-ready 6-9 months after first pilot success. GA-ready 12-18 months after that.

**The standards work is real but bounded**. Every spec we cite has a clear conformance path. The hardening work is mostly engineering hygiene, not novel research. The 22 conformance gaps in `STANDARDS.md` are addressable in the existing 30-week PLAN-v1 roadmap with disciplined execution.

**The one thing that can derail this**: rushing pilot deployment without completing the Tier 1 checklist. Operators tolerate slow vendors. They do not tolerate compliance failures or security incidents. Better to take 20 weeks and pass than 14 weeks and fail.
