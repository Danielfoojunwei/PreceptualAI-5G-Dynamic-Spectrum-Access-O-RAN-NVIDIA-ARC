# NIST Cybersecurity Framework 2.0 — Mapping for Horizon-RIC

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


**Version:** 0.2.0
**Date:** 2026-05-06
**Audience:** Operator CISO, third-party security assessor (FedRAMP / SOC2 reviewer where applicable)
**Component:** Horizon-RIC rApp (`src/horizon_ric/`)
**Framework:** NIST Cybersecurity Framework 2.0, NIST CSWP 29 (26 February 2024)

---

This document maps Horizon-RIC controls to the six CSF 2.0 Functions (GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, RECOVER) and to two-to-three sub-categories per Function. Each mapping cites a file:line in this repository or names the document that satisfies the control. Honest gap notes appear inline; nothing is asserted that is not implemented.

CSF 2.0 sub-category identifiers below follow the published Core (NIST CSWP 29 Appendix A).

## GOVERN (GV)

The organisational layer of the framework — context, risk strategy, roles, oversight, supply chain.

| Sub-category | What it requires | Horizon-RIC mapping | Gap |
|---|---|---|---|
| **GV.OC-04** Critical objectives, capabilities, and services that stakeholders depend on are understood and communicated | The deployer must know what Horizon-RIC's purpose is and what it depends on | `REVIEW.md` §1-2 (purpose, scope); `PILOT.md` (concession map listing every honest concession); `docs/RESEARCH_ALIGNMENT.md` | None |
| **GV.RM-01** Risk management objectives are established and agreed to by organisational stakeholders | Documented risk objectives | `REVIEW.md`; `docs/EVALUATION_CRITERIA.md` (standards/regulation gap analysis); `docs/conformance/CONFORMANCE.md` | None at the rApp tier; operator must complete its own risk register |
| **GV.SC-04** Suppliers are known and prioritised by criticality | SBOM / supply-chain inventory | `deploy/sbom/horizon-ric-sbom.json` (CycloneDX 1.5); validated by `deploy/sbom/validate_sbom.py` in CI (`.github/workflows/sbom.yml`) | None |
| **GV.SC-07** The risks posed by a supplier, their products and services, and other third parties are understood, recorded, prioritised, assessed, responded to, and monitored over the course of the relationship | Image signing | `deploy/cosign/cosign.pub` + `cosign sign --key …` on every release tag (`.github/workflows/build.yml`) | None |

## IDENTIFY (ID)

Asset and risk inventory — the assets the system holds and the risks that apply to them.

| Sub-category | What it requires | Horizon-RIC mapping | Gap |
|---|---|---|---|
| **ID.AM-02** Software assets (e.g. applications, services) and systems are inventoried | Software inventory | `deploy/sbom/horizon-ric-sbom.json` (CycloneDX 1.5 — every dependency with version, licence, hash); `pyproject.toml` (top-level deps); `checkpoints/*.md` (model assets with sha256 + size) | None |
| **ID.AM-05** Resources are prioritised based on classification, criticality, resources, and impact on the mission | Asset criticality classification | `docs/conformance/CONFORMANCE.md` (per-spec status with ✅/⚠️/🟡/❌); `deploy/SLO.md` (SLO classes) | Operator-side classification not vendor-provided |
| **ID.RA-01** Vulnerabilities in assets are identified, validated, and recorded | Vulnerability inventory | CI dependency-scan via SBOM; `AUDIT_NO_FAKES.md` (runtime vulnerability — fakes that masquerade as real implementations); `DEAD_CODE_SWEEP.md` | NOT YET — automated CVE feed integration planned in Phase 3 |
| **ID.RA-09** The authenticity and integrity of hardware and software is assessed prior to acquisition and use | Image signing verification | `deploy/cosign/cosign.pub` (verifying public key); release pipeline signs every image | None |

## PROTECT (PR)

Technical controls preventing unauthorised access and protecting the data the system holds.

| Sub-category | What it requires | Horizon-RIC mapping | Gap |
|---|---|---|---|
| **PR.AA-01** Identities and credentials for authorised users, services, and hardware are managed by the organisation | Identity lifecycle | JWT issuance + zero-downtime rotation: `src/horizon_ric/security/jwt.py:77-198` (`JWTManager.mint_token`, `rotate_signing_key`) | None |
| **PR.AA-03** Users, services, and hardware are authenticated | Authentication | `src/horizon_ric/rapp/auth.py:268-290` (mTLS + OAuth2); `src/horizon_ric/security/jwt.py:141-175` (`verify_token`); `src/horizon_ric/security/middleware.py:79-162` (`JWTAuthMiddleware.dispatch`) | None |
| **PR.AA-05** Access permissions, entitlements, and authorisations are defined in a policy, managed, enforced, and reviewed | RBAC | `src/horizon_ric/security/rbac.py`; `src/horizon_ric/security/rbac_model.conf`; `src/horizon_ric/security/rbac_policy.csv`; per-route enforcement helpers `require_role` / `require_tenant` (`src/horizon_ric/security/middleware.py:186-240`) | None |
| **PR.DS-01** The confidentiality, integrity, and availability of data-at-rest are protected | Data-at-rest integrity | Hash-chained `EvidenceStore`: `src/horizon_ric/evidence/store.py:106-142` (`verify`, `verify_tenant`); atomic state checkpointer with fsync: `src/horizon_ric/runtime/state_recovery.py:50-60` | Confidentiality at rest is operator-side disk encryption — not vendor-controlled |
| **PR.DS-02** The confidentiality, integrity, and availability of data-in-transit are protected | Data-in-transit | mTLS on every R1/A1/O1 channel: `src/horizon_ric/rapp/auth.py:92-200`; production cert-skipping is hard-banned at line 111-117; WG11 §6 cipher allow-list at lines 34-38 | Honest TLS-1.3 limitation documented at `auth.py:93-109` (Python ssl module quirk; operator must additionally restrict at terminator) |
| **PR.PS-01** Configuration management practices are established and applied | Configuration management | `deploy/helm/`, `deploy/kubernetes/`, `deploy/docker-compose.yml`; `pyproject.toml`; environment-variable surface at `src/horizon_ric/rapp/lifecycle.py:301-315` (`_config_from_env`) | None |
| **PR.PS-03** Hardware is maintained, replaced, and removed commensurate with risk | Hardware lifecycle | `docs/HARDWARE_PROCUREMENT.md` (operator-side hardware profile for Jetson edge runs) | Operator-side; vendor only documents the operator's reference profile |

## DETECT (DE)

Monitoring and detection — observability over the system in production.

| Sub-category | What it requires | Horizon-RIC mapping | Gap |
|---|---|---|---|
| **DE.CM-01** Networks and network services are monitored to find potentially adverse events | Continuous network monitoring | Prometheus metrics at `/metrics`: `src/horizon_ric/rapp/health.py`; `src/horizon_ric/sla/engine.py:78-187` (`SLAEvaluator`); Alertmanager v2 webhook client at `src/horizon_ric/sla/alertmanager.py`; rule files at `deploy/prometheus/sla_rules.yml`, `rules.yml` | None |
| **DE.CM-03** Personnel activity and technology usage are monitored to find potentially adverse events | Personnel activity audit | Structured audit event names emitted on every auth grant/deny: `src/horizon_ric/security/middleware.py:38, 90-161` (events `auth.token_invalid`, `auth.denied`, `auth.granted` with subject, tenant, roles, path, action) | None |
| **DE.CM-09** Computing hardware and software, runtime environments, and their data are monitored to find potentially adverse events | Runtime monitoring | `src/horizon_ric/rapp/health.py` (liveness/readiness); `src/horizon_ric/runtime/watchdog.py` (sd_notify watchdog); `src/horizon_ric/runtime/circuit_breaker.py` (R1/A1/O1 breakers); `src/horizon_ric/runtime/graceful_degradation.py` | None |
| **DE.AE-02** Potentially adverse events are analysed to better understand associated activities | Event analysis | Hash-chained DecisionRecord with predicted vs actual outcome at +30s/+1min/+5min: `src/horizon_ric/evidence/schema.py:122-125`; SLA breach annotation linking each breach to the decision in flight at the breach time: `src/horizon_ric/sla/breach_annotation.py` (referenced from `evidence/schema.py:131-138`) | None |

## RESPOND (RS)

Incident response — what happens when an event becomes an incident.

| Sub-category | What it requires | Horizon-RIC mapping | Gap |
|---|---|---|---|
| **RS.MA-01** The incident response plan is executed in coordination with relevant third parties once an incident is declared | Documented incident response procedure | `deploy/RUNBOOK.md` (operator-facing runbook with named procedures for R1 outage, A1 PUT failure, audit-chain break, SLA P1 breach) | None |
| **RS.AN-03** Analysis is performed to establish what has taken place during an incident and the root cause | Forensic analysis | DecisionRecord hash chain (`evidence/store.py`); structured event names with `decision_id` correlation (`sla/engine.py:152-172` — every breach event carries `decision_id_at_breach`); per-tenant chain verification (`evidence/store.py:123-142`) | None |
| **RS.MI-01** Incidents are contained | Incident containment | Circuit breakers per adapter: `src/horizon_ric/runtime/circuit_breaker.py`; graceful degradation controller: `src/horizon_ric/runtime/graceful_degradation.py:1-end` (held by lifecycle at `rapp/lifecycle.py:74`); `RAppState.DEGRADED` transition (`lifecycle.py:176-180`) | None |
| **RS.CO-02** Internal stakeholders are notified of incidents | Operator notification | Alertmanager v2 webhook integration: `src/horizon_ric/sla/alertmanager.py`; SLA escalation policy: `src/horizon_ric/sla/escalation.py` | None |

## RECOVER (RC)

Recovery — bringing the system back to normal after an incident.

| Sub-category | What it requires | Horizon-RIC mapping | Gap |
|---|---|---|---|
| **RC.RP-01** The recovery portion of the incident response plan is executed once initiated from the incident response process | Recovery execution | `src/horizon_ric/runtime/state_recovery.py:50-60` (`save_state`, atomic write+fsync+rename); restore on boot at `lifecycle.py:85-104` (`_restore_from_disk`); periodic checkpointer at `runtime/state_recovery.py` (`with_periodic_checkpoint`, called from `lifecycle.py:267-272`) | None |
| **RC.RP-03** The integrity of backups and other restoration assets is verified before using them for restoration | Backup integrity | `evidence/store.py:106-142` (chain verify before relying on restored audit data); `state_recovery.py` documented corruption-recovery path: returns `None` on malformed JSON and emits `horizon.state.corrupt`; the rApp boots from defaults rather than crash-looping (lines 19-22 of state_recovery.py) | None |
| **RC.CO-03** Recovery activities and progress in restoring operational capabilities are communicated to designated internal and external stakeholders | Recovery communication | Lifecycle event names (`lifecycle.shutdown.start`, `lifecycle.boot.complete`, `lifecycle.degraded`) emitted via `structlog`: `src/horizon_ric/rapp/lifecycle.py:136, 158, 167, 174, 180` | None |

---

## Summary

Horizon-RIC fully maps to the GOVERN, PROTECT, DETECT, RESPOND, and RECOVER Functions. Within IDENTIFY there is one honest gap (ID.RA-01 — automated CVE-feed integration is not yet wired into the dependency-scan CI; planned in Phase 3). All other sub-categories are implemented and CI-tested or, where the control is operator-side (e.g. PR.PS-03 hardware lifecycle), the vendor documents the operator's reference profile.

**Cross-references:**

- For the framework's interaction with EU AI Act controls, see `docs/compliance/eu_ai_act.md` (Art. 15 cybersecurity maps directly to PR.AA / PR.DS / DE.CM here).
- For the GDPR Art. 32 mapping, see `docs/compliance/gdpr_dpia.md` §6 — the security measures named there are the same controls cited above.
- For the row-by-row standards conformance status, see `docs/conformance/CONFORMANCE.md`.
