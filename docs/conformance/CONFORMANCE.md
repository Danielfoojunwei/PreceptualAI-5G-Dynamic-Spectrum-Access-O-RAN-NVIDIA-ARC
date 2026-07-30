# Horizon-RIC Conformance Report

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


**Version:** 0.3.0
**Date:** 2026-07-30
**Component:** [`docs/oda/component.yaml`](../oda/component.yaml)
**OpenAPI:** [`docs/openapi/horizon-ric-rapp.yaml`](../openapi/horizon-ric-rapp.yaml)
**SBOM:** [`deploy/sbom/horizon-ric-sbom.json`](../../deploy/sbom/horizon-ric-sbom.json)
(CycloneDX 1.5)
**Image signing key:** [`deploy/cosign/cosign.pub`](../../deploy/cosign/cosign.pub)

Status legend:

| Glyph | Meaning |
| ----- | ------- |
| ✅ | Implemented and continuously tested in CI |
| ⚠️ | Partially implemented — gaps tracked in `docs/EVALUATION_CRITERIA.md` |
| ❌ | Not applicable — out of scope for this component |
| 🟡 | Planned — design exists, code not yet landed |

---

## O-RAN.WG2.R1AP-v06.00 — R1 Service Exposure

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §5.3.1 | rApp registration with SME | ⚠️ | `src/horizon_ric/rapp/r1_adapter.py:98` (`R1Adapter.register`) |
| §5.4 | Service heartbeat / liveness | ✅ | `src/horizon_ric/rapp/health.py` (Prometheus + sd_notify) |
| §5.5 | Service deregistration on shutdown | ⚠️ | `src/horizon_ric/rapp/lifecycle.py` |
| §6 | Authentication (OAuth2 + bearer JWT) | ✅ | `src/horizon_ric/rapp/auth.py`, `src/horizon_ric/rapp/api.py` (BearerJWT scheme) |

## O-RAN.WG2.A1AP-v05.00 — A1 Policy

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §6.3 (PUT policy) | Policy emission per type | ✅ | `src/horizon_ric/rapp/a1_adapter.py:301` (`emit_policy`) |
| §6.4 (GET status) | Policy status retrieval | ✅ | `src/horizon_ric/rapp/a1_adapter.py:492` (`get_policy_status`) |
| §6.5 (DELETE) | Policy rollback | ✅ | `src/horizon_ric/rapp/a1_adapter.py:584` (`rollback_policy`) |
| §7 (A1-EI) | Enrichment Information jobs | ✅ | `src/horizon_ric/rapp/a1_adapter.py:539` (`create_ei_job`) |
| Annex A | Default policy types registered: `horizon.qos.priority`=20001, `horizon.traffic.steering`=20002, `horizon.admission.control`=20003, `horizon.spectrum.reservation`=20004. All four are registered; only the first three are ever **emitted** by the decision pipeline (`src/horizon_ric/rapp/pipeline.py:251`) — 20004 is exercised over the wire only by `scripts/osc_a1_live_smoke.py` | ✅ | `src/horizon_ric/rapp/a1_adapter.py:56` (`DEFAULT_POLICY_TYPES`) |

## O-RAN.WG2.O1-v06.00 — O1 NETCONF/YANG

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §6.2 NETCONF 1.1 over SSH | Session establishment | ✅ | `src/horizon_ric/rapp/o1_adapter.py:103` (`connect`) |
| §6.3 get-config | Datastore retrieval (XPath/subtree) | ✅ | `src/horizon_ric/rapp/o1_adapter.py:145` (`get_config`) |
| §6.4 edit-config | Datastore mutation | ✅ | `src/horizon_ric/rapp/o1_adapter.py:165` (`edit_config`) |
| §7 NRM YANG | TS 28.541 NRM module ingest | ⚠️ | `src/horizon_ric/rapp/o1_adapter.py:25` (NRM scope), `deploy/yang/` |

## O-RAN.WG3 E2 — E2 Service Models

Horizon-RIC is a Non-RT RIC rApp and is **receive-only** on E2. The package
states it directly: it "deliberately contains NO E2AP/SCTP transport of its own"
(`src/horizon_ric/e2/__init__.py:5`).

| Element | Requirement | Status | Reference |
| --- | --- | --- | --- |
| E2SM-KPM v03.00 | `E2SM-KPM-IndicationMessage` decode (Format 1/2 → one event, Format 3 → one event per `ueMeasReportList` entry) and `E2SM-KPM-IndicationHeader` Format 1 (`colletStartTime`, `senderName`) | ✅ | `src/horizon_ric/e2/kpm_bridge.py:363` (`from_e2sm_kpm_indication`), `tests/test_e2_kpm_bridge.py` |
| E2SM-KPM v03.00 | Committed ASN.1 module, sha256-pinned to the FlexRIC original | ✅ | `src/horizon_ric/e2/asn1/e2sm_kpm_v03.00_standard.asn1`, `src/horizon_ric/e2/asn1/PROVENANCE.md`, `tests/test_e2_kpm_bridge.py::test_committed_asn1_spec_matches_flexric_provenance` |
| E2AP / SCTP | E2 association, RIC Subscription, ActionDefinition, EventTriggerDefinition | ❌ | Out of scope by design: the near-RT RIC owns the E2 association and Horizon consumes the E2SM payloads it surfaces (`src/horizon_ric/e2/__init__.py:3`) |
| E2SM-RC | RIC Control (near-real-time enforcement) | 🟡 | WP4 roadmap, no code and no vendored ASN.1 today. See [`ASSURANCE_PROFILE.md`](ASSURANCE_PROFILE.md) §7.2 |

## O-RAN.WG11 Security Specification

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §6.1 transport | mTLS for all R1/A1/O1 channels | ✅ | `src/horizon_ric/rapp/auth.py` (`AuthConfig`, `build_secure_async_client`) |
| §6.3 authn | OAuth2 + JWT bearer | ✅ | `src/horizon_ric/rapp/auth.py`, `src/horizon_ric/rapp/api.py` |
| §6.4 authz | RBAC (Casbin) | ✅ | `src/horizon_ric/security/rbac.py`, `src/horizon_ric/security/rbac_model.conf`, `src/horizon_ric/security/rbac_policy.csv` |
| §6.5 supply chain | Image signing + SBOM | ✅ | `deploy/cosign/`, `deploy/sbom/` |

## 3GPP TS 28.105 — AI/ML Management

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §7 model description card | name/version/training_data/metrics/sha256 | ✅ | `src/horizon_ric/observability/model_card.py` |
| §8 inference reporting | DecisionRecord schema (TS 28.105 §7.4) | ✅ | `src/horizon_ric/evidence/schema.py` (`DecisionRecord`) |
| §9 lifecycle | training / serving separation, version tracking | ✅ | `src/horizon_ric/evidence/schema.py` (`ModelVersions`) |

## 3GPP TS 28.541 — NRM (YANG modules accepted)

| Module | Status | Reference |
| --- | --- | --- |
| `_3gpp-common-managed-element` | ⚠️ | parsed via `o1_adapter.get_config(filter=…)` |
| `_3gpp-nr-nrm-gnbcucpfunction` | ⚠️ | `deploy/yang/` |
| `_3gpp-nr-nrm-gnbcuupfunction` | ⚠️ | `deploy/yang/` |
| `_3gpp-nr-nrm-gnbdufunction` | ⚠️ | `deploy/yang/` |
| `_3gpp-nr-nrm-rrmpolicy`     | ⚠️ | `deploy/yang/` |

## 3GPP TS 28.552 — KPIs emitted

| KPI | Status | Reference |
| --- | --- | --- |
| §6.3.1 SLA breach probability | ✅ | `src/horizon_ric/policy/counterfactual.py` (`sla_risk_30s/1min/5min`), emitted as the `sla_breach_count` span attribute in `src/horizon_ric/observability/tracing.py` |
| §6.3.5 Energy KWh per decision | ✅ | `src/horizon_ric/evidence/schema.py` (`PredictedOutcome.energy_kwh`) |

## TM Forum ODA / Open APIs

| Spec | Status | Reference |
| --- | --- | --- |
| TMF640 Service Activation | 🟡 | declared in `docs/oda/component.yaml` |
| TMF638 Service Inventory | 🟡 | declared in `docs/oda/component.yaml` |
| TMF724 Intent Management | 🟡 | declared in `docs/oda/component.yaml` |
| TMF630 API design guidelines | ✅ | `src/horizon_ric/rapp/api.py` (OpenAPI 3.1, JSON, RFC 7807-style errors via FastAPI) |

## Horizon-RIC assurance profile (own draft, not an external specification)

The rows above map the repository onto specifications other people publish. The
row below is the reverse: a draft profile of the assurance layer in this
repository, offered as candidate input to a standardisation discussion. It is not
an adopted standard and no vendor has implemented it.

| Document | Scope | Status | Reference |
| --- | --- | --- | --- |
| Draft assurance conformance profile v0.1.0 | Planner action contract, invariant contract, safety-certificate schema and canonical signing form, A1 binding (implemented vs proposed), E2 binding (KPM only), third-party conformance criteria, known gaps | ⚠️ draft | [`docs/conformance/ASSURANCE_PROFILE.md`](ASSURANCE_PROFILE.md) |

---

## Continuous validation

These claims are gated by CI:

| Check | Workflow | What it asserts |
| --- | --- | --- |
| Unit + integration tests | `.github/workflows/test.yml` | `pytest tests/` green |
| SBOM validity | `deploy/sbom/validate_sbom.py` (run in `.github/workflows/sbom.yml`) | CycloneDX 1.5 strict schema |
| Image signing | `.github/workflows/build.yml` | `cosign sign --key …` on every release tag |
