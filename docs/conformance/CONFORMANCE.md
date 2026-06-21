# PreceptualAI Conformance Report

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


**Version:** 0.2.0
**Date:** 2026-05-06
**Component:** [`docs/oda/component.yaml`](../oda/component.yaml)
**OpenAPI:** [`docs/openapi/horizon-ric-rapp.yaml`](../openapi/horizon-ric-rapp.yaml)
**SBOM:** [`deploy/sbom/horizon-ric-sbom.json`](../../deploy/sbom/horizon-ric-sbom.json)
(CycloneDX 1.5)
**Image signing key:** [`deploy/cosign/cosign.pub`](../../deploy/cosign/cosign.pub)

Status legend:

| Glyph | Meaning |
| ----- | ------- |
| ✅ | Implemented and continuously tested in CI |
| ⚠️ | Partially implemented — gaps tracked in `GAPS_TO_PILOT.md` |
| ❌ | Not applicable — out of scope for this component |
| 🟡 | Planned — design exists, code not yet landed |

---

## O-RAN.WG2.R1AP-v06.00 — R1 Service Exposure

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §5.3.1 | rApp registration with SME | ⚠️ | `src/horizon_ric/rapp/r1_adapter.py:99` (`R1Adapter.register`) |
| §5.4 | Service heartbeat / liveness | ✅ | `src/horizon_ric/rapp/health.py` (Prometheus + sd_notify) |
| §5.5 | Service deregistration on shutdown | ⚠️ | `src/horizon_ric/rapp/lifecycle.py` |
| §6 | Authentication (OAuth2 + bearer JWT) | ✅ | `src/horizon_ric/rapp/auth.py`, `src/horizon_ric/rapp/api.py` (BearerJWT scheme) |

## O-RAN.WG2.A1AP-v05.00 — A1 Policy

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §6.3 (PUT policy) | Policy emission per type | ✅ | `src/horizon_ric/rapp/a1_adapter.py:190` (`emit_policy`) |
| §6.4 (GET status) | Policy status retrieval | ✅ | `src/horizon_ric/rapp/a1_adapter.py:273` (`get_policy_status`) |
| §6.5 (DELETE) | Policy rollback | ✅ | `src/horizon_ric/rapp/a1_adapter.py:365` (`rollback_policy`) |
| §7 (A1-EI) | Enrichment Information jobs | ✅ | `src/horizon_ric/rapp/a1_adapter.py:320` (`create_ei_job`) |
| Annex A | Default policy types (1, 2, 20000) | ✅ | `src/horizon_ric/rapp/a1_adapter.py` (`DEFAULT_POLICY_TYPES`) |

## O-RAN.WG2.O1-v06.00 — O1 NETCONF/YANG

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §6.2 NETCONF 1.1 over SSH | Session establishment | ✅ | `src/horizon_ric/rapp/o1_adapter.py:103` (`connect`) |
| §6.3 get-config | Datastore retrieval (XPath/subtree) | ✅ | `src/horizon_ric/rapp/o1_adapter.py:145` (`get_config`) |
| §6.4 edit-config | Datastore mutation | ✅ | `src/horizon_ric/rapp/o1_adapter.py:165` (`edit_config`) |
| §7 NRM YANG | TS 28.541 NRM module ingest | ⚠️ | `src/horizon_ric/rapp/o1_adapter.py:25` (NRM scope), `deploy/yang/` |

## O-RAN.WG11 Security Specification

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §6.1 transport | mTLS for all R1/A1/O1 channels | ✅ | `src/horizon_ric/rapp/auth.py` (`AuthConfig`, `build_secure_async_client`) |
| §6.3 authn | OAuth2 + JWT bearer | ✅ | `src/horizon_ric/rapp/auth.py`, `src/horizon_ric/rapp/api.py` |
| §6.4 authz | RBAC (Casbin) | ✅ | `src/horizon_ric/security/rbac.py`, `rbac_model.conf`, `rbac_policy.csv` |
| §6.5 supply chain | Image signing + SBOM | ✅ | `deploy/cosign/`, `deploy/sbom/` |

## 3GPP TS 28.105 — AI/ML Management

| Subsection | Requirement | Status | Reference |
| --- | --- | --- | --- |
| §7 model description card | name/version/training_data/metrics/sha256 | ✅ | `checkpoints/*.md` (canonical block), `tests/test_ts28105_model_card_emit.py` |
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
| §6.3.1 SLA breach probability | ✅ | `src/horizon_ric/heads/sla_risk.py` (multi-horizon head) |
| §6.3.2 Throughput trend | ✅ | `src/horizon_ric/sla/engine.py` |
| §6.3.5 Energy KWh per decision | ✅ | `src/horizon_ric/evidence/schema.py` (`PredictedOutcome.energy_kwh`) |
| §6.4 SLA-policy associations | ✅ | `src/horizon_ric/sla/policy.py` |

## ITU-R Propagation

| Recommendation | Status | Reference |
| --- | --- | --- |
| ITU-R P.525-4 (free-space path loss) | ✅ | `src/horizon_ric/planner/physics/propagation.py:53` (`free_space_path_loss_dB`) |
| ITU-R P.676-13 (gas attenuation, Annex 2) | ✅ | `src/horizon_ric/planner/physics/propagation.py:95` (`gas_attenuation_dB`) |
| ITU-R P.838-3 (rain k/α) | ✅ | `src/horizon_ric/planner/physics/propagation.py:151` (`rain_attenuation_dB`), Table 1 in `_itu_tables.py` |
| ITU-R P.618-13 (slant-path) | ✅ | `src/horizon_ric/planner/physics/propagation.py` (path-length reduction in `rain_attenuation_dB`) |

## ITU-R Spectrum / NGSO

| Recommendation | Status | Reference |
| --- | --- | --- |
| ITU-R S.1428-1 (ES antenna mask) | ✅ | `src/horizon_ric/planner/physics/s1428.py` |
| ITU-R S.1503-3 (EPFD aggregation) | ✅ | `src/horizon_ric/planner/physics/epfd.py` (Annex 1 Eq. (1) at line 91) |

## TM Forum ODA / Open APIs

| Spec | Status | Reference |
| --- | --- | --- |
| TMF640 Service Activation | 🟡 | declared in `docs/oda/component.yaml` |
| TMF638 Service Inventory | 🟡 | declared in `docs/oda/component.yaml` |
| TMF724 Intent Management | 🟡 | declared in `docs/oda/component.yaml` |
| TMF630 API design guidelines | ✅ | `src/horizon_ric/rapp/api.py` (OpenAPI 3.1, JSON, RFC 7807-style errors via FastAPI) |

---

## Continuous validation

These claims are gated by CI:

| Check | Workflow | What it asserts |
| --- | --- | --- |
| Unit + integration tests | `.github/workflows/test.yml` | `pytest tests/` green |
| OpenAPI spec validity | `tests/test_conformance.py::test_openapi_spec_is_valid_3_1` | OpenAPI 3.1, BearerJWT scheme present |
| TS 28.105 cards | `tests/test_ts28105_model_card_emit.py` | sha256 + size match every shipped checkpoint |
| Prom metric naming | `tests/test_conformance.py::test_prometheus_metric_naming_convention` | `<namespace>_<subsystem>_<name>_<unit>` |
| SBOM validity | `deploy/sbom/validate_sbom.py` (run in `.github/workflows/sbom.yml`) | CycloneDX 1.5 strict schema |
| Image signing | `.github/workflows/build.yml` | `cosign sign --key …` on every release tag |
