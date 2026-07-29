# SMO Integration Guide

> *Canonical-to-v3-trust-layer-wave: 2026-05-08. See [`README.md`](../README.md) for the 49-section deep dive of current state, performance, tests, and roadmap.*


Horizon-RIC ships as a vendor-portable non-RT RIC rApp. Operators wire
it into their SMO of choice by selecting an A1 *dialect* on
`A1AdapterConfig`. A dialect is the (URL surface + body shape + auth
mechanism) tuple that a specific SMO vendor publishes for the A1
PolicyManagement interface — the rApp's policy logic is unchanged
across vendors; only the wire shape differs.

This document is the canonical reference for:

  * the four A1 dialects Horizon-RIC supports today
  * what an operator changes to add a fifth (or N+1) SMO vendor
  * the per-capability × per-dialect compatibility matrix
  * the exact URL Horizon-RIC PUTs a policy to, for each dialect

## 1. Dialect switch

`A1AdapterConfig.dialect` selects one of:

| Dialect   | Vendor / SMO                                   | Auth                   |
|-----------|------------------------------------------------|------------------------|
| `legacy`  | Historical near-RT-RIC A1AP mirror             | plain HTTP (dev only)  |
| `osc`     | OSC NONRTRIC PMS reference                     | OAuth2 / mTLS optional |
| `eiap`    | Ericsson EIAP rApp SDK / EIAP A1 PolicyManagement | OAuth2 client-credentials with the Ericsson IDP |
| `mantaray`| Nokia MantaRay SMO via SDN-R                   | Keycloak-issued JWT    |

The dialect drives:
  * `A1Adapter._policy_create_url(...)` — where to PUT a new policy
  * `A1Adapter._policy_instance_url(...)` — where to GET / DELETE one
  * `A1Adapter._policy_status_url(...)` — where to read enforcement state
  * `A1Adapter._policy_list_url(...)` — list endpoint shape
  * the body schema (snake_case for OSC, camelCase for EIAP / MantaRay)

Nothing else in the rApp changes between dialects — the policy types,
JSON schemas, decision-record contract, and evidence chain are dialect-
agnostic.

## 2. Adding a new SMO vendor

A new vendor is **one new dialect, one URL builder, one schema map**.
Concretely, add a fifth dialect (call it `acme`) like so:

  1. **Extend the enum**: add `acme` to the documented values of
     `A1AdapterConfig.dialect` (no Enum class is enforced — strings are
     used for forward-compat).
  2. **URL builders**: add `acme` branches to the four `_policy_*_url`
     methods on `A1Adapter` (file
     `src/horizon_ric/rapp/a1_adapter.py`). Each branch returns the
     vendor-specific path.
  3. **Body shape**: add an `elif self.cfg.dialect == "acme":` branch
     in `emit_policy._do_put` that constructs the body in the shape
     the vendor publishes.
  4. **Optional context fields**: if the vendor carries a non-standard
     identifier (e.g. EIAP carries `ricId`, MantaRay carries `rappId`),
     add a per-vendor field to `A1AdapterConfig` (e.g. `acme_ric_id`).
  5. **Auth**: if the vendor uses a non-standard auth mechanism, wire
     it through `horizon_ric.rapp.auth.AuthConfig`. OAuth2
     client-credentials, static bearer, and mTLS are already supported;
     adding a vendor-specific header injector is a one-line custom
     `httpx.Auth` subclass.
  6. **Tests**: add a per-dialect test module following the pattern of
     the existing `tests/test_a1_osc_dialect.py`,
     `tests/test_a1_mantaray_dialect.py`, and
     `tests/test_a1_eiap_dialect.py` — at least four
     `httpx.MockTransport`-backed assertions (URL, body shape, auth
     header, success status) — and add one row in
     `tests/test_multi_vendor_integration.py::DIALECTS`.

That is the complete contract. Schema maps for the four
Horizon-RIC-default policy types (`horizon.qos.priority`,
`horizon.traffic.steering`, `horizon.admission.control`,
`horizon.spectrum.reservation`) are vendor-independent — every dialect
ships the same JSON Schema for the policy payload.

## 3. Compatibility matrix

| Capability                                   | legacy | osc | eiap | mantaray |
|----------------------------------------------|:------:|:---:|:----:|:--------:|
| A1AP policy create / status / delete         | ✅     | ✅  | ✅   | ✅       |
| A1-EI enrichment job (PUT/DELETE)            | ✅     | ⚠️* | ⚠️*  | ⚠️*      |
| OAuth2 client-credentials                    | n/a    | ✅  | ✅   | ✅       |
| mTLS (WG11 §6 cipher allow-list)             | ✅     | ✅  | ✅   | ✅       |
| Tamper-evident evidence store                | ✅     | ✅  | ✅   | ✅       |
| Federated weights-only learning              | ✅     | ✅  | ✅   | ✅       |
| Cosign-signed audit export                   | ✅     | ✅  | ✅   | ✅       |
| O-RAN.WG2.O1-v06.00 PM bulk-data ingest      | ✅     | ✅  | ✅   | ✅       |

\* A1-EI is currently emitted on the legacy path (`/A1-EI/v1/eijobs/...`)
regardless of dialect because all four SMOs ship A1-EI compatibility
shims under that path. A vendor-specific A1-EI URL switch is on the
roadmap and would follow the same one-branch-per-dialect pattern.

## 4. Exact URL Horizon-RIC PUTs a policy to, per dialect

These are the exact URL paths emitted by `A1Adapter.emit_policy(...)`
in Horizon-RIC v0.2.0 for the default `horizon.qos.priority` policy
type (policy_type_id = 20001). The base URL is
`A1AdapterConfig.near_rt_ric_base_url` and is prepended in every case.

| Dialect    | Method | URL path                                                                |
|------------|--------|-------------------------------------------------------------------------|
| `legacy`   | PUT    | `/A1-P/v2/policytypes/20001/policies/{policy_id}`                       |
| `osc`      | PUT    | `/a1-policy/v2/policies`                                                |
| `eiap`     | PUT    | `/A1-PolicyManagement/v2/policies`                                      |
| `mantaray` | PUT    | `/sdn-r/api/v1/policies`                                                |

The corresponding bodies:

  * **legacy**: the policy JSON itself, conforming to the registered
    type schema (no envelope).
  * **osc**: `{policy_id, policytype_id, ric_id, service_id, transient,
    policy_data}` per `pms-api.json` in
    `o-ran-sc/nonrtric-plt-a1policymanagementservice`.
  * **eiap**: `{policyId, policyTypeId, ricId, policyData}` (camelCase,
    per the public Ericsson rApp SDK contract).
  * **mantaray**: `{policyId, policyTypeId, ricId, rappId, policyData}`
    (camelCase plus `rappId` for SDN-R rApp catalogue correlation).

## 5. Multi-tenant isolation

Tenant isolation is enforced in the security layer, independent of the
SMO dialect:

  * `TenantScope` (`horizon_ric.security.tenant`) — drives the active
    `tenant_id` for the duration of one request.
  * Per-tenant SHA-256 hash chain in the evidence store
    (`horizon_ric.evidence.store`) — a tamper of tenant A's chain
    leaves tenant B's chain intact.

## 6. Honest blockers

  * **Ericsson EIAP**: the full EIAP rApp SDK reference requires an
    Ericsson Developer Hub login. The dialect implementation and these
    tests pin the **publicly-documented** portion of the contract.
  * **Nokia MantaRay**: same caveat — the full MantaRay administrator
    guide and DAC listing schema live behind a Nokia partner portal
    login. The MantaRay dialect honours the public SDN-R URL surface
    and JWT auth flow.

In both cases Horizon-RIC's payloads are validated against the
public surface; a real partner integration replaces the mock transport
with the vendor's live endpoint and adds the partner-portal-only
fields. Neither replacement requires a code change in
Horizon-RIC itself — the dialect switch is the seam.
