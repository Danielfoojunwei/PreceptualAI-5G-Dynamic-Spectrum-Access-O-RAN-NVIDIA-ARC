# Ericsson EIAP onboarding package (skeleton)

This directory is the onboarding skeleton for running Horizon-RIC as an
rApp on the Ericsson Intelligent Automation Platform (EIAP). It exists to
make the *verifiable* part of the integration explicit and to be
scrupulously honest about the part that still requires an EIAP tenant.

**There has been no live Ericsson validation.** Nothing in this repository
has been run against an EIAP tenant, an Ericsson lab, or any Ericsson
software. Do not read anything below as a claim otherwise.

## What is validated today

The **A1 wire contract** — the exact bytes Horizon-RIC puts on the wire
when configured with `HORIZON_A1_DIALECT=eiap` — is pinned against the
documented EIAP A1 PolicyManagement surface **over a real local HTTP
socket** (uvicorn on `127.0.0.1`, ephemeral port; no `httpx.MockTransport`,
no `unittest.mock`):

- `PUT /A1-PolicyManagement/v2/policies` with the exact camelCase body
  `{policyId, policyTypeId, ricId, policyData}`, where `ricId` comes from
  `A1AdapterConfig.eiap_ric_id` (env: `HORIZON_A1_EIAP_RIC_ID`).
- `PUT /A1-PolicyManagement/v2/policy-types/{policyTypeId}` for all four
  default Horizon-RIC policy types (20001–20004).
- `GET .../policies/{policyId}/status` and `DELETE .../policies/{policyId}`
  URL shapes (offline contract tier).
- Auth headers as the server actually receives them: a static bearer token
  (`Authorization: Bearer <token>`) and a full OAuth2 client-credentials
  round-trip against a real `/oauth/token` endpoint, exercising the same
  `horizon_ric.rapp.auth.AuthConfig` path an EIAP Keycloak would use.

Tests: `tests/test_multi_vendor_integration.py`
(`TestRealSocketVendorContracts` = real socket; the module's Tier 1 tests
are the offline `httpx.MockTransport` contract checks). The URL surface
itself is documented in `docs/SMO_INTEGRATION.md`.

## What still requires an EIAP tenant

- **Descriptor/package schema confirmation.** `rapp-descriptor.yaml` here
  is a neutral-YAML skeleton of the onboarding facts. The authoritative
  EIAP rApp package format (ASD-based CSAR layout, TOSCA metadata,
  Helm-chart embedding rules) is published behind the Ericsson developer
  portal login and must be confirmed against a tenant's App Manager.
- **CSAR packaging specifics** — archive layout, manifest digests, and
  whatever the tenant's onboarding pipeline validates.
- **Catalogue signing** — EIAP package signing keys/certificates are
  issued per tenant; we cannot sign anything here.
- **Keycloak realm wiring** — the real token endpoint URL, client
  provisioning, and scopes for `HORIZON_A1_OAUTH_TOKEN_URL` /
  `HORIZON_A1_OAUTH_CLIENT_ID` / `HORIZON_A1_OAUTH_CLIENT_SECRET`.
- **Live acceptance** — actual A1 policy create/status/delete against an
  EIAP-fronted Near-RT RIC, rApp lifecycle management from the EIAP
  console, and any Ericsson conformance/interoperability checks.

## Files

- `rapp-descriptor.yaml` — onboarding skeleton: rApp identity
  (`horizon-ric-rapp`, version 0.2.0), container image ref
  (`horizonric/rapp:0.2.0`), A1 dialect `eiap`, required env, OAuth2 env
  names, health/metrics ports (8081/8082).

See also `docs/VENDOR_ONBOARDING.md` for the dialect-by-dialect validation
matrix and the full environment-variable reference.
