# Nokia MantaRay onboarding package (skeleton)

This directory is the onboarding skeleton for registering Horizon-RIC in
a Nokia MantaRay SMO's rApp catalogue via the SDN-R northbound. As with
the EIAP package (`../eiap/`), the point is to be explicit about what is
verifiable from this repository and honest about what is not.

**There has been no live Nokia validation.** Nothing here has been run
against a MantaRay SMO, a Nokia lab, or any Nokia software. The
registration payload is a skeleton, not an artefact accepted by a real
catalogue.

## What is validated today

The **A1 wire contract** for `HORIZON_A1_DIALECT=mantaray` is pinned
against the documented MantaRay SDN-R URL surface **over a real local
HTTP socket** (uvicorn on `127.0.0.1`, ephemeral port; no
`httpx.MockTransport`, no `unittest.mock`):

- `PUT /sdn-r/api/v1/policies` with the exact camelCase body
  `{policyId, policyTypeId, ricId, rappId, policyData}` — including the
  `rappId` catalogue-correlation field (`horizon-ric-rapp`) that
  distinguishes the MantaRay body from the EIAP one. `ricId` comes from
  `A1AdapterConfig.mantaray_ric_id` (env: `HORIZON_A1_MANTARAY_RIC_ID`).
- `PUT /sdn-r/api/v1/policy-types/{policyTypeId}` for all four default
  Horizon-RIC policy types (20001–20004).
- `GET .../policies/{policyId}/status` and `DELETE .../policies/{policyId}`
  URL shapes (offline contract tier).
- Auth as the server actually receives it: a real OAuth2
  client-credentials round-trip against a live local `/oauth/token`
  endpoint (the same `horizon_ric.rapp.auth.AuthConfig` path a
  Keycloak-fronted MantaRay deployment would use), and the static-bearer
  fallback.

Tests: `tests/test_multi_vendor_integration.py`
(`TestRealSocketVendorContracts` = real socket; Tier 1 of the same module
is the offline `httpx.MockTransport` contract checks). URL surface:
`docs/SMO_INTEGRATION.md`.

## What still requires a MantaRay deployment

- **Registration schema confirmation.** `rapp-registration.json` mirrors
  the publicly documented SDN-R surface; the authoritative MantaRay rApp
  catalogue schema sits behind the Nokia administrator documentation
  login and must be confirmed against a real deployment.
- **Catalogue registration flow** — the actual endpoint, method and
  response codes for submitting this payload, plus any Nokia-side
  validation/approval steps.
- **Keycloak realm wiring** — real token endpoint, client provisioning
  and scopes for `HORIZON_A1_OAUTH_TOKEN_URL` /
  `HORIZON_A1_OAUTH_CLIENT_ID` / `HORIZON_A1_OAUTH_CLIENT_SECRET`.
- **Live acceptance** — A1 policy create/status/delete against a
  MantaRay-fronted Near-RT RIC and operator interoperability testing.

## Files

- `rapp-registration.json` — catalogue registration skeleton: rApp
  identity (`horizon-ric-rapp`, version 0.2.0), container image ref,
  `a1Endpoint` with dialect `mantaray`, JWT/Keycloak auth env names,
  health/metrics ports (8081/8082).

See also `docs/VENDOR_ONBOARDING.md` for the dialect-by-dialect validation
matrix and the full environment-variable reference.
