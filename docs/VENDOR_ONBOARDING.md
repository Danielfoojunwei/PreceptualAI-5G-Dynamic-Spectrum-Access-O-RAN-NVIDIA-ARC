# Vendor onboarding — A1 dialects, validation levels, configuration

Horizon-RIC emits A1 policies through one adapter
(`src/horizon_ric/rapp/a1_adapter.py`) with five URL/body dialects. This
page states, per dialect, exactly *how far* each integration is validated
— live simulator, real-socket wire-contract, or offline mock-contract —
and where the onboarding artefacts live. The claim discipline matches the
rest of the repo: no dialect is described as vendor-validated unless a
vendor stack was actually in the loop (today: none are).

## Validation levels

| Level | Meaning |
|---|---|
| **Live OSC simulator** | CI builds the official o-ran-sc simulator/reference from a pinned commit and drives it over real HTTP. Strongest evidence available without a vendor tenant. |
| **Real-socket wire-contract** | A local FastAPI/uvicorn server on `127.0.0.1` (ephemeral port) records the actual bytes received over a genuine TCP socket — exact URL, JSON body, and `Authorization` header. No `httpx.MockTransport`, no `unittest.mock`. Proves the emitted wire format, not vendor interoperability. |
| **Offline mock-contract** | `httpx.MockTransport` in-process contract checks of URL shape and body keys. No socket is opened. |

## Dialect × validation matrix

| Dialect | Vendor surface | Highest validation today | Evidence |
|---|---|---|---|
| `legacy` | Historical near-RT-RIC A1AP mirror (`/A1-P/v2/...`) | **Live official A1 mediator + hw-python xApp** — full pipeline 12/12 accepted, 12/12 `enforceStatus=ENFORCED`, three-witness proof pass (see [Live E2E results](#live-e2e-results)) | `deploy/XAPP_E2E_PROOF.md`, `scripts/xapp_e2e_proof.py`, `tests/test_multi_vendor_integration.py` (Tier 1), `tests/test_rapp_lifecycle.py` |
| `osc` | OSC NONRTRIC Policy Management Service (`/a1-policy/v2/...`) | Real-socket wire-contract (recorded docker-compose run against the OSC reference images; historical record) | `deploy/OSC_NONRTRIC_PROOF.md`, `tests/test_a1_osc_dialect.py`, Tier 1 |
| `osc_a1` | OSC A1 2.1.0 Near-RT RIC interface (`/a1-p/...`) | **Live OSC simulator (CI-gated)** — official `o-ran-sc/sim-a1-interface`, pinned commit | `.github/workflows/osc-a1-integration.yml`, `scripts/osc_a1_live_smoke.py`, `tests/test_a1_osc_a1_dialect.py` |
| `eiap` | Ericsson EIAP A1 PolicyManagement (`/A1-PolicyManagement/v2/...`) | Real-socket wire-contract (local uvicorn; exact camelCase body + bearer/OAuth2 headers) — **no live Ericsson validation** | `tests/test_multi_vendor_integration.py::TestRealSocketVendorContracts`, `tests/test_a1_eiap_dialect.py`, [`deploy/onboarding/eiap/`](../deploy/onboarding/eiap/README.md) |
| `mantaray` | Nokia MantaRay SMO via SDN-R (`/sdn-r/api/v1/...`) | Real-socket wire-contract (local uvicorn; exact camelCase body incl. `rappId` + OAuth2 round-trip) — **no live Nokia validation** | `tests/test_multi_vendor_integration.py::TestRealSocketVendorContracts`, `tests/test_a1_mantaray_dialect.py`, [`deploy/onboarding/mantaray/`](../deploy/onboarding/mantaray/README.md) |

Onboarding packages (descriptor/registration skeletons plus an honest
statement of what still requires vendor tenant access):

- Ericsson EIAP: [`deploy/onboarding/eiap/`](../deploy/onboarding/eiap/README.md)
  (`rapp-descriptor.yaml`)
- Nokia MantaRay: [`deploy/onboarding/mantaray/`](../deploy/onboarding/mantaray/README.md)
  (`rapp-registration.json`)

Wire-level details of every dialect (URL tables, body schemas):
[`docs/SMO_INTEGRATION.md`](SMO_INTEGRATION.md).

## Environment-variable configuration

Common to all vendors:

| Variable | Maps to | Meaning |
|---|---|---|
| `HORIZON_A1_DIALECT` | `A1AdapterConfig.dialect` | One of `legacy`, `osc`, `osc_a1`, `eiap`, `mantaray`. |
| `HORIZON_NEAR_RT_RIC_URL` | `A1AdapterConfig.near_rt_ric_base_url` | Base URL of the A1 termination (SMO/PMS or Near-RT RIC). |
| `HORIZON_A1_RIC_ID` | `A1AdapterConfig.osc_ric_id` (and default for vendor RIC ids) | RIC identifier carried in dialects whose body names a RIC. |
| `HORIZON_A1_SERVICE_ID` | `A1AdapterConfig.osc_service_id` | `service_id` field of the OSC PMS body. |
| `HORIZON_RAPP_ID` | `A1AdapterConfig.rapp_id` | rApp identity; emitted as `rappId` in the MantaRay body. |

Vendor-specific overrides:

| Variable | Maps to | Used by |
|---|---|---|
| `HORIZON_A1_EIAP_RIC_ID` | `A1AdapterConfig.eiap_ric_id` | `eiap` — camelCase `ricId` body field. |
| `HORIZON_A1_MANTARAY_RIC_ID` | `A1AdapterConfig.mantaray_ric_id` | `mantaray` — camelCase `ricId` body field. |

Authentication (all consumed through `horizon_ric.rapp.auth.AuthConfig`;
static token wins over OAuth2 when both are set):

| Variable | Maps to | Meaning |
|---|---|---|
| `A1_CLIENT_TOKEN` | `AuthConfig.static_bearer_token` | Static bearer token (Helm: `secret.data.a1ClientToken`). Sent as `Authorization: Bearer <token>`. |
| `HORIZON_A1_OAUTH_TOKEN_URL` | `AuthConfig.token_url` | OAuth2 token endpoint (EIAP/MantaRay: the tenant Keycloak's `.../protocol/openid-connect/token`). |
| `HORIZON_A1_OAUTH_CLIENT_ID` | `AuthConfig.client_id` | OAuth2 client-credentials client id. |
| `HORIZON_A1_OAUTH_CLIENT_SECRET` | `AuthConfig.client_secret` | OAuth2 client secret — mount from a Secret. |
| `HORIZON_A1_CLIENT_CERT_PATH` | `AuthConfig.client_cert_path` | PEM client certificate for mTLS (O-RAN WG11 §6). |
| `HORIZON_A1_CLIENT_KEY_PATH` | `AuthConfig.client_key_path` | PEM client key for mTLS. |
| `HORIZON_A1_CA_BUNDLE_PATH` | `AuthConfig.ca_bundle_path` | CA bundle used to verify the SMO endpoint. |

Per-vendor quick reference:

- **OSC (PMS)**: `HORIZON_A1_DIALECT=osc`, `HORIZON_A1_RIC_ID`,
  `HORIZON_A1_SERVICE_ID`; auth optional on the reference sandbox.
- **OSC (direct A1 simulator)**: `HORIZON_A1_DIALECT=osc_a1`; no body-level
  RIC id (nested URL addressing).
- **Ericsson EIAP**: `HORIZON_A1_DIALECT=eiap`, `HORIZON_A1_EIAP_RIC_ID`,
  plus the three `HORIZON_A1_OAUTH_*` variables pointing at the EIAP
  Keycloak. See `deploy/onboarding/eiap/rapp-descriptor.yaml`.
- **Nokia MantaRay**: `HORIZON_A1_DIALECT=mantaray`,
  `HORIZON_A1_MANTARAY_RIC_ID`, `HORIZON_RAPP_ID` (catalogue-registered id),
  plus `HORIZON_A1_OAUTH_*` (Keycloak JWT) or `A1_CLIENT_TOKEN`.
  See `deploy/onboarding/mantaray/rapp-registration.json`.

## What "real-socket wire-contract" actually pins

From `tests/test_multi_vendor_integration.py::TestRealSocketVendorContracts`
(uvicorn on `127.0.0.1`, ephemeral port, adapter client unmodified):

- EIAP emit — `PUT /A1-PolicyManagement/v2/policies`, body exactly
  `{"policyId": ..., "policyTypeId": "20001", "ricId": <eiap_ric_id>,
  "policyData": {...}}`.
- MantaRay emit — `PUT /sdn-r/api/v1/policies`, body exactly
  `{"policyId": ..., "policyTypeId": "20001", "ricId": <mantaray_ric_id>,
  "rappId": "horizon-ric-rapp", "policyData": {...}}`.
- Policy-type registration — `PUT .../policy-types/{20001..20004}` on both
  surfaces.
- Static bearer — server receives `Authorization: Bearer <token>` with
  `AuthConfig(production_mode=True)`.
- OAuth2 client-credentials — adapter POSTs
  `grant_type=client_credentials` to a real local `/oauth/token`, then the
  policy PUT arrives with `Authorization: Bearer <access_token>`.

## Live E2E results

Measured 2026-07-27 on a clean Linux host; full narrative and raw
artifacts in [`deploy/XAPP_E2E_PROOF.md`](../deploy/XAPP_E2E_PROOF.md)
and [`deploy/xapp-e2e/results/`](../deploy/xapp-e2e/results/).

| Dialect | Live target | Result |
| --- | --- | --- |
| `osc_a1` | Official `sim-a1-interface` @ `be2943f5…` (OSC_2.1.0), built from source | Smoke **pass** (register 20001–20004, create 202, status, delete) **and** full pipeline `--once`: 12/12 events → accepted policies, 0 blocked, 0 failed, audit chain intact |
| `legacy` | Official Go A1 mediator @ `09a757b4…` + RMR 4.9.4 + **official `hw-python` xApp** @ `a6d00525…` | Full pipeline: **12/12 accepted, 12/12 `enforceStatus=ENFORCED`** after the real xApp ACKed each policy over RMR (`handler_id: "hw-python"`); three-witness proof `scripts/xapp_e2e_proof.py` **pass** |
| `osc` | — (NONRTRIC PMS registry unreachable from this host) | Unchanged: recorded real-socket proof (`deploy/OSC_NONRTRIC_PROOF.md`) + contract tests |
| `eiap`, `mantaray` | — (no vendor tenant) | Unchanged: real-socket wire-contract tests only; **no live vendor claim** |
